"""Authenticated, outbound-only transport for Android model workers.

The registry and inference routes share one process-local WebSocket broker.
Connections are leases: a process restart or a missed heartbeat makes the node
unhealthy, and nodes must reconnect with their node-scoped token.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import httpx
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app import db
from app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
HEARTBEAT_LEASE_SECONDS = 60
HEARTBEAT_INTERVAL_SECONDS = 20
HELLO_TIMEOUT_SECONDS = 15
MAX_MESSAGE_BYTES = 64 * 1024
MAX_REQUEST_BYTES = 1024 * 1024
MAX_REQUEST_CHUNK_BYTES = 8 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_GENERATION_TOKENS = 1024
MAX_CONCURRENT_REQUESTS = 1
RESPONSE_QUEUE_SIZE = 8


class MobileTransportError(httpx.HTTPError):
    pass


@dataclass
class PendingRequest:
    request_id: str
    streaming: bool
    queue: asyncio.Queue
    next_sequence: int = 0
    response_bytes: int = 0
    result_bytes_expected: int | None = None
    result_sequence: int = 0
    result_buffer: bytearray = field(default_factory=bytearray)
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    terminal: bool = False


@dataclass
class MobileConnection:
    node_id: UUID
    websocket: WebSocket
    model_name: str
    model_format: str
    runtime_name: str
    runtime_version: str
    runtime_capabilities: frozenset[str]
    last_seen: float = field(default_factory=time.monotonic)
    pending: dict[str, PendingRequest] = field(default_factory=dict)
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    inflight: int = 0
    closed: bool = False


class MobileResponseStream(httpx.AsyncByteStream):
    def __init__(self, manager: "MobileConnectionManager", connection: MobileConnection,
                 pending: PendingRequest, first: dict[str, Any]):
        self.manager = manager
        self.connection = connection
        self.pending = pending
        self.first = first
        self.finished = False

    async def __aiter__(self):
        try:
            frame = self.first
            while True:
                if frame is None:
                    frame = await asyncio.wait_for(
                        self.pending.queue.get(), settings.forward_read_timeout_seconds,
                    )
                if isinstance(frame, BaseException):
                    raise httpx.ReadError(str(frame))
                if frame == "done":
                    self.finished = True
                    yield b"data: [DONE]\n\n"
                    return
                chunk = json.dumps(frame["chunk"], separators=(",", ":"), ensure_ascii=False)
                yield f"data: {chunk}\n\n".encode("utf-8")
                frame = None
        except asyncio.TimeoutError as exc:
            raise httpx.ReadTimeout("Android worker response timed out") from exc
        finally:
            await self.manager.finish_request(
                self.connection, self.pending, cancel=not self.finished,
            )

    async def aclose(self) -> None:
        if not self.finished:
            await self.manager.finish_request(self.connection, self.pending, cancel=True)


class MobileConnectionManager:
    def __init__(self) -> None:
        self.connections: dict[UUID, MobileConnection] = {}
        self._lock = asyncio.Lock()

    async def attach(self, connection: MobileConnection) -> None:
        async with self._lock:
            previous = self.connections.get(connection.node_id)
            self.connections[connection.node_id] = connection
        if previous is not None:
            await self._close_connection(previous, "reconnected")

    async def detach(self, connection: MobileConnection) -> bool:
        async with self._lock:
            if self.connections.get(connection.node_id) is not connection:
                return False
            del self.connections[connection.node_id]
        await self._close_connection(connection, "disconnected")
        return True

    async def disconnect_node(self, node_id: UUID, reason: str = "deregistered") -> None:
        async with self._lock:
            connection = self.connections.pop(node_id, None)
        if connection is not None:
            await self._close_connection(connection, reason)

    async def _close_connection(self, connection: MobileConnection, reason: str) -> None:
        if connection.closed:
            return
        connection.closed = True
        for pending in list(connection.pending.values()):
            while pending.queue.full():
                try:
                    pending.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            pending.queue.put_nowait(MobileTransportError(reason))
            pending.cancelled.set()
        try:
            await connection.websocket.close(code=1001, reason=reason[:100])
        except (RuntimeError, WebSocketDisconnect):
            pass

    def is_live(self, node_id: UUID) -> bool:
        connection = self.connections.get(node_id)
        return bool(
            connection and not connection.closed
            and time.monotonic() - connection.last_seen <= HEARTBEAT_LEASE_SECONDS
        )

    async def expire_stale(self) -> list[UUID]:
        now = time.monotonic()
        async with self._lock:
            stale = [
                node_id for node_id, connection in self.connections.items()
                if now - connection.last_seen > HEARTBEAT_LEASE_SECONDS
            ]
            connections = [self.connections.pop(node_id) for node_id in stale]
        for connection in connections:
            await self._close_connection(connection, "heartbeat lease expired")
        return stale

    async def forward(self, node: dict, body: dict[str, Any], streaming: bool) -> httpx.Response:
        connection = self.connections.get(node["id"])
        if connection is None or not self.is_live(node["id"]):
            raise httpx.ConnectError("Android worker is offline")
        if connection.inflight >= MAX_CONCURRENT_REQUESTS:
            raise httpx.ConnectError("Android worker is at its concurrency limit")

        outgoing = dict(body)
        outgoing["model"] = connection.model_name
        outgoing["stream"] = streaming
        outgoing["max_tokens"] = self._bounded_max_tokens(outgoing.get("max_tokens"))
        outgoing.pop("max_completion_tokens", None)
        outgoing["n"] = 1
        encoded = json.dumps(outgoing, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if len(encoded) > MAX_REQUEST_BYTES:
            raise httpx.WriteError("request exceeds the Android worker payload limit")

        request_id = str(uuid4())
        pending = PendingRequest(request_id, streaming, asyncio.Queue(RESPONSE_QUEUE_SIZE))
        connection.pending[request_id] = pending
        connection.inflight += 1
        try:
            async with connection.send_lock:
                await self._send(connection, {
                    "type": "request.start",
                    "request_id": request_id,
                    "protocol_version": PROTOCOL_VERSION,
                    "model": connection.model_name,
                    "model_format": connection.model_format,
                    "stream": streaming,
                    "body_bytes": len(encoded),
                })
                for sequence, offset in enumerate(range(0, len(encoded), MAX_REQUEST_CHUNK_BYTES)):
                    chunk = encoded[offset:offset + MAX_REQUEST_CHUNK_BYTES]
                    await self._send(connection, {
                        "type": "request.data",
                        "request_id": request_id,
                        "sequence": sequence,
                        "data": base64.b64encode(chunk).decode("ascii"),
                    })
                await self._send(connection, {"type": "request.end", "request_id": request_id})
            first = await asyncio.wait_for(
                pending.queue.get(), settings.forward_read_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            await self.finish_request(connection, pending, cancel=True)
            raise httpx.ReadTimeout("Android worker did not start responding") from exc
        except asyncio.CancelledError:
            await self.finish_request(connection, pending, cancel=True)
            raise
        except Exception as exc:
            await self.finish_request(connection, pending, cancel=True)
            if isinstance(exc, httpx.WriteError):
                await self.disconnect_node(connection.node_id, "channel write failed")
            raise

        if isinstance(first, BaseException):
            await self.finish_request(connection, pending, cancel=False)
            raise httpx.ConnectError(str(first))

        if streaming:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=MobileResponseStream(self, connection, pending, first),
            )

        try:
            if first.get("type") != "response.result":
                raise httpx.ProtocolError("Android worker returned a stream for a non-stream request")
            result = first.get("result")
            content = json.dumps(result, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            if len(content) > MAX_RESPONSE_BYTES:
                raise httpx.ReadError("Android worker response exceeds the payload limit")
            return httpx.Response(200, content=content, headers={"content-type": "application/json"})
        finally:
            await self.finish_request(connection, pending, cancel=False)

    @staticmethod
    def _bounded_max_tokens(value: Any) -> int:
        if isinstance(value, bool):
            return MAX_GENERATION_TOKENS
        try:
            requested = int(value) if value is not None else MAX_GENERATION_TOKENS
        except (TypeError, ValueError, OverflowError):
            return MAX_GENERATION_TOKENS
        return max(1, min(requested, MAX_GENERATION_TOKENS))

    async def _send(self, connection: MobileConnection, payload: dict[str, Any]) -> None:
        wire = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        if len(wire.encode("utf-8")) > MAX_MESSAGE_BYTES:
            raise httpx.WriteError("outbound Android protocol frame exceeds the message limit")
        try:
            await asyncio.wait_for(
                connection.websocket.send_text(wire),
                settings.forward_write_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise httpx.WriteTimeout("Android worker channel stopped accepting data") from exc
        except (RuntimeError, WebSocketDisconnect) as exc:
            raise httpx.WriteError("Android worker channel closed during send") from exc

    async def receive(self, connection: MobileConnection, message: str) -> bool:
        if len(message.encode("utf-8")) > MAX_MESSAGE_BYTES:
            raise ValueError("oversized protocol message")
        try:
            frame = json.loads(message)
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise ValueError("malformed protocol JSON") from exc
        if not isinstance(frame, dict) or not isinstance(frame.get("type"), str):
            raise ValueError("protocol frame must be a JSON object with a type")

        connection.last_seen = time.monotonic()
        frame_type = frame["type"]
        if frame_type == "heartbeat":
            await self._send(connection, {"type": "heartbeat.ack", "lease_seconds": HEARTBEAT_LEASE_SECONDS})
            return True
        if frame_type == "disconnect":
            return False
        request_id = frame.get("request_id")
        pending = connection.pending.get(request_id) if isinstance(request_id, str) else None
        if pending is None:
            return True
        if pending.terminal:
            return True

        if frame_type == "response.chunk":
            if (
                not pending.streaming or isinstance(frame.get("sequence"), bool)
                or not isinstance(frame.get("sequence"), int)
                or frame.get("sequence") != pending.next_sequence
            ):
                raise ValueError("unexpected response sequence or non-stream chunk")
            chunk = frame.get("chunk")
            if not isinstance(chunk, dict):
                raise ValueError("response chunk must be an object")
            encoded = json.dumps(chunk, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            pending.response_bytes += len(encoded)
            if pending.response_bytes > MAX_RESPONSE_BYTES:
                await self._queue_response(
                    pending, MobileTransportError("response exceeds the Android worker payload limit"),
                )
            else:
                await self._queue_response(pending, {"type": frame_type, "chunk": chunk})
                pending.next_sequence += 1
            return True
        if frame_type == "response.done":
            if not pending.streaming:
                raise ValueError("unexpected stream terminator")
            pending.terminal = True
            await self._queue_response(pending, "done")
            return True
        if frame_type == "response.result.start":
            body_bytes = frame.get("body_bytes")
            if (
                pending.streaming or pending.result_bytes_expected is not None
                or isinstance(body_bytes, bool) or not isinstance(body_bytes, int)
                or body_bytes < 1 or body_bytes > MAX_RESPONSE_BYTES
            ):
                raise ValueError("invalid non-stream result size")
            pending.result_bytes_expected = body_bytes
            return True
        if frame_type == "response.result.data":
            if (
                pending.streaming or pending.result_bytes_expected is None
                or isinstance(frame.get("sequence"), bool)
                or not isinstance(frame.get("sequence"), int)
                or frame.get("sequence") != pending.result_sequence
            ):
                raise ValueError("unexpected non-stream result sequence")
            try:
                data = base64.b64decode(frame.get("data", ""), validate=True)
            except (ValueError, TypeError) as exc:
                raise ValueError("malformed non-stream result data") from exc
            if not data or len(data) > MAX_REQUEST_CHUNK_BYTES:
                raise ValueError("invalid non-stream result chunk size")
            pending.result_buffer.extend(data)
            if len(pending.result_buffer) > pending.result_bytes_expected:
                raise ValueError("non-stream result exceeds its declared size")
            pending.result_sequence += 1
            return True
        if frame_type == "response.result.end":
            if (
                pending.streaming or pending.result_bytes_expected is None
                or len(pending.result_buffer) != pending.result_bytes_expected
            ):
                raise ValueError("incomplete non-stream result")
            try:
                result = json.loads(pending.result_buffer)
            except (json.JSONDecodeError, UnicodeError) as exc:
                raise ValueError("malformed non-stream result JSON") from exc
            if not isinstance(result, dict):
                raise ValueError("non-stream result must be a JSON object")
            pending.terminal = True
            await self._queue_response(pending, {"type": "response.result", "result": result})
            return True
        if frame_type == "response.error":
            error = frame.get("message")
            if not isinstance(error, str) or len(error) > 512:
                raise ValueError("malformed worker error")
            pending.terminal = True
            await self._queue_response(pending, MobileTransportError(error))
            return True
        raise ValueError("unknown protocol frame type")

    @staticmethod
    async def _queue_response(pending: PendingRequest, value: Any) -> None:
        if pending.cancelled.is_set():
            return
        put_task = asyncio.create_task(pending.queue.put(value))
        cancel_task = asyncio.create_task(pending.cancelled.wait())
        done, _ = await asyncio.wait(
            (put_task, cancel_task), return_when=asyncio.FIRST_COMPLETED,
        )
        if cancel_task in done:
            put_task.cancel()
            await asyncio.gather(put_task, return_exceptions=True)
        else:
            cancel_task.cancel()
            await asyncio.gather(cancel_task, return_exceptions=True)

    async def finish_request(self, connection: MobileConnection, pending: PendingRequest,
                             *, cancel: bool) -> None:
        if connection.pending.pop(pending.request_id, None) is None:
            return
        connection.inflight = max(0, connection.inflight - 1)
        pending.cancelled.set()
        if cancel and not connection.closed:
            try:
                async with connection.send_lock:
                    await self._send(connection, {
                        "type": "request.cancel", "request_id": pending.request_id,
                    })
            except (httpx.HTTPError, RuntimeError, WebSocketDisconnect):
                await self.disconnect_node(connection.node_id, "cancel send failed")

    async def serve(self, connection: MobileConnection) -> None:
        try:
            while not connection.closed:
                message = await connection.websocket.receive_text()
                if not await self.receive(connection, message):
                    break
        except WebSocketDisconnect:
            pass
        except ValueError as exc:
            logger.info("closing malformed Android worker channel for node %s: %s",
                        connection.node_id, exc)
            try:
                await connection.websocket.close(code=1009 if "oversized" in str(exc) else 1008)
            except RuntimeError:
                pass
        except RuntimeError:
            logger.info("closing Android worker channel for node %s after invalid message type",
                        connection.node_id)
            try:
                await connection.websocket.close(code=1003)
            except RuntimeError:
                pass
        finally:
            was_current = await self.detach(connection)
            if was_current:
                async with db.pool().acquire() as conn:
                    await conn.execute(
                        "update nodes set healthy = false where id = $1 and transport = 'android_ws'",
                        connection.node_id,
                    )


manager = MobileConnectionManager()


def _bearer_token(websocket: WebSocket) -> str | None:
    token = websocket.headers.get("x-common-node-token")
    if token:
        return token.strip()
    authorization = websocket.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip() or None
    return None


def _valid_hello(frame: Any, row: Any) -> bool:
    return (
        isinstance(frame, dict)
        and frame.get("type") == "hello"
        and frame.get("protocol_version") == PROTOCOL_VERSION
        and frame.get("model_name") == row["model_name"]
        and frame.get("model_format") == row["model_format"]
        and frame.get("runtime_name") == row["runtime_name"]
        and frame.get("runtime_version") == row["runtime_version"]
        and isinstance(frame.get("runtime_capabilities"), list)
        and all(isinstance(capability, str) for capability in frame["runtime_capabilities"])
        and set(frame["runtime_capabilities"]) == set(row["runtime_capabilities"] or [])
    )


@router.websocket("/nodes/{node_id}/connect")
async def android_node_connect(websocket: WebSocket, node_id: UUID) -> None:
    token = _bearer_token(websocket)
    async with db.pool().acquire() as conn:
        row = await conn.fetchrow(
            "select id, node_token, transport, model_name, model_format, runtime_name, "
            "runtime_version, runtime_capabilities from nodes where id = $1",
            node_id,
        )
    if (
        row is None or row["transport"] != "android_ws" or not token
        or not secrets.compare_digest((row["node_token"] or "").encode(), token.encode())
    ):
        await websocket.close(code=4401, reason="node authorization failed")
        return

    await websocket.accept()
    connection = None
    try:
        message = await asyncio.wait_for(websocket.receive_text(), HELLO_TIMEOUT_SECONDS)
        if len(message.encode("utf-8")) > MAX_MESSAGE_BYTES:
            raise ValueError("oversized hello")
        hello = json.loads(message)
        if not _valid_hello(hello, row):
            await websocket.close(code=4403, reason="runtime or model does not match registration")
            return
        connection = MobileConnection(
            node_id=node_id,
            websocket=websocket,
            model_name=row["model_name"],
            model_format=row["model_format"],
            runtime_name=row["runtime_name"],
            runtime_version=row["runtime_version"],
            runtime_capabilities=frozenset(row["runtime_capabilities"] or []),
        )
        await manager.attach(connection)
        async with db.pool().acquire() as conn:
            await conn.execute(
                "update nodes set healthy = true, last_heartbeat = now(), "
                "last_seen_healthy = now() where id = $1 and transport = 'android_ws' "
                "and node_token = $2",
                node_id, token,
            )
        await manager._send(connection, {
            "type": "ready",
            "protocol_version": PROTOCOL_VERSION,
            "heartbeat_interval_seconds": HEARTBEAT_INTERVAL_SECONDS,
            "lease_seconds": HEARTBEAT_LEASE_SECONDS,
            "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
            "max_message_bytes": MAX_MESSAGE_BYTES,
        })
        await manager.serve(connection)
    except (asyncio.TimeoutError, ValueError, json.JSONDecodeError):
        await websocket.close(code=1008, reason="invalid or missing hello")
    finally:
        if connection is not None and await manager.detach(connection):
            async with db.pool().acquire() as conn:
                await conn.execute(
                    "update nodes set healthy = false where id = $1 and transport = 'android_ws'",
                    node_id,
                )
