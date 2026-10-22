"""Protocol and lifecycle tests for the outbound mobile worker transport."""
import asyncio
import base64
import json
import sys
import time
import unittest
from pathlib import Path
from unittest import mock
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
from fastapi import HTTPException  # noqa: E402

from app import mobile_transport, upstream  # noqa: E402
from app import catalogue as catalogue_module  # noqa: E402
from app import health as health_module  # noqa: E402
from app.catalogue import AssignRequest, HardwareProfile  # noqa: E402
from app.models import NodeCreate  # noqa: E402
from app import registry as registry_module  # noqa: E402
from app.registry import validate_mobile_node  # noqa: E402
from app.config import settings  # noqa: E402


class FakeWebSocket:
    def __init__(self, headers=None, incoming=()):
        self.headers = headers or {}
        self.incoming = list(incoming)
        self.sent = []
        self.closed = None
        self.accepted = False

    async def send_text(self, text):
        self.sent.append(json.loads(text))

    async def receive_text(self):
        if not self.incoming:
            raise mobile_transport.WebSocketDisconnect()
        return self.incoming.pop(0)

    async def accept(self):
        self.accepted = True

    async def close(self, code=1000, reason=None):
        self.closed = (code, reason)


class FakeDbConnection:
    def __init__(self, row):
        self.row = row
        self.executed = []
        self.handlers = {}

    async def fetchrow(self, query, *args):
        if "from catalogue_models" in query:
            return self.handlers["catalogue"](query, *args)
        if "select node_token" in query:
            return self.handlers["existing"](query, *args)
        if "returning *" in query:
            return self.handlers["write"](query, *args)
        return self.row

    async def execute(self, *args):
        self.executed.append(args)
        return self.handlers.get("execute", lambda *_: "DELETE 1")(*args)

    async def fetch(self, query, *args):
        handler = self.handlers.get("fetch")
        if handler:
            return handler(query, *args)
        if "catalogue_models" in query:
            return self.handlers.get("catalogue_rows", [])
        return []


class FakeAcquire:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, *_args):
        return False


class FakePool:
    def __init__(self, connection):
        self.connection = connection

    def acquire(self):
        return FakeAcquire(self.connection)


class MobileTransportTests(unittest.TestCase):
    def run_async(self, coro):
        return asyncio.run(coro)

    def make_connection(self, manager, node_id=None, websocket=None):
        node_id = node_id or uuid4()
        connection = mobile_transport.MobileConnection(
            node_id=node_id, websocket=websocket or FakeWebSocket(),
            model_name="bonsai2-4b", model_format="Q2_0",
            runtime_name="android-test-runtime", runtime_version="1",
            runtime_capabilities=frozenset({"gguf:q2_0", "ternary:hadamard-activations"}),
        )
        return connection

    def test_authentication_registration_and_explicit_disconnect(self):
        token = "node-token-0123456789"
        node_id = uuid4()
        row = {
            "id": node_id, "node_token": token, "transport": "android_ws",
            "model_name": "bonsai2-4b", "model_format": "Q2_0",
            "runtime_name": "android-test-runtime", "runtime_version": "1",
            "runtime_capabilities": ["gguf:q2_0", "ternary:hadamard-activations"],
        }
        hello = {
            "type": "hello", "protocol_version": 1,
            "model_name": "bonsai2-4b", "model_format": "Q2_0",
            "runtime_name": "android-test-runtime", "runtime_version": "1",
            "runtime_capabilities": row["runtime_capabilities"],
        }
        db_connection = FakeDbConnection(row)

        async def exercise():
            manager = mobile_transport.MobileConnectionManager()
            bad_socket = FakeWebSocket({"authorization": "Bearer wrong"}, [])
            with mock.patch.object(mobile_transport.db, "pool", return_value=FakePool(db_connection)):
                await mobile_transport.mobile_node_connect(bad_socket, node_id)
                self.assertEqual(bad_socket.closed[0], 4401)
                self.assertFalse(bad_socket.accepted)

                socket = FakeWebSocket(
                    {"authorization": f"Bearer {token}"},
                    [json.dumps(hello), json.dumps({"type": "heartbeat"}),
                     json.dumps({"type": "disconnect"})],
                )
                with mock.patch.object(mobile_transport, "manager", manager):
                    await mobile_transport.mobile_node_connect(socket, node_id)
            self.assertTrue(socket.accepted)
            self.assertEqual(socket.sent[0]["type"], "ready")
            self.assertEqual(socket.sent[1]["type"], "heartbeat.ack")
            self.assertFalse(manager.is_live(node_id))
            self.assertTrue(any("healthy = true" in q[0] for q in db_connection.executed))
            self.assertTrue(any("healthy = false" in q[0] for q in db_connection.executed))

        self.run_async(exercise())

    def test_hello_rejects_mismatched_format_and_capabilities(self):
        row = {
            "model_name": "bonsai2-4b", "model_format": "Q2_0",
            "runtime_name": "runtime", "runtime_version": "1",
            "runtime_capabilities": ["gguf:q2_0"],
        }
        hello = {
            "type": "hello", "protocol_version": 1,
            "model_name": "bonsai2-4b", "model_format": "Q2_0",
            "runtime_name": "runtime", "runtime_version": "1",
            "runtime_capabilities": ["gguf:q2_0"],
        }
        self.assertTrue(mobile_transport._valid_hello(hello, row))
        hello["model_format"] = "PTQ1_0"
        self.assertFalse(mobile_transport._valid_hello(hello, row))

    def test_ios_node_uses_authenticated_mobile_channel(self):
        token = "ios-node-token-0123456789"
        node_id = uuid4()
        row = {
            "id": node_id, "node_token": token, "transport": "ios_ws",
            "model_name": "bonsai2-4b", "model_format": "Q2_0",
            "runtime_name": "llama.cpp-ios", "runtime_version": "b11555",
            "runtime_capabilities": ["gguf:q2_0", "ternary:hadamard-activations"],
        }
        hello = {
            "type": "hello", "protocol_version": 1,
            "model_name": row["model_name"], "model_format": row["model_format"],
            "runtime_name": row["runtime_name"], "runtime_version": row["runtime_version"],
            "runtime_capabilities": row["runtime_capabilities"],
        }
        db_connection = FakeDbConnection(row)
        socket = FakeWebSocket(
            {"authorization": f"Bearer {token}"},
            [json.dumps(hello), json.dumps({"type": "disconnect"})],
        )

        async def exercise():
            manager = mobile_transport.MobileConnectionManager()
            with mock.patch.object(mobile_transport.db, "pool", return_value=FakePool(db_connection)), \
                 mock.patch.object(mobile_transport, "manager", manager):
                await mobile_transport.mobile_node_connect(socket, node_id)
            self.assertTrue(socket.accepted)
            self.assertEqual(socket.sent[0]["type"], "ready")
            self.assertTrue(any("healthy = true" in query[0] for query in db_connection.executed))
            self.assertFalse(manager.is_live(node_id))

        self.run_async(exercise())

    def test_catalogue_gates_android_models_and_formats(self):
        common = dict(
            name="phone", model_name="bonsai2-4b", catalogue_id="bonsai2-4b",
            capability_text="Bonsai model", transport="android_ws",
            runtime_name="llama.cpp-android", runtime_version="b11555",
            model_format="Q2_0",
            runtime_capabilities=["gguf:q2_0", "ternary:hadamard-activations"],
        )
        node = NodeCreate(**common)
        model = {
            "id": "bonsai2-4b", "source": "gguf:https://host/model-Q2_0.gguf",
            "weight_format": "Q2_0", "android_supported": True,
            "runtime_requirements": ["gguf:q2_0", "ternary:hadamard-activations"],
        }
        validate_mobile_node(node, model)
        for invalid, catalogue in (
            (NodeCreate(**{**common, "model_format": "PTQ1_0"}), model),
            (node, {**model, "android_supported": False}),
            (NodeCreate(**{**common, "runtime_capabilities": ["gguf:q2_0"]}), model),
            (NodeCreate(**{**common, "endpoint_url": "https://phone.example"}), model),
        ):
            with self.assertRaises(HTTPException):
                validate_mobile_node(invalid, catalogue)

        self.assertEqual(AssignRequest(
            hardware=HardwareProfile(total_ram_gb=8, available_ram_gb=8),
            platform="android", runtime_capabilities=["gguf:q2_0"],
        ).platform, "android")
        ios_node = NodeCreate(**{**common, "transport": "ios_ws"})
        ios_model = {**model, "android_supported": False, "ios_supported": True}
        validate_mobile_node(ios_node, ios_model)
        with self.assertRaises(HTTPException):
            validate_mobile_node(ios_node, model)
        self.assertEqual(AssignRequest(
            hardware=HardwareProfile(total_ram_gb=8, available_ram_gb=8),
            platform="ios", runtime_capabilities=["gguf:q2_0"],
        ).platform, "ios")

    def test_assign_filters_android_runtime_and_preserves_catalogue_format(self):
        models = [
            {
                "id": "bonsai2-2b", "display_name": "Bonsai 2 2B",
                "source": "gguf:https://models.example/model-Q2_0.gguf",
                "domain_tags": ["general"], "capability_text": "general",
                "params_b": 2, "min_ram_gb": 2, "min_vram_gb": 0,
                "needs_gpu": False, "verified_in_lane": False,
                "lane_benchmark": None, "licence": "apache-2.0", "added_at": None,
                "weight_format": "Q2_0",
                "runtime_requirements": ["gguf:q2_0", "ternary:hadamard-activations"],
                "runtime_notes": "stock llama.cpp b11555+", "android_supported": True,
                "ios_supported": True,
            },
            {
                "id": "ordinary-model", "display_name": "Unsupported",
                "source": "gguf:https://models.example/ordinary.gguf",
                "domain_tags": ["general"], "capability_text": "general",
                "params_b": 1, "min_ram_gb": 1, "min_vram_gb": 0,
                "needs_gpu": False, "verified_in_lane": True,
                "lane_benchmark": None, "licence": "apache-2.0", "added_at": None,
                "weight_format": "Q4_K_M", "runtime_requirements": [],
                "runtime_notes": None, "android_supported": False, "ios_supported": False,
            },
        ]
        db_connection = FakeDbConnection(None)
        db_connection.handlers["catalogue_rows"] = models

        async def exercise():
            request = AssignRequest(
                hardware=HardwareProfile(total_ram_gb=4, available_ram_gb=2.5),
                platform="android",
                runtime_capabilities=["gguf:q2_0", "ternary:hadamard-activations"],
            )
            with mock.patch.object(catalogue_module.db, "pool", return_value=FakePool(db_connection)):
                assigned = await catalogue_module.assign(request)
                self.assertEqual(assigned.catalogue_id, "bonsai2-2b")
                self.assertEqual(assigned.weight_format, "Q2_0")
                self.assertEqual(assigned.runtime_requirements, models[0]["runtime_requirements"])
                unsupported = request.model_copy(update={"runtime_capabilities": ["gguf:ptq1_0"]})
                with self.assertRaises(HTTPException) as raised:
                    await catalogue_module.assign(unsupported)
                self.assertEqual(raised.exception.status_code, 422)

                ios_request = request.model_copy(update={"platform": "ios"})
                ios_assigned = await catalogue_module.assign(ios_request)
                self.assertEqual(ios_assigned.catalogue_id, "bonsai2-2b")

        self.run_async(exercise())

    def test_registration_reconnect_and_deregistration_keep_node_scope(self):
        node_id = uuid4()
        catalogue_row = {
            "id": "bonsai2-4b", "source": "gguf:https://models.example/model-Q2_0.gguf",
            "weight_format": "Q2_0", "runtime_requirements": [
                "gguf:q2_0", "ternary:hadamard-activations",
            ], "android_supported": True,
        }
        registered_row = {
            "id": node_id, "name": "phone", "operator": None, "model_name": "bonsai2-4b",
            "region": None, "cost_per_1k": 0, "avg_latency_ms": 0, "healthy": False,
            "last_heartbeat": None, "last_seen_healthy": None, "capability_text": "Bonsai",
            "domain_tags": ["general"], "catalogue_id": "bonsai2-4b",
            "transport": "android_ws", "node_token": "original-node-token",
            "endpoint_url": None, "group_status": None,
            "group_members_total": None, "group_members_up": None,
        }
        db_connection = FakeDbConnection(None)
        db_connection.handlers["catalogue"] = lambda *_: catalogue_row
        db_connection.handlers["existing"] = lambda *_: {"node_token": "original-node-token"}
        db_connection.handlers["write"] = lambda *_: registered_row
        node = NodeCreate(
            name="phone", model_name="bonsai2-4b", catalogue_id="bonsai2-4b",
            capability_text="Bonsai", transport="android_ws",
            runtime_name="runtime", runtime_version="1", model_format="Q2_0",
            runtime_capabilities=["gguf:q2_0", "ternary:hadamard-activations"],
        )

        async def exercise():
            with mock.patch.object(registry_module.db, "pool", return_value=FakePool(db_connection)), \
                 mock.patch.object(registry_module.embedder, "embed", return_value=[0.0] * 384), \
                 mock.patch.object(registry_module.manager, "disconnect_node", new_callable=mock.AsyncMock) as disconnect:
                result = await registry_module.register_node(
                    node, "original-node-token", None,
                )
                self.assertEqual(result.node_token, "original-node-token")
                self.assertEqual(result.transport, "android_ws")
                self.assertFalse(hasattr(result, "endpoint_url"))
                disconnect.assert_awaited_with(node_id, "node re-registered")

                with self.assertRaises(HTTPException) as raised:
                    await registry_module.register_node(node, "wrong-token", None)
                self.assertEqual(raised.exception.status_code, 409)

            with mock.patch.object(registry_module.db, "pool", return_value=FakePool(db_connection)), \
                 mock.patch.object(registry_module.manager, "disconnect_node", new_callable=mock.AsyncMock) as disconnect:
                await registry_module.delete_node(node_id, "original-node-token")
                disconnect.assert_awaited_once_with(node_id)

        self.run_async(exercise())

    def test_reconnect_replaces_old_channel_and_expiry_marks_offline(self):
        async def exercise():
            manager = mobile_transport.MobileConnectionManager()
            first = self.make_connection(manager)
            second = self.make_connection(manager, first.node_id)
            await manager.attach(first)
            self.assertTrue(manager.is_live(first.node_id))
            await manager.attach(second)
            self.assertEqual(first.websocket.closed[0], 1001)
            self.assertTrue(manager.is_live(first.node_id))
            self.assertFalse(await manager.detach(first))
            second.last_seen = time.monotonic() - mobile_transport.HEARTBEAT_LEASE_SECONDS - 1
            self.assertEqual(await manager.expire_stale(), [first.node_id])
            self.assertFalse(manager.is_live(first.node_id))
            await manager.attach(self.make_connection(manager, first.node_id))
            await manager.disconnect_node(first.node_id, "explicit disconnect")
            self.assertFalse(manager.is_live(first.node_id))

        self.run_async(exercise())

    def test_health_never_marks_a_missing_mobile_lease_healthy(self):
        node_id = uuid4()
        db_connection = FakeDbConnection(None)
        db_connection.handlers["fetch"] = lambda query, *_: (
            [{"id": node_id}] if "transport in ('android_ws', 'ios_ws')" in query else []
        )

        class OfflineManager:
            async def expire_stale(self):
                return []

            def is_live(self, _node_id):
                return False

        async def exercise():
            with mock.patch.object(health_module.db, "pool", return_value=FakePool(db_connection)), \
                 mock.patch.object(health_module, "manager", OfflineManager()):
                await health_module.run_health_checks_once()
            mobile_updates = [
                call for call in db_connection.executed
                if "transport in ('android_ws', 'ios_ws')" in call[0]
            ]
            self.assertEqual(len(mobile_updates), 1)
            self.assertFalse(mobile_updates[0][1])

        self.run_async(exercise())

    def test_stream_correlation_backpressure_and_cancellation(self):
        async def exercise():
            manager = mobile_transport.MobileConnectionManager()
            first = self.make_connection(manager)
            second = self.make_connection(manager)
            await manager.attach(first)
            await manager.attach(second)
            first_task = asyncio.create_task(manager.forward(
                {"id": first.node_id}, {"messages": [{"role": "user", "content": "one"}],
                                        "max_tokens": 9999, "max_completion_tokens": 8888,
                                        "n": 99}, True))
            second_task = asyncio.create_task(manager.forward(
                {"id": second.node_id}, {"messages": [{"role": "user", "content": "two"}]}, True))
            async def wait_until_sent(connection, task):
                while not any(frame["type"] == "request.end" for frame in connection.websocket.sent):
                    if task.done():
                        await task
                    await asyncio.sleep(0)

            await asyncio.wait_for(asyncio.gather(
                wait_until_sent(first, first_task),
                wait_until_sent(second, second_task),
            ), timeout=1)
            first_id = first.websocket.sent[0]["request_id"]
            second_id = second.websocket.sent[0]["request_id"]
            self.assertNotEqual(first_id, second_id)
            self.assertEqual(first.websocket.sent[0]["type"], "request.start")
            self.assertEqual(first.websocket.sent[-1]["type"], "request.end")
            encoded_input = b"".join(
                base64.b64decode(frame["data"])
                for frame in first.websocket.sent if frame["type"] == "request.data"
            )
            outgoing_body = json.loads(encoded_input)
            self.assertEqual(outgoing_body["max_tokens"], mobile_transport.MAX_GENERATION_TOKENS)
            self.assertNotIn("max_completion_tokens", outgoing_body)
            self.assertEqual(outgoing_body["n"], 1)
            self.assertEqual(outgoing_body["model"], first.model_name)

            chunk_one = {"id": "one", "choices": [{"delta": {"content": "one"}}]}
            chunk_two = {"id": "two", "choices": [{"delta": {"content": "two"}}]}
            await manager.receive(second, json.dumps({
                "type": "response.chunk", "request_id": second_id, "sequence": 0, "chunk": chunk_two,
            }))
            await manager.receive(first, json.dumps({
                "type": "response.chunk", "request_id": second_id, "sequence": 0, "chunk": chunk_two,
            }))
            await manager.receive(first, json.dumps({
                "type": "response.chunk", "request_id": first_id, "sequence": 0, "chunk": chunk_one,
            }))
            await manager.receive(first, json.dumps({
                "type": "response.done", "request_id": first_id,
            }))
            await manager.receive(second, json.dumps({
                "type": "response.chunk", "request_id": second_id, "sequence": 1, "chunk": chunk_two,
            }))
            first_response, second_response = await asyncio.gather(first_task, second_task)
            self.assertIn(b'"id":"one"', b"".join([part async for part in first_response.aiter_raw()]))
            await second_response.aclose()
            self.assertIn({"type": "request.cancel", "request_id": second_id}, second.websocket.sent)

        self.run_async(exercise())

    def test_cancellation_before_first_chunk_and_full_queue_unblocks_receiver(self):
        async def exercise():
            manager = mobile_transport.MobileConnectionManager()
            connection = self.make_connection(manager)
            await manager.attach(connection)

            async def wait_for_request_end(count):
                async def wait():
                    while len([
                        frame for frame in connection.websocket.sent
                        if frame["type"] == "request.end"
                    ]) < count:
                        await asyncio.sleep(0)
                await asyncio.wait_for(wait(), timeout=1)

            aborted = asyncio.create_task(manager.forward(
                {"id": connection.node_id}, {"messages": []}, True))
            await wait_for_request_end(1)
            aborted_id = connection.websocket.sent[0]["request_id"]
            aborted.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await aborted
            self.assertEqual(connection.inflight, 0)
            self.assertIn(
                {"type": "request.cancel", "request_id": aborted_id},
                connection.websocket.sent,
            )

            response_task = asyncio.create_task(manager.forward(
                {"id": connection.node_id}, {"messages": []}, True))
            await wait_for_request_end(2)
            request_id = [
                frame for frame in connection.websocket.sent if frame["type"] == "request.start"
            ][1]["request_id"]
            for sequence in range(mobile_transport.RESPONSE_QUEUE_SIZE + 1):
                await manager.receive(connection, json.dumps({
                    "type": "response.chunk", "request_id": request_id,
                    "sequence": sequence, "chunk": {"choices": []},
                }))
                if sequence == 0:
                    response = await asyncio.wait_for(response_task, timeout=1)

            blocked_receive = asyncio.create_task(manager.receive(connection, json.dumps({
                "type": "response.chunk", "request_id": request_id,
                "sequence": mobile_transport.RESPONSE_QUEUE_SIZE + 1,
                "chunk": {"choices": []},
            })))
            await asyncio.sleep(0)
            self.assertFalse(blocked_receive.done())
            await response.aclose()
            await asyncio.wait_for(blocked_receive, timeout=1)
            self.assertIn(
                {"type": "request.cancel", "request_id": request_id},
                connection.websocket.sent,
            )

        self.run_async(exercise())

    def test_concurrency_limit_timeout_malformed_and_oversized_frames(self):
        original_timeout = settings.forward_read_timeout_seconds

        async def exercise():
            manager = mobile_transport.MobileConnectionManager()
            connection = self.make_connection(manager)
            await manager.attach(connection)
            first = asyncio.create_task(manager.forward(
                {"id": connection.node_id}, {"messages": []}, False))
            while not connection.websocket.sent:
                await asyncio.sleep(0)
            request_id = connection.websocket.sent[0]["request_id"]
            with self.assertRaises(httpx.ConnectError):
                await manager.forward({"id": connection.node_id}, {"messages": []}, False)
            result = {"choices": [], "content": "x" * 80_000}
            encoded_result = json.dumps(result, separators=(",", ":")).encode()
            await manager.receive(connection, json.dumps({
                "type": "response.result.start", "request_id": request_id,
                "body_bytes": len(encoded_result),
            }))
            for sequence, offset in enumerate(range(0, len(encoded_result), 8 * 1024)):
                await manager.receive(connection, json.dumps({
                    "type": "response.result.data", "request_id": request_id,
                    "sequence": sequence,
                    "data": base64.b64encode(encoded_result[offset:offset + 8 * 1024]).decode(),
                }))
            await manager.receive(connection, json.dumps({
                "type": "response.result.end", "request_id": request_id,
            }))
            response = await first
            self.assertEqual(response.json(), result)

            settings.forward_read_timeout_seconds = 0.01
            with self.assertRaises(httpx.ReadTimeout):
                await manager.forward({"id": connection.node_id}, {"messages": []}, False)
            self.assertTrue(any(frame["type"] == "request.cancel" for frame in connection.websocket.sent))
            with self.assertRaises(ValueError):
                await manager.receive(connection, "{")
            with self.assertRaises(ValueError):
                await manager.receive(connection, " " * (mobile_transport.MAX_MESSAGE_BYTES + 1))

        try:
            self.run_async(exercise())
        finally:
            settings.forward_read_timeout_seconds = original_timeout

    def test_ios_worker_routes_through_shared_mobile_broker(self):
        response = httpx.Response(200, json={"ios": True})

        async def mobile_forward(node, body, stream):
            self.assertEqual(node["transport"], "ios_ws")
            return response

        with mock.patch.object(upstream.manager, "forward", new=mobile_forward):
            actual = self.run_async(upstream.forward(
                {"transport": "ios_ws", "model_name": "bonsai2-4b"}, {"model": "auto"}, False))
        self.assertEqual(actual.json(), {"ios": True})

    def test_http_worker_forwarding_path_remains(self):
        class FakeClient:
            async def send(self, request, stream=False):
                self.request = request
                self.stream = stream
                return httpx.Response(200, json={"ok": True}, request=request)

            def build_request(self, *args, **kwargs):
                return httpx.Request(*args, **kwargs)

        client = FakeClient()
        with mock.patch.object(upstream.httpx, "AsyncClient", return_value=client):
            response = self.run_async(upstream.forward(
                {"endpoint_url": "https://node.example", "model_name": "desktop-model",
                 "worker_token": "worker-token"}, {"model": "auto"}, False))
        self.assertEqual(response.json(), {"ok": True})
        self.assertEqual(client.request.url.path, "/chat/completions")
        self.assertEqual(client.request.headers["authorization"], "Bearer worker-token")
        self.assertEqual(json.loads(client.request.content)["model"], "desktop-model")


if __name__ == "__main__":
    unittest.main()
