#!/usr/bin/env python3
"""Authenticated, bounded text-inference proxy for Common. AGPL-3.0.

No management API or content logging. This does not hide text from the PC's
administrator. Bind loopback for tunnels; explicitly bind a LAN interface for
trusted LAN use. LAN HTTP is not encrypted: use a VPN or TLS for that hop.
"""
from __future__ import annotations
import argparse
import json
import secrets
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

DEFAULT_OLLAMA = 'http://127.0.0.1:11434'
UPSTREAM_TIMEOUT = 120
MAX_BODY = 2_000_000
MAX_RESPONSE = 8_000_000
MAX_TOKENS = 8192
MAX_JOB_SECONDS = 300


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class BoundedServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, *args, **kwargs):
        self.slots = threading.BoundedSemaphore(16)
        super().__init__(*args, **kwargs)

    def process_request(self, request, address):
        if not self.slots.acquire(blocking=False):
            request.close()
            return
        try:
            super().process_request(request, address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.slots.release()

    def handle_error(self, request, client_address):
        # No traceback with attacker-supplied strings or inference content.
        pass


class WorkerConfig:
    def __init__(self, token, model, ollama_url=DEFAULT_OLLAMA, max_jobs=1):
        if not 16 <= len(token) <= 128 or not token.isascii():
            raise ValueError('worker token must be 16–128 ASCII characters')
        parsed = urlsplit(ollama_url)
        if parsed.hostname not in {'localhost', '127.0.0.1', '::1'} or parsed.scheme != 'http' or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ('', '/'):
            raise ValueError('Ollama must use an HTTP loopback endpoint')
        self.token, self.model = token, model
        self.ollama_url = ollama_url.rstrip('/')
        self.jobs = threading.BoundedSemaphore(max_jobs)


def validate_body(body, model):
    if not isinstance(body, dict):
        raise ValueError('body must be an object')
    requested = body.get('model')
    if requested and requested not in (model, 'auto'):
        raise ValueError('this worker serves only its configured model')
    messages = body.get('messages')
    if not isinstance(messages, list) or not 1 <= len(messages) <= 130:
        raise ValueError('messages must be a non-empty bounded list')
    for message in messages:
        if (not isinstance(message, dict) or message.get('role') not in {'user', 'assistant', 'system', 'developer', 'tool'}
                or not isinstance(message.get('content'), str)):
            raise ValueError('messages must contain a supported role and text')
    if 'stream' in body and type(body['stream']) is not bool:
        raise ValueError('stream must be boolean')
    allowed = {'model', 'messages', 'stream', 'stream_options', 'temperature', 'top_p',
               'seed', 'stop', 'max_tokens', 'max_completion_tokens', 'frequency_penalty',
               'presence_penalty', 'response_format', 'logprobs', 'top_logprobs'}
    if set(body) - allowed:
        raise ValueError('unsupported inference options')
    for field in ('max_tokens', 'max_completion_tokens'):
        value = body.get(field, MAX_TOKENS)
        if type(value) is not int or not 1 <= value <= MAX_TOKENS:
            raise ValueError('output token limit exceeded')
    body = dict(body)
    body['model'] = model
    if 'max_tokens' not in body and 'max_completion_tokens' not in body:
        body['max_tokens'] = MAX_TOKENS
    return body


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'common-worker'
    sys_version = ''

    def setup(self):
        super().setup()
        self.connection.settimeout(10)
        def expire():
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.request_timer = threading.Timer(15, expire)
        self.request_timer.daemon = True
        self.request_timer.start()

    def finish(self):
        self.request_timer.cancel()
        super().finish()

    def log_message(self, *args):
        pass

    def _json(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Connection', 'close')
        self.end_headers()
        self.close_connection = True
        try:
            self.wfile.write(data)
        except OSError:
            pass

    def _error(self, code, message):
        self._json(code, {'error': {'message': message, 'type': 'worker_error'}})

    def _authorised(self):
        values = self.headers.get_all('Authorization', [])
        if len(values) != 1:
            return False
        scheme, _, token = values[0].partition(' ')
        return scheme.lower() == 'bearer' and secrets.compare_digest(token.strip().encode(), self.config.token.encode())

    def _open(self, req, timeout=UPSTREAM_TIMEOUT):
        return urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect()).open(req, timeout=timeout)

    def do_GET(self):
        if self.path.rstrip('/') != '/v1/models':
            return self._error(404, 'not found')
        if not self._authorised():
            return self._error(401, 'worker credential required')
        # Query metadata, without loading the model. A live worker is not proof
        # that Ollama is alive or the model is still installed.
        try:
            with self._open(self.config.ollama_url + '/v1/models', 5) as upstream:
                raw = upstream.read(65537)
                if len(raw) > 65536:
                    raise ValueError()
                listing = json.loads(raw)
                names = {self.config.model}
                if ':' not in self.config.model.rsplit('/', 1)[-1]:
                    names.add(self.config.model + ':latest')
                if not any(isinstance(m, dict) and m.get('id') in names for m in listing.get('data', [])):
                    raise ValueError()
        except (OSError, ValueError, AttributeError):
            return self._error(503, 'configured model is not ready')
        self._json(200, {'object': 'list', 'data': [{'id': self.config.model, 'object': 'model', 'owned_by': 'common-network'}]})

    def do_POST(self):
        if self.path.rstrip('/') != '/v1/chat/completions':
            return self._error(404, 'not found')
        if not self._authorised():
            return self._error(401, 'worker credential required')
        lengths = self.headers.get_all('Content-Length', [])
        if len(lengths) != 1 or self.headers.get('Transfer-Encoding') or self.headers.get('Content-Encoding', 'identity') != 'identity':
            return self._error(413, 'one content length and an uncompressed body are required')
        try:
            length = int(lengths[0])
        except ValueError:
            length = 0
        if not 0 < length <= MAX_BODY:
            return self._error(413, 'body must be present and under 2MB')
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError()
            def invalid_constant(value):
                raise ValueError()
            body = validate_body(json.loads(raw, parse_constant=invalid_constant), self.config.model)
        except (ValueError, RecursionError, OSError):
            return self._error(400, 'invalid or unsupported text inference request')
        self.request_timer.cancel()
        if not self.config.jobs.acquire(blocking=False):
            return self._error(429, 'worker busy; retry later')
        try:
            self._proxy_chat(body)
        finally:
            self.config.jobs.release()

    def _proxy_chat(self, body):
        req = urllib.request.Request(self.config.ollama_url + '/v1/chat/completions',
            data=json.dumps(body, allow_nan=False).encode(), headers={'Content-Type': 'application/json'}, method='POST')
        try:
            upstream = self._open(req)
        except (OSError, ValueError):
            return self._error(502, 'local inference failed')
        # Closing the upstream on timeout/disconnect also stops reading its job.
        # Ollama cancellation remains best-effort, dependent on that backend.
        def abort_relay():
            # Socket shutdown interrupts a read even when makefile wrappers
            # still hold references. close() alone does not do that.
            sockets = [self.connection]
            try:
                sockets.append(upstream.fp.raw._sock)  # stdlib HTTPResponse/SocketIO
            except AttributeError:
                pass
            for sock in sockets:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        deadline = threading.Timer(MAX_JOB_SECONDS, abort_relay)
        deadline.daemon = True
        deadline.start()
        with upstream:
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream' if body.get('stream') else 'application/json')
            self.send_header('Transfer-Encoding', 'chunked')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Connection', 'close')
            self.end_headers()
            self.close_connection = True
            total = 0
            end = time.monotonic() + MAX_JOB_SECONDS
            try:
                while time.monotonic() < end:
                    chunk = upstream.read1(8192)
                    if not chunk:
                        self.wfile.write(b'0\r\n\r\n')
                        self.wfile.flush()
                        return
                    total += len(chunk)
                    if total > MAX_RESPONSE:
                        return
                    self.wfile.write(f'{len(chunk):X}\r\n'.encode() + chunk + b'\r\n')
                    self.wfile.flush()
            except OSError:
                return
            finally:
                deadline.cancel()


def serve(token, model, port=11435, ollama_url=DEFAULT_OLLAMA, host='127.0.0.1', max_jobs=1):
    if not 1 <= max_jobs <= 4:
        raise ValueError('max_jobs must be between 1 and 4')
    handler = type('BoundHandler', (Handler,), {'config': WorkerConfig(token, model, ollama_url, max_jobs)})
    server = BoundedServer((host, port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main():
    import getpass
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--port', type=int, default=11435)
    p.add_argument('--bind', default='127.0.0.1')
    p.add_argument('--ollama', default=DEFAULT_OLLAMA)
    p.add_argument('--max-jobs', type=int, default=1)
    args = p.parse_args()
    token = getpass.getpass('Worker token (hidden): ')
    serve(token, args.model, args.port, args.ollama, args.bind, args.max_jobs)
    print('Common worker running. Press Ctrl+C to stop.')
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
