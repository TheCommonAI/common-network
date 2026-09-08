"""Security regressions with synthetic secrets and local HTTP servers only."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch, AsyncMock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'gateway'))
sys.path.insert(0, str(ROOT / 'join'))
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from app import network, upstream, ratelimit, gateway, privacy, admin, registry
from app.config import settings
from app.credentials import token_digest, token_matches
from app.limits import RequestLimits, validate_chat
import worker

TOKEN = 'fake-worker-credential-0123456789'
MODEL = 'test:1b'


def load_script(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Backend(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def log_message(self, *a): pass
    def do_GET(self):
        self.server.hits.append((self.path, dict(self.headers)))
        payload = json.dumps({'data': [{'id': MODEL}]} if self.server.ready else {'data': []}).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.hits.append((self.path, body))
        if self.server.delay:
            time.sleep(self.server.delay)
        if body.get('stream'):
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Transfer-Encoding', 'chunked')
            self.end_headers()
            frames = [b'data: {"choices":[{"delta":{"content":"one"}}]}\n\n', b'data: [DONE]\n\n']
            for frame in frames:
                self.wfile.write(f'{len(frame):X}\r\n'.encode()+frame+b'\r\n')
                self.wfile.flush()
                time.sleep(.15)
            self.wfile.write(b'0\r\n\r\n')
        else:
            payload = json.dumps({'choices': [{'message': {'content': 'synthetic reply'}}]}).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)


class SecurityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = ThreadingHTTPServer(('127.0.0.1', 0), Backend)
        cls.backend.hits, cls.backend.ready, cls.backend.delay = [], True, 0
        threading.Thread(target=cls.backend.serve_forever, daemon=True).start()
        cls.url = f'http://127.0.0.1:{cls.backend.server_port}'
        cls.worker = worker.serve(TOKEN, MODEL, port=0, ollama_url=cls.url)
        cls.worker_url = f'http://127.0.0.1:{cls.worker.server_port}'
        cls.scripts = [load_script('test_cli_'+str(i), path) for i,path in enumerate(
            ['common/common.py', 'chat/chat.py', 'join/join.py'])]

    @classmethod
    def tearDownClass(cls):
        for server in (cls.worker, cls.backend):
            server.shutdown()
            server.server_close()

    def setUp(self):
        self.original = settings.model_dump()
        settings.allow_loopback_node_endpoints = False
        settings.allowed_node_cidrs = ''
        settings.trusted_proxy_cidrs = ''
        settings.api_key_destinations = {}
        self.backend.hits.clear()
        self.backend.ready, self.backend.delay = True, 0
        ratelimit._buckets.clear()

    def tearDown(self):
        for key, value in self.original.items():
            setattr(settings, key, value)
        self.backend.delay = 0

    def test_endpoint_validation(self):
        for url in ['https://user:pass@host.test/v1', 'https://host.test/v1?x=y',
                    'https://host.test/a/../v1', 'https://host.test/%2e%2e/v1',
                    'https://169.254.169.254/v1', 'https://127.0.0.1/v1',
                    'https://[::ffff:127.0.0.1]/v1', 'http://8.8.8.8/v1',
                    'https://192.168.1.4/v1', 'https://0.0.0.0/v1']:
            with self.subTest(url=url), self.assertRaises(HTTPException):
                network.validate_endpoint_url(url)
        settings.allowed_node_cidrs = '192.168.1.0/24'
        network.validate_endpoint_url('http://192.168.1.4:11435/v1')
        with self.assertRaises(HTTPException):
            network.validate_endpoint_url('http://192.168.2.4:11435/v1')

    def test_pinned_connection_and_original_host(self):
        settings.allow_loopback_node_endpoints = True
        # DNS only returns loopback to the validator. If transport re-resolves
        # the original hostname, this .invalid URL cannot succeed.
        async def run():
            async with httpx.AsyncClient(trust_env=False) as client:
                with patch.object(network, '_resolved_addresses', return_value=[__import__('ipaddress').ip_address('127.0.0.1')]) as resolve:
                    req = await network.pinned_request(client, 'GET', f'http://test.invalid:{self.backend.server_port}/v1/models')
                    response = await client.send(req)
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(resolve.call_count, 1)
                    self.assertEqual(req.extensions['sni_hostname'], 'test.invalid')
        asyncio.run(run())
        self.assertEqual(self.backend.hits[-1][1]['Host'], f'test.invalid:{self.backend.server_port}')

    def test_key_is_bound_to_exact_destination(self):
        settings.api_key_destinations = {'DEMO_KEY': ['https://approved.example/v1']}
        with patch.dict(os.environ, {'DEMO_KEY': 'fake-key'}):
            self.assertEqual(upstream.resolve_api_key('DEMO_KEY', 'https://approved.example/v1/'), 'fake-key')
            for dest in ['https://attacker.example/v1', 'https://approved.example/other',
                         'http://approved.example/v1', 'https://approved.example:444/v1']:
                self.assertIsNone(upstream.resolve_api_key('DEMO_KEY', dest))

    def test_credential_redirects_blocked_in_each_client(self):
        target = self.url
        class Redirect(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                self.send_response(302)
                self.send_header('Location', target+'/sink')
                self.end_headers()
        server = ThreadingHTTPServer(('127.0.0.1', 0), Redirect)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            for module in self.scripts:
                req = module.urllib.request.Request(f'http://127.0.0.1:{server.server_port}/', data=b'{}', headers={'X-Common-Node-Token': 'synthetic-node-token'})
                with self.assertRaises(module.urllib.error.HTTPError):
                    module.safe_urlopen(req, timeout=2)
            self.assertEqual(self.backend.hits, [])
        finally:
            server.shutdown(); server.server_close()

    def test_no_automatic_update_and_no_execution(self):
        with patch.dict(os.environ, {'COMMON_ALLOW_UNVERIFIED_UPDATES': '0'}):
            for module in self.scripts:
                with patch.object(module, 'safe_urlopen', side_effect=AssertionError('network contacted')):
                    fn = getattr(module, 'self_update', None) or module.fetch_update
                    fn()
        self.assertFalse(self.scripts[0]._THESIS_EXEC)
        self.assertIsNone(self.scripts[0]._exec_check('print(1)', []))
        sys.path.insert(0, str(ROOT))
        from bench.sandbox import run_program
        self.assertFalse(run_program("raise RuntimeError('must not execute')")[0])

    def test_terminal_controls_are_removed(self):
        for module in self.scripts:
            text = module.safe_text('hello\x1b]52;c;FAKE\x07\x9b\u202e world\n')
            self.assertNotIn('\x1b', text)
            self.assertNotIn('\x07', text)
            self.assertNotIn('\u202e', text)
            self.assertTrue(text.endswith('\n'))

    def test_proxy_trust_and_verified_rate_identity(self):
        self.assertEqual(ratelimit.client_key({'x-forwarded-for': '1.2.3.4'}, '10.1.1.1'), '10.1.1.1')
        settings.trusted_proxy_cidrs = '10.0.0.0/8'
        self.assertEqual(ratelimit.client_key({'x-forwarded-for': '9.9.9.9, 1.2.3.4, 10.0.0.2'}, '10.1.1.1'), '1.2.3.4')
        settings.rate_limit_requests_per_minute = 1
        req = types.SimpleNamespace(state=types.SimpleNamespace(verified_token='fake-member'), headers={}, client=types.SimpleNamespace(host='1.2.3.4'))
        self.assertEqual(ratelimit.check_rate_limit(req, 1), 0)
        req.headers['x-forwarded-for'] = '8.8.8.8'
        req.client.host = '8.8.8.8'
        self.assertGreater(ratelimit.check_rate_limit(req, 1), 0)
        self.assertNotIn('fake-member', str(ratelimit._buckets))

    def test_hashes_and_legacy_credentials(self):
        digest = token_digest(TOKEN)
        self.assertTrue(token_matches(digest, TOKEN))
        self.assertTrue(token_matches(TOKEN, TOKEN))
        self.assertFalse(token_matches(digest, digest))
        self.assertFalse(token_matches(digest, TOKEN[:-1]))

    def test_worker_json_health_and_concurrency(self):
        headers = {'Authorization': 'Bearer '+TOKEN}
        with httpx.Client(trust_env=False) as client:
            for data in [[], None, {'messages': 'text'}, {'messages':[{'role':'user','content':[]}]}]:
                self.assertEqual(client.post(self.worker_url+'/v1/chat/completions', content=json.dumps(data), headers=headers).status_code, 400)
            self.backend.ready = False
            self.assertEqual(client.get(self.worker_url+'/v1/models', headers=headers).status_code, 503)
            self.backend.ready = True
            self.assertEqual(client.get(self.worker_url+'/v1/models', headers=headers).status_code, 200)
            self.worker.RequestHandlerClass.config.jobs.acquire()
            try:
                self.assertEqual(client.post(self.worker_url+'/v1/chat/completions', json={'messages':[{'role':'user','content':'synthetic'}]}, headers=headers).status_code, 429)
            finally:
                self.worker.RequestHandlerClass.config.jobs.release()

    def test_worker_streams_before_completion(self):
        with httpx.Client(trust_env=False) as client:
            with client.stream('POST', self.worker_url+'/v1/chat/completions', json={'stream':True,'messages':[{'role':'user','content':'synthetic'}]}, headers={'Authorization':'Bearer '+TOKEN}) as response:
                start = time.monotonic()
                first = next(response.iter_bytes())
                self.assertIn(b'one', first)
                self.assertLess(time.monotonic()-start, .25)

    def test_byte_limits_and_security_headers(self):
        app = FastAPI()
        app.add_middleware(RequestLimits)
        @app.post('/echo')
        async def echo(request: __import__('fastapi').Request):
            return await request.json()
        settings.max_request_bytes = 100
        with TestClient(app) as client:
            self.assertEqual(client.post('/echo', content=b'x'*101).status_code, 413)
            self.assertEqual(client.post('/echo', content=b'{}', headers={'content-encoding':'gzip'}).status_code, 415)
            ok = client.post('/echo', json={'a':'b'})
            self.assertEqual(ok.json(), {'a':'b'})
            self.assertIn('no-store', ok.headers['cache-control'])
            self.assertEqual(ok.headers['x-content-type-options'], 'nosniff')
        for body in [[], {'messages': []}, {'messages':[{'role':'user','content':'x'}], 'max_tokens':1000000}, {'messages':[{'role':'user','content':'x'}], 'temperature':float('nan')}]:
            with self.assertRaises(HTTPException): validate_chat(body)

    def test_retention_and_no_embeddings_in_decision(self):
        writes = []
        class Conn:
            async def execute(self, sql, *args): writes.append((sql,args))
        class Pool:
            def acquire(self): return self
            async def __aenter__(self): return Conn()
            async def __aexit__(self, *a): pass
        settings.retain_request_embeddings = False
        with patch.object(gateway.db, 'pool', return_value=Pool()):
            asyncio.run(gateway._record_decision([.123]*384, None, None, 10, True, compose_reason={'reason':'SYNTHETIC_PRIVATE_MARKER'}))
            self.assertIsNone(writes[0][1][0])
            self.assertIsNone(writes[0][1][10])
            self.assertNotIn('SYNTHETIC_PRIVATE_MARKER', str(writes))
            asyncio.run(privacy.purge_expired())
            self.assertTrue(any('delete from decisions' in sql for sql,args in writes))
            self.assertTrue(any('request_embed = null' in sql for sql,args in writes))

    def test_response_byte_limit(self):
        settings.max_response_bytes = 10
        async def run():
            response = httpx.Response(200, content=b'x'*11)
            with self.assertRaises(httpx.ReadError):
                await upstream.read_limited(response)
        asyncio.run(run())

    def test_gateway_chat_streaming_and_allowed_recipient(self):
        from app.router import ScoredNode
        from app.compose import PanelPlan
        from uuid import uuid4
        node = {'id': uuid4(), 'name': 'local-test', 'model_name': MODEL,
                'endpoint_url': self.worker_url+'/v1', 'worker_token': TOKEN,
                'api_key_ref': None, 'domain_tags': ['general']}
        score = ScoredNode(node=node, score=1, sim=1, cost_term=0, lat_term=0, region_term=0)
        class Conn:
            async def fetchrow(self, sql, *args):
                return {'present': 1} if args and args[0] == token_digest('fake-node') else None
            async def execute(self, *a): pass
        class Pool:
            def acquire(self): return self
            async def __aenter__(self): return Conn()
            async def __aexit__(self, *a): pass
        app = FastAPI()
        app.add_middleware(RequestLimits)
        app.include_router(gateway.router)
        settings.allow_loopback_node_endpoints = True
        settings.require_contribution = True
        headers = {'X-Common-Node-Token': 'fake-node'}
        with patch.object(gateway.db, 'pool', return_value=Pool()), \
             patch.object(gateway, '_fetch_healthy_nodes', AsyncMock(return_value=[node])), \
             patch.object(gateway.embedder, 'embed', return_value=[1.0]*384), \
             patch.object(gateway, 'score_nodes', return_value=[score]), \
             patch.object(gateway, 'best_matched_domain', return_value='general'), \
             patch.object(gateway.compose, 'plan_panel', return_value=PanelPlan(compose=False, reason='test')), \
             TestClient(app) as client:
            body = {'messages':[{'role':'user','content':'SYNTHETIC_ONLY'}]}
            self.assertEqual(client.post('/v1/chat/completions', json=body).status_code, 401)
            blocked = client.post('/v1/chat/completions', json=body, headers={**headers, 'X-Common-Allowed-Nodes': 'someone-else'})
            self.assertEqual(blocked.status_code, 503)
            self.assertEqual(self.backend.hits, [])
            response = client.post('/v1/chat/completions', json=body, headers=headers)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()['choices'][0]['message']['content'], 'synthetic reply')
            self.assertEqual(response.headers['x-common-node'], 'local-test')
            self.assertEqual(self.backend.hits[-1][1]['model'], MODEL)
            response = client.post('/v1/chat/completions', json={**body, 'stream':True}, headers=headers)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIn('data: [DONE]', response.text)
            self.assertEqual(response.headers['content-type'], 'text/event-stream; charset=utf-8')

    def test_identity_is_atomic_and_private(self):
        import tempfile
        module = self.scripts[2]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'common'/'identity.json'
            with patch.object(module, 'IDENTITY_PATH', path):
                module.write_identity('https://gateway.example', 'test', 'id', None, None, 'fake-node')
                self.assertEqual(json.loads(path.read_text())['node_token'], 'fake-node')
                if os.name != 'nt':
                    self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                    self.assertEqual(path.parent.stat().st_mode & 0o777, 0o700)
                self.assertEqual(len(list(path.parent.iterdir())), 1)

    def test_detail_endpoints_require_admin_but_login_page_does_not(self):
        from app.decisions import router
        app = FastAPI()
        app.include_router(router)
        app.include_router(admin.router)
        settings.admin_token = 'fake-admin'
        settings.public_decision_details = False
        with TestClient(app) as client:
            self.assertEqual(client.get('/admin').status_code, 200)
            self.assertEqual(client.get('/admin/state?token=fake-admin').status_code, 401)
            self.assertEqual(client.get('/decisions/recent').status_code, 401)
            self.assertEqual(client.get('/decisions/recent', headers={'X-Common-Node-Token':'fake-node'}).status_code, 401)

    @unittest.skipUnless(__import__('shutil').which('openssl'), 'openssl is required for local TLS fixture')
    def test_tls_pin_preserves_certificate_hostname_verification(self):
        import ssl
        import tempfile
        import subprocess
        import ipaddress
        settings.allow_loopback_node_endpoints = True
        with tempfile.TemporaryDirectory() as directory:
            key, cert = Path(directory)/'key.pem', Path(directory)/'cert.pem'
            subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
                            '-keyout', str(key), '-out', str(cert), '-days', '1',
                            '-subj', '/CN=test.invalid', '-addext', 'subjectAltName=DNS:test.invalid'],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            server = ThreadingHTTPServer(('127.0.0.1', 0), Backend)
            server.hits, server.ready, server.delay = [], True, 0
            tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            tls.load_cert_chain(cert, key)
            server.socket = tls.wrap_socket(server.socket, server_side=True)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            trust = ssl.create_default_context(cafile=str(cert))
            async def run():
                async with httpx.AsyncClient(verify=trust, trust_env=False) as client:
                    with patch.object(network, '_resolved_addresses', return_value=[ipaddress.ip_address('127.0.0.1')]):
                        req = await network.pinned_request(client, 'GET', f'https://test.invalid:{server.server_port}/v1/models')
                        self.assertEqual((await client.send(req)).status_code, 200)
                        wrong = await network.pinned_request(client, 'GET', f'https://wrong.invalid:{server.server_port}/v1/models')
                        with self.assertRaises(httpx.ConnectError):
                            await client.send(wrong)
            try:
                asyncio.run(run())
            finally:
                server.shutdown(); server.server_close()


if __name__ == '__main__':
    unittest.main()
