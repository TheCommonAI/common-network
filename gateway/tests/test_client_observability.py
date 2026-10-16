"""Exercise the actual intake routes; replace only the DB and upstream probe."""
import asyncio
import json
import os
import sys
import types
import unittest
from pathlib import Path
from uuid import uuid4
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Intake never uses an embedding model. Avoid installing/downloading one here.
if 'sentence_transformers' not in sys.modules:
    stub = types.ModuleType('sentence_transformers')
    stub.SentenceTransformer = object
    sys.modules['sentence_transformers'] = stub

import httpx
from fastapi import FastAPI
from app import client_observability as obs
from app.config import settings


class Context:
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        pass


class Connection(Context):
    def __init__(self):
        self.writes = []
        self.row = None
        self.fail = False
    def transaction(self):
        return Context()
    async def execute(self, sql, *args):
        if self.fail:
            raise RuntimeError('Database unavailable')
        self.writes.append((sql, args))
        return 'UPDATE 0' if sql.startswith('update nodes set healthy=false') and args[1] != 'owner' else 'INSERT 0 1'
    async def fetchrow(self, sql, *args):
        return self.row
    async def fetch(self, sql, *args):
        return []
    def acquire(self):
        return self


class IntakeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        obs._buckets.clear()
        self.conn = Connection()
        self.patch = patch.object(obs.db, 'pool', return_value=self.conn)
        self.patch.start()
        self.env_patch = patch.dict(os.environ, {"ALL_PROXY": "", "HTTPS_PROXY": "", "HTTP_PROXY": ""})
        self.env_patch.start()
        app = FastAPI()
        app.include_router(obs.router)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url='http://test')
    async def asyncTearDown(self):
        await self.client.aclose()
        self.patch.stop()
        self.env_patch.stop()
    async def test_telemetry_allowlist_and_durable_ack(self):
        body = {'installationId': str(uuid4()), 'events': [{'event': 'chat_completed', 'prompt': 'SECRET', 'response': 'SECRET', 'model': 'alice/private', 'completionTokens': 3}]}
        r = await self.client.post('/client/telemetry', json=body)
        self.assertEqual(r.json(), {'accepted': True})
        payload = self.conn.writes[0][1][1]
        self.assertNotIn('SECRET', payload)
        self.assertNotIn('alice', payload)
        self.assertEqual(json.loads(payload)[0]['completionTokens'], 3)
        self.conn.fail = True
        r = await self.client.post('/client/telemetry', json=body)
        self.assertEqual(r.status_code, 500)
    async def test_reports_ack_id_and_keep_only_requested_diagnostics(self):
        body = {'id': 'COMMON-20260921-1842-A7F31234', 'mode': 'feedback', 'category': 'general', 'text': 'Explicit user feedback', 'rating': 5, 'diagnostics': {'prompt': 'SECRET'}}
        r = await self.client.post('/client/reports', json=body)
        self.assertEqual(r.json(), {'accepted': True, 'id': body['id']})
        payload = json.loads(self.conn.writes[0][1][1])
        self.assertIsNone(payload['diagnostics'])
        body['mode'] = 'problem'
        body['diagnostics'] = {'worker': {'status': 'unreachable', 'worker_token': 'SECRET'}}
        await self.client.post('/client/reports', json=body)
        self.assertNotIn('SECRET', str(self.conn.writes))
    async def test_invalid_input_and_size_limits(self):
        for body in ({}, {'installationId': 'bad', 'events': []}, {'installationId': str(uuid4()), 'events': [{'event': 'prompt'}]}):
            self.assertEqual((await self.client.post('/client/telemetry', json=body)).status_code, 400)
        self.assertEqual((await self.client.post('/client/reports', content='x' * 140000)).status_code, 413)
        self.assertEqual(self.conn.writes, [])
    async def test_admin_data_is_not_public(self):
        original = settings.admin_token
        try:
            settings.admin_token = ''
            self.assertEqual((await self.client.get('/admin/client-reports')).status_code, 404)
            settings.admin_token = 'private'
            self.assertEqual((await self.client.get('/admin/client-telemetry')).status_code, 401)
            self.assertEqual((await self.client.get('/admin/client-reports', headers={'X-Common-Admin-Token': 'private'})).status_code, 200)
        finally:
            settings.admin_token = original
    async def test_node_probe_requires_ownership_and_uses_worker_credentials(self):
        node_id = str(uuid4())
        self.conn.row = {'id': node_id, 'node_token': 'owner', 'worker_token': 'worker', 'endpoint_url': 'https://example.com/v1', 'api_key_ref': None}
        with patch.object(obs, '_check_one', new=AsyncMock(return_value=True)) as probe:
            r = await self.client.post('/nodes/' + node_id + '/health', headers={'X-Common-Node-Token': 'wrong'})
            self.assertEqual(r.status_code, 404)
            probe.assert_not_called()
            r = await self.client.post('/nodes/' + node_id + '/health', headers={'X-Common-Node-Token': 'owner'})
            self.assertEqual(r.json(), {'healthy': True})
            self.assertEqual(probe.call_args.args[1]['worker_token'], 'worker')
            self.assertNotIn('worker', r.text)
    async def test_intake_is_rate_limited(self):
        for _ in range(20):
            await self.client.post('/client/telemetry', json={})
        self.assertEqual((await self.client.post('/client/telemetry', json={})).status_code, 429)
    def test_nested_sanitiser_bounds_and_secret_fields(self):
        self.assertEqual(obs.sanitise({'events': [{'event': 'job_failed', 'code': 'INFERENCE_FAILED', 'environment': {'PASSWORD': 'SECRET'}, 'filename': '/home/alice/file'}]}), {'events': [{'event': 'job_failed', 'code': 'INFERENCE_FAILED'}]})

    async def test_worker_health_rejects_malformed_or_wrong_model_responses(self):
        from app.health import _check_one
        node = {'endpoint_url': 'https://fixture.example/v1', 'worker_token': 'secret-worker', 'api_key_ref': None, 'model_name': 'fixture'}
        for body, expected in [('not json', False), ('{"data":[]}', False), ('{"data":[{"id":"wrong"}]}', False), ('{"data":[{"id":"fixture"}]}', True)]:
            def handler(request):
                self.assertEqual(request.headers['authorization'], 'Bearer secret-worker')
                return httpx.Response(200, text=body)
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                with patch('app.health.validate_endpoint_url'):
                    self.assertEqual(await _check_one(client, node), expected)

    async def test_pause_preserves_identity_but_requires_ownership(self):
        url = '/nodes/' + str(uuid4()) + '/pause'
        self.assertEqual((await self.client.post(url, headers={'X-Common-Node-Token': 'wrong'})).status_code, 404)
        self.assertEqual((await self.client.post(url, headers={'X-Common-Node-Token': 'owner'})).json(), {'paused': True})
        self.assertTrue(all('delete' not in sql for sql, _ in self.conn.writes))

    # --- reporting a bad response -------------------------------------------
    #
    # The exchange is the one field on this route that is user-attached text
    # rather than automatically-gathered diagnostics, and the tests below are
    # mostly about the difference. The load-bearing one is
    # test_exchange_survives_verbatim: `sanitise()` is a strict allowlist that
    # contains neither 'exchange', 'question' nor 'answer', so anyone who
    # "tidied up" by running it through there would get {} back and delete the
    # whole feature while every other test still passed.

    def _problem_report(self, **overrides):
        body = {'id': 'COMMON-20261017-1432-A1B2C3D4', 'mode': 'problem',
                'category': 'chat', 'text': 'that answer was made up'}
        body.update(overrides)
        return body

    async def test_exchange_survives_verbatim(self):
        exchange = {
            'question': 'What is the capital of South Australia?',
            'answer': 'The capital of South Australia is Sydney, founded in 1836.',
            'answer_truncated': False,
            'routing': {'node': 'au-adelaide-laptop', 'topology': 'single', 'candidates': 3,
                        'score': 0.5741, 'decision_id': str(uuid4())},
        }
        r = await self.client.post('/client/reports', json=self._problem_report(exchange=exchange))
        self.assertEqual(r.status_code, 200)
        stored = json.loads(self.conn.writes[0][1][1])['exchange']
        # Byte-for-byte, not "contains something". A sanitiser that emptied it
        # or a truncation that quietly shortened it both fail here.
        self.assertEqual(stored['question'], exchange['question'])
        self.assertEqual(stored['answer'], exchange['answer'])
        self.assertEqual(stored['routing']['node'], 'au-adelaide-laptop')
        self.assertEqual(stored['routing']['decision_id'], exchange['routing']['decision_id'])
        self.assertEqual(stored['routing']['candidates'], 3)

    async def test_exchange_is_coupled_to_problem_mode(self):
        r = await self.client.post('/client/reports', json=self._problem_report(
            mode='feedback', exchange={'question': 'q', 'answer': 'a'}))
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(json.loads(self.conn.writes[0][1][1])['exchange'])

    async def test_an_empty_answer_is_a_reportable_failure(self):
        r = await self.client.post('/client/reports', json=self._problem_report(
            exchange={'question': 'why did this return nothing?', 'answer': ''}))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(json.loads(self.conn.writes[0][1][1])['exchange']['answer'], '')

    async def test_exchange_bounds_and_types(self):
        cases = [
            {'exchange': 'not a dict'},
            {'exchange': {'question': '', 'answer': 'a'}},               # empty question
            {'exchange': {'question': 'q' * 4001, 'answer': 'a'}},       # question over 4000 chars
            {'exchange': {'question': 'q', 'answer': 'a' * 32769}},      # answer over 32KiB
            {'exchange': {'question': 'q', 'answer': 'a', 'answer_truncated': 'yes'}},
            {'exchange': {'question': 'q', 'answer': 'a', 'routing': 'single'}},
            {'exchange': {'question': 'q', 'answer': 'a', 'routing': {'decision_id': 'not-a-uuid'}}},
            {'exchange': {'question': 'q', 'answer': 'a', 'routing': {'panel': ['ok'] * 33}}},
            {'exchange': {'question': 'q', 'answer': 'a', 'routing': {'node': 'n' * 201}}},
        ]
        for body in cases:
            with self.subTest(body=str(body)[:60]):
                self.assertEqual((await self.client.post('/client/reports', json=self._problem_report(**body))).status_code, 400)
        self.assertEqual(self.conn.writes, [])
        # Multi-byte text is measured in bytes, so 11k emoji is over the cap
        # even though it is well under it in characters.
        over = {'exchange': {'question': 'q', 'answer': '\U0001f600' * 11000}}
        self.assertEqual((await self.client.post('/client/reports', json=self._problem_report(**over))).status_code, 400)

    async def test_routing_drops_unknown_keys_and_junk_numbers(self):
        exchange = {'question': 'q', 'answer': 'a', 'routing': {
            'node': 'au-adelaide', 'topology': 'single',
            'injected': 'arbitrary text the client made up',
            'score': float('nan'),
            'candidates': 10 ** 16,
            'compose_reason': {'compose': False, 'reason': 'no second domain', 'nested': {'deep': 1}},
        }}
        # Sent as raw content, not via `json=`: httpx encodes with
        # allow_nan=False and refuses to put NaN on the wire at all. A client
        # that does send it is the case being tested, so the bytes are built
        # here directly rather than through the helper.
        r = await self.client.post('/client/reports', content=json.dumps(
            self._problem_report(exchange=exchange)).encode(),
            headers={'Content-Type': 'application/json'})
        self.assertEqual(r.status_code, 200)
        routing = json.loads(self.conn.writes[0][1][1])['exchange']['routing']
        self.assertEqual(routing['node'], 'au-adelaide')
        self.assertNotIn('injected', routing)
        self.assertNotIn('score', routing)          # NaN is not a number we store
        self.assertNotIn('candidates', routing)     # nor is 1e16
        self.assertEqual(routing['compose_reason'], {'compose': False, 'reason': 'no second domain'})

    async def test_reason_is_a_closed_list(self):
        # `writes[0]` rather than `[-1]`: the route's second statement is the
        # 90-day prune, which takes no arguments, so the tail of the list is
        # not the insert.
        for reason in sorted(obs.REASONS):
            self.conn.writes.clear()
            r = await self.client.post('/client/reports', json=self._problem_report(reason=reason))
            self.assertEqual(r.status_code, 200)
            self.assertEqual(json.loads(self.conn.writes[0][1][1])['reason'], reason)
        for bad in ('hallucinated', 'WRONG', '', 3, ['wrong'], {'reason': 'wrong'}):
            self.conn.writes.clear()
            with self.subTest(bad=bad):
                self.assertEqual((await self.client.post('/client/reports', json=self._problem_report(reason=bad))).status_code, 400)
            self.assertEqual(self.conn.writes, [])

    async def test_absent_reason_is_null_not_missing(self):
        r = await self.client.post('/client/reports', json=self._problem_report())
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(json.loads(self.conn.writes[0][1][1])['reason'])

    def test_sanitise_would_have_deleted_an_exchange(self):
        """The reason _valid_exchange exists, asserted rather than remembered."""
        exchange = {'question': 'q', 'answer': 'a', 'routing': {'node': 'n'}}
        self.assertEqual(obs.sanitise(exchange), {})


if __name__ == '__main__':
    unittest.main()
