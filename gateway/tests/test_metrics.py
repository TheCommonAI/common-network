"""The timing log: what the gateway records, and what it refuses to believe.

Three things are worth a test here and they are not the usual ones.

The first is that the gateway's own write can never damage a request. Metrics
are a side effect of answering, and a network that fails to answer because it
could not write a latency row would have its priorities badly wrong.

The second is the parse boundary. `node_report` reads headers sent by a
machine that registered itself on a permissionless registry, so every value in
it is attacker-controlled in the ordinary case -- not because anyone is
attacking, but because nothing stops them. A parse that accepts "+5" or
"  12  " accepts a value that was never meant, and one that accepts a
two-billion-millisecond load lets a single liar move a p95.

The third is the null. A node that did not report a model load and a node that
reported a model load of zero are different facts, and the schema keeps them
apart on purpose. A summary that renders the first as 0 would tell every
reader that the whole fleet loads instantly.

No database and no network: the pool is a canned row and the point is what the
endpoint does with it.
"""
import asyncio
import contextlib
import io
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Nothing here touches an embedding model; don't make the suite download one.
if 'sentence_transformers' not in sys.modules:
    stub = types.ModuleType('sentence_transformers')
    stub.SentenceTransformer = object
    sys.modules['sentence_transformers'] = stub

import httpx
from fastapi import FastAPI
from app import metrics


# --- The pool, canned ------------------------------------------------------

class _Ctx:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *args):
        pass


class _Pool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        return _Ctx(self.conn)


# Every column the summary SQL selects. All null but the counts, so a test that
# wants a value sets exactly that value and nothing else can quietly supply one.
_SUMMARY_KEYS = [
    'requests', 'succeeded', 'failed', 'streaming', 'non_streaming',
    'reported', 'cold_count', 'warm_count',
    'ttft_p50', 'ttft_p95', 'ttft_min', 'ttft_max',
    'total_p50', 'total_p95', 'total_min', 'total_max',
    'route_p50', 'route_p95', 'route_min', 'route_max',
    'total_stream_p50', 'total_nonstream_p50',
    'load_p50', 'load_p95', 'cold_load_p50', 'cold_load_p95', 'warm_load_p50',
    'prompt_eval_p50', 'eval_p50', 'tps_p50', 'tps_p95',
]


def summary_row(**overrides):
    row = {k: None for k in _SUMMARY_KEYS}
    row.update(requests=0, succeeded=0, failed=0, streaming=0, non_streaming=0,
               reported=0, cold_count=0, warm_count=0)
    row.update(overrides)
    return row


_NODE_KEYS = ['node_name', 'model_name', 'requests', 'succeeded', 'reported',
              'cold_count', 'warm_count', 'ttft_p50', 'ttft_p95', 'total_p50',
              'route_p50', 'load_p50', 'tps_p50']


def node_row(**overrides):
    row = {k: None for k in _NODE_KEYS}
    row.update(requests=0, succeeded=0, reported=0, cold_count=0, warm_count=0)
    row.update(overrides)
    return row


class Conn:
    """Records the SQL it was handed and returns whatever the test staged."""

    def __init__(self):
        self.row = summary_row()
        self.rows = []
        self.recent = []
        self.queries = []

    async def fetchrow(self, sql, *args):
        self.queries.append((sql, list(args)))
        return self.row

    async def fetch(self, sql, *args):
        self.queries.append((sql, list(args)))
        return self.rows if 'group by' in sql else self.recent


# --- Header parsing: the untrusted boundary --------------------------------

class NodeReportTests(unittest.TestCase):
    def test_integer_header_rejects_anything_that_is_not_a_clean_count(self):
        bad = ['+5', '5_0', '5.0', '-1', '', '  ', 'abc', '0x10',
               '1e3', '5ms', '٥', '9999999999999999']
        for value in bad:
            with self.subTest(value=repr(value)):
                self.assertIsNone(
                    metrics._int_header({'X': value}, 'X'),
                    f'{value!r} parsed; it should have been treated as absent')

    def test_padding_is_tolerated_because_the_http_layer_strips_it_first(self):
        # Not a loosening. Every HTTP parser removes optional whitespace around
        # a field value before this function sees it, so rejecting padding here
        # would reject well-formed input that can never actually arrive padded.
        self.assertEqual(metrics._int_header({'X': ' 5 '}, 'X'), 5)

    def test_integer_header_accepts_a_plain_count_and_drops_an_absurd_one(self):
        self.assertEqual(metrics._int_header({'X': '1998'}, 'X'), 1998)
        self.assertEqual(metrics._int_header({'X': '0'}, 'X'), 0)
        self.assertIsNone(metrics._int_header({}, 'X'))
        # One node claiming a two-day load would move a p95 on its own, so the
        # ceiling is enforced at the boundary rather than trusted downstream.
        self.assertIsNone(metrics._int_header({'X': str(metrics._MAX_MS + 1)}, 'X'))
        self.assertIsNone(
            metrics._int_header({'X': str(metrics._MAX_TOKENS + 1)}, 'X',
                                maximum=metrics._MAX_TOKENS))

    def test_boolean_header_accepts_only_the_two_spellings(self):
        for value, expected in [('true', True), ('1', True), ('TRUE', True),
                                (' false ', False), ('0', False)]:
            with self.subTest(value=repr(value)):
                self.assertIs(metrics._bool_header({'X': value}, 'X'), expected)
        # Not `bool(raw)`: every non-empty string is truthy, so accepting these
        # would have recorded a node saying "no" as cold.
        for value in ['yes', 'no', 'on', '', '2', 'cold', 'FALSEY']:
            with self.subTest(value=repr(value)):
                self.assertIsNone(metrics._bool_header({'X': value}, 'X'))

    def test_a_node_that_sent_nothing_is_not_reported_on(self):
        shapes = set()
        for headers in (None, {}, {'X-Common-Node-Load-Ms': 'soon',
                                   'X-Common-Node-Cold': 'maybe'}):
            with self.subTest(headers=headers):
                out = metrics.node_report(headers)
                self.assertFalse(out['node_reported'])
                self.assertIsNone(out['load_ms'])
                self.assertIsNone(out['cold'])
                shapes.add(tuple(sorted(out)))
        # No headers and unusable headers must produce the same keys, so a
        # caller can index the result rather than .get it without the
        # no-headers path raising KeyError.
        self.assertEqual(len(shapes), 1, f'inconsistent result shapes: {shapes}')

    def test_a_node_that_reported_reads_back_whole(self):
        out = metrics.node_report({
            'X-Common-Node-Load-Ms': '1998',
            'X-Common-Node-Prompt-Eval-Ms': '88',
            'X-Common-Node-Eval-Ms': '1420',
            'X-Common-Node-Prompt-Tokens': '412',
            'X-Common-Node-Prompt-Cached-Tokens': '400',
            'X-Common-Node-Completion-Tokens': '44',
            'X-Common-Node-Cold': 'true',
        })
        self.assertTrue(out['node_reported'])
        self.assertEqual(out['load_ms'], 1998)
        self.assertEqual(out['prompt_eval_ms'], 88)
        self.assertEqual(out['eval_ms'], 1420)
        self.assertEqual(out['prompt_tokens'], 412)
        self.assertEqual(out['prompt_cached_tokens'], 400)
        self.assertEqual(out['completion_tokens'], 44)
        self.assertIs(out['cold'], True)

    def test_one_good_header_among_garbage_still_counts_as_reported(self):
        out = metrics.node_report({'X-Common-Node-Load-Ms': '12',
                                   'X-Common-Node-Eval-Ms': 'nonsense'})
        self.assertTrue(out['node_reported'])
        self.assertEqual(out['load_ms'], 12)
        self.assertIsNone(out['eval_ms'])


# --- The write path -------------------------------------------------------

class WritePathTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        metrics._pending.clear()

    async def test_a_failed_write_never_reaches_the_request(self):
        def unavailable():
            raise RuntimeError('database unavailable')

        captured = io.StringIO()
        with patch.object(metrics.db, 'pool', unavailable):
            # Must return immediately rather than raise, and must not require
            # the caller to await anything.
            self.assertIsNone(metrics.record_request(node_name='n', total_ms=5))
            pending = list(metrics._pending)
            self.assertEqual(len(pending), 1, 'the write should have been scheduled')
            with contextlib.redirect_stdout(captured):
                results = await asyncio.gather(*pending, return_exceptions=True)

        self.assertEqual(results, [None],
                         'a failed metrics write must be swallowed, not raised')
        self.assertIn('could not record request metrics', captured.getvalue())

    async def test_the_write_survives_a_client_hanging_up(self):
        # The detached task is the whole reason a cancelled request still gets
        # recorded, so a write must complete after its requester is gone.
        class Recording(Conn):
            def __init__(self):
                super().__init__()
                self.executed = []

            async def execute(self, sql, *args):
                self.executed.append(args)
                return 'INSERT 0 1'

        conn = Recording()
        with patch.object(metrics.db, 'pool', return_value=_Pool(conn)):
            metrics.record_request(node_name='n', total_ms=5, ttft_ms=4)
            await asyncio.gather(*list(metrics._pending))
        self.assertEqual(len(conn.executed), 1)
        self.assertEqual(len(conn.executed[0]), 18, 'all columns are bound')


# --- The read path --------------------------------------------------------

class SummaryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.conn = Conn()
        self.pool = patch.object(metrics.db, 'pool', return_value=_Pool(self.conn))
        self.pool.start()
        app = FastAPI()
        app.include_router(metrics.router)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url='http://test')

    async def asyncTearDown(self):
        await self.client.aclose()
        self.pool.stop()

    async def test_window_and_topology_are_bound_never_interpolated(self):
        await self.client.get('/metrics/summary?window_hours=6&topology=panel')
        sql, args = next(q for q in self.conn.queries if 'percentile_cont' in q[0])
        self.assertIn('make_interval(hours => $1::int)', sql)
        self.assertIn('topology = $2', sql)
        # The value must not reach the statement text. `topology` is a query
        # parameter; interpolating it would be an injection.
        self.assertNotIn('panel', sql)
        self.assertEqual(args, [6, 'panel'])

    async def test_no_topology_filter_binds_only_the_window(self):
        await self.client.get('/metrics/summary?window_hours=1')
        sql, args = next(q for q in self.conn.queries if 'percentile_cont' in q[0])
        self.assertNotIn('topology =', sql)
        self.assertEqual(args, [1])

    async def test_the_window_is_validated(self):
        for hours in ('0', '-1', '8761', 'soon', ''):
            with self.subTest(hours=repr(hours)):
                r = await self.client.get(f'/metrics/summary?window_hours={hours}')
                self.assertEqual(r.status_code, 422)

    async def test_unreported_node_columns_stay_null_and_are_not_zeroed(self):
        # Today's reality: a window full of real requests, no node reporting.
        self.conn.row = summary_row(
            requests=12, succeeded=11, failed=1, streaming=9, non_streaming=3,
            ttft_p50=847.2, ttft_p95=3192.4, ttft_min=91, ttft_max=3192,
            total_p50=1204.6, route_p50=23.4,
            total_stream_p50=1204.6, total_nonstream_p50=847.0)
        self.conn.rows = [node_row(node_name='miles-macbook', model_name='llama3.2:1b',
                                   requests=12, succeeded=11, ttft_p50=847.2)]

        body = (await self.client.get('/metrics/summary')).json()

        self.assertEqual(body['node_reports']['coverage'], 0.0)
        self.assertEqual(body['node_reports']['reported'], 0)
        for key in ('p50', 'p95', 'cold_p50', 'cold_p95', 'warm_p50'):
            self.assertIsNone(body['node_reports']['model_load_ms'][key],
                              f'model_load_ms.{key} must be absent, not zero')
        self.assertIsNone(body['node_reports']['tokens_per_second']['p50'])
        self.assertIsNone(body['node_reports']['cold_fraction'])
        self.assertIsNone(body['by_node'][0]['load_p50_ms'])
        self.assertIsNone(body['by_node'][0]['tokens_per_second_p50'])
        # And the measured half is untouched by any of that.
        self.assertEqual(body['latency_ms']['ttft']['p50'], 847.2)
        self.assertEqual(body['latency_ms']['routing']['p50'], 23.4)

    async def test_coverage_reflects_how_much_of_the_traffic_a_node_reported_on(self):
        self.conn.row = summary_row(requests=12, reported=7, cold_count=4, warm_count=3,
                                    cold_load_p50=1998.0, tps_p50=31.7)
        body = (await self.client.get('/metrics/summary')).json()
        self.assertEqual(body['node_reports']['coverage'], 0.583)
        self.assertEqual(body['node_reports']['cold_fraction'], 0.571)
        self.assertEqual(body['node_reports']['model_load_ms']['cold_p50'], 1998.0)
        self.assertEqual(body['node_reports']['tokens_per_second']['p50'], 31.7)

    async def test_cold_fraction_is_null_rather_than_zero_when_nothing_was_reported(self):
        # A window with requests but no cold/warm flag at all. 0% would be a
        # claim; the honest answer is that nobody said.
        self.conn.row = summary_row(requests=4, cold_count=0, warm_count=0)
        body = (await self.client.get('/metrics/summary')).json()
        self.assertIsNone(body['node_reports']['cold_fraction'])

    async def test_an_empty_window_does_not_divide_by_zero(self):
        self.conn.row = summary_row()
        body = (await self.client.get('/metrics/summary')).json()
        self.assertEqual(body['totals']['requests'], 0)
        self.assertEqual(body['node_reports']['coverage'], 0.0)
        self.assertIsNone(body['latency_ms']['ttft']['p50'])

    async def test_every_summary_carries_the_provenance_caveat(self):
        # The dashboard renders this, so it must not be quietly dropped.
        self.conn.row = summary_row(requests=1, succeeded=1)
        body = (await self.client.get('/metrics/summary')).json()
        self.assertIn('unverified', body['caveat'])
        self.assertIn('strangers', body['caveat'])


class RecentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.conn = Conn()
        self.pool = patch.object(metrics.db, 'pool', return_value=_Pool(self.conn))
        self.pool.start()
        app = FastAPI()
        app.include_router(metrics.router)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url='http://test')

    async def asyncTearDown(self):
        await self.client.aclose()
        self.pool.stop()

    def _row(self, **overrides):
        from datetime import datetime, timezone
        row = dict(created_at=datetime.now(timezone.utc), node_name='n', model_name='m',
                   topology='single', stream=True, route_ms=20, ttft_ms=900,
                   total_ms=1500, node_reported=True, load_ms=None, prompt_eval_ms=None,
                   eval_ms=None, prompt_tokens=None, prompt_cached_tokens=None,
                   completion_tokens=None, cold=None, ok=True, error=None)
        row.update(overrides)
        return row

    async def test_tokens_per_second_is_derived_and_guarded(self):
        self.conn.recent = [
            self._row(eval_ms=1420, completion_tokens=44),
            # A node reporting a duration but no tokens, and one reporting
            # tokens but a zero duration: neither may divide by zero or invent
            # a throughput from half the inputs.
            self._row(eval_ms=1420, completion_tokens=None),
            self._row(eval_ms=0, completion_tokens=44),
            self._row(eval_ms=None, completion_tokens=44),
        ]
        body = (await self.client.get('/metrics/recent')).json()
        self.assertEqual(body[0]['tokens_per_second'], 31.0)
        self.assertIsNone(body[1]['tokens_per_second'])
        self.assertIsNone(body[2]['tokens_per_second'])
        self.assertIsNone(body[3]['tokens_per_second'])

    async def test_only_reported_filters_in_the_query(self):
        await self.client.get('/metrics/recent?limit=5&only_reported=true')
        sql, args = self.conn.queries[-1]
        self.assertIn('not $2::boolean or node_reported', sql)
        self.assertEqual(args, [5, True])

    async def test_an_unreported_row_keeps_its_measured_columns(self):
        self.conn.recent = [self._row(node_reported=False, ttft_ms=847, total_ms=900)]
        body = (await self.client.get('/metrics/recent')).json()
        self.assertFalse(body[0]['node_reported'])
        self.assertEqual(body[0]['ttft_ms'], 847)
        self.assertEqual(body[0]['total_ms'], 900)


if __name__ == '__main__':
    unittest.main()
