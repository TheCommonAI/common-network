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
from datetime import datetime, timezone
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
    # The phase columns. The endpoint indexes the row for each of these by
    # name, so one missing here raises KeyError and 500s every test in
    # SummaryTests at once -- which is exactly what happened when they were
    # added to the SQL and not to this list.
    'phase_count',
    'phase_route_p50', 'phase_route_p95',
    'phase_ttft_p50', 'phase_ttft_p95',
    'phase_gen_p50', 'phase_gen_p95',
]


def summary_row(**overrides):
    row = {k: None for k in _SUMMARY_KEYS}
    row.update(requests=0, succeeded=0, failed=0, streaming=0, non_streaming=0,
               reported=0, cold_count=0, warm_count=0, phase_count=0)
    row.update(overrides)
    return row


def decomposed(which, route_ms, ttft_ms, total_ms):
    """One row of the median/p95-request query, with the sum already checked.

    `generation_ms` is derived here the same way the SQL derives it, so a test
    staging a row that does not add up is staging a row the query could not
    have produced.
    """
    return {'which': which, 'route_ms': route_ms, 'ttft_ms': ttft_ms,
            'total_ms': total_ms,
            'generation_ms': total_ms - ttft_ms - route_ms}


def no_decomposition(which):
    """What the query returns for a quantile when nothing was decomposable."""
    return {'which': which, 'route_ms': None, 'ttft_ms': None,
            'total_ms': None, 'generation_ms': None}


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
        self.phases = []
        self.timeseries = []
        self.distribution = []
        self.versions = []
        self.by_model = []
        self.queries = []

    async def fetchrow(self, sql, *args):
        self.queries.append((sql, list(args)))
        return self.row

    async def fetch(self, sql, *args):
        self.queries.append((sql, list(args)))
        # Dispatched on a marker unique to each statement rather than on
        # `'group by' in sql`. That shortcut was fine while there were two
        # queries; with five it silently hands the timeseries result to the
        # by-node reader and back, and the test passes for the wrong reason.
        if 'targets(which' in sql:
            return self.phases
        if 'width_bucket' in sql:
            return self.distribution
        if 'bucket_start' in sql:
            return self.timeseries
        # Checked before 'group by client', which is a prefix of it.
        if 'group by client, model_name' in sql:
            return self.by_model
        if 'group by client' in sql:
            return self.versions
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
        self.assertEqual(len(conn.executed[0]), 19, 'all columns are bound')

    async def test_the_client_is_snapshotted_onto_the_row(self):
        # The node's client string at request time, not a join to nodes later:
        # an upgraded node's older timings must stay under the older version.
        class Recording(Conn):
            def __init__(self):
                super().__init__()
                self.executed = []

            async def execute(self, sql, *args):
                self.executed.append((sql, args))
                return 'INSERT 0 1'

        conn = Recording()
        with patch.object(metrics.db, 'pool', return_value=_Pool(conn)):
            metrics.record_request(node_name='n', client='common-cli/0.1.5', total_ms=5)
            await asyncio.gather(*list(metrics._pending))
        sql, args = conn.executed[0]
        self.assertIn('client)', sql)
        self.assertEqual(args[-1], 'common-cli/0.1.5')


# --- The read path --------------------------------------------------------

class SummaryTests(unittest.IsolatedAsyncioTestCase):
    # The cache is off so each request reads the rows staged for it.
    cache_ttl = 0.0

    async def asyncSetUp(self):
        metrics._cache.clear()
        self.ttl = patch.object(metrics, '_CACHE_TTL_S', self.cache_ttl)
        self.ttl.start()
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
        self.ttl.stop()

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

    async def test_a_phase_bar_is_one_request_not_three_percentiles(self):
        """The regression guard for the whole decomposition.

        The wrong way to build the stacked bar is to stack the three per-phase
        percentiles. p50 of a sum is not the sum of the p50s, so those segments
        would add up to a duration no request ever had -- and on live data the
        gap was 12 seconds on a 50-second request. Staged so the two answers
        differ, and asserted on both: the independent sum is NOT the bar, and
        the row that is the bar really does add up.
        """
        self.conn.row = summary_row(
            requests=9, phase_count=9,
            phase_route_p50=1000.0, phase_ttft_p50=2000.0, phase_gen_p50=3000.0,
            total_p50=12000.0)
        self.conn.phases = [decomposed('p50', 100, 900, 12000),
                            decomposed('p95', 400, 5000, 40000)]

        phases = (await self.client.get('/metrics/summary')).json()['phases']

        independent = (phases['route_ms']['p50'] + phases['ttft_ms']['p50']
                       + phases['generation_ms']['p50'])
        self.assertEqual(independent, 6000.0)

        median = phases['median_request']
        self.assertEqual(median['total_ms'], 12000)          # the real median row
        self.assertNotEqual(independent, median['total_ms'])  # and NOT the sum
        self.assertTrue(median['additive'])
        self.assertEqual(median['route_ms'] + median['ttft_ms'] + median['generation_ms'],
                         median['total_ms'])
        self.assertEqual(median['generation_ms'], 11000)

        # Both quantiles come back, each an actual request.
        self.assertEqual(phases['p95_request']['total_ms'], 40000)
        self.assertEqual(phases['p95_request']['route_ms'], 400)
        self.assertTrue(phases['p95_request']['additive'])

    async def test_a_phase_count_is_reported_beside_the_phases(self):
        # Every per-phase percentile is over the decomposable subset only, so
        # the size of that subset has to travel with them or a decomposition of
        # 2 requests out of 900 reads as the whole window.
        self.conn.row = summary_row(requests=900, phase_count=2,
                                    phase_route_p50=50.0)
        body = (await self.client.get('/metrics/summary')).json()
        self.assertEqual(body['phases']['count'], 2)
        self.assertEqual(body['totals']['requests'], 900)

    async def test_a_window_with_nothing_streaming_has_no_decomposition(self):
        # No request streamed, so there is no first token to split on and the
        # phase query returns two all-null rows. Null, not zero: a zero-length
        # routing phase is a claim that routing was instant.
        self.conn.row = summary_row(requests=5, phase_count=0)
        self.conn.phases = [no_decomposition('p50'), no_decomposition('p95')]

        phases = (await self.client.get('/metrics/summary')).json()['phases']

        self.assertEqual(phases['count'], 0)
        self.assertIsNone(phases['median_request'])
        self.assertIsNone(phases['p95_request'])
        for name in ('route_ms', 'ttft_ms', 'generation_ms'):
            self.assertIsNone(phases[name]['p50'], f'{name}.p50 must be absent, not 0')

    async def test_a_window_with_no_rows_at_all_reports_no_phases(self):
        # The default canned row plus an empty phase result: the case a fresh
        # database and a narrow window both produce.
        body = (await self.client.get('/metrics/summary')).json()
        self.assertEqual(body['phases']['count'], 0)
        self.assertIsNone(body['phases']['median_request'])

    async def test_the_caveat_warns_that_the_phase_percentiles_do_not_stack(self):
        # The dashboard reads this text; a future reader of the JSON needs to be
        # told not to add the three numbers up.
        body = (await self.client.get('/metrics/summary')).json()
        self.assertIn('do NOT sum', body['caveat'])
        self.assertIn('median_request', body['caveat'])
        # And the ordering surprise between the two clocks is stated rather than
        # left to look like a bug.
        self.assertIn('different origins', body['caveat'])


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


def ts_row(**overrides):
    row = {'bucket_start': datetime.now(timezone.utc), 'requests': 0, 'succeeded': 0,
           'streaming': 0, 'total_p50': None, 'total_p95': None,
           'ttft_p50': None, 'route_p50': None}
    row.update(overrides)
    return row


class _ReadPathTests(unittest.IsolatedAsyncioTestCase):
    """Shared setup: the router over a canned pool, no database."""

    # Off by default, so a test that stages rows and asks again reads the new
    # rows. CacheTests turns it back on.
    cache_ttl = 0.0

    async def asyncSetUp(self):
        metrics._cache.clear()
        self.ttl = patch.object(metrics, '_CACHE_TTL_S', self.cache_ttl)
        self.ttl.start()
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
        self.ttl.stop()


class TimeseriesTests(_ReadPathTests):

    async def test_the_grid_is_counted_so_the_point_count_is_exact(self):
        """`generate_series(start, stop, step)` would emit buckets + 1.

        It is inclusive of its stop, and the window start is truncated to the
        minute, so the span reaches just past the end and one extra step fits.
        Verified against live Postgres: asking for 24 returned 25. Counting to
        `buckets - 1` cannot drift, so the statement is asserted to do that --
        the row count itself is a property of the SQL, not of this fake.
        """
        await self.client.get('/metrics/timeseries?window_hours=1&buckets=24')
        sql, _ = next(q for q in self.conn.queries if 'bucket_start' in q[0])
        self.assertIn('generate_series(0, $2::int - 1)', sql)
        self.assertIn('bucket_end', sql)

    async def test_an_empty_bucket_reports_zero_requests_not_one(self):
        """The count(*) trap, pinned.

        A bucket with no traffic still produces one all-null joined row, so
        `count(*)` reports 1 request where there were none -- on live data it
        reported 30 for a window holding 6. The statement must count a column of
        the joined table, and an empty bucket must read as zero with null
        percentiles rather than a percentile of nothing.
        """
        await self.client.get('/metrics/timeseries')
        sql, _ = next(q for q in self.conn.queries if 'bucket_start' in q[0])
        self.assertIn('count(m.id)', sql)
        self.assertNotIn('count(*)', sql)

        self.conn.timeseries = [ts_row(requests=0), ts_row(requests=4, total_p50=910.0)]
        body = (await self.client.get('/metrics/timeseries')).json()
        self.assertEqual(body['points'][0]['requests'], 0)
        self.assertIsNone(body['points'][0]['total_p50_ms'])
        self.assertEqual(body['points'][1]['requests'], 4)
        self.assertEqual(body['points'][1]['total_p50_ms'], 910.0)

    async def test_window_buckets_and_topology_are_all_bound(self):
        await self.client.get('/metrics/timeseries?window_hours=6&buckets=12&topology=panel')
        sql, args = next(q for q in self.conn.queries if 'bucket_start' in q[0])
        self.assertEqual(args, [6, 12, 'panel'])
        self.assertNotIn('panel', sql)
        self.assertIn('$3::text is null or m.topology = $3::text', sql)

    async def test_a_topology_value_can_never_reach_the_statement_as_syntax(self):
        # Bound and null-tested rather than interpolated, exactly as
        # /metrics/recent does it.
        nasty = "single'; drop table request_metrics; --"
        r = await self.client.get('/metrics/timeseries', params={'topology': nasty})
        self.assertEqual(r.status_code, 200)
        sql, args = next(q for q in self.conn.queries if 'bucket_start' in q[0])
        self.assertNotIn('drop table', sql)
        self.assertEqual(args[2], nasty)

    async def test_the_window_and_the_bucket_count_are_validated(self):
        for query in ('window_hours=0', 'window_hours=8761', 'buckets=0',
                      'buckets=201', 'window_hours=soon', 'buckets='):
            with self.subTest(query=query):
                r = await self.client.get(f'/metrics/timeseries?{query}')
                self.assertEqual(r.status_code, 422)

    async def test_an_empty_window_still_returns_every_bucket(self):
        self.conn.timeseries = [ts_row() for _ in range(24)]
        body = (await self.client.get('/metrics/timeseries?buckets=24')).json()
        self.assertEqual(len(body['points']), 24)
        self.assertTrue(all(p['requests'] == 0 for p in body['points']))
        self.assertTrue(all(p['total_p50_ms'] is None for p in body['points']))


class DistributionTests(_ReadPathTests):

    @staticmethod
    def _rows(bins, counts, hi=1000, total=None):
        if total is None:
            total = sum(counts)
        return [{'bin': i + 1, 'requests': c, 'min_ms': None, 'max_ms': None,
                 'hi': hi, 'total': total} for i, c in enumerate(counts)]

    async def test_one_point_per_bin_and_one_more_edge_than_bins(self):
        await self.client.get('/metrics/distribution?bins=6')
        sql, _ = next(q for q in self.conn.queries if 'width_bucket' in q[0])
        self.assertIn('generate_series(1, $2::int)', sql)

        self.conn.distribution = self._rows(6, [0, 1, 2, 0, 0, 3])
        body = (await self.client.get('/metrics/distribution?bins=6')).json()
        self.assertEqual(len(body['points']), 6)
        self.assertEqual(len(body['edge_ms']), 7)
        self.assertEqual(body['edge_ms'][0], 0)
        self.assertEqual(body['edge_ms'][-1], body['hi_ms'])
        self.assertEqual(body['total'], 6)

    async def test_bins_ascend_and_span_the_range(self):
        self.conn.distribution = self._rows(4, [1, 1, 1, 1], hi=1000)
        body = (await self.client.get('/metrics/distribution?bins=4')).json()
        edges = body['edge_ms']
        self.assertEqual(edges, sorted(edges))
        self.assertEqual(edges, [0, 250, 500, 750, 1000])

    async def test_a_handful_of_requests_is_flagged_sparse(self):
        # A histogram drawn from three requests looks identical to one drawn
        # from three thousand. The flag is what stops the dashboard presenting
        # the first as though it were the second.
        self.conn.distribution = self._rows(6, [0, 1, 1, 0, 1, 1])
        body = (await self.client.get('/metrics/distribution?bins=6')).json()
        self.assertTrue(body['sparse'])
        self.assertEqual(body['min_for_shape'], 20)

        self.conn.distribution = self._rows(6, [10, 10, 10, 10, 10, 10])
        body = (await self.client.get('/metrics/distribution?bins=6')).json()
        self.assertFalse(body['sparse'])

    async def test_an_empty_window_has_no_shape_to_draw(self):
        self.conn.distribution = self._rows(6, [0, 0, 0, 0, 0, 0], hi=1, total=0)
        body = (await self.client.get('/metrics/distribution?bins=6')).json()
        self.assertEqual(body['total'], 0)
        self.assertTrue(body['sparse'])
        self.assertTrue(all(p['requests'] == 0 for p in body['points']))

    async def test_the_bin_count_is_validated(self):
        for query in ('bins=0', 'bins=51', 'bins=soon', 'bins='):
            with self.subTest(query=query):
                r = await self.client.get(f'/metrics/distribution?{query}')
                self.assertEqual(r.status_code, 422)


# --- Speed by client version ----------------------------------------------

class ClientVersionTests(unittest.TestCase):
    def test_a_client_string_splits_into_program_and_version(self):
        self.assertEqual(metrics.parse_client('common-cli/0.1.6'), ('common-cli', '0.1.6'))
        self.assertEqual(metrics.parse_client('common-desktop/0.1.0'), ('common-desktop', '0.1.0'))

    def test_a_string_that_is_not_name_slash_version_is_kept_whole(self):
        # The string is self-reported. Anything that does not parse is shown as
        # it was sent, with no version invented for it.
        for raw in ('curl/8.4.0', 'common-cli', 'common-cli/latest', 'common-cli/', '/0.1.0'):
            with self.subTest(raw=raw):
                program, version = metrics.parse_client(raw)
                if version is None:
                    self.assertEqual(program, raw)

    def test_missing_client_is_none_not_a_label(self):
        self.assertEqual(metrics.parse_client(None), (None, None))
        self.assertEqual(metrics.parse_client(''), (None, None))

    def test_versions_compare_numerically(self):
        # Lexically, '0.1.10' sorts before '0.1.9'. The release after 0.1.9 must not.
        self.assertGreater(metrics.version_key('0.1.10'), metrics.version_key('0.1.9'))
        self.assertLess(metrics.version_key(None), metrics.version_key('0.0.1'))


class ByClientTests(_ReadPathTests):
    @staticmethod
    def _version(client, requests, total_p50, **extra):
        row = {'client': client, 'nodes': 1, 'requests': requests, 'succeeded': requests,
               'reported': 0, 'total_p50': total_p50, 'total_p95': total_p50 * 2,
               'ttft_p50': None, 'route_p50': 10.0, 'tps_p50': None}
        row.update(extra)
        return row

    async def test_newest_version_leads_within_a_program(self):
        self.conn.versions = [
            self._version('common-cli/0.1.9', 5, 900.0),
            self._version('common-cli/0.1.10', 5, 800.0),
            self._version('common-desktop/0.1.0', 5, 700.0),
        ]
        body = (await self.client.get('/metrics/by_client')).json()
        self.assertEqual([v['client'] for v in body['versions']], [
            'common-cli/0.1.10', 'common-cli/0.1.9', 'common-desktop/0.1.0'])

    async def test_a_version_without_a_client_string_sorts_after_the_known_ones(self):
        self.conn.versions = [self._version(None, 3, 500.0), self._version('common-cli/0.1.0', 3, 400.0)]
        body = (await self.client.get('/metrics/by_client')).json()
        self.assertEqual([v['client'] for v in body['versions']], ['common-cli/0.1.0', None])
        self.assertIsNone(body['versions'][1]['program'])

    async def test_each_version_carries_the_model_mix_behind_it(self):
        self.conn.versions = [self._version('common-cli/0.1.6', 10, 1200.0)]
        self.conn.by_model = [
            {'client': 'common-cli/0.1.6', 'model_name': 'bonsai2-4b', 'requests': 8,
             'total_p50': 1000.0, 'total_p95': 1900.0, 'ttft_p50': None, 'tps_p50': 31.5},
            {'client': 'common-cli/0.1.6', 'model_name': 'bonsai-8b', 'requests': 2,
             'total_p50': 2400.0, 'total_p95': 2500.0, 'ttft_p50': None, 'tps_p50': None},
        ]
        body = (await self.client.get('/metrics/by_client')).json()
        models = body['versions'][0]['models']
        self.assertEqual([m['model_name'] for m in models], ['bonsai2-4b', 'bonsai-8b'])
        self.assertEqual(models[0]['requests'], 8)
        self.assertEqual(models[0]['total_p95_ms'], 1900.0)
        self.assertIsNone(models[1]['tokens_per_second_p50'])

    async def test_a_version_with_no_reports_says_so_rather_than_zero(self):
        self.conn.versions = [self._version('common-cli/0.1.6', 4, 900.0, reported=0,
                                            tps_p50=None)]
        body = (await self.client.get('/metrics/by_client')).json()
        self.assertEqual(body['versions'][0]['reported'], 0)
        self.assertIsNone(body['versions'][0]['tokens_per_second_p50'])

    async def test_the_window_is_bound_as_a_parameter_not_interpolated(self):
        await self.client.get('/metrics/by_client?window_hours=168')
        sql, args = next(q for q in self.conn.queries if 'group by client' in q[0])
        self.assertIn('make_interval(hours => $1::int)', sql)
        self.assertEqual(args, [168])

    async def test_the_window_is_validated(self):
        for query in ('window_hours=0', 'window_hours=8761', 'window_hours=soon'):
            with self.subTest(query=query):
                r = await self.client.get(f'/metrics/by_client?{query}')
                self.assertEqual(r.status_code, 422)


class CacheTests(_ReadPathTests):
    cache_ttl = 15.0

    async def test_a_second_poll_inside_the_ttl_does_not_query_again(self):
        await self.client.get('/metrics/by_client?window_hours=24')
        await self.client.get('/metrics/by_client?window_hours=24')
        self.assertEqual(sum('group by client' in q[0] for q in self.conn.queries), 2,
                         'one version query and one model query, once, not twice')

    async def test_a_different_window_is_a_different_entry(self):
        await self.client.get('/metrics/by_client?window_hours=1')
        await self.client.get('/metrics/by_client?window_hours=24')
        self.assertEqual(sum('group by client' in q[0] for q in self.conn.queries), 4)

    async def test_an_expired_entry_is_recomputed(self):
        await self.client.get('/metrics/by_client?window_hours=24')
        key = ('by_client', 24)
        stamp, value = metrics._cache[key]
        metrics._cache[key] = (stamp - metrics._CACHE_TTL_S - 1, value)
        await self.client.get('/metrics/by_client?window_hours=24')
        self.assertEqual(sum('group by client' in q[0] for q in self.conn.queries), 4)

    async def test_the_summary_is_cached_as_well(self):
        await self.client.get('/metrics/summary?window_hours=24')
        await self.client.get('/metrics/summary?window_hours=24')
        self.assertEqual(sum('created_at' in q[0] and 'phase_count' in q[0]
                             for q in self.conn.queries), 1)


if __name__ == '__main__':
    unittest.main()
