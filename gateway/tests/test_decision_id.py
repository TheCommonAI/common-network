"""The decision id: what turns a complaint into a diagnosis.

`_record_decision` used to be a bare `conn.execute` whose failure propagated.
It now uses `fetchrow` with `returning id` and hands that id back, so the
gateway can put it on the response as X-Common-Decision-Id and a client can
attach it to a report about that answer. An operator then joins the report to
the row and gets the embedding, the chosen and runner-up nodes, the score, the
topology, compose_reason and the verification counters -- none of which the
client had to send, because the network already kept it.

Two properties are worth pinning, and neither is obvious from reading it:

1. `execute` cannot return rows. Swapping it back -- or "simplifying" the call
   -- silently discards `returning id` and every report loses its
   cross-reference while everything still appears to work. Asserted by
   checking the call actually made.
2. It is best-effort. This runs *before* the response is built, so a raise here
   fails a request the node has already answered. A lost id is a lost
   cross-reference; a lost answer would be the feature breaking the product.

No database and no embedder: the pool is replaced, and `app.embedder` imports
sentence-transformers lazily so nothing downloads.
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import UUID, uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import gateway  # noqa: E402
from app.router import ScoredNode  # noqa: E402


class Context:
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        pass


class Conn(Context):
    """Records the call and returns one canned row."""

    def __init__(self):
        self.calls = []
        self.row = {'id': uuid4()}
        self.raises = None

    def acquire(self):
        return self

    async def fetchrow(self, sql, *args):
        self.calls.append((sql, args))
        if self.raises:
            raise self.raises
        return self.row

    async def execute(self, sql, *args):
        # If the implementation ever goes back to execute(), it lands here and
        # `calls` stays empty -- which is what the first test asserts against.
        self.calls.append(('EXECUTE ' + sql, args))
        return 'INSERT 0 1'


def scored(name: str, score: float) -> ScoredNode:
    return ScoredNode(node={'id': name, 'name': name, 'domain_tags': ['general']},
                      score=score, sim=score, cost_term=0.0, lat_term=0.0,
                      region_term=0.0, tag_term=0.0)


class DecisionIdTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.conn = Conn()
        self.patch = patch.object(gateway.db, 'pool', return_value=self.conn)
        self.patch.start()

    async def asyncTearDown(self):
        self.patch.stop()

    async def test_the_row_is_written_and_its_id_comes_back(self):
        chosen, runner = scored('au-adelaide', 0.61), scored('au-melbourne', 0.58)
        result = await gateway._record_decision(
            [0.1] * 8, chosen, runner, 412, True, 'maths', topology='single',
            compose_reason={'compose': False, 'reason': 'one domain'},
        )
        self.assertEqual(result, self.conn.row['id'])
        self.assertIsInstance(result, UUID)
        (sql, args), = self.conn.calls
        # The whole point: fetchrow, and an insert that asks for the id back.
        self.assertNotIn('EXECUTE', sql)
        self.assertIn('returning id', sql.lower())
        self.assertEqual(args[1], 'au-adelaide')
        self.assertEqual(args[3], 'au-melbourne')
        self.assertEqual(args[4], 412)
        self.assertIs(args[5], True)
        self.assertEqual(args[6], 'maths')
        self.assertEqual(args[7], 'single')

    async def test_a_failed_write_loses_the_id_and_nothing_else(self):
        self.conn.raises = RuntimeError('Database unavailable')
        with patch('builtins.print') as printed:
            result = await gateway._record_decision([0.1], scored('n', 0.5), None, 10, True)
        self.assertIsNone(result)
        # Reported, not swallowed in silence: a fleet where every decision
        # insert is failing should say so somewhere.
        self.assertIn('could not record this decision', printed.call_args.args[0])

    async def test_a_write_that_returns_nothing_is_not_an_error(self):
        self.conn.row = None
        self.assertIsNone(await gateway._record_decision([0.1], None, None, 10, False))

    async def test_a_failed_request_is_still_recorded_as_a_failure(self):
        self.conn.row = None
        await gateway._record_decision([0.1], None, None, 9000, False, 'maths')
        (sql, args), = self.conn.calls
        self.assertIn('insert into decisions', sql)
        self.assertIs(args[5], False)

    def test_the_header_is_absent_rather_than_empty(self):
        # An empty string would read as an id that failed to parse. Absent
        # means "there is nothing to join against", which is the truth.
        self.assertEqual(gateway._decision_header(None), {})
        self.assertEqual(gateway._decision_header(''), {})
        decision = uuid4()
        self.assertEqual(gateway._decision_header(decision),
                         {'X-Common-Decision-Id': str(decision)})

    def test_the_header_name_matches_what_the_client_reads(self):
        # Pinned as a literal because the CLI reads this exact spelling in
        # common/common.py; renaming it here would silently stop matching.
        self.assertEqual(list(gateway._decision_header(uuid4())), ['X-Common-Decision-Id'])


if __name__ == '__main__':
    unittest.main()
