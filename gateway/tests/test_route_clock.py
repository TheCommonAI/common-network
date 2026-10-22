"""The routing clock: what "routing" is allowed to contain.

The dashboard's "Routing (median)" is `route_ms`, which the gateway means as its
own share of a request: access control, the node list, the embedding and the
score. Two things were wrong with it, and both are pinned here.

1. The embedding ran synchronously on the event loop. One slow encode stalled
   every request in flight, and every one of them then reported the stall as
   routing. Asserted by running a ticker across an encode that blocks: a
   blocking encode lets the ticker run at most once, an off-loop one lets it
   keep time.

2. When every panel member failed, the gateway fell back to single routing and
   sampled the route clock afterwards. So the whole failed panel attempt, which
   is member model loads and timeouts, was counted as routing. Asserted by
   making the panel take half a second and failing, then checking the
   X-Common-Route-Ms the client is handed.

No database, no embedding model, no live nodes: the pool and the upstream are
replaced, and `app.embedder` imports sentence-transformers lazily so nothing
downloads.
"""
import asyncio
import json
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import compose, embedder, gateway, metrics, ratelimit, upstream  # noqa: E402
from app.compose import PanelPlan  # noqa: E402
from app.config import settings  # noqa: E402
from app.router import ScoredNode  # noqa: E402

PANEL_SECONDS = 0.5


def node(name: str) -> dict:
    return {'id': name, 'name': name, 'model_name': 'm', 'domain_tags': ['general']}


def scored_node(n: dict, score: float) -> ScoredNode:
    return ScoredNode(node=n, score=score, sim=score, cost_term=0.0, lat_term=0.0,
                      region_term=0.0, tag_term=0.0)


class FakeRequest:
    def __init__(self, body: dict):
        self._body = body
        self.headers = {}
        self.client = None

    async def json(self):
        return self._body

    async def stream(self):
        yield json.dumps(self._body).encode()


class FakeResponse:
    """Enough of an httpx.Response for the non-streaming relay."""

    status_code = 200
    headers = {'content-type': 'application/json'}

    def __init__(self):
        self.extensions = {'_client': self}

    async def aread(self) -> bytes:
        return b'{"ok": true}'

    async def aclose(self) -> None:
        pass


async def no_op(*args, **kwargs):
    return None


class EmbedOffTheLoop(unittest.IsolatedAsyncioTestCase):
    async def test_a_blocking_encode_does_not_stall_the_loop(self):
        def blocking_encode(text):
            time.sleep(0.3)
            return [0.0]

        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        with patch.object(embedder, 'embed', blocking_encode):
            task = asyncio.create_task(ticker())
            vector = await embedder.embed_async('anything')
            task.cancel()

        self.assertEqual(vector, [0.0])
        # About 30 ticks in 0.3s if the loop was free. A blocked loop gets
        # one or two, so 10 separates the two with room to spare.
        self.assertGreater(ticks, 10)


class FailedPanelIsNotRouting(unittest.IsolatedAsyncioTestCase):
    async def test_a_failed_panel_is_not_counted_as_routing(self):
        nodes = [node('a'), node('b')]
        scored = [scored_node(nodes[0], 0.9), scored_node(nodes[1], 0.5)]

        async def fetch_nodes():
            return nodes

        async def fake_embed(text):
            return [0.1, 0.2, 0.3, 0.4]

        async def panel_that_fails(*args, **kwargs):
            # Members took this long to fail. Every one of them failed.
            await asyncio.sleep(PANEL_SECONDS)
            return None

        async def forward(candidate, body, stream):
            return FakeResponse()

        body = {'messages': [{'role': 'user', 'content': 'hello'}]}

        with patch.object(settings, 'require_contribution', False), \
                patch.object(ratelimit, 'check_rate_limit', lambda request: 0.0), \
                patch.object(gateway, '_fetch_healthy_nodes', fetch_nodes), \
                patch.object(embedder, 'embed_async', fake_embed), \
                patch.object(gateway, 'best_matched_domain', lambda *a, **k: None), \
                patch.object(gateway, 'score_nodes', lambda *a, **k: scored), \
                patch.object(compose, 'plan_panel',
                             lambda *a, **k: PanelPlan(compose=True, reason='test panel')), \
                patch.object(gateway, '_handle_composed', panel_that_fails), \
                patch.object(gateway, 'choose_primary', lambda s, t: (s[0], s[1])), \
                patch.object(upstream, 'forward', forward), \
                patch.object(gateway, '_record_decision', no_op), \
                patch.object(gateway, '_update_latency', no_op), \
                patch.object(metrics, 'record_request', lambda **k: None):
            response = await gateway.chat_completions(FakeRequest(body))

        # The fallback really happened: the answer came from single routing.
        self.assertEqual(response.headers['x-common-topology'], 'single')
        # The panel took half a second. The routing clock must not have seen it.
        route_ms = int(response.headers['x-common-route-ms'])
        self.assertLess(route_ms, PANEL_SECONDS * 1000 / 2)


if __name__ == '__main__':
    unittest.main()
