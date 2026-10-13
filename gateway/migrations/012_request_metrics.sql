-- What a request actually cost, in the four places the time goes.
--
-- `decisions` already carries a `latency_ms`, and it is not enough. Under
-- composition it holds one row per panel member, so averaging it answers a
-- question about nodes rather than about requests; and it is a single number
-- for something that is at least three -- gateway routing, the node getting its
-- model into memory, and the node generating tokens. Those have different
-- owners. Only one of them is fixable by the person reading the dashboard.
--
-- So this is a second table rather than more columns on `decisions`, and it is
-- one row per user-facing request whatever the topology. `decisions` stays the
-- routing log; this is the timing log. They are deliberately not joined: a
-- panel request writes one decision per member plus one for the aggregator, so
-- there is no single decision row to point at, and a foreign key that is
-- sometimes null and sometimes arbitrary is worse than no key.
--
-- --- What the gateway can measure on its own -----------------------------
--
-- `ttft_ms` is a genuine time-to-first-token *for a streaming request*, and null
-- otherwise. Not a proxy for one, where it exists at all. Measured against
-- Ollama (0.34.0) with a cold 1.3GB model: 1998ms to response headers, 1998ms to
-- the first content delta, 3192ms to the last byte. Headers and first token
-- land together, because Ollama does not send response headers until the model
-- is resident and generation has begun. A warm request on the same model was
-- 91ms. So for a stream the wait for the first byte *is* the wait for the first
-- token, and it carries the model load inside it -- 22x the warm figure in that
-- measurement, which is the effect a donor feels and the reason decomposing it
-- matters.
--
-- Non-streaming is the case that has no first token to have waited for, and it
-- must be null rather than approximated. Measured the same way, warm, 44 tokens
-- generated: headers at 847ms and the full body at 847ms -- identical, because
-- Ollama buffers a non-streaming response and sends headers with the whole
-- thing. There is no moment at which "the first token arrived" is observable to
-- anyone, so the column is left null. Filling it with the time to headers would
-- put whole generations into a distribution of first tokens and quietly inflate
-- every TTFT figure the network publishes.
--
-- `total_ms` is measured to the last byte the gateway relays, not to the last
-- byte it receives from the node. The number worth showing a user is how long
-- until they have the whole answer.
--
-- What the gateway cannot see is *inside* ttft_ms. `ttft_ms - route_ms` is
-- model load plus prompt evaluation, and nothing measured at the gateway can
-- split those two: the node's first byte is the only event, and it happens
-- after both. Hence the node-reported block below.
--
-- --- What the node reports, and what it is worth -------------------------
--
-- Every node_ column is self-reported by a machine that registered itself on a
-- permissionless registry. None of it is verified, and a node that wanted to
-- look fast could simply lie. It is kept because these are the only honest
-- numbers available for model load and throughput -- Ollama exposes
-- load_duration, prompt_eval_duration, eval_duration and eval_count on its
-- native /api/chat and on nothing else; the OpenAI-compatible endpoint the
-- worker proxies today returns none of them (verified: the compat response has
-- `usage` with token counts and no durations at all). Read them as claims, and
-- say so wherever they are shown.
--
-- Nullable throughout, and null means "this node did not report", which today
-- is every node. A dashboard that renders null as zero would claim every node
-- loads instantly, so the column is nullable rather than defaulted and the
-- callers must handle the gap.
create table if not exists request_metrics (
  id                   uuid primary key default gen_random_uuid(),
  created_at           timestamptz not null default now(),

  -- Attribution, denormalised on purpose. A node that has since deregistered
  -- leaves a gap in the routing log rather than erasing the record that it
  -- answered; the same holds for timings. A metrics row whose node_name is
  -- null is still a real measurement of a real request.
  node_id              uuid references nodes(id) on delete set null,
  node_name            text,
  model_name           text,
  -- 'single' | 'panel' | 'degraded', matching decisions.topology.
  topology             text not null default 'single',
  -- Whether the *client* asked for a stream. Kept because ttft_ms means
  -- something slightly different in each case -- see the note in gateway.py --
  -- and a latency distribution that mixes the two is not one distribution.
  stream               boolean not null default false,

  -- --- measured by the gateway -------------------------------------------
  -- The gateway's own share: access control, the node list, the embedding and
  -- the score. Tens of milliseconds. This is the number to watch if routing
  -- ever feels slow, and the one number here that the gateway is fully
  -- responsible for.
  route_ms             integer,
  -- To the first token. Streaming requests only; null for a non-streaming
  -- request, which has no observable first token (see above). Where set, it is
  -- routing + model load + prompt evaluation, which is why it cannot be
  -- decomposed without the node's own numbers below.
  ttft_ms              integer,
  -- To the last byte relayed to the client.
  total_ms             integer,

  -- --- reported by the node (unverified; see above) ----------------------
  -- True when the node sent any of the block below, so a reader can tell
  -- "reported zero" from "did not report" without testing five columns.
  node_reported        boolean not null default false,
  -- Ollama's load_duration: getting weights into memory. The number that
  -- explains a cold node, and the one a donor can act on.
  load_ms              integer,
  -- Ollama's prompt_eval_duration: reading the prompt. Distinct from load and
  -- from generation, and the term that grows with context size.
  prompt_eval_ms       integer,
  -- Ollama's eval_duration: generating the output. eval_ms and
  -- completion_tokens together give tokens/second.
  eval_ms              integer,
  prompt_tokens        integer,
  -- prompt_eval_cached_count. Worth its own column on a network that re-sends
  -- conversation history: a high value means the node's prefix cache absorbed
  -- the prompt and ttft_ms is small for a reason that has nothing to do with
  -- the model being fast.
  prompt_cached_tokens integer,
  completion_tokens    integer,
  -- The node's own claim that this request paid a model load. Derivable from
  -- load_ms > 0, but kept separately because a node can know it is cold
  -- without being able to time the load exactly.
  cold                 boolean,

  ok                   boolean not null default true,
  -- Short, and never request content. Enough to tell a timeout from a refused
  -- connection in a distribution of failures.
  error                text
);

-- The dashboard reads a window of recent rows, newest first, and the
-- aggregates group by node within a window. Both are served by this one index.
create index if not exists idx_request_metrics_created_at
  on request_metrics (created_at desc);

create index if not exists idx_request_metrics_node_created
  on request_metrics (node_id, created_at desc);

-- Partial index over the rows that actually carry a node report. Every row is
-- null there today, so the index is nearly empty and costs nothing until the
-- worker starts filling it in -- at which point the tok/s and model-load
-- aggregates stop scanning rows that have nothing to contribute.
create index if not exists idx_request_metrics_reported
  on request_metrics (created_at desc)
  where node_reported;
