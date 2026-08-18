-- v0.1.1: composition.
--
-- v0.1 was one request -> one node. This migration records the other case:
-- one request -> a panel of specialists answering in parallel -> an aggregator
-- synthesising one answer.
--
-- Everything here is additive and nullable. A v0.1 row is a valid v0.1.1 row
-- with topology='single', so the existing decisions log stays readable and
-- the two topologies remain directly comparable in the same table -- which is
-- the point, since whether composition actually beats single-routing is the
-- open question this version exists to answer.

-- 'single'   -- v0.1 behaviour, one node answered.
-- 'panel'    -- >=2 specialists in parallel, aggregator synthesised.
-- 'degraded' -- a panel was selected but only one member returned, so its
--               answer was passed through unaggregated. Recorded distinctly
--               because averaging it into 'panel' would quietly credit
--               composition for answers nothing was composed into.
alter table decisions add column if not exists topology text not null default 'single';

-- Panel members, in the order they were selected. Deliberately a plain uuid[]
-- rather than a join table with FKs: a node that deregisters must not take the
-- historical record of what it answered with it. Resolve names at read time
-- and tolerate misses.
alter table decisions add column if not exists panel uuid[];
alter table decisions add column if not exists aggregator_node uuid references nodes(id) on delete set null;

-- Why this request was composed (or wasn't) -- the domains that cleared the
-- floor, the domination check, the per-member scores. Free-form jsonb so the
-- explanation can get richer without another migration.
alter table decisions add column if not exists compose_reason jsonb;

-- Deterministic verification outcome for this request (see app/verify.py).
-- checks_run / checks_failed are the fire rate: the seam findings measured
-- overrides firing on 42% of handoffs, and a rate far from that is a signal
-- the extractor is broken rather than the models being unusually correct.
alter table decisions add column if not exists checks_run integer;
alter table decisions add column if not exists checks_failed integer;
alter table decisions add column if not exists disagreements integer;

create index if not exists idx_decisions_topology on decisions (topology, created_at desc);

-- A node may declare itself willing to aggregate. Aggregation is a different
-- job from answering in a lane -- it needs breadth and instruction-following,
-- not depth in one domain -- so it is a separate flag rather than another
-- domain tag, which would otherwise pull the node into topical routing.
alter table nodes add column if not exists can_aggregate boolean not null default false;

-- Overlay specialists.
--
-- v0.1's catalogue had exactly two kinds of entry: `ollama:*` (pull a model,
-- run it) and `api:*` (a hosted service, which common-join explicitly refused
-- to install). That made CGLA-Legal unreachable for a contributor -- it was
-- listed as `api:cgla-legal` and the joiner rejected it, so the network's only
-- verified-in-lane specialist could never actually be contributed by anyone.
--
-- It was never really an API. CGLA is a causal graph in SQLite plus a
-- *swappable* LLM that already runs against local Ollama; the hosted deployment
-- is one packaging of it, not the thing itself. So a third kind of entry:
--
--   source:     'graph:cgla-legal'      -- what this specialist is
--   base_model: 'ollama:llama3.1:8b'    -- the weights it drives
--   overlay:    {repo, entrypoint, ...} -- the reasoning layer over them
--
-- This matters beyond one model: it is how a specialist can be *better than its
-- own weights*. The overlay is where the domain knowledge lives, and it is
-- small, auditable, and contributable in a way fine-tuned weights are not.
alter table catalogue_models add column if not exists base_model text;
alter table catalogue_models add column if not exists overlay jsonb;

-- Which catalogue entries a node should NOT be assigned because the network
-- already has enough of them. Not a column -- computed in app/catalogue.py --
-- but the index that makes the coverage query cheap belongs here.
create index if not exists idx_nodes_domain_tags on nodes using gin (domain_tags);
