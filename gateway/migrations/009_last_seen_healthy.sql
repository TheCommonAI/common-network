-- When the gateway last saw a node actually answer.
--
-- `last_heartbeat` does not record this and never did. The health loop stamps
-- it `now()` on every pass regardless of the result, so it means "when we last
-- asked", and every row -- a node that answered a second ago, a school laptop
-- that was lent for one afternoon in July -- carries the same timestamp within
-- a second of each other, forever. Anything downstream trying to tell a
-- machine that is coming back from one that has left had nothing to read.
--
-- This column is stamped only by a check that passed. `healthy` alone can't
-- substitute: it is a single bit with no age, so it cannot separate a laptop
-- that shut its lid two minutes ago from a contributor who stopped in July.
--
-- Nullable, and null is load-bearing: it means this gateway has never seen the
-- node answer. For a row registered before this migration that never came back
-- healthy, that is the literal truth and the right thing for a peer list to
-- act on. Currently-healthy rows are backfilled from `last_heartbeat`, which
-- for a node passing checks right now is accurate to one health-check interval.
alter table nodes add column if not exists last_seen_healthy timestamptz;

update nodes
   set last_seen_healthy = last_heartbeat
 where healthy = true
   and last_seen_healthy is null;
