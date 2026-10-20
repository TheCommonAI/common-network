-- Distributed model groups: what the gateway shows about a group's state.
--
-- A distributed group is one registered node. Its coordinator registers under
-- the model's name through the normal path, so the gateway routes to it like any
-- other node, and it has no per-member rows. These columns are the only way the
-- gateway learns that the group is restarting or how many of its members are up.
-- See docs/v02-gateway-groups.md (section 1). There is no groups table in v1.
--
-- Written by the group's coordinator on POST /nodes, under the node token that
-- already guards the row. Display only: nothing routes on these values except
-- the one rule that a restarting group is never marked healthy (health.py) and
-- that a request with no other candidate gets a 503 (gateway.py). The coordinator
-- sends what it observes; the gateway does not verify it.
--
-- `ready` means the runtime is serving with all members up. `restarting` means a
-- member or the runtime is down and the coordinator is bringing it back in place.
-- There is no "degraded" state, because a partial group cannot serve.
--
-- All three are nullable, and null is the correct value for every single node,
-- including every row written before this migration.
alter table nodes add column if not exists group_status text
  check (group_status in ('ready', 'restarting'));

alter table nodes add column if not exists group_members_total integer
  check (group_members_total >= 0);

alter table nodes add column if not exists group_members_up integer
  check (group_members_up >= 0);
