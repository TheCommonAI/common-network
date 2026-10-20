-- A distributed model group, as the gateway sees it: one registered node whose
-- runtime is split across several machines.
--
-- No groups table. A group is still one node row, registered by its coordinator
-- under the model's own domain tag, and routed like any other node. These three
-- columns are the coordinator's own report of its state. They are display data
-- and the one restart flag in health.py; they never choose a node on their own.
-- A coordinator that does not report leaves them null, which is what a
-- single-machine node looks like.
--
-- group_status: 'ready' (serving, every member up) or 'restarting' (a member or
-- the runtime is down and being brought back). Null for a single-machine node.
-- Deliberately no 'degraded': a partial group cannot serve, so there is no third
-- state to show.
--
-- group_members_total / group_members_up: the member count the coordinator was
-- configured with, and how many answered at its last report. Self-reported, so
-- they are a claim, not a measurement.
--
-- group_reported_at: when the coordinator last sent group_status. A 'restarting'
-- claim is only honoured while it is fresh, so a coordinator that dies mid-restart
-- cannot leave its node unroutable forever.
--
-- All nullable. Rows that predate this migration are single-machine nodes.
alter table nodes add column if not exists group_status text
  check (group_status in ('ready', 'restarting'));
alter table nodes add column if not exists group_members_total integer;
alter table nodes add column if not exists group_members_up integer;
alter table nodes add column if not exists group_reported_at timestamptz;
