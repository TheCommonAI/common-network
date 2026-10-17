-- The client string a node was registered under, copied onto each timing row.
--
-- `nodes.client` says which program a node was registered by, but it is the
-- node's *current* value and `request_metrics` is a history. A node that
-- upgrades from common-cli/0.1.5 to 0.1.6 must keep its 0.1.5 timings under
-- 0.1.5, so the value is snapshotted at request time rather than joined in
-- later. Same reasoning as node_name above: denormalised on purpose.
--
-- Nullable. Null means the node did not say, or the row predates this
-- migration -- kept distinct from any invented label.
alter table request_metrics add column if not exists client text;

-- No index on this column, deliberately. The by-version query filters on
-- created_at, which idx_request_metrics_created_at already serves, and a
-- plain CREATE INDEX here would hold a write lock on the live timing table
-- for the length of the build.
