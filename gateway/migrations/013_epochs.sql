-- Release epochs: tag request metrics by release version for before/after comparison.
--
-- An epoch is a named period, typically tied to a release label like "v0.2.8".
-- Starting a new epoch does NOT delete old data — it just changes which label
-- new requests get tagged with, so all historical metrics remain queryable.
--
-- The `current_epoch()` function returns the most recently started epoch's label,
-- defaulting to 'baseline' if none exists. This is what `request_metrics.epoch`
-- gets populated with on insert.

-- The epochs table: one row per release/comparison point.
create table if not exists epochs (
  id           serial primary key,
  label        text not null unique,
  started_at   timestamptz not null default now(),
  notes        text
);

-- Seed the baseline epoch for any existing data.
-- This runs idempotently: if 'baseline' already exists, nothing happens.
insert into epochs (label, notes)
values ('baseline', 'Initial epoch — all requests before epoch tracking was added')
on conflict (label) do nothing;

-- Add epoch column to request_metrics.
-- Default is 'baseline' so existing rows get a sensible value and new rows
-- before the gateway is updated still have something.
alter table request_metrics
  add column if not exists epoch text not null default 'baseline';

-- Index for efficient per-epoch queries.
create index if not exists idx_request_metrics_epoch
  on request_metrics (epoch, created_at desc);

-- Function to get the current (most recently started) epoch label.
-- Used by the gateway when recording new requests.
create or replace function current_epoch() returns text as $$
  select coalesce(
    (select label from epochs order by started_at desc limit 1),
    'baseline'
  );
$$ language sql stable;
