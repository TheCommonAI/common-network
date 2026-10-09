-- No request content, tokens, IPs or device names in telemetry. Reports contain
-- only user-submitted text/contact plus an optional allowlisted diagnostic blob.
create table if not exists client_telemetry (
    id bigserial primary key,
    received_at timestamptz not null default now(),
    installation_id uuid not null,
    payload jsonb not null
);
create index if not exists client_telemetry_received on client_telemetry(received_at);
create table if not exists client_reports (
    id text primary key,
    received_at timestamptz not null default now(),
    payload jsonb not null
);
create index if not exists client_reports_received on client_reports(received_at);

-- A paused row keeps its identity but must never be revived by an in-flight
-- health response. Resume is an authenticated re-registration.
alter table nodes add column if not exists paused boolean not null default false;
