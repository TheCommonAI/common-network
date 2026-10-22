-- Outbound-only mobile workers use an authenticated gateway WebSocket rather
-- than a public endpoint. Keep the old HTTP transport as the default.
alter table nodes add column if not exists transport text not null default 'http';
alter table nodes alter column endpoint_url drop not null;
alter table nodes add column if not exists runtime_name text;
alter table nodes add column if not exists runtime_version text;
alter table nodes add column if not exists model_format text;
alter table nodes add column if not exists runtime_capabilities text[] not null default '{}';

alter table catalogue_models add column if not exists weight_format text;
alter table catalogue_models add column if not exists runtime_requirements text[] not null default '{}';
alter table catalogue_models add column if not exists runtime_notes text;
alter table catalogue_models add column if not exists android_supported boolean not null default false;

do $$
begin
  if not exists (
    select 1 from pg_constraint
    where conname = 'nodes_transport_check' and conrelid = 'nodes'::regclass
  ) then
    alter table nodes add constraint nodes_transport_check
      check (transport in ('http', 'android_ws'));
  end if;
end
$$;
