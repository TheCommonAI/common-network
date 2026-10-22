-- iOS workers use the same outbound WebSocket protocol and lease model as
-- Android and iOS workers, with separate catalogue eligibility.
alter table catalogue_models add column if not exists ios_supported boolean not null default false;

alter table nodes drop constraint if exists nodes_transport_check;
alter table nodes add constraint nodes_transport_check
  check (transport in ('http', 'android_ws', 'ios_ws'));
