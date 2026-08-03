-- The catalogue seed file is the source of truth for what the network will
-- auto-install (see seed_catalogue_from_file), so removing an entry from the
-- YAML must actually delete the row. nodes.catalogue_id referenced
-- catalogue_models(id) with the default NO ACTION, which made that delete
-- fail for any model a node had ever provisioned -- exactly the models most
-- likely to be retired.
--
-- A node outliving its catalogue entry is normal and fine: it keeps serving
-- the model it already pulled, it just no longer points at a catalogue row.
-- Same reasoning as 002 for decisions/nodes.

alter table nodes drop constraint if exists nodes_catalogue_id_fkey;
alter table nodes add constraint nodes_catalogue_id_fkey
  foreign key (catalogue_id) references catalogue_models(id) on delete set null;
