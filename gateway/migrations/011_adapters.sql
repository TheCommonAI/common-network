-- v0.1.1: on-demand specialists from LoRA adapters.
--
-- The catalogue answers "what specialists do we know about" -- a fixed list
-- somebody curated, and a node picks one off it at onboarding. This is the
-- other direction: given a cluster of demand no node covers, which small
-- public LoRA adapters sit nearest it, and which node on the matching base
-- could fuse them into a specialist that did not exist an hour ago.
--
-- The gateway never builds anything. It recommends; the node's owner runs
-- `common adapters build` and re-registers on the result. That is not a
-- limitation to work around, it is the security boundary -- see the note on
-- adapters_mode in app/config.py.
--
-- A separate table, not an extension of catalogue_models. The catalogue
-- seeder ends in `delete from catalogue_models where id <> all($1::text[])`,
-- and catalogue.seed.yaml calls itself the source of truth whose removed
-- entries are deleted on next boot. An adapter living there would be wiped
-- the first time somebody trimmed that YAML. Separate table, separate seed
-- file, same convention, no inherited wipe.

create table if not exists adapters (
  id                text primary key,          -- e.g. 'math-12k', 'code-r16v3'
  display_name      text not null,

  -- Where the adapter actually is. Two refs because they are two different
  -- artifacts: hf_repo is the PEFT directory (what the Stage 1 experiment
  -- blends, and what llama.cpp converts), gguf_ref is a pre-converted GGUF
  -- where somebody has already published one.
  --
  -- Ollama's ADAPTER directive takes either, but its safetensors adapter
  -- support is documented for the Llama, Mistral and Gemma families -- Qwen
  -- is not among them, and every adapter the experiment used is on Qwen2.5.
  -- So the reliable path is a GGUF, and conversion is an out-of-band operator
  -- step: the stdlib-only CLI can *download* a GGUF but cannot produce one.
  -- Nullable, because most published adapters have no GGUF and that is a fact
  -- about the ecosystem rather than a gap in the row.
  hf_repo           text not null,
  gguf_ref          text,

  -- The commit the adapter was verified at. Stage 1 pinned every revision to a
  -- SHA before measuring anything, for one reason: `main` moving would silently
  -- change the bytes a later run loads, and the whole justification for this
  -- feature is that a specific adapter was measured on a specific question set.
  -- A plan that named a mutable repo would let the operator build a different
  -- artifact from the one the experiment scored, which is the one thing this
  -- table cannot afford to be vague about.
  --
  -- Nullable: an adapter somebody adds by hand may have no pinned revision, and
  -- NULL means "unpinned", which the CLI reports rather than pretending.
  revision          text,

  -- NOT NULL, and deliberately stricter than nodes.base_model below. A LoRA
  -- is meaningless without the weights it was trained against: two adapters
  -- can only be averaged when they share r, lora_alpha and target modules,
  -- and that is a property of the base they were fitted on. An adapter row
  -- whose base is unknown cannot be recommended for anything, so it should
  -- not be storable.
  --
  -- The namespace is the Hugging Face repo id -- 'Qwen/Qwen2.5-1.5B-Instruct'
  -- -- because that is the identifier the adapter was fitted against and the
  -- one `nodes.base_model` is asked to report. It is NOT the same namespace as
  -- catalogue_models.base_model, whose convention is an Ollama pull-name
  -- ('ollama:llama3.1:8b': what to *fetch*, not what the weights are). The two
  -- are compared in app/adapters.py, which handles the difference by name
  -- rather than ignoring it -- see resolve_node_base.
  base_model        text not null,

  -- The profile text embedded into domain_embed. Routing compares a cluster's
  -- centroid against this, in numpy, exactly as everywhere else in the repo.
  domain_text       text not null,
  domain_embed      vector(384),

  -- What fusing costs the donor's machine. size_mb is the artifact; min_ram_gb
  -- is the headroom the fused model needs resident, which is the number that
  -- actually matters -- a blend is a distinct Ollama model with its own
  -- residency slot, so a node running a blend and its base keeps both loaded.
  size_mb           integer,
  min_ram_gb        integer,

  -- Which adapters may be averaged *together*.
  --
  -- A label, not the three numbers behind it, and that is the point. Averaging
  -- requires r, lora_alpha and target modules to match on every adapter in the
  -- set -- `W + (alpha/r)(sum(lambda_i B_i) @ sum(lambda_i A_i))` is only a
  -- LoRA if the B's and A's are the same shape and the alpha/r scaling is the
  -- same number for all of them. Re-deriving that at selection time from three
  -- columns means the rule lives in two places and can disagree with itself;
  -- one shared string cannot.
  --
  -- It matters now, not hypothetically: six of the seven known adapters for
  -- Qwen2.5-1.5B-Instruct are r=16/alpha=32/7 targets and can be averaged,
  -- while the medical one is r=64/alpha=16/4 targets and cannot join any of
  -- them. The value records the class, e.g. 'qwen2.5-1.5b-r16-a32-7t'.
  --
  -- NULL means "blends with nothing", which is how a single-only adapter is
  -- recorded. An adapter with a NULL group can still be recommended -- as one
  -- adapter, fused alone -- it just can never be a member of a multi-adapter
  -- set. Selection reads this rather than inferring compatibility.
  blend_group       text,

  domain_tags       text[],
  licence           text,
  added_at          timestamptz default now()
);

-- Which adapters a node is currently running, and the base they were fused
-- onto.
--
-- Text[] and not a join table with FKs, following 006's precedent for
-- decisions.panel: a node that deregisters, or an adapter that is later
-- withdrawn from the seed file, must not take the historical record with it.
-- Resolve at read time and tolerate misses.
--
-- base_model is load-bearing rather than descriptive. While a node is running
-- a blend its model_name is the blend tag, not a real model -- so base_model
-- is the only thing that survives the swap and says what the node can still
-- be fused from. NULL means "unknown", and the rule is refuse to assign,
-- never guess: most current seed entries predate this column and the demo
-- nodes have no catalogue_id to derive it from either.
--
-- A node's self-reported base_model is a hint; nodes.catalogue_id ->
-- catalogue_models.base_model (added in 006) is preferred where it exists,
-- because that one is ours.
alter table nodes add column if not exists base_model text;
alter table nodes add column if not exists adapter_ids text[];

-- "Has this adapter set already been built on a node on this base?" is the
-- query the one-blend-per-node rule reads, and it is a containment test on
-- the array.
create index if not exists idx_nodes_adapter_ids on nodes using gin (adapter_ids);
