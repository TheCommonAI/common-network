from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql://localhost/common_network"

    # Bounds on every database wait. asyncpg has no defaults here, and an
    # unbounded await on a dead socket is a silent permanent hang rather than
    # an error -- see the note in app/db.py, which this gateway hit in
    # production. Generous enough never to fire in normal operation.
    db_connect_timeout_seconds: float = 20.0
    db_command_timeout_seconds: float = 30.0

    embed_model_name: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384

    # Routing weights. Must not be relied on to sum to 1 exactly; kept simple and tunable.
    w_sim: float = 0.7
    w_cost: float = 0.15
    w_lat: float = 0.15
    region_bonus: float = 0.05

    health_check_interval_seconds: int = 30
    health_check_timeout_seconds: float = 5.0
    forward_timeout_seconds: float = 60.0

    seed_file: str = "nodes.seed.yaml"
    seed_on_startup: bool = True

    # Local dev runs uvicorn from gateway/, where catalogue/ is a sibling
    # directory (../catalogue). The Docker image copies catalogue/ in as a
    # child of /app instead -- Railway sets CATALOGUE_SEED_FILE=catalogue/...
    # to override this default for deployment.
    catalogue_seed_file: str = "../catalogue/catalogue.seed.yaml"
    catalogue_seed_on_startup: bool = True

    # Routing refinement (v0.3): below this *topical* score (similarity + tag
    # overlap only, not cost/latency -- see ScoredNode.topical_score), prefer
    # a generalist node over a low-confidence specialist match. Calibrated
    # empirically: observed topical scores clustered ~0.35-0.40 for vague/
    # generic queries and ~0.47+ for clearly on-topic ones in local testing.
    routing_confidence_threshold: float = 0.40
    # Tag overlap contributes this weight alongside the existing similarity/
    # cost/latency score (see app/router.py).
    w_tag_overlap: float = 0.15

    # Node onboarding: only ever assign a catalogue model if it leaves this
    # much RAM headroom, so a specialist never swaps and times out.
    assignment_ram_headroom: float = 0.8

    # --- Composition (v0.1.1) -------------------------------------------
    #
    # 'auto'   — compose only where the preconditions from seam-findings.md
    #            hold: the request spans domains, and different nodes are best
    #            at them. The default, because composing unconditionally is
    #            precisely what Experiment 2 did, and it lost.
    # 'never'  — v0.1 behaviour exactly. Also the control arm for any A/B.
    # 'always' — compose whenever two nodes declare two matched domains,
    #            skipping the confidence and single-domain gates (the
    #            domination gate still applies — composing a node with one it
    #            dominates is not a stricter setting, it is a broken one).
    #            For experiments, not for production.
    compose_mode: str = "auto"

    # How big a drop-off separates "the domains this request is about" from the
    # rest. Domains are ranked by similarity and the gateway looks for an elbow
    # in the first few positions; everything above the elbow is relevant.
    #
    # A standard-deviation test was tried first and has a structural ceiling
    # that makes it unusable here: with n declared domains of which k are
    # relevant, the largest achievable z is sqrt((n-k)/k) — so on a four-lane
    # network a genuine two-domain request tops out at z=1.0 exactly, and any
    # threshold that rejects flat requests also rejects the real ones. The
    # elbow is scale-free and has no such ceiling.
    #
    # Measured against the real embedder: off-topic requests ("write me a poem
    # about the sea") produce a largest early gap of 0.026-0.038, while a
    # genuinely on-topic request produces 0.172. 0.05 sits in that gap.
    compose_domain_gap: float = 0.05

    # Below this many declared domains there is no background to measure an
    # elbow against, and the gateway falls back to the absolute floor.
    compose_domain_min_tags_for_relative: int = 4

    # Domain tags that count as "the arithmetic lane". When a request is about
    # one clear domain AND asks for a calculation, the best node holding one of
    # these is seated alongside — the second lane the embedder cannot see.
    compose_quantitative_tags: str = "math,arithmetic,quantitative,calculation,word-problems,maths"

    # A sanity floor, not a discriminator -- it only rejects domains that are
    # unrelated on any reading. The work is done by the relative test above.
    compose_domain_floor: float = 0.30

    # Below this many matched domains there is nothing to compose -- one
    # specialist covers the request and a panel adds cost, latency and
    # dilution without adding coverage.
    compose_min_domains: int = 2

    # Tags that describe breadth rather than a lane, and so can never earn a
    # panel seat. Without this exclusion a generalist matches every request
    # moderately well, `general` clears the floor on almost everything, and the
    # gateway seats a generalist next to a specialist *in that specialist's own
    # domain* — which is Experiment 2's exact mistake, rebuilt as a default.
    # A generalist's job in a panel is to aggregate, and it is still chosen for
    # that (see compose._pick_aggregator).
    compose_excluded_domains: str = "general,conversation,chat,assistant,instruction-following"

    # Hard ceiling on panel size. Every member is a full inference on someone's
    # donated laptop, and the seam findings give no reason to expect returns
    # from breadth beyond the domains a request actually spans.
    compose_max_panel: int = 3

    # Specialists are asked in parallel, so the panel costs one specialist's
    # latency, not the sum -- but the slowest member sets the pace. Deliberately
    # shorter than forward_timeout_seconds: a panel member that has not answered
    # by now is dropped and the rest proceed, because a degraded answer beats a
    # timed-out one. Development is on a 16GB laptop where a cold 7B model can
    # take ~57s, so this is generous on purpose.
    compose_member_timeout_seconds: float = 90.0


settings = Settings()
