# Stability & compatibility

probelock's promise is a **lockfile you commit** and gate against for the life of a
project. That is only credible if upgrading probelock does not silently change what your
committed lockfile means. This document is the contract, effective from **1.0.0**, and it
follows [semantic versioning](https://semver.org/): within a `1.x` series nothing listed
as *stable* below breaks.

## The stable core (frozen within 1.x)

These will not change in a backward-incompatible way before 2.0:

- **Lockfile format.** A lockfile written by any `1.x` carries `lockfile_format: 1` and is
  readable by every `1.x`. A reader refuses a lockfile whose format is *newer* than it
  understands (clean error, exit 2) rather than mis-scoring a gate. New optional fields may
  be added without a format bump; a format bump only happens on a breaking change and only
  at a major release.
- **The schema-derived capability set and its scoring.** The eleven schema-derived
  capabilities (`tool_selection`, `tool_discrimination`, `needle_in_tools`,
  `arity_robustness`, `arg_validity`, `required_args`, `structured_output`,
  `format_adherence`, `tool_restraint`, `tool_permission`, `no_hallucinated_tool`) keep
  their names and their scoring semantics. **The score a stable capability assigns to a
  given (probe, response) is frozen within `1.x`** — a change that would move scores on
  unchanged inputs is breaking and waits for `2.0`. This is what lets a committed baseline
  stay meaningful across upgrades.
- **`diff`, `gate`, `trend`, `derive`, `probe`, `init`, `version`** and their documented
  flags and exit codes. Exit `1` = regression (gate), exit `2` = invalid input, exit `0` =
  ok — across all of them.
- **Determinism.** The scored content of a lockfile — the capability scores and per-probe
  results — is a pure function of the inputs, so the same inputs produce the same scores
  and the same diff (only the recorded `generated_at` timestamp varies run to run). The
  only model-dependent component is the `Client`.

## Beta subsystems (may evolve in a `1.x` minor)

These shipped and are validated (see [`VALIDATION.md`](VALIDATION.md) and
[`VALIDATION-TRACES.md`](VALIDATION-TRACES.md)), but are younger than the core and may
change in a minor release — always with a [`CHANGELOG.md`](CHANGELOG.md) note. Each
subsystem that writes a file versions that file (noted below), so a future reader can
tell what it is looking at:

- **Trace ingestion** — `probelock ingest`, the log adapters (`trace-v1`, `openai-jsonl`,
  `anthropic-jsonl`, `otel-genai`), the confirmed-good mining rules, and the mined-probe
  file format (`version: 1`). Mining heuristics may be tuned; re-mine after a minor.
- **Recording proxy** — `probelock proxy` and the `trace-v1` record shape (`v: 1`).
- **`traced_*` capabilities** (`traced_schema_validity`, `traced_tool_selection`,
  `traced_no_tool`) come from `probe --mined` and are reported separately from the core.
- **Curated trace-export probes** (`probe --traces`) replay a hand-assembled export;
  they score under the *stable core* capability buckets (e.g. `tool_selection`,
  `structured_output`), not the `traced_*` ones. The `--traces` feature is beta, but the
  buckets its probes land in are the frozen core.
- **`json_mode` probe** (`--json-mode`) and **embeddings clustering** (`--cluster
  embeddings`, itself explicitly non-deterministic).
- **`probelock doctor`** output and codes.

## What counts as a breaking change (→ next major)

- A lockfile-format bump, or removing/renaming a stable field.
- Removing or renaming a stable capability, or changing its scoring on unchanged inputs.
- Removing or renaming a stable command or documented flag, or changing an exit code.

## What is *not* breaking (→ minor or patch)

- **Adding a capability.** New capabilities are additive: they show as `added` in a diff
  and never retroactively fail an existing baseline (an added capability is not a
  regression). Re-mint your baseline to include it.
- Adding a command, flag, adapter, or beta-subsystem capability.
- Any change to a beta subsystem (noted in the changelog).
- A **scoring correction**: if a stable scorer is demonstrably wrong, it may be fixed in a
  minor — but only as an explicit, prominently-noted [`CHANGELOG.md`](CHANGELOG.md) entry, because a
  knowingly-wrong score is worse than the churn. Re-mint baselines when one lands.

## Upgrading

Pin probelock in CI (`uvx probelock==1.x` or a lockfile) so a gate run is reproducible.
After a minor that touches a beta subsystem or lands a scoring correction, re-run `probe`
to refresh your committed baseline, and use `probelock doctor` to catch a toolset that has
drifted away from your frozen probes.
