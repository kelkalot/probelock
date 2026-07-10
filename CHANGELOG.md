# Changelog

All notable changes to probelock are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and probelock adheres to
[semantic versioning](https://semver.org/) — see [STABILITY.md](STABILITY.md) for what a
major, minor, or patch bump means for a committed lockfile.

## [1.0.0]

The stabilization release: the schema-derived core is now a committed compatibility
contract. See [STABILITY.md](STABILITY.md).

### Added
- **`probelock doctor`** — health-check a toolset (too few tools, unconstrained arguments
  that under-report regressions) and detect **schema drift**: when a trace-mined or
  trace-export probe's frozen tool schema no longer matches the live `--tools` (a renamed
  or removed tool, or a changed parameter schema). Exit 1 on any error-level finding (a
  removed pinned tool, a duplicate tool name), `--strict` to fail on warnings too.
- **[STABILITY.md](STABILITY.md)** — the compatibility contract: the stable core vs the
  beta subsystems, what counts as a breaking change, and the frozen-scoring guarantee.
- **Explicit lockfile format version** (`lockfile_format: 1`). A reader refuses a lockfile
  written by a newer probelock rather than mis-scoring a gate; pre-1.0 lockfiles read as
  format 1.
- This changelog.

### Changed
- Marked `Development Status :: 5 - Production/Stable`.
- Documentation now distinguishes the stable core from the beta subsystems (trace
  ingestion, proxy, embeddings clustering, json_mode) instead of labeling them
  "experimental".
- The composite GitHub Action moves to the `v1` major tag (was `v0`) — pin
  `kelkalot/probelock@v1`.

### Fixed
- Malformed OpenTelemetry documents (non-dict elements in the OTLP nesting) and other
  bad ingest input now exit 2 with a clean message rather than a traceback.

## [0.4.0] — 2026-07-09

### Added
- **`probelock trend`** — track each capability across N lockfiles (a quant ladder or a
  timeline), with `regressed` / `improved` / `unstable` / `stable` / `removed` annotations
  and table/markdown/json/html output.
- Ingest adapters for **Anthropic Messages API logs** (`anthropic-jsonl`) and
  **OpenTelemetry GenAI spans** (`otel-genai`, scoped to the `gen_ai.*` semantic
  convention).
- **`--cluster embeddings`** — opt-in near-duplicate dedup via an OpenAI-compatible
  `/v1/embeddings` endpoint (non-deterministic; falls back to hash clustering on failure).
- **`--json-mode`** — a `json_mode` probe that exercises the server's native
  `response_format` API beside the strict-prompt `structured_output`.

### Fixed
- `diff`/`gate` no longer read a within-model runtime swap (`qwen3.5:9b` vs
  `qwen3.5:9b-mlx`) as a cross-model comparison — model-family normalization strips
  runtime/quant markers baked into the model id.

## [0.3.1] — 2026-07-04

### Fixed
- `--tools` is optional on `probe`/`derive` when `--traces` or `--mined` supply probes
  (they carry their own embedded tool definitions).

## [0.3.0] — 2026-07-03

### Added
- **`probelock proxy`** — a recording reverse proxy that captures OpenAI-compatible
  chat traffic as `trace-v1` logs for `ingest` (streaming reassembly, session stitching,
  rotation, strict passthrough-on-error).
- `ingest` accepts multiple rotated log segments as one corpus.

### Changed
- PyPI publishing uses `uv publish` with Trusted Publishing.

## [0.2.0] — 2026-07-02

### Added
- **Trace ingestion** — `probelock ingest` mines deterministic probes from real agent
  traffic (the `openai-jsonl` adapter, confirmed-good filtering, redaction, the
  `traces review` gate, and replay). Mined probes replay under the `traced_*`
  capabilities, reported separately from the synthetic battery.
- `--traces` curated trace-export probes, which replay a hand-assembled export under the
  core capability buckets.

## [0.1.0] — 2026-06-30

### Added
- Initial release: the schema-derived capability battery (eleven deterministic
  capabilities, no LLM judge), `diff`/`gate`, provider routing (`--endpoint` / `--via`),
  and `init` scaffolding.
