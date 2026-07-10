# probelock

[![PyPI](https://img.shields.io/pypi/v/probelock.svg)](https://pypi.org/project/probelock/)
[![Python](https://img.shields.io/pypi/pyversions/probelock.svg)](https://pypi.org/project/probelock/)
[![CI](https://github.com/kelkalot/probelock/actions/workflows/ci.yml/badge.svg)](https://github.com/kelkalot/probelock/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**A capability lockfile for local models.** It records what a model does on a set
of tool-calling and output checks, and fails CI when a model/quant/runtime swap
lowers a score.

```
llama-3.1-8b @ Q8_0 (ollama)  →  llama-3.1-8b @ Q4_K_M (ollama)
Capability            Baseline   Candidate     Δ   Status
arg_validity              1.00        0.67  -0.33  REGRESSION
arity_robustness          1.00        0.67  -0.33  REGRESSION
format_adherence          1.00        1.00  +0.00  ok
needle_in_tools           1.00        0.33  -0.67  REGRESSION
no_hallucinated_tool      1.00        0.67  -0.33  REGRESSION
required_args             1.00        1.00  +0.00  ok
structured_output         1.00        0.33  -0.67  REGRESSION
tool_discrimination       1.00        0.33  -0.67  REGRESSION
tool_permission           1.00        0.67  -0.33  REGRESSION
tool_restraint            1.00        0.67  -0.33  REGRESSION
tool_selection            1.00        0.67  -0.33  REGRESSION

FAIL — capabilities regressed or removed: arg_validity, arity_robustness,
needle_in_tools, no_hallucinated_tool, structured_output, tool_discrimination,
tool_permission, tool_restraint, tool_selection
```

Here the Q4 quant scores 0.33–0.67 on several capabilities where Q8 scored 1.00.
`probelock gate` exits non-zero when a capability drops past the threshold.

## How it differs from promptfoo

> **promptfoo is a test framework you author. probelock is a lockfile you commit.**

1. **Probes are derived from your tool schemas.** Point it at the OpenAI-style
   tool definitions your agent already ships, and it generates a fixed,
   reproducible battery of capability checks. You write no test cases.
2. **No LLM judge.** Every probe is scored by code: JSON-schema validation, exact
   match, or a tool-name check. Run it twice on the same model and the numbers
   match. (promptfoo relies on assertions you write and often on model-graded
   evals, which vary across runs.)
3. **It compares a model against its own baseline,** across a model/quant/runtime
   swap, rather than producing an absolute leaderboard. You only ever compare like
   with like, on your box, with your tools, so the "benchmarks are
   gameable/hardware-dependent" objection does not apply.

## Install & run (only needs [uv](https://docs.astral.sh/uv/))

Run it without installing, or install it into the current environment:

```bash
uvx probelock --help          # run the latest release
pip install probelock         # or install it
```

To run an unreleased revision straight from git:

```bash
uvx --from git+https://github.com/kelkalot/probelock probelock --help
```

The examples below use `uv run` from a checkout of this repo. No model is required
for the demo — a deterministic `SimulatedClient` stands in for two quant levels of
the same model:

```bash
# from the probelock/ project dir
uv run probelock derive --tools examples/agent_tools.json          # see the probe battery
uv run probelock probe  --tools examples/agent_tools.json --simulate fixtures/profile_q8.json -o q8.lock
uv run probelock probe  --tools examples/agent_tools.json --simulate fixtures/profile_q4.json -o q4.lock
uv run probelock diff   q8.lock q4.lock
uv run probelock gate   --baseline q8.lock --candidate q4.lock     # exits non-zero
```

Against a local model, swap `--simulate` for an OpenAI-compatible endpoint:

```bash
uv run probelock probe --tools examples/agent_tools.json \
    --endpoint http://localhost:11434/v1 --model llama3.1:8b-instruct-q4_K_M \
    --quant Q4_K_M --runtime ollama --timeout 120 -o q4.lock
```

A probe the model rejects (e.g. "model does not support tools") or that times out
scores **0** for that capability and the run continues, so a model that cannot
tool-call still produces a lockfile. An unreachable server, a 404 (wrong model or
URL), or a run where every probe fails aborts the run, so a misconfiguration never
becomes a poisoned all-zeros baseline.

`examples/agent_tools.json` is a 3-tool schema for the walkthrough above, not a
sensitivity benchmark — validation testing found it insensitive to real capability
drift that a 10-tool schema with overlapping tool names and richer argument
constraints caught cleanly (see [`VALIDATION.md`](VALIDATION.md)). A schema with too
few tools, or arguments with no real constraints to violate, under-reports
regressions. Point `--tools` at your own agent's actual tool definitions before
trusting `gate` in CI.

## Providers & frameworks

probelock speaks one protocol — OpenAI `/v1/chat/completions` with OpenAI-style
tools — so anything that exposes it works with `--endpoint`. For providers that
do not (Anthropic, Gemini, …), route through a unified SDK with `--via`. Every path
is deterministic; none of them put an LLM in the loop.

| You have… | Use |
|-----------|-----|
| Ollama, **vLLM**, **llama.cpp** server, LM Studio, HF TGI, OpenAI, OpenRouter, Together… | `--endpoint <url>/v1 --model <name>` (vLLM needs `--enable-auto-tool-choice`; llama.cpp needs `--jinja`) |
| **Anthropic / Gemini / Mistral / Bedrock / …** (any-llm) | `--via anyllm --model anthropic/claude-3-5-sonnet` |
| Any of 100+ providers (litellm SDK) | `--via litellm --model anthropic/claude-3-5-sonnet` |
| A running **LiteLLM proxy** | `--endpoint http://litellm:4000/v1 --model <name>` (no extra) |
| In-process HF `transformers` / MLX (no server) | not yet — add a small `Client` adapter |

```bash
pip install 'probelock[anyllm]'   # or 'probelock[litellm]'
probelock probe --tools tools.json --via anyllm --model mistral/mistral-large-latest \
    --samples 5 --temperature 0.7 -o candidate.lock
```

`--via` clients reuse the same caching, sampling, and error semantics as
`--endpoint`; they are thin adapters over each SDK's OpenAI-shaped response. Add a
new backend by implementing the tiny `Client` protocol — that is the only seam.

### Recorded demo (Ollama)

[`demo/`](demo/) has runs against a local Ollama server: a committed `qwen3.5:9b`
baseline vs a `gemma3:1b` candidate (which does not support tool-calling). See
[`demo/DEMO.md`](demo/DEMO.md) for the transcript, or replay it:

```bash
asciinema play demo/probelock-demo.cast   # or: bash demo/demo.sh
```

The tool-calling capabilities drop `1.00 → 0.00` and the gate exits non-zero.
`tool_restraint`, `tool_permission`, and `no_hallucinated_tool` stay `1.00` (a
model that cannot call tools cannot misuse one), and `gemma3:1b` scores `1.00` on
`format_adherence` vs `0.50` for `qwen3.5:9b`. The diff is per-capability.

Also committed: `qwen3.5:9b` vs `lfm2.5-thinking:1.2b`:

```bash
uv run probelock diff demo/qwen3.5-9b.lock demo/lfm2.5-thinking.lock
```

The 1.2B model matches `qwen3.5:9b` on tool selection, discrimination,
`needle_in_tools`, `arg_validity`, `required_args`, and the three safety probes;
`structured_output` and `arity_robustness` drop `1.00 → 0.33`.

## The capabilities (all scored deterministically)

| Capability            | What it checks                                            | Scorer |
|-----------------------|-----------------------------------------------------------|--------|
| `tool_selection`      | Calls the right tool for the task                         | tool-name match |
| `tool_discrimination` | Calls the right tool **and no other** (picks precisely)   | tool-name set |
| `needle_in_tools`     | Finds the right tool when many (15+) are offered          | tool-name match |
| `arg_validity`        | Emitted args validate against the tool's JSON schema      | `jsonschema` |
| `required_args`       | All required args present and non-empty                   | key presence |
| `arity_robustness`    | Fills **every** parameter (required + optional) when asked | all-present |
| `structured_output`   | Emits schema-valid JSON on demand (no tools, no fences)   | parse + `jsonschema` |
| `json_mode` *(opt-in, `--json-mode`)* | Same, but via the server's native `response_format` API instead of a prompt | parse + `jsonschema` |
| `tool_restraint`      | Does **not** call a tool for a task that needs none (over-trigger) | no tool call |
| `tool_permission`     | Does **not** call a tool it was explicitly forbidden to use | forbidden tool absent |
| `no_hallucinated_tool`| Does **not** fabricate a call to a tool that was not offered | called ⊆ offered |
| `format_adherence`    | Follows an exact output constraint                        | exact match |

Three are **negative** probes (a higher score means the bad behavior did not
happen): `tool_restraint` (over-triggering), `tool_permission` (calling a forbidden
tool), and `no_hallucinated_tool` (fabricating a tool). All probes are derived from
the tool schemas, not hand-authored.

## Architecture

```
tool schemas ──▶ derive probes ──▶ Client ──▶ ResponseMessage ──▶ deterministic scorer ──▶ Lockfile
 (your agent)    (zero authoring)  (model)    (the only model      (no LLM judge)          (commit it)
                                              -touching part)
                                                                    Lockfile + Lockfile ──▶ diff / gate
```

The only nondeterministic part is the `Client`; everything else is pure, so the
same inputs produce the same lockfile and the same diff. At temperature 0 the
client caches identical requests, so the probes that share one request (the tool
checks for a given tool) hit the network once. The `SimulatedClient` crafts correct or
incorrect responses that the real scorers grade, so the scoring path runs even
with no model present.

## Deriving probes from real traces (beta)

Schema-derived probes are single-turn and synthetic — great for catching schema-level
regressions, blind to what breaks after several turns of real context, a tool result
feeding back in, or ambiguous phrasing. `--traces` adds a second source: real,
already-recorded agent decision points (e.g. exported from litellm's OpenTelemetry
callback), replayed through the exact same deterministic scorers.

```bash
uv run probelock derive --tools tools.json --traces traces.json      # see what gets added
uv run probelock probe  --tools tools.json --traces traces.json \
    --endpoint http://localhost:11434/v1 --model llama3.1:8b -o candidate.lock
```

`--tools` is optional here: traced probes replay their own embedded tool definitions,
so a trace-only run (`probe --traces traces.json ...`) needs no schema file. The same
holds for `--mined` below. Provide `--tools` when you also want the synthetic battery.

A traces file is a small, stable JSON schema probelock defines itself — **not** raw
OpenTelemetry — because OTel's own span attribute layout is not stable across libraries or
versions (litellm has already changed where it puts request/response attributes once, and
has a newer, differently-shaped opt-in integration). Converting your export into this shape
is a one-time step you own; see
[`examples/otel_traces_to_probelock.py`](examples/otel_traces_to_probelock.py) for a
documented starting point and [`fixtures/sample_traces.json`](fixtures/sample_traces.json)
for the target shape:

```json
{
  "version": 1,
  "records": [
    {
      "id": "checkout-flow-turn-3",
      "messages": [{"role": "user", "content": "..."}],
      "tools": [ /* OpenAI-style tool defs actually offered at this turn */ ],
      "response": {"content": null, "tool_calls": [{"name": "...", "arguments": "{...}"}]}
    }
  ]
}
```

Trace-derived probes join the *same* capabilities as schema-derived ones — `tool_selection`,
`tool_discrimination`, `arg_validity`, `required_args`, and `structured_output` — since these
map cleanly onto "replay this real context, check the candidate still behaves validly" (probe
ids carry a `::traced::` marker if you want to inspect the split). The rest stay purely
schema-derived: `needle_in_tools`, `tool_permission`, `no_hallucinated_tool`, and
`tool_restraint` need a synthetic perturbation (an injected distractor tool, a forbidden-tool
instruction, a removed tool) that a passively recorded trace does not naturally contain;
`format_adherence` needs an exact-text prompt, not a tool-calling decision point; and
`arity_robustness` needs its own explicit "fill EVERY parameter, including optional ones"
instruction to mean anything — a real conversation was never asked for that, so replaying it
would only test whichever optional fields happened to get filled in that one exchange, not
robustness.

**Unlike a tool schema, a traces file contains real conversation content.** `probe --traces`
prints a warning every time, and the lockfile records a `traces_fingerprint` so a `diff`
flags a baseline/candidate pair whose trace inputs differ — but review and redact the file
yourself before committing it, the same way you would review any fixture with real data in it.

Tested against a real llama.cpp regression (commit-level, not synthetic): `gate` fails on
the regressed commit and passes on an adjacent, unrelated commit. See
[`VALIDATION.md`](VALIDATION.md) for the test setup and results, and
[`fixtures/gptoss_regression_trace.json`](fixtures/gptoss_regression_trace.json) to reproduce it.

## Recording traffic (`probelock proxy`) (beta)

If your stack does not already log requests, the recording proxy captures them with one
line changed in the agent:

```bash
probelock proxy --listen 127.0.0.1:8484 \
                --upstream http://127.0.0.1:11434 \
                --out traces/agent.jsonl
# agent side: base_url = "http://127.0.0.1:8484/v1"
```

Every request is forwarded to the upstream unchanged (streaming included — SSE flows
token by token and is reassembled for the record afterwards, tool-call deltas and all);
each completed chat-completions exchange is appended asynchronously as one `trace-v1`
JSONL record. Recording is strictly non-invasive: on any internal logging error the
request is still served and a warning goes to stderr. Failed or truncated exchanges
(upstream errors, mid-stream disconnects) are logged with a failing status so `ingest`
skips them instead of mining half-generated responses. Multi-turn conversations are
stitched into sessions without any agent cooperation (restarting the proxy mid-conversation
splits that conversation into two sessions — harmless, but it weakens confirmation
evidence, so prefer restarting between runs), `--max-size` / `--max-age` rotate
the log, and the file is created `0600` — it holds **verbatim conversation content**;
keep it out of version control (redaction happens later, at `ingest`).

## Mining probes from raw agent logs (beta)

`--traces` (above) replays a *curated* export you assembled by hand. `probelock ingest`
goes one step earlier: point it at a raw request/response log of real agent traffic —
the proxy's output, or your own logging — and it mines probes for you: multi-turn,
realistic regression tests with near-zero authoring effort, still scored by the same
deterministic checks (LLMs may appear in *trace generation* — that is your own agent —
but never in *scoring*).

```bash
probelock ingest traces/agent.jsonl --out probes/mined.json   # everything lands "pending"
probelock traces review probes/mined.json                     # activate probes (y/n/e/a/s/q)
probelock probe --tools tools.json --mined probes/mined.json \
    --endpoint http://localhost:11434/v1 --model llama3.1:8b -o candidate.lock
```

`ingest` accepts several logs at once (`probelock ingest agent.jsonl agent-*.jsonl`) —
pass a rotated set together so sessions spanning a rotation boundary keep their
confirmation evidence.

Several input formats are supported (`--format`, or `auto`):

| `--format` | Shape |
|---|---|
| `trace-v1` | the native record the recording proxy writes (one JSON object per line, `request`/`response.message`) |
| `openai-jsonl` | the verbatim chat-completions request body next to the verbatim response, per line |
| `anthropic-jsonl` | logged Anthropic Messages API calls (`request`/`response`); content blocks, `tool_use`/`tool_result`, and `system` are translated to the canonical shape |
| `otel-genai` | an OTLP-JSON span export, read via the OpenTelemetry **GenAI semantic-convention** attributes (`gen_ai.prompt`/`gen_ai.completion`, blob or indexed form) — scoped to the spec, not any one library's layout; spans without those attributes are skipped and counted |

`auto` detects the JSONL shapes and OTel documents. See the `fixtures/sample_*` files for
each. For OTel exporters that do not follow the semantic convention,
[`examples/otel_traces_to_probelock.py`](examples/otel_traces_to_probelock.py) remains the
conversion recipe.

Deduplication is exact-hash by default (deterministic). `--cluster embeddings
--embed-endpoint URL --embed-model NAME` instead groups *near-duplicate* contexts by
embedding cosine similarity (via an OpenAI-compatible `/v1/embeddings` endpoint you
already run). This is opt-in and **not deterministic** — the grouping depends on the
embedding model and version, so probelock prints a caveat and records `cluster:
embeddings` in each affected probe's provenance. Everything downstream (scoring, gating)
stays deterministic; only which contexts merged does not.

Raw traffic includes model mistakes, so **provenance determines trust** — every probe
records how many sessions support it and which rule confirmed it, and that decides how
much review it needs:

| Category | Check at replay | Mined from | Review |
|---|---|---|---|
| `traced_schema_validity` | some call's args validate against the called tool's schema | every tool-calling exchange (no inference) | `--auto-accept schema_validity` is safe |
| `traced_tool_selection` | calls the confirmed tool | exchanges confirmed good: the result fed back and the conversation moved on (no error payload, no corrected-args retry, no re-ask), or the same context produced the same call in ≥ `--min-agreement` distinct sessions | review, or `--auto-accept-all --i-know-what-im-doing` |
| `traced_no_tool` | answers in text, calls nothing | unanimous text answers across ≥ `--min-agreement-notool` (default 3) distinct sessions, no re-ask, preferring contexts lexically distant from the tools | **individual review only** — a mislabeled probe freezes a model mistake as expected behavior |

The traced capabilities are deliberately separate names in the lockfile: a drop in
multi-turn trace probes while single-turn synthetic probes hold steady is itself
diagnostic ("context-length-sensitive regression").

Privacy defaults are conservative. Argument values in frozen contexts are always
replaced with structure-preserving placeholders (`"query": "<str:47ch>"`); message
content stays verbatim (that is what makes replay realistic), so mined probes carry
`"sensitive": true` and `probe -o` refuses to include them in a written lockfile
without `--allow-sensitive`. If you want committable probes, opt in to scrubbing with
`ingest --redact-patterns emails,phones,paths`. Identical contexts are deduplicated
(timestamps stripped, whitespace collapsed), sampling keeps up to `--per-capability`
probes per (tool, category) preferring longer contexts and later turns, and everything
the pipeline skips (failed calls, forced `tool_choice`, oversized contexts, ambiguous
agreement) is counted and reported — never silently dropped.

The full pipeline is validated against real agent traffic on real local models —
including a runtime swap the gate catches and real frozen-mistake probes the review
step rejects — in [`VALIDATION-TRACES.md`](VALIDATION-TRACES.md).

## Trends across a ladder

`diff` compares two lockfiles; `trend` compares N, in the order you give them — a
quantization ladder or the same model over time — so you can see *where* a capability
holds and *where* it cliffs:

```bash
uv run probelock trend Q8_0.lock Q6_K.lock Q5_K_M.lock Q4_K_M.lock Q3_K_M.lock Q2_K.lock
```

```
Capability          Q8_0   Q6_K   Q5_K_M   Q4_K_M   Q3_K_M   Q2_K       Δ   Trend
structured_output   1.00   1.00     1.00     0.67     0.33   0.33   -0.67   ↓ regressed
tool_restraint      1.00   1.00     1.00     1.00     1.00   1.00   +0.00   = stable
tool_selection      1.00   1.00     0.67     1.00     0.67   1.00   +0.00   ~ unstable
```

Each row is annotated by its whole-ladder behavior: `regressed` (net drop past
`--max-drop`), `improved`, `unstable` (net-flat but it dipped along the way — a signal a
two-point diff of the endpoints would miss), `stable`, `removed` (present early but gone
from the last rung — a dropped capability, counted as a regression), or `partial`
(present in fewer than two lockfiles). `--format markdown|json|html` mirror `diff`; the
HTML view draws a sparkline per capability. `trend` never fails on a regression (use
`gate` pairwise for CI); it exits non-zero only on bad input.

The filename stem is each column's header, so name your lockfiles for the axis
(`Q8_0.lock`, `Q4_K_M.lock`).

## Sampling & noisy gates

With one sample per probe, a capability backed by 3 tools quantizes to
`{0, 0.33, 0.67, 1.0}` — a single flip moves it 0.33, far past the default 0.05
gate. So:

- **`probe --samples N [--temperature T]`** runs each probe N times and records the
  pass-*rate* (raise the temperature for sampling variance).
- **`gate --confidence 0.95`** only fails on a drop that is statistically
  significant for the recorded trial count (a one-sided two-proportion test).
  Sub-significant drops are shown as **`noisy ↓`** and do **not** fail the gate;
  raise `--samples` to confirm or clear them.

A total collapse (e.g. `1.00 → 0.00`) is significant even at low N; a single-flip
`1.00 → 0.67` over 3 trials is `noisy` until you raise `--samples`.

## In CI

`probelock init` scaffolds a `probelock.tools.json` and a
`.github/workflows/probelock.yml` to start from. Commit a baseline lockfile, then
gate each candidate:

```yaml
- run: uvx probelock probe
       --tools tools.json --endpoint $LLM_URL --model $MODEL --samples 5 --temperature 0.7 -o candidate.lock
- run: uvx probelock gate
       --baseline probelock.lock --candidate candidate.lock --max-drop 0.05 --confidence 0.95
```

Or use the composite GitHub Action ([`action.yml`](action.yml)), which wraps those
two steps end-to-end:

```yaml
- uses: kelkalot/probelock@v1
  with:
    tools: tools.json
    baseline: probelock.lock
    endpoint: ${{ secrets.LLM_ENDPOINT }}
    model: ${{ vars.LLM_MODEL }}
```

To show the result on a pull request, render the diff as Markdown (or `--format html`
for a self-contained page):

```bash
probelock diff probelock.lock candidate.lock --format markdown >> "$GITHUB_STEP_SUMMARY"
```

## Stability

probelock follows [semantic versioning](https://semver.org/), and a committed lockfile is
a compatibility contract — see [STABILITY.md](STABILITY.md). In short: the schema-derived
capability battery, their **scoring**, the lockfile format, and `diff`/`gate`/`trend` are
**stable** and will not change incompatibly within `1.x`, so a committed baseline stays
meaningful across upgrades. The trace subsystems above (marked *beta*) — `ingest`, the log
adapters, `proxy`, embeddings clustering, `json_mode` — are validated but may evolve in a
minor release, always with a [CHANGELOG.md](CHANGELOG.md) note.

`probelock doctor` checks a toolset for weaknesses and detects when your committed
trace-mined probes have drifted from the live toolset (a gate failure that is really
drift, not a regression):

```bash
probelock doctor --tools tools.json --mined probes/mined.json
```

## Roadmap (post-1.0)

- Proxy hardening: a static Go/Rust binary beside the reference Python implementation,
  and streaming-reassembly edge cases (multi-line SSE events, resume-after-disconnect).
- In-process backends (HF `transformers` / MLX) via a small `Client` adapter, no server required.
- Emit OpenTelemetry spans from `probe` runs, so a probe run shows up alongside your other
  agent traces in whatever backend you already use — a follow-on to trace-derived probes
  above (that direction consumes traces; this one produces them).

## License

Apache-2.0 — see [LICENSE](LICENSE).

## Acknowledgements

Built with [Claude Code](https://claude.com/claude-code).
