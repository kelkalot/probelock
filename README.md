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
   gameable/hardware-dependent" objection doesn't apply.

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
scores **0** for that capability and the run continues, so a model that can't
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
don't (Anthropic, Gemini, …), route through a unified SDK with `--via`. Every path
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
`--endpoint`; they're thin adapters over each SDK's OpenAI-shaped response. Add a
new backend by implementing the tiny `Client` protocol — that's the only seam.

### Recorded demo (Ollama)

[`demo/`](demo/) has runs against a local Ollama server: a committed `qwen3.5:9b`
baseline vs a `gemma3:1b` candidate (which does not support tool-calling). See
[`demo/DEMO.md`](demo/DEMO.md) for the transcript, or replay it:

```bash
asciinema play demo/probelock-demo.cast   # or: bash demo/demo.sh
```

The tool-calling capabilities drop `1.00 → 0.00` and the gate exits non-zero.
`tool_restraint`, `tool_permission`, and `no_hallucinated_tool` stay `1.00` (a
model that can't call tools can't misuse one), and `gemma3:1b` scores `1.00` on
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
| `tool_restraint`      | Does **not** call a tool for a task that needs none (over-trigger) | no tool call |
| `tool_permission`     | Does **not** call a tool it was explicitly forbidden to use | forbidden tool absent |
| `no_hallucinated_tool`| Does **not** fabricate a call to a tool that wasn't offered | called ⊆ offered |
| `format_adherence`    | Follows an exact output constraint                        | exact match |

Three are **negative** probes (a higher score means the bad behavior didn't
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

## Deriving probes from real traces (experimental)

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

A traces file is a small, stable JSON schema probelock defines itself — **not** raw
OpenTelemetry — because OTel's own span attribute layout isn't stable across libraries or
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
instruction, a removed tool) that a passively recorded trace doesn't naturally contain;
`format_adherence` needs an exact-text prompt, not a tool-calling decision point; and
`arity_robustness` needs its own explicit "fill EVERY parameter, including optional ones"
instruction to mean anything — a real conversation was never asked for that, so replaying it
would only test whichever optional fields happened to get filled in that one exchange, not
robustness.

**Unlike a tool schema, a traces file contains real conversation content.** `probe --traces`
prints a warning every time, and the lockfile records a `traces_fingerprint` so a `diff`
flags a baseline/candidate pair whose trace inputs differ — but review and redact the file
yourself before committing it, the same way you'd review any fixture with real data in it.

Tested against a real llama.cpp regression (commit-level, not synthetic): `gate` fails on
the regressed commit and passes on an adjacent, unrelated commit. See
[`VALIDATION.md`](VALIDATION.md) for the test setup and results, and
[`fixtures/gptoss_regression_trace.json`](fixtures/gptoss_regression_trace.json) to reproduce it.

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
- uses: kelkalot/probelock@v0
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

## Roadmap

- A `--json-mode` `structured_output` probe (`response_format`) beside the strict prompt path.
- Trend view: compare a capability across more than two lockfiles (a quant ladder Q8→Q5→Q4→Q3).
- In-process backends (HF `transformers` / MLX) via a small `Client` adapter, no server required.
- Emit OpenTelemetry spans from `probe` runs, so a probe run shows up alongside your other
  agent traces in whatever backend you already use — a follow-on to trace-derived probes
  above (that direction consumes traces; this one produces them).

## License

Apache-2.0 — see [LICENSE](LICENSE).

## Acknowledgements

Built with [Claude Code](https://claude.com/claude-code).
