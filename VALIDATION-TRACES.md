# Validating the trace pipeline (capture → mine → review → replay)

[`VALIDATION.md`](VALIDATION.md) validates the *synthetic* battery against a real
llama.cpp regression. This file validates the *trace* pipeline shipped in v0.2.0–v0.3.1
— `probelock proxy`, `probelock ingest`, `probelock traces review`, and `--mined`
replay — against real agent traffic on real local models. Date: 2026-07-04, probelock
v0.3.1.

Three claims were under test:

1. The pipeline works end to end on real traffic: a live model behind the proxy, real
   multi-turn tool loops, mining with default thresholds, review, replay.
2. The mandatory review gate is necessary, not ceremony: confirmed-good filtering is
   heuristic and real traffic freezes real model mistakes.
3. Trace-derived capabilities measure an axis the synthetic battery does not, and the
   two move independently across a swap (the design document's core pitch).

All three held. Details below.

## Setup

- **Traffic**: `examples/record_agent_sessions.py` — a scripted agent loop where every
  assistant turn is real model behavior. 31 sessions against `qwen3.5:9b` (Ollama,
  temperature 0.2): 8 tool tasks × 2 reruns with tool results fed back and a follow-up
  turn, 4 no-tool questions × 3 sessions, 3 ambiguous "trap" asks. Every user turn
  carries a timestamp prefix, as real traffic does — reruns are distinct sessions to
  the proxy (raw bytes differ) yet cluster together after ingest normalization strips
  the timestamps.
- **Toolset**: `fixtures/hard_agent_tools.json` — the 10-tool schema VALIDATION.md
  found sensitive (overlapping calendar/messaging/records tools).
- **Capture**: `probelock proxy --upstream http://127.0.0.1:11434` recording to a
  trace-v1 JSONL file.
- **Models**: baseline `qwen3.5:9b`; candidate A `qwen3.5:9b-mlx` (same weights,
  different runtime — the within-model swap probelock exists for); candidate B
  `lfm2.5-thinking:1.2b` (cross-model capability gap, informational).

## Capture and mining

31 sessions produced 71 trace records (multi-turn tool loops produce several records
per session). `probelock ingest` with default thresholds:

| | count |
|---|--:|
| records read / skipped | 71 / 0 |
| sessions recognized | 31 (exactly the driver's count — stitching needed no agent cooperation) |
| distinct contexts | 55 |
| `schema_validity` mined | 19 |
| `tool_selection` mined (all via the continuation rule) | 17 |
| `no_tool` mined (all four questions reached the 3-session bar) | 4 |
| tool-calling contexts NOT confirmed good | 4 (the traps and fumbles — filtered as designed) |

## Review: the gate caught real label noise

`--auto-accept schema_validity` accepted 19 probes (no correctness inference). The 17
`tool_selection` and 4 `no_tool` probes were reviewed individually. **3 of the 17
inferred tool-selection probes (18%) were frozen model mistakes** and were rejected:

| the user asked | the model called | correct tool |
|---|---|---|
| "When is Sarah free this week…" | `search_records(record_type="contact")` | `find_availability` |
| "Book the first free slot…" | `search_records(record_type="contact")` | `create_event` |
| "Move my 'Board prep' event…" | `search_records(record_type="task")` | `update_event` |

All three passed the continuation rule because the environment cooperated: the driver
returned a plausible result for whatever tool was called and the conversation moved on.
Real agent environments do the same. This is exactly the failure mode the design
document predicts for heuristic confirmation, and it is why `tool_selection` probes
have no unattended auto-accept path. Final battery: 37 accepted probes.

## Replay

Combined battery (synthetic from the 10-tool schema + 37 mined probes, 132 probes
total, temperature 0, one sample per probe):

| capability | qwen3.5:9b (baseline) | qwen3.5:9b-mlx | lfm2.5-thinking:1.2b |
|---|--:|--:|--:|
| `tool_selection` | 1.00 | **0.60** | 1.00 |
| `tool_discrimination` | 1.00 | **0.60** | 1.00 |
| `arg_validity` | 1.00 | **0.60** | 0.90 |
| `required_args` | 1.00 | **0.60** | 1.00 |
| `structured_output` | 1.00 | 1.00 | **0.50** |
| `arity_robustness` | 1.00 | 1.00 | 0.90 |
| `needle_in_tools` / safety probes | 1.00 | 1.00 | 1.00 |
| `format_adherence` | 0.50 | 0.50 | 0.50 |
| `traced_schema_validity` | 0.68 | 0.84 | 0.74 |
| `traced_tool_selection` | 0.71 | **0.50** | 0.79 |
| `traced_no_tool` | 1.00 | 1.00 | 1.00 |

`gate` fails both candidates (exit 1), for different reasons.

## Findings

1. **The pipeline works end to end on real traffic.** No skipped records, exact
   session stitching, every mining category populated, and both confirmation rules
   exercised — with zero manual trace curation.
2. **Review is load-bearing.** An 18% frozen-mistake rate among inferred
   tool-selection probes on a well-behaved 9B model means unattended auto-accept
   would have poisoned the baseline with probes that punish candidates for fixing
   the mistakes.
3. **Trace probes measure a different axis.** The baseline model scores 1.00 on
   every synthetic tool capability yet 0.68/0.71 on its own recorded multi-turn
   decisions replayed at temperature 0. Long contexts with tool feedback are
   materially harder than single-turn synthetic prompts — the coverage gap the
   feature exists to close is real and visible.
4. **The axes move independently, in both directions.** The MLX runtime swap
   regresses synthetic tool capabilities (1.00 → 0.60) and `traced_tool_selection`
   (0.71 → 0.50) while `traced_schema_validity` *improves* (0.68 → 0.84); the 1.2B
   model holds synthetic `tool_selection` at 1.00 while `structured_output` collapses
   and its traced scores improve slightly. A single-axis battery would have told an
   incomplete story in both cases; reporting trace-derived scores separately (the
   design decision in §5) is vindicated.

## Notes

- `qwen3.5:9b` and `qwen3.5:9b-mlx` are the same weights under different runtimes,
  but they carry different model ids, so `diff` adds its cross-model warning to what
  is genuinely a within-model runtime swap. Cosmetic, but worth knowing: the note
  keys on the id, and Ollama encodes the runtime in the id.
- One sample per probe quantizes traced capability scores in steps of 1/14 for the
  14-probe `traced_tool_selection` group. The standard guidance from the README's
  sampling section applies unchanged: raise `--samples` with a temperature above zero
  and gate with `--confidence` for production use. Sub-1.0 *baseline* traced scores
  are expected and harmless — the gate compares candidates against the baseline, not
  against 1.0.
- The rejected probes stay in the mined file with `status: "rejected"` — re-running
  `ingest` on the same log would re-mine them, so keep the reviewed file, not the
  raw log, as the artifact of record.

## Reproducing

```bash
# terminal 1 — record (verbatim conversation content lands in traces/agent.jsonl)
uv run probelock proxy --upstream http://127.0.0.1:11434 --out traces/agent.jsonl

# terminal 2 — traffic, mining, review, replay
uv run python examples/record_agent_sessions.py --model qwen3.5:9b
uv run probelock ingest traces/agent.jsonl --out probes/mined.json
uv run probelock traces review probes/mined.json --auto-accept schema_validity
uv run probelock traces review probes/mined.json          # review the rest honestly
uv run probelock probe --tools fixtures/hard_agent_tools.json \
    --mined probes/mined.json --allow-sensitive \
    --endpoint http://127.0.0.1:11434/v1 --model qwen3.5:9b -o baseline.lock
# swap --model for the candidate, then:
uv run probelock gate -b baseline.lock -c candidate.lock
```

Model outputs vary run to run (capture happens at temperature 0.2), so mined counts
and exact scores will differ; the shape of the findings should not.
