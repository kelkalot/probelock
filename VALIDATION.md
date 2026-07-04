# Validation

Three tests of probelock's regression detection against real local models on a
48 GB Apple Silicon Mac:

- **Quantization ladder** — does the signal track quantization's known,
  roughly-monotonic effect on capability?
- **Runtime backend comparison** — GGUF vs MLX at matched quant.
- **Regression replay** — a real, documented, commit-level llama.cpp
  regression, as ground truth.

(The trace pipeline — proxy capture, mining, review, replay — has its own
validation in [`VALIDATION-TRACES.md`](VALIDATION-TRACES.md), kept separate so each
round keeps its own overview.)

The regression replay ships a reproducible fixture in this repo
([`fixtures/gptoss_regression_trace.json`](fixtures/gptoss_regression_trace.json)),
as does the harder tool schema used in the quantization ladder follow-up
([`fixtures/hard_agent_tools.json`](fixtures/hard_agent_tools.json)). Both
still require an external model download to run against (`gpt-oss-20b-GGUF`
and `Qwen2.5-7B-Instruct-GGUF` respectively) — the schema/trace files are
what's bundled, not the model weights. The runtime comparison is reported here
with the exact commands used but has no bundled fixture.

## Quantization ladder

Setup: `bartowski/Qwen2.5-7B-Instruct-GGUF`, F16 → Q2_K, served via local
`llama-server` (Metal), 10 samples/probe at temperature 0.7, gated against the
F16 baseline at 95% confidence.

```bash
llama-server --jinja --flash-attn auto -hf bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M \
    --alias qwen25-7b --port 8080
uv run probelock probe --tools examples/agent_tools.json \
    --endpoint http://localhost:8080/v1 --model qwen25-7b \
    --quant Q4_K_M --runtime llama.cpp --samples 10 --temperature 0.7 -o q4_k_m.lock
uv run probelock gate --baseline f16.lock --candidate q4_k_m.lock --confidence 0.95
```

| Capability | F16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_M | Q2_K |
|---|--:|--:|--:|--:|--:|--:|--:|
| tool_selection | .967 | .967 | .967 | .967 | 1.00 | 1.00 | 1.00 |
| tool_discrimination | .967 | .967 | 1.00 | .933 | 1.00 | 1.00 | 1.00 |
| needle_in_tools | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| arg_validity | 1.00 | .967 | 1.00 | .967 | 1.00 | 1.00 | 1.00 |
| required_args | .933 | 1.00 | 1.00 | .967 | 1.00 | 1.00 | 1.00 |
| arity_robustness | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| structured_output | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| tool_restraint | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| tool_permission | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| no_hallucinated_tool | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| format_adherence | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.90 |

`gate --confidence 0.95` against the F16 baseline passes at every quant down to
Q2_K. The one drop that appears at all (`format_adherence`, 1.00 → 0.90 at
Q2_K) is one flipped sample out of ten on a two-probe capability; the
statistical gate marks it `noisy ↓` rather than failing the build over it.

No published per-quant perplexity numbers exist for this GGUF, so this isn't a
precise comparison against the known perplexity curve. On this 3-tool schema,
capability scores don't move meaningfully down to 2-bit quantization. That result
is ambiguous on its own: it could mean tool-calling is inherently robust to
quantization, or it could mean a 3-tool schema with near-ceiling scores on most
capabilities has no room left to show a drop. The follow-up below distinguishes
the two.

### Follow-up: harder schema, higher sample count

Same ladder, same cached weights, no new downloads. Two changes: the tool
schema goes from 3 tools to 10
([`fixtures/hard_agent_tools.json`](fixtures/hard_agent_tools.json)), with
three semantically overlapping clusters (event operations, messaging, record
operations) and richer argument constraints (enums, `format`, `multipleOf`,
nested arrays of formatted strings); and `--samples` goes from 10 to 15. This
raises trials per capability from 30 (3 probes × 10 samples) to 150 (10 probes
× 15 samples).

```bash
uv run probelock probe --tools fixtures/hard_agent_tools.json \
    --endpoint http://localhost:8080/v1 --model qwen25-7b \
    --quant Q4_K_M --runtime llama.cpp --samples 15 --temperature 0.7 -o q4_k_m.lock
```

| Capability | F16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_M | Q2_K |
|---|--:|--:|--:|--:|--:|--:|--:|
| arg_validity | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| arity_robustness | 0.913 | 0.940 | 0.907 | 0.907 | 0.973 | 0.920 | 0.967 |
| format_adherence | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.700 |
| needle_in_tools | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| no_hallucinated_tool | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| required_args | 1.00 | 0.987 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| structured_output | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| tool_discrimination | 0.993 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| tool_permission | 0.927 | 0.913 | 0.933 | 0.940 | 0.853 | 0.993 | 1.00 |
| tool_restraint | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| tool_selection | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

Two capabilities baseline below 1.00 on this schema (`arity_robustness` at
0.913, `tool_permission` at 0.927 on F16), which gives the ladder room to move
in either direction — unlike the 3-tool schema, where most capabilities were
already pinned at ceiling. `gate --confidence 0.95` against the F16 baseline
flags two statistically significant regressions:

- **Q4_K_M**: `tool_permission` 0.927 → 0.853 (Δ −0.073, significant). Not the
  most aggressive quant in the ladder — Q3_K_M and Q2_K score higher on the
  same capability. Non-monotonic relative to bit-width.
- **Q2_K**: `format_adherence` 1.00 → 0.700 (Δ −0.300, significant). At the
  most aggressive quant, consistent with a genuine floor effect. The 3-tool
  run showed the same capability dip at the same quant (1.00 → 0.90) but
  didn't have enough samples to clear the significance bar.

Provenance note on the Q2_K result: bartowski's quants, including Q2_K, are
built with imatrix calibration (llama.cpp b3772, calibration dataset linked on
the model card), not naive round-to-nearest quantization. Imatrix calibration
generally reduces low-bit quality loss relative to an uncalibrated quant of the
same size. The −0.300 drop is observed under that more favorable recipe — it's
a real result for this specific GGUF, not necessarily representative of Q2_K
quantization in general; an uncalibrated Q2_K, or one calibrated on a
differently-composed dataset, could show a larger or smaller effect.

This resolves the ambiguity from the 3-tool run: the probe battery does detect
routine, non-catastrophic capability drift, including at a non-terminal point
in the ladder — the 3-tool result was a sensitivity limit of that particular
schema and sample count, not a property of the scoring method. The
non-monotonic `tool_permission` result is itself a relevant data point for how
probelock is meant to be used: a candidate one step down the ladder from a
passing one is not guaranteed to also pass, which is the argument for gating
every quant/runtime swap against baseline rather than assuming a smooth
degradation curve.

## GGUF vs MLX, same nominal quant

Setup: `qwen3.5:9b` (Ollama's default GGUF backend) vs `qwen3.5:9b-mlx`, both
Q4_K_M, 5 samples at temperature 0.7. Isolates an inference-engine effect
(chat-template rendering, tool-call parsing) that a quant-only comparison
doesn't cover.

```bash
ollama pull qwen3.5:9b && ollama pull qwen3.5:9b-mlx
uv run probelock probe --tools examples/agent_tools.json \
    --endpoint http://localhost:11434/v1 --model qwen3.5:9b \
    --quant Q4_K_M --runtime ollama-gguf --samples 5 --temperature 0.7 -o gguf.lock
uv run probelock probe --tools examples/agent_tools.json \
    --endpoint http://localhost:11434/v1 --model qwen3.5:9b-mlx \
    --quant Q4_K_M --runtime ollama-mlx --samples 5 --temperature 0.7 -o mlx.lock
uv run probelock diff gguf.lock mlx.lock
```

| Capability | GGUF | MLX | Δ | Status |
|---|--:|--:|--:|---|
| arg_validity | 1.00 | 0.47 | −0.53 | REGRESSION |
| required_args | 1.00 | 0.33 | −0.67 | REGRESSION |
| tool_permission | 0.87 | 1.00 | +0.13 | improved |
| tool_discrimination | 0.93 | 1.00 | +0.07 | improved |
| structured_output | 0.87 | 0.93 | +0.07 | improved |
| arity_robustness | 0.93 | 1.00 | +0.07 | improved |
| tool_selection / needle_in_tools / no_hallucinated_tool / tool_restraint / format_adherence | — | — | +0.00 | ok |

`arg_validity` and `required_args` drop sharply on MLX while several other
capabilities move the other way. At 5 samples (15 trials/capability), a swing
this size still needs a confidence check before it's more than a suggestive
delta — rerun below.

### Confirming at higher sample count

Same two models, same quant, `--samples 15` (45 trials/capability) instead of
5, `gate`d at 95% confidence instead of read off the raw diff.

```bash
uv run probelock probe --tools examples/agent_tools.json \
    --endpoint http://localhost:11434/v1 --model qwen3.5:9b \
    --quant Q4_K_M --runtime ollama-gguf --samples 15 --temperature 0.7 -o gguf.lock
uv run probelock probe --tools examples/agent_tools.json \
    --endpoint http://localhost:11434/v1 --model qwen3.5:9b-mlx \
    --quant Q4_K_M --runtime ollama-mlx --samples 15 --temperature 0.7 -o mlx.lock
uv run probelock diff gguf.lock mlx.lock --confidence 0.95
```

| Capability | GGUF | MLX | Δ | Status | Significant |
|---|--:|--:|--:|---|---|
| arg_validity | 1.00 | 0.33 | −0.67 | REGRESSION | yes |
| required_args | 1.00 | 0.33 | −0.67 | REGRESSION | yes |
| tool_permission | 0.93 | 1.00 | +0.07 | improved | — |
| arity_robustness | 0.98 | 1.00 | +0.02 | ok | — |
| structured_output | 1.00 | 1.00 | +0.00 | ok | — |
| tool_discrimination | 1.00 | 1.00 | +0.00 | ok | — |
| tool_selection / needle_in_tools / no_hallucinated_tool / tool_restraint / format_adherence | — | — | ~+0.00 | ok | — |

`arg_validity` and `required_args` hold at the same magnitude and clear the
significance bar — a real, confirmed regression, not a small-sample artifact.
Two of the four "improved" deltas from the 5-sample run (`tool_discrimination`,
`structured_output`) collapsed to `ok` with more data; only `tool_permission`
held. That's the expected shape when re-running with more power: real effects
persist, marginal ones regress toward no difference.

Note: `diff` prints `⚠ different models` for this comparison, because
`qwen3.5:9b` and `qwen3.5:9b-mlx` are different Ollama manifest names —
probelock has no way to know two differently-tagged models are the same
underlying weights on different backends. Technically correct given what the
tool can observe. A way to mark two differently-named lockfiles as comparable
would remove this false flag for the runtime-swap use case specifically.

## Regression replay

Bug: [ggml-org/llama.cpp#19703](https://github.com/ggml-org/llama.cpp/issues/19703) —
gpt-oss Jinja crash on multi-turn history when an assistant message carries
both `reasoning_content` and `tool_calls`. Fixed in
[PR #19704](https://github.com/ggml-org/llama.cpp/pull/19704), reintroduced by
[PR #18675](https://github.com/ggml-org/llama.cpp/pull/18675)
(commit `566059a26b0ce8faec4ea053605719d399c64cc5`), per
[ggml-org/llama.cpp#20500](https://github.com/ggml-org/llama.cpp/issues/20500).

Builds: `llama-server` at the commit before the regression ("good") and at the
regression commit ("bad"), plus a control pair at an unrelated, adjacent commit
(`ba2fd11c`, a CPU/ROPE cache change) to check for false positives.

Initial reproduction attempt did not trigger the crash on either build.
Two causes, found by reading the server logs and the relevant Jinja template
rather than guessing:

1. The handler with the bug (`common_chat_params_init_gpt_oss`) only activates
   when the model's chat template contains `<|channel|>`. The GGUF hosted
   today (`ggml-org/gpt-oss-20b-GGUF`) embeds a different template, so both
   builds used a newer, unaffected code path. Fixed by forcing the template
   explicitly with `--chat-template-file`.
2. The crash condition in the template is
   `{%- if message.content and message.thinking %}`. Jinja's `and` is a
   truthiness test; the first fixture used `"content": ""`, which is falsy, so
   the condition never fired regardless of code path. Fixed by setting
   `content` to a non-empty string on the assistant message that carries
   `reasoning_content` + `tool_calls`.

With both fixed, the regression commit returns the documented error verbatim
(`Cannot pass both content and thinking in an assistant message with tool
calls!`). All four trace-derived probes for that record score `0.0`, with the
HTTP 500 body captured in `ProbeResult.error`. The pre-regression build
returns a correct `check_calendar` tool call on the same request. The 32
schema-derived probes, which don't touch multi-turn history, are unaffected on
either build.

| | gate flags regression | gate does not flag regression |
|---|---|---|
| regressed commit (`566059a2`) | yes — `arg_validity`, `required_args`, `tool_selection`, `tool_discrimination` | — |
| control commit pair (`ba2fd11c`) | — | yes — PASS, +0.00 on every capability |

### Reproducing

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git worktree add ../llama.cpp-good 566059a26b0ce8faec4ea053605719d399c64cc5~1
git worktree add ../llama.cpp-bad  566059a26b0ce8faec4ea053605719d399c64cc5
# build each: cmake -B build -DGGML_METAL=ON && cmake --build build --target llama-server

# from this repo, against either build:
llama-server --jinja --flash-attn auto -hf ggml-org/gpt-oss-20b-GGUF --alias gptoss --port 8090
uv run probelock probe --tools examples/agent_tools.json \
    --traces fixtures/gptoss_regression_trace.json \
    --endpoint http://localhost:8090/v1 --model gptoss -o result.lock
```

If the model's own chat template no longer contains `<|channel|>` (as with the
current `gpt-oss-20b-GGUF`), add `--chat-template-file` pointing at
`models/templates/openai-gpt-oss-120b.jinja` from the llama.cpp checkout to
force the code path this test targets.

## Notes

- `llama-server -fa` requires an explicit value (`--flash-attn auto`) on
  recent builds; the bare flag misparses a following `-hf` argument as its
  value.
- `probelock probe --traces` required `--tools` at the time of this validation,
  even though traced probes carry their own embedded tool definitions. Fixed in
  v0.3.1: `--tools` is now optional whenever `--traces` or `--mined` supply probes.
- Running two `llama-server` instances concurrently on one GPU crashed one of
  them (Metal contention) mid-probe during testing. Don't run concurrent GPU
  inference workloads on a single-GPU host.
