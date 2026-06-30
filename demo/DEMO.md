# probelock — recorded demo (Ollama)

These runs use a local Ollama server, over HTTP via `probelock`'s `HttpClient`.

- **Replay the recording:** `asciinema play demo/probelock-demo.cast`
- **Re-run it live** (needs Ollama with `gemma3:1b` pulled): `bash demo/demo.sh`

## qwen3.5:9b → lfm2.5-thinking:1.2b

`qwen3.5:9b` against a 1.2B model that supports tool-calling. Both lockfiles are
committed, so this reproduces offline:

```bash
uv run probelock diff demo/qwen3.5-9b.lock demo/lfm2.5-thinking.lock
```

```
          qwen3.5:9b @ native (ollama)  →  lfm2.5-thinking:1.2b @ native (ollama)
 Capability            Baseline   Candidate       Δ   Status
 arg_validity              1.00        1.00   +0.00   ok
 arity_robustness          1.00        0.33   -0.67   REGRESSION
 format_adherence          0.50        0.50   +0.00   ok
 needle_in_tools           1.00        1.00   +0.00   ok
 no_hallucinated_tool      1.00        1.00   +0.00   ok
 required_args             1.00        1.00   +0.00   ok
 structured_output         1.00        0.33   -0.67   REGRESSION
 tool_discrimination       1.00        1.00   +0.00   ok
 tool_permission           1.00        1.00   +0.00   ok
 tool_restraint            1.00        1.00   +0.00   ok
 tool_selection            1.00        1.00   +0.00   ok
```

The 1.2B model matches `qwen3.5:9b` on selection, discrimination, `needle_in_tools`
(the toolset is padded to 18), `arg_validity`, `required_args`, and the three
safety probes. `structured_output` and `arity_robustness` drop `1.00 → 0.33`. It is
a thinking model; this run exercised the `<think>`-stripping path, and 0 probes
errored.

## qwen3.5:9b → gemma3:1b (no tool support)

`gemma3:1b` does not support tool-calling, so every capability that requires
emitting a call scores 0 (a 400 is recorded as a per-probe 0). The gate exits
non-zero:

```
          qwen3.5:9b @ native (ollama)  →  gemma3:1b @ native (ollama)
 Capability            Baseline   Candidate       Δ   Status
 arg_validity              1.00        0.00   -1.00   REGRESSION
 arity_robustness          1.00        0.00   -1.00   REGRESSION
 format_adherence          0.50        1.00   +0.50   improved
 needle_in_tools           1.00        0.00   -1.00   REGRESSION
 no_hallucinated_tool      1.00        1.00   +0.00   ok
 required_args             1.00        0.00   -1.00   REGRESSION
 structured_output         1.00        0.00   -1.00   REGRESSION
 tool_discrimination       1.00        0.00   -1.00   REGRESSION
 tool_permission           1.00        1.00   +0.00   ok
 tool_restraint            1.00        1.00   +0.00   ok
 tool_selection            1.00        0.00   -1.00   REGRESSION

FAIL — capabilities regressed or removed: arg_validity, arity_robustness,
needle_in_tools, required_args, structured_output, tool_discrimination, tool_selection
```

The three negative (safety) probes stay `1.00`: a model that can't call tools
can't over-trigger, call a forbidden tool, or hallucinate one. `gemma3:1b` scores
`1.00` on `format_adherence` vs `0.50` for `qwen3.5:9b`. The probes that send tools
report:

```
27 probe(s) errored at the API level. e.g. HTTP 400:
{"error":{"message":"...gemma3:1b does not support tools",...}}
```

## Shareable HTML report

`diff --format html` emits a self-contained page (no external assets) for posting a
regression to a teammate. A sample is committed at
[`report-qwen-vs-lfm.html`](report-qwen-vs-lfm.html):

```bash
probelock diff demo/qwen3.5-9b.lock demo/lfm2.5-thinking.lock --format html > report.html
```

## Cross-model vs within-model

Both comparisons above are **cross-model** (the box has one quant per model), so
`diff`/`gate` print the ⚠ different-MODELS warning, and `gate --require-same-model`
would turn that into a hard exit-2. probelock's main use is a within-model swap
(same model, different quant/runtime). The simulated example uses one model,
`llama-3.1-8b`, at Q8 → Q4:

```bash
uv run probelock probe --tools examples/agent_tools.json --simulate fixtures/profile_q8.json -o q8.lock
uv run probelock probe --tools examples/agent_tools.json --simulate fixtures/profile_q4.json -o q4.lock
uv run probelock gate --baseline q8.lock --candidate q4.lock --confidence 0.95
```

## Provenance — how the committed lockfiles were produced

The three `.lock` files in this folder were produced with:

```bash
# strong baseline (6.6 GB; cold load is slow, hence --timeout)
uv run probelock probe --tools examples/agent_tools.json \
  --endpoint http://localhost:11434/v1 --model qwen3.5:9b --runtime ollama \
  --timeout 240 -o demo/qwen3.5-9b.lock

# small thinking model that tool-calls
uv run probelock probe --tools examples/agent_tools.json \
  --endpoint http://localhost:11434/v1 --model lfm2.5-thinking:1.2b --runtime ollama \
  --timeout 240 -o demo/lfm2.5-thinking.lock

# model with no tool support
uv run probelock probe --tools examples/agent_tools.json \
  --endpoint http://localhost:11434/v1 --model gemma3:1b --runtime ollama \
  -o demo/gemma3-1b.lock
```

Point `--endpoint` at any OpenAI-compatible server (llama.cpp, LM Studio, vLLM, an
MLX server) to probe your own models.
