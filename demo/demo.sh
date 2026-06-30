#!/usr/bin/env bash
# probelock — real demo against a local Ollama server.
#
# Leads with the realistic finding: a strong baseline (qwen3.5:9b) vs a small
# model that tool-calls well but can't emit structured output (lfm2.5-thinking:1.2b),
# then proves the live HTTP path and the all-or-nothing case (gemma3:1b, no tools).
# Every lockfile here is real; see DEMO.md for how they were produced.
set -u
cd "$(dirname "$0")/.."   # -> probelock project root

TOOLS=examples/agent_tools.json
QWEN=demo/qwen3.5-9b.lock
LFM=demo/lfm2.5-thinking.lock
GEMMA=demo/gemma3-1b.lock
ENDPOINT=${PROBELOCK_ENDPOINT:-http://localhost:11434/v1}

echo "### 1) Probes are DERIVED from your tool schemas — no test authoring"
uv run probelock derive --tools "$TOOLS"

echo
echo "### 2) The realistic finding: qwen3.5:9b -> lfm2.5-thinking:1.2b (both real)"
echo "#   A 1.2B model that tool-calls as well as a 9B — only structured_output drops."
uv run probelock diff "$QWEN" "$LFM"
uv run probelock gate --baseline "$QWEN" --candidate "$LFM" \
  || echo "(gate exited non-zero — one specific, real regression caught)"

echo
echo "### 3) Live over HTTP + the all-or-nothing case: gemma3:1b can't tool-call"
uv run probelock probe --tools "$TOOLS" --endpoint "$ENDPOINT" \
  --model gemma3:1b --runtime ollama -o "$GEMMA"
uv run probelock gate --baseline "$QWEN" --candidate "$GEMMA" \
  || echo "(gate exited non-zero — every tool capability collapsed)"
