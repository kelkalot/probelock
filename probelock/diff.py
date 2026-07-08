"""Diff two capability lockfiles — the core of probelock.

The unit of comparison is a model against *itself* across a swap (a new version,
a different quantization, a different runtime). A capability that drops by more
than ``max_drop`` is a regression. This is what promptfoo's absolute,
hand-authored cross-provider matrix does not give you.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .models import Lockfile
from .scoring import NEGATIVE_CAPABILITIES
from .stats import is_significant_regression

# Runtime/backend and quantization markers that some registries (Ollama especially)
# bake into the model id as a trailing "-<token>". They describe HOW a model runs, not
# WHICH model it is — so a within-model runtime or quant swap must not read as a
# cross-model comparison just because the id string changed.
_RUNTIME_MARKERS = frozenset({
    "mlx", "gguf", "ggml", "metal", "cuda", "rocm", "cpu", "vulkan", "vllm",
    "mps", "coreml", "onnx", "tensorrt", "sycl", "hip",
})
# Quantization tags, including the importance-matrix "IQ" family (iq2_xs, iq4_nl, …)
# that Ollama/llama.cpp bake into the id — missing them made a within-model quant swap
# read as cross-model.
_QUANT_MARKER = re.compile(r"^(iq\d.*|q\d.*|fp?\d+|f\d+|bf\d+|int\d+|awq|gptq)$")


def model_family(model: str, quant: str = "", runtime: str = "") -> str:
    """Model identity with runtime/quant variant markers stripped from the id.

    So 'qwen3.5:9b' and 'qwen3.5:9b-mlx' compare equal (a runtime swap), and
    'llama3.1:8b-q8_0' and 'llama3.1:8b-q4_K_M' compare equal (a quant swap), while
    genuinely different models ('qwen3.5:9b' vs 'qwen3.5:32b') stay distinct. The
    first hyphen token (the base name:tag) is never stripped; only trailing tokens
    that are known runtime markers, look like a quant tag, or equal the lockfile's
    own quant/runtime field are dropped."""
    tokens = model.strip().lower().split("-")
    fields = {f.strip().lower() for f in (quant, runtime) if f.strip()}
    core = tokens[:1]
    for tok in tokens[1:]:
        if tok in _RUNTIME_MARKERS or _QUANT_MARKER.match(tok) or tok in fields:
            continue
        core.append(tok)
    return "-".join(core)


@dataclass(frozen=True)
class DiffRow:
    capability: str
    baseline: Optional[float]
    candidate: Optional[float]
    delta: Optional[float]
    status: str  # "ok"|"regression"|"noisy"|"improved"|"added"|"removed"
    significant: Optional[bool] = None  # set when --confidence was applied


@dataclass(frozen=True)
class DiffResult:
    rows: List[DiffRow]
    tools_changed: bool
    traces_changed: bool
    max_drop: float

    # A dropped capability ("removed") is a silent regression too — the gate
    # exists to catch a swap that loses a capability, not just one that weakens it.
    _FAIL_STATUSES = ("regression", "removed")

    @property
    def regressed(self) -> bool:
        return any(r.status in self._FAIL_STATUSES for r in self.rows)

    @property
    def regressions(self) -> List[DiffRow]:
        return [r for r in self.rows if r.status in self._FAIL_STATUSES]


def _trials(lock: Lockfile, cap: str) -> int:
    """Bernoulli trials backing a capability score: probes-in-cap x samples."""
    n = sum(1 for r in lock.results if r.capability == cap)
    return n * max(getattr(lock, "samples", 1) or 1, 1)


def error_derived_negative_capabilities(lock: Lockfile) -> Dict[str, Tuple[int, int]]:
    """Negative capabilities (tool_restraint, tool_permission, no_hallucinated_tool) whose
    score includes at least one probe that errored at the API level.

    A negative capability scores 1.0 on a ProbeError under the rationale "a model that
    can't even accept tools can't misbehave" (see scoring.NEGATIVE_CAPABILITIES) — genuine
    on its own, but that 1.0 measures nothing if the endpoint simply rejected the request
    rather than the model demonstrating restraint. Surfacing this lets a diff/gate flag a
    baseline whose "perfect" safety score is an artifact of a broken endpoint, not a
    property worth holding a later, genuinely-responding candidate to.

    Returns {capability: (errored_count, total_probes)} for affected capabilities only.
    """
    by_cap: Dict[str, List[bool]] = {}
    for r in lock.results:
        if r.capability in NEGATIVE_CAPABILITIES:
            by_cap.setdefault(r.capability, []).append(bool(r.error))
    return {
        cap: (sum(flags), len(flags))
        for cap, flags in by_cap.items()
        if any(flags)
    }


def diff_lockfiles(
    baseline: Lockfile,
    candidate: Lockfile,
    max_drop: float = 0.05,
    confidence: Optional[float] = None,
) -> DiffResult:
    """Per-capability deltas. With ``confidence`` set, a drop past ``max_drop`` is
    only a "regression" if it is statistically significant for the trial count;
    otherwise it is "noisy" (shown, but not gated on)."""
    caps = sorted(set(baseline.capabilities) | set(candidate.capabilities))
    rows: List[DiffRow] = []
    for cap in caps:
        b = baseline.capabilities.get(cap)
        c = candidate.capabilities.get(cap)
        significant = None
        if b is None:
            status, delta = "added", None
        elif c is None:
            status, delta = "removed", None
        else:
            delta = round(c - b, 4)
            if delta < -max_drop:
                if confidence is None:
                    status = "regression"
                else:
                    significant = is_significant_regression(
                        b, _trials(baseline, cap), c, _trials(candidate, cap), confidence
                    )
                    status = "regression" if significant else "noisy"
            elif delta > max_drop:
                status = "improved"
            else:
                status = "ok"
        rows.append(DiffRow(cap, b, c, delta, status, significant))
    return DiffResult(
        rows=rows,
        tools_changed=baseline.tools_fingerprint != candidate.tools_fingerprint,
        traces_changed=baseline.traces_fingerprint != candidate.traces_fingerprint,
        max_drop=max_drop,
    )
