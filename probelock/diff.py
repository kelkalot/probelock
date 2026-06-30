"""Diff two capability lockfiles — the core of probelock.

The unit of comparison is a model against *itself* across a swap (a new version,
a different quantization, a different runtime). A capability that drops by more
than ``max_drop`` is a regression. This is what promptfoo's absolute,
hand-authored cross-provider matrix does not give you.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .models import Lockfile
from .stats import is_significant_regression


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
        max_drop=max_drop,
    )
