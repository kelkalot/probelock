"""Track a capability across an ordered ladder of lockfiles (N-way, not pairwise).

``diff`` answers "did this one swap regress anything". ``trend`` answers the local-models
question ``diff`` cannot: given a quant ladder (F16 → Q8 → Q6 → Q5 → Q4 → Q3 → Q2), or the
same model over several dates, *where* does each capability hold and *where* does it
cliff. The lockfiles are compared in the order given — that order is the axis; this
module does not reorder them.

Everything here is pure and deterministic, like the rest of probelock: no LLM, no
network. A row's ``status`` summarizes the whole ladder for that capability:

  * ``regressed`` — net drop (last minus first) worse than ``max_drop``
  * ``improved``  — net gain beyond ``max_drop``
  * ``unstable``  — net change within ``max_drop`` but some adjacent step dropped past
    it (dipped and recovered — noise a two-point diff of the endpoints would miss)
  * ``stable``    — holds across the whole ladder
  * ``removed``   — present on an earlier rung but gone from the LAST one; the clearest
    possible regression, and what ``diff``/``gate`` already call "removed"
  * ``partial``   — present in fewer than two lockfiles, so no trend can be computed
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .models import Lockfile


@dataclass(frozen=True)
class TrendRow:
    capability: str
    values: List[Optional[float]]  # one per lockfile, in ladder order; None if absent
    delta: Optional[float]         # last present minus first present; None if < 2 present
    worst_step: Optional[float]    # most negative adjacent step among present values
    status: str
    monotonic: bool                # among present values (non-increasing or non-decreasing)


@dataclass(frozen=True)
class TrendResult:
    labels: List[str]              # column headers, one per lockfile
    rows: List[TrendRow]
    max_drop: float

    # A capability dropped from the last rung is a silent regression too, exactly as
    # diff/gate treat a "removed" capability.
    _FAIL_STATUSES = ("regressed", "removed")

    @property
    def regressed(self) -> List[TrendRow]:
        return [r for r in self.rows if r.status in self._FAIL_STATUSES]

    @property
    def unstable(self) -> List[TrendRow]:
        return [r for r in self.rows if r.status == "unstable"]


def trend_lockfiles(
    locks: List[Lockfile], labels: List[str], max_drop: float = 0.05
) -> TrendResult:
    """Build the per-capability trend across ``locks`` (already in ladder order)."""
    caps = sorted({cap for lock in locks for cap in lock.capabilities})
    rows: List[TrendRow] = []
    for cap in caps:
        values = [lock.capabilities.get(cap) for lock in locks]
        present = [v for v in values if v is not None]
        if values[-1] is None and any(v is not None for v in values[:-1]):
            # Scored earlier, absent from the final rung: the capability was dropped.
            # Filtering None out of `present` would otherwise hide this as stable/
            # improved — the exact opposite of the truth.
            rows.append(TrendRow(cap, values, None, None, "removed", True))
            continue
        if len(present) < 2:
            rows.append(TrendRow(cap, values, None, None, "partial", True))
            continue
        delta = round(present[-1] - present[0], 4)
        steps = [round(present[i + 1] - present[i], 4) for i in range(len(present) - 1)]
        worst_step = min(steps)
        non_increasing = all(s <= 0 for s in steps)
        non_decreasing = all(s >= 0 for s in steps)
        monotonic = non_increasing or non_decreasing
        if delta < -max_drop:
            status = "regressed"
        elif delta > max_drop:
            status = "improved"
        elif worst_step < -max_drop:
            # net-flat but dipped along the way — a real signal the endpoints hide.
            status = "unstable"
        else:
            status = "stable"
        rows.append(TrendRow(cap, values, delta, worst_step, status, monotonic))
    return TrendResult(labels, rows, max_drop)
