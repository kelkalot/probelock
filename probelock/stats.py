"""Significance test for the statistically-aware gate.

A capability score is a proportion backed by ``trials`` Bernoulli trials
(probes-in-capability x samples-per-probe). With few trials, a score is noisy:
3 probes x 1 sample quantizes to {0, 0.33, 0.67, 1.0}, so a single flip moves a
capability by 0.33 — far past the default 0.05 gate. This module lets the gate
require that a drop is not just past the threshold but statistically real for the
trial count, using a one-sided two-proportion z-test.
"""

from __future__ import annotations

import math
from statistics import NormalDist


def is_significant_regression(
    p_baseline: float,
    trials_baseline: int,
    p_candidate: float,
    trials_candidate: int,
    confidence: float,
) -> bool:
    """True if the candidate proportion is significantly BELOW the baseline at the
    given one-sided confidence, per a pooled two-proportion z-test.

    When trial counts are unknown (0), we cannot test, so we return True
    (fail safe: do not silence a possible regression for lack of metadata)."""
    if trials_baseline <= 0 or trials_candidate <= 0:
        return True
    if p_candidate >= p_baseline:
        return False
    # Defensive: NormalDist.inv_cdf is only defined on (0, 1). Callers validate,
    # but clamp here too so a stray value can never raise instead of decide.
    confidence = min(max(confidence, 1e-9), 1 - 1e-9)

    passes_b = round(p_baseline * trials_baseline)
    passes_c = round(p_candidate * trials_candidate)
    pooled = (passes_b + passes_c) / (trials_baseline + trials_candidate)
    var = pooled * (1 - pooled) * (1 / trials_baseline + 1 / trials_candidate)
    if var <= 0:
        # No pooled variance only happens at p=0 or p=1 for both; with a genuine
        # drop the pool sits strictly between, so treat any drop here as real.
        return True
    z = (p_baseline - p_candidate) / math.sqrt(var)
    return z > NormalDist().inv_cdf(confidence)
