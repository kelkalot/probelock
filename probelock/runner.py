"""Run a probe battery against a client and build a capability lockfile."""

from __future__ import annotations

from typing import Dict, List, Optional

from .clients import ProbeError
from .models import Lockfile, Probe, ProbeResult
from .scoring import NEGATIVE_CAPABILITIES, score


def run_probes(
    client,
    probes: List[Probe],
    tools_fingerprint: str,
    version: str,
    samples: int = 1,
    traces_fingerprint: Optional[str] = None,
) -> Lockfile:
    client.prepare(probes)
    # Only count samples that are genuinely independent. A deterministic client
    # (temp 0, or the simulator) returns identical results, so N>1 there would
    # record phantom trials that inflate the significance test. Clamp to 1.
    samples = max(1, samples) if getattr(client, "produces_variance", False) else 1

    results: List[ProbeResult] = []
    for probe in probes:
        # Run the probe `samples` times; the score is the pass RATE. A per-sample
        # ProbeError counts as a 0 for that sample; the probe is only error-tagged
        # if every sample failed at the API level (a fatal ClientError still aborts).
        scores: List[float] = []
        errors = 0
        last_error = None
        for _ in range(samples):
            try:
                scores.append(score(probe, client.complete(probe)))
            except ProbeError as exc:
                errors += 1
                last_error = str(exc)
                # Negative probes (restraint/permission/no-hallucination) measure the
                # ABSENCE of bad behavior: a model that can't even accept tools cannot
                # misbehave, so an API rejection scores 1.0. It is STILL counted as an
                # API error above, so the all-errored fatal guard sees a broken run.
                scores.append(1.0 if probe.capability in NEGATIVE_CAPABILITIES else 0.0)
        error = last_error if errors == samples else None
        results.append(
            ProbeResult(probe.id, probe.capability, sum(scores) / len(scores), error=error)
        )

    aggregate: Dict[str, List[float]] = {}
    for r in results:
        aggregate.setdefault(r.capability, []).append(r.score)
    capabilities = {
        cap: round(sum(scores) / len(scores), 4)
        for cap, scores in sorted(aggregate.items())
    }

    md = client.metadata
    return Lockfile(
        label=md.get("label", ""),
        model=md.get("model", ""),
        quant=md.get("quant", ""),
        runtime=md.get("runtime", ""),
        tools_fingerprint=tools_fingerprint,
        probelock_version=version,
        capabilities=capabilities,
        results=results,
        n_probes=len(results),
        samples=samples,
        traces_fingerprint=traces_fingerprint,
    )
