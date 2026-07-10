"""Core data model for probelock.

Two halves, deliberately separated:

  * A :class:`Client` talks to a model and returns a :class:`ResponseMessage`.
    This is the only nondeterministic, model-touching part.
  * Everything else — deriving :class:`Probe` objects from tool schemas, scoring
    a response, building a :class:`Lockfile`, diffing two lockfiles — is pure and
    deterministic. No LLM judge anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Probe:
    """One deterministic capability check, derived from a tool schema."""

    id: str
    capability: str
    description: str
    messages: List[Dict[str, Any]]
    tools: List[Dict[str, Any]]
    expected_tool: Optional[str] = None
    schema: Optional[Dict[str, Any]] = None
    expected_text: Optional[str] = None
    # OpenAI `response_format` for this probe, if any — the native structured-output API
    # path (json_mode capability). None for every schema/prompt-based probe. Real
    # clients forward it verbatim; the simulator ignores it (it crafts by capability).
    response_format: Optional[Dict[str, Any]] = None
    # Simulator-only reference (the scorer never reads this): how a *correct*
    # answer looks, so the SimulatedClient can craft pass/fail responses.
    reference: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: str  # raw JSON string, exactly as an OpenAI-style API returns it


@dataclass(frozen=True)
class ResponseMessage:
    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)


@dataclass(frozen=True)
class ProbeResult:
    probe_id: str
    capability: str
    score: float  # 0.0 .. 1.0, deterministic
    error: Optional[str] = None  # set when the probe failed at the API level


# Lockfile on-disk schema version — distinct from probelock_version (the tool version).
# Bumped only on a BREAKING format change, per STABILITY.md; a reader older than the
# file's version refuses it rather than silently mis-parsing. Committed lockfiles across
# a 1.x series all read as LOCKFILE_FORMAT 1.
LOCKFILE_FORMAT = 1


@dataclass
class Lockfile:
    label: str
    model: str
    quant: str
    runtime: str
    tools_fingerprint: str
    probelock_version: str
    capabilities: Dict[str, float]
    results: List[ProbeResult]
    n_probes: int
    samples: int = 1  # samples per probe (>1 makes per-probe scores pass-rates)
    generated_at: Optional[str] = None
    lockfile_format: int = LOCKFILE_FORMAT
    # Fingerprint of the trace-export file (probelock/traces.py), if any traced probes
    # were included — None when the battery is purely schema-derived. Lets a diff flag a
    # baseline/candidate pair whose real-trace inputs differ, the same way tools_fingerprint
    # already flags a changed toolset.
    traces_fingerprint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lockfile_format": self.lockfile_format,
            "probelock_version": self.probelock_version,
            "label": self.label,
            "model": self.model,
            "quant": self.quant,
            "runtime": self.runtime,
            "tools_fingerprint": self.tools_fingerprint,
            "traces_fingerprint": self.traces_fingerprint,
            "n_probes": self.n_probes,
            "samples": self.samples,
            "generated_at": self.generated_at,
            "capabilities": self.capabilities,
            "results": [
                {
                    "probe_id": r.probe_id,
                    "capability": r.capability,
                    "score": r.score,
                    **({"error": r.error} if r.error else {}),
                }
                for r in self.results
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Lockfile":
        """Parse a lockfile dict defensively. Raises ValueError on bad shape so
        callers can report a clean message instead of a stray KeyError/TypeError."""
        if not isinstance(data, dict):
            raise ValueError("lockfile must be a JSON object")

        # A file written by a NEWER probelock (higher format) may use fields this reader
        # cannot honor — refuse it clearly rather than mis-scoring a gate. A missing
        # field means a pre-1.0 lockfile, which is format 1.
        try:
            fmt = int(data.get("lockfile_format", LOCKFILE_FORMAT))
        except (TypeError, ValueError):
            fmt = LOCKFILE_FORMAT
        if fmt > LOCKFILE_FORMAT:
            raise ValueError(
                f"lockfile format {fmt} is newer than this probelock supports "
                f"(max {LOCKFILE_FORMAT}); upgrade probelock to read it"
            )

        caps_raw = data.get("capabilities") or {}
        if not isinstance(caps_raw, dict):
            raise ValueError("lockfile 'capabilities' must be an object")
        capabilities = {}
        for key, value in caps_raw.items():
            try:
                capabilities[str(key)] = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"capability '{key}' has a non-numeric score: {value!r}")

        results = []
        for entry in data.get("results") or []:
            if not isinstance(entry, dict):
                continue
            try:
                score = float(entry.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            results.append(
                ProbeResult(
                    str(entry.get("probe_id", "")),
                    str(entry.get("capability", "")),
                    score,
                    entry.get("error"),
                )
            )

        try:
            n_probes = int(data.get("n_probes", len(results)))
        except (TypeError, ValueError):
            n_probes = len(results)
        try:
            samples = max(1, int(data.get("samples", 1)))
        except (TypeError, ValueError):
            samples = 1

        return cls(
            label=str(data.get("label", "")),
            model=str(data.get("model", "")),
            quant=str(data.get("quant", "")),
            runtime=str(data.get("runtime", "")),
            tools_fingerprint=str(data.get("tools_fingerprint", "")),
            probelock_version=str(data.get("probelock_version", "")),
            capabilities=capabilities,
            results=results,
            n_probes=n_probes,
            samples=samples,
            generated_at=data.get("generated_at"),
            traces_fingerprint=data.get("traces_fingerprint"),
            lockfile_format=fmt,
        )
