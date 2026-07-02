"""The frozen, reviewable probe format produced by ``probelock ingest``.

A mined probe is a decision point lifted out of raw agent traffic: the verbatim context
(messages + tools) a model saw, plus one deterministic check describing what the recorded
model did there. It is NOT yet part of the battery — correctness inference from raw
traces is heuristic, so every mined probe lands with ``status: "pending"`` and only
becomes replayable after review (``probelock traces review``) flips it to ``accepted``.

Three categories, ordered from safest to most heuristic (this ordering drives the
auto-accept rules below — provenance determines trust):

  * ``schema_validity``  — replaying the context, the candidate's tool-call arguments
    must validate against the schema of whichever offered tool it calls. No correctness
    inference was needed to mine it (every tool-calling exchange qualifies), so it is
    the one category ``--auto-accept`` may activate without a human in the loop.
  * ``tool_selection``   — the candidate must call the specific tool the recorded model
    was confirmed to have used successfully. Mining inferred that confirmation
    (continuation / min-agreement, see ingest.py), so review or an explicit
    ``--auto-accept-all --i-know-what-im-doing`` is required.
  * ``no_tool``          — the candidate must answer in text, calling nothing. A
    mislabeled probe here freezes a model *mistake* as expected behavior and would
    punish candidates that fix it, so this category has NO auto-accept path at all:
    each probe must be individually accepted by a human.

Accepted probes convert to ordinary :class:`~probelock.models.Probe` objects under
dedicated ``traced_*`` capability names — deliberately distinct from the synthetic
capabilities so lockfiles, diffs, and gates report trace-derived scores separately
(a drop in multi-turn trace probes with stable synthetic probes is itself diagnostic).

Field note vs the design doc: the doc's frozen format carries a domain-style
``"capability": "calendar"`` grouping field. probelock has no domain concept; the
deterministic equivalent is the tool name, stored here as ``tool`` (it is also the
per-category sampling group key in ingest.py).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import Probe

CATEGORIES = ("schema_validity", "tool_selection", "no_tool")
STATUSES = ("pending", "accepted", "rejected")

CATEGORY_CAPABILITIES = {
    "schema_validity": "traced_schema_validity",
    "tool_selection": "traced_tool_selection",
    "no_tool": "traced_no_tool",
}

# The auto-accept trust ladder (see module docstring). AUTO_ACCEPT_SAFE may be activated
# by a bare --auto-accept; everything else except NEVER_AUTO_ACCEPT needs the explicit
# --auto-accept-all --i-know-what-im-doing pair; NEVER_AUTO_ACCEPT has no batch path.
AUTO_ACCEPT_SAFE = frozenset({"schema_validity"})
NEVER_AUTO_ACCEPT = frozenset({"no_tool"})


@dataclass
class MinedProbe:
    """One frozen decision point. Mutable on purpose: review flips ``status`` in place."""

    id: str  # "trace:<session12>:t<turn>" — unique per (id, category), not per id alone
    category: str
    messages: List[Dict[str, Any]]
    tools: List[Dict[str, Any]]
    tool: Optional[str] = None  # expected tool (tool_selection) / recorded tool (schema_validity)
    status: str = "pending"
    provenance: Dict[str, Any] = field(default_factory=dict)
    sensitive: bool = True
    # Recorded behavior (redacted), for the SimulatedClient and the review display only —
    # real scoring never reads this, same contract as Probe.reference.
    reference: Dict[str, Any] = field(default_factory=dict)

    def check(self) -> Dict[str, Any]:
        """The deterministic check, derived from (category, tool) — kept computed rather
        than stored so the two can never disagree in memory; serialization writes it out
        for human readers of the JSON file."""
        if self.category == "tool_selection":
            return {"type": "calls_tool", "tool": self.tool}
        if self.category == "no_tool":
            return {"type": "no_tool_call"}
        return {"type": "schema_valid_call"}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "status": self.status,
            # `tool` also appears top-level: for schema_validity the check doesn't pin a
            # tool (any offered one may be called), but the RECORDED tool still matters —
            # it's the sampling group key and the simulator's target.
            "tool": self.tool,
            "check": self.check(),
            "context": {"messages": self.messages, "tools": self.tools},
            "provenance": self.provenance,
            "sensitive": self.sensitive,
            "reference": self.reference,
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_mined(path) -> List[MinedProbe]:
    """Load and validate a mined-probes file. Raises FileNotFoundError / ValueError /
    json.JSONDecodeError on malformed input — callers wrap these into a clean CLI exit,
    the same convention as load_trace_records."""
    data = json.loads(Path(path).read_text())
    _require(
        isinstance(data, dict) and isinstance(data.get("probes"), list),
        "mined probes file must be a JSON object with a 'probes' array",
    )
    probes: List[MinedProbe] = []
    seen = set()
    for i, entry in enumerate(data["probes"]):
        _require(isinstance(entry, dict), f"probe #{i} must be an object")
        category = entry.get("category")
        _require(category in CATEGORIES, f"probe #{i} has unknown category {category!r}")
        status = entry.get("status", "pending")
        _require(status in STATUSES, f"probe #{i} has unknown status {status!r}")
        probe_id = entry.get("id")
        _require(bool(probe_id) and isinstance(probe_id, str), f"probe #{i} needs a string 'id'")
        key = (probe_id, category)
        _require(key not in seen, f"duplicate probe (id, category): {key!r}")
        seen.add(key)
        context = entry.get("context")
        _require(
            isinstance(context, dict) and isinstance(context.get("messages"), list),
            f"probe #{i} needs a 'context' object with a 'messages' array",
        )
        tools = context.get("tools") or []
        _require(
            isinstance(tools, list) and all(isinstance(t, dict) for t in tools),
            f"probe #{i} 'context.tools' must be a list of tool objects",
        )
        check = entry.get("check") if isinstance(entry.get("check"), dict) else {}
        tool = entry.get("tool") or check.get("tool")
        # A tool_selection probe without an expected tool would convert to
        # expected_tool=None, which the scorer treats as "any call matches" — a probe
        # that can never fail. Reject the file instead.
        _require(
            category != "tool_selection" or bool(tool),
            f"probe #{i} is tool_selection but names no expected tool",
        )
        probes.append(
            MinedProbe(
                id=probe_id,
                category=category,
                messages=list(context.get("messages") or []),
                tools=list(tools),
                tool=str(tool) if tool is not None else None,
                status=status,
                provenance=dict(entry.get("provenance") or {}),
                # Unknown sensitivity is treated as sensitive — the safe direction for
                # a hand-edited file that dropped the flag.
                sensitive=bool(entry.get("sensitive", True)),
                reference=dict(entry.get("reference") or {}),
            )
        )
    return probes


def save_mined(probes: List[MinedProbe], path, header: Optional[Dict[str, Any]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"version": 1, **(header or {})}
    payload["probes"] = [p.to_dict() for p in probes]
    path.write_text(json.dumps(payload, indent=2) + "\n")


def mined_fingerprint(probes: List[MinedProbe]) -> str:
    """Order-invariant hash of the probes' replayable identity (context + check), NOT of
    review metadata — re-reviewing without changing which probes exist keeps it stable.
    Which probes were *included* in a run is captured by membership: callers fingerprint
    the accepted-and-included subset, mirroring traces_fingerprint's role of letting a
    diff flag baseline/candidate pairs whose trace inputs differ."""
    canonical = [
        {
            "id": p.id,
            "category": p.category,
            "tool": p.tool,
            "messages": p.messages,
            "tools": p.tools,
        }
        for p in probes
    ]
    canonical.sort(key=lambda rec: json.dumps(rec, sort_keys=True, separators=(",", ":")))
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _tool_parameters(tools: List[Dict[str, Any]], name: Optional[str]) -> Optional[Dict[str, Any]]:
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function")
        if isinstance(fn, dict) and fn.get("name") == name:
            params = fn.get("parameters")
            return params if isinstance(params, dict) else {}
    return None


def to_probe(mp: MinedProbe) -> Probe:
    """Convert an accepted mined probe into an ordinary Probe. The capability name is the
    traced_* variant so scores aggregate separately from the synthetic battery."""
    capability = CATEGORY_CAPABILITIES[mp.category]
    schema = _tool_parameters(mp.tools, mp.tool)
    return Probe(
        id=f"{capability}::{mp.id}",
        capability=capability,
        description=(
            f"[mined] {mp.category} from {mp.provenance.get('sessions', '?')} session(s) "
            f"via {mp.provenance.get('rule', '?')}"
        ),
        messages=mp.messages,
        tools=mp.tools,
        expected_tool=mp.tool if mp.category == "tool_selection" else None,
        # For schema_validity the scorer looks schemas up from probe.tools (the candidate
        # may validly call any offered tool); probe.schema carries the RECORDED tool's
        # schema anyway so the SimulatedClient can craft pass/fail responses against it.
        schema=schema if mp.category in ("schema_validity", "tool_selection") else None,
        reference=mp.reference,
    )


def accepted_probes(probes: List[MinedProbe]) -> List[MinedProbe]:
    return [p for p in probes if p.status == "accepted"]


def auto_accept(
    probes: List[MinedProbe], categories: frozenset, accept_all: bool = False
) -> Dict[str, int]:
    """Batch-accept pending probes per the trust ladder. Returns counts per category,
    plus a ``no_tool_skipped`` entry when accept_all ran into the no-auto-accept wall —
    callers surface that count so the skip is never silent."""
    counts: Dict[str, int] = {}
    for p in probes:
        if p.status != "pending":
            continue
        if p.category in NEVER_AUTO_ACCEPT:
            if accept_all:
                counts["no_tool_skipped"] = counts.get("no_tool_skipped", 0) + 1
            continue
        if accept_all or p.category in categories:
            p.status = "accepted"
            p.provenance["review"] = "auto-accepted"
            counts[p.category] = counts.get(p.category, 0) + 1
    return counts


def edit_expected_tool(mp: MinedProbe, new_tool: str) -> None:
    """Review's ``e`` key: repoint a tool_selection probe at a different expected tool.
    Only tools actually offered in the frozen context are valid targets — the probe
    replays that exact toolset, so anything else could never pass."""
    if mp.category != "tool_selection":
        raise ValueError("only tool_selection probes have an expected tool to edit")
    offered = [t.get("function", {}).get("name") for t in mp.tools]
    if new_tool not in offered:
        raise ValueError(
            f"'{new_tool}' is not among the tools offered in this context "
            f"({', '.join(n for n in offered if n) or 'none'})"
        )
    if new_tool != mp.tool:
        mp.provenance["edited_from"] = mp.tool
        mp.tool = new_tool
    mp.status = "accepted"
    mp.provenance["review"] = "edited"
