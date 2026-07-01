"""Derive probes from real, already-recorded agent conversations — an addition to
derive_probes()'s synthetic battery, not a replacement.

Why a separate, tiny schema instead of parsing OpenTelemetry directly: OTel's own span
attribute layout is not stable across libraries/versions (e.g. litellm moved where it puts
request/response attributes in v1.81.0, and has a newer, differently-shaped opt-in "OTel v2"
integration), and there's no simple static export target — OTel is normally piped live to a
collector. Coupling probelock's core to that wire format would be a maintenance liability.

So probelock defines its own minimal, stable trace-record schema (mirroring Probe /
ResponseMessage directly) and treats "getting from your OTel backend to this schema" as a
one-time conversion you own — see examples/otel_traces_to_probelock.py for a documented
recipe. A trace-export file looks like:

    {
      "version": 1,
      "source": "litellm-otel",
      "records": [
        {
          "id": "checkout-flow-turn-3",
          "messages": [{"role": "user", "content": "..."}, ...],
          "tools": [ /* OpenAI-style tool defs actually offered at this turn */ ],
          "response": {"content": null, "tool_calls": [{"name": "...", "arguments": "{...}"}]}
        }
      ]
    }

Trust model: the recorded response is treated as ground truth (mine traces from a model
you already trust). Unlike a tool schema, a trace-export file contains real conversation
content — review and redact it before committing, the same way you'd review any fixture
containing real data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonschema

from .models import Probe, ResponseMessage, ToolCall

# Capabilities built from a recorded TOOL CALL, actually iterated by derive_traced_probes
# below (a single source of truth — not a separately-maintained list that could drift).
# structured_output is built from a recorded TEXT response instead (no tool call at all),
# so it has its own branch and isn't part of this tuple; TRACED_CAPABILITIES below is the
# union of both, for anything that just wants the full "what capabilities can traces
# populate" answer (e.g. the README, or a future `derive` summary).
#
# Left to the synthetic battery (probes.derive_probes) instead:
#   - needle_in_tools / tool_permission / no_hallucinated_tool / tool_restraint need a
#     synthetic perturbation (an injected distractor tool, a forbidden-tool instruction, a
#     removed tool) that a passively recorded trace doesn't naturally contain.
#   - format_adherence needs an exact-text prompt, not a tool-calling decision point.
#   - arity_robustness needs its own explicit "fill EVERY parameter, including optional
#     ones" instruction to mean anything; a real conversation's messages were never asked
#     for that, so replaying them would just test whichever optional fields happened to
#     get filled in that one exchange, not robustness.
_TOOL_CALL_CAPABILITIES = ("tool_selection", "tool_discrimination", "arg_validity", "required_args")
TRACED_CAPABILITIES = _TOOL_CALL_CAPABILITIES + ("structured_output",)


@dataclass(frozen=True)
class TraceRecord:
    """One real agent decision point: the context a model saw, and what it actually did."""

    id: str
    messages: List[Dict[str, Any]]
    tools: List[Dict[str, Any]]
    response: ResponseMessage


def _parse_response(data: Dict[str, Any]) -> ResponseMessage:
    calls = []
    for tc in data.get("tool_calls") or []:
        # `or` (not `.get(key, default)`) on purpose: an explicit null/empty name or
        # arguments means the same thing as absent here, and `.get(key, default)` only
        # applies the default when the key is missing, not when it's present-but-null —
        # that would otherwise turn a null name into the literal string "None".
        name = tc.get("name") or ""
        args = tc.get("arguments") or "{}"
        if not isinstance(args, str):  # tolerate a dict, as clients.py already does
            args = json.dumps(args)
        calls.append(ToolCall(str(name), args))
    return ResponseMessage(content=data.get("content"), tool_calls=calls)


def _hash_json(obj: Any, length: int) -> str:
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:length]


def _default_id(entry: Dict[str, Any]) -> str:
    return _hash_json(entry, 12)


def load_trace_records(path) -> List[TraceRecord]:
    """Load and validate a committed trace-export file (see module docstring). Raises
    FileNotFoundError / ValueError / json.JSONDecodeError on malformed input — callers
    wrap these into a clean CLI exit, the same convention as _load_tools/_load_json_or_exit
    in cli.py."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or not isinstance(data.get("records"), list):
        raise ValueError("traces file must be a JSON object with a 'records' array")
    records = []
    seen_ids = set()
    for i, entry in enumerate(data["records"]):
        if not isinstance(entry, dict):
            raise ValueError(f"record #{i} must be an object")
        if not isinstance(entry.get("response"), dict):
            raise ValueError(f"record #{i} is missing an object 'response'")
        # Default only when "id" is truly absent or null — an explicit falsy id (0, "",
        # False) is still the author's id, not a signal to fall back to a content hash.
        raw_id = entry.get("id")
        record_id = str(raw_id) if raw_id is not None else _default_id(entry)
        if record_id in seen_ids:
            # Probe ids are f"{capability}::traced::{id}"; duplicate record ids would
            # collide the same way derive_probes() guards against duplicate tool names.
            raise ValueError(f"duplicate trace record id: {record_id!r}")
        seen_ids.add(record_id)
        records.append(
            TraceRecord(
                id=record_id,
                messages=list(entry.get("messages") or []),
                tools=list(entry.get("tools") or []),
                response=_parse_response(entry["response"]),
            )
        )
    return records


def traces_fingerprint(records: List[TraceRecord]) -> str:
    """Stable hash of a trace-record set (order-invariant), mirroring
    probes.tools_fingerprint — so a diff can flag when the underlying real-trace input
    changes, the same way it already flags a changed toolset."""
    canonical_records = [
        {
            "id": r.id,
            "messages": r.messages,
            "tools": r.tools,
            "response": {
                "content": r.response.content,
                "tool_calls": [
                    {"name": c.name, "arguments": c.arguments} for c in r.response.tool_calls
                ],
            },
        }
        for r in records
    ]
    # Sort by each record's own canonical JSON, not just its id: load_trace_records()
    # rejects duplicate ids, but this function is public and can be called on
    # hand-built records too, and Python's stable sort would otherwise let two same-id
    # entries' relative input order leak into the hash, breaking order-invariance.
    canonical_records.sort(key=lambda rec: json.dumps(rec, sort_keys=True, separators=(",", ":")))
    return _hash_json(canonical_records, 16)


def _tool_schema(tools: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    for t in tools:
        fn = t.get("function", {})
        if fn.get("name") == name:
            return fn.get("parameters") or {}
    return None


def _parse_json_object(text: Any) -> Dict[str, Any]:
    if isinstance(text, dict):
        return text
    if isinstance(text, str):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _traced_probe(capability, record, expected_tool, schema, reference=None) -> Probe:
    return Probe(
        id=f"{capability}::traced::{record.id}",
        capability=capability,
        description=f"[traced] reproduces the recorded decision in '{record.id}'",
        messages=record.messages,
        tools=record.tools,
        expected_tool=expected_tool,
        schema=schema,
        # The recorded behavior IS the reference (trust model: mine traces from a model
        # you already trust). Real scoring never reads this — only SimulatedClient does,
        # so a traced probe still demos/tests sensibly through `--simulate`, the same way
        # a synthetic one does, before you point --traces at a real endpoint.
        reference=reference or {},
    )


def derive_traced_probes(records: List[TraceRecord]) -> List[Probe]:
    """Turn recorded decision points into probes for TRACED_CAPABILITIES."""
    probes: List[Probe] = []
    for r in records:
        calls = r.response.tool_calls
        if calls:
            call = calls[0]  # first call only; parallel-tool-call replay is out of scope
            # The matched tool's own "parameters" — possibly {} for a genuinely zero-arg
            # tool, or missing because the export didn't capture it. Either way, scoring
            # already treats an empty/missing schema as a trivial pass (see
            # score_required_args), the same as it does for a synthetic zero-property
            # tool, so there's no vacuous-probe risk here. None means the tool actually
            # called isn't in this record's own offered set at all — an incomplete record
            # that can't meaningfully test even tool_selection, so skip it entirely.
            schema = _tool_schema(r.tools, call.name)
            if schema is None:
                continue
            reference = {"valid_args": _parse_json_object(call.arguments)}
            for cap in _TOOL_CALL_CAPABILITIES:
                probes.append(_traced_probe(cap, r, call.name, schema, reference))
        elif r.response.content:
            # No tool call: only usable for structured_output, and only if the recorded
            # text is itself schema-valid JSON against one of the tools actually offered
            # at that turn — the trace-analog of "the model was asked for JSON matching a
            # schema and didn't call a tool." Note this keeps `tools=record.tools` (via
            # _traced_probe), unlike derive_probes()'s synthetic structured_output probes
            # which always set tools=[]: a traced probe replays the tool as it really was
            # offered, so a candidate that instead calls the tool (a defensible choice in
            # an ambiguous exchange) scores 0 here — a real, sometimes-surprising signal,
            # not a bug. Curate structured_output traces from exchanges that
            # unambiguously call for text, not a tool call (see fixtures/sample_traces.json).
            try:
                payload = json.loads(r.response.content)
            except json.JSONDecodeError:
                continue
            for t in r.tools:
                schema = t.get("function", {}).get("parameters") or {}
                if not schema.get("properties"):
                    continue  # no real constraint to match against — would trivially
                    # "validate" any payload and mask the tool actually intended
                try:
                    jsonschema.validate(payload, schema)
                except (jsonschema.ValidationError, jsonschema.SchemaError):
                    continue
                probes.append(
                    _traced_probe("structured_output", r, None, schema, {"valid_args": payload})
                )
                break  # first matching tool schema only
    return probes
