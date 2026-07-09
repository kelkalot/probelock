"""Deterministic scorers — no LLM judge, no randomness.

Every capability is scored by code: did it call the right tool, do the arguments
validate against the JSON schema, did it emit parseable JSON, did the text match
exactly. This is what makes a probelock lockfile reproducible and diffable: run
it twice on the same model and you get the same numbers.
"""

from __future__ import annotations

import json
from typing import List, Optional

import jsonschema

from .models import Probe, ResponseMessage, ToolCall


def _matching_calls(resp: ResponseMessage, name: Optional[str]) -> List[ToolCall]:
    return [c for c in resp.tool_calls if name is None or c.name == name]


def _parse_args(call: ToolCall):
    raw = call.arguments
    if isinstance(raw, dict):  # some servers return arguments already parsed
        return raw
    try:
        args = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return args if isinstance(args, dict) else None


def _forced_empty(prop_schema, empty_value) -> bool:
    """True if `prop_schema` structurally cannot be satisfied by anything but the given
    empty value (const/enum pinned to it, or a max-size bound of 0) — i.e. an empty
    response for this property is the CORRECT answer, not a lazy non-answer."""
    if not isinstance(prop_schema, dict):
        return False
    if "const" in prop_schema:
        return prop_schema["const"] == empty_value
    if prop_schema.get("enum"):
        return all(v == empty_value for v in prop_schema["enum"])
    if empty_value == [] and prop_schema.get("maxItems") == 0:
        return True
    if empty_value == "" and prop_schema.get("maxLength") == 0:
        return True
    return False


def _is_unfilled(value, prop_schema=None) -> bool:
    """True if `value` looks like a lazy non-answer (missing, or empty where the schema
    doesn't force emptiness) rather than a genuine filled-in argument."""
    if value is None:
        return True
    if isinstance(value, (str, list, dict)) and len(value) == 0:
        return not _forced_empty(prop_schema, value)
    return False


def _all_filled(props_schema, args) -> bool:
    """True if every property in `props_schema` (required + optional) is present and
    filled in `args`, recursing into nested object-typed properties so an optional field
    nested inside a filled-in object argument is checked too, not just top-level keys."""
    if not isinstance(args, dict):
        return False
    for key, sub in (props_schema or {}).items():
        if key not in args or _is_unfilled(args[key], sub):
            return False
        if isinstance(sub, dict) and sub.get("type") == "object" and sub.get("properties"):
            if not _all_filled(sub["properties"], args[key]):
                return False
    return True


def score_tool_selection(probe: Probe, resp: ResponseMessage) -> float:
    return 1.0 if _matching_calls(resp, probe.expected_tool) else 0.0


def score_tool_discrimination(probe: Probe, resp: ResponseMessage) -> float:
    # Precise selection: the expected tool is called AND no OTHER tool is called.
    names = {c.name for c in resp.tool_calls}
    if probe.expected_tool not in names:
        return 0.0
    return 1.0 if not (names - {probe.expected_tool}) else 0.0


def score_tool_restraint(probe: Probe, resp: ResponseMessage) -> float:
    # Over-trigger check: a task needing no tool must produce no tool call.
    return 0.0 if resp.tool_calls else 1.0


def score_tool_permission(probe: Probe, resp: ResponseMessage) -> float:
    # The expected_tool here is the FORBIDDEN one: pass iff it was not called.
    called = {c.name for c in resp.tool_calls}
    return 0.0 if probe.expected_tool in called else 1.0


def score_no_hallucinated_tool(probe: Probe, resp: ResponseMessage) -> float:
    # Pass iff every called tool was actually offered (no fabricated/absent tool).
    offered = {t.get("function", {}).get("name") for t in (probe.tools or [])}
    return 0.0 if any(c.name not in offered for c in resp.tool_calls) else 1.0


def score_needle_in_tools(probe: Probe, resp: ResponseMessage) -> float:
    # Selection under distraction: the right tool, hidden among many, still called.
    return 1.0 if _matching_calls(resp, probe.expected_tool) else 0.0


def score_arity_robustness(probe: Probe, resp: ResponseMessage) -> float:
    # Did the model fill EVERY parameter (required + optional) when asked?
    matches = _matching_calls(resp, probe.expected_tool)
    if not matches:
        return 0.0  # the tool was never called at all: no credit, even for a 0-arg tool
    props = (probe.schema or {}).get("properties") or {}
    if not props:
        return 1.0  # tool was called; nothing to fill
    for call in matches:
        args = _parse_args(call)
        if args is not None and _all_filled(props, args):
            return 1.0
    return 0.0


def score_arg_validity(probe: Probe, resp: ResponseMessage) -> float:
    # Any matching call with schema-valid args counts (consistent with
    # tool_selection's any-match), so a corrected later call is not ignored.
    for call in _matching_calls(resp, probe.expected_tool):
        args = _parse_args(call)
        if args is None:
            continue
        try:
            jsonschema.validate(args, probe.schema or {})
            return 1.0
        except jsonschema.ValidationError:
            continue
    return 0.0


def score_required_args(probe: Probe, resp: ResponseMessage) -> float:
    required = (probe.schema or {}).get("required", [])
    props = (probe.schema or {}).get("properties") or {}
    for call in _matching_calls(resp, probe.expected_tool):
        args = _parse_args(call)
        if args is None:
            continue
        if all(key in args and not _is_unfilled(args[key], props.get(key)) for key in required):
            return 1.0
    return 0.0


def score_structured_output(probe: Probe, resp: ResponseMessage) -> float:
    if not resp.content:
        return 0.0
    try:
        payload = json.loads(resp.content.strip())
    except json.JSONDecodeError:
        return 0.0
    try:
        jsonschema.validate(payload, probe.schema or {})
    except jsonschema.ValidationError:
        return 0.0
    return 1.0


def score_format_adherence(probe: Probe, resp: ResponseMessage) -> float:
    # Exact match (surrounding whitespace ignored). Case matters: this capability
    # exists to catch "reply with exactly X" failures, including wrong case.
    if not resp.content or probe.expected_text is None:
        return 0.0
    return 1.0 if resp.content.strip() == probe.expected_text.strip() else 0.0


def score_traced_schema_validity(probe: Probe, resp: ResponseMessage) -> float:
    # Trace-mined schema validity: no expected tool was inferred (that would need
    # correctness inference), so the candidate may call ANY offered tool — the check is
    # that some call's arguments validate against the schema of the tool it names.
    # Any-match across calls, consistent with arg_validity; a call naming an un-offered
    # tool has no declared schema to hold it to and cannot earn the pass.
    schemas = {}
    for t in probe.tools or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if isinstance(fn, dict):
            params = fn.get("parameters")
            schemas[fn.get("name")] = params if isinstance(params, dict) else {}
    for call in resp.tool_calls:
        schema = schemas.get(call.name)
        args = _parse_args(call)
        if schema is None or args is None:
            continue
        try:
            jsonschema.validate(args, schema)
            return 1.0
        except (jsonschema.ValidationError, jsonschema.SchemaError):
            continue
    return 0.0


SCORERS = {
    "tool_selection": score_tool_selection,
    "tool_discrimination": score_tool_discrimination,
    "needle_in_tools": score_needle_in_tools,
    "arity_robustness": score_arity_robustness,
    "arg_validity": score_arg_validity,
    "required_args": score_required_args,
    "structured_output": score_structured_output,
    # json_mode replays the native response_format path but the SUCCESS test is
    # identical to structured_output: content must be JSON valid against the schema.
    "json_mode": score_structured_output,
    "format_adherence": score_format_adherence,
    "tool_restraint": score_tool_restraint,
    "tool_permission": score_tool_permission,
    "no_hallucinated_tool": score_no_hallucinated_tool,
    # Trace-mined capabilities (probelock ingest). Deliberately distinct names, not the
    # synthetic ones, so lockfiles/diffs report trace-derived scores separately — a drop
    # in multi-turn trace probes with stable synthetic probes is itself diagnostic.
    # tool_selection and no_tool replay through the same deterministic checks as their
    # synthetic counterparts; only schema_validity needs its own scorer (no pinned tool).
    "traced_schema_validity": score_traced_schema_validity,
    "traced_tool_selection": score_tool_selection,
    "traced_no_tool": score_tool_restraint,
}

# Negative probes: the score measures the ABSENCE of bad behavior. An API error
# means the bad behavior couldn't happen, so the runner scores these 1.0 on error
# (but still error-tags them, so the all-errored fatal guard keeps working).
# traced_no_tool is restraint mined from real traffic — same absence-of-behavior logic.
NEGATIVE_CAPABILITIES = frozenset(
    {"tool_restraint", "tool_permission", "no_hallucinated_tool", "traced_no_tool"}
)


def score(probe: Probe, resp: ResponseMessage) -> float:
    scorer = SCORERS.get(probe.capability)
    if scorer is None:  # pragma: no cover - guarded by derivation
        raise ValueError(f"No scorer for capability '{probe.capability}'")
    return scorer(probe, resp)
