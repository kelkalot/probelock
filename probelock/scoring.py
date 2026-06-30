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
    props = list(((probe.schema or {}).get("properties") or {}).keys())
    if not props:
        return 1.0  # nothing to fill
    for call in _matching_calls(resp, probe.expected_tool):
        args = _parse_args(call)
        if args is None:
            continue
        if all(key in args and args[key] not in (None, "", [], {}) for key in props):
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
    for call in _matching_calls(resp, probe.expected_tool):
        args = _parse_args(call)
        if args is None:
            continue
        if all(key in args and args[key] not in (None, "", [], {}) for key in required):
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


SCORERS = {
    "tool_selection": score_tool_selection,
    "tool_discrimination": score_tool_discrimination,
    "needle_in_tools": score_needle_in_tools,
    "arity_robustness": score_arity_robustness,
    "arg_validity": score_arg_validity,
    "required_args": score_required_args,
    "structured_output": score_structured_output,
    "format_adherence": score_format_adherence,
    "tool_restraint": score_tool_restraint,
    "tool_permission": score_tool_permission,
    "no_hallucinated_tool": score_no_hallucinated_tool,
}

# Negative probes: the score measures the ABSENCE of bad behavior. An API error
# means the bad behavior couldn't happen, so the runner scores these 1.0 on error
# (but still error-tags them, so the all-errored fatal guard keeps working).
NEGATIVE_CAPABILITIES = frozenset({"tool_restraint", "tool_permission", "no_hallucinated_tool"})


def score(probe: Probe, resp: ResponseMessage) -> float:
    scorer = SCORERS.get(probe.capability)
    if scorer is None:  # pragma: no cover - guarded by derivation
        raise ValueError(f"No scorer for capability '{probe.capability}'")
    return scorer(probe, resp)
