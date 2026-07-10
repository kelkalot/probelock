"""``probelock doctor`` — health-check a toolset and detect probe/tool drift.

Two diagnostics, both pure and deterministic (no model, no network):

  * **Toolset health.** A schema-derived battery is only as sensitive as the toolset it
    is built from. VALIDATION.md found a 3-tool schema with unconstrained arguments blind
    to real regressions a richer schema caught. ``doctor`` warns before you trust a gate:
    duplicate names, empty parameter schemas, arguments with no constraint to violate,
    and a toolset too small for the discrimination probes to bite.

  * **Drift.** Trace-mined and trace-export probes freeze a tool's schema *verbatim* at
    mint time (see mined.py / traces.py). When the real agent's toolset later changes —
    a tool renamed, an argument added — the frozen probe silently replays a stale schema,
    and a gate failure then looks like a regression when it is really drift. ``doctor``
    compares each frozen probe's pinned schema against the live ``--tools`` and flags a
    removed/renamed tool (the probe can never match) or a changed schema (re-mine).

This is the trust tool the "commit a lockfile" promise needs: it tells you when a
committed probe no longer means what it did the day it was minted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import jsonschema

# A toolset smaller than this makes needle_in_tools / tool_discrimination weak — there is
# little to discriminate among.
_MIN_TOOLS = 3
# Schema keywords the scorers' jsonschema.validate ACTUALLY enforces, so a value carrying
# one is falsifiable and arg_validity can fail. Deliberately excludes "format": default
# jsonschema does not assert it (no format_checker is passed), so a format-only property
# would give false confidence. "type" is included — a wrong-typed argument does fail.
_ASSERTED_KEYS = ("type", "enum", "const", "pattern", "minimum", "maximum",
                  "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength",
                  "minItems", "maxItems", "multipleOf", "uniqueItems")
# JSON Schema composition. The probe generator resolves these ($ref/$defs, allOf, …) and
# the scorer validates against them, so a constraint can live entirely behind them
# (Pydantic emits exactly this). doctor does not re-resolve composition; it stays
# CONSERVATIVE — presence of any composition keyword means "assume constrained", so a
# real, richly-constrained toolset is never falsely flagged weak.
_COMPOSITION_KEYS = ("$ref", "allOf", "anyOf", "oneOf", "not")
# Arrays JSON Schema treats as unordered (a reordering is not a real change), so drift
# canonicalization must sort them or a re-serialized-but-identical schema reads as drift.
_UNORDERED_ARRAY_KEYS = ("required", "enum", "type", "allOf", "anyOf", "oneOf")


@dataclass(frozen=True)
class Finding:
    level: str  # "error" | "warn"
    code: str
    message: str


def _params(tool: Any) -> Optional[Dict[str, Any]]:
    fn = tool.get("function") if isinstance(tool, dict) else None
    if not isinstance(fn, dict):
        return None
    params = fn.get("parameters")
    return params if isinstance(params, dict) else {}


def _tool_schema(tools: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    for t in tools:
        fn = t.get("function") if isinstance(t, dict) else None
        if isinstance(fn, dict) and fn.get("name") == name:
            return fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
    return None


def _normalize(node: Any) -> Any:
    """Sort every array JSON Schema treats as unordered (``required``, ``enum``, a
    list-form ``type``, and the ``allOf``/``anyOf``/``oneOf`` member lists) so a
    cosmetically-reordered but semantically identical schema is not read as drift."""
    if isinstance(node, dict):
        out: Dict[str, Any] = {}
        for k, v in node.items():
            if k in _UNORDERED_ARRAY_KEYS and isinstance(v, list):
                members = [_normalize(x) for x in v]
                out[k] = sorted(members, key=lambda x: json.dumps(x, sort_keys=True))
            else:
                out[k] = _normalize(v)
        return out
    if isinstance(node, list):
        return [_normalize(v) for v in node]
    return node


def _canon(schema: Dict[str, Any]) -> str:
    return json.dumps(_normalize(schema), sort_keys=True, separators=(",", ":"))


def _falsifiable(node: Any) -> bool:
    """True if a schema node could fail the scorers' validation — it carries an enforced
    keyword directly, uses composition (assumed constrained — see _COMPOSITION_KEYS),
    forbids extra keys, or nests an object/array whose sub-schema does."""
    if not isinstance(node, dict):
        return False
    if any(k in node for k in _ASSERTED_KEYS) or any(k in node for k in _COMPOSITION_KEYS):
        return True
    if node.get("additionalProperties") is False:
        return True
    for sub in (node.get("properties") or {}).values():
        if _falsifiable(sub):
            return True
    items = node.get("items")
    if isinstance(items, dict) and _falsifiable(items):
        return True
    if isinstance(items, list) and any(_falsifiable(i) for i in items):  # tuple-form items
        return True
    return False


def _args_falsifiable(schema: Dict[str, Any]) -> bool:
    """True if the tool's ARGUMENTS can meaningfully fail a probe — a required field
    (required_args can fail), an object-level const/enum/composition, or any property the
    scorer would reject a bad value for (arg_validity can fail). The object wrapper's own
    ``type: object`` does not count; only the arguments do."""
    if not isinstance(schema, dict):
        return False
    if schema.get("required"):
        return True
    if any(k in schema for k in ("const", "enum")) or any(k in schema for k in _COMPOSITION_KEYS):
        return True
    return any(_falsifiable(p) for p in (schema.get("properties") or {}).values())


def toolset_health(tools: List[Dict[str, Any]]) -> List[Finding]:
    """Warn about a toolset too weak to catch real regressions (never fatal on its own)."""
    findings: List[Finding] = []
    names = [
        t["function"]["name"] for t in tools
        if isinstance(t, dict) and isinstance(t.get("function"), dict)
        and t["function"].get("name")
    ]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        findings.append(Finding("error", "duplicate-tool",
                                f"duplicate tool name(s): {', '.join(dupes)}"))
    if len(names) < _MIN_TOOLS:
        findings.append(Finding("warn", "few-tools",
                                f"only {len(names)} tool(s) — needle_in_tools and "
                                f"tool_discrimination have little to discriminate among; "
                                f"a richer toolset detects more drift (see VALIDATION.md)"))
    for t in tools:
        fn = t.get("function") if isinstance(t, dict) else None
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        name = fn["name"]
        schema = _params(t) or {}
        if _args_falsifiable(schema):
            continue  # something the scorer enforces — not weak
        if not (schema.get("properties") or {}):
            findings.append(Finding("warn", "no-args",
                                    f"'{name}' has no parameters — structured_output and "
                                    f"arg_validity probes for it are trivial passes"))
        else:
            findings.append(Finding("warn", "unconstrained-args",
                                    f"'{name}' arguments carry nothing the scorer enforces "
                                    f"(no type/required/enum/bounds) — arg_validity and "
                                    f"required_args cannot fail, under-reporting regressions"))
    return findings


def drift(live_tools: List[Dict[str, Any]], frozen: List[Tuple[str, Dict[str, Any]]]) -> List[Finding]:
    """Compare frozen (tool_name, schema) pairs from mined/trace probes against the live
    toolset. ``frozen`` may repeat a name (several probes pin the same tool)."""
    by_name: Dict[str, List[Dict[str, Any]]] = {}
    for name, schema in frozen:
        if name:
            by_name.setdefault(name, []).append(schema)
    findings: List[Finding] = []
    for name in sorted(by_name):
        schemas = by_name[name]
        n = len(schemas)
        live = _tool_schema(live_tools, name)
        if live is None:
            findings.append(Finding("error", "tool-removed",
                f"'{name}' is no longer in the toolset (renamed or removed) — "
                f"{n} frozen probe(s) can never match; re-mine or restore the tool"))
            continue
        live_canon = _canon(live)
        stale = sum(1 for s in schemas if _canon(s) != live_canon)
        if stale:
            findings.append(Finding("warn", "schema-drift",
                f"'{name}' schema changed since mint — {stale} of {n} frozen probe(s) pin "
                f"the old schema; re-mine to refresh (or the gate may read drift as regression)"))
    return findings


def mined_frozen_tools(mined_probes) -> List[Tuple[str, Dict[str, Any]]]:
    """The (tool_name, pinned_schema) pairs a set of MinedProbe objects depend on."""
    out: List[Tuple[str, Dict[str, Any]]] = []
    for mp in mined_probes:
        if not mp.tool:
            continue
        schema = _tool_schema(mp.tools, mp.tool)
        if schema is not None:
            out.append((mp.tool, schema))
    return out


def traces_frozen_tools(records) -> List[Tuple[str, Dict[str, Any]]]:
    """The (tool_name, pinned_schema) pairs a set of TraceRecord objects depend on.

    Mirrors traces.derive_traced_probes: a tool-call record pins the called tool's
    schema; a content-only record pins the tool whose parameters the recorded JSON text
    validates against (the structured_output traced probe). Both freeze a schema that can
    drift, so drift must see both."""
    out: List[Tuple[str, Dict[str, Any]]] = []
    for r in records:
        calls = r.response.tool_calls
        if calls:
            name = calls[0].name
            schema = _tool_schema(r.tools, name)
            if schema is not None:
                out.append((name, schema))
        elif r.response.content:
            try:
                payload = json.loads(r.response.content)
            except (json.JSONDecodeError, TypeError):
                continue
            for t in r.tools:
                fn = t.get("function") if isinstance(t, dict) else None
                schema = fn.get("parameters") if isinstance(fn, dict) else None
                if not (isinstance(schema, dict) and schema.get("properties")):
                    continue
                try:
                    jsonschema.validate(payload, schema)
                except (jsonschema.ValidationError, jsonschema.SchemaError):
                    continue
                out.append((fn.get("name"), schema))
                break
    return out
