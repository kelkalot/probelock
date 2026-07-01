#!/usr/bin/env python3
"""Example: convert an OpenTelemetry span export into probelock's trace-record schema.

This is a documented STARTING POINT you adapt, not a maintained adapter shipped with
probelock. litellm's OTel attribute layout has already changed once (v1.81.0 moved where
request/response attributes live) and there's a newer, differently-shaped opt-in "OTel v2"
integration — so the attribute names below are a best-effort guess based on litellm's
current docs, not a stable contract. Run with --inspect against your own export first to
see what attribute names your setup actually produces, and adjust the *_ATTR_CANDIDATES
lists below to match.

Input: a JSON file that is either
  - a JSON array of span objects, each with an "attributes" dict (e.g. from an OTLP JSON
    file exporter, or your own logging of litellm's otel callback), or
  - an object with a top-level "spans" list, or
  - a raw OTLP export ({"resourceSpans": [...]}) — flattened best-effort; adjust
    _flatten_spans() if your exporter's nesting differs.

Output: probelock's own trace-export schema (see probelock/traces.py's module docstring),
written to --out (default: probelock.traces.json). Review the output before committing it —
unlike a tool schema, it contains real conversation content.

Usage:
    python examples/otel_traces_to_probelock.py otel_export.json --inspect
    python examples/otel_traces_to_probelock.py otel_export.json -o probelock.traces.json
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional

# Attribute names to try, in order, for each field. litellm has used different names across
# versions/configs: gen_ai.* is the newer semantic-convention name; llm.openai.*/raw_request/
# raw_response are the legacy shape. Run with --inspect to see what YOUR export actually has,
# then reorder/extend these to match.
MESSAGES_ATTR_CANDIDATES = ["gen_ai.input.messages", "llm.openai.messages", "raw_request"]
RESPONSE_ATTR_CANDIDATES = ["gen_ai.output.messages", "llm.openai.response", "raw_response"]
TOOLS_ATTR_CANDIDATES = ["gen_ai.request.tools", "llm.openai.tools"]


def _flatten_spans(data: Any) -> List[Dict[str, Any]]:
    """Best-effort flatten of a few common export shapes into a flat list of span dicts,
    each with a plain {attr_name: value} "attributes" dict."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("spans"), list):
        return data["spans"]
    if isinstance(data, dict) and isinstance(data.get("resourceSpans"), list):
        # Raw OTLP/JSON export: resourceSpans -> scopeSpans -> spans, with attributes as
        # {"key": "...", "value": {"stringValue": "..."}} pairs (protobuf-JSON shape).
        spans = []
        for rs in data["resourceSpans"]:
            for ss in rs.get("scopeSpans", []):
                for span in ss.get("spans", []):
                    attrs = {}
                    for kv in span.get("attributes", []):
                        value = kv.get("value", {})
                        attrs[kv["key"]] = (
                            value.get("stringValue")
                            or value.get("intValue")
                            or value.get("boolValue")
                            or json.dumps(value)
                        )
                    spans.append({"name": span.get("name"), "attributes": attrs})
        return spans
    raise ValueError(
        "Unrecognized export shape — expected a JSON array of spans, {'spans': [...]}, "
        "or a raw OTLP {'resourceSpans': [...]} export. Adjust _flatten_spans() to match "
        "whatever your exporter actually produces."
    )


def _first_attr(attrs: Dict[str, Any], candidates: List[str]) -> Optional[Any]:
    for key in candidates:
        if attrs.get(key):
            return attrs[key]
    return None


def _as_list(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        value = json.loads(value)
    return value if isinstance(value, list) else []


def _to_response(raw: Any) -> Dict[str, Any]:
    """Normalize whatever shape the response attribute holds into probelock's
    {"content": ..., "tool_calls": [{"name": ..., "arguments": ...}]}."""
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, list):  # gen_ai.output.messages: a list of role/parts messages
        raw = raw[-1] if raw else {}
    message = raw
    if isinstance(raw, dict) and "choices" in raw:  # a raw ChatCompletion-shaped payload
        message = raw["choices"][0]["message"]
    if not isinstance(message, dict):
        return {"content": None, "tool_calls": []}
    tool_calls = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", tc)
        tool_calls.append({"name": fn.get("name", ""), "arguments": fn.get("arguments", "{}")})
    return {"content": message.get("content"), "tool_calls": tool_calls}


def convert(spans: List[Dict[str, Any]]) -> Dict[str, Any]:
    records = []
    for i, span in enumerate(spans):
        attrs = span.get("attributes") or {}
        messages = _as_list(_first_attr(attrs, MESSAGES_ATTR_CANDIDATES))
        response_raw = _first_attr(attrs, RESPONSE_ATTR_CANDIDATES)
        if not messages or response_raw is None:
            continue  # not a chat-completion span (e.g. a guardrail/router span) — skip it
        records.append({
            "id": span.get("name") or f"span-{i}",
            "messages": messages,
            "tools": _as_list(_first_attr(attrs, TOOLS_ATTR_CANDIDATES)),
            "response": _to_response(response_raw),
        })
    return {"version": 1, "source": "litellm-otel", "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export", help="Path to your OTel/litellm export JSON.")
    parser.add_argument("-o", "--out", default="probelock.traces.json")
    parser.add_argument(
        "--inspect", action="store_true",
        help="Print each span's attribute keys instead of converting, so you can update "
        "the *_ATTR_CANDIDATES lists above to match your setup.",
    )
    args = parser.parse_args()

    with open(args.export) as fh:
        data = json.load(fh)
    spans = _flatten_spans(data)

    if args.inspect:
        for span in spans[:20]:
            print(span.get("name"), sorted((span.get("attributes") or {}).keys()))
        return

    payload = convert(spans)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {len(payload['records'])} record(s) to {args.out}")
    print("Review this file before committing it — it may contain real conversation content.")


if __name__ == "__main__":
    main()
