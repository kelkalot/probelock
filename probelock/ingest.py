"""Mine deterministic probes from raw agent traffic logs (``probelock ingest``).

Where traces.py replays a *curated* trace export (each record hand-picked, the recorded
response trusted as ground truth), this module handles *raw* request/response logs — the
kind a recording proxy or an existing logging layer appends to blindly. Raw traffic
includes model mistakes, retries, duplicates, and real user data, so between "log line"
and "probe" sits a pipeline:

    adapt → stitch sessions → cluster (dedup) → infer confirmed-good → sample → redact

Every mined probe lands ``pending`` and must pass human review (``probelock traces
review``) before it joins the battery — see mined.py for the trust ladder. The one rule
applied throughout: provenance determines trust. Each probe records how many sessions
support it and which confirmation rule fired, and that provenance decides how much
review it needs.

Everything here is deterministic and dependency-free: confirmation is structural
(continuation and cross-session agreement), "semantic distance" for no-tool mining is
lexical overlap, and dedup is normalized-context hashing — no embeddings, no LLM judge.

Session stitching: records written by a cooperating logger may carry a ``session_id``
(the future proxy tags one); records without one are stitched by containment — exchange
B belongs to A's session when A's normalized messages are a prefix of (or equal to)
B's. Containment-based stitching is conservative: byte-identical records — duplicate
conversations, or one exchange logged twice by at-least-once shipping — collapse into
one session, so they never inflate the distinct-session agreement counts that
confirmed-good filtering relies on.
"""

from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jsonschema

from .mined import MinedProbe
from .models import ResponseMessage, ToolCall
from .probes import synth_args, synth_value

FORMATS = ("auto", "trace-v1", "openai-jsonl", "anthropic-jsonl", "otel-genai")

# --- input records ----------------------------------------------------------


@dataclass
class Exchange:
    """One logged request/response pair, normalized to a single internal shape.
    ``session_id`` and ``turn`` are filled during stitching when the log didn't
    provide them."""

    messages: List[Dict[str, Any]]
    tools: List[Dict[str, Any]]
    response: ResponseMessage
    tool_choice: Any = None
    session_id: Optional[str] = None
    ts: str = ""
    model: str = ""
    status: int = 200
    line_no: int = 0
    turn: int = 0
    norm: List[Dict[str, Any]] = field(default_factory=list)  # normalized messages, cached


@dataclass
class MiningConfig:
    min_agreement: int = 2
    min_agreement_notool: int = 3
    per_capability: int = 8
    max_context_tokens: int = 8192
    redact_patterns: Tuple[str, ...] = ()
    source: str = ""
    # Clustering (dedup) mode. "hash" is the default deterministic path: identical
    # normalized contexts collapse. "embeddings" groups NEAR-duplicate contexts by
    # cosine similarity of an embedding from embed_endpoint — opt-in and NOT
    # deterministic (the grouping depends on the embedding model/version), so the CLI
    # warns and provenance records it.
    cluster: str = "hash"
    embed_endpoint: str = ""
    embed_model: str = ""
    cluster_threshold: float = 0.92


@dataclass
class MiningSummary:
    """Counters for everything the pipeline saw and everything it dropped. Dropping is
    fine — silently dropping is not, so the CLI renders all of this."""

    records: int = 0
    skipped: Dict[str, int] = field(default_factory=dict)
    sessions: int = 0
    clusters: int = 0
    candidates: Dict[str, int] = field(default_factory=dict)
    emitted: Dict[str, int] = field(default_factory=dict)
    ambiguous_tool_selection: int = 0
    unconfirmed_tool_clusters: int = 0  # calls that only qualified for schema_validity

    def skip(self, reason: str, n: int = 1) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + n


# --- adapters ---------------------------------------------------------------


def _clean_tools(tools: Any) -> List[Dict[str, Any]]:
    """Only OpenAI-shaped tool entries survive: anything without a dict 'function' can
    neither answer a schema lookup nor replay, and one garbage entry must not abort a
    whole ingest run."""
    return [
        t for t in (tools if isinstance(tools, list) else [])
        if isinstance(t, dict) and isinstance(t.get("function"), dict)
    ]


def _message_to_response(msg: Dict[str, Any]) -> ResponseMessage:
    """Map a wire-format assistant message to a ResponseMessage, tolerating both the
    OpenAI nesting ({"function": {"name", "arguments"}}) and the already-flat shape
    probelock's own trace export uses."""
    calls: List[ToolCall] = []
    for tc in msg.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
        name = fn.get("name") or ""
        args = fn.get("arguments") or "{}"
        if not isinstance(args, str):
            args = json.dumps(args)
        calls.append(ToolCall(str(name), args))
    content = msg.get("content")
    if not isinstance(content, str) or not content:
        content = None
    return ResponseMessage(content=content, tool_calls=calls)


def _parse_trace_v1(obj: Dict[str, Any], line_no: int) -> Exchange:
    """The native record schema (one JSON object per line) — what the recording proxy
    writes and what every other adapter normalizes into."""
    req, resp = obj.get("request"), obj.get("response")
    if not isinstance(req, dict) or not isinstance(resp, dict):
        raise ValueError("record needs 'request' and 'response' objects")
    message = resp.get("message")
    if not isinstance(message, dict):
        raise ValueError("record needs 'response.message'")
    meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
    try:
        status = int(meta.get("status", 200))
    except (TypeError, ValueError):
        status = 200
    return Exchange(
        messages=list(req.get("messages") or []),
        tools=_clean_tools(req.get("tools")),
        response=_message_to_response(message),
        tool_choice=req.get("tool_choice"),
        session_id=str(obj["session_id"]) if obj.get("session_id") else None,
        ts=str(obj.get("ts") or ""),
        model=str(obj.get("model") or ""),
        status=status,
        line_no=line_no,
    )


def _parse_openai_jsonl(obj: Dict[str, Any], line_no: int) -> Exchange:
    """Adapter for the common roll-your-own log: the verbatim chat-completions request
    body next to the verbatim response object, one pair per line."""
    req, resp = obj.get("request"), obj.get("response")
    if not isinstance(req, dict) or not isinstance(resp, dict):
        raise ValueError("record needs 'request' and 'response' objects")
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("record needs 'response.choices[0]'")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("record needs 'response.choices[0].message'")
    try:
        status = int(obj.get("status", 200))
    except (TypeError, ValueError):
        status = 200
    return Exchange(
        messages=list(req.get("messages") or []),
        tools=_clean_tools(req.get("tools")),
        response=_message_to_response(message),
        tool_choice=req.get("tool_choice"),
        session_id=str(obj["session_id"]) if obj.get("session_id") else None,
        ts=str(obj.get("ts") or resp.get("created") or ""),
        model=str(req.get("model") or resp.get("model") or ""),
        status=status,
        line_no=line_no,
    )


# --- anthropic adapter ------------------------------------------------------
# Anthropic's Messages API is a different shape (content blocks, tool_use/tool_result,
# a top-level system field). Translate it to probelock's canonical OpenAI shape at
# ingest time so the mined probe's frozen context replays through the one existing path.


def _anthropic_tools_to_openai(tools: Any) -> List[Dict[str, Any]]:
    out = []
    for t in tools if isinstance(tools, list) else []:
        if isinstance(t, dict) and t.get("name"):
            out.append({"type": "function", "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema") if isinstance(t.get("input_schema"), dict) else {},
            }})
    return out


def _anthropic_blocks(content: Any):
    """Split Anthropic message content (a string or a list of blocks) into
    (text, openai_tool_calls, openai_tool_messages)."""
    if isinstance(content, str):
        return (content or None), [], []
    text_parts: List[str] = []
    calls: List[Dict[str, Any]] = []
    tool_msgs: List[Dict[str, Any]] = []
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text" and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
        elif btype == "tool_use":
            calls.append({
                "id": str(block.get("id") or ""),
                "type": "function",
                "function": {"name": str(block.get("name") or ""),
                             "arguments": json.dumps(block.get("input") or {})},
            })
        elif btype == "tool_result":
            rc = block.get("content")
            if isinstance(rc, list):  # tool_result content may itself be blocks
                rc = " ".join(str(b.get("text", "")) for b in rc if isinstance(b, dict))
            content = rc if isinstance(rc, str) else json.dumps(rc)
            if block.get("is_error"):
                # Anthropic marks a failed tool call with is_error: true and often
                # human-readable content ("No matching files.") that would otherwise
                # look successful. Translate the structured flag into probelock's
                # error-content convention so confirms_continuation does NOT confirm a
                # failed call as good tool selection.
                content = f"Error: {content}"
            tool_msgs.append({
                "role": "tool",
                "tool_call_id": str(block.get("tool_use_id") or ""),
                "content": content,
            })
    return ("".join(text_parts) or None), calls, tool_msgs


def _anthropic_messages_to_openai(system: Any, messages: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if system:
        sys_text = system if isinstance(system, str) else " ".join(
            b.get("text", "") for b in system if isinstance(b, dict)
        )
        out.append({"role": "system", "content": sys_text})
    for m in messages if isinstance(messages, list) else []:
        if not isinstance(m, dict):
            continue
        text, calls, tool_msgs = _anthropic_blocks(m.get("content"))
        out.extend(tool_msgs)  # tool_result blocks become standalone tool messages
        if m.get("role") == "assistant":
            msg: Dict[str, Any] = {"role": "assistant", "content": text}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        elif m.get("role") == "user" and text is not None:
            out.append({"role": "user", "content": text})
    return out


def _anthropic_tool_choice(tc: Any) -> Any:
    # Anthropic: {"type": "auto"|"any"|"tool"}. Only "auto" is a free model decision;
    # "any"/"tool" force a call, so map to a non-auto sentinel (mining skips forced).
    if isinstance(tc, dict):
        return "auto" if tc.get("type") == "auto" else "required"
    return None


def _parse_anthropic_jsonl(obj: Dict[str, Any], line_no: int) -> Exchange:
    """Adapter for logged Anthropic Messages API calls: {"request": <body>,
    "response": <message>} per line."""
    req, resp = obj.get("request"), obj.get("response")
    if not isinstance(req, dict) or not isinstance(resp, dict):
        raise ValueError("record needs 'request' and 'response' objects")
    if not isinstance(resp.get("content"), (list, str)):
        raise ValueError("record needs a 'response.content' (Anthropic Messages API)")
    try:
        status = int(obj.get("status", 200))
    except (TypeError, ValueError):
        status = 200
    text, calls, _ = _anthropic_blocks(resp.get("content"))
    response = ResponseMessage(
        content=text,
        tool_calls=[ToolCall(c["function"]["name"], c["function"]["arguments"]) for c in calls],
    )
    return Exchange(
        messages=_anthropic_messages_to_openai(req.get("system"), req.get("messages")),
        tools=_anthropic_tools_to_openai(req.get("tools")),
        response=response,
        tool_choice=_anthropic_tool_choice(req.get("tool_choice")),
        session_id=str(obj["session_id"]) if obj.get("session_id") else None,
        ts=str(obj.get("ts") or ""),
        model=str(req.get("model") or resp.get("model") or ""),
        status=status,
        line_no=line_no,
    )


_ADAPTERS = {
    "trace-v1": _parse_trace_v1,
    "openai-jsonl": _parse_openai_jsonl,
    "anthropic-jsonl": _parse_anthropic_jsonl,
}


def _detect_format(obj: Dict[str, Any]) -> str:
    resp = obj.get("response")
    if isinstance(resp, dict):
        if isinstance(resp.get("message"), dict):
            return "trace-v1"
        if isinstance(resp.get("choices"), list):
            return "openai-jsonl"
        if resp.get("type") == "message" or (
            isinstance(resp.get("content"), (list, str)) and resp.get("role") == "assistant"
        ):
            return "anthropic-jsonl"
    raise ValueError("unrecognized record shape")


# --- OpenTelemetry GenAI adapter --------------------------------------------
# Scoped DELIBERATELY to the GenAI semantic-convention attributes (gen_ai.*), NOT to any
# one library's private span layout — see the traces.py docstring for why probelock
# otherwise avoids parsing OTel. Reads an OTLP-JSON export (a resourceSpans document, a
# list, or a bare span). Spans without gen_ai prompt attributes are skipped and counted;
# the examples/otel_traces_to_probelock.py recipe remains for non-conforming exporters.


def _iter_spans(doc: Any):
    """Yield span dicts from an OTLP-JSON structure (resourceSpans/scopeSpans/spans), a
    list of such, or a single bare span."""
    if isinstance(doc, dict):
        if isinstance(doc.get("resourceSpans"), list):
            for rs in doc["resourceSpans"]:
                scopes = rs.get("scopeSpans") or rs.get("instrumentationLibrarySpans") or []
                for ss in scopes if isinstance(scopes, list) else []:
                    yield from (ss.get("spans") or [])
        elif isinstance(doc.get("scopeSpans"), list):
            for ss in doc["scopeSpans"]:
                yield from (ss.get("spans") or [])
        elif isinstance(doc.get("spans"), list):
            yield from doc["spans"]
        elif ("spanId" in doc or "traceId" in doc) and ("name" in doc or "attributes" in doc):
            # A bare span must carry OTLP span identity. Requiring spanId/traceId (not
            # just a "name" key) keeps a roll-your-own single JSONL record — which may
            # well have a top-level "name" — from being misread as an OTel span and
            # hard-failing the default --format auto path.
            yield doc
    elif isinstance(doc, list):
        for item in doc:
            yield from _iter_spans(item)


def _span_attributes(span: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a span's attributes to {key: python-value}, tolerating both the OTLP
    list-of-{key,value:{stringValue,...}} form and a plain flat dict."""
    attrs = span.get("attributes")
    if isinstance(attrs, dict):
        return attrs
    out: Dict[str, Any] = {}
    for kv in attrs if isinstance(attrs, list) else []:
        if not isinstance(kv, dict) or kv.get("key") is None:
            continue
        val = kv.get("value")
        if isinstance(val, dict):
            for vk in ("stringValue", "intValue", "doubleValue", "boolValue"):
                if vk in val:
                    out[kv["key"]] = val[vk]
                    break
            else:
                out[kv["key"]] = val  # arrayValue / kvlistValue: keep raw
        else:
            out[kv["key"]] = val
    return out


def _as_messages(value: Any) -> List[Dict[str, Any]]:
    """A gen_ai.prompt / gen_ai.completion attribute is a JSON-string array of messages
    (or already a list). Return the message dicts, else []."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return [m for m in value if isinstance(m, dict)] if isinstance(value, list) else []


def _otel_indexed_messages(attrs: Dict[str, Any], prefix: str) -> List[Dict[str, Any]]:
    """Reconstruct messages from OpenLLMetry-style indexed attributes
    (gen_ai.prompt.0.role / gen_ai.prompt.0.content / ...).

    Tool calls arrive EITHER as a single JSON-string attribute (gen_ai.completion.0.
    tool_calls) OR, in OpenLLMetry's own convention, as further-indexed sub-attributes
    (gen_ai.completion.0.tool_calls.0.name / .arguments / .id, optionally under a
    .function. segment). Both are reassembled into the canonical OpenAI tool_calls
    shape, so the frozen replay context is valid."""
    idx: Dict[int, Dict[str, Any]] = {}
    calls: Dict[int, Dict[int, Dict[str, str]]] = {}  # msg index -> call index -> fields
    marker = prefix + "."
    for key, val in attrs.items():
        if not key.startswith(marker):
            continue
        parts = key[len(marker):].split(".")
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        i = int(parts[0])
        msg = idx.setdefault(i, {})
        field = parts[1:]
        if field[0] == "tool_calls":
            if len(field) == 1 and isinstance(val, str):
                try:  # a JSON-blob list of tool calls
                    msg["tool_calls"] = json.loads(val)
                except json.JSONDecodeError:
                    pass
            elif len(field) >= 3 and field[1].isdigit():
                # tool_calls.{j}.name|arguments|id, or .{j}.function.name|arguments
                sub = field[3] if field[2] == "function" and len(field) >= 4 else field[2]
                calls.setdefault(i, {}).setdefault(int(field[1]), {})[sub] = val
        else:
            msg[".".join(field)] = val
    for i, per_call in calls.items():
        assembled = [
            {"id": per_call[j].get("id", ""), "type": "function",
             "function": {"name": per_call[j].get("name", ""),
                          "arguments": per_call[j].get("arguments", "{}")}}
            for j in sorted(per_call)
        ]
        if assembled:
            idx[i]["tool_calls"] = assembled
    return [idx[i] for i in sorted(idx)]


def _otel_indexed_tools(attrs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Reconstruct offered tools from OpenLLMetry-style indexed tool DEFINITIONS
    (gen_ai.request.functions.{i}.name/.description/.parameters, or the older
    llm.request.functions.{i}.*). Without this, indexed-form spans mine nothing: the
    reassembled tool CALL would name a tool that appears un-offered."""
    for prefix in ("gen_ai.request.functions", "llm.request.functions"):
        marker = prefix + "."
        idx: Dict[int, Dict[str, Any]] = {}
        for key, val in attrs.items():
            if not key.startswith(marker):
                continue
            parts = key[len(marker):].split(".", 1)
            if len(parts) != 2 or not parts[0].isdigit():
                continue
            idx.setdefault(int(parts[0]), {})[parts[1]] = val
        if not idx:
            continue
        tools = []
        for i in sorted(idx):
            fn = idx[i]
            params = fn.get("parameters")
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except json.JSONDecodeError:
                    params = {}
            tools.append({"type": "function", "function": {
                "name": str(fn.get("name") or ""),
                "description": str(fn.get("description") or ""),
                "parameters": params if isinstance(params, dict) else {},
            }})
        return _clean_tools(tools)
    return []


def _parse_otel_span(span: Dict[str, Any]) -> Optional[Exchange]:
    attrs = _span_attributes(span)
    prompt = _as_messages(attrs.get("gen_ai.prompt")) or _otel_indexed_messages(attrs, "gen_ai.prompt")
    if not prompt:
        return None  # not a GenAI chat span we can read
    completion = _as_messages(attrs.get("gen_ai.completion")) or _otel_indexed_messages(
        attrs, "gen_ai.completion"
    )
    # The assistant turn is the completion; fall back to an empty message so an
    # error/blank span still parses (and lands in failed_status, never mined).
    assistant = next((m for m in reversed(completion) if m.get("role") in (None, "assistant")), {})
    tools = _clean_tools(_as_messages_raw(attrs.get("gen_ai.request.tools"))) or _otel_indexed_tools(attrs)
    status_code = ((span.get("status") or {}).get("code")
                   if isinstance(span.get("status"), dict) else None)
    # OTLP status code 2 (or "STATUS_CODE_ERROR") means the span errored.
    errored = status_code in (2, "STATUS_CODE_ERROR", "ERROR")
    return Exchange(
        messages=prompt,
        tools=tools,
        response=_message_to_response(assistant),
        tool_choice="auto",
        session_id=str(span.get("traceId") or span.get("trace_id") or "") or None,
        ts=str(attrs.get("gen_ai.timestamp") or ""),
        model=str(attrs.get("gen_ai.request.model") or attrs.get("gen_ai.response.model") or ""),
        status=500 if errored else 200,
        line_no=0,
    )


def _as_messages_raw(value: Any) -> Any:
    """tools may be a JSON string or an already-parsed list; hand _clean_tools a list."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return value


def _load_otel(path, summary: MiningSummary) -> List[Exchange]:
    doc = json.loads(Path(path).read_text())
    exchanges: List[Exchange] = []
    for span in _iter_spans(doc):
        summary.records += 1
        ex = _parse_otel_span(span) if isinstance(span, dict) else None
        if ex is None:
            summary.skip("no_genai_attrs")
        elif ex.status >= 400:
            summary.skip("failed_status")
        elif not ex.messages:
            summary.skip("no_messages")
        else:
            exchanges.append(ex)
    return exchanges


def _looks_like_otel(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped.startswith(("{", "[")):
        return False
    try:
        doc = json.loads(stripped)
    except json.JSONDecodeError:
        return False
    return next(_iter_spans(doc), None) is not None


def load_exchanges(path, fmt: str = "auto") -> Tuple[List[Exchange], MiningSummary]:
    """Read a traffic log, adapting each record to an Exchange. Unparseable records and
    failed calls are skipped and counted, never silently dropped; a file that yields
    NOTHING raises ValueError (almost certainly the wrong --format). JSONL formats are
    read line-by-line; otel-genai reads the whole file as one OTLP-JSON document."""
    if fmt not in FORMATS:
        raise ValueError(f"unknown format '{fmt}' (use {' | '.join(FORMATS)})")
    summary = MiningSummary()

    text = Path(path).read_text()
    if fmt == "otel-genai" or (fmt == "auto" and _looks_like_otel(text)):
        exchanges = _load_otel(path, summary)
        if summary.records and not exchanges and not any(
            k != "no_genai_attrs" for k in summary.skipped
        ) and summary.skipped.get("no_genai_attrs") == summary.records:
            raise ValueError(f"no GenAI spans in {path} — is this really otel-genai format?")
        return exchanges, summary

    exchanges = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        summary.records += 1
        try:
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError("line is not a JSON object")
            kind = _detect_format(obj) if fmt == "auto" else fmt
            ex = _ADAPTERS[kind](obj, line_no)
        except (json.JSONDecodeError, ValueError, TypeError):
            summary.skip("malformed")
            continue
        if ex.status >= 400:
            summary.skip("failed_status")  # §3.2: failed upstream calls are never mined
            continue
        if not ex.messages:
            summary.skip("no_messages")
            continue
        exchanges.append(ex)
    if summary.records and not exchanges and summary.skipped.get("malformed") == summary.records:
        raise ValueError(
            f"no parseable records in {path} — is this really "
            f"{fmt if fmt != 'auto' else 'a supported'} format?"
        )
    return exchanges, summary


# --- normalization & hashing ------------------------------------------------

_TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?"
    r"|\b\d{4}-\d{2}-\d{2}\b|\b\d{2}:\d{2}:\d{2}\b"
)
_WS_RE = re.compile(r"\s+")


def _norm_text(text: str) -> str:
    return _WS_RE.sub(" ", _TIMESTAMP_RE.sub("<ts>", text)).strip()


def _norm_value(value: Any) -> Any:
    """Recursively normalize every string in a JSON-ish structure (timestamps out,
    whitespace collapsed) so clustering and prefix matching survive the cosmetic
    per-request differences (a clock in the system prompt, reflowed text) that would
    otherwise split identical contexts into distinct clusters."""
    if isinstance(value, str):
        return _norm_text(value)
    if isinstance(value, dict):
        return {k: _norm_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_norm_value(v) for v in value]
    return value


def _hash_json(obj: Any, length: int = 16) -> str:
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:length]


def _context_key(ex: Exchange) -> str:
    return _hash_json({"messages": ex.norm, "tools": _norm_value(ex.tools)})


def _norm_args(call: ToolCall) -> str:
    """Canonical form of a call's arguments for retry comparison: parsed and
    re-serialized when possible so key order never masquerades as 'corrected args'."""
    try:
        parsed = json.loads(call.arguments)
    except (json.JSONDecodeError, TypeError):
        return _norm_text(str(call.arguments))
    return json.dumps(_norm_value(parsed), sort_keys=True, separators=(",", ":"))


def _estimate_tokens(ex: Exchange) -> int:
    # chars/4 — deliberately a tokenizer-free estimate; this bounds replay cost, it
    # doesn't need to be exact.
    blob = json.dumps({"messages": ex.messages, "tools": ex.tools})
    return len(blob) // 4


# --- session stitching ------------------------------------------------------


def _is_prefix(shorter: List[Any], longer: List[Any]) -> bool:
    return len(shorter) < len(longer) and longer[: len(shorter)] == shorter


def stitch_sessions(exchanges: List[Exchange]) -> List[List[Exchange]]:
    """Group exchanges into ordered sessions and assign session_id/turn in place.

    Provided session_ids win. The rest are chained by containment against the current
    tail of every open session — provided-id sessions included, so a conversation whose
    logger tags only some records still stitches whole. A tail-EQUAL exchange (the same
    record logged twice by at-least-once shipping) joins the session instead of opening
    a phantom one: a duplicate must never mint a second "distinct session" for the
    min-agreement counts. Conversations grow linearly; a branched conversation starts a
    new session at the branch point, which only makes confirmation more conservative."""
    for ex in exchanges:
        ex.norm = _norm_value(ex.messages)

    by_id: Dict[str, List[Exchange]] = {}
    unlabeled: List[Exchange] = []
    for ex in exchanges:
        if ex.session_id:
            by_id.setdefault(ex.session_id, []).append(ex)
        else:
            unlabeled.append(ex)

    open_sessions: List[List[Exchange]] = []
    for group in by_id.values():
        group.sort(key=lambda e: (len(e.messages), e.ts, e.line_no))
        open_sessions.append(group)

    # Shortest first, so an extension always finds its base — and a duplicate always
    # finds its twin as some session's tail — already in place.
    unlabeled.sort(key=lambda e: (len(e.messages), e.ts, e.line_no))
    for ex in unlabeled:
        best: Optional[List[Exchange]] = None
        for sess in open_sessions:
            tail = sess[-1]
            if _is_prefix(tail.norm, ex.norm) or tail.norm == ex.norm:
                if best is None or len(tail.norm) > len(best[-1].norm):
                    best = sess
        if best is None:
            open_sessions.append([ex])
        else:
            best.append(ex)

    for sess in open_sessions:
        if not sess[0].session_id:
            sid = "sha256:" + _hash_json(sess[0].norm)
            for ex in sess:
                ex.session_id = sid
        else:
            for ex in sess:  # unlabeled records adopted into a provided-id session
                ex.session_id = sess[0].session_id
        sess.sort(key=lambda e: (len(e.norm), e.ts, e.line_no))
        for turn, ex in enumerate(sess):
            ex.turn = turn
    return open_sessions


# --- confirmed-good inference (§ design: provenance determines trust) --------

_ERROR_PREFIX_RE = re.compile(r"^\s*(error|exception|traceback)\b", re.IGNORECASE)


def _looks_like_error(content: Any) -> bool:
    """Deterministic 'this tool result is an error payload' check — a JSON object with a
    truthy 'error' key, or text that opens like a stack trace. Deliberately narrow: a
    result merely *containing* the word error ('no errors found') must not count."""
    if not isinstance(content, str) or not content.strip():
        return False
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return bool(_ERROR_PREFIX_RE.match(content))
    return isinstance(parsed, dict) and bool(parsed.get("error") or parsed.get("errors"))


def _trailing_user(norm_messages: List[Dict[str, Any]]) -> Optional[Any]:
    for m in reversed(norm_messages):
        if isinstance(m, dict) and m.get("role") == "user":
            return m.get("content")
    return None


def _call_names(msg: Dict[str, Any]) -> List[str]:
    names = []
    for tc in msg.get("tool_calls") or []:
        if isinstance(tc, dict):
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
            names.append(fn.get("name") or "")
    return names


def confirms_continuation(ex: Exchange, session: List[Exchange]) -> bool:
    """Confirmed-good rule 1: the call executed, its result fed back, and the
    conversation moved on. Disqualified by an error payload in the result, by a
    same-tool retry with corrected arguments anywhere in the tool loop before the
    user's next turn, or by that next user turn re-asking the same question.

    The scan must walk ALL extensions of this exchange up to the first new user turn:
    in a real agent loop the immediate continuation is just [assistant(call),
    tool(result)] with no user message in it — the retry and re-ask evidence only
    appears one or more exchanges later."""
    if not ex.response.tool_calls:
        return False
    call = ex.response.tool_calls[0]
    extensions = sorted(
        (e for e in session if _is_prefix(ex.norm, e.norm)), key=lambda e: len(e.norm)
    )
    if not extensions:
        return False
    ext = extensions[0].messages[len(ex.messages):]
    fed_back = any(
        isinstance(m, dict) and m.get("role") == "assistant" and call.name in _call_names(m)
        for m in ext
    )
    results = [m for m in ext if isinstance(m, dict) and m.get("role") == "tool"]
    if not fed_back or not results:
        return False
    # The result for THIS call: match by tool name when the log carries one, else the
    # first tool message in the extension (single-call agents, the common case).
    result = next((m for m in results if m.get("name") == call.name), results[0])
    if _looks_like_error(result.get("content")):
        return False

    last_user = _trailing_user(ex.norm)
    for e in extensions:
        new_users = [
            m for m in e.norm[len(ex.norm):]
            if isinstance(m, dict) and m.get("role") == "user"
        ]
        if new_users:
            # The user's next turn decides: an identical question means the call did
            # not do the job; anything else means the conversation moved on.
            return not (last_user is not None and new_users[0].get("content") == last_user)
        # Still inside the tool loop (no user intervention yet): a same-tool call with
        # different args is the agent correcting itself — a retry.
        for c in e.response.tool_calls:
            if c.name == call.name and _norm_args(c) != _norm_args(call):
                return False
    return True


def _reasked_later(ex: Exchange, session: List[Exchange]) -> bool:
    """True when the same user turn shows up again later in the session — for no-tool
    mining, evidence the text answer did NOT end the task."""
    last_user = _trailing_user(ex.norm)
    if last_user is None:
        return False
    for e in session:
        if not _is_prefix(ex.norm, e.norm):
            continue
        for m in e.norm[len(ex.norm):]:
            if isinstance(m, dict) and m.get("role") == "user" and m.get("content") == last_user:
                return True
    return False


def _tool_overlap(ex: Exchange) -> float:
    """Lexical overlap between the final user turn and the offered tools' names and
    descriptions — the dependency-free stand-in for semantic distance. Low overlap means
    the tools are clearly unrelated to the query, exactly the contexts where no-tool
    restraint is unambiguous (§ design: mined preferentially)."""
    user = _trailing_user(ex.norm)
    user_tokens = set(re.findall(r"[a-z0-9_]+", str(user or "").lower()))
    if not user_tokens:
        return 1.0
    tool_text = " ".join(
        f"{t.get('function', {}).get('name', '')} {t.get('function', {}).get('description', '')}"
        for t in ex.tools
    )
    tool_tokens = set(re.findall(r"[a-z0-9_]+", tool_text.lower()))
    return len(user_tokens & tool_tokens) / len(user_tokens)


# --- redaction (§ design: on by default) --------------------------------------

REDACT_PATTERNS: Dict[str, re.Pattern] = {
    "emails": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    # >= 7 digits/separators total, so short national formats (8-digit mobiles) are
    # caught too; scrubbing errs toward over-matching — this is the committable path.
    "phones": re.compile(r"\+?\d[\d\s().-]{5,}\d"),
    "paths": re.compile(r"(?:[A-Za-z]:)?[\\/](?:[\w.~-]+[\\/])+[\w.~-]+"),
}
_CONSTRAINED_STRING_KEYS = ("pattern", "format", "minLength", "maxLength")


def _redact_arg_value(value: Any, prop_schema: Optional[Dict[str, Any]]) -> Any:
    """Structure-preserving redaction of one argument value. Free-text strings become
    '<str:NNch>' placeholders; strings whose schema constrains their shape get a
    deterministic synthetic value instead (a placeholder would break schema validity,
    and these enum-like/formatted fields are not free text); const/enum and non-string
    scalars pass through — they are the fields a check could depend on."""
    schema = prop_schema if isinstance(prop_schema, dict) else {}
    if "const" in schema or schema.get("enum"):
        return value
    if isinstance(value, str):
        if any(k in schema for k in _CONSTRAINED_STRING_KEYS):
            return synth_value(schema)
        return f"<str:{len(value)}ch>"
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        return {k: _redact_arg_value(v, props.get(k)) for k, v in value.items()}
    if isinstance(value, list):
        items = schema.get("items")
        return [_redact_arg_value(v, items) for v in value]
    return value


def redact_args(args: Dict[str, Any], schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    props = (schema.get("properties") if isinstance(schema, dict) else None) or {}
    return {k: _redact_arg_value(v, props.get(k)) for k, v in args.items()}


def _scrub_text(text: str, patterns: Tuple[str, ...]) -> str:
    for name in patterns:
        text = REDACT_PATTERNS[name].sub(f"<{name[:-1]}>", text)
    return text


def _scrub_value(value: Any, patterns: Tuple[str, ...]) -> Any:
    """Pattern-scrub every string inside a JSON-ish structure. Message content isn't
    always a plain string — the OpenAI format also allows a list of parts
    ([{"type": "text", "text": ...}]), and a committable probe must not leak through
    that shape either."""
    if isinstance(value, str):
        return _scrub_text(value, patterns)
    if isinstance(value, dict):
        return {k: _scrub_value(v, patterns) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub_value(v, patterns) for v in value]
    return value


def _tool_schema(tools: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function")
        if isinstance(fn, dict) and fn.get("name") == name:
            params = fn.get("parameters")
            # A non-dict 'parameters' (seen in sloppy logs) can't be validated against;
            # treat it as the empty schema rather than crashing downstream.
            return params if isinstance(params, dict) else {}
    return None


def redact_context(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    patterns: Tuple[str, ...],
) -> List[Dict[str, Any]]:
    """The frozen-context redaction pass: historical tool-call ARGUMENTS are always
    placeholder-redacted (schema-aware where the tool is known); message CONTENT is kept
    verbatim unless the user opted into --redact-patterns scrubbing — content is what
    makes replay realistic, which is why verbatim probes carry sensitive: true."""
    redacted = copy.deepcopy(messages)
    for msg in redacted:
        if not isinstance(msg, dict):
            continue
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    fn["arguments"] = "{}"
                    continue
            # Some servers log arguments already parsed (a dict), and some models emit
            # non-object argument payloads (a bare string, an array) — every shape must
            # come out redacted, none may pass through verbatim.
            if args is None:
                fn["arguments"] = "{}"
            elif isinstance(args, dict):
                schema = _tool_schema(tools, fn.get("name") or "")
                fn["arguments"] = json.dumps(redact_args(args, schema))
            else:
                fn["arguments"] = json.dumps(_redact_arg_value(args, None))
        if patterns and "content" in msg:
            msg["content"] = _scrub_value(msg["content"], patterns)
    return redacted


def _reference_args(call: ToolCall, schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Redacted recorded args for the probe's simulator reference — falling back to
    synthesized args when the redacted (or original) ones don't validate, so a
    'passing' simulated response actually passes the scorer."""
    try:
        recorded = json.loads(call.arguments)
    except (json.JSONDecodeError, TypeError):
        recorded = {}
    if not isinstance(recorded, dict):
        recorded = {}
    args = redact_args(recorded, schema)
    try:
        jsonschema.validate(args, schema or {})
    except (jsonschema.ValidationError, jsonschema.SchemaError):
        args = synth_args(schema or {})
    return args


# --- mining -------------------------------------------------------------------


@dataclass
class _Candidate:
    category: str
    exchange: Exchange  # cluster representative (members are normalized-identical)
    tool: Optional[str]
    rule: str
    sessions: int
    overlap: float = 0.0  # no_tool only: lexical query↔tools overlap, lower is better


def _mineable(ex: Exchange) -> bool:
    # A forced tool_choice means the recorded outcome reflects the FORCE, not a model
    # decision — and replay carries only messages+tools, never tool_choice, so a probe
    # minted from a forced exchange would hold the candidate to a decision the recorded
    # model never made.
    return ex.tool_choice in (None, "auto")


def _cluster(exchanges: List[Exchange]) -> Dict[str, List[Exchange]]:
    clusters: Dict[str, List[Exchange]] = {}
    for ex in exchanges:
        clusters.setdefault(_context_key(ex), []).append(ex)
    return clusters


def _context_text(ex: Exchange) -> str:
    return json.dumps({"messages": ex.norm, "tools": _norm_value(ex.tools)},
                      sort_keys=True, separators=(",", ":"))


def _embed(texts: List[str], config: MiningConfig) -> List[List[float]]:
    """Fetch embeddings from an OpenAI-compatible /embeddings endpoint (stdlib only)."""
    import urllib.request

    base = config.embed_endpoint.rstrip("/")
    body = json.dumps({"model": config.embed_model, "input": texts}).encode()
    req = urllib.request.Request(
        f"{base}/embeddings", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as fh:  # noqa: S310
        payload = json.loads(fh.read())
    # Validate the OpenAI shape defensively: any deviation (a bare array, raw-vector
    # rows, a null index) raises ValueError, which _cluster_by_embeddings catches and
    # degrades to deterministic hash clustering rather than crashing the ingest run.
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) != len(texts):
        raise ValueError(
            f"embeddings endpoint returned an unexpected shape for {len(texts)} inputs"
        )

    def _row_index(row):
        i = row.get("index") if isinstance(row, dict) else None
        return i if isinstance(i, int) else 0

    vectors = []
    for row in sorted(rows, key=_row_index):  # preserve request order
        emb = row.get("embedding") if isinstance(row, dict) else None
        if not isinstance(emb, list):
            raise ValueError("embeddings endpoint returned a row without an 'embedding' list")
        vectors.append(emb)
    return vectors


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _cluster_by_embeddings(
    exchanges: List[Exchange], config: MiningConfig
) -> Dict[str, List[Exchange]]:
    """Group NEAR-duplicate contexts by embedding cosine >= cluster_threshold. Exact
    hash-duplicates are collapsed first (one embedding per distinct context, cheaper and
    stable), then representatives are greedily grouped. Deterministic given the same
    embedding vectors; the embeddings themselves are model-dependent — hence the caveat.
    Falls back to exact-hash clustering if the endpoint is unreachable."""
    exact = _cluster(exchanges)  # {context_key: [exchanges]}
    keys = sorted(exact)  # deterministic order for greedy grouping
    if len(keys) < 2:
        return exact
    reps = [exact[k][0] for k in keys]
    try:
        vectors = _embed([_context_text(ex) for ex in reps], config)
    except Exception as exc:  # noqa: BLE001 - opt-in convenience: any failure degrades
        # The documented contract is "never crash the run": an unreachable endpoint, a
        # non-JSON body, or any unexpected 200 shape all fall back to deterministic
        # hash clustering rather than aborting ingest.
        _warn_embed(f"embedding failed ({exc}); falling back to exact-hash clustering")
        return exact

    merged: Dict[str, List[Exchange]] = {}
    canonical: List[Tuple[str, List[float]]] = []  # (group key, its representative vector)
    for key, vec in zip(keys, vectors):
        match = next((gk for gk, gv in canonical
                      if _cosine(vec, gv) >= config.cluster_threshold), None)
        if match is None:
            canonical.append((key, vec))
            merged[key] = list(exact[key])
        else:
            merged[match].extend(exact[key])
    return merged


def _warn_embed(message: str) -> None:
    import sys
    print(f"probelock ingest: {message}", file=sys.stderr)


def _candidates_for_cluster(
    members: List[Exchange],
    sessions_by_id: Dict[str, List[Exchange]],
    config: MiningConfig,
    summary: MiningSummary,
) -> List[_Candidate]:
    members = sorted(members, key=lambda e: (e.session_id or "", e.turn, e.line_no))
    rep = members[0]
    callers = [m for m in members if m.response.tool_calls]
    texters = [m for m in members if not m.response.tool_calls and m.response.content]
    out: List[_Candidate] = []

    if callers:
        # Only calls to tools actually in the exchange's own offered set count — for
        # schema_validity there is no declared schema to hold the candidate to
        # otherwise, and for tool_selection a hallucinated name confirmed as "expected"
        # would mint a probe no correct candidate could ever pass (the same rule
        # mined.edit_expected_tool enforces at review time).
        valid_callers = [
            m for m in callers if _tool_schema(m.tools, m.response.tool_calls[0].name) is not None
        ]
        if not valid_callers:
            summary.skip("called_tool_not_offered")
        else:
            sess = {m.session_id for m in valid_callers}
            out.append(
                _Candidate(
                    "schema_validity", valid_callers[0], valid_callers[0].response.tool_calls[0].name,
                    "schema-validity", len(sess),
                )
            )

            # Tool selection: needs confirmed-good. Rule 1 (continuation) on any member,
            # else rule 2 (cross-session agreement on ONE name).
            confirmed = next(
                (m for m in valid_callers if confirms_continuation(m, sessions_by_id[m.session_id])),
                None,
            )
            if confirmed is not None:
                name = confirmed.response.tool_calls[0].name
                sess = {m.session_id for m in valid_callers
                        if m.response.tool_calls[0].name == name}
                out.append(_Candidate("tool_selection", confirmed, name, "continuation", len(sess)))
            else:
                by_name: Dict[str, set] = {}
                for m in valid_callers:
                    by_name.setdefault(m.response.tool_calls[0].name, set()).add(m.session_id)
                agreeing = [n for n, s in by_name.items() if len(s) >= config.min_agreement]
                if len(agreeing) == 1:
                    name = agreeing[0]
                    out.append(
                        _Candidate(
                            "tool_selection",
                            next(m for m in valid_callers
                                 if m.response.tool_calls[0].name == name),
                            name, "min-agreement", len(by_name[name]),
                        )
                    )
                elif len(agreeing) > 1:
                    summary.ambiguous_tool_selection += 1
                else:
                    summary.unconfirmed_tool_clusters += 1

    # No-tool restraint: strictest rules (a mislabeled probe here freezes a model
    # mistake as expected behavior). Unanimous no-tool across the cluster, agreement
    # from >= min_agreement_notool distinct sessions, tools actually offered (restraint
    # is only meaningful when the model COULD have called something), and no member's
    # user turn re-asked later.
    if texters and not callers and rep.tools:
        sess = {m.session_id for m in texters}
        if len(sess) >= config.min_agreement_notool and not any(
            _reasked_later(m, sessions_by_id[m.session_id]) for m in texters
        ):
            out.append(
                _Candidate("no_tool", rep, None, "min-agreement-notool", len(sess),
                           overlap=_tool_overlap(rep))
            )
    return out


def _sample(candidates: List[_Candidate], config: MiningConfig) -> List[_Candidate]:
    """Up to per_capability probes per (tool, category), preferring longer contexts and
    later turns — the coverage synthetic probes lack — and, for no_tool, low lexical
    overlap first (the unambiguous-restraint contexts)."""
    groups: Dict[Tuple[str, str], List[_Candidate]] = {}
    for c in candidates:
        # no_tool candidates have no tool name; bucket them per offered TOOLSET so a
        # log spanning several agents doesn't squeeze all restraint probes — the
        # category quantization breaks first — through one shared cap.
        group_tool = c.tool or _hash_json(_norm_value(c.exchange.tools), 8)
        groups.setdefault((group_tool, c.category), []).append(c)
    kept: List[_Candidate] = []
    for group in groups.values():
        group.sort(
            key=lambda c: (
                c.overlap,
                -len(c.exchange.messages),
                -c.exchange.turn,
                c.exchange.session_id or "",
                c.exchange.line_no,
            )
        )
        kept.extend(group[: config.per_capability])
    return kept


def _freeze(candidate: _Candidate, config: MiningConfig, mined_at: str) -> MinedProbe:
    ex = candidate.exchange
    session = (ex.session_id or "").removeprefix("sha256:")
    # Abbreviate long bare content hashes for readable ids; structured ids (the
    # proxy's "pxy:<run>:<n>") stay whole — truncating those would collide every
    # session in a run onto one prefix and leave probe identity to the
    # order-dependent dedup guard.
    if len(session) > 16 and all(c in "0123456789abcdef" for c in session):
        session = session[:12]
    sess12 = session
    reference: Dict[str, Any] = {}
    if candidate.category == "no_tool":
        reference = {"content": "(answered in text, no tool call)"}
    elif ex.response.tool_calls:
        call = ex.response.tool_calls[0]
        schema = _tool_schema(ex.tools, call.name)
        reference = {"tool": call.name, "valid_args": _reference_args(call, schema)}
    provenance: Dict[str, Any] = {
        "sessions": candidate.sessions,
        "rule": candidate.rule,
        "mined_at": mined_at,
        "model": ex.model,
        "source": config.source,
    }
    if config.redact_patterns:
        provenance["redact_patterns"] = list(config.redact_patterns)
    if config.cluster != "hash":
        # Non-deterministic dedup: record it so a probe's provenance shows the grouping
        # was not the reproducible hash path.
        provenance["cluster"] = config.cluster
    return MinedProbe(
        id=f"trace:{sess12}:t{ex.turn}",
        category=candidate.category,
        messages=redact_context(ex.messages, ex.tools, config.redact_patterns),
        tools=ex.tools,
        tool=candidate.tool,
        status="pending",
        provenance=provenance,
        # Verbatim real conversation content is sensitive by default; --redact-patterns
        # is the user's explicit committable-probes path (recorded in provenance, so the
        # trust decision stays traceable). Argument redaction alone doesn't clear the
        # flag — message content is the leak surface.
        sensitive=not config.redact_patterns,
        reference=reference,
    )


def mine_exchanges(
    exchanges: List[Exchange], config: MiningConfig, summary: Optional[MiningSummary] = None
) -> Tuple[List[MinedProbe], MiningSummary]:
    summary = summary or MiningSummary()
    for name in config.redact_patterns:
        if name not in REDACT_PATTERNS:
            raise ValueError(
                f"unknown redact pattern '{name}' (use {', '.join(sorted(REDACT_PATTERNS))})"
            )
    if config.cluster not in ("hash", "embeddings"):
        raise ValueError(f"unknown cluster mode '{config.cluster}' (use hash | embeddings)")
    if config.cluster == "embeddings" and not (config.embed_endpoint and config.embed_model):
        raise ValueError("--cluster embeddings requires --embed-endpoint and --embed-model")

    sessions = stitch_sessions(exchanges)
    summary.sessions = len({e.session_id for s in sessions for e in s})
    sessions_by_id: Dict[str, List[Exchange]] = {}
    for sess in sessions:
        # Provided-id sessions and containment-stitched ones can share an id (identical
        # roots hash identically); merge so confirmation sees the whole conversation.
        sessions_by_id.setdefault(sess[0].session_id, []).extend(sess)

    mineable: List[Exchange] = []
    for ex in exchanges:
        if not _mineable(ex):
            summary.skip("forced_tool_choice")
        elif _estimate_tokens(ex) > config.max_context_tokens:
            summary.skip("over_token_cap")
        else:
            mineable.append(ex)

    if config.cluster == "embeddings":
        clusters = _cluster_by_embeddings(mineable, config)
    else:
        clusters = _cluster(mineable)
    summary.clusters = len(clusters)

    candidates: List[_Candidate] = []
    for members in clusters.values():
        candidates.extend(_candidates_for_cluster(members, sessions_by_id, config, summary))
    for c in candidates:
        summary.candidates[c.category] = summary.candidates.get(c.category, 0) + 1

    mined_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
    probes: List[MinedProbe] = []
    seen_ids = set()
    for c in _sample(candidates, config):
        probe = _freeze(c, config, mined_at)
        # Distinct clusters can collapse to one (session, turn) id only if stitching
        # split what normalization later merged; suffix rather than drop.
        while (probe.id, probe.category) in seen_ids:
            probe.id += "+"
        seen_ids.add((probe.id, probe.category))
        probes.append(probe)
    probes.sort(key=lambda p: (p.category, p.id))
    for p in probes:
        summary.emitted[p.category] = summary.emitted.get(p.category, 0) + 1
    return probes, summary


def ingest_files(paths, fmt: str, config: MiningConfig) -> Tuple[List[MinedProbe], MiningSummary]:
    """Load one or more JSONL logs as a SINGLE corpus and mine — the `probelock
    ingest` entry point. Multiple paths exist for rotated proxy logs: a session that
    spans a rotation boundary keeps its continuation evidence only when the segments
    are stitched together in one load. Raises FileNotFoundError / ValueError /
    json.JSONDecodeError on bad input, per the cli.py error-wrapping convention."""
    summary = MiningSummary()
    exchanges: List[Exchange] = []
    for path in paths:
        loaded, part = load_exchanges(path, fmt)
        exchanges.extend(loaded)
        summary.records += part.records
        for reason, n in part.skipped.items():
            summary.skip(reason, n)
    if not config.source:
        config.source = ",".join(sorted({Path(p).name for p in paths}))
    return mine_exchanges(exchanges, config, summary)


def ingest_file(path, fmt: str, config: MiningConfig) -> Tuple[List[MinedProbe], MiningSummary]:
    return ingest_files([path], fmt, config)
