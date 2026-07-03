"""Recording reverse proxy (``probelock proxy``) — Layer 1 of trace ingestion.

A local HTTP proxy speaking the OpenAI-compatible chat API. The agent changes one line
(``base_url = "http://127.0.0.1:8484/v1"``); the proxy forwards everything to the
upstream and appends one trace-v1 record — exactly the shape ``probelock ingest``
reads — per completed chat-completions exchange.

Prime directive: **strictly non-invasive.** Recording is asynchronous (bounded queue +
writer thread) and every capture step is wrapped so that on ANY internal logging error
the request is still forwarded and a warning goes to stderr. A record may be dropped;
a request must never be.

Stdlib only, matching clients.py: a ThreadingHTTPServer is plenty for the local
single-agent case this reference implementation targets. Hard-won details encoded here
(each was a confirmed plan-review finding — change them knowingly):

  * The relay loop uses ``resp.read1()``, never ``resp.read(n)``: ``read(n)`` buffers
    ACROSS transfer-encoding chunks until n bytes fill, which turns token-by-token SSE
    into multi-second 8 KB bursts. ``read1()`` returns per-chunk.
  * ``Accept-Encoding`` is stripped on the upstream leg (http.client then negotiates
    identity itself): a gzip-capable upstream would otherwise make every capture
    unparseable while the proxy looks perfectly healthy. This is a deliberate
    deviation from "forward headers unchanged".
  * Framing: http.client transparently de-chunks upstream bodies, so the relayed
    response must never copy ``Transfer-Encoding``. Content-Length is passed through
    when the upstream sent one; otherwise the response is close-delimited
    (``Connection: close``).
  * Mid-stream failures get synthetic failed statuses — 499 when the client hung up,
    502 when the upstream died after its 200 — so a truncated half-generated message
    lands in ingest's ``failed_status`` bucket instead of being mined as a good
    exchange. On client disconnect the upstream connection is closed immediately:
    draining it would burn the model slot generating tokens nobody will read.
  * Session ids are opaque unique ids, NOT content hashes (the design doc's literal
    prefix-hash formula cannot group a conversation's exchanges, and hashing the
    transcript would collapse reruns of the same task — silently defeating
    cross-session min-agreement). The stitcher keys its map on the post-response
    transcript prefix, which a genuine continuation must contain and a fresh rerun
    cannot; identical requests inside a short TTL are treated as HTTP-level retries
    (same session), after it as reruns (new session).
"""

from __future__ import annotations

import datetime as _dt
import http.client
import itertools
import json
import os
import queue
import sys
import threading
import time
import urllib.parse
from collections import OrderedDict
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .ingest import _hash_json, _norm_value

_RELAY_CHUNK = 65536
_MAX_CAPTURE = 64 * 1024 * 1024  # capture safety cap; the response itself still relays
_EMPTY_MESSAGE: Dict[str, Any] = {"role": "assistant", "content": None, "tool_calls": []}

_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "trailers", "transfer-encoding", "upgrade",
}
# content-length is recomputed per leg; expect would stall http.client; accept-encoding
# per the module docstring. server/date are re-issued by our own send_response — relaying
# the upstream's copies too would emit duplicates.
_STRIP_REQUEST = _HOP_BY_HOP | {"host", "content-length", "accept-encoding", "expect"}
_STRIP_RESPONSE = _HOP_BY_HOP | {"content-length", "server", "date"}


def _warn(message: str) -> None:
    print(f"probelock proxy: {message}", file=sys.stderr)


@dataclass
class ProxyConfig:
    upstream: str
    out: Path
    listen_host: str = "127.0.0.1"
    listen_port: int = 8484
    timeout: float = 300.0  # upstream read; local generations are slow
    connect_timeout: float = 10.0  # fail fast when the upstream is down/black-holed
    max_size_mb: float = 100.0
    max_age_min: float = 0.0  # 0 = never rotate by age
    retry_ttl: float = 300.0  # identical request inside this window = HTTP retry


# --- async trace writer -------------------------------------------------------


class TraceWriter:
    """Bounded-queue JSONL appender with rotation. Runs in its own thread; ``put``
    never blocks and never raises — the request path must not depend on the disk."""

    def __init__(self, path, max_size_mb: float = 100.0, max_age_min: float = 0.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.written = 0
        self.dropped = 0
        self.rotated = 0
        self._max_bytes = int(max_size_mb * 1024 * 1024) if max_size_mb else 0
        self._max_age = max_age_min * 60.0 if max_age_min else 0.0
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=1024)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._fh = None
        self._opened_at = 0.0
        self._thread = threading.Thread(
            target=self._run, name="probelock-trace-writer", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def put(self, record: Dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            with self._lock:
                self.dropped += 1
            _warn("trace queue full — dropped a record")

    def close(self) -> None:
        """Stop accepting the loop and drain whatever is queued. Records from requests
        still mid-flight at shutdown may be lost — a documented, bounded trade-off of
        daemonized handler threads."""
        self._stop.set()
        self._thread.join(timeout=10)

    # --- writer thread ---------------------------------------------------------

    def _open(self) -> None:
        preexisting = self.path.exists() and self.path.stat().st_size > 0
        # 0o600: this file is where verbatim conversation content first lands on disk.
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        self._fh = os.fdopen(fd, "w", encoding="utf-8")
        # Treat pre-existing content as at least as old as its last write, so --max-age
        # doesn't restart the clock every time the proxy does.
        self._opened_at = self.path.stat().st_mtime if preexisting else time.time()

    def _rotate_target(self) -> Path:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        target = self.path.with_name(f"{self.path.stem}-{stamp}{self.path.suffix}")
        n = 1
        while target.exists():  # same-second rotations must not clobber a segment
            target = self.path.with_name(f"{self.path.stem}-{stamp}-{n}{self.path.suffix}")
            n += 1
        return target

    def _maybe_rotate(self) -> None:
        """Checked BEFORE each append (and periodically while idle, so --max-age fires
        on a quiet proxy rather than stamping stale epochs into a fresh record's file)."""
        if self._fh is None:
            return
        size = os.fstat(self._fh.fileno()).st_size
        if size <= 0:
            return
        too_big = bool(self._max_bytes) and size >= self._max_bytes
        too_old = bool(self._max_age) and (time.time() - self._opened_at) >= self._max_age
        if not (too_big or too_old):
            return
        self._fh.close()
        self._fh = None
        try:
            self.path.rename(self._rotate_target())
            with self._lock:
                self.rotated += 1
        except OSError as exc:  # keep appending to the same file rather than dying
            _warn(f"rotation failed: {exc}")

    def _append(self, record: Dict[str, Any]) -> None:
        try:
            if self._fh is None:
                self._open()
            line = json.dumps(record, ensure_ascii=False)
            try:
                line.encode("utf-8")
            except UnicodeEncodeError:
                # json.loads happily produces lone surrogates (a "\\ud800" escape in
                # someone's message content) and ensure_ascii=False re-emits them,
                # which only explodes at write time; ensure_ascii re-escapes them so
                # the record survives instead of killing the writer thread.
                line = json.dumps(record, ensure_ascii=True)
        except Exception as exc:
            with self._lock:
                self.dropped += 1
            _warn(f"trace record not serializable: {exc}")
            return
        try:
            self._fh.write(line + "\n")
            self._fh.flush()
            with self._lock:
                self.written += 1
        except Exception as exc:  # the writer thread must never die (prime directive)
            with self._lock:
                self.dropped += 1
            _warn(f"trace write failed: {exc}")

    def _run(self) -> None:
        try:
            # Eager open so the very first age check sees the pre-existing file's
            # clock — a proxy restarting over an over-age file must rotate BEFORE
            # stamping a fresh record into the stale segment.
            self._open()
        except OSError as exc:
            _warn(f"could not open trace file: {exc}")
        while True:
            try:
                record = self._queue.get(timeout=1.0)
            except queue.Empty:
                if self._stop.is_set():
                    break
                self._maybe_rotate()  # age-based rotation must fire on idle too
                continue
            self._maybe_rotate()
            self._append(record)
        while True:  # drain what arrived before the stop
            try:
                record = self._queue.get_nowait()
            except queue.Empty:
                break
            self._append(record)
        if self._fh is not None:
            self._fh.close()
            self._fh = None


# --- session stitching ----------------------------------------------------------


class SessionStitcher:
    """Opaque per-conversation session ids without agent cooperation.

    ``begin`` is called with a request's messages, ``complete`` after the response is
    assembled. The only stored lookup keys are post-response transcript prefixes
    (``hash(norm(messages + [assistant]))``) — a genuine continuation's messages start
    with exactly that, while a rerun of the same task cannot contain the assistant
    turn it never saw. See the module docstring for why ids are not content hashes.
    """

    def __init__(self, retry_ttl: float = 300.0, max_entries: int = 4096):
        self._retry_ttl = retry_ttl
        self._max = max_entries
        self._prefixes: "OrderedDict[str, str]" = OrderedDict()
        self._recent: "OrderedDict[str, Tuple[str, float]]" = OrderedDict()
        self._lock = threading.Lock()
        self._count = itertools.count(1)
        # Stamp + salt: distinct proxy runs mint distinct ids even across restarts
        # over the same log file.
        self._run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + os.urandom(3).hex()

    def _new_sid(self) -> str:
        return f"pxy:{self._run_id}:{next(self._count):05d}"

    def begin(self, messages: List[Any]) -> Tuple[str, List[Any]]:
        norm = _norm_value(messages)
        # Retry detection keys on the RAW messages: a transport-level retry resends
        # identical bytes, while a rerun of the same task typically differs somewhere
        # (a timestamp, an id) that normalization would have erased.
        request_key = _hash_json(messages)
        now = time.monotonic()
        with self._lock:
            hit = self._recent.get(request_key)
            if hit is not None and now - hit[1] <= self._retry_ttl:
                # The same bytes again this soon is an HTTP-level retry, not new
                # agreement evidence. The window is anchored at FIRST sight and never
                # slides — otherwise identical periodic requests (a cron-style agent)
                # would collapse into one session forever, silently making
                # cross-session min-agreement unreachable.
                sid = hit[0]
            else:
                sid = None
                for k in range(len(norm) - 1, 0, -1):  # longest PROPER prefix wins
                    found = self._prefixes.get(_hash_json(norm[:k]))
                    if found is not None:
                        sid = found
                        break
                if sid is None:
                    sid = self._new_sid()
                self._recent[request_key] = (sid, now)
                while len(self._recent) > self._max:
                    self._recent.popitem(last=False)
        return sid, norm

    def complete(self, sid: str, norm_messages: List[Any], assistant_message: Dict[str, Any]) -> None:
        key = _hash_json(list(norm_messages) + [_norm_value(assistant_message)])
        with self._lock:
            self._prefixes[key] = sid
            self._prefixes.move_to_end(key)
            while len(self._prefixes) > self._max:
                self._prefixes.popitem(last=False)


# --- response capture -------------------------------------------------------------


def _sse_data_events(body: bytes) -> List[str]:
    """The ``data:`` payloads of an SSE stream. Line-based on purpose: every
    OpenAI-compatible server emits single-line data events; multi-line events are
    v0.3.x hardening territory."""
    events = []
    for raw in body.split(b"\n"):
        line = raw.strip()
        if line.startswith(b"data:"):
            events.append(line[5:].strip().decode("utf-8", "replace"))
    return events


def assemble_sse(body: bytes) -> Optional[Dict[str, Any]]:
    """Reassemble chat.completion.chunk deltas into one assistant message.

    Tolerates the shapes real streams contain: ``[DONE]``, garbage lines, and the
    empty-``choices`` usage chunk that ``stream_options: {"include_usage": true}``
    appends (indexing ``choices[0]`` blindly would throw and silently drop every
    streamed record from such agents). Returns None when nothing assembles.
    """
    content_parts: List[str] = []
    calls: "OrderedDict[Any, Dict[str, Any]]" = OrderedDict()
    finish_reason = None
    saw_delta = False
    for event in _sse_data_events(body):
        if event == "[DONE]":
            continue
        try:
            obj = json.loads(event)
        except json.JSONDecodeError:
            continue
        choices = obj.get("choices") if isinstance(obj, dict) else None
        if not isinstance(choices, list):
            continue
        # Only choice 0 is the recorded decision: with n>1 every choice's deltas
        # arrive as separate chunks, and merging them would weave two sampled
        # answers into one garbage message recorded as a good exchange.
        first = next(
            (c for c in choices if isinstance(c, dict) and int(c.get("index") or 0) == 0),
            None,
        )
        if first is None:
            continue  # e.g. the include_usage summary chunk (empty choices)
        if first.get("finish_reason"):
            finish_reason = first["finish_reason"]
            saw_delta = True
        delta = first.get("delta") if isinstance(first.get("delta"), dict) else {}
        if isinstance(delta.get("content"), str):
            content_parts.append(delta["content"])
            saw_delta = True
        for tc in delta.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            saw_delta = True
            # Slot key: the index when the server sends one; else the call id (some
            # servers omit index — two parallel calls must not merge into one slot).
            if "index" in tc and tc["index"] is not None:
                key: Any = int(tc["index"])
            else:
                key = tc.get("id") or 0
            slot = calls.setdefault(key, {"id": None, "name": None, "args": []})
            if tc.get("id"):
                slot["id"] = tc["id"]
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            if fn.get("name"):
                slot["name"] = fn["name"]
            if isinstance(fn.get("arguments"), str):
                slot["args"].append(fn["arguments"])
    if not saw_delta:
        return None
    message: Dict[str, Any] = {"role": "assistant", "content": "".join(content_parts) or None}
    if calls:
        # A server indexes all calls or none: numeric indexes sort (their chunks may
        # interleave out of order); id-keyed slots keep arrival order.
        if all(isinstance(k, int) for k in calls):
            ordered = sorted(calls.items())
        else:
            ordered = list(calls.items())
        message["tool_calls"] = [
            {
                "id": slot["id"] or f"call_{i}",
                "type": "function",
                "function": {"name": slot["name"] or "", "arguments": "".join(slot["args"]) or "{}"},
            }
            for i, (_key, slot) in enumerate(ordered)
        ]
    return {"message": message, "finish_reason": finish_reason}


def parse_completion(body: bytes) -> Optional[Dict[str, Any]]:
    """The non-streaming counterpart: choices[0].message of a chat.completion object."""
    try:
        obj = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    choices = obj.get("choices") if isinstance(obj, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return None
    return {"message": message, "finish_reason": choices[0].get("finish_reason")}


def build_record(
    capture: Dict[str, Any],
    payload: Optional[Dict[str, Any]],
    status: int,
    latency_ms: int,
    truncated: bool = False,
) -> Dict[str, Any]:
    """One trace-v1 line (§3.3) — the exact shape ingest's trace-v1 adapter parses.
    Failed or truncated exchanges carry an EMPTY assistant message: they must parse
    (landing in ingest's failed_status bucket), never mine."""
    request = capture["body"]
    if payload is not None:
        response = {"message": payload["message"], "finish_reason": payload.get("finish_reason")}
    else:
        response = {"message": dict(_EMPTY_MESSAGE), "finish_reason": None}
    meta: Dict[str, Any] = {"status": status, "latency_ms": latency_ms}
    if truncated:
        meta["truncated"] = True
    return {
        "v": 1,
        "ts": capture["ts"],
        "session_id": capture["sid"],
        "model": str(request.get("model") or ""),
        "request": {
            "messages": request.get("messages") or [],
            "tools": request.get("tools") or [],
            "tool_choice": request.get("tool_choice"),
            "temperature": request.get("temperature"),
        },
        "response": response,
        "meta": meta,
    }


# --- the proxy itself ---------------------------------------------------------------


class ProxyServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, config: ProxyConfig, writer: TraceWriter, stitcher: SessionStitcher):
        upstream = urllib.parse.urlsplit(config.upstream)
        try:
            # urlsplit defers port validation to the .port property; touch it HERE so
            # a bad port fails startup with a clean message instead of killing every
            # request with an uncaught ValueError.
            upstream_port = upstream.port
        except ValueError:
            upstream_port = -1
        if (
            upstream.scheme not in ("http", "https")
            or not upstream.hostname
            or upstream_port == -1
        ):
            raise ValueError(
                f"--upstream must be an http(s) base URL, got '{config.upstream}'"
            )
        self.config = config
        self.writer = writer
        self.stitcher = stitcher
        self.upstream = upstream
        self.capture_failures = 0
        self.inflight = 0
        self._counter_lock = threading.Lock()
        super().__init__((config.listen_host, config.listen_port), _ProxyHandler)

    def count_capture_failure(self, why: str) -> None:
        with self._counter_lock:
            self.capture_failures += 1
        _warn(f"recording skipped ({why}) — the request itself was served")

    def track(self, delta: int) -> None:
        with self._counter_lock:
            self.inflight += delta


class _ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 60  # reading the CLIENT's request must not hang a handler thread forever
    server: ProxyServer

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet: one line per token
        return

    def do_GET(self) -> None:
        self._proxy()

    do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = do_GET

    # --- plumbing ----------------------------------------------------------------

    def _error_response(self, status: int, message: str) -> None:
        payload = json.dumps(
            {"error": {"message": f"probelock proxy: {message}", "type": "proxy_error"}}
        ).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.close_connection = True
            self.end_headers()
            self.wfile.write(payload)
        except OSError:
            pass

    def _begin_capture(self, body: bytes, ts: str) -> Optional[Dict[str, Any]]:
        if self.command != "POST":
            return None
        path = self.path.split("?", 1)[0].rstrip("/")
        if not path.endswith("/chat/completions"):
            return None
        try:
            parsed = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(parsed, dict) or not isinstance(parsed.get("messages"), list):
            return None
        sid, norm = self.server.stitcher.begin(parsed["messages"])
        return {"body": parsed, "sid": sid, "norm": norm, "ts": ts}

    def _finish_capture(
        self,
        capture: Dict[str, Any],
        status: int,
        payload: Optional[Dict[str, Any]],
        started: float,
        truncated: bool = False,
    ) -> None:
        try:
            latency_ms = int((time.monotonic() - started) * 1000)
            record = build_record(capture, payload, status, latency_ms, truncated)
            if payload is not None and status < 400:
                self.server.stitcher.complete(capture["sid"], capture["norm"], payload["message"])
            self.server.writer.put(record)
        except Exception as exc:  # prime directive: recording never breaks serving
            self.server.count_capture_failure(f"record build: {exc}")

    @staticmethod
    def _strip(headers, extra: set) -> Dict[str, str]:
        """Drop hop-by-hop headers plus anything the Connection header names."""
        named = {
            token.strip().lower()
            for value in headers.get_all("Connection") or []
            for token in value.split(",")
        }
        drop = extra | named
        return {k: v for k, v in headers.items() if k.lower() not in drop}

    # --- request lifecycle ----------------------------------------------------------

    def _proxy(self) -> None:
        self.server.track(+1)
        try:
            self._proxy_inner()
        finally:
            self.server.track(-1)

    def _proxy_inner(self) -> None:
        config = self.server.config
        started = time.monotonic()
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()

        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            # BaseHTTPRequestHandler doesn't de-chunk request bodies; OpenAI-style
            # clients always send Content-Length for JSON. Documented limitation.
            self._error_response(411, "chunked request bodies are not supported")
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        body = self.rfile.read(length) if length > 0 else b""

        capture = None
        try:
            capture = self._begin_capture(body, ts)
        except Exception as exc:
            self.server.count_capture_failure(f"capture setup: {exc}")

        upstream = self.server.upstream
        conn_cls = (
            http.client.HTTPSConnection if upstream.scheme == "https"
            else http.client.HTTPConnection
        )
        conn = conn_cls(upstream.hostname, upstream.port, timeout=config.connect_timeout)
        try:
            conn.connect()
            if conn.sock is not None:  # connect fast; read patiently
                conn.sock.settimeout(config.timeout)
            conn.request(
                self.command,
                upstream.path.rstrip("/") + self.path,
                body=body or None,
                headers=self._strip(self.headers, _STRIP_REQUEST),
            )
            response = conn.getresponse()
        except (OSError, http.client.HTTPException) as exc:
            # HTTPException covers BadStatusLine/LineTooLong — an upstream that isn't
            # speaking HTTP at all must still yield a clean 502 + failed record, not a
            # dead handler thread.
            conn.close()
            self._error_response(502, f"upstream unreachable: {exc}")
            if capture is not None:
                self._finish_capture(capture, 502, None, started)
            return

        content_length = response.getheader("Content-Length")
        # Strip hop-by-hop headers on this leg too, including anything the upstream's
        # own Connection header names — those are for OUR hop, not the client's.
        response_drop = _STRIP_RESPONSE | {
            token.strip().lower()
            for value in response.msg.get_all("Connection") or []
            for token in value.split(",")
        }
        try:
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() not in response_drop:
                    self.send_header(key, value)
            if content_length is not None:
                self.send_header("Content-Length", content_length)
            else:
                # http.client de-chunked the body; close-delimit rather than re-chunk.
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
        except OSError:
            # Headers may be half-sent: a kept-alive client would misparse whatever
            # follows, so the connection must die with this request — and the client
            # is gone, so the exchange is recorded as such rather than lost.
            self.close_connection = True
            conn.close()
            if capture is not None:
                self._finish_capture(capture, 499, None, started, truncated=True)
            return

        captured = bytearray()
        relayed = 0
        overflow = False
        client_gone = False
        upstream_died = False
        try:
            while True:
                try:
                    chunk = response.read1(_RELAY_CHUNK)
                except (OSError, http.client.HTTPException):
                    upstream_died = True
                    break
                if not chunk:
                    break
                relayed += len(chunk)
                if capture is not None and not overflow:
                    if len(captured) + len(chunk) > _MAX_CAPTURE:
                        overflow = True
                        captured.clear()
                    else:
                        captured.extend(chunk)
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except OSError:
                    client_gone = True
                    break
        finally:
            # On client disconnect this also tears down the upstream request instead
            # of draining it — no point burning the model slot on unread tokens.
            conn.close()
        if not (client_gone or upstream_died) and content_length is not None:
            try:
                promised = int(content_length)
            except ValueError:
                promised = relayed
            if relayed < promised:
                # read1 reports a premature upstream EOF as a clean end; the client
                # is still owed bytes under the relayed Content-Length and would hang
                # on keep-alive. Treat it as the upstream death it is.
                upstream_died = True
        if client_gone or upstream_died:
            self.close_connection = True

        if capture is None:
            return
        try:
            if client_gone:
                status = 499  # nginx convention: client closed request
            elif upstream_died:
                status = 502
            else:
                status = response.status
            payload = None
            truncated = client_gone or upstream_died
            if status < 400 and not truncated:
                if overflow:
                    self.server.count_capture_failure("response larger than the capture cap")
                    return
                data = bytes(captured)
                # The response's OWN content type decides the parser — servers are free
                # to ignore the request's stream flag.
                is_sse = "text/event-stream" in (response.getheader("Content-Type") or "").lower()
                payload = assemble_sse(data) if is_sse else parse_completion(data)
                if payload is None:
                    self.server.count_capture_failure("upstream response did not parse")
                    return
                if is_sse and payload.get("finish_reason") is None:
                    # Every healthy OpenAI-compatible stream ends with a finish_reason
                    # chunk; a clean EOF without one means the upstream died at an
                    # event boundary. Half a message must never look mineable.
                    status, payload, truncated = 502, None, True
            self._finish_capture(capture, status, payload, started, truncated=truncated)
        except Exception as exc:
            self.server.count_capture_failure(f"capture: {exc}")


def start_proxy(config: ProxyConfig) -> ProxyServer:
    """Start the writer thread, the stitcher, and the serving thread. Returns the
    server; callers own shutdown via stop_proxy()."""
    writer = TraceWriter(config.out, config.max_size_mb, config.max_age_min)
    stitcher = SessionStitcher(retry_ttl=config.retry_ttl)
    server = ProxyServer(config, writer, stitcher)
    writer.start()
    thread = threading.Thread(target=server.serve_forever, name="probelock-proxy", daemon=True)
    thread.start()
    server._serve_thread = thread
    return server


def stop_proxy(server: ProxyServer) -> None:
    server.shutdown()
    server.server_close()
    # Handler threads are daemons; give in-flight requests a moment to finish their
    # capture work so their records make the writer's final drain.
    deadline = time.monotonic() + 5.0
    while server.inflight and time.monotonic() < deadline:
        time.sleep(0.05)
    server.writer.close()
