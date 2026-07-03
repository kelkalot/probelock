"""Tests for the recording proxy: reassembly/stitching/writer units plus real
HTTP integration through a stub upstream (ephemeral ports, stdlib client)."""

import http.client
import json
import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from probelock.ingest import MiningConfig, load_exchanges, mine_exchanges, stitch_sessions
from probelock.proxy import (
    ProxyConfig,
    SessionStitcher,
    TraceWriter,
    assemble_sse,
    parse_completion,
    start_proxy,
    stop_proxy,
)

# --- unit: SSE reassembly ---------------------------------------------------------


def _sse(events):
    out = b""
    for e in events:
        payload = e if isinstance(e, str) else json.dumps(e)
        out += f"data: {payload}\n\n".encode()
    return out


def test_assemble_sse_concatenates_content_deltas():
    body = _sse([
        {"choices": [{"delta": {"role": "assistant", "content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        "[DONE]",
    ])
    got = assemble_sse(body)
    assert got == {"message": {"role": "assistant", "content": "Hello"},
                   "finish_reason": "stop"}


def test_assemble_sse_merges_parallel_tool_calls_by_index():
    body = _sse([
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "a", "function": {"name": "one", "arguments": '{"x"'}},
            {"index": 1, "id": "b", "function": {"name": "two", "arguments": '{"y"'}},
        ]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 1, "function": {"arguments": ": 2}"}},
            {"index": 0, "function": {"arguments": ": 1}"}},
        ]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ])
    calls = assemble_sse(body)["message"]["tool_calls"]
    assert [(c["function"]["name"], c["function"]["arguments"]) for c in calls] == [
        ("one", '{"x": 1}'), ("two", '{"y": 2}')]


def test_assemble_sse_tolerates_usage_chunk_and_garbage():
    body = _sse([
        {"choices": [{"delta": {"content": "ok"}}]},
        {"choices": [], "usage": {"total_tokens": 9}},  # stream_options include_usage
        "not json at all",
        "[DONE]",
    ])
    assert assemble_sse(body)["message"]["content"] == "ok"


def test_assemble_sse_returns_none_when_nothing_assembles():
    assert assemble_sse(b"") is None
    assert assemble_sse(_sse([{"choices": [], "usage": {}}, "[DONE]"])) is None


def test_parse_completion_rejects_malformed():
    assert parse_completion(b"not json") is None
    assert parse_completion(json.dumps({"choices": []}).encode()) is None
    good = {"choices": [{"message": {"role": "assistant", "content": "hi"},
                         "finish_reason": "stop"}]}
    assert parse_completion(json.dumps(good).encode())["message"]["content"] == "hi"


# --- unit: session stitching ---------------------------------------------------------


U1 = {"role": "user", "content": "find the report"}
A1 = {"role": "assistant", "content": None,
      "tool_calls": [{"id": "c1", "type": "function",
                      "function": {"name": "search", "arguments": '{"q": "report"}'}}]}
T1 = {"role": "tool", "name": "search", "content": "found it"}
U2 = {"role": "user", "content": "open it"}


def test_stitcher_groups_a_continuation_under_one_sid():
    st = SessionStitcher()
    sid1, norm1 = st.begin([U1])
    st.complete(sid1, norm1, A1)
    sid2, _ = st.begin([U1, A1, T1, U2])
    assert sid2 == sid1


def test_stitcher_gives_reruns_fresh_sids_but_retries_the_same_one():
    quick = SessionStitcher(retry_ttl=300.0)
    a, _ = quick.begin([U1])
    b, _ = quick.begin([U1])  # identical request right away = HTTP retry
    assert b == a

    strict = SessionStitcher(retry_ttl=0.0)
    a, norm = strict.begin([U1])
    strict.complete(a, norm, A1)
    b, _ = strict.begin([U1])  # past the TTL = a rerun: NEW agreement evidence
    assert b != a


def test_stitcher_keeps_unrelated_conversations_apart_and_bounds_its_maps():
    st = SessionStitcher(max_entries=2)
    sids = set()
    for i in range(5):
        sid, norm = st.begin([{"role": "user", "content": f"question {i}"}])
        st.complete(sid, norm, {"role": "assistant", "content": f"answer {i}"})
        sids.add(sid)
    assert len(sids) == 5
    assert len(st._prefixes) <= 2  # bounded, oldest evicted, no crash


# --- unit: writer & rotation ----------------------------------------------------------


def test_writer_rotates_by_size_without_clobbering(tmp_path):
    out = tmp_path / "t.jsonl"
    w = TraceWriter(out, max_size_mb=0.0001)  # ~100 bytes
    w.start()
    for i in range(4):
        w.put({"v": 1, "i": i, "pad": "x" * 120})
        time.sleep(0.05)
    w.close()
    segments = sorted(tmp_path.glob("t-*.jsonl"))
    assert segments, "expected at least one rotated segment"
    total = sum(len(p.read_text().splitlines()) for p in [out, *segments] if p.exists())
    assert total == 4  # every record survives rotation, none clobbered
    assert w.written == 4 and w.dropped == 0


# --- integration helpers ----------------------------------------------------------------


class _Upstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    gate = None          # test-controlled event for incremental streaming
    saw_headers = []     # captured request headers, for the accept-encoding test

    def log_message(self, *args):
        return

    def do_GET(self):
        payload = json.dumps({"data": ["stub-model"]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _stream(self, events, tail=True):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        for e in events:
            self.wfile.write(f"data: {json.dumps(e)}\n\n".encode())
            self.wfile.flush()
        if tail:
            self.wfile.write(b"data: [DONE]\n\n")
        self.close_connection = True

    def do_POST(self):
        type(self).saw_headers.append(dict(self.headers))
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        marker = str(body.get("messages", [{}])[-1].get("content") or "")

        if "MODE:500" in marker:
            payload = json.dumps({"error": {"message": "boom"}}).encode()
            self.send_response(500)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if "MODE:no-finish" in marker:
            # dies at an event boundary: clean EOF, but no finish_reason chunk ever
            self._stream([{"choices": [{"delta": {"content": "half a mess"}}]}], tail=False)
            return
        if "MODE:gated-stream" in marker:
            self._stream([{"choices": [{"delta": {"content": "first"}}]}], tail=False)
            type(self).gate.wait(timeout=10)
            self._continue_stream()
            return
        if "MODE:slow-drip" in marker:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for i in range(20):
                try:
                    self.wfile.write(
                        f"data: {json.dumps({'choices': [{'delta': {'content': 'x' * 512}}]})}\n\n".encode())
                    self.wfile.flush()
                except OSError:
                    return
                time.sleep(0.05)
            self.close_connection = True
            return
        if "MODE:text" in marker:
            payload = json.dumps({
                "model": "stub-model",
                "choices": [{"message": {"role": "assistant", "content": "All done."},
                             "finish_reason": "stop"}],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if body.get("stream"):
            self._stream([
                {"choices": [{"delta": {"role": "assistant", "tool_calls": [
                    {"index": 0, "id": "c1",
                     "function": {"name": "get_weather", "arguments": '{"ci'}}]}}]},
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "function": {"arguments": 'ty": "Oslo"}'}}]}}]},
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
                {"choices": [], "usage": {"total_tokens": 42}},
            ])
            return

        message = {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c9", "type": "function",
             "function": {"name": "search_files", "arguments": '{"query": "report"}'}}]}
        if "MODE:text" in marker or any(m.get("role") == "tool" for m in body.get("messages", [])):
            message = {"role": "assistant", "content": "All done."}
        payload = json.dumps({
            "model": "stub-model",
            "choices": [{"message": message,
                         "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _continue_stream(self):
        try:
            self.wfile.write(
                f"data: {json.dumps({'choices': [{'delta': {'content': ' second'}, 'finish_reason': 'stop'}]})}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except OSError:
            pass


def _run_pair(tmp_path, **cfg):
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    out = tmp_path / "trace.jsonl"
    server = start_proxy(ProxyConfig(
        upstream=f"http://127.0.0.1:{upstream.server_address[1]}", out=out,
        listen_port=0, **cfg))
    return upstream, server, out


def _teardown(upstream, server):
    stop_proxy(server)
    upstream.shutdown()
    upstream.server_close()


def _post(port, payload, timeout=10):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request("POST", "/v1/chat/completions", body=json.dumps(payload),
                 headers={"Content-Type": "application/json",
                          "Accept-Encoding": "gzip, deflate"})
    response = conn.getresponse()
    data = response.read()
    conn.close()
    return response.status, data


def _records(out):
    return [json.loads(line) for line in out.read_text().splitlines()]


# --- integration tests --------------------------------------------------------------------


def test_proxy_passthrough_and_capture(tmp_path):
    upstream, server, out = _run_pair(tmp_path)
    try:
        port = server.server_address[1]
        status, body = _post(port, {"model": "m",
                                    "messages": [{"role": "user", "content": "find it"}]})
        assert status == 200
        assert json.loads(body)["model"] == "stub-model"  # byte-level pass-through
        # non-chat paths forwarded but never recorded
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request("GET", "/v1/models")
        assert conn.getresponse().status == 200
        conn.close()
    finally:
        _teardown(upstream, server)
    records = _records(out)
    assert len(records) == 1
    rec = records[0]
    assert rec["v"] == 1 and rec["meta"]["status"] == 200
    assert rec["request"]["messages"][0]["content"] == "find it"
    assert rec["response"]["message"]["tool_calls"][0]["function"]["name"] == "search_files"
    assert oct(out.stat().st_mode & 0o777) == "0o600"  # verbatim content: private by default


def test_proxy_strips_accept_encoding_upstream(tmp_path):
    # A gzip-capable upstream would make every capture unparseable; with the header
    # stripped, http.client itself asks for identity.
    _Upstream.saw_headers = []
    upstream, server, out = _run_pair(tmp_path)
    try:
        _post(server.server_address[1],
              {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    finally:
        _teardown(upstream, server)
    assert _Upstream.saw_headers[-1].get("Accept-Encoding") == "identity"
    assert _records(out)  # and the capture parsed


def test_proxy_streams_incrementally_not_in_buffered_bursts(tmp_path):
    # Regression for the read()/read1() blocker: the client must see the first SSE
    # event BEFORE the upstream emits the second. The upstream holds event 2 behind a
    # gate that only this test opens — a buffering proxy would deadlock into timeouts.
    _Upstream.gate = threading.Event()
    upstream, server, out = _run_pair(tmp_path)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        conn.request("POST", "/v1/chat/completions", body=json.dumps({
            "model": "m", "stream": True,
            "messages": [{"role": "user", "content": "MODE:gated-stream"}]}),
            headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        first = b""
        while b"first" not in first:
            chunk = response.read1(65536)
            assert chunk, "stream ended before the first event arrived"
            first += chunk
        _Upstream.gate.set()  # only now may the upstream continue
        rest = first
        while b"[DONE]" not in rest:
            chunk = response.read1(65536)
            if not chunk:
                break
            rest += chunk
        conn.close()
        assert b" second" in rest
    finally:
        _teardown(upstream, server)
        _Upstream.gate = None
    rec = _records(out)[0]
    assert rec["response"]["message"]["content"] == "first second"
    assert rec["response"]["finish_reason"] == "stop"


def test_proxy_streaming_reassembles_tool_calls(tmp_path):
    upstream, server, out = _run_pair(tmp_path)
    try:
        status, body = _post(server.server_address[1], {
            "model": "m", "stream": True,
            "messages": [{"role": "user", "content": "weather in Oslo?"}]})
        assert status == 200
        assert body.count(b"data:") == 5  # all events relayed, usage chunk included
    finally:
        _teardown(upstream, server)
    call = _records(out)[0]["response"]["message"]["tool_calls"][0]["function"]
    assert call == {"name": "get_weather", "arguments": '{"city": "Oslo"}'}


def test_upstream_500_is_relayed_and_logged_as_failed(tmp_path):
    upstream, server, out = _run_pair(tmp_path)
    try:
        status, _ = _post(server.server_address[1], {
            "model": "m", "messages": [{"role": "user", "content": "MODE:500"}]})
        assert status == 500
    finally:
        _teardown(upstream, server)
    rec = _records(out)[0]
    assert rec["meta"]["status"] == 500
    assert rec["response"]["message"] == {"role": "assistant", "content": None,
                                          "tool_calls": []}
    exchanges, summary = load_exchanges(out)
    assert exchanges == [] and summary.skipped == {"failed_status": 1}  # never mined


def test_unreachable_upstream_yields_502_and_failed_record(tmp_path):
    out = tmp_path / "trace.jsonl"
    # a port from a just-closed listener: nothing is listening there
    probe_sock = socket.socket()
    probe_sock.bind(("127.0.0.1", 0))
    dead_port = probe_sock.getsockname()[1]
    probe_sock.close()
    server = start_proxy(ProxyConfig(
        upstream=f"http://127.0.0.1:{dead_port}", out=out, listen_port=0,
        connect_timeout=2.0))
    try:
        status, body = _post(server.server_address[1], {
            "model": "m", "messages": [{"role": "user", "content": "hi"}]})
        assert status == 502
        assert json.loads(body)["error"]["type"] == "proxy_error"
    finally:
        stop_proxy(server)
    assert _records(out)[0]["meta"]["status"] == 502


def test_client_disconnect_mid_stream_records_499_not_a_good_exchange(tmp_path):
    upstream, server, out = _run_pair(tmp_path)
    try:
        sock = socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=10)
        # RST on close, so the proxy's next write fails immediately (deterministic)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        payload = json.dumps({"model": "m", "stream": True,
                              "messages": [{"role": "user", "content": "MODE:slow-drip"}]})
        sock.sendall(
            f"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n\r\n"
            f"{payload}".encode())
        sock.recv(1024)  # got the status line + maybe a first chunk: now hang up
        sock.close()
        deadline = time.time() + 10
        while time.time() < deadline and not (
            out.exists() and out.read_text().strip()
        ):
            time.sleep(0.1)
    finally:
        _teardown(upstream, server)
    rec = _records(out)[0]
    assert rec["meta"]["status"] == 499
    assert rec["meta"]["truncated"] is True
    assert rec["response"]["message"]["content"] is None  # partial output never recorded
    exchanges, summary = load_exchanges(out)
    assert exchanges == [] and summary.skipped == {"failed_status": 1}


def test_round_trip_proxy_to_continuation_confirmed_probe(tmp_path):
    """The acceptance gate: a 2-turn tool conversation recorded by the proxy must
    stitch to ONE session and confirm the first call via the continuation rule when
    mined — proxy output is first-class miner input, end to end."""
    tools = [{"type": "function", "function": {
        "name": "search_files", "description": "Search the workspace",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}}]
    upstream, server, out = _run_pair(tmp_path)
    try:
        port = server.server_address[1]
        first = [{"role": "user", "content": "find the report"}]
        status, body = _post(port, {"model": "m", "messages": first, "tools": tools})
        assert status == 200
        assistant = json.loads(body)["choices"][0]["message"]
        second = first + [assistant,
                          {"role": "tool", "name": "search_files", "content": "found 3 files"},
                          {"role": "user", "content": "great, open the first one"}]
        status, _ = _post(port, {"model": "m", "messages": second, "tools": tools})
        assert status == 200
    finally:
        _teardown(upstream, server)

    exchanges, _ = load_exchanges(out)
    assert len(exchanges) == 2
    sessions = stitch_sessions(exchanges)
    assert len({e.session_id for s in sessions for e in s}) == 1  # stitched, no agent help

    exchanges, summary = load_exchanges(out)
    probes, _ = mine_exchanges(exchanges, MiningConfig())
    selection = [p for p in probes if p.category == "tool_selection"]
    assert len(selection) == 1
    assert selection[0].tool == "search_files"
    assert selection[0].provenance["rule"] == "continuation"


def test_concurrent_requests_all_recorded_with_distinct_sessions(tmp_path):
    upstream, server, out = _run_pair(tmp_path)
    try:
        port = server.server_address[1]
        results = []

        def one(i):
            results.append(_post(port, {
                "model": "m",
                "messages": [{"role": "user", "content": f"question number {i}"}]})[0])

        threads = [threading.Thread(target=one, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert results == [200, 200, 200, 200]
    finally:
        _teardown(upstream, server)
    records = _records(out)
    assert len(records) == 4
    assert len({r["session_id"] for r in records}) == 4


def test_stream_ending_without_finish_reason_is_recorded_as_truncated(tmp_path):
    # A clean EOF with no finish_reason chunk means the upstream died at an event
    # boundary — half a message must land in the failed bucket, never mine.
    upstream, server, out = _run_pair(tmp_path)
    try:
        status, body = _post(server.server_address[1], {
            "model": "m", "stream": True,
            "messages": [{"role": "user", "content": "MODE:no-finish"}]})
        assert status == 200 and b"half a mess" in body  # the relay itself is untouched
    finally:
        _teardown(upstream, server)
    rec = _records(out)[0]
    assert rec["meta"]["status"] == 502
    assert rec["meta"]["truncated"] is True
    assert rec["response"]["message"]["content"] is None
    exchanges, summary = load_exchanges(out)
    assert exchanges == [] and summary.skipped == {"failed_status": 1}


def test_writer_rotates_by_age_even_while_idle(tmp_path):
    out = tmp_path / "t.jsonl"
    w = TraceWriter(out, max_size_mb=0, max_age_min=0.5 / 60)  # 0.5 s
    w.start()
    w.put({"v": 1, "epoch": "old"})
    time.sleep(2.5)  # > age + at least one idle 1 s tick: rotation must fire unprompted
    w.put({"v": 1, "epoch": "new"})
    time.sleep(0.2)
    w.close()
    segments = sorted(tmp_path.glob("t-*.jsonl"))
    assert segments, "idle age rotation never fired"
    assert "old" in segments[0].read_text()
    assert out.exists() and "new" in out.read_text()  # fresh epoch in a fresh file


# --- round-2 review regressions --------------------------------------------------------


def test_assemble_sse_only_follows_choice_zero():
    # n>1 streams interleave per-choice chunks; merging them would weave two sampled
    # answers into one garbage message recorded as a good exchange.
    body = _sse([
        {"choices": [{"index": 0, "delta": {"content": "keep"}}]},
        {"choices": [{"index": 1, "delta": {"content": "DISCARD"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ])
    assert assemble_sse(body)["message"]["content"] == "keep"


def test_assemble_sse_keys_indexless_parallel_calls_by_id():
    # some servers omit "index": two parallel calls must not merge into one slot
    body = _sse([
        {"choices": [{"delta": {"tool_calls": [
            {"id": "a", "function": {"name": "one", "arguments": '{"x"'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"id": "b", "function": {"name": "two", "arguments": '{"y": 2}'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"id": "a", "function": {"arguments": ': 1}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ])
    calls = assemble_sse(body)["message"]["tool_calls"]
    assert [(c["function"]["name"], c["function"]["arguments"]) for c in calls] == [
        ("one", '{"x": 1}'), ("two", '{"y": 2}')]


def test_stitcher_retry_window_does_not_slide_and_keys_on_raw_bytes():
    st = SessionStitcher(retry_ttl=300.0)
    a, _ = st.begin([U1])
    # identical periodic requests: the window is anchored at first sight, so a
    # cron-style agent cannot collapse into one session forever
    st._recent[next(iter(st._recent))] = (a, time.monotonic() - 301.0)  # age the entry
    b, _ = st.begin([U1])
    assert b != a
    # normalization must NOT bridge retry detection: same normalized content with a
    # different timestamp is a rerun (new evidence), not a transport retry
    t1 = {"role": "user", "content": "It is 2026-07-03T08:00:00Z. weather in Oslo?"}
    t2 = {"role": "user", "content": "It is 2026-07-03T09:30:00Z. weather in Oslo?"}
    c, _ = st.begin([t1])
    d, _ = st.begin([t2])
    assert c != d


def test_writer_survives_unwritable_path_and_lone_surrogates(tmp_path):
    target = tmp_path / "t.jsonl"
    target.mkdir()  # the out path IS a directory: every open in the thread fails
    w = TraceWriter(target)
    try:
        w.start()
        w.put({"v": 1})
        time.sleep(1.5)  # past the eager open and at least one queue tick
        assert w.dropped == 1  # dropped and counted, no exception escaped
        assert w._thread.is_alive()
    finally:
        w.close()

    ok = TraceWriter(tmp_path / "ok.jsonl")
    ok.start()
    ok.put({"v": 1, "content": "\ud800 lone surrogate from someone's log"})
    ok.put({"v": 1, "after": "still alive"})
    ok.close()
    lines = (tmp_path / "ok.jsonl").read_text().splitlines()
    assert len(lines) == 2  # surrogate record re-escaped, not dropped; writer survived
    assert json.loads(lines[0])["content"].startswith("\ud800")


def test_bad_upstream_port_fails_at_startup_not_per_request(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        start_proxy(ProxyConfig(upstream="http://127.0.0.1:99999",
                                out=tmp_path / "t.jsonl", listen_port=0))


def test_non_http_upstream_yields_502_and_failed_record(tmp_path):
    # an upstream speaking non-HTTP bytes (BadStatusLine is HTTPException, not OSError)
    def smtp_ish(server_sock):
        conn, _ = server_sock.accept()
        conn.recv(1024)
        conn.sendall(b"220 smtp.example.com ESMTP ready\r\n")
        conn.close()

    lst = socket.socket()
    lst.bind(("127.0.0.1", 0))
    lst.listen(1)
    threading.Thread(target=smtp_ish, args=(lst,), daemon=True).start()
    out = tmp_path / "trace.jsonl"
    server = start_proxy(ProxyConfig(upstream=f"http://127.0.0.1:{lst.getsockname()[1]}",
                                     out=out, listen_port=0))
    try:
        status, body = _post(server.server_address[1], {
            "model": "m", "messages": [{"role": "user", "content": "hi"}]})
        assert status == 502
        assert json.loads(body)["error"]["type"] == "proxy_error"
    finally:
        stop_proxy(server)
        lst.close()
    assert _records(out)[0]["meta"]["status"] == 502


def test_upstream_dying_mid_body_records_502_truncated(tmp_path):
    # Content-Length promised, connection closed short: read1 reports it as a clean
    # EOF, but the client is owed bytes — the proxy must close the connection and the
    # record must land in the failed bucket, never mine.
    class ShortBody(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            return

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "1000")
            self.end_headers()
            self.wfile.write(b'{"partial": ')  # 12 of 1000 promised bytes
            self.wfile.flush()
            self.connection.close()

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), ShortBody)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    out = tmp_path / "trace.jsonl"
    server = start_proxy(ProxyConfig(upstream=f"http://127.0.0.1:{upstream.server_address[1]}",
                                     out=out, listen_port=0, timeout=5))
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
        conn.request("POST", "/v1/chat/completions",
                     body=json.dumps({"model": "m",
                                      "messages": [{"role": "user", "content": "hi"}]}),
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        got = b""
        try:
            while True:
                chunk = response.read1(65536)
                if not chunk:
                    break
                got += chunk
        except (http.client.HTTPException, OSError):
            pass  # a truncated body on the client leg is expected here
        conn.close()
        assert b'{"partial": ' in got  # what arrived was relayed, nothing invented
    finally:
        stop_proxy(server)
        upstream.shutdown()
        upstream.server_close()
    rec = _records(out)[0]
    assert rec["meta"]["status"] == 502
    assert rec["meta"]["truncated"] is True
    exchanges, summary = load_exchanges(out)
    assert exchanges == [] and summary.skipped == {"failed_status": 1}


def test_parser_follows_response_content_type_not_request_stream_flag(tmp_path):
    # stream:true requested, but the upstream ignores it and answers plain JSON: the
    # capture must parse it as a completion object (Content-Type decides).
    upstream, server, out = _run_pair(tmp_path)
    try:
        status, _ = _post(server.server_address[1], {
            "model": "m", "stream": True,
            "messages": [{"role": "user", "content": "MODE:text please"}]})
        assert status == 200
    finally:
        _teardown(upstream, server)
    rec = _records(out)[0]
    assert rec["meta"]["status"] == 200
    assert rec["response"]["message"]["content"] == "All done."
