"""Real-socket tests for HttpClient against a localhost server (no external deps).

These exercise the actual HTTP path — request building and OpenAI-response
parsing — without needing a model. They run in CI.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from probelock.clients import ClientError, HttpClient, ProbeError
from probelock.models import Probe


def _make_server(handler_fn):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence test server logging
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            status, payload = handler_fn(body)
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture
def serve():
    servers = []

    def _make(handler_fn, temperature=0.0):
        srv = _make_server(handler_fn)
        servers.append(srv)
        host, port = srv.server_address
        return HttpClient(base_url=f"http://{host}:{port}/v1", model="m", temperature=temperature)

    yield _make
    for srv in servers:
        srv.shutdown()


def _probe(tools=None):
    return Probe(
        id="p",
        capability="tool_selection",
        description="",
        messages=[{"role": "user", "content": "hi"}],
        tools=tools or [],
        expected_tool="t",
    )


def test_parses_tool_calls(serve):
    def handler(_body):
        return 200, {
            "choices": [
                {"message": {"content": None, "tool_calls": [
                    {"function": {"name": "t", "arguments": '{"x": 1}'}}
                ]}}
            ]
        }

    resp = serve(handler).complete(_probe())
    assert resp.content is None
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "t"
    assert json.loads(resp.tool_calls[0].arguments) == {"x": 1}


def test_parses_plain_content(serve):
    resp = serve(lambda _b: (200, {"choices": [{"message": {"content": "hello"}}]})).complete(_probe())
    assert resp.content == "hello"
    assert resp.tool_calls == []


def test_dict_arguments_are_normalized_to_json_string(serve):
    def handler(_body):
        return 200, {"choices": [{"message": {"tool_calls": [
            {"function": {"name": "t", "arguments": {"x": 1}}}  # dict, not a string
        ]}}]}

    resp = serve(handler).complete(_probe())
    assert isinstance(resp.tool_calls[0].arguments, str)
    assert json.loads(resp.tool_calls[0].arguments) == {"x": 1}


def test_request_shape_includes_tools_and_temperature(serve):
    seen = {}

    def handler(body):
        seen.update(body)
        return 200, {"choices": [{"message": {"content": "ok"}}]}

    tools = [{"type": "function", "function": {"name": "t", "parameters": {}}}]
    serve(handler).complete(_probe(tools=tools))
    assert seen["model"] == "m"
    assert seen["temperature"] == 0
    assert seen["tool_choice"] == "auto"
    assert seen["tools"][0]["function"]["name"] == "t"


def test_404_is_fatal_not_recoverable(serve):
    # A 404 (model not pulled / wrong URL) must be fatal, not a per-probe 0,
    # so it can't poison an all-zeros baseline lockfile.
    client = serve(lambda _b: (404, {"error": "model not found"}))
    with pytest.raises(ClientError) as excinfo:
        client.complete(_probe())
    assert not isinstance(excinfo.value, ProbeError)


def test_400_is_recoverable_probe_error(serve):
    # 400 "model does not support tools" is a real capability failure -> recoverable.
    with pytest.raises(ProbeError):
        serve(lambda _b: (400, {"error": "does not support tools"})).complete(_probe())


def test_strips_leading_think_block(serve):
    def handler(_b):
        return 200, {"choices": [{"message": {"content": "<think>plan</think>\nACK"}}]}

    assert serve(handler).complete(_probe()).content == "ACK"


def test_falls_back_to_reasoning_field_when_content_empty(serve):
    def handler(_b):
        return 200, {"choices": [{"message": {"content": "", "reasoning": "ANSWER"}}]}

    assert serve(handler).complete(_probe()).content == "ANSWER"


def test_bad_response_shape_raises_probe_error(serve):
    with pytest.raises(ProbeError):
        serve(lambda _b: (200, {"unexpected": True})).complete(_probe())


def _counting_handler(counter):
    def handler(_body):
        counter["n"] += 1
        return 200, {"choices": [{"message": {"content": "ok"}}]}
    return handler


def test_temperature_0_caches_identical_requests(serve):
    counter = {"n": 0}
    client = serve(_counting_handler(counter))  # temperature 0
    probe = _probe()
    client.complete(probe)
    client.complete(probe)
    client.complete(probe)
    assert counter["n"] == 1  # 3 identical requests -> 1 network call


def test_nonzero_temperature_does_not_cache(serve):
    counter = {"n": 0}
    client = serve(_counting_handler(counter), temperature=0.7)
    probe = _probe()
    client.complete(probe)
    client.complete(probe)
    assert counter["n"] == 2  # independent samples, no caching


def test_produces_variance_tracks_temperature():
    assert HttpClient(model="m", temperature=0.0).produces_variance is False
    assert HttpClient(model="m", temperature=0.7).produces_variance is True


def test_probe_error_is_not_cached(serve):
    # A transient failure must not be cached, or it would poison the other probes
    # that share the request at temperature 0.
    state = {"n": 0}

    def handler(_body):
        state["n"] += 1
        if state["n"] == 1:
            return 500, {"error": "transient"}
        return 200, {"choices": [{"message": {"content": "ok"}}]}

    client = serve(handler)  # temperature 0
    probe = _probe()
    with pytest.raises(ProbeError):
        client.complete(probe)
    assert client.complete(probe).content == "ok"  # retried, not a cached error


def test_unreachable_server_raises_client_error_not_probe_error():
    # A closed port is a fatal ClientError (config issue) -> never scored as a per-probe
    # 0 that would silently continue the run.
    client = HttpClient(base_url="http://127.0.0.1:1/v1", model="m", timeout=3)
    with pytest.raises(ClientError) as excinfo:
        client.complete(_probe())
    assert not isinstance(excinfo.value, ProbeError)


def test_read_timeout_raises_probe_error_not_a_crash(serve):
    def handler(_b):
        time.sleep(0.3)
        return 200, {"choices": [{"message": {"content": "slow"}}]}

    client = serve(handler)
    client.timeout = 0.05
    with pytest.raises(ProbeError):
        client.complete(_probe())


def test_non_json_response_raises_probe_error_not_a_crash():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            body = b"not json at all"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    host, port = srv.server_address
    client = HttpClient(base_url=f"http://{host}:{port}/v1", model="m")
    try:
        with pytest.raises(ProbeError):
            client.complete(_probe())
    finally:
        srv.shutdown()


def test_truncated_response_raises_probe_error_not_a_crash():
    # A Content-Length that overpromises what's actually written (e.g. the model
    # server crashes or is OOM-killed mid-generation) raises http.client.IncompleteRead,
    # which must be classified as a recoverable per-probe failure, not left uncaught.
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            body = b'{"choices": [{"message": {"content": "ok"}}]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body) + 50))  # lie about length
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    host, port = srv.server_address
    client = HttpClient(base_url=f"http://{host}:{port}/v1", model="m")
    try:
        with pytest.raises(ProbeError):
            client.complete(_probe())
    finally:
        srv.shutdown()


def test_api_key_is_redacted_from_error_detail(serve):
    def handler(_b):
        return 400, {"error": "bad request, Authorization: Bearer super-secret-key"}

    client = serve(handler)
    client.api_key = "super-secret-key"
    with pytest.raises(ProbeError) as excinfo:
        client.complete(_probe())
    assert "super-secret-key" not in str(excinfo.value)
    assert "[REDACTED]" in str(excinfo.value)


def test_deterministic_4xx_error_is_cached(serve):
    # A 400 recurs identically, so it's cached to de-dup the shared-request group.
    state = {"n": 0}

    def handler(_body):
        state["n"] += 1
        return 400, {"error": "model does not support tools"}

    client = serve(handler)  # temperature 0
    probe = _probe()
    for _ in range(3):
        with pytest.raises(ProbeError):
            client.complete(probe)
    assert state["n"] == 1  # deterministic 4xx cached after the first call
