"""Tests for the any-llm / litellm SDK clients, with the completion fn mocked
(no real provider install needed). Both return OpenAI ChatCompletion objects."""

import importlib.util
import json
from types import SimpleNamespace as NS

import pytest

from probelock.clients import (
    AnyLlmClient,
    ClientError,
    LiteLlmClient,
    ProbeError,
    _openai_object_to_response,
)
from probelock.models import Probe


def _probe(tools=None):
    return Probe(id="p", capability="tool_selection", description="",
                 messages=[{"role": "user", "content": "hi"}], tools=tools or [], expected_tool="t")


def _fake_resp(content=None, tool_calls=None):
    tcs = [NS(function=NS(name=n, arguments=a)) for n, a in (tool_calls or [])]
    return NS(choices=[NS(message=NS(content=content, tool_calls=tcs))])


def _completion_factory(counter):
    def fake(**kwargs):
        counter["n"] += 1
        counter["last"] = kwargs
        return _fake_resp(content="ok")
    return fake


def test_openai_object_mapping():
    r = _openai_object_to_response(_fake_resp(tool_calls=[("t", '{"x": 1}')]))
    assert r.content is None
    assert r.tool_calls[0].name == "t" and json.loads(r.tool_calls[0].arguments) == {"x": 1}


def test_openai_object_strips_think_and_normalizes_dict_args():
    r = _openai_object_to_response(_fake_resp(content="<think>plan</think>\nhi",
                                              tool_calls=[("t", {"x": 1})]))
    assert r.content == "hi"
    assert isinstance(r.tool_calls[0].arguments, str)
    assert json.loads(r.tool_calls[0].arguments) == {"x": 1}


def test_openai_object_bad_shape_raises_probe_error():
    with pytest.raises(ProbeError):
        _openai_object_to_response(NS(nope=True))


def test_openai_object_reasoning_field_fallback():
    # Parity with HttpClient: fall back to a separate reasoning field when content
    # is empty (some reasoning models split it out).
    r1 = _openai_object_to_response(NS(choices=[NS(message=NS(content="", reasoning="the answer", tool_calls=[]))]))
    assert r1.content == "the answer"
    r2 = _openai_object_to_response(NS(choices=[NS(message=NS(content=None, reasoning_content="rc", tool_calls=[]))]))
    assert r2.content == "rc"


def test_sdk_client_caches_at_temp0(monkeypatch):
    counter = {"n": 0}
    monkeypatch.setattr(AnyLlmClient, "_import_completion",
                        staticmethod(lambda: _completion_factory(counter)))
    c = AnyLlmClient(model="ollama/llama3", temperature=0.0)
    assert c.produces_variance is False
    c.complete(_probe())
    c.complete(_probe())
    assert counter["n"] == 1  # identical request cached at temp 0
    assert c.metadata["runtime"] == "any-llm"


def test_sdk_client_no_cache_at_temp_nonzero(monkeypatch):
    counter = {"n": 0}
    monkeypatch.setattr(LiteLlmClient, "_import_completion",
                        staticmethod(lambda: _completion_factory(counter)))
    c = LiteLlmClient(model="anthropic/claude", temperature=0.7)
    assert c.produces_variance is True
    c.complete(_probe())
    c.complete(_probe())
    assert counter["n"] == 2
    assert c.metadata["runtime"] == "litellm"


def test_sdk_client_forwards_tools_and_api_key(monkeypatch):
    counter = {"n": 0}
    monkeypatch.setattr(AnyLlmClient, "_import_completion",
                        staticmethod(lambda: _completion_factory(counter)))
    c = AnyLlmClient(model="m", api_key="sk-x", temperature=0.0)
    tools = [{"type": "function", "function": {"name": "t", "parameters": {}}}]
    c.complete(_probe(tools=tools))
    assert counter["last"]["model"] == "m"
    assert counter["last"]["tools"] == tools
    assert counter["last"]["tool_choice"] == "auto"
    assert counter["last"]["api_key"] == "sk-x"


def test_sdk_client_exception_becomes_recoverable_probe_error(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(AnyLlmClient, "_import_completion", staticmethod(lambda: boom))
    with pytest.raises(ProbeError):
        AnyLlmClient(model="m", temperature=0.0).complete(_probe())


def test_sdk_client_redacts_api_key_from_exception_message(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("connection failed with api_key=sk-secret-123 in headers")

    monkeypatch.setattr(AnyLlmClient, "_import_completion", staticmethod(lambda: boom))
    client = AnyLlmClient(model="m", api_key="sk-secret-123", temperature=0.0)
    with pytest.raises(ProbeError) as excinfo:
        client.complete(_probe())
    assert "sk-secret-123" not in str(excinfo.value)
    assert "[REDACTED]" in str(excinfo.value)


def test_anyllm_missing_package_is_clean_error():
    if importlib.util.find_spec("any_llm") is not None:
        pytest.skip("any_llm installed")
    with pytest.raises(ClientError):
        AnyLlmClient(model="anthropic/claude")


def test_litellm_missing_package_is_clean_error():
    if importlib.util.find_spec("litellm") is not None:
        pytest.skip("litellm installed")
    with pytest.raises(ClientError):
        LiteLlmClient(model="anthropic/claude")
