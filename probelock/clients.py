"""Clients — the only model-touching part of probelock.

  * HttpClient — talks to any OpenAI-compatible local endpoint (Ollama,
    llama.cpp, LM Studio, vLLM, an MLX server). Uses temperature 0 so a probe run
    is as reproducible as the server allows.
  * AnyLlmClient / LiteLlmClient — route to any provider (Anthropic, Gemini,
    Mistral, …) through a unified SDK that returns OpenAI ChatCompletion objects.
    A transport layer, not an LLM judge: still fully deterministic.
  * SimulatedClient — a deterministic stand-in driven by a quality profile, so
    the whole pipeline (derive -> probe -> lock -> diff -> gate) runs and is
    tested with no network and no model. It crafts genuinely correct/incorrect
    responses that the real scorers then grade — the scoring path is exercised
    for real, not mocked.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List

from .models import Probe, ResponseMessage, ToolCall

# A leading <think>...</think> reasoning block, which some servers (llama.cpp,
# LM Studio, vLLM, MLX) inline into message content for reasoning models.
_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _clean_content(msg: Dict[str, Any]):
    """Message text with a leading think-block stripped, falling back to a
    separate reasoning field when content is empty."""
    content = msg.get("content")
    if not content:
        content = msg.get("reasoning") or msg.get("reasoning_content")
    if isinstance(content, str):
        content = _THINK_BLOCK.sub("", content)
    return content


class ClientError(RuntimeError):
    """A fatal, user-facing error talking to a model endpoint (e.g. unreachable)."""


class ProbeError(ClientError):
    """A single probe failed at the API level (e.g. the model rejects tools).

    Recoverable: the runner records it as a 0 score for that probe and continues,
    so "this model can't tool-call at all" is measured rather than crashing.

    ``transient`` distinguishes a one-off failure (timeout, socket error, 5xx) from
    a deterministic one (a 4xx the request will always get). Only deterministic
    errors are cached, so a transient blip never poisons the shared-request group.
    """

    def __init__(self, message: str, transient: bool = True):
        super().__init__(message)
        self.transient = transient


def _openai_object_to_response(resp: Any) -> ResponseMessage:
    """Map an OpenAI ChatCompletion object (as any-llm / litellm return it) to a
    ResponseMessage. Attribute access, since these are SDK objects, not dicts."""
    try:
        message = resp.choices[0].message
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise ProbeError(f"unexpected response shape: {str(resp)[:200]}") from exc
    content = getattr(message, "content", None)
    if not content:  # parity with _clean_content: some reasoning models split this out
        content = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
    if isinstance(content, str):
        content = _THINK_BLOCK.sub("", content)
    calls: List[ToolCall] = []
    for tc in getattr(message, "tool_calls", None) or []:
        fn = getattr(tc, "function", None)
        if fn is None:
            continue
        args = getattr(fn, "arguments", "{}")
        if not isinstance(args, str):
            args = json.dumps(args)
        calls.append(ToolCall(getattr(fn, "name", "") or "", args))
    return ResponseMessage(content=content, tool_calls=calls)


class _EndpointClient:
    """Shared caching + error semantics for real (non-simulated) clients.

    Subclasses implement ``_request(probe) -> ResponseMessage`` (raising ProbeError
    on a per-probe failure). At temperature 0, identical requests — and
    deterministic 4xx errors — are cached, so probes that share a request hit the
    backend once; a transient failure is never cached.
    """

    def __init__(self, model: str, temperature: float, metadata: Dict[str, Any]):
        self.model = model
        self.temperature = temperature
        self.metadata = metadata
        self._cache: Dict[str, Any] = {}

    @property
    def produces_variance(self) -> bool:
        # Only temperature>0 yields independent samples; at temp 0 the backend is
        # deterministic and identical requests are cached, so N samples collapse to
        # one. The runner uses this to record an honest sample count.
        return self.temperature > 0

    def prepare(self, probes: List[Probe]) -> None:  # no-op; here for the protocol
        return None

    def _cache_key(self, probe: Probe) -> str:
        return json.dumps(
            [self.model, probe.messages, probe.tools], sort_keys=True, separators=(",", ":")
        )

    def complete(self, probe: Probe) -> ResponseMessage:
        key = self._cache_key(probe) if self.temperature == 0 else None
        if key is not None and key in self._cache:
            cached = self._cache[key]
            if isinstance(cached, ProbeError):
                raise cached
            return cached
        try:
            result = self._request(probe)
        except ProbeError as exc:
            if key is not None and not exc.transient:
                self._cache[key] = exc
            raise
        if key is not None:
            self._cache[key] = result
        return result

    def _request(self, probe: Probe) -> ResponseMessage:
        raise NotImplementedError


class HttpClient(_EndpointClient):
    """OpenAI-compatible /chat/completions client (stdlib only)."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "",
        api_key: str = "",
        timeout: float = 60.0,
        quant: str = "",
        runtime: str = "",
        temperature: float = 0.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        super().__init__(model, temperature, {
            "label": f"{model} @ {quant or 'native'} ({runtime or base_url})",
            "model": model,
            "quant": quant,
            "runtime": runtime or base_url,
        })

    def _request(self, probe: Probe) -> ResponseMessage:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": probe.messages,
            "temperature": self.temperature,
        }
        if probe.tools:
            body["tools"] = probe.tools
            body["tool_choice"] = "auto"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as fh:  # noqa: S310
                payload = json.loads(fh.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300].strip()
            if exc.code in (401, 403, 404):
                # Configuration error (model not pulled, wrong URL, bad auth):
                # FATAL. Scoring it as a per-probe 0 would write an all-zeros
                # lockfile that, committed as a baseline, neuters the gate.
                hint = ""
                if exc.code == 404:
                    hint = (
                        f" — is --model pulled, and does the base URL end in /v1? "
                        f"(got {self.base_url})"
                    )
                raise ClientError(f"HTTP {exc.code} from {self.base_url}: {detail}{hint}") from exc
            # Other 4xx/5xx (e.g. 400 "model does not support tools") are a real
            # per-probe capability failure: scored 0, not fatal. A 4xx is
            # deterministic (cacheable); a 5xx may be transient.
            raise ProbeError(f"HTTP {exc.code}: {detail}", transient=exc.code >= 500) from exc
        except urllib.error.URLError as exc:
            # Could not reach the server at all -> fatal; nothing can be measured.
            raise ClientError(
                f"Could not reach {self.base_url} ({exc.reason}). Is the model server running?"
            ) from exc
        except (TimeoutError, OSError) as exc:
            # Read timeout or transient socket error on one probe -> recoverable.
            raise ProbeError(
                f"request timed out after {self.timeout:g}s (try --timeout): {exc}"
            ) from exc
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProbeError(f"Non-JSON response from {self.base_url}: {exc}") from exc

        try:
            msg = payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProbeError(
                f"Unexpected response shape: {str(payload)[:200]}"
            ) from exc

        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments", "{}")
            # Most servers send arguments as a JSON string; some send a dict.
            if not isinstance(args, str):
                args = json.dumps(args)
            calls.append(ToolCall(fn.get("name", ""), args))
        return ResponseMessage(content=_clean_content(msg), tool_calls=calls)


def _bucket(probe_id: str) -> int:
    """Stable 0..999 bucket for a probe id (deterministic, no randomness)."""
    return int(hashlib.sha256(probe_id.encode()).hexdigest(), 16) % 1000


class SimulatedClient:
    """A deterministic model whose per-capability success rates come from a profile.

    profile = {
      "label": "...", "model": "...", "quant": "...", "runtime": "...",
      "capabilities": {"tool_selection": 0.67, "arg_validity": 1.0, ...}
    }

    For each capability, round(rate * n) of its probes (half-up) are made to pass,
    chosen by stable hash order — so the measured score equals the target rate
    (to that quantization) and is identical on every run. Fail responses are
    crafted to be genuinely failing for the probe's schema.
    """

    produces_variance = False  # deterministic per probe id, regardless of N

    def __init__(self, profile: Dict[str, Any]):
        self.profile = profile
        caps = profile.get("capabilities", {})
        self.metadata = {
            "label": profile.get("label", profile.get("model", "simulated")),
            "model": profile.get("model", "simulated"),
            "quant": profile.get("quant", ""),
            "runtime": profile.get("runtime", "simulated"),
        }
        self._rates = caps
        self._passing: set[str] = set()

    def prepare(self, probes: List[Probe]) -> None:
        by_cap: Dict[str, List[Probe]] = {}
        for p in probes:
            by_cap.setdefault(p.capability, []).append(p)
        self._passing = set()
        for cap, group in by_cap.items():
            rate = float(self._rates.get(cap, 1.0))
            k = int(rate * len(group) + 0.5)  # half-up; avoids round()'s banker's rounding
            ordered = sorted(group, key=lambda p: _bucket(p.id))
            for p in ordered[:k]:
                self._passing.add(p.id)

    # --- response crafting -------------------------------------------------
    def complete(self, probe: Probe) -> ResponseMessage:
        passing = probe.id in self._passing
        cap = probe.capability
        valid_args = probe.reference.get("valid_args", {})

        if cap == "tool_selection":
            if passing:
                return self._call(probe.expected_tool, valid_args)
            return ResponseMessage(content="Sure, I can help you with that.")

        if cap == "tool_discrimination":
            if passing:
                return self._call(probe.expected_tool, valid_args)
            others = [n for n in self._tool_names(probe) if n != probe.expected_tool]
            if others:  # fail by calling the WRONG tool
                return self._call(others[0], {})
            return ResponseMessage(content="(declined to call a tool)")

        if cap == "needle_in_tools":
            if passing:
                return self._call(probe.expected_tool, valid_args)
            others = [n for n in self._tool_names(probe) if n != probe.expected_tool]
            if others:  # fail by getting lost and calling a distractor
                return self._call(others[0], {})
            return ResponseMessage(content="(no tool called)")

        if cap == "arity_robustness":
            full = probe.reference.get("full_args", {})
            if passing:
                return self._call(probe.expected_tool, full)
            partial = dict(full)
            if partial:  # fail by dropping a parameter (not every field filled)
                partial.pop(next(iter(partial)))
            return self._call(probe.expected_tool, partial)

        if cap == "tool_restraint":
            if passing:  # benign task -> answer, no tool call
                return ResponseMessage(content="Paris is the capital of France.")
            names = self._tool_names(probe)  # fail by over-triggering a tool
            return self._call(names[0] if names else "__spurious__", {})

        if cap == "tool_permission":
            if passing:  # respect the prohibition -> don't call the forbidden tool
                return ResponseMessage(content="I'm not permitted to use that tool.")
            return self._call(probe.expected_tool, valid_args)  # call forbidden -> fail

        if cap == "no_hallucinated_tool":
            if passing:  # the needed tool isn't offered -> decline, don't invent one
                return ResponseMessage(content="No available tool fits that request.")
            absent = probe.reference.get("absent_tool", "__ghost_tool__")
            return self._call(absent, {})  # fabricate a call to a not-offered tool -> fail

        if cap == "arg_validity":
            if passing:
                return self._call(probe.expected_tool, valid_args)
            return self._call(probe.expected_tool, self._type_break(probe.schema, valid_args))

        if cap == "required_args":
            if passing:
                return self._call(probe.expected_tool, valid_args)
            required = (probe.schema or {}).get("required", [])
            if required:
                dropped = {k: v for k, v in valid_args.items() if k != required[0]}
                return self._call(probe.expected_tool, dropped)
            # No required args => can't fail by omission; omit the call entirely
            # so the probe deterministically scores 0 (guaranteed-failing response).
            return ResponseMessage(content="(declined to call a tool)")

        if cap == "structured_output":
            if passing:
                return ResponseMessage(content=json.dumps(valid_args))
            return ResponseMessage(
                content=f"Sure! Here is the JSON:\n```json\n{json.dumps(valid_args)}\n```"
            )

        if cap == "format_adherence":
            text = probe.expected_text or ""
            if passing:
                return ResponseMessage(content=text)
            return ResponseMessage(content=f"{text} — happy to help!")

        return ResponseMessage(content="")  # pragma: no cover

    @staticmethod
    def _call(name, args) -> ResponseMessage:
        return ResponseMessage(tool_calls=[ToolCall(name=name, arguments=json.dumps(args))])

    @staticmethod
    def _tool_names(probe) -> List[str]:
        return [t.get("function", {}).get("name", "") for t in (probe.tools or [])]

    @staticmethod
    def _type_break(schema, valid_args):
        """Return args that are GUARANTEED to fail arg_validity for this schema."""
        props = (schema or {}).get("properties", {})
        for key in ((schema or {}).get("required", []) or list(props)):
            sub = props.get(key, {})
            if sub.get("enum"):
                return {**valid_args, key: "__not_in_enum__"}
            t = sub.get("type")
            wrong = {
                "string": 99999,
                "integer": "not-a-number",
                "number": "not-a-number",
                "boolean": "not-a-bool",
                "array": "not-an-array",
                "object": "not-an-object",
            }.get(t)
            if wrong is not None:
                return {**valid_args, key: wrong}
        # No property carries a falsifiable constraint: make the whole payload
        # structurally invalid (a non-object) so arg_validity still fails.
        return ["__invalid_args__"]


class _SdkClient(_EndpointClient):
    """Route through a unified-provider Python SDK (any-llm / litellm) that returns
    OpenAI ChatCompletion objects. The model string carries the provider, e.g.
    ``anthropic/claude-3-5-sonnet`` or ``ollama/llama3.1``. One dependency, many
    providers — still deterministic (a transport layer, not an LLM judge)."""

    runtime = "sdk"

    def __init__(self, model: str, api_key: str = "", temperature: float = 0.0, quant: str = ""):
        self._completion = self._import_completion()
        self._api_key = api_key or None
        super().__init__(model, temperature, {
            "label": f"{model} @ {quant or 'native'} ({self.runtime})",
            "model": model,
            "quant": quant,
            "runtime": self.runtime,
        })

    @staticmethod
    def _import_completion():  # pragma: no cover - trivial import shim
        raise NotImplementedError

    def _request(self, probe: Probe) -> ResponseMessage:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": probe.messages,
            "temperature": self.temperature,
        }
        if probe.tools:
            kwargs["tools"] = probe.tools
            kwargs["tool_choice"] = "auto"
        if self._api_key:
            kwargs["api_key"] = self._api_key
        try:
            resp = self._completion(**kwargs)
        except Exception as exc:  # noqa: BLE001 - SDKs raise varied provider errors
            # Recoverable per-probe (transient by default); an all-failed run is
            # still caught by the all-errored fatal guard.
            raise ProbeError(f"{self.runtime}: {exc}") from exc
        return _openai_object_to_response(resp)


class AnyLlmClient(_SdkClient):
    """Provider-agnostic via mozilla-ai any-llm (`pip install 'probelock[anyllm]'`)."""

    runtime = "any-llm"

    @staticmethod
    def _import_completion():
        try:
            from any_llm import completion
        except ImportError as exc:
            raise ClientError(
                "any-llm is not installed — `pip install 'probelock[anyllm]'` "
                "(plus the provider SDK, e.g. 'any-llm-sdk[anthropic]')."
            ) from exc
        return completion


class LiteLlmClient(_SdkClient):
    """Provider-agnostic via litellm (`pip install 'probelock[litellm]'`).

    For a running LiteLLM *proxy*, no extra is needed — point --endpoint at it."""

    runtime = "litellm"

    @staticmethod
    def _import_completion():
        try:
            from litellm import completion
        except ImportError as exc:
            raise ClientError(
                "litellm is not installed — `pip install 'probelock[litellm]'`."
            ) from exc
        return completion
