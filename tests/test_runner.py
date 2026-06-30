import json
from pathlib import Path

from probelock.clients import ProbeError, SimulatedClient
from probelock.diff import diff_lockfiles
from probelock.lockfile import read_lockfile, write_lockfile
from probelock.models import ResponseMessage
from probelock.probes import derive_probes, tools_fingerprint
from probelock.runner import run_probes


class _VarianceClient:
    """A stochastic client stand-in (independent samples)."""
    produces_variance = True
    metadata = {"label": "v", "model": "v", "quant": "", "runtime": "v"}

    def prepare(self, probes):
        pass

    def complete(self, probe):
        return ResponseMessage(content="x")


class _AllErrorClient:
    produces_variance = False
    metadata = {"label": "e", "model": "e", "quant": "", "runtime": "e"}

    def prepare(self, probes):
        pass

    def complete(self, probe):
        raise ProbeError("model does not support tools")

ROOT = Path(__file__).resolve().parents[1]
TOOLS = json.loads((ROOT / "examples" / "agent_tools.json").read_text())
Q8 = json.loads((ROOT / "fixtures" / "profile_q8.json").read_text())
Q4 = json.loads((ROOT / "fixtures" / "profile_q4.json").read_text())


def run(profile):
    return run_probes(SimulatedClient(profile), derive_probes(TOOLS), tools_fingerprint(TOOLS), "0.1.0")


def test_q8_all_capabilities_perfect():
    lock = run(Q8)
    assert lock.n_probes == 32  # 3 tools x 9 per-tool caps + 3 restraint + 2 format
    assert all(v == 1.0 for v in lock.capabilities.values())


def test_deterministic_client_clamps_samples_to_1():
    # SimulatedClient (and any temp-0 endpoint) is deterministic, so N samples are
    # identical; recording samples=N would inflate the significance test's trials.
    lock = run_probes(SimulatedClient(Q8), derive_probes(TOOLS), tools_fingerprint(TOOLS), "0", samples=4)
    assert lock.samples == 1
    assert all(v == 1.0 for v in lock.capabilities.values())


def test_variance_client_keeps_requested_samples():
    lock = run_probes(_VarianceClient(), derive_probes(TOOLS), tools_fingerprint(TOOLS), "0", samples=3)
    assert lock.samples == 3


_NEGATIVE = ("tool_restraint", "tool_permission", "no_hallucinated_tool")


def test_negative_capabilities_pass_when_model_cannot_tool_call():
    # An API rejection isn't a negative-probe failure: a model that can't accept
    # tools can't over-trigger, call a forbidden tool, or hallucinate one. The
    # negatives -> 1.0; the real tool capabilities -> 0.0.
    lock = run_probes(_AllErrorClient(), derive_probes(TOOLS), tools_fingerprint(TOOLS), "0")
    for cap in _NEGATIVE:
        assert lock.capabilities[cap] == 1.0
    assert lock.capabilities["tool_selection"] == 0.0
    # ...but they are STILL error-tagged, so the all-errored guard sees a broken run.
    negs = [r for r in lock.results if r.capability in _NEGATIVE]
    assert all(r.score == 1.0 and r.error is not None for r in negs)


def test_all_errored_run_tags_every_probe():
    # The fatal all-errored guard counts error-tagged probes; tool_restraint scoring
    # 1.0 must not hide a totally-broken run from it.
    lock = run_probes(_AllErrorClient(), derive_probes(TOOLS), tools_fingerprint(TOOLS), "0")
    assert len([r for r in lock.results if r.error]) == lock.n_probes


def test_simulated_tool_restraint_fails_with_empty_toolset():
    probes = derive_probes([])  # no tools -> only restraint + format probes
    lock = run_probes(
        SimulatedClient({"capabilities": {"tool_restraint": 0.0, "format_adherence": 1.0}}),
        probes, "fp", "0",
    )
    assert lock.capabilities["tool_restraint"] == 0.0


def test_runs_are_deterministic():
    assert run(Q8).capabilities == run(Q8).capabilities
    assert run(Q4).capabilities == run(Q4).capabilities


def test_q4_regresses_against_q8():
    result = diff_lockfiles(run(Q8), run(Q4), max_drop=0.05)
    regressed = {r.capability for r in result.regressions}
    assert {"tool_selection", "arg_validity", "structured_output"} <= regressed
    assert result.regressed is True


def test_q4_holds_robust_capabilities():
    caps = run(Q4).capabilities
    assert caps["required_args"] == 1.0
    assert caps["format_adherence"] == 1.0


CONSTRAINED_TOOL = [{
    "type": "function",
    "function": {
        "name": "set_unit",
        "description": "Set the temperature unit",
        "parameters": {
            "type": "object",
            "properties": {"unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
            "required": ["unit"],
        },
    },
}]
_ALL_CAPS = [
    "tool_selection", "tool_discrimination", "needle_in_tools", "arity_robustness",
    "tool_permission", "no_hallucinated_tool", "arg_validity", "required_args",
    "structured_output", "tool_restraint", "format_adherence",
]


def test_pass_profile_scores_1_for_enum_tool():
    # Proves synth_value honors enum: the simulated PASS path emits {"unit":"celsius"},
    # which validates -> arg_validity/structured_output = 1.0 (was 0.0 before the fix).
    probes = derive_probes(CONSTRAINED_TOOL)
    lock = run_probes(SimulatedClient({"capabilities": {c: 1.0 for c in _ALL_CAPS}}), probes, "fp", "0")
    assert lock.capabilities["arg_validity"] == 1.0
    assert lock.capabilities["structured_output"] == 1.0


def test_fail_profile_scores_0_for_enum_tool():
    # Proves the SimulatedClient fail-paths are genuinely failing for a constrained schema.
    probes = derive_probes(CONSTRAINED_TOOL)
    lock = run_probes(SimulatedClient({"capabilities": {c: 0.0 for c in _ALL_CAPS}}), probes, "fp", "0")
    assert all(v == 0.0 for v in lock.capabilities.values())


def test_lockfile_roundtrip(tmp_path):
    lock = run(Q8)
    path = tmp_path / "q8.lock"
    write_lockfile(lock, path)
    back = read_lockfile(path)
    assert back.capabilities == lock.capabilities
    assert back.n_probes == lock.n_probes
    assert back.tools_fingerprint == lock.tools_fingerprint
