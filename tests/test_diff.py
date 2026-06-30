from probelock.diff import diff_lockfiles
from probelock.models import Lockfile, ProbeResult


def lock(caps, fingerprint="fp1"):
    return Lockfile(
        label="l",
        model="m",
        quant="q",
        runtime="r",
        tools_fingerprint=fingerprint,
        probelock_version="0",
        capabilities=caps,
        results=[],
        n_probes=0,
    )


def lock_with(caps, samples=1, probes_per_cap=3):
    """A lockfile with per-capability results so trial counts are well-defined."""
    results = [
        ProbeResult(f"{cap}::{i}", cap, score)
        for cap, score in caps.items()
        for i in range(probes_per_cap)
    ]
    return Lockfile(
        label="l", model="m", quant="q", runtime="r", tools_fingerprint="fp",
        probelock_version="0", capabilities=dict(caps), results=results,
        n_probes=len(results), samples=samples,
    )


def test_regression_detected():
    result = diff_lockfiles(lock({"a": 1.0, "b": 1.0}), lock({"a": 0.6, "b": 0.98}), max_drop=0.05)
    status = {r.capability: r.status for r in result.rows}
    assert status["a"] == "regression"
    assert status["b"] == "ok"
    assert result.regressed is True
    assert [r.capability for r in result.regressions] == ["a"]


def test_no_regression_within_threshold():
    assert diff_lockfiles(lock({"a": 1.0}), lock({"a": 0.97}), max_drop=0.05).regressed is False


def test_improved_and_added_are_not_regressions():
    result = diff_lockfiles(lock({"a": 0.5}), lock({"a": 0.9, "only_cand": 1.0}), max_drop=0.05)
    status = {r.capability: r.status for r in result.rows}
    assert status["a"] == "improved"
    assert status["only_cand"] == "added"
    assert result.regressed is False


def test_removed_capability_is_a_regression():
    # Losing a whole capability across a swap is exactly what the gate must catch.
    result = diff_lockfiles(lock({"a": 1.0, "gone": 1.0}), lock({"a": 1.0}), max_drop=0.05)
    status = {r.capability: r.status for r in result.rows}
    assert status["gone"] == "removed"
    assert result.regressed is True


def test_tools_changed_flag():
    assert diff_lockfiles(lock({"a": 1.0}, "x"), lock({"a": 1.0}, "y")).tools_changed is True
    assert diff_lockfiles(lock({"a": 1.0}, "x"), lock({"a": 1.0}, "x")).tools_changed is False


def test_confidence_marks_small_sample_drop_noisy():
    # 1.0 -> 0.667 across 3 probes x 1 sample = 3 trials: not significant -> 'noisy', no fail.
    result = diff_lockfiles(
        lock_with({"tool_selection": 1.0}, samples=1),
        lock_with({"tool_selection": 0.667}, samples=1),
        max_drop=0.05, confidence=0.95,
    )
    row = {r.capability: r for r in result.rows}["tool_selection"]
    assert row.status == "noisy"
    assert result.regressed is False


def test_confidence_flags_well_sampled_drop_as_regression():
    # Same drop with 10 samples (30 trials) is statistically real -> regression.
    result = diff_lockfiles(
        lock_with({"tool_selection": 1.0}, samples=10),
        lock_with({"tool_selection": 0.667}, samples=10),
        max_drop=0.05, confidence=0.95,
    )
    row = {r.capability: r for r in result.rows}["tool_selection"]
    assert row.status == "regression"
    assert result.regressed is True


def test_without_confidence_threshold_behaviour_unchanged():
    result = diff_lockfiles(lock_with({"a": 1.0}), lock_with({"a": 0.667}), max_drop=0.05)
    assert result.regressed is True
