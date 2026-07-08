"""Unit tests for N-way trend analysis and model-family normalization."""

from probelock.diff import model_family
from probelock.models import Lockfile
from probelock.trend import trend_lockfiles


def _lock(caps, model="m", quant="", runtime="", tools_fp="fp", samples=1):
    return Lockfile(
        label="", model=model, quant=quant, runtime=runtime,
        tools_fingerprint=tools_fp, probelock_version="test",
        capabilities=caps, results=[], n_probes=len(caps), samples=samples,
    )


def _row(result, cap):
    return next(r for r in result.rows if r.capability == cap)


# --- model_family -------------------------------------------------------------------


def test_model_family_strips_runtime_and_quant_markers():
    # runtime swap: same family
    assert model_family("qwen3.5:9b", "", "ollama") == model_family("qwen3.5:9b-mlx", "", "ollama-mlx")
    # quant swap: same family
    assert (model_family("llama3.1:8b-instruct-q8_0", "Q8_0", "ollama")
            == model_family("llama3.1:8b-instruct-q4_K_M", "Q4_K_M", "ollama"))


def test_model_family_keeps_genuinely_different_models_distinct():
    assert model_family("qwen3.5:9b") != model_family("qwen3.5:32b")  # size is identity
    assert model_family("llama3.1:8b") != model_family("mistral:7b")


def test_model_family_strips_iq_and_backend_markers():
    # importance-matrix (IQ) quants and less-common backends must count as variant
    # markers too, or a within-model quant/runtime swap reads as cross-model
    base = model_family("llama3.1:8b-instruct-q4_K_M", "Q4_K_M", "ollama")
    assert model_family("llama3.1:8b-instruct-iq2_xs", "", "") == base
    assert model_family("qwen2.5:7b-iq2_xs") == model_family("qwen2.5:7b-iq3_m")  # IQ vs IQ
    assert model_family("phi3:mini-coreml", "", "coreml") == model_family("phi3:mini", "", "cpu")


def test_model_family_never_strips_the_base_tag():
    # the first hyphen token is the base name:tag and must survive even if it looks
    # quant-ish after the colon
    assert model_family("q4:latest") == "q4:latest"


def test_model_family_strips_field_valued_suffix():
    # a suffix that equals the recorded runtime/quant field is stripped even if it is
    # not in the known-marker set
    assert model_family("custommodel-turbo", runtime="turbo") == "custommodel"


# --- trend_lockfiles ----------------------------------------------------------------


def test_trend_flags_monotonic_regression_across_a_ladder():
    locks = [_lock({"tool_selection": v}) for v in (1.0, 1.0, 0.67, 0.33)]
    result = trend_lockfiles(locks, ["Q8", "Q6", "Q4", "Q2"], max_drop=0.05)
    row = _row(result, "tool_selection")
    assert row.values == [1.0, 1.0, 0.67, 0.33]
    assert row.delta == -0.67
    assert row.status == "regressed"
    assert row.monotonic is True
    assert row.worst_step == -0.34
    assert [r.capability for r in result.regressed] == ["tool_selection"]


def test_trend_marks_stable_when_it_holds():
    locks = [_lock({"tool_restraint": 1.0}) for _ in range(4)]
    result = trend_lockfiles(locks, ["a", "b", "c", "d"])
    row = _row(result, "tool_restraint")
    assert row.status == "stable"
    assert row.delta == 0.0
    assert result.regressed == []


def test_trend_detects_a_dip_that_endpoints_hide():
    # net-flat (1.0 -> 1.0) but cliffs to 0.4 in the middle: a pairwise diff of the
    # endpoints would call this stable; the ladder shows it is not.
    locks = [_lock({"structured_output": v}) for v in (1.0, 0.4, 1.0)]
    row = _row(trend_lockfiles(locks, ["a", "b", "c"], max_drop=0.05), "structured_output")
    assert row.delta == 0.0
    assert row.status == "unstable"
    assert row.worst_step == -0.6
    assert row.monotonic is False


def test_trend_marks_improvement():
    locks = [_lock({"arg_validity": v}) for v in (0.5, 0.8, 1.0)]
    row = _row(trend_lockfiles(locks, ["a", "b", "c"]), "arg_validity")
    assert row.status == "improved"
    assert row.delta == 0.5


def test_trend_partial_when_capability_only_appears_at_the_end():
    # present in only the LAST lockfile (added late): a single data point, no trend to
    # compute, and not a drop — so "partial", never counted as a regression
    locks = [_lock({"tool_selection": 1.0}),
             _lock({"tool_selection": 1.0, "new_cap": 0.9})]
    result = trend_lockfiles(locks, ["a", "b"])
    row = _row(result, "new_cap")
    assert row.values == [None, 0.9]
    assert row.delta is None
    assert row.status == "partial"
    assert "new_cap" not in {r.capability for r in result.regressed}


def test_trend_marks_a_dropped_capability_removed_not_stable():
    # present early, gone on the final rung: must be a regression, not stable/improved
    # (a plain present-only delta would have hidden it)
    locks = [_lock({"tool_selection": 0.9}), _lock({"tool_selection": 0.9}), _lock({})]
    result = trend_lockfiles(locks, ["a", "b", "c"])
    row = _row(result, "tool_selection")
    assert row.values == [0.9, 0.9, None]
    assert row.status == "removed"
    assert row.delta is None
    assert row.capability in {r.capability for r in result.regressed}

    # even a rising-then-dropped capability is removed, never "improved"
    rising = [_lock({"x": 0.3}), _lock({"x": 0.9}), _lock({})]
    assert _row(trend_lockfiles(rising, ["a", "b", "c"]), "x").status == "removed"


def test_trend_interior_gap_is_not_removed():
    # absent in the middle but present at the end: a cosmetic gap, not a drop
    locks = [_lock({"x": 0.9}), _lock({}), _lock({"x": 0.9})]
    row = _row(trend_lockfiles(locks, ["a", "b", "c"]), "x")
    assert row.status == "stable"
    assert row.values == [0.9, None, 0.9]


def test_trend_capabilities_are_the_sorted_union():
    locks = [_lock({"b_cap": 1.0}), _lock({"a_cap": 1.0, "b_cap": 1.0})]
    result = trend_lockfiles(locks, ["x", "y"])
    assert [r.capability for r in result.rows] == ["a_cap", "b_cap"]
