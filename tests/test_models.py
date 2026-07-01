import pytest

from probelock.models import Lockfile


def test_from_dict_rejects_non_numeric_capability_score():
    with pytest.raises(ValueError):
        Lockfile.from_dict({"capabilities": {"tool_selection": "not-a-number"}})


def test_from_dict_rejects_non_object():
    with pytest.raises(ValueError):
        Lockfile.from_dict(["not", "a", "dict"])


def test_from_dict_rejects_non_object_capabilities():
    with pytest.raises(ValueError):
        Lockfile.from_dict({"capabilities": ["not", "a", "dict"]})


def test_from_dict_is_lenient_on_result_entries():
    lf = Lockfile.from_dict(
        {
            "capabilities": {"a": 1.0},
            "results": [
                {"probe_id": "p1", "capability": "a", "score": 1.0},
                "junk-non-dict-entry",
                {"capability": "a", "score": "bad"},  # bad score -> 0.0, not a crash
            ],
        }
    )
    assert lf.capabilities == {"a": 1.0}
    assert len(lf.results) == 2  # the non-dict entry is skipped
    assert any(r.score == 0.0 for r in lf.results)


def test_roundtrip_preserves_fields():
    original = Lockfile.from_dict(
        {
            "label": "m @ Q4 (ollama)",
            "model": "m",
            "quant": "Q4",
            "runtime": "ollama",
            "tools_fingerprint": "abc123",
            "capabilities": {"a": 0.5, "b": 1.0},
            "results": [{"probe_id": "p", "capability": "a", "score": 0.5}],
            "n_probes": 1,
        }
    )
    back = Lockfile.from_dict(original.to_dict())
    assert back.capabilities == original.capabilities
    assert back.model == "m" and back.quant == "Q4"
    assert back.tools_fingerprint == "abc123"


def test_roundtrip_preserves_per_result_error_field():
    original = Lockfile.from_dict({
        "capabilities": {"tool_restraint": 1.0},
        "results": [
            {"probe_id": "p", "capability": "tool_restraint", "score": 1.0, "error": "boom"},
            {"probe_id": "p2", "capability": "tool_restraint", "score": 1.0},
        ],
        "n_probes": 2,
    })
    back = Lockfile.from_dict(original.to_dict())
    errors = {r.probe_id: r.error for r in back.results}
    assert errors["p"] == "boom"
    assert errors["p2"] is None
