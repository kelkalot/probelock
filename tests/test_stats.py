from probelock.stats import is_significant_regression


def test_small_sample_drop_is_not_significant():
    # 1.0 -> 0.667 over only 3 trials each is within sampling noise at 95%.
    assert is_significant_regression(1.0, 3, 0.667, 3, 0.95) is False


def test_same_drop_with_more_samples_is_significant():
    # The identical drop over 30 trials each is real.
    assert is_significant_regression(1.0, 30, 0.667, 30, 0.95) is True


def test_no_drop_is_never_significant():
    assert is_significant_regression(0.8, 100, 0.9, 100, 0.95) is False


def test_unknown_trials_fail_safe_true():
    # Can't test without trial counts -> don't silence a possible regression.
    assert is_significant_regression(1.0, 0, 0.0, 0, 0.95) is True
