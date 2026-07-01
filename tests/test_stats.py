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


def test_zero_pooled_variance_fail_safe_true():
    # p_baseline=0.01 over 10 trials rounds to 0 passes; p_candidate=0.0 is also 0
    # passes, so the pooled proportion is exactly 0 and variance is 0 -- the z-score
    # is undefined. This is a genuine (if tiny) drop, so the defensive branch must not
    # silence it.
    assert is_significant_regression(0.01, 10, 0.0, 10, 0.95) is True
