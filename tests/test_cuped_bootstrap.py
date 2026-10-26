import numpy as np
import pytest

from abkit.bootstrap import bootstrap_diff
from abkit.cuped import cuped_diff, theta


def _correlated(rng, n, rho, effect=0.0):
    x = rng.normal(0, 1, n)
    y = rho * x + np.sqrt(1 - rho ** 2) * rng.normal(0, 1, n) + effect
    return y, x


def test_theta_recovers_slope():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 50000)
    y = 3.0 * x + rng.normal(0, 1, 50000)
    assert theta(y, x) == pytest.approx(3.0, abs=0.02)


def test_cuped_variance_reduction_is_rho_squared():
    rng = np.random.default_rng(1)
    yc, xc = _correlated(rng, 20000, 0.7)
    yt, xt = _correlated(rng, 20000, 0.7, effect=0.05)
    r = cuped_diff(yc, xc, yt, xt)
    assert r.variance_reduction == pytest.approx(0.49, abs=0.03)
    assert r.estimate.se < r.raw.se


def test_cuped_is_unbiased():
    rng = np.random.default_rng(2)
    diffs = []
    for _ in range(300):
        yc, xc = _correlated(rng, 2000, 0.6)
        yt, xt = _correlated(rng, 2000, 0.6, effect=0.1)
        diffs.append(cuped_diff(yc, xc, yt, xt).estimate.diff)
    se_mean = np.std(diffs) / np.sqrt(len(diffs))
    assert abs(np.mean(diffs) - 0.1) < 4 * se_mean


def test_cuped_with_useless_covariate_changes_little():
    rng = np.random.default_rng(3)
    yc, yt = rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)
    xc, xt = rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)
    r = cuped_diff(yc, xc, yt, xt)
    assert abs(r.variance_reduction) < 0.01


def test_bootstrap_ci_matches_normal_ci():
    rng = np.random.default_rng(4)
    c, t = rng.normal(10, 2, 3000), rng.normal(10.2, 2, 3000)
    point, lo, hi, reps = bootstrap_diff(c, t, n_boot=4000, seed=5)
    assert point == pytest.approx(t.mean() - c.mean())
    se = np.sqrt(c.var(ddof=1) / 3000 + t.var(ddof=1) / 3000)
    assert hi - lo == pytest.approx(2 * 1.96 * se, rel=0.08)
    assert lo < point < hi
    assert reps.shape == (4000,)


def test_bootstrap_is_deterministic_with_seed_and_chunking():
    rng = np.random.default_rng(6)
    c, t = rng.exponential(1, 500), rng.exponential(1, 500)
    a = bootstrap_diff(c, t, n_boot=300, seed=9)
    b = bootstrap_diff(c, t, n_boot=300, seed=9)
    assert a[1:3] == b[1:3]
    small = bootstrap_diff(c, t, n_boot=300, seed=9, chunk_bytes=8 * 500 * 7)
    assert small[3].shape == (300,)


def test_bootstrap_relative():
    rng = np.random.default_rng(7)
    c, t = rng.exponential(10, 4000), rng.exponential(12, 4000)
    point, lo, hi, _ = bootstrap_diff(c, t, n_boot=1000, seed=1, relative=True)
    assert lo < 0.2 < hi
