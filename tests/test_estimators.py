import numpy as np
import pytest

from abkit.estimators import mean_diff, ratio_diff, ratio_variance, required_sample_size


def test_mean_diff_hand_computed():
    c = [1.0, 2.0, 3.0, 4.0]
    t = [2.0, 4.0, 6.0, 8.0]
    e = mean_diff(c, t)
    assert e.diff == pytest.approx(2.5)
    # var(c)=5/3, var(t)=20/3; se = sqrt(5/12 + 20/12)
    assert e.se == pytest.approx(np.sqrt(25 / 12))
    assert e.rel_lift == pytest.approx(1.0)
    assert e.ci_low < e.diff < e.ci_high


def test_mean_diff_needs_two_obs():
    with pytest.raises(ValueError):
        mean_diff([1.0], [1.0, 2.0])


def test_relative_lift_se_matches_simulation():
    rng = np.random.default_rng(1)
    n = 4000
    rels, ses = [], []
    for _ in range(400):
        c = rng.exponential(10.0, n)
        t = rng.exponential(11.0, n)
        e = mean_diff(c, t)
        rels.append(e.rel_lift)
        ses.append(e.rel_se)
    assert np.std(rels) == pytest.approx(np.mean(ses), rel=0.1)


def test_ratio_variance_matches_simulation():
    # Users have clustered sessions; empirical spread of the ratio across
    # replications should match the delta-method SE.
    rng = np.random.default_rng(2)
    ratios, ses = [], []
    for _ in range(400):
        den = rng.poisson(rng.gamma(2.0, 2.0, 3000)).astype(float)
        num = rng.binomial(den.astype(int), 0.1).astype(float)
        r, v = ratio_variance(num, den)
        ratios.append(r)
        ses.append(np.sqrt(v))
    assert np.std(ratios) == pytest.approx(np.mean(ses), rel=0.1)


def test_ratio_diff_detects_effect():
    rng = np.random.default_rng(3)
    den_c = rng.poisson(5, 5000).astype(float)
    den_t = rng.poisson(5, 5000).astype(float)
    num_c = rng.binomial(den_c.astype(int), 0.10).astype(float)
    num_t = rng.binomial(den_t.astype(int), 0.12).astype(float)
    e = ratio_diff(num_c, den_c, num_t, den_t)
    assert e.p_value < 1e-6
    assert e.ci_low < 0.02 < e.ci_high


def test_required_sample_size_achieves_power():
    sd, mde = 1.0, 0.1
    n = required_sample_size(sd, mde)
    assert 1560 <= n <= 1580  # textbook 2*(1.96+0.8416)^2/0.01 = 1569.8
    rng = np.random.default_rng(4)
    hits = sum(mean_diff(rng.normal(0, sd, n), rng.normal(mde, sd, n)).p_value < 0.05
               for _ in range(1000))
    assert 0.75 < hits / 1000 < 0.85
