import numpy as np
import pytest

from abkit.checks import benjamini_hochberg, holm, srm_test
from abkit.sequential import SequentialTest, msprt_lambda


def test_srm_balanced_passes():
    r = srm_test({"a": 10000, "b": 10050})
    assert not r.mismatch
    assert r.p_value > 0.5


def test_srm_detects_imbalance():
    r = srm_test({"a": 10000, "b": 10600})
    assert r.mismatch
    assert r.p_value < 1e-4


def test_srm_with_weights_and_three_arms():
    r = srm_test({"a": 5000, "b": 2500, "c": 2500}, weights={"a": 2, "b": 1, "c": 1})
    assert r.chi2 == pytest.approx(0.0)
    assert r.p_value == pytest.approx(1.0)
    assert r.expected["a"] == pytest.approx(5000)


def test_bh_known_values():
    p = [0.01, 0.04, 0.03, 0.005]
    # sorted: .005,.01,.03,.04 -> *4/rank = .02,.02,.04,.04
    assert benjamini_hochberg(p) == pytest.approx([0.02, 0.04, 0.04, 0.02])


def test_holm_known_values():
    p = [0.01, 0.04, 0.03, 0.005]
    # sorted: .005*4=.02, .01*3=.03, .03*2=.06, .04*1=.04 -> monotone .06
    assert holm(p) == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_corrections_are_monotone_and_bounded():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=50)
    for f in (benjamini_hochberg, holm):
        q = f(p)
        assert np.all(q >= p - 1e-15)
        assert np.all(q <= 1)
        order = np.argsort(p)
        assert np.all(np.diff(q[order]) >= -1e-15)


def test_msprt_lambda_shrinks_without_signal():
    assert msprt_lambda(0.0, 1.0, 1.0) < 1.0
    assert msprt_lambda(5.0, 1.0, 1.0) > msprt_lambda(1.0, 1.0, 1.0)


def test_sequential_pvalue_is_monotone_and_detects_effect():
    rng = np.random.default_rng(1)
    st = SequentialTest(tau2=0.01)
    for _ in range(50):
        st.update(rng.normal(0, 1, 200), rng.normal(0.15, 1, 200))
    h = st.history
    assert all(b <= a for a, b in zip(h, h[1:]))
    assert st.rejected


def test_sequential_controls_type1_under_peeking():
    rng = np.random.default_rng(2)
    sims, looks = 300, 30
    msprt_fp = naive_fp = 0
    for _ in range(sims):
        st = SequentialTest(tau2=0.01)
        naive = False
        for _ in range(looks):
            st.update(rng.normal(0, 1, 100), rng.normal(0, 1, 100))
            naive |= st.naive_p_value() < 0.05
        msprt_fp += st.rejected
        naive_fp += naive
    assert msprt_fp / sims <= 0.07
    assert naive_fp / sims > 0.15
