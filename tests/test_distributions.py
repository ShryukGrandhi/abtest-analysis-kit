import math

import pytest

from abkit.distributions import chi2_sf, gamma_q, norm_cdf, norm_ppf, two_sided_p


@pytest.mark.parametrize("x,df,expected", [
    (3.841458820694124, 1, 0.05),
    (6.634896601021214, 1, 0.01),
    (5.991464547107979, 2, 0.05),
    (7.814727903251178, 3, 0.05),
    (18.307038053275146, 10, 0.05),
    (124.34211340400407, 100, 0.05),
])
def test_chi2_sf_matches_table_quantiles(x, df, expected):
    assert chi2_sf(x, df) == pytest.approx(expected, rel=1e-8)


def test_chi2_df2_is_exponential():
    for x in (0.1, 1.0, 4.0, 30.0):
        assert chi2_sf(x, 2) == pytest.approx(math.exp(-x / 2), rel=1e-12)


def test_chi2_df1_matches_normal():
    # chi2(1) is the square of a standard normal.
    for z in (0.3, 1.0, 2.5, 4.0):
        assert chi2_sf(z * z, 1) == pytest.approx(two_sided_p(z), rel=1e-9)


def test_gamma_q_edges():
    assert gamma_q(3.0, 0.0) == 1.0
    assert chi2_sf(-1.0, 3) == 1.0
    with pytest.raises(ValueError):
        gamma_q(0, 1)


def test_normal_roundtrip():
    for p in (0.001, 0.025, 0.5, 0.975):
        assert norm_cdf(norm_ppf(p)) == pytest.approx(p, abs=1e-12)
    assert two_sided_p(1.959963984540054) == pytest.approx(0.05, rel=1e-9)
