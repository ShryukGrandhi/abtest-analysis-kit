"""Small set of distribution functions so the package only depends on numpy.

The normal functions wrap ``statistics.NormalDist``. The chi-square survival
function is computed from the regularized incomplete gamma function using the
series / continued-fraction split from Numerical Recipes (section 6.2).
"""

import math
from statistics import NormalDist

_STD_NORMAL = NormalDist()

_EPS = 1e-15
_MAX_ITER = 500
_TINY = 1e-300


def norm_cdf(x):
    return _STD_NORMAL.cdf(x)


def norm_ppf(p):
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    return _STD_NORMAL.inv_cdf(p)


def two_sided_p(z):
    """Two-sided p-value for a standard normal test statistic."""
    if math.isnan(z):
        return float("nan")
    return 2.0 * norm_cdf(-abs(z))


def _gamma_p_series(a, x):
    term = 1.0 / a
    total = term
    ap = a
    for _ in range(_MAX_ITER):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * _EPS:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_q_contfrac(a, x):
    # Modified Lentz evaluation of the continued fraction for Q(a, x).
    b = x + 1.0 - a
    c = 1.0 / _TINY
    d = 1.0 / b
    h = d
    for i in range(1, _MAX_ITER):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _TINY:
            d = _TINY
        c = b + an / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gamma_q(a, x):
    """Regularized upper incomplete gamma function Q(a, x)."""
    if a <= 0:
        raise ValueError("a must be positive")
    if x < 0:
        raise ValueError("x must be non-negative")
    if x == 0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _gamma_p_series(a, x)
    return _gamma_q_contfrac(a, x)


def chi2_sf(x, df):
    """P(X >= x) for X ~ chi-square(df)."""
    if df <= 0:
        raise ValueError("df must be positive")
    if x <= 0:
        return 1.0
    return gamma_q(df / 2.0, x / 2.0)
