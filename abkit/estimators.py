"""Two-sample estimators for experiment metrics.

All tests use the normal approximation (Welch-style unpooled variance), which
is the standard choice for online experiments with thousands of users per arm.
Relative lift and ratio metrics use the delta method.
"""

import math
from dataclasses import dataclass

import numpy as np

from .distributions import norm_ppf, two_sided_p


@dataclass
class Estimate:
    control: float          # control-arm point value (mean or ratio)
    treatment: float        # treatment-arm point value
    diff: float             # treatment - control
    se: float               # standard error of diff
    ci_low: float
    ci_high: float
    p_value: float
    rel_lift: float         # treatment / control - 1
    rel_se: float
    rel_ci_low: float
    rel_ci_high: float
    n_control: int
    n_treatment: int

    def as_dict(self):
        return dict(self.__dict__)


def _z_crit(alpha):
    return norm_ppf(1.0 - alpha / 2.0)


def _build(mc, mt, var_c, var_t, rel_var, nc, nt, alpha):
    diff = mt - mc
    se = math.sqrt(var_c + var_t)
    z = diff / se if se > 0 else (0.0 if diff == 0 else math.copysign(math.inf, diff))
    zc = _z_crit(alpha)
    if mc != 0:
        rel = mt / mc - 1.0
        rel_se = math.sqrt(rel_var)
    else:
        rel, rel_se = float("nan"), float("nan")
    return Estimate(
        control=mc, treatment=mt, diff=diff, se=se,
        ci_low=diff - zc * se, ci_high=diff + zc * se,
        p_value=two_sided_p(z),
        rel_lift=rel, rel_se=rel_se,
        rel_ci_low=rel - zc * rel_se, rel_ci_high=rel + zc * rel_se,
        n_control=nc, n_treatment=nt,
    )


def mean_diff(control, treatment, alpha=0.05) -> Estimate:
    """Difference in means with a Welch z-test and delta-method relative lift.

    Var(mt/mc) ~= var_t/(n_t mc^2) + mt^2 var_c/(n_c mc^4)
    """
    c = np.asarray(control, dtype=float)
    t = np.asarray(treatment, dtype=float)
    nc, nt = c.size, t.size
    if nc < 2 or nt < 2:
        raise ValueError("each arm needs at least 2 observations")
    mc, mt = c.mean(), t.mean()
    vm_c = c.var(ddof=1) / nc   # variance of the control mean
    vm_t = t.var(ddof=1) / nt
    rel_var = (vm_t / mc ** 2 + mt ** 2 * vm_c / mc ** 4) if mc != 0 else float("nan")
    return _build(mc, mt, vm_c, vm_t, rel_var, nc, nt, alpha)


def ratio_variance(num, den):
    """Delta-method variance of sum(num)/sum(den) where units are the rows.

    Used for metrics like revenue per session, where the analysis unit (user)
    differs from the denominator unit (session) and naive per-session variance
    would understate the true variance.
    """
    x = np.asarray(num, dtype=float)
    y = np.asarray(den, dtype=float)
    n = x.size
    mx, my = x.mean(), y.mean()
    if my == 0:
        raise ValueError("ratio denominator has zero mean")
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    cxy = np.cov(x, y, ddof=1)[0, 1]
    r = mx / my
    var = (vx - 2 * r * cxy + r * r * vy) / (n * my * my)
    return r, max(var, 0.0)


def ratio_diff(num_c, den_c, num_t, den_t, alpha=0.05) -> Estimate:
    """Compare a ratio metric sum(num)/sum(den) between two arms."""
    nc, nt = len(num_c), len(num_t)
    if nc < 2 or nt < 2:
        raise ValueError("each arm needs at least 2 observations")
    rc, var_c = ratio_variance(num_c, den_c)
    rt, var_t = ratio_variance(num_t, den_t)
    rel_var = (var_t / rc ** 2 + rt ** 2 * var_c / rc ** 4) if rc != 0 else float("nan")
    return _build(rc, rt, var_c, var_t, rel_var, nc, nt, alpha)


def required_sample_size(baseline_sd, mde, alpha=0.05, power=0.8):
    """Users per arm for a two-sided z-test to detect absolute effect ``mde``."""
    if mde <= 0 or baseline_sd <= 0:
        raise ValueError("mde and baseline_sd must be positive")
    z = _z_crit(alpha) + norm_ppf(power)
    return int(math.ceil(2 * (z * baseline_sd / mde) ** 2))
