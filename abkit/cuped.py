"""CUPED variance reduction (Deng, Xu, Kohavi, Walker 2013).

Y_adj = Y - theta * (X - mean(X)),  theta = Cov(X, Y) / Var(X)

X is a pre-experiment covariate (usually the same metric measured before the
experiment). theta is estimated on the pooled data of all arms so the
adjustment is identical for each arm and the estimator stays unbiased. The
variance of the mean drops by a factor of (1 - corr(X, Y)^2).
"""

from dataclasses import dataclass

import numpy as np

from .estimators import Estimate, mean_diff


@dataclass
class CupedResult:
    estimate: Estimate
    theta: float
    variance_reduction: float   # 1 - Var(adjusted diff) / Var(raw diff)
    raw: Estimate


def theta(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    vx = x.var(ddof=1)
    if vx == 0:
        return 0.0
    return float(np.cov(x, y, ddof=1)[0, 1] / vx)


def adjust(y, x, th, x_mean):
    return np.asarray(y, dtype=float) - th * (np.asarray(x, dtype=float) - x_mean)


def cuped_diff(y_c, x_c, y_t, x_t, alpha=0.05) -> CupedResult:
    y_all = np.concatenate([y_c, y_t])
    x_all = np.concatenate([x_c, x_t])
    th = theta(y_all, x_all)
    xm = x_all.mean()
    raw = mean_diff(y_c, y_t, alpha)
    est = mean_diff(adjust(y_c, x_c, th, xm), adjust(y_t, x_t, th, xm), alpha)
    vr = 1.0 - (est.se ** 2) / (raw.se ** 2) if raw.se > 0 else 0.0
    return CupedResult(estimate=est, theta=th, variance_reduction=vr, raw=raw)
