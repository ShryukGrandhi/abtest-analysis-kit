"""Experiment health checks and multiple-testing corrections."""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from .distributions import chi2_sf


@dataclass
class SrmResult:
    observed: Dict[str, int]
    expected: Dict[str, float]
    chi2: float
    p_value: float
    mismatch: bool


def srm_test(counts: Dict[str, int], weights: Optional[Dict[str, float]] = None,
             threshold=0.001) -> SrmResult:
    """Sample ratio mismatch: chi-square goodness of fit of arm sizes.

    A very small p-value means the assignment or logging is broken and the
    metric results should not be trusted. The conventional threshold is 0.001
    because the test is run on every experiment.
    """
    names = list(counts)
    if len(names) < 2:
        raise ValueError("need at least two variants")
    if weights is None:
        weights = {k: 1.0 for k in names}
    w = np.array([float(weights[k]) for k in names])
    w = w / w.sum()
    obs = np.array([counts[k] for k in names], dtype=float)
    exp = obs.sum() * w
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    p = chi2_sf(chi2, len(names) - 1)
    return SrmResult(
        observed=dict(counts),
        expected={k: float(e) for k, e in zip(names, exp)},
        chi2=chi2, p_value=p, mismatch=p < threshold,
    )


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """BH-adjusted p-values (q-values); controls the false discovery rate."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    return out


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values; controls the family-wise error rate."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * (m - np.arange(m))
    ranked = np.maximum.accumulate(ranked)
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    return out
