"""Poisson bootstrap confidence intervals.

Instead of resampling indices with replacement (multinomial weights), each
observation gets an independent Poisson(1) weight per replicate. The two are
asymptotically equivalent, but Poisson weights need no knowledge of n and can
be generated per row, which is why this variant is used for streaming and
sharded data. Replicates are processed in chunks to bound memory.
"""

import numpy as np


def _weighted_means(x, n_boot, rng, chunk):
    x = np.asarray(x, dtype=float)
    out = np.empty(n_boot)
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        w = rng.poisson(1.0, size=(b, x.size)).astype(float)
        wsum = w.sum(axis=1)
        # A replicate with all-zero weights is vanishingly rare for n > 20;
        # fall back to the sample mean rather than divide by zero.
        wsum[wsum == 0] = np.nan
        means = (w @ x) / wsum
        means[np.isnan(means)] = x.mean()
        out[done:done + b] = means
        done += b
    return out


def bootstrap_diff(control, treatment, n_boot=2000, alpha=0.05, seed=None,
                   relative=False, chunk_bytes=32 << 20):
    """Percentile CI for mean(treatment) - mean(control) (or relative lift).

    Returns (point_estimate, ci_low, ci_high, replicates).
    """
    rng = np.random.default_rng(seed)
    c = np.asarray(control, dtype=float)
    t = np.asarray(treatment, dtype=float)
    chunk = max(1, int(chunk_bytes // (8 * max(c.size, t.size))))
    mc = _weighted_means(c, n_boot, rng, chunk)
    mt = _weighted_means(t, n_boot, rng, chunk)
    if relative:
        reps = mt / mc - 1.0
        point = t.mean() / c.mean() - 1.0
    else:
        reps = mt - mc
        point = t.mean() - c.mean()
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi), reps
