"""Synthetic experiment data with known ground truth.

Used by the tests and the benchmark: because the true effect is known, we can
measure false positive rate, power and CI coverage of every estimator.

Per user the model is

    sessions_pre  ~ Poisson(lambda_u),  sessions_post ~ Poisson(lambda_u)
    lambda_u      ~ Gamma(shape, mean / shape)       (heterogeneous activity)
    p_u           ~ Beta with mean convert_p          (heterogeneous propensity)
    orders        ~ Binomial(sessions, p_u * lift[variant])
    revenue       = sum of orders x LogNormal(spend_mu + s_u, spend_sigma)

so pre and post metrics are correlated through the user-level lambda_u, p_u
and s_u, which is what CUPED exploits, and revenue is zero-inflated and
right-skewed like real revenue.
"""

import csv
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class SimConfig:
    n_users: int = 20000
    variants: tuple = ("control", "treatment")
    weights: Optional[tuple] = None          # actual assignment probabilities
    session_mean: float = 3.0
    session_shape: float = 2.0
    convert_p: float = 0.08
    convert_concentration: float = 2.0       # Beta a+b; lower = more spread
    spend_mu: float = 3.0                    # log-scale mean of order value
    spend_user_sd: float = 0.5               # per-user shift of log order value
    spend_sigma: float = 0.6
    # multiplicative effect on conversion probability per variant (1.0 = none)
    conversion_lift: Optional[Dict[str, float]] = None


def simulate_users(cfg: SimConfig, rng: np.random.Generator):
    """Return dict of per-user arrays: variant code, sessions/orders/revenue pre and post."""
    k = len(cfg.variants)
    w = np.full(k, 1.0 / k) if cfg.weights is None else np.asarray(cfg.weights, float)
    w = w / w.sum()
    codes = rng.choice(k, size=cfg.n_users, p=w)
    lam = rng.gamma(cfg.session_shape, cfg.session_mean / cfg.session_shape, cfg.n_users)
    a = cfg.convert_p * cfg.convert_concentration
    p_user = rng.beta(a, cfg.convert_concentration - a, cfg.n_users)
    spend_shift = rng.normal(0.0, cfg.spend_user_sd, cfg.n_users)
    lift = np.ones(k)
    if cfg.conversion_lift:
        for name, m in cfg.conversion_lift.items():
            lift[cfg.variants.index(name)] = m
    out = {"variant": codes}
    for period, mult in (("pre", np.ones(k)), ("post", lift)):
        sessions = rng.poisson(lam)
        p = np.clip(p_user * mult[codes], 0, 1)
        orders = rng.binomial(sessions, p)
        # Sum of `orders` lognormal draws per user, vectorised via repeat + bincount.
        owner = np.repeat(np.arange(cfg.n_users), orders)
        spend = rng.lognormal(cfg.spend_mu + spend_shift[owner], cfg.spend_sigma)
        revenue = np.bincount(owner, weights=spend, minlength=cfg.n_users)
        out[f"sessions_{period}"] = sessions.astype(float)
        out[f"orders_{period}"] = orders.astype(float)
        out[f"revenue_{period}"] = revenue
    return out


def write_events_csv(path, cfg: SimConfig, seed=0, dirty_fraction=0.0):
    """Write a long-format event log (see abkit.data) and return the user dict.

    Each metric value is split into one row per session for sessions and
    orders and one row per order for revenue, so row counts look like a real
    event stream. ``dirty_fraction`` injects malformed/unassigned/mismatched
    rows to exercise the data-quality checks.
    """
    rng = np.random.default_rng(seed)
    users = simulate_users(cfg, rng)
    names = cfg.variants
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["user_id", "variant", "period", "metric", "value"])
        for i in range(cfg.n_users):
            uid = f"u{i:07d}"
            v = names[users["variant"][i]]
            wr.writerow([uid, v, "assign", "", ""])
            for period in ("pre", "post"):
                for _ in range(int(users[f"sessions_{period}"][i])):
                    wr.writerow([uid, v, period, "sessions", 1])
                rev = users[f"revenue_{period}"][i]
                n_orders = int(users[f"orders_{period}"][i])
                for _ in range(n_orders):
                    wr.writerow([uid, v, period, "orders", 1])
                if n_orders:
                    # Split revenue across orders; the per-user sum is exact.
                    parts = rng.dirichlet(np.ones(n_orders)) * rev
                    for amt in parts:
                        wr.writerow([uid, v, period, "revenue", f"{amt:.6f}"])
        if dirty_fraction > 0:
            n_dirty = int(cfg.n_users * dirty_fraction)
            for j in range(n_dirty):
                kind = j % 3
                if kind == 0:
                    wr.writerow([f"u{j:07d}", names[0], "post"])            # too short
                elif kind == 1:
                    wr.writerow([f"ghost{j}", names[0], "post", "revenue", 5])  # never assigned
                else:
                    uid = f"u{j:07d}"
                    other = names[(users["variant"][j] + 1) % len(names)]
                    wr.writerow([uid, other, "post", "revenue", 5])        # wrong arm
    return users
