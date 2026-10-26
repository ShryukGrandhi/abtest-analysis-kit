"""Monte Carlo accuracy benchmark and throughput measurement for abkit.

Every estimator is run many times on simulated experiments whose true effect is
known, so we can measure what matters for an analysis tool: false positive
rate, power, CI coverage and variance reduction. The last section measures
how fast the CSV -> per-user aggregation -> report pipeline runs.

    python bench/benchmark.py            # full run (numbers in README)
    python bench/benchmark.py --quick    # smaller run used in CI
"""

import argparse
import json
import os
import platform
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from abkit.bootstrap import bootstrap_diff  # noqa: E402
from abkit.checks import benjamini_hochberg, holm, srm_test  # noqa: E402
from abkit.cuped import cuped_diff  # noqa: E402
from abkit.data import load_events  # noqa: E402
from abkit.estimators import mean_diff  # noqa: E402
from abkit.report import analyze  # noqa: E402
from abkit.sequential import SequentialTest  # noqa: E402
from abkit.simulate import SimConfig, simulate_users, write_events_csv  # noqa: E402

ALPHA = 0.05


def wilson(k, n):
    """95% Wilson interval for a simulated rate, so readers see MC noise."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def rate(k, n):
    lo, hi = wilson(k, n)
    return {"rate": k / n, "ci95": [lo, hi], "sims": n}


def arms(u, metric, period="post"):
    v = u["variant"]
    x = u[f"{metric}_{period}"]
    return x[v == 0], x[v == 1]


def bench_type1_and_power(rng, sims, n_users, lift_mult):
    """A/A false positive rate and A/B power, raw z-test vs CUPED."""
    out = {}
    for label, lift in (("aa", None), ("ab", {"treatment": lift_mult})):
        cfg = SimConfig(n_users=n_users, conversion_lift=lift)
        hits = {m: {"raw": 0, "cuped": 0} for m in ("orders", "revenue")}
        vr = {m: [] for m in ("sessions", "orders", "revenue")}
        for _ in range(sims):
            u = simulate_users(cfg, rng)
            for m in ("sessions", "orders", "revenue"):
                yc, yt = arms(u, m)
                xc, xt = arms(u, m, "pre")
                r = cuped_diff(yc, xc, yt, xt)
                vr[m].append(r.variance_reduction)
                if m in hits:
                    hits[m]["raw"] += r.raw.p_value < ALPHA
                    hits[m]["cuped"] += r.estimate.p_value < ALPHA
        out[label] = {
            m: {k: rate(v, sims) for k, v in h.items()} for m, h in hits.items()
        }
        out[label]["cuped_variance_reduction_mean"] = {m: float(np.mean(v)) for m, v in vr.items()}
    return out


def bench_peeking(rng, sims, looks, batch):
    """Stopping at the first p < 0.05 over `looks` interim checks."""
    out = {}
    for label, effect in (("null", 0.0), ("effect_0.05sd", 0.05)):
        naive_fp = msprt_fp = 0
        stop_looks = []
        for _ in range(sims):
            st = SequentialTest(tau2=0.05 ** 2)
            naive = False
            stopped = None
            for k in range(looks):
                st.update(rng.normal(0, 1, batch), rng.normal(effect, 1, batch))
                naive |= st.naive_p_value() < ALPHA
                if stopped is None and st.rejected:
                    stopped = k + 1
            naive_fp += naive
            msprt_fp += stopped is not None
            if stopped is not None:
                stop_looks.append(stopped)
        fixed = 0
        # Reference: one fixed-horizon test at the final sample size.
        for _ in range(sims):
            n = looks * batch
            fixed += mean_diff(rng.normal(0, 1, n), rng.normal(effect, 1, n)).p_value < ALPHA
        out[label] = {
            "naive_peeking_reject": rate(naive_fp, sims),
            "msprt_reject": rate(msprt_fp, sims),
            "fixed_horizon_reject": rate(fixed, sims),
            "msprt_mean_stopping_look": float(np.mean(stop_looks)) if stop_looks else None,
        }
    out["looks"] = looks
    out["users_per_arm_per_look"] = batch
    return out


def bench_coverage(rng, sims, n_users, n_boot):
    """95% CI coverage of relative revenue lift: delta method vs Poisson bootstrap."""
    true_lift = 0.0  # A/A, so the true relative lift is exactly 0
    cfg = SimConfig(n_users=n_users)
    delta_cover = boot_cover = 0
    widths_d, widths_b = [], []
    for _ in range(sims):
        u = simulate_users(cfg, rng)
        c, t = arms(u, "revenue")
        e = mean_diff(c, t)
        delta_cover += e.rel_ci_low <= true_lift <= e.rel_ci_high
        widths_d.append(e.rel_ci_high - e.rel_ci_low)
        _, lo, hi, _ = bootstrap_diff(c, t, n_boot=n_boot, relative=True,
                                      seed=int(rng.integers(1 << 31)))
        boot_cover += lo <= true_lift <= hi
        widths_b.append(hi - lo)
    return {
        "users": n_users,
        "bootstrap_replicates": n_boot,
        "delta_method": {**rate(delta_cover, sims), "mean_width": float(np.mean(widths_d))},
        "poisson_bootstrap": {**rate(boot_cover, sims), "mean_width": float(np.mean(widths_b))},
    }


def bench_srm(rng, sims, n_users):
    out = {}
    for label, p in (("balanced_50_50", 0.5), ("skewed_50.5_49.5", 0.505)):
        flagged = 0
        for _ in range(sims):
            a = int(rng.binomial(n_users, p))
            flagged += srm_test({"a": a, "b": n_users - a}).mismatch
        out[label] = rate(flagged, sims)
    out["users"] = n_users
    out["threshold"] = 0.001
    return out


def bench_multiple_testing(rng, sims, n_metrics, n_true, n):
    """FDR / FWER over a family of metrics where only n_true have an effect."""
    false_disc_prop = {"none": [], "bh": [], "holm": []}
    any_false = {"none": 0, "bh": 0, "holm": 0}
    power = {"none": [], "bh": [], "holm": []}
    effects = np.zeros(n_metrics)
    effects[:n_true] = 0.12
    for _ in range(sims):
        p = np.array([mean_diff(rng.normal(0, 1, n), rng.normal(e, 1, n)).p_value
                      for e in effects])
        for name, adj in (("none", p), ("bh", benjamini_hochberg(p)), ("holm", holm(p))):
            rej = adj < ALPHA
            fd = int(rej[n_true:].sum())
            false_disc_prop[name].append(fd / max(int(rej.sum()), 1))
            any_false[name] += fd > 0
            power[name].append(rej[:n_true].mean())
    return {
        "metrics": n_metrics, "true_effects": n_true, "users_per_arm": n,
        **{name: {"fdr": float(np.mean(false_disc_prop[name])),
                  "fwer": rate(any_false[name], sims),
                  "power": float(np.mean(power[name]))} for name in any_false},
    }


def bench_throughput(n_users, repeats):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "events.csv")
        write_events_csv(path, SimConfig(n_users=n_users, conversion_lift={"treatment": 1.05}),
                         seed=1, dirty_fraction=0.01)
        size = os.path.getsize(path)
        with open(path) as fh:
            rows = sum(1 for _ in fh) - 1
        load_t, analyze_t = [], []
        for _ in range(repeats):
            t0 = time.perf_counter()
            table = load_events(path)
            t1 = time.perf_counter()
            analyze(table, "control", ratios=[("revenue", "sessions"), ("orders", "sessions")])
            t2 = time.perf_counter()
            load_t.append(t1 - t0)
            analyze_t.append(t2 - t1)
    best_load = min(load_t)
    return {
        "users": n_users, "rows": rows, "file_mb": size / 1e6, "repeats": repeats,
        "load_aggregate_seconds_best": best_load,
        "rows_per_second": rows / best_load,
        "analyze_seconds_best": min(analyze_t),
    }


def pct(x):
    return f"{x * 100:.1f}%"


def r(x):
    lo, hi = x["ci95"]
    return f"{pct(x['rate'])} ({pct(lo)}-{pct(hi)})"


def to_markdown(res):
    L = []
    t = res["type1_power"]
    L.append(f"### Fixed-horizon test, {res['config']['power_users']} users, "
             f"{res['config']['power_sims']} simulated experiments")
    L.append("")
    L.append(f"| metric | A/A false positive rate (raw) | A/A FPR (CUPED) | "
             f"power at +{(res['config']['power_lift'] - 1) * 100:.0f}% conversion (raw) | power (CUPED) |")
    L.append("|---|---|---|---|---|")
    for m in ("orders", "revenue"):
        L.append(f"| {m} | {r(t['aa'][m]['raw'])} | {r(t['aa'][m]['cuped'])} | "
                 f"{r(t['ab'][m]['raw'])} | {r(t['ab'][m]['cuped'])} |")
    vr = t["aa"]["cuped_variance_reduction_mean"]
    L.append("")
    L.append("Mean CUPED variance reduction (pre-period covariate): " +
             ", ".join(f"{m} {pct(v)}" for m, v in vr.items()))
    p = res["peeking"]
    L += ["", f"### Peeking: {p['looks']} interim looks, {p['users_per_arm_per_look']} users/arm per look, "
          f"{res['config']['peek_sims']} simulations", "",
          "| truth | naive z-test, stop at first p<0.05 | mSPRT always-valid p | single test at the end | mSPRT mean stopping look |",
          "|---|---|---|---|---|"]
    for k in ("null", "effect_0.05sd"):
        x = p[k]
        sl = f"{x['msprt_mean_stopping_look']:.1f}" if k != "null" else "-"
        L.append(f"| {k} | {r(x['naive_peeking_reject'])} | {r(x['msprt_reject'])} | "
                 f"{r(x['fixed_horizon_reject'])} | {sl} |")
    c = res["coverage"]
    L += ["", f"### 95% CI coverage for relative revenue lift (A/A, {c['users']} users, "
          f"{res['config']['coverage_sims']} simulations)", "",
          "| method | coverage | mean CI width |", "|---|---|---|",
          f"| delta method | {r(c['delta_method'])} | {pct(c['delta_method']['mean_width'])} |",
          f"| Poisson bootstrap ({c['bootstrap_replicates']} replicates) | {r(c['poisson_bootstrap'])} | "
          f"{pct(c['poisson_bootstrap']['mean_width'])} |"]
    s = res["srm"]
    L += ["", f"### Sample ratio mismatch check ({s['users']} users, p < {s['threshold']}, "
          f"{res['config']['srm_sims']} simulations)", "",
          f"- flagged when split is truly 50/50: {r(s['balanced_50_50'])}",
          f"- flagged when split is 50.5/49.5: {r(s['skewed_50.5_49.5'])}"]
    m = res["multiple_testing"]
    L += ["", f"### Multiple metrics: {m['metrics']} metrics, {m['true_effects']} with a real effect, "
          f"{res['config']['mt_sims']} simulations", "",
          "| correction | false discovery rate | P(any false positive) | power |", "|---|---|---|---|"]
    for k, name in (("none", "none"), ("bh", "Benjamini-Hochberg"), ("holm", "Holm")):
        L.append(f"| {name} | {pct(m[k]['fdr'])} | {r(m[k]['fwer'])} | {pct(m[k]['power'])} |")
    th = res["throughput"]
    L += ["", f"### Pipeline throughput ({th['users']} users, {th['rows']} event rows, "
          f"{th['file_mb']:.1f} MB CSV, best of {th['repeats']})", "",
          f"- CSV parse + validation + per-user aggregation: {th['load_aggregate_seconds_best']:.2f} s "
          f"({th['rows_per_second']:,.0f} rows/s)",
          f"- analysis of 5 metrics (CUPED, ratios, BH): {th['analyze_seconds_best'] * 1000:.1f} ms"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results.json"))
    args = ap.parse_args()
    q = args.quick
    cfg = {
        "power_sims": 200 if q else 1000, "power_users": 40000, "power_lift": 1.10,
        "peek_sims": 300 if q else 2000,
        "coverage_sims": 100 if q else 500, "coverage_users": 2000, "boot": 500 if q else 1000,
        "srm_sims": 1000 if q else 5000, "srm_users": 200000,
        "mt_sims": 200 if q else 1000,
        "tp_users": 20000 if q else 200000, "tp_repeats": 1 if q else 3,
    }
    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    res = {"config": cfg, "seed": args.seed, "quick": q,
           "python": platform.python_version(), "numpy": np.__version__,
           "machine": f"{platform.system()} {platform.machine()}"}
    res["type1_power"] = bench_type1_and_power(rng, cfg["power_sims"], cfg["power_users"],
                                                cfg["power_lift"])
    res["peeking"] = bench_peeking(rng, cfg["peek_sims"], looks=20, batch=500)
    res["coverage"] = bench_coverage(rng, cfg["coverage_sims"], cfg["coverage_users"], cfg["boot"])
    res["srm"] = bench_srm(rng, cfg["srm_sims"], cfg["srm_users"])
    res["multiple_testing"] = bench_multiple_testing(rng, cfg["mt_sims"], 20, 4, 1000)
    res["throughput"] = bench_throughput(cfg["tp_users"], cfg["tp_repeats"])
    res["wall_seconds"] = time.perf_counter() - t0
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(to_markdown(res))
    print(f"\nTotal benchmark time: {res['wall_seconds']:.1f} s "
          f"(python {res['python']}, numpy {res['numpy']}, {res['machine']})")


if __name__ == "__main__":
    main()
