"""Full experiment analysis: health checks, per-metric results, FDR correction."""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .checks import benjamini_hochberg, srm_test
from .cuped import cuped_diff
from .data import UnitTable
from .estimators import mean_diff, ratio_diff


@dataclass
class MetricResult:
    metric: str
    kind: str                       # "mean" or "ratio"
    variant: str
    control: float
    treatment: float
    diff: float
    ci_low: float
    ci_high: float
    rel_lift: float
    rel_ci_low: float
    rel_ci_high: float
    p_value: float
    cuped: bool = False
    variance_reduction: float = 0.0
    q_value: float = float("nan")


@dataclass
class Analysis:
    control: str
    counts: Dict[str, int]
    srm_p: float
    srm_mismatch: bool
    quality: Dict[str, int]
    results: List[MetricResult] = field(default_factory=list)

    def as_dict(self):
        return {
            "control": self.control,
            "counts": self.counts,
            "srm_p_value": self.srm_p,
            "srm_mismatch": self.srm_mismatch,
            "data_quality_issues": self.quality,
            "results": [r.__dict__ for r in self.results],
        }


def analyze(table: UnitTable, control: str, metrics: Optional[Sequence[str]] = None,
            ratios: Sequence[Tuple[str, str]] = (), weights=None, alpha=0.05,
            use_cuped=True) -> Analysis:
    if control not in table.variants:
        raise ValueError(f"control {control!r} not in variants {table.variants}")
    counts = table.counts()
    srm = srm_test(counts, weights)
    metrics = sorted(table.post) if metrics is None else list(metrics)
    out = Analysis(control, counts, srm.p_value, srm.mismatch, table.quality.issues())

    for variant in table.variants:
        if variant == control:
            continue
        for m in metrics:
            yc, yt = table.metric(m, control), table.metric(m, variant)
            if use_cuped and m in table.pre:
                cr = cuped_diff(yc, table.metric(m, control, "pre"),
                                yt, table.metric(m, variant, "pre"), alpha)
                e, vr, adj = cr.estimate, cr.variance_reduction, True
                # Report raw arm means; the adjusted means differ by a constant.
                cm, tm = cr.raw.control, cr.raw.treatment
            else:
                e = mean_diff(yc, yt, alpha)
                vr, adj = 0.0, False
                cm, tm = e.control, e.treatment
            out.results.append(MetricResult(
                m, "mean", variant, cm, tm, e.diff, e.ci_low, e.ci_high,
                e.rel_lift, e.rel_ci_low, e.rel_ci_high, e.p_value, adj, vr))
        for num, den in ratios:
            e = ratio_diff(table.metric(num, control), table.metric(den, control),
                           table.metric(num, variant), table.metric(den, variant), alpha)
            out.results.append(MetricResult(
                f"{num}/{den}", "ratio", variant, e.control, e.treatment, e.diff,
                e.ci_low, e.ci_high, e.rel_lift, e.rel_ci_low, e.rel_ci_high, e.p_value))

    q = benjamini_hochberg([r.p_value for r in out.results])
    for r, qv in zip(out.results, q):
        r.q_value = float(qv)
    return out


def _fmt(x, pct=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x * 100:+.2f}%" if pct else f"{x:.4g}"


def to_markdown(a: Analysis) -> str:
    lines = [f"Control: {a.control}", ""]
    lines.append("Users per variant: " + ", ".join(f"{k}={v}" for k, v in a.counts.items()))
    flag = "MISMATCH - do not trust results" if a.srm_mismatch else "ok"
    lines.append(f"Sample ratio check: p={a.srm_p:.4g} ({flag})")
    if a.quality:
        lines.append("Data quality issues: " + ", ".join(f"{k}={v}" for k, v in a.quality.items()))
    lines += ["", "| variant | metric | control | treatment | lift | lift 95% CI | p | q (BH) | CUPED var. red. |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in a.results:
        vr = f"{r.variance_reduction * 100:.1f}%" if r.cuped else "-"
        lines.append(
            f"| {r.variant} | {r.metric} | {_fmt(r.control)} | {_fmt(r.treatment)} | "
            f"{_fmt(r.rel_lift, True)} | [{_fmt(r.rel_ci_low, True)}, {_fmt(r.rel_ci_high, True)}] | "
            f"{r.p_value:.4g} | {r.q_value:.4g} | {vr} |")
    return "\n".join(lines)
