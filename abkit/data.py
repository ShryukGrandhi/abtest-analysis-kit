"""Load event-level experiment logs and aggregate them to one row per user.

Input is a long-format CSV with the header::

    user_id,variant,period,metric,value

``period`` is one of:

* ``assign`` - the user was assigned to ``variant`` (metric/value ignored)
* ``pre``    - a metric event from before the experiment started (used by CUPED)
* ``post``   - a metric event during the experiment

Every assigned user is part of the analysis population even if they have no
metric events (their metric totals are 0). Rows that cannot be used are
counted in :class:`QualityReport` rather than silently dropped.
"""

import csv
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

import numpy as np

REQUIRED_COLUMNS = ("user_id", "variant", "period", "metric", "value")
PERIODS = ("assign", "pre", "post")


@dataclass
class QualityReport:
    rows_read: int = 0
    rows_used: int = 0
    malformed_rows: int = 0
    unknown_period_rows: int = 0
    unassigned_user_rows: int = 0
    variant_mismatch_rows: int = 0
    multi_assigned_users: int = 0

    def issues(self) -> Dict[str, int]:
        keys = ("malformed_rows", "unknown_period_rows", "unassigned_user_rows",
                "variant_mismatch_rows", "multi_assigned_users")
        return {k: getattr(self, k) for k in keys if getattr(self, k)}


@dataclass
class UnitTable:
    """Per-user aggregates. Arrays are aligned on ``user_ids``."""

    user_ids: List[str]
    variants: List[str]                # variant names, index = code
    variant_codes: np.ndarray          # int code per user
    post: Dict[str, np.ndarray]        # metric -> per-user sum during experiment
    pre: Dict[str, np.ndarray]         # metric -> per-user sum before experiment
    quality: QualityReport = field(default_factory=QualityReport)

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    def counts(self) -> Dict[str, int]:
        c = np.bincount(self.variant_codes, minlength=len(self.variants))
        return {v: int(c[i]) for i, v in enumerate(self.variants)}

    def metric(self, name: str, variant: str, period: str = "post") -> np.ndarray:
        table = self.post if period == "post" else self.pre
        if name not in table:
            raise KeyError(f"no {period} metric named {name!r}")
        code = self.variants.index(variant)
        return table[name][self.variant_codes == code]


def _iter_csv(path) -> Iterable[List[str]]:
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{path}: empty file")
        header = [h.strip() for h in header]
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        idx = [header.index(c) for c in REQUIRED_COLUMNS]
        width = max(idx) + 1
        for row in reader:
            if len(row) < width:
                yield None
            else:
                yield [row[i] for i in idx]


def aggregate_rows(rows: Iterable[List[str]]) -> UnitTable:
    """Aggregate parsed rows ``[user_id, variant, period, metric, value]``.

    Two passes over the materialised rows: the first resolves assignments, the
    second accumulates metric sums with ``np.add.at`` per (period, metric).
    """
    q = QualityReport()
    assignment: Dict[str, str] = {}
    conflicted = set()
    events = []

    for row in rows:
        q.rows_read += 1
        if row is None:
            q.malformed_rows += 1
            continue
        user, variant, period, metric, value = row
        if not user or not variant:
            q.malformed_rows += 1
            continue
        if period == "assign":
            prev = assignment.get(user)
            if prev is None:
                assignment[user] = variant
            elif prev != variant:
                conflicted.add(user)
            q.rows_used += 1
            continue
        if period not in ("pre", "post"):
            q.unknown_period_rows += 1
            continue
        try:
            val = float(value)
        except ValueError:
            q.malformed_rows += 1
            continue
        if not metric or not math.isfinite(val):
            q.malformed_rows += 1
            continue
        events.append((user, variant, period, metric, val))

    # Users assigned to more than one variant are excluded entirely: their
    # outcome cannot be attributed to a single arm.
    for user in conflicted:
        del assignment[user]
    q.multi_assigned_users = len(conflicted)

    user_ids = sorted(assignment)
    user_index = {u: i for i, u in enumerate(user_ids)}
    variants = sorted(set(assignment.values()))
    variant_index = {v: i for i, v in enumerate(variants)}
    codes = np.fromiter((variant_index[assignment[u]] for u in user_ids),
                        dtype=np.int64, count=len(user_ids))

    buckets: Dict[tuple, tuple] = {}
    for user, variant, period, metric, val in events:
        i = user_index.get(user)
        if i is None:
            if user in conflicted:
                q.variant_mismatch_rows += 1
            else:
                q.unassigned_user_rows += 1
            continue
        if assignment[user] != variant:
            q.variant_mismatch_rows += 1
            continue
        key = (period, metric)
        b = buckets.get(key)
        if b is None:
            b = ([], [])
            buckets[key] = b
        b[0].append(i)
        b[1].append(val)
        q.rows_used += 1

    post: Dict[str, np.ndarray] = {}
    pre: Dict[str, np.ndarray] = {}
    for (period, metric), (idx, vals) in buckets.items():
        arr = np.zeros(len(user_ids))
        np.add.at(arr, np.asarray(idx, dtype=np.int64), np.asarray(vals))
        (post if period == "post" else pre)[metric] = arr
    # A metric observed in only one period still gets a zero array in the other
    # so downstream code can rely on aligned keys for CUPED.
    for metric in list(pre):
        post.setdefault(metric, np.zeros(len(user_ids)))

    return UnitTable(user_ids, variants, codes, post, pre, q)


def load_events(path) -> UnitTable:
    return aggregate_rows(_iter_csv(path))
