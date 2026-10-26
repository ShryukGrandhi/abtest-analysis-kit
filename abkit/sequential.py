"""Sequential testing with the mixture SPRT (Johari, Pekelis, Walsh 2017).

Checking a fixed-horizon p-value every day and stopping at the first p < 0.05
("peeking") inflates the false positive rate well above alpha. The mSPRT gives
an *always-valid* p-value: it may be checked after every batch and the
probability of ever crossing alpha under the null stays <= alpha.

For a difference in means with estimated variance V = s_c^2/n_c + s_t^2/n_t
and a N(0, tau^2) mixing prior on the effect, the likelihood ratio is

    Lambda = sqrt(V / (V + tau^2)) * exp(tau^2 * d^2 / (2 V (V + tau^2)))

and the always-valid p-value is p_n = min(p_{n-1}, 1 / Lambda_n).
"""

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np

from .distributions import two_sided_p


def msprt_lambda(diff, var, tau2):
    if var <= 0:
        return math.inf if diff != 0 else 1.0
    log_l = 0.5 * math.log(var / (var + tau2)) + tau2 * diff * diff / (2 * var * (var + tau2))
    return math.exp(min(log_l, 700.0))


@dataclass
class _Arm:
    n: int = 0
    total: float = 0.0
    total_sq: float = 0.0

    def add(self, x):
        x = np.asarray(x, dtype=float)
        self.n += x.size
        self.total += float(x.sum())
        self.total_sq += float((x * x).sum())

    @property
    def mean(self):
        return self.total / self.n

    @property
    def var(self):
        if self.n < 2:
            return float("nan")
        m = self.mean
        return max((self.total_sq - self.n * m * m) / (self.n - 1), 0.0)


@dataclass
class SequentialTest:
    """Streaming two-arm test. Feed batches with :meth:`update`.

    ``tau2`` is the variance of the mixing prior on the effect size in the
    metric's units; a reasonable default is the square of the smallest effect
    worth detecting.
    """

    tau2: float
    alpha: float = 0.05
    control: _Arm = field(default_factory=_Arm)
    treatment: _Arm = field(default_factory=_Arm)
    p_value: float = 1.0
    history: List[float] = field(default_factory=list)

    def update(self, control_batch, treatment_batch):
        self.control.add(control_batch)
        self.treatment.add(treatment_batch)
        if self.control.n < 2 or self.treatment.n < 2:
            self.history.append(self.p_value)
            return self.p_value
        diff = self.treatment.mean - self.control.mean
        var = self.control.var / self.control.n + self.treatment.var / self.treatment.n
        lam = msprt_lambda(diff, var, self.tau2)
        self.p_value = min(self.p_value, 1.0 / lam)
        self.history.append(self.p_value)
        return self.p_value

    @property
    def rejected(self):
        return self.p_value <= self.alpha

    def naive_p_value(self):
        """Fixed-horizon z-test p-value at the current look (invalid if peeked)."""
        diff = self.treatment.mean - self.control.mean
        se = math.sqrt(self.control.var / self.control.n + self.treatment.var / self.treatment.n)
        return two_sided_p(diff / se) if se > 0 else 1.0
