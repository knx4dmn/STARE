from __future__ import annotations

import math
from typing import Iterable

import pandas as pd


class EdgeDataProfiler:
    def __init__(self) -> None:
        from datasketch import HyperLogLog
        from runstats import Statistics
        from tdigest import TDigest

        self._welford = Statistics()
        self._digest = TDigest()
        self._hll = HyperLogLog(p=12)
        self._n_total = 0
        self._n_missing = 0

    def update(self, value: float) -> None:
        self._n_total += 1
        if value is None or (isinstance(value, float) and math.isnan(value)):
            self._n_missing += 1
            return
        self._welford.push(value)
        self._digest.update(value)
        self._hll.update(str(round(value, 4)).encode())

    def update_batch(self, values: Iterable[float]) -> None:
        for v in values:
            self.update(v)

    def get_profile(self) -> dict:
        n = len(self._welford)
        mean = self._welford.mean() if n > 0 else float("nan")
        var = self._welford.variance() if n > 1 else float("nan")
        skew = self._welford.skewness() if n > 2 and var > 0 else float("nan")
        kurt = self._welford.kurtosis() if n > 3 and var > 0 else float("nan")
        std = math.sqrt(var) if var == var and var > 0 else 0.0

        q1, q3 = (self._digest.percentile(25), self._digest.percentile(75)) if n > 0 else (0.0, 0.0)
        iqr = q3 - q1
        if iqr > 0:
            n_outliers = n * (self._digest.cdf(q1 - 3 * iqr) + 1.0 - self._digest.cdf(q3 + 3 * iqr))
        else:
            n_outliers = 0.0
        cardinality = self._hll.count() if n > 0 else 0.0
        missing_ratio = self._n_missing / self._n_total if self._n_total else 0.0

        shape = _shape(skew)
        noise = _noise(mean, std)
        outlier = _outlier(n_outliers, n)
        missing = _missing(missing_ratio)

        constraints = ["univariate"]
        if shape != "symmetric" and shape != "unknown":
            constraints.append("asymmetric_distribution")
        if noise == "high":
            constraints.append("high_variance_stream")
        if outlier == "severe":
            constraints.append("extreme_outliers_present")
        if kurt == kurt and kurt > 3.0:
            constraints.append("heavy_tailed")
        if n > 0 and cardinality > 0.8 * n:
            constraints.append("high_cardinality")
        if n > 0 and cardinality < 0.05 * n:
            constraints.append("low_cardinality")
        if self._n_missing > 0:
            constraints.append("contains_missing")

        return {"distribution_shape": shape, "noise_level": noise, "outlier_severity": outlier,
                "missing_data": missing, "constraints": constraints}


def _shape(skew: float) -> str:
    if skew != skew:
        return "unknown"
    if abs(skew) < 0.5:
        return "symmetric"
    return "right-skewed" if skew > 0 else "left-skewed"


def _noise(mean: float, std: float) -> str:
    if mean != mean or mean == 0:
        return "high"
    cv = std / abs(mean)
    if cv < 0.05:
        return "low"
    if cv < 0.20:
        return "moderate"
    return "high"


def _outlier(n_outliers: float, n: int) -> str:
    if n == 0 or n_outliers < 1:
        return "none"
    return "mild" if n_outliers / n < 0.01 else "severe"


def _missing(ratio: float) -> str:
    if ratio == 0.0:
        return "complete"
    return "minor_gaps" if ratio < 0.05 else "severe_loss"


def profile_stream(series: pd.Series) -> dict:
    profiler = EdgeDataProfiler()
    profiler.update_batch(series)
    return profiler.get_profile()


_SHAPE_PHRASE = {
    "symmetric": "symmetric distribution", "right-skewed": "right-skewed distribution",
    "left-skewed": "left-skewed distribution", "unknown": "unknown distribution shape",
}
_NOISE_PHRASE   = {"low": "low noise", "moderate": "moderate noise", "high": "high noise"}
_OUTLIER_PHRASE = {"none": "no outliers", "mild": "mild outliers", "severe": "severe outliers"}
_MISSING_PHRASE = {
    "complete": "complete data", "minor_gaps": "minor gaps and sparse missing values",
    "severe_loss": "severe missing data loss",
}


def format_query(profile: dict, task_category: str) -> str:
    constraints = [c.replace("_", " ") for c in profile["constraints"]]
    return (f"Broad Category: {task_category}. "
            f"Stream with {_SHAPE_PHRASE[profile['distribution_shape']]}, "
            f"{_NOISE_PHRASE[profile['noise_level']]}, {_OUTLIER_PHRASE[profile['outlier_severity']]}, "
            f"{_MISSING_PHRASE[profile['missing_data']]}. "
            f"Constraints: {', '.join(constraints)}.")
