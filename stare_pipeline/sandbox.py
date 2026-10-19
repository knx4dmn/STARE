from __future__ import annotations

import numpy as np

from .tools import TOOL_REGISTRY

LABELED_TASKS = {"anomaly_detection", "burst_detection", "change_point_detection"}

TASK_TO_TOOLS: dict[str, list[str]] = {
    "anomaly_detection":       ["IQR", "CUSUM", "Z-Score"],
    "burst_detection":         ["Z-Score", "IQR"],
    "change_point_detection":  ["CUSUM", "Hypothesis Test", "IQR", "Z-Score"],
    "forecasting":             ["Moving Average", "EWMA"],
    "spectral_analysis":       ["FFT", "Moving Average", "EWMA"],
    "data_imputation":         ["Imputation", "Quality Check"],
    "data_quality_assessment": ["Quality Check", "Goodness-of-Fit", "Imputation"],
}


class SandboxEvaluator:
    def __init__(self, alpha: float = 0.5, holdout_frac: float = 0.20, mask_frac: float = 0.15,
                 stuck_frac: float = 0.20, spectral_noise_scale: float = 2.0, seed: int = 42):
        self.alpha = alpha
        self.holdout_frac = holdout_frac
        self.mask_frac = mask_frac
        self.stuck_frac = stuck_frac
        self.spectral_noise_scale = spectral_noise_scale
        self.seed = seed

    def evaluate(self, stream: np.ndarray, task: str, labels: np.ndarray | None = None,
                 tool_names: list[str] | None = None) -> dict[str, float]:
        data = np.asarray(stream, dtype=float).ravel()
        tools = {name: TOOL_REGISTRY[name] for name in (tool_names or TASK_TO_TOOLS[task])}
        rng = np.random.default_rng(self.seed)

        if task in LABELED_TASKS:
            if labels is None:
                raise ValueError(f"{task} fitness requires benchmark labels.")
            return self._labeled_detection(data, np.asarray(labels, dtype=bool), tools)
        if task == "forecasting":
            return self._holdout_forecast(data, tools)
        if task == "spectral_analysis":
            return self._spectral_fit(data, tools, rng)
        if task == "data_imputation":
            return self._masked_imputation(data, tools)
        if task == "data_quality_assessment":
            return self._stuck_block(data, tools)
        raise ValueError(f"Unknown task {task!r}.")

    def _labeled_detection(self, data, labels, tools):
        return {name: self._recall_minus_fpr(_run(fn, data).get("anomaly_mask"), labels)
                for name, fn in tools.items()}

    def _holdout_forecast(self, data, tools):
        n = len(data)
        split = max(2, int((1.0 - self.holdout_frac) * n))
        actual = data[split:]
        masked = data.copy()
        masked[split:] = np.nan
        var = float(np.nanvar(actual)) + 1e-12
        scores = {}
        for name, fn in tools.items():
            fitted = np.asarray(_run(fn, masked).get("fitted_signal", []), dtype=float)
            scores[name] = _exp_mse(actual, fitted[split:] if len(fitted) >= n else np.array([]), var)
        return scores

    def _spectral_fit(self, data, tools, rng):
        valid = data[~np.isnan(data)]
        var = float(np.var(valid)) + 1e-12
        noisy = data + rng.standard_normal(len(data)) * self.spectral_noise_scale * float(np.std(valid))
        return {name: _exp_mse(data, np.asarray(_run(fn, noisy).get("fitted_signal", []), dtype=float), var)
                for name, fn in tools.items()}

    def _masked_imputation(self, data, tools):
        block = max(5, int(self.mask_frac * len(data)))
        start = _find_clean_block(data, block)
        if start is None:
            return {name: 0.0 for name in tools}
        original = data[start:start + block].copy()
        masked = data.copy()
        masked[start:start + block] = np.nan
        var = float(np.var(original)) + 1e-12
        scores = {}
        for name, fn in tools.items():
            fitted = np.asarray(_run(fn, masked).get("fitted_signal", []), dtype=float)
            scores[name] = _exp_mse(original, fitted[start:start + block] if len(fitted) >= start + block
                                    else np.array([]), var)
        return scores

    def _stuck_block(self, data, tools):
        block = max(5, int(self.stuck_frac * len(data)))
        start = _find_clean_block(data, block)
        if start is None:
            return {name: 0.0 for name in tools}
        stuck = data.copy()
        stuck[start:start + block] = float(np.nanmedian(data))
        labels = np.zeros(len(data), dtype=bool)
        labels[start:start + block] = True
        return {name: self._recall_minus_fpr(_run(fn, stuck).get("anomaly_mask"), labels)
                for name, fn in tools.items()}

    def _recall_minus_fpr(self, pred, labels) -> float:
        if pred is None or labels.sum() == 0:
            return 0.0
        pred = np.asarray(pred, dtype=bool)
        recall = float((pred & labels).sum() / labels.sum())
        n_neg = int((~labels).sum())
        fpr = float((pred & ~labels).sum() / n_neg) if n_neg else 0.0
        return max(0.0, recall - self.alpha * fpr)


def _run(fn, data) -> dict:
    try:
        return fn(data)
    except Exception:
        return {}


def _exp_mse(actual, predicted, var) -> float:
    k = min(len(actual), len(predicted))
    if k == 0:
        return 0.0
    valid = ~np.isnan(actual[:k]) & ~np.isnan(predicted[:k])
    if valid.sum() == 0:
        return 0.0
    mse = float(np.mean((actual[:k][valid] - predicted[:k][valid]) ** 2))
    return float(np.exp(-mse / var))


def _find_clean_block(data, block) -> int | None:
    margin = max(5, int(0.05 * len(data)))
    run = 0
    for i in range(margin, len(data)):
        run = 0 if np.isnan(data[i]) else run + 1
        if run >= block:
            return i - block + 1
    return None
