from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import medfilt


def _nan_fill_linear(arr: np.ndarray) -> np.ndarray:
    out = arr.copy().astype(float)
    nan_mask = np.isnan(out)
    if not nan_mask.any():
        return out
    idx = np.arange(len(out))
    out[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], out[~nan_mask])
    return out


def _nan_ffill(arr: np.ndarray) -> np.ndarray:
    out = arr.copy().astype(float)
    for i in range(1, len(out)):
        if np.isnan(out[i]):
            out[i] = out[i - 1]
    for i in range(len(out) - 2, -1, -1):
        if np.isnan(out[i]):
            out[i] = out[i + 1]
    return out


def iqr_outlier(stream: np.ndarray, k: float = 3.0) -> dict:
    s = np.asarray(stream, dtype=float)
    valid = s[~np.isnan(s)]
    q25, q75 = float(np.percentile(valid, 25)), float(np.percentile(valid, 75))
    iqr_val = q75 - q25
    lower_fence, upper_fence = q25 - k * iqr_val, q75 + k * iqr_val

    s_filled = _nan_fill_linear(s)
    win = max(3, int(0.05 * len(s_filled)))
    if win % 2 == 0:
        win += 1
    win = min(win, len(s_filled) if len(s_filled) % 2 == 1 else len(s_filled) - 1)
    fitted = medfilt(s_filled, kernel_size=win)

    anomaly_mask = ((s < lower_fence) | (s > upper_fence)) & ~np.isnan(s)
    return {
        "fitted_signal": fitted,
        "anomaly_mask": anomaly_mask,
        "internal_metrics": {"q25": q25, "q75": q75, "iqr": float(iqr_val),
                             "lower_fence": float(lower_fence), "upper_fence": float(upper_fence),
                             "k": k, "n_outliers": int(anomaly_mask.sum())},
    }


def cusum(stream: np.ndarray, target: float = None, slack: float = 0.2,
          threshold: float = 3.0) -> dict:
    s = np.asarray(stream, dtype=float)
    s_filled = _nan_fill_linear(s)
    n = len(s_filled)

    warmup_n = max(5, int(0.20 * n))
    warmup = s_filled[:warmup_n]
    mu = float(np.mean(warmup)) if target is None else float(target)
    sigma = float(np.std(warmup, ddof=1)) if len(warmup) > 1 else 1.0
    global_std = float(np.nanstd(s_filled)) if n > 1 else 1.0
    sigma = max(sigma, global_std * 1e-6)

    k_val, h_val = slack * sigma, threshold * sigma
    cusum_pos, cusum_neg = np.zeros(n), np.zeros(n)
    for i in range(1, n):
        dev = s_filled[i] - mu
        cusum_pos[i] = max(0.0, cusum_pos[i - 1] + dev - k_val)
        cusum_neg[i] = max(0.0, cusum_neg[i - 1] - dev - k_val)

    anomaly_mask = (cusum_pos > h_val) | (cusum_neg > h_val)
    change_points = list(np.where(np.diff(anomaly_mask.astype(int)) == 1)[0] + 1)

    fitted = np.full(n, mu)
    seg_start = 0
    for bp in change_points + [n]:
        if bp > seg_start:
            fitted[seg_start:bp] = np.mean(s_filled[seg_start:bp])
        seg_start = bp

    return {
        "fitted_signal": fitted,
        "anomaly_mask": anomaly_mask,
        "internal_metrics": {"target_mean": mu, "sigma": sigma, "slack_k": float(k_val),
                             "threshold_h": float(h_val), "warmup_n": warmup_n,
                             "change_points": change_points,
                             "n_change_points": len(change_points)},
    }


def moving_average(stream: np.ndarray, window: int = 0) -> dict:
    s = np.asarray(stream, dtype=float)
    s_filled = _nan_fill_linear(s)
    if window <= 0:
        window = max(3, int(0.05 * len(s_filled)))
    kernel = np.ones(window) / window
    padded = np.pad(s_filled, window // 2, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")[: len(s)]
    residuals = s_filled - smoothed
    return {
        "fitted_signal": smoothed,
        "anomaly_mask": np.zeros(len(s), dtype=bool),
        "internal_metrics": {"window": window,
                             "residuals_std": float(np.std(residuals, ddof=1)),
                             "residuals_mean": float(np.mean(residuals))},
    }


def ewma(stream: np.ndarray, alpha: float = 0.1) -> dict:
    s = np.asarray(stream, dtype=float)
    n = len(s)
    smoothed = np.empty(n, dtype=float)
    first_valid = next((float(v) for v in s if not np.isnan(v)), 0.0)
    smoothed[0] = first_valid if np.isnan(s[0]) else float(s[0])
    for i in range(1, n):
        smoothed[i] = smoothed[i - 1] if np.isnan(s[i]) else alpha * s[i] + (1.0 - alpha) * smoothed[i - 1]
    residuals = np.where(np.isnan(s), np.nan, s - smoothed)
    return {
        "fitted_signal": smoothed,
        "anomaly_mask": np.zeros(n, dtype=bool),
        "internal_metrics": {"alpha": alpha, "final_ewma": float(smoothed[-1]),
                             "residuals_std": float(np.nanstd(residuals, ddof=1)),
                             "residuals_mean": float(np.nanmean(residuals))},
    }


def imputation(stream: np.ndarray, method: str = "spline") -> dict:
    s = np.asarray(stream, dtype=float)
    nan_mask = np.isnan(s)
    nan_idx = list(np.where(nan_mask)[0])
    n_nan = int(nan_mask.sum())

    if method == "spline":
        idx = np.arange(len(s))
        valid_idx, valid_vals = idx[~nan_mask], s[~nan_mask]
        if len(valid_idx) > 3:
            cs = CubicSpline(valid_idx, valid_vals, extrapolate=True)
            filled = s.copy()
            filled[nan_mask] = cs(idx[nan_mask])
        else:
            filled = _nan_fill_linear(s)
    elif method == "ffill":
        filled = _nan_ffill(s)
    elif method == "mean":
        filled = s.copy()
        filled[nan_mask] = float(np.nanmean(s))
    else:
        filled = _nan_fill_linear(s)

    return {
        "fitted_signal": filled,
        "anomaly_mask": nan_mask.copy(),
        "internal_metrics": {"n_imputed": n_nan, "nan_fraction": float(n_nan / len(s)),
                             "imputation_method": method, "nan_indices": nan_idx},
    }


def quality_check_stuck(stream: np.ndarray, stuck_threshold: int = 3,
                        nan_flag: bool = False) -> dict:
    s = np.asarray(stream, dtype=float)
    n = len(s)
    anomaly_mask = np.zeros(n, dtype=bool)
    nan_indices = list(np.where(np.isnan(s))[0])
    if nan_flag:
        anomaly_mask[np.isnan(s)] = True

    stuck_segments: list[tuple] = []
    max_run_length = 0
    i = 0
    while i < n - 1:
        if not np.isnan(s[i]) and s[i] == s[i + 1]:
            j = i + 1
            while j < n and s[j] == s[i]:
                j += 1
            run_len = j - i
            max_run_length = max(max_run_length, run_len)
            if run_len >= stuck_threshold:
                stuck_segments.append((int(i), int(j - 1), float(s[i]), run_len))
                anomaly_mask[i:j] = True
            i = j
        else:
            i += 1

    return {
        "fitted_signal": s.copy(),
        "anomaly_mask": anomaly_mask,
        "internal_metrics": {"nan_count": len(nan_indices), "stuck_segments": stuck_segments,
                             "stuck_count": len(stuck_segments), "max_run_length": max_run_length,
                             "max_run_ratio": float(max_run_length / n) if n > 0 else 0.0,
                             "quality_score": float(1.0 - anomaly_mask.sum() / n)},
    }


def hypothesis_test(stream: np.ndarray, alpha: float = 0.05) -> dict:
    from scipy.stats import ttest_ind
    s = np.asarray(stream, dtype=float)
    n = len(s)
    half = n // 2
    a, b = s[:half][~np.isnan(s[:half])], s[half:][~np.isnan(s[half:])]
    if len(a) < 2 or len(b) < 2:
        return {"fitted_signal": s.copy(), "anomaly_mask": np.zeros(n, dtype=bool),
                "internal_metrics": {"p_value": 1.0}}
    stat, p_value = ttest_ind(a, b, equal_var=False)
    fitted = np.concatenate([np.full(half, a.mean()), np.full(n - half, b.mean())])
    anomaly_mask = np.zeros(n, dtype=bool)
    if p_value < alpha:
        anomaly_mask[half:] = ~np.isnan(s[half:])
    return {
        "fitted_signal": fitted,
        "anomaly_mask": anomaly_mask,
        "internal_metrics": {"t_stat": float(stat), "p_value": float(p_value), "alpha": alpha},
    }


def goodness_of_fit(stream: np.ndarray, alpha: float = 0.05) -> dict:
    from scipy.stats import kstest
    s = np.asarray(stream, dtype=float)
    valid = s[~np.isnan(s)]
    mu = float(np.mean(valid)) if len(valid) else 0.0
    sigma = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
    if sigma <= 0:
        return {"fitted_signal": np.full(len(s), mu), "anomaly_mask": np.zeros(len(s), dtype=bool),
                "internal_metrics": {"p_value": 0.0}}
    stat, p_value = kstest(valid, "norm", args=(mu, sigma))
    return {
        "fitted_signal": np.full(len(s), mu),
        "anomaly_mask": np.zeros(len(s), dtype=bool),
        "internal_metrics": {"ks_stat": float(stat), "p_value": float(p_value),
                             "fits_normal": bool(p_value >= alpha)},
    }


def fft_analysis(stream: np.ndarray, n_components: int = 2,
                 threshold_factor: float = 3.0) -> dict:
    s = np.asarray(stream, dtype=float)
    s_filled = _nan_fill_linear(s)
    n = len(s_filled)
    dc_mean = float(np.mean(s_filled))

    fft_vals = np.fft.rfft(s_filled - dc_mean)
    fft_freqs = np.fft.rfftfreq(n)
    dominant_bins = np.argsort(np.abs(fft_vals)[1:])[::-1][:n_components] + 1

    fft_rec = np.zeros_like(fft_vals)
    fft_rec[0] = fft_vals[0]
    for idx in dominant_bins:
        fft_rec[idx] = fft_vals[idx]
    fitted = np.fft.irfft(fft_rec, n=n) + dc_mean

    residuals = np.where(np.isnan(s), np.nan, s_filled - fitted)
    resid_std = float(np.nanstd(residuals, ddof=1))
    anom_thresh = threshold_factor * max(resid_std, float(np.nanstd(s_filled)) * 1e-6)
    anomaly_mask = (np.abs(residuals) > anom_thresh) & ~np.isnan(s)
    return {
        "fitted_signal": fitted,
        "anomaly_mask": anomaly_mask,
        "internal_metrics": {"dominant_frequencies_hz": [float(fft_freqs[b]) for b in dominant_bins],
                             "dominant_amplitudes": [float(2.0 * np.abs(fft_vals[b]) / n) for b in dominant_bins],
                             "n_components": n_components, "dc_mean": dc_mean,
                             "reconstruction_error_std": resid_std,
                             "anomaly_threshold": float(anom_thresh),
                             "n_anomalies": int(anomaly_mask.sum())},
    }


def z_score_burst(stream: np.ndarray, threshold: float = 2.5) -> dict:
    s = np.asarray(stream, dtype=float)
    valid = s[~np.isnan(s)]
    mu = float(np.mean(valid)) if len(valid) > 0 else 0.0
    sigma = float(np.std(valid, ddof=1)) if len(valid) > 1 else 1.0
    data_range = float(np.nanmax(valid) - np.nanmin(valid)) if len(valid) >= 2 else 1.0
    sigma = max(sigma, max(data_range, 1e-30) * 1e-6)

    z_scores = (s - mu) / sigma
    anomaly_mask = (np.abs(z_scores) > threshold) & ~np.isnan(s)
    return {
        "fitted_signal": s.copy(),
        "anomaly_mask": anomaly_mask,
        "internal_metrics": {"mean": mu, "std": sigma, "threshold": threshold,
                             "n_anomalies": int(anomaly_mask.sum())},
    }


TOOL_REGISTRY: dict[str, callable] = {
    "Z-Score":         z_score_burst,
    "IQR":             iqr_outlier,
    "Moving Average":  moving_average,
    "EWMA":            ewma,
    "CUSUM":           cusum,
    "Imputation":      imputation,
    "FFT":             fft_analysis,
    "Hypothesis Test": hypothesis_test,
    "Quality Check":   quality_check_stuck,
    "Goodness-of-Fit": goodness_of_fit,
}
