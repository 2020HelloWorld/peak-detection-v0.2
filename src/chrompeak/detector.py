#!/usr/bin/env python3
"""Conservative chromatogram peak detector trained from the supplied references.

The detector learns retention-time, width, and symmetry envelopes from the five
reference chromatograms, preprocesses each curve for impulses, baseline drift,
and noise, then classifies features as confirmed template peaks, negative peaks,
overlapping/peak-on-hump cases, broad humps, electrical spikes, or uncertain
noise/interference.

Chemical component names are intentionally external configuration: the supplied
CSV files do not contain a channel/component-to-retention-time mapping.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
VENDOR = PROJECT_ROOT / ".vendor"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pybaselines import Baseline
from scipy.cluster.hierarchy import fclusterdata
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, peak_widths, savgol_filter

from .core import (
    Curve,
    EPS,
    local_noise_sigma,
    odd_at_most,
    read_curves,
    symmetric_baseline,
)

from . import __version__


@dataclass
class Preprocessed:
    signal_range: float
    impulse_clean: np.ndarray
    impulse_mask: np.ndarray
    baseline: np.ndarray
    hump_baseline: np.ndarray
    diagnostic_baseline: np.ndarray
    positive: np.ndarray
    hump_positive: np.ndarray
    signed_baseline: np.ndarray
    signed: np.ndarray
    noise_positive: np.ndarray
    noise_hump: np.ndarray
    noise_signed: np.ndarray
    structured_noise_positive: np.ndarray
    baseline_model_uncertainty: np.ndarray
    baseline_model_disagreement: np.ndarray
    bump_mask: np.ndarray
    global_noise: float
    baseline_drift: float
    endpoint_drift_ratio: float
    baseline_excursion_ratio: float
    baseline_curvature: float
    bump_duration: float
    bump_flag: bool
    bump_strength: str
    bump_is_open: bool
    bump_count: int


@dataclass(frozen=True)
class DetectorConfig:
    """User-visible thresholds for candidate filtering and confirmation."""

    confirmation_threshold: float = 0.75
    artifact_threshold: float = 0.45
    normal_relative_prominence_floor: float = 0.0008
    high_noise_relative_prominence_floor: float = 0.01
    negative_bilateral_depth_floor: float = 0.005
    negative_signed_depth_floor: float = 0.008
    high_noise_negative_bilateral_floor: float = 0.01
    high_noise_negative_signed_floor: float = 0.02
    hump_relative_prominence_floor: float = 0.005
    hump_weak_density_limit: int = 2
    hump_density_radius_min: float = 0.50
    hump_bilateral_depth_floor: float = 0.0015
    hump_candidate_width_limit_min: float = 0.80
    electrical_spike_score_threshold: float = 0.45
    apex_spike_peak_body_floor: float = 0.72
    apex_spike_confidence_multiplier: float = 0.90

    @classmethod
    def from_json(cls, path: Path | None) -> "DetectorConfig":
        if path is None or not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        allowed = cls.__dataclass_fields__.keys()
        unknown = sorted(set(raw) - set(allowed))
        if unknown:
            raise ValueError(f"Unknown detector config fields: {unknown}")
        config = cls(**raw)
        if not 0 < config.artifact_threshold < config.confirmation_threshold < 1:
            raise ValueError("Require 0 < artifact_threshold < confirmation_threshold < 1")
        if (
            config.hump_relative_prominence_floor <= 0
            or config.hump_weak_density_limit < 2
            or config.hump_density_radius_min <= 0
            or config.hump_bilateral_depth_floor <= 0
            or config.hump_candidate_width_limit_min <= 0
        ):
            raise ValueError("Hump residual thresholds must be positive and non-trivial")
        if not (
            0 < config.electrical_spike_score_threshold < 1
            and 0 < config.apex_spike_peak_body_floor < 1
            and 0 < config.apex_spike_confidence_multiplier <= 1
        ):
            raise ValueError("Electrical-interference score thresholds must be in (0, 1]")
        return config


FEATURE_COLUMNS = [
    "folder",
    "file",
    "sha256",
    "feature_id",
    "feature_type",
    "feature_type_cn",
    "status",
    "status_cn",
    "sign",
    "template_slot",
    "component",
    "apex_time_min",
    "corrected_apex_value",
    "raw_apex_value",
    "start_time_min",
    "end_time_min",
    "width_min",
    "fwhm_min",
    "width_samples",
    "fwhm_samples",
    "top_width_ratio",
    "rise_time_min",
    "fall_time_min",
    "rise_continuity",
    "fall_continuity",
    "normalized_max_slope",
    "max_step_ratio",
    "slope_asymmetry",
    "top3_energy_fraction",
    "apex_core_area_fraction",
    "area_shape_factor",
    "apex_spike_flag",
    "impulse_overlap",
    "apex_impulse_ratio",
    "apex_quadratic_excess_ratio",
    "despiked_height_retention",
    "despiked_prominence_retention",
    "despiked_area_retention",
    "peak_body_survives_despike",
    "file_impulse_density_per_min",
    "electrical_assessment_applicable",
    "peak_body_score",
    "electrical_interference_score",
    "height",
    "prominence",
    "relative_prominence",
    "snr",
    "random_snr",
    "random_noise",
    "structured_noise",
    "baseline_model_uncertainty",
    "baseline_model_disagreement",
    "effective_noise",
    "width_to_fwhm",
    "apex_in_bump",
    "bump_overlap_fraction",
    "candidate_density",
    "weak_candidate_density",
    "structured_background_residual_flag",
    "ripple_group",
    "area",
    "symmetry",
    "bilateral_depth",
    "bilateral_depth_relative",
    "signed_depth_relative",
    "baseline_change_ratio",
    "edge_distance_min",
    "width_ratio_to_template",
    "rt_error_min",
    "peak_confidence",
    "peak_confidence_percent",
    "template_confidence",
    "template_confidence_percent",
    "score_snr",
    "score_prominence",
    "score_width",
    "score_symmetry",
    "score_nonflat",
    "score_signed_depth",
    "score_bilateral_depth",
    "confirmation_threshold",
    "confidence",
    "reasons",
]


FEATURE_TYPE_CN = {
    "normal_positive_peak": "普通正峰",
    "narrow_positive_peak": "窄正峰",
    "narrow_peak_or_interference": "窄峰或电干扰",
    "unassigned_positive_peak": "保留时间未映射的正峰",
    "positive_peak_on_hump": "鼓包或漂移背景上的正峰",
    "overlapping_positive_peak": "重叠正峰",
    "secondary_or_overlapping_candidate": "同一窗口的次峰或重叠候选",
    "broad_or_overlapped_peak": "宽峰或未分离重叠峰",
    "broad_positive_peak": "宽正峰",
    "broad_hump_or_baseline": "宽鼓包、宽峰或基线",
    "gentle_broad_peak_candidate": "平缓宽峰候选",
    "peak_with_apex_spike": "带峰顶尖点的色谱峰",
    "electrical_interference_candidate": "电信号干扰候选",
    "electrical_spike": "正向毛刺/电尖峰",
    "negative_electrical_spike": "负向毛刺/电尖峰",
    "negative_peak": "负峰",
    "broad_negative_peak": "宽负峰",
    "interpeak_valley_or_negative_peak": "峰间谷底或负峰",
    "uncertain_peak_or_noise": "峰或噪声（待复核）",
    "structured_background_residual": "结构化背景残差",
}

STATUS_CN = {"confirmed": "确认", "review": "待复核", "artifact": "干扰/伪峰"}


def independent_curves(curves: list[Curve]) -> list[Curve]:
    seen: set[str] = set()
    result: list[Curve] = []
    for curve in curves:
        if curve.sha256 in seen:
            continue
        seen.add(curve.sha256)
        result.append(curve)
    return result


def configure_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def robust_sigma(values: np.ndarray) -> float:
    center = float(np.median(values))
    sigma = 1.4826 * float(np.median(np.abs(values - center)))
    return max(sigma, EPS)


def local_sigma(residual: np.ndarray, dt_min: float, signal_range: float) -> np.ndarray:
    window = odd_at_most(round(0.40 / dt_min), len(residual))
    center = median_filter(residual, size=window, mode="nearest")
    mad = median_filter(np.abs(residual - center), size=window, mode="nearest")
    sigma = 1.4826 * mad
    global_value = robust_sigma(residual)
    floor = max(global_value * 0.5, signal_range * 1e-6, EPS)
    return np.maximum(sigma, floor)


def _contiguous_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive index intervals for every True run in a boolean mask."""
    if not mask.any():
        return []
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def baseline_diagnostics(
    x: np.ndarray,
    y: np.ndarray,
    main_baseline: np.ndarray,
    rolling_baseline: np.ndarray,
    signed_baseline: np.ndarray,
) -> dict:
    """Measure endpoint drift and non-linear, time-local baseline excursions.

    ``baseline_curvature`` is deliberately a robust global curvature proxy:
    the 1--99% spread around the line joining the two endpoint levels. Direct
    second derivatives were rejected because they are sampling-rate and
    smoothing sensitive.
    """
    dt = float(np.median(np.diff(x)))
    duration = max(float(x[-1] - x[0]), dt)
    signal_range = max(float(np.quantile(y, 0.999) - np.quantile(y, 0.001)), EPS)
    diagnostic = np.median(
        np.vstack([main_baseline, rolling_baseline, signed_baseline]), axis=0
    )
    smooth_window = odd_at_most(round(0.20 / dt), len(x), minimum=5)
    if smooth_window >= 5:
        diagnostic = savgol_filter(diagnostic, smooth_window, 2, mode="interp")

    endpoint_window_min = float(np.clip(0.02 * duration, 0.10, 0.25))
    start_mask = x <= x[0] + endpoint_window_min
    end_mask = x >= x[-1] - endpoint_window_min
    start_level = float(np.median(diagnostic[start_mask]))
    end_level = float(np.median(diagnostic[end_mask]))
    endpoint_drift_ratio = (end_level - start_level) / signal_range
    baseline_excursion_ratio = float(np.ptp(diagnostic) / signal_range)
    endpoint_line = start_level + (end_level - start_level) * (
        (x - x[0]) / duration
    )
    nonlinear = diagnostic - endpoint_line
    baseline_curvature = float(
        (np.quantile(nonlinear, 0.99) - np.quantile(nonlinear, 0.01))
        / signal_range
    )

    intervals: list[tuple[int, int]] = []
    retained_prominence_ratios: list[float] = []
    minimum_width_samples = max(3, round(0.50 / dt))
    distance_samples = max(3, round(0.30 / dt))
    # A single global 3%-of-range threshold misses real low-amplitude humps in
    # files that also contain a very large analyte peak. Search down to 0.5%
    # of the robust range, then reject weak baseline candidates whose interval
    # is visibly contaminated by a strong narrow analyte event. Candidates
    # above the original 3% floor remain eligible because large humps can carry
    # real peaks on top of them (A4/A5/A11).
    bump_indices, bump_properties = find_peaks(
        diagnostic,
        prominence=0.005 * signal_range,
        width=(minimum_width_samples, None),
        distance=distance_samples,
    )
    if len(bump_indices):
        _, _, left90, right90 = peak_widths(
            diagnostic, bump_indices, rel_height=0.90
        )
        for candidate_index, (apex, left, right) in enumerate(
            zip(bump_indices, left90, right90)
        ):
            left_index = max(0, int(math.floor(left)))
            right_index = min(len(x) - 1, int(math.ceil(right)))
            prominence_ratio = float(
                bump_properties["prominences"][candidate_index] / signal_range
            )
            analyte_excess = np.abs(
                y[left_index : right_index + 1]
                - diagnostic[left_index : right_index + 1]
            ) / signal_range
            apex_excess_ratio = abs(float(y[apex] - diagnostic[apex])) / signal_range
            analyte_overlap_fraction = float(np.mean(analyte_excess > 0.03))
            weak_candidate_polluted = bool(
                prominence_ratio < 0.03
                and (
                    apex_excess_ratio > 0.03
                    or analyte_overlap_fraction > 0.10
                )
            )
            if weak_candidate_polluted:
                continue
            intervals.append((left_index, right_index))
            retained_prominence_ratios.append(prominence_ratio)

    bump_is_open = False
    if not intervals:
        positive_nonlinear = np.maximum(nonlinear, 0.0)
        maximum = float(np.max(positive_nonlinear))
        if maximum / signal_range >= 0.035:
            threshold = max(0.01 * signal_range, 0.10 * maximum)
            open_mask = positive_nonlinear >= threshold
            open_intervals = [
                (start, end)
                for start, end in _contiguous_intervals(open_mask)
                if x[end] - x[start] >= 0.50
            ]
            if open_intervals:
                intervals.extend(open_intervals)
                bump_is_open = True

    intervals.sort()
    merged: list[list[int]] = []
    maximum_gap_samples = max(1, round(0.20 / dt))
    for start, end in intervals:
        if merged and start - merged[-1][1] - 1 <= maximum_gap_samples:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    candidate_duration = max(
        (float(x[end] - x[start]) for start, end in merged), default=0.0
    )
    bump_flag = bool(merged and candidate_duration >= 0.50)
    strong_bump_flag = bool(
        bump_flag
        and (
            (
                baseline_excursion_ratio >= 0.05
                and baseline_curvature >= 0.04
            )
            or any(value >= 0.03 for value in retained_prominence_ratios)
        )
    )
    bump_strength = "strong" if strong_bump_flag else "candidate" if bump_flag else "none"
    bump_duration = candidate_duration if bump_flag else 0.0
    bump_mask = np.zeros(len(x), dtype=bool)
    if bump_flag:
        for start, end in merged:
            bump_mask[start : end + 1] = True

    return {
        "signal_range": signal_range,
        "diagnostic_baseline": diagnostic,
        "endpoint_drift_ratio": float(endpoint_drift_ratio),
        "baseline_excursion_ratio": baseline_excursion_ratio,
        "baseline_curvature": baseline_curvature,
        "bump_duration": bump_duration,
        "bump_flag": bump_flag,
        "bump_strength": bump_strength,
        "bump_is_open": bool(bump_is_open and bump_flag),
        "bump_count": len(merged) if bump_flag else 0,
        "bump_mask": bump_mask,
    }


def structured_background_sigma(
    signal: np.ndarray, dt_min: float, signal_range: float
) -> np.ndarray:
    """Estimate smooth ripple/model-error scale without erasing candidates."""
    trend_window = odd_at_most(round(0.20 / dt_min), len(signal), minimum=5)
    local_trend = median_filter(signal, size=trend_window, mode="nearest")
    return local_sigma(signal - local_trend, dt_min, signal_range)


def _integral_between(
    x: np.ndarray,
    y: np.ndarray,
    left: float,
    right: float,
) -> float:
    """Integrate a sampled curve on physical-time bounds.

    Boundary interpolation makes this metric depend on minutes rather than on
    an arbitrary number of samples.  That is important when chromatograms from
    different instruments use different acquisition rates.
    """

    if len(x) < 2:
        return 0.0
    left = max(float(left), float(x[0]))
    right = min(float(right), float(x[-1]))
    if right <= left:
        return 0.0
    interior = (x > left) & (x < right)
    bounded_x = np.concatenate(([left], x[interior], [right]))
    bounded_y = np.concatenate(
        (
            [float(np.interp(left, x, y))],
            y[interior],
            [float(np.interp(right, x, y))],
        )
    )
    return float(np.trapezoid(bounded_y, bounded_x))


def _quadratic_apex_excess_ratio(
    x: np.ndarray,
    raw_branch: np.ndarray,
    provisional_impulse_mask: np.ndarray,
    anomaly_index: int,
    prominence: float,
    fwhm_min: float,
) -> float:
    """Measure a point/cusp above the locally predicted chromatographic body.

    The three-point median is deliberately only a *provisional* despiking
    operation: a normal narrow Gaussian also differs from that median at its
    apex.  Here the candidate point is excluded, the surrounding signal is fit
    on a physical-time window, and only the unexplained positive excess is
    reported.  A coarse-sampling fallback additionally requires an isolated
    slope reversal, preventing a five-sample real peak from being called a
    one-sample impulse merely because it has few samples.
    """

    center_time = float(x[anomaly_index])
    half_window = max(0.010, 0.25 * float(fwhm_min))
    fit_mask = (
        (np.abs(x - center_time) <= half_window)
        & ~provisional_impulse_mask
    )
    fit_mask[anomaly_index] = False
    fit_indices = np.flatnonzero(fit_mask)
    predicted: float | None = None
    if len(fit_indices) >= 4:
        offsets = x[fit_indices] - center_time
        try:
            coefficients = np.polyfit(offsets, raw_branch[fit_indices], 2)
            predicted = float(np.polyval(coefficients, 0.0))
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            predicted = None

    if predicted is None:
        if anomaly_index < 2 or anomaly_index + 2 >= len(raw_branch):
            return 0.0
        predicted = 0.5 * float(
            raw_branch[anomaly_index - 1] + raw_branch[anomaly_index + 1]
        )
        center_excess = max(0.0, float(raw_branch[anomaly_index]) - predicted)
        shoulder_step = max(
            abs(
                float(
                    raw_branch[anomaly_index - 1]
                    - raw_branch[anomaly_index - 2]
                )
            ),
            abs(
                float(
                    raw_branch[anomaly_index + 1]
                    - raw_branch[anomaly_index + 2]
                )
            ),
            EPS,
        )
        if center_excess / shoulder_step < 0.65:
            return 0.0

    unexplained_excess = max(
        0.0, float(raw_branch[anomaly_index]) - float(predicted)
    )
    return unexplained_excess / max(float(prominence), EPS)


def preprocess(curve: Curve) -> Preprocessed:
    x, y = curve.x, curve.y
    dt = float(np.median(np.diff(x)))
    signal_range = max(float(np.quantile(y, 0.999) - np.quantile(y, 0.001)), EPS)

    # Remove only isolated one-sample impulses. Wider events are never erased;
    # they remain available to the peak/interference classifier.
    median3 = median_filter(y, size=3, mode="nearest")
    residual3 = y - median3
    impulse_threshold = max(8 * robust_sigma(residual3), 0.004 * signal_range)
    impulse_mask = np.abs(residual3) > impulse_threshold
    impulse_clean = y.copy()
    impulse_clean[impulse_mask] = median3[impulse_mask]

    # This slow, sign-symmetric estimate is not pulled preferentially toward a
    # negative event. It is also the baseline for the negative-peak branch.
    signed_baseline = symmetric_baseline(y, dt)

    fitter = Baseline(x_data=x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # arPLS is the main baseline because it follows slow drift without
        # swallowing the supplied broad chromatographic peaks. rolling-ball is
        # retained as a second, more local background model for peak-on-hump
        # evidence.
        arpls_baseline, _ = fitter.arpls(impulse_clean, lam=1e7, max_iter=50)
        rolling_baseline, _ = fitter.rolling_ball(
            impulse_clean,
            half_window=max(20, round(0.50 / dt)),
            smooth_half_window=max(2, round(0.025 / dt)),
        )
    # A strong negative peak can drag lower-envelope algorithms down for
    # minutes. The upper envelope prevents that failure while arPLS continues
    # to handle ordinary slow drift in the positive-peak cases.
    baseline = np.maximum(arpls_baseline, signed_baseline)
    # Keep the local rolling-ball branch genuinely independent. It is used as
    # evidence and uncertainty, not as the primary positive-peak baseline.
    hump_baseline = rolling_baseline
    positive_raw = impulse_clean - baseline
    hump_raw = impulse_clean - hump_baseline
    smooth_window = odd_at_most(round(0.015 / dt), len(y), minimum=5)
    positive = savgol_filter(positive_raw, smooth_window, 3, mode="interp")
    hump_positive = savgol_filter(hump_raw, smooth_window, 3, mode="interp")

    # A sign-symmetric branch preserves legitimate negative peaks.
    signed_raw = y - signed_baseline
    signed = savgol_filter(signed_raw, smooth_window, 3, mode="interp")

    # Use both short-scale smoothing residuals and robust first differences.
    # The maximum avoids unrealistically huge SNR values on smooth references
    # and adapts to noisy sections of a trace.
    noise_positive = np.maximum(
        local_sigma(positive_raw - positive, dt, signal_range),
        local_noise_sigma(positive_raw, dt),
    )
    noise_hump = np.maximum(
        local_sigma(hump_raw - hump_positive, dt, signal_range),
        local_noise_sigma(hump_raw, dt),
    )
    noise_signed = np.maximum(
        local_sigma(signed_raw - signed, dt, signal_range),
        local_noise_sigma(signed_raw, dt),
    )
    structured_noise_positive = structured_background_sigma(
        positive, dt, signal_range
    )
    baseline_model_disagreement = np.abs(baseline - hump_baseline)
    baseline_model_uncertainty = local_sigma(
        baseline - hump_baseline, dt, signal_range
    )
    diagnostics = baseline_diagnostics(
        x,
        y,
        baseline,
        hump_baseline,
        signed_baseline,
    )
    # Compare noise between files only after normalizing by that file's useful
    # signal range. Absolute detector units vary by orders of magnitude here.
    global_noise = float(np.median(noise_positive) / signal_range)
    baseline_drift = float(baseline[-1] - baseline[0])
    return Preprocessed(
        signal_range=diagnostics["signal_range"],
        impulse_clean=impulse_clean,
        impulse_mask=impulse_mask,
        baseline=baseline,
        hump_baseline=hump_baseline,
        diagnostic_baseline=diagnostics["diagnostic_baseline"],
        positive=positive,
        hump_positive=hump_positive,
        signed_baseline=signed_baseline,
        signed=signed,
        noise_positive=noise_positive,
        noise_hump=noise_hump,
        noise_signed=noise_signed,
        structured_noise_positive=structured_noise_positive,
        baseline_model_uncertainty=baseline_model_uncertainty,
        baseline_model_disagreement=baseline_model_disagreement,
        bump_mask=diagnostics["bump_mask"],
        global_noise=global_noise,
        baseline_drift=baseline_drift,
        endpoint_drift_ratio=diagnostics["endpoint_drift_ratio"],
        baseline_excursion_ratio=diagnostics["baseline_excursion_ratio"],
        baseline_curvature=diagnostics["baseline_curvature"],
        bump_duration=diagnostics["bump_duration"],
        bump_flag=diagnostics["bump_flag"],
        bump_strength=diagnostics["bump_strength"],
        bump_is_open=diagnostics["bump_is_open"],
        bump_count=diagnostics["bump_count"],
    )


def candidate_features(
    curve: Curve,
    processed: Preprocessed,
    sign: int,
    min_relative_prominence: float = 0.0005,
) -> list[dict]:
    x = curve.x
    work = processed.positive if sign > 0 else -processed.signed
    if sign > 0:
        raw_branch = curve.y - processed.baseline
        despiked_branch = processed.impulse_clean - processed.baseline
    else:
        raw_branch = -(curve.y - processed.signed_baseline)
        despiked_branch = raw_branch.copy()
    random_noise_array = (
        processed.noise_positive if sign > 0 else processed.noise_signed
    )
    dt = float(np.median(np.diff(x)))
    duration = max(float(x[-1] - x[0]), dt)
    file_impulse_density_per_min = float(
        len(_contiguous_intervals(processed.impulse_mask)) / duration
    )
    signal_range = max(float(np.quantile(curve.y, 0.999) - np.quantile(curve.y, 0.001)), EPS)
    raw_window = odd_at_most(round(0.015 / dt), len(curve.y), minimum=5)
    raw_smooth = savgol_filter(curve.y, raw_window, 3, mode="interp")
    raw_branch_smooth = savgol_filter(
        raw_branch, raw_window, 3, mode="interp"
    )
    repaired_branch_smooth = work
    minimum_prominence = np.maximum(
        3.0 * random_noise_array, min_relative_prominence * signal_range
    )
    wlen = odd_at_most(min(round(2.0 / dt), len(work) - 1), len(work))
    indices, props = find_peaks(
        work,
        height=np.maximum(
            2.0 * random_noise_array, min_relative_prominence * signal_range
        ),
        prominence=minimum_prominence,
        width=(3, None),
        distance=max(3, round(0.008 / dt)),
        wlen=wlen,
    )
    if not len(indices):
        return []
    widths95, _, left95, right95 = peak_widths(work, indices, rel_height=0.95)
    widths50, _, left50, right50 = peak_widths(work, indices, rel_height=0.50)
    widths10, _, _, _ = peak_widths(work, indices, rel_height=0.10)
    features: list[dict] = []
    for i, apex in enumerate(indices):
        li = max(0, int(math.floor(left95[i])))
        ri = min(len(x) - 1, int(math.ceil(right95[i])))
        left_half = max(float(apex - left50[i]), EPS)
        right_half = max(float(right50[i] - apex), EPS)
        symmetry = min(left_half, right_half) / max(left_half, right_half)
        start = float(x[li])
        end = float(x[ri])
        peak_signal = work[li : ri + 1]
        peak_x = x[li : ri + 1]
        area = float(np.trapezoid(np.maximum(peak_signal, 0), peak_x)) if len(peak_x) > 1 else 0.0
        prominence = float(props["prominences"][i])
        width_samples = float(widths95[i])
        fwhm_samples = float(widths50[i])
        width_min = float(width_samples * dt)
        fwhm_min = float(fwhm_samples * dt)
        rise_time_min = float(left_half * dt)
        fall_time_min = float(right_half * dt)
        apex_neighborhood = np.abs(x - float(x[apex])) <= 0.005 + EPS
        provisional_indices = np.flatnonzero(
            processed.impulse_mask & apex_neighborhood
        )
        impulse_overlap = bool(len(provisional_indices))
        if impulse_overlap:
            repair_magnitude = np.abs(curve.y - processed.impulse_clean)
            anomaly_index = int(
                provisional_indices[
                    np.argmax(repair_magnitude[provisional_indices])
                ]
            )
            impulse_amplitude = float(repair_magnitude[anomaly_index])
            apex_quadratic_excess_ratio = (
                _quadratic_apex_excess_ratio(
                    x,
                    raw_branch,
                    processed.impulse_mask,
                    anomaly_index,
                    prominence,
                    fwhm_min,
                )
                if sign > 0
                else 0.0
            )
        else:
            impulse_amplitude = 0.0
            apex_quadratic_excess_ratio = 0.0
        apex_impulse_ratio = impulse_amplitude / max(prominence, EPS)
        apex_spike_flag = bool(
            sign > 0
            and impulse_overlap
            and apex_quadratic_excess_ratio >= 0.015
        )

        if sign > 0:
            # Counterfactual retention compares like with like: both branches
            # use the same smoothing, interval, base rule, and area rule.
            raw_region = raw_branch_smooth[li : ri + 1]
            despiked_region = repaired_branch_smooth[li : ri + 1]
            raw_apex_local = int(np.argmax(raw_region))
            despiked_apex_local = int(np.argmax(despiked_region))
            raw_height = max(float(raw_region[raw_apex_local]), EPS)
            despiked_height = max(
                0.0, float(despiked_region[despiked_apex_local])
            )
            raw_prominence = max(
                raw_height
                - max(
                    float(np.min(raw_region[: raw_apex_local + 1])),
                    float(np.min(raw_region[raw_apex_local:])),
                ),
                EPS,
            )
            despiked_prominence = max(
                despiked_height
                - max(
                    float(
                        np.min(
                            despiked_region[: despiked_apex_local + 1]
                        )
                    ),
                    float(np.min(despiked_region[despiked_apex_local:])),
                ),
                0.0,
            )
            raw_area = (
                float(np.trapezoid(np.maximum(raw_region, 0), peak_x))
                if len(peak_x) > 1
                else 0.0
            )
            despiked_height_retention = clip01(despiked_height / raw_height)
            despiked_prominence_retention = clip01(
                despiked_prominence / raw_prominence
            )
            despiked_area_retention = clip01(area / max(raw_area, EPS))
        else:
            despiked_height_retention = float("nan")
            despiked_prominence_retention = float("nan")
            despiked_area_retention = float("nan")

        positive_samples = np.maximum(peak_signal, 0)
        top_count = min(3, len(positive_samples))
        top3_energy_fraction = (
            float(np.sort(positive_samples)[-top_count:].sum())
            / max(float(positive_samples.sum()), EPS)
            if top_count
            else 1.0
        )
        apex_core_area_fraction = clip01(
            _integral_between(
                peak_x,
                positive_samples,
                float(x[apex]) - 0.005,
                float(x[apex]) + 0.005,
            )
            / max(area, EPS)
        )
        area_shape_factor = area / max(prominence * float(widths50[i] * dt), EPS)

        left_slope_edge = max(li, int(math.floor(left50[i])))
        right_slope_edge = min(len(x) - 1, int(math.ceil(right50[i])))
        slope_steps = np.diff(
            repaired_branch_smooth[left_slope_edge : right_slope_edge + 1]
        )
        max_step_ratio = (
            float(np.max(np.abs(slope_steps))) / max(prominence, EPS)
            if len(slope_steps)
            else 0.0
        )
        normalized_max_slope = max_step_ratio / max(dt, EPS)
        rise_steps = np.diff(repaired_branch_smooth[li : apex + 1])
        fall_steps = -np.diff(repaired_branch_smooth[apex : ri + 1])
        monotonic_tolerance = 0.01 * prominence
        rise_continuity = (
            float(np.mean(rise_steps >= -monotonic_tolerance))
            if len(rise_steps)
            else 0.0
        )
        fall_continuity = (
            float(np.mean(fall_steps >= -monotonic_tolerance))
            if len(fall_steps)
            else 0.0
        )
        rise_rate = (
            float(np.quantile(np.maximum(rise_steps, 0), 0.75)) / max(dt, EPS)
            if len(rise_steps)
            else 0.0
        )
        fall_rate = (
            float(np.quantile(np.maximum(fall_steps, 0), 0.75)) / max(dt, EPS)
            if len(fall_steps)
            else 0.0
        )
        slope_asymmetry = (
            min(rise_rate, fall_rate) / max(rise_rate, fall_rate, EPS)
        )
        baseline_local = processed.hump_baseline[li : ri + 1]
        baseline_change_ratio = (
            float(np.ptp(baseline_local)) / max(float(work[apex]), EPS)
            if len(baseline_local)
            else 0.0
        )
        relative_prominence = prominence / signal_range
        random_noise = float(random_noise_array[apex])
        if sign > 0:
            structured_noise = float(processed.structured_noise_positive[apex])
            model_uncertainty = float(processed.baseline_model_uncertainty[apex])
            model_disagreement = float(processed.baseline_model_disagreement[apex])
            # Strong events are not allowed to inflate their own local
            # structured-noise estimate. The conservative multi-scale floor is
            # activated for weak events, where smooth hump ripple is the known
            # false-positive mode.
            effective_noise = (
                max(random_noise, structured_noise, model_uncertainty)
                if relative_prominence < 0.003
                else random_noise
            )
        else:
            structured_noise = 0.0
            model_uncertainty = 0.0
            model_disagreement = 0.0
            effective_noise = random_noise
        width_to_fwhm = float(widths95[i] / max(widths50[i], EPS))
        bump_overlap_fraction = float(
            np.mean(processed.bump_mask[li : ri + 1])
        )
        # A genuine valley must be lower than the raw signal on both sides.
        # This rejects the negative side-lobes produced when a positive peak is
        # removed by a symmetric baseline.
        shoulder_points = max(4, round(0.04 / dt))
        left_edge = max(1, int(math.floor(left50[i])))
        right_edge = min(len(x) - 2, int(math.ceil(right50[i])))
        left_shoulder = raw_smooth[max(0, left_edge - shoulder_points) : left_edge]
        right_shoulder = raw_smooth[
            right_edge + 1 : min(len(x), right_edge + 1 + shoulder_points)
        ]
        if len(left_shoulder) and len(right_shoulder):
            if sign < 0:
                left_depth = float(np.median(left_shoulder) - raw_smooth[apex])
                right_depth = float(np.median(right_shoulder) - raw_smooth[apex])
            else:
                left_depth = float(raw_smooth[apex] - np.median(left_shoulder))
                right_depth = float(raw_smooth[apex] - np.median(right_shoulder))
            bilateral_depth = max(0.0, min(left_depth, right_depth))
        else:
            bilateral_depth = 0.0
        features.append(
            {
                "sign": "positive" if sign > 0 else "negative",
                "apex_index": int(apex),
                "apex_time_min": float(x[apex]),
                "start_time_min": start,
                "end_time_min": end,
                "width_min": width_min,
                "fwhm_min": fwhm_min,
                "width_samples": width_samples,
                "fwhm_samples": fwhm_samples,
                "width_to_fwhm": width_to_fwhm,
                "top_width_ratio": float(widths10[i] / max(widths50[i], EPS)),
                "rise_time_min": rise_time_min,
                "fall_time_min": fall_time_min,
                "rise_continuity": rise_continuity,
                "fall_continuity": fall_continuity,
                "normalized_max_slope": normalized_max_slope,
                "max_step_ratio": max_step_ratio,
                "slope_asymmetry": slope_asymmetry,
                "top3_energy_fraction": top3_energy_fraction,
                "apex_core_area_fraction": apex_core_area_fraction,
                "area_shape_factor": area_shape_factor,
                "height": float(work[apex]),
                "prominence": prominence,
                "relative_prominence": relative_prominence,
                "random_noise": random_noise,
                "structured_noise": structured_noise,
                "baseline_model_uncertainty": model_uncertainty,
                "baseline_model_disagreement": model_disagreement,
                "effective_noise": effective_noise,
                "random_snr": float(
                    props["prominences"][i] / max(random_noise, EPS)
                ),
                "snr": float(
                    props["prominences"][i] / max(effective_noise, EPS)
                ),
                "apex_in_bump": bool(processed.bump_mask[apex]),
                "bump_overlap_fraction": bump_overlap_fraction,
                "candidate_density": 0,
                "weak_candidate_density": 0,
                "structured_background_residual_flag": False,
                "ripple_group": "",
                "area": area,
                "symmetry": float(symmetry),
                "bilateral_depth": bilateral_depth,
                "bilateral_depth_relative": bilateral_depth / signal_range,
                "signed_depth_relative": (
                    max(0.0, -float(processed.signed[apex])) / signal_range
                    if sign < 0
                    else max(0.0, float(processed.positive[apex])) / signal_range
                ),
                "impulse_overlap": impulse_overlap,
                "apex_spike_flag": apex_spike_flag,
                "apex_impulse_ratio": apex_impulse_ratio,
                "apex_quadratic_excess_ratio": apex_quadratic_excess_ratio,
                "despiked_height_retention": despiked_height_retention,
                "despiked_prominence_retention": despiked_prominence_retention,
                "despiked_area_retention": despiked_area_retention,
                "peak_body_survives_despike": False,
                "file_impulse_density_per_min": file_impulse_density_per_min,
                "electrical_assessment_applicable": bool(sign > 0),
                "peak_body_score": 0.0 if sign > 0 else float("nan"),
                "electrical_interference_score": (
                    0.0 if sign > 0 else float("nan")
                ),
                "baseline_change_ratio": baseline_change_ratio,
                "edge_distance_min": float(min(x[apex] - x[0], x[-1] - x[apex])),
            }
        )
    return features


def train_template(reference_curves: list[Curve]) -> tuple[list[dict], dict[str, Preprocessed]]:
    samples: list[dict] = []
    processed_map: dict[str, Preprocessed] = {}
    for curve in reference_curves:
        processed = preprocess(curve)
        processed_map[curve.sha256] = processed
        for feature in candidate_features(curve, processed, +1, 0.0002):
            if (
                feature["snr"] >= 8
                and feature["relative_prominence"] >= 0.003
                and 0.025 <= feature["width_min"] <= 0.80
            ):
                samples.append({**feature, "sha256": curve.sha256, "file": curve.name})
    if not samples:
        raise RuntimeError("No stable positive peaks found in reference chromatograms")
    frame = pd.DataFrame(samples)
    labels = fclusterdata(
        frame[["apex_time_min"]].to_numpy(),
        # The supplied references show up to about 0.10 min drift for the
        # ~1.8 min component. 0.12 min keeps that family together while the
        # complete-linkage span still separates the earlier adjacent peaks.
        t=0.120,
        criterion="distance",
        method="complete",
    )
    frame["cluster"] = labels
    clusters: list[dict] = []
    for _, group in frame.groupby("cluster"):
        support = int(group["sha256"].nunique())
        if support < max(4, len(reference_curves) - 1):
            continue
        rt = float(group["apex_time_min"].median())
        width = float(group["width_min"].median())
        symmetry = float(group["symmetry"].median())
        clusters.append(
            {
                "retention_time_min": rt,
                "rt_tolerance_min": min(
                    0.22,
                    max(0.045, 0.05 * rt, 0.5 * float(np.ptp(group["apex_time_min"])) + 0.02),
                ),
                "width_median_min": width,
                "width_low_min": max(0.018, float(group["width_min"].min()) * 0.75),
                "width_high_min": float(group["width_min"].max()) * 1.30,
                "symmetry_median": symmetry,
                "symmetry_low": max(0.20, float(group["symmetry"].min()) * 0.70),
                "support": support,
                "reference_count": int(len(group)),
                "component": "",
            }
        )
    clusters.sort(key=lambda item: item["retention_time_min"])
    for index, cluster in enumerate(clusters, 1):
        cluster["slot"] = f"T{index}"
    if len(clusters) < 4:
        raise RuntimeError(f"Only {len(clusters)} stable reference peak clusters were learned")
    return clusters, processed_map


def load_component_map(path: Path | None, template: list[dict]) -> dict[str, str]:
    mapping = {item["slot"]: item.get("component", "") for item in template}
    if path is not None and path.exists():
        user = json.loads(path.read_text(encoding="utf-8"))
        for key, value in user.items():
            if key in mapping:
                mapping[key] = str(value)
    return mapping


def nearest_slot(feature: dict, template: list[dict]) -> tuple[dict | None, float]:
    best = None
    best_normalized = float("inf")
    for slot in template:
        error = abs(feature["apex_time_min"] - slot["retention_time_min"])
        normalized = error / slot["rt_tolerance_min"]
        if normalized < best_normalized:
            best = slot
            best_normalized = normalized
    if best is None or best_normalized > 1.0:
        return None, float("nan")
    return best, abs(feature["apex_time_min"] - best["retention_time_min"])


def clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def log_evidence(value: float, weak: float, strong: float) -> float:
    """Map evidence spanning orders of magnitude onto a stable 0..1 score."""
    if value <= weak:
        return 0.0
    if value >= strong:
        return 1.0
    return clip01(math.log(value / weak) / math.log(strong / weak))


def log_ratio_score(value: float, reference: float, factor: float = 1.8) -> float:
    ratio = max(value, EPS) / max(reference, EPS)
    return float(math.exp(-0.5 * (math.log(ratio) / math.log(factor)) ** 2))


def best_width_score(feature: dict, template: list[dict]) -> float:
    if not template:
        return 0.0
    return max(
        log_ratio_score(feature["width_min"], slot["width_median_min"])
        for slot in template
    )


def template_match_scores(feature: dict, slot: dict | None) -> tuple[float, float, float]:
    if slot is None:
        return 0.0, float("nan"), 0.0
    rt_error = abs(feature["apex_time_min"] - slot["retention_time_min"])
    rt_score = clip01(1.0 - rt_error / slot["rt_tolerance_min"])
    width_ratio = feature["width_min"] / max(slot["width_median_min"], EPS)
    width_score = log_ratio_score(feature["width_min"], slot["width_median_min"])
    symmetry_score = clip01(feature["symmetry"] / max(slot["symmetry_median"], 0.25))
    nonflat_score = clip01(1.0 - (feature["top_width_ratio"] - 0.42) / 0.25)
    confidence = 0.50 * rt_score + 0.30 * width_score + 0.15 * symmetry_score + 0.05 * nonflat_score
    return clip01(confidence), float(width_ratio), rt_score


def electrical_morphology_assessment(
    feature: dict,
    slot: dict | None,
    config: DetectorConfig,
) -> dict[str, float | bool]:
    """Jointly score a chromatographic peak body and electrical-spike evidence.

    An isolated impulse at the apex is not a hard veto.  A candidate is rescued
    as ``peak_with_apex_spike`` only when a conservative, multi-parameter peak
    body remains after despiking.  Conversely, a retention-time match never
    overrides strong spike morphology because this assessment is performed
    before the final template/category routing.  A learned template contributes
    only relative width support; retention time alone can never rescue a spike.
    """

    continuity = min(feature["rise_continuity"], feature["fall_continuity"])
    retention = float(
        np.mean(
            [
                feature["despiked_height_retention"],
                feature["despiked_prominence_retention"],
                feature["despiked_area_retention"],
            ]
        )
    )
    template_width_support = (
        log_ratio_score(feature["width_min"], slot["width_median_min"])
        if slot is not None
        else 0.0
    )
    body_components = {
        "duration": max(
            clip01((feature["fwhm_min"] - 0.035) / 0.045),
            template_width_support,
        ),
        "slope_balance": clip01((feature["slope_asymmetry"] - 0.20) / 0.65),
        "symmetry": clip01((feature["symmetry"] - 0.45) / 0.40),
        "area_shape": clip01((feature["area_shape_factor"] - 0.75) / 0.55),
        "nonflat": clip01((0.55 - feature["top_width_ratio"]) / 0.25),
        "baseline_stability": clip01((0.40 - feature["baseline_change_ratio"]) / 0.40),
        "continuity": clip01(continuity),
        "energy_spread": clip01(
            (0.42 - feature["apex_core_area_fraction"]) / 0.30
        ),
        "despike_retention": clip01(retention),
        "mild_apex_impulse": clip01(
            (0.25 - feature["apex_quadratic_excess_ratio"]) / 0.25
        ),
    }
    peak_body_score = (
        0.14 * body_components["duration"]
        + 0.10 * body_components["slope_balance"]
        + 0.12 * body_components["symmetry"]
        + 0.16 * body_components["area_shape"]
        + 0.08 * body_components["nonflat"]
        + 0.08 * body_components["baseline_stability"]
        + 0.12 * body_components["continuity"]
        + 0.08 * body_components["energy_spread"]
        + 0.08 * body_components["despike_retention"]
        + 0.04 * body_components["mild_apex_impulse"]
    )

    electrical_components = {
        "time_narrow": clip01((0.080 - feature["fwhm_min"]) / 0.060),
        "top_concentration": clip01(
            (feature["apex_core_area_fraction"] - 0.12) / 0.30
        ),
        "slope_jump": clip01(
            (feature["normalized_max_slope"] - 18.0) / 35.0
        ),
        "apex_impulse": (
            clip01(
                (feature["apex_quadratic_excess_ratio"] - 0.015) / 0.085
            )
            if feature["apex_spike_flag"]
            else 0.0
        ),
        "impulse_train": clip01(
            (feature["file_impulse_density_per_min"] - 0.30) / 0.50
        ),
        "low_symmetry": clip01((0.60 - feature["symmetry"]) / 0.50),
        "flat_top": clip01((feature["top_width_ratio"] - 0.50) / 0.20),
        "slope_imbalance": clip01(
            (0.45 - feature["slope_asymmetry"]) / 0.40
        ),
        "despike_loss": clip01((0.85 - retention) / 0.60),
    }
    electrical_interference_score = (
        0.18 * electrical_components["time_narrow"]
        + 0.14 * electrical_components["top_concentration"]
        + 0.14 * electrical_components["slope_jump"]
        + 0.10 * electrical_components["apex_impulse"]
        + 0.18 * electrical_components["impulse_train"]
        + 0.06 * electrical_components["low_symmetry"]
        + 0.06 * electrical_components["flat_top"]
        + 0.08 * electrical_components["slope_imbalance"]
        + 0.06 * electrical_components["despike_loss"]
    )

    absolute_body_shape = bool(
        feature["fwhm_min"] >= 0.060
        and feature["area_shape_factor"] >= 1.15
    )
    template_relative_body_shape = bool(
        slot is not None
        and 0.75 * slot["width_low_min"]
        <= feature["width_min"]
        <= 1.25 * slot["width_high_min"]
        and feature["area_shape_factor"] >= 0.85
    )

    # The absolute route conservatively protects A7-like complete triangular
    # bodies.  The template-relative route protects a legitimately narrow T1
    # body, but only together with every independent shape/retention check.
    peak_body_survives_despike = bool(
        (absolute_body_shape or template_relative_body_shape)
        and feature["symmetry"] >= 0.75
        and feature["top_width_ratio"] <= 0.45
        and feature["baseline_change_ratio"] <= 0.25
        and continuity >= 0.80
        and feature["apex_core_area_fraction"] <= 0.35
        and feature["despiked_height_retention"] >= 0.65
        and feature["despiked_prominence_retention"] >= 0.65
        and feature["despiked_area_retention"] >= 0.65
        and feature["apex_quadratic_excess_ratio"] <= 0.25
    )
    apex_spike_on_peak = bool(
        feature["apex_spike_flag"]
        and peak_body_survives_despike
        and peak_body_score >= config.apex_spike_peak_body_floor
        and peak_body_score >= electrical_interference_score
    )
    pure_electrical_spike = bool(
        electrical_interference_score
        >= config.electrical_spike_score_threshold
        and not peak_body_survives_despike
        and (feature["apex_spike_flag"] or feature["fwhm_min"] < 0.018)
    )
    ambiguous_apex_anomaly = bool(
        feature["apex_spike_flag"]
        and not apex_spike_on_peak
        and not pure_electrical_spike
    )
    return {
        "peak_body_score": clip01(peak_body_score),
        "electrical_interference_score": clip01(electrical_interference_score),
        "peak_body_survives_despike": peak_body_survives_despike,
        "apex_spike_on_peak": apex_spike_on_peak,
        "pure_electrical_spike": pure_electrical_spike,
        "ambiguous_apex_anomaly": ambiguous_apex_anomaly,
    }


def composite_peak_scores(
    feature: dict,
    template: list[dict],
    sign: int,
    high_noise: bool,
    config: DetectorConfig,
    width_score_scale: float = 1.0,
) -> dict[str, float]:
    snr_score = log_evidence(feature["snr"], 3.0, 30.0)
    prominence_score = log_evidence(
        feature["relative_prominence"],
        config.normal_relative_prominence_floor,
        0.03,
    )
    width_score = best_width_score(feature, template) * width_score_scale
    symmetry_score = clip01((feature["symmetry"] - 0.20) / 0.70)
    nonflat_score = clip01(1.0 - (feature["top_width_ratio"] - 0.42) / 0.25)
    signed_depth_score = log_evidence(feature["signed_depth_relative"], 0.008, 0.08)
    bilateral_score = log_evidence(feature["bilateral_depth_relative"], 0.005, 0.08)

    if sign > 0:
        confidence = (
            0.25 * snr_score
            + 0.25 * prominence_score
            + 0.20 * width_score
            + 0.15 * symmetry_score
            + 0.15 * nonflat_score
        )
    else:
        confidence = (
            0.18 * snr_score
            + 0.15 * prominence_score
            + 0.12 * width_score
            + 0.10 * symmetry_score
            + 0.20 * signed_depth_score
            + 0.25 * bilateral_score
        )

    if high_noise and feature["relative_prominence"] < config.high_noise_relative_prominence_floor:
        confidence *= 0.75
    if feature["edge_distance_min"] < max(0.12, 0.5 * feature["width_min"]):
        confidence *= 0.65
    return {
        "peak_confidence": clip01(confidence),
        "score_snr": snr_score,
        "score_prominence": prominence_score,
        "score_width": width_score,
        "score_symmetry": symmetry_score,
        "score_nonflat": nonflat_score,
        "score_signed_depth": signed_depth_score,
        "score_bilateral_depth": bilateral_score,
    }


def status_from_confidence(
    confidence: float,
    feature_type: str,
    config: DetectorConfig,
) -> str:
    if confidence >= config.confirmation_threshold:
        return "confirmed"
    if confidence < config.artifact_threshold and (
        "electrical" in feature_type or feature_type == "uncertain_peak_or_noise"
    ):
        return "artifact"
    return "review"


def confidence_reasons(
    scores: dict[str, float],
    template_confidence: float,
    rt_score: float,
    config: DetectorConfig,
) -> list[str]:
    return [
        f"snr_score={scores['score_snr']:.2f}",
        f"prominence_score={scores['score_prominence']:.2f}",
        f"width_score={scores['score_width']:.2f}",
        f"symmetry_score={scores['score_symmetry']:.2f}",
        f"nonflat_score={scores['score_nonflat']:.2f}",
        f"signed_depth_score={scores['score_signed_depth']:.2f}",
        f"bilateral_depth_score={scores['score_bilateral_depth']:.2f}",
        f"rt_score={rt_score:.2f}",
        f"template_confidence={template_confidence:.3f}",
        f"peak_confidence={scores['peak_confidence']:.3f}",
        f"confirmation_threshold={config.confirmation_threshold:.2f}",
    ]


def high_noise_condition(processed: Preprocessed, reference_noise: float) -> bool:
    """Require both a material absolute noise floor and a reference increase."""
    return processed.global_noise > max(10.0 * max(reference_noise, EPS), 5e-5)


def annotate_positive_candidate_context(
    features: list[dict], config: DetectorConfig
) -> None:
    """Attach local density and config-consistent effective-noise evidence."""
    if not features:
        return
    times = np.asarray([feature["apex_time_min"] for feature in features])
    weak = np.asarray(
        [
            feature["relative_prominence"]
            < config.hump_relative_prominence_floor
            for feature in features
        ],
        dtype=bool,
    )
    for index, feature in enumerate(features):
        local = np.abs(times - times[index]) <= config.hump_density_radius_min
        feature["candidate_density"] = int(np.count_nonzero(local))
        feature["weak_candidate_density"] = int(np.count_nonzero(local & weak))
        feature["effective_noise"] = (
            max(
                feature["random_noise"],
                feature["structured_noise"],
                feature["baseline_model_uncertainty"],
            )
            if weak[index]
            else feature["random_noise"]
        )
        feature["snr"] = feature["prominence"] / max(
            feature["effective_noise"], EPS
        )


def nonstationary_background_condition(processed: Preprocessed) -> bool:
    """Use one definition everywhere a non-stationary background is reported."""
    return bool(
        processed.baseline_excursion_ratio >= 0.08
        or processed.baseline_curvature >= 0.04
        or abs(processed.endpoint_drift_ratio) >= 0.05
    )


def is_structured_background_residual(
    feature: dict,
    processed: Preprocessed,
    config: DetectorConfig,
) -> bool:
    """Identify weak ripple in a globally non-stationary baseline.

    This is a confirmation gate, not deletion. The calibrated conjunction is
    intentionally conservative: all conditions must hold, and the candidate
    remains in the audit output as ``review``.
    """
    return bool(
        nonstationary_background_condition(processed)
        and feature["relative_prominence"]
        < config.hump_relative_prominence_floor
        and feature["width_min"] < config.hump_candidate_width_limit_min
        and feature["weak_candidate_density"]
        >= config.hump_weak_density_limit
        and feature["bilateral_depth_relative"]
        < config.hump_bilateral_depth_floor
    )


def assign_ripple_groups(classified: list[dict], maximum_gap_min: float = 0.35) -> None:
    """Group nearby structured residuals for compact reporting and plotting."""
    residuals = sorted(
        (
            item
            for item in classified
            if item.get("structured_background_residual_flag", False)
        ),
        key=lambda item: item["apex_time_min"],
    )
    group_index = 0
    previous_time: float | None = None
    for item in residuals:
        if previous_time is None or item["apex_time_min"] - previous_time > maximum_gap_min:
            group_index += 1
        item["ripple_group"] = f"R{group_index}"
        previous_time = item["apex_time_min"]


def is_gentle_broad_peak_candidate(feature: dict, slot: dict | None) -> bool:
    """Conservatively distinguish a smooth, slow peak from obvious drift.

    This rule intentionally applies only outside learned component windows.
    Inside a template window, an abnormally wide event remains a broad or
    overlapped candidate until a future deconvolution stage can separate it.
    """
    width_to_fwhm = feature["width_min"] / max(feature["fwhm_min"], EPS)
    return bool(
        slot is None
        and not feature["impulse_overlap"]
        and feature["width_min"] >= 0.80
        and feature["fwhm_min"] >= 0.30
        and width_to_fwhm <= 6.0
        and feature["symmetry"] >= 0.25
        and feature["top_width_ratio"] <= 0.65
        and feature["relative_prominence"] >= 0.003
        and feature["snr"] >= 8.0
        and feature["baseline_change_ratio"] <= 3.0
        and feature["bilateral_depth_relative"] >= 0.002
        and feature["edge_distance_min"] >= max(0.12, 0.5 * feature["width_min"])
    )


def classify_curve(
    curve: Curve,
    processed: Preprocessed,
    template: list[dict],
    component_map: dict[str, str],
    reference_noise: float,
    config: DetectorConfig,
) -> list[dict]:
    positive = candidate_features(curve, processed, +1)
    negative = candidate_features(curve, processed, -1)
    annotate_positive_candidate_context(positive, config)
    high_noise = high_noise_condition(processed, reference_noise)
    classified: list[dict] = []

    for feature in positive:
        slot, rt_error = nearest_slot(feature, template)
        electrical_assessment = electrical_morphology_assessment(
            feature, slot, config
        )
        feature.update(
            {
                "peak_body_score": electrical_assessment["peak_body_score"],
                "electrical_interference_score": electrical_assessment[
                    "electrical_interference_score"
                ],
                "peak_body_survives_despike": electrical_assessment[
                    "peak_body_survives_despike"
                ],
            }
        )
        pure_electrical_spike = bool(
            electrical_assessment["pure_electrical_spike"]
        )
        apex_spike_on_peak = bool(electrical_assessment["apex_spike_on_peak"])
        ambiguous_apex_anomaly = bool(
            electrical_assessment["ambiguous_apex_anomaly"]
        )
        background_residual_candidate = is_structured_background_residual(
            feature, processed, config
        )
        feature["structured_background_residual_flag"] = False
        outside_floor = (
            config.high_noise_relative_prominence_floor
            if high_noise
            else config.normal_relative_prominence_floor
        )
        # Outside learned retention windows, tiny ripples are represented by
        # the file-level noise metric rather than emitted as hundreds of fake
        # "peaks". Low-amplitude peaks inside a learned slot remain eligible.
        if (
            slot is None
            and feature["relative_prominence"] < outside_floor
            and not feature["impulse_overlap"]
            and feature["width_min"] < 0.80
        ):
            continue
        if (
            slot is None
            and feature["width_min"] >= 0.80
            and feature["relative_prominence"] < 0.003
        ):
            continue
        component = ""
        template_confidence, width_ratio, rt_score = template_match_scores(feature, slot)
        scores = composite_peak_scores(feature, template, +1, high_noise, config)
        morphology_multiplier = 1.0
        on_detected_bump = bool(
            feature["apex_in_bump"] or feature["bump_overlap_fraction"] >= 0.50
        )
        if slot is not None:
            component = component_map.get(slot["slot"], "")
            width_ok = slot["width_low_min"] <= feature["width_min"] <= slot["width_high_min"]
            symmetry_ok = feature["symmetry"] >= slot["symmetry_low"]
            if pure_electrical_spike:
                # A retention-time match is supporting chemical evidence, not
                # permission for a strong multi-parameter electrical transient.
                feature_type = "electrical_interference_candidate"
                morphology_multiplier = 0.25
            elif apex_spike_on_peak:
                feature_type = "peak_with_apex_spike"
                morphology_multiplier = config.apex_spike_confidence_multiplier
            elif ambiguous_apex_anomaly:
                feature_type = "electrical_interference_candidate"
                morphology_multiplier = 0.55
            elif width_ok and symmetry_ok:
                feature_type = "normal_positive_peak"
                if (
                    feature["baseline_change_ratio"] >= 0.25
                    or on_detected_bump
                ):
                    feature_type = "positive_peak_on_hump"
            elif feature["width_min"] < slot["width_low_min"]:
                if (
                    feature["top_width_ratio"] >= 0.55
                    or not symmetry_ok
                ):
                    feature_type = "electrical_interference_candidate"
                    morphology_multiplier = 0.45
                else:
                    feature_type = "narrow_positive_peak"
            elif feature["width_min"] > slot["width_high_min"]:
                feature_type = "broad_or_overlapped_peak"
            else:
                feature_type = "uncertain_peak_or_noise"
        else:
            if pure_electrical_spike:
                feature_type = "electrical_spike"
                morphology_multiplier = 0.25
            elif apex_spike_on_peak:
                feature_type = "peak_with_apex_spike"
                morphology_multiplier = config.apex_spike_confidence_multiplier
            elif ambiguous_apex_anomaly:
                feature_type = "electrical_interference_candidate"
                morphology_multiplier = 0.55
            elif feature["width_min"] > 0.80:
                feature_type = (
                    "gentle_broad_peak_candidate"
                    if is_gentle_broad_peak_candidate(feature, slot)
                    else "broad_hump_or_baseline"
                )
                morphology_multiplier = 0.80
            elif scores["score_width"] >= 0.45 and feature["symmetry"] >= 0.20:
                feature_type = "unassigned_positive_peak"
                if (
                    feature["baseline_change_ratio"] >= 0.25
                    or on_detected_bump
                ):
                    feature_type = "positive_peak_on_hump"
            else:
                feature_type = "uncertain_peak_or_noise"

        scores["peak_confidence"] *= morphology_multiplier
        scores["peak_confidence"] = clip01(scores["peak_confidence"])
        base_status = status_from_confidence(
            scores["peak_confidence"], feature_type, config
        )
        structured_residual = bool(
            background_residual_candidate
            and base_status != "artifact"
            and "electrical" not in feature_type
        )
        if structured_residual:
            # Recompute the whole score with a reduced reference-width reward,
            # so high-noise/edge multipliers keep their original mathematics.
            scores = composite_peak_scores(
                feature,
                template,
                +1,
                high_noise,
                config,
                width_score_scale=0.50,
            )
            scores["peak_confidence"] *= morphology_multiplier
            feature_type = "structured_background_residual"
            scores["peak_confidence"] = min(
                scores["peak_confidence"],
                config.confirmation_threshold - 0.01,
            )
            feature["structured_background_residual_flag"] = True

        scores["peak_confidence"] = clip01(scores["peak_confidence"])
        status = status_from_confidence(scores["peak_confidence"], feature_type, config)
        if feature_type == "broad_hump_or_baseline" and status == "confirmed":
            feature_type = "broad_positive_peak"
        if feature_type == "uncertain_peak_or_noise" and status == "confirmed":
            feature_type = "unassigned_positive_peak"
        reasons = confidence_reasons(scores, template_confidence, rt_score, config)
        if feature["baseline_change_ratio"] >= 0.25:
            reasons.append("local_baseline_changes_across_peak")
        if on_detected_bump:
            reasons.append("candidate_overlaps_detected_bump_region")
        if feature["relative_prominence"] < config.hump_relative_prominence_floor:
            reasons.append(f"effective_snr={feature['snr']:.2f}")
        if structured_residual:
            reasons.extend(
                [
                    "structured_background_residual",
                    f"weak_candidate_density={feature['weak_candidate_density']}",
                    f"baseline_excursion_ratio={processed.baseline_excursion_ratio:.3f}",
                    f"baseline_curvature={processed.baseline_curvature:.3f}",
                    f"confirmation_cap={config.confirmation_threshold - 0.01:.2f}",
                ]
            )
        if feature_type == "gentle_broad_peak_candidate":
            reasons.append("conservative_gentle_broad_peak_rule")
        reasons.extend(
            [
                f"peak_body_score={feature['peak_body_score']:.3f}",
                f"electrical_interference_score={feature['electrical_interference_score']:.3f}",
            ]
        )
        if apex_spike_on_peak:
            reasons.append("apex_spike_on_surviving_peak_body")
        elif pure_electrical_spike:
            reasons.append("multi_parameter_electrical_spike")
        elif ambiguous_apex_anomaly:
            reasons.append("apex_anomaly_without_reliable_peak_body")
        elif feature["impulse_overlap"]:
            reasons.append("provisional_impulse_overlap_without_spike_evidence")
        if feature["edge_distance_min"] < max(0.12, 0.5 * feature["width_min"]):
            reasons.append("acquisition_edge_penalty=0.65")
        classified.append(
            {
                **feature,
                "feature_type": feature_type,
                "status": status,
                "template_slot": "" if slot is None else slot["slot"],
                "component": component,
                "width_ratio_to_template": width_ratio,
                "rt_error_min": rt_error,
                **scores,
                "peak_confidence_percent": 100.0 * scores["peak_confidence"],
                "template_confidence": template_confidence,
                "template_confidence_percent": 100.0 * template_confidence,
                "confirmation_threshold": config.confirmation_threshold,
                "confidence": scores["peak_confidence"],
                "reasons": reasons,
            }
        )

    for feature in negative:
        bilateral_floor = (
            config.high_noise_negative_bilateral_floor
            if high_noise
            else config.negative_bilateral_depth_floor
        )
        signed_floor = (
            config.high_noise_negative_signed_floor
            if high_noise
            else config.negative_signed_depth_floor
        )
        if (
            feature["bilateral_depth_relative"] < bilateral_floor
            or feature["signed_depth_relative"] < signed_floor
        ):
            continue
        scores = composite_peak_scores(feature, template, -1, high_noise, config)
        if feature["width_min"] < 0.018:
            feature_type = "negative_electrical_spike"
            scores["peak_confidence"] *= 0.25
        else:
            feature_type = "negative_peak" if feature["width_min"] <= 0.80 else "broad_negative_peak"
            if feature["impulse_overlap"]:
                scores["peak_confidence"] *= 0.65
        scores["peak_confidence"] = clip01(scores["peak_confidence"])
        status = status_from_confidence(scores["peak_confidence"], feature_type, config)
        reasons = confidence_reasons(scores, 0.0, 0.0, config)
        reasons.extend(["signed_peak_detected", "raw_signal_has_bilateral_valley"])
        if feature["impulse_overlap"]:
            reasons.append("spike_or_negative_ambiguous")
        if feature["edge_distance_min"] < max(0.12, 0.5 * feature["width_min"]):
            reasons.append("acquisition_edge_penalty=0.65")
        classified.append(
            {
                **feature,
                "feature_type": feature_type,
                "status": status,
                "template_slot": "",
                "component": "",
                "width_ratio_to_template": float("nan"),
                "rt_error_min": float("nan"),
                **scores,
                "peak_confidence_percent": 100.0 * scores["peak_confidence"],
                "template_confidence": 0.0,
                "template_confidence_percent": 0.0,
                "confirmation_threshold": config.confirmation_threshold,
                "confidence": scores["peak_confidence"],
                "reasons": reasons,
            }
        )

    # A shallow valley bracketed by positive peaks can look negative after
    # baseline removal. Keep it visible, but do not state that it is certainly
    # a chemical negative peak without the component/retention map.
    positive_times = sorted(
        item["apex_time_min"] for item in classified if item["sign"] == "positive"
    )
    for item in classified:
        if item["sign"] != "negative":
            continue
        if item["signed_depth_relative"] >= 0.05:
            continue
        time = item["apex_time_min"]
        left_exists = any(0 < time - other <= 0.8 for other in positive_times)
        right_exists = any(0 < other - time <= 0.8 for other in positive_times)
        if left_exists and right_exists:
            item["feature_type"] = "interpeak_valley_or_negative_peak"
            item["peak_confidence"] = clip01(item["peak_confidence"] * 0.65)
            item["peak_confidence_percent"] = 100.0 * item["peak_confidence"]
            item["confidence"] = item["peak_confidence"]
            item["status"] = status_from_confidence(
                item["peak_confidence"], item["feature_type"], config
            )
            item["reasons"].append("positive_peaks_on_both_sides")
            item["reasons"].append("interpeak_valley_penalty=0.65")

    # Multiple real peaks may occupy one template window. Preserve peak-status
    # decisions and only mark the lower template-confidence entries as
    # secondary/overlap candidates.
    for slot in [item["slot"] for item in template]:
        indices = [
            i
            for i, item in enumerate(classified)
            if item["template_slot"] == slot and item["status"] == "confirmed"
        ]
        if len(indices) > 1:
            winner = max(
                indices,
                key=lambda i: (
                    classified[i]["template_confidence"],
                    classified[i]["peak_confidence"],
                ),
            )
            for i in indices:
                if i == winner:
                    continue
                if classified[i]["feature_type"] != "peak_with_apex_spike":
                    classified[i]["feature_type"] = "secondary_or_overlapping_candidate"
                classified[i]["reasons"].append("multiple_candidates_in_one_template_slot")

    # Mark overlapping confirmed peak windows without losing the base type.
    confirmed = [
        item
        for item in classified
        if item["status"] == "confirmed" and item["sign"] == "positive"
    ]
    confirmed.sort(key=lambda item: item["apex_time_min"])
    for left, right in zip(confirmed, confirmed[1:]):
        if left["end_time_min"] >= right["start_time_min"]:
            # Keep the apex-spike quality class visible; overlap is still
            # recorded explicitly in reasons.  Otherwise this late pass would
            # erase the very distinction made by the joint morphology model.
            if left["feature_type"] != "peak_with_apex_spike":
                left["feature_type"] = "overlapping_positive_peak"
            if right["feature_type"] != "peak_with_apex_spike":
                right["feature_type"] = "overlapping_positive_peak"
            left["reasons"].append("window_overlaps_next_confirmed_peak")
            right["reasons"].append("window_overlaps_previous_confirmed_peak")
    assign_ripple_groups(classified)
    return classified


def feature_rows(curve: Curve, classified: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for index, feature in enumerate(sorted(classified, key=lambda item: item["apex_time_min"]), 1):
        row = {
            "folder": curve.folder,
            "file": curve.name,
            "sha256": curve.sha256,
            "feature_id": index,
            "feature_type": feature["feature_type"],
            "feature_type_cn": FEATURE_TYPE_CN.get(feature["feature_type"], feature["feature_type"]),
            "status": feature["status"],
            "status_cn": STATUS_CN[feature["status"]],
            "sign": feature["sign"],
            "template_slot": feature["template_slot"],
            "component": feature["component"],
            "apex_time_min": feature["apex_time_min"],
            "corrected_apex_value": (
                feature["height"]
                if feature["sign"] == "positive"
                else -feature["height"]
            ),
            "raw_apex_value": float(
                np.interp(feature["apex_time_min"], curve.x, curve.y)
            ),
            "start_time_min": feature["start_time_min"],
            "end_time_min": feature["end_time_min"],
            "width_min": feature["width_min"],
            "fwhm_min": feature["fwhm_min"],
            "width_samples": feature["width_samples"],
            "fwhm_samples": feature["fwhm_samples"],
            "top_width_ratio": feature["top_width_ratio"],
            "rise_time_min": feature["rise_time_min"],
            "fall_time_min": feature["fall_time_min"],
            "rise_continuity": feature["rise_continuity"],
            "fall_continuity": feature["fall_continuity"],
            "normalized_max_slope": feature["normalized_max_slope"],
            "max_step_ratio": feature["max_step_ratio"],
            "slope_asymmetry": feature["slope_asymmetry"],
            "top3_energy_fraction": feature["top3_energy_fraction"],
            "apex_core_area_fraction": feature["apex_core_area_fraction"],
            "area_shape_factor": feature["area_shape_factor"],
            "apex_spike_flag": feature["apex_spike_flag"],
            "impulse_overlap": feature["impulse_overlap"],
            "apex_impulse_ratio": feature["apex_impulse_ratio"],
            "apex_quadratic_excess_ratio": feature[
                "apex_quadratic_excess_ratio"
            ],
            "despiked_height_retention": feature["despiked_height_retention"],
            "despiked_prominence_retention": feature[
                "despiked_prominence_retention"
            ],
            "despiked_area_retention": feature["despiked_area_retention"],
            "peak_body_survives_despike": feature[
                "peak_body_survives_despike"
            ],
            "file_impulse_density_per_min": feature[
                "file_impulse_density_per_min"
            ],
            "electrical_assessment_applicable": feature[
                "electrical_assessment_applicable"
            ],
            "peak_body_score": feature["peak_body_score"],
            "electrical_interference_score": feature[
                "electrical_interference_score"
            ],
            "height": feature["height"],
            "prominence": feature["prominence"],
            "relative_prominence": feature["relative_prominence"],
            "snr": feature["snr"],
            "random_snr": feature["random_snr"],
            "random_noise": feature["random_noise"],
            "structured_noise": feature["structured_noise"],
            "baseline_model_uncertainty": feature["baseline_model_uncertainty"],
            "baseline_model_disagreement": feature["baseline_model_disagreement"],
            "effective_noise": feature["effective_noise"],
            "width_to_fwhm": feature["width_to_fwhm"],
            "apex_in_bump": feature["apex_in_bump"],
            "bump_overlap_fraction": feature["bump_overlap_fraction"],
            "candidate_density": feature["candidate_density"],
            "weak_candidate_density": feature["weak_candidate_density"],
            "structured_background_residual_flag": feature[
                "structured_background_residual_flag"
            ],
            "ripple_group": feature["ripple_group"],
            "area": feature["area"],
            "symmetry": feature["symmetry"],
            "bilateral_depth": feature["bilateral_depth"],
            "bilateral_depth_relative": feature["bilateral_depth_relative"],
            "signed_depth_relative": feature["signed_depth_relative"],
            "baseline_change_ratio": feature["baseline_change_ratio"],
            "edge_distance_min": feature["edge_distance_min"],
            "width_ratio_to_template": feature["width_ratio_to_template"],
            "rt_error_min": feature["rt_error_min"],
            "peak_confidence": feature["peak_confidence"],
            "peak_confidence_percent": feature["peak_confidence_percent"],
            "template_confidence": feature["template_confidence"],
            "template_confidence_percent": feature["template_confidence_percent"],
            "score_snr": feature["score_snr"],
            "score_prominence": feature["score_prominence"],
            "score_width": feature["score_width"],
            "score_symmetry": feature["score_symmetry"],
            "score_nonflat": feature["score_nonflat"],
            "score_signed_depth": feature["score_signed_depth"],
            "score_bilateral_depth": feature["score_bilateral_depth"],
            "confirmation_threshold": feature["confirmation_threshold"],
            "confidence": feature["confidence"],
            "reasons": ";".join(feature["reasons"]),
        }
        rows.append(row)
    return rows


def _annotation_priority(row: dict) -> int:
    if row["status"] == "confirmed":
        return 6
    if row["template_slot"]:
        return 5
    if row["feature_type"] == "gentle_broad_peak_candidate" or "electrical" in row["feature_type"]:
        return 4
    if row["feature_type"] in {"negative_peak", "broad_negative_peak"}:
        return 3
    return 2


def _confirmed_boundary_specs(rows: list[dict]) -> list[dict]:
    """Return validated plot specifications for confirmed peak boundaries only."""
    specs: list[dict] = []
    for row in rows:
        if row["status"] != "confirmed":
            continue
        start = float(row["start_time_min"])
        apex = float(row["apex_time_min"])
        corrected_apex_value = float(row["corrected_apex_value"])
        end = float(row["end_time_min"])
        if not all(
            math.isfinite(value)
            for value in (start, apex, end, corrected_apex_value)
        ):
            raise ValueError("Confirmed peak plot specification contains a non-finite value")
        if not start < end or not start <= apex <= end:
            raise ValueError(
                "Confirmed peak boundary must satisfy start < end and start <= apex <= end"
            )
        specs.append(
            {
                "folder": row.get("folder"),
                "file": row.get("file"),
                "feature_id": row.get("feature_id"),
                "sign": row["sign"],
                "start_time_min": start,
                "apex_time_min": apex,
                "corrected_apex_value": corrected_apex_value,
                "end_time_min": end,
                "apex_text": (
                    f"顶 {apex:.4f} min / 校正值 {corrected_apex_value:.5g}"
                ),
                "text": f"起 {start:.4f} / 止 {end:.4f} min",
            }
        )
    return specs


def _place_peak_annotations(axis, specs: list[dict]) -> None:
    """Place labels into non-overlapping axes-fraction rails.

    Text width is measured by Matplotlib's renderer, converted back to time
    units, and used for interval scheduling. This is deterministic, avoids a
    third-party layout dependency, and remains readable for very dense traces.
    """
    if not specs:
        return
    count = len(specs)
    fontsize = 7.0 if count <= 15 else 6.5 if count <= 30 else 6.0
    annotations: list[tuple[object, dict]] = []
    for spec in specs:
        annotation = axis.annotate(
            spec["text"],
            (spec["x"], spec["y"]),
            xytext=(spec["x"], 0.90 if spec["direction"] > 0 else 0.10),
            textcoords=axis.get_xaxis_transform(),
            ha="center",
            va="center",
            fontsize=fontsize,
            linespacing=0.95,
            annotation_clip=True,
            bbox={
                "boxstyle": "round,pad=0.14",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": spec["color"],
                "lw": 0.45,
                "alpha": 0.65,
                "shrinkA": 1.5,
                "shrinkB": 2.0,
            },
            zorder=7,
        )
        annotations.append((annotation, spec))

    figure = axis.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axes_box = axis.get_window_extent(renderer)
    x_low, x_high = axis.get_xlim()
    data_per_pixel = (x_high - x_low) / max(axes_box.width, 1.0)
    rail_positions = {
        "top": [0.91, 0.79, 0.67, 0.55],
        "bottom": [0.09, 0.21, 0.33, 0.45],
    }
    occupied: dict[tuple[str, int], list[tuple[float, float]]] = {
        (side, index): []
        for side, rails in rail_positions.items()
        for index in range(len(rails))
    }
    ordered = sorted(
        annotations,
        key=lambda item: (-item[1]["priority"], item[1]["x"]),
    )
    for annotation, spec in ordered:
        initial_box = annotation.get_bbox_patch().get_window_extent(renderer)
        half_width = 0.5 * initial_box.width * data_per_pixel
        padding = 6.0 * data_per_pixel
        preferred = "top" if spec["direction"] > 0 else "bottom"
        side_order = [preferred, "bottom" if preferred == "top" else "top"]
        shift_unit = max(2.0 * half_width + padding, 0.025 * (x_high - x_low))
        shifts = [0.0, -0.75, 0.75, -1.5, 1.5, -2.25, 2.25, -3.0, 3.0]
        placed = False
        for side in side_order:
            for rail_index, rail_y in enumerate(rail_positions[side]):
                intervals = occupied[(side, rail_index)]
                for shift in shifts:
                    label_x = float(
                        np.clip(
                            spec["x"] + shift * shift_unit,
                            x_low + half_width + padding,
                            x_high - half_width - padding,
                        )
                    )
                    interval = (
                        label_x - half_width - padding,
                        label_x + half_width + padding,
                    )
                    if any(interval[0] < right and interval[1] > left for left, right in intervals):
                        continue
                    annotation.set_position((label_x, rail_y))
                    annotation.update_positions(renderer)
                    intervals.append(interval)
                    intervals.sort()
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break
        if not placed:
            if spec.get("required", False):
                # Confirmed labels contain the requested start/end values and
                # must remain visible. This fallback is only reached if every
                # collision-free rail is full; confirmed labels are otherwise
                # placed first because they have the highest priority.
                fallback_side = preferred
                fallback_index = min(
                    range(len(rail_positions[fallback_side])),
                    key=lambda index: len(occupied[(fallback_side, index)]),
                )
                fallback_x = float(
                    np.clip(
                        spec["x"],
                        x_low + half_width + padding,
                        x_high - half_width - padding,
                    )
                )
                annotation.set_position(
                    (fallback_x, rail_positions[fallback_side][fallback_index])
                )
                annotation.update_positions(renderer)
            else:
                # The symbol and complete CSV row remain even when a
                # lower-priority text rail is full.
                annotation.set_visible(False)


def plot_result(
    curve: Curve,
    processed: Preprocessed,
    rows: list[dict],
    template: list[dict],
    output: Path,
) -> None:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.25]},
    )

    # curve.y is the Curvel column read directly from the CSV. This panel is
    # intentionally raw-only: no filtering, smoothing, impulse replacement,
    # or baseline subtraction is applied to the displayed line.
    axes[0].plot(curve.x, curve.y, color="0.25", lw=0.75, label="raw samples (unprocessed)")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].set_ylabel("raw signal")
    axes[0].set_title(f"{curve.key}\nraw CSV trace (no preprocessing)")

    # Preprocessing diagnostics are separate from the raw-only panel so the
    # baseline estimates cannot be mistaken for original measurements.
    axes[1].plot(curve.x, curve.y, color="0.75", lw=0.65, label="raw reference")
    axes[1].plot(
        curve.x,
        processed.baseline,
        color="#d62728",
        lw=1.0,
        label="robust main baseline",
    )
    axes[1].plot(
        curve.x,
        processed.hump_baseline,
        color="#9467bd",
        lw=0.7,
        alpha=0.75,
        label="rolling-ball local background",
    )
    for interval_index, (start, end) in enumerate(
        _contiguous_intervals(processed.bump_mask)
    ):
        axes[1].axvspan(
            curve.x[start],
            curve.x[end],
            color="#f2c94c",
            alpha=0.10,
            label=(
                f"broad-background region ({processed.bump_strength})"
                if interval_index == 0
                else None
            ),
        )
    axes[1].legend(loc="best", fontsize=8)
    axes[1].set_ylabel("signal / baseline")

    detection_axis = axes[2]
    detection_axis.plot(
        curve.x,
        processed.positive,
        color="#1f77b4",
        lw=0.8,
        label="preprocessed positive",
    )
    detection_axis.plot(
        curve.x,
        processed.signed,
        color="0.55",
        lw=0.55,
        alpha=0.75,
        label="signed branch",
    )
    for slot_index, slot in enumerate(template):
        detection_axis.axvspan(
            slot["retention_time_min"] - slot["rt_tolerance_min"],
            slot["retention_time_min"] + slot["rt_tolerance_min"],
            color="#66bb6a",
            alpha=0.065,
            label=(
                "T1-T6 template RT windows (not detected peaks)"
                if slot_index == 0
                else None
            ),
        )
    for interval_index, (start, end) in enumerate(
        _contiguous_intervals(processed.bump_mask)
    ):
        detection_axis.axvspan(
            curve.x[start],
            curve.x[end],
            color="#f2c94c",
            alpha=0.085,
            label=(
                f"broad-background region ({processed.bump_strength})"
                if interval_index == 0
                else None
            ),
        )
    confirmed_boundaries = {
        int(spec["feature_id"]): spec
        for spec in _confirmed_boundary_specs(rows)
        if spec["feature_id"] is not None
    }
    styles = {
        "confirmed": ("#2ca02c", "o", "confirmed"),
        "review": ("#ff7f0e", "^", "review"),
        "artifact": ("#d62728", "x", "artifact / interference"),
        "likely_noise": ("#7f7f7f", "D", "likely noise"),
        "background_residual": (
            "#8c8c8c",
            ".",
            "structured background residual",
        ),
    }
    short_plot_names = {
        "normal_positive_peak": "正峰",
        "narrow_positive_peak": "窄正峰",
        "unassigned_positive_peak": "未映射正峰",
        "positive_peak_on_hump": "鼓包上小峰",
        "overlapping_positive_peak": "重叠正峰",
        "secondary_or_overlapping_candidate": "次峰/重叠候选",
        "broad_or_overlapped_peak": "宽峰/重叠峰",
        "broad_positive_peak": "宽正峰",
        "broad_hump_or_baseline": "宽峰/基线",
        "gentle_broad_peak_candidate": "平缓宽峰候选",
        "peak_with_apex_spike": "真实峰/峰顶尖点",
        "negative_peak": "负峰",
        "broad_negative_peak": "宽负峰",
        "interpeak_valley_or_negative_peak": "谷底/负峰",
        "electrical_interference_candidate": "电干扰",
        "electrical_spike": "正向毛刺/电尖峰",
        "negative_electrical_spike": "负向毛刺/电尖峰",
        "uncertain_peak_or_noise": "峰/噪声",
    }
    used: set[str] = set()
    annotation_specs: list[dict] = []
    for row in rows:
        plot_class = row["status"]
        if row["feature_type"] == "structured_background_residual":
            plot_class = "background_residual"
        elif row["status"] == "artifact" and row["feature_type"] == "uncertain_peak_or_noise":
            plot_class = "likely_noise"
        color, marker, legend_text = styles[plot_class]
        y_value = np.interp(row["apex_time_min"], curve.x, processed.positive)
        if row["sign"] == "negative":
            y_value = np.interp(row["apex_time_min"], curve.x, processed.signed)
        boundary = confirmed_boundaries.get(int(row["feature_id"]))
        if boundary is not None:
            boundary_signal = (
                processed.positive if boundary["sign"] == "positive" else processed.signed
            )
            start_y = float(
                np.interp(boundary["start_time_min"], curve.x, boundary_signal)
            )
            end_y = float(np.interp(boundary["end_time_min"], curve.x, boundary_signal))
            boundary_label = (
                "confirmed boundary: start > / < end"
                if "confirmed_boundary" not in used
                else None
            )
            used.add("confirmed_boundary")
            # The two hollow markers point inward. They identify the exact
            # algorithmic boundary samples without drawing full-height lines
            # or shading very wide automatic windows across the whole plot.
            detection_axis.scatter(
                boundary["start_time_min"],
                start_y,
                marker=">",
                s=34,
                facecolors="white",
                edgecolors="#00897b",
                linewidths=0.9,
                label=boundary_label,
                zorder=5,
            )
            detection_axis.scatter(
                boundary["end_time_min"],
                end_y,
                marker="<",
                s=34,
                facecolors="white",
                edgecolors="#00897b",
                linewidths=0.9,
                zorder=5,
            )
        label = legend_text if plot_class not in used else None
        used.add(plot_class)
        detection_axis.scatter(
            row["apex_time_min"],
            y_value,
            c=color,
            marker=marker,
            s=15 if plot_class == "background_residual" else 28,
            label=label,
            zorder=5,
        )
        if row["feature_type"] == "gentle_broad_peak_candidate":
            gentle_label = "gentle broad peak candidate" if "gentle_type" not in used else None
            used.add("gentle_type")
            detection_axis.scatter(
                row["apex_time_min"],
                y_value,
                facecolors="none",
                edgecolors="#9467bd",
                marker="s",
                s=62,
                linewidths=1.15,
                label=gentle_label,
                zorder=6,
            )
        if row["feature_type"] == "peak_with_apex_spike":
            apex_spike_label = (
                "chromatographic peak with apex spike"
                if "apex_spike_type" not in used
                else None
            )
            used.add("apex_spike_type")
            detection_axis.scatter(
                row["apex_time_min"],
                y_value,
                facecolors="none",
                edgecolors="#6a1b9a",
                marker="P",
                s=72,
                linewidths=1.2,
                label=apex_spike_label,
                zorder=7,
            )
        annotate = (
            row["status"] == "confirmed"
            or bool(row["template_slot"])
            or row["feature_type"] == "gentle_broad_peak_candidate"
            or "electrical" in row["feature_type"]
            or row["feature_type"] in {
                "negative_peak",
                "broad_negative_peak",
                "negative_electrical_spike",
            }
            or (
                row["feature_type"] == "interpeak_valley_or_negative_peak"
                and row["relative_prominence"] >= 0.02
            )
            or (row["status"] == "review" and row["relative_prominence"] >= 0.01)
        )
        if annotate:
            short_type = short_plot_names.get(row["feature_type"], row["feature_type_cn"])
            if row["feature_type"] in {
                "gentle_broad_peak_candidate",
                "peak_with_apex_spike",
                "electrical_interference_candidate",
                "electrical_spike",
                "negative_electrical_spike",
            }:
                base_text = short_type
            else:
                base_text = row["component"] or row["template_slot"] or short_type
            annotation_text = f"{base_text}\n{row['peak_confidence_percent']:.0f}%"
            if boundary is not None:
                annotation_text = (
                    f"{base_text} {row['peak_confidence_percent']:.0f}%\n"
                    f"{boundary['apex_text']}\n"
                    f"{boundary['text']}"
                )
            annotation_specs.append(
                {
                    "text": annotation_text,
                    "x": row["apex_time_min"],
                    "y": y_value,
                    "direction": -1 if row["sign"] == "negative" else 1,
                    "priority": _annotation_priority(row),
                    "color": color,
                    "required": boundary is not None,
                }
            )
    detection_axis.axhline(0, color="0.25", lw=0.5)
    lower, upper = detection_axis.get_ylim()
    label_padding = 0.18 * max(upper - lower, EPS)
    detection_axis.set_ylim(lower - label_padding, upper + label_padding)
    detection_axis.set_xlabel("time (min)")
    detection_axis.set_ylabel("corrected signal")
    detection_axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        fontsize=7.2,
        ncol=6,
    )
    fig.tight_layout()
    _place_peak_annotations(detection_axis, annotation_specs)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _raw_zoom_segment_bounds(
    x: np.ndarray,
    segment_count: int = 4,
) -> list[tuple[float, float]]:
    """Return contiguous, equal-duration display intervals for a raw trace."""

    if segment_count < 1:
        raise ValueError("segment_count must be at least 1")
    if len(x) < 2 or not np.all(np.isfinite(x)) or not np.all(np.diff(x) > 0):
        raise ValueError("x must contain at least two finite, increasing samples")
    edges = np.linspace(float(x[0]), float(x[-1]), segment_count + 1)
    return [
        (float(edges[index]), float(edges[index + 1]))
        for index in range(segment_count)
    ]


def _robust_display_limits(values: np.ndarray) -> tuple[float, float]:
    """Compute an outlier-resistant y view without changing plotted samples.

    This deliberately controls only the Matplotlib axis.  The corresponding
    plot still receives the original values, including samples outside this
    display window.  The full-range raw panel in ``plot_result`` remains the
    authoritative view for those extremes.
    """

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        raise ValueError("values must contain at least one finite sample")

    q05, q25, median, q75, q95 = np.quantile(
        finite, [0.05, 0.25, 0.50, 0.75, 0.95]
    )
    iqr = float(q75 - q25)
    low = float(min(q05, q25 - 3.0 * iqr))
    high = float(max(q95, q75 + 3.0 * iqr))

    # Constant or almost-constant segments need a visible, deterministic span.
    scale = max(
        abs(float(median)),
        float(np.ptp(finite)) * 1e-6,
        np.finfo(float).eps * 100.0,
    )
    minimum_span = max(scale * 1e-6, np.finfo(float).eps * 1000.0)
    if high - low < minimum_span:
        low = float(median - minimum_span / 2.0)
        high = float(median + minimum_span / 2.0)

    padding = 0.08 * (high - low)
    return float(low - padding), float(high + padding)


def plot_raw_zoom(
    curve: Curve,
    output: Path,
    segment_count: int = 4,
) -> list[dict]:
    """Plot the untouched raw samples in independent equal-time zoom panels.

    Only the axes are zoomed.  No smoothing, interpolation, impulse removal,
    baseline subtraction, or sample replacement is performed here.
    """

    bounds = _raw_zoom_segment_bounds(curve.x, segment_count)
    fig, axes = plt.subplots(
        segment_count,
        1,
        figsize=(14, 2.35 * segment_count + 1.0),
        squeeze=False,
    )
    axes_1d = axes[:, 0]
    segment_records: list[dict] = []
    for index, ((start, end), axis) in enumerate(zip(bounds, axes_1d)):
        if index + 1 == segment_count:
            mask = (curve.x >= start) & (curve.x <= end)
        else:
            mask = (curve.x >= start) & (curve.x < end)
        x_segment = curve.x[mask]
        y_segment = curve.y[mask]
        if not len(x_segment):
            # This is only relevant to unusually sparse, non-uniform inputs.
            nearest = int(np.argmin(np.abs(curve.x - (start + end) / 2.0)))
            x_segment = curve.x[nearest : nearest + 1]
            y_segment = curve.y[nearest : nearest + 1]

        y_low, y_high = _robust_display_limits(y_segment)
        clipped = int(np.count_nonzero((y_segment < y_low) | (y_segment > y_high)))
        axis.plot(
            x_segment,
            y_segment,
            color="0.22",
            lw=0.72,
            label="raw samples (unprocessed)",
        )
        axis.set_xlim(start, end)
        axis.set_ylim(y_low, y_high)
        axis.set_ylabel("raw signal")
        axis.grid(color="0.85", linewidth=0.45, alpha=0.55)
        axis.set_title(
            f"segment {index + 1}/{segment_count}: {start:.4f}-{end:.4f} min | "
            f"display-axis zoom; {clipped} raw sample(s) outside y view",
            fontsize=9,
        )
        if index == 0:
            axis.legend(loc="best", fontsize=8)
        segment_records.append(
            {
                "segment": index + 1,
                "start_time_min": start,
                "end_time_min": end,
                "sample_count": int(len(y_segment)),
                "display_y_min": y_low,
                "display_y_max": y_high,
                "samples_outside_display_y": clipped,
            }
        )

    fig.suptitle(
        f"{curve.key}\nraw samples, no preprocessing; only display-axis zoom",
        fontsize=12,
    )
    axes_1d[-1].set_xlabel("time (min)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return segment_records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--zip",
        type=Path,
        default=PROJECT_ROOT / "data" / "typical_chromatograms.zip",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reliable_detector_results",
    )
    parser.add_argument(
        "--component-map",
        type=Path,
        default=PROJECT_ROOT / "configs" / "component_map.json",
        help="Optional JSON mapping such as {'T1':'H2', ...}",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "detector_config.json",
        help="JSON thresholds for confidence and candidate filtering",
    )
    args = parser.parse_args()
    config = DetectorConfig.from_json(args.config)
    configure_font()
    args.out.mkdir(parents=True, exist_ok=True)
    plot_dir = args.out / "plots"
    plot_dir.mkdir(exist_ok=True)
    raw_zoom_dir = args.out / "raw_zoom_plots"
    raw_zoom_dir.mkdir(exist_ok=True)

    curves, duplicates = read_curves(args.zip)
    curves = independent_curves(curves)
    references = [curve for curve in curves if curve.folder.startswith("1-")]
    template, reference_processed = train_template(references)
    component_map = load_component_map(args.component_map, template)
    for slot in template:
        slot["component"] = component_map.get(slot["slot"], "")
    reference_noise = float(
        np.median([reference_processed[curve.sha256].global_noise for curve in references])
    )

    all_rows: list[dict] = []
    file_rows: list[dict] = []
    raw_zoom_rows: list[dict] = []
    for curve in curves:
        processed = reference_processed.get(curve.sha256) or preprocess(curve)
        classified = classify_curve(
            curve,
            processed,
            template,
            component_map,
            reference_noise,
            config,
        )
        rows = feature_rows(curve, classified)
        all_rows.extend(rows)
        full_signal_range = max(float(np.ptp(curve.y)), EPS)
        file_rows.append(
            {
                "folder": curve.folder,
                "file": curve.name,
                "sha256": curve.sha256,
                "confirmed_peaks": sum(row["status"] == "confirmed" for row in rows),
                "review_features": sum(row["status"] == "review" for row in rows),
                "artifacts": sum(row["status"] == "artifact" for row in rows),
                "structured_background_residuals": sum(
                    row["structured_background_residual_flag"] for row in rows
                ),
                "confirmed_confidence_mean": float(
                    np.mean([row["peak_confidence"] for row in rows if row["status"] == "confirmed"])
                )
                if any(row["status"] == "confirmed" for row in rows)
                else float("nan"),
                "review_confidence_max": float(
                    np.max([row["peak_confidence"] for row in rows if row["status"] == "review"])
                )
                if any(row["status"] == "review" for row in rows)
                else float("nan"),
                "negative_peaks": sum(
                    row["feature_type"]
                    in {
                        "negative_peak",
                        "broad_negative_peak",
                        "interpeak_valley_or_negative_peak",
                    }
                    for row in rows
                ),
                "electrical_features": sum("electrical" in row["feature_type"] for row in rows),
                "apex_spike_peaks": sum(
                    row["feature_type"] == "peak_with_apex_spike" for row in rows
                ),
                "broad_features": sum(
                    row["feature_type"]
                    in {
                        "broad_or_overlapped_peak",
                        "broad_positive_peak",
                        "broad_hump_or_baseline",
                        "gentle_broad_peak_candidate",
                        "broad_negative_peak",
                    }
                    for row in rows
                ),
                "global_noise_fraction": processed.global_noise,
                "noise_ratio_vs_reference": processed.global_noise / max(reference_noise, EPS),
                "high_noise_flag": high_noise_condition(processed, reference_noise),
                "baseline_drift": processed.baseline_drift,
                "endpoint_drift_ratio": processed.endpoint_drift_ratio,
                "endpoint_drift_abs_ratio": abs(processed.endpoint_drift_ratio),
                "baseline_excursion_ratio": processed.baseline_excursion_ratio,
                "baseline_curvature": processed.baseline_curvature,
                "bump_duration": processed.bump_duration,
                "bump_flag": processed.bump_flag,
                "bump_strength": processed.bump_strength,
                "bump_is_open": processed.bump_is_open,
                "bump_count": processed.bump_count,
                "nonstationary_baseline_flag": nonstationary_background_condition(
                    processed
                ),
                "robust_signal_range": processed.signal_range,
                # Preserve the pre-v0.5 definition for downstream spreadsheets.
                "baseline_drift_fraction_of_range": abs(
                    processed.baseline_drift
                ) / full_signal_range,
                "baseline_drift_flag": abs(processed.baseline_drift)
                / full_signal_range
                >= 0.05,
            }
        )
        safe = f"{curve.folder.split('-', 1)[0]}_{Path(curve.name).stem}_{curve.sha256[:8]}.png"
        plot_result(curve, processed, rows, template, plot_dir / safe)
        zoom_segments = plot_raw_zoom(curve, raw_zoom_dir / safe)
        raw_zoom_rows.append(
            {
                "folder": curve.folder,
                "file": curve.name,
                "sha256": curve.sha256,
                "path": (Path("raw_zoom_plots") / safe).as_posix(),
                "segments": json.dumps(zoom_segments, ensure_ascii=False),
                "说明": (
                    "直接绘制CSV原始采样点，未平滑、未插值、未去毛刺、未扣基线；"
                    "仅将时间等分为4段并为每段独立设置稳健显示纵轴，超出纵轴的原始点未被删除。"
                ),
            }
        )

    features = pd.DataFrame(all_rows, columns=FEATURE_COLUMNS)
    features.to_csv(args.out / "all_detected_features.csv", index=False, encoding="utf-8-sig")
    features[features["status"] == "confirmed"].to_csv(
        args.out / "confirmed_component_peaks.csv", index=False, encoding="utf-8-sig"
    )
    features[features["status"] == "confirmed"].to_csv(
        args.out / "confirmed_peaks.csv", index=False, encoding="utf-8-sig"
    )
    features[features["status"] == "review"].to_csv(
        args.out / "review_required.csv", index=False, encoding="utf-8-sig"
    )
    features[features["status"] != "artifact"].to_csv(
        args.out / "detected_peaks_for_validation.csv", index=False, encoding="utf-8-sig"
    )
    features[features["status"] == "artifact"].to_csv(
        args.out / "interference_candidates.csv", index=False, encoding="utf-8-sig"
    )
    features[features["feature_type"] == "peak_with_apex_spike"].to_csv(
        args.out / "apex_spike_peaks.csv", index=False, encoding="utf-8-sig"
    )
    residuals = features[features["structured_background_residual_flag"]].copy()
    residuals.to_csv(
        args.out / "structured_background_residuals.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ripple_region_columns = [
        "folder",
        "file",
        "ripple_group",
        "start_time_min",
        "end_time_min",
        "candidate_count",
        "maximum_peak_confidence",
        "maximum_relative_prominence",
    ]
    if len(residuals):
        ripple_regions = (
            residuals.groupby(["folder", "file", "ripple_group"], dropna=False)
            .agg(
                start_time_min=("start_time_min", "min"),
                end_time_min=("end_time_min", "max"),
                candidate_count=("feature_id", "size"),
                maximum_peak_confidence=("peak_confidence", "max"),
                maximum_relative_prominence=("relative_prominence", "max"),
            )
            .reset_index()
        )
    else:
        ripple_regions = pd.DataFrame(columns=ripple_region_columns)
    ripple_regions.to_csv(
        args.out / "background_ripple_regions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    confidence_statistics = (
        features.groupby(["feature_type", "feature_type_cn", "status", "status_cn"], dropna=False)
        .agg(
            count=("peak_confidence", "size"),
            confidence_mean=("peak_confidence", "mean"),
            confidence_median=("peak_confidence", "median"),
            confidence_min=("peak_confidence", "min"),
            confidence_max=("peak_confidence", "max"),
            template_confidence_mean=("template_confidence", "mean"),
        )
        .reset_index()
        .sort_values(["status", "confidence_mean"], ascending=[True, False])
    )
    confidence_statistics.to_csv(
        args.out / "confidence_statistics.csv", index=False, encoding="utf-8-sig"
    )
    validation = features[features["status"] != "artifact"].copy()
    validation["human_is_real_peak"] = ""
    validation["human_feature_type"] = ""
    validation["human_apex_time_min"] = ""
    validation["human_start_time_min"] = ""
    validation["human_end_time_min"] = ""
    validation["human_component"] = ""
    validation["human_comment"] = ""
    validation.to_csv(
        args.out / "validation_labels_template.csv", index=False, encoding="utf-8-sig"
    )
    validation[validation["feature_type"] != "uncertain_peak_or_noise"].to_csv(
        args.out / "priority_validation_set.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(file_rows).to_csv(
        args.out / "file_quality_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        raw_zoom_rows,
        columns=["folder", "file", "sha256", "path", "segments", "说明"],
    ).to_csv(
        args.out / "raw_zoom_index.csv", index=False, encoding="utf-8-sig"
    )
    (args.out / "learned_peak_template.json").write_text(
        json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "algorithm_version": __version__,
        "independent_curves": len(curves),
        "reference_curves": len(references),
        "learned_template_slots": len(template),
        "duplicate_groups": duplicates,
        "confirmation_threshold": config.confirmation_threshold,
        "confidence_definition": "peak_confidence is the combined probability-like score for chromatographic peak evidence; template_confidence is reported separately.",
        "detector_config": config.__dict__,
        "baseline_diagnostic_policy": {
            "robust_signal_quantiles": [0.001, 0.999],
            "diagnostic_smoothing_min": 0.20,
            "endpoint_window_fraction": 0.02,
            "endpoint_window_min_limits": [0.10, 0.25],
            "wide_background_sensitive_prominence_fraction": 0.005,
            "wide_background_strong_prominence_fraction": 0.03,
            "wide_background_min_fwhm_min": 0.50,
            "wide_background_min_distance_min": 0.30,
            "weak_candidate_analyte_excess_fraction": 0.03,
            "weak_candidate_max_analyte_overlap_fraction": 0.10,
            "bump_flag_duration_floor_min": 0.50,
            "strong_bump_excursion_floor": 0.05,
            "strong_bump_curvature_floor": 0.04,
            "nonstationary_excursion_floor": 0.08,
            "nonstationary_curvature_floor": 0.04,
            "nonstationary_endpoint_abs_floor": 0.05,
            "ripple_group_max_gap_min": 0.35,
        },
        "electrical_interference_policy": {
            "decision": "Joint physical-time morphology, slope, apex-fit, and counterfactual despiking evidence replace an impulse-overlap hard veto. A retention-time match alone cannot rescue strong electrical morphology.",
            "electrical_score_threshold": config.electrical_spike_score_threshold,
            "peak_body_score_floor_for_apex_spike_rescue": config.apex_spike_peak_body_floor,
            "apex_spike_confidence_multiplier": config.apex_spike_confidence_multiplier,
            "apex_core_half_window_min": 0.005,
            "apex_quadratic_excess_threshold": 0.015,
            "sampling_invariance": {
                "fwhm_samples": "audit_only",
                "top3_energy_fraction": "audit_only",
                "classification_uses": "fwhm_min and apex_core_area_fraction on fixed physical-time bounds",
                "file_impulse_density": "contiguous provisional impulse runs per minute",
            },
            "peak_body_rescue_gate": {
                "absolute_route": {
                    "fwhm_min": 0.060,
                    "area_shape_factor_min": 1.15,
                },
                "template_relative_route": {
                    "width_low_multiplier": 0.75,
                    "width_high_multiplier": 1.25,
                    "area_shape_factor_min": 0.85,
                },
                "symmetry_min": 0.75,
                "top_width_ratio_max": 0.45,
                "baseline_change_ratio_max": 0.25,
                "rise_fall_continuity_min": 0.80,
                "apex_core_area_fraction_max": 0.35,
                "despiked_height_retention_min": 0.65,
                "despiked_prominence_retention_min": 0.65,
                "despiked_area_retention_min": 0.65,
                "apex_quadratic_excess_ratio_max": 0.25,
            },
            "negative_branch_scope": "The v0.5.2 joint electrical morphology scores apply to positive candidates. Negative rows leave those score/retention fields blank and retain the signed-valley rules.",
        },
        "raw_zoom_policy": {
            "directory": "raw_zoom_plots",
            "index": "raw_zoom_index.csv",
            "segments_per_curve": 4,
            "data_transform": "none; original curve.x and curve.y are plotted directly",
            "display_only": "each equal-time segment has an independent robust y-axis; clipped samples remain in the raw data and are counted in the segment title/index",
        },
        "reliability_policy": "Positive and negative events use the same 0.75 confirmation threshold. Every weak positive event uses effective multi-scale noise. On a non-stationary background, weak dense candidates with shallow bilateral evidence become structured background residuals: they stay visible for review and cannot self-confirm. Positive electrical interference uses joint physical width, symmetry, rise/fall continuity and slope balance, fixed-time apex energy, quadratic apex excess, counterfactual despike retention, and impulse-context evidence. Sampling-point counts are audit fields only. Only a conservatively surviving chromatographic body can be labeled peak_with_apex_spike.",
        "chemical_name_limitation": "Fill component_map.json to convert T1..Tn into H2/CO/CH4/etc.",
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
