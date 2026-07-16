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
    impulse_clean: np.ndarray
    impulse_mask: np.ndarray
    baseline: np.ndarray
    hump_baseline: np.ndarray
    positive: np.ndarray
    hump_positive: np.ndarray
    signed_baseline: np.ndarray
    signed: np.ndarray
    noise_positive: np.ndarray
    noise_hump: np.ndarray
    noise_signed: np.ndarray
    global_noise: float
    baseline_drift: float


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
    "start_time_min",
    "end_time_min",
    "width_min",
    "fwhm_min",
    "top_width_ratio",
    "height",
    "prominence",
    "relative_prominence",
    "snr",
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
    "electrical_interference_candidate": "电信号干扰候选",
    "electrical_spike": "正向电尖峰",
    "negative_electrical_spike": "负向电尖峰",
    "negative_peak": "负峰",
    "broad_negative_peak": "宽负峰",
    "interpeak_valley_or_negative_peak": "峰间谷底或负峰",
    "uncertain_peak_or_noise": "峰或噪声（待复核）",
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
    hump_baseline = np.maximum(rolling_baseline, signed_baseline)
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
    # Compare noise between files only after normalizing by that file's useful
    # signal range. Absolute detector units vary by orders of magnitude here.
    global_noise = float(np.median(noise_positive) / signal_range)
    baseline_drift = float(baseline[-1] - baseline[0])
    return Preprocessed(
        impulse_clean,
        impulse_mask,
        baseline,
        hump_baseline,
        positive,
        hump_positive,
        signed_baseline,
        signed,
        noise_positive,
        noise_hump,
        noise_signed,
        global_noise,
        baseline_drift,
    )


def candidate_features(
    curve: Curve,
    processed: Preprocessed,
    sign: int,
    min_relative_prominence: float = 0.0005,
) -> list[dict]:
    x = curve.x
    work = processed.positive if sign > 0 else -processed.signed
    noise = processed.noise_positive if sign > 0 else processed.noise_signed
    dt = float(np.median(np.diff(x)))
    signal_range = max(float(np.quantile(curve.y, 0.999) - np.quantile(curve.y, 0.001)), EPS)
    raw_window = odd_at_most(round(0.015 / dt), len(curve.y), minimum=5)
    raw_smooth = savgol_filter(curve.y, raw_window, 3, mode="interp")
    minimum_prominence = np.maximum(3.0 * noise, min_relative_prominence * signal_range)
    wlen = odd_at_most(min(round(2.0 / dt), len(work) - 1), len(work))
    indices, props = find_peaks(
        work,
        height=np.maximum(2.0 * noise, min_relative_prominence * signal_range),
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
        impulse_overlap = bool(
            processed.impulse_mask[max(0, apex - 3) : min(len(x), apex + 4)].any()
        )
        baseline_local = processed.hump_baseline[li : ri + 1]
        baseline_change_ratio = (
            float(np.ptp(baseline_local)) / max(float(work[apex]), EPS)
            if len(baseline_local)
            else 0.0
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
                "width_min": float(widths95[i] * dt),
                "fwhm_min": float(widths50[i] * dt),
                "top_width_ratio": float(widths10[i] / max(widths50[i], EPS)),
                "height": float(work[apex]),
                "prominence": float(props["prominences"][i]),
                "relative_prominence": float(props["prominences"][i] / signal_range),
                "snr": float(props["prominences"][i] / max(noise[apex], EPS)),
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


def composite_peak_scores(
    feature: dict,
    template: list[dict],
    sign: int,
    high_noise: bool,
    config: DetectorConfig,
) -> dict[str, float]:
    snr_score = log_evidence(feature["snr"], 3.0, 30.0)
    prominence_score = log_evidence(
        feature["relative_prominence"],
        config.normal_relative_prominence_floor,
        0.03,
    )
    width_score = best_width_score(feature, template)
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
    high_noise = high_noise_condition(processed, reference_noise)
    classified: list[dict] = []

    for feature in positive:
        slot, rt_error = nearest_slot(feature, template)
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
        if slot is not None:
            component = component_map.get(slot["slot"], "")
            width_ok = slot["width_low_min"] <= feature["width_min"] <= slot["width_high_min"]
            symmetry_ok = feature["symmetry"] >= slot["symmetry_low"]
            if width_ok and symmetry_ok:
                feature_type = "normal_positive_peak"
                if feature["baseline_change_ratio"] >= 0.25:
                    feature_type = "positive_peak_on_hump"
            elif feature["width_min"] < slot["width_low_min"]:
                if (
                    feature["impulse_overlap"]
                    or feature["top_width_ratio"] >= 0.55
                    or not symmetry_ok
                ):
                    feature_type = "electrical_interference_candidate"
                    scores["peak_confidence"] *= 0.45
                else:
                    feature_type = "narrow_positive_peak"
            elif feature["width_min"] > slot["width_high_min"]:
                feature_type = "broad_or_overlapped_peak"
            else:
                feature_type = "uncertain_peak_or_noise"
        else:
            if feature["width_min"] < 0.025 or feature["impulse_overlap"]:
                feature_type = "electrical_spike"
                scores["peak_confidence"] *= 0.25
            elif feature["width_min"] > 0.80:
                feature_type = "broad_hump_or_baseline"
                scores["peak_confidence"] *= 0.80
            elif scores["score_width"] >= 0.45 and feature["symmetry"] >= 0.20:
                feature_type = "unassigned_positive_peak"
                if feature["baseline_change_ratio"] >= 0.25:
                    feature_type = "positive_peak_on_hump"
            else:
                feature_type = "uncertain_peak_or_noise"

        scores["peak_confidence"] = clip01(scores["peak_confidence"])
        status = status_from_confidence(scores["peak_confidence"], feature_type, config)
        if feature_type == "broad_hump_or_baseline" and status == "confirmed":
            feature_type = "broad_positive_peak"
        if feature_type == "uncertain_peak_or_noise" and status == "confirmed":
            feature_type = "unassigned_positive_peak"
        reasons = confidence_reasons(scores, template_confidence, rt_score, config)
        if feature["baseline_change_ratio"] >= 0.25:
            reasons.append("local_baseline_changes_across_peak")
        if feature["impulse_overlap"]:
            reasons.append("impulse_overlap_penalty")
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
            left["feature_type"] = "overlapping_positive_peak"
            right["feature_type"] = "overlapping_positive_peak"
            left["reasons"].append("window_overlaps_next_confirmed_peak")
            right["reasons"].append("window_overlaps_previous_confirmed_peak")
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
            "start_time_min": feature["start_time_min"],
            "end_time_min": feature["end_time_min"],
            "width_min": feature["width_min"],
            "fwhm_min": feature["fwhm_min"],
            "top_width_ratio": feature["top_width_ratio"],
            "height": feature["height"],
            "prominence": feature["prominence"],
            "relative_prominence": feature["relative_prominence"],
            "snr": feature["snr"],
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


def plot_result(
    curve: Curve,
    processed: Preprocessed,
    rows: list[dict],
    template: list[dict],
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(curve.x, curve.y, color="0.30", lw=0.75, label="raw")
    axes[0].plot(curve.x, processed.baseline, color="#d62728", lw=1.0, label="arPLS baseline")
    axes[0].plot(
        curve.x,
        processed.hump_baseline,
        color="#9467bd",
        lw=0.7,
        alpha=0.75,
        label="rolling-ball local background",
    )
    axes[0].legend(loc="best", fontsize=8)
    axes[0].set_ylabel("signal")
    axes[0].set_title(curve.key)

    axes[1].plot(curve.x, processed.positive, color="#1f77b4", lw=0.8, label="preprocessed positive")
    axes[1].plot(curve.x, processed.signed, color="0.55", lw=0.55, alpha=0.75, label="signed branch")
    for slot in template:
        axes[1].axvspan(
            slot["retention_time_min"] - slot["rt_tolerance_min"],
            slot["retention_time_min"] + slot["rt_tolerance_min"],
            color="#2ca02c",
            alpha=0.045,
        )
    styles = {
        "confirmed": ("#2ca02c", "o"),
        "review": ("#ff7f0e", "^"),
        "artifact": ("#d62728", "x"),
    }
    short_plot_names = {
        "unassigned_positive_peak": "未映射正峰",
        "secondary_or_overlapping_candidate": "次峰/重叠候选",
        "broad_or_overlapped_peak": "宽峰/重叠峰",
        "broad_hump_or_baseline": "宽峰/基线",
        "interpeak_valley_or_negative_peak": "谷底/负峰",
        "electrical_interference_candidate": "电干扰",
        "uncertain_peak_or_noise": "峰/噪声",
    }
    used: set[str] = set()
    last_annotation_time = -float("inf")
    annotation_level = 0
    for row in rows:
        color, marker = styles[row["status"]]
        y_value = np.interp(row["apex_time_min"], curve.x, processed.positive)
        if row["sign"] == "negative":
            y_value = np.interp(row["apex_time_min"], curve.x, processed.signed)
        label = row["status"] if row["status"] not in used else None
        used.add(row["status"])
        axes[1].scatter(row["apex_time_min"], y_value, c=color, marker=marker, s=28, label=label, zorder=5)
        annotate = (
            row["status"] == "confirmed"
            or bool(row["template_slot"])
            or row["relative_prominence"] >= 0.003
            or "negative_peak" in row["feature_type"]
            or "electrical" in row["feature_type"]
        )
        if annotate:
            if row["apex_time_min"] - last_annotation_time < 0.28:
                annotation_level = (annotation_level + 1) % 4
            else:
                annotation_level = 0
            last_annotation_time = row["apex_time_min"]
            base_text = (
                row["component"]
                or row["template_slot"]
                or short_plot_names.get(row["feature_type"], row["feature_type_cn"])
            )
            text = f"{base_text} {row['peak_confidence_percent']:.0f}%"
            direction = 1 if y_value >= 0 else -1
            offset = direction * (7 + 11 * annotation_level)
            axes[1].annotate(
                text,
                (row["apex_time_min"], y_value),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if direction > 0 else "top",
                fontsize=7,
            )
    axes[1].axhline(0, color="0.25", lw=0.5)
    lower, upper = axes[1].get_ylim()
    label_padding = 0.12 * max(upper - lower, EPS)
    axes[1].set_ylim(lower - label_padding, upper + label_padding)
    axes[1].set_xlabel("time (min)")
    axes[1].set_ylabel("corrected signal")
    axes[1].legend(loc="best", fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


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
        signal_range = max(float(np.ptp(curve.y)), EPS)
        file_rows.append(
            {
                "folder": curve.folder,
                "file": curve.name,
                "sha256": curve.sha256,
                "confirmed_peaks": sum(row["status"] == "confirmed" for row in rows),
                "review_features": sum(row["status"] == "review" for row in rows),
                "artifacts": sum(row["status"] == "artifact" for row in rows),
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
                "broad_features": sum("broad" in row["feature_type"] or "hump" in row["feature_type"] for row in rows),
                "global_noise_fraction": processed.global_noise,
                "noise_ratio_vs_reference": processed.global_noise / max(reference_noise, EPS),
                "high_noise_flag": high_noise_condition(processed, reference_noise),
                "baseline_drift": processed.baseline_drift,
                "baseline_drift_fraction_of_range": abs(processed.baseline_drift) / signal_range,
                "baseline_drift_flag": abs(processed.baseline_drift) / signal_range >= 0.05,
            }
        )
        safe = f"{curve.folder.split('-', 1)[0]}_{Path(curve.name).stem}_{curve.sha256[:8]}.png"
        plot_result(curve, processed, rows, template, plot_dir / safe)

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
        "reliability_policy": "Positive and negative events use the same composite confidence threshold; morphology-based interference penalties lower the peak confidence.",
        "chemical_name_limitation": "Fill component_map.json to convert T1..Tn into H2/CO/CH4/etc.",
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
