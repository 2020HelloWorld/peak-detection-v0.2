#!/usr/bin/env python3
"""Run three peak-recognition routes and compare their peak-level outputs.

Routes:
1. adaptive_arpls: arPLS baseline + local MAD noise + prominence/width rules.
2. cwt_multiscale: rolling/symmetric baseline + CWT multiscale candidates.
3. hplc_py: hplc-py SNIP correction + skew-normal mixture fitting.

The output schema is intentionally common so a fourth, external/reference
algorithm can later be imported and compared with the same matching code.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
VENDOR = PROJECT_ROOT / ".vendor"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks_cwt, peak_prominences, peak_widths

from run_open_algorithms import (
    Curve,
    EPS,
    analyze_curve,
    local_noise_sigma,
    odd_at_most,
    read_curves,
)


ALGORITHMS = ["adaptive_arpls", "cwt_multiscale", "hplc_py"]
PEAK_COLUMNS = [
    "algorithm",
    "folder",
    "file",
    "sha256",
    "sign",
    "class",
    "apex_time_min",
    "start_time_min",
    "end_time_min",
    "width_min",
    "height",
    "prominence",
    "snr",
    "area",
    "quality_flags",
]


def independent_curves(curves: list[Curve]) -> list[Curve]:
    seen: set[str] = set()
    output: list[Curve] = []
    for curve in curves:
        if curve.sha256 in seen:
            continue
        seen.add(curve.sha256)
        output.append(curve)
    return output


def integrate_peak(
    x: np.ndarray,
    corrected: np.ndarray,
    start: float,
    end: float,
    sign: str,
) -> float:
    mask = (x >= start) & (x <= end)
    if mask.sum() < 2:
        return 0.0
    values = corrected[mask] if sign == "positive" else -corrected[mask]
    return float(np.trapezoid(np.maximum(values, 0), x[mask]))


def adaptive_rows(
    curve: Curve, peaks: pd.DataFrame, arrays: dict[str, np.ndarray]
) -> list[dict]:
    if peaks.empty:
        return []
    output: list[dict] = []
    selected = peaks[peaks["screened_candidate"]]
    for _, peak in selected.iterrows():
        signed = arrays["positive_corrected"] if peak["sign"] == "positive" else arrays["signed_corrected"]
        flags = []
        if bool(peak.get("hampel_overlap", False)):
            flags.append("hampel_overlap")
        if bool(peak.get("ambiguous_spike_or_negative", False)):
            flags.append("spike_or_negative_ambiguous")
        output.append(
            {
                "algorithm": "adaptive_arpls",
                "folder": curve.folder,
                "file": curve.name,
                "sha256": curve.sha256,
                "sign": peak["sign"],
                "class": peak["class"],
                "apex_time_min": float(peak["retention_time_min"]),
                "start_time_min": float(peak["start_time_min"]),
                "end_time_min": float(peak["end_time_min"]),
                "width_min": float(peak["width_min_95pct"]),
                "height": float(peak["height"]),
                "prominence": float(peak["prominence"]),
                "snr": float(peak["snr"]),
                "area": integrate_peak(
                    curve.x,
                    signed,
                    float(peak["start_time_min"]),
                    float(peak["end_time_min"]),
                    peak["sign"],
                ),
                "quality_flags": ";".join(flags),
            }
        )
    return output


def refine_cwt_indices(work: np.ndarray, candidates: list[int]) -> np.ndarray:
    refined: set[int] = set()
    for candidate in candidates:
        lo = max(0, int(candidate) - 6)
        hi = min(len(work), int(candidate) + 7)
        if hi <= lo:
            continue
        refined.add(lo + int(np.argmax(work[lo:hi])))
    return np.array(sorted(refined), dtype=int)


def cwt_one_sign(
    curve: Curve,
    corrected: np.ndarray,
    noise: np.ndarray,
    sign: int,
    spike_mask: np.ndarray,
) -> list[dict]:
    x = curve.x
    work = corrected if sign > 0 else -corrected
    dt = float(np.median(np.diff(x)))
    widths = np.unique(np.round(np.geomspace(4, min(240, len(work) // 6), 13)).astype(int))
    candidates = find_peaks_cwt(
        work,
        widths,
        max_distances=np.maximum(1, widths / 2),
        gap_thresh=2,
        min_length=3,
        min_snr=2.0,
        noise_perc=20,
    )
    indices = refine_cwt_indices(work, candidates)
    if not len(indices):
        return []
    wlen = odd_at_most(min(1201, len(work) - 1), len(work))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prominences, _, _ = peak_prominences(work, indices, wlen=wlen)
        widths95, _, left_ips, right_ips = peak_widths(work, indices, rel_height=0.95)
    full_scale = max(float(np.quantile(curve.y, 0.999) - np.quantile(curve.y, 0.001)), EPS)
    rows: list[dict] = []
    for i, apex in enumerate(indices):
        if work[apex] <= 2 * noise[apex] or prominences[i] <= 3 * noise[apex]:
            continue
        relative = float(prominences[i] / full_scale)
        width_min = float(widths95[i] * dt)
        if relative < 0.003:
            continue
        if width_min < 0.020:
            peak_class = "spike_candidate" if sign > 0 else "negative_spike_candidate"
        elif width_min > 0.80:
            peak_class = "broad_hump" if sign > 0 else "broad_negative"
        else:
            peak_class = "candidate_peak" if sign > 0 else "negative_candidate"
        # Preserve broad candidates for 5.3 comparison but reject positive
        # spike-like objects from the component-peak comparison.
        if peak_class == "spike_candidate":
            continue
        li = max(0, int(math.floor(left_ips[i])))
        ri = min(len(x) - 1, int(math.ceil(right_ips[i])))
        overlap = bool(spike_mask[max(0, apex - 3) : min(len(x), apex + 4)].any())
        flags = []
        if overlap:
            flags.append("hampel_overlap")
        if sign < 0 and overlap:
            flags.append("spike_or_negative_ambiguous")
        start = float(x[li])
        end = float(x[ri])
        rows.append(
            {
                "algorithm": "cwt_multiscale",
                "folder": curve.folder,
                "file": curve.name,
                "sha256": curve.sha256,
                "sign": "positive" if sign > 0 else "negative",
                "class": peak_class,
                "apex_time_min": float(x[apex]),
                "start_time_min": start,
                "end_time_min": end,
                "width_min": width_min,
                "height": float(work[apex]),
                "prominence": float(prominences[i]),
                "snr": float(prominences[i] / max(noise[apex], EPS)),
                "area": integrate_peak(
                    x,
                    corrected,
                    start,
                    end,
                    "positive" if sign > 0 else "negative",
                ),
                "quality_flags": ";".join(flags),
            }
        )
    return rows


def cwt_rows(curve: Curve, arrays: dict[str, np.ndarray]) -> list[dict]:
    positive = curve.y - arrays["rolling_ball"]
    signed = arrays["signed_corrected"]
    pos_noise = local_noise_sigma(positive, float(np.median(np.diff(curve.x))))
    signed_noise = arrays["signed_noise"]
    return cwt_one_sign(
        curve, positive, pos_noise, +1, arrays["spike_mask"]
    ) + cwt_one_sign(curve, signed, signed_noise, -1, arrays["spike_mask"])


def hplc_rows(curve: Curve) -> tuple[list[dict], dict, np.ndarray | None]:
    from hplc.quant import Chromatogram

    start_time = time.perf_counter()
    result = {
        "algorithm": "hplc_py",
        "folder": curve.folder,
        "file": curve.name,
        "sha256": curve.sha256,
        "status": "ok",
        "error": "",
        "runtime_sec": 0.0,
    }
    frame = pd.DataFrame({"time": curve.x, "signal": curve.y})
    chrom = Chromatogram(frame)
    rows: list[dict] = []
    baseline = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            peaks = chrom.fit_peaks(
                prominence=0.03,
                rel_height=0.95,
                approx_peak_width=0.25,
                buffer=5,
                verbose=False,
                max_iter=100000,
                peak_kwargs={"width": (5, 900)},
            )
        if "estimated_background" in chrom.df:
            baseline = chrom.df["estimated_background"].to_numpy(dtype=float)
        dt = float(np.median(np.diff(curve.x)))
        for _, peak in peaks.iterrows():
            negative = bool(float(peak["area"]) < 0 or float(peak["amplitude"]) < 0)
            sign = "negative" if negative else "positive"
            scale = abs(float(peak["scale"]))
            width = max(2.355 * scale, 3 * dt)
            apex = float(peak["retention_time"])
            start = max(float(curve.x[0]), apex - 3 * scale)
            end = min(float(curve.x[-1]), apex + 3 * scale)
            if width < 0.020:
                peak_class = "negative_spike_candidate" if negative else "spike_candidate"
            elif width > 0.80:
                peak_class = "broad_negative" if negative else "broad_hump"
            else:
                peak_class = "negative_candidate" if negative else "candidate_peak"
            flags = []
            if negative:
                flags.append("hplc_negative_unreliable")
            if abs(float(peak["area"])) * dt < np.ptp(curve.y) * dt * 1e-4:
                flags.append("near_zero_fit")
            rows.append(
                {
                    "algorithm": "hplc_py",
                    "folder": curve.folder,
                    "file": curve.name,
                    "sha256": curve.sha256,
                    "sign": sign,
                    "class": peak_class,
                    "apex_time_min": apex,
                    "start_time_min": start,
                    "end_time_min": end,
                    "width_min": width,
                    "height": abs(float(peak["signal_maximum"]))
                    if abs(float(peak["signal_maximum"])) > EPS
                    else abs(float(peak["amplitude"])),
                    "prominence": np.nan,
                    "snr": np.nan,
                    "area": abs(float(peak["area"])) * dt,
                    "quality_flags": ";".join(flags),
                }
            )
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["runtime_sec"] = time.perf_counter() - start_time
    result["peak_count"] = len(rows)
    return rows, result, baseline


def match_pair(
    left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str
) -> list[dict]:
    rows: list[dict] = []
    for sign in ("positive", "negative"):
        a = left[left["sign"] == sign].reset_index(drop=True)
        b = right[right["sign"] == sign].reset_index(drop=True)
        if a.empty or b.empty:
            continue
        cost = np.abs(
            a["apex_time_min"].to_numpy()[:, None]
            - b["apex_time_min"].to_numpy()[None, :]
        )
        ia, ib = linear_sum_assignment(cost)
        for i, j in zip(ia, ib):
            tolerance = min(
                0.20,
                max(
                    0.03,
                    0.25 * (float(a.loc[i, "width_min"]) + float(b.loc[j, "width_min"])),
                ),
            )
            if cost[i, j] > tolerance:
                continue
            rows.append(
                {
                    "folder": a.loc[i, "folder"],
                    "file": a.loc[i, "file"],
                    "sha256": a.loc[i, "sha256"],
                    "sign": sign,
                    "algorithm_a": left_name,
                    "algorithm_b": right_name,
                    "apex_a_min": float(a.loc[i, "apex_time_min"]),
                    "apex_b_min": float(b.loc[j, "apex_time_min"]),
                    "apex_difference_min": float(cost[i, j]),
                    "width_a_min": float(a.loc[i, "width_min"]),
                    "width_b_min": float(b.loc[j, "width_min"]),
                    "area_a": float(a.loc[i, "area"]),
                    "area_b": float(b.loc[j, "area"]),
                    "match_tolerance_min": tolerance,
                }
            )
    return rows


def compare_files(peaks: pd.DataFrame, curves: list[Curve]) -> tuple[pd.DataFrame, pd.DataFrame]:
    match_rows: list[dict] = []
    summary_rows: list[dict] = []
    for curve in curves:
        per_file = peaks[peaks["sha256"] == curve.sha256]
        summary = {
            "folder": curve.folder,
            "file": curve.name,
            "sha256": curve.sha256,
        }
        for algorithm in ALGORITHMS:
            subset = per_file[per_file["algorithm"] == algorithm]
            summary[f"{algorithm}_peaks"] = len(subset)
            summary[f"{algorithm}_positive"] = int((subset["sign"] == "positive").sum())
            summary[f"{algorithm}_negative"] = int((subset["sign"] == "negative").sum())
            summary[f"{algorithm}_broad"] = int(subset["class"].str.startswith("broad").sum())
        for left_name, right_name in combinations(ALGORITHMS, 2):
            left = per_file[per_file["algorithm"] == left_name]
            right = per_file[per_file["algorithm"] == right_name]
            pair_matches = match_pair(left, right, left_name, right_name)
            match_rows.extend(pair_matches)
            summary[f"matched_{left_name}_vs_{right_name}"] = len(pair_matches)
        summary_rows.append(summary)
    return pd.DataFrame(summary_rows), pd.DataFrame(match_rows)


def problem_label(folder: str) -> str:
    if folder.startswith("1-"):
        return "reference"
    if folder.startswith("2-"):
        return "5.3_broad_and_5.4_overlap"
    if folder.startswith("3-"):
        return "5.1_dynamic_noise"
    if folder.startswith("4-"):
        return "5.6_negative_peak"
    if folder.startswith("5-"):
        return "5.5_electrical_spike"
    return "unclassified"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip",
        type=Path,
        default=PROJECT_ROOT / "data" / "typical_chromatograms.zip",
    )
    parser.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "outputs" / "algorithm_comparison"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    curves, duplicates = read_curves(args.zip)
    curves = independent_curves(curves)
    all_rows: list[dict] = []
    run_rows: list[dict] = []
    baseline_rows: list[dict] = []

    for index, curve in enumerate(curves, 1):
        t0 = time.perf_counter()
        summary, adaptive, arrays = analyze_curve(curve)
        adaptive_output = adaptive_rows(curve, adaptive, arrays)
        cwt_output = cwt_rows(curve, arrays)
        run_rows.append(
            {
                "algorithm": "adaptive_arpls+cwt_multiscale",
                "folder": curve.folder,
                "file": curve.name,
                "sha256": curve.sha256,
                "status": "ok",
                "error": "",
                "runtime_sec": time.perf_counter() - t0,
                "peak_count": len(adaptive_output) + len(cwt_output),
            }
        )
        hplc_output, hplc_run, hplc_baseline = hplc_rows(curve)
        run_rows.append(hplc_run)
        all_rows.extend(adaptive_output)
        all_rows.extend(cwt_output)
        all_rows.extend(hplc_output)

        baselines = {
            "arpls": arrays["arpls"],
            "rolling_ball": arrays["rolling_ball"],
            "snip_pybaselines": arrays["snip"],
        }
        if hplc_baseline is not None:
            baselines["snip_hplc_py"] = hplc_baseline
        for name, baseline in baselines.items():
            baseline_rows.append(
                {
                    "folder": curve.folder,
                    "file": curve.name,
                    "sha256": curve.sha256,
                    "baseline_algorithm": name,
                    "start": float(baseline[0]),
                    "end": float(baseline[-1]),
                    "change": float(baseline[-1] - baseline[0]),
                    "range": float(np.ptp(baseline)),
                    "mean_abs_residual": float(np.mean(np.abs(curve.y - baseline))),
                }
            )
        print(f"[{index:02d}/{len(curves)}] {curve.key}")

    peaks = pd.DataFrame(all_rows, columns=PEAK_COLUMNS)
    peaks.to_csv(args.out / "all_algorithm_peaks.csv", index=False, encoding="utf-8-sig")
    runs = pd.DataFrame(run_rows)
    runs.to_csv(args.out / "algorithm_runs.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(baseline_rows).to_csv(
        args.out / "baseline_comparison.csv", index=False, encoding="utf-8-sig"
    )

    file_comparison, matches = compare_files(peaks, curves)
    file_comparison["problem_group"] = file_comparison["folder"].map(problem_label)
    file_comparison.to_csv(
        args.out / "file_algorithm_comparison.csv", index=False, encoding="utf-8-sig"
    )
    matches.to_csv(args.out / "pairwise_peak_matches.csv", index=False, encoding="utf-8-sig")

    folder_summary = (
        peaks.assign(problem_group=peaks["folder"].map(problem_label))
        .groupby(["problem_group", "algorithm"], as_index=False)
        .agg(
            total_peaks=("apex_time_min", "size"),
            files_with_peaks=("sha256", "nunique"),
            positive_peaks=("sign", lambda x: int((x == "positive").sum())),
            negative_peaks=("sign", lambda x: int((x == "negative").sum())),
            broad_peaks=("class", lambda x: int(x.str.startswith("broad").sum())),
            median_width_min=("width_min", "median"),
        )
    )
    folder_summary.to_csv(
        args.out / "problem_algorithm_summary.csv", index=False, encoding="utf-8-sig"
    )

    manifest = {
        "input_zip": str(args.zip),
        "independent_curves": len(curves),
        "algorithms": ALGORITHMS,
        "duplicate_groups": duplicates,
        "ground_truth_available": False,
        "comparison_meaning": "Agreement is not correctness; pairwise matches only show algorithm consistency.",
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
