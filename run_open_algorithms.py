#!/usr/bin/env python3
"""Prototype open-source chromatogram algorithms on the supplied CSV archive.

This is an evaluation harness, not a claim of validated peak identification.  It
uses pybaselines for background estimation, SciPy for robust filtering and
signed peak detection, and hplc-py for chromatogram-specific SNIP correction
and skew-normal mixture fitting on representative cases.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / ".vendor"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pybaselines import Baseline
from scipy.ndimage import binary_dilation, median_filter
from scipy.signal import find_peaks, peak_widths, savgol_filter


EPS = np.finfo(float).eps


@dataclass
class Curve:
    folder: str
    name: str
    sha256: str
    x: np.ndarray
    y: np.ndarray

    @property
    def key(self) -> str:
        return f"{self.folder}/{self.name}"


def odd_at_most(value: int, n: int, minimum: int = 3) -> int:
    """Return an odd window length that is valid for an array of length n."""
    value = max(minimum, int(value))
    value += (value + 1) % 2
    cap = n if n % 2 else n - 1
    return max(minimum, min(value, cap))


def read_curves(zip_path: Path) -> tuple[list[Curve], list[list[str]]]:
    curves: list[Curve] = []
    hash_groups: dict[str, list[str]] = {}
    with ZipFile(zip_path) as archive:
        for member in archive.namelist():
            if not member.lower().endswith(".csv"):
                continue
            raw = archive.read(member)
            digest = hashlib.sha256(raw).hexdigest()
            frame = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
            if list(frame.columns) != ["x", "Curvel"]:
                raise ValueError(f"Unexpected columns in {member}: {list(frame.columns)}")
            x = frame["x"].to_numpy(dtype=float)
            y = frame["Curvel"].to_numpy(dtype=float)
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
                raise ValueError(f"Non-finite values in {member}")
            folder, name = member.split("/", 1)
            curves.append(Curve(folder, name, digest, x, y))
            hash_groups.setdefault(digest, []).append(member)
    duplicates = [names for names in hash_groups.values() if len(names) > 1]
    return curves, duplicates


def estimate_spikes(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hampel-style isolated impulse detector using rolling median and MAD."""
    n = len(y)
    med_window = odd_at_most(11, n)
    scale_window = odd_at_most(31, n)
    local_median = median_filter(y, size=med_window, mode="nearest")
    residual = np.abs(y - local_median)
    local_mad = median_filter(residual, size=scale_window, mode="nearest")
    nonzero = residual[residual > 0]
    if len(nonzero):
        # The range-based floor prevents a perfectly smooth chromatographic
        # apex from being called an impulse merely because its local MAD is 0.
        floor = max(float(np.quantile(nonzero, 0.20)), np.ptp(y) * 5e-4, EPS)
    else:
        floor = max(np.ptp(y) * 5e-4, EPS)
    sigma = 1.4826 * np.maximum(local_mad, floor)
    raw_mask = residual > 8.0 * sigma
    # Expand one sample so that an impulse edge is not left in the working signal.
    mask = binary_dilation(raw_mask, iterations=1)
    cleaned = y.copy()
    cleaned[mask] = local_median[mask]
    return cleaned, mask


def symmetric_baseline(y: np.ndarray, dt_min: float) -> np.ndarray:
    """Median plus Savitzky-Golay baseline that does not privilege peak sign."""
    n = len(y)
    # A one-minute median window removes normal narrow peaks of either sign while
    # retaining slow drift and broad humps.
    median_points = odd_at_most(round(1.0 / dt_min), n)
    smooth_points = odd_at_most(round(0.35 / dt_min), n)
    base = median_filter(y, size=median_points, mode="nearest")
    if smooth_points >= 5:
        base = savgol_filter(base, smooth_points, polyorder=2, mode="interp")
    return base


def local_noise_sigma(signal: np.ndarray, dt_min: float) -> np.ndarray:
    """Estimate time-varying noise from the robust scale of first differences."""
    n = len(signal)
    diff = np.diff(signal, prepend=signal[0])
    window = odd_at_most(round(0.6 / dt_min), n)
    local_center = median_filter(diff, size=window, mode="nearest")
    local_mad = median_filter(np.abs(diff - local_center), size=window, mode="nearest")
    sigma = local_mad / (0.67448975 * math.sqrt(2.0))
    positive = sigma[sigma > 0]
    if len(positive):
        floor = max(float(np.quantile(positive, 0.10)), np.ptp(signal) * 1e-9, EPS)
    else:
        floor = max(np.ptp(signal) * 1e-9, EPS)
    return np.maximum(sigma, floor)


def detect_signed_peaks(
    x: np.ndarray,
    corrected: np.ndarray,
    noise: np.ndarray,
    sign: int,
    spike_mask: np.ndarray,
    full_scale: float,
) -> list[dict[str, float | int | str]]:
    """Detect positive or negative candidates with a local three-sigma threshold."""
    work = corrected if sign > 0 else -corrected
    height = 2.0 * noise
    prominence = 3.0 * noise
    wlen = odd_at_most(min(1201, len(work) - 1), len(work))
    indices, props = find_peaks(
        work,
        height=height,
        prominence=prominence,
        width=(3, None),
        distance=3,
        wlen=wlen,
    )
    if not len(indices):
        return []
    widths, _, left_ips, right_ips = peak_widths(work, indices, rel_height=0.95)
    dt_min = float(np.median(np.diff(x)))
    records: list[dict[str, float | int | str]] = []
    for i, apex in enumerate(indices):
        li = max(0, int(math.floor(left_ips[i])))
        ri = min(len(x) - 1, int(math.ceil(right_ips[i])))
        width_min = float(widths[i] * dt_min)
        spike_overlap = bool(spike_mask[max(0, apex - 3) : min(len(x), apex + 4)].any())
        relative_prominence = float(props["prominences"][i] / max(full_scale, EPS))
        if width_min < 0.020:
            label = "spike_candidate" if sign > 0 else "negative_spike_candidate"
        elif width_min > 0.80:
            label = "broad_hump" if sign > 0 else "broad_negative"
        else:
            label = "candidate_peak" if sign > 0 else "negative_candidate"
        screened = bool(
            label in {"candidate_peak", "negative_candidate"}
            and props["prominences"][i] / max(noise[apex], EPS) >= 5.0
            and relative_prominence >= 0.003
            # A known-component negative peak can look exactly like a Hampel
            # impulse; keep it visible and mark the ambiguity for later use of
            # component/retention-time constraints.
            and (sign < 0 or not spike_overlap)
        )
        records.append(
            {
                "sign": "positive" if sign > 0 else "negative",
                "class": label,
                "apex_index": int(apex),
                "retention_time_min": float(x[apex]),
                "height": float(work[apex]),
                "prominence": float(props["prominences"][i]),
                "relative_prominence": relative_prominence,
                "snr": float(props["prominences"][i] / max(noise[apex], EPS)),
                "width_min_95pct": width_min,
                "start_time_min": float(x[li]),
                "end_time_min": float(x[ri]),
                "hampel_overlap": spike_overlap,
                "ambiguous_spike_or_negative": bool(sign < 0 and spike_overlap),
                "screened_candidate": screened,
            }
        )
    return records


def analyze_curve(curve: Curve) -> tuple[dict, pd.DataFrame, dict[str, np.ndarray]]:
    x, y = curve.x, curve.y
    dt_min = float(np.median(np.diff(x)))
    cleaned, spike_mask = estimate_spikes(y)
    fitter = Baseline(x_data=x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arpls, _ = fitter.arpls(cleaned, lam=1e7, max_iter=50)
        rolling, _ = fitter.rolling_ball(
            cleaned,
            half_window=max(10, round(0.50 / dt_min)),
            smooth_half_window=max(2, round(0.03 / dt_min)),
        )
        snip, _ = fitter.snip(
            cleaned,
            max_half_window=max(10, round(0.25 / dt_min)),
            smooth_half_window=max(1, round(0.01 / dt_min)),
        )
    # Do not destructively despike the signed branch: a narrow, legitimate
    # negative chromatographic peak can resemble an electrical impulse.  The
    # Hampel output is therefore retained only as a feature/flag here.
    signed_base = symmetric_baseline(y, dt_min)
    positive_corrected = cleaned - arpls
    signed_corrected = y - signed_base
    pos_noise = local_noise_sigma(positive_corrected, dt_min)
    signed_noise = local_noise_sigma(signed_corrected, dt_min)
    full_scale = float(np.quantile(cleaned, 0.999) - np.quantile(cleaned, 0.001))
    peak_records = detect_signed_peaks(
        x, positive_corrected, pos_noise, +1, spike_mask, full_scale
    ) + detect_signed_peaks(
        x, signed_corrected, signed_noise, -1, spike_mask, full_scale
    )
    peaks = pd.DataFrame(peak_records)
    if not peaks.empty:
        peaks.insert(0, "file", curve.name)
        peaks.insert(0, "folder", curve.folder)
        peaks.insert(2, "sha256", curve.sha256)
    summary = {
        "folder": curve.folder,
        "file": curve.name,
        "sha256": curve.sha256,
        "n_points": len(x),
        "duration_min": float(x[-1] - x[0]),
        "sample_interval_sec": dt_min * 60.0,
        "y_min": float(y.min()),
        "y_max": float(y.max()),
        "spike_points": int(spike_mask.sum()),
        "noise_sigma_median": float(np.median(pos_noise)),
        "baseline_change": float(arpls[-1] - arpls[0]),
        "positive_candidates": int(
            0 if peaks.empty else (peaks["sign"] == "positive").sum()
        ),
        "negative_candidates": int(
            0 if peaks.empty else (peaks["sign"] == "negative").sum()
        ),
        "broad_candidates": int(
            0 if peaks.empty else peaks["class"].str.startswith("broad").sum()
        ),
        "spike_candidates": int(
            0 if peaks.empty else peaks["class"].str.contains("spike").sum()
        ),
        "screened_positive": int(
            0
            if peaks.empty
            else ((peaks["sign"] == "positive") & peaks["screened_candidate"]).sum()
        ),
        "screened_negative": int(
            0
            if peaks.empty
            else ((peaks["sign"] == "negative") & peaks["screened_candidate"]).sum()
        ),
    }
    arrays = {
        "cleaned": cleaned,
        "spike_mask": spike_mask,
        "arpls": arpls,
        "rolling_ball": rolling,
        "snip": snip,
        "signed_baseline": signed_base,
        "positive_corrected": positive_corrected,
        "signed_corrected": signed_corrected,
        "positive_noise": pos_noise,
        "signed_noise": signed_noise,
    }
    return summary, peaks, arrays


def safe_stem(curve: Curve) -> str:
    prefix = curve.folder.split("-", 1)[0]
    digest = curve.sha256[:8]
    return f"type{prefix}_{Path(curve.name).stem}_{digest}"


def configure_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def detail_plot(
    curve: Curve,
    peaks: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    output: Path,
) -> None:
    x, y = curve.x, curve.y
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(x, y, color="0.25", lw=0.9, label="raw")
    axes[0].plot(x, arrays["arpls"], color="#d62728", lw=1.2, label="arPLS")
    axes[0].plot(
        x,
        arrays["signed_baseline"],
        color="#2ca02c",
        lw=1.0,
        label="symmetric median",
    )
    axes[0].plot(
        x, arrays["rolling_ball"], color="#1f77b4", lw=0.9, label="rolling-ball"
    )
    spike_idx = np.flatnonzero(arrays["spike_mask"])
    if len(spike_idx):
        axes[0].scatter(x[spike_idx], y[spike_idx], s=8, c="#ff7f0e", label="Hampel flags")
    axes[0].legend(loc="best", ncol=4, fontsize=8)
    axes[0].set_ylabel("signal")
    axes[0].set_title(curve.key)

    axes[1].plot(x, arrays["positive_corrected"], color="#1f77b4", lw=0.8)
    axes[1].fill_between(
        x,
        -3 * arrays["positive_noise"],
        3 * arrays["positive_noise"],
        color="0.7",
        alpha=0.3,
        label="local ±3σ",
    )
    if not peaks.empty:
        pos = peaks[(peaks["sign"] == "positive") & peaks["screened_candidate"]]
        axes[1].scatter(
            pos["retention_time_min"],
            pos["height"],
            c=np.where(pos["class"].str.contains("spike"), "#ff7f0e", "#d62728"),
            s=24,
            zorder=4,
        )
    axes[1].axhline(0, color="0.3", lw=0.6)
    axes[1].set_ylabel("arPLS corrected")
    axes[1].legend(loc="best", fontsize=8)

    axes[2].plot(x, arrays["signed_corrected"], color="#2ca02c", lw=0.8)
    axes[2].fill_between(
        x,
        -3 * arrays["signed_noise"],
        3 * arrays["signed_noise"],
        color="0.7",
        alpha=0.3,
    )
    if not peaks.empty:
        neg = peaks[(peaks["sign"] == "negative") & peaks["screened_candidate"]]
        axes[2].scatter(
            neg["retention_time_min"],
            -neg["height"],
            c="#9467bd",
            marker="v",
            s=28,
            zorder=4,
            label="negative candidates",
        )
    axes[2].axhline(0, color="0.3", lw=0.6)
    axes[2].set_ylabel("signed corrected")
    axes[2].set_xlabel("time (min)")
    if not peaks.empty and (peaks["sign"] == "negative").any():
        axes[2].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def category_sheet(
    folder: str,
    rows: list[tuple[Curve, pd.DataFrame, dict[str, np.ndarray]]],
    output: Path,
) -> None:
    cols = 3
    nrows = math.ceil(len(rows) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(16, 3.2 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, (curve, peaks, arrays) in zip(axes.ravel(), rows):
        ax.set_visible(True)
        scale = max(float(np.nanpercentile(np.abs(curve.y), 99.5)), EPS)
        ax.plot(curve.x, curve.y / scale, color="0.35", lw=0.65)
        ax.plot(curve.x, arrays["arpls"] / scale, color="#d62728", lw=0.8)
        if not peaks.empty:
            pos = peaks[(peaks["sign"] == "positive") & peaks["screened_candidate"]]
            neg = peaks[(peaks["sign"] == "negative") & peaks["screened_candidate"]]
            ax.scatter(
                pos["retention_time_min"],
                np.interp(pos["retention_time_min"], curve.x, curve.y) / scale,
                c="#1f77b4",
                s=11,
            )
            ax.scatter(
                neg["retention_time_min"],
                np.interp(neg["retention_time_min"], curve.x, curve.y) / scale,
                c="#9467bd",
                marker="v",
                s=13,
            )
        ax.set_title(curve.name, fontsize=8)
        ax.set_xlabel("min", fontsize=7)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{folder} | raw normalized, arPLS baseline, signed candidates", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output, dpi=140)
    plt.close(fig)


def choose_representatives(curves: list[Curve]) -> list[Curve]:
    patterns = [
        "2-819机型-基准2#标气",
        "A1-C2H2鼓包",
        "A10-CH4赶在鼓包上",
        "A14-双8#柱通道1赶上鼓包",
        "B10-1噪音、波动比较多",
        "B11-噪音、波动比较多",
        "H1-信号干扰向上谱图",
        "H2-信号干扰向下干扰",
        "H3-C2H2位置受干扰",
        "F1-H2负峰-处理",
        "F2-CO2负峰1",
    ]
    selected: list[Curve] = []
    for pattern in patterns:
        match = next((c for c in curves if pattern in c.name), None)
        if match is not None:
            selected.append(match)
    return selected


def run_hplc_case(curve: Curve, output_dir: Path) -> dict:
    """Run unmodified hplc-py peak fitting on one representative curve."""
    from hplc.quant import Chromatogram

    x, y = curve.x, curve.y
    # Feed hplc-py the original curve so this benchmark reflects the library's
    # own baseline and detection behavior, without our preprocessing helping it.
    frame = pd.DataFrame({"time": x, "signal": y})
    chrom = Chromatogram(frame)
    result = {
        "folder": curve.folder,
        "file": curve.name,
        "sha256": curve.sha256,
        "status": "ok",
        "peak_count": 0,
        "error": "",
    }
    try:
        peaks = chrom.fit_peaks(
            prominence=0.03,
            rel_height=0.95,
            approx_peak_width=0.25,
            buffer=5,
            verbose=False,
            max_iter=100000,
            peak_kwargs={"width": (5, 900)},
        )
        result["peak_count"] = int(len(peaks))
        peaks.to_csv(output_dir / f"{safe_stem(curve)}_hplc_peaks.csv", index=False)
        fig, ax = plt.subplots(figsize=(14, 4.8))
        ax.plot(x, y, color="0.75", lw=0.7, label="raw")
        if "estimated_background" in chrom.df:
            ax.plot(x, chrom.df["estimated_background"], color="#d62728", lw=1.0, label="SNIP baseline")
        ax.plot(x, chrom.df[chrom.int_col], color="#1f77b4", lw=0.8, label="corrected")
        if chrom.unmixed_chromatograms is not None:
            for i in range(chrom.unmixed_chromatograms.shape[1]):
                ax.plot(x, chrom.unmixed_chromatograms[:, i], lw=0.75, alpha=0.8)
        if not peaks.empty:
            ax.scatter(
                peaks["retention_time"],
                peaks["signal_maximum"],
                c="#111111",
                marker="x",
                s=28,
                label="hplc-py fitted peaks",
            )
        ax.set_title(f"hplc-py 0.2.8 | {curve.key}")
        ax.set_xlabel("time (min)")
        ax.set_ylabel("signal")
        ax.legend(loc="best", fontsize=8, ncol=4)
        fig.tight_layout()
        fig.savefig(output_dir / f"{safe_stem(curve)}_hplc.png", dpi=150)
        plt.close(fig)
    except Exception as exc:  # Keep batch evaluation going and record failures.
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip",
        type=Path,
        default=Path(r"D:\色谱峰处理算法\一些典型的谱图.zip"),
        help="Path to the supplied ZIP archive",
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "algorithm_results", help="Output directory"
    )
    parser.add_argument(
        "--skip-hplc", action="store_true", help="Skip slower hplc-py representative fits"
    )
    args = parser.parse_args()
    configure_chinese_font()
    args.out.mkdir(parents=True, exist_ok=True)
    detail_dir = args.out / "details"
    detail_dir.mkdir(exist_ok=True)
    hplc_dir = args.out / "hplc_py"
    hplc_dir.mkdir(exist_ok=True)

    curves, duplicates = read_curves(args.zip)
    seen: set[str] = set()
    independent = []
    for curve in curves:
        if curve.sha256 not in seen:
            independent.append(curve)
            seen.add(curve.sha256)

    summaries: list[dict] = []
    all_peaks: list[pd.DataFrame] = []
    analyzed: dict[str, tuple[Curve, pd.DataFrame, dict[str, np.ndarray]]] = {}
    for curve in independent:
        summary, peaks, arrays = analyze_curve(curve)
        summaries.append(summary)
        if not peaks.empty:
            all_peaks.append(peaks)
        analyzed[curve.sha256] = (curve, peaks, arrays)

    summary_df = pd.DataFrame(summaries)
    reference_noise = float(
        summary_df.loc[summary_df["folder"].str.startswith("1-"), "noise_sigma_median"].median()
    )
    summary_df["noise_ratio_vs_reference"] = summary_df["noise_sigma_median"] / max(reference_noise, EPS)
    summary_df["high_noise_flag"] = summary_df["noise_ratio_vs_reference"] >= 3.0
    summary_df.to_csv(args.out / "file_summary.csv", index=False, encoding="utf-8-sig")
    if all_peaks:
        peak_df = pd.concat(all_peaks, ignore_index=True)
        peak_df.to_csv(
            args.out / "peak_candidates.csv", index=False, encoding="utf-8-sig"
        )
        peak_df[peak_df["screened_candidate"]].to_csv(
            args.out / "screened_peak_candidates.csv",
            index=False,
            encoding="utf-8-sig",
        )
    else:
        pd.DataFrame().to_csv(args.out / "peak_candidates.csv", index=False)
        pd.DataFrame().to_csv(args.out / "screened_peak_candidates.csv", index=False)

    representatives = choose_representatives(independent)
    for curve in representatives:
        _, peaks, arrays = analyzed[curve.sha256]
        detail_plot(curve, peaks, arrays, detail_dir / f"{safe_stem(curve)}.png")

    folders = sorted({curve.folder for curve in independent})
    for folder in folders:
        rows = [analyzed[c.sha256] for c in independent if c.folder == folder]
        prefix = folder.split("-", 1)[0]
        category_sheet(folder, rows, args.out / f"category_{prefix}.png")

    hplc_rows: list[dict] = []
    if not args.skip_hplc:
        hplc_patterns = [
            "2-819机型-基准2#标气",
            "A1-C2H2鼓包",
            "A10-CH4赶在鼓包上",
            "B10-1噪音、波动比较多",
            "H3-C2H2位置受干扰",
            "F1-H2负峰-处理",
        ]
        for pattern in hplc_patterns:
            curve = next((c for c in independent if pattern in c.name), None)
            if curve is not None:
                hplc_rows.append(run_hplc_case(curve, hplc_dir))
    pd.DataFrame(hplc_rows).to_csv(
        hplc_dir / "hplc_summary.csv", index=False, encoding="utf-8-sig"
    )

    manifest = {
        "input_zip": str(args.zip),
        "csv_count": len(curves),
        "independent_curve_count": len(independent),
        "duplicate_groups": duplicates,
        "libraries": {
            "hplc-py": "0.2.8",
            "pybaselines": "1.2.1",
            "scipy": "1.18.0",
        },
        "important_limitation": "No per-peak ground truth was supplied; outputs are candidates for manual review, not validated identifications.",
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Wrote results to {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
