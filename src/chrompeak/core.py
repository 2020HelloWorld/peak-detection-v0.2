"""Core data structures and signal utilities used by the detector."""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter


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
    """Read and validate every x/Curvel CSV stored in a ZIP archive."""
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
            if len(x) < 5 or not np.all(np.diff(x) > 0):
                raise ValueError(f"Time axis must be strictly increasing in {member}")
            folder, name = member.split("/", 1)
            curves.append(Curve(folder, name, digest, x, y))
            hash_groups.setdefault(digest, []).append(member)
    duplicates = [names for names in hash_groups.values() if len(names) > 1]
    return curves, duplicates


def symmetric_baseline(y: np.ndarray, dt_min: float) -> np.ndarray:
    """Estimate a slow baseline without preferring positive or negative peaks."""
    n = len(y)
    median_points = odd_at_most(round(1.0 / dt_min), n)
    smooth_points = odd_at_most(round(0.35 / dt_min), n)
    baseline = median_filter(y, size=median_points, mode="nearest")
    if smooth_points >= 5:
        baseline = savgol_filter(baseline, smooth_points, polyorder=2, mode="interp")
    return baseline


def local_noise_sigma(signal: np.ndarray, dt_min: float) -> np.ndarray:
    """Estimate time-varying noise from robust first-difference statistics."""
    n = len(signal)
    diff = np.diff(signal, prepend=signal[0])
    window = odd_at_most(round(0.6 / dt_min), n)
    center = median_filter(diff, size=window, mode="nearest")
    mad = median_filter(np.abs(diff - center), size=window, mode="nearest")
    sigma = mad / (0.67448975 * math.sqrt(2.0))
    positive = sigma[sigma > 0]
    if len(positive):
        floor = max(float(np.quantile(positive, 0.10)), np.ptp(signal) * 1e-9, EPS)
    else:
        floor = max(np.ptp(signal) * 1e-9, EPS)
    return np.maximum(sigma, floor)
