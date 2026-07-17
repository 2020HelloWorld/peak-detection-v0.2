from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chrompeak.detector import (
    DetectorConfig,
    _confirmed_boundary_specs,
    classify_curve,
    feature_rows,
    high_noise_condition,
    independent_curves,
    preprocess,
    train_template,
)
from chrompeak.core import EPS, read_curves


class DetectorSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        curves, _ = read_curves(PROJECT_ROOT / "data" / "typical_chromatograms.zip")
        cls.curves = independent_curves(curves)
        references = [curve for curve in cls.curves if curve.folder.startswith("1-")]
        cls.template, cls.reference_processed = train_template(references)
        cls.reference_noise = float(
            np.median(
                [cls.reference_processed[curve.sha256].global_noise for curve in references]
            )
        )
        cls.config = DetectorConfig.from_json(
            PROJECT_ROOT / "configs" / "detector_config.json"
        )

    def classify(self, prefix: str) -> list[dict]:
        curve = next(curve for curve in self.curves if curve.name.startswith(prefix))
        processed = self.reference_processed.get(curve.sha256) or preprocess(curve)
        return classify_curve(
            curve,
            processed,
            self.template,
            {},
            max(self.reference_noise, EPS),
            self.config,
        )

    def test_reference_template_has_six_slots(self) -> None:
        self.assertEqual(6, len(self.template))

    def test_f1_strong_negative_and_positive_are_confirmed(self) -> None:
        rows = self.classify("F1-H2")
        negative = min(rows, key=lambda row: abs(row["apex_time_min"] - 0.3267))
        positive = min(rows, key=lambda row: abs(row["apex_time_min"] - 0.3983))
        later = min(rows, key=lambda row: abs(row["apex_time_min"] - 2.3550))
        self.assertEqual("negative_peak", negative["feature_type"])
        self.assertEqual("confirmed", negative["status"])
        self.assertGreaterEqual(negative["peak_confidence"], self.config.confirmation_threshold)
        self.assertEqual("confirmed", positive["status"])
        self.assertGreaterEqual(positive["peak_confidence"], self.config.confirmation_threshold)
        self.assertEqual("confirmed", later["status"])

    def test_h3_flat_top_interference_stays_artifact(self) -> None:
        rows = self.classify("H3-C2H2")
        event = min(rows, key=lambda row: abs(row["apex_time_min"] - 4.3633))
        self.assertEqual("electrical_interference_candidate", event["feature_type"])
        self.assertEqual("artifact", event["status"])
        self.assertLess(event["peak_confidence"], self.config.artifact_threshold)

    def test_conservative_gentle_broad_peak_candidate(self) -> None:
        rows = self.classify("F3-CO2")
        event = min(rows, key=lambda row: abs(row["apex_time_min"] - 2.6633))
        self.assertEqual("gentle_broad_peak_candidate", event["feature_type"])
        self.assertEqual("review", event["status"])

    def test_positive_electrical_spike_is_an_artifact(self) -> None:
        rows = self.classify("H1-")
        event = min(rows, key=lambda row: abs(row["apex_time_min"] - 3.9333))
        self.assertEqual("electrical_spike", event["feature_type"])
        self.assertEqual("artifact", event["status"])

    def test_plot_boundaries_are_emitted_for_confirmed_peaks_only(self) -> None:
        selected_rows: list[dict] = []
        for prefix in ("F1-H2", "F3-CO2", "H1-"):
            curve = next(curve for curve in self.curves if curve.name.startswith(prefix))
            selected_rows.extend(feature_rows(curve, self.classify(prefix)))

        specs = _confirmed_boundary_specs(selected_rows)
        confirmed = [row for row in selected_rows if row["status"] == "confirmed"]
        self.assertEqual(len(confirmed), len(specs))
        self.assertTrue(specs)
        for spec in specs:
            self.assertLess(spec["start_time_min"], spec["end_time_min"])
            self.assertLessEqual(spec["start_time_min"], spec["apex_time_min"])
            self.assertLessEqual(spec["apex_time_min"], spec["end_time_min"])
            self.assertIn("起 ", spec["text"])
            self.assertIn("止 ", spec["text"])
            self.assertTrue(spec["text"].endswith(" min"))

        excluded = {
            (row["file"], row["feature_id"])
            for row in selected_rows
            if row["status"] != "confirmed"
        }
        plotted = {(spec["file"], spec["feature_id"]) for spec in specs}
        self.assertTrue(excluded.isdisjoint(plotted))

    def test_high_noise_flag_is_not_used_as_a_blanket_rejection(self) -> None:
        curve = next(curve for curve in self.curves if curve.name.startswith("B10-"))
        processed = preprocess(curve)
        self.assertTrue(high_noise_condition(processed, self.reference_noise))
        rows = classify_curve(
            curve,
            processed,
            self.template,
            {},
            self.reference_noise,
            self.config,
        )
        confirmed_times = [row["apex_time_min"] for row in rows if row["status"] == "confirmed"]
        self.assertTrue(any(abs(time - 0.5067) < 0.01 for time in confirmed_times))


if __name__ == "__main__":
    unittest.main()
