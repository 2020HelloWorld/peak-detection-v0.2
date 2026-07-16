from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chrompeak.detector import (
    DetectorConfig,
    classify_curve,
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
