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
    _raw_zoom_segment_bounds,
    _robust_display_limits,
    classify_curve,
    feature_rows,
    high_noise_condition,
    independent_curves,
    preprocess,
    train_template,
)
from chrompeak.core import Curve, EPS, read_curves


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
        for expected in (1.4367, 1.9200):
            event = min(rows, key=lambda row: abs(row["apex_time_min"] - expected))
            self.assertLess(abs(event["apex_time_min"] - expected), 0.012)
            self.assertEqual("confirmed", event["status"])

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
        for expected in (3.9333, 6.1400, 6.5333):
            event = min(rows, key=lambda row: abs(row["apex_time_min"] - expected))
            self.assertLess(abs(event["apex_time_min"] - expected), 0.012)
            self.assertEqual("electrical_spike", event["feature_type"])
            self.assertEqual("artifact", event["status"])
            self.assertIn("multi_parameter_electrical_spike", event["reasons"])

    def test_a7_apex_spike_does_not_erase_surviving_peak_body(self) -> None:
        rows = self.classify("A7-")
        for expected in (0.433333333333333, 0.565):
            event = min(rows, key=lambda row: abs(row["apex_time_min"] - expected))
            self.assertLess(abs(event["apex_time_min"] - expected), 0.004)
            self.assertEqual("peak_with_apex_spike", event["feature_type"])
            self.assertEqual("confirmed", event["status"])
            self.assertTrue(event["apex_spike_flag"])
            self.assertTrue(event["peak_body_survives_despike"])
            self.assertGreaterEqual(event["despiked_height_retention"], 0.90)
            self.assertGreaterEqual(
                event["peak_body_score"], self.config.apex_spike_peak_body_floor
            )
            self.assertLess(
                event["electrical_interference_score"],
                self.config.electrical_spike_score_threshold,
            )
            self.assertIn("apex_spike_on_surviving_peak_body", event["reasons"])

    def test_reference_narrow_peaks_are_not_mistaken_for_electrical_spikes(self) -> None:
        references = [curve for curve in self.curves if curve.folder.startswith("1-")]
        for curve in references:
            rows = classify_curve(
                curve,
                self.reference_processed[curve.sha256],
                self.template,
                {},
                max(self.reference_noise, EPS),
                self.config,
            )
            for slot in ("T1", "T2"):
                matches = [row for row in rows if row["template_slot"] == slot]
                self.assertTrue(matches, msg=f"{curve.name} should contain {slot}")
                event = max(matches, key=lambda row: row["peak_confidence"])
                self.assertEqual("confirmed", event["status"])
                self.assertNotIn("electrical", event["feature_type"])

    def test_template_rt_match_cannot_rescue_irregular_narrow_spike(self) -> None:
        x = np.arange(0.0, 1.000001, 1.0 / 600.0)
        y = 1e-4 * np.sin(17.0 * x)
        center = int(np.argmin(np.abs(x - 0.365)))
        y[center - 2 : center + 3] += np.array([0.05, 0.35, 1.0, 0.22, 0.04])
        curve = Curve(
            "synthetic",
            "T1_coincident_spike.csv",
            "synthetic-t1-spike",
            x,
            y,
        )
        processed = preprocess(curve)
        rows = classify_curve(
            curve,
            processed,
            self.template,
            {},
            max(self.reference_noise, EPS),
            self.config,
        )
        event = min(rows, key=lambda row: abs(row["apex_time_min"] - 0.365))
        self.assertLess(abs(event["apex_time_min"] - 0.365), 0.004)
        self.assertEqual("T1", event["template_slot"])
        self.assertIn("electrical", event["feature_type"])
        self.assertEqual("artifact", event["status"])
        self.assertLess(event["peak_confidence"], self.config.artifact_threshold)

    def test_t1_width_matched_gaussian_is_not_an_apex_spike(self) -> None:
        slot = next(slot for slot in self.template if slot["slot"] == "T1")
        # scipy's width_min is measured at 95% relative height. Choose sigma so
        # the ideal Gaussian has exactly the learned T1 median width at that
        # level, rather than confusing FWHM with the detector's width_min.
        sigma = slot["width_median_min"] / (2.0 * np.sqrt(2.0 * np.log(20.0)))
        x = np.arange(0.0, 1.000001, 1.0 / 600.0)
        y = np.exp(-0.5 * ((x - slot["retention_time_min"]) / sigma) ** 2)
        curve = Curve(
            "synthetic",
            "T1_width_matched_gaussian.csv",
            "synthetic-t1-width-matched-gaussian",
            x,
            y,
        )
        rows = classify_curve(
            curve,
            preprocess(curve),
            self.template,
            {},
            max(self.reference_noise, EPS),
            self.config,
        )
        event = min(
            rows,
            key=lambda row: abs(
                row["apex_time_min"] - slot["retention_time_min"]
            ),
        )

        self.assertLess(
            abs(event["apex_time_min"] - slot["retention_time_min"]),
            0.004,
        )
        self.assertAlmostEqual(
            event["width_min"], slot["width_median_min"], delta=0.004
        )
        self.assertEqual("T1", event["template_slot"])
        self.assertFalse(event["apex_spike_flag"])
        self.assertNotIn("electrical", event["feature_type"])
        self.assertEqual("confirmed", event["status"])

    def test_t1_gaussian_with_one_apex_spike_keeps_real_peak_body(self) -> None:
        slot = next(slot for slot in self.template if slot["slot"] == "T1")
        sigma = slot["width_median_min"] / (2.0 * np.sqrt(2.0 * np.log(20.0)))
        x = np.arange(0.0, 1.000001, 1.0 / 600.0)
        y = np.exp(-0.5 * ((x - slot["retention_time_min"]) / sigma) ** 2)
        apex = int(np.argmin(np.abs(x - slot["retention_time_min"])))
        y[apex] += 0.20
        curve = Curve(
            "synthetic",
            "T1_gaussian_with_apex_spike.csv",
            "synthetic-t1-gaussian-with-apex-spike",
            x,
            y,
        )
        rows = classify_curve(
            curve,
            preprocess(curve),
            self.template,
            {},
            max(self.reference_noise, EPS),
            self.config,
        )
        event = min(
            rows,
            key=lambda row: abs(
                row["apex_time_min"] - slot["retention_time_min"]
            ),
        )

        self.assertTrue(event["apex_spike_flag"])
        self.assertTrue(event["peak_body_survives_despike"])
        self.assertEqual("peak_with_apex_spike", event["feature_type"])
        self.assertEqual("confirmed", event["status"])
        self.assertIn("apex_spike_on_surviving_peak_body", event["reasons"])

    def test_gaussian_classification_is_sampling_rate_invariant(self) -> None:
        center = 1.0
        fwhm_min = 0.08
        sigma = fwhm_min / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        feature_types: set[str] = set()

        for samples_per_min in (60, 120, 600):
            dt = 1.0 / samples_per_min
            x = np.arange(0.0, 2.0 + 0.5 * dt, dt)
            y = np.exp(-0.5 * ((x - center) / sigma) ** 2)
            curve = Curve(
                "synthetic",
                f"gaussian_fwhm_008_{samples_per_min}_samples_per_min.csv",
                f"synthetic-gaussian-fwhm-008-{samples_per_min}",
                x,
                y,
            )
            rows = classify_curve(
                curve,
                preprocess(curve),
                self.template,
                {},
                max(self.reference_noise, EPS),
                self.config,
            )
            event = min(rows, key=lambda row: abs(row["apex_time_min"] - center))

            self.assertLess(
                abs(event["apex_time_min"] - center),
                dt + EPS,
                msg=f"apex mismatch at {samples_per_min} samples/min",
            )
            self.assertAlmostEqual(
                event["fwhm_min"],
                fwhm_min,
                delta=max(0.01, dt),
                msg=f"FWHM mismatch at {samples_per_min} samples/min",
            )
            self.assertFalse(
                event["apex_spike_flag"],
                msg=f"smooth Gaussian flagged as apex spike at {samples_per_min} samples/min",
            )
            self.assertNotIn(
                "electrical",
                event["feature_type"],
                msg=f"smooth Gaussian classified as electrical at {samples_per_min} samples/min",
            )
            self.assertEqual(
                "confirmed",
                event["status"],
                msg=f"smooth Gaussian not confirmed at {samples_per_min} samples/min",
            )
            feature_types.add(event["feature_type"])

        self.assertEqual(
            1,
            len(feature_types),
            msg=f"sampling rate changed Gaussian type: {sorted(feature_types)}",
        )

    def test_h2_multipoint_interference_near_point_six_is_not_auto_confirmed(self) -> None:
        rows = self.classify("H2-")
        event = min(rows, key=lambda row: abs(row["apex_time_min"] - 0.6000))

        self.assertLess(abs(event["apex_time_min"] - 0.6000), 0.012)
        self.assertEqual("electrical_interference_candidate", event["feature_type"])
        self.assertEqual("review", event["status"])
        self.assertIn("apex_anomaly_without_reliable_peak_body", event["reasons"])

    def test_a3_returning_bump_is_not_missed_by_endpoint_drift(self) -> None:
        a3 = next(curve for curve in self.curves if curve.name.startswith("A3-"))
        processed = preprocess(a3)
        self.assertLess(abs(processed.endpoint_drift_ratio), 0.05)
        self.assertGreater(processed.baseline_excursion_ratio, 0.25)
        self.assertGreater(processed.baseline_curvature, 0.20)
        self.assertGreater(processed.bump_duration, 6.0)
        self.assertTrue(processed.bump_flag)

        b10 = next(curve for curve in self.curves if curve.name.startswith("B10-"))
        flat = preprocess(b10)
        self.assertLess(flat.baseline_excursion_ratio, 0.02)
        self.assertLess(flat.baseline_curvature, 0.02)
        self.assertEqual(0.0, flat.bump_duration)
        self.assertFalse(flat.bump_flag)

    def test_sensitive_broad_background_regions_cover_known_hump_samples(self) -> None:
        expected_coverage = {
            "A1-": 6.90,
            "A2-": 3.56,
            "A8-": 5.03,
            "A10-": 8.44,
            "A12-": 6.74,
            "A14-": 3.64,
        }
        for prefix, expected_time in expected_coverage.items():
            curve = next(curve for curve in self.curves if curve.name.startswith(prefix))
            processed = preprocess(curve)
            sample = int(np.argmin(np.abs(curve.x - expected_time)))
            self.assertTrue(
                bool(processed.bump_mask[sample]),
                msg=f"{prefix} should mark the broad background near {expected_time:.2f} min",
            )
            self.assertTrue(processed.bump_flag)
            self.assertIn(processed.bump_strength, {"candidate", "strong"})

        # A14's former open-drift fallback shaded a stable early segment. The
        # local 3-5 min bump must now win and that early false region must stay clear.
        a14 = next(curve for curve in self.curves if curve.name.startswith("A14-"))
        a14_processed = preprocess(a14)
        early_sample = int(np.argmin(np.abs(a14.x - 0.30)))
        self.assertFalse(bool(a14_processed.bump_mask[early_sample]))

        # A flat/noisy trace protects against a blunt global threshold reduction.
        b10 = next(curve for curve in self.curves if curve.name.startswith("B10-"))
        b10_processed = preprocess(b10)
        self.assertFalse(bool(np.any(b10_processed.bump_mask)))
        self.assertEqual("none", b10_processed.bump_strength)

    def test_a3_structured_background_ripple_cannot_self_confirm(self) -> None:
        rows = self.classify("A3-")
        suspicious = (5.0000, 5.2000, 5.3000, 6.4000, 7.2667, 9.3350)
        for expected in suspicious:
            event = min(rows, key=lambda row: abs(row["apex_time_min"] - expected))
            self.assertLess(abs(event["apex_time_min"] - expected), 0.012)
            self.assertEqual("structured_background_residual", event["feature_type"])
            self.assertTrue(event["structured_background_residual_flag"])
            self.assertEqual("review", event["status"])
            self.assertLess(event["peak_confidence"], self.config.confirmation_threshold)
            self.assertIn("structured_background_residual", event["reasons"])

        confirmed_times = [
            row["apex_time_min"] for row in rows if row["status"] == "confirmed"
        ]
        self.assertEqual(2, len(confirmed_times))
        self.assertTrue(any(abs(time - 0.3317) < 0.012 for time in confirmed_times))
        self.assertTrue(any(abs(time - 1.8000) < 0.012 for time in confirmed_times))
        self.assertFalse(any(time >= 4.5 for time in confirmed_times))

        for row in rows:
            self.assertGreaterEqual(row["effective_noise"], row["random_noise"])
            self.assertAlmostEqual(
                row["snr"],
                row["prominence"] / max(row["effective_noise"], EPS),
            )
            self.assertAlmostEqual(
                row["width_to_fwhm"],
                row["width_min"] / max(row["fwhm_min"], EPS),
            )

    def test_real_peaks_on_hump_are_preserved(self) -> None:
        protected = {
            "A9-": (1.9717, 3.0983),
            "A11-": (2.1833, 2.9333, 3.5800, 5.1250),
            "A12-": (1.1300, 3.2633, 4.3017),
            "A14-": (1.5967, 2.0967, 2.4683),
        }
        for prefix, expected_times in protected.items():
            rows = self.classify(prefix)
            for expected in expected_times:
                event = min(
                    rows, key=lambda row: abs(row["apex_time_min"] - expected)
                )
                self.assertLess(abs(event["apex_time_min"] - expected), 0.012)
                self.assertEqual("confirmed", event["status"])
                self.assertGreaterEqual(
                    event["peak_confidence"], self.config.confirmation_threshold
                )

    def test_electrical_artifact_classification_has_priority_over_background(self) -> None:
        protected = {
            "H1-": (6.1400, "electrical_spike"),
            "A4-": (2.8633, "electrical_interference_candidate"),
        }
        for prefix, (expected_time, expected_type) in protected.items():
            rows = self.classify(prefix)
            event = min(
                rows, key=lambda row: abs(row["apex_time_min"] - expected_time)
            )
            self.assertLess(abs(event["apex_time_min"] - expected_time), 0.012)
            self.assertEqual(expected_type, event["feature_type"])
            self.assertEqual("artifact", event["status"])
            self.assertFalse(event["structured_background_residual_flag"])

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
            self.assertTrue(np.isfinite(spec["corrected_apex_value"]))
            self.assertIn("顶 ", spec["apex_text"])
            self.assertIn("校正值 ", spec["apex_text"])
            self.assertIn("min", spec["apex_text"])

        f1_curve = next(curve for curve in self.curves if curve.name.startswith("F1-H2"))
        f1_specs = _confirmed_boundary_specs(
            feature_rows(f1_curve, self.classify("F1-H2"))
        )
        negative = min(
            f1_specs, key=lambda spec: abs(spec["apex_time_min"] - 0.3267)
        )
        positive = min(
            f1_specs, key=lambda spec: abs(spec["apex_time_min"] - 0.3983)
        )
        self.assertLess(negative["corrected_apex_value"], 0.0)
        self.assertGreater(positive["corrected_apex_value"], 0.0)

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

    def test_raw_zoom_uses_equal_time_segments_and_robust_display_only(self) -> None:
        x = np.linspace(2.0, 10.0, 801)
        bounds = _raw_zoom_segment_bounds(x, segment_count=4)
        self.assertEqual(
            [(2.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 10.0)],
            bounds,
        )

        # One extreme point must not flatten the near-baseline display.  The
        # raw point is intentionally left untouched and merely falls outside
        # the returned display-axis limits.
        raw = 0.01 * np.sin(np.linspace(0.0, 20.0, 1001))
        raw_copy = raw.copy()
        raw[0] = 1000.0
        low, high = _robust_display_limits(raw)
        self.assertTrue(np.array_equal(raw[1:], raw_copy[1:]))
        self.assertEqual(1000.0, raw[0])
        self.assertLess(low, -0.005)
        self.assertGreater(high, 0.005)
        self.assertLess(high, 1.0)


if __name__ == "__main__":
    unittest.main()
