import unittest

import pandas as pd

from calibration import (
    CALIBRATION_COLUMNS,
    calibration_template,
    fit_taylor_model,
    matching_calibration_trials,
)


class TaylorCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.context = {
            "pairing_key": "Ti-6Al-4V | PVD TiAlN Carbide",
            "machine_name": "Test machine",
            "coolant_name": "Test coolant",
            "operation_name": "Turning",
            "milling_tooling_name": "",
            "holder_name": "Test holder",
            "overhang_ratio": 3.0,
            "is_roughing": False,
        }
        rows = []
        for speed in (60.0, 90.0):
            for feed in (0.08, 0.16):
                for depth in (0.5, 1.5):
                    for repeat in range(2):
                        life = (80.0 / (speed * feed**0.4 * depth**0.2)) ** (1 / 0.25)
                        rows.append({
                            **self.context,
                            "cutting_speed_m_min": speed,
                            "feed_rate_mm": feed,
                            "depth_of_cut_mm": depth,
                            "measured_tool_life_min": life * 1.7,
                        })
        self.trials = pd.DataFrame(rows, columns=CALIBRATION_COLUMNS)

    def test_template_contains_context_and_measurement_columns(self):
        template = calibration_template(self.context)
        self.assertEqual(list(template.columns), list(CALIBRATION_COLUMNS))
        self.assertEqual(template.iloc[0]["pairing_key"], self.context["pairing_key"])

    def test_fit_recovers_coefficients_and_normalizes_setup_factor(self):
        fit = fit_taylor_model(self.trials, self.context, life_multiplier=1.7)
        self.assertAlmostEqual(fit.taylor_C, 80.0, places=6)
        self.assertAlmostEqual(fit.taylor_n, 0.25, places=6)
        self.assertAlmostEqual(fit.taylor_x, 0.4, places=6)
        self.assertAlmostEqual(fit.taylor_y, 0.2, places=6)
        self.assertEqual(fit.runs_used, 16)
        self.assertAlmostEqual(fit.r_squared, 1.0, places=10)
        self.assertAlmostEqual(fit.loo_r_squared, 1.0, places=10)
        self.assertAlmostEqual(fit.loo_log_rmse, 0.0, places=10)

    def test_rejects_unmatched_setup(self):
        context = {**self.context, "machine_name": "Different machine"}
        with self.assertRaisesRegex(ValueError, "No trial rows match"):
            fit_taylor_model(self.trials, context, life_multiplier=1.0)

    def test_matching_trial_rows_are_filtered_to_exact_setup(self):
        other_setup = self.trials.iloc[[0]].copy()
        other_setup["machine_name"] = "Different machine"
        selected = matching_calibration_trials(
            pd.concat([self.trials.iloc[:2], other_setup], ignore_index=True),
            self.context,
        )
        self.assertEqual(len(selected), 2)
        self.assertTrue((selected["machine_name"] == self.context["machine_name"]).all())

    def test_rejects_insufficient_measurements(self):
        with self.assertRaisesRegex(ValueError, "At least 8"):
            fit_taylor_model(self.trials.iloc[:4], self.context, life_multiplier=1.0)

    def test_rejects_trials_without_independent_parameter_variation(self):
        trials = self.trials.copy()
        trials["cutting_speed_m_min"] = 80.0
        with self.assertRaisesRegex(ValueError, "do not vary cutting speed"):
            fit_taylor_model(trials, self.context, life_multiplier=1.0)


if __name__ == "__main__":
    unittest.main()
