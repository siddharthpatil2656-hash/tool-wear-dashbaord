import unittest

from gcode_optimizer import cutting_targets, update_spindle_and_feed


class GCodeOptimizerTests(unittest.TestCase):
    def test_turning_targets_for_g94(self):
        rpm, feed = cutting_targets(
            cutting_speed_m_min=100,
            diameter_mm=50,
            feed_value=0.2,
            feed_unit="mm/rev",
            teeth=1,
            feed_mode="G94",
        )
        self.assertEqual(rpm, 637)
        self.assertAlmostEqual(feed, 127.4, places=1)

    def test_milling_feed_per_tooth_for_g95(self):
        _, feed = cutting_targets(
            cutting_speed_m_min=100,
            diameter_mm=10,
            feed_value=0.05,
            feed_unit="mm/tooth",
            teeth=4,
            feed_mode="G95",
        )
        self.assertAlmostEqual(feed, 0.2)

    def test_update_preserves_coordinates_and_comments(self):
        result = update_spindle_and_feed(
            "G01 X10.0 Y5.0 S3000 F250 ; keep this comment",
            4000,
            325.5,
            "G94",
        )
        self.assertEqual(result, "G01 X10.0 Y5.0 S4000 F325.5 ; keep this comment")

    def test_missing_s_and_f_are_added_before_comment(self):
        result = update_spindle_and_feed(
            "G01 X10 (comment) ; tail",
            4000,
            250,
            "G94",
        )
        self.assertEqual(result, "G01 X10 S4000 F250 (comment) ; tail")

    def test_unsafe_or_unsupported_modes_are_rejected(self):
        for block in (
            "G01 X10 F20 G20",
            "G01 X10 S100 G96",
            "G02 X10 Y10 S100 F20",
        ):
            with self.subTest(block=block), self.assertRaises(ValueError):
                update_spindle_and_feed(block, 3000, 200, "G94")

    def test_conflicting_feed_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            update_spindle_and_feed("G01 G95 X10 F0.2", 3000, 200, "G94")


if __name__ == "__main__":
    unittest.main()
