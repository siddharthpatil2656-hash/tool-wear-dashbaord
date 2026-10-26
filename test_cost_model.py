import unittest

from cost_model import build_cost_curve, cost_per_part


class CostModelTests(unittest.TestCase):
    def test_cost_includes_tool_consumption_and_change_time(self):
        cost, machining, cycle = cost_per_part(
            part_volume_cm3=10,
            mrr_cm3_min=5,
            tool_life_min=20,
            tool_edge_cost=12,
            machine_rate_per_hour=60,
            operator_rate_per_hour=30,
            tool_change_min=4,
        )
        self.assertAlmostEqual(machining, 2.0)
        self.assertAlmostEqual(cycle, 2.4)
        self.assertAlmostEqual(cost, 3.6 + 1.2)

    def test_speed_sweep_returns_candidates_in_input_order(self):
        points = build_cost_curve(
            cutting_speeds=[50, 100],
            feed=0.1,
            depth_of_cut=1,
            part_volume_cm3=10,
            tool_edge_cost=10,
            machine_rate_per_hour=60,
            operator_rate_per_hour=0,
            tool_change_min=2,
            tool_life_fn=lambda speed, _feed, _depth: 1000 / speed,
            mrr_fn=lambda speed, _feed, _depth: speed / 10,
        )
        self.assertEqual([point.cutting_speed_m_min for point in points], [50, 100])
        self.assertGreater(points[1].machining_time_min, 0)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            cost_per_part(
                part_volume_cm3=0,
                mrr_cm3_min=1,
                tool_life_min=1,
                tool_edge_cost=1,
                machine_rate_per_hour=1,
                operator_rate_per_hour=1,
                tool_change_min=0,
            )


if __name__ == "__main__":
    unittest.main()
