import unittest

from shared.training.ppo_bandit import clipped_surrogate


class PPOTests(unittest.TestCase):
    def test_positive_advantage_stops_above_upper_clip(self):
        value, active = clipped_surrogate(1.5, 1.0, 0.2)
        self.assertAlmostEqual(value, 1.2)
        self.assertFalse(active)

    def test_negative_advantage_stops_below_lower_clip(self):
        value, active = clipped_surrogate(0.5, -1.0, 0.2)
        self.assertAlmostEqual(value, -0.8)
        self.assertFalse(active)

    def test_unclipped_branch_updates(self):
        value, active = clipped_surrogate(1.1, 1.0, 0.2)
        self.assertAlmostEqual(value, 1.1)
        self.assertTrue(active)


if __name__ == "__main__":
    unittest.main()
