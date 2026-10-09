import unittest

from shared.training.grpo_bandit import group_advantages, reference_kl_gradient


class GRPOTests(unittest.TestCase):
    def test_group_advantages_center_and_scale(self):
        result = group_advantages([1.0, -2.0])
        self.assertAlmostEqual(result[0], 1.0)
        self.assertAlmostEqual(result[1], -1.0)

    def test_tied_group_has_zero_advantage(self):
        self.assertEqual(group_advantages([1.0, 1.0]), [0.0, 0.0])

    def test_uniform_policy_has_zero_reference_gradient(self):
        self.assertEqual(reference_kl_gradient([0.0, 0.0], [[1, 0], [0, -1]]), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
