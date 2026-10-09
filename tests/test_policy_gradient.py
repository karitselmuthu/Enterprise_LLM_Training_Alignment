import unittest

from shared.training.policy_gradient import probabilities, reinforce_update


class PolicyGradientTests(unittest.TestCase):
    def test_probabilities_sum_to_one(self):
        self.assertAlmostEqual(sum(probabilities([0.0, 0.0], [[1, 0], [0, -2]])), 1.0)

    def test_positive_reward_raises_sampled_action_probability(self):
        case = {"rewards": [{"required_coverage": 1.0, "forbidden_hits": 0},
                            {"required_coverage": 0.0, "forbidden_hits": 1}]}
        before = probabilities([0.0, 0.0], [[1.0, 0.0], [0.0, -1.0]])[0]
        weights = reinforce_update([0.0, 0.0], case, 0, 1.0, 0.1)
        after = probabilities(weights, [[1.0, 0.0], [0.0, -1.0]])[0]
        self.assertGreater(after, before)


if __name__ == "__main__":
    unittest.main()
