import unittest

from shared.training.support_bandit import rule_reward


class RewardDesignTests(unittest.TestCase):
    def setUp(self):
        self.ticket = {"required_terms": ["administrator", "30 days"],
                       "forbidden_terms": ["any time"]}

    def test_approved_language_scores_higher(self):
        approved = rule_reward(self.ticket, "An administrator can restore within 30 days.")
        rejected = rule_reward(self.ticket, "Anyone can restore at any time.")
        self.assertGreater(approved["score"], rejected["score"])

    def test_keyword_stuffing_exposes_reward_gaming(self):
        gaming = rule_reward(self.ticket, "administrator 30 days")
        self.assertEqual(gaming["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
