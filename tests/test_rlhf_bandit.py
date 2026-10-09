import unittest

from shared.training.rlhf_bandit import attach_model_scores


class RLHFTests(unittest.TestCase):
    def test_scores_follow_candidate_order(self):
        case = {"ticket_id": "T", "approved_index": 1}
        summary = {"final": {"train": {"details": [{"ticket_id": "T",
                                                      "chosen_score": 3.0,
                                                      "rejected_score": 1.0}]}}}
        result = attach_model_scores({"train": [case]}, summary)
        self.assertEqual(result["train"][0]["model_scores"], [1.0, 3.0])

    def test_mismatched_ticket_ids_rejected(self):
        summary = {"final": {"train": {"details": []}}}
        with self.assertRaises(ValueError):
            attach_model_scores({"train": [{"ticket_id": "T"}]}, summary)


if __name__ == "__main__":
    unittest.main()
