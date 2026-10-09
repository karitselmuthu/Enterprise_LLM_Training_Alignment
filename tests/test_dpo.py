import unittest

from shared.training.dpo import build_pairs
from test_sft_format import FakeTokenizer


class DPOTests(unittest.TestCase):
    def test_pairs_share_prompt_and_mask_it(self):
        rows = [{"ticket_id": "T-1", "prompt": "Help?", "chosen": "Use SSO.",
                 "rejected": "Share your password."}]
        pair = build_pairs(rows, "Follow policy.", FakeTokenizer(), 200)[0]
        self.assertEqual(pair["chosen"]["prompt_ids"], pair["rejected"]["prompt_ids"])
        for label in ("chosen", "rejected"):
            example = pair[label]
            self.assertEqual(example["labels"][:len(example["prompt_ids"])],
                             [-100] * len(example["prompt_ids"]))

    def test_identical_responses_rejected(self):
        rows = [{"ticket_id": "T-1", "prompt": "Help?", "chosen": "Same", "rejected": "Same"}]
        with self.assertRaises(ValueError):
            build_pairs(rows, "Policy", FakeTokenizer(), 100)


if __name__ == "__main__":
    unittest.main()
