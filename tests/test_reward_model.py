import unittest

from shared.training.reward_model import encode_pair, validate_pair


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]


class RewardModelTests(unittest.TestCase):
    def test_pair_contains_the_same_prompt_with_distinct_answers(self):
        row = {"ticket_id": "T-1", "prompt": "Reset password?",
               "chosen": "Use the reset link.", "rejected": "Share your password."}
        encoded = encode_pair(row, "Use approved knowledge.", FakeTokenizer(), 200)
        chosen = "".join(map(chr, encoded["chosen"]))
        rejected = "".join(map(chr, encoded["rejected"]))
        self.assertIn("### User\nReset password?", chosen)
        self.assertIn("### User\nReset password?", rejected)
        self.assertTrue(chosen.endswith("Use the reset link."))
        self.assertTrue(rejected.endswith("Share your password."))

    def test_identical_preference_responses_are_rejected(self):
        row = {"ticket_id": "T-1", "prompt": "Question",
               "chosen": "Same", "rejected": "Same"}
        with self.assertRaisesRegex(ValueError, "identical"):
            validate_pair(row)


if __name__ == "__main__":
    unittest.main()
