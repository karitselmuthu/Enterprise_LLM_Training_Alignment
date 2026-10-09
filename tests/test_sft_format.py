import unittest

from shared.training.sft import encode_example


class FakeTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        self.last_add_special_tokens = add_special_tokens
        return [ord(character) for character in text]


class SFTFormatTests(unittest.TestCase):
    def test_only_assistant_response_contributes_to_loss(self):
        tokenizer = FakeTokenizer()
        row = {"ticket_id": "T-1", "messages": [
            {"role": "system", "content": "Use evidence."},
            {"role": "user", "content": "Where is the setting?"},
            {"role": "assistant", "content": "In Settings."}]}
        result = encode_example(row, tokenizer, max_length=200)
        prompt_length = len(result["prompt_ids"])
        self.assertEqual(result["labels"][:prompt_length], [-100] * prompt_length)
        self.assertEqual(result["labels"][prompt_length:],
                         [ord(character) for character in "In Settings."] + [0])
        self.assertFalse(tokenizer.last_add_special_tokens)


if __name__ == "__main__":
    unittest.main()
