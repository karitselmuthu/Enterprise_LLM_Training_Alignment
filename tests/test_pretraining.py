import tempfile
import unittest
from pathlib import Path

from shared.training.char_bigram import generate, load_checkpoint, train


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "shared/datasets/sample_tickets.jsonl"


class PretrainingTests(unittest.TestCase):
    def test_training_reduces_loss_and_checkpoint_reproduces_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = train(SAMPLE, output, epochs=40, seed=11)
            self.assertEqual(summary["held_out_test_documents"], 1)
            self.assertLess(summary["final_train_loss"], summary["initial_train_loss"])
            self.assertLess(summary["final_validation_loss"], summary["initial_validation_loss"])
            vocab, weights = load_checkpoint(output / "checkpoint.json")
            self.assertEqual(generate(vocab, weights, 11), summary["samples"][0])


if __name__ == "__main__":
    unittest.main()
