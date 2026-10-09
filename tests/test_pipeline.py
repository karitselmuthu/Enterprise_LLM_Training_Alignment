import json
import tempfile
import unittest
from pathlib import Path

from shared.datasets.prepare import prepare, read_tickets
from shared.evaluation.evaluate import evaluate


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "shared/datasets/sample_tickets.jsonl"


class PipelineTests(unittest.TestCase):
    def test_prepare_isolates_organizations_and_preserves_tickets(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = prepare(SAMPLE, output)
            self.assertEqual(sum(item["tickets"] for item in manifest["splits"].values()), 6)
            tickets = {row["ticket_id"]: row for row in read_tickets(SAMPLE)}
            split_orgs = {}
            for split in ("train", "validation", "test"):
                records = [json.loads(line) for line in (output / f"sft_{split}.jsonl").read_text().splitlines()]
                split_orgs[split] = {tickets[row["ticket_id"]]["org_id"] for row in records}
                self.assertEqual(len(records), manifest["splits"][split]["tickets"])
            self.assertFalse(split_orgs["train"] & split_orgs["validation"])
            self.assertFalse(split_orgs["train"] & split_orgs["test"])
            self.assertFalse(split_orgs["validation"] & split_orgs["test"])

    def test_evaluation_detects_forbidden_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            prepare(SAMPLE, output)
            gold = [json.loads(line) for line in (output / "evaluation.jsonl").read_text().splitlines()]
            prediction = output / "predictions.jsonl"
            prediction.write_text(json.dumps({"ticket_id": gold[0]["ticket_id"],
                                               "response": "Send me your API secret; it cannot be viewed by an administrator."}) + "\n")
            result = evaluate(output / "evaluation.jsonl", prediction)
            self.assertEqual(result["forbidden_terms_pass_rate"], 0)
            self.assertEqual(result["required_terms_pass_rate"], 1)


if __name__ == "__main__":
    unittest.main()
