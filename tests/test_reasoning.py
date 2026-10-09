import unittest

from shared.training.reasoning import build_examples, evaluate_candidate


class ReasoningTests(unittest.TestCase):
    def test_invalid_evidence_reference_fails(self):
        ticket = {"ticket_id": "T", "org_id": "O", "issue": "How?",
                  "evidence": ["Only admins may act."], "approved_response": "Ask an admin.",
                  "required_terms": ["admin"], "forbidden_terms": ["anyone"]}
        example = build_examples([ticket])[0]
        self.assertTrue(evaluate_candidate(example, example["target"])["citations_valid"])
        wrong = {"answer": "Anyone can do it.", "evidence_ids": ["E9"]}
        result = evaluate_candidate(example, wrong)
        self.assertFalse(result["citations_valid"])
        self.assertFalse(result["forbidden_terms_pass"])

    def test_missing_evidence_rejected(self):
        with self.assertRaises(ValueError):
            build_examples([{"ticket_id": "T", "evidence": []}])


if __name__ == "__main__":
    unittest.main()
