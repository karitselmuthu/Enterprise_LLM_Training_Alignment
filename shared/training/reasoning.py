"""Build and evaluate evidence-grounded support reasoning examples."""

import argparse
import json
from pathlib import Path

from shared.training.sft import read_jsonl, write_jsonl


def build_examples(tickets: list[dict]) -> list[dict]:
    examples = []
    for ticket in tickets:
        evidence = ticket["evidence"]
        if not evidence or not all(isinstance(item, str) and item.strip() for item in evidence):
            raise ValueError(f"{ticket['ticket_id']}: evidence is required")
        examples.append({
            "ticket_id": ticket["ticket_id"], "org_id": ticket["org_id"],
            "question": ticket["issue"],
            "evidence": [{"id": f"E{index}", "text": text}
                         for index, text in enumerate(evidence, start=1)],
            "target": {"evidence_ids": [f"E{index}" for index in range(1, len(evidence) + 1)],
                       "answer": ticket["approved_response"]},
            "required_terms": ticket["required_terms"],
            "forbidden_terms": ticket["forbidden_terms"],
        })
    return examples


def evaluate_candidate(example: dict, candidate: dict) -> dict:
    valid_ids = {item["id"] for item in example["evidence"]}
    cited = candidate.get("evidence_ids", [])
    answer = candidate.get("answer", "")
    if not isinstance(cited, list) or not isinstance(answer, str):
        raise ValueError("candidate requires evidence_ids list and answer string")
    lower = answer.casefold()
    return {"ticket_id": example["ticket_id"],
            "citations_valid": bool(cited) and all(item in valid_ids for item in cited),
            "required_terms_pass": all(term.casefold() in lower for term in example["required_terms"]),
            "forbidden_terms_pass": all(term.casefold() not in lower for term in example["forbidden_terms"])}


def run(source: Path, data_dir: Path, output: Path, baseline: Path | None = None) -> dict:
    tickets = read_jsonl(source)
    split_by_id = {}
    for split in ("train", "validation", "test"):
        for row in read_jsonl(data_dir / f"sft_{split}.jsonl"):
            if row["ticket_id"] in split_by_id:
                raise ValueError("duplicate ticket across splits")
            split_by_id[row["ticket_id"]] = split
    examples = build_examples(tickets)
    if {item["ticket_id"] for item in examples} != set(split_by_id):
        raise ValueError("reasoning examples differ from prepared split")
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        write_jsonl(output / f"{split}.jsonl",
                    [item for item in examples if split_by_id[item["ticket_id"]] == split])
    # Gold-format smoke check detects broken evidence links and banned language.
    checks = [evaluate_candidate(item, item["target"]) for item in examples]
    result = {"cases": len(checks), "split_counts": {split: sum(
        split_by_id[item["ticket_id"]] == split for item in examples)
        for split in ("train", "validation", "test")},
        "gold_valid_citations": sum(row["citations_valid"] for row in checks),
        "gold_required_terms_pass": sum(row["required_terms_pass"] for row in checks),
        "gold_forbidden_terms_pass": sum(row["forbidden_terms_pass"] for row in checks)}
    if baseline is not None:
        by_id = {item["ticket_id"]: item for item in examples}
        baseline_rows = read_jsonl(baseline)
        if {row["ticket_id"] for row in baseline_rows} != {
                item["ticket_id"] for item in examples if split_by_id[item["ticket_id"]] == "test"}:
            raise ValueError("baseline IDs differ from test split")
        baseline_checks = [evaluate_candidate(by_id[row["ticket_id"]],
                                             {"answer": row["response"], "evidence_ids": []})
                           for row in baseline_rows]
        result["sft_test_baseline"] = {key: sum(row[key] for row in baseline_checks)
                                       for key in ("citations_valid", "required_terms_pass",
                                                   "forbidden_terms_pass")}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("shared/datasets/sample_tickets.jsonl"))
    parser.add_argument("--data-dir", type=Path, default=Path("experiments/sample"))
    parser.add_argument("--output", type=Path, default=Path("experiments/phase_3_6"))
    parser.add_argument("--baseline", type=Path,
                        default=Path("experiments/phase_3_2/predictions.jsonl"))
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.data_dir, args.output, args.baseline), indent=2))


if __name__ == "__main__":
    main()
