"""Synthetic support response selection and explicit reward rules for RL lessons."""

import argparse
import json
from pathlib import Path

from shared.training.sft import read_jsonl, write_jsonl


def rule_reward(ticket: dict, response: str) -> dict:
    if not isinstance(response, str) or not response.strip():
        raise ValueError("response must be nonempty")
    text = response.casefold()
    required = ticket["required_terms"]
    forbidden = ticket["forbidden_terms"]
    coverage = sum(term.casefold() in text for term in required) / len(required)
    violations = sum(term.casefold() in text for term in forbidden)
    # Both components are visible so reward gaming can be inspected.
    score = coverage - violations
    return {"required_coverage": coverage, "forbidden_hits": violations, "score": score}


def make_cases(tickets: list[dict], data_dir: Path) -> dict[str, list[dict]]:
    by_id = {ticket["ticket_id"]: ticket for ticket in tickets}
    if len(by_id) != len(tickets):
        raise ValueError("duplicate ticket ID")
    cases = {}
    all_ids = []
    for split in ("train", "validation", "test"):
        rows = read_jsonl(data_dir / f"sft_{split}.jsonl")
        cases[split] = []
        for row in rows:
            ticket = by_id[row["ticket_id"]]
            candidates = [ticket["approved_response"], ticket["rejected_response"]]
            # Alternate order to prevent the policy from using position alone.
            if int(ticket["ticket_id"].split("-")[-1]) % 2:
                candidates.reverse()
            cases[split].append({"ticket_id": ticket["ticket_id"], "org_id": ticket["org_id"],
                                 "prompt": ticket["issue"], "required_terms": ticket["required_terms"],
                                 "forbidden_terms": ticket["forbidden_terms"],
                                 "candidates": candidates,
                                 "approved_index": candidates.index(ticket["approved_response"]),
                                 "rewards": [rule_reward(ticket, candidate) for candidate in candidates]})
            all_ids.append(ticket["ticket_id"])
    if set(all_ids) != set(by_id) or len(all_ids) != len(by_id):
        raise ValueError("case IDs differ from prepared split")
    return cases


def run(source: Path, data_dir: Path, output: Path) -> dict:
    cases = make_cases(read_jsonl(source), data_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in cases.items():
        write_jsonl(output / f"{split}.jsonl", rows)
    all_cases = [row for rows in cases.values() for row in rows]
    preferred_higher = sum(row["rewards"][row["approved_index"]]["score"] >
                           row["rewards"][1 - row["approved_index"]]["score"] for row in all_cases)
    summary = {"cases": len(all_cases), "split_counts": {key: len(value) for key, value in cases.items()},
               "approved_reward_higher": preferred_higher,
               "mean_approved_reward": sum(row["rewards"][row["approved_index"]]["score"]
                                           for row in all_cases) / len(all_cases),
               "mean_rejected_reward": sum(row["rewards"][1 - row["approved_index"]]["score"]
                                           for row in all_cases) / len(all_cases)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("shared/datasets/sample_tickets.jsonl"))
    parser.add_argument("--data-dir", type=Path, default=Path("experiments/sample"))
    parser.add_argument("--output", type=Path, default=Path("experiments/phase_4_1"))
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.data_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
