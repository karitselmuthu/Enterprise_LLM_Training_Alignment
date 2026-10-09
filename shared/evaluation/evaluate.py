"""Measure explicit phrase assertions on held-out support tickets."""

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def evaluate(gold_path: Path, prediction_path: Path) -> dict:
    gold = load_jsonl(gold_path)
    predictions = load_jsonl(prediction_path)
    ids = [row["ticket_id"] for row in gold]
    prediction_ids = [row["ticket_id"] for row in predictions]
    if len(ids) != len(set(ids)) or len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("duplicate ticket_id in gold or predictions")
    if set(ids) != set(prediction_ids):
        raise ValueError(f"ticket_id mismatch: missing={sorted(set(ids) - set(prediction_ids))}, "
                         f"unexpected={sorted(set(prediction_ids) - set(ids))}")
    by_id = {row["ticket_id"]: row for row in predictions}
    details = []
    for row in gold:
        response = by_id[row["ticket_id"]].get("response")
        if not isinstance(response, str):
            raise ValueError(f"{row['ticket_id']}: response must be a string")
        normalized = response.casefold()
        missing = [term for term in row["required_terms"] if term.casefold() not in normalized]
        present_forbidden = [term for term in row["forbidden_terms"] if term.casefold() in normalized]
        details.append({"ticket_id": row["ticket_id"], "required_terms_pass": not missing,
                        "forbidden_terms_pass": not present_forbidden,
                        "missing_required_terms": missing, "present_forbidden_terms": present_forbidden})
    total = len(details)
    if not total:
        raise ValueError("gold contains no evaluation cases")
    return {"cases": total,
            "required_terms_pass_rate": sum(row["required_terms_pass"] for row in details) / total,
            "forbidden_terms_pass_rate": sum(row["forbidden_terms_pass"] for row in details) / total,
            "details": details}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.gold, args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()
