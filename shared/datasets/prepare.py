"""Validate support tickets and export reproducible training splits."""

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED_STRINGS = (
    "ticket_id", "org_id", "product", "issue", "approved_response", "rejected_response"
)
REQUIRED_LISTS = ("evidence", "required_terms", "forbidden_terms")


def read_tickets(path: Path) -> list[dict]:
    tickets = []
    seen = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                ticket = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc
            if not isinstance(ticket, dict):
                raise ValueError(f"line {line_number}: expected a JSON object")
            for field in REQUIRED_STRINGS:
                if not isinstance(ticket.get(field), str) or not ticket[field].strip():
                    raise ValueError(f"line {line_number}: {field} must be a nonempty string")
            for field in REQUIRED_LISTS:
                value = ticket.get(field)
                if not isinstance(value, list) or (field == "evidence" and not value):
                    raise ValueError(f"line {line_number}: {field} must be a list")
                if any(not isinstance(item, str) or not item.strip() for item in value):
                    raise ValueError(f"line {line_number}: {field} contains an empty or non-string item")
            if ticket["ticket_id"] in seen:
                raise ValueError(f"line {line_number}: duplicate ticket_id {ticket['ticket_id']}")
            if ticket["approved_response"] == ticket["rejected_response"]:
                raise ValueError(f"line {line_number}: chosen and rejected responses match")
            seen.add(ticket["ticket_id"])
            tickets.append(ticket)
    if not tickets:
        raise ValueError("input contains no tickets")
    return tickets


def split_by_org(tickets: list[dict]) -> dict[str, list[dict]]:
    orgs = sorted({item["org_id"] for item in tickets},
                  key=lambda value: hashlib.sha256(value.encode()).hexdigest())
    if len(orgs) < 3:
        raise ValueError("at least three distinct org_id values are needed for train/validation/test")
    test_count = max(1, round(len(orgs) * 0.1))
    validation_count = max(1, round(len(orgs) * 0.1))
    assignments = {org: "train" for org in orgs}
    for org in orgs[:test_count]:
        assignments[org] = "test"
    for org in orgs[test_count:test_count + validation_count]:
        assignments[org] = "validation"
    result = {name: [] for name in ("train", "validation", "test")}
    for ticket in tickets:
        result[assignments[ticket["org_id"]]].append(ticket)
    return result


def prompt_for(ticket: dict) -> str:
    evidence = "\n".join(f"- {item}" for item in ticket["evidence"])
    return f"Product: {ticket['product']}\nIssue: {ticket['issue']}\nApproved knowledge:\n{evidence}"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare(input_path: Path, output_dir: Path) -> dict:
    tickets = read_tickets(input_path)
    splits = split_by_org(tickets)
    output_dir.mkdir(parents=True, exist_ok=True)
    system = "Answer enterprise support questions using only the provided approved knowledge. State uncertainty and suggest human escalation when knowledge is insufficient."
    manifest = {"source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                "split_strategy": "sha256(org_id), 10% test, 10% validation, remainder train",
                "splits": {}}
    evaluation = []
    for name, rows in splits.items():
        sft = []
        preferences = []
        for ticket in rows:
            prompt = prompt_for(ticket)
            sft.append({"ticket_id": ticket["ticket_id"], "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": ticket["approved_response"]}]})
            preferences.append({"ticket_id": ticket["ticket_id"], "prompt": prompt,
                                "chosen": ticket["approved_response"],
                                "rejected": ticket["rejected_response"]})
            if name == "test":
                evaluation.append({"ticket_id": ticket["ticket_id"],
                                   "required_terms": ticket["required_terms"],
                                   "forbidden_terms": ticket["forbidden_terms"]})
        write_jsonl(output_dir / f"sft_{name}.jsonl", sft)
        write_jsonl(output_dir / f"preferences_{name}.jsonl", preferences)
        manifest["splits"][name] = {"tickets": len(rows),
                                    "organizations": len({row["org_id"] for row in rows})}
    write_jsonl(output_dir / "evaluation.jsonl", evaluation)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
