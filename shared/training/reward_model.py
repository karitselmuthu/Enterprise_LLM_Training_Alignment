"""Train a pairwise reward model from the phase 3.2 SFT checkpoint."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from shared.training.sft import read_jsonl


def validate_pair(row: dict) -> None:
    for field in ("ticket_id", "prompt", "chosen", "rejected"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f"preference pair needs nonempty {field}")
    if row["chosen"] == row["rejected"]:
        raise ValueError(f"{row['ticket_id']}: chosen and rejected are identical")


def render_candidate(system: str, prompt: str, response: str) -> str:
    return (f"### System\n{system}\n\n### User\n{prompt}\n\n"
            f"### Assistant\n{response}")


def encode_pair(row: dict, system: str, tokenizer, max_length: int) -> dict:
    validate_pair(row)
    encoded = {"ticket_id": row["ticket_id"]}
    for label in ("chosen", "rejected"):
        text = render_candidate(system, row["prompt"], row[label])
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > max_length:
            raise ValueError(f"{row['ticket_id']} {label}: {len(tokens)} tokens exceeds {max_length}")
        encoded[label] = tokens
    return encoded


def score_pair(model, pair: dict, device, torch):
    chosen = torch.tensor([pair["chosen"]], device=device)
    rejected = torch.tensor([pair["rejected"]], device=device)
    return (model(input_ids=chosen).logits.squeeze(),
            model(input_ids=rejected).logits.squeeze())


def assess(model, pairs: list[dict], device, torch) -> dict:
    model.eval()
    details = []
    with torch.no_grad():
        for pair in pairs:
            chosen, rejected = score_pair(model, pair, device, torch)
            margin = (chosen - rejected).item()
            details.append({"ticket_id": pair["ticket_id"],
                            "chosen_score": chosen.item(), "rejected_score": rejected.item(),
                            "margin": margin, "preferred_ranked_higher": margin > 0,
                            "pairwise_loss": torch.nn.functional.softplus(
                                torch.tensor(-margin)).item()})
    return {"pairs": len(details),
            "ranking_accuracy": sum(item["preferred_ranked_higher"] for item in details) / len(details),
            "mean_margin": sum(item["margin"] for item in details) / len(details),
            "mean_pairwise_loss": sum(item["pairwise_loss"] for item in details) / len(details),
            "details": details}


def train(data_dir: Path, sft_checkpoint: Path, sft_summary: Path, output_dir: Path,
          epochs: int = 8, learning_rate: float = 1e-3, max_length: int = 384,
          seed: int = 7) -> dict:
    try:
        import torch
        import transformers
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install requirements-sft.txt in a Python environment") from exc
    if epochs < 1 or learning_rate <= 0 or max_length < 16:
        raise ValueError("invalid training settings")
    if not (sft_checkpoint / "model.safetensors").is_file():
        raise ValueError("phase 3.2 SFT checkpoint is required")
    reference = json.loads(sft_summary.read_text(encoding="utf-8"))
    manifest_sha = hashlib.sha256((data_dir / "manifest.json").read_bytes()).hexdigest()
    if manifest_sha != reference["data_manifest_sha256"]:
        raise ValueError("preference data manifest differs from SFT checkpoint")
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint, local_files_only=True)
    sft_train = read_jsonl(data_dir / "sft_train.jsonl")
    system = sft_train[0]["messages"][0]["content"]
    pairs = {name: [encode_pair(row, system, tokenizer, max_length)
                    for row in read_jsonl(data_dir / f"preferences_{name}.jsonl")]
             for name in ("train", "validation", "test")}
    if any(not rows for rows in pairs.values()):
        raise ValueError("train, validation, and test preference splits must be nonempty")
    for name in pairs:
        sft_ids = {row["ticket_id"] for row in read_jsonl(data_dir / f"sft_{name}.jsonl")}
        if {row["ticket_id"] for row in pairs[name]} != sft_ids:
            raise ValueError(f"{name} preference ticket IDs differ from SFT split")
    model = AutoModelForSequenceClassification.from_pretrained(
        sft_checkpoint, num_labels=1, local_files_only=True)
    model.config.pad_token_id = tokenizer.eos_token_id
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.score.parameters():
        parameter.requires_grad_(True)
    model.to(device)
    initial = {name: assess(model, rows, device, torch)
               for name, rows in pairs.items()}
    optimizer = torch.optim.AdamW(model.score.parameters(), lr=learning_rate)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(pairs["train"])))
        random.shuffle(order)
        losses = []
        for position in order:
            chosen, rejected = score_pair(model, pairs["train"][position], device, torch)
            loss = torch.nn.functional.softplus(-(chosen - rejected))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        validation = assess(model, pairs["validation"], device, torch)
        history.append({"epoch": epoch, "mean_training_loss": sum(losses) / len(losses),
                        "validation_accuracy": validation["ranking_accuracy"],
                        "validation_margin": validation["mean_margin"]})
    final = {name: assess(model, rows, device, torch)
             for name, rows in pairs.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "checkpoint"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    reloaded = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, local_files_only=True).to(device)
    reloaded_test = assess(reloaded, pairs["test"], device, torch)
    if abs(reloaded_test["mean_margin"] - final["test"]["mean_margin"]) > 1e-5:
        raise RuntimeError("reloaded reward model changed test ranking margin")
    summary = {"source_model": reference["model"],
               "source_revision": reference["model_revision"],
               "sft_manifest_sha256": manifest_sha, "device": str(device),
               "torch_version": torch.__version__,
               "transformers_version": transformers.__version__,
               "seed": seed, "epochs": epochs, "learning_rate": learning_rate,
               "max_length": max_length,
               "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
               "total_parameters": sum(p.numel() for p in model.parameters()),
               "initial": initial, "final": final,
               "reload_test_margin": reloaded_test["mean_margin"],
               "history": history}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("experiments/sample"))
    parser.add_argument("--sft-checkpoint", type=Path,
                        default=Path("experiments/phase_3_2/checkpoint"))
    parser.add_argument("--sft-summary", type=Path,
                        default=Path("experiments/phase_3_2/summary.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/phase_3_4"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    summary = train(args.data_dir, args.sft_checkpoint, args.sft_summary, args.output,
                    args.epochs, args.learning_rate, args.max_length, args.seed)
    compact = {key: value for key, value in summary.items() if key not in ("initial", "final")}
    compact["initial"] = {name: {key: value for key, value in result.items() if key != "details"}
                          for name, result in summary["initial"].items()}
    compact["final"] = {name: {key: value for key, value in result.items() if key != "details"}
                        for name, result in summary["final"].items()}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
