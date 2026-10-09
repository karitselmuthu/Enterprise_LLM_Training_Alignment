"""Full-parameter supervised fine-tuning on prepared support conversations."""

import argparse
import hashlib
import json
import random
from pathlib import Path


DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def render_prompt(messages: list[dict]) -> str:
    if [item.get("role") for item in messages] != ["system", "user", "assistant"]:
        raise ValueError("expected system, user, assistant messages")
    return ("### System\n" + messages[0]["content"] + "\n\n"
            "### User\n" + messages[1]["content"] + "\n\n### Assistant\n")


def encode_example(row: dict, tokenizer, max_length: int) -> dict:
    messages = row["messages"]
    prompt_ids = tokenizer.encode(render_prompt(messages), add_special_tokens=False)
    answer_ids = tokenizer.encode(messages[2]["content"], add_special_tokens=False)
    answer_ids.append(tokenizer.eos_token_id)
    input_ids = prompt_ids + answer_ids
    if len(input_ids) > max_length:
        raise ValueError(f"{row['ticket_id']}: {len(input_ids)} tokens exceeds --max-length {max_length}")
    return {"ticket_id": row["ticket_id"], "input_ids": input_ids,
            "labels": [-100] * len(prompt_ids) + answer_ids,
            "prompt_ids": prompt_ids, "answer_tokens": len(answer_ids)}


def score(model, examples: list[dict], device, torch) -> float:
    model.eval()
    weighted_loss = 0.0
    tokens = 0
    with torch.no_grad():
        for example in examples:
            inputs = torch.tensor([example["input_ids"]], device=device)
            labels = torch.tensor([example["labels"]], device=device)
            loss = model(input_ids=inputs, labels=labels).loss.item()
            weighted_loss += loss * example["answer_tokens"]
            tokens += example["answer_tokens"]
    return weighted_loss / tokens


def predict(model, tokenizer, examples: list[dict], device, torch,
            max_new_tokens: int) -> list[dict]:
    model.eval()
    outputs = []
    with torch.no_grad():
        for example in examples:
            prompt = torch.tensor([example["prompt_ids"]], device=device)
            generated = model.generate(input_ids=prompt, max_new_tokens=max_new_tokens,
                                       do_sample=False, pad_token_id=tokenizer.eos_token_id)
            response_ids = generated[0, prompt.shape[1]:].tolist()
            response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            outputs.append({"ticket_id": example["ticket_id"], "response": response})
    return outputs


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def train(data_dir: Path, output_dir: Path, model_name: str = DEFAULT_MODEL,
          revision: str = DEFAULT_REVISION,
          epochs: int = 2, learning_rate: float = 5e-5, max_length: int = 384,
          max_new_tokens: int = 64, seed: int = 7, local_files_only: bool = False) -> dict:
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install requirements-sft.txt in a Python environment") from exc
    if epochs < 1 or learning_rate <= 0 or max_length < 16 or max_new_tokens < 1:
        raise ValueError("invalid training settings")
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision,
                                               local_files_only=local_files_only)
    if tokenizer.eos_token_id is None:
        raise ValueError("model tokenizer needs an EOS token")
    split_rows = {name: read_jsonl(data_dir / f"sft_{name}.jsonl")
                  for name in ("train", "validation", "test")}
    examples = {name: [encode_example(row, tokenizer, max_length) for row in rows]
                for name, rows in split_rows.items()}
    if any(not rows for rows in examples.values()):
        raise ValueError("train, validation, and test splits must be nonempty")
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision,
                                                 local_files_only=local_files_only)
    model.to(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_validation_loss = score(model, examples["validation"], device, torch)
    baseline_test_loss = score(model, examples["test"], device, torch)
    baseline_predictions = predict(model, tokenizer, examples["test"], device, torch,
                                   max_new_tokens)
    write_jsonl(output_dir / "baseline_predictions.jsonl", baseline_predictions)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(examples["train"])))
        random.shuffle(order)
        losses = []
        for position in order:
            example = examples["train"][position]
            inputs = torch.tensor([example["input_ids"]], device=device)
            labels = torch.tensor([example["labels"]], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=inputs, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        history.append({"epoch": epoch, "training_loss": sum(losses) / len(losses),
                        "validation_loss": score(model, examples["validation"], device, torch)})
    final_test_loss = score(model, examples["test"], device, torch)
    predictions = predict(model, tokenizer, examples["test"], device, torch, max_new_tokens)
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    checkpoint = output_dir / "checkpoint"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    reloaded = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True).to(device)
    reload_validation_loss = score(reloaded, examples["validation"], device, torch)
    if abs(reload_validation_loss - history[-1]["validation_loss"]) > 1e-4:
        raise RuntimeError("reloaded checkpoint changed validation loss")
    from shared.evaluation.evaluate import evaluate
    gold = data_dir / "evaluation.jsonl"
    baseline_checks = evaluate(gold, output_dir / "baseline_predictions.jsonl")
    final_checks = evaluate(gold, output_dir / "predictions.jsonl")
    summary = {"model": model_name, "model_revision": revision,
               "data_manifest_sha256": hashlib.sha256((data_dir / "manifest.json").read_bytes()).hexdigest(),
               "device": str(device), "torch_version": torch.__version__,
               "transformers_version": transformers.__version__, "seed": seed,
               "epochs": epochs, "learning_rate": learning_rate,
               "max_length": max_length, "max_new_tokens": max_new_tokens,
               "train_examples": len(examples["train"]),
               "validation_examples": len(examples["validation"]),
               "test_examples": len(examples["test"]),
               "baseline_validation_loss": baseline_validation_loss,
               "final_validation_loss": history[-1]["validation_loss"],
               "reloaded_validation_loss": reload_validation_loss,
               "baseline_test_loss": baseline_test_loss,
               "final_test_loss": final_test_loss,
               "baseline_checks": {k: v for k, v in baseline_checks.items() if k != "details"},
               "final_checks": {k: v for k, v in final_checks.items() if k != "details"},
               "history": history}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("experiments/sample"))
    parser.add_argument("--output", type=Path, default=Path("experiments/phase_3_2"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train(args.data_dir, args.output, args.model, args.revision, args.epochs,
                           args.learning_rate, args.max_length, args.max_new_tokens,
                           args.seed, args.local_files_only), indent=2))


if __name__ == "__main__":
    main()
