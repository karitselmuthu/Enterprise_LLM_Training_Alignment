"""Train a LoRA SFT adapter and compare it with the phase 3.2 full SFT run."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from shared.evaluation.evaluate import evaluate
from shared.training.sft import encode_example, predict, read_jsonl, score, write_jsonl


def read_reference(path: Path, data_dir: Path) -> dict:
    reference = json.loads(path.read_text(encoding="utf-8"))
    digest = hashlib.sha256((data_dir / "manifest.json").read_bytes()).hexdigest()
    if reference["data_manifest_sha256"] != digest:
        raise ValueError("full SFT reference used a different data manifest")
    for name in ("model", "model_revision", "epochs", "learning_rate", "max_length",
                 "max_new_tokens", "seed"):
        if name not in reference:
            raise ValueError(f"full SFT reference lacks {name}")
    return reference


def artifact_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def train(data_dir: Path, full_reference: Path, output_dir: Path,
          full_checkpoint: Path, local_files_only: bool = False) -> dict:
    try:
        import peft
        import torch
        import transformers
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install requirements-sft.txt in a Python environment") from exc
    reference = read_reference(full_reference, data_dir)
    if not (full_checkpoint / "model.safetensors").is_file():
        raise ValueError("full SFT checkpoint is missing model.safetensors")
    torch.manual_seed(reference["seed"])
    random.seed(reference["seed"])
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(reference["model"],
                                               revision=reference["model_revision"],
                                               local_files_only=local_files_only)
    examples = {name: [encode_example(row, tokenizer, reference["max_length"])
                       for row in read_jsonl(data_dir / f"sft_{name}.jsonl")]
                for name in ("train", "validation", "test")}
    for name in examples:
        if len(examples[name]) != reference[f"{name}_examples"]:
            raise ValueError(f"{name} example count differs from full SFT run")
    base = AutoModelForCausalLM.from_pretrained(reference["model"],
                                                revision=reference["model_revision"],
                                                local_files_only=local_files_only)
    config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
                        target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(base, config).to(device)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    baseline_validation_loss = score(model, examples["validation"], device, torch)
    if abs(baseline_validation_loss - reference["baseline_validation_loss"]) > 1e-3:
        raise RuntimeError("LoRA run does not start from the same pretrained baseline")
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters()
                                   if parameter.requires_grad), lr=reference["learning_rate"])
    history = []
    start = time.perf_counter()
    for epoch in range(1, reference["epochs"] + 1):
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
            torch.nn.utils.clip_grad_norm_((parameter for parameter in model.parameters()
                                            if parameter.requires_grad), 1.0)
            optimizer.step()
            losses.append(loss.item())
        history.append({"epoch": epoch, "training_loss": sum(losses) / len(losses),
                        "validation_loss": score(model, examples["validation"], device, torch)})
    training_seconds = time.perf_counter() - start
    test_loss = score(model, examples["test"], device, torch)
    predictions = predict(model, tokenizer, examples["test"], device, torch,
                          reference["max_new_tokens"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    fresh_base = AutoModelForCausalLM.from_pretrained(reference["model"],
                                                      revision=reference["model_revision"],
                                                      local_files_only=local_files_only)
    reloaded = PeftModel.from_pretrained(fresh_base, adapter_dir).to(device)
    reloaded_validation_loss = score(reloaded, examples["validation"], device, torch)
    if abs(reloaded_validation_loss - history[-1]["validation_loss"]) > 1e-4:
        raise RuntimeError("reloaded LoRA adapter changed validation loss")
    checks = evaluate(data_dir / "evaluation.jsonl", output_dir / "predictions.jsonl")
    summary = {"model": reference["model"], "model_revision": reference["model_revision"],
               "data_manifest_sha256": reference["data_manifest_sha256"],
               "seed": reference["seed"], "epochs": reference["epochs"],
               "learning_rate": reference["learning_rate"],
               "max_length": reference["max_length"],
               "max_new_tokens": reference["max_new_tokens"],
               "device": str(device), "torch_version": torch.__version__,
               "transformers_version": transformers.__version__,
               "peft_version": peft.__version__,
               "lora": {"rank": 8, "alpha": 16, "dropout": 0.0,
                        "target_modules": ["q_proj", "v_proj"]},
               "trainable_parameters": trainable, "total_parameters": total,
               "trainable_fraction": trainable / total,
               "training_seconds": training_seconds,
               "adapter_bytes": artifact_bytes(adapter_dir),
               "full_checkpoint_bytes": artifact_bytes(full_checkpoint),
               "baseline_validation_loss": baseline_validation_loss,
               "full_validation_loss": reference["final_validation_loss"],
               "lora_validation_loss": history[-1]["validation_loss"],
               "reloaded_validation_loss": reloaded_validation_loss,
               "full_test_loss": reference["final_test_loss"],
               "lora_test_loss": test_loss,
               "full_checks": reference["final_checks"],
               "lora_checks": {key: value for key, value in checks.items() if key != "details"},
               "history": history}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("experiments/sample"))
    parser.add_argument("--full-reference", type=Path,
                        default=Path("experiments/phase_3_2/summary.json"))
    parser.add_argument("--full-checkpoint", type=Path,
                        default=Path("experiments/phase_3_2/checkpoint"))
    parser.add_argument("--output", type=Path, default=Path("experiments/phase_3_3"))
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train(args.data_dir, args.full_reference, args.output,
                           args.full_checkpoint, args.local_files_only), indent=2))


if __name__ == "__main__":
    main()
