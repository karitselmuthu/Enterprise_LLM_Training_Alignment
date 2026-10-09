"""Small DPO learning run on support preference pairs."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from shared.evaluation.evaluate import evaluate
from shared.training.sft import encode_example, predict, read_jsonl, write_jsonl


def build_pairs(rows: list[dict], system: str, tokenizer, max_length: int) -> list[dict]:
    result = []
    for row in rows:
        if not row["chosen"].strip() or not row["rejected"].strip() or row["chosen"] == row["rejected"]:
            raise ValueError(f"{row['ticket_id']}: invalid preference pair")
        variants = {}
        for label in ("chosen", "rejected"):
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": row["prompt"]},
                        {"role": "assistant", "content": row[label]}]
            variants[label] = encode_example({"ticket_id": row["ticket_id"],
                                               "messages": messages}, tokenizer, max_length)
        result.append({"ticket_id": row["ticket_id"], **variants})
    return result


def response_logprob(model, example: dict, torch):
    inputs = torch.tensor([example["input_ids"]], device=next(model.parameters()).device)
    labels = torch.tensor([example["labels"]], device=inputs.device)[:, 1:]
    logits = model(input_ids=inputs).logits[:, :-1, :].float()
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    selected = log_probs.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (selected * (labels != -100)).sum()


def assess(model, pairs: list[dict], reference_logps: dict, beta: float, torch) -> dict:
    model.eval()
    details = []
    with torch.no_grad():
        for pair in pairs:
            chosen = response_logprob(model, pair["chosen"], torch).item()
            rejected = response_logprob(model, pair["rejected"], torch).item()
            ref_chosen, ref_rejected = reference_logps[pair["ticket_id"]]
            margin = beta * ((chosen - rejected) - (ref_chosen - ref_rejected))
            details.append({"ticket_id": pair["ticket_id"], "implicit_reward_margin": margin,
                            "preferred_ranked_higher": margin > 0})
    return {"pairs": len(details),
            "implicit_reward_accuracy": sum(item["preferred_ranked_higher"] for item in details) / len(details),
            "mean_implicit_reward_margin": sum(item["implicit_reward_margin"] for item in details) / len(details),
            "details": details}


def train(data_dir: Path, sft_checkpoint: Path, sft_summary: Path, output_dir: Path,
          epochs: int = 2, learning_rate: float = 1e-4, beta: float = 0.1,
          seed: int = 7) -> dict:
    try:
        import peft
        import torch
        import transformers
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install requirements-sft.txt in a Python environment") from exc
    if epochs < 1 or learning_rate <= 0 or beta <= 0:
        raise ValueError("invalid DPO settings")
    reference_summary = json.loads(sft_summary.read_text(encoding="utf-8"))
    manifest_sha = hashlib.sha256((data_dir / "manifest.json").read_bytes()).hexdigest()
    if manifest_sha != reference_summary["data_manifest_sha256"]:
        raise ValueError("preference data manifest differs from SFT checkpoint")
    if not (sft_checkpoint / "model.safetensors").is_file():
        raise ValueError("phase 3.2 SFT checkpoint is required")
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint, local_files_only=True)
    system = read_jsonl(data_dir / "sft_train.jsonl")[0]["messages"][0]["content"]
    pairs = {name: build_pairs(read_jsonl(data_dir / f"preferences_{name}.jsonl"),
                               system, tokenizer, reference_summary["max_length"])
             for name in ("train", "validation", "test")}
    if any(not rows for rows in pairs.values()):
        raise ValueError("all preference splits must be nonempty")
    for name, rows in pairs.items():
        sft_ids = {row["ticket_id"] for row in read_jsonl(data_dir / f"sft_{name}.jsonl")}
        if {row["ticket_id"] for row in rows} != sft_ids:
            raise ValueError(f"{name} preference IDs differ from SFT split")
    reference = AutoModelForCausalLM.from_pretrained(sft_checkpoint, local_files_only=True).to(device)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    reference_logps = {}
    with torch.no_grad():
        for rows in pairs.values():
            for pair in rows:
                reference_logps[pair["ticket_id"]] = (
                    response_logprob(reference, pair["chosen"], torch).item(),
                    response_logprob(reference, pair["rejected"], torch).item())
    base = AutoModelForCausalLM.from_pretrained(sft_checkpoint, local_files_only=True)
    config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
                        target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM)
    policy = get_peft_model(base, config).to(device)
    initial = {name: assess(policy, rows, reference_logps, beta, torch)
               for name, rows in pairs.items()}
    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad),
                                  lr=learning_rate)
    history = []
    for epoch in range(1, epochs + 1):
        policy.train()
        order = list(range(len(pairs["train"])))
        random.shuffle(order)
        losses = []
        for position in order:
            pair = pairs["train"][position]
            chosen = response_logprob(policy, pair["chosen"], torch)
            rejected = response_logprob(policy, pair["rejected"], torch)
            ref_chosen, ref_rejected = reference_logps[pair["ticket_id"]]
            margin = beta * ((chosen - rejected) - (ref_chosen - ref_rejected))
            loss = torch.nn.functional.softplus(-margin)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in policy.parameters() if p.requires_grad), 1.0)
            optimizer.step()
            losses.append(loss.item())
        validation = assess(policy, pairs["validation"], reference_logps, beta, torch)
        history.append({"epoch": epoch, "mean_training_loss": sum(losses) / len(losses),
                        "validation_margin": validation["mean_implicit_reward_margin"]})
    final = {name: assess(policy, rows, reference_logps, beta, torch)
             for name, rows in pairs.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = predict(policy, tokenizer, [pair["chosen"] for pair in pairs["test"]],
                          device, torch, reference_summary["max_new_tokens"])
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    checks = evaluate(data_dir / "evaluation.jsonl", output_dir / "predictions.jsonl")
    adapter_dir = output_dir / "adapter"
    policy.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    fresh_base = AutoModelForCausalLM.from_pretrained(sft_checkpoint, local_files_only=True)
    reloaded = PeftModel.from_pretrained(fresh_base, adapter_dir).to(device)
    reload_test = assess(reloaded, pairs["test"], reference_logps, beta, torch)
    if abs(reload_test["mean_implicit_reward_margin"] - final["test"]["mean_implicit_reward_margin"]) > 1e-4:
        raise RuntimeError("reloaded DPO adapter changed test margin")
    summary = {"source_model": reference_summary["model"],
               "source_revision": reference_summary["model_revision"],
               "sft_manifest_sha256": manifest_sha, "seed": seed,
               "epochs": epochs, "learning_rate": learning_rate, "beta": beta,
               "device": str(device), "torch_version": torch.__version__,
               "transformers_version": transformers.__version__, "peft_version": peft.__version__,
               "trainable_parameters": sum(p.numel() for p in policy.parameters() if p.requires_grad),
               "initial": initial, "final": final, "history": history,
               "reload_test_margin": reload_test["mean_implicit_reward_margin"],
               "generated_answer_checks": {k: v for k, v in checks.items() if k != "details"},
               "sft_generated_answer_checks": reference_summary["final_checks"]}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("experiments/sample"))
    parser.add_argument("--sft-checkpoint", type=Path,
                        default=Path("experiments/phase_3_2/checkpoint"))
    parser.add_argument("--sft-summary", type=Path,
                        default=Path("experiments/phase_3_2/summary.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/phase_3_5"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    result = train(args.data_dir, args.sft_checkpoint, args.sft_summary, args.output,
                   args.epochs, args.learning_rate, args.beta, args.seed)
    compact = {key: value for key, value in result.items() if key not in ("initial", "final")}
    for stage in ("initial", "final"):
        compact[stage] = {name: {k: v for k, v in details.items() if k != "details"}
                          for name, details in result[stage].items()}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
