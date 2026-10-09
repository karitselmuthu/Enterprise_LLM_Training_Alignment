"""Small logit-distillation exercise from SmolLM2-360M to SmolLM2-135M."""

import argparse
import json
import time
from pathlib import Path

from shared.evaluation.evaluate import evaluate
from shared.training.sft import encode_example, predict, read_jsonl, score, write_jsonl


def masked_kl(student_logits, teacher_logits, labels, torch, temperature: float):
    mask = labels[:, 1:] != -100
    student = student_logits[:, :-1, :].float()[mask] / temperature
    teacher = teacher_logits[:, :-1, :].float()[mask] / temperature
    return torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student, dim=-1),
        torch.nn.functional.softmax(teacher, dim=-1), reduction="batchmean") * temperature ** 2


def timed_loss(model, examples, device, torch):
    start = time.perf_counter()
    loss = score(model, examples, device, torch)
    return {"answer_token_loss": loss, "elapsed_seconds": time.perf_counter() - start}


def train(data_dir: Path, output: Path, teacher_name: str = "HuggingFaceTB/SmolLM2-360M",
          student_name: str = "HuggingFaceTB/SmolLM2-135M", temperature: float = 2.0,
          alpha: float = 0.5, learning_rate: float = 1e-4, seed: int = 7) -> dict:
    try:
        import peft
        import torch
        import transformers
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install requirements-sft.txt in a Python environment") from exc
    if temperature <= 0 or not 0 <= alpha <= 1 or learning_rate <= 0:
        raise ValueError("invalid distillation settings")
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    student_tokenizer = AutoTokenizer.from_pretrained(student_name, local_files_only=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_name, local_files_only=True)
    rows = {name: read_jsonl(data_dir / f"sft_{name}.jsonl")
            for name in ("train", "validation", "test")}
    for group in rows.values():
        for row in group:
            student_example = encode_example(row, student_tokenizer, 384)
            teacher_example = encode_example(row, teacher_tokenizer, 384)
            if student_example["input_ids"] != teacher_example["input_ids"]:
                raise ValueError("teacher/student token IDs differ")
    examples = {name: [encode_example(row, student_tokenizer, 384) for row in group]
                for name, group in rows.items()}
    teacher = AutoModelForCausalLM.from_pretrained(teacher_name, local_files_only=True).to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    base = AutoModelForCausalLM.from_pretrained(student_name, local_files_only=True).to(device)
    initial = {name: timed_loss(base, examples[name], device, torch)
               for name in ("validation", "test")}
    teacher_results = {name: timed_loss(teacher, examples[name], device, torch)
                       for name in ("validation", "test")}
    config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
                        target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM)
    student = get_peft_model(base, config)
    trainable_parameters = sum(p.numel() for p in student.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad),
                                  lr=learning_rate)
    losses = []
    start = time.perf_counter()
    for example in examples["train"]:
        inputs = torch.tensor([example["input_ids"]], device=device)
        labels = torch.tensor([example["labels"]], device=device)
        with torch.no_grad():
            target_logits = teacher(input_ids=inputs).logits
        student.train()
        result = student(input_ids=inputs, labels=labels)
        soft_loss = masked_kl(result.logits, target_logits, labels, torch, temperature)
        loss = alpha * soft_loss + (1 - alpha) * result.loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append({"hard_loss": result.loss.item(), "soft_kl": soft_loss.item(),
                       "combined_loss": loss.item()})
    training_seconds = time.perf_counter() - start
    final = {name: timed_loss(student, examples[name], device, torch)
             for name in ("validation", "test")}
    output.mkdir(parents=True, exist_ok=True)
    predictions = predict(student, student_tokenizer, examples["test"], device, torch, 64)
    write_jsonl(output / "predictions.jsonl", predictions)
    checks = evaluate(data_dir / "evaluation.jsonl", output / "predictions.jsonl")
    adapter = output / "adapter"
    student.save_pretrained(adapter)
    student_tokenizer.save_pretrained(adapter)
    del student, base
    fresh_base = AutoModelForCausalLM.from_pretrained(student_name, local_files_only=True).to(device)
    reloaded = PeftModel.from_pretrained(fresh_base, adapter).to(device)
    reload_loss = score(reloaded, examples["test"], device, torch)
    if abs(reload_loss - final["test"]["answer_token_loss"]) > 1e-4:
        raise RuntimeError("reloaded adapter changed test loss")
    summary = {"teacher": teacher_name, "student": student_name, "device": str(device),
               "torch_version": torch.__version__, "transformers_version": transformers.__version__,
               "peft_version": peft.__version__, "seed": seed, "temperature": temperature,
               "soft_loss_weight": alpha, "learning_rate": learning_rate,
               "teacher_parameters": sum(p.numel() for p in teacher.parameters()),
               "student_parameters": sum(p.numel() for p in fresh_base.parameters()),
               "trainable_parameters": trainable_parameters,
               "initial": initial, "teacher_eval": teacher_results, "final": final,
               "train_steps": losses, "training_seconds": training_seconds,
               "reload_test_loss": reload_loss,
               "generated_answer_checks": {k: v for k, v in checks.items() if k != "details"}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("experiments/sample"))
    parser.add_argument("--output", type=Path, default=Path("experiments/phase_3_7"))
    args = parser.parse_args()
    print(json.dumps(train(args.data_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
