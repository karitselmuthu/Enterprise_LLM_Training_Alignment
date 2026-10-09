"""Group-relative clipped updates on pairs of fixed support responses."""

import argparse
import json
import math
from pathlib import Path

from shared.training.policy_gradient import evaluate, features, probabilities
from shared.training.ppo_bandit import update
from shared.training.sft import read_jsonl


def group_advantages(scores: list[float]) -> list[float]:
    if len(scores) < 2:
        raise ValueError("group needs at least two responses")
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
    if variance == 0:
        return [0.0] * len(scores)
    scale = math.sqrt(variance)
    return [(score - mean) / scale for score in scores]


def reference_kl_gradient(weights: list[float], rows: list[list[float]]) -> list[float]:
    probs = probabilities(weights, rows)
    reference = 1 / len(rows)
    log_ratios = [math.log(prob / reference) for prob in probs]
    kl = sum(prob * log_ratio for prob, log_ratio in zip(probs, log_ratios))
    return [sum(prob * (log_ratio - kl) * row[dim]
                for prob, log_ratio, row in zip(probs, log_ratios, rows))
            for dim in range(len(weights))]


def run(data_dir: Path, output: Path, iterations: int = 50, update_epochs: int = 4,
        epsilon: float = 0.2, learning_rate: float = 0.05,
        reference_kl_beta: float = 0.01) -> dict:
    if iterations < 1 or update_epochs < 1 or learning_rate <= 0 or reference_kl_beta < 0:
        raise ValueError("invalid GRPO settings")
    cases = {split: read_jsonl(data_dir / f"{split}.jsonl")
             for split in ("train", "validation", "test")}
    weights = [0.0, 0.0]
    initial = {split: evaluate(weights, rows) for split, rows in cases.items()}
    clipped = 0
    for _ in range(iterations):
        old = weights[:]
        groups = []
        for case in cases["train"]:
            old_probs = probabilities(old, features(case))
            rewards = [item["score"] for item in case["rewards"]]
            groups.append((case, old_probs, group_advantages(rewards)))
        for _ in range(update_epochs):
            for case, old_probs, advantages in groups:
                for action in range(2):
                    weights, was_clipped = update(weights, case, action, old_probs[action],
                                                  advantages[action], epsilon, learning_rate)
                    clipped += was_clipped
                penalty = reference_kl_gradient(weights, features(case))
                weights = [weight - learning_rate * reference_kl_beta * gradient
                           for weight, gradient in zip(weights, penalty)]
    final = {split: evaluate(weights, rows) for split, rows in cases.items()}
    result = {"algorithm": "group-relative clipped response selection",
              "iterations": iterations, "update_epochs": update_epochs,
              "group_size": 2, "epsilon": epsilon, "learning_rate": learning_rate,
              "reference_kl_beta": reference_kl_beta, "clipped_steps": clipped,
              "weights": weights, "initial": initial, "final": final}
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("experiments/phase_4_1"))
    parser.add_argument("--output", type=Path, default=Path("experiments/phase_4_5"))
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
