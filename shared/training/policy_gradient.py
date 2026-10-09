"""REINFORCE on the two-response synthetic support bandit."""

import argparse
import json
import math
import random
from pathlib import Path

from shared.training.sft import read_jsonl


def features(case: dict) -> list[list[float]]:
    return [[reward["required_coverage"], -float(reward["forbidden_hits"])]
            for reward in case["rewards"]]


def probabilities(weights: list[float], candidate_features: list[list[float]]) -> list[float]:
    scores = [sum(weight * value for weight, value in zip(weights, row))
              for row in candidate_features]
    offset = max(scores)
    exp_scores = [math.exp(score - offset) for score in scores]
    total = sum(exp_scores)
    return [value / total for value in exp_scores]


def evaluate(weights: list[float], cases: list[dict]) -> dict:
    chosen = []
    for case in cases:
        probs = probabilities(weights, features(case))
        selected = max(range(2), key=lambda index: probs[index])
        chosen.append({"ticket_id": case["ticket_id"], "approved_probability": probs[case["approved_index"]],
                       "greedy_approved": selected == case["approved_index"],
                       "greedy_reward": case["rewards"][selected]["score"]})
    return {"cases": len(chosen),
            "mean_approved_probability": sum(row["approved_probability"] for row in chosen) / len(chosen),
            "greedy_approved_rate": sum(row["greedy_approved"] for row in chosen) / len(chosen),
            "mean_greedy_reward": sum(row["greedy_reward"] for row in chosen) / len(chosen)}


def reinforce_update(weights: list[float], case: dict, action: int, reward: float,
                     learning_rate: float) -> list[float]:
    rows = features(case)
    probs = probabilities(weights, rows)
    expected = [sum(probs[index] * rows[index][dim] for index in range(2))
                for dim in range(len(weights))]
    return [weight + learning_rate * reward * (rows[action][dim] - expected[dim])
            for dim, weight in enumerate(weights)]


def run(data_dir: Path, output: Path, epochs: int = 100, learning_rate: float = 0.1,
        seed: int = 7) -> dict:
    if epochs < 1 or learning_rate <= 0:
        raise ValueError("invalid policy-gradient settings")
    cases = {split: read_jsonl(data_dir / f"{split}.jsonl")
             for split in ("train", "validation", "test")}
    rng = random.Random(seed)
    weights = [0.0, 0.0]
    initial = {split: evaluate(weights, rows) for split, rows in cases.items()}
    sampled_rewards = []
    for _ in range(epochs):
        for case in cases["train"]:
            probs = probabilities(weights, features(case))
            action = 0 if rng.random() < probs[0] else 1
            reward = case["rewards"][action]["score"]
            weights = reinforce_update(weights, case, action, reward, learning_rate)
            sampled_rewards.append(reward)
    final = {split: evaluate(weights, rows) for split, rows in cases.items()}
    result = {"algorithm": "REINFORCE", "epochs": epochs, "learning_rate": learning_rate,
              "seed": seed, "updates": len(sampled_rewards), "weights": weights,
              "mean_sampled_reward_first_40": sum(sampled_rewards[:40]) / 40,
              "mean_sampled_reward_last_40": sum(sampled_rewards[-40:]) / 40,
              "initial": initial, "final": final}
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("experiments/phase_4_1"))
    parser.add_argument("--output", type=Path, default=Path("experiments/phase_4_2"))
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
