"""Clipped PPO updates for the synthetic two-action response selector."""

import argparse
import json
import math
import random
from pathlib import Path

from shared.training.policy_gradient import evaluate, features, probabilities
from shared.training.sft import read_jsonl


def clipped_surrogate(ratio: float, advantage: float, epsilon: float) -> tuple[float, bool]:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    clipped = max(1 - epsilon, min(1 + epsilon, ratio))
    active = not ((advantage > 0 and ratio > 1 + epsilon) or
                  (advantage < 0 and ratio < 1 - epsilon))
    return min(ratio * advantage, clipped * advantage), active


def update(weights: list[float], case: dict, action: int, old_probability: float,
           advantage: float, epsilon: float, learning_rate: float) -> tuple[list[float], bool]:
    rows = features(case)
    probs = probabilities(weights, rows)
    ratio = probs[action] / old_probability
    _, active = clipped_surrogate(ratio, advantage, epsilon)
    if not active:
        return weights, True
    expected = [sum(probs[index] * rows[index][dim] for index in range(2))
                for dim in range(len(weights))]
    gradient = [ratio * advantage * (rows[action][dim] - expected[dim])
                for dim in range(len(weights))]
    return [weight + learning_rate * value for weight, value in zip(weights, gradient)], False


def run(data_dir: Path, output: Path, rollouts: int = 50, update_epochs: int = 4,
        epsilon: float = 0.2, learning_rate: float = 0.05, seed: int = 7) -> dict:
    if rollouts < 1 or update_epochs < 1 or learning_rate <= 0:
        raise ValueError("invalid PPO settings")
    cases = {split: read_jsonl(data_dir / f"{split}.jsonl")
             for split in ("train", "validation", "test")}
    weights = [0.0, 0.0]
    rng = random.Random(seed)
    initial = {split: evaluate(weights, rows) for split, rows in cases.items()}
    sampled_rewards = []
    clipped_steps = 0
    optimizer_steps = 0
    rollout_kls = []
    for _ in range(rollouts):
        old_weights = weights[:]
        batch = []
        for case in cases["train"]:
            probs = probabilities(old_weights, features(case))
            action = 0 if rng.random() < probs[0] else 1
            rewards = [item["score"] for item in case["rewards"]]
            baseline = sum(p * reward for p, reward in zip(probs, rewards))
            batch.append((case, action, probs[action], rewards[action] - baseline))
            sampled_rewards.append(rewards[action])
        for _ in range(update_epochs):
            for case, action, old_probability, advantage in batch:
                weights, clipped = update(weights, case, action, old_probability,
                                          advantage, epsilon, learning_rate)
                clipped_steps += clipped
                optimizer_steps += 1
        kls = []
        for case in cases["train"]:
            old = probabilities(old_weights, features(case))
            new = probabilities(weights, features(case))
            kls.append(sum(p * math.log(p / q) for p, q in zip(old, new)))
        rollout_kls.append(sum(kls) / len(kls))
    final = {split: evaluate(weights, rows) for split, rows in cases.items()}
    result = {"algorithm": "clipped PPO response selection", "rollouts": rollouts,
              "update_epochs": update_epochs, "epsilon": epsilon, "learning_rate": learning_rate,
              "seed": seed, "sampled_actions": len(sampled_rewards),
              "optimizer_steps": optimizer_steps, "clipped_steps": clipped_steps,
              "mean_rollout_kl": sum(rollout_kls) / len(rollout_kls),
              "mean_sampled_reward_first_40": sum(sampled_rewards[:40]) / 40,
              "mean_sampled_reward_last_40": sum(sampled_rewards[-40:]) / 40,
              "weights": weights, "initial": initial, "final": final}
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("experiments/phase_4_1"))
    parser.add_argument("--output", type=Path, default=Path("experiments/phase_4_3"))
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
