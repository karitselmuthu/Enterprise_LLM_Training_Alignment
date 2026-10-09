"""Exact-answer reinforcement learning on tiny addition problems."""

import argparse
import json
import math
import random
from pathlib import Path


MAX_OPERAND = 4
ANSWERS = 2 * MAX_OPERAND + 1
FEATURES = 2 * (MAX_OPERAND + 1)


def verify(a: int, b: int, answer: int) -> int:
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in (a, b, answer)):
        raise ValueError("integers are required")
    return int(answer == a + b)


def probabilities(weights: list[list[float]], a: int, b: int) -> list[float]:
    logits = [row[a] + row[MAX_OPERAND + 1 + b] for row in weights]
    offset = max(logits)
    values = [math.exp(logit - offset) for logit in logits]
    total = sum(values)
    return [value / total for value in values]


def cases() -> dict[str, list[tuple[int, int]]]:
    all_cases = [(a, b) for a in range(MAX_OPERAND + 1) for b in range(MAX_OPERAND + 1)]
    test = [(a, b) for a, b in all_cases if (a * 7 + b * 11) % 5 == 0]
    train = [case for case in all_cases if case not in test]
    return {"train": train, "test": test}


def evaluate(weights: list[list[float]], rows: list[tuple[int, int]]) -> dict:
    correct = 0
    expected = 0.0
    for a, b in rows:
        probs = probabilities(weights, a, b)
        correct += verify(a, b, max(range(ANSWERS), key=lambda index: probs[index]))
        expected += probs[a + b]
    return {"cases": len(rows), "greedy_exact_accuracy": correct / len(rows),
            "mean_exact_answer_probability": expected / len(rows)}


def reinforce_update(weights: list[list[float]], a: int, b: int, action: int,
                     reward: float, baseline: float, learning_rate: float) -> None:
    probs = probabilities(weights, a, b)
    for answer in range(ANSWERS):
        gradient = (int(answer == action) - probs[answer]) * (reward - baseline)
        weights[answer][a] += learning_rate * gradient
        weights[answer][MAX_OPERAND + 1 + b] += learning_rate * gradient


def run(output: Path, epochs: int = 1000, learning_rate: float = 0.1,
        seed: int = 7) -> dict:
    if epochs < 1 or learning_rate <= 0:
        raise ValueError("invalid RLVR settings")
    rng = random.Random(seed)
    weights = [[0.0] * FEATURES for _ in range(ANSWERS)]
    splits = cases()
    initial = {name: evaluate(weights, rows) for name, rows in splits.items()}
    rewards = []
    for _ in range(epochs):
        order = splits["train"][:]
        rng.shuffle(order)
        for a, b in order:
            probs = probabilities(weights, a, b)
            draw = rng.random()
            cumulative = 0.0
            action = ANSWERS - 1
            for index, prob in enumerate(probs):
                cumulative += prob
                if draw < cumulative:
                    action = index
                    break
            reward = verify(a, b, action)
            # Baseline is the current policy's expected verifiable reward.
            reinforce_update(weights, a, b, action, reward, probs[a + b], learning_rate)
            rewards.append(reward)
    final = {name: evaluate(weights, rows) for name, rows in splits.items()}
    result = {"algorithm": "REINFORCE with exact-answer reward", "seed": seed,
              "epochs": epochs, "learning_rate": learning_rate,
              "training_examples": len(splits["train"]), "held_out_examples": len(splits["test"]),
              "updates": len(rewards), "mean_reward_first_100": sum(rewards[:100]) / 100,
              "mean_reward_last_100": sum(rewards[-100:]) / 100,
              "initial": initial, "final": final,
              "held_out_predictions": [{"a": a, "b": b, "expected": a + b,
                                        "predicted": max(range(ANSWERS), key=lambda answer:
                                                         probabilities(weights, a, b)[answer])}
                                       for a, b in splits["test"]]}
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiments/phase_4_6"))
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
