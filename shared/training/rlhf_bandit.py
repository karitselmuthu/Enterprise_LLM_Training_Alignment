"""Optimize a response selector using scores from the learned phase 3.4 reward model."""

import argparse
import json
import random
from pathlib import Path

from shared.training.policy_gradient import evaluate, features, probabilities
from shared.training.ppo_bandit import update
from shared.training.sft import read_jsonl


def attach_model_scores(cases: dict, reward_summary: dict) -> dict:
    scored = {}
    for split, rows in cases.items():
        details = {item["ticket_id"]: item for item in reward_summary["final"][split]["details"]}
        if set(details) != {row["ticket_id"] for row in rows}:
            raise ValueError(f"{split} reward scores differ from case IDs")
        scored[split] = []
        for case in rows:
            item = details[case["ticket_id"]]
            scores = [0.0, 0.0]
            scores[case["approved_index"]] = item["chosen_score"]
            scores[1 - case["approved_index"]] = item["rejected_score"]
            scored[split].append({**case, "model_scores": scores})
    return scored


def model_evaluation(weights: list[float], cases: list[dict]) -> dict:
    correct = 0
    expected = []
    for case in cases:
        probs = probabilities(weights, features(case))
        selected = max(range(2), key=lambda index: probs[index])
        correct += selected == case["approved_index"]
        expected.append(sum(p * score for p, score in zip(probs, case["model_scores"])))
    return {"greedy_approved_rate": correct / len(cases),
            "mean_expected_model_score": sum(expected) / len(expected)}


def run(data_dir: Path, reward_summary_path: Path, output: Path,
        rollouts: int = 50, update_epochs: int = 4, seed: int = 7) -> dict:
    if rollouts < 1 or update_epochs < 1:
        raise ValueError("invalid RLHF settings")
    cases = {split: read_jsonl(data_dir / f"{split}.jsonl")
             for split in ("train", "validation", "test")}
    summary = json.loads(reward_summary_path.read_text(encoding="utf-8"))
    cases = attach_model_scores(cases, summary)
    weights = [0.0, 0.0]
    rng = random.Random(seed)
    initial = {split: model_evaluation(weights, rows) for split, rows in cases.items()}
    sampled = []
    clipped = 0
    for _ in range(rollouts):
        old = weights[:]
        batch = []
        for case in cases["train"]:
            probs = probabilities(old, features(case))
            action = 0 if rng.random() < probs[0] else 1
            scores = case["model_scores"]
            baseline = sum(p * score for p, score in zip(probs, scores))
            batch.append((case, action, probs[action], scores[action] - baseline))
            sampled.append(scores[action])
        for _ in range(update_epochs):
            for case, action, old_probability, advantage in batch:
                weights, was_clipped = update(weights, case, action, old_probability,
                                              advantage, 0.2, 0.05)
                clipped += was_clipped
    final = {split: model_evaluation(weights, rows) for split, rows in cases.items()}
    rule_checks = {split: evaluate(weights, rows) for split, rows in cases.items()}
    result = {"algorithm": "learned-reward PPO response selection", "reward_source": str(reward_summary_path),
              "reward_model_validation_ranking_accuracy": summary["final"]["validation"]["ranking_accuracy"],
              "rollouts": rollouts, "update_epochs": update_epochs, "seed": seed,
              "sampled_actions": len(sampled), "clipped_steps": clipped,
              "mean_sampled_model_score_first_40": sum(sampled[:40]) / 40,
              "mean_sampled_model_score_last_40": sum(sampled[-40:]) / 40,
              "weights": weights, "initial": initial, "final": final, "rule_checks": rule_checks}
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("experiments/phase_4_1"))
    parser.add_argument("--reward-summary", type=Path,
                        default=Path("experiments/phase_3_4/summary.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/phase_4_4"))
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.reward_summary, args.output), indent=2))


if __name__ == "__main__":
    main()
