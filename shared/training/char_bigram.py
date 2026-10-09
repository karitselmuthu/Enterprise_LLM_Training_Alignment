"""Train a small character bigram language model without external packages.

The model learns P(next character | current character). It is deliberately much
simpler than a Transformer, but uses the same next-token cross-entropy objective.
"""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from shared.datasets.prepare import read_tickets, split_by_org


BOS = "\u0002"
EOS = "\u0003"
UNK = "\ufffd"


def documents(rows: list[dict]) -> list[str]:
    return ["\n".join([row["product"], row["issue"], *row["evidence"],
                       row["approved_response"]]) for row in rows]


def vocabulary(train_documents: list[str]) -> list[str]:
    return [BOS, EOS, UNK] + sorted(set("".join(train_documents)) - {BOS, EOS, UNK})


def transition_counts(texts: list[str], vocab: list[str]) -> tuple[list[list[int]], int]:
    index = {character: position for position, character in enumerate(vocab)}
    counts = [[0] * len(vocab) for _ in vocab]
    total = 0
    for text in texts:
        tokens = [BOS] + [character if character in index else UNK for character in text] + [EOS]
        for current, following in zip(tokens, tokens[1:]):
            counts[index[current]][index[following]] += 1
            total += 1
    return counts, total


def probabilities(logits: list[float]) -> list[float]:
    maximum = max(logits)
    scaled = [math.exp(value - maximum) for value in logits]
    total = sum(scaled)
    return [value / total for value in scaled]


def cross_entropy(weights: list[list[float]], counts: list[list[int]]) -> float:
    total = sum(sum(row) for row in counts)
    if total == 0:
        raise ValueError("no character transitions to score")
    loss = 0.0
    for logits, targets in zip(weights, counts):
        if not any(targets):
            continue
        values = probabilities(logits)
        loss -= sum(count * math.log(max(values[index], 1e-300))
                    for index, count in enumerate(targets) if count)
    return loss / total


def fit(train_counts: list[list[int]], validation_counts: list[list[int]],
        epochs: int, learning_rate: float) -> tuple[list[list[float]], list[dict]]:
    if epochs < 1 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    size = len(train_counts)
    weights = [[0.0] * size for _ in range(size)]
    history = [{"epoch": 0, "train_loss": cross_entropy(weights, train_counts),
                "validation_loss": cross_entropy(weights, validation_counts)}]
    for epoch in range(1, epochs + 1):
        for row_index, targets in enumerate(train_counts):
            row_total = sum(targets)
            if row_total == 0:
                continue
            values = probabilities(weights[row_index])
            for column in range(size):
                gradient = values[column] - targets[column] / row_total
                weights[row_index][column] -= learning_rate * gradient
        history.append({"epoch": epoch,
                        "train_loss": cross_entropy(weights, train_counts),
                        "validation_loss": cross_entropy(weights, validation_counts)})
    return weights, history


def generate(vocab: list[str], weights: list[list[float]], seed: int,
             max_chars: int = 200) -> str:
    rng = random.Random(seed)
    index = {character: position for position, character in enumerate(vocab)}
    current = BOS
    output = []
    for _ in range(max_chars):
        character = rng.choices(vocab, weights=probabilities(weights[index[current]]), k=1)[0]
        if character == EOS:
            break
        if character == BOS:
            continue
        output.append(character)
        current = character
    return "".join(output)


def load_checkpoint(path: Path) -> tuple[list[str], list[list[float]]]:
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if checkpoint.get("architecture") != "character_bigram":
        raise ValueError("checkpoint has an unsupported architecture")
    vocab = checkpoint["vocabulary"]
    weights = checkpoint["weights"]
    if len(weights) != len(vocab) or any(len(row) != len(vocab) for row in weights):
        raise ValueError("checkpoint weight dimensions do not match vocabulary")
    return vocab, weights


def train(ticket_path: Path, output_dir: Path, epochs: int = 100,
          learning_rate: float = 1.0, seed: int = 7) -> dict:
    splits = split_by_org(read_tickets(ticket_path))
    train_texts = documents(splits["train"])
    validation_texts = documents(splits["validation"])
    vocab = vocabulary(train_texts)
    training_counts, training_transitions = transition_counts(train_texts, vocab)
    validation_counts, validation_transitions = transition_counts(validation_texts, vocab)
    weights, history = fit(training_counts, validation_counts, epochs, learning_rate)
    checkpoint = {"architecture": "character_bigram", "vocabulary": vocab,
                  "weights": weights, "source_sha256": hashlib.sha256(ticket_path.read_bytes()).hexdigest(),
                  "epochs": epochs, "learning_rate": learning_rate, "seed": seed}
    summary = {"architecture": checkpoint["architecture"],
               "training_documents": len(train_texts),
               "validation_documents": len(validation_texts),
               "held_out_test_documents": len(splits["test"]),
               "training_transitions": training_transitions,
               "validation_transitions": validation_transitions,
               "vocabulary_size": len(vocab),
               "parameters": len(vocab) ** 2,
               "initial_train_loss": history[0]["train_loss"],
               "final_train_loss": history[-1]["train_loss"],
               "initial_validation_loss": history[0]["validation_loss"],
               "final_validation_loss": history[-1]["validation_loss"],
               "final_validation_perplexity": math.exp(history[-1]["validation_loss"]),
               "samples": [generate(vocab, weights, seed + offset) for offset in range(3)]}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoint.json").write_text(json.dumps(checkpoint, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickets", type=Path, default=Path("shared/datasets/sample_tickets.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("experiments/phase_3_1"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--load-checkpoint", type=Path,
                        help="load a trained checkpoint and generate samples")
    args = parser.parse_args()
    if args.load_checkpoint:
        vocab, weights = load_checkpoint(args.load_checkpoint)
        print(json.dumps({"samples": [generate(vocab, weights, args.seed + offset)
                                      for offset in range(3)]}, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(train(args.tickets, args.output, args.epochs, args.learning_rate, args.seed),
                         indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
