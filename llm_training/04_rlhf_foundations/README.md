# Phase 3.4 — RLHF foundations: reward modeling

**Status:** Complete for the learning deliverable (2026-10-09). See the [phase tracker](../../docs/project_register.md#phase-progress).

## Goal

Train a reward model to score a response given its support-ticket prompt. Each preference pair contains a `chosen` response and a `rejected` response. The model learns to make `score(chosen) > score(rejected)` using the [pairwise reward-model loss](https://huggingface.co/docs/trl/reward_trainer): `softplus(-(score(chosen) - score(rejected)))`.

This exercise starts from the phase 3.2 SFT checkpoint, freezes its Transformer backbone, and trains a new scalar scoring head with 576 parameters. This keeps the local exercise small and makes the preference objective easy to inspect. It is a limited reward model; full reward-model fine-tuning could behave differently.

## Run

From the repository root, prepare data and complete phase 3.2 first. Its checkpoint and summary are generated files and are not stored in Git.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-sft.txt
python3 -m shared.datasets.prepare --input shared/datasets/sample_tickets.jsonl --output experiments/sample
.venv/bin/python -m shared.training.sft --data-dir experiments/sample --output experiments/phase_3_2 --epochs 2 --max-new-tokens 48
.venv/bin/python -m shared.training.reward_model --data-dir experiments/sample --sft-checkpoint experiments/phase_3_2/checkpoint --sft-summary experiments/phase_3_2/summary.json --output experiments/phase_3_4 --epochs 8 --learning-rate 1e-3
```

The script verifies that the preference files and SFT checkpoint use the same data manifest and ticket splits. It writes a reward-model checkpoint and `summary.json` under `experiments/phase_3_4/`, then reloads the checkpoint and checks its test ranking margin. The model revision is inherited from the phase 3.2 checkpoint.

## Verified run

The run used SmolLM2-135M after phase 3.2 SFT, CPU, PyTorch 2.14.0, Transformers 5.17.0, seed 7, eight epochs, and learning rate `1e-3`. The data contains four training pairs, one validation pair, and one held-out test pair.

| Split | Initial ranking accuracy | Final ranking accuracy | Initial mean margin | Final mean margin |
| --- | ---: | ---: | ---: | ---: |
| Train (4 pairs) | 3/4 | 4/4 | 0.076 | 2.781 |
| Validation (1 pair) | 0/1 | 0/1 | -0.328 | -0.375 |
| Test (1 pair) | 1/1 | 1/1 | 0.016 | 1.688 |

The final mean pairwise loss was `0.063` on training pairs, `0.898` on the validation pair, and `0.170` on the test pair. Reloading the checkpoint reproduced the test margin `1.6875`. All seven repository tests pass.

The training ranking improved, while the validation pair remained incorrectly ranked and its margin worsened. The test set has only one pair, so `1/1` does not establish reliability. This reward model should not be used to judge enterprise support answers or to optimize a policy; it is an RLHF learning exercise.

## Inspect and experiment

1. Open `experiments/phase_3_4/summary.json` and compare each pair's chosen and rejected scores.
2. Plot or inspect `history` to see training loss fall while validation ranking stays wrong.
3. Try fewer epochs with the same split and seed. Record the validation margin; do not select hyperparameters using the held-out test pair.

**Next:** Phase 3.5 teaches DPO using the same preference data. Later PPO experiments can reuse the reward-model interface after a larger, reviewed preference dataset is available.
