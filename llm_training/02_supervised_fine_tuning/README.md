# Phase 3.2 — Supervised fine-tuning

**Status:** Complete for the learning deliverable (2026-10-09). See the [phase tracker](../../docs/project_register.md#phase-progress).

## Goal

Fine-tune a pretrained causal language model on approved support responses. The loss applies only to the assistant response tokens; the system instruction and user ticket remain in the prompt as context. This run updates all model parameters. Phase 3.3 will compare this full-parameter setup with LoRA on matching data.

## Run

From the repository root, create a Python environment with the optional phase dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-sft.txt
python3 -m shared.datasets.prepare --input shared/datasets/sample_tickets.jsonl --output experiments/sample
.venv/bin/python -m shared.training.sft --data-dir experiments/sample --output experiments/phase_3_2 --epochs 2 --max-new-tokens 48
```

The model revision is pinned in `shared/training/sft.py`. If the model is already cached, add `--local-files-only`. The command writes a full checkpoint, before/after test predictions, and `summary.json` under `experiments/phase_3_2/`. The generated files are ignored by Git. Package downloads were unavailable in the execution sandbox, so the verified run used an existing local Python environment with the pinned package versions.

## Verified run

Model: [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. Device: CPU on the 8 GB Apple-silicon Mac. Library versions: PyTorch 2.14.0 and Transformers 5.17.0. Seed: 7. Training examples: 4. Validation examples: 1. Held-out test examples: 1. Two epochs, learning rate `5e-5`, maximum sequence length 384, maximum generated length 48 tokens.

| Measure | Pretrained baseline | After SFT |
| --- | ---: | ---: |
| Validation answer-token loss | 3.1469 | 2.9230 |
| Held-out answer-token loss | 2.7208 | 2.2279 |
| Required phrase check on 1 test ticket | Pass | Fail |
| Forbidden phrase check on 1 test ticket | Pass | Pass |

The saved checkpoint reloaded with the same validation loss (`2.9230`). All four repository tests pass, including the assistant-only loss mask check.

The baseline response mostly repeated the supplied ticket context. After SFT, the response repeated `Can view API secrets?` and omitted a required term. This is a useful failure case: lower teacher-forced loss did not produce a better generated support answer. Four training conversations and one test case are too small for a quality conclusion.

## Inspect and experiment

1. Compare `experiments/phase_3_2/baseline_predictions.jsonl` with `predictions.jsonl`.
2. Compare answer-token loss with the explicit phrase checks in `summary.json`.
3. Change one variable, such as the epoch count or learning rate, and keep the same data split and seed. Record whether generated answers improve or deteriorate. Do not tune repeatedly on the one held-out test ticket.

**Next:** Phase 3.3 compares full SFT with LoRA using the same model revision, split, prompt format, and evaluation.
