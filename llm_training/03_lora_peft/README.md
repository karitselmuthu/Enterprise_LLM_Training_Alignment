# Phase 3.3 — LoRA / PEFT comparison

**Status:** Complete for the learning deliverable (2026-10-09). See the [phase tracker](../../docs/project_register.md#phase-progress).

## Goal

Compare full supervised fine-tuning with [LoRA](https://huggingface.co/docs/peft/quicktour) on the same pretrained checkpoint and support conversations. LoRA freezes the base model and trains small adapter matrices in the attention query and value projections. This exercise uses rank 8, alpha 16, and zero adapter dropout.

## Run

From the repository root, prepare data and complete phase 3.2 first because its generated summary and full checkpoint are the comparison reference:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-sft.txt
python3 -m shared.datasets.prepare --input shared/datasets/sample_tickets.jsonl --output experiments/sample
.venv/bin/python -m shared.training.sft --data-dir experiments/sample --output experiments/phase_3_2 --epochs 2 --max-new-tokens 48
.venv/bin/python -m shared.training.lora_compare --data-dir experiments/sample --full-reference experiments/phase_3_2/summary.json --full-checkpoint experiments/phase_3_2/checkpoint --output experiments/phase_3_3
```

Add `--local-files-only` to the model commands if the pinned model revision is cached. The LoRA script checks the data manifest, model revision, seed, epochs, learning rate, token limits, and example counts from the full SFT run. It fails if the LoRA baseline differs from the full SFT baseline. It saves an adapter and verifies that loading the adapter onto a fresh base model reproduces validation loss. Generated checkpoints and summaries live under `experiments/` and are ignored by Git.

## Verified comparison

Both runs used SmolLM2-135M revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, four training conversations, one validation conversation, one test conversation, seed 7, two epochs, learning rate `5e-5`, maximum sequence length 384, and 48 generated tokens. Runs used CPU, PyTorch 2.14.0, Transformers 5.17.0, and PEFT 0.21.2. The LoRA training loop took 66.5 seconds; a matching full-run time was not recorded.

| Measure | Full SFT | LoRA SFT |
| --- | ---: | ---: |
| Trainable parameters | 134,515,008 | 460,800 (0.34% of adapter-wrapped model) |
| Saved weight bytes | 269,060,552 | 1,858,776 |
| Validation answer-token loss | 2.9230 | 3.1294 |
| Test answer-token loss | 2.2279 | 2.6820 |
| Required phrase check on one test ticket | Fail | Pass |
| Forbidden phrase check on one test ticket | Pass | Pass |

The pretrained baseline validation loss was `3.1469` for both paths. Reloading the LoRA adapter reproduced validation loss `3.1294`. The full checkpoint directory was 272,585,091 bytes including tokenizer files; the LoRA adapter directory was 5,388,719 bytes including tokenizer files.

Full SFT reduced loss more under these shared settings. Its generated response repeated an incorrect question. The LoRA response mostly echoed the supplied ticket context, which happened to contain the required phrases. Neither response demonstrated useful support-answer quality. One test ticket and four training examples cannot support a general claim that either method is better. An optimized LoRA learning rate could differ from the matched rate used here.

## Inspect and experiment

1. Compare `experiments/phase_3_2/predictions.jsonl` with `experiments/phase_3_3/predictions.jsonl`.
2. Inspect `summary.json` in each run and compare loss, phrase checks, and checkpoint size.
3. For a follow-up experiment, change LoRA rank or learning rate while keeping the data split and model revision fixed. Record that this is a tuned comparison rather than the controlled one above.

**Next:** Phase 3.4 uses the prepared preference pairs to teach reward modeling.
