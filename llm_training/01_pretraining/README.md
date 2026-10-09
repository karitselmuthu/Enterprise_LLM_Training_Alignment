# Phase 3.1 — Pretraining a tiny language model

**Status:** Complete for the introductory learning deliverable (2026-10-09). See the [phase tracker](../../docs/project_register.md#phase-progress).

## Goal

Learn the next-token objective by training a character bigram model from scratch. The model has one learnable logit for each possible `(current character, next character)` pair and minimizes cross-entropy. It is a language model, but it has no attention, longer context, or Transformer blocks.

## Run

From the repository root:

```bash
python3 -m shared.training.char_bigram --tickets shared/datasets/sample_tickets.jsonl --output experiments/phase_3_1 --epochs 100 --learning-rate 1 --seed 7
python3 -m shared.training.char_bigram --load-checkpoint experiments/phase_3_1/checkpoint.json --seed 7
python3 -m unittest discover -s tests -v
```

The trainer uses only the standard library. It takes the approved knowledge and response fields from the four training organizations, scores one validation organization, and leaves the test organization unused. It writes `checkpoint.json`, `history.json`, and `summary.json` under `experiments/phase_3_1/`.

## Observed run

| Metric | Start | After 100 epochs |
| --- | ---: | ---: |
| Training loss, nats/character | 3.8918 | 2.0013 |
| Validation loss, nats/character | 3.8918 | 3.0707 |
| Validation perplexity | 49.0 | 21.56 |

The training corpus has 4 documents and 1,109 character transitions. Validation has 1 document and 305 transitions. The vocabulary has 49 symbols, giving 2,401 learned parameters. The exact run is reproducible with the command above; its outputs are generated files and are ignored by Git.

One generated sample begins `Afr in g d Cat astinvata?`. This is expected: a bigram model knows only the previous character and the corpus is tiny. Lower loss here demonstrates fitting the training distribution, not useful support-answer quality.

## Try changing one thing

Run with `--epochs 10`, then `--epochs 300`, and compare training and validation loss in `history.json`. Keep the same split and seed. Explain why the training loss and generated answer quality are different measurements.

**Next:** Phase 3.2 fine-tunes a pretrained model on support conversations; it requires an external model and training library.
