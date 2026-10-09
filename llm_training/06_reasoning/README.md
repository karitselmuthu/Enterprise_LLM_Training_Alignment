# Phase 3.6 — Reasoning-focused support data

This exercise builds an evidence-grounded answer format from the six synthetic tickets. Each example contains a question, numbered evidence statements, and a target with evidence IDs and an approved final answer. It preserves the prepared organization-isolated train/validation/test split. This is a data and evaluation exercise; it does not train a reasoning model or collect private chain-of-thought traces.

```bash
python3 -m shared.training.reasoning
python3 -m unittest discover -s tests -q
```

The run writes ignored files to `experiments/phase_3_6/`. All six gold targets had valid citation IDs, contained their required terms, and avoided forbidden terms. The phase 3.2 SFT response on the one held-out test ticket had no evidence citation and missed its required terms; it avoided forbidden terms. The evaluator also rejects a deliberately invalid evidence ID and forbidden language in tests.

**Training method to explore next:** An evidence-conditioned SFT run could train the model to emit `evidence_ids` and `answer`. Preference learning could rank valid citations and approved answers above unsupported ones. Both need more reviewed examples and independent evidence verification. Citation ID validity only checks that an ID exists; it does not prove the cited statement supports the answer. Gold checks are a dataset integrity check, not a model quality result.
