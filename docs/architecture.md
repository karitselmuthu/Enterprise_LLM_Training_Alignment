# Architecture and implementation plan

## Data contract

Each input JSONL row represents one approved synthetic or de-identified ticket. Required fields are `ticket_id`, `org_id`, `product`, `issue`, `evidence` (nonempty list of strings), `approved_response`, `rejected_response`, `required_terms`, and `forbidden_terms`. All identifiers must be unique where appropriate. The preparation command checks types, nonempty content, and duplicate ticket IDs.

Keep customer secrets, personal data, internal credentials, and unapproved ticket text out of training data. Review data rights and retention before using real tickets. `org_id` is used for splitting and excluded from model prompts. Keep the held-out test set fixed across experiments.

## Training sequence

1. **Base model and tokenizer:** Record model revision, license, tokenizer, context length, and inference constraints. Pretraining is a teaching module; the practical platform should start from a permitted small base model.
2. **SFT:** Train on the prepared `messages` examples. Choose full fine-tuning or LoRA/PEFT as the optimization method. Record seed, hyperparameters, data manifest, and checkpoint hash.
3. **Preference alignment:** Start each branch from the same SFT checkpoint. DPO consumes `prompt`/`chosen`/`rejected` pairs. A separate reward-model experiment can score those pairs before PPO. GRPO requires an explicit, audited reward function and multiple generated completions per prompt.
4. **Evaluation:** Compare checkpoints on the same test ticket IDs. Measure explicit policy checks, evidence attribution, answer quality, refusal behavior, latency, and cost. Human reviewers must assess factual correctness and policy compliance; the included phrase checks cover only known assertions.
5. **Integration:** Expose a reviewed model through a ticket-system adapter with access control, logging, retrieval of current approved knowledge, and a human handoff path. No adapter is implemented yet because the ticket-system API is unspecified.

## Current milestone

The repository currently provides the data contract, deterministic organization-isolated splits, SFT and preference export, synthetic sample data, and a small automated evaluation baseline. Training jobs and ticket-system integration are subsequent milestones.
