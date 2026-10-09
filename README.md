# Enterprise AI Support Assistant Training Platform

A small, reproducible workspace for training and assessing an enterprise support assistant. The example data is synthetic. No ticket-system connection or pretrained model is bundled.

## Pipeline

```text
Synthetic or approved support tickets
  -> validated, organization-isolated splits
  -> SFT examples (full fine-tuning OR LoRA/PEFT)
  -> preference pairs
       -> DPO
       -> reward model -> PPO
       -> GRPO with a defined reward function (when applicable)
  -> fixed evaluation set
  -> reviewed model artifact -> ticket-system integration
```

LoRA is a parameter-efficient way to perform SFT, not an extra model stage. DPO and reward-model-based RL are alternative alignment experiments sharing the same SFT checkpoint. GRPO can use a programmatic reward instead of a learned reward model. RLVR is suitable only for tasks with verifiable outcomes; it is not a general substitute for support-response review.

## Run the foundation

Requires Python 3.10 or newer; the preparation and evaluation commands use only the standard library.

```bash
python3 -m shared.datasets.prepare --input shared/datasets/sample_tickets.jsonl --output experiments/sample
python3 -m shared.evaluation.evaluate --gold experiments/sample/evaluation.jsonl --predictions shared/evaluation/sample_predictions.jsonl --output reports/sample_metrics.json
python3 -m unittest discover -s tests -v
```

The preparation command writes `sft_{train,validation,test}.jsonl`, `preferences_{train,validation,test}.jsonl`, `evaluation.jsonl`, and `manifest.json`. Splits are deterministic and isolated by `org_id` to reduce customer leakage. The evaluation command measures explicit required and forbidden phrase checks. These are narrow automated checks, not a measure of factual correctness or complete policy compliance.

## Repository map

- `llm_training/`: pretraining, SFT, PEFT, preference learning, reasoning, and distillation lessons.
- `reinforcement_learning/`: reward design and RL alignment lessons.
- `shared/`: dataset contracts, reusable model/training interfaces, and evaluation.
- `configs/`: experiment settings.
- `experiments/`: generated datasets and future training runs.
- `reports/`: generated evaluation outputs.
- `docs/`: architecture and milestone plan.

See [Project decision and recommendation register](docs/project_register.md) for the current choices, future enterprise recommendations, and open questions. See [docs/architecture.md](docs/architecture.md) for the boundaries between stages and the next implementation steps.

Start with [phase 3.1 pretraining](llm_training/01_pretraining/README.md). The [phase progress tracker](docs/project_register.md#phase-progress) records completion evidence for each lesson.

Continue with [phase 3.2 supervised fine-tuning](llm_training/02_supervised_fine_tuning/README.md). It uses the optional dependencies in `requirements-sft.txt`.
