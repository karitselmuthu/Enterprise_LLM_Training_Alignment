# Lecture 03: LLM training

| Phase | Topic | Hands-on deliverable |
| --- | --- | --- |
| 3.1 | Pretraining | Train a tiny language model on a small dataset; report validation loss and generated samples. |
| 3.2 | SFT | Fine-tune a permitted pretrained model on support conversations; save a reproducible checkpoint. |
| 3.3 | LoRA / PEFT | Compare full SFT and LoRA SFT with matched data, model, and evaluation. |
| 3.4 | RLHF foundations | Build preference pairs and train a reward model; measure held-out pair ranking. |
| 3.5 | DPO | Train from the SFT checkpoint on chosen/rejected responses; compare against SFT. |
| 3.6 | Reasoning | Study a curated reasoning dataset and evaluate visible answer quality without exposing hidden reasoning in support replies. |
| 3.7 | Distillation | Transfer reviewed teacher responses to a smaller student; compare quality, latency, and cost. |

The shared preparation command already exports SFT and preference files for phases 3.2–3.5. Each lesson should state its model, compute budget, dataset license, baseline, seed, and evaluation criteria before training code is added.
