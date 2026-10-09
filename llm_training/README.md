# Lecture 03: LLM training

| Phase | Topic | Hands-on deliverable |
| --- | --- | --- |
| 3.1 | Pretraining | Train a tiny language model on a small dataset; report validation loss and generated samples. |
| 3.2 | SFT | Fine-tune a permitted pretrained model on support conversations; save a reproducible checkpoint. |
| 3.3 | LoRA / PEFT | Compare full SFT and LoRA SFT with matched data, model, and evaluation. |
| 3.4 | RLHF foundations | Build preference pairs and train a reward model; measure held-out pair ranking. |
| 3.5 | DPO | Train from the SFT checkpoint on chosen/rejected responses; compare against SFT. |
| 3.6 | Reasoning | Build evidence-linked final-answer examples and check a held-out SFT response. |
| 3.7 | Distillation | Transfer a larger teacher's token probabilities to a smaller student; compare loss and local CPU time. |

All lecture 03 phases are complete as learning exercises. The [phase tracker](../docs/project_register.md#phase-progress) links each result and completion commit. The shared preparation command exports SFT and preference files for phases 3.2–3.5. Sample data is synthetic; these runs do not establish enterprise support quality.
