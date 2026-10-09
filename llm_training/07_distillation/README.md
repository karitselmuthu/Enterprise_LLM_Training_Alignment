# Phase 3.7 — Teacher to student distillation

This exercise uses a frozen SmolLM2-360M teacher and a SmolLM2-135M student. It checks that their token IDs match on every training and evaluation conversation. A LoRA adapter on the student receives both teacher soft-token targets (temperature 2) and approved-answer cross-entropy, weighted equally. Only assistant response tokens contribute to the KL loss. Four synthetic conversations provide four update steps.

```bash
HF_HUB_OFFLINE=1 python -m shared.training.distill
python3 -m unittest discover -s tests -q
```

The local run used CPU, PyTorch 2.14.0, Transformers 5.17.0, PEFT 0.21.2, seed 7, and learning rate 0.0001. Training took 43.1 seconds. The teacher has 361,821,120 parameters; the student backbone has 134,975,808; 460,800 LoRA parameters were updated.

| Model | Validation answer-token loss | Test answer-token loss | Validation forward time | Test forward time |
| --- | ---: | ---: | ---: | ---: |
| Teacher | 2.8865 | 2.2775 | 2.66 s | 1.59 s |
| Student before training | 3.1469 | 2.7208 | 1.00 s | 0.57 s |
| Student after training | 3.1169 | 2.6868 | 1.04 s | 0.85 s |

The reloaded adapter reproduced test loss 2.6868. Its one generated test response passed both required- and forbidden-term checks. These are single forward-pass timings on a shared local CPU, not a production throughput or cost benchmark. The student remains worse than the teacher on answer-token loss. The apparent phrase-check success is one synthetic case and does not establish answer quality. Model weights, tokenizer, prediction, and summary are written under ignored `experiments/phase_3_7/`.
