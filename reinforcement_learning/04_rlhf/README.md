# Phase 4.4 — Learned-reward RLHF

This lesson connects phase 3.4's trained reward model to the phase 4.3 clipped PPO response selector. It reads saved reward-model scores for the approved and rejected responses, verifies ticket IDs and candidate order, and optimizes the selection policy on training tickets only. No LLM tokens are generated or updated in this phase.

```bash
python3 -m shared.training.rlhf_bandit
python3 -m unittest discover -s tests -q
```

With seed 7, 50 rollouts sampled 200 actions; 53 optimizer steps were clipped. Mean sampled learned reward rose from 3.916 over the first 40 actions to 4.477 over the last 40. On the one held-out test pair, the final policy selected the approved response and assigned it probability 0.9973. The run summary is in ignored `experiments/phase_4_4/`.

**Reward-model error is visible:** Phase 3.4 ranked the one validation pair backward (0/1). As the policy favored the approved response, its expected learned reward on that pair fell from 2.25 to 2.064. Optimizing this learned score is therefore not a reliable quality signal. The selector also sees phrase-derived features, so this run cannot isolate learned reward from those features or demonstrate generative LLM RLHF.
