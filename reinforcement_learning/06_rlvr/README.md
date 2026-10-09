# Phase 4.6 — Reinforcement learning with verifiable rewards

This exercise separates reward verification from the support phrase checker. A small softmax policy sees two integers from 0 through 4 and samples one answer from 0 through 8. The verifier returns 1 only for the exact sum and 0 otherwise. Twenty input pairs train the policy; five distinct pairs are held out. The policy has learned weights for each answer and operand value and is updated with REINFORCE and an expected-reward baseline.

```bash
python3 -m shared.training.rlvr_arithmetic
python3 -m unittest discover -s tests -q
```

Seed 7 and 1,000 passes produced 20,000 sampled updates. Mean exact-match reward rose from 0.10 in the first 100 samples to 0.66 in the last 100. Greedy training accuracy ended at 14/20, while held-out accuracy was 0/5 (initial tie-based accuracy was 1/5). The held-out answer probability fell from 1/9 to 0.00017. Predictions and metrics are in ignored `experiments/phase_4_6/summary.json`.

**Limit:** This policy did not learn addition as a transferable rule. It shows that a perfectly verifiable reward can still produce poor generalization with an inadequate policy and tiny data. It is a one-step arithmetic policy, not an LLM RLVR run or evidence of support-assistant improvement.
