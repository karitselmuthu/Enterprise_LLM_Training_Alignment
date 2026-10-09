# Phase 4.2 — Policy gradients

A two-action softmax policy selects one of two fixed responses per synthetic ticket. Candidate features are required-term coverage and negative forbidden-term count from phase 4.1. REINFORCE samples an action and applies its reward times the gradient of log action probability. This directly exposes the on-policy update; it does not generate tokens.

```bash
python3 -m shared.training.policy_gradient
python3 -m unittest discover -s tests -q
```

With seed 7, 100 passes over four training tickets produced 400 updates. Mean sampled reward rose from 0.7 in the first 40 updates to 1.0 in the last 40. Mean probability of the approved response on the one held-out test ticket rose from 0.5 to 0.9994; greedy selection was approved on that ticket. The final weights and metrics are in ignored `experiments/phase_4_2/summary.json`.

**Limit:** The features expose the same required/forbidden phrase signals used by the reward. The held-out result shows the optimizer learned this tiny rule-based bandit, not that it generalized support reasoning or overcame reward gaming. A zero-weight tie chooses the first candidate, so initial greedy accuracy reflects candidate order.
