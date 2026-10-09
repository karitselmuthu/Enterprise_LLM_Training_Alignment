# Phase 4.5 — Group-relative policy optimization

For each training ticket, this exercise treats its two fixed response candidates as a group. It centers and scales their rule rewards within the group, then performs clipped probability-ratio updates with a small KL penalty toward a uniform reference policy. No learned value model is used.

```bash
python3 -m shared.training.grpo_bandit
python3 -m unittest discover -s tests -q
```

Fifty iterations with four update passes produced 774 clipped steps. The mean approved-response probability on the one held-out test ticket rose from 0.5 to about 0.9999996, and its greedy selection was approved. The summary is in ignored `experiments/phase_4_5/`.

**Limit:** Real GRPO samples multiple generated outputs from the current LLM policy. This lesson reuses two static responses and phrase-based rewards to isolate group-relative advantages, clipping, and reference KL. It provides no evidence of generative support quality, and the near-perfect probability reflects an easy rule signal rather than a robust answer policy.
