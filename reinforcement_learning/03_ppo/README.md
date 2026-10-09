# Phase 4.3 — PPO

This exercise applies a clipped PPO surrogate to the two-response support bandit. Each rollout samples with a frozen copy of the current policy. Four optimization passes reuse the sampled actions and their old probabilities. Advantages subtract the old policy's expected rule reward. The code tracks clipped updates and old-to-new policy KL.

```bash
python3 -m shared.training.ppo_bandit
python3 -m unittest discover -s tests -q
```

Seed 7 produced 50 rollouts, 200 sampled actions, and 800 optimizer steps. The surrogate clipped 50 steps; mean rollout KL was 0.00315. Mean sampled reward rose from 0.4 in the first 40 actions to 1.0 in the last 40. Mean approved-response probability on the one held-out test case rose from 0.5 to 0.9960. The run summary is in ignored `experiments/phase_4_3/`.

**Limit:** This is PPO on a two-action, one-step bandit, not token-level PPO on an LLM. Its features are derived from the same phrase checks as the reward. The single held-out case and easy reward cannot establish robust support behavior.
