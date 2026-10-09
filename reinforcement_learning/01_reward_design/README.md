# Phase 4.1 — Reward design

This lesson turns each synthetic ticket into a two-response selection case, preserving the prepared organization-isolated split. The approved and rejected responses alternate positions. A transparent rule reward gives fractional credit for required terms and subtracts one point per forbidden term.

```bash
python3 -m shared.training.support_bandit
python3 -m unittest discover -s tests -q
```

All six approved responses scored higher than their rejected partners. Mean approved score was 1.0 and mean rejected score was -2.0. The run writes cases and metrics to ignored `experiments/phase_4_1/`.

**Reward failure:** The meaningless response `administrator 30 days` earns the full 1.0 on the sample retention rule. This reward only detects phrases. It does not establish factual grounding, usefulness, authorization, or full policy compliance. Later RL phases optimize response selection in this deliberately narrow environment and must not be read as training a support LLM.
