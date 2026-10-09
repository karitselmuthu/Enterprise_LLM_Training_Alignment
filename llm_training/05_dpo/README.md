# Phase 3.5 — Direct preference optimization

This lesson starts from the phase 3.2 SFT checkpoint, keeps a frozen copy as the reference policy, and trains a LoRA adapter to raise chosen-response likelihood relative to rejected-response likelihood. The response-only log probability excludes the shared system and user prompt. The code checks that preference ticket IDs match the SFT split and that the data manifest is unchanged.

```bash
HF_HUB_OFFLINE=1 python -m shared.training.dpo \
  --data-dir experiments/sample \
  --sft-checkpoint experiments/phase_3_2/checkpoint \
  --sft-summary experiments/phase_3_2/summary.json \
  --output experiments/phase_3_5 --epochs 2
python3 -m unittest discover -s tests -v
```

The run used CPU, PyTorch 2.14.0, Transformers 5.17.0, PEFT 0.21.2, seed 7, learning rate 0.0001, and DPO beta 0.1. It trained 460,800 LoRA parameters on four synthetic preference pairs. The initial implicit reward margin is zero by construction because policy and reference are identical. After two epochs:

| Split | Pairs ranked chosen above rejected | Mean implicit reward margin |
| --- | ---: | ---: |
| Train | 4/4 | 0.0728 |
| Validation | 0/1 | -0.0229 |
| Test | 0/1 | -0.0149 |

The reloaded adapter reproduced the test margin within 0.0001. The one generated test answer passed the forbidden-term check and failed the required-term check, matching the SFT checkpoint's phrase-check rates. The adapter and full run summary are written to ignored `experiments/phase_3_5/`.

**Limit:** Four training pairs and one case per held-out split cannot establish generalization. The run demonstrates the DPO objective and artifact reload, not an improved support assistant. The response phrase checks are narrow and do not assess factual grounding or complete policy compliance.
