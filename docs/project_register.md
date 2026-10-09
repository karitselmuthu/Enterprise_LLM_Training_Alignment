# Project decision and recommendation register

This is the single place to record project decisions, recommendations for a future enterprise pilot, and unresolved questions. Update it when a material choice is made in a project discussion. A recommendation becomes a decision only when the user explicitly accepts it or the project proceeds on that basis with the choice recorded. Keep superseded entries for history.

## Status key

- **Decided:** Explicitly chosen by the user or implemented as the current project direction.
- **Recommended:** Proposed; awaiting a decision or validation.
- **Open:** A question that can affect implementation.
- **Superseded:** Replaced by a later entry, with a link to it.

## Register

| ID | Date (IST) | Kind | Status | Summary | Revisit when |
| --- | --- | --- | --- | --- | --- |
| D-001 | 2026-10-09 | Decision | Decided | Build this as a learning project first. | Enterprise pilot work begins. |
| D-002 | 2026-10-09 | Decision | Decided | Use a shared, organization-isolated dataset split and fixed evaluation cases for comparisons. | The source data or leakage risks change. |
| D-003 | 2026-10-09 | Implementation decision | Decided | Teach phase 3.1 with a standard-library character bigram model and explicitly show its limits. | A Transformer pretraining exercise is added. |
| D-004 | 2026-10-09 | Implementation decision | Decided | Use SmolLM2-135M and assistant-response-only loss for phase 3.2 SFT; compare baseline and trained checkpoints on the same held-out ticket. | A larger dataset or model is selected. |
| D-005 | 2026-10-09 | Workflow decision | Decided | Push a separate completion commit for each finished phase using `Complete phase X.Y: <topic>`. | The user changes the repository workflow. |
| D-006 | 2026-10-09 | Implementation decision | Decided | Compare LoRA against phase 3.2 full SFT with matched model, data, seed, optimizer rate, epochs, prompt format, and evaluation. | A larger dataset or tuned-method comparison is needed. |
| D-007 | 2026-10-09 | Implementation decision | Decided | Teach pairwise reward modeling with a frozen phase 3.2 backbone and a trained scalar score head. | A larger preference dataset supports full reward-model tuning. |
| D-008 | 2026-10-09 | Implementation decision | Decided | Teach DPO from the phase 3.2 SFT checkpoint with a frozen reference and LoRA policy, then report held-out implicit reward margins. | More reviewed preference pairs support a meaningful comparison. |
| R-001 | 2026-10-09 | Model recommendation | Recommended | Use a tiny model from scratch for pretraining; SmolLM2-135M for controlled training comparisons; consider Qwen3-0.6B with LoRA for an assistant pilot. | A model benchmark, memory test, or license review changes the choice. |
| R-002 | 2026-10-09 | Hardware recommendation | Recommended | Run early exercises locally on the 8 GB Apple-silicon Mac; use a suitable GPU environment for later online RL experiments. | Profiling shows local runs are too slow or memory-limited. |
| R-003 | 2026-10-09 | Data recommendation | Recommended | For a meaningful pilot, seek roughly 1,000 reviewed responses, 300 independent preference pairs, and 100 held-out tickets across organizations. These are planning targets, not validated minimums. | Real data availability and learning curves are measured. |
| R-004 | 2026-10-09 | Enterprise recommendation | Recommended | Before real-ticket training, obtain approved, de-identified data and current policy/knowledge documents; establish rights, retention, and human review. | Real enterprise data is introduced. |
| R-005 | 2026-10-09 | Enterprise recommendation | Recommended | Evaluate factual grounding, policy adherence, escalation, and human-rated answer quality on a fixed test set. The current phrase checker is a narrow baseline. | A model is considered for ticket-system use. |
| R-006 | 2026-10-09 | Enterprise recommendation | Recommended | Integrate through current approved knowledge retrieval, access control, logging, and human handoff after model review. | Ticket-system API and deployment requirements are known. |
| O-001 | 2026-10-09 | Question | Open | Which enterprise policies and ticket-system API would govern a later pilot? | Pilot planning begins. |

## Phase progress

**Completion rule:** Mark a phase **Complete** only when its hands-on artifact runs, its stated checks pass, and its results and limitations are documented. Use **In progress** while implementing or checking it. Later phases remain **Not started** until work begins.

| Phase | Topic | Status | Completion evidence or next gate |
| --- | --- | --- | --- |
| 3.1 | Pretraining | **Complete** (2026-10-09) | [Results](../llm_training/01_pretraining/README.md) · [Completion commit `3ad9f15`](https://github.com/karitselmuthu/Enterprise_LLM_Training_Alignment/commit/3ad9f15bc7fe76b68640da38065d35c8bad879c4); checkpoint reload and all 3 tests pass. |
| 3.2 | Supervised fine-tuning | **Complete** (2026-10-09) | [Results](../llm_training/02_supervised_fine_tuning/README.md) · [Completion commit `d06eafa`](https://github.com/karitselmuthu/Enterprise_LLM_Training_Alignment/commit/d06eafa2d2f534b879d117d5e0aa3c5f14838313); checkpoint reload matched validation loss; all 4 tests pass. |
| 3.3 | LoRA / PEFT | **Complete** (2026-10-09) | [Results](../llm_training/03_lora_peft/README.md) · [Completion commit `272d049`](https://github.com/karitselmuthu/Enterprise_LLM_Training_Alignment/commit/272d049b37979c5de8495eadaebbbde3cf4755e2); adapter reload matched validation loss; all 5 tests pass. |
| 3.4 | RLHF foundations | **Complete** (2026-10-09) | [Results](../llm_training/04_rlhf_foundations/README.md) · [Completion commit `4778802`](https://github.com/karitselmuthu/Enterprise_LLM_Training_Alignment/commit/47788028d99bb46471f945ed97a95ae7ac8d2c1d); checkpoint reload matched test margin; all 7 tests pass. |
| 3.5 | DPO | **Complete** (2026-10-09) | [Results](../llm_training/05_dpo/README.md) · [Completion commit `d8e1fc6`](https://github.com/karitselmuthu/Enterprise_LLM_Training_Alignment/commit/d8e1fc6119dda66bb02ea8cb72090a19bd4f0712); adapter reload matched test margin; all 9 tests pass. |
| 3.6 | Reasoning | Not started | Evaluate a reasoning-focused training exercise. |
| 3.7 | Distillation | Not started | Compare teacher and student quality, latency, and cost. |
| 4.1 | Reward design | Not started | Define and test task rewards. |
| 4.2 | Policy gradients | Not started | Run a small policy-gradient exercise. |
| 4.3 | PPO | Not started | Train and evaluate a PPO policy. |
| 4.4 | RLHF | Not started | Connect reward modeling and policy optimization. |
| 4.5 | GRPO | Not started | Train and evaluate a GRPO policy. |
| 4.6 | RLVR | Not started | Train on a task with verifiable rewards. |

## Decision notes

### D-001 — Learning project first

**Source:** User decision in the project discussion on 2026-10-09. The immediate goal is hands-on understanding of pretraining, SFT, LoRA, preference learning, DPO, reasoning, distillation, and RL concepts. Enterprise deployment is a later possibility.

**Consequence:** Synthetic data is adequate for wiring and teaching exercises. No current result should be presented as proof of enterprise readiness.

### D-002 — Shared splits and evaluation

**Source:** Implemented in `shared/datasets/prepare.py` and `shared/evaluation/evaluate.py` on 2026-10-09. The split groups by `org_id`; the sample contains only six synthetic tickets.

**Consequence:** Future model comparisons can reuse the same held-out ticket IDs. The current phrase metrics do not establish factual accuracy or full policy compliance.

### D-003 — Introductory pretraining model

**Source:** Implementation choice made on 2026-10-09 to start the user-requested step-by-step learning path using the available standard-library Python environment.

**Rationale and evidence:** The local environment has no NumPy, PyTorch, or MLX installed. The character bigram trainer completed a deterministic run: training loss 3.8918 to 2.0013 and validation loss 3.8918 to 3.0707. The generated text is mostly incoherent. The checkpoint reload check and all three repository tests passed.

**Consequence:** Phase 3.1 demonstrates next-token training, validation, checkpointing, and generation. It does not teach attention or Transformer architecture; that remains a possible extension.

### D-004 — Supervised fine-tuning baseline

**Source:** Implementation choice made on 2026-10-09 for the user's next learning phase. SmolLM2-135M is already cached locally, and the prepared SFT split has four training conversations, one validation conversation, and one held-out test conversation.

**Rationale:** A small permitted pretrained model allows full-parameter training on constrained hardware. Masking the system and user tokens focuses the loss on the approved assistant response. The same test ticket is used for before/after generation and explicit phrase checks.

**Run evidence:** Full-parameter CPU run with PyTorch 2.14.0 and Transformers 5.17.0 completed for two epochs. Validation answer-token loss fell from 3.1469 to 2.9230 and test answer-token loss fell from 2.7208 to 2.2279. The saved checkpoint reloaded with identical validation loss. On the single held-out ticket, the trained response became repetitive and failed a required phrase check that the baseline passed. [Detailed results](../llm_training/02_supervised_fine_tuning/README.md).

**Consequence:** Phase 3.2 demonstrates SFT mechanics and checkpoint comparison. The result does not show support-answer improvement or generalization. Revisit the model and dataset when larger, reviewed examples become available.

### D-005 — Phase completion commits

**Source:** User request on 2026-10-09 to push the project to `karitselmuthu/Enterprise_LLM_Training_Alignment` and repeat the same commit-message pattern for every completed phase.

**Rationale:** Separate, consistently named commits show which phase was completed and make progress reviewable in Git history.

**Consequence:** The phase 3.1 and 3.2 completion commits use `Complete phase X.Y: <topic>`. Future phase completion work follows the same pattern and is pushed to `origin/main` under the user's authorization.

### D-006 — Controlled LoRA comparison

**Source:** User request on 2026-10-09 to proceed with phase 3.3. Phase 3.2 supplied the full SFT reference and the same prepared ticket split.

**Rationale and evidence:** The LoRA script verified the phase 3.2 manifest and settings and reproduced the pretrained baseline validation loss (3.1469). It trained 460,800 parameters (0.34% of the adapter-wrapped model) and reloaded the adapter with the same validation loss (3.1294). Full SFT had lower validation and test losses; LoRA passed the one-ticket required phrase check because its generated response mostly echoed the prompt. [Detailed comparison](../llm_training/03_lora_peft/README.md).

**Consequence:** Phase 3.3 demonstrates parameter and checkpoint savings plus the limits of a tiny controlled comparison. Neither model produced a good support response; no enterprise-quality or method-superiority conclusion follows.

### D-007 — Pairwise reward-model exercise

**Source:** User request on 2026-10-09 to proceed with phase 3.4. The prepared preference splits and phase 3.2 checkpoint are reused.

**Rationale and evidence:** A frozen SFT backbone plus a 576-parameter scalar head makes the Bradley–Terry preference objective runnable on the available CPU. Training ranked 4/4 pairs correctly, but the one validation pair remained incorrect (0/1) and its margin worsened from -0.328 to -0.375. The one test pair ranked correctly (1/1); checkpoint reload preserved its 1.6875 margin. [Detailed results](../llm_training/04_rlhf_foundations/README.md).

**Consequence:** Phase 3.4 demonstrates preference-data validation, pairwise loss, ranking metrics, and checkpoint verification. This model is not validated for policy optimization or enterprise use; more reviewed pairs and broader held-out evaluation are needed.

### D-008 — DPO exercise

**Source:** User request on 2026-10-09 to complete the remaining phases in order. The phase 3.2 SFT checkpoint and the same organization-isolated preference splits are reused.

**Rationale and evidence:** A frozen SFT reference and a LoRA policy expose the DPO log-probability objective without a separate reward model. The adapter trained on four pairs (4/4 correctly ranked), but validation and test each remained 0/1. The reloaded adapter matched the test implicit reward margin (-0.0149), and the generated test response still failed the required-term check. [Detailed results](../llm_training/05_dpo/README.md).

**Consequence:** DPO mechanics are demonstrated; no support-quality gain is established. Revisit with a larger reviewed preference set.

### R-001 and R-002 — Model and compute path

**Rationale:** A tiny scratch model teaches pretraining mechanics. A small pretrained base model enables controlled full-versus-LoRA comparisons on constrained hardware. A somewhat larger model can test support response quality with LoRA. Online RL adds generation and training memory costs.

**Evidence checked 2026-10-09:** [SmolLM2-135M model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B), [MLX LM documentation](https://github.com/ml-explore/mlx-lm), [TRL trainer documentation](https://huggingface.co/docs/trl/index). Actual fit and runtime have not been measured in this project.

### R-003 to R-006 — Future enterprise path

**Rationale:** A deployment decision needs representative approved data, current organizational rules, meaningful evaluation, and operational controls. These recommendations are conditional; they do not turn the learning project into a deployment project.

## Add an entry

1. Assign the next `D-`, `R-`, or `O-` ID and add a row to the register.
2. Record the date, who made or accepted the choice, the reason and evidence, its consequence, and the trigger for revisiting it.
3. Keep recommendations marked **Recommended** until a decision is made. Mark replaced entries **Superseded** and link to the replacement.
4. Update **Phase progress** after each phase milestone, with a link to its artifact and verified results.
