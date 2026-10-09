# Project memory for agents

Read `docs/project_register.md` before making project architecture, model, data, hardware, evaluation, or deployment choices.

Update that register in the same work session whenever a material decision, recommendation, or open question emerges. Distinguish explicit user decisions from assistant recommendations and working assumptions. Record a date, rationale, evidence, consequence, and revisit trigger. Preserve history by marking replaced entries superseded; do not silently rewrite prior decisions.

Keep `README.md` linked to the register so the user has one discoverable place for project decisions and future enterprise recommendations.

For learning phases, update the **Phase progress** table in `docs/project_register.md`. Mark a phase complete only after its artifact runs, checks pass, and results and limitations are documented.

When a phase is completed, commit its work using `Complete phase X.Y: <topic>` (for example, `Complete phase 3.2: supervised fine-tuning`) and push it to `origin/main` when the user has authorized repository updates. Keep one completion commit per phase and preserve earlier phase history.

After the completion commit is pushed, add its full GitHub commit URL to the corresponding row in the phase tracker and push that small documentation update.
