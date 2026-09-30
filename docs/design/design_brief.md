# Design brief, round 2

Input to the second design round. Delete it once the PI accepts the design.

The PI reviewed round 1 (`DESIGN.md`, "rent-or-buy consolidation") and changed direction. Their comments are inline in `DESIGN.md` (text between hyphens, `-like this-`) and on the last line of `../FINDINGS_AND_PLAN.md`. Read them yourself; below is the orchestrator's reading, with the PI's own words where they are open to interpretation.

## What the PI wants (fixed)

1. **Aim.** Build the best agent for the three core problems: lifelong memory far beyond any context window; continual learning without forgetting; a reward that makes it interact with its environment and evolve. The contribution is a working agent and its lifelong evidence, not a new principle. Existing ideas and code are preferred ("If all ideas existed, that is even better"). Rent-or-buy is no longer the claim.
2. **End state.** A personal lifelong assistant: the PI keeps giving it tasks; it finishes them and hands them back, learns the PI's habits, confirms what the PI needs, and builds and uses its own tools. CADWorld is only its first environment.
3. **A life like a human's.**
   - Time is irreversible: "no revert would happen in higher aspect, like false would be a cost as well … we can not revert back the time." No rollback of the agent's weights, memory or world. Its environment persists and changes across tasks.
   - Every experience is learning, evaluation included. There are no feedback-free evaluation episodes.
   - Death is natural; do not author hazard tasks. "When model facing some non critical fault, it will learn from the fault. But when critical fault happen, human will be death and no restart. … another model (restart, like another human) can learn from the summarization from the [severe] critical experience from previous model."
4. **Access.** Only fundamental tools (GUI, shell, Python) and full freedom inside its VM or container; it builds everything else. Open internet ("it needs to learn and grab knowledge"). It may contact humans and ask for help ("no boundary").
5. **Learning like a human.**
   - Reflection (replay): "how can the model more efficiently learn from the limited trajectories."
   - Evolution by comparison: "Either at same time or in series. Model evolve (like advantage over other rollout) and whoever have the better performance adapt (dominated but maybe not suddenly change all, some decay)."
   - A sleep rhythm: "I actually like this idea about model decide when to adapt or evolve, but maybe we can define a hyper parameter on this? As like a day is 24 hours and human evolved to do the 'general replay' during sleep time, but that is adapted to the environment setting."
   - The agent may evolve its base model too, not only adapters.
6. **Benchmark as is.** Use CADWorld's tasks, evaluators and references unchanged. Build no task variants or reference constructors.
7. **Numbers.** Give every constant a one-line reason, or mark it as a tunable default and state the rule that tunes it; the PI rejected round 1's unexplained thresholds. Accepted by the PI: general drift ≤ 1 pp per update and ≤ 2 pp cumulative; retention ≥ best − 5 pp. Energy and reward constants are yours to choose; performance is what matters.
8. **Documents.** The PI reads only a short plain-language summary. No jargon without a plain explanation.

Orchestrator decision: the internet is open except for the benchmark's answer key (the CADWorld repository and its evaluator code), and all traffic is logged. Otherwise CADWorld results mean nothing.

## Inputs

- `../FINDINGS_AND_PLAN.md` §6: environment, model and compute facts, including the 2026-09-29 smoke test.
- `DESIGN.md`: round 1 with the PI's comments. Keep whatever survives the new direction.
- `../survey/parametric_memory.md` and the papers in `../../Papers/`.
- `../../third_party/CADWORLD` (pinned): check how a task is set up and scored, to decide whether tasks can run inside one persistent VM.

## Questions the design must answer

1. **A day in the life.** What a day is (episodes, wall-clock or energy) and its default length, with the reason. What happens awake and asleep. What the agent decides and what the rhythm fixes. Whether parallel VMs are several bodies sharing one brain.
2. **Reflection.** The concrete pipeline that gets the most learning out of few trajectories, e.g. verifier stage reports, contrast of successes and failures, hindsight, self-set practice, reading, asking a human. What is kept as text, what is trained into weights, and when.
3. **Evolution.** The learning operator (RL with group-relative advantage, self-distillation, SFT, replay, or a mix), and how "better behavior dominates gradually" is implemented. How old skills and general ability are protected without rollback. When and how the base model itself changes.
4. **No-revert world.** How CADWorld tasks reach a persistent home; what persists (files, tools, notes, installed software); what the agent can break and what follows. How updates stay safe when they cannot be undone. Is testing a candidate update before adopting it compatible with "no revert"? Argue it.
5. **Death and succession.** Critical versus non-critical faults, and how each is detected. What the successor inherits (which weights, if any; testament contents; tools) and how it learns from the testament. How the lineage and the scientific record survive a death.
6. **Reward and energy.** Keep, simplify or replace round 1's energy ledger. How "a mistake is a cost" enters. Anti-gaming that survives open internet and human help.
7. **Measurement.** With every episode also a lesson, how progress, forgetting, transfer and cost are measured (e.g., first-attempt success on never-seen tasks over the lifetime). The fewest comparison agents needed to show this agent beats the same base model with the same tools and a strong context memory, and how they run without violating "no revert".
8. **Stream.** Task order and recurrence over CADWorld's 200 tasks (no variants), then later environments and the PI's own tasks. How the agent gets its first successes when the start model rarely succeeds.
9. **Build.** A reuse-first map of existing open-source frameworks (agentic RL, memory, serving) and what we still write. Milestones with go/no-go gates. Compute on 2 GPUs now and 4 later.
10. **Risks.** The top risks and pivots, and at most 3 decisions that truly need the PI.

Where a PI wish conflicts with a working agent (for example, a permanent death could erase months of learning), say so plainly and propose the least harmful way to honor it.

## Deliverable

Write `round2_<your letter>.md` in this directory, in two parts:

1. **For the PI:** at most one page, in plain words and human analogies, with no symbols. Every number carries its reason.
2. **Internal:** exact rules, equations and numbers, each with its reason. Prefer tables to prose. At most 3,000 words.
