# SelfEvolveLLM: findings and plan

- Last updated: 2026-09-29
- Owner: PI (Zihan Dong)
- Status: research design; implementation has not started

This is the project source of truth. Keep it short and update it when a decision changes. Survey notes are supporting evidence, not the plan.

## 1. Goal

Build a general agent that learns continually from verified experience. Given an arbitrary task in an isolated shell-and-internet sandbox, it should try, use verifier feedback, recover from failure, and consolidate reusable experience into its own weights. After consolidation it should:

1. solve the same task more directly after all episode context is removed;
2. transfer to related unseen tasks;
3. retain earlier skills and general model capabilities; and
4. spend less inference energy with practice.

This is not a computer-use project. The research contribution must work across domains. The agent may create tools inside its sandbox; tools are external procedural artifacts, while knowing when and how to use them may be consolidated into weights.

## 2. Scientific thesis and novelty bar

### Core problem to resolve:

Life time long context, Continual learning without fail the previous knowlegde. Reward design to make model interact with enviroment and evolve.

### Working hypothesis

An agent optimizing lifetime verified reward minus inference cost should consolidate experience in proportion to its recurrence, surprise, and cost to re-derive. Rare events remain episodic; recurring structure moves through fast weights into protected slow weights; unused structure decays. This predicts reduced reasoning/action cost with practice, retention under interference, and transfer through abstraction.

### Required contribution

Combining existing methods is not sufficient for a top conference or journal. The project needs one clear new mechanism or principle and decisive evidence that it produces a behavior unavailable from context memory, ordinary fine-tuning, replay, or existing self-evolving agents.

The minimum convincing result is a long, heterogeneous task stream showing all of the following:

- context-free re-solving after a memory wipe;
- near and far transfer to held-out task families;
- bounded forgetting and sustained late-stream learning;
- lower lifetime inference cost than strong retrieval/context baselines; and
- causal ablations connecting the result to the proposed consolidation mechanism.

A main-journal claim requires a broadly important scientific finding, not only a better agent system. A top ML conference is the primary method-paper target; a broader journal submission follows only if large-scale results reveal a robust general learning phenomenon.

## 3. Current evidence

### Relevant prior work from this group (You can find more background)

- **SCoL** learns where to write streamed textual knowledge with LoRA and rewards acquisition minus forgetting. It supplies a learned-plasticity idea but does not learn procedural agent experience.
- **TENSE** treats writable capacity as finite and routes infeasible edits to side memory. It motivates an admission/capacity gate.
- **Titans** supplies surprise-driven updates, momentum, and adaptive forgetting. Its principles may inform adapter updates; its architecture is not directly attachable to a pretrained Qwen model.

### External evidence

The focused survey in `survey/parametric_memory.md` supports four design choices:

- on-policy/context self-distillation is a promising way to write only the teacher-student residual;
- fast and slow parameter timescales need replay and regression control;
- synthetic “dreams” can improve transfer when they are selected and verified; and
- every accepted update needs an admission test because repeated self-training can collapse.

The nearest competitors must be rechecked immediately before claims are written. Current high-priority comparisons include Language Models Need Sleep, SOLO, LifeSkill, PEAM, SDFT/SDPO/OPSD, SEAL, and OpenClaw-RL.

### Project independence and reuse boundary

This project is independent of other projects on this machine. Do not assume, import, edit, or depend on code outside this repository. 

Open-source projects may be evaluated and imported into this repository when useful. Record their origin, revision, license, and local modifications. All project artifacts, caches, environments, models, and experimental outputs must stay under this repository or another location explicitly approved by the PI; never modify unrelated folders.

## 4. Minimal system design

### Wake loop

1. Attempt a verifier-backed task in an isolated sandbox.
2. Retry with verifier feedback while the expected success value exceeds the remaining energy cost.
3. Store successful and failed trajectories plus a compact lesson in an episodic buffer.
4. Score the episode by reward prediction error, teacher-student gap, recurrence, and estimated future inference cost.
5. Admit only updates that improve held-out acquisition without violating retention limits.

### Memory hierarchy

| Memory | Contents | Lifetime |
|---|---|---|
| Working context | current attempt and feedback | one episode |
| Episodic buffer | trajectories, lessons, verifier evidence | until consolidated or stale |
| Tool library | agent-created executable procedures | persistent, versioned |
| Fast adapter | recent high-value residuals | frequent decay/reset |
| Slow adapter | replay-tested abstractions | long-lived, protected |
| Base model | fixed during pilots | unchanged |

### Write and sleep

The default write operator is context distillation: an experience-conditioned teacher and context-free student are the same model, and the student learns the teacher-student residual on verified outputs. Compare this directly with successful-trajectory SFT, off-policy distillation, and verifier-driven RL.

During periodic sleep:

- replay new episodes interleaved with retained older cases and general-capability samples;
- generate related task variants, but train only on variants whose answers or executions can be verified;
- distill accepted fast-adapter knowledge into the slow adapter;
- reject or roll back updates that fail acquisition, retention, or general-capability gates; and
- decay/reset fast memory after successful consolidation.

Start with simple adapter separation and explicit evaluation. Add MAS/EWC protection, learned layer selection, TENSE-style projection, or adaptive decay only when an ablation justifies the complexity.

### Energy objective

For episode \(i\):

\[
R_i = R_{success}V_i
      - c_{think}N_{think,i}
      - c_{act}N_{act,i}
      - c_{env}N_{env,i},
\qquad c_{think} < c_{act}.
\]

`V_i` is produced by a hardened executable verifier. Compare methods by cumulative verified reward and total cost over the same task stream, not just endpoint accuracy. Calibrate costs so failure cannot be made attractive by quitting early. Keep the verifier and its protected files outside the agent-writable sandbox.

## 5. Experiments

### Pilot: kill the riskiest assumption first

Use procedurally generated, contamination-resistant CLI/API worlds with hidden but shared rules. Train on initially failed tasks and evaluate after clearing prompts, retrieval memory, and tools from the test context.

The pilot must compare:

- no memory;
- retrieval/context memory at equal inference-token budget;
- successful-trajectory SFT;
- context/on-policy distillation; and
- distillation with replay plus an admission gate.

Measure exact-task re-solving, held-out near/far transfer, backward transfer, general-capability drift, tokens/actions per success, and wall-clock/GPU cost. Use multiple task seeds and report confidence intervals.

**Go/no-go:** proceed to the full architecture only if a weight-based method improves held-out transfer or lifetime cost over context memory while staying within a predeclared forgetting budget. Otherwise simplify or pivot the write operator.

### Full study

Scale to hundreds or thousands of mixed-domain episodes. The critical hypotheses are:

- consolidation improves context-free re-solving and transfer;
- replay plus admission control bounds forgetting;
- late-stream acquisition does not degrade relative to early-stream acquisition;
- verified dreaming improves transfer more than exact-task retention; and
- energy per successful solve decreases predictably with repeated exposure.

A realistic benchmark is selected only after it passes four filters: executable verification, reproducible isolation, affordable repeated trials, and measurable cross-task structure.

## 6. Models and compute

- **Pilot:** benchmark a current approximately 8B Qwen-family model first for rapid LoRA iteration. Pin the exact model revision after a smoke test of training, serving, tool use, and license compatibility.
- **Scale-up:** use [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) as the current 27B candidate. Its official model card describes a 27B dense model compatible with Transformers, vLLM, and SGLang. Pin a commit; do not silently track “latest.”
- **Hardware:** 4× RTX PRO 6000 Blackwell (96 GB), plus one 8 GB A1000. Always set `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- Prefer one rollout GPU and one training GPU for the pilot. Use additional GPUs only after the single-node loop and admission tests are reliable.

## 7. Execution plan

1. Initialize version control and add a minimal ignore policy.
2. Finish the competitor/benchmark survey only to the level needed to choose a pilot.
3. Freeze the pilot protocol, baselines, metrics, and forgetting threshold before training.
4. Build the smallest rollout → verifier → trajectory → adapter update → regression-gate loop.
5. Run the pilot and make the go/no-go decision.
6. Add fast/slow memory, replay, dreaming, and adaptive protection one at a time behind ablations.
7. Scale the task stream only after a simple method passes the pilot.

## 8. Documentation rules

- `FINDINGS_AND_PLAN.md` is the only living plan.
- `design/design_brief.md` is the current design-review prompt.
- `survey/` contains compact evidence notes, not operational plans.
- Delete stale status logs and superseded plans instead of appending indefinitely.
- Mark unverified claims, pin versions, and link evidence needed for later citation.
- Do not include paths or operational details from unrelated projects.

## 9. Open PI decisions

1. Should the first realistic domain be terminal/coding, research, or a balanced mixture?
Lets start with the CADWORLD benchmark. but the model should be able to remeber more in future benchmakrs and the model should not be reset. we let it run forever from now even across benchmarks. Which means the model should have strong continual learning ability and be able to adapt new knowledge without forget the old.
