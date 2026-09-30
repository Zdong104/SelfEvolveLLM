# Design B, round 2: one lived life

Work by day, sleep by night, and earn your place. Draft for PI review, 2026-09-30. Part 1 is for the PI; Part 2 is the internal specification. [U] marks something unverified, to be measured in M0–M1.

## Part 1. For the PI

**The idea.** The agent lives one continuous life. By day it works in computers that are never reset. By night it sleeps and turns the day into skill. A changed brain must earn its place before it takes over. Nothing is undone: mistakes stay on the record and cost it.

**A day** is an energy allowance, like a person's daily food. It is about what 160 task attempts cost the starting model. That is the smallest day that shows, within about four days, whether a new brain got worse. Four bodies work on each task at once: four computers sharing one brain and one memory. Four is the fewest that let two brain versions each try the task twice. New tasks come first. The agent decides the rest: retries, practice, reading, questions, notes, and when to stop. Once your tasks begin, its day will be yours.

**Reflection.** After each graded attempt, the agent writes down what went wrong and one lesson. In the evening it notes what the better attempt at the same task did differently. At night a wiser self re-reads every step: the same brain, shown the grade, the lessons and any success. The brain then learns to act like it without the notes. Facts stay in the notebook; skills move into the brain.

**Evolution.** Better attempts are copied a little, never all at once. Each night the brain may change only a little: more after a success, less after a failure. The proven brain and a learning brain share the work. The learner gets more of it while it does better and less while it does worse. A clear winner replaces the proven brain. A clear loser fades, and a new learner grows from the proven one, keeping all memories. That is how people adopt habits. Every ten days or so, proven learning is fused into the base model itself.

**No going back.** The computers are never reset, so yesterday's leftovers are its problem. Old brains and memories are never restored. Trying a new brain on part of the real work first is looking before leaping, not going back, and those tries count.

**Death.** Ordinary faults are lessons with a cost. If the agent breaks its computer so badly that it cannot act even after a restart, it dies. Its successor gets the fused base brain, the family library of tools and notes (including what it knows about you), and a testament: a nightly will and a record of its last moments. It loses personal memories and at most ten days of learning, which is why fusing happens every ten days. Restarting from the original model would throw away months of work; we advise against it.

**Reward.** A success earns a point. Wrong work handed back costs half a point: enough to make it check, but not enough to stop it trying. Thinking is cheap. An action costs about one step of thinking, because actions cannot be taken back. A question to a person costs a whole attempt, because people's time is precious.

**How we will know.** Each task is new to the agent exactly once. We count its first-try successes on never-seen tasks over its life, against a twin with the same model, tools and notebook, living the same days, whose brain never changes. Old skills must stay within 5 points of their best, and general ability within the 1 and 2 points you agreed.

**Three decisions for you.** How far it may reach into the outside world. What a successor inherits. Who answers its questions during CADWorld, since you know the answers.

## Part 2. Internal specification

Ẽ is the start model's median attempt energy, from the census in §7. An attempt is one body on one task, until DONE, FAIL or 100 steps.

### 0. Facts and conflicts

Checked in CADWorld `e5d0eba`:
1. Setup is the task's `config` only: upload a fixture to `/home/user/Precondition_Unnamed.FCStd` (100 tasks) and `rm -f /home/user/Unnamed.FCStd` (all 200). It assumes a fresh VM with FreeCAD auto-started.
2. Grading is host-side and reads files only; of the trajectory, only DONE/FAIL counts. A lived-in VM is therefore gradable while its control server answers.
3. Docker "reset" destroys the container. The image declares `VOLUME /storage`, so a named volume per body should keep the disk [U].
4. The geometry details in `evaluation.json` hold expected values; raw, they leak the answer key.
5. All tasks share two paths, so a persistent FreeCAD can hold a stale document under the task's file name [U].

| PI wish | Conflict | Least harmful way |
|---|---|---|
| permanent death | erases learning | fuse proven learning into the base every ~10 days (§5) |
| no feedback-free evaluation | measurement needs untouched items | first encounters and first review tries (feedback comes after), against a twin; check-ups read a copy of the weights |
| no revert | a harmful update would stick | test before acting (§4) |
| benchmark as is | CADWorld assumes a fresh VM | each task's own setup in a lived-in VM, grader and "Use GUI," unchanged; not leaderboard-comparable |
| fundamental tools only | slower start | an onboarding day; the agent builds its own tools |
| open internet; unbounded human contact | leaks; other AIs doing the work; harm | a logging proxy; labeled help (decisions 1, 3) |

### 1. Day and night (Q1)

| Item | Rule | Reason |
|---|---|---|
| Day | ends when E_day = 160·Ẽ is spent or the agent stops; no carry-over | holds 8 new + ≤ 24 review groups of 4, plus slack; ~32 groups catch a 10-point drop of the learner in ~4 days (§4.3), 16 never do; a usual RL batch |
| Bodies | 8 per agent (2 groups at once): persistent VMs sharing one brain, notebook and library | parallel attempts are the "at the same time" comparison |
| Order | new tasks, then due reviews, then the agent's choices | first encounters fall on the same days as the twin's |
| Night | after every day; both GPUs train | the morning brain has last night's learning |
| Clock (2 GPUs) | day ≈ 3 h, night ≈ 3.3 h: ≈ 3.5 days per calendar day [U] | 16 bodies, ~50 attempts/h per GPU [U] |
| PI's tasks | day = the PI's day | human time |

**Awake.** Deliver, attempt, grade (R1), review, and decide on retries. The prompt holds the danger list, core blocks, the top-5 notes (≤ 1,500 tokens, under one screenshot) and the library index. A group's notes appear only after it ends, so its 4 attempts stay independent.

**Asleep, in order:** evening (vLLM still up: R4, R5, R8 for agent and twin; teacher scoring), reboot bodies with files kept, train, check-up, then at dawn shares, promotion or retirement, and any due merge.

| The rhythm fixes | The agent decides |
|---|---|
| day length, new-task rate, reviews | retries (≤ 2 per task per day, since grinding spends the day), giving up, handing back |
| night size, check-ups, competition | notes, tags for the night, study material |
| nightly reboot | practice, reading, tools, asking, ending the day |

### 2. A home that is never reset (Q4)

**Delivery.** Our runner replaces `DesktopEnv.reset()` with the task's own `SetupController.setup(config)` inside the running VM, then sends the instruction and images verbatim; `evaluate()` and `run_task_diagnostics` are unchanged. A provider subclass gives each body a named `/storage` volume, restarts rather than removes containers, and binds ports to localhost.

**What persists.** Per body: files, packages and FreeCAD settings; open documents until the nightly reboot. Shared: the library (git, synced at task start and end). On the host: the notebook and the diary.

| The agent breaks | What follows |
|---|---|
| its work or notes | the task fails; it rebuilds |
| its software (FreeCAD settings, packages) | it repairs with sudo and internet, paying energy |
| its session (hung GUI, killed control server) | a reboot keeps the files; the attempt fails; a "faint" joins the danger list |
| its body for good (disabled control service, unbootable or full disk) | death (§5) |
| the library (a bad commit) | reaches all bodies; restorable from its own git history |

**Safe updates without undo.** Safety comes before action. A new learner passes a check-up on a copy, changes a capped amount per night, does at most 3 of 4 attempts until promoted, and retires within ~4 days if worse (§4). The base changes only after a promotion.

**Testing before adopting keeps "no revert."** A revert undoes what happened; a test decides what will happen. The trial is lived: its attempts are real, graded, remembered and trained on, whatever the verdict. The proven version is no restored snapshot; it has worked all along. Only a night's weights that fail the check-up are dropped, before acting, like an unexecuted plan.

### 3. Reflection (Q2)

| Stage | When | Input → output | Kept as |
|---|---|---|---|
| R1 grade | each attempt | report → pass/fail, failure class, stage flags, names of failed checks; never expected values | diary |
| R2 review | after R1 | → ≤ 150 tokens: what happened, one lesson | note (task, tags) |
| R3 retry | after R2 | the agent's choice; a retry sees R1 and R2 | diary |
| R4 contrast | evening, per group | best vs worst attempt → the first decisive difference (ReasoningBank MaTTS) | note |
| R5 merge | evening | merge duplicates; a lesson shared by ≥ 2 tasks becomes a skill note; notes unused 30 days are archived | notebook |
| R6 wiser self | night | teacher = learner + best hint of {R1; +R2; +R4; + sibling success digest}, by top-k overlap (OpenClaw-RL) → per-token targets on the student's steps | weights |
| R7 compare | night | rewards → advantages (§4.1) | weights |
| R8 will | night | ≤ 1,000 tokens: skills, open problems, warnings, what it knows of the PI | testament |
| practice, reading, asking | free time | self-checked exercises, docs, answers | notes; practice trains only through R6, because the agent grades itself |

| Content | Text | Weights |
|---|---|---|
| facts (paths, names, the PI's preferences) | at once, until contradicted | never directly (TENSE: facts go to side memory) |
| lessons, skills | at once (R2, R4), abstracted in R5 | that night, as R6 hints; a note behind ≥ 2 successes also joins replay |
| tools | library, at once | when to call them, through the attempts that used them |
| dangers | always in context | every night, first |
| raw episodes | diary, forever, searchable | replay source |

≥ 2 because one success may be luck. 30 days because gaps reach 16–32 days, so the note missed a full review cycle; it stays searchable.

### 4. Evolution (Q3)

**4.1 Advantages.** Group g is the 4 simultaneous first attempts: A_i = r_i − mean_g r, without division by the standard deviation (Dr. GRPO). A retry, the series comparison, gets A = r_retry − mean_g r. After faints and deaths, the last 10 steps take the fault report as R6 hint.

**4.2 Night loss.** θ is the learner, β the version that acted, and ρ = π_θ/π_β per token.

L = (L_RL + L_OPD)[today's selected steps, 50% of tokens] + L_replay[30%] + L_anchor[20%]

| Part | Rule | Reason |
|---|---|---|
| L_RL | −min(ρA, clip(ρ, 0.8, 1.28)A) | DAPO/OpenClaw-RL clip |
| L_OPD | OpenClaw-RL eq. 1: top-20 tokens, Δ = clip(log π_T − log π_old, ±1), 1:1 with L_RL; teacher from vLLM `prompt_logprobs` | their defaults |
| L_replay | −log π_θ on stored verified successes (each solved task at least every 3 nights; dangers nightly) | keeps what worked |
| L_anchor | top-20 KL(π_θ0‖π_θ) on 2,000 general prompts answered by the original Qwen3.8 (θ0) | protects general ability |
| Selection | A ≠ 0, teacher–student KL above the day's median, or tagged; ≤ 1,500 steps | ≈ 2.3 h with 2-GPU DDP (10.8 s per 8K step, smoke test) |
| Mix | 50/30/20; +10 points to replay after a retention failure, to anchor after a check-up failure | today first; 30/20 from round 1 A |
| LoRA | r = 64, all language linear layers including Gated DeltaNet; vision frozen; lr 2e-5 [U] | smoke-tested |
| Cap | stop when top-20 KL(tonight ‖ last night) on held-out states reaches δ; δ₀ = half a full night's KL (M2); ×1.5 per promotion, ×0.5 per retirement, in [0.01, 0.2] nat/token | Rechenberg's success rule; the cap, not the lr, bounds change |

**4.3 Two living versions: gradual dominance.** P is proven and L is learning. Each group gives round(4s) bodies to L, with s ∈ [0.25, 0.75].
- z_d = mean_g Δ_g / (sd(Δ_g)/√G), where Δ_g = L's mean first-attempt r − P's.
- Z_d = 0.7·Z_{d−1} + z_d, and s = 1/(1 + e^(−Z/2)).
- Promote (P ← L) at Z ≥ 3. Retire at Z ≤ −3: the new L copies P, and δ halves.

Why: 0.7 halves old evidence every 2 days, since L changes nightly. With no real difference, Z's spread is 1.4, so it passes ±3 on under 2% of days each way. A 10-point gap (≈ 1.2 standard errors a day at 32 groups [U]) crosses on day 4. This is replicator dynamics with decay: shares follow relative fitness, and nothing switches suddenly. L trains on both versions' attempts; ρ corrects for P's.

**4.4 Check-up.** It runs on a copy, like a blood test, not as an experience.
- Parity: trainer vs vLLM mean |Δ log p| ≤ 0.02 nat (under 5% of a trained adapter's 0.45 nat). Adapter-to-base KL ≥ 1e-3, against silent no-ops.
- General: 600 fixed greedy items (MMLU-Pro 200, IFEval 150, GPQA-D 100, ScreenSpot-Pro 100, BFCL 50). Drop ≤ 1 point vs P and ≤ 2 vs θ0 (PI), and ≤ 3 on any benchmark, because averages hide narrow losses.
- Memory: the action-token log-likelihood of 200 stored successes falls ≤ 10% vs P (thinking is excluded; shorter is welcome).
- Failure: tonight's weights are dropped unused; the next night gets +10 points of anchor and replay.

**4.5 How the base model changes.**
- Genome merge: at the first promotion ≥ 10 days after the last, if cumulative drift ≤ 2 points. W ← W + ΔW_P, the adapters are zeroed (behavior unchanged up to bf16; canary re-run), and vLLM restarts. This bounds a death's loss and keeps base edits rare; the anchor stays θ0.
- Base swap to a new family (the PI's call): schooled on the diary's successes and the library, the newcomer enters as L and must win §4.3.
- Study material the agent nominates becomes R6 hints, subject to the check-up.

### 5. Death and succession (Q5)

| | Rule | Reason |
|---|---|---|
| Vital signs | after each action and delivery: the control server answers, the screenshot is not blank, `true` runs, ≥ 1 GB is free | the minimum needed to act |
| Faint | signs fail 3 times over 90 s → reboot, files kept → signs return within 5 min | ordinary recovery |
| Death | signs still fail after the reboot, in any body | it can never act again; one brain chose the act, so the individual dies |
| Weights | latest genome, zero adapters | bounded loss |
| Inherited | library (tools; curated notes, including the PI profile), will (R8), black box (last 50 actions), danger list | culture and testament |
| Not inherited | adapters, raw diary, bodies (it gets fresh VMs) | personal memories die |
| Learning the testament | at birth it writes a death lesson into the danger list; first night: R6 on the fatal steps with that lesson as hint, plus −log(1 − π_θ(fatal action)) | one-trial learning |
| Record | append-only ledger of all attempts with lineage IDs, plus read-only archives of weights, memory and disks; agents never see it | science survives death |

Natural death may never come in CADWorld; the M5 base-swap rehearsal exercises the same path.

### 6. Reward, energy, anti-gaming (Q6)

r = V + 0.25(1−V)S − 0.5·D(1−V) + 0.2·V·clip((Ē_succ − E)/Ē_succ, ±1)

| Term | Meaning | Reason |
|---|---|---|
| V | grader score | benchmark as is |
| S | share of applicable grader stages passed | nearer misses guide; ≤ 0.25, so any success beats any failure |
| D | claimed DONE but failed | mistakes cost; break-even belief ≈ ⅓, so it still tries. ×2 if under half of DONEs pass after 10 days; ×0.5 if it abandons over half of the tasks it later solves |
| efficiency | among a group's successes only | failing fast never pays |

Mistakes also cost through permanence: every attempt stays on record, and damage persists until repaired.

**Energy**, simplified from round 1 (no GPU-second calibration, starvation or reservoir), in generated-token equivalents:

E = N_gen + 0.05·N_prompt,uncached + 500·N_act + Ẽ·N_ask

0.05 is the decode-to-prefill cost ratio [U]. An action costs 500, one step's typical thinking, per the PI's rule that acting costs more; a planned script thus beats ten clicks. A question costs Ẽ, because people's time is precious. Energy sets the day, the efficiency term and the cost metrics; nights are free.

**Anti-gaming.**
1. The grader, fixtures and references stay on the host. The VM gets only the instruction, its images and the precondition file.
2. A transparent logging proxy (mitmproxy) blocks CADWorld's repository, dataset, site and mirrors by host, path and content signature (task IDs, rule strings, file hashes), plus known LLM services and the LAN. A leak marks the affected tasks as contaminated.
3. R1 never shows expected values.
4. Every success is recomputed in host FreeCAD 1.1.3. A mismatch with the stored properties scores V = 0, D = 1 and is logged as cheating.
5. Help is logged; the task becomes "assisted", outside the unaided statistic.
6. Rewards, energy, the stream, the grader and the record are out of the agent's reach.

### 7. Measurement (Q7)

| Quantity | Measure |
|---|---|
| progress | first-attempt success on never-seen tasks over the lifetime (4 per task), agent − twin, paired by task |
| near vs far | new tasks sharing a coverage tag with a solved task vs none; the next environment vs the twin |
| forgetting | first try at each review vs the task's best earlier rate; gap ≤ 5 points (PI) |
| weights-only memory | 10% of review groups closed-book (no notebook), graded as usual |
| general | check-up drift |
| cost | lifetime energy per verified success, nights included; energy at the n-th solve |

**Comparison agents.**
- **Twin T**, the strong context-memory control, shares everything except weight updates: the model, prompts, tools, bodies, notebook, R1–R5, R8, library, help rules, energy day and new-task days. It lives its own never-reset life on the same GPUs, so nothing is reverted.
- **Census C**, the memory-less base model, makes 4 attempts per task once before birth. Like a published baseline, it is not a life. It gives task difficulty and Ẽ.
- Ablation twins wait for 4 GPUs.

**Statistics (pre-registered).** logit(success) ~ agent + position + agent:position + (1|task), on first attempts. Primary: the agent effect on tasks 101–200, one-sided 95%; it detects ≈ 8–9 points at 80% power near a 25% base rate [U].

### 8. Stream (Q8)

| Rule | Reason |
|---|---|
| seed-2026 permutation of all 200; 8 new per day, same days for the twin | 32 of ~160 daily attempts; all met in 25 days (a 25-night curve); random order stays unbiased |
| unsolved tasks return after 1, 2, 4, 8, 16 days | learnable as the agent grows, without flooding days |
| solved: success doubles the gap (≤ 32 days), failure resets it to 1 (Leitner) | practice cost grows logarithmically; forgetting is repaired fast |
| ≤ 24 review groups per day, most overdue first | slack for retries and practice |
| free time: re-request any seen task (graded) or set an exercise | self-chosen curriculum; unseen tasks untouched |
| days 26–40 mastery; then the next environment (OSWorld-Verified [U]) with CADWorld reviews; PI tasks from M4 | retention across environments; the PI's replies become grades |

**First successes** come from an onboarding day (explore FreeCAD, read docs, build tools, self-checked exercises), 4 bodies per task, retries with feedback, partial credit, and R6, which learns even when every attempt in a group failed. Docs, questions and common easy tasks ("sketch one 10 mm line") help too. There is no difficulty ordering, so the twin comparison stays clean.

### 9. Build (Q9)

| Need | Reuse | We write |
|---|---|---|
| tasks, grader | CADWorld `e5d0eba`, unchanged | persistent provider, delivery, R1 filter, recompute audit |
| serving | vLLM 0.30 (multi-LoRA, `prompt_logprobs`) | adapter rotation |
| training | PEFT, transformers, flash-linear-attention (smoke-tested), DDP; losses from OpenClaw-RL and SDPO (Apache-2.0) | trainer and data builder (~700 LOC) |
| RL frameworks | verl-agent (GiGPO), OpenClaw-RL (slime, 8 GPUs by default), TRL, ms-swift, ART | none: our update is an offline nightly batch, and PEFT handles Gated DeltaNet; revisit at 4 GPUs |
| memory | ReasoningBank and MaTTS (Apache-2.0), Letta-style core blocks, Qwen3-Embedding, SQLite | ~500 LOC |
| boundary, probes | mitmproxy (MIT), lm-evaluation-harness (MIT), VLMEvalKit (Apache-2.0) | rules, subsets |
| life manager | — | scheduler, meter, watchdog, competition, merge, succession, ledger (~1,500 LOC) |

| Milestone | Go if |
|---|---|
| M0 build, 3 weeks | 50 consecutive tasks in one body without host help; a reboot keeps files; canary passes; ≥ 40 attempts/h per GPU; the proxy blocks the repo and a test fork |
| M1 census, onboarding, days 1–5 | ≥ 10 distinct tasks solved by day 5 |
| M2 days 6–12, full loop | ≥ 1 promotion; no unexplained check-up failure |
| M3 days 13–40, CADWorld | agent − twin on tasks 101–200 > 0 (one-sided 95%), estimate ≥ 5 points; budgets hold |
| M4 days 41–100, next environment and PI tasks | CADWorld retention ≥ 90% of day 40 after 1,000 foreign attempts; first attempts ≥ twin |
| M5, 4 GPUs: ablation twins (no R6; no competition), base-swap and succession rehearsal | each part beats its ablation |

**Compute.** On 2 GPUs, the day serves 16 bodies (8 agent, 8 twin) and the night trains with DDP. At ≈ 3.5 days per calendar day [U], M3 ends in about 6 weeks. 4 GPUs double the bodies (8 per new task) or add an ablation twin.

### 10. Risks, pivots, decisions (Q10)

| Risk | Signal | Pivot |
|---|---|---|
| too few successes | < 10 tasks by day 5 | longer onboarding, more thinking, more help, 8 bodies per new task |
| clutter drowns skill | lived-in success < half the census rate for 10 days | reboot at each delivery, files kept |
| weights add nothing | agent ≈ twin | report it; more R6; longer nights |
| self-training collapse | 3 retirements in a row | 5-day rest: replay and anchor only, minimum δ |
| noisy competition | promotions and retirements alternate | 8 bodies per group, or decay 0.8 |
| answer-key leak | signature hit | mark tasks contaminated; report |
| frequent deaths | > 1 per 20 days | shorter merge interval |
| GDN LoRA breaks | canary | MLP-only LoRA |

**Decisions for the PI.**
1. **Outside world.** Recommended: read-mostly internet through the proxy; no e-mail, posting, payments or accounts; other AIs blocked, since they would do the work; humans through one lab channel, widened later. Without a boundary, a root agent on the open internet can harm others and the lab.
2. **Inheritance.** Recommended: evolved base, library and testament, losing at most ~10 days. The alternative, the original Qwen plus library and testament, is a true newborn that loses all learned weights.
3. **CADWorld helpers.** Recommended: lab members who do not know the grader rules answer general FreeCAD questions only, at most 3 a day, logged. You help only outside CADWorld.

### From round 1

- **Kept:** LoRA r64, canary, probe, budgets, replay, anchor; lessons, ledger and tools, now the notebook, diary and library.
- **Changed:** admission plus rollback became check-up plus competition; energy in token units; GRPO+OPD writes.
- **Dropped:** rent-or-buy, variants, `fc()`, the feedback-free panel, hazards, starvation.
