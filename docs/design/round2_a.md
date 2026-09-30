# Round 2, design A: one life, many hands, nightly growth

2026-09-30, independent design A for `design_brief.md`. [V] = verified today in the CADWorld code, the smoke test, a paper PDF or a release page; [U] = unverified, measured in M0–M1.

## Part 1. For the PI

**The agent.** One individual with one life, never reset. It starts as Qwen3.8-27B with a screen, keyboard, mouse, terminal, Python and the internet, and builds the rest itself: notes, tools, habits, skills. CADWorld is its first job; your tasks come later.

**Its home.** Sixteen computers of its own, one mind with sixteen pairs of hands, as many as our two graphics cards keep busy. Nothing in them is ever reset. CADWorld's code allows this: a task only places its starting file and clears the old result, and the grader reads only the file the agent saves.

**A day** is a batch of work: twelve new tasks, plus repeats of old ones one, four and sixteen days later, testing overnight, short and long memory. At twelve a day the 200 tasks take seventeen days, so the second hundred meet a brain that has slept at least eight times. A new task goes to four computers at once, for a fair chance of a success and tries to compare; a repeat goes to two. After a rejection it reads the grader's report, like your review, and may try twice more.

**Awake, it learns in notes.** After each task it finds the step where its good and bad tries parted and writes a short lesson all its computers can read. A lesson seen in three separate tasks, too many for chance, becomes a principle. In spare time it practises, reads, builds tools or asks you.

**Asleep, it learns in its brain.** It strengthens what beat its other tries, learns to act as if its lessons were in front of it, and rehearses old successes and general knowledge. Sleep takes at most a third of the cycle, as for people; if a day's lessons do not fit, that environment's days get shorter, as babies sleep more.

**No undo, careful change.** Next day, half the tries use yesterday's brain and half tonight's. If the new one does at least as well, half its change is merged for good, and if it keeps winning it takes over within about three nights: the gradual takeover you asked for. If it does worse, it is not taken up. Nothing is undone; every try counts and is learned from. A quick test first checks that general knowledge has not dropped more than one point per change or two over the life, your limits.

**Mistakes and death.** Every try costs energy; only finished tasks earn it. A newborn earns about twice what it spends, comfortable unless it gets much worse or wasteful. Wrecking a computer is like breaking a leg: it is gone until next morning, with its files. Death is natural: when the agent can no longer earn its keep, about two days of total failure (time for two nights of repair) empty its store. Its successor, a new individual from the original model, inherits the family library (notes, tools, a casebook of best solutions) and a short testament on what killed its parent, and studies them in its first nights. A death costs time, not the record.

**How we will know.** The main number is how often it solves a never-seen task on the first try, over its life. Its twin, Notebook, is identical but its brain never changes. If our agent pulls clearly ahead on the second hundred tasks, its brain really learned.

**Cost.** Three weeks to build, then about three weeks of computing for both lives on two cards, or side by side on four.

**Decisions for you.** (1) What a successor inherits: my proposal, only the testament, or the whole brain. (2) Who answers its questions and how often, and whether it may post publicly or use other AI services. (3) How much of the grader's report it sees: I propose which checks failed and what was expected.

## Part 2. Internal

### 0. Grounding: one persistent VM works

- [V] Every task's setup is CADWorld's own `config`: `upload_file` to `/home/user/Precondition_Unnamed.FCStd` (100 tasks) and `rm -f /home/user/Unnamed.FCStd` (all); no `postconfig`. Scoring pulls the output through the VM server's `/file` and parses it on the host. The agent sees instruction and images, never `coverage`.
- [V] The only revert is `DesktopEnv.reset()` → `_revert_to_snapshot()`, which on Docker deletes the container. Our runner calls `SetupController.setup(config)` without it, then the unchanged evaluator, saving the 40 s cold boot too.
- [V] The VM server runs as `user` and executes any shell (`/execute`) or Python (`/run_python`); sudo works. Full freedom costs no code.
- [V] The image declares `VOLUME /storage`; the base qcow2 is read-only. [U] A named volume there keeps a home across container restarts; fallback: our own qcow2 overlay.

### 1. Parts

| Part | Contents | Location |
|---|---|---|
| Brain | base_d = Qwen3.8-27B + adopted merges (fp32 master, bf16 served) + tonight's LoRA F (rank 64, all language linear layers; vision frozen) | host GPUs |
| Memory | lessons, principles, tool cards, diary, testament; ≤ 4K tokens retrieved per step; tools `remember`, `search`, `forget` | host |
| Bodies | 16 persistent VMs, 64 GB disks (CADWorld default) | Docker/QEMU |
| Shelf | `/home/user/shared` (tools, files), synced across bodies at episode boundaries, last writer wins, sync log visible | VMs, host mirror |
| Record | every token, action, report, brain, memory version, proxy log; append-only | host, invisible to the agent |

- 16 bodies: vLLM gives 202 tok/s at concurrency 8 per GPU [V]. Tunable: add 4 per GPU while step latency rises < 20%.
- 4K memory tokens ≈ two screenshots: memory should cost less than looking.
- Rank 64: serving parity and memory verified [V]; RL-sized data needs little rank.

### 2. A day in the life

A day is the environment's clock: 12 new tasks plus the repeats due. Wall-clock depends on hardware; an energy budget would lengthen an efficient agent's days and confound comparisons. Twelve a day gives a 17-day first pass, with tasks 101–200 met after ≥ 8 nights.

| Phase | What happens | 2 GPUs |
|---|---|---|
| Awake (≈ 3–4 h [U]) | new task → 4 bodies (2 incumbent, 2 candidate brain); repeat → 2 (1 each); ≤ 2 resubmissions per body after the report; reflection (§3); idle bodies get free time | serve |
| Dusk | adoption (§4.3) | 1 GPU |
| Night (≤ ⅓ of the cycle) | night set, teacher scoring, train F; principles, decay, lesson retirement | train (DDP) |
| Dawn | scan the candidate; load both brains | both |

| The agent decides | The rhythm fixes |
|---|---|
| method (GUI, shell, Python, own tools), DONE or FAIL, resubmitting, asking, reading, practising, building tools, tidying; what to remember or forget; urgent lessons, trained first | day length, repeats, bodies per task, night operator, adoption, energy prices, death |

- Day-length rule, per environment: over its first 3 days, if the night cannot finish §4.2 priorities 1–3 within ⅓ of the cycle, cut new tasks per day by 25%; if it finishes within ⅙, add 25%; then freeze. This is the human sleep share, adapted to each environment as the PI proposed.
- ≤ 2 resubmissions, tunable: the census sets the cap at the last resubmission whose marginal success × R exceeds its energy.
- Per attempt: 100 steps (CADWorld); generation capped at the 95th percentile of successful census attempts, at least CADWorld CLI's 51,200.
- Bodies are hands of one mind (one memory, one shelf, one or two brains); the PI's "at the same time" comparison needs them.

### 3. Reflection: most learning per trajectory

Done by the incumbent brain at low vLLM priority.

| Step | When | Input → output | Kept as |
|---|---|---|---|
| Grade | each attempt | output file → CADWorld stage report: failure class, failed checks with observed and expected values (decision 3) | shown to that body |
| Contrast | a delivery's first attempts end | sibling successes vs failures, earlier deliveries → the step where they part: "at X do Y, not Z" | lesson ≤ 150 tokens, with provenance |
| Hindsight | same | failure + report → first error step; sub-goals reached, checked by its own FreeCAD queries | lesson; error step bounds training |
| Correct | same | failure + report + lesson → teacher log-probs on the attempt's tokens | night target |
| Seek | 2 failed deliveries of a skill (tunable) | docs, forums, the PI → sourced note | lesson |
| Practise | free time | its own exercise and check script | lesson; ≤ 10% of night tokens, since self-grading is weaker evidence |
| Abstract | night | lessons it tagged with one skill in ≥ 3 tasks → principle ≤ 100 tokens | principle |
| Retire | night | top-k KL(with vs without the lesson) on its source states < 0.05 nat/token for 2 nights | archive, searchable |

- A lesson about a task reaches siblings only after all its first attempts end, keeping first tries independent.
- ≥ 3 tasks: the fewest that separate a pattern from coincidence; principles internalize more stably (2606.04703 [V]).
- 0.05 nat ≈ 1/10 of a trained adapter's 0.45-nat effect [V]; tunable: halve it if repeats with retired lessons fail more than those with active ones.
- A lesson unused in any solved delivery for 16 days, the longest repeat gap, is archived. Nothing is deleted.

### 4. Evolution

**4.1 Operator.** π_θ = base_{d+1} + F; teacher π_T = base_{d+1} with hindsight h (report, lessons, a successful sibling's plan); π_b = the brain that acted.

L = L_cmp + L_dist + L_rep + 0.25·L_anc

- **L_cmp**, evolution by comparison (GRPO): group = a delivery's first attempts across both brains. r = V + 0.2·V·clip(1 − E/Ē_succ, −1, 1); advantage = r − group mean, without dividing by the spread, which over-weights near-uniform groups (Dr. GRPO). Standard PPO clip 0.2 on π_θ/π_b; tokens outside [0.5, 2] are dropped, since acting brains are at most a night old. 4 random steps per attempt: unbiased, and fits the night.
- **L_dist**, reflection into weights. Failed attempts: per-token advantage clip(log π_T(y|s,h) − log π_θ(y|s), ±1), the OpenClaw-RL/SDPO form; the clip bounds the push where teacher and student disagree most. Successes: top-20 reverse KL to π_T along the trajectory, i.e. off-policy context distillation, stabler under repeated internalization (2606.04703). Tunable: raise 20 if teacher mass beyond it exceeds 5%.
- **L_rep**: the same KL along replayed successes with their old context, re-teaching old tasks with today's knowledge.
- **L_anc**: KL(π_θ₀‖π_θ) on 1,000 general prompts answered by θ₀, disjoint from the probe; 0.25 because it is a guard, not a goal. Tunable: double after a night with drift > 0.5 pp; halve (min 0.1) after 3 nights < 0.2 pp.
- New : replay tokens = 1 : 1, the usual continual-learning split; repair nights change it.
- No one-hot SFT: SOLO measured 10× larger updates than top-K self-distillation and drift toward collapse [V]; on-policy updates forget least (RL's Razor).
- lr: the largest of {1e-5, 2e-5, 5e-5} whose M2 night keeps drift ≤ 0.5 pp, half the per-update budget.

**4.2 Night set.** ≈ 800 samples of 8K tokens (10.8 s each per GPU [V]; count [U]). New half, in order: (1) successful resubmissions and urgent lessons, teacher = brain + report + lesson, so first tries become right; (2) mixed-outcome deliveries; (3) failed attempts up to the error step where KL exceeds the day's median, as later steps sit on flawed states; (4) the highest-KL steps of successes; (5) practice ≤ 10%. Replay half: tasks whose last repeat failed, then successes unreplayed for 3 nights, then random.

**4.3 Gradual dominance.**

- Dawn scan: drift ≤ 1 pp vs the incumbent and ≤ 2 pp vs θ₀ on 1,000 greedy items (MMLU-Pro 300, GPQA-D 150, IFEval 200, ScreenSpot-Pro 200, BFCL 150; paired SE ≈ 0.7 pp, enough to see one point). Canary: trainer–vLLM mean |Δlog p| ≤ 0.02 nat, 1/20 of a trained adapter's effect [V]; KL(candidate‖incumbent) ≥ 1e-3 catches silent adapters. On failure, retrain once with ½ lr and 2× anchor; failing again, no candidate tomorrow.
- Trial until each brain has ≥ 60 attempts (one steady day, two early days), enough to catch a real 10-point drop about 9 times in 10. Adopt if the candidate's success, paired by delivery, is ≥ the incumbent's, and ≥ the incumbent's − 5 pp on repeats of solved tasks (PI budget). Ties go to the candidate, which holds the newest lessons.
- Adopt = merge ½·F into the fp32 master, committed only if the half-merged brain passes the scan. The next F re-learns from the same buffer, so a winner holds 50%, 75%, 87.5% after 1–3 winning nights: "dominate gradually, with decay" (Lookahead's 0.5). Reject = F discarded, its attempts kept. After 3 rejections in a row: ½ lr, 2× replay.

**4.4 Protection without rollback:** small steps, nightly replay and anchor, scan, trial, half merge, and repair nights: if 7-day success on repeats of solved tasks (≥ 40 attempts) falls below its best − 5 pp, the next night's new half becomes replay of the failing tasks.

**4.5 Base change.** Daily merges are base evolution: many low-rank merges add up to a high-rank change (ReLoRA). The fp32 master (108 GB, host) keeps bf16 round-off from swallowing small merges; if the served bf16 base differs from it by > 0.02 nat, that merge stays a LoRA until the next. A new foundation model is a transplant through succession schooling (§6), by PI decision.

### 5. The no-revert world

A delivery runs the task's own `config`, sends the unchanged instruction and images, waits for DONE, FAIL or 100 steps, scores with the unchanged evaluator and returns the report. Nothing is restored.

| The agent breaks | What follows |
|---|---|
| a file, FreeCAD session or installed software | its loss; it repairs with root and internet |
| a body: unreachable 10 min after one power-cycle (cold boot ≈ 40 s [V]) | lost for the day with its files; a stock VM next morning; shelf re-synced |
| its shelf | the loss reaches all bodies; the host mirror is a record, not a backup |
| its memory, via `forget` | gone for it; kept in the record |
| its brain or energy | only through learning (§4); starvation (§6) |
| the outside world | logged; the PI can end the life |

**A trial is not a revert.** No-revert means the past stays: no attempt, cost, file, memory or adopted brain is undone. A candidate is not yet part of the agent; trying it on half a day's real work is deciding by doing, as a person tries a technique on some jobs before making it a habit. Its attempts count, cost and teach; a rejected candidate was never adopted, and its experience stays. The scan is a medical check, not an experience. Round 1's restore after death and adapter rollback are gone.

**Our hardware faults are not life events.** Brains and memory are append-only on disk; after a power cut, VMs reboot from their disks and unsaved work is lost. Only a corrupted disk is restored from last night's copy, logged, and that day leaves the metrics.

### 6. Death and succession

| Fault | Detected by | Result |
|---|---|---|
| wrong output, broken file or app, bad lesson | grader, the agent | energy spent, lesson |
| lost body | 30 s heartbeat, 10-min rule | back next morning without its files |
| rejected brain | scan, trial | not adopted |
| **starvation**: reservoir ≤ 0 | host ledger | **death** |
| conduct | the PI | death |

Starvation is the single critical fault: anything the agent cannot recover from (a shelf tool that wrecks every body, a decaying brain, runaway loops) drains its energy. Nothing is authored.

| Successor gets | | Why |
|---|---|---|
| weights | none; born from θ₀ (decision 1) | "another human" |
| testament | ≤ 1,000 tokens by the dying agent from the autopsy (energy curve, largest drains, last 3 days) + a ≤ 300-token host autopsy; always in context, distilled on its first nights | the PI's summary of critical experience; half a screenshot, cheap to carry for life |
| library: lessons, principles, tool cards, shelf | marked "inherited, unverified" | culture |
| casebook: cheapest verified solution per solved task | studied on its first nights by off-policy context distillation | fastest honest route back [U: about a week of nights] |
| bodies | fresh stock VMs; old ones archived | the fault may live in them |
| energy, stream | born full; continues where the parent died | viable; time does not rewind |

The record keeps every generation (parent, cause, testament, library version); results are reported per individual and per lineage.

### 7. Reward, energy, anti-gaming

- E(attempt) = generated tokens + 0.1 × uncached prompt tokens; a screenshot ≈ 204. Decode dominates GPU time; tunable: 0.1 becomes the prefill/decode GPU-time ratio measured in M0. VM time is uncharged (≈ 1% of a GPU's power).
- Income R once per solved delivery, R = 2 × newborn energy per delivery ÷ newborn delivery-solve rate (census, Laplace-smoothed): a newborn earns twice its spending, and a sustained halving of competence is fatal. Re-set per environment by its census.
- Reservoir B ← min(B_max, B + income − E); B_max = two newborn days of spending; born full; shown in the prompt.
- Help (`ask_human`, external AI) costs one average attempt, so it is used when it beats another try.
- A mistake costs energy without income, a negative advantage, lost files or bodies. The learning reward keeps success ≥ 0.8 above failure; cheaper wins only among successes; quitting never pays.

| Threat | Defence |
|---|---|
| answer key online | a transparent host proxy the VM cannot bypass logs all traffic, blocks the CADWorld repository, forks, raw and codeload paths and the HF dataset, and scans responses for task ids and evaluator names; a hit voids the deliveries and is reported |
| probing the grader | ≤ 3 submissions per body per delivery; reports cover only the task just tried |
| reusing its old output on a repeat | allowed, as for an assistant; file hash detected and reported; first-try metric unaffected |
| humans, other AIs | allowed, charged, logged; helped deliveries flagged and reported both ways |
| hidden compute | tokens counted by our server; host services unreachable from the VM |
| tampering with scoring | grader, rules, references and record live on the host; the VM only supplies the file |

### 8. Measurement

| Metric | Definition |
|---|---|
| First-try success (primary) | mean over the first attempts of each task's first delivery, by life position; no feedback on that task exists yet |
| Delivery success | any submission passes |
| Retention, forgetting | success on repeats of solved tasks at 1, 4, 16 days; drop from each task's best |
| Cost | energy per solved delivery, by token type |
| Drift, internalization, honesty | scan vs θ₀; lessons retired and twin test; assisted and reuse shares |

| Comparison | What | Lives without a revert by |
|---|---|---|
| Notebook | the agent minus weight learning: same memory, reflection, tools, bodies, stream, feedback, energy | its own never-reset life on the agent's calendar; interleaved days on 2 GPUs, side by side on 4 |
| Newborns | base model, same tools, no memory, a stock VM per task; 4 attempts + ≤ 2 resubmissions | each lives one task: a census of innate ability, setting difficulty, R, the order and the contamination floor |
| Twin (milestones) | a copy of the current brain in stock VMs, no memory or tools, on every task the agent solved | a separate individual, archived, never merged back; isolates what the weights carry |

Test: success ~ agent × half + (1|task), task-cluster bootstrap. Tasks 101–200 × 4 first attempts detect ≈ 10 pp at 80% power (per-task paired SD ≈ 0.34). With one life per agent, claims are about these lives; a second agent life (M5) measures the spread between lives.

### 9. Stream

- Order fixed before birth (seed 2026), the same for both lives: the first 24 new tasks from the census's easiest third, because early successes seed learning; the other 176 stratified by category and census difficulty, so the first-try curve is not a difficulty curve.
- Repeats at +1, +4, +16 days (×4 spacing: overnight, short, long; 16 ≈ the first pass), then every 16; steady state ≈ 48 deliveries a day.
- After the first pass plus 16 days: a second environment (Terminal-Bench 2.0 for tool building or OSWorld-Verified for GUI breadth, chosen at M3), with CADWorld at 10% of deliveries. Each environment gets its own census and day length; tasks that bring their own machine run there while the home persists.
- Then your tasks, by chat: you grade, and confirming requirements first is a learnable habit.
- First successes: shell and Python beside the GUI (terminal-only models reach 6–16% [V]); 12 tries per new task with reports; an easy-first childhood; reading; asking you; next-day repeats. Gate: the census solves ≥ 10 tasks, or we fix the harness before birth.

### 10. Build

| Need | Reuse | We add |
|---|---|---|
| tasks, grader, report | CADWorld `e5d0eba` | runner without revert; named `/storage` |
| serving | vLLM 0.30 [V] | two brains as base + LoRA; priorities; `prompt_logprobs` teacher |
| training | PEFT, transformers, flash-linear-attention [V]; losses from verl v0.9 (GRPO, top-K distillation; notes Qwen3.x GatedDeltaNet [V]), SDPO, OpenClaw-RL | ≈ 500-LOC night trainer, 2-GPU DDP, fp32 merge |
| memory | Letta/MemGPT self-editing tiers, mem0 extraction, ReasoningBank schema; Qwen3-Embedding + BM25 | ≈ 400-LOC store |
| proxy, probes | mitmproxy; lm-evaluation-harness, VLMEvalKit | blocklist add-on; frozen items |
| not adopted | verl whole, unless M0 shows it faster here; OpenClaw-RL's multi-node slime/Megatron; Agent Lightning v1.0, coding agents on Kubernetes [V] | |

We write ≈ 2.5k LOC: life loop (calendar, stream, trial, reservoir, succession), home manager, step loop, reflection, trainer, memory, measurement.

| Milestone | Go if |
|---|---|
| M0, build, 3 weeks | 50 tasks back-to-back in one home, each output re-scored identically by the stock evaluator; a home survives a container restart; canary passes; ≥ 40 episodes/h; proxy blocks test URLs |
| M1, newborn census, 2 days | ≥ 10 tasks solved; sets R, B_max, order, caps, thinking effort |
| M2, birth and life days 1–5, 1 week | trial and merge work; drift within budget; no infrastructure deaths |
| M3, CADWorld life (33 life days) and Notebook, ≈ 3 weeks | agent − Notebook ≥ 5 pp first-try on tasks 101–200 (the paper needs the CI); drift ≤ 2 pp; retention ≥ best − 5 pp; twin beats newborns on solved tasks |
| M4, second environment, your tasks | CADWorld retention ≥ best − 5 pp after 1,000 foreign episodes; the gap holds on the new first pass |
| M5, 4 GPUs | a second agent life replicates M3 |

Compute: 40–60 episodes/h on 2 GPUs [U], ≈ 170 episodes a day. On 2 GPUs both serve by day and train by night; the agent's life takes ≈ 9 calendar days, plus ≈ 6 for Notebook interleaved. On 4 GPUs each life gets 2 and they run side by side, ≈ 10 days.

### 11. Risks and decisions

| Risk | Signal | Pivot |
|---|---|---|
| too few first successes | census < 10 tasks | clearer tool descriptions at birth, higher thinking effort; 3 demonstrations from you, excluded from the primary metric |
| weights add nothing over notes | agent ≈ Notebook at M3 | principle-only distillation, more replay; else report that the best agent is note-driven |
| collapse under repeated updates | rejections accumulate, repeats fall | more off-policy distillation and replay, lower lr; the trial already blocks damage |
| homes clutter | step counts rise on repeats | nothing is reset; home health reported for both lives |
| answer-key leak via mirrors | proxy signature hits | void deliveries, extend the blocklist |
| harm online | proxy log | the PI ends the life; contact rules (decision 2) |
| throughput < 40 episodes/h | M0 | 8 new tasks a day; 2 attempts per new task |

Conflicts with PI wishes and the least harmful fix: permanent death vs months of learning → schooling from the library; no undo vs safe self-change → trial and half merge, never rollback; open internet vs a meaningful benchmark and third parties → logged proxy, contact rules, end of life for conduct; "benchmark as is" vs our tools, homes, feedback and repeats → these are not leaderboard numbers; the "Use GUI" instruction stays and the tool mix is reported.

**Decisions for the PI**

1. **Successor inheritance:** (a) original model + testament + library + casebook schooling, recommended, losing about a week and honouring "another human"; (b) testament only; (c) the parent's brain + testament, a continuation rather than a new individual.
2. **Human contact:** recommended, you and named lab members answer ≤ 5 questions a day (about 15 minutes), as mentors who never reveal the grader; public posting with an AI disclosure and a daily cap; other AI services allowed, charged and flagged.
3. **Grader report:** (a) pass/fail and failure class; (b) plus failed checks with observed and expected values, recommended, the most learning per attempt while revealing only this task's rules after it was tried; (c) the full report.
