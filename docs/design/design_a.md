# Design A: Amortized Residual Consolidation (ARC)

2026-09-29, for PI review. "Verified" means checked today against the paper, the repository at `e5d0eba`, the model card or the issue tracker. **[U]** marks claims I could not verify.

## 0. Verified facts that change the plan

1. **Terminal-only is not 0%.** On a 50-task stratified subset, terminal-only agents succeed on 8.6% overall: Opus 4.8 16%, GPT-5.4 14% and Qwen3.6 6% (paper v2, Fig. 5). Most failures are F5, "wrong document structure". The 0% belongs to GPT-5.4 without its computer-use harness. Fix plan §6.
2. **Qwen3.8-27B:** HF sha `1d4bf0f`, `Qwen3_5ForConditionalGeneration`, 48 Gated DeltaNet + 16 gated-attention layers, one MTP layer. The card claims 84.3% on OSWorld-Verified, so near-zero CADWorld success is an inference from Qwen3.6 and must be measured.
3. **vLLM LoRA on this family has two known failures.** PEFT adapters saved under transformers 5 can load and silently do nothing (their keys lack the `language_model` prefix). A partial LoRA on the packed GDN group (`in_proj_qkv`/`in_proj_z`) crashes vLLM 0.21–0.24; PR #47640 fixes it.
4. **Tasks:** every instruction starts "Use GUI,". 116 tasks share numeric parameters between the instruction and the evaluator (Part 69, Sketch 39), so parametric variants are feasible. About 45 of them also check derived quantities (volume, bounding box, centre of mass), which a variant must recompute.
5. **PEAM is the nearest mechanism.** Its worthiness score is call frequency × code length + stability − redundancy − a forgetting proxy, with grid-searched weights, and it triggers on a failure-rate z-test.

## 1. Thesis, prediction, novelty

**Thesis.** A lifelong agent should write an experience into its weights once the energy it keeps paying for not having it there exceeds the price of writing it. That cost is the leaky-integrated, verified residual between its experience-conditioned self and its context-free self. This amortization rule, ARC, yields more verified success per unit of lifetime energy, with bounded forgetting, than retrieval, writing everything, or heuristic gating.

**Mechanism.** ARC is rent-or-buy (ski-rental) consolidation over one energy ledger.
- **Rent** is the verified residual per use.
- **Price** is training plus admission-test energy plus predicted interference.
- The same inequality decides retries, sleep, consolidation, eviction and survival learning.

Two published laws make the ledger computable before any write:
- the teacher–student gap predicts self-distillation gain linearly (2605.30070);
- forgetting scales with the KL divergence to the pre-update policy (RL's Razor, 2509.04259).

With known costs and no leak, buying once the rent reaches the price costs at most 2× the hindsight optimum.

**Pre-registered predictions** (on the P2 stream):
- **H1 (economy):** at the calibrated price, ARC's lifetime energy per verified success (EPS, including sleep) is at most 0.8 × the better of retrieval-only and consolidate-all. The upper bound of the 95% CI of the ratio must be below 1.
- **H2 (price law):** with the price of writing ×0.1, ×1 and ×10, ARC is within 5% of the better baseline at the extremes and beats both at ×1. The occurrence at which a family is first consolidated scales with price, with a log–log slope between 0.5 and 1.5.
- **H3 (competence):** after a memory wipe, success on families with at least 8 occurrences is at least consolidate-all's minus 3 pp, and forgetting stays within budget.

**Falsified if** H1 fails, or if a PEAM-style gate or a count-only gate matches ARC at all three prices. Then pricing adds nothing beyond a recurrence heuristic.

| Work | Already does | ARC adds |
|---|---|---|
| Sleep 2606.03979 | distillation into new experts; RL dreaming; fixed schedule; not agentic | measured-energy when/what/forget; verifier; eviction |
| SOLO 2609.34321 | recurring GUI streams; judge; one attempt; windowed self-distillation | verifier retries; long-term replay; price gate |
| LifeSkill 2606.04815 | verifier-rewarded skills; online internalization | gating economics; forgetting; energy |
| PEAM 2605.27762 | tuned worthiness score and z-trigger; isolated per-category LoRA | measured prices instead of tuned weights; competitive bound; KL-priced interference; price-shift test |
| SDFT/SDPO/OPSD, OpenClaw-RL | write operators; clipped hint-conditioned OPD | reused, not claimed |
| SEAL 2506.10943 | RL-trained self-edits | not claimed |

**Not new:** self-distillation, replay, admission gates, LoRA and sleep.

**Journal bar:** the same prices and rule, across at least 3 domains and 2 base models, must:
- predict when consolidation happens;
- yield a power law of practice in energy per success;
- yield a double dissociation between declarative and procedural memory under lesions;
- yield one-trial learning of lethal events;
- survive a base-model swap.

## 2. Architecture and data flow

```
seeded stream -> scheduler -> agent = base (+) S --actions--> CADWorld VM (GUI + shell + FreeCAD-Py)
                  context: task, screenshots, energy B_t,        | .FCStd, VM health
                  [experience c_u], [tools]                      v
episode store <-- trajectories, lessons, diagnostics <-- host verifier V ; energy meter E ; watchdog D
     |
ARC ledger (A_u, C_u) --sleep--> compile Delta (train GPU) --> admission
                         accept: S <- S (+) Delta, hot-swap | reject: rollback, bisect, C_u x2
```

| Timescale | Store | Forgetting |
|---|---|---|
| short | episode context | cleared per episode |
| medium | episodes, lessons, tools (`/home/user/.agent`, host-synced) | evict an episode once its family is consolidated and verified, or after 1,000 episodes unused; archive unused tools |
| long | LoRA `S` (rank 64) | replay eviction (R10), then natural decay |
| near-permanent | base | changed only by a PI-approved swap or merge |
| portable | experience genome `G` | never deleted |

**Base-model swap.** `G` holds templates, verified variants and trajectories, lessons, diagnostics, tools, the ledger and retention panels; weights are a compiled cache of `G`.

To swap:
1. Test the new base context-free on each consolidated family.
2. Recompute the gaps; families the new base already solves have no rent and are skipped.
3. Re-compile the rest with the same operator. The old trunk may act as an extra logit teacher if the tokenizers match.
4. Accept if retention is at least the old trunk's minus 3 pp and the general suite is at least the new base's minus 1 pp.

## 3. Exact rules

**Notation.**
- `u` is a family (a template plus its variants); `k` is an occurrence.
- Teacher `π_T` = base ⊕ S plus experience `c_u`: the best verified trajectory, a principle-level lesson and past failure diagnostics. Student `π_S` = the same weights without `c_u`.
- `p̂⁰_u` and `p̂ᵀ_u` are Beta posteriors of context-free and teacher-mode success; the gap is `g_u = p̂ᵀ_u − p̂⁰_u`.
- `KL̂_u` is the mean top-k KL(π_T‖π_S) on u's recent states.
- Energy is in units of `E_ref`.

**Rules.**
- **R1 (rent):** `ρ_{u,k} = max{0, [R_s·p̂ʷ_u − Êʷ_u] − [R_s·V_{u,k} − E_{u,k}]}`, where `p̂ʷ_u = p̂⁰_u + κ·g_u`.
  - `Êʷ_u` is the median energy of u's verified successes, excluding memory tokens.
  - `κ` is fitted online (prior 0.5).
- **R2 (leaky account):** `A_u ← λ^{Δt}·A_u + ρ_{u,k}`, with `λ = 2^{−1/300}` per episode. This is the PI's fast-weight persistence criterion, applied in the ledger rather than in the weights.
- **R3 (price):** `C_u = c_gpu·ŝ_u + c_adm/|P| + ι·KL̂_u`. `ŝ_u` is the GPU-seconds for u's samples; `ι` is regressed from measured retention loss.
- **R4 (sleep):** the due set is `P = {u : A_u ≥ C_u}`. Sleep when `Σ_{u∈P}(A_u − C_u) ≥ C_fix`, or when a due family has waited more than 150 episodes.
- **R5 (retry):** at most 4 attempts per occurrence, each capped at `E_cap`.
  - Attempt 1 is context-free if u is consolidated or `p̂⁰_u ≥ p̂ᵀ_u − 0.1`; otherwise it runs in teacher mode.
  - Continue while `p̂ᵀ_u·(R_s + F_u) > Ê_next`. `F_u = n̂_u·ρ̄_u` (leaky occurrence count × mean rent) until u first succeeds, then 0.
- **R6 (write, "compile"):** `L = Σ_{S_on} −clip(log π_T(y) − log π_S(y), ±2)·log π_θ(y) + 0.5·CE_{S_off}(a*) + 0.1·KL(π_θ‖π_base)`, the last term on general prompts.
  - `S_on` is the student's own steps with top-k KL ≥ 0.1 nat. In failed episodes it stops at the first step where `π_T(student action) < 0.1`.
  - `S_off` is the verified steps the student never reached, with context removed.
  - LoRA r = 64 on MLP plus o_proj/out_proj, all layers. This avoids a partial packed-group LoRA. The ViT is frozen; lr 5e-5.
- **R7 (replay):** 50% due families, 30% consolidated families (weighted by holding value), 20% general anchor. All are re-scored by the current teacher.
- **R8 (admission):** `S ⊕ Δ` must pass all four tests.
  - **Acquisition:** each due family solves at least 1 of 2 new in-range variants context-free, or its posterior mean rises by at least 0.15. A family that fails is dropped and its `C_u` doubles.
  - **Retention:** 16 consolidated families lose no more than 2 successes. Each family's verified-trajectory log-likelihood drops by at most 10%.
  - **General:** a 500-item probe drops by at most 1 pp per sleep and 2 pp cumulatively.
  - **Parity:** the §8 canary passes.
- **R9 (rollback):** if retention, general or parity fails, keep `S` and retry the lower-`KL̂` half of `P` at the next sleep. A family rejected twice stays episodic-only (TENSE-style) for 3 more occurrences.
- **R10 (explicit forgetting):** a consolidated family's realized savings accrue in a leaky account `Σ_v`. Drop `v` from replay when `Σ_v` is below its replay GPU-cost over the half-life. Lethal-hazard families accrue `R_D` per recurrence and are therefore kept.

## 4. Energy, death, anti-gaming

**Energy** (in decode-token equivalents): `E_i = N_think + N_out + c_in·N_in^new + c_kv·N_in^cached + c_act·N_act + c_env·T_vm + c_cpu·CPU_tool`.
- Prices come from a day-0 NVML calibration and are then frozen.
- `c_think < c_act` holds per unit; expect `c_act` ≈ 10–50 [U].

**Reward.** `r_i = R_s·V_i − E_i/E_ref − R_D·D_i`, with `R_s = 5`, `R_D = 10`, and `E_ref` = the median day-0 successful-episode energy.
- `E_cap = 4·E_ref` ends an attempt, and the verifier scores whatever was saved.
- Any success earns at least +1; any failure scores below 0.

**Survival reservoir.** `B_{t+1} = min(25, B_t + r_t)`, with sleep energy charged. The prompt shows "energy x%".

**Death.** `D = 1` when any of these occur:
- VM, X or FreeCAD is unrecoverable, or the VM shuts down;
- anything under `/home/user/.agent` is written or deleted;
- disk use exceeds 95%, or an OOM or fork storm occurs;
- `B ≤ 0`.

A death costs `R_D`, loses unsynced tools and notes, restores the VM snapshot, and resets `B` to 10.

**Hazards.** 5% of stream episodes are verifiable hazard probes, such as "free disk space in /home/user". Held-out probe types test transfer. ARC predicts one-trial consolidation of lethal events, because their rent `ρ = R_D` exceeds the typical price `C_u`.

**Anti-gaming.**
1. The energy term enters gradients only in groups with at least one verified success, so the policy cannot learn to quit.
2. The verifier, evaluator JSONs, ledger, prices, scheduler and reward code live on the host. The agent cannot write them or choose tasks.
3. Token counts come from vLLM and timings from the host; tool CPU time is charged.
4. No internet in the pilot. Later, a logged proxy blocks CADWorld's GitHub, site and Hugging Face assets, and any hit voids the episode.
5. Variants come only from trusted templates; agent "dreams" may choose parameters, never evaluators.
6. Evaluations assert an empty store, no tools, the day-0 prompt hash and hashed held-out items.
7. All held-out successes and 5% of stream successes are re-scored (`reevaluate.py`) and spot-checked for gaming.

## 5. One never-reset agent with controlled comparisons

**State.** Σ = (base sha, `S`, store, tools, ledger, `B`, RNG, VM snapshot, commit), content-addressed.

**Trunk.** The never-reset agent; it lives the default arm, seed 0, of each experiment. A rejected sleep rolls back an update, never the agent's life.

**Branches.** Copy-on-write forks of a trunk snapshot that run other arms or seeds on the same seeded stream. They never merge back. Only the PI can adopt a new method into the trunk, at a snapshot boundary.

**Pairing.** Arms share the snapshot, the stream seed (order and variant parameters) and the evaluation items. Use at least 3 seeds; P1 uses 2.

**Isolating the weights' effect.**
- A post-wipe lesion matrix of {weights, retrieval, tools} × {on, off} at each milestone.
- A sham adapter (same compute, other families' data), which isolates generic harness fluency.
- The day-0 census as the contamination baseline.

**Statistics.** Hierarchical bootstrap (family → seed → episode); mixed-effects logistic regression `success ~ arm + (1|family) + (1|seed)`; EPS as a ratio of sums; Holm correction over H1–H3. The protocol file is hashed before P1.

## 6. CADWorld protocol

**Interface (recommended).** The trunk gets GUI (`pyautogui`), plus `shell(cmd)` through the VM server's `/execute`, plus FreeCAD Python, plus tools. The "Use GUI," prefix is removed.

Why:
- the agent is general;
- scripting already scores above zero;
- tools need an execution substrate;
- the verifier checks native structure (F5), so scripting is no shortcut.

A fixed 40-task GUI-only panel at each milestone provides CADWorld-comparable numbers and tests transfer from script to GUI.

**Splits** (frozen before the census; stratified; counts from a seed-0 simulation):

| Set | Tasks | Use |
|---|---|---|
| F far | Assembly 25, FEM 3, Mesh 3, Cloudpoint 3, TechDraw 2 | evaluation only until M5 |
| Z frontier | CAM 15 (0% for every published agent) | practised from Stage 3 |
| T train | 110 (Part 56, Sketch 48, other 6); 86 parametric | stream families |
| N near | 39, each sharing a non-generic coverage tag with T | evaluation only |

**Verified variants.**
- **Q (parametric).** A template has instruction slots with training and extrapolation ranges, evaluator slots, and a trusted FreeCAD builder.
  - Derived expectations come from CADWorld's own getters run on the builder's artifact.
  - **Per variant:** the builder's artifact passes; the original reference fails; a +10% perturbation fails; the shape is valid.
  - **Per template:** the builder at the original parameters passes the original evaluator.
- **P (perceptual).** Same evaluator, with screen resolution {1920×1080, 1600×900, 1366×768}, a light or dark stylesheet and a toolbar reset.
- **C (compositional, Stage 3).** Two chained templates with a joint evaluator.

**Authoring.** The team writes about 56 templates, optionally from offline coding-model drafts reviewed by a human; the agent never does.

## 7. Smallest decisive pilot

**P0 census.** All 200 tasks × 4 samples on the hybrid interface, plus 60 GUI-only tasks × 2. A family is "learnable" if pass@4 > 0, or if it is solved within 4 teacher-mode retries with diagnostics.

**P1: one wake, many sleeps** (does any weight write work? 2 seeds). The trunk (base + retrieval + diagnostics + retries) plays 24 learnable families × 4 variant occurrences, 25% of them perceptual; about 250 episodes. All arms train on this identical log:

| Arm | Description |
|---|---|
| A0 | base, no memory |
| A1 | retrieval in context, at equal inference-token budget |
| W1 | SFT on verified successes |
| W2 | off-policy logit context distillation |
| W3 | compile (R6) + replay |
| SH | sham |

Evaluation after the wipe:

| Set | Contents | Samples |
|---|---|---|
| E1 | trained variants | ×2 |
| E2 | new in-range variants | ×2 |
| E3 | extrapolation | ×1 |
| E4 | perceptual | ×1 |
| E5 | N siblings | ×2 |
| E6 | Assembly | ×1 |
| E7 | static suite: MMLU-Pro, GPQA-D, IFEval, MMMU and ScreenSpot-Pro subsets | — |

**G1** (all must hold):
- On E2+E4+E5, the best W arm is at least A1 − 5 pp with EPS ≤ 0.8 × A1's, or it is at least A1 + 5 pp.
- On E1+E2, the best W arm beats SH by at least 10 pp.
- Drift is at most 2 pp.
- E6 is at least A0 − 3 pp.

**P2: ARC online** (3 seeds, branched from the trunk after P1).

*Stream.* 32 new families with Zipf recurrence (4×16, 4×8, 8×4, 8×2, 8×1; 152 occurrences) plus 8 hazard probes. The P1 families serve as retention probes.

*Arms.*
- A1 retrieval-only;
- A4 consolidate-all: best W with replay and admission, sleeping every 24 occurrences;
- ARC ×1;
- price sweep: ARC ×0.1 and ×10. The non-adaptive arms are re-priced from their logs, so they need no new runs.

*Metrics.* EPS; cumulative reward; post-wipe success by recurrence bin; P1 retention; drift; death rate; first-consolidation index against price. **G2** = H1–H3.

**Stage-2 arms** (4 GPUs, or after G2): PEAM-PV gate; count-only gate (k = 2); GRPO-LoRA with the energy reward at an equal episode budget; SOLO-style single-attempt self-distillation.

**Stage-3 ablations**, one at a time:
- no leak;
- gap-only gating;
- no KL price;
- no replay;
- no anchor;
- a fast LoRA (kept only if it cuts time to first context-free success by ≥ 25% without extra forgetting);
- dreaming;
- decay-to-base;
- Fisher/usage protection;
- tools off.

**Power.** About 144 E2+E4+E5 episodes per arm per seed. With a design effect of about 2, a 10 pp difference near a 20% base rate is detectable at 80% power (approximate).

## 8. Reuse/build map

| Component | Source (license) | Change |
|---|---|---|
| Environment and verifier | CADWorld `e5d0eba` (no license file) | shell tool, P-variant setup, variant JSONs, watchdog, energy hooks |
| Serving | vLLM with PR #47640 (Apache-2.0) | runtime LoRA loading; `prompt_logprobs` for teacher scoring |
| Training | transformers ≥ 5.9, PEFT, flash-linear-attention ≥ 0.4.2, causal-conv1d | compile loss; canary |
| RL baseline | ms-swift (Apache-2.0; Qwen3.5 GRPO/GKD; 27B [U]) or TRL | config only |
| Loss code | OpenClaw-RL, SDPO (Apache-2.0), OPSD [license U] | port about 200 LOC |
| Retrieval baseline | ReasoningBank schema (Apache-2.0) + Qwen3-Embedding | equal-budget top-k |
| General evaluation | lm-evaluation-harness (MIT), VLMEvalKit (Apache-2.0) | frozen subsets |
| **Build** | ledger, branch manager, builders, admission tester, meter, PEAM-PV reimplementation | about 2k LOC + 56 builders |

**Parity canary** (after every adapter load):
- mean |Δ log p| between the trainer and vLLM ≤ 0.02 nat on 1k tokens;
- KL(adapter‖base) ≥ 1e-3, which catches silent no-op adapters.

## 9. Compute

**Assumptions** [U; replace with P0 measurements]:
- **Serving:** one bf16 server per GPU drives 8 VMs at about 60 episodes/h (range 30–100). Serving and training need separate GPUs.
- **Training:** an 8k-token LoRA sample plus its teacher pass costs about 2.2 PFLOP, about 11 s at 200 TFLOPS effective (504 TFLOPS dense BF16 peak, verified).
- **Sleep:** about 2 h of training plus 1 h of admission episodes.

| Phase | Episodes | Train GPU-h | 2 GPUs (1 rollout + 1 train) | 4 GPUs (2 + 2) |
|---|---|---|---|---|
| P0 | 920 | 0 | 0.7 d | 0.4 d |
| P1 | 3,100 | 12 | 3 d | 1.5 d |
| P2 + sweep | ~10,000 | ~126 | 11 d | 5 d |
| Pilot total, with overhead | ~14k | ~140 | 3–4 weeks | 1.5–2 weeks |
| Stage 3 (full CADWorld trunk) | ~15k | ~250 | 6–8 weeks | 3–4 weeks |

## 10. Milestones and gates

| Milestone | Weeks (2 GPUs) | Content | Gate |
|---|---|---|---|
| M0 | 1 | serving, hot-swap, training, canary, harness, meter, watchdog, snapshots | canary passes; ≥ 30 episodes/h; VM restore ≤ 60 s |
| M1 | 2 | census; frozen splits and templates | ≥ 40 learnable families, else shrink P1/P2 or pivot R1 |
| M2 | 3–4 | P1 | G1 |
| M3 | 5–8 | P2 + price sweep | G2 (thesis) |
| M4 | 9–16 | all of T, then CAM; Stage-2 arms; ablations | late/early acquisition ≥ 0.8; retention ≥ 0.9; drift ≤ 2 pp; deaths falling |
| M5 | 17–22 | terminal/SWE domain on the same trunk; F far set | CADWorld retention ≥ 0.9; H1/H2 replicate |
| M6 | +4 | base swap through `G` | ≥ old − 3 pp on retention at ≤ 30% of the original training energy |

## 11. Top risks and pivots

| Risk | Signal | Pivot |
|---|---|---|
| R1: too few successes | < 40 learnable families | short tasks and simplified parameters first; higher teacher reasoning effort; earlier second domain |
| R2: LoRA broken on GDN | canary fails; slow fla fallback | MLP-only LoRA; pinned versions; merged-weight evaluation |
| R3: cached context memory is cheaper | ARC rarely buys at ×1 | report the crossover; stress retrieval with a context budget, perceptual variants and composition; reframe as "when does parametric memory pay" |
| R4: collapse over iterations (2606.04703) | acceptance rate or E1 falling | principle-level lessons; more off-policy weight; stricter admission |
| R5: gains are harness fluency | W ≈ SH | say so; the transfer claim rests on W − SH |
| R6: ARC ≈ PEAM-PV at every price | Stage-2 result | drop the pricing claim; a systems contribution (weaker) |
| R7: episodes 2× slower than assumed | P0 | FP8 serving after the canary; 16 families per phase |

## 12. Decisions for the PI

1. **Interface:** hybrid for the trunk, with GUI-only as a fixed evaluation panel?
2. **Claim:** ARC with price-shift validation as the paper's claim, with the write operator chosen in P1?
3. **Constants:** NVML-calibrated frozen prices; `R_s = 5`, `R_D = 10`, `E_cap = 4·E_ref`, `B_max = 25`; training energy charged to the agent?
4. **Forgetting budget:** drift ≤ 2 pp and retention ≥ 90% per consolidated family?
5. **Splits:** F/Z/T/N as above, reporting CADWorld results as transfer and GUI-panel numbers, never as leaderboard scores?
6. **Internet:** none in the pilot, then a logged denylist proxy?
7. **Resources:** GPUs 2–3 when they are free, and about 2 RA-weeks for templates?

**Not verified:**
- Qwen3.8's CADWorld success;
- vLLM LoRA on the 27B multimodal checkpoint;
- flash-linear-attention kernels on sm_120;
- throughput and energy numbers;
- LifeSkill and SOLO beyond their abstracts;
- OpenClaw-RL LoRA support;
- PEAM code;
- the OPSD license.
