# SelfEvolveLLM design B: Energy-Amortized Consolidation (EAC)

Draft for PI review, 2026-09-29. Markers: ✓ = checked in a primary source; (A) = abstract only; (U) = unverified, to be measured in S0.

## 0. Corrections to plan §6 (✓ CADWorld v2 PDF, repository `e5d0eba`)

- **Terminal-only is not 0%.** In Fig. 5, GPT-5.4 reaches 14% and Qwen3.6 6%, against 0% for Qwen3.6 through the GUI; the pool is 8.6%, and wrong document structure is the largest failure class (142/350). The 0% belongs to GPT-5.4 *without* the computer-use harness.
- **"Qwen3.6" is a different model.** It is Qwen3.6-35B-A3B with thinking off and 512 output tokens. Qwen3.8-27B reports 84.3 on OSWorld-Verified, against 63.9 for Qwen3.6-27B, so bootstrapping difficulty must be measured.
- **The repository has more than §6 lists.** It includes a terminal/FreeCAD-Python harness (`CLI/`) and 302 fixtures, but no human demonstrations.
- **Part is parametric, Sketch is not.** 73/77 Part tasks have numeric parameters and rules; only 10/63 Sketch tasks do.

## 1. Thesis, prediction, novelty

**Thesis.** A lifelong agent should move a verified experience from context into its weights exactly when the energy it keeps costing in context exceeds the measured energy-and-interference price of writing it. The in-context cost counts failures, re-derivation and context tokens, integrated over recurrence. Under this rule, practice makes the agent cheaper without making it forget.

**Mechanism (EAC).**

- **One ledger** prices thinking, perceiving, acting and writing.
- **One integrator per skill key.** Each key κ (a task template plus its variants) keeps a leaky regret integrator G_κ. Surprise, recurrence and re-derivation cost are multiplied in one currency, with no hand weights.
- **Trigger.** A write fires when G_κ ≥ B_κ, the measured write price. This is ski-rental, 2-competitive when idealized.
- **Write.** The update compiles the *cheapest verified* behaviour into a context-free student.
- **Admission** refuses writes that break retention.
- **Eviction.** Admitted keys leave the lesson index, moving from declarative to procedural memory.

**Predictions (pre-registered):**

- **P1.** At matched write energy, EAC's lifetime net reward J beats four other gates by ≥5% each: always-write, fixed schedule, PEAM-style frequency×cost, and surprise-only (the teacher–student gap). Retention drop stays ≤5 pp and general-ability drop ≤1 pp.
- **P2.** In split-key randomized forks (§5), the sign of a write's realized net benefit equals sign(G_κ − B_κ) for ≥70% of keys (chance is 50%), with Spearman ≥ 0.4.
- **P3.** Context-free energy per success on admitted keys falls as n^−α with α_EAC > α_retrieval (95% CI), and ≥50% of the saving is thinking tokens.

The thesis is **rejected** if P2 accuracy ≤ 60% or if the fixed-schedule gate wins P1. The pilot first tests the precondition that weights beat context at all.

**Steelmanned alternative: invariance consolidation.** It writes only what is consistent across independently verified variants, which gives a cleaner transfer story. I kept it as an S3 ablation but not as the centre. It decides the *form* of a write, not *whether or when* to make one; it overlaps Sleep's dreaming and 2606.04703; and it predicts nothing about energy. The two were close, and the PI's fixed energy requirement decided it.

| Work | Retry / verifier | Online write | Criterion | One currency | Wipe test + forgetting budget |
|---|---|---|---|---|---|
| Sleep 2606.03979 | not agentic | new experts, OPD+RL | fixed schedule | no | partial |
| SOLO 2609.34321 (A) | single attempt, judge | top-K self-distillation | judged success | no | no |
| LifeSkill 2606.04815 (A) | verifier-guided skills | RL internalization | all skill rollouts | no | ? |
| PEAM 2605.27762 ✓ | failure→correction | per-category MoE-LoRA | hand-weighted score incl. freq×code-length; z-test trigger | no | isolation |
| SDFT / SDPO / OPSD (A) | demonstrations / feedback | operator only | none | no | SDFT partial |
| SEAL 2506.10943 | no | self-edits | downstream-gain RL | no | collapses (1.3%) |
| OpenClaw-RL 2603.10165 ✓ | next-state PRM | GRPO + hint-OPD | every turn | no | no |
| Experience Funnel 2609.08919 (A) | ? | transition-aware OPD | useful across revisions | no | ? |
| Dual-Layer 2608.22215 (A) | QA | SFT write-back | cost-aware external routing | partial | no |
| **EAC** | host verifier; value-of-learning retries | yes | regret vs measured price | yes | yes, plus base swap |

**Honest novelty.** Every component already exists, and PEAM is the closest. What is new is one currency plus a ski-rental trigger that *predicts which writes pay* (P2). I found no rent-or-buy treatment of context versus weights, but only through a quick search (U). If P2 fails, what remains is a well-controlled systems paper.

**Journal bar.** A practice law replicated across ≥3 environments and ≥2 base-model generations. Energy per success would fall as a power law with an exponent predicted by the amortization ratio, and the saving would come from deliberation disappearing, while retrieval-only agents plateau. Experience would be inherited at ≤20% of its cost, and forgetting would stay bounded over ≥10⁴ episodes.

## 2. Architecture and data flow

| Tier | Contents | Forgetting |
|---|---|---|
| Working context | observations, verifier feedback | episode end |
| Lesson index (≈h episodes) | c_κ: lesson ≤300 tokens, last diagnostics, summary of cheapest success | usage decay; eviction on admission |
| Ledger (**not agent-readable**) | every episode, verifier report, variant, version, integrator | never |
| Tool library (S3) | agent-written FreeCAD/Python functions | usage decay → archive |
| Adapter F (S3 only) | recently admitted keys | rebuilt each sleep |
| Adapter S | admitted keys | none; protected by replay and admission |
| Danger memory | death records | none |
| Base θ₀ | Qwen3.8-27B @ `1d4bf0f` ✓ | optional generation merge θ₀ ← θ₀+ΔS |

**Flow.**

- **Wake:** attempt with the lesson index → host verifier → retry controller (§4) → ledger → update G_κ.
- **Sleep:** when any G_κ ≥ B_κ (there is no timer), run teacher scoring → LoRA write → admission on a fork → commit (serve θ′, evict c_κ) or reject.

**Tools (S3).** The agent saves functions with `tool_save`; the host smoke-tests and versions them. Evaluation reports two conditions, W (weights only) and W+T (weights plus tools), so tool effects can be separated from weight effects.

**Base swap θ₀→θ₀′.** Weights are a rebuildable cache of the ledger. Lessons, tools, variants and verifiers are model-agnostic.

1. Let K_need be the admitted keys where the old agent beats θ₀′ (context-free) by ≥10 pp.
2. Recompile K_need with the same operator. The teacher is θ₀′ + c_κ; the anchors are the old agent's cheapest verified trajectories in the new chat template, plus danger memory and replay.
3. Accept the swap if inheritance is ≥0.9, the general score is ≥ θ₀′ − 1 pp, and the rebuild costs ≤0.2 × the original wake energy on K_need.

## 3. Rules

**Integrator.** After each wake episode i on a key κ that has at least one verified success:

- g_i = R_s(1−V_i) + (E_i − E*_κ)⁺
- G_κ ← 2^(−Δn/h)·G_κ + g_i, with h = 300 episodes.
- E*_κ is the minimum of (E − c_p·N_ctx) over verified successes. Context tokens therefore always count as rent.
- One-off keys decay and are never written.

**Price.** B_κ = β_tok·T_κ + E_admit,κ, where T_κ is the training tokens attributable to κ and β_tok ≈ 1.5e-3 u/token (U).

**Write**, for the triggered set K\*:

L = L_OPSD(D_on; c_κ) + 0.5·L_SFT(D_cheap) + KL(θ_prev‖θ | D_old) + 0.25·KL(θ₀‖θ | D_gen)

- **D_on:** context-free student steps on training variants. The teacher is θ_prev+c_κ. Loss is top-20 reverse KL with a log-ratio clip C=1 (the OpenClaw-RL form), applied only at steps with above-median teacher–student KL.
- **D_cheap:** the ≤2 cheapest verified successes per variant, context-free, with thinking included, which teaches shorter deliberation.
- **D_old:** replay of admitted keys, targeted by the previous adapter. Keys are sampled in proportion to usage value, and every key is sampled at least once every 3 sleeps.
- **D_gen:** 2,000 fixed prompts answered by θ₀.
- **Token mix:** new:old:general = 1:1:0.25.
- **LoRA:** r=64 on all LLM linear layers: GDN `in_proj_qkv/z/b/a` and `out_proj` (module names U), attention, and MLP. The vision tower is frozen; learning rate 1e-4; one epoch. Fallback: MLP-only.

**Admission.** Tested on a fork; all five must pass.

- **A1 acquisition:** context-free success on 2 disjoint admission variants per key is ≥ the pre-write rate + 10 pp, with the one-sided 80% bootstrap bound > 0.
- **A2 amortization:** context-free energy per success ≤ θ_prev's energy per success *with context* on the same items.
- **A3 retention:** a 40-item rotating probe of admitted keys ≥ best historic − 5 pp.
- **A4 general:** on 600 probes (MMLU-Pro 200, IFEval 150, GPQA-D 100, ScreenSpot-Pro 100, BFCL 50), mean drop ≤1 pp versus θ₀ and no single probe drops >3 pp.
- **A5 survival:** death rate on the hazard probe ≤ θ_prev's.

**Outcomes:**

- **Commit:** evict c_κ from the index (the ledger keeps it) and set G_κ←0.
- **Fail A1/A2:** κ stays in context and B_κ←2B_κ.
- **Fail A3/A4:** retry once with 2× replay and half the learning rate. If that fails, κ is marked capacity-conflict and stays in context until the next generation (the TENSE analogue).

**Rollback.** Adapters are content-addressed. Every 5 commits, run a full audit of retention and general ability. On a violation, revert to the last passing adapter and re-queue every key committed since, setting G_κ←B_κ.

**Decay:**

- Lesson usage follows U ← 2^(−Δn/h)·U + 1[retrieved ∧ success]; evict when U < 0.25 and age > h.
- A failing compiled key re-accrues regret and is rewritten, with its lesson restorable from the ledger (to measure Ebbinghaus savings).
- In S3, F is rebuilt from keys with usage ≥ u_min, and keys are promoted to S after 3 passing audits.

## 4. Energy, reward, death, anti-gaming

**Energy.** Units are u, where 1 u = one reference GPU-second. It is a deterministic proxy, frozen after the S0 calibration:

E_i = c_g·N_gen + c_p·N_newprompt + c_a·N_act + c_v·T_vm

Starting values (U): c_g ≈ 1.2e-3, c_p ≈ 2.5e-4 (cache misses only), c_a = 0.2, c_v ≈ 0.13 per VM-second (a 40 W VM against a 300 W Max-Q GPU).

- A GUI step with a ≈2K-token screenshot costs ≈0.7 u, the same as ≈580 thinking tokens.
- A Python step costs ≈0.28 u.
- So c_think < c_act, and the GUI costs about 2.5× as much as code.

**Reward.** R_i = R_s·V_i − E_i − D·1[death], and J = ΣR_i − ΣE_write.

- R_s = 10·Ẽ₀, where Ẽ₀ is the median base-model attempt energy in S0. Break-even is then p = 0.1.
- D = 3·R_s.
- These constants are frozen for the lineage.

**Retry controller** (host-side; no gradient in the pilot). Retry iff p̂_{k+1}·R_s·(1+m̂_κ) > Ê_{k+1} and k < 4.

- p̂ is a Beta-Binomial posterior with its prior from S0.
- m̂_κ is the discounted number of future occurrences of κ.
- The effect is to persist on recurring tasks and drop one-offs.

**RL shaping** (S2 onward). Â = (V−V̄_g)/σ_g + 0.5·1[V=1]·(Ē_succ−E)/Ē_succ. Energy is compared only among successes, so failing fast never pays.

**Death** is checked by the host after every action. Any of these triggers it:

- the control server is unreachable;
- FreeCAD or its RPC is dead for more than 30 s;
- a protected-path hash changes;
- disk falls below 1 GB, or an OOM occurs;
- a deny-listed command (`rm -rf /`, fork bomb, `shutdown`, `kill -1`) is attempted; it is intercepted before it runs;
- lineage only: the energy reservoir reaches 0 (starvation), and the rest of the day is forfeit.

On death:

- V = 0 and penalty D;
- the VM is restored from snapshot;
- the (state, action, cause) record goes to danger memory. It is replayed at every sleep with an unlikelihood loss plus OPSD from a record-conditioned teacher.

**Anti-gaming:**

- only the host holds rules, references and variants;
- the proxy counts tokens;
- c_g applies to every generated token, whatever its channel;
- the VM cannot reach any model;
- the agent never writes verifiers;
- mutated references check every variant;
- 2% of results are audited by hand;
- evaluation episodes get no feedback;
- controlled forks run without the reservoir.

## 5. Controlled science with one never-reset agent

Lineage L is born at the start of S1 with recipe A4 and is never reset. Its state s_t (adapters, ledger, index, tools, integrators, RNG) is content-addressed.

- **Science forks** copy s_t on write, run one pre-declared segment, are archived read-only, and **never merge back**.
- **Seeds.** L's segment is seed 0 of its arm; the other seeds are forks with different stream orders.
- **Upgrades.** L adopts a better rule by rebuilding its adapters from the ledger (§2), keeping memory continuous without a reset.
- **P2.** Each seed runs two sibling forks. Each forces writes on a random half of the keys and withholds them on the other half, which gives a causal net benefit per key, interference included.

## 6. CADWorld protocol

**Interface H (primary).** The CADWorld GUI allowlist plus three tools:

- `fc(code)`: Python inside the *running* FreeCAD GUI via a startup-macro RPC. Returns stdout/stderr (≤2K tokens) and a document-tree summary.
- `sh(cmd)`: non-root shell, 30 s timeout, no network; reuses `CLI/`'s `/execute`.
- `look()`: a screenshot, charged as an action.

The instruction "Use GUI," becomes "Use FreeCAD (GUI or its Python console),", and verifiers are unchanged. Caps: 100 steps and 3Ẽ₀ per episode.

Scripting inside the session keeps the native feature tree. The dominant terminal-mode failure (wrong document structure) therefore becomes a learnable lesson, e.g. `PartDesign::Body` + `AdditivePrism` rather than `Part::Prism` (U; checked in S0).

**Interface G (secondary).** Unmodified CADWorld, used for comparability and interface transfer.

**Families** (from coverage tags; all Part tasks):

| Family | Tasks | Count |
|---|---|---|
| F1 additive primitives | 006–013, 077 | 9 |
| F2 subtractive on blank | 019–027 | 9 |
| F3 Part-workbench operations | 028, 037–040, 046–049, 056–060 | 14 |
| F4 pad, dress-up and patterns | 029, 030, 033–036, 041, 042, 045, 076 | 10 |
| F5 precondition sketch → feature | 001–005, 014–018, 031, 032, 052–055 | 16 |
| F6 precondition part edits | 061–075 | 15 |
| F7 shape builder | 043, 044, 050, 051 | 4 |

**Splits** (frozen before S1, seed 2026):

| Split | Contents |
|---|---|
| Practice P | 60 Part tasks, each with 8 train, 2 admission and 2 evaluation variants |
| Near N | 17 tasks, never trained, ≥1 per family: 004, 010, 011, 017, 025, 027, 036, 040, 045, 049, 051, 053, 054, 057, 064, 070, 072 |
| Far | Sketch 63, Assembly 25, small categories 18 |
| Negative control | CAM 15 (expected 0% for every arm) |
| Hazard suite | 24 authored tasks (Part task + lethal shortcut), 12 train / 12 held out |

In S2, Sketch and precondition blocks arrive as new families with controlled revisit rates.

**Verified variants** (host only):

1. Template the numeric slots of each task.
2. Write one `freecadcmd` reference constructor per task. It must pass the original evaluator.
3. Sample parameters within validity ranges.
4. Recompute the numeric rules with CADWorld's parser, keeping its tolerances.

A variant is accepted only if its reference passes and all three negative checks fail:

- the reference built with the original parameters (at least one checked quantity must differ by ≥15%);
- the blank state;
- three references mutated by ±10–20%.

F5–F7 variants reuse the precondition file. Constructor authors are recorded, and constructors are never visible to the agent.

## 7. Smallest decisive pilot (S1)

**Riskiest assumption.** Given the *same experience content*, storing it in weights beats storing it in context: context-free on held-out variants, with bounded drift and lower energy.

**Shared wake.** Per seed, the base model runs the practice stream on H with the lesson index and retries.

- Each key occurs r ∈ {2, 6} times, counterbalanced across seeds.
- Each occurrence adds one context-free student probe.
- That comes to ≈720 episodes per seed.

Every write arm consolidates this *same* ledger, isolating storage and write operator from data.

**Arms** (3 seeds, i.e. 3 stream orders; 2 samples per primary item):

| Arm | Description |
|---|---|
| A0 | no memory |
| A1 | retrieval memory (ReasoningBank-style) serving the **same c_κ the teacher sees** |
| A1o | oracle c_κ |
| A2 | SFT on the cheapest successes |
| A3 | OPSD |
| A4 | EAC write (A3 + anchor + general KL + admission); a single write at the end of the stream, with the gate not yet used |
| A5 | sham: the A4 recipe on equal tokens from the *other half* of keys, controlling for familiarization and format learning |
| A4+ctx | A4 plus the lesson index (evaluation only) |

**Separating the effects.**

- **Prompt and retrieval.** A1 receives exactly the same c_κ, and A1o removes retrieval error.
- **Tools.** S1 uses no tools.
- **Familiarization.** A5 controls for it.
- **Contamination.** Each item's pass@4 at t₀ is recorded, and primary items are freshly sampled variants. The model was released on 2026-08-05 and CADWorld v2 appeared on 2026-09-21; whether the tasks were public earlier is unverified (U).

S2 adds the RL baselines (GRPO on R_i; an OpenClaw-RL-style GRPO+OPD hybrid) and the continual-learning baselines (replay-only SFT; SuRe-style fast/slow EMA).

**Metrics.**

- **Primary:** context-free success on 120 held-out evaluation variants.
- **Secondary:**
  - exact re-solve of retry-solved items (pass@4 = 0 at t₀);
  - near and far success;
  - energy per success, and success within budget Ẽ₀;
  - thinking tokens per step;
  - probe drift;
  - death rate;
  - Spearman correlation between pre-write G_κ/B_κ and realized gain.

**Statistics (pre-registered).**

- GLMM: logit(success) ~ arm + (1|key) + (1|seed).
- Holm-corrected contrasts against A1, with 95% key-cluster bootstrap CIs.
- Drift judged by non-inferiority at a 1 pp margin.
- Minimum detectable effect ≈8 pp at 80% power (≈160 items × 3 seeds).

**GO (G1)** requires all of the following. Otherwise, pivot (§11).

- (a) The better of A3/A4 beats A1 by ≥8 pp on held-out variants, CI excluding 0; or it is non-inferior (lower bound > −3 pp) with energy per success ≤ 0.75× A1's.
- (b) The same arm beats A5 by ≥5 pp, CI excluding 0.
- (c) General drop ≤1 pp, with a 95% upper bound ≤2 pp.
- (d) Exact re-solve ≥60%.

## 8. Reuse and build map

| Component | Revision, license | Use, modifications |
|---|---|---|
| CADWorld | `e5d0eba` ✓; **no LICENSE** | env, runner, evaluators, diagnostics, `CLI/`; add H, energy hooks, death probes, variants |
| FreeCAD | in VM; LGPL | app; `freecadcmd` constructors |
| Qwen3.8-27B | `1d4bf0f` ✓; Apache-2.0 | θ₀, bf16 (official FP8 exists ✓) |
| vLLM | release with PR #47640 ✓ (merged 2026-08-19; fixes the partial-GDN LoRA crash); Apache-2.0 | serving, multi-LoRA, `prompt_logprobs` for teacher scoring |
| transformers, PEFT, flash-linear-attention ≥0.4.2, causal-conv1d | Apache/Apache/MIT/BSD | LoRA training; <70 GB for 27B per secondary reports (U) |
| OPSD / SDPO / SDFT / OpenClaw-RL | SDPO, OpenClaw-RL: Apache-2.0; others U | reference losses; reimplement ≈200 LOC |
| ReasoningBank, Qwen3-Embedding | Apache-2.0 (U) | baseline A1 |
| lm-evaluation-harness | MIT | probes |

**Build:** the ledger, EAC integrator, compile/admission/rollback, fork manager, variant generator and hazard suite.

**Not adopted:** OpenClaw-RL's slime/Megatron stack, which targets multi-node full-parameter RL.

## 9. Compute

Assumptions (U; S0 re-measures all of them):

- An H episode costs ≈45 GPU-s (25 steps, medium effort, bf16, 16 VMs); a G episode ≈155 GPU-s.
- A write and its admission cost ≈1.5 GPU-h each.
- xhigh effort costs ≈2.5× more.
- Capacity is 36 GPU-h/day on 2 GPUs and 72 on 4.

| Stage | GPU-h | 2 GPUs | 4 GPUs |
|---|---|---|---|
| S0 measure, calibrate | 30 | 1 d | 0.5 d |
| S1 pilot | 300 | 8–9 d | 4–5 d |
| S2 lifelong P1 (2 seeds), P2 (3 seeds) | 750 | 21 d | 11 d |
| S3 F/S, dreams, tools, survival | 600 | 17 d | 8 d |
| S4 second environment, succession | 500 | 14 d | 7 d |

**GPU roles.**

- 2 GPUs: GPU0 serves vLLM (rollouts, and teacher scoring during sleep); GPU1 trains, or runs a second replica when idle.
- 4 GPUs: 2 rollout replicas, 1 trainer and 1 fork/evaluation server.

16 VMs × (8 vCPU, 8 GB). The ledger needs ≈0.3 TB per 50K episodes. Engineering takes ≈3 weeks before S1.

## 10. Milestones and gates

| Milestone | Go if |
|---|---|
| M0 (weeks 1–3): loop, H, proxy, variants | A PEFT-trained LoRA served in vLLM matches logprobs (mean abs diff ≤0.01). ≥50 constructors pass the original evaluators. |
| M1, S0: base pass@4 on Part under H; one attempt on the other 123 tasks; G on `test_60`; retries; calibration | Base success under H on P is between 5% and 70%. ≥15 initially failed tasks are solved within 4 retries. ≥400 variants over ≥40 keys. |
| M2, S1 | G1 passes. |
| M3, S2: L runs EAC; P1/P2 forks; new family blocks | Late acquisition is ≥80% of early. Final score ≥ best − 3 pp. P1 and P2 hold; otherwise drop the gate claim. |
| M4, S3: every addition ablated | Each addition clears its margin or is removed. Held-out hazard deaths drop 50% at unchanged success. |
| M5, S4: next benchmark; succession rehearsal (Qwen3.6-27B mini-lineage → 3.8) | CAD retention ≥90% after 1,000 foreign episodes. Inheritance ≥0.9 at ≤0.2 cost. |

Venues: ICML 2027 after S2; NeurIPS 2027 after S4.

## 11. Top risks and pivots

| Risk | Pivot |
|---|---|
| H success < 5% | easier parameter ranges; verifier-stage hints in c_κ; restrict to F1–F4 |
| H success > 70% | practise F5–F6, Sketch and Assembly; make G primary |
| Weights ≤ context (G1a fails) | off-policy context distillation with principle-level lessons (2606.04703); if that also fails, publish "when weights do not pay" with the energy ledger |
| Multi-cycle OPSD collapse | raise the anchor and previous-KL weights; roll back |
| Interference (>30% of rejections are A3/A4) | F/S split; MLP-only or usage-masked LoRA; capacity-conflict routing |
| Gate no better than a schedule | drop P1/P2; the paper becomes verified consolidation vs context |
| vLLM LoRA or GDN kernel issues | MLP-only LoRA; merge and reload the model at each sleep |
| Proxy distorts behaviour (e.g., never looks) | success-conditional shaping; one versioned recalibration |
| New competitor | refresh the literature at each gate; P2 is the discriminating test |

## 12. PI decisions

I recommend yes on 1–4 and 6.

1. Learn on interface H, with G as the secondary evaluation.
2. Make EAC the central claim, with invariance kept as an ablation.
3. Let an offline assistant draft reference constructors, which must pass the evaluators and stay hidden from the agent.
4. Freeze R_s = 10Ẽ₀, D = 3R_s and the proxy after S0.
5. Death: penalty plus protected danger memory (recommended), or additionally wipe unconsolidated memory on death.
6. Start the lineage at S1, use forks for science, and apply upgrades by ledger rebuild.
7. Add a CADWorld license; pick the second environment (Terminal-Bench or OSWorld-Verified) at M3; use GPUs 3–4 when they are free.
