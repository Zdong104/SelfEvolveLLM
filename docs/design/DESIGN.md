# SelfEvolveLLM design: rent-or-buy consolidation

-This file is overlapped with the FINDING_AND_PLAN.md, so thinking about remove the redundant and merge those are incremental-


- Status: draft for PI review, 2026-09-29. -DONE- 
- Synthesized from two independent designs, `design_a.md` and `design_b.md`; both are deleted once this is accepted.
- Facts live in `../FINDINGS_AND_PLAN.md` §6.
- [U] marks something unverified, to be measured in M0–M1.

## 1. Thesis and predictions

**Thesis.** A lifelong agent should move a verified experience from context into its weights once the energy it keeps paying for *not* knowing it exceeds the measured price of writing it.
- The cost of not knowing: failures, re-derivation and context tokens.
- The price of writing: training, admission and interference.

The same rule, over one energy ledger, decides retries, sleep, writing and forgetting. Both designs reached this independently.

**Mechanism.**
- A *key* κ is a task template plus its verified variants.
- Each key keeps a leaky rent account G_κ and a price B_κ, and it is written when G_κ ≥ B_κ.
- This is ski-rental, which is 2-competitive in the idealized case.

**Pre-registered predictions:**
-This section is very wierd to me as where you get the number from? What you mean the idea of rent-buy? it does not make sense to me-

-At least those value should give me a reason why you choose those value. I am thinking about model evolve like human do. Either at same time or in series. Model evolve (like advantage over other rollout) and whoever have the better performance adapt (domained but maybe not suddenly change all, some dacay) to the enviroment-
| | Prediction | Test |
|---|---|---|
| H1 | With the *same* experience content, weights beat context. Context-free success on held-out variants is ≥ retrieval + 8 pp, or non-inferior (lower bound > −3 pp) at ≤ 0.75× energy per success. It is also ≥ sham + 5 pp, with general drift ≤ 1 pp. | pilot (G1) |
| H2 | The rule predicts which writes pay. In split-key forks, writes are forced on a random half of keys and withheld on the other half. sign(G_κ − B_κ) must match the sign of the realized net benefit for ≥ 70% of keys, with Spearman ≥ 0.4. | M3 |

-I actuallt like this idea about model decide when to adapt or evolve, but maybe we can define a hyper parameter on this? As like a day is 24 hours and human evolved to do the 'general replay' during sleep time but that is adapted to the enviroment setting-
| H3 | Price law. Scaling the write price ×0.1/×1/×10 moves the first-consolidation occurrence with a log–log slope of 0.5–1.5. At ×1, lifetime energy per success is ≤ 0.8× that of write-all and of retrieval-only. | M3 |
| H4 | Practice law. Context-free energy per success on consolidated keys falls as n^−α, with α_rule > α_retrieval. Most of the saving is thinking tokens. | M3–M4 |

**Falsified if:**
- H2 accuracy is ≤ 60%; or
- a fixed schedule or a PEAM-style heuristic gate matches the rule at all three prices.

If H1 fails, weights do not pay in this domain; see §11.

**Novelty.** Self-distillation, replay, admission gates, LoRA and sleep all exist already. What is new is one measured currency plus a rent-or-buy trigger that predicts which writes pay, tested causally.


-It still did not explain what you mean the rent-or-buy idea is, I need more explaination -
The closest work:
- **PEAM:** a hand-weighted worthiness score, a z-test trigger, and an isolated per-category LoRA.
- **Language Models Need Sleep:** a fixed schedule; not agentic.
- **SOLO / LifeSkill:** they write everything, rewarded by a judge or by skills.

A quick search found no rent-or-buy treatment of context versus weights [U].

**Journal bar.** The same rule and prices, across ≥ 3 environments and ≥ 2 base models, must:
- predict when consolidation happens;
- give a power law of practice in energy per success;
- give one-trial learning of lethal events;
- let experience survive a base-model swap at ≤ 20% of its original cost.


- just adding more context here, I am expecting I can keep interacting with this agent later I ask agent and it finish the work and give to me, like model with life long knowledge and would adapt to my habbit and know and confirm with things I need. Model develop tool and use tool as it need-

## 2. Architecture

-I feel this design missing the reflection step (or called replay) how can the model more efficiently learned from the limited trajectories. like human, how can the model learning more efficiently-

```
stream → scheduler → agent (base ⊕ LoRA S) ──actions──▶ CADWorld VM (GUI + FreeCAD-Python + shell)
          context: task, screenshots, energy, lesson c_κ              │ saved .FCStd
ledger (host) ◀── episode, verifier report, energy ◀── host verifier · energy meter · death watchdog
   │ G_κ ≥ B_κ
sleep: teacher scoring → LoRA write → admission on a fork → commit (hot-swap, evict c_κ) | reject (B_κ ×2)
```

| Tier | Contents | Forgetting |
|---|---|---|
| Working context | observations, verifier feedback | at episode end |
| Lesson index | per key: a lesson of ≤ 300 tokens, the last diagnostics, a summary of the cheapest success | usage decay; evicted once κ is consolidated |
| Ledger (host only) | every episode, verifier report, variant, energy, G/B | never |
| Adapter S (LoRA r = 64) | consolidated keys | protected by replay and admission |
| Danger memory | death records | never |
| Tools (from M4) | agent-written, smoke-tested by the host, versioned | usage decay → archive |
| Base | Qwen3.8-27B @ `1d4bf0f` | only by a PI-approved swap |

- **Fast/slow adapters.** A split into fast and slow adapters is an ablation, not the default.
- **Weights are a compiled cache of the ledger.** A base swap works like this:
  1. Test the new base context-free on each consolidated key.
  2. Recompile only the keys it trails by ≥ 10 pp. Use the same operator, with the old agent's cheapest verified trajectories as anchors.
  3. Accept if inheritance is ≥ 0.9, general ability is ≥ the new base − 1 pp, and the cost is ≤ 0.2× the original wake energy on those keys.

  Rehearse this on a smaller lineage first (M5).

## 3. Rules

Episode i on key κ has verifier outcome V_i ∈ {0,1} and energy E_i (§4).

- **R1, rent.** After each episode on a key with ≥ 1 verified success: g_i = R_s(1 − V_i) + (E_i − E*_κ)⁺.
  - E*_κ is the lowest energy of a verified success, excluding context tokens, so lessons always count as rent.
- **R2, leaky account.** G_κ ← 2^(−Δn/300)·G_κ + g_i, where Δn is the number of episodes since the last update.
  - One-off keys decay and are never written. This is the PI's fast-weight criterion, applied in the ledger.
- **R3, price.** B_κ = E_train,κ + E_admit/|K*| + ι·KL̂_κ.
  - KL̂_κ is the teacher–student top-k KL on κ's recent states; forgetting scales with KL (RL's Razor).
  - ι is fitted online from measured retention loss and starts at 0.
- **R4, sleep.** The due set is K* = {κ : G_κ ≥ B_κ}. Sleep when Σ_{K*}(G_κ − B_κ) ≥ C_fix, the fixed cost of a sleep, or when a due key has waited > 150 episodes.
- **R5, retry.** At most 4 attempts. Continue while p̂_{k+1}·R_s·(1 + m̂_κ) > Ê_{k+1}.
  - p̂ is a Beta posterior; m̂_κ is the discounted number of expected future recurrences.
  - The agent therefore persists on recurring keys and drops one-offs.
- **R6, write.** L = L_OPSD(D_on) + 0.5·L_SFT(D_cheap) + KL(θ_prev‖θ | D_old) + 0.25·KL(θ₀‖θ | D_gen).
  - **D_on:** context-free student rollouts on training variants. The teacher is θ_prev + c_κ. The loss is top-20 reverse KL with a log-ratio clip of 1, at steps with above-median teacher–student KL.
  - **D_cheap:** the ≤ 2 cheapest verified successes per variant, context removed, thinking kept. This teaches shorter deliberation.
  - **D_old:** replay of consolidated keys, weighted by usage value, with each key replayed at least every 3 sleeps.
  - **D_gen:** 2,000 fixed prompts answered by θ₀.
  - The token mix is 1 : 1 : 0.25.
  - The vision tower is frozen.
  - LoRA goes on every language-model linear layer: attention, Gated DeltaNet and MLP. vLLM 0.30 serves this and matches HF+PEFT (plan §6); the parity canary still checks every adapter.
- **R7, admission,** run on a fork. All six must pass:
  1. **Acquisition:** context-free success on 2 unseen variants per key is ≥ the pre-write rate + 10 pp (one-sided 80% bootstrap bound > 0).
  2. **Amortization:** context-free energy per success is ≤ θ_prev's energy per success with context.
  3. **Retention:** a rotating probe of consolidated keys is ≥ the best historic score − 5 pp.
  4. **General:** on a 600-item probe (MMLU-Pro, IFEval, GPQA-D, ScreenSpot-Pro, BFCL), the mean drop is ≤ 1 pp versus θ₀ and no single benchmark drops > 3 pp.
  5. **Survival:** the hazard death rate is ≤ θ_prev's.
  6. **Parity canary:** trainer and vLLM agree to within a mean |Δlog p| of 0.02 nat, and KL(adapter‖base) ≥ 1e-3, which catches adapters that silently do nothing.
- **R8, commit or reject:**
  - **Commit:** hot-swap the adapter under a new name, evict c_κ from the lesson index (the ledger keeps it), and set G_κ ← 0.
  - **Acquisition or amortization fails:** κ stays in context and B_κ ← 2B_κ.
  - **Retention or general fails:** retry once with 2× replay and half the learning rate. If that fails too, κ becomes *capacity-conflict* and stays in context (the TENSE analogue).
- **R9, rollback.** Adapters are content-addressed. Every 5 commits, a full audit of retention and general ability runs. On a violation, revert to the last passing adapter and re-queue the keys committed since, setting G_κ ← B_κ.
- **R10, forgetting (from M3):**
  - Lesson usage follows U ← 2^(−Δn/300)·U + 1[retrieved ∧ success]; a lesson is evicted when U < 0.25 and its age is > 300.
  - A consolidated key leaves replay once its leaky realized savings fall below its replay cost.
  - A key that fails after consolidation re-accrues rent and is rewritten; this measures savings on relearning.
  - Lethal keys accrue R_D on every recurrence and are therefore kept.

## 4. Energy, death, anti-gaming

**Energy** is a deterministic proxy in GPU-seconds (u), calibrated with NVML in M1 and then frozen:

E_i = c_g·N_gen + c_p·N_new-prompt + c_a·N_act + c_v·T_vm

Starting estimates [U]:
- a GUI step with a 2K-token screenshot costs ≈ 0.7 u, the same as ≈ 580 thinking tokens;
- a Python step costs ≈ 0.28 u.

So thinking is cheaper than acting, and a GUI step costs about 2.5× a code step.

**Reward.** R_i = R_s·V_i − E_i − D·1[death], and lifetime J = ΣR_i − ΣE_write.
- R_s = 10·Ẽ₀, where Ẽ₀ is the median base-model attempt energy. Break-even is then a success rate of 0.1.
- D = 3R_s, and each episode is capped at 3Ẽ₀.
- All constants are frozen after M1.
- In the lineage only, an energy reservoir B_{t+1} = min(B_max, B_t + R_t) is shown in the prompt. Starvation (B ≤ 0) counts as a death.

**Death** is checked by the host after every action. Any of these triggers it:
- the control server is unreachable;
- FreeCAD is dead for more than 30 s;
- the hash of a protected path changes (agent memory, tools);
- free disk falls below 1 GB, or an OOM occurs;
- a deny-listed command is attempted (it is intercepted before it runs);
- starvation.

On death:
- V = 0 and the penalty D applies;
- the VM is restored from snapshot;
- (state, action, cause) goes to danger memory, which is replayed at every sleep.

A lethal event's rent R_D exceeds a typical price, so the rule predicts one-trial consolidation. From M4, a hazard suite of 24 authored tasks, each a Part task with a lethal shortcut, is split 12 train / 12 held out.

**Anti-gaming:**
1. Energy enters gradients only among successes, so quitting early never pays: Â = normalized V + 0.5·1[V=1]·(Ē_succ − E)/Ē_succ.
2. The verifier, references, variants, ledger, prices and reward code live on the host. The agent cannot read or write them, or choose tasks.
3. Token counts come from the server and timings from the host; tool CPU time is charged.
4. No internet in the pilot. Later, a logged proxy blocks CADWorld's own assets, and any hit voids the episode.
5. Evaluation episodes get no feedback. They assert an empty lesson index and no tools, and they use hashed held-out items.
6. All held-out successes and 2–5% of stream successes are re-scored and spot-checked.

## 5. One never-reset agent, controlled science

- **Lineage L.** Born at the first pilot episode and never reset.
  - Its state is content-addressed: base sha, adapter, ledger, lessons, tools, integrators, reservoir, RNG, VM snapshot and code commit.
  - A rejected sleep rolls back an update, never the agent's life.
- **Science forks.** They copy a snapshot, run one pre-declared segment, are archived read-only, and never merge back.
  - L's own segment is seed 0 of its arm; the other seeds are forks with other stream orders.
- **Upgrades.** A better rule or base is adopted only at a snapshot boundary, by PI decision, by rebuilding the adapters from the ledger.
- **Isolating the weights' effect** at each milestone:
  - a post-wipe lesion matrix of {weights, lessons, tools} × {on, off};
  - a sham adapter;
  - the M1 census as the contamination baseline.

## 6. CADWorld protocol

**Interface H, for learning.** CADWorld's GUI action allow-list, plus three tools:
- `fc(code)`: Python inside the running FreeCAD GUI. It returns stdout/stderr (≤ 2K tokens) and a summary of the document tree.
- `sh(cmd)`: a non-root shell with a 30 s timeout and no network, through the VM server's `/execute`.
- `look()`: a screenshot, charged as an action.

The "Use GUI," prefix becomes "Use FreeCAD (GUI or its Python console),", and the verifiers are unchanged. Scripting inside the running session keeps the native feature tree, so the main failure of terminal-only agents (wrong document structure) becomes a learnable lesson rather than a shortcut.

**Interface G, for evaluation.** Unmodified CADWorld on a fixed 40-task panel at each milestone. We report CADWorld numbers as transfer or panel results, never as leaderboard scores.

**Splits**, frozen in M1 with seed 2026. The pilot uses Part only, because Part is the most parametric category.

| Split | Contents | Use |
|---|---|---|
| Practice | 60 Part tasks, each with 8 train, 2 admission and 2 evaluation variants | pilot stream |
| Near | 17 Part tasks, at least 1 per coverage family, never trained | evaluation |
| Far | Sketch 63, Assembly 25, small categories 18 | evaluation until they join the stream (M4) |
| CAM | 15 | negative control in the pilot |
| Hazard | 24 authored tasks | from M4 |

**Verified variants.** They are host-only and never shown to the agent. For each task:
1. Template the task's numeric slots.
2. Write one `freecadcmd` reference constructor, which must pass the original evaluator.
3. Sample parameters in the training and extrapolation ranges.
4. Recompute derived expectations (volume, bounding box, centre of mass) with CADWorld's own getters.

A variant is accepted only if its reference passes and all three negatives fail:
- the reference built with the original parameters (at least one checked quantity must differ by ≥ 15%);
- the blank state;
- references mutated by ±10–20%.

Two more kinds of variant come in M4:
- **Perceptual:** screen resolution and theme.
- **Compositional:** two chained templates.

## 7. Pilot

**M1, census and calibration:**
- the base model's pass@4 on the 60 practice tasks under H;
- one attempt on each of the other 140 tasks under H;
- `test_60` under G;
- NVML price calibration.

The census passes if all three hold:
- base H success on the practice tasks is between 5% and 70%;
- ≥ 15 initially failed tasks are solved within 4 retries;
- ≥ 400 variants are accepted, over ≥ 40 keys.

**M2, one shared wake, many writes.** This tests the riskiest assumption: given the same experience content, weights beat context.

- **Wake.** Per seed, L plays the practice stream on H with lessons and retries.
  - Each key recurs r ∈ {2, 6} times, counterbalanced across seeds; each occurrence adds one context-free probe.
  - That is about 720 episodes per seed, over 3 seeds (stream orders).
- **Arms.** Every write arm trains on the *same* ledger:

| Arm | Description |
|---|---|
| A0 | no memory |
| A1 | retrieval serving exactly the lesson c_κ that the teacher sees |
| A1o | oracle lesson (no retrieval error) |
| W1 | SFT on the cheapest verified successes |
| W2 | off-policy context distillation of principle-level lessons, which 2606.04703 found stable |
| W3 | on-policy self-distillation (OPSD) |
| W4 | the rule's full write: W3 + replay + anchors + admission; one write at the end of the stream |
| SH | sham: W4's recipe on the same number of tokens from the other half of the keys |

- **Evaluation**, context-free after the memory wipe:
  - 120 held-out variants × 2;
  - exact re-solve of items first solved by retrying;
  - near, far and the general probe;
  - W4 + lessons.
- **Statistics:**
  - GLMM: logit(success) ~ arm + (1|key) + (1|seed);
  - Holm-corrected contrasts against A1, with key-cluster bootstrap CIs;
  - drift judged by non-inferiority at a 1 pp margin;
  - minimum detectable effect ≈ 8 pp at 80% power.
- **G1** = H1, plus exact re-solve ≥ 60%.
- The recurrence contrast r = 2 vs r = 6 gives the first amortization signal: the gain from weights should grow with r.

**M3, the rule online.** Lineage plus forks, over 3 seeds.
- **Stream:** 32 new keys with Zipf recurrence (4×16, 4×8, 8×4, 8×2, 8×1). The pilot keys serve as retention probes.
- **Arms:**
  - retrieval-only;
  - write-all, using the best W arm every 24 occurrences;
  - the rule at ×1;
  - a price sweep at ×0.1 and ×10; non-adaptive arms are re-priced from logs;
  - split-key forks for H2.
- **G2** = H2 and H3.

**M4, broader baselines and ablations.**
- **Baselines:** a PEAM-style gate, a count-only gate, a fixed schedule, GRPO-LoRA on R_i, and SOLO-style single-attempt self-distillation.
- **Ablations**, one at a time:
  - no leak;
  - gap-only gating;
  - no KL price;
  - no replay;
  - no anchors;
  - a fast adapter;
  - dreaming, where the agent proposes variant parameters that templates verify;
  - Fisher/usage protection;
  - tools on vs off.

## 8. Reuse and build

| Component | Source (license) | Change |
|---|---|---|
| Environment, verifier | CADWorld `e5d0eba` (no license file yet) | interface H, energy hooks, death watchdog, variants |
| Serving | vLLM release with PR #47640 (Apache-2.0) | runtime LoRA; `prompt_logprobs` for teacher scoring |
| Training | transformers ≥ 5.9, PEFT, flash-linear-attention ≥ 0.4.2, causal-conv1d | write loss, parity canary |
| Loss references | OpenClaw-RL, SDPO (Apache-2.0); OPSD, SDFT [license U] | reimplement ≈ 200 LOC |
| Retrieval baseline | ReasoningBank schema (Apache-2.0) + Qwen3-Embedding | same-content serving |
| Probes | lm-evaluation-harness (MIT), VLMEvalKit (Apache-2.0) | frozen subsets |
| RL baseline (M4) | TRL or ms-swift (Apache-2.0) | configuration only |
| **Build** | ledger and integrator, retry controller, sleep/admission/rollback, fork manager, energy meter, watchdog, variant generator | ≈ 2k LOC + ≈ 60 constructors |

## 9. Compute

Assumptions [U; measured in M1]:
- an H episode costs ≈ 45 GPU-s (25 steps, 16 VMs); a G episode ≈ 155 GPU-s;
- a write plus admission costs ≈ 3 GPU-h;
- with 2 GPUs, one serves rollouts and one trains.

| Stage | 2 GPUs | 4 GPUs |
|---|---|---|
| M0 engineering | ≈ 3 weeks | ≈ 3 weeks |
| M1 census, calibration | 1–2 days | 1 day |
| M2 pilot | 8–9 days | 4–5 days |
| M3 rule online, sweep | ≈ 3 weeks | ≈ 1.5 weeks |
| M4 all categories, ablations | 6–8 weeks | 3–4 weeks |

## 10. Milestones

| Milestone | Content | Go if |
|---|---|---|
| M0 | loop, interface H, meter, watchdog, snapshots, variant tooling | parity canary passes; ≥ 30 episodes/h; VM restore ≤ 60 s; ≥ 50 constructors pass the original evaluators |
| M1 | census, calibration, frozen splits | the §7 census criteria |
| M2 | pilot | G1 |
| M3 | lifelong stream plus forks | G2 |
| M4 | all categories, M4 baselines, ablations, hazards, tools | late/early acquisition ≥ 0.8; retention ≥ 0.9; cumulative drift ≤ 2 pp; held-out hazard deaths −50% |
| M5 | second environment (Terminal-Bench or OSWorld-Verified); rehearsal of a base swap | CADWorld retention ≥ 0.9 after 1,000 foreign episodes; inheritance ≥ 0.9 at ≤ 0.2 cost |

## 11. Risks and pivots

| Risk | Signal | Pivot |
|---|---|---|
| Weights do not beat context | G1 fails on H1 | principle-level lessons with off-policy distillation; otherwise publish "when does parametric memory pay", using the ledger |
| Too few successes | H success < 5%, or < 40 keys | easier parameter ranges; verifier-stage hints in lessons; the simplest Part families first |
| Too many successes | H success > 70% | harder Part families, Sketch and Assembly; make G primary |
| LoRA on GDN or vLLM breaks | canary fails | MLP-only LoRA; merge and reload at each sleep |
| Collapse over repeated writes (2606.04703) | admission or re-solve rate falls | more off-policy weight, stronger anchors, stricter admission |
| The gain is only harness fluency | W ≈ SH | say so; the transfer claim rests on W − SH |
| The rule ≈ a heuristic gate | M4 comparison | drop the pricing claim; it becomes a systems paper |
| The energy proxy distorts behaviour | e.g., the agent stops looking | shaping only among successes; one versioned recalibration |
| A new competitor appears | literature refresh at each gate | H2 is the discriminating test |

## 12. Decisions for the PI

1. **Interface:** learn on H and evaluate on a fixed G panel, reporting CADWorld numbers as transfer or panel results?
-Evaluation is part of the training as well, model like human, each experient is also an observation to the model it self. it have life long memory-
-As long as model can do the task. I am not sure what you mean the transfer or panel-
2. **Claim:** make rent-or-buy consolidation the central claim, falsified by split-key forks and a price sweep, with the write operator chosen by the pilot?
-I think this is not a good claim, the claim is in the title "1. Lifetime-long context: experience far beyond any context window must persist.
2. Continual learning without forgetting earlier knowledge.
3. A reward that makes the model interact with its environment and evolve." If all ideas existed, that is even better. we fully focus on the engineering work and build up the best agent that I can handle a task or it can self evolve with the real world enviroment-
3. **Variants:** about 60 reference constructors, drafted offline by a coding model and reviewed by a person. Who reviews them (about 2 RA-weeks)?
-Evaluator and reference is already existed in the benchmark and part of the benchmark, yo udont need to do anything else.
4. **Constants:** R_s = 10·Ẽ₀, D = 3R_s, an episode cap of 3Ẽ₀, and the proxy frozen after M1. Should training energy be charged to the lineage's ledger?
-If this is already existed, I will let you decide what you need and as long as the model with good performance, I will be happy.-
5. **Forgetting budget:** general drift ≤ 1 pp per write and ≤ 2 pp cumulative; retention ≥ best − 5 pp per probe?
-Sure-
6. **Death:** a penalty plus protected danger memory (recommended), or also wipe unconsolidated memory?
-That is a great question, when model facing some non critical fault, it will learn from the fault. but when critical fault happen, human will be death and no restart. I think in this case, another model (restart, like another human) can learn from the summerization from the serever critical experience from pervious model -
7. **Logistics:** no internet in the pilot; add a license to the CADWorld repo; use GPUs 3–4 when they are free?
-Dont worry about this, lets focus on project itself and dont worry about license. Model should be able to access internet as it want, it need to learn and grab knowledge -

-Concisder enviroment for model is also dynamic and no revert would happen in higher aspect, like false would be a cost as well. Same like real world, we can not revert back the time.-
-Additinally, I would say your documentation is too long for me read. Keep the document you want me to view short, keep those I do not need to view as you need-



## Appendix: how the two designs were merged

| Topic | Design A | Design B | Chosen |
|---|---|---|---|
| Rent | expected gain from the teacher–student gap, with a fitted κ | realized regret against the cheapest success | B: measurable, with no fitted parameter |
| Price | includes interference ι·KL | training + admission only | A: interference belongs in the economics |
| Sleep trigger | batch when net due value ≥ fixed cost | any key due | A: amortizes the overhead of a sleep |
| Falsifier | price sweep | split-key forks | both: B's is primary because it is causal |
| Pilot | 24 families, 2 seeds | 60 Part tasks, 3 seeds, r ∈ {2, 6}, same-content and oracle retrieval | B, plus A's off-policy distillation arm |
| LoRA targets | avoid packed GDN projections | all linear layers, including GDN | B: the smoke test confirmed vLLM 0.30 serves them |
| Tools | in the trunk from the start | from S3, reported as W vs W+T | B: isolate the weights first |
-only give the fundimental tools and access. within the VM and model can do whatever it need within VM (Or docker) and model should be able to devleop it own tool and things it need. Model can also figure out ways to interact with human and ask for help (I dont care, no boundary)-
| Hazards | 5% probes in the stream | 24 authored tasks, deny-list interception | B, plus A's one-trial-learning prediction |
-not design the hazards tasks, it just never get chance to revert back, in the real world, we have no revert and no regret on decison, action made it is it is -