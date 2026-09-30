# Survey: parametric memory, sleep/consolidation, closest competitors

Updated 2026-09-29. This is supporting evidence, not an implementation plan. Numbers mostly come from arXiv abstracts; only the Sleep paper was checked against its PDF. Verify the full paper and current competitor landscape before citing. IDs with a PDF in `../../Papers/` were checked against their arXiv titles and abstracts on 2026-09-29.

## Key takeaways
1. **The "Google sleep" paper** is **Behrouz, Hashemi, Javanmard, Mirrokni, *Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories*** (Google Research + Cornell, arXiv 2606.03979, v1 Jun 2026, v2 Jul 2026; an ICLR 2026 OpenReview submission).
   - The "data augmentation" is its **Dreaming** step. No code was found. It is not agentic.
2. **Its mechanism**
   - **Wake:** the fast CMS blocks absorb the context.
   - **Sleep (i), consolidation:** the model adds new low-rank MoE experts and does "Knowledge Seeding", on-policy distillation (GKD) into only the new experts with old weights frozen. A "Learning-to-Imitate" RL step follows.
   - **Sleep (ii), Dreaming:** the model generates synthetic dreams with the context in its prompt, adding recombination by routing to a random expert. It keeps the top-k dreams by gradient importance plus some random ones. It rewards a dream with 1 if LoRA-SFT on it improves a downstream metric (ReST-EM, as in SEAL).
   - **Schedule:** sleep runs at fixed chunk boundaries; it is not triggered by surprise.
   - **Results:**
     - SQuAD no-context 48.9 / 46.2 vs SEAL 46.7 / 43.2 (their reproduction).
     - Without Dreaming: 35.7 / 36.2.
     - Qwen3-8B AIME24: 79.2 vs OPSD 76.6.
     - Sequential Kalamang + Manchu: little forgetting.
     - BABILong up to 10M tokens.
   - **Gap:** no explicit forgetting metric, and not agentic.
3. **Best-supported write operator: on-policy self-distillation** (RL's Razor, SDFT, SDPO, OPSD). The teacher is the same model *with* the trajectory, reflection or verifier feedback in context; the student sees none of it. Only the teacher–student gap is written, which is predictive coding in practice.
4. **Surprise gate candidates:** NLL surprise (SuRe), the teacher–student gap (it predicts OPSD gains linearly, arXiv 2605.30070), and gradient importance (Sleep).
5. **Where to write**
   - All-layer LoRA, especially on MLPs, is enough (LoRA Without Regret).
   - Protect weights by usage, as in Sparse Memory Finetuning (Meta 2510.15103): 11% forgetting vs 71% for LoRA. This is MAS-like.
   - Or protect by Fisher information; SCoL's layers already align with it.
6. **Fast weights in practice:** a fast LoRA and a slow LoRA merged by EMA (SuRe 2511.22367), or CMS multi-frequency updates (HOPE). Add periodic expansion or resets against plasticity loss.
7. **Replay mix:** self-generated variants of solved tasks (Sleep dreams, SEAL self-edits, TT-SI 2510.07841), plus context-free synthetic replay (2505.13811) to protect general ability.
8. **Warnings**
   - Repeated experience internalization **collapses** over iterations (2606.04703). That paper finds principle-level experience and off-policy context distillation more stable, which conflicts with SDFT; we must test both.
   - Agent optimizers compound only with **regression control** (Terminal-Bench 2.0, 2607.14004).
   - → We need an **admission test** before accepting each update.
9. **Crowded 2026 field:** SOLO, LifeSkill, PEAM, TMEM, aTTT, OpenClaw-RL, SEAgent. None that could be verified combines all four of: open sandbox, retry until the verifier passes, online consolidation into weights, and joint measurement of re-solve, transfer and forgetting.
10. **SCoL citations:** none found; the Sleep paper does not cite SCoL. **The NeurIPS 2026 TTCL workshop** is a natural venue: Zekun Wang co-organizes it and MacLellan is an invited speaker.

## Reusable methods and code (✓ = code available)

**Architectures and test-time training**
- Titans (lucidrains/titans-pytorch ✓ MIT, unofficial)
- MIRAS (framing: attentional bias + retention regularizer)
- ATLAS (window-level memory)
- HOPE/CMS (community reimplementations only)
- TTT-E2E 2512.23675 (✓ JAX; meta-learned init)
- TTT-Discover 2601.16175 (✓ MIT; RL on a single problem)

**Self-directed updates**
- SEAL 2506.10943 (✓ MIT; self-edits via ReST-EM; forgets under sequential edits)
- Generalized Neural Memory 2602.23201 (✓)

**Update rules that forget less**
- **SDFT 2601.19897** (✓ github.com/idanshen/Self-Distillation). Qwen2.5-7B prior capabilities stay at 64.5 vs 53.4 for SFT. Costs about 2.5× the FLOPs of SFT.
- **SDPO 2601.20802** (✓ Apache-2.0, lasgroup/SDPO; ICML 2026). The teacher is conditioned on feedback such as errors or test logs; needs 3× fewer attempts at test time.
- **OPSD 2601.18734** (✓)
- iSDFT 2609.24646
- On-policy distillation (✓ tinker-cookbook, Apache-2.0). It recovered IFEval from 45% to 83% after mid-training → **a "sleep" restore step**.
- RL's Razor 2509.04259: forgetting ∝ KL to the base model on the new task; on-policy is KL-minimal.

**Amortized context → weights**
- Context distillation 2209.15189
- Cartridges (✓ Apache-2.0)
- Doc-to-LoRA 2602.15902 (✓ MIT). Sub-second hypernetwork write, a candidate **fast "hippocampal" write**.
- Text-to-LoRA (✓ Apache-2.0)
- Generative Adapter
- MemoryLLM / M+ (✓ MIT; baseline)

**Editing** (EasyEdit ✓ MIT): AlphaEdit, WISE, GRACE, UltraEdit, MEMOIR. Route discrete facts to editing or side memory (TENSE-style) and procedures to LoRA.

**Plasticity**
- Dohare et al., Nature 2024 (continual backprop ✓ MIT).
- "Can Scale Save Us" 2606.24752: plasticity loss persists in LMs.

**Classic continual-learning methods at LLM scale**
- EWC helps Gemma2 continual pretraining (2505.05946). No strong evidence for MAS or SI at LLM scale (unverified).
- Replay plus LR re-warming works (2403.08763).

**Complementary learning systems:** ICML 2026 spotlight "Modular Memory is the Key to Continual Learning Agents" (2603.01761); Experience Funnel (2609.08919); Dual-Layer Agentic Memory (2608.22215).

## Closest competitors (full texts must be read before claiming novelty)

| Work | What it does | Closeness |
|---|---|---|
| **SOLO** 2609.34321 (28 Sep 2026) | GUI agents on recurring streams; judge-selected successes plus relabeled failed prefixes; adapter updated by top-K self-distillation; +3–6 points; beats memory baselines | **High**, but a single attempt, a judge rather than a verifier, and forgetting not central |
| **LifeSkill** 2606.04815 | Skill extraction after failure, retry, online internalization of skill-conditioned trajectories into the policy | **High** |
| **PEAM** 2605.27762 | Minecraft; failed→corrected pairs train an isolated MoE-LoRA; self-triggered consolidation | **High** (but embodied) |
| SEAgent 2508.04700 | Batch-phase training; OSWorld novel apps 11.3→34.5% | Medium-High |
| **OpenClaw-RL** 2603.10165 | Async live RL on terminal, GUI and SWE with LoRA (✓ Apache-2.0, 5.7k stars) | Medium-High; **possible infrastructure to reuse** |
| TMEM 2606.04536 | Within-episode LoRA fast weights | Medium |
| aTTT 2607.03441 | Within-episode LoRA TTT; ALFWorld +5.0 pp, SWE-bench Lite +4.9 pp | Medium |
| Dennis et al. 2605.24657 | Nightly LoRA consolidation of knowledge | Medium |
| EVAF 2606.26806 | Surprise- and valence-gated LoRA; toy models only | Medium-Low |

**Other points from the report**
- A skill library on OSWorld sometimes failed to re-solve the task its skill came from (2609.04869). This is an argument for storing experience in weights.
- Non-parametric baselines to beat: ReasoningBank 2509.25140 (✓ Apache-2.0), ReMem/Evo-Memory, and Anthropic's Dreaming.
- Evaluation protocols to adopt: AgentStream 2608.00155, and "Do Self-Evolving Agents Forget?" 2605.09315.

## The gap a new paper can claim
1. **Setting:** start from initially failed tasks (pass@k=0) in an **open sandbox** (shell, web); retry with reflection until a **verifier** passes; consolidate **online into weights**.
2. **Proof the gain is in the weights:** re-solve after a full **context and memory wipe**, show near and far transfer, and keep forgetting bounded over hundreds or more cycles.
3. **Compounding:** gains compound rather than collapse (an admission test and regression control); compare at equal inference-token budgets against non-parametric memory.
4. **Neuroscience-tied ablations:** prediction-error gating, fast/slow timescales, usage protection, and sleep with vs without dreaming. Hypothesis: sleep matters for transfer, not for re-solving.
5. **Open-source code**; most competitors have none.

## Maintenance

Keep only details that affect a design choice, baseline or novelty claim, and remove superseded listings at each literature refresh. Paper PDFs are in `../../Papers/`; the reuse boundary is in `../FINDINGS_AND_PLAN.md` §3.
