# SelfEvolveLLM: findings and plan

- Last updated: 2026-09-30
- Owner: PI (Zihan Dong)
- Status: the PI reviewed design round 1 (§9, decisions 6–15). Round 2 is under way; implementation has not started.

This is the project source of truth. Keep it short and update it when a decision changes. Survey notes are supporting evidence, not the plan.

## 1. Goal

Build a general agent that learns continually from verified experience. Given an arbitrary task in an isolated shell-and-internet sandbox, it should try, use verifier feedback, recover from failure, and consolidate reusable experience into its own weights. After consolidation it should:

1. solve the same task more directly after all episode context is removed;
2. transfer to related unseen tasks;
3. retain earlier skills and general model capabilities; and
4. spend less inference energy with practice.

The agent is general, not a computer-use specialist: CADWorld is only the first environment of a never-reset lifelong stream (§9), and the contribution must hold across domains. The agent may create tools inside its sandbox; tools are external procedural artifacts, while knowing when and how to use them may be consolidated into weights.

## 2. Scientific thesis and novelty bar

### Core problems

1. Lifetime-long context: experience far beyond any context window must persist.
2. Continual learning without forgetting earlier knowledge.
3. A reward that makes the model interact with its environment and evolve.

### Working hypothesis

Consolidate an experience into weights once the energy the agent keeps paying for not knowing it exceeds the measured price of writing it (rent-or-buy). One energy ledger then decides retries, sleep, writing and forgetting. Predictions and falsification tests are in `design/DESIGN.md` §1.

### Required contribution

The best working agent for the three core problems, with lifelong evidence (PI decision 6, §9). Existing methods are welcome; a new principle is not required.

The minimum convincing result is a long, heterogeneous task stream showing all of the following:

- context-free re-solving after a memory wipe;
- near and far transfer to held-out task families;
- bounded forgetting and sustained late-stream learning;
- lower lifetime inference cost than strong retrieval/context baselines; and
- causal ablations connecting the result to the proposed consolidation mechanism.

A main-journal claim requires a broadly important scientific finding, not only a better agent system. A top ML conference is the primary method-paper target; a broader journal submission follows only if large-scale results reveal a robust general learning phenomenon.

## 3. Current evidence

### Relevant prior work from this group (PDFs in `Papers/`)

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

## 4. Design

See `design/DESIGN.md`, which covers the architecture, the exact rules, energy and death, and the never-reset lineage with forks.

## 5. Experiments

See `design/DESIGN.md` §7 (pilot) and §10 (milestones and gates).

## 6. Environment, model and compute

**CADWorld**, the first environment ([paper](https://arxiv.org/abs/2609.16251), [repo](https://github.com/Zdong104/CADWORLD) at `e5d0eba`; the repo has no license file yet):

- 200 FreeCAD tasks. The paper counts 183 knowledge points; the task files carry 227 distinct `coverage` tags. The categories are skewed: Part 77, Sketch 63, Assembly 25, CAM 15, and seven categories with 2–3 tasks each.
- Paper baselines (v2; recheck before citing):
  - The best agent scores 17.5% (GPT-5.4 computer-use); the expert reference scores 87%.
  - Open-weight models score at most 1.5%; Qwen3.6 scores 0%.
  - Per task: 72–105 actions and 0.63–0.88M input tokens, with a 100-step limit.
  - Terminal-only agents on a 50-task subset (Fig. 5) score 8.6% pooled: Opus 4.8 16%, GPT-5.4 14%, Qwen3.6 6%. Wrong document structure is their main failure. Without its native computer-use harness, GPT-5.4 solved no task.
  - The "Qwen3.6" GUI baseline is Qwen3.6-35B-A3B with thinking off and 512 output tokens. It says little about Qwen3.8-27B, which reports 84.3% on OSWorld-Verified.
  - **Implication:** our start model will rarely succeed, so the first design problem is getting any verified success to learn from.
- **Setup, done 2026-09-29:**
  - Submodule `third_party/CADWORLD`.
  - Ubuntu VM image of 26.4 GiB, SHA-256 verified, run under QEMU/KVM in Docker.
  - Host FreeCAD 1.1.3 for CAM checks, in `.cache/tools/`; override the profile's `FREECAD_CMD` to use it.
  - Our runs coexist with the other job; ports are scanned under a `/tmp` lock.
  - Known flaws: ports bind 0.0.0.0, and each container leaks an anonymous Docker volume.
- **Interface:**
  - The model sees a 1920×1080 screenshot and replies with JSON `{reason, action(s)}`: literal `pyautogui` calls, or WAIT/DONE/FAIL.
  - The stock agent allow-lists those literals, but the VM server executes raw Python, so a custom agent can add shell or FreeCAD-Python tools.
  - `CADWORLD_MAX_TOKENS` defaults to 512 tokens, thinking included; raise it for Qwen thinking.
- **Verifier:**
  - It runs on the host and scores the saved `.FCStd` file 0 or 1.
  - `evaluation.json` gives pass/fail per stage (saved, valid, precondition, structure, geometry, process, strict) plus a failure class. This is a candidate dense signal.
- **Reset:** there are no VM snapshots; each reset cold-boots the container. That costs about 40 s per task including runner waits, plus about 15 s per evaluation.
- Task lists are `test_easy`, `test_small`, `test_60` and `test_all`. There is no train/test split, no task generator and no human demonstrations. The repository also ships a terminal/FreeCAD-Python harness (`CLI/`) and 302 fixtures. Every instruction starts "Use GUI,". About 70 of 77 Part tasks are parametric.

**Model:** start from [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0, released Aug 2026). It is a dense 27B vision-language model with 64 layers that mix Gated DeltaNet and gated attention, 262K native context, and adjustable thinking. The pinned revision is `1d4bf0f` (`Qwen3_5ForConditionalGeneration`: 48 Gated DeltaNet + 16 gated-attention layers, plus one MTP layer). **Smoke test, 2026-09-29.** Stack: vLLM 0.30.0, torch 2.13.0 (CUDA 13.0), transformers 5.17.0, PEFT 0.21.1, flash-linear-attention 0.5.2, Python 3.12. Logs are in `runs/smoke/`.

- **Serving on one GPU:** 85 GiB (53 GiB weights; 28 GiB KV cache ≈ 430K tokens); 192 s startup. It needs `--max-num-seqs 256` and `VLLM_USE_FLASHINFER_SAMPLER=0`, which `env.sh` sets.
- **Throughput:** 27 decode tokens/s at concurrency 1 and 202 at concurrency 8, about 18% lower with a LoRA active.
- **Inputs:** a 1920×1080 screenshot costs 2,042 image tokens. Thinking is on by default (`reasoning_effort` low/medium/xhigh) and can be turned off per request.
- **LoRA serving:**
  - vLLM accepts adapters on every language-model linear layer, including the Gated DeltaNet projections (`in_proj_z` only together with `in_proj_qkv`), and its outputs match HF+PEFT.
  - Hot-loading takes 0.2–1.4 s.
  - Give each adapter version a new name, because the prefix cache is keyed by name.
  - Avoid `load_inplace`, which re-reads the adapter from disk on every request.
- **Training (rank 64, one GPU, bf16):** 10.8 s/step at 8K tokens (67 GiB) and 50 s/step at 32K (79 GiB) with flash-linear-attention. Without it, 8K runs 3× slower and 32K runs out of memory.
- **Train → serve:** a trained adapter raises the log-probability of its training text by 0.45 nat, identically in vLLM and HF (correlation 0.998).
- **Known pitfalls:** older vLLM crashed on a partial GDN LoRA (fixed by PR #47640), and PEFT adapters can load but silently do nothing. The parity canary (`design/DESIGN.md` R7) still guards both.

**Hardware:**

- 4× RTX PRO 6000 Blackwell (96 GB). Never use the 8 GB A1000. Always set `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- Only GPUs 0 and 1 are free today. Another job (a vLLM server plus a CADWorld container) holds GPUs 3 and 4; do not disturb it.
- 240 CPU threads, 503 GB RAM, 2.1 TB free disk.
- Pilot on one rollout GPU and one training GPU. Add GPUs only after the single-node loop and admission tests are reliable.

## 7. Execution plan

1. Done: version control initialized (2026-09-29).
2. Done: first benchmark chosen, CADWorld (§9).
3. Done: two independent deep-reasoner designs (`design/design_a.md`, `design/design_b.md`) synthesized into `design/DESIGN.md`. CADWorld set up (§6).
4. Done: PI review of round 1 (§9, decisions 6–15). The smoke tests passed (§6).
5. **Now:** design round 2, with two independent deep-reasoner designs (`design/round2_a.md`, `design/round2_b.md`) to be synthesized into the design.
6. Then engineering, followed by the milestones that round 2 sets.

## 8. Documentation rules

- Keep every doc concise and state each fact once; other docs link to it.
- `FINDINGS_AND_PLAN.md` is the only living plan. `design/DESIGN.md` holds the design; it is a draft until the PI accepts it.
- `design/design_brief.md`, `design_a.md` and `design_b.md` are the inputs to this design round. Delete them once `DESIGN.md` is accepted.
- `survey/` holds compact evidence notes, not plans.
- Edit decisions in place and delete superseded text; git keeps the history.
- Mark unverified claims, pin versions, and link the evidence needed for later citation.
- Do not include paths or operational details from unrelated projects.

## 9. PI decisions

Decided 2026-09-29:

1. **First environment: CADWorld.** The agent is never reset. One run continues from now on, across CADWorld and every later benchmark, so it must absorb new knowledge without forgetting old knowledge.
2. **Model:** start from Qwen3.8-27B (§6).
3. **Survival:** while evolving, the agent must learn to recognize and avoid actions that would kill it.
4. **Hardware:** never use the A1000.
5. **Base-model upgrades:** a stronger base model may later replace Qwen3.8-27B, and learned experience must carry over.

Decided 2026-09-30, on reviewing design round 1:

6. **Aim.** The claim is the three core problems (§2), solved by the best agent we can build. Existing ideas are preferred; rent-or-buy is no longer the claim.
7. **End state.** A personal lifelong assistant: it takes the PI's tasks and hands back the results, learns the PI's habits, confirms what the PI needs, and builds and uses its own tools.
8. **Irreversible life.** No rollback of the agent's weights, memory or world. Mistakes are costs, and the environment persists and changes.
9. **Evaluation is experience.** Every episode, evaluation included, is also learning.
10. **Death is natural.** There are no authored hazard tasks. A non-critical fault is a lesson. A critical fault ends the individual for good, and a successor learns from a summary of its critical experience.
11. **Access.** Only fundamental tools (GUI, shell, Python), full freedom inside its VM or container, open internet, and it may ask humans for help.
12. **Learning.**
    - A reflection step gets more out of few trajectories.
    - Among compared attempts, the better behavior wins gradually.
    - Sleep follows a day length set per environment, and the agent decides what to consolidate.
    - The agent may evolve its base model as well as its adapters.
13. **Benchmark as is.** No task variants or reference constructors.
14. **Forgetting budget.** General drift of at most 1 pp per update and 2 pp cumulative; retention of at least the best score minus 5 pp.
15. **Delegated.** Energy and reward constants, logistics and licenses are left to the design.

Design round 2 is under way from `design/design_brief.md`.

