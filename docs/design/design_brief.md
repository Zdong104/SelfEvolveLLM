# Independent design brief

Produce an executable research design for SelfEvolveLLM. Be concise, critical, and specific.

## Objective

Design a general agent that learns from verified experience in an isolated shell-and-internet sandbox. It should retry failed tasks, consolidate reusable experience into weights, re-solve after episode context is removed, transfer to related tasks, retain old abilities, and reduce inference cost with practice. It should continual learning and not forget the knowledge learned before. 

This is an independed project from any other on the computer. 

## Scientific standard

The method must make a contribution stronger than assembling existing components. Center the design on one falsifiable scientific claim and identify the new mechanism or principle responsible for it. The evidence must separate weight learning from prompt, retrieval, tool, and benchmark-contamination effects.

Treat a top ML conference as the primary target. State what qualitatively stronger finding would justify a broad journal submission.

## Fixed requirements

- General, verifier-backed tasks in a reproducible sandbox.
- Short-, medium-, and long-timescale memory with explicit forgetting.
- A parametric write path, not only retrieval or prompt memory.
- This should related to systematic continual learning, with interacting with enviroment and be able to make the model stronger with RL and efficient reply.
- Energy-aware evaluation: verified success minus thinking, action, and environment cost. It is just intuitive action cause consequence and token cost enery for survive. Model should adapt to the enviroment like human do.
- Agent-created tools stored and use those tool to help agent get stronger.
- Strong context/retrieval, continual-learning, self-distillation, and RL baselines.
- Context-free evaluation after memory wipe, plus near/far transfer, retention, general-capability drift, and plasticity over time.

## Starting hypotheses, not commitments

- Context/on-policy self-distillation may write the residual between an experience-conditioned teacher and a context-free student more safely than ordinary SFT.
- Fast and slow LoRA adapters may provide practical memory timescales.
- Surprise, recurrence, and future inference cost may determine what is worth consolidating.
- Interleaved replay and verified synthetic variants may make sleep useful for transfer.
- Usage/Fisher protection or a capacity gate may reduce interference.

Reject or simplify any of these if the evidence does not justify its complexity.

## Relevant foundations

- SCoL: learned where-to-write and acquisition-minus-forgetting reward for streamed textual knowledge.
- TENSE: finite writable capacity and routing infeasible updates away from protected weights.
- Titans: surprise, momentum, and adaptive forgetting principles.
- Survey digest: `../survey/parametric_memory.md`.
- Current plan and constraints: `../FINDINGS_AND_PLAN.md`.

The closest external work must be checked before novelty claims. At minimum compare with Language Models Need Sleep, SOLO, LifeSkill, PEAM, SDFT/SDPO/OPSD, SEAL, and OpenClaw-RL.

## Resources and boundary

- 4× RTX PRO 6000 Blackwell (96 GB), Do not use the 8 GB A1000, 240 CPU threads, and 503 GB RAM.
- Start with Candidate: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) and model should be able to evolve and choose the candidate. When evolving, agent should be able to recognize action that kills itself and evolve etc.
- All dependencies, imported repositories, environments, data, and outputs must live inside this repository unless the PI explicitly approves another location.
- Online reuse is encouraged. Record source revision, license, and modifications. Do not inspect, modify, or depend on unrelated local projects.

## Deliverable

Write one compact design containing:

1. one-sentence thesis, falsifiable prediction, and honest novelty comparison;
2. minimal architecture and data flow;
3. exact write, decay, replay, admission, and rollback rules;
4. energy/reward formula and anti-gaming controls;
5. smallest decisive pilot, baselines, ablations, metrics, and statistics;
6. reuse/build map restricted to in-repository or online dependencies;
7. compute estimate for two and four large GPUs;
8. staged milestones with explicit go/no-go gates;
9. top risks and pivots; and
10. no more than seven decisions for the PI.

Prefer equations, tables, and measurable thresholds over prose. Aim for roughly 2,000–3,000 words; omit background already captured in the plan or survey.
