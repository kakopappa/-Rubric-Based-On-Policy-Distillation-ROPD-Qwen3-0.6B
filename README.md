# Rubric-Based On-Policy Distillation (ROPD) — Qwen3-0.6B

Instead of minimising token-level KL against a teacher, we score each student rollout
with a **rubric** — checkable criteria for format and factual correctness — and train
with **GRPO** (Group Relative Policy Optimization).
