# Rubric-Based On-Policy Distillation (ROPD) — Qwen3-0.6B

Continue from https://github.com/kakopappa/qwen-sft-on-policy-distillation-experiment

Instead of minimising token-level KL against a teacher, we score each student rollout
with a **rubric** — checkable criteria for format and factual correctness — and train
with **GRPO** (Group Relative Policy Optimization).

```
question
   │
   ▼
STUDENT (no manual) ──► G rollouts
                              │
                         rubric score each
                         ┌──────────────────────────┐
                         │ format:   </think> present?  +1.0 / -1.0  │
                         │           answer non-empty?  +0.5         │
                         │           think block short? +0.3         │
                         │ factual:  ref keywords found? 0..+1.0     │
                         └──────────────────────────┘
                              │
                    advantage = reward - mean(group rewards)
                              │
                    REINFORCE: loss = -advantage * log_prob(rollout)
                              │
                    backprop into student LoRA
```
