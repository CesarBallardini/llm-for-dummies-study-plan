# Path 3 — Improve an existing model

**Goal.** Finish able to take a released open model and make it measurably better
at something you care about. Concretely: fine-tune it on your own data with LoRA
on a single GPU; align it to preferences with DPO, and know where RLHF differs;
train it to reason with reinforcement learning against a verifiable reward;
measure all of it with a harness rather than by reading samples; and read the
paper describing a new method well enough to reproduce it next month.

This is the research and training path. It is the only one whose last chapter
never finishes: [Chapter 24 — The Research Reading Loop](../chapter-24-research-reading-loop/index.md) is a routine, not a subject.

## Prerequisites

Everything through [Chapter 13 — Tokens and Embeddings](../chapter-13-tokenization-embeddings/index.md), and in particular:

- [Chapter 4 — Probability and Statistics](../chapter-04-probability-statistics/index.md) — *expectation and variance; maximum likelihood estimation; entropy, cross-entropy, KL divergence and perplexity; confidence intervals and the bootstrap*. DPO and PPO are written in exactly this vocabulary — a KL term against a reference policy is the heart of both — and Chapter 20 reports every score with a standard error.
- [Chapter 9 — Deep Learning: Concepts and Practice](../chapter-09-deep-learning/index.md) — *the training loop: mini-batches, forward and backward pass, optimizer step, validation; optimizers and learning-rate schedules*. Every method on this path is a variation on that loop with a different loss.
- [Chapter 12 — The Transformer Architecture](../chapter-12-transformer-architecture/index.md) — *scaled dot-product attention; the Transformer block; the decoder-only design; parameter counts*. Enough to know what a LoRA adapter is attached to and which layers are worth adapting.
- [Chapter 13 — Tokens and Embeddings](../chapter-13-tokenization-embeddings/index.md) — *special tokens; the token embedding table and weight tying*. Chat templates in Chapter 17 are special tokens, and a template applied inconsistently is the most common silent failure in fine-tuning.

[Chapter 14 — Build an LLM from Scratch](../chapter-14-llm-from-scratch/index.md) is not strictly required, but a reader who has built a model from scratch
debugs a fine-tuning run considerably faster than one who has not.

## The sequence

| Step | Chapter | Why this path needs it | Time |
| --- | --- | --- | --- |
| 1 | [Chapter 16 — Distributed Training and Systems](../chapter-16-distributed-training-systems/index.md) | **Part of the chapter.** The memory of a training step; mixed precision; gradient accumulation and activation recomputation; `torch.compile` and FlashAttention. These are what make a fine-tune fit on one GPU. The parallelism half — ZeRO, FSDP, tensor and pipeline parallelism — is Path 2's material and can wait. | 6–8 weeks |
| 2 | [Chapter 17 — Fine-Tuning and Parameter-Efficient Methods](../chapter-17-finetuning/index.md) | Supervised fine-tuning, LoRA and QLoRA. The cheapest way to change a model's behaviour, and the base layer for everything after it. | 4–6 weeks |
| 3 | [Chapter 18 — Alignment, RLHF and DPO](../chapter-18-alignment-rlhf/index.md) | Preference tuning: reward models, RLHF and DPO. Where a model stops merely answering in the right format and starts answering the way people prefer. | 5–7 weeks |
| 4 | [Chapter 19 — Reasoning Models and Reinforcement Learning](../chapter-19-reasoning-rl/index.md) | Reinforcement learning against verifiable rewards, long chains of thought, GRPO, and the reasoning-model recipe. | 5–7 weeks |
| 5 | [Chapter 20 — Evaluation and Benchmarks](../chapter-20-evaluation-benchmarks/index.md) | Perplexity, benchmarks, contamination, LLM-as-judge and their failure modes. Without this chapter the path produces changes you cannot call improvements. | 3–4 weeks |
| 6 | [Chapter 24 — The Research Reading Loop](../chapter-24-research-reading-loop/index.md) | Selecting papers, three-pass reading, reproducing a result, keeping notes. What keeps the path usable after the book ends. | ongoing |

!!! tip "Read Chapter 20 early"

    It is listed fifth because that is where it sits in the book, but nothing
    stops you from reading it straight after Chapter 17. Every experiment in
    Chapters 18 and 19 is worth more if you can measure it, and fixing the
    evaluation before running the training is the most useful habit on this path.

### Two places this path leans outside itself

[Chapter 16 — Distributed Training and Systems](../chapter-16-distributed-training-systems/index.md)'s milestone trains "nanoGPT or the Chapter 14 model" and its throughput
discussion is framed against the training runs of [Chapter 15 — Pretraining at Scale](../chapter-15-pretraining-at-scale/index.md). Neither is in this
path, and neither needs to be: substitute the model you are fine-tuning for the
model being trained, and read the numbers as a ceiling on your own run rather
than as a reproduction target.

[Chapter 24 — The Research Reading Loop](../chapter-24-research-reading-loop/index.md) has one topic — open-source releases: model cards, technical reports and
the release notes of training and inference libraries — that reads against
[Chapter 21 — Inference Optimization and Serving](../chapter-21-inference-serving/index.md) and [Chapter 22 — Operating and Improving an Open-Source LLM](../chapter-22-operate-improve-open-llm/index.md). It is one topic out of eleven; skip it, or pick it up if you
later take [Path 4](serve-and-operate.md).

## The milestones that prove it

Chapters 18 and 19 both end in a training run that fits a free Colab or Kaggle T4
GPU with LoRA: a small SFT model tuned on preferences, and a small
instruction-tuned model taught to reason. [Chapter 20 — Evaluation and Benchmarks](../chapter-20-evaluation-benchmarks/index.md) asks you to run
lm-evaluation-harness across several benchmarks on two open base models and
compare them properly. Pair them — run the evaluation before and after your own
fine-tune and report the difference, which is what the rest of the field would
ask you for.

## Estimated time

**23–32 weeks** at 10 hours per week for Chapters 16 to 20, plus [Chapter 24 — The Research Reading Loop](../chapter-24-research-reading-loop/index.md),
which is ongoing by design.
