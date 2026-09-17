# Path 2 — Pretrain an open-source LLM

**Goal.** Finish able to plan and execute a pretraining run: choose a model size
and a token budget that fit a compute budget, assemble and filter a data mixture,
train the tokenizer, run the training loop across more than one GPU without
wasting most of them, watch the loss curve and know which of its shapes mean
trouble, and end with base model weights and a fine-tuned checkpoint you can hand
to someone else.

This is the most demanding path in the book and the only one with a real hardware
bill. The chapters are written so the milestones stay inside a free Colab or
Kaggle GPU; the capstone in [Chapter 23 — Capstone: Train, Ship and Improve Your Own Model](../chapter-23-capstone/index.md) is where a budget of $15–$100 turns the plan
into a released model.

## Prerequisites

Everything through [Chapter 13 — Tokens and Embeddings](../chapter-13-tokenization-embeddings/index.md), and in particular:

- [Chapter 12 — The Transformer Architecture](../chapter-12-transformer-architecture/index.md) — *the whole chapter*. The architecture being trained, including the changes in current open models — SwiGLU, grouped-query attention and mixture-of-experts layers.
- [Chapter 13 — Tokens and Embeddings](../chapter-13-tokenization-embeddings/index.md) — *byte pair encoding: training, encoding and decoding; vocabulary size, special tokens and compression ratio; the dependence of a tokenizer on its training corpus; tokenized datasets and sliding-window input-target pairs*. A pretraining run begins by training its own tokenizer on its own corpus, so these are operational sections here, not background.
- [Chapter 10 — Neural Networks from Scratch](../chapter-10-neural-networks-from-scratch/index.md) — *reverse-mode automatic differentiation; gradient checking with finite differences; vanishing and exploding gradients, and the scale of the initial weights*. A run that diverges at step 3,000 is debugged at this level, not at the level of the training script.
- [Chapter 5 — Optimization and Numerical Methods](../chapter-05-optimization-numerical-methods/index.md) — *stochastic and mini-batch gradient descent; momentum and adaptive methods (Adam, AdamW); learning-rate schedules with warmup and decay; floating-point representation, overflow and underflow; precision formats and mixed-precision training*. Every one of these is a hyperparameter or a failure mode of the run you are about to launch.
- [Chapter 7 — GPUs, Hardware and the Training Environment](../chapter-07-gpus-hardware/index.md) — *VRAM, memory bandwidth and the cost of moving data; FLOPs and FLOP/s; bytes per parameter; the introduction to training memory*. What sets the model size you can afford, and the arithmetic behind the compute estimate in Chapter 15.

## The sequence

| Step | Chapter | Why this path needs it | Time |
| --- | --- | --- | --- |
| 1 | [Chapter 14 — Build an LLM from Scratch](../chapter-14-llm-from-scratch/index.md) | Build and train the model itself: embeddings, blocks, the training loop, sampling. The thing being scaled up in the rest of the path. | 6–8 weeks |
| 2 | [Chapter 15 — Pretraining at Scale](../chapter-15-pretraining-at-scale/index.md) | Data mixtures and filtering, scaling laws, token budgets, compute estimates from C ≈ 6ND, and reading a loss curve. | 6–8 weeks |
| 3 | [Chapter 16 — Distributed Training and Systems](../chapter-16-distributed-training-systems/index.md) | Multi-GPU training: data, tensor and pipeline parallelism, ZeRO and FSDP, mixed precision, torch.compile, fused attention kernels, throughput measurement. | 6–8 weeks |
| 4 | [Chapter 17 — Fine-Tuning and Parameter-Efficient Methods](../chapter-17-finetuning/index.md) | Supervised fine-tuning and LoRA. A base model is not usable on its own; this is what turns the run's output into something that answers. | 4–6 weeks |

## The milestones that prove it

[Chapter 15 — Pretraining at Scale](../chapter-15-pretraining-at-scale/index.md) asks for a two-page training plan for a 1B-parameter model: data sources
and mixture, one compute-optimal and one over-trained token budget, compute in
FLOPs, and a cost estimate. [Chapter 16 — Distributed Training and Systems](../chapter-16-distributed-training-systems/index.md) asks you to measure tokens per second and peak
memory while switching on mixed precision, `torch.compile` and a fused attention
kernel, which is the difference between a run that finishes and one that does
not. Take them in that order — the plan tells you how much throughput you need.

## Where to go next

[Chapter 23 — Capstone: Train, Ship and Improve Your Own Model](../chapter-23-capstone/index.md) turns this path into a released artifact: a small chat model trained,
evaluated, served and published under a fixed budget. After that,
[Path 3](improve-an-existing-model.md) is how the model gets better.

## Estimated time

**22–30 weeks** at 10 hours per week for the four chapters, after the
prerequisites and not counting the capstone.
