# Path 4 — Serve, deploy and operate an LLM

**Goal.** Finish able to put an open model in front of users and keep it there.
Concretely: quantize a model and state what the quantization cost you in
accuracy; choose between llama.cpp, vLLM and the alternatives for a given
workload and defend the choice in terms of prefill, decode, batching and the KV
cache; size the hardware for a target latency and a target cost per million
tokens; and run the operating loop — select, adapt, serve, evaluate, improve —
instead of shipping once and hoping.

## Prerequisites

This path needs less mathematics than the others and more systems knowledge, so
its prerequisite list is shorter than "everything before Chapter 21":

- [Chapter 7 — GPUs, Hardware and the Training Environment](../chapter-07-gpus-hardware/index.md) — *compute-bound, memory-bound and overhead-bound workloads; VRAM, memory bandwidth and the cost of moving data; bytes per parameter in fp32, fp16, bf16 and int8; measuring GPU work with `nvidia-smi` and the PyTorch memory counters*. Chapter 21 *defines* prefill and decode in these workload classes and asks you to predict decode speed from memory bandwidth divided by model size. Essential, and the largest single dependency of this path.
- [Chapter 6 — Python and the Scientific Stack](../chapter-06-python-scientific-stack/index.md) — *virtual environments, the command line, Git; NumPy and PyTorch basics*. Essential: the deployment work is ordinary software engineering, and this is where the book covers it.
- [Chapter 13 — Tokens and Embeddings](../chapter-13-tokenization-embeddings/index.md) — *vocabulary size and compression ratio; the dependence of a tokenizer on its corpus and the resulting cost differences between languages; special tokens*. Essential: tokens are the unit you are billed in, the unit latency is quoted in, and the reason a non-English deployment costs more than you budgeted.
- [Chapter 20 — Evaluation and Benchmarks](../chapter-20-evaluation-benchmarks/index.md) — *perplexity and bits per byte; scoring methods; evaluation settings that change scores; statistical uncertainty*. Essential in practice: the Chapter 21 milestone prices quantization in perplexity, and Chapter 22 selects a base model on benchmark results.
- [Chapter 12 — The Transformer Architecture](../chapter-12-transformer-architecture/index.md) — *scaled dot-product attention; multi-head attention and its tensor shapes; grouped-query attention; the quadratic time and memory cost of attention in the sequence length*. Recommended, and these four sections are the whole of it: the KV cache, its growth with context length, and the methods that shrink it are all statements about this material.
- [Chapter 16 — Distributed Training and Systems](../chapter-16-distributed-training-systems/index.md) — *mixed-precision formats; attention kernels that minimize memory traffic (FlashAttention); tensor parallelism*. Recommended, three sections only. Chapter 21 uses each by name and does not re-derive them.
- [Chapter 17 — Fine-Tuning and Parameter-Efficient Methods](../chapter-17-finetuning/index.md) — *choosing between fine-tuning, prompting and retrieval; chat templates and special tokens; LoRA and QLoRA*. Recommended: Chapter 22's improvement cycle fine-tunes the model you are serving, and QLoRA is where fine-tuning meets the quantization of Chapter 21.
- [Chapter 9 — Deep Learning: Concepts and Practice](../chapter-09-deep-learning/index.md) — *`nn.Module`, autograd, the training loop*. Recommended: enough PyTorch to read what a serving stack is doing to your model.

From Part I, two sections carry most of the weight: matrix multiplication and
low-rank approximation from [Chapter 3 — Linear Algebra](../chapter-03-linear-algebra/index.md), and entropy, cross-entropy and perplexity from
[Chapter 4 — Probability and Statistics](../chapter-04-probability-statistics/index.md). The calculus of [Chapter 2 — Calculus I and II](../chapter-02-calculus/index.md) and the optimizer theory of [Chapter 5 — Optimization and Numerical Methods](../chapter-05-optimization-numerical-methods/index.md) matter far more for
training than for serving, and a reader whose goal is to operate models can take
them later. That is an editorial judgement about this route, not a claim that the
book's order is optional.

## The sequence

| Step | Chapter | Why this path needs it | Time |
| --- | --- | --- | --- |
| 1 | [Chapter 21 — Inference Optimization and Serving](../chapter-21-inference-serving/index.md) | Prefill and decode, the KV cache, continuous batching, speculative decoding, quantization formats, and the serving stacks themselves. | 4–5 weeks |
| 2 | [Chapter 22 — Operating and Improving an Open-Source LLM](../chapter-22-operate-improve-open-llm/index.md) | The operating cycle: pick a base model, adapt it, serve it, evaluate it, improve it. Retrieval, cost control and monitoring live here. | 6–8 weeks |
| 3 | [Chapter 23 — Capstone: Train, Ship and Improve Your Own Model](../chapter-23-capstone/index.md) | The capstone: train, evaluate, serve and publish a small model end to end within a fixed budget, which is this path rehearsed on something you own. | 4–8 weeks |

### What Chapter 23 assumes

The capstone is the one step here that is not about serving. Its milestone trains
a tokenizer, pretrains a model and fine-tunes it with SFT *before* anything is
served or published, which brings [Chapter 14 — Build an LLM from Scratch](../chapter-14-llm-from-scratch/index.md) and [Chapter 15 — Pretraining at Scale](../chapter-15-pretraining-at-scale/index.md) with it on top of the
[Chapter 13 — Tokens and Embeddings](../chapter-13-tokenization-embeddings/index.md) and [Chapter 17 — Fine-Tuning and Parameter-Efficient Methods](../chapter-17-finetuning/index.md) already listed above — several months of work that operating a
model does not otherwise require. Its DPO or RL stage ([Chapter 18 — Alignment, RLHF and DPO](../chapter-18-alignment-rlhf/index.md) and [Chapter 19 — Reasoning Models and Reinforcement Learning](../chapter-19-reasoning-rl/index.md)) is
marked optional in the chapter itself and can be skipped outright on this path.
[Chapter 22 — Operating and Improving an Open-Source LLM](../chapter-22-operate-improve-open-llm/index.md) describes itself as combining the methods of Chapters 17 to 21, but only
Chapter 17 is load-bearing there: alignment and reasoning appear as named
adaptation options, not as material you must have read.

Two ways through it. Take it as written if you want to have trained the thing you
operate; the pretraining stages are then Path 2's territory and worth the detour.
Or take it in its adapt-an-existing-model form: substitute a released open base
model for the tokenizer and pretraining stages and keep the evaluate, serve,
publish and iterate half, which is the part this path is actually about. The
budget, the plan, the cost record and the one measured change all survive the
substitution.

## The milestones that prove it

[Chapter 21 — Inference Optimization and Serving](../chapter-21-inference-serving/index.md) asks you to run a 7–8B model with llama.cpp at three quantization levels
and compare quality against memory and speed, which is the trade-off every
deployment decision comes down to. [Chapter 22 — Operating and Improving an Open-Source LLM](../chapter-22-operate-improve-open-llm/index.md) asks for a question-answering system
over at least 100 pages of documents built on a small open instruction-tuned
model: the smallest honest version of a production system.

## Estimated time

**14–21 weeks** at 10 hours per week for the three chapters, after the
prerequisites.

## Related paths

[Path 5](build-applications.md) shares Chapters 21 and 22 with this path and
differs only at the end: it finishes with the reading loop rather than the
capstone, and is the better choice when the model is a component of a product
rather than the product itself. [Path 3](improve-an-existing-model.md) is where
the "improve" half of Chapter 22 is treated in full.
