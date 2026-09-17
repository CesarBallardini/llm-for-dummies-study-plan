# Path 5 — Build applications on open models

**Goal.** Finish able to ship something people use on top of an open model rather
than a vendor API. Concretely: run the model yourself and know what it costs per
million tokens; choose a base model on size, licence, context length and serving
cost rather than on a leaderboard position; design around the context window and
the tokenizer instead of being surprised by them; add retrieval when the model
needs facts it was never trained on; put guardrails and monitoring around it;
evaluate the application rather than the model; and keep the stack current as
better open models are released every few months.

This is the shortest path in the book, and the one that leans hardest on [Chapter 24 — The Research Reading Loop](../chapter-24-research-reading-loop/index.md):
an application built on a six-month-old open model is usually one release behind
a cheaper option.

## Prerequisites

- [Chapter 7 — GPUs, Hardware and the Training Environment](../chapter-07-gpus-hardware/index.md) — *compute-bound, memory-bound and overhead-bound workloads; VRAM and memory bandwidth; bytes per parameter in fp32, fp16, bf16 and int8; CPU against GPU for running a model, and the cases in which a laptop is enough*. Essential. Chapter 21 defines prefill and decode in these workload classes, and the last section is what decides whether your application needs a GPU at all.
- [Chapter 6 — Python and the Scientific Stack](../chapter-06-python-scientific-stack/index.md) — *virtual environments, the command line, Git; NumPy and PyTorch basics*. Essential: shipping an application is ordinary software engineering, and this is where the book covers the tooling.
- [Chapter 13 — Tokens and Embeddings](../chapter-13-tokenization-embeddings/index.md) — *vocabulary size and compression ratio; the dependence of a tokenizer on its corpus and the cost differences between languages; model failures caused by tokenization; the token embedding table and the context window*. Essential: the context window shapes the product, tokens are what you are billed in, and the tokenization failures section explains a whole class of bug reports.
- [Chapter 20 — Evaluation and Benchmarks](../chapter-20-evaluation-benchmarks/index.md) — *perplexity; scoring methods; evaluation settings that change scores; statistical uncertainty; LLM-as-judge and its failure modes*. Essential in practice. Chapter 22 selects a base model on benchmark results and evaluates a retrieval system, and the judge section is what you will use to evaluate your own application.
- [Chapter 12 — The Transformer Architecture](../chapter-12-transformer-architecture/index.md) — *scaled dot-product attention; multi-head and grouped-query attention; the quadratic cost of attention in the sequence length*. Recommended, three sections only: they are what the KV cache of Chapter 21 is, and why a long context costs what it does.
- [Chapter 16 — Distributed Training and Systems](../chapter-16-distributed-training-systems/index.md) — *mixed-precision formats; attention kernels that minimize memory traffic (FlashAttention); tensor parallelism*. Recommended, three sections only. Chapter 21 uses FlashAttention by name in prefill and decode, and tensor parallelism when a model is served across several GPUs.
- [Chapter 17 — Fine-Tuning and Parameter-Efficient Methods](../chapter-17-finetuning/index.md) — *choosing between fine-tuning, prompting and retrieval; chat templates and special tokens; LoRA*. Recommended: Chapter 22 treats these as adaptation options and fine-tuning is often cheaper than prompt engineering for a narrow task.
- [Chapter 9 — Deep Learning: Concepts and Practice](../chapter-09-deep-learning/index.md) — *`nn.Module`, autograd, the training loop*. Recommended: enough PyTorch to read what a serving stack is doing.

## The sequence

| Step | Chapter | Why this path needs it | Time |
| --- | --- | --- | --- |
| 1 | [Chapter 21 — Inference Optimization and Serving](../chapter-21-inference-serving/index.md) | Run the model yourself: quantization, serving stacks, batching, latency, cost per token. The engineering floor under any application. | 4–5 weeks |
| 2 | [Chapter 22 — Operating and Improving an Open-Source LLM](../chapter-22-operate-improve-open-llm/index.md) | The application chapter: choosing a base model, prompt design, retrieval-augmented generation, agents, guardrails, versioning, monitoring, and turning logged failures into evaluation cases. | 6–8 weeks |
| 3 | [Chapter 24 — The Research Reading Loop](../chapter-24-research-reading-loop/index.md) | The reading loop, which for this path is what keeps the model choice and the serving stack from going stale. | ongoing |

[Chapter 22 — Operating and Improving an Open-Source LLM](../chapter-22-operate-improve-open-llm/index.md) describes itself as combining the methods of Chapters 17 to 21. Of those,
only [Chapter 17 — Fine-Tuning and Parameter-Efficient Methods](../chapter-17-finetuning/index.md) is load-bearing and it is listed above; alignment ([Chapter 18 — Alignment, RLHF and DPO](../chapter-18-alignment-rlhf/index.md)) and
reasoning ([Chapter 19 — Reasoning Models and Reinforcement Learning](../chapter-19-reasoning-rl/index.md)) appear in that chapter as named adaptation options rather than
as material you need to have read first.

## The optional finisher

[Chapter 23 — Capstone: Train, Ship and Improve Your Own Model](../chapter-23-capstone/index.md) is the capstone, and it is worth knowing what it actually is before
adding it to this path: a **training** project. Its milestone trains a tokenizer,
pretrains a model, fine-tunes it with SFT, evaluates, serves and publishes it
within a budget of $15–$100, which brings [Chapter 13 — Tokens and Embeddings](../chapter-13-tokenization-embeddings/index.md) through [Chapter 17 — Fine-Tuning and Parameter-Efficient Methods](../chapter-17-finetuning/index.md) with it — several
months of work that an application does not require.

Take it if you want to own the weights you ship. If you only want to ship the
application, substitute "adapt an existing open model" for the pretraining stages
and you have [Chapter 22 — Operating and Improving an Open-Source LLM](../chapter-22-operate-improve-open-llm/index.md)'s cycle, which is already in this path's core. Budget a
further 4–8 weeks if you take the capstone as written.

## The milestone that proves it

The [Chapter 22 — Operating and Improving an Open-Source LLM](../chapter-22-operate-improve-open-llm/index.md) milestone is this path's: a question-answering system over a document
collection of at least 100 pages, built on an open instruction-tuned model of 1–3
billion parameters. Pair it with the [Chapter 21 — Inference Optimization and Serving](../chapter-21-inference-serving/index.md) milestone — the same 7–8B model run at
three quantization levels, with quality plotted against speed — which is the
measurement that tells you what your application will cost to run.

## Estimated time

**10–13 weeks** at 10 hours per week for Chapters 21 and 22, plus [Chapter 24 — The Research Reading Loop](../chapter-24-research-reading-loop/index.md),
which is ongoing, and 4–8 more if you take the [Chapter 23 — Capstone: Train, Ship and Improve Your Own Model](../chapter-23-capstone/index.md) capstone.

## Related paths

[Path 4](serve-and-operate.md) shares Chapters 21 and 22 and continues into the
capstone instead of the reading loop: take it if running the model *is* the job.
[Path 3](improve-an-existing-model.md) is the route to take when prompting and
retrieval stop being enough and the model itself has to change.
