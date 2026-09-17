# Path 1 — Understand how LLMs work internally

**Goal.** Finish able to take any open model's card and architecture diagram and
say what each part does and why it is there. Concretely: trace a prompt from raw
characters through the tokenizer, the embedding table, the stack of Transformer
blocks and the final softmax to a sampled token; explain what the KV cache holds
and why it grows with context length; read a training run's hyperparameters and
say what each one controls; and use a scaling law to explain why a model of a
given size was trained on the number of tokens it was.

This path is about understanding rather than production. It ends with a model you
built yourself that reproduces GPT-2's outputs, which is a stronger test of
understanding than any amount of reading.

## Prerequisites

This path sits in the middle of the book and assumes the chapters before it,
particularly:

- [Chapter 12 — The Transformer Architecture](../chapter-12-transformer-architecture/index.md) — *the whole chapter*. This is the architecture you are about to build; nothing in it is surplus to this path.
- [Chapter 10 — Neural Networks from Scratch](../chapter-10-neural-networks-from-scratch/index.md) — *computational graphs and reverse-mode automatic differentiation; the hand-written backward pass verified against PyTorch autograd; vanishing and exploding gradients*. Without these the training loop in Chapter 14 stays a black box that either works or does not.
- [Chapter 9 — Deep Learning: Concepts and Practice](../chapter-09-deep-learning/index.md) — *`nn.Module`, `Dataset` and `DataLoader`, autograd; the training loop; softmax with cross-entropy; layer normalization and residual connections*. The PyTorch vocabulary Chapter 14 is written in. The convolutional and vision material of Chapter 9 is not needed here.
- [Chapter 7 — GPUs, Hardware and the Training Environment](../chapter-07-gpus-hardware/index.md) — *VRAM and bytes per parameter in fp32, fp16 and bf16; the introduction to training memory (weights, gradients, optimizer states, activations)*. Chapters 15 and 16 are an extended argument about memory, and these two sections are the terms it is conducted in.
- [Chapter 4 — Probability and Statistics](../chapter-04-probability-statistics/index.md) — *entropy, cross-entropy, KL divergence and perplexity*. The loss you will be reading is a cross-entropy, and the number reported at the end of a run is a perplexity.

[Chapter 11 — NLP Foundations and Word Representations](../chapter-11-nlp-foundations/index.md) is worth reading for context — perplexity, word embeddings and
pre-Transformer attention all reappear later — but it is not on the critical
path. [Path 0](foundations-first.md) covers everything earlier.

## The sequence

| Step | Chapter | Why this path needs it | Time |
| --- | --- | --- | --- |
| 1 | [Chapter 13 — Tokens and Embeddings](../chapter-13-tokenization-embeddings/index.md) | How text becomes numbers: byte-level BPE, vocabularies, embedding tables. Most surprising model behaviour starts here. | 3–4 weeks |
| 2 | [Chapter 14 — Build an LLM from Scratch](../chapter-14-llm-from-scratch/index.md) | Assemble Chapters 12 and 13 into a complete GPT-style model and train it. The centre of this path. | 6–8 weeks |
| 3 | [Chapter 15 — Pretraining at Scale](../chapter-15-pretraining-at-scale/index.md) | Why a released model differs from the one you just built: data mixtures, scaling laws, token budgets and the cost of a real run. | 6–8 weeks |
| 4 | [Chapter 16 — Distributed Training and Systems](../chapter-16-distributed-training-systems/index.md) | How a model too large for one GPU is trained on many: data, tensor and pipeline parallelism, ZeRO, mixed precision, and what actually makes a run fast. | 6–8 weeks |

## The milestone that proves it

[Chapter 14 — Build an LLM from Scratch](../chapter-14-llm-from-scratch/index.md) asks you to load the released GPT-2 (124M) weights into your own
from-scratch implementation and verify that its logits match Hugging Face's
`GPT2LMHeadModel` on the same input. If your implementation reproduces a real
model's numbers, you have understood the architecture — there is nowhere left for
a misunderstanding to hide.

## What this path leaves out

Post-training ([Chapter 17 — Fine-Tuning and Parameter-Efficient Methods](../chapter-17-finetuning/index.md) to [Chapter 19 — Reasoning Models and Reinforcement Learning](../chapter-19-reasoning-rl/index.md)) and serving ([Chapter 21 — Inference Optimization and Serving](../chapter-21-inference-serving/index.md), [Chapter 22 — Operating and Improving an Open-Source LLM](../chapter-22-operate-improve-open-llm/index.md)) are not needed to
understand how a model works internally. Come back for them through
[Path 3](improve-an-existing-model.md) or [Path 4](serve-and-operate.md) when you
want to change or run a model rather than explain it.

## Estimated time

**21–28 weeks** at 10 hours per week for the four chapters of the path, after
the prerequisites.
