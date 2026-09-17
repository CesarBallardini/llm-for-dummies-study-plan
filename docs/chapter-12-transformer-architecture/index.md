# Chapter 12 — The Transformer Architecture, Deep Dive

> Part IV — Transformers and LLMs · 4–6 weeks

## What you will learn

This chapter describes the Transformer, the neural network architecture used by
almost all current large language models (LLMs), at the level of detail needed to
implement it. It covers scaled dot-product attention, causal masking, multi-head
attention, positional encodings, residual connections, layer normalization, and the
feed-forward layer, first in the encoder-decoder model of the original paper and
then in the decoder-only stack used by GPT-style models. It also summarizes the
changes that current open models make to the 2017 design — rotary position
embeddings (RoPE), root mean square normalization (RMSNorm), gated feed-forward
layers (SwiGLU), grouped-query attention (GQA), and mixture-of-experts (MoE)
layers — because the model reports read from Chapter 15 onward assume them. The
milestone implements multi-head attention; the tokenizer follows in Chapter 13, and
the full model with its training loop in Chapter 14.

## Topics

- The limits of recurrence and the design goals of "Attention Is All You Need"
- Scaled dot-product attention: queries, keys, values, and the scaling factor
- Self-attention, cross-attention, and causal masking
- Multi-head attention and its tensor shapes
- Positional encodings: sinusoidal, learned absolute, and rotary (RoPE)
- The Transformer block: residual connections, normalization (LayerNorm and
  RMSNorm, pre-norm and post-norm placement), and the feed-forward layer
- Encoder-only (BERT), encoder-decoder (T5), and decoder-only (GPT) models, and the
  reasons LLMs use the decoder-only design
- Parameter counts, and the quadratic time and memory cost of attention in the
  sequence length
- Changes in current open LLMs: SwiGLU, grouped-query attention (GQA), and
  mixture-of-experts (MoE) layers (overview)

## Resources

**Suggested path.** Start with the two 3Blue1Brown videos and *The Illustrated
Transformer*, then watch CS224N 2024 lecture 8 for the derivation. Next, watch
Karpathy's "Let's build GPT" while typing the code, and work through Raschka
Chapter 3 (or Prince Chapter 12, which is free) for a second implementation of
attention. Read "Attention Is All You Need" after that, with *Formal Algorithms for
Transformers* open as the reference for shapes and pseudocode. CS336 lectures 3 and
4 cover the changes made by current models and fit at the end; when time is short,
omit CS25, *The Annotated Transformer*, and the optional papers.

### University courses

- Stanford — [CS224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/)
  by Diyi Yang and Yejin Choi (free; Winter 2026 slides, notes, and assignments +
  [Spring 2024 lecture videos](https://www.youtube.com/playlist?list=PLoROMvodv4rOaMFbaqxPDoLWjDaRAdP9D)
  by Christopher Manning; start here: 2024 lecture 8 "Transformers" derives
  self-attention from the limits of the recurrent neural networks (RNNs) of
  Chapter 11, and lecture 9 "Pretraining" compares encoder, encoder-decoder, and
  decoder models; Assignment 3 covers self-attention and Transformers).
- Stanford — [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
  by Tatsunori Hashimoto and Percy Liang (free; Spring 2026
  [lecture videos](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)
  and assignments; advanced: lecture 3 "Architectures, hyperparameters" and lecture 4
  "Attention alternatives and mixture of experts" survey the design choices of
  current LLMs; Assignment 1 is scheduled in Chapters 13 and 14).
- Stanford — [CS25: Transformers United](https://web.stanford.edu/class/cs25/)
  (free seminar; optional; sixth edition in Spring 2026; the
  [recordings on YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM)
  include the overview lectures, one of them by Andrej Karpathy, followed by guest
  talks on current research).

### Online courses (MOOCs)

- Andrej Karpathy — [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)
  (free; 1h56m; Lecture 7 of [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html),
  after the six lectures used in Chapter 10; start here: builds a character-level,
  decoder-only Transformer on Tiny Shakespeare in about 200 lines of PyTorch;
  Chapter 14 reuses this code and does not require a second viewing).
- DeepLearning.AI — [How Transformer LLMs Work](https://www.deeplearning.ai/short-courses/how-transformer-llms-work/)
  by Jay Alammar and Maarten Grootendorst (free; 1h44m; the lessons from
  "Architectural Overview" to "Mixture of Experts" cover the Transformer block,
  self-attention, the key-value cache, grouped-query attention, and MoE; the
  tokenizer lessons belong to Chapter 13).
- Hugging Face — [LLM Course, Chapter 1: Transformer models](https://huggingface.co/learn/llm-course/chapter1/4)
  (free; about 2 hours; sections 4–6 describe the encoder-only, decoder-only, and
  encoder-decoder families and the tasks each is used for).

### Books

- Book: Sebastian Raschka, *Build a Large Language Model (From Scratch)* (Manning,
  2024) — [official page](https://www.manning.com/books/build-a-large-language-model-from-scratch)
  (paid; free code at [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
  and free [companion videos](https://www.youtube.com/playlist?list=PLTKMiZHVd_2IIEsoJrWACkIxLRdfMlw11);
  start here: Chapter 3 "Coding Attention Mechanisms", from simplified
  self-attention to causal multi-head attention; Chapter 4 is used in Chapter 14).
- Book: Simon J.D. Prince, *Understanding Deep Learning* (MIT Press, 2023) —
  [official page](https://udlbook.github.io/udlbook/) (free PDF; Chapter 12
  "Transformers", with notebooks on self-attention, multi-head attention, and
  tokenization).

### Lectures, papers and articles

- Article: Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
  (2018; start here: an illustrated walkthrough of the original encoder-decoder
  model; the sequel [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)
  covers the decoder-only stack and masked self-attention).
- Video: 3Blue1Brown, [Transformers, the tech behind LLMs](https://www.3blue1brown.com/lessons/gpt)
  and [Attention in transformers, step-by-step](https://www.3blue1brown.com/lessons/attention)
  (2024; Chapters 5 and 6 of the Deep Learning series; about 50 minutes in total;
  animated explanations of embeddings, queries, keys, values, and masking).
- Paper: Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  (2017; the original paper; Section 3 defines the architecture).
- Article: Sasha Rush et al., Harvard NLP, [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
  (2018, updated 2022; the paper re-implemented section by section in PyTorch).
- Paper: Phuong and Hutter, [Formal Algorithms for Transformers](https://arxiv.org/abs/2207.09238)
  (2022; pseudocode with explicit tensor shapes for every component; a reference
  to keep open while coding).
- Paper: Devlin et al., [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
  (2018; the encoder-only design, for comparison with the decoder-only GPT-2
  report read in Chapter 14).
- Paper: Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
  (2021; RoPE, the positional encoding used by most current open LLMs).
- Paper: Xiong et al., [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
  (2020) and Zhang and Sennrich, [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
  (2019; optional: the reasons current models use pre-norm placement and RMSNorm).

## Milestone

Implement multi-head self-attention with a causal mask from scratch in PyTorch, copy
its weights into `nn.MultiheadAttention`, and verify that both modules produce the
same output to within 1e-5. Verify causality by changing the last input token and
checking that all earlier output positions are unchanged. Measure the size of the
attention matrix at sequence lengths 512 and 2,048, and explain in one paragraph why
time and memory grow as O(n²).

## Estimated time

4–6 weeks.
