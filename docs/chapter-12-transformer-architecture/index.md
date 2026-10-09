# Chapter 12 — The Transformer Architecture, Deep Dive

> Part IV — Transformers and LLMs · 4–6 weeks

## What you will learn

This chapter describes the Transformer, the neural network architecture used by
almost all current large language models (LLMs), at the level of detail needed to
implement it. It covers scaled dot-product attention, causal masking, multi-head
attention, positional encodings, residual connections, layer normalization, and the
feed-forward layer, first in the encoder-decoder model of the original paper and then
in the decoder-only stack used by GPT-style models. It also summarizes the changes
that current open models make to the 2017 design — rotary position embeddings (RoPE),
root mean square normalization (RMSNorm), gated feed-forward layers (SwiGLU),
grouped-query attention (GQA), multi-head latent attention (MLA), and
mixture-of-experts (MoE) layers — because the model reports read from Chapter 15
onward assume them. The milestone implements multi-head attention; the tokenizer
follows in Chapter 13, and the full model with its training loop in Chapter 14.

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
- Changes in current open LLMs: SwiGLU, grouped-query attention (GQA), multi-head
  latent attention (MLA), and mixture-of-experts (MoE) layers (overview)

## Resources

**Suggested path.** Start with the two 3Blue1Brown videos and *The Illustrated
Transformer*, then watch CS224N 2024 lecture 8 for the derivation. Next, watch
Karpathy's "Let's build GPT" while typing the code, and work through Raschka Chapter
3 (or Prince Chapter 12, which is free) for a second implementation of attention.
Read "Attention Is All You Need" after that, with *Formal Algorithms for
Transformers* open as the reference for shapes and pseudocode. CS336 lectures 3 and 4
cover the changes made by current models and fit at the end; when time is short, omit
CS25, *The Annotated Transformer*, and the optional papers. An optional alternative
start is lecture 1 of CME295, a 1h45m survey of this chapter and of Chapter 11; the
rest of that course previews Chapters 13–22.

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
- Stanford — [CME295: Transformers & Large Language
  Models](https://cme295.stanford.edu/) by Afshine Amidi and Shervine Amidi (free;
  optional; nine lectures of about 1h45m, with slides and recordings and no
  assignments; the [Fall 2026 syllabus](https://cme295.stanford.edu/syllabus/) posts
  each lecture as the quarter runs, and the complete [Autumn 2025
  recordings](https://www.youtube.com/playlist?list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy)
  are on YouTube; a survey of Chapters 12–22 in one quarter — the Transformer, MoE,
  GQA, and RoPE, pretraining, SFT, LoRA, RLHF, and DPO, RL for reasoning, training
  and inference systems, agents, and evaluation — to watch before the deep dives or
  as a review after Chapter 22; the free
  [cheatsheet](https://github.com/afshinea/stanford-cme-295-transformers-large-language-models)
  (MIT license; 15 languages, Spanish included) is the written companion, and the
  authors' paid *Super Study Guide: Transformers & Large Language Models* is not
  required).

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
- Book: Dan Jurafsky and James H. Martin, *Speech and Language Processing* (3rd ed.
  draft, August 2026 release) — [official
  page](https://web.stanford.edu/~jurafsky/slp3/) (free PDF; Chapter 7 "Transformers
  and Pretraining", sections 7.1–7.5: attention, the Transformer block, the
  single-matrix form of the computation, token and position embeddings, and the
  language modeling head; sections 7.6–7.8, on decoding, pretraining, and current
  models, belong to Chapter 14).

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
- Paper: DeepSeek-AI, [DeepSeek-V2: A Strong, Economical, and Efficient
  Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) (2024;
  optional: introduces multi-head latent attention (MLA), which compresses the keys
  and values into a low-rank latent vector to shrink the key-value cache of Chapter
  21, and the MoE design inherited by DeepSeek-V3 (Chapter 15) and DeepSeek-R1
  (Chapter 19); Section 2 suffices).
- Article: Sebastian Raschka, [The Big LLM Architecture
  Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)
  (2025, updated 2026; the released open architectures side by side — DeepSeek V3,
  OLMo 2 and 3, Gemma 3 and 4, Llama 4, Qwen3, SmolLM3, Kimi K2, gpt-oss — with the
  design choices each one makes: MLA, sliding-window attention, MoE with shared
  experts, QK-norm, and NoPE; the reference to keep open when reading a model card).

## Milestone

Implement multi-head self-attention with a causal mask from scratch in PyTorch, copy
its weights into `nn.MultiheadAttention`, and verify that both modules produce the
same output to within 1e-5. Verify causality by changing the last input token and
checking that all earlier output positions are unchanged. Measure the size of the
attention matrix at sequence lengths 512 and 2,048, and explain in one paragraph why
time and memory grow as O(n²).

## Estimated time

4–6 weeks.
