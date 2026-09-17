# Chapter 14 — Build an LLM from Scratch

> Part IV — Transformers and LLMs · 6–8 weeks

## What you will learn

This chapter assembles the components of Chapters 12 and 13 into a complete
GPT-style large language model (LLM) and trains it. It covers the model definition
(embedding tables, a stack of Transformer blocks, and the output layer that predicts
the next token), the cross-entropy training objective, and a training loop with the
AdamW optimizer, learning-rate warmup and cosine decay, gradient clipping, and
periodic evaluation on held-out data. It then covers text generation with greedy
decoding, temperature, and top-k and top-p sampling. The implementation is verified
by loading the released GPT-2 weights into it, and the trained model is saved and
published on the Hugging Face Hub. Chapters 15 and 16 scale the same procedure up
in data, model size, and hardware.

## Topics

- The GPT model: token and positional embeddings, Transformer blocks, final
  normalization, and the output layer
- Parameter counting, weight tying between the embedding table and the output
  layer, and weight initialization
- The next-token prediction objective, the cross-entropy loss, and the
  training/validation split
- Batches of token sequences: context length and batch size
- The training loop: AdamW, weight decay, learning-rate warmup and cosine decay,
  and gradient clipping
- Monitoring of training and validation loss, overfitting, and generated samples
- Text generation: the autoregressive loop, greedy decoding, temperature, and top-k
  and top-p sampling
- Saving and loading checkpoints (model and optimizer state)
- Loading the released GPT-2 weights into the implementation as a correctness check
- Publishing a model on the Hugging Face Hub

## Resources

**Suggested path.** Raschka Chapters 4 and 5 with Appendix D are the main text of
this chapter: type and run the code chapter by chapter, with the companion videos as
a second explanation. Karpathy's "Let's build GPT" was watched in Chapter 12 and is
not repeated here; continue with the first section of "Let's reproduce GPT-2 (124M)"
for the model definition and the weight-loading check, and with its optimizer
section for AdamW, gradient clipping, and the learning-rate schedule, with nanoGPT
open as a reference implementation. Read Holtzman et al. and "How to generate
text" before writing the sampling code, the TinyStories paper before choosing the
milestone corpus, and Chapter 4 of the Hugging Face LLM Course before publishing
the checkpoint. CS336 lectures 2–3 and the Transformer part of Assignment 1 are a
stricter, test-driven route to the same result; when time is short, omit them
together with *Hands-On Large Language Models*, the GPT-3 paper, nanochat, and
llm.c.

### University courses

- Stanford — [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
  by Tatsunori Hashimoto and Percy Liang (free; Spring 2026
  [lecture videos](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)
  and assignments; start here: lecture 2 "PyTorch (einops), resource accounting" and
  lecture 3 "Architectures, hyperparameters", first assigned in Chapter 12; in
  [Assignment 1 (Basics)](https://github.com/stanford-cs336/assignment1-basics)
  the Transformer language model, cross-entropy loss, AdamW, training loop, and
  decoding are written without the ready-made `torch.nn` layers and checked by unit
  tests; the tokenizer part was covered in Chapter 13).

### Online courses (MOOCs)

- Andrej Karpathy — [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)
  (free; 4 hours; the sequel to Lecture 7 of [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html),
  with step-by-step code in [build-nanogpt](https://github.com/karpathy/build-nanogpt);
  start here: section 1, about the first 80 minutes, defines the GPT-2 model, loads
  the released weights, and writes the training loop; the section on AdamW,
  gradient clipping, and the learning-rate schedule starts at about 2h15m; the
  sections on speed and multi-GPU training belong to Chapter 16).
- Hugging Face — [LLM Course, Chapter 4: Sharing models and tokenizers](https://huggingface.co/learn/llm-course/chapter4/1)
  (free; about 2 hours; the Hugging Face Hub, uploading a checkpoint, and writing a
  model card, as required by the milestone; Chapter 2 "Using Transformers" covers
  loading and saving models).

### Books

- Book: Sebastian Raschka, *Build a Large Language Model (From Scratch)* (Manning,
  2024) — [official page](https://www.manning.com/books/build-a-large-language-model-from-scratch)
  (paid; free code at [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
  and free [companion videos](https://www.youtube.com/playlist?list=PLTKMiZHVd_2IIEsoJrWACkIxLRdfMlw11);
  start here: Chapter 4 "Implementing a GPT Model from Scratch to Generate Text",
  Chapter 5 "Pretraining on Unlabeled Data", which includes decoding strategies and
  loading the GPT-2 weights, and Appendix D on warmup, cosine decay, and gradient
  clipping; Chapters 2 and 3 were used in Chapters 13 and 12).
- Book: Jay Alammar and Maarten Grootendorst, *Hands-On Large Language Models*
  (O'Reilly, 2024) — [official page](https://llm-book.com/) (paid; optional; free
  notebooks on [GitHub](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models);
  Chapter 3 "Looking Inside Transformer LLMs" follows one forward pass of a released
  model, including the decoding strategy and the key-value cache).

### Lectures, papers and articles

- Paper: Radford et al., [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
  (GPT-2, 2019; start here: the report of the model reproduced in this chapter;
  Section 2.3 lists its changes to the original Transformer).
- Paper: Brown et al., [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
  (GPT-3, 2020; optional: the same design scaled to 175B parameters; Table 2.1
  lists model sizes, batch sizes, and learning rates that later work reuses, and
  Appendix B gives the optimizer settings).
- Paper: Holtzman et al., [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
  (2019; shows that greedy and beam search produce repetitive text, and introduces
  top-p (nucleus) sampling).
- Article: Patrick von Platen, [How to generate text: using different decoding methods for language generation with Transformers](https://huggingface.co/blog/how-to-generate)
  (Hugging Face blog, 2020; greedy search, beam search, and top-k and top-p
  sampling compared in code).
- Paper: Eldan and Li, [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759)
  (2023; a synthetic corpus of short stories on which models below 10M parameters
  produce fluent text; the [dataset](https://huggingface.co/datasets/roneneldan/TinyStories)
  makes the milestone feasible on a single GPU).

### Tools and hands-on

- Code: Andrej Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT) (start
  here: a GPT model definition of about 300 lines and a training loop of about 300
  lines; the README now points to nanochat as the successor, but the code remains
  a compact reference for this chapter).
- Code: Andrej Karpathy, [nanochat](https://github.com/karpathy/nanochat) (2025;
  advanced: tokenizer, pretraining, fine-tuning, evaluation, and inference for a
  small chat model, designed to run on one node of eight H100 GPUs).
- Code: Andrej Karpathy, [llm.c](https://github.com/karpathy/llm.c) (2024;
  optional: GPT-2 training written in plain C and CUDA without PyTorch; shows every
  forward and backward kernel explicitly).

## Milestone

Load the released GPT-2 (124M) weights into the from-scratch implementation and
verify that its logits match those of the Hugging Face `GPT2LMHeadModel` on the same
input. Then train a decoder-only model of 10–50M parameters on a public corpus such
as TinyStories, plot the training and validation loss, and compare samples generated
with greedy decoding, temperature, top-k sampling, and top-p sampling. Publish the
checkpoint on the Hugging Face Hub with a model card that states the data, the
hyperparameters, and the final validation loss.

## Estimated time

6–8 weeks.
