# Chapter 17 — Fine-Tuning and Parameter-Efficient Methods

> Part V — Post-Training · 4–6 weeks

## What you will learn

This chapter covers the adaptation of a pretrained open large language model (LLM)
to follow instructions or to perform a single task. It treats supervised
fine-tuning (SFT): training on prompts paired with reference responses, formatted
with the chat template of the model, and the evidence that a few thousand curated
examples can match a much larger unfiltered set. It explains why full fine-tuning
requires graphics processing unit (GPU) memory equal to several times the size of
the model, and how low-rank adaptation (LoRA), which trains small low-rank
matrices added to frozen weights, and quantized LoRA (QLoRA), which applies LoRA
to a base model quantized to 4 bits, reduce that requirement to the memory of one
consumer GPU. It also covers the criteria for choosing fine-tuning over prompting
or retrieval, and the measurement of both the task performance gained and the
general capability lost. Alignment (Chapter 18), reasoning (Chapter 19), and the
capstone (Chapter 23) start from a model fine-tuned with these methods.

## Topics

- What fine-tuning changes: continued pretraining, SFT, and instruction tuning
- Choosing between fine-tuning, prompting, and retrieval-augmented generation
  (RAG, Chapter 22)
- Instruction datasets: Alpaca and chat-message formats, synthetic data, and data
  quality versus dataset size
- Chat templates and special tokens
- The SFT loss: next-token prediction computed on the response tokens only, and
  sequence packing
- Full fine-tuning and its memory cost (weights, gradients, optimizer states)
- Parameter-efficient fine-tuning (PEFT): freezing layers, adapters, and prefix
  and prompt tuning
- LoRA: low-rank updates, the choice of layers to adapt, and merging the adapter
  into the base weights
- LoRA hyperparameters (rank, alpha, learning rate, dropout)
- QLoRA: LoRA on a 4-bit quantized base model (quantization is treated in full in
  Chapter 21)
- Training on consumer hardware: mixed precision, gradient accumulation, and
  gradient checkpointing
- Overfitting, catastrophic forgetting, and evaluation before and after
  fine-tuning

## Resources

**Suggested path.** Start with chapter 11 of the Hugging Face LLM Course (a few
evenings), then implement the same components from scratch with chapters 6–7 and
appendix E of *Build a Large Language Model (From Scratch)*, the main text of this
chapter. Read the LoRA and QLoRA papers alongside appendix E, and "LoRA Without
Regret" before choosing hyperparameters for the milestone; CS336 lecture 15 fits
at any point and covers instruction data and SFT practice at research-lab scale.
When time is short, skip chapter 6 of *Build a Large Language Model (From
Scratch)*, the FLAN and LIMA papers, and the other three books, and complete the
milestone with the Hugging Face Transformer Reinforcement Learning (TRL) library
or Unsloth, starting from the LLM Course notebooks.

### University courses

- Stanford —
  [CS336 Language Modeling from Scratch (Spring 2026)](https://cs336.stanford.edu/)
  by Tatsunori Hashimoto and Percy Liang (free;
  [videos](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV) +
  assignments; start here: lecture 15, "Mid/post-training (SFT/RLHF)", first half,
  on instruction data and SFT practice; advanced: the SFT part of
  [assignment 5](https://github.com/stanford-cs336/assignment5-alignment), which
  fine-tunes a model on math problems and is sized for a data-center GPU).

### Online courses (MOOCs)

- Hugging Face —
  [LLM Course, Chapter 11: Supervised Fine-Tuning](https://huggingface.co/learn/llm-course/chapter11/1)
  (free; a few hours; start here: four sections on chat templates, SFT with the
  TRL `SFTTrainer`, LoRA, and evaluation, each with a notebook).
- Hugging Face — [smol course](https://huggingface.co/learn/smol-course/unit0/1)
  (free; v2, 2025; self-paced, based on SmolLM3 and TRL; unit 1, "Instruction
  Tuning", provides further practice with chat templates and SFT, with graded
  exercises and a free certificate).

### Books

- Book: Sebastian Raschka, *Build a Large Language Model (From Scratch)* (Manning, 2024)
  — [official
  page](https://www.manning.com/books/build-a-large-language-model-from-scratch)
  (paid; free code at
  [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch); start here:
  chapter 6, "Fine-tuning for classification", chapter 7, "Fine-tuning to follow
  instructions", and appendix E, which implements LoRA from scratch, all on the GPT-2
  model built in Chapter 14).
- Book: Jay Alammar and Maarten Grootendorst, *Hands-On Large Language Models*
  (O'Reilly, 2024) — [official page](https://llm-book.com/) (paid; free notebooks on
  [GitHub](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models); chapter 12,
  "Fine-Tuning Generation Models", demonstrates QLoRA instruction tuning of TinyLlama
  with the Hugging Face libraries, followed by preference tuning, which belongs to
  Chapter 18).
- Book: Chip Huyen, *AI Engineering* (O'Reilly, 2025) —
  [book repo](https://github.com/chiphuyen/aie-book) (paid; free chapter
  summaries; chapter 7, "Finetuning", covers the choice between fine-tuning,
  prompting, and RAG, the memory arithmetic, PEFT, and model merging; chapter 8,
  "Dataset Engineering", covers curating and synthesizing training data).
- Book: Nathan Lambert, *Reinforcement Learning from Human Feedback* (Manning,
  2026) — [official page](https://rlhfbook.com/) (free online, paid print; chapter
  4, "Instruction Fine-Tuning", is a short account of chat templates, prompt
  masking, and SFT practice in current post-training pipelines; the rest of the
  book belongs to Chapter 18).

### Lectures, papers and articles

- Paper: Edward J. Hu et al.,
  [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
  (2021; start here: defines the low-rank update used by most fine-tunes of open
  models; read sections 1–4).
- Paper: Tim Dettmers et al.,
  [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
  (2023; LoRA on a base model quantized to the 4-bit NormalFloat (NF4) format,
  with paged optimizers; reduces the memory needed to fine-tune a
  7-billion-parameter model to that of one consumer GPU).
- Article: John Schulman and Thinking Machines Lab,
  [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) (2025; the
  conditions under which LoRA matches full fine-tuning: adapters on all layers,
  including the multilayer perceptron (MLP) layers, a rank sufficient for the
  dataset size, and a learning rate about 10 times that of full fine-tuning).
- Paper: Dan Biderman et al.,
  [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673) (2024;
  measures the target-domain performance gained by LoRA and by full fine-tuning,
  and the base-model capability that each loses).
- Paper: Jason Wei et al.,
  [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652)
  (2021, FLAN; shows that instruction tuning on many tasks improves zero-shot
  performance on unseen tasks).
- Paper: Chunting Zhou et al.,
  [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) (2023;
  fine-tunes a 65-billion-parameter base model on 1,000 curated examples and
  obtains assistant-style answers of competitive quality; the standard reference
  for data quality versus dataset size).

### Tools and hands-on

- Hugging Face — [PEFT](https://huggingface.co/docs/peft/index) and
  [TRL](https://huggingface.co/docs/trl/index) documentation (free, open source;
  start here: the libraries most widely used for LoRA and QLoRA adapters (PEFT)
  and for SFT (TRL); the PEFT conceptual guides also cover adapters, prefix and
  prompt tuning, and LoRA variants such as weight-decomposed low-rank adaptation
  (DoRA); the [`SFTTrainer`](https://huggingface.co/docs/trl/sft_trainer) page
  documents the trainer used in the milestone).
- [Unsloth](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide),
  [Axolotl](https://github.com/axolotl-ai-cloud/axolotl), and
  [LlamaFactory](https://github.com/hiyouga/LlamaFactory) (free, open source;
  optional: Unsloth provides LoRA and QLoRA training with reduced memory use and
  training time, and ready-made Colab notebooks; Axolotl and LlamaFactory
  configure a complete fine-tuning run from one configuration file).
- Hugging Face — [SmolLM](https://github.com/huggingface/smollm) (free, open
  weights; the SmolLM2 models with 135M, 360M, and 1.7B parameters and the 3B
  SmolLM3 model can be fine-tuned on a free-tier GPU; the training recipes are
  published).

## Milestone

Fine-tune a small base model with LoRA (SmolLM2-360M, or SmolLM2-135M on a slow
GPU; both fit a free Colab or Kaggle T4 GPU) on a few thousand examples of an open
instruction dataset, formatted with a chat template. Report, before and after
fine-tuning: the loss on a held-out slice of the dataset, a blind pass/fail
grading by hand of the answers to 30 held-out prompts, and, as a check on
catastrophic forgetting, the perplexity on plain text such as WikiText-2. Repeat
the run once with a different rank or learning rate and explain the difference.

## Estimated time

4–6 weeks.
