# Chapter 15 — Pretraining at Scale: Data, Scaling Laws, and Training Runs

> Part IV — Transformers and LLMs · 6–8 weeks

## What you will learn

This chapter covers the differences between the small model of Chapter 14 and a
released open large language model (LLM): the data, the model size, and the length
and cost of the training run. It describes how a pretraining corpus is built from
web crawls and curated sources by text extraction, filtering, deduplication, and
mixing, and how small ablation models are used to choose among data recipes. It
presents the scaling laws of Kaplan et al. and of Hoffmann et al. (Chinchilla),
which relate the loss to parameters, tokens, and compute, and the estimate C ≈ 6ND,
which converts a parameter count N and a token count D into floating-point
operations (FLOPs), GPU-hours, and cost. It also covers the configuration of a long
training run — batch size, learning-rate schedule, checkpointing, and recovery from
loss spikes — as documented in the reports of open models such as OLMo 2, Llama 3,
and SmolLM3.

## Topics

- Sources of pretraining text: Common Crawl, code, books, papers, and curated
  corpora
- Legal and privacy constraints on training data: licenses, copyright, and removal
  of personal information
- Corpus construction: text extraction, language identification, quality filtering,
  deduplication, and removal of benchmark data (decontamination)
- Open corpora: The Pile, RefinedWeb, Dolma, FineWeb, and FineWeb-Edu
- Data mixtures, synthetic data, and ablations on small models to choose among them
- Training compute: C ≈ 6ND, FLOPs, GPU-hours, model FLOPs utilization (MFU), and
  cost
- Scaling laws (Kaplan, Chinchilla) and compute-optimal training
- Token budgets in practice: training beyond the compute-optimal point to reduce
  inference cost
- Hyperparameters at scale: batch size, learning rate, and schedules (cosine,
  warmup-stable-decay)
- Training stability: loss spikes, their mitigations, checkpointing, and restarts
- The effect of data quality on downstream evaluation results

## Resources

**Suggested path.** Start with *The Smol Training Playbook*, which follows one open
training run from the decision to pretrain through ablations, data mixing, and
failure recovery; it is book-length, so read it over several weeks alongside the
other material. In parallel, watch CS336 lectures 9 and 11 with the Kaplan and
Chinchilla papers, then lectures 13 and 14 with the FineWeb paper; *Transformer
Math 101* supplies the compute formulas for the milestone, and the final part of
Karpathy's "Let's reproduce GPT-2 (124M)", listed in Chapter 14, shows a complete
run on 10B tokens of FineWeb-Edu. Of the model reports, read at least the
pretraining sections of OLMo 2 and Llama 3. When time is short, omit CS324, the
DeepLearning.AI short course, Raschka's bonus material, The Pile, RefinedWeb,
Dolma, and DeepSeek-V3.

### University courses

- Stanford — [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
  by Tatsunori Hashimoto and Percy Liang (free; Spring 2026
  [lecture videos](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)
  and assignments; start here: lectures 9 and 11 "Scaling laws", lecture 13 "Data
  (sources, datasets)", and lecture 14 "Data (filtering, deduplication, mixing,
  synthetic data)"; lecture 2, first assigned in Chapter 14, supplies the FLOPs
  accounting; [Assignment 3 (Scaling)](https://github.com/stanford-cs336/assignment3-scaling)
  has instructions for non-students, because its hosted training API is reserved
  for enrolled students; the filtering and deduplication parts of
  [Assignment 4 (Data)](https://github.com/stanford-cs336/assignment4-data) run
  offline, and its final 8-GPU training run can be omitted).
- Stanford — [CS324: Large Language Models](https://stanford-cs324.github.io/winter2022/)
  by Percy Liang, Tatsunori Hashimoto, and Christopher Ré (free; Winter 2022
  lecture notes; optional: the notes on Data, Training, and Scaling laws are a
  short written summary of the same topics, without the developments after 2022).

### Online courses (MOOCs)

- DeepLearning.AI and Upstage — [Pretraining LLMs](https://www.deeplearning.ai/short-courses/pretraining-llms/)
  by Sung Kim and Lucy Park (free; 1h29m; optional: notebooks on data preparation,
  packing of token sequences, model initialization, a short training run, and
  evaluation).

### Books

- Book: Sebastian Raschka, *Build a Large Language Model (From Scratch)* (Manning,
  2024) — [official page](https://www.manning.com/books/build-a-large-language-model-from-scratch)
  (paid; free code at [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
  and free [companion videos](https://www.youtube.com/playlist?list=PLTKMiZHVd_2IIEsoJrWACkIxLRdfMlw11);
  optional: the pretraining chapter of the book was completed in Chapter 14; the
  repository's bonus material [Pretraining GPT on the Project Gutenberg Dataset](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/03_bonus_pretraining_on_gutenberg)
  extends the Chapter 5 training loop to a multi-gigabyte corpus).

### Lectures, papers and articles

- Article: Loubna Ben Allal et al. (Hugging Face), [The Smol Training Playbook: The Secrets to Building World-Class LLMs](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook)
  (2025; start here: a book-length account of training SmolLM3, a 3B-parameter
  model, on 11T tokens: whether to pretrain at all, ablations, architecture, data
  mixing, loss spikes, and infrastructure failures).
- Article: Quentin Anthony, Stella Biderman, and Hailey Schoelkopf,
  [Transformer Math 101](https://blog.eleuther.ai/transformer-math/) (EleutherAI,
  2023; first used in Chapter 7; the sections "Compute Requirements" and "Parameter
  vs Dataset Tradeoffs" give C ≈ 6PD, in which P is the parameter count, and its
  use for cost estimates; the sections on training memory and distributed training
  are used in Chapter 16).
- Paper: Kaplan et al., [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
  (2020; power laws relating the loss of Transformer language models to
  parameters, data, and compute).
- Paper: Hoffmann et al., [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
  (Chinchilla, 2022; revises Kaplan et al.: parameters and tokens are scaled
  together, at roughly 20 tokens per parameter; Section 3.3 gives the parametric
  loss fit used in the milestone).
- Paper: Gao et al., [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027)
  (2020; an open corpus assembled from 22 curated sources, on which the GPT-NeoX
  and Pythia models were trained).
- Paper: Lee et al., [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)
  (2021; exact-substring and near-duplicate (MinHash) deduplication, and their
  effect on memorization and evaluation).
- Paper: Penedo et al., [The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116)
  (2023; reports that models trained on filtered and deduplicated web data alone
  match models trained on curated corpora).
- Paper: Penedo et al., [The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale](https://arxiv.org/abs/2406.17557)
  (2024; the pipeline, with an ablation for each step, behind the open
  [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) corpus and its
  classifier-filtered subset FineWeb-Edu).
- Paper: Soldaini et al., [Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research](https://arxiv.org/abs/2402.00159)
  (2024; a documented data pipeline, including the removal of personal
  information, released together with its tools).
- Paper: Groeneveld et al., [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838)
  (2024) and Team OLMo, [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656)
  (2024; open models released with data, code, logs, and checkpoints; the OLMo 2
  report has a section on training stability).
- Paper: Meta, [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
  (2024; Section 3 "Pre-Training" describes the data pipeline, the scaling-law
  experiments, the 15T-token training recipe, and the MFU achieved on 16,000
  GPUs).
- Paper: DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
  (2024; advanced: a 671B-parameter mixture-of-experts (MoE) model trained in 8-bit
  floating point (FP8), with its pretraining data, hyperparameters, and compute
  cost in GPU-hours).

## Milestone

Write a two-page training plan for a 1B-parameter open model: data sources and
mixture, token budget (one compute-optimal and one over-trained variant), compute in
FLOPs from C ≈ 6ND, GPU-hours and cost on a named GPU at a stated MFU, and the loss
predicted by the Chinchilla parametric fit. Validate the arithmetic against
published figures: reproduce the 3.8 × 10²⁵ FLOPs that the Llama 3 report states
for its 405B model, and compute the MFU implied by the GPU count and training
duration reported for SmolLM3.

## Estimated time

6–8 weeks.
