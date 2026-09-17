# Chapter 16 — Distributed Training and LLM Systems

> Part IV — Transformers and LLMs · 6–8 weeks

## What you will learn

This chapter describes how a large language model (LLM) that does not fit in the
memory of one graphics processing unit (GPU), or whose training on one GPU would take
years, is trained on many GPUs. It first accounts for the memory of one training step
— weights, gradients, optimizer states, and activations — and covers the single-GPU
techniques that reduce that memory or raise the speed: mixed precision, gradient
accumulation, activation recomputation, kernel fusion, and FlashAttention. It then
covers the collective communication operations (all-reduce, all-gather, and
reduce-scatter) and the forms of parallelism built on them: data parallelism, sharded
data parallelism with the Zero Redundancy Optimizer (ZeRO) and PyTorch Fully Sharded
Data Parallel (FSDP), and tensor, pipeline, sequence, context, and expert
parallelism. For each method the chapter states what is divided among the GPUs, what
is communicated, and how the result is measured as throughput and as model FLOPs
utilization (MFU), the fraction of the peak floating-point operations per second of
the GPUs that the training run achieves. The model reports of Chapter 15 describe
their training infrastructure in these terms, and Chapters 17 and 21 reuse the memory
accounting and the attention kernels.

## Topics

- The memory of a training step: weights, gradients, optimizer states, and
  activations
- Mixed precision training: float16 (fp16) with loss scaling, bfloat16 (bf16), and
  float32 (fp32) master weights
- Gradient accumulation and activation recomputation (gradient checkpointing)
- GPU performance: the memory hierarchy, kernel fusion, `torch.compile`, and
  profiling
- Attention kernels that minimize GPU memory traffic (FlashAttention)
- Collective communication: all-reduce, all-gather, and reduce-scatter, the NVIDIA
  Collective Communications Library (NCCL), and interconnect bandwidth
- Data parallelism with PyTorch DistributedDataParallel (DDP)
- Sharded data parallelism: ZeRO stages 1–3 and FSDP
- Tensor parallelism and pipeline parallelism
- Sequence, context, and expert parallelism, and the conditions that require each
- Combined parallelism: choosing a configuration for a given model size and GPU
  count
- Measurement: tokens per second, scaling efficiency, and MFU
- The functions automated by DeepSpeed, PyTorch FSDP, and Megatron-LM

## Resources

**Suggested path.** Read the article by Lilian Weng for an overview and reread the
article by Horace He from Chapter 7, then work through sections 2 and 3 of Karpathy's
"Let's reproduce GPT-2 (124M)" with the code open, from mixed precision to DDP; the
milestone repeats those steps. *The Ultra-Scale Playbook* is the main text: read it
in order over several weeks, with CS336 lectures 5–8 in parallel and the PyTorch DDP
and FSDP tutorials when the corresponding section is reached; its appendix A0
explains the collective operations used in the milestone. Read each paper after the
Playbook section on the same method; the abstract, the method section, and the main
results table are sufficient on a first pass. When time is short, omit CMU 11-868,
*How to Scale Your Model*, CS336 Assignment 2, and the FSDP, FlashAttention-2, and
FlashAttention-3 papers.

### University courses

- Stanford — [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
  by Tatsunori Hashimoto and Percy Liang (free; Spring 2026
  [lecture videos](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)
  and assignments; start here: lecture 5 "GPUs, TPUs", lecture 6 "Kernels, Triton",
  and lectures 7 and 8 "Parallelism"; in
  [Assignment 2 (Systems)](https://github.com/stanford-cs336/assignment2-systems)
  the Assignment 1 model is profiled, FlashAttention-2 is written in Triton, and
  data-parallel training with a sharded optimizer state is implemented; the
  assignment is advanced and optional).
- CMU — [11-868: Large Language Model Systems](https://llmsystem.github.io/llmsystem2026spring/)
  by Lei Li (free; Spring 2026 slides and homework; optional: GPU programming,
  three lectures on distributed training, memory optimization, mixture-of-experts
  (MoE) models, and attention kernels; a second presentation of the topics of this
  chapter).

### Online courses (MOOCs)

- Andrej Karpathy — [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)
  (free; 4 hours; the sequel to Lecture 7 of [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html),
  with step-by-step code in [build-nanogpt](https://github.com/karpathy/build-nanogpt);
  start here: section 2, from about 1h22m to 2h15m, adds TF32 and bf16 mixed
  precision, `torch.compile`, and FlashAttention one change at a time and measures
  the speedup of each; gradient accumulation starts at about 2h34m and DDP on 8
  GPUs at about 2h47m; section 1 was used in Chapter 14).

### Books

- Book: Nouamane Tazi, Ferdinand Mom et al. (Hugging Face), *The Ultra-Scale
  Playbook: Training LLMs on GPU Clusters* (2025) — [free online edition](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
  (free online; start here: the main text of this chapter, first used in Chapter 7;
  covers training memory on one GPU, data parallelism and ZeRO, tensor, context,
  pipeline, and expert parallelism, the choice of a configuration, and GPU kernels,
  each with benchmark measurements; appendix A0 covers the collective operations).
- Book: Stas Bekman, *Machine Learning Engineering Open Book* (continuously
  updated) — [official page](https://github.com/stas00/ml-engineering) (free; a
  practitioner's reference, first used in Chapter 7; the Training chapters "Model
  parallelism", "Performance", and "Fault tolerance" and the Network chapter cover
  the choice of a parallelism method, throughput tuning, and interconnects).
- Book: Jacob Austin et al. (Google DeepMind), *How to Scale Your Model: A
  Systems View of LLMs on TPUs* (2025) — [free online edition](https://jax-ml.github.io/scaling-book/)
  (free online; advanced; optional: the sections on rooflines, Transformer math,
  and training parallelism derive the compute and communication cost of each
  method, followed by a worked example for Llama 3; written for Google's Tensor
  Processing Units (TPUs) and JAX, with a later section on GPUs).

### Lectures, papers and articles

- Article: Lilian Weng, [How to Train Really Large Models on Many GPUs?](https://lilianweng.github.io/posts/2021-09-25-train-large/)
  (2021; start here: a summary of data, pipeline, and tensor parallelism, MoE
  models, activation recomputation, and mixed precision, with figures from the
  source papers).
- Article: Horace He, [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)
  (2022; a second reading of the article from Chapter 7: compute, memory bandwidth,
  and overhead as the three bottlenecks of GPU work, and operator fusion as the
  method that reduces memory traffic).
- Article: Quentin Anthony, Stella Biderman, and Hailey Schoelkopf,
  [Transformer Math 101](https://blog.eleuther.ai/transformer-math/) (EleutherAI,
  2023; the section "Memory Requirements" gives formulas for the memory of weights,
  gradients, optimizer states, and activations, and the section "Distributed
  Training" shows how ZeRO and combined parallelism divide that memory among GPUs;
  the compute sections were used in Chapter 15).
- Docs: PyTorch, [PyTorch Distributed Overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html)
  (current docs; lists the distributed APIs and the use case of each), then the
  [DDP tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)
  and [Getting Started with FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
  (runnable examples; FSDP1 is deprecated, and the tutorial uses its replacement,
  `fully_shard`).
- Paper: Micikevicius et al., [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
  (2017; fp16 weights and activations with an fp32 master copy of the weights and
  loss scaling; the abstract was read in Chapter 7, the full paper is read here).
- Paper: Rajbhandari et al., [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
  (2019; partitions optimizer states, gradients, and parameters among the
  data-parallel GPUs in three stages; the DeepSpeed
  [ZeRO tutorial](https://www.deepspeed.ai/tutorials/zero/) shows the three stages
  and offload to CPU memory, configured in a JSON file).
- Paper: Zhao et al., [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277)
  (2023; optional: the design of the PyTorch implementation of ZeRO stage 3
  sharding).
- Paper: Shoeybi et al., [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
  (2019; tensor parallelism inside each Transformer layer), Narayanan et al.,
  [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
  (2021; pipeline schedules and the combination of data, tensor, and pipeline
  parallelism), and Korthikanti et al.,
  [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)
  (2022; sequence parallelism and selective activation recomputation; reports
  MFU).
- Paper: Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
  (2022; exact attention computed in blocks that fit in on-chip memory, so that the
  full attention matrix is never stored; the follow-ups
  [FlashAttention-2](https://arxiv.org/abs/2307.08691) (Dao, 2023) and
  [FlashAttention-3](https://arxiv.org/abs/2407.08608) (Shah et al., 2024) are
  advanced and optional).

### Tools and hands-on

- Code: Andrej Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT) (start
  here: a training loop of about 300 lines with mixed precision, `torch.compile`,
  gradient accumulation, and DDP launched with `torchrun`; its README reports a
  GPT-2 (124M) reproduction on one node of eight A100 GPUs in about 4 days, and now
  points to nanochat as the successor; the code base for the milestone, first
  listed in Chapter 14).

## Milestone

Train nanoGPT or the Chapter 14 model on one GPU and record tokens per second and
peak memory as mixed precision, `torch.compile`, and the fused attention kernel of
`scaled_dot_product_attention` are enabled one at a time. Run the same configuration with DDP on 2–4 GPUs — the two T4 GPUs of a
free Kaggle notebook are sufficient — and report tokens per second, scaling
efficiency relative to one GPU, and MFU; with a single GPU, use gradient
accumulation instead and verify that the loss curve matches that of the equivalent
larger batch. Compute the data volume that one ring all-reduce of the gradients
sends from each GPU, 2(N − 1)/N times the gradient size for N GPUs, estimate its
duration at the bandwidth of the interconnect used, and compare it with the
measured step time.

## Estimated time

6–8 weeks.
