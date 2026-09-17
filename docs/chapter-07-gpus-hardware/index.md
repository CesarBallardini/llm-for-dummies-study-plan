# Chapter 7 — GPUs, Hardware, and the Training Environment

> Part II — Programming and Tools · 2–4 weeks

## What you will learn

This chapter describes what a graphics processing unit (GPU) is, why it multiplies
matrices much faster than a central processing unit (CPU), and how the capacity and
bandwidth of its memory (VRAM) limit the models that can be run on it. It explains
how the number formats float32 (fp32), float16 (fp16), and bfloat16 (bf16) determine
the memory and speed cost of a model, and how to estimate that cost as the number of
parameters times the bytes per parameter. It also covers setting up a GPU
environment — Google Colab or Kaggle at no cost, a rented cloud GPU, or a local
machine — and measuring memory use and speed in it. Neural networks are not
introduced until Chapter 9, so the training topics (mixed precision, gradients,
optimizer states) are presented here as an introduction and are treated in full in
Chapters 9 and 16.

## Topics

- CPU vs GPU: a few fast cores vs thousands of simple ones, and why matrix
  multiplication suits a GPU
- The software stack: NVIDIA drivers, the CUDA platform, and PyTorch devices (CPU,
  CUDA, Apple MPS)
- VRAM, memory bandwidth, and the cost of moving data between CPU and GPU
- Floating-point operations (FLOPs), FLOP/s, and throughput in tokens per second
- Compute-bound, memory-bound, and overhead-bound workloads
- Number formats and bytes per parameter: fp32, fp16, bf16, and 8-bit integers (int8)
- Measuring GPU work: `nvidia-smi`, PyTorch memory counters, and timing
  asynchronous code
- CPU vs GPU for running a model: the cases in which a laptop is sufficient
- Choosing an environment: Google Colab and Kaggle notebooks, a rented cloud GPU,
  or a local machine
- An introduction to training memory: weights, gradients, optimizer states, and
  activations
- An introduction to mixed precision training (autocast and loss scaling)

## Resources

**Suggested path.** Start this chapter after Chapter 6, still in parallel with
Part I; it requires only arithmetic and the PyTorch tensors of Chapter 6. Watch the
Branch Education video, read Horace He's article, then open Colab and reproduce the
main points — move tensors to the GPU, change their dtype, and time the operations —
with the PyTorch CUDA notes as reference. Read Dettmers and the "AI Accelerators"
section of *AI Engineering* for the hardware vocabulary, and the bytes-per-parameter
sections of Transformer Math 101 for the memory arithmetic. Items marked advanced
assume neural networks or C++: omit them when time is short, and return to them
from Chapters 9, 15, and 16 as indicated in each note.

### University courses

- Stanford — [CS149: Parallel Computing (Fall 2025)](https://gfxcourses.stanford.edu/cs149/fall25)
  by Kayvon Fatahalian and Kunle Olukotun (free slides + assignments; the
  [2023 lecture videos](https://www.youtube.com/playlist?list=PLoROMvodv4rMp7MTFr4hQsDEcX7Bx6Odp)
  are on YouTube; start here: lecture 7 only, "GPU Architecture and CUDA
  Programming"; advanced: the rest of the course is systems programming in C++ and
  fits after Chapter 16, as preparation for writing GPU code).
- Stanford — [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
  by Percy Liang and Tatsunori Hashimoto (free; Spring 2026 lecture videos on a
  [YouTube playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)
  and public assignments; advanced: lecture 2 on resource accounting counts the
  memory and FLOPs of training, and lecture 5 covers GPUs for large language model
  (LLM) work; both assume the transformer architecture, so they fit in Chapter 15).

### Online courses (MOOCs)

- GPU MODE — [lecture series on YouTube](https://www.youtube.com/@GPUMODE) with
  [slides and code on GitHub](https://github.com/gpu-mode/lectures) (free; 100+
  community talks on GPU programming; advanced: lecture 3, "Getting Started With
  CUDA" by Jeremy Howard, is aimed at Python programmers and needs no further
  background; the other lectures fit after Chapter 16).

### Books

- Book: Chip Huyen, *AI Engineering* (O'Reilly, 2025) — [book repo](https://github.com/chiphuyen/aie-book)
  (paid book, free chapter summaries; start here: within chapter 9, "Inference
  Optimization", only the "AI Accelerators" section, which defines an accelerator
  and describes it by memory size, memory bandwidth, and FLOP/s; the rest of that
  chapter belongs to Chapter 21).
- Book: Stas Bekman, *Machine Learning Engineering Open Book* (continuously
  updated) — [official page](https://github.com/stas00/ml-engineering) (free; a
  practitioner's reference; "Compute" under Hardware covers accelerator
  specifications, and "Tensor precision / Data types" under Training covers fp32,
  fp16, and bf16; the other Training chapters belong to Chapter 16).
- Book: Nouamane Tazi, Ferdinand Mom et al. (Hugging Face), *The Ultra-Scale
  Playbook: Training LLMs on GPU Clusters* (2025) — [free online edition](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
  (free online; advanced: the opening part on training with one GPU shows how
  memory is divided among weights, gradients, optimizer states, and activations;
  the whole book is read in Chapter 16).
- Book: Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj, *Programming Massively
  Parallel Processors* (4th ed., 2022) —
  [official page](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0)
  (paid; advanced; optional: the standard CUDA textbook, abbreviated *PMPP*;
  chapters 1–6 run through memory architecture and performance; requires C, so it
  fits after Chapter 16).

### Lectures, papers and articles

- Video: Branch Education, [How do Graphics Cards Work? Exploring GPU Architecture](https://www.youtube.com/watch?v=h9Z4oGN89MU)
  (2024; 28 minutes; start here: an animated description of a graphics card — its
  cores, its VRAM, and why thousands of simple cores suit matrix arithmetic;
  assumes no background).
- Article: Horace He, [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)
  (2022; identifies compute, memory bandwidth, and overhead as the three
  bottlenecks of GPU work, without CUDA code; the neural-network layer names it
  mentions are defined in Chapter 9, and a second reading fits in Chapter 16).
- Article: Tim Dettmers, [Which GPU(s) to Get for Deep Learning](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/)
  (2023; which specifications matter, from Tensor Cores to memory bandwidth and
  VRAM, and when renting costs less than buying; the GPU models discussed are
  dated, the selection criteria still apply).
- Article: Quentin Anthony, Stella Biderman, and Hailey Schoelkopf,
  [Transformer Math 101](https://blog.eleuther.ai/transformer-math/) (EleutherAI,
  2023; the sections on bytes per parameter in fp32, fp16/bf16, and int8 belong to
  this chapter; the C = 6PD compute rule and the training-memory sections belong to
  Chapter 15).
- Docs: PyTorch, [CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)
  (current docs; devices, memory management, and asynchronous execution, including
  why unsynchronized GPU timings are wrong and how to correct them), then
  [Automatic Mixed Precision package - torch.amp](https://docs.pytorch.org/docs/stable/amp.html)
  (how `autocast` and `GradScaler` work; a preview of Chapter 9).
- Paper: Paulius Micikevicius et al., [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
  (2017, ICLR 2018; advanced: fp16 training with an fp32 master copy of the
  weights and loss scaling; the abstract suffices here, the full paper fits in
  Chapter 16).
- Article: Mark Harris, [An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
  (NVIDIA Technical Blog, updated 2025; optional: a first CUDA kernel in C++ —
  threads, blocks, and unified memory — showing the layer that PyTorch abstracts
  away).

### Tools and hands-on

- Hands-on: Google, [Colab](https://colab.research.google.com) (free tier; start
  here: open a notebook, switch the runtime to a GPU, run `!nvidia-smi`, and move
  a PyTorch tensor to `"cuda"`; the [Colab FAQ](https://research.google.com/colaboratory/faq.html)
  explains the usage limits).
- Hands-on: [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks) (free GPU
  notebooks with a weekly quota) for work that exceeds the Colab limits; beyond
  that, GPUs can be rented from [Lambda](https://lambda.ai/),
  [RunPod](https://www.runpod.io/), or [Vast.ai](https://vast.ai/) (paid; billed by
  the hour whether or not the GPU is in use, so shut the instance down after each
  session).

## Milestone

In Colab or on any GPU, write a short PyTorch script that creates a 4096 × 4096
matrix in fp32, fp16, and bf16 and verifies that the memory footprint of each tensor
(`element_size() * nelement()`) matches a hand calculation. Measure the time of an
fp32 matrix multiplication on the CPU and on the GPU — with a warm-up run and
`torch.cuda.synchronize()` before each clock reading — then repeat on the GPU in
fp16 and report both speedups. Finally, compute by hand how many gigabytes the
weights of a 7-billion-parameter model occupy in fp32 and in bf16, and state whether
they fit in the VRAM of the GPU used.

## Estimated time

2–4 weeks.
