# Chapter 23 — Capstone: Train, Ship, and Improve Your Own Model

> Part VI — Evaluation, Deployment, and the Ecosystem · 4–8 weeks, flexible based on budget

## What you will learn

This chapter is a capstone project: a small chat model is trained, evaluated, served,
and published through the same sequence of stages that produces a released open large
language model (LLM), within a fixed budget of about $15 to $100 of rented graphics
processing unit (GPU) time. The project applies the earlier chapters in order:
tokenizer training (Chapter 13), pretraining (Chapters 14–16), supervised fine-tuning
(SFT, Chapter 17), an optional stage of Direct Preference Optimization (DPO) or
reinforcement learning (RL) (Chapters 18 and 19), evaluation (Chapter 20), serving
(Chapter 21), and publication with a model card (Chapter 22). It adds the
project-level work that the single-topic chapters do not require: a written plan
with a compute and cost estimate, a rehearsal run at reduced scale, experiment
tracking, a cost record, and one controlled change to the pipeline whose effect is
measured by re-evaluation. The result is a public repository on the Hugging Face Hub
that contains the checkpoints, a model card, and a technical write-up.

## Topics

- Project plan: model size, token budget, compute estimated in floating-point
  operations (FLOPs) from C ≈ 6ND, GPU-hours, and a cost ceiling (Chapter 15)
- Choice of compute: a rented node of eight GPUs, a single rented GPU, and spot
  instances (Chapters 7 and 16)
- Pipeline design: the stages, the artifact that each stage produces, and a rehearsal
  run at reduced scale before the full run
- Experiment tracking and reproducibility: configurations, random seeds, logs, and a
  record of cost
- Tokenizer training and measurement of its compression ratio (Chapter 13)
- Pretraining: data download, monitoring of loss and throughput, checkpointing, and
  restarts (Chapters 14–16)
- Mid-training (a short stage on curated or task-specific data between pretraining
  and SFT) and SFT with a chat template (Chapter 17)
- An optional stage of DPO or of RL with verifiable rewards (Chapters 18 and 19)
- Evaluation of the base and chat checkpoints against a published baseline
  (Chapter 20)
- Serving: an inference engine with a key-value cache, a chat interface, and
  measurement of generation throughput (Chapter 21)
- Iteration: one controlled change to the pipeline, followed by re-evaluation
  (Chapter 22)
- Publication: checkpoints, model card, license, and technical write-up (Chapters 14
  and 22)

## Resources

**Suggested path.** Watch "Intro to Large Language Models" as a review of the whole
pipeline, then read the nanochat README, its `runs/speedrun.sh` script, and the guide
"Beating GPT-2 for <<$100", and write the project plan. Run the pipeline first at
reduced scale — `runs/runcpu.sh` on a laptop, or a 12-layer model on one rented GPU —
so that every stage and every evaluation script has completed once before the
eight-GPU node is rented; then run the full pipeline, with the sections of *The Smol
Training Playbook* on ablations and loss spikes as the reference for monitoring. In
parallel with the runs, work through the smol course units on instruction tuning and
preference alignment, and use the SmolLM3 article and the model cards guide as
templates for the write-up. CS336 and *Build a Large Language Model (From Scratch)*
support a project written from scratch in place of nanochat; when time is short,
omit them together with "Deep Dive into LLMs like ChatGPT", the DataComp-LM paper,
and modded-nanogpt.

### University courses

- Stanford — [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
  by Tatsunori Hashimoto and Percy Liang (free; Spring 2026
  [lecture videos](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)
  and assignments; optional: the five assignments — basics, systems, scaling, data,
  and alignment — cover the same stages as this capstone with unit tests, and were
  assigned in Chapters 13–19; they form an alternative project when the model is
  written from scratch).

### Online courses (MOOCs)

- Hugging Face — [smol course](https://huggingface.co/learn/smol-course/unit0/1)
  (free; v2, 2025; self-paced, based on SmolLM3 and TRL; start here: the units on
  instruction tuning, preference alignment, and evaluation rehearse the
  post-training half of the capstone on a small model).
- Andrej Karpathy — [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)
  (free; 4 hours; step-by-step code in [build-nanogpt](https://github.com/karpathy/build-nanogpt);
  first assigned in Chapter 14; the final section, a complete pretraining run on 10B
  tokens of FineWeb-Edu with a HellaSwag evaluation, is the part relevant here).

### Books

- Book: Sebastian Raschka, *Build a Large Language Model (From Scratch)* (Manning,
  2024) — [official page](https://www.manning.com/books/build-a-large-language-model-from-scratch)
  (paid; free code at [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch);
  optional: the book was completed in Chapters 12–17; the repository's bonus
  material on the key-value cache, DPO, and the Llama 3 and Qwen3 architectures is a
  reference for a project written from scratch).

### Lectures, papers and articles

- Video: Andrej Karpathy, [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g)
  (2023; 1 hour; start here: an overview of pretraining, fine-tuning, and
  deployment, used as a review before the project plan is written), and
  [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI)
  (2025; 3.5 hours; optional: the same pipeline from pretraining data to
  post-training and RL in more detail).
- Article: Loubna Ben Allal et al. (Hugging Face), [The Smol Training Playbook: The Secrets to Building World-Class LLMs](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook)
  (2025; first read in Chapter 15; the sections on whether to train at all,
  ablations, data mixtures, loss spikes, and restarts apply directly to the project
  plan and to monitoring the run).
- Article: Hugging Face, [SmolLM3: smol, multilingual, long-context reasoner](https://huggingface.co/blog/smollm3)
  (2025; a published recipe with architecture, a three-stage data mixture,
  mid-training, post-training, and evaluation tables; a template for the structure
  of the write-up).
- Article: Sam Dobson, [I trained a ChatGPT clone for £75](https://samdobson.uk/posts/training-a-chatgpt-clone-for-cheap/)
  (2025; an independent account of the original nanochat run on a rented 8xH100
  node — 4 hours, about $100 — with the resulting benchmark scores; the current
  speedrun is shorter).
- Paper: Li et al., [DataComp-LM: In search of the next generation of training sets for language models](https://arxiv.org/abs/2406.11794)
  (2024; optional: defines the CORE score, the mean centered accuracy over 22
  tasks, which nanochat reports for its base model and compares with GPT-2).
- Paper: Mitchell et al., [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)
  (2018; the sections of a model card: intended use, training data, evaluation
  data, metrics, and limitations), with the Hugging Face
  [model cards guide](https://huggingface.co/docs/hub/model-cards) (the metadata
  format of a `README.md` on the Hub) and
  [Spaces documentation](https://huggingface.co/docs/hub/spaces) (hosting of a
  demonstration application).

### Tools and hands-on

- Code: Andrej Karpathy, [nanochat](https://github.com/karpathy/nanochat) (free,
  open source, MIT license; start here: one codebase for tokenizer training,
  pretraining, SFT, RL, evaluation, and inference with a key-value cache and a
  command-line chat; as of 2026 `runs/speedrun.sh` trains a model with the CORE
  score of GPT-2 in about 2 hours on an 8xH100 node, about $48 on demand or about
  $15 on spot instances; the same code runs on one GPU in about eight times the
  time; the guide [Beating GPT-2 for <<$100: the nanochat journey](https://github.com/karpathy/nanochat/discussions/481)
  documents the current recipe).
- Code: Andrej Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT) (free;
  pretraining only; the README points to nanochat as the successor; the training
  loop of about 300 lines, used in Chapters 14 and 16, is a smaller base for a
  project that adds the other stages by hand).
- Code: Keller Jordan et al., [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
  (advanced; a public speedrun of GPT-2 (124M) training derived from Karpathy's
  reproduction; its record log lists each optimizer and architecture change with
  the measured training time, and nanochat adopts several of them).
- Tool: EleutherAI, [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
  (free, open source; the harness of Chapter 20; applies to checkpoints in Hugging
  Face format; nanochat uses its own scripts for the CORE score, ARC, MMLU, GSM8K,
  and HumanEval).
- Tool: [Weights & Biases](https://docs.wandb.ai/) documentation (free tier for
  personal projects; experiment tracking; nanochat logs loss, throughput, and
  model FLOPs utilization (MFU) to it when `WANDB_RUN` is set).
- Hands-on: GPUs rented by the hour from [Lambda](https://lambda.ai/),
  [RunPod](https://www.runpod.io/), or [Vast.ai](https://vast.ai/) (paid; billed
  whether or not the GPU is in use; copy the checkpoints and logs off the instance
  and shut it down after each session).

## Milestone

Write a one-page plan that states the model size, the token budget, the compute in
FLOPs from C ≈ 6ND, and a cost ceiling between $15 and $100; then, within that
ceiling, train a tokenizer, pretrain a model, and fine-tune it with SFT (a DPO or RL
stage is optional). Evaluate the base checkpoint on one benchmark that has a
published GPT-2 baseline, such as the CORE score or HellaSwag, and the chat checkpoint
on at least two tasks, such as ARC and GSM8K, and report each accuracy with its
standard error. Serve the final checkpoint through a chat interface and measure its
generation throughput in tokens per second. Publish the checkpoints, a model card,
and a write-up on the Hugging Face Hub; the write-up states the data, the
hyperparameters, the loss curves, the actual cost, the evaluation results, and one
change made after the first evaluation together with its measured effect.

## Estimated time

4–8 weeks, flexible based on budget.
