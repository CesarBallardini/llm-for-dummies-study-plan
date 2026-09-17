# Chapter 18 — Alignment: RLHF, DPO, and Getting Models to Follow Instructions

> Part V — Post-Training · 5–7 weeks

## What you will learn

A pretrained large language model (LLM) continues text, and the supervised
fine-tuning (SFT) of Chapter 17 teaches it the format of an assistant answer but
provides no signal about which of two answers people prefer. This chapter covers
how that preference is trained into a model: the collection of human or AI
judgments between pairs of answers, the reward model that converts those judgments
into a scalar score, and reinforcement learning from human feedback (RLHF), which
uses the reinforcement learning (RL) algorithm proximal policy optimization (PPO)
to increase that score while a Kullback–Leibler (KL) divergence penalty keeps the
model close to its starting point. It then covers direct preference optimization
(DPO), which optimizes the same objective with a classification loss on the
preference pairs and no RL loop, and which the milestone applies to a small model.
The chapter closes with reward hacking and over-optimization, in which the reward
score rises while answer quality falls, and with the evaluation of whether a tuned
model is more helpful and no less safe.

## Topics

- The difference between a base model and an assistant model, and the
  post-training pipeline (SFT, preference tuning, RL)
- Human preference data: pairwise comparisons, annotation guidelines, and open
  datasets (the Anthropic Helpful and Harmless dataset (HH-RLHF), UltraFeedback)
- Reward models and the Bradley-Terry preference model
- RLHF with PPO: policy, reward, and advantage at a conceptual level (the RL
  foundations are in Chapter 19)
- The KL penalty that keeps the policy close to the reference model
- Rejection sampling and best-of-N sampling as baselines
- DPO and its variants: identity preference optimization (IPO), Kahneman-Tversky
  optimization (KTO), odds ratio preference optimization (ORPO), and simple
  preference optimization (SimPO)
- Offline vs online preference optimization
- RL from AI feedback (RLAIF) and Constitutional AI
- Reward hacking and over-optimization
- Evaluation of helpfulness and harmlessness: win rates, LLM judges, refusals, and
  sycophancy
- Open end-to-end recipes (Tülu 3)

## Resources

**Suggested path.** Start with the article "Illustrating RLHF" and the lecture
"State of GPT" for an overview of the pipeline, then use *Reinforcement Learning
from Human Feedback* by Nathan Lambert and its companion lectures as the main text
of this chapter, reading the InstructGPT and DPO papers when the book reaches
those topics. Work through unit 3 of the smol course in parallel as preparation
for the milestone. This chapter requires PPO only at a conceptual level: the RL
foundations (Sutton and Barto, Berkeley CS 285, David Silver's course, Spinning
Up) are listed in Chapter 19, which derives the algorithms. When time is short,
skip the university courses other than CS336 lecture 15, the PPO paper, and the
two Anthropic papers, and read only the summary sections of the Tülu 3 paper.

### University courses

- Stanford —
  [CS336 Language Modeling from Scratch (Spring 2026)](https://cs336.stanford.edu/)
  by Tatsunori Hashimoto and Percy Liang (free;
  [videos](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV) +
  assignments; start here: lecture 15, "Mid/post-training (SFT/RLHF)", second
  half, on preference data, PPO, and DPO; advanced: the optional safety-alignment
  part of [assignment 5](https://github.com/stanford-cs336/assignment5-alignment)
  implements DPO).
- Stanford — [CS234 Reinforcement Learning](https://web.stanford.edu/class/cs234/)
  by Emma Brunskill (free; Winter 2026 slides on the course site and
  [Spring 2024 videos](https://www.youtube.com/playlist?list=PLoROMvodv4rN4wG6Nk6sNpTEbuOSosZdX);
  optional: in the 2024 videos, lecture 9 is a guest lecture on DPO by its
  authors, and lectures 5–7, "Policy Search", derive policy gradients and PPO, the
  theory that Chapter 19 covers).
- Carnegie Mellon —
  [11-667 Large Language Models: Methods and Applications (Fall 2025)](https://2025.cmu-llms.org/)
  by Jaromir Savelka and Yubin Kim (free slides, no videos; optional: the lectures
  on task-oriented fine-tuning and on RLHF place this chapter and Chapter 17 in
  the context of the full LLM pipeline).

### Online courses (MOOCs)

- Nathan Lambert — [The Post-Training Course](https://rlhfbook.com/course) (free;
  [YouTube lectures](https://www.youtube.com/playlist?list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y)
  with slides that follow *Reinforcement Learning from Human Feedback* chapter by
  chapter; start here: lectures 1–4 on the overview, instruction tuning and reward
  models, and RL, then lecture 6 on DPO and lectures 8–9 on preference data and
  over-optimization).
- Hugging Face — [smol course](https://huggingface.co/learn/smol-course/unit0/1)
  (free; v2, 2025; self-paced, based on SmolLM3 and the Hugging Face Transformer
  Reinforcement Learning (TRL) library; unit 3, "Preference Alignment", trains a
  small model with DPO and prepares for the milestone).

### Books

- Book: Nathan Lambert, *Reinforcement Learning from Human Feedback* (Manning,
  2026) — [official page](https://rlhfbook.com/) (free online, paid print; start
  here: the main text of this chapter; read chapters 3–6 and 8, which cover the
  training overview, instruction tuning, reward modeling, RL, and direct alignment
  algorithms, then chapters 11 and 14 on preference data and over-optimization).

### Lectures, papers and articles

- Article: Nathan Lambert, Louis Castricato, Leandro von Werra, and Alex Havrilla,
  [Illustrating Reinforcement Learning from Human Feedback (RLHF)](https://huggingface.co/blog/rlhf)
  (2022, Hugging Face blog; start here: a short description of the three RLHF
  stages and the KL penalty).
- Lecture: Andrej Karpathy,
  [State of GPT](https://www.youtube.com/watch?v=bZQun8Y4L2A) (2023, Microsoft
  Build; 40 minutes; the full assistant pipeline from pretraining through SFT,
  reward modeling, and RLHF, and what each stage contributes).
- Paper: Long Ouyang et al.,
  [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
  (2022, InstructGPT; the three-stage recipe of SFT, reward modeling, and PPO on
  which ChatGPT was based; read sections 1–3).
- Paper: Rafael Rafailov et al.,
  [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
  (2023; rewrites the KL-constrained RLHF objective as a classification loss on
  preference pairs, with no separate reward model and no RL loop).
- Paper: Yuntao Bai et al.,
  [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862)
  (2022, Anthropic; released the HH-RLHF preference dataset and measured the
  trade-off between helpfulness and harmlessness).
- Paper: Yuntao Bai et al.,
  [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
  (2022, Anthropic; replaces human harmlessness labels with AI feedback guided by
  written principles; the paper that introduced RLAIF).
- Paper: Nathan Lambert et al.,
  [Tülu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124)
  (2024, Ai2; an open recipe of SFT, DPO, and RL with verifiable rewards (RLVR,
  Chapter 19), published with data, code, and evaluations).
- Article: Lilian Weng,
  [Reward Hacking in Reinforcement Learning](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)
  (2024; how policies exploit imperfect reward models, with a long section on
  reward hacking in the RLHF of language models).
- Paper: John Schulman et al.,
  [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
  (2017; advanced: the clipped policy-gradient algorithm used in the original RLHF
  recipes; assumes the RL foundations of Chapter 19).

### Tools and hands-on

- Hugging Face — [TRL](https://huggingface.co/docs/trl/index) documentation (free,
  open source; start here: the
  [`DPOTrainer`](https://huggingface.co/docs/trl/dpo_trainer) page documents the
  trainer used in the milestone, including the logged reward accuracy and reward
  margins; the [`RewardTrainer`](https://huggingface.co/docs/trl/reward_trainer)
  page documents the training of a Bradley-Terry reward model on the same pairs).

## Milestone

Start from a small SFT model — the model from the Chapter 17 milestone or
SmolLM2-360M-Instruct; with low-rank adaptation (LoRA) the run fits a free Colab
or Kaggle T4 graphics processing unit (GPU) — and run DPO with TRL on a few
thousand pairs from an open preference dataset such as UltraFeedback, once with a
small value of the DPO coefficient beta and once with a large one. Report the
preference accuracy on held-out pairs (the fraction of pairs for which the tuned
model assigns the chosen answer a higher implicit reward than the rejected one)
and a blind comparison by hand of the SFT and DPO answers to 30 held-out prompts.
For each value of beta, also estimate the KL divergence from the reference model
(the mean log-probability ratio between the two models on answers sampled from the
tuned one), and explain how beta trades the gain in preference accuracy against
drift from the reference model.

## Estimated time

5–7 weeks.
