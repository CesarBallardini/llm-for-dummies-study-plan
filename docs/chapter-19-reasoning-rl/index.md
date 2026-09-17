# Chapter 19 — Reasoning Models and Reinforcement Learning

> Part V — Post-Training · 5–7 weeks

## What you will learn

Reasoning models such as OpenAI o1 and DeepSeek-R1 solve difficult mathematics and
coding problems by generating a long chain of thought before the final answer, and
this chapter covers how such models are built. It begins with the methods that
require no training: prompting a large language model (LLM) for step-by-step
answers, and inference-time scaling, which spends more compute per question by
sampling many answers and selecting one by majority vote or with a verifier. It
then develops the reinforcement learning (RL) foundations that Chapter 18 used at
a conceptual level — Markov decision processes (MDPs), policy gradients, and
proximal policy optimization (PPO) — and applies them to language models with
rewards that a program can verify, such as a correct final answer or passing unit
tests, using group relative policy optimization (GRPO), the algorithm used to
train DeepSeek-R1. It also covers the distillation of a large reasoning model into
a small one. The milestone trains a small model with RL on a single graphics
processing unit (GPU) and measures the change in its reasoning accuracy.

## Topics

- Chain-of-thought prompting and the effect of intermediate steps on accuracy
- Evaluating reasoning: answer extraction, exact-match checkers, and pass@k (the
  probability that at least one of k sampled answers is correct)
- Inference-time scaling (self-consistency, best-of-N sampling, verifier-guided
  search)
- Outcome and process reward models as verifiers
- RL foundations: MDPs, returns, value functions, and advantages
- Policy gradients: REINFORCE, baselines, actor-critic, and the clipped objective
  of PPO
- RL with verifiable rewards (RLVR) and its differences from reinforcement
  learning from human feedback (RLHF, Chapter 18)
- Reward design for math and code (answer and format rewards, unit tests) and
  reward hacking
- GRPO: group-relative advantages without a value network
- Reasoning models: DeepSeek-R1-Zero, DeepSeek-R1, and o1-style models
- Distillation of reasoning traces into small models

## Resources

**Suggested path.** Read "Understanding Reasoning LLMs" by Sebastian Raschka for
an overview, then divide each week between two tracks. The first track is RL
foundations: the three-part introduction of Spinning Up, chapters 3 and 13 of
Sutton and Barto, and the policy-gradient lectures of CS 285 (or David Silver's
course, which proceeds at a slower pace). The second track is reasoning models:
*Build a Reasoning Model (From Scratch)* is the main text of this chapter; read
the chain-of-thought and self-consistency papers alongside its chapters 4–5, and
the DeepSeekMath and DeepSeek-R1 papers, CS336 lecture 16, and chapter 12 of the
Hugging Face LLM Course alongside its chapters 6–7. When time is short, skip David
Silver's course, the remaining CS 285 lectures, and the process-reward and
test-time-compute papers, and complete the milestone with the Hugging Face
Transformer Reinforcement Learning (TRL) library or Unsloth instead of a
hand-written training loop.

### University courses

- Stanford —
  [CS336 Language Modeling from Scratch (Spring 2026)](https://cs336.stanford.edu/)
  by Tatsunori Hashimoto and Percy Liang (free;
  [videos](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV) +
  assignments; start here: lecture 16, "Post-training - RLVR", on policy gradients
  for language models, GRPO, and the DeepSeek-R1 recipe; advanced:
  [assignment 5](https://github.com/stanford-cs336/assignment5-alignment)
  implements supervised fine-tuning (SFT), expert iteration, and GRPO for math
  reasoning and is sized for a data-center GPU).
- UC Berkeley —
  [CS 185/285 Deep Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse/)
  by Sergey Levine (free; Spring 2026 slides and homework on the course site and
  [Fall 2023 videos](https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps),
  whose numbering differs; the relevant lectures in the 2026 schedule are 4–6 on
  RL basics, policy gradients, and actor-critic, 9–10 on advanced policy
  gradients, and 14, "LLM RL").
- UCL and DeepMind — [Reinforcement Learning course
  (2015)](https://www.davidsilver.uk/teaching/) by David Silver (free;
  [videos](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ) +
  slides; optional: a ten-lecture introduction at a slower pace than CS 285; the
  relevant lectures are 1–3 on MDPs and value functions and 7 on policy gradients).

### Online courses (MOOCs)

- Hugging Face —
  [LLM Course, Chapter 12: Open R1 for Students](https://huggingface.co/learn/llm-course/chapter12/1)
  (free; a few hours; start here: covers RL for language models, a reading of the
  DeepSeek-R1 paper, GRPO in TRL, and a GRPO training exercise).
- OpenAI — [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/)
  (free; 2018; a concise introduction to RL terminology and policy optimization,
  with reference implementations of vanilla policy gradient and PPO; read
  "Introduction to RL" parts 1–3).

### Books

- Book: Sebastian Raschka, *Build a Reasoning Model (From Scratch)* (Manning,
  2026) — [official hub](https://sebastianraschka.com/reasoning-from-scratch/)
  (paid; free code in
  [the GitHub repo](https://github.com/rasbt/reasoning-from-scratch); start here:
  chapter 3 on evaluating with verifiers, chapters 4–5 on inference-time scaling,
  chapters 6–7 on RLVR and GRPO, and chapter 8 on distillation, all implemented in
  PyTorch on a small Qwen3 model).
- Book: Richard S. Sutton and Andrew G. Barto, *Reinforcement Learning: An
  Introduction* (2nd ed., MIT Press, 2018) —
  [official page](http://incompleteideas.net/book/the-book-2nd.html) (free PDF;
  the standard RL textbook; chapter 3, "Finite Markov Decision Processes", and
  chapter 13, "Policy Gradient Methods", provide the terminology and the
  mathematics underlying PPO and GRPO).
- Book: Nathan Lambert, *Reinforcement Learning from Human Feedback* (Manning,
  2026) — [official page](https://rlhfbook.com/) (free online, paid print; chapter
  6, "Reinforcement Learning", derives the policy-gradient algorithms from
  REINFORCE to PPO and GRPO as they are applied to language models, and chapter 7,
  "Reasoning and Inference-Time Scaling", surveys RLVR and the reasoning-model
  recipes; lecture 5 of the companion
  [Post-Training Course](https://rlhfbook.com/course) covers chapter 7).

### Lectures, papers and articles

- Article: Sebastian Raschka,
  [Understanding Reasoning LLMs](https://magazine.sebastianraschka.com/p/understanding-reasoning-llms)
  (2025; start here: describes four approaches to building a reasoning model —
  inference-time scaling, pure RL, SFT followed by RL, and distillation — with
  DeepSeek-R1 as the case study).
- Paper: Jason Wei et al.,
  [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
  (2022; shows that prompting a large model to write out intermediate steps
  improves its accuracy on multi-step arithmetic, commonsense, and symbolic
  problems).
- Paper: Xuezhi Wang et al.,
  [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)
  (2022; samples many chains of thought and takes a majority vote over the final
  answers; a basic form of inference-time scaling).
- Paper: Zhihong Shao et al.,
  [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
  (2024; introduces GRPO, a PPO variant that omits the value network and computes
  advantages relative to a group of sampled answers; read section 4).
- Paper: DeepSeek-AI,
  [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
  (2025; DeepSeek-R1-Zero acquires long chains of thought from RL alone with
  rule-based rewards; DeepSeek-R1 adds a cold-start SFT stage and is distilled
  into smaller models).
- Paper: Hunter Lightman et al.,
  [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) (2023, OpenAI;
  process reward models, which grade each reasoning step, select correct solutions
  to problems of the MATH benchmark more often than outcome reward models).
- Paper: Charlie Snell et al.,
  [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314)
  (2024; advanced: compares the allocation of inference compute between
  verifier-guided search and sequential revisions as a function of question
  difficulty).

### Tools and hands-on

- Hugging Face — [TRL](https://huggingface.co/docs/trl/index) documentation (free,
  open source; start here: the
  [`GRPOTrainer`](https://huggingface.co/docs/trl/grpo_trainer) page documents a
  widely used open-source GRPO implementation, its custom reward functions, and
  the logged reward and completion length that the milestone plots; optional: the
  [Unsloth](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide)
  RL guide wraps the same trainer, explains reward-function design, and links free
  Colab notebooks that train models of up to 1.5B parameters with low-rank
  adaptation (LoRA) in about 5 GB of GPU memory).

## Milestone

Train a small instruction-tuned model (Qwen2.5-0.5B-Instruct or Qwen3-0.6B; with
LoRA the run fits a free Colab or Kaggle T4 GPU, and a 1.5B-parameter model fits a
24 GB GPU) with GRPO — the TRL `GRPOTrainer` or a hand-written loop based on
chapters 6–7 of *Build a Reasoning Model (From Scratch)* — on the training set of
GSM8K, a dataset of grade-school math word problems, with a rule-based reward for
the answer format and for an exact-match final answer. Plot the mean reward, the
mean completion length, and the accuracy on a fixed 200-question slice of the
GSM8K test set against training steps. Compare the final accuracy with that of the
model before RL training under greedy decoding and under self-consistency with 8
samples, and state which of the two gains required more compute.

## Estimated time

5–7 weeks.
