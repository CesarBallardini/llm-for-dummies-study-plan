# Learning paths

The book has 24 chapters in a fixed order, and a reader who works through all of
them in sequence will not go wrong. A learning path is the other way in: it starts
from something you want to be able to do and names the chapters that get you
there, in the order to take them.

Each path states a **goal**, the **prerequisites** it assumes, the **sequence** of
chapters with a line on why each one is in the path, and the **milestone** that
proves you reached the goal. No path contains new material: every one of them is a
route through the same 24 chapters.

| Path | Goal | Core chapters | Time |
| --- | --- | --- | --- |
| [0 — Foundations first](foundations-first.md) | Reach the point where every later chapter is readable. | [1](../chapter-01-college-algebra/index.md), [2](../chapter-02-calculus/index.md), [3](../chapter-03-linear-algebra/index.md), [4](../chapter-04-probability-statistics/index.md), [5](../chapter-05-optimization-numerical-methods/index.md), [6](../chapter-06-python-scientific-stack/index.md), [7](../chapter-07-gpus-hardware/index.md) | 29–48 weeks |
| [1 — Understand how LLMs work internally](understand-llms-internally.md) | Follow a prompt from characters to sampled token, and read any model card for what it is. | [13](../chapter-13-tokenization-embeddings/index.md), [14](../chapter-14-llm-from-scratch/index.md), [15](../chapter-15-pretraining-at-scale/index.md), [16](../chapter-16-distributed-training-systems/index.md) | 21–28 weeks |
| [2 — Pretrain an open-source LLM](pretrain-an-llm.md) | Plan and run a pretraining run end to end, and finish with a base model of your own. | [14](../chapter-14-llm-from-scratch/index.md), [15](../chapter-15-pretraining-at-scale/index.md), [16](../chapter-16-distributed-training-systems/index.md), [17](../chapter-17-finetuning/index.md) | 22–30 weeks |
| [3 — Improve an existing model](improve-an-existing-model.md) | Take a released open model, make it measurably better, and prove the gain. | [16](../chapter-16-distributed-training-systems/index.md), [17](../chapter-17-finetuning/index.md), [18](../chapter-18-alignment-rlhf/index.md), [19](../chapter-19-reasoning-rl/index.md), [20](../chapter-20-evaluation-benchmarks/index.md), [24](../chapter-24-research-reading-loop/index.md) | 23–32 weeks |
| [4 — Serve, deploy and operate an LLM](serve-and-operate.md) | Run an open model in production at a known cost and latency, and keep it improving. | [21](../chapter-21-inference-serving/index.md), [22](../chapter-22-operate-improve-open-llm/index.md), [23](../chapter-23-capstone/index.md) | 14–21 weeks |
| [5 — Build applications on open models](build-applications.md) | Ship a product on top of an open model, and keep up with what replaces it. | [21](../chapter-21-inference-serving/index.md), [22](../chapter-22-operate-improve-open-llm/index.md), [24](../chapter-24-research-reading-loop/index.md) | 10–13 weeks |

## Choosing one

- Starting from zero, take [Path 0](foundations-first.md). Everything else assumes
  most of it, and its Chapter 6 runs in parallel with the mathematics rather than
  after it.
- Wanting to know how the machine works rather than to operate one, take
  [Path 1](understand-llms-internally.md).
- Wanting to train a model of your own from raw text, take
  [Path 2](pretrain-an-llm.md). It is the most expensive path in both time and
  money.
- Wanting to make an existing open model better, take
  [Path 3](improve-an-existing-model.md). This is the research path.
- Wanting to run a model in production, take [Path 4](serve-and-operate.md).
- Wanting to build a product on someone else's open weights, take
  [Path 5](build-applications.md), the shortest route here, which treats the
  Chapter 23 capstone as an optional finisher rather than a requirement.

Paths 4 and 5 share Chapters 21 and 22 and part company at the end — the capstone
for one, the reading loop for the other. Paths 2 and 3 are the two halves of
training: one before a base model exists, one after.

## How the paths relate to the chapters

The core chapters of each path are the rows of the
[map of goals to chapters](../index.md#map-of-goals-to-chapters), with
[Path 0](foundations-first.md) added for readers starting from zero. The
prerequisite lists and the reading notes are editorial: they say what a path leans
on most heavily, which is not the same as permission to skip the chapters in
between. A chapter left out of a path is left out of *that route*, not declared
optional.
