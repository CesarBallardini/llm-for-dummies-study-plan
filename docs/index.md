# LLM for Dummies

This book is a self-study course that starts at **college algebra** and ends at
**building, training, operating, and improving open-source large language models
(LLMs)**. Each chapter ends with a milestone: a practical, hands-on exercise that
is used to check progress before the next chapter begins.

All required material is free or low-cost, and every resource is linked directly.
The university courses listed publish their lecture videos, slides, and
assignments openly, even when enrollment for credit is restricted.

!!! tip "Two ways in"

    **Straight through**, in chapter order: the book is built for it, and each
    chapter assumes the ones before it. Start at
    [Chapter 1](chapter-01-college-algebra/index.md).

    **From a goal**: the six [learning paths](learning-paths/index.md) each name a
    destination — understand how LLMs work, pretrain one, improve an existing
    model, serve and operate one, or build an application on open weights — and
    give the chapters that lead there in order, the prerequisites they assume down
    to the section, and the milestone that proves you arrived. Beginning from zero,
    take [Path 0 — Foundations first](learning-paths/foundations-first.md).

This book is released under the [MIT License](license.md). That license covers the
book and the code written for its milestones, and nothing else: every course,
textbook, paper, video, dataset, and model linked from the chapters keeps the terms
of its own author or publisher.

## How to use this book

The book has six parts and 24 chapters. The chapters are designed to be followed
in order, because each one assumes the material of the chapters before it. The
exception is Part II (programming and tools), which runs in parallel with Part I:
[Chapter 6](chapter-06-python-scientific-stack/index.md) can start in the first
week. Every chapter has the same structure:

- **What you will learn** — the scope of the chapter and how the rest of the course
  uses it.
- **Topics** — the syllabus, in study order.
- **Resources** — a suggested path, followed by university courses, online courses
  (MOOCs), books, and lectures, papers, and articles. The recommended starting
  point in each group is marked "start here"; optional and advanced items are
  labeled as such.
- **Milestone** — a hands-on exercise that applies the main ideas of the chapter
  and produces a result that can be verified.
- **Estimated time** — based on about 10 hours of study per week.

A reader who already knows a topic can skim the chapter and use the milestone as a
diagnostic test. The milestones are the course's mechanism for checking progress,
so completing each one before moving on is recommended.

The estimated total is 24–36 months at 10 hours per week, or 12–18 months at 20
hours per week. Part I accounts for a large share of that time because every later
part depends on its mathematics.

!!! tip "Suggested first week"

    1. Watch Andrej Karpathy's [Intro to Large Language
       Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) (1 hour) for an
       overview of the subject.
    2. Start Khan Academy College Algebra ([Chapter 1](chapter-01-college-algebra/index.md)):
       30–45 minutes per day.
    3. Start CS50P ([Chapter 6](chapter-06-python-scientific-stack/index.md)):
       45–60 minutes per day.
    4. At the end of the week, re-read this page and set a weekly time budget. A
       fixed weekly schedule is easier to sustain over a multi-year course than
       irregular long sessions.

## Map of goals to chapters

Each row below is the core of a [learning path](learning-paths/index.md): a page
that states the goal, the chapters that lead to it in order, what each one
contributes, and the milestone that proves you got there. Start with
[Path 0 - Foundations first](learning-paths/foundations-first.md) if you are
beginning from zero.

| Goal | Relevant chapters |
| --- | --- |
| Understand how LLMs work internally | [13](chapter-13-tokenization-embeddings/index.md), [14](chapter-14-llm-from-scratch/index.md), [15](chapter-15-pretraining-at-scale/index.md), [16](chapter-16-distributed-training-systems/index.md) |
| Pretrain an open-source LLM | [14](chapter-14-llm-from-scratch/index.md), [15](chapter-15-pretraining-at-scale/index.md), [16](chapter-16-distributed-training-systems/index.md), [17](chapter-17-finetuning/index.md) |
| Improve an existing model (training and research) | [16](chapter-16-distributed-training-systems/index.md), [17](chapter-17-finetuning/index.md), [18](chapter-18-alignment-rlhf/index.md), [19](chapter-19-reasoning-rl/index.md), [20](chapter-20-evaluation-benchmarks/index.md), [24](chapter-24-research-reading-loop/index.md) |
| Serve, deploy, and operate an LLM | [21](chapter-21-inference-serving/index.md), [22](chapter-22-operate-improve-open-llm/index.md), [23](chapter-23-capstone/index.md) |
| Build applications on top of open models | [21](chapter-21-inference-serving/index.md), [22](chapter-22-operate-improve-open-llm/index.md), [24](chapter-24-research-reading-loop/index.md) |

## The core bookshelf

The books below are referenced in several chapters. Local PDFs of the freely distributable titles are kept in the
repository `books/free/` folder; `books/free/README.md` is the manifest (source, license, and the relevant
chapters of each title). The links point to the official pages, and free titles are marked.

| Book | Where it fits | Access |
| --- | --- | --- |
| OpenStax — *Precalculus* (2e, 2021) | Chapter 1 | [Free PDF and web](https://openstax.org/details/books/precalculus-2e) |
| Strang — *Calculus* (3rd ed.) | Chapter 2 | [Free (MIT OCW)](https://ocw.mit.edu/courses/res-18-001-calculus-fall-2023/) |
| Deisenroth, Faisal, Ong — *Mathematics for Machine Learning* | Chapters 2–5 | [Free PDF](https://mml-book.github.io) |
| Strang — *Introduction to Linear Algebra* (6th ed.) | Chapter 3 | Companion to MIT 18.06 |
| Axler — *Linear Algebra Done Right* | Chapter 3, deeper rigor | [Free online](https://linear.axler.net) |
| Blitzstein & Hwang — *Introduction to Probability* | Chapter 4 | [Free](https://probabilitybook.net/) |
| Downey — *Think Python* (3rd ed.) | Chapter 6 | [Free notebooks](https://allendowney.github.io/ThinkPython/) |
| Goodfellow, Bengio, Courville — *Deep Learning* | Chapters 5, 9, 10 | [Free](https://www.deeplearningbook.org) |
| James, Witten, Hastie, Tibshirani, Taylor — *An Introduction to Statistical Learning* | Chapter 8 | [Free](https://www.statlearning.com) |
| Prince — *Understanding Deep Learning* | Chapters 8–10, 12 | [Free](https://udlbook.github.io/udlbook) |
| Zhang, Lipton, Li, Smola — *Dive into Deep Learning* | Chapters 5, 9, 11 | [Free](https://d2l.ai/) |
| Nielsen — *Neural Networks and Deep Learning* | Chapters 9, 10 | [Free](http://neuralnetworksanddeeplearning.com/) |
| Jurafsky & Martin — *Speech and Language Processing, 3rd ed.* | Chapters 11, 13 | [Free](https://web.stanford.edu/~jurafsky/slp3) |
| Raschka — *Build a Large Language Model (From Scratch)* | Chapters 12–15, 17, 23 | [Official page](https://www.manning.com/books/build-a-large-language-model-from-scratch), free code and videos |
| Alammar & Grootendorst — *Hands-On Large Language Models* | Chapters 13, 14, 17, 22 | [Site + free code](https://llm-book.com/) |
| Tazi, Mom, et al. (Hugging Face) — *The Ultra-Scale Playbook* | Chapters 7, 16 | [Free](https://huggingface.co/spaces/nanotron/ultrascale-playbook) |
| Ben Allal et al. (Hugging Face) — *The Smol Training Playbook* | Chapters 15, 23 | [Free](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) |
| Lambert — *Reinforcement Learning from Human Feedback* | Chapters 17–19 | [Free online](https://rlhfbook.com/), paid print |
| Raschka — *Build a Reasoning Model (From Scratch)* | Chapter 19 | [Hub + free videos](https://sebastianraschka.com/reasoning-from-scratch) |
| Kamath et al. — *Large Language Models: A Deep Dive* | Reference | [Springer](https://link.springer.com/book/10.1007/978-3-031-65647-7) |
| Huyen — *AI Engineering* | Chapters 7, 17, 20–22 | [Chapter summaries](https://github.com/chiphuyen/aie-book) |

## The course shelf

| Course | Provider | Chapter | Access |
| --- | --- | --- | --- |
| College Algebra / Pre-calculus | Khan Academy | [1](chapter-01-college-algebra/index.md) | Free, khanacademy.org |
| 18.01 / 18.02 Calculus | MIT OCW | [2](chapter-02-calculus/index.md) | Free, ocw.mit.edu |
| 18.06 Linear Algebra | MIT OCW | [3](chapter-03-linear-algebra/index.md) | Free, ocw.mit.edu |
| Statistics 110 | Harvard | [4](chapter-04-probability-statistics/index.md) | Free, YouTube |
| 18.065 Matrix Methods in Data Analysis, Signal Processing, and ML | MIT OCW | [5](chapter-05-optimization-numerical-methods/index.md) | Free, ocw.mit.edu |
| CS50P / Python for Everybody | Harvard / PY4E | [6](chapter-06-python-scientific-stack/index.md) | Free |
| CS229 Machine Learning | Stanford | [8](chapter-08-machine-learning/index.md) | Free |
| 6.S191 Introduction to Deep Learning | MIT | [9](chapter-09-deep-learning/index.md) | Free |
| Practical Deep Learning for Coders | fast.ai | [9](chapter-09-deep-learning/index.md) | Free (optional) |
| CS231n Deep Learning for Computer Vision | Stanford | [10](chapter-10-neural-networks-from-scratch/index.md) | Free |
| Zero to Hero | Karpathy | [10](chapter-10-neural-networks-from-scratch/index.md), [12](chapter-12-transformer-architecture/index.md), [13](chapter-13-tokenization-embeddings/index.md), [14](chapter-14-llm-from-scratch/index.md), [16](chapter-16-distributed-training-systems/index.md) | Free |
| CS224N NLP with Deep Learning | Stanford | [11](chapter-11-nlp-foundations/index.md), [12](chapter-12-transformer-architecture/index.md) | Free (videos) |
| CS25 Transformers United | Stanford | [12](chapter-12-transformer-architecture/index.md) | Free (videos, optional) |
| CS336 Language Modeling from Scratch | Stanford | [12](chapter-12-transformer-architecture/index.md), [13](chapter-13-tokenization-embeddings/index.md), [14](chapter-14-llm-from-scratch/index.md), [15](chapter-15-pretraining-at-scale/index.md), [16](chapter-16-distributed-training-systems/index.md), [17](chapter-17-finetuning/index.md), [18](chapter-18-alignment-rlhf/index.md), [19](chapter-19-reasoning-rl/index.md), [20](chapter-20-evaluation-benchmarks/index.md), [21](chapter-21-inference-serving/index.md), [23](chapter-23-capstone/index.md) | Free |
| The LLM Course | Hugging Face | [12](chapter-12-transformer-architecture/index.md), [13](chapter-13-tokenization-embeddings/index.md), [14](chapter-14-llm-from-scratch/index.md), [17](chapter-17-finetuning/index.md), [19](chapter-19-reasoning-rl/index.md) | Free, huggingface.co/learn |
| CS324 Foundation Models | Stanford | [15](chapter-15-pretraining-at-scale/index.md) | Free (optional) |
| 11-868 Large Language Model Systems | CMU | [16](chapter-16-distributed-training-systems/index.md), [21](chapter-21-inference-serving/index.md) | Free, syllabus online |
| 11-667 LLM Methods and Applications | CMU | [18](chapter-18-alignment-rlhf/index.md) | Free, syllabus online |
| The Post-Training Course | Nathan Lambert | [18](chapter-18-alignment-rlhf/index.md), [19](chapter-19-reasoning-rl/index.md) | Free |
| smol course | Hugging Face | [17](chapter-17-finetuning/index.md), [18](chapter-18-alignment-rlhf/index.md), [20](chapter-20-evaluation-benchmarks/index.md), [23](chapter-23-capstone/index.md) | Free |
| Efficiently Serving LLMs | DeepLearning.AI | [21](chapter-21-inference-serving/index.md) | Free |
| LLM Bootcamp | Full Stack Deep Learning | [22](chapter-22-operate-improve-open-llm/index.md) | Free |
| CS329S Machine Learning Systems Design | Stanford | [22](chapter-22-operate-improve-open-llm/index.md) | Free |
