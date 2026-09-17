# Path 0 — Foundations first

**Goal.** Finish able to read the rest of the book without stopping at the
notation. Concretely: differentiate a composed function by hand and say what a
gradient is; multiply matrices, and read an eigendecomposition or an SVD as a
statement about directions and scaling; write down a likelihood and take its
logarithm without hesitating; explain what gradient descent does and why a
learning rate matters; write and debug NumPy and PyTorch code inside a virtual
environment under version control; and say how much VRAM a model of a given size
needs in fp32, fp16 and bf16, and why that number decides what you can train.

This is the path taken by a reader starting from zero. Every other path assumes
most of it, so the honest order is to come here first and then choose a goal — or
to run this path as the backbone and read a later path's chapters as the reward
for finishing each part.

## Prerequisites

None beyond secondary-school algebra. [Chapter 1 — College Algebra and Pre-calculus](../chapter-01-college-algebra/index.md) starts from functions and graphs and
reviews exponents, radicals and factoring as it goes.

## The sequence

| Step | Chapter | Why this path needs it | Time |
| --- | --- | --- | --- |
| 1 | [Chapter 1 — College Algebra and Pre-calculus](../chapter-01-college-algebra/index.md) | The exponentials, logarithms, trigonometry and sigma notation that every later formula is written in. | 3–6 weeks |
| 2 | [Chapter 2 — Calculus I and II](../chapter-02-calculus/index.md) | Derivatives, partial derivatives and the multivariable chain rule. Backpropagation is that chain rule applied to a computational graph. | 6–10 weeks |
| 3 | [Chapter 3 — Linear Algebra](../chapter-03-linear-algebra/index.md) | Vectors, matrices and factorizations. A model represents each token as a vector and each layer as a matrix product; this is the language of the whole field. | 8–12 weeks |
| 4 | [Chapter 4 — Probability and Statistics](../chapter-04-probability-statistics/index.md) | Distributions, likelihood and information theory. A language model *is* a distribution over the next token, and its loss is a cross-entropy. | 6–10 weeks |
| 5 | [Chapter 5 — Optimization and Numerical Methods](../chapter-05-optimization-numerical-methods/index.md) | Gradient descent and its variants, plus the finite-precision arithmetic that decides whether a training run converges or produces NaNs. | 4–6 weeks |
| 6 | [Chapter 6 — Python and the Scientific Stack](../chapter-06-python-scientific-stack/index.md) | Python, the command line, virtual environments, Git, NumPy and PyTorch: the tools every later milestone is written with. | 8–12 weeks |
| 7 | [Chapter 7 — GPUs, Hardware and the Training Environment](../chapter-07-gpus-hardware/index.md) | What a GPU is, why it multiplies matrices fast, and how VRAM caps the model you can train or serve. | 2–4 weeks |

## How to run it

[Chapter 6 — Python and the Scientific Stack](../chapter-06-python-scientific-stack/index.md) is designed to run **in parallel** with Part I, not after it: start it in
the first week and give it 45–60 minutes a day alongside the mathematics. The
book's suggested first week does exactly that. [Chapter 7 — GPUs, Hardware and the Training Environment](../chapter-07-gpus-hardware/index.md) is short and can be read as
soon as you have a Colab or Kaggle account to try it on.

Four milestones prove this path: matrix multiplication written by hand and
checked against NumPy (Chapter 3), a maximum likelihood estimate derived and
verified by simulation (Chapter 4), a logistic regression fitted with a gradient
you derived yourself (Chapter 5), and a small loop-free NumPy library in its own
repository (Chapter 6).

## Estimated time

About **29–48 weeks** at 10 hours per week with Chapter 6 running in
parallel with the mathematics, or 37–60 weeks taken strictly in
sequence. Part I is the largest single block of time in the course, and it is why
the total estimate is measured in years rather than months.

## Where to go next

Every other path continues from here. If you do not yet know which one you want,
read [Chapter 12 — The Transformer Architecture](../chapter-12-transformer-architecture/index.md) and [Chapter 14 — Build an LLM from Scratch](../chapter-14-llm-from-scratch/index.md) first: they are the centre of the book, and which of them
you enjoy more is a fair signal for whether to head towards
[pretraining](pretrain-an-llm.md) or towards
[serving and operating](serve-and-operate.md).
