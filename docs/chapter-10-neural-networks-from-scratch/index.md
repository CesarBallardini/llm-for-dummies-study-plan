# Chapter 10 — Neural Networks from Scratch: Backprop and the Gradient Flow

> Part III — Machine Learning and Deep Learning · 4–6 weeks

## What you will learn

This chapter implements from first principles the training machinery that Chapter 9
used through PyTorch. It covers computational graphs, reverse-mode automatic
differentiation, and backpropagation as the chain rule applied to a graph, first for
scalar operations and then for the matrix operations of a network layer. The
implementation is a small automatic differentiation (autograd) engine modeled on
Karpathy's micrograd, verified against finite differences and against PyTorch, and
used to train a multilayer perceptron (MLP); a character-level language model is
then trained with a hand-written backward pass. The chapter also covers how the
magnitude of the gradients changes with the depth of a network — vanishing and
exploding gradients — and how initialization, normalization, residual connections,
and gradient clipping control it, using measured statistics of activations and
gradients. The character-level model is the starting point of the Transformer
language model built in Chapters 12 and 14, and the gradient diagnostics are used to
debug training runs in the later chapters.

## Topics

- Computational graphs: an expression as a directed acyclic graph of elementary
  operations
- The chain rule on a graph: forward-mode and reverse-mode automatic differentiation
- A scalar autograd engine: value nodes, local derivatives, topological ordering,
  and gradient accumulation
- Gradient checking with finite differences
- Backpropagation in vector and matrix form: linear layers, activation functions,
  and softmax with cross-entropy
- A neural network library on top of the engine: neurons, layers, an MLP, and
  parameter updates by stochastic gradient descent (SGD)
- A character-level language model: bigram counts, then an MLP over learned
  character vectors (embeddings)
- A hand-written backward pass for the MLP, including batch normalization, verified
  against PyTorch autograd
- Vanishing and exploding gradients: saturating activation functions, depth, and
  the scale of the initial weights
- Diagnostics: statistics of activations and gradients, and the ratio of update size
  to parameter size
- Remedies: Xavier and He initialization, normalization, residual connections, and
  gradient clipping
- PyTorch autograd as the same design: dynamic graphs and vector-Jacobian products

## Resources

**Suggested path.** Watch chapters 3 and 4 of the 3Blue1Brown series and read the
article by Olah, then follow lecture 1 of Neural Networks: Zero to Hero while typing
the code, which produces the autograd engine of the milestone. Continue with
lectures 2–5 for the character-level language model, the gradient diagnostics, and
the hand-written backward pass; in parallel, read chapter 2 of Nielsen or the CS231n
backpropagation note for the derivation, and the handout by Johnson before lecture 5.
Read chapter 5 of Nielsen and the paper by Glorot and Bengio for vanishing gradients,
and finish with the PyTorch autograd tutorial. When time is short, skip lecture 6,
CSC321, MiniTorch, Deep Learning Systems, and the items marked advanced.

### University courses

- Stanford — [CS231n: Deep Learning for Computer Vision](https://cs231n.stanford.edu/)
  by Fei-Fei Li, Ehsan Adeli, Justin Johnson, and Zane Durante (free; start here:
  the course note [Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/)
  covers local gradients, gates, staged computation, and patterns in the backward
  flow; in [assignment 1 (2025)](https://cs231n.github.io/assignments2025/assignment1/),
  questions 3 and 5 implement the forward and backward passes of fully connected
  networks in NumPy, with gradient checks provided).
- University of Toronto — [CSC321: Intro to Neural Networks and Machine Learning (Winter 2018)](https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/)
  by Roger Grosse (free slides and lecture notes; lecture 6 derives backpropagation
  on computational graphs, and lecture 10 describes the implementation of an
  automatic differentiation system).
- Cornell Tech — [MiniTorch](https://minitorch.github.io/) by Sasha Rush (free;
  optional; a teaching library in which the Torch API is reimplemented module by
  module — scalar automatic differentiation, tensors, efficient operations, and
  networks — with unit tests that check each step; a longer second project after
  the milestone).
- CMU — [10-714: Deep Learning Systems](https://dlsyscourse.org/) by Tianqi Chen and
  Tim Dettmers (advanced; free slides, notebooks, and assignments, with recordings
  of the 2022 offering on YouTube; the assignments build "Needle", a deep learning
  library with automatic differentiation, layers, optimizers, and a graphics
  processing unit (GPU) backend; requires C++, so it fits after Chapter 16).

### Online courses (MOOCs)

- Andrej Karpathy — [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
  (free; 8 lectures of 1–2.5 hours; start here: lecture 1 builds
  [micrograd](https://github.com/karpathy/micrograd), a scalar autograd engine of
  about 100 lines with a neural network library of about 50 lines; lectures 2–3
  build makemore, a character-level language model, first from bigram counts and
  then as an MLP; lecture 4 measures activation and gradient statistics and adds
  batch normalization; lecture 5, "Becoming a Backprop Ninja", writes the backward
  pass of that MLP by hand; lecture 6 is optional; lectures 7 and 8, on a generative
  pre-trained transformer (GPT) and its tokenizer, are used in Chapters 12 and 13;
  notebooks on [GitHub](https://github.com/karpathy/nn-zero-to-hero), videos as a
  [YouTube playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)).

### Books

- Book: Michael Nielsen, *Neural Networks and Deep Learning* (Determination Press,
  2015) — [official page](http://neuralnetworksanddeeplearning.com/) (free online;
  start here: chapter 2, "How the backpropagation algorithm works", derives the four
  equations of backpropagation for an MLP, and chapter 5, "Why are deep neural
  networks hard to train?", measures vanishing and exploding gradients in a deep
  network; the code samples are written in Python 2).
- Book: Simon J.D. Prince, *Understanding Deep Learning* (MIT Press, 2023) —
  [official page](https://udlbook.github.io/udlbook/) (free PDF; chapter 7,
  "Gradients and initialization", with notebooks 7.1–7.3, which implement
  backpropagation for a toy model and for a deep network and then compare
  initialization schemes).
- Book: Christopher M. Bishop and Hugh Bishop, *Deep Learning: Foundations and
  Concepts* (Springer, 2024) — [official page](https://www.bishopbook.com/)
  (free to read online; optional; chapter 8, "Backpropagation", including Jacobian
  matrices and automatic differentiation).
- Book: Ian Goodfellow, Yoshua Bengio, and Aaron Courville, *Deep Learning* (MIT
  Press, 2016) — [official page](https://www.deeplearningbook.org) (free online;
  advanced; section 6.5, "Back-Propagation and Other Differentiation Algorithms", a
  formal treatment of computational graphs and reverse-mode differentiation).

### Lectures, papers and articles

- Lecture series: Grant Sanderson (3Blue1Brown), [Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
  (2017; free; start here: chapters 3 and 4, "Backpropagation, intuitively" and
  "Backpropagation calculus", about 25 minutes in total; chapters 1 and 2, on the
  structure of a network and on gradient descent, belong to Chapter 9; the chapters
  added in 2024 cover large language models and the Transformer and are listed in
  Chapter 12).
- Article: Christopher Olah, [Calculus on Computational Graphs: Backpropagation](https://colah.github.io/posts/2015-08-Backprop/)
  (2015; forward-mode and reverse-mode differentiation on a graph, and why
  reverse mode computes the derivatives of one output with respect to all inputs in
  a single pass).
- Article: Justin Johnson, [Derivatives, Backpropagation, and Vectorization](https://cs231n.stanford.edu/handouts/derivatives.pdf)
  (Stanford CS231n handout, 2017; 7 pages; extends the scalar chain rule to vectors,
  matrices, and tensors, and derives the gradient of a linear layer without forming
  the full Jacobian).
- Paper: David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams,
  [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0)
  (Nature, 1986; paid access; optional; four pages; the paper that introduced
  backpropagation to neural network research).
- Papers: Xavier Glorot and Yoshua Bengio, [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html)
  (International Conference on Artificial Intelligence and Statistics, 2010), and
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun,
  [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)
  (International Conference on Computer Vision, 2015; measurements of how sigmoid
  layers saturate and how gradients shrink with depth, and the derivations of
  Xavier initialization and of He initialization for networks of rectified linear
  units (ReLU); section 2.2 of the second paper suffices).
- Paper: Atılım Güneş Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, and
  Jeffrey Mark Siskind, [Automatic differentiation in machine learning: a survey](https://arxiv.org/abs/1502.05767)
  (Journal of Machine Learning Research, 2018; advanced; distinguishes automatic
  differentiation from numerical and symbolic differentiation and surveys forward
  and reverse mode and their implementations).
- Tutorial: PyTorch, [A Gentle Introduction to torch.autograd](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
  (current docs; the dynamic graph and the vector-Jacobian products of PyTorch;
  intended for reading after the milestone engine works, for comparison with it).

## Milestone

Implement a scalar reverse-mode autograd engine in plain Python, modeled on
micrograd (addition, multiplication, powers, `exp`, `log`, and `tanh` or ReLU), and
verify its gradients on at least three expressions against finite differences and
against PyTorch, to within 1e-6. Use the engine to train an MLP on a two-class toy
dataset such as scikit-learn's `make_moons` to at least 95% accuracy, recording the
loss at every step. Then train a character-level MLP language model on the names
dataset of makemore with PyTorch tensors and a hand-written backward pass (no call
to `loss.backward()`), check every gradient against autograd, and report a
validation loss lower than that of a bigram count model on the same split.

## Estimated time

4–6 weeks.
