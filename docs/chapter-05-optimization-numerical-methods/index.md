# Chapter 5 — Optimization, Numerical Methods, and Math for ML Consolidation

> Part I — Mathematical Foundations · 4–6 weeks

## What you will learn

This chapter covers the optimization algorithms that train neural networks and the
numerical issues that arise when they run in finite-precision arithmetic, and it
consolidates the mathematics of Chapters 2–4 for machine learning (ML). The
optimization part treats gradient descent and its variants — stochastic gradient
descent (SGD), momentum, and the adaptive methods Adam and AdamW — together with
learning-rate schedules, convexity, and conditioning. The numerical part treats
floating-point arithmetic (rounding, overflow, and underflow) and the number formats
in which large language models (LLMs) are trained and served: 32-bit floating point
(fp32) as the baseline, and 16-bit floating point (fp16), bfloat16 (bf16), and 8-bit
integers (int8) to reduce memory and computation. The consolidation part combines
calculus, linear algebra, and probability in three standard problems — least squares,
maximum likelihood estimation (MLE), and principal component analysis (PCA) — each
formulated and solved as an optimization problem.

## Topics

- Objective functions, minima, and convex versus non-convex functions
- Gradient descent, step size, and convergence
- Conditioning and scaling of a problem
- Newton's method and second-order information
- Stochastic and mini-batch gradient descent
- Momentum and adaptive methods (RMSProp, Adam, AdamW)
- Learning-rate schedules (warmup, decay)
- Loss surfaces of neural networks: local minima, saddle points, and plateaus
- Floating-point representation, rounding error, overflow, and underflow
- Numerically stable computation: the softmax function (exponential normalization of
  scores into probabilities), the log-sum-exp trick, and gradient checking with
  finite differences
- Precision formats (fp32, fp16, bf16, int8) and mixed-precision training
- Consolidation: least squares, maximum likelihood, and PCA as optimization problems

## Resources

**Suggested path.** Start with chapter 7 of *Mathematics for Machine Learning* and
the overview by Ruder, then watch lectures 21–23 and 25 of MIT 18.065 and complete
week 2 of Improving Deep Neural Networks. For the numerical topics, read chapter 4 of
*Deep Learning*, the first two sections of the article by Goldberg, the note on
bfloat16 by Higham, and the Mixed Precision Training paper. For consolidation, run
the Imperial College London specialization in parallel: complete course 3 (PCA) in
full and use the quizzes of courses 1 and 2 as a test of Chapters 3 and 2; the
programming labs and the milestone assume the basic Python and NumPy of Chapter 6.
When time is short, skip EE364a, *Convex Optimization*, *Dive into Deep Learning*,
and the DeepLearning.AI specialization.

### University courses

- MIT OpenCourseWare — [18.065 Matrix Methods in Data Analysis, Signal Processing,
  and Machine
  Learning](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/)
  by Gilbert Strang (Spring 2018; free; lecture videos + problem sets; start here:
  lectures 21–23 and 25 cover step-by-step minimization, gradient descent, momentum,
  and SGD, building on the linear algebra of Chapter 3; lectures 26 and 27 introduce
  neural networks and backpropagation).
- Stanford — [EE364a Convex Optimization
  I](https://www.youtube.com/playlist?list=PLoROMvodv4rMJqxxviPa4AmDClvcbHi6h) by
  Stephen Boyd (2023 lectures; free videos; optional; slides and exercises on the
  [course site](https://web.stanford.edu/class/ee364a/); the standard course on
  convex sets, convex functions, and duality; shows why convex problems can be solved
  reliably, which non-convex problems such as neural network training cannot).

### Online courses (MOOCs)

- DeepLearning.AI (Coursera) — [Improving Deep Neural Networks: Hyperparameter
  Tuning, Regularization and
  Optimization](https://www.coursera.org/learn/deep-neural-network) by Andrew Ng
  (free to audit, paid certificate; 3 weeks, about 20 hours; start here: week 2
  develops mini-batch gradient descent, momentum, RMSProp, Adam, and learning-rate
  decay step by step; course 2 of the Deep Learning Specialization listed in Chapter
  9).
- Imperial College London (Coursera) — [Mathematics for Machine Learning
  Specialization](https://www.coursera.org/specializations/mathematics-machine-learning)
  by David Dye, Samuel J. Cooper, A. Freddie Page, and Marc Peter Deisenroth (free to
  audit per course, paid certificate, financial aid available; 3 courses of about 20
  hours each: linear algebra, multivariate calculus, and PCA; the consolidation
  course of Part I: course 2 applies gradients to curve fitting and to training a
  small neural network, and course 3 derives PCA from projections and variances and
  implements it in NumPy).
- DeepLearning.AI (Coursera) — [Mathematics for Machine Learning and Data Science
  Specialization](https://www.coursera.org/specializations/mathematics-for-machine-learning-and-data-science)
  by Luis Serrano (free to audit per course, paid certificate, financial aid
  available; 3 courses, about 90 hours: linear algebra, calculus, and probability and
  statistics; a slower-paced alternative to the Imperial College London
  specialization, with Python labs; course 2 covers gradient descent and Newton's
  method, and course 3 is the probability course listed in Chapter 4).

### Books

- Book: Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong, *Mathematics for
  Machine Learning* (Cambridge University Press, 2020) — [official
  page](https://mml-book.github.io) (free PDF; start here: chapter 7, "Continuous
  Optimization": gradient descent, constrained optimization, and convexity; for
  consolidation, chapters 2–7 restate the mathematics of Chapters 2–4 in one
  notation, and chapters 8–10 apply it to model fitting, linear regression, and PCA).
- Book: Ian Goodfellow, Yoshua Bengio, and Aaron Courville, *Deep Learning* (MIT
  Press, 2016) — [official page](https://www.deeplearningbook.org) (free online;
  chapter 4, "Numerical Computation", on overflow, underflow, and conditioning, and
  chapter 8, "Optimization for Training Deep Models").
- Book: Aston Zhang, Zachary C. Lipton, Mu Li, and Alexander J. Smola, *Dive into
  Deep Learning* (Cambridge University Press, 2023) — [chapter 12, "Optimization
  Algorithms"](https://d2l.ai/chapter_optimization/index.html) (free online; runnable
  code for SGD, momentum, Adagrad, RMSProp, Adam, and learning-rate scheduling).
- Book: Stephen Boyd and Lieven Vandenberghe, *Convex Optimization* (Cambridge
  University Press, 2004) — [official page](https://web.stanford.edu/~boyd/cvxbook/)
  (free PDF; optional; the textbook of EE364a; chapters 2–5 for convexity and
  duality, chapter 9 for unconstrained minimization and Newton's method).

### Lectures, papers and articles

- Paper: Sebastian Ruder, [An overview of gradient descent optimization
  algorithms](https://arxiv.org/abs/1609.04747) (2016; start here: a survey of SGD,
  momentum, Nesterov, Adagrad, RMSProp, and Adam, with all update rules in one
  notation).
- Article: Gabriel Goh, [Why Momentum Really
  Works](https://distill.pub/2017/momentum/) (Distill, 2017; interactive analysis of
  momentum on a quadratic function, relating the convergence rate to the eigenvalues
  and the condition number).
- Papers: Diederik P. Kingma and Jimmy Ba, [Adam: A Method for Stochastic
  Optimization](https://arxiv.org/abs/1412.6980) (International Conference on
  Learning Representations (ICLR), 2015), and Ilya Loshchilov and Frank Hutter,
  [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) (ICLR,
  2019; the Adam optimizer and its variant AdamW, which separates weight decay from
  the gradient update and is the usual optimizer for LLM pretraining).
- Paper: David Goldberg, [What Every Computer Scientist Should Know About
  Floating-Point
  Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) (ACM
  Computing Surveys, 1991; rounding error, the IEEE 754 standard, overflow, and
  underflow; the first two sections are sufficient).
- Paper: Paulius Micikevicius et al., [Mixed Precision
  Training](https://arxiv.org/abs/1710.03740) (ICLR, 2018; fp16 training with an fp32
  master copy of the weights and loss scaling; the basis of current fp16 and bf16
  training recipes).
- Article: Nicholas J. Higham, [What Is Bfloat16
  Arithmetic?](https://nhigham.com/2020/06/02/what-is-bfloat16-arithmetic/) (2020; a
  short note that compares the range and precision of bf16, fp16, and fp32).
- Article: Rich Sutton, [The Bitter
  Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) (2019; a short
  essay arguing that general methods that scale with computation have outperformed
  methods built on hand-coded domain knowledge; background for the scaling approach
  of Chapter 15).

## Milestone

Fit a logistic regression with two parameters (a logistic function applied to
`w x + b`) to a small synthetic dataset in NumPy: derive the gradient of the
cross-entropy loss by hand and verify it with finite differences. Implement gradient
descent, momentum, and Adam from scratch, and plot or animate the three trajectories
on the contour plot of the two-dimensional loss surface. Then compute a softmax in
fp16 with inputs large enough to overflow, and correct it with the log-sum-exp trick.

## Estimated time

4–6 weeks.
