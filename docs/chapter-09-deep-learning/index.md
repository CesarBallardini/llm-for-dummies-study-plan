# Chapter 9 — Deep Learning: Concepts and Practice

> Part III — Machine Learning and Deep Learning · 8–12 weeks

## What you will learn

This chapter covers neural networks: functions composed of layers, each a weighted
sum followed by a nonlinear activation function, whose parameters are fitted jointly
by gradient descent on a loss function. It treats the multilayer perceptron (MLP),
the output layers and loss functions for regression and classification,
backpropagation as the algorithm that computes the gradients, and the optimizers of
Chapter 5: stochastic gradient descent (SGD), Adam, and AdamW. It then covers the
methods that keep the training of deep networks stable and limit overfitting — weight
initialization, normalization, residual connections, dropout, weight decay, and early
stopping — and a procedure for diagnosing a training run from its loss curves. All
models are implemented in PyTorch, including the training loop, execution on a
graphics processing unit (GPU), and the mixed precision training introduced in
Chapter 7. Except for attention, every component of the Transformer of Chapter 12
(linear layers, Gaussian error linear unit (GELU) activations, layer normalization,
residual connections, dropout, and AdamW) is introduced here; backpropagation is
implemented by hand in Chapter 10.

## Topics

- The artificial neuron, the perceptron, and the multilayer perceptron (MLP):
  layers, weights, and biases
- Activation functions — sigmoid, tanh, the rectified linear unit (ReLU), and GELU —
  and the universal approximation theorem
- Output layers and loss functions: mean squared error for regression, softmax with
  cross-entropy for classification
- Backpropagation: the chain rule applied layer by layer to compute the gradient of
  the loss (implemented by hand in Chapter 10)
- PyTorch model building: `nn.Module`, `Dataset` and `DataLoader`, and automatic
  differentiation (autograd)
- The training loop: mini-batches, epochs, forward pass, backward pass, optimizer
  step, and evaluation on a validation set
- Optimizers (SGD with momentum, Adam, AdamW) and learning-rate schedules (warmup,
  decay)
- Weight initialization (Xavier and He) and normalization (batch normalization,
  layer normalization)
- Residual connections and the training of deep networks
- Regularization (weight decay, dropout, early stopping, data augmentation)
- Training on a GPU and mixed precision training (`torch.autocast`, loss scaling)
- Hyperparameter tuning and the diagnosis of training runs from training and
  validation loss curves
- Convolutional networks in overview (recurrent networks follow in Chapter 11)

## Resources

**Suggested path.** Watch lecture 1 of MIT 6.S191 and the first two videos of the
3Blue1Brown series listed in Chapter 10 as an overview, then read chapters 3–9 of
*Understanding Deep Learning* as the main text and work through its notebooks. In
parallel, take courses 1 and 2 of the Deep Learning Specialization, which continue
the Machine Learning Specialization of Chapter 8, and practice PyTorch with units
3, 4, 6, and 9 of Deep Learning Fundamentals or chapters 3–6 of *Dive into Deep
Learning*. Before building the milestone, read A Recipe for Training Neural Networks
and the CS231n note "Learning and Evaluation". When time is short, skip the fast.ai
course, the books by Bishop and by Goodfellow et al., the original papers, and
courses 3–5 of the Specialization (project strategy, convolutional networks, and
sequence models).

### University courses

- MIT — [6.S191: Introduction to Deep Learning](https://introtodeeplearning.com/)
  by Alexander Amini and Ava Amini (free; slides, videos, and three software labs,
  renewed every year; start here: lecture 1 of the 2026 edition, about one hour,
  surveys this chapter — the perceptron, MLPs, loss functions, backpropagation,
  optimizers, and regularization; the later lectures cover sequence models, computer
  vision, and generative models, and lab 3 fine-tunes a large language model (LLM);
  the [2020 edition](https://ocw.mit.edu/courses/6-s191-introduction-to-deep-learning-january-iap-2020/)
  is archived on MIT OpenCourseWare).
- Stanford — [CS231n: Deep Learning for Computer Vision](https://cs231n.stanford.edu/)
  by Fei-Fei Li, Ehsan Adeli, Justin Johnson, and Zane Durante (free
  [course notes](https://cs231n.github.io/) and
  [2025 lecture videos](https://www.youtube.com/playlist?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16);
  the three "Neural Networks" notes cover activation functions, data preprocessing,
  weight initialization, regularization, gradient checks, and the monitoring of a
  training run; the third note, "Learning and Evaluation", is the reference for the
  milestone; the notes on convolutional networks cover the overview topic of this
  chapter; assignment 1 is used in Chapter 10).

### Online courses (MOOCs)

- DeepLearning.AI (Coursera) — [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)
  by Andrew Ng, Younes Bensouda Mourri, and Kian Katanforoosh (free to audit, paid
  certificate; 5 courses, about 3 months at 10 h/week; start here: course 1, "Neural
  Networks and Deep Learning", develops the MLP and its training step by step, and
  course 2, "Improving Deep Neural Networks: Hyperparameter Tuning, Regularization
  and Optimization", covers initialization, regularization, optimizers, batch
  normalization, and hyperparameter tuning; week 2 of course 2 is used in Chapter 5;
  the programming exercises use NumPy and TensorFlow, so PyTorch practice comes from
  the other resources).
- Lightning AI — [Deep Learning Fundamentals](https://lightning.ai/pages/courses/deep-learning-fundamentals/)
  by Sebastian Raschka (free; 10 units of short videos with PyTorch code and
  exercises; units 3 and 4 train logistic regression and MLPs in PyTorch with data
  loaders; unit 6 covers early stopping, learning-rate schedulers, activation
  functions, batch normalization, dropout, and the debugging of networks; unit 9
  covers mixed precision training; unit 5 introduces the Lightning library, which
  the later units use and this book does not).
- fast.ai — [Practical Deep Learning for Coders](https://course.fast.ai) by Jeremy
  Howard (free; optional; Part 1, 2022: 9 lessons of about 90 minutes; an
  alternative route that trains complete models with the fastai library from the
  first lesson and introduces the theory afterwards; the companion book, Jeremy
  Howard and Sylvain Gugger, *Deep Learning for Coders with fastai and PyTorch*
  (O'Reilly, 2020), is free as [Jupyter notebooks](https://github.com/fastai/fastbook)).

### Books

- Book: Simon J.D. Prince, *Understanding Deep Learning* (MIT Press, 2023) —
  [official page](https://udlbook.github.io/udlbook/) (free PDF; start here:
  chapters 3–9 cover shallow and deep networks, loss functions, fitting models,
  gradients and initialization, measuring performance, and regularization, with
  Python notebooks for each chapter; chapter 10 covers convolutional networks, and
  chapter 11 residual networks and batch normalization).
- Book: Aston Zhang, Zachary C. Lipton, Mu Li, and Alexander J. Smola, *Dive into
  Deep Learning* (Cambridge University Press, 2023) — [official page](https://d2l.ai/)
  (free online; chapters 3–6 implement linear regression, softmax regression, MLPs,
  and custom PyTorch modules, first from scratch and then with the library;
  sections 8.5 and 8.6 cover batch normalization and residual networks; chapter 12,
  on optimization algorithms, is listed in Chapter 5; every example runs in
  PyTorch).
- Book: Aurélien Géron, *Hands-On Machine Learning with Scikit-Learn and PyTorch*
  (O'Reilly, 1st ed., 2025) — [official notebooks](https://github.com/ageron/handson-mlp)
  (paid book, free notebooks; the continuation of the chapters used in Chapter 8:
  chapter 9 introduces artificial neural networks, chapter 10 builds them in
  PyTorch, and chapter 11 covers initialization, normalization, optimizers,
  learning-rate schedules, and regularization; appendix B covers mixed precision).
- Book: Christopher M. Bishop and Hugh Bishop, *Deep Learning: Foundations and
  Concepts* (Springer, 2024) — [official page](https://www.bishopbook.com/)
  (free to read online; optional; chapters 6–9: deep neural networks, gradient
  descent, backpropagation, and regularization, with more probability theory than
  the other books listed).
- Book: Ian Goodfellow, Yoshua Bengio, and Aaron Courville, *Deep Learning* (MIT
  Press, 2016) — [official page](https://www.deeplearningbook.org) (free online;
  advanced; chapters 6–8 on feedforward networks, regularization, and optimization,
  and chapter 11, "Practical Methodology"; chapters 4 and 8 are listed in Chapter 5).

### Lectures, papers and articles

- Article: Andrej Karpathy, [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)
  (2019; start here: a procedure for building a model in stages — inspect the data,
  establish a baseline, overfit a single batch, then regularize and tune — designed
  to expose errors that do not raise exceptions; the milestone follows this order).
- Article: Varun Godbole, George E. Dahl, Justin Gilmer, Christopher J. Shallue, and
  Zachary Nado, [Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook)
  (Google Research, 2023; free; a procedure for hyperparameter tuning: which
  hyperparameters to tune first, how to choose the batch size, and how to read
  training curves).
- Papers: Diederik P. Kingma and Jimmy Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
  (International Conference on Learning Representations (ICLR), 2015), and Ilya
  Loshchilov and Frank Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
  (ICLR, 2019; also listed in Chapter 5; Adam and its variant AdamW, the default
  optimizers from this chapter onward).
- Paper: Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and
  Ruslan Salakhutdinov, [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html)
  (Journal of Machine Learning Research, 2014; defines dropout and interprets it as
  the averaging of an ensemble of thinned networks).
- Papers: Sergey Ioffe and Christian Szegedy, [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
  (International Conference on Machine Learning, 2015); Jimmy Lei Ba, Jamie Ryan
  Kiros, and Geoffrey E. Hinton, [Layer Normalization](https://arxiv.org/abs/1607.06450)
  (2016); and Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun,
  [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
  (Conference on Computer Vision and Pattern Recognition, 2016; optional; the
  original descriptions of batch normalization, of layer normalization, and of
  residual connections; the last two are components of every Transformer block in
  Chapter 12).
- Docs: PyTorch, [Automatic Mixed Precision package - torch.amp](https://docs.pytorch.org/docs/stable/amp.html)
  (current docs; `autocast` and `GradScaler`, introduced in Chapter 7 and applied
  here to a training loop; the method is defined in Paulius Micikevicius et al.,
  [Mixed Precision Training](https://arxiv.org/abs/1710.03740) (ICLR, 2018), listed
  in Chapter 5).

## Milestone

Train an MLP in PyTorch to classify the sentiment of the
[IMDb movie review dataset](https://huggingface.co/datasets/stanfordnlp/imdb)
(25,000 training and 25,000 test reviews), using as input bag-of-words counts of the
10,000 most frequent words computed with scikit-learn's `CountVectorizer`, and reach
a test accuracy of at least 85%, reported next to a logistic regression baseline on
the same features. Follow the order of A Recipe for Training Neural Networks: verify
first that the model can overfit a single batch, then train on the full training set
with a held-out validation set and plot the training and validation loss curves.
Produce one run that overfits and one in which dropout, weight decay, or early
stopping reduces the gap between the two curves, and state which change had the
largest effect.

## Estimated time

8–12 weeks.
