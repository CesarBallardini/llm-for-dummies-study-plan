# Chapter 8 — Machine Learning Foundations

> Part III — Machine Learning and Deep Learning · 6–8 weeks

## What you will learn

This chapter introduces machine learning: the parameters of a model are fitted to
data by minimizing a loss function (training), and the fitted model is then used to
make predictions on new data (inference). The chapter covers linear regression,
logistic and softmax regression, decision trees and their ensembles, and two
unsupervised methods, k-means clustering and principal component analysis (PCA), all
implemented with scikit-learn. It also covers the methodology that applies to every
model: training, validation, and test splits, overfitting and regularization,
cross-validation, and evaluation metrics. The same concepts apply to large language
models (LLMs): a language model is a classifier that predicts the next unit of text
(a token) from a fixed vocabulary, it is trained with the cross-entropy loss of
Chapter 4, and it is evaluated on held-out data.

## Topics

- Types of learning (supervised, unsupervised, and reinforcement learning)
- Models, parameters, loss functions, and the difference between training and inference
- Linear regression, fitted by least squares and by gradient descent
- Logistic regression and softmax regression for binary and multiclass classification
- Feature preparation (scaling, one-hot encoding) and scikit-learn pipelines
- Training, validation, and test splits, overfitting, underfitting, and generalization
- Regularization (L1 and L2 penalties)
- The bias-variance trade-off, cross-validation, and hyperparameter selection
- Decision trees and ensembles (random forests, gradient boosting)
- Unsupervised learning (k-means clustering and PCA)
- Evaluation metrics (accuracy, precision, recall, F1 score, confusion matrix, ROC
  curve), baselines, and class imbalance

## Resources

**Suggested path.** Take courses 1 and 2 of the Machine Learning Specialization as
the main course, and read *An Introduction to Statistical Learning* in parallel:
chapters 2–6 first, then chapters 8 and 12. Use the Kaggle Learn course or the Géron
notebooks for scikit-learn practice, read the article by Domingos after the first
weeks of the course, and consult the scikit-learn metrics guide while building the
milestone. Readers who are comfortable with the mathematics of Part I can replace the
Specialization with the CS229 lectures and notes. When time is short, skip course 3
of the Specialization, Learning From Data, and the books marked advanced.

### University courses

- Stanford — [CS229: Machine Learning (Autumn 2018 lectures)](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU)
  by Andrew Ng (free videos; start here: 20 lectures with full derivations of linear
  and logistic regression, generalized linear models, regularization, k-means, and
  PCA; the current [CS229 lecture notes](https://cs229.stanford.edu/main_notes.pdf)
  by Andrew Ng and Tengyu Ma are a free PDF, of which Part I, chapters 1–2, and
  Part III, on generalization and regularization, apply here).
- MIT — [6.390 Introduction to Machine Learning lecture notes](https://introml.mit.edu/notes/)
  (free; notes on regression, gradient descent, classification, and feature
  representation; the earlier [6.036 (Fall 2020) on OCW](https://ocw.mit.edu/courses/6-036-introduction-to-machine-learning-fall-2020/)
  by Leslie Kaelbling, Tomás Lozano-Pérez, Isaac Chuang, and Duane Boning adds
  videos and exercises).
- Caltech — [Learning From Data](https://work.caltech.edu/telecourse) by Yaser
  Abu-Mostafa (free; optional; 18 recorded lectures, 8 homework sets, and a final;
  covers the theory of generalization: VC dimension, bias-variance, overfitting, and
  validation).

### Online courses (MOOCs)

- DeepLearning.AI and Stanford Online (Coursera) — [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction)
  by Andrew Ng (free to audit, paid certificate; 3 courses, about 2 months at
  10 h/week; start here: Python-based and less mathematical than CS229; course 1
  covers linear and logistic regression, course 2 neural networks, decision trees,
  and model evaluation, and course 3 clustering, recommender systems, and
  reinforcement learning).
- Google — [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
  (free; 12 modules with interactive visualizations and exercises; linear and
  logistic regression, classification metrics, data preparation, and overfitting).
- Kaggle Learn — [Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning)
  (free; about 3 hours; decision trees, validation, underfitting and overfitting,
  and random forests in hands-on notebooks).

### Books

- Book: Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, and Jonathan
  Taylor, *An Introduction to Statistical Learning with Applications in Python*
  (Springer, 2023; the R edition is the 2nd ed., 2021) —
  [official page](https://www.statlearning.com) (free PDF; start here: chapters 2–6
  on statistical learning, linear regression, classification, resampling methods,
  and regularization, then chapter 8 on tree-based methods and chapter 12 on
  unsupervised learning).
- Book: Aurélien Géron, *Hands-On Machine Learning with Scikit-Learn and PyTorch*
  (O'Reilly, 1st ed., 2025) — [official notebooks](https://github.com/ageron/handson-mlp)
  (paid book, free notebooks; chapters 1–8: an end-to-end scikit-learn project,
  classification metrics, training linear models, decision trees, ensembles,
  dimensionality reduction, and clustering; chapters 9–11 are used in Chapter 9).
- Book: Andriy Burkov, *The Hundred-Page Machine Learning Book* (2019) —
  [official page](https://themlbook.com/) (free to read online under the author's
  "read first, buy later" policy; a short overview of the whole field, usable as a
  preview before a course or as a review after it).
- Book: Simon J.D. Prince, *Understanding Deep Learning* (MIT Press, 2023) —
  [official page](https://udlbook.github.io/udlbook/) (free PDF; chapter 2,
  "Supervised learning", chapter 5, "Loss functions", and chapter 8, "Measuring
  performance"; the same book is the main text of Chapter 9).
- Book: Trevor Hastie, Robert Tibshirani, and Jerome Friedman, *The Elements of
  Statistical Learning* (Springer, 2nd ed., 2009) —
  [official page](https://hastie.su.domains/ElemStatLearn/) (advanced; free PDF;
  chapters 2–4 and chapter 7, "Model Assessment and Selection"; the graduate-level
  counterpart of *An Introduction to Statistical Learning*).

### Lectures, papers and articles

- Article: Pedro Domingos, [A Few Useful Things to Know About Machine Learning](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)
  (Communications of the ACM, 2012; start here: twelve practical lessons on
  generalization, overfitting, feature engineering, and the curse of
  dimensionality; assumes the vocabulary of the first weeks of a course).
- Docs: scikit-learn, [Metrics and scoring: quantifying the quality of predictions](https://scikit-learn.org/stable/modules/model_evaluation.html)
  (current docs; reference for accuracy, precision, recall, F1 score, confusion
  matrices, ROC curves, and dummy baselines, as required by the milestone).

## Milestone

Train a logistic regression classifier on scikit-learn's digits dataset with a
stratified 80/20 train/test split, select the regularization strength by 5-fold
cross-validation on the training set only, and report the test accuracy (at least
95%), the confusion matrix, and the per-class precision and recall next to a
`DummyClassifier` baseline. Then relabel the task as "digit 9 versus all other
digits", and explain in one paragraph, using the measured accuracy, precision, and
recall of the majority-class baseline, why accuracy alone is misleading on
imbalanced data.

## Estimated time

6–8 weeks.
