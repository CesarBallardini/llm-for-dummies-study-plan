# Study Plan: From College Algebra to Building, Training, Operating and Improving Open-Source LLMs

This plan is a curriculum that starts at college algebra and calculus, builds the mathematics and programming foundations, continues through machine learning and deep learning, and then covers Transformer-based large language models (LLMs) in depth: pretraining, fine-tuning, alignment, evaluation, serving, and the open-source ecosystem.

All required material is free or low-cost, and every resource is linked directly. The university courses listed publish their lecture videos, slides, and assignments openly, even when enrollment for credit is restricted.

## How to Use This Plan

The plan has six parts and 24 chapters. Every chapter has the same structure:

- **Goals**: the scope of the chapter and how the rest of the course uses it.
- **Topics**: the syllabus, in study order.
- **Resources**: a suggested path, followed by university courses, online courses (MOOCs), books, and lectures, papers, and articles. The recommended starting point in each group is marked "start here"; optional and advanced items are labeled as such.
- **Milestone**: a hands-on exercise that applies the main ideas of the chapter and produces a result that can be verified.
- **Time**: an estimate based on about 10 hours of study per week.

The chapters are designed to be followed in order, because each one assumes the material of the chapters before it. The exception is Part II, which runs in parallel with Part I. A reader who already knows a topic can skim the chapter and use the milestone as a diagnostic test. The milestones are the course's mechanism for checking progress, so completing each one before moving on is recommended.

The estimated total is 24–36 months at 10 hours per week, or 12–18 months at 20 hours per week. Part I accounts for a large share of that time because every later part depends on its mathematics.

### Map of Goals to Chapters

| Goal | Relevant chapters |
| --- | --- |
| Understand how LLMs work internally | 13, 14, 15, 16 |
| Pretrain an open-source LLM | 14, 15, 16, 17 |
| Improve an existing model (training and research) | 16, 17, 18, 19, 20, 24 |
| Serve, deploy, and operate an LLM | 21, 22, 23 |
| Build applications on top of open models | 21, 23, 24 |

---

# Part I — Mathematical Foundations

## Chapter 1 — College Algebra and Pre-calculus

**Goals.** This chapter covers the algebra and precalculus that every later chapter assumes: functions and their graphs; polynomial, rational, exponential, and logarithmic functions; trigonometric functions; summation notation; and an informal notion of a limit. Exponentials and logarithms are the components of the functions that convert the raw outputs of a large language model (LLM) into probabilities, and of the loss function used to train it. Sines and cosines are used by the Transformer architecture of Chapter 12 to encode the position of each word, and summation notation appears in nearly every formula of the machine learning literature. The informal treatment of limits prepares the definitions of the derivative and the integral in Chapter 2.

**Topics.** Real numbers, exponents, radicals, and factoring; equations and inequalities; functions: notation, domain, range, and graphs; linear and quadratic functions; polynomial and rational functions; function transformations, composition, and inverses; exponential and logarithmic functions and the laws of logarithms; trigonometric functions, the unit circle, and the basic identities; sequences, series, and summation (sigma) notation; an informal notion of limits, as preparation for Chapter 2.

**Resources.**

**Suggested path.** Use Khan Academy College Algebra as the main course, with the matching chapters of the OpenStax *Precalculus* textbook as the written reference and source of extra exercises. Continue with the Khan Academy Precalculus units on composite and inverse functions, trigonometry, series, and limits and continuity; the Lockdown Math lectures can run in parallel with the trigonometry and logarithm units. Finish with the Algebra/Trig Review of Paul's Online Math Notes as a readiness check for Chapter 2, and with lesson 1 of The Essence of Calculus. When time is short, skip the Coursera specialization, the Stewart textbook, and the Khan Academy Precalculus units on conic sections, vectors, and matrices (Chapter 3 covers vectors and matrices).

### Online courses (MOOCs)

- Khan Academy — [College Algebra](https://www.khanacademy.org/math/college-algebra) (free; videos + self-grading exercises, unit tests, and a course challenge; start here: linear, quadratic, polynomial, rational, exponential, and logarithmic functions; the [Khan Academy math hub](https://www.khanacademy.org/math) lists the earlier algebra courses for review of any weak skill).
- Khan Academy — [Precalculus](https://www.khanacademy.org/math/precalculus) (free; videos + self-grading exercises and unit tests; the second pass: composite and inverse functions, trigonometry, complex numbers, series, and limits and continuity).
- Professor Leonard — [Precalculus: College Algebra/Trigonometry](https://www.youtube.com/playlist?list=PLDesaqWTN6ESsmwELdrzhcGiRhk5DjwLP) (free YouTube playlist; more than 70 hours of full-length classroom lectures with worked examples; an alternative to the Khan Academy videos).
- Johns Hopkins University (Coursera) — [Algebra: Elementary to Advanced](https://www.coursera.org/specializations/algebra-elementary-to-advanced) by Joseph W. Cutrone (free to audit, paid certificate; optional; 3 courses, about 4 weeks at 10 hours/week; equations, functions, polynomials, and roots with graded quizzes).

### Books

- Book: Jay Abramson et al., *Precalculus* (2nd ed., OpenStax, 2021) — [official page](https://openstax.org/details/books/precalculus-2e) (free PDF and web; start here: chapters 1–4 for functions, polynomials, exponentials, and logarithms, 5–7 for trigonometry, 11 for sequences and series, and 12 for an introduction to limits).
- Book: Jay Abramson et al., *College Algebra* (2nd ed., OpenStax, 2021) — [official page](https://openstax.org/details/books/college-algebra-2e) (free PDF and web; chapter 1, "Prerequisites", and chapter 2, "Equations and Inequalities", review exponents, radicals, and factoring).
- Book: Carl Stitz and Jeff Zeager, *Precalculus* (3rd corrected ed., 2013) — [official page](https://www.stitz-zeager.com/) (free PDF, CC BY-NC-SA; more rigorous than the OpenStax text, with a large set of exercises and answers; chapters 1–6 for functions and logarithms, 9 for sequences, 10–11 for trigonometry).
- Book: James Stewart, Lothar Redlin, and Saleem Watson, *Precalculus: Mathematics for Calculus* (8th ed., Cengage, 2024) — [author page](https://www.stewartmath.com/) (paid; optional; a widely adopted US precalculus textbook, in the same notation as Stewart's *Calculus*, which is cited in Chapter 2).
- Book (reference): Paul Dawkins, *Paul's Online Math Notes* (Lamar University) — [Algebra](https://tutorial.math.lamar.edu/Classes/Alg/Alg.aspx) and the [Algebra/Trig Review](https://tutorial.math.lamar.edu/Extras/AlgebraTrigReview/AlgebraTrigIntro.aspx) (free web notes with worked problems; the review is a diagnostic set of the algebra and trigonometry problems that Calculus I assumes).

### Lectures, papers and articles

- Lecture series: Grant Sanderson (3Blue1Brown), [Lockdown Math](https://www.youtube.com/playlist?list=PLZHQObOWTQDP5CVelJJ1bNDouqrAhVPev) (2020; free; recorded live lectures; start here: quadratics, trigonometry, complex numbers, logarithms, and the number *e*, presented graphically).
- Lecture series: Grant Sanderson (3Blue1Brown), [The Essence of Calculus](https://www.3blue1brown.com/lessons/essence-of-calculus) (2017; free; lesson 1, watched at the end of this chapter, previews limits and rates of change; Chapter 2 uses the full series).

**Milestone.** Score at least 90% on the Khan Academy College Algebra course challenge and on the Khan Academy Precalculus unit tests for trigonometry, series, and limits and continuity. Then solve on paper, without notes: expand `log(a^2 b / c)` with the laws of logarithms and solve `3 · 2^x = 48` for `x`; write the sum of `k · 2^k` for `k` from 1 to 5 in sigma notation and evaluate it; and sketch `y = 2 sin(x − π/2) + 1` starting from the graph of `sin x`. Verify each answer with a graphing calculator or a computer algebra system.

**Time.** 3–6 weeks.

## Chapter 2 — Calculus I and II (Single and Multivariable)

**Goals.** This chapter covers single-variable and multivariable calculus: limits, derivatives, integrals, Taylor series, partial derivatives, gradients, Jacobians, Hessians, and the multivariable chain rule. A neural network is trained by gradient descent, which changes each parameter in the direction that lowers a loss function (a number that measures the error of the model); the required gradients are computed by backpropagation, which is the multivariable chain rule applied layer by layer. Integrals are required for the continuous probability distributions of Chapter 4, and Taylor approximations and Hessians for the optimization methods of Chapter 5. The emphasis is on differentiation and on functions of several variables; techniques of integration and the calculus of vector fields are not used in later chapters.

**Topics.** Limits and continuity; derivatives and the rules of differentiation, including the chain rule; optimization of single-variable functions; integrals and the fundamental theorem of calculus; sequences, series, and Taylor approximations; vectors and functions of several variables; partial derivatives; gradients and directional derivatives; the chain rule in several variables; Jacobians and Hessians; multivariable optimization: critical points, the second-derivative test, and Lagrange multipliers; double integrals, at the level needed for the joint probability densities of Chapter 4.

**Resources.**

**Suggested path.** Watch The Essence of Calculus first for the geometric interpretation. Follow 18.01SC units 1–3 and 5 (differentiation, applications of differentiation, the definite integral, and series), then 18.02SC units 1–2 and the double-integral sessions of unit 3, and work the Khan Academy exercises on the same topics in parallel. Finish with the article by Parr and Howard and chapter 5 of *Mathematics for Machine Learning*, which restate the chapter in the notation of machine learning. When time is short, skip 18.01SC unit 4 (techniques of integration), the 18.02SC sessions on line, triple, and surface integrals, and every item marked optional or advanced.

### University courses

- MIT OpenCourseWare — [18.01SC Single Variable Calculus](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/) by David Jerison (Fall 2010; free; lecture and recitation videos, problem sets and exams with solutions; start here: the OpenCourseWare Scholar version, arranged for independent study; the original course page is [18.01 Fall 2006](https://ocw.mit.edu/courses/18-01-single-variable-calculus-fall-2006/)).
- MIT OpenCourseWare — [18.02SC Multivariable Calculus](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/) by Denis Auroux (Fall 2010; free; lecture and recitation videos, problem sets and exams with solutions; vectors, partial derivatives, gradients, Lagrange multipliers, and multiple integrals; the original course page is [18.02 Fall 2007](https://ocw.mit.edu/courses/18-02-multivariable-calculus-fall-2007/)).
- MIT OpenCourseWare — [RES.18-005 Highlights of Calculus](https://ocw.mit.edu/courses/res-18-005-highlights-of-calculus-spring-2010/) by Gilbert Strang (Spring 2010; free; optional; short videos; an overview of the main ideas of differential and integral calculus, usable as a preview or as a refresher).

### Online courses (MOOCs)

- Khan Academy — [Calculus 1](https://www.khanacademy.org/math/calculus-1), [Calculus 2](https://www.khanacademy.org/math/calculus-2), and [Multivariable calculus](https://www.khanacademy.org/math/multivariable-calculus) (free; videos + self-grading exercises; start here: exercise sets for every differentiation and integration rule; the multivariable videos are by Grant Sanderson of 3Blue1Brown).
- MITx (MIT Learn) — [Calculus 1A: Differentiation](https://learn.mit.edu/courses/course-v1:MITxT+18.01.1x), [1B: Integration](https://learn.mit.edu/courses/course-v1:MITxT+18.01.2x), [1C: Coordinate Systems & Infinite Series](https://learn.mit.edu/courses/course-v1:MITxT+18.01.3x), and [Multivariable Calculus 1: Vectors and Derivatives](https://learn.mit.edu/courses/course-v1:MITxT+18.02.1x) (free to audit when a session is open, paid certificate; about one semester each; the content of 18.01 and of the first part of 18.02 with automatically graded problems).
- Professor Leonard — [Calculus 1](https://www.youtube.com/playlist?list=PLF797E961509B4EB5), [Calculus 2](https://www.youtube.com/playlist?list=PLDesaqWTN6EQ2J4vgsN1HyBeRADEh4Cw-), and [Calculus 3](https://www.youtube.com/playlist?list=PLDesaqWTN6ESk16YRmzuJ8f6-rnuy0Ry7) (free YouTube playlists; complete classroom lecture courses with worked examples, at a slower pace than the MIT lectures).
- Imperial College London (Coursera) — [Mathematics for Machine Learning: Multivariate Calculus](https://www.coursera.org/learn/multivariate-calculus-machine-learning), and DeepLearning.AI (Coursera) — [Calculus for Machine Learning and Data Science](https://www.coursera.org/learn/machine-learning-calculus) (free to audit, paid certificates; optional here; the calculus courses of the two mathematics-for-machine-learning specializations that Chapter 5 describes and uses for consolidation).

### Books

- Book: Gilbert Strang, *Calculus* (3rd ed., Wellesley-Cambridge Press, 2017) — [MIT OpenCourseWare page](https://ocw.mit.edu/courses/res-18-001-calculus-fall-2023/) (free PDF, CC BY-NC-SA; start here: chapters 1–8 and 10 for single-variable calculus and series, 11 and 13 for vectors and partial derivatives).
- Book: Gilbert Strang, Edwin Herman et al., *Calculus* Volumes 1–3 (OpenStax, 2016) — [Volume 1](https://openstax.org/details/books/calculus-volume-1), [Volume 2](https://openstax.org/details/books/calculus-volume-2), and [Volume 3](https://openstax.org/details/books/calculus-volume-3) (free PDF and web; all of Volume 1, Volume 2 chapters 5–6 for series and Taylor series, Volume 3 chapters 2 and 4 for vectors and multivariable differentiation).
- Book: James Stewart, Daniel Clegg, and Saleem Watson, *Calculus: Early Transcendentals* (9th ed., Cengage, 2020) — [author page](https://www.stewartcalculus.com/) (paid; optional; a widely adopted university calculus textbook; chapters 1–5, 11, 12, and 14).
- Book (reference): Paul Dawkins, *Paul's Online Math Notes* (Lamar University) — [Calculus I](https://tutorial.math.lamar.edu/Classes/CalcI/CalcI.aspx), [Calculus II](https://tutorial.math.lamar.edu/Classes/CalcII/CalcII.aspx), and [Calculus III](https://tutorial.math.lamar.edu/Classes/CalcIII/CalcIII.aspx) (free web notes with worked problems; a reference for individual techniques).
- Book: Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong, *Mathematics for Machine Learning* (Cambridge University Press, 2020) — [official page](https://mml-book.github.io) (free PDF; chapter 5, "Vector Calculus": gradients of vector- and matrix-valued functions, backpropagation, automatic differentiation, and the multivariate Taylor series).

### Lectures, papers and articles

- Lecture series: Grant Sanderson (3Blue1Brown), [The Essence of Calculus](https://www.3blue1brown.com/lessons/essence-of-calculus) (2017; free; 12 lessons; start here: the geometric interpretation of derivatives, integrals, the chain rule, and Taylor series; watch before or alongside the first course).
- Paper/Article: Terence Parr and Jeremy Howard, [The Matrix Calculus You Need For Deep Learning](https://arxiv.org/abs/1802.01528) (2018; connects this chapter to neural networks: Jacobians, derivatives of element-wise operations, and the vector chain rule).
- Paper/Article: Atilim Gunes Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, and Jeffrey Mark Siskind, [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767) (Journal of Machine Learning Research, 2018; advanced; how frameworks such as PyTorch apply the chain rule automatically).

**Milestone.** Compute by hand the gradient and the Hessian of `f(x, y) = x^2 y + e^(xy)`, and verify both with SymPy. For the composition `L = (s(w x + b) − y)^2`, where `s(z) = 1 / (1 + e^(−z))` is the logistic function, derive `dL/dw` and `dL/db` with the chain rule, and confirm the result numerically with central finite differences at one chosen point.

**Time.** 6–10 weeks.

## Chapter 3 — Linear Algebra

**Goals.** This chapter covers vectors, matrices, and the matrix factorizations built on them, in both their algebraic and their geometric interpretation. A large language model (LLM) represents each token (a unit of text) as a vector called an embedding, measures the relevance of one token to another with dot products (the attention scores of Chapter 12), and applies its weights by matrix multiplication, so every later chapter uses this material. Orthogonality, projections, and least squares underlie the regression methods of Chapter 8, and eigenvalues and positive definite matrices describe the curvature of the loss functions studied in Chapter 5. The singular value decomposition (SVD) and low-rank approximation are the basis of principal component analysis (PCA) and of the low-rank fine-tuning methods of Chapter 17.

**Topics.** Vectors, dot products, norms, and angles (cosine similarity); linear combinations, span, and linear independence; systems of linear equations and elimination (the LU factorization); matrices as linear transformations; matrix multiplication and inverses; vector spaces, bases, dimension, and rank; the four fundamental subspaces; orthogonality, projections, and least squares; Gram-Schmidt and the QR factorization; determinants; eigenvalues, eigenvectors, and diagonalization; symmetric and positive definite matrices; the singular value decomposition; low-rank approximation and principal component analysis.

**Resources.**

**Suggested path.** Watch Essence of Linear Algebra first, then follow 18.06SC from start to finish, with Strang's *Introduction to Linear Algebra* or the free text by Boyd and Vandenberghe as the written reference; the Khan Academy videos give slower derivations of individual topics where needed. Read chapters 2–4 and 10 of *Mathematics for Machine Learning* near the end, before the milestone. When time is short, skip the 18.06SC sessions on differential equations, Markov matrices and Fourier series, and complex matrices and the fast Fourier transform, and leave 18.065, Axler, and the other optional and advanced items for a second pass.

### University courses

- MIT OpenCourseWare — [18.06SC Linear Algebra](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/) by Gilbert Strang (Fall 2011; free; lecture and recitation videos, problem sets and exams with solutions; start here: the OpenCourseWare Scholar version of 18.06, arranged for independent study; elimination, the four fundamental subspaces, orthogonality, determinants, eigenvalues, and the SVD; the original course page is [18.06 Spring 2010](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)).
- MIT OpenCourseWare — [18.065 Matrix Methods in Data Analysis, Signal Processing, and Machine Learning](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/) by Gilbert Strang (Spring 2018; free; lecture videos + problem sets; advanced; the follow-on course to 18.06: the SVD, low-rank approximation, and PCA, followed by optimization and neural networks; Chapter 5 uses lectures 21–25).
- MIT OpenCourseWare — [RES.18-010 A Vision of Linear Algebra](https://ocw.mit.edu/courses/res-18-010-a-2020-vision-of-linear-algebra-spring-2020/) by Gilbert Strang (Spring 2020, with later additions; free; optional; short videos that summarize the subject through the factorization A = CR, the four fundamental subspaces, and the SVD; a review after 18.06SC).

### Online courses (MOOCs)

- Khan Academy — [Linear Algebra](https://www.khanacademy.org/math/linear-algebra) (free; videos; start here: step-by-step derivations of span, null space, column space, and change of basis, as a supplement to the 18.06SC lectures).
- Imperial College London (Coursera) — [Mathematics for Machine Learning: Linear Algebra](https://www.coursera.org/learn/linear-algebra-machine-learning), and DeepLearning.AI (Coursera) — [Linear Algebra for Machine Learning and Data Science](https://www.coursera.org/learn/machine-learning-linear-algebra) (free to audit, paid certificates; optional here; the linear algebra courses of the two mathematics-for-machine-learning specializations that Chapter 5 describes and uses for consolidation).
- fast.ai — [Computational Linear Algebra for Coders](https://github.com/fastai/numerical-linear-algebra) by Rachel Thomas (2017; free; optional; Jupyter notebooks + a [lecture video playlist](https://www.youtube.com/playlist?list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY); the SVD, PCA, non-negative matrix factorization, and QR implemented in NumPy on applied problems).

### Books

- Book: Gilbert Strang, *Introduction to Linear Algebra* (6th ed., Wellesley-Cambridge Press, 2023) — [official page](https://math.mit.edu/~gs/linearalgebra/ila6/indexila6.html) (paid; sample sections and selected solutions free; start here: the textbook of 18.06; chapters 1–7 cover every topic of this chapter, through the SVD).
- Book: Stephen Boyd and Lieven Vandenberghe, *Introduction to Applied Linear Algebra: Vectors, Matrices, and Least Squares* (Cambridge University Press, 2018) — [official page](https://web.stanford.edu/~boyd/vmls/) (free PDF, with Julia and Python companions, slides, and lecture videos from Stanford ENGR108; an applied, data-oriented introduction to norms, clustering, and least squares; does not cover eigenvalues or the SVD).
- Book: Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong, *Mathematics for Machine Learning* (Cambridge University Press, 2020) — [official page](https://mml-book.github.io) (free PDF; chapters 2, "Linear Algebra", 3, "Analytic Geometry", and 4, "Matrix Decompositions", then chapter 10 on PCA, the subject of the milestone).
- Book: David C. Lay, Steven R. Lay, and Judi J. McDonald, *Linear Algebra and Its Applications* (6th ed., Pearson, 2020) — [publisher page](https://www.pearson.com/en-us/subject-catalog/p/linear-algebra-and-its-applications/P200000006235/9780136880929) (paid; optional; a widely assigned first-course textbook in US universities; computational, with many applications).
- Book: Jim Hefferon, *Linear Algebra* (4th ed., 2020) — [official page](https://hefferon.net/linearalgebra/) (free PDF, with a free book of worked answers to all exercises; optional; a proof-oriented first course that prepares for Axler).
- Book: Sheldon Axler, *Linear Algebra Done Right* (4th ed., Springer, 2024) — [official page](https://linear.axler.net) (free PDF, open access under CC BY-NC; advanced; a standard second course, which develops the theory through linear maps and introduces determinants only near the end).
- Book: Gilbert Strang, *Linear Algebra and Learning from Data* (Wellesley-Cambridge Press, 2019) — [official page](https://math.mit.edu/~gs/learningfromdata/) (paid; advanced; the textbook of 18.065: the SVD, low-rank approximation, optimization, and neural networks).

### Lectures, papers and articles

- Lecture series: Grant Sanderson (3Blue1Brown), [Essence of Linear Algebra](https://www.3blue1brown.com/lessons/eola-preview) (2016; free; 16 lessons; start here: matrices as transformations of space, determinants as scaling factors of area, and eigenvectors, all shown geometrically; watch before or alongside 18.06SC).

**Milestone.** Implement matrix multiplication and the transpose with plain Python loops and verify them against the NumPy operators `@` and `.T`. Compute the dominant eigenvector of a small symmetric matrix by power iteration and verify it against `numpy.linalg.eigh`. Then perform PCA on a small dataset such as Iris in two ways — from the eigenvectors of the covariance matrix and from the SVD of the centered data — and show that both give the same two-dimensional projection up to sign.

**Time.** 8–12 weeks.

## Chapter 4 — Probability and Statistics

**Goals.** This chapter covers probability theory, the elements of statistical inference, and basic information theory. A large language model (LLM) is a probabilistic model: it outputs a probability distribution over the next token (a unit of text), the probability of a whole text follows from the chain rule of probability, and text is generated by sampling from these distributions. Training an LLM is maximum likelihood estimation (MLE), and the loss function that is minimized is the cross-entropy between the training data and the model. The Kullback–Leibler (KL) divergence measures the difference between two distributions and is used as a penalty term by the alignment methods of Chapter 18. Confidence intervals and hypothesis tests determine whether the difference between two evaluation scores in Chapter 20 exceeds sampling noise.

**Topics.** Sample spaces, events, counting, and the axioms of probability; conditional probability, independence, and Bayes' rule; random variables and discrete distributions (Bernoulli, binomial, categorical, Poisson); continuous distributions (uniform, exponential, normal); expectation, variance, and covariance; joint, marginal, and conditional distributions, and the chain rule of probability; the law of large numbers and the central limit theorem; sampling and Monte Carlo estimation; Markov chains, at the level needed for the n-gram language models of Chapter 11; maximum likelihood estimation and Bayesian updating; confidence intervals, hypothesis tests, and the bootstrap; entropy, cross-entropy, KL divergence, and perplexity.

**Resources.**

**Suggested path.** Use the Statistics 110 lectures with the textbook by Blitzstein and Hwang (chapters 1–7 and 9–11) as the main course, and work the Stat110x or textbook exercises in parallel. Then take the statistics topics (MLE, Bayesian updating, confidence intervals, hypothesis tests, and the bootstrap) from MIT 18.05, and finish with Visual Information Theory and the two 3Blue1Brown lessons on entropy and cross-entropy. When time is short, replace 18.05 with chapter 6 of *Mathematics for Machine Learning* and the last two weeks of the DeepLearning.AI course, and skip 6.041SC, CS109, and the books marked optional or advanced.

### University courses

- Harvard — [Statistics 110: Probability (lecture playlist)](https://www.youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo) by Joe Blitzstein (free; 35 videos, with handouts and practice problems on the course site; start here: a first course in probability, from counting to Markov chains, taught through worked problems).
- MIT OpenCourseWare — [6.041SC Probabilistic Systems Analysis and Applied Probability](https://ocw.mit.edu/courses/6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2013/) by John Tsitsiklis (Fall 2013; free; lecture, recitation, and tutorial videos + problem sets with solutions; an engineering-oriented alternative to Statistics 110 that ends with Bayesian and classical inference).
- Stanford — [CS109 Probability for Computer Scientists](https://web.stanford.edu/class/cs109/) (free slides + problem sets; the companion [course reader](https://chrispiech.github.io/probabilityForComputerScientists/en/) by Chris Piech covers counting through maximum likelihood, information theory, and logistic regression, with Python examples).
- MIT OpenCourseWare — [18.05 Introduction to Probability and Statistics](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/) by Jeremy Orloff and Jennifer French Kamrin (Spring 2022; free; readings + problem sets and exams with solutions, no videos; the source for the statistics topics: Bayesian updating, MLE, confidence intervals, hypothesis tests, and the bootstrap).

### Online courses (MOOCs)

- HarvardX (edX) — [Introduction to Probability (Stat110x)](https://www.edx.org/learn/probability/harvard-university-introduction-to-probability) by Joe Blitzstein (free to audit, paid certificate; start here: adds automatically graded exercises to the Statistics 110 lectures).
- DeepLearning.AI (Coursera) — [Probability & Statistics for Machine Learning & Data Science](https://www.coursera.org/learn/machine-learning-probability-and-statistics) by Luis Serrano (free to audit, paid certificate; 4 weeks, about 30 hours; course 3 of the specialization described in Chapter 5; an introductory treatment of distributions, the central limit theorem, MLE, confidence intervals, and hypothesis tests, with Python labs).
- Khan Academy — [Statistics and probability](https://www.khanacademy.org/math/statistics-probability) (free; videos + self-grading exercises; review of basic probability, descriptive statistics, and significance tests).

### Books

- Book: Joseph K. Blitzstein and Jessica Hwang, *Introduction to Probability* (2nd ed., Chapman & Hall/CRC, 2019) — [official free PDF](https://probabilitybook.net/) (free PDF, paid print edition; start here: the textbook of Statistics 110; chapters 1–7, 9, and 10 alongside the lectures, and chapter 11 for Markov chains).
- Book: Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong, *Mathematics for Machine Learning* (Cambridge University Press, 2020) — [official page](https://mml-book.github.io) (free PDF; chapter 6, "Probability and Distributions": a summary of the probability used in machine learning, including the Gaussian distribution and Bayes' rule).
- Book: Dimitri P. Bertsekas and John N. Tsitsiklis, *Introduction to Probability* (2nd ed., Athena Scientific, 2008) — [official page](http://www.athenasc.com/probbook.html) (paid; the textbook of 6.041; chapters 1–5, then 8–9 on Bayesian and classical inference).
- Book: Larry Wasserman, *All of Statistics: A Concise Course in Statistical Inference* (Springer, 2004) — [official page](https://link.springer.com/book/10.1007/978-0-387-21736-9) (paid; a compact statistics course written for computer science and machine learning students; chapters 1–6 and 8–11 cover convergence, the bootstrap, MLE, hypothesis testing, and Bayesian inference).
- Book: Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction* (MIT Press, 2022) — [official page](https://probml.github.io/pml-book/book1.html) (free PDF; chapters 2–6: probability, statistics, decision theory, and information theory, including cross-entropy and KL divergence, in machine learning notation).
- Book: David J. C. MacKay, *Information Theory, Inference, and Learning Algorithms* (Cambridge University Press, 2003) — [official page](https://www.inference.org.uk/itila/book.html) (free PDF for on-screen reading; optional; chapters 1–4 for entropy and source coding, the theory that relates prediction to data compression).
- Book: Christopher M. Bishop, *Pattern Recognition and Machine Learning* (Springer, 2006) — [official page](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) (free PDF; advanced; chapters 1 and 2 on probability theory, information theory, and probability distributions).

### Lectures, papers and articles

- Article: Christopher Olah, [Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/) (2015; start here: entropy, cross-entropy, and KL divergence explained with diagrams and optimal codes).
- Lecture series: Grant Sanderson (3Blue1Brown), Compression is Intelligence — [Reinventing Entropy](https://www.3blue1brown.com/lessons/entropy/) and [But what is Cross-Entropy?](https://www.3blue1brown.com/lessons/cross-entropy/) (2026; free; derives entropy and cross-entropy from data compression and relates them to the training loss of an LLM); related single lessons cover [Bayes' theorem](https://www.3blue1brown.com/lessons/bayes-theorem/) and the [central limit theorem](https://www.3blue1brown.com/lessons/clt/).

**Milestone.** Derive by hand the maximum likelihood estimates of the bias of a coin and of the mean and variance of a Gaussian, and verify them by simulation in NumPy. Build a character-level bigram model of a small text by counting (the normalized counts are the maximum likelihood estimate; add-one smoothing removes zero probabilities), sample text from it, and show that its cross-entropy and perplexity on held-out text are lower than those of a unigram model and of a uniform model. Verify numerically on a small example that cross-entropy equals entropy plus KL divergence.

**Time.** 6–10 weeks.

## Chapter 5 — Optimization, Numerical Methods, and Math for ML Consolidation

**Goals.** This chapter covers the optimization algorithms that train neural networks and the numerical issues that arise when they run in finite-precision arithmetic, and it consolidates the mathematics of Chapters 2–4 for machine learning (ML). The optimization part treats gradient descent and its variants — stochastic gradient descent (SGD), momentum, and the adaptive methods Adam and AdamW — together with learning-rate schedules, convexity, and conditioning. The numerical part treats floating-point arithmetic (rounding, overflow, and underflow) and the number formats in which large language models (LLMs) are trained and served: 32-bit floating point (fp32) as the baseline, and 16-bit floating point (fp16), bfloat16 (bf16), and 8-bit integers (int8) to reduce memory and computation. The consolidation part combines calculus, linear algebra, and probability in three standard problems — least squares, maximum likelihood estimation (MLE), and principal component analysis (PCA) — each formulated and solved as an optimization problem.

**Topics.** Objective functions, minima, and convex versus non-convex functions; gradient descent, step size, and convergence; conditioning and scaling of a problem; Newton's method and second-order information; stochastic and mini-batch gradient descent; momentum and adaptive methods (RMSProp, Adam, AdamW); learning-rate schedules (warmup, decay); loss surfaces of neural networks: local minima, saddle points, and plateaus; floating-point representation, rounding error, overflow, and underflow; numerically stable computation: the softmax function (exponential normalization of scores into probabilities), the log-sum-exp trick, and gradient checking with finite differences; precision formats (fp32, fp16, bf16, int8) and mixed-precision training; consolidation: least squares, maximum likelihood, and PCA as optimization problems.

**Resources.**

**Suggested path.** Start with chapter 7 of *Mathematics for Machine Learning* and the overview by Ruder, then watch lectures 21–23 and 25 of MIT 18.065 and complete week 2 of Improving Deep Neural Networks. For the numerical topics, read chapter 4 of *Deep Learning*, the first two sections of the article by Goldberg, the note on bfloat16 by Higham, and the Mixed Precision Training paper. For consolidation, run the Imperial College London specialization in parallel: complete course 3 (PCA) in full and use the quizzes of courses 1 and 2 as a test of Chapters 3 and 2; the programming labs and the milestone assume the basic Python and NumPy of Chapter 6. When time is short, skip EE364a, *Convex Optimization*, *Dive into Deep Learning*, and the DeepLearning.AI specialization.

### University courses

- MIT OpenCourseWare — [18.065 Matrix Methods in Data Analysis, Signal Processing, and Machine Learning](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/) by Gilbert Strang (Spring 2018; free; lecture videos + problem sets; start here: lectures 21–23 and 25 cover step-by-step minimization, gradient descent, momentum, and SGD, building on the linear algebra of Chapter 3; lectures 26 and 27 introduce neural networks and backpropagation).
- Stanford — [EE364a Convex Optimization I](https://www.youtube.com/playlist?list=PLoROMvodv4rMJqxxviPa4AmDClvcbHi6h) by Stephen Boyd (2023 lectures; free videos; optional; slides and exercises on the [course site](https://web.stanford.edu/class/ee364a/); the standard course on convex sets, convex functions, and duality; shows why convex problems can be solved reliably, which non-convex problems such as neural network training cannot).

### Online courses (MOOCs)

- DeepLearning.AI (Coursera) — [Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization](https://www.coursera.org/learn/deep-neural-network) by Andrew Ng (free to audit, paid certificate; 3 weeks, about 20 hours; start here: week 2 develops mini-batch gradient descent, momentum, RMSProp, Adam, and learning-rate decay step by step; course 2 of the Deep Learning Specialization listed in Chapter 9).
- Imperial College London (Coursera) — [Mathematics for Machine Learning Specialization](https://www.coursera.org/specializations/mathematics-machine-learning) by David Dye, Samuel J. Cooper, A. Freddie Page, and Marc Peter Deisenroth (free to audit per course, paid certificate, financial aid available; 3 courses of about 20 hours each: linear algebra, multivariate calculus, and PCA; the consolidation course of Part I: course 2 applies gradients to curve fitting and to training a small neural network, and course 3 derives PCA from projections and variances and implements it in NumPy).
- DeepLearning.AI (Coursera) — [Mathematics for Machine Learning and Data Science Specialization](https://www.coursera.org/specializations/mathematics-for-machine-learning-and-data-science) by Luis Serrano (free to audit per course, paid certificate, financial aid available; 3 courses, about 90 hours: linear algebra, calculus, and probability and statistics; a slower-paced alternative to the Imperial College London specialization, with Python labs; course 2 covers gradient descent and Newton's method, and course 3 is the probability course listed in Chapter 4).

### Books

- Book: Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong, *Mathematics for Machine Learning* (Cambridge University Press, 2020) — [official page](https://mml-book.github.io) (free PDF; start here: chapter 7, "Continuous Optimization": gradient descent, constrained optimization, and convexity; for consolidation, chapters 2–7 restate the mathematics of Chapters 2–4 in one notation, and chapters 8–10 apply it to model fitting, linear regression, and PCA).
- Book: Ian Goodfellow, Yoshua Bengio, and Aaron Courville, *Deep Learning* (MIT Press, 2016) — [official page](https://www.deeplearningbook.org) (free online; chapter 4, "Numerical Computation", on overflow, underflow, and conditioning, and chapter 8, "Optimization for Training Deep Models").
- Book: Aston Zhang, Zachary C. Lipton, Mu Li, and Alexander J. Smola, *Dive into Deep Learning* (Cambridge University Press, 2023) — [chapter 12, "Optimization Algorithms"](https://d2l.ai/chapter_optimization/index.html) (free online; runnable code for SGD, momentum, Adagrad, RMSProp, Adam, and learning-rate scheduling).
- Book: Stephen Boyd and Lieven Vandenberghe, *Convex Optimization* (Cambridge University Press, 2004) — [official page](https://web.stanford.edu/~boyd/cvxbook/) (free PDF; optional; the textbook of EE364a; chapters 2–5 for convexity and duality, chapter 9 for unconstrained minimization and Newton's method).

### Lectures, papers and articles

- Paper: Sebastian Ruder, [An overview of gradient descent optimization algorithms](https://arxiv.org/abs/1609.04747) (2016; start here: a survey of SGD, momentum, Nesterov, Adagrad, RMSProp, and Adam, with all update rules in one notation).
- Article: Gabriel Goh, [Why Momentum Really Works](https://distill.pub/2017/momentum/) (Distill, 2017; interactive analysis of momentum on a quadratic function, relating the convergence rate to the eigenvalues and the condition number).
- Papers: Diederik P. Kingma and Jimmy Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) (International Conference on Learning Representations (ICLR), 2015), and Ilya Loshchilov and Frank Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) (ICLR, 2019; the Adam optimizer and its variant AdamW, which separates weight decay from the gradient update and is the usual optimizer for LLM pretraining).
- Paper: David Goldberg, [What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) (ACM Computing Surveys, 1991; rounding error, the IEEE 754 standard, overflow, and underflow; the first two sections are sufficient).
- Paper: Paulius Micikevicius et al., [Mixed Precision Training](https://arxiv.org/abs/1710.03740) (ICLR, 2018; fp16 training with an fp32 master copy of the weights and loss scaling; the basis of current fp16 and bf16 training recipes).
- Article: Nicholas J. Higham, [What Is Bfloat16 Arithmetic?](https://nhigham.com/2020/06/02/what-is-bfloat16-arithmetic/) (2020; a short note that compares the range and precision of bf16, fp16, and fp32).
- Article: Rich Sutton, [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) (2019; a short essay arguing that general methods that scale with computation have outperformed methods built on hand-coded domain knowledge; background for the scaling approach of Chapter 15).

**Milestone.** Fit a logistic regression with two parameters (a logistic function applied to `w x + b`) to a small synthetic dataset in NumPy: derive the gradient of the cross-entropy loss by hand and verify it with finite differences. Implement gradient descent, momentum, and Adam from scratch, and plot or animate the three trajectories on the contour plot of the two-dimensional loss surface. Then compute a softmax in fp16 with inputs large enough to overflow, and correct it with the log-sum-exp trick.

**Time.** 4–6 weeks.

---

# Part II — Programming and Tools

Part II runs in parallel with Part I. Programming does not depend on the later mathematics chapters, and implementing the mathematics in code reinforces it. Chapter 6 can start in the first week, alongside Chapter 1; Chapter 7 follows Chapter 6.

## Chapter 6 — Python and the Scientific Computing Stack

**Goals.** This chapter covers programming in Python, the language in which nearly all large language model (LLM) tools are written, together with the standard working environment: the command line, virtual environments, Git, and Jupyter notebooks. It covers functions and classes, file input and output, and the testing and debugging of code. It then introduces NumPy, which applies mathematical operations to whole arrays at once, and Matplotlib, which plots the results; both are used to implement the mathematics of Part I. The chapter closes with an introduction to PyTorch, the library used from Chapter 7 onward, whose tensors are NumPy-style arrays that can run on a graphics processing unit (GPU) and compute derivatives automatically.

**Topics.** The command line, a code editor, and running Python scripts; Python syntax and built-in data structures (lists, tuples, dictionaries, sets); functions, modules, and classes; file I/O (text, CSV, and JSON); errors, debugging, and unit tests (tracebacks, pytest); virtual environments and packages (venv, pip, uv); version control with Git (commit, branch, merge, push); Jupyter notebooks as a working environment (locally and on Google Colab); NumPy arrays: shapes, dtypes, indexing, broadcasting, and vectorized math; plotting with Matplotlib; an introduction to PyTorch tensors and automatic differentiation (autograd).

**Resources.**

**Suggested path.** Start in the first week, alongside Chapter 1. Use CS50P as the single main course (or Python for Everybody for a slower pace) at 45–60 minutes per day next to the Part I mathematics, and read the matching chapter of *Think Python* when a lecture needs a second explanation. In parallel, watch the shell and Git lectures of the Missing Semester, read *Pro Git* chapters 1–3, and keep the problem sets in a Git repository from the first week. After the course, work through the NumPy beginners' page and the CS231n tutorial, finish with PyTorch's Learn the Basics or Raschka's one-hour tutorial, and build the milestone. When time is short, omit 6.100L, *Automate the Boring Stuff*, *Fluent Python*, and the official Python Tutorial.

### University courses

- Harvard — [CS50P: Introduction to Programming with Python](https://cs50.harvard.edu/python/) by David J. Malan (free; videos + problem sets + final project; start here: ten weeks from variables to classes, including libraries and pip, unit tests with pytest, and file I/O; assumes no programming experience).
- MIT — [The Missing Semester of Your CS Education](https://missing.csail.mit.edu/) by Anish Athalye, Jon Gjengset, and Jose Javier Gonzalez Ortiz (free; 2026 edition, videos + notes + exercises; the lectures on the shell, the command-line environment, debugging, Git, and packaging cover tools that introductory Python courses omit).
- MIT OpenCourseWare — [6.100L Introduction to CS and Programming Using Python (Fall 2022)](https://ocw.mit.edu/courses/6-100l-introduction-to-cs-and-programming-using-python-fall-2022/) by Ana Bell (free; videos + notes + problem sets; optional: an alternative first course with more computer science content: algorithms, recursion, and complexity).

### Online courses (MOOCs)

- University of Michigan (Coursera) — [Python for Everybody Specialization](https://www.coursera.org/specializations/python) by Charles Severance (free to audit, paid certificate; 5 courses; start here as the alternative to CS50P: a slower pace, with less on classes and nothing on testing; the same lectures, book, and autograded exercises are free at [py4e.com](https://www.py4e.com)).

### Books

- Book: Allen B. Downey, *Think Python* (3rd ed., 2024) — [official page](https://allendowney.github.io/ThinkPython/) (free online as Jupyter notebooks that run on Colab; start here: the companion text to the main course; each chapter is a notebook, so reading it also gives practice with Jupyter).
- Book: Al Sweigart, *Automate the Boring Stuff with Python* (3rd ed., 2025) — [official page](https://automatetheboringstuff.com) (free online; optional: the file, CSV, and JSON chapters add practice with file I/O on realistic tasks).
- Book: Jake VanderPlas, *Python Data Science Handbook* (1st ed. free online; 2nd ed., 2022, in print) — [official page](https://jakevdp.github.io/PythonDataScienceHandbook/) (free; chapter 1 on IPython and Jupyter, chapter 2, "Introduction to NumPy", and chapter 4, "Visualization with Matplotlib").
- Book: Scott Chacon and Ben Straub, *Pro Git* (2nd ed., 2014) — [official page](https://git-scm.com/book/en/v2) (free; chapters 1–3 cover all the Git this course requires: commits, branches, merges, and remotes).
- Book: Luciano Ramalho, *Fluent Python* (2nd ed., 2022) — [official page](https://www.fluentpython.com/) (paid; advanced: Part I, "Data Structures", explains how idiomatic Python uses the data model, sequences, and dictionaries; intended for a second pass after the main course).

### Lectures, papers and articles

- Tutorial: NumPy developers, [NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html) (current docs; start here for NumPy: arrays, shapes, indexing, broadcasting, and aggregation on one page).
- Tutorial: Justin Johnson (Stanford CS231n), [Python Numpy Tutorial (with Jupyter and Colab)](https://cs231n.github.io/python-numpy-tutorial/) (a compact Python, NumPy, and Matplotlib refresher written for deep-learning students; usable as a self-test after the main course).
- Tutorial: PyTorch, [Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) (official; runnable in Colab; the Tensors and Autograd pages belong to this chapter, and the Autograd page requires derivatives from Chapter 2; the model-building pages belong to Chapter 9).
- Article: Sebastian Raschka, [PyTorch in One Hour: From Tensors to Training Neural Networks on Multiple GPUs](https://sebastianraschka.com/teaching/pytorch-1h/) (2025; covers the same ground as Appendix A of his *Build a Large Language Model (From Scratch)*; sections 1–4 on tensors and autograd belong to this chapter, the training-loop and GPU sections to Chapters 7 and 9).
- Tutorial: Python Software Foundation, [The Python Tutorial](https://docs.python.org/3/tutorial/) (current Python 3 docs; optional: the official tour of the language, written for readers who can already program; suited to a second pass).

### Tools and hands-on

- Docs: Astral, [uv: First steps](https://docs.astral.sh/uv/getting-started/first-steps/) (free; uv is a package and project manager that replaces venv and pip; together with the Projects guide on the same site, the page shows how to create a project, add NumPy and pytest, and run tests in an isolated environment).

**Milestone.** In a dedicated Git repository and virtual environment, build a small NumPy library with loop-free functions for the dot product, the matrix–vector product, mean squared error, and a softmax that operates on each row of a 2-D array and does not overflow on large inputs (subtract the row maximum before exponentiating). Write pytest unit tests that pass, including one test per function that verifies the result against the PyTorch equivalent with `np.allclose`.

**Time.** 8–12 weeks, overlapping with math.

## Chapter 7 — GPUs, Hardware, and the Training Environment

**Goals.** This chapter describes what a graphics processing unit (GPU) is, why it multiplies matrices much faster than a central processing unit (CPU), and how the capacity and bandwidth of its memory (VRAM) limit the models that can be run on it. It explains how the number formats float32 (fp32), float16 (fp16), and bfloat16 (bf16) determine the memory and speed cost of a model, and how to estimate that cost as the number of parameters times the bytes per parameter. It also covers setting up a GPU environment — Google Colab or Kaggle at no cost, a rented cloud GPU, or a local machine — and measuring memory use and speed in it. Neural networks are not introduced until Chapter 9, so the training topics (mixed precision, gradients, optimizer states) are presented here as an introduction and are treated in full in Chapters 9 and 16.

**Topics.** CPU vs GPU: a few fast cores vs thousands of simple ones, and why matrix multiplication suits a GPU; the software stack: NVIDIA drivers, the CUDA platform, and PyTorch devices (CPU, CUDA, Apple MPS); VRAM, memory bandwidth, and the cost of moving data between CPU and GPU; floating-point operations (FLOPs), FLOP/s, and throughput in tokens per second; compute-bound, memory-bound, and overhead-bound workloads; number formats and bytes per parameter: fp32, fp16, bf16, and 8-bit integers (int8); measuring GPU work: `nvidia-smi`, PyTorch memory counters, and timing asynchronous code; CPU vs GPU for running a model: the cases in which a laptop is sufficient; choosing an environment: Google Colab and Kaggle notebooks, a rented cloud GPU, or a local machine; an introduction to training memory: weights, gradients, optimizer states, and activations; an introduction to mixed precision training (autocast and loss scaling).

**Resources.**

**Suggested path.** Start this chapter after Chapter 6, still in parallel with Part I; it requires only arithmetic and the PyTorch tensors of Chapter 6. Watch the Branch Education video, read Horace He's article, then open Colab and reproduce the main points — move tensors to the GPU, change their dtype, and time the operations — with the PyTorch CUDA notes as reference. Read Dettmers and the "AI Accelerators" section of *AI Engineering* for the hardware vocabulary, and the bytes-per-parameter sections of Transformer Math 101 for the memory arithmetic. Items marked advanced assume neural networks or C++: omit them when time is short, and return to them from Chapters 9, 15, and 16 as indicated in each note.

### University courses

- Stanford — [CS149: Parallel Computing (Fall 2025)](https://gfxcourses.stanford.edu/cs149/fall25) by Kayvon Fatahalian and Kunle Olukotun (free slides + assignments; the [2023 lecture videos](https://www.youtube.com/playlist?list=PLoROMvodv4rMp7MTFr4hQsDEcX7Bx6Odp) are on YouTube; start here: lecture 7 only, "GPU Architecture and CUDA Programming"; advanced: the rest of the course is systems programming in C++ and fits after Chapter 16, as preparation for writing GPU code).
- Stanford — [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) by Percy Liang and Tatsunori Hashimoto (free; Spring 2026 lecture videos on a [YouTube playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV) and public assignments; advanced: lecture 2 on resource accounting counts the memory and FLOPs of training, and lecture 5 covers GPUs for large language model (LLM) work; both assume the transformer architecture, so they fit in Chapter 15).

### Online courses (MOOCs)

- GPU MODE — [lecture series on YouTube](https://www.youtube.com/@GPUMODE) with [slides and code on GitHub](https://github.com/gpu-mode/lectures) (free; 100+ community talks on GPU programming; advanced: lecture 3, "Getting Started With CUDA" by Jeremy Howard, is aimed at Python programmers and needs no further background; the other lectures fit after Chapter 16).

### Books

- Book: Chip Huyen, *AI Engineering* (O'Reilly, 2025) — [book repo](https://github.com/chiphuyen/aie-book) (paid book, free chapter summaries; start here: within chapter 9, "Inference Optimization", only the "AI Accelerators" section, which defines an accelerator and describes it by memory size, memory bandwidth, and FLOP/s; the rest of that chapter belongs to Chapter 21).
- Book: Stas Bekman, *Machine Learning Engineering Open Book* (continuously updated) — [official page](https://github.com/stas00/ml-engineering) (free; a practitioner's reference; "Compute" under Hardware covers accelerator specifications, and "Tensor precision / Data types" under Training covers fp32, fp16, and bf16; the other Training chapters belong to Chapter 16).
- Book: Nouamane Tazi, Ferdinand Mom et al. (Hugging Face), *The Ultra-Scale Playbook: Training LLMs on GPU Clusters* (2025) — [free online edition](https://huggingface.co/spaces/nanotron/ultrascale-playbook) (free online; advanced: the opening part on training with one GPU shows how memory is divided among weights, gradients, optimizer states, and activations; the whole book is read in Chapter 16).
- Book: Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj, *Programming Massively Parallel Processors* (4th ed., 2022) — [official page](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0) (paid; advanced; optional: the standard CUDA textbook, abbreviated *PMPP*; chapters 1–6 run through memory architecture and performance; requires C, so it fits after Chapter 16).

### Lectures, papers and articles

- Video: Branch Education, [How do Graphics Cards Work? Exploring GPU Architecture](https://www.youtube.com/watch?v=h9Z4oGN89MU) (2024; 28 minutes; start here: an animated description of a graphics card — its cores, its VRAM, and why thousands of simple cores suit matrix arithmetic; assumes no background).
- Article: Horace He, [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) (2022; identifies compute, memory bandwidth, and overhead as the three bottlenecks of GPU work, without CUDA code; the neural-network layer names it mentions are defined in Chapter 9, and a second reading fits in Chapter 16).
- Article: Tim Dettmers, [Which GPU(s) to Get for Deep Learning](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/) (2023; which specifications matter, from Tensor Cores to memory bandwidth and VRAM, and when renting costs less than buying; the GPU models discussed are dated, the selection criteria still apply).
- Article: Quentin Anthony, Stella Biderman, and Hailey Schoelkopf, [Transformer Math 101](https://blog.eleuther.ai/transformer-math/) (EleutherAI, 2023; the sections on bytes per parameter in fp32, fp16/bf16, and int8 belong to this chapter; the C = 6PD compute rule and the training-memory sections belong to Chapter 15).
- Docs: PyTorch, [CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html) (current docs; devices, memory management, and asynchronous execution, including why unsynchronized GPU timings are wrong and how to correct them), then [Automatic Mixed Precision package - torch.amp](https://docs.pytorch.org/docs/stable/amp.html) (how `autocast` and `GradScaler` work; a preview of Chapter 9).
- Paper: Paulius Micikevicius et al., [Mixed Precision Training](https://arxiv.org/abs/1710.03740) (2017, ICLR 2018; advanced: fp16 training with an fp32 master copy of the weights and loss scaling; the abstract suffices here, the full paper fits in Chapter 16).
- Article: Mark Harris, [An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/) (NVIDIA Technical Blog, updated 2025; optional: a first CUDA kernel in C++ — threads, blocks, and unified memory — showing the layer that PyTorch abstracts away).

### Tools and hands-on

- Hands-on: Google, [Colab](https://colab.research.google.com) (free tier; start here: open a notebook, switch the runtime to a GPU, run `!nvidia-smi`, and move a PyTorch tensor to `"cuda"`; the [Colab FAQ](https://research.google.com/colaboratory/faq.html) explains the usage limits).
- Hands-on: [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks) (free GPU notebooks with a weekly quota) for work that exceeds the Colab limits; beyond that, GPUs can be rented from [Lambda](https://lambda.ai/), [RunPod](https://www.runpod.io/), or [Vast.ai](https://vast.ai/) (paid; billed by the hour whether or not the GPU is in use, so shut the instance down after each session).

**Milestone.** In Colab or on any GPU, write a short PyTorch script that creates a 4096 × 4096 matrix in fp32, fp16, and bf16 and verifies that the memory footprint of each tensor (`element_size() * nelement()`) matches a hand calculation. Measure the time of an fp32 matrix multiplication on the CPU and on the GPU — with a warm-up run and `torch.cuda.synchronize()` before each clock reading — then repeat on the GPU in fp16 and report both speedups. Finally, compute by hand how many gigabytes the weights of a 7-billion-parameter model occupy in fp32 and in bf16, and state whether they fit in the VRAM of the GPU used.

**Time.** 2–4 weeks.

---

# Part III — Machine Learning and Deep Learning Fundamentals

## Chapter 8 — Machine Learning Foundations

**Goals.** This chapter introduces machine learning: the parameters of a model are fitted to data by minimizing a loss function (training), and the fitted model is then used to make predictions on new data (inference). The chapter covers linear regression, logistic and softmax regression, decision trees and their ensembles, and two unsupervised methods, k-means clustering and principal component analysis (PCA), all implemented with scikit-learn. It also covers the methodology that applies to every model: training, validation, and test splits, overfitting and regularization, cross-validation, and evaluation metrics. The same concepts apply to large language models (LLMs): a language model is a classifier that predicts the next unit of text (a token) from a fixed vocabulary, it is trained with the cross-entropy loss of Chapter 4, and it is evaluated on held-out data.

**Topics.** Types of learning (supervised, unsupervised, and reinforcement learning); models, parameters, loss functions, and the difference between training and inference; linear regression, fitted by least squares and by gradient descent; logistic regression and softmax regression for binary and multiclass classification; feature preparation (scaling, one-hot encoding) and scikit-learn pipelines; training, validation, and test splits, overfitting, underfitting, and generalization; regularization (L1 and L2 penalties); the bias-variance trade-off, cross-validation, and hyperparameter selection; decision trees and ensembles (random forests, gradient boosting); unsupervised learning (k-means clustering and PCA); evaluation metrics (accuracy, precision, recall, F1 score, confusion matrix, ROC curve), baselines, and class imbalance.

**Resources.**

**Suggested path.** Take courses 1 and 2 of the Machine Learning Specialization as the main course, and read *An Introduction to Statistical Learning* in parallel: chapters 2–6 first, then chapters 8 and 12. Use the Kaggle Learn course or the Géron notebooks for scikit-learn practice, read the article by Domingos after the first weeks of the course, and consult the scikit-learn metrics guide while building the milestone. Readers who are comfortable with the mathematics of Part I can replace the Specialization with the CS229 lectures and notes. When time is short, skip course 3 of the Specialization, Learning From Data, and the books marked advanced.

### University courses

- Stanford — [CS229: Machine Learning (Autumn 2018 lectures)](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU) by Andrew Ng (free videos; start here: 20 lectures with full derivations of linear and logistic regression, generalized linear models, regularization, k-means, and PCA; the current [CS229 lecture notes](https://cs229.stanford.edu/main_notes.pdf) by Andrew Ng and Tengyu Ma are a free PDF, of which Part I, chapters 1–2, and Part III, on generalization and regularization, apply here).
- MIT — [6.390 Introduction to Machine Learning lecture notes](https://introml.mit.edu/notes/) (free; notes on regression, gradient descent, classification, and feature representation; the earlier [6.036 (Fall 2020) on OCW](https://ocw.mit.edu/courses/6-036-introduction-to-machine-learning-fall-2020/) by Leslie Kaelbling, Tomás Lozano-Pérez, Isaac Chuang, and Duane Boning adds videos and exercises).
- Caltech — [Learning From Data](https://work.caltech.edu/telecourse) by Yaser Abu-Mostafa (free; optional; 18 recorded lectures, 8 homework sets, and a final; covers the theory of generalization: VC dimension, bias-variance, overfitting, and validation).

### Online courses (MOOCs)

- DeepLearning.AI and Stanford Online (Coursera) — [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) by Andrew Ng (free to audit, paid certificate; 3 courses, about 2 months at 10 h/week; start here: Python-based and less mathematical than CS229; course 1 covers linear and logistic regression, course 2 neural networks, decision trees, and model evaluation, and course 3 clustering, recommender systems, and reinforcement learning).
- Google — [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course) (free; 12 modules with interactive visualizations and exercises; linear and logistic regression, classification metrics, data preparation, and overfitting).
- Kaggle Learn — [Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning) (free; about 3 hours; decision trees, validation, underfitting and overfitting, and random forests in hands-on notebooks).

### Books

- Book: Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, and Jonathan Taylor, *An Introduction to Statistical Learning with Applications in Python* (Springer, 2023; the R edition is the 2nd ed., 2021) — [official page](https://www.statlearning.com) (free PDF; start here: chapters 2–6 on statistical learning, linear regression, classification, resampling methods, and regularization, then chapter 8 on tree-based methods and chapter 12 on unsupervised learning).
- Book: Aurélien Géron, *Hands-On Machine Learning with Scikit-Learn and PyTorch* (O'Reilly, 1st ed., 2025) — [official notebooks](https://github.com/ageron/handson-mlp) (paid book, free notebooks; chapters 1–8: an end-to-end scikit-learn project, classification metrics, training linear models, decision trees, ensembles, dimensionality reduction, and clustering; chapters 9–11 are used in Chapter 9).
- Book: Andriy Burkov, *The Hundred-Page Machine Learning Book* (2019) — [official page](https://themlbook.com/) (free to read online under the author's "read first, buy later" policy; a short overview of the whole field, usable as a preview before a course or as a review after it).
- Book: Simon J.D. Prince, *Understanding Deep Learning* (MIT Press, 2023) — [official page](https://udlbook.github.io/udlbook/) (free PDF; chapter 2, "Supervised learning", chapter 5, "Loss functions", and chapter 8, "Measuring performance"; the same book is the main text of Chapter 9).
- Book: Trevor Hastie, Robert Tibshirani, and Jerome Friedman, *The Elements of Statistical Learning* (Springer, 2nd ed., 2009) — [official page](https://hastie.su.domains/ElemStatLearn/) (advanced; free PDF; chapters 2–4 and chapter 7, "Model Assessment and Selection"; the graduate-level counterpart of *An Introduction to Statistical Learning*).

### Lectures, papers and articles

- Article: Pedro Domingos, [A Few Useful Things to Know About Machine Learning](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf) (Communications of the ACM, 2012; start here: twelve practical lessons on generalization, overfitting, feature engineering, and the curse of dimensionality; assumes the vocabulary of the first weeks of a course).
- Docs: scikit-learn, [Metrics and scoring: quantifying the quality of predictions](https://scikit-learn.org/stable/modules/model_evaluation.html) (current docs; reference for accuracy, precision, recall, F1 score, confusion matrices, ROC curves, and dummy baselines, as required by the milestone).

**Milestone.** Train a logistic regression classifier on scikit-learn's digits dataset with a stratified 80/20 train/test split, select the regularization strength by 5-fold cross-validation on the training set only, and report the test accuracy (at least 95%), the confusion matrix, and the per-class precision and recall next to a `DummyClassifier` baseline. Then relabel the task as "digit 9 versus all other digits", and explain in one paragraph, using the measured accuracy, precision, and recall of the majority-class baseline, why accuracy alone is misleading on imbalanced data.

**Time.** 6–8 weeks.

## Chapter 9 — Deep Learning: Concepts and Practice

**Goals.** This chapter covers neural networks: functions composed of layers, each a weighted sum followed by a nonlinear activation function, whose parameters are fitted jointly by gradient descent on a loss function. It treats the multilayer perceptron (MLP), the output layers and loss functions for regression and classification, backpropagation as the algorithm that computes the gradients, and the optimizers of Chapter 5: stochastic gradient descent (SGD), Adam, and AdamW. It then covers the methods that keep the training of deep networks stable and limit overfitting — weight initialization, normalization, residual connections, dropout, weight decay, and early stopping — and a procedure for diagnosing a training run from its loss curves. All models are implemented in PyTorch, including the training loop, execution on a graphics processing unit (GPU), and the mixed precision training introduced in Chapter 7. Except for attention, every component of the Transformer of Chapter 12 (linear layers, Gaussian error linear unit (GELU) activations, layer normalization, residual connections, dropout, and AdamW) is introduced here; backpropagation is implemented by hand in Chapter 10.

**Topics.** The artificial neuron, the perceptron, and the multilayer perceptron (MLP): layers, weights, and biases; activation functions — sigmoid, tanh, the rectified linear unit (ReLU), and GELU — and the universal approximation theorem; output layers and loss functions: mean squared error for regression, softmax with cross-entropy for classification; backpropagation: the chain rule applied layer by layer to compute the gradient of the loss (implemented by hand in Chapter 10); PyTorch model building: `nn.Module`, `Dataset` and `DataLoader`, and automatic differentiation (autograd); the training loop: mini-batches, epochs, forward pass, backward pass, optimizer step, and evaluation on a validation set; optimizers (SGD with momentum, Adam, AdamW) and learning-rate schedules (warmup, decay); weight initialization (Xavier and He) and normalization (batch normalization, layer normalization); residual connections and the training of deep networks; regularization (weight decay, dropout, early stopping, data augmentation); training on a GPU and mixed precision training (`torch.autocast`, loss scaling); hyperparameter tuning and the diagnosis of training runs from training and validation loss curves; convolutional networks in overview (recurrent networks follow in Chapter 11).

**Resources.**

**Suggested path.** Watch lecture 1 of MIT 6.S191 and the first two videos of the 3Blue1Brown series listed in Chapter 10 as an overview, then read chapters 3–9 of *Understanding Deep Learning* as the main text and work through its notebooks. In parallel, take courses 1 and 2 of the Deep Learning Specialization, which continue the Machine Learning Specialization of Chapter 8, and practice PyTorch with units 3, 4, 6, and 9 of Deep Learning Fundamentals or chapters 3–6 of *Dive into Deep Learning*. Before building the milestone, read A Recipe for Training Neural Networks and the CS231n note "Learning and Evaluation". When time is short, skip the fast.ai course, the books by Bishop and by Goodfellow et al., the original papers, and courses 3–5 of the Specialization (project strategy, convolutional networks, and sequence models).

### University courses

- MIT — [6.S191: Introduction to Deep Learning](https://introtodeeplearning.com/) by Alexander Amini and Ava Amini (free; slides, videos, and three software labs, renewed every year; start here: lecture 1 of the 2026 edition, about one hour, surveys this chapter — the perceptron, MLPs, loss functions, backpropagation, optimizers, and regularization; the later lectures cover sequence models, computer vision, and generative models, and lab 3 fine-tunes a large language model (LLM); the [2020 edition](https://ocw.mit.edu/courses/6-s191-introduction-to-deep-learning-january-iap-2020/) is archived on MIT OpenCourseWare).
- Stanford — [CS231n: Deep Learning for Computer Vision](https://cs231n.stanford.edu/) by Fei-Fei Li, Ehsan Adeli, Justin Johnson, and Zane Durante (free [course notes](https://cs231n.github.io/) and [2025 lecture videos](https://www.youtube.com/playlist?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16); the three "Neural Networks" notes cover activation functions, data preprocessing, weight initialization, regularization, gradient checks, and the monitoring of a training run; the third note, "Learning and Evaluation", is the reference for the milestone; the notes on convolutional networks cover the overview topic of this chapter; assignment 1 is used in Chapter 10).

### Online courses (MOOCs)

- DeepLearning.AI (Coursera) — [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) by Andrew Ng, Younes Bensouda Mourri, and Kian Katanforoosh (free to audit, paid certificate; 5 courses, about 3 months at 10 h/week; start here: course 1, "Neural Networks and Deep Learning", develops the MLP and its training step by step, and course 2, "Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization", covers initialization, regularization, optimizers, batch normalization, and hyperparameter tuning; week 2 of course 2 is used in Chapter 5; the programming exercises use NumPy and TensorFlow, so PyTorch practice comes from the other resources).
- Lightning AI — [Deep Learning Fundamentals](https://lightning.ai/pages/courses/deep-learning-fundamentals/) by Sebastian Raschka (free; 10 units of short videos with PyTorch code and exercises; units 3 and 4 train logistic regression and MLPs in PyTorch with data loaders; unit 6 covers early stopping, learning-rate schedulers, activation functions, batch normalization, dropout, and the debugging of networks; unit 9 covers mixed precision training; unit 5 introduces the Lightning library, which the later units use and this book does not).
- fast.ai — [Practical Deep Learning for Coders](https://course.fast.ai) by Jeremy Howard (free; optional; Part 1, 2022: 9 lessons of about 90 minutes; an alternative route that trains complete models with the fastai library from the first lesson and introduces the theory afterwards; the companion book, Jeremy Howard and Sylvain Gugger, *Deep Learning for Coders with fastai and PyTorch* (O'Reilly, 2020), is free as [Jupyter notebooks](https://github.com/fastai/fastbook)).

### Books

- Book: Simon J.D. Prince, *Understanding Deep Learning* (MIT Press, 2023) — [official page](https://udlbook.github.io/udlbook/) (free PDF; start here: chapters 3–9 cover shallow and deep networks, loss functions, fitting models, gradients and initialization, measuring performance, and regularization, with Python notebooks for each chapter; chapter 10 covers convolutional networks, and chapter 11 residual networks and batch normalization).
- Book: Aston Zhang, Zachary C. Lipton, Mu Li, and Alexander J. Smola, *Dive into Deep Learning* (Cambridge University Press, 2023) — [official page](https://d2l.ai/) (free online; chapters 3–6 implement linear regression, softmax regression, MLPs, and custom PyTorch modules, first from scratch and then with the library; sections 8.5 and 8.6 cover batch normalization and residual networks; chapter 12, on optimization algorithms, is listed in Chapter 5; every example runs in PyTorch).
- Book: Aurélien Géron, *Hands-On Machine Learning with Scikit-Learn and PyTorch* (O'Reilly, 1st ed., 2025) — [official notebooks](https://github.com/ageron/handson-mlp) (paid book, free notebooks; the continuation of the chapters used in Chapter 8: chapter 9 introduces artificial neural networks, chapter 10 builds them in PyTorch, and chapter 11 covers initialization, normalization, optimizers, learning-rate schedules, and regularization; appendix B covers mixed precision).
- Book: Christopher M. Bishop and Hugh Bishop, *Deep Learning: Foundations and Concepts* (Springer, 2024) — [official page](https://www.bishopbook.com/) (free to read online; optional; chapters 6–9: deep neural networks, gradient descent, backpropagation, and regularization, with more probability theory than the other books listed).
- Book: Ian Goodfellow, Yoshua Bengio, and Aaron Courville, *Deep Learning* (MIT Press, 2016) — [official page](https://www.deeplearningbook.org) (free online; advanced; chapters 6–8 on feedforward networks, regularization, and optimization, and chapter 11, "Practical Methodology"; chapters 4 and 8 are listed in Chapter 5).

### Lectures, papers and articles

- Article: Andrej Karpathy, [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) (2019; start here: a procedure for building a model in stages — inspect the data, establish a baseline, overfit a single batch, then regularize and tune — designed to expose errors that do not raise exceptions; the milestone follows this order).
- Article: Varun Godbole, George E. Dahl, Justin Gilmer, Christopher J. Shallue, and Zachary Nado, [Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook) (Google Research, 2023; free; a procedure for hyperparameter tuning: which hyperparameters to tune first, how to choose the batch size, and how to read training curves).
- Papers: Diederik P. Kingma and Jimmy Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) (International Conference on Learning Representations (ICLR), 2015), and Ilya Loshchilov and Frank Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) (ICLR, 2019; also listed in Chapter 5; Adam and its variant AdamW, the default optimizers from this chapter onward).
- Paper: Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov, [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html) (Journal of Machine Learning Research, 2014; defines dropout and interprets it as the averaging of an ensemble of thinned networks).
- Papers: Sergey Ioffe and Christian Szegedy, [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167) (International Conference on Machine Learning, 2015); Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton, [Layer Normalization](https://arxiv.org/abs/1607.06450) (2016); and Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (Conference on Computer Vision and Pattern Recognition, 2016; optional; the original descriptions of batch normalization, of layer normalization, and of residual connections; the last two are components of every Transformer block in Chapter 12).
- Docs: PyTorch, [Automatic Mixed Precision package - torch.amp](https://docs.pytorch.org/docs/stable/amp.html) (current docs; `autocast` and `GradScaler`, introduced in Chapter 7 and applied here to a training loop; the method is defined in Paulius Micikevicius et al., [Mixed Precision Training](https://arxiv.org/abs/1710.03740) (ICLR, 2018), listed in Chapter 5).

**Milestone.** Train an MLP in PyTorch to classify the sentiment of the [IMDb movie review dataset](https://huggingface.co/datasets/stanfordnlp/imdb) (25,000 training and 25,000 test reviews), using as input bag-of-words counts of the 10,000 most frequent words computed with scikit-learn's `CountVectorizer`, and reach a test accuracy of at least 85%, reported next to a logistic regression baseline on the same features. Follow the order of A Recipe for Training Neural Networks: verify first that the model can overfit a single batch, then train on the full training set with a held-out validation set and plot the training and validation loss curves. Produce one run that overfits and one in which dropout, weight decay, or early stopping reduces the gap between the two curves, and state which change had the largest effect.

**Time.** 8–12 weeks.

## Chapter 10 — Neural Networks from Scratch: Backprop and the Gradient Flow

**Goals.** This chapter implements from first principles the training machinery that Chapter 9 used through PyTorch. It covers computational graphs, reverse-mode automatic differentiation, and backpropagation as the chain rule applied to a graph, first for scalar operations and then for the matrix operations of a network layer. The implementation is a small automatic differentiation (autograd) engine modeled on Karpathy's micrograd, verified against finite differences and against PyTorch, and used to train a multilayer perceptron (MLP); a character-level language model is then trained with a hand-written backward pass. The chapter also covers how the magnitude of the gradients changes with the depth of a network — vanishing and exploding gradients — and how initialization, normalization, residual connections, and gradient clipping control it, using measured statistics of activations and gradients. The character-level model is the starting point of the Transformer language model built in Chapters 12 and 14, and the gradient diagnostics are used to debug training runs in the later chapters.

**Topics.** Computational graphs: an expression as a directed acyclic graph of elementary operations; the chain rule on a graph: forward-mode and reverse-mode automatic differentiation; a scalar autograd engine: value nodes, local derivatives, topological ordering, and gradient accumulation; gradient checking with finite differences; backpropagation in vector and matrix form: linear layers, activation functions, and softmax with cross-entropy; a neural network library on top of the engine: neurons, layers, an MLP, and parameter updates by stochastic gradient descent (SGD); a character-level language model: bigram counts, then an MLP over learned character vectors (embeddings); a hand-written backward pass for the MLP, including batch normalization, verified against PyTorch autograd; vanishing and exploding gradients: saturating activation functions, depth, and the scale of the initial weights; diagnostics: statistics of activations and gradients, and the ratio of update size to parameter size; remedies: Xavier and He initialization, normalization, residual connections, and gradient clipping; PyTorch autograd as the same design: dynamic graphs and vector-Jacobian products.

**Resources.**

**Suggested path.** Watch chapters 3 and 4 of the 3Blue1Brown series and read the article by Olah, then follow lecture 1 of Neural Networks: Zero to Hero while typing the code, which produces the autograd engine of the milestone. Continue with lectures 2–5 for the character-level language model, the gradient diagnostics, and the hand-written backward pass; in parallel, read chapter 2 of Nielsen or the CS231n backpropagation note for the derivation, and the handout by Johnson before lecture 5. Read chapter 5 of Nielsen and the paper by Glorot and Bengio for vanishing gradients, and finish with the PyTorch autograd tutorial. When time is short, skip lecture 6, CSC321, MiniTorch, Deep Learning Systems, and the items marked advanced.

### University courses

- Stanford — [CS231n: Deep Learning for Computer Vision](https://cs231n.stanford.edu/) by Fei-Fei Li, Ehsan Adeli, Justin Johnson, and Zane Durante (free; start here: the course note [Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/) covers local gradients, gates, staged computation, and patterns in the backward flow; in [assignment 1 (2025)](https://cs231n.github.io/assignments2025/assignment1/), questions 3 and 5 implement the forward and backward passes of fully connected networks in NumPy, with gradient checks provided).
- University of Toronto — [CSC321: Intro to Neural Networks and Machine Learning (Winter 2018)](https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/) by Roger Grosse (free slides and lecture notes; lecture 6 derives backpropagation on computational graphs, and lecture 10 describes the implementation of an automatic differentiation system).
- Cornell Tech — [MiniTorch](https://minitorch.github.io/) by Sasha Rush (free; optional; a teaching library in which the Torch API is reimplemented module by module — scalar automatic differentiation, tensors, efficient operations, and networks — with unit tests that check each step; a longer second project after the milestone).
- CMU — [10-714: Deep Learning Systems](https://dlsyscourse.org/) by Tianqi Chen and Tim Dettmers (advanced; free slides, notebooks, and assignments, with recordings of the 2022 offering on YouTube; the assignments build "Needle", a deep learning library with automatic differentiation, layers, optimizers, and a graphics processing unit (GPU) backend; requires C++, so it fits after Chapter 16).

### Online courses (MOOCs)

- Andrej Karpathy — [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) (free; 8 lectures of 1–2.5 hours; start here: lecture 1 builds [micrograd](https://github.com/karpathy/micrograd), a scalar autograd engine of about 100 lines with a neural network library of about 50 lines; lectures 2–3 build makemore, a character-level language model, first from bigram counts and then as an MLP; lecture 4 measures activation and gradient statistics and adds batch normalization; lecture 5, "Becoming a Backprop Ninja", writes the backward pass of that MLP by hand; lecture 6 is optional; lectures 7 and 8, on a generative pre-trained transformer (GPT) and its tokenizer, are used in Chapters 12 and 13; notebooks on [GitHub](https://github.com/karpathy/nn-zero-to-hero), videos as a [YouTube playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)).

### Books

- Book: Michael Nielsen, *Neural Networks and Deep Learning* (Determination Press, 2015) — [official page](http://neuralnetworksanddeeplearning.com/) (free online; start here: chapter 2, "How the backpropagation algorithm works", derives the four equations of backpropagation for an MLP, and chapter 5, "Why are deep neural networks hard to train?", measures vanishing and exploding gradients in a deep network; the code samples are written in Python 2).
- Book: Simon J.D. Prince, *Understanding Deep Learning* (MIT Press, 2023) — [official page](https://udlbook.github.io/udlbook/) (free PDF; chapter 7, "Gradients and initialization", with notebooks 7.1–7.3, which implement backpropagation for a toy model and for a deep network and then compare initialization schemes).
- Book: Christopher M. Bishop and Hugh Bishop, *Deep Learning: Foundations and Concepts* (Springer, 2024) — [official page](https://www.bishopbook.com/) (free to read online; optional; chapter 8, "Backpropagation", including Jacobian matrices and automatic differentiation).
- Book: Ian Goodfellow, Yoshua Bengio, and Aaron Courville, *Deep Learning* (MIT Press, 2016) — [official page](https://www.deeplearningbook.org) (free online; advanced; section 6.5, "Back-Propagation and Other Differentiation Algorithms", a formal treatment of computational graphs and reverse-mode differentiation).

### Lectures, papers and articles

- Lecture series: Grant Sanderson (3Blue1Brown), [Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) (2017; free; start here: chapters 3 and 4, "Backpropagation, intuitively" and "Backpropagation calculus", about 25 minutes in total; chapters 1 and 2, on the structure of a network and on gradient descent, belong to Chapter 9; the chapters added in 2024 cover large language models and the Transformer and are listed in Chapter 12).
- Article: Christopher Olah, [Calculus on Computational Graphs: Backpropagation](https://colah.github.io/posts/2015-08-Backprop/) (2015; forward-mode and reverse-mode differentiation on a graph, and why reverse mode computes the derivatives of one output with respect to all inputs in a single pass).
- Article: Justin Johnson, [Derivatives, Backpropagation, and Vectorization](https://cs231n.stanford.edu/handouts/derivatives.pdf) (Stanford CS231n handout, 2017; 7 pages; extends the scalar chain rule to vectors, matrices, and tensors, and derives the gradient of a linear layer without forming the full Jacobian).
- Paper: David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams, [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0) (Nature, 1986; paid access; optional; four pages; the paper that introduced backpropagation to neural network research).
- Papers: Xavier Glorot and Yoshua Bengio, [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html) (International Conference on Artificial Intelligence and Statistics, 2010), and Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852) (International Conference on Computer Vision, 2015; measurements of how sigmoid layers saturate and how gradients shrink with depth, and the derivations of Xavier initialization and of He initialization for networks of rectified linear units (ReLU); section 2.2 of the second paper suffices).
- Paper: Atılım Güneş Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, and Jeffrey Mark Siskind, [Automatic differentiation in machine learning: a survey](https://arxiv.org/abs/1502.05767) (Journal of Machine Learning Research, 2018; advanced; distinguishes automatic differentiation from numerical and symbolic differentiation and surveys forward and reverse mode and their implementations).
- Tutorial: PyTorch, [A Gentle Introduction to torch.autograd](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) (current docs; the dynamic graph and the vector-Jacobian products of PyTorch; intended for reading after the milestone engine works, for comparison with it).

**Milestone.** Implement a scalar reverse-mode autograd engine in plain Python, modeled on micrograd (addition, multiplication, powers, `exp`, `log`, and `tanh` or ReLU), and verify its gradients on at least three expressions against finite differences and against PyTorch, to within 1e-6. Use the engine to train an MLP on a two-class toy dataset such as scikit-learn's `make_moons` to at least 95% accuracy, recording the loss at every step. Then train a character-level MLP language model on the names dataset of makemore with PyTorch tensors and a hand-written backward pass (no call to `loss.backward()`), check every gradient against autograd, and report a validation loss lower than that of a bigram count model on the same split.

**Time.** 4–6 weeks.

---

# Part IV — Transformers and LLMs

## Chapter 11 — NLP Foundations and Word Representations

**Goals.** This chapter covers the methods that natural language processing (NLP) used before the Transformer. Several of them — next-word prediction, perplexity, word embeddings, and attention — remain in use in large language models (LLMs). The chapter defines language modeling as next-word prediction, with cross-entropy and perplexity as the measures of quality, and follows language models from n-gram counts to neural networks. It describes how word2vec and GloVe represent words as dense vectors, how recurrent neural networks (RNNs) and long short-term memory networks (LSTMs) process a sequence one step at a time, and how sequence-to-sequence (seq2seq) encoder-decoder models map one sequence to another. The chapter ends with attention, which was introduced to remove the fixed-size bottleneck of encoder-decoder models, and with the limits of recurrence that the Transformer of Chapter 12 removes.

**Topics.** Language modeling as next-word prediction, measured by cross-entropy and perplexity; n-gram language models, smoothing, and data sparsity; the distributional hypothesis and count-based word vectors; word embeddings: word2vec, with its skip-gram and continuous bag-of-words (CBOW) models and negative sampling, and GloVe; evaluation of embeddings (similarity, analogies) and the social biases they encode; neural language models, from fixed-window networks to RNNs; backpropagation through time, vanishing gradients, LSTMs, and gated recurrent units (GRUs); seq2seq (encoder-decoder) models, teacher forcing, and greedy and beam-search decoding; attention, introduced to remove the fixed-size bottleneck of the encoder; contextual embeddings (ELMo) and the limits of recurrence that motivate the Transformer.

**Resources.**

**Suggested path.** Start with Lena Voita's NLP Course (word embeddings, language modeling, then seq2seq and attention) and read the matching chapters of Jurafsky and Martin in parallel. Continue with CS224N Spring 2024 lectures 1, 2, 5, and 6, using the two Alammar articles and Olah's LSTM article as visual companions to those lectures. Implement the milestone with the code of *Dive into Deep Learning* as a reference. When time is short, omit the DeepLearning.AI specialization, CMU 11-711, Eisenstein, and all original papers except word2vec and Bahdanau et al.

### University courses

- Stanford — [CS224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/) by Diyi Yang and Yejin Choi (free; Winter 2026 slides, notes, and assignments + [Spring 2024 lecture videos](https://www.youtube.com/playlist?list=PLoROMvodv4rOaMFbaqxPDoLWjDaRAdP9D) by Christopher Manning; start here: 2024 lectures 1–2 on word vectors and language models, 5 on RNNs, and 6 on sequence-to-sequence models and machine translation; Assignment 1 explores count-based and GloVe word vectors).
- CMU — [11-711 Advanced NLP, Fall 2024 lectures](https://www.youtube.com/playlist?list=PL8PYTP1V4I8D4BeyjwWczukWq9d8PNyZp) by Graham Neubig (free; videos; optional: a second, research-oriented treatment of word representation, language modeling, and sequence models; the [Spring 2026 course page](https://cmu-l3.github.io/anlp-spring2026/) by Sean Welleck has current slides and readings).

### Online courses (MOOCs)

- Lena Voita — [NLP Course | For You](https://lena-voita.github.io/nlp_course.html) (free; self-paced; start here: illustrated lectures with exercises on word embeddings, language modeling, and seq2seq with attention, in the same order as this chapter's topics).
- DeepLearning.AI on Coursera — [Natural Language Processing Specialization](https://www.coursera.org/specializations/natural-language-processing) by Younes Bensouda Mourri and Łukasz Kaiser (paid, financial aid available; optional; 4 courses, about 3 months at 10 hours a week; an alternative to CS224N with graded labs on word vectors, n-gram autocomplete, RNNs, LSTMs, and attention).

### Books

- Book: Dan Jurafsky and James H. Martin, *Speech and Language Processing* (3rd ed. draft, August 2026 release) — [official page](https://web.stanford.edu/~jurafsky/slp3/) (free PDF; start here: Chapters 2 "Words and Tokens", 3 "N-gram Language Models", 5 "Embeddings", 6 "Neural Networks", 13 "Machine Translation", and 14 "RNNs and LSTMs").
- Book: Aston Zhang, Zachary C. Lipton, Mu Li, and Alexander J. Smola, *Dive into Deep Learning* (Cambridge University Press, 2023) — [official page](https://d2l.ai/) (free online with runnable PyTorch code; Chapters 9–11 on RNNs, LSTMs, GRUs, encoder-decoder models, beam search, and attention, and Chapter 15 on word2vec and GloVe; the code reference for the milestone).
- Book: Jacob Eisenstein, *Introduction to Natural Language Processing* (MIT Press, 2019) — [free draft PDF](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/notes/eisenstein-nlp-notes.pdf) (free; optional: a more mathematical treatment; Chapter 6 "Language models", Chapter 14 "Distributional and distributed semantics", and Chapter 18 "Machine translation").

### Lectures, papers and articles

- Article: Jay Alammar, [The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/) (2019; start here: an illustrated explanation of embeddings, skip-gram, and negative sampling).
- Paper: Bengio et al., [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/v3/bengio03a.html) (2003; a feed-forward language model that learns word embeddings jointly with the network weights; the fixed-window model that RNNs replaced).
- Paper: Mikolov et al., [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) and [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) (2013; word2vec: skip-gram, CBOW, and negative sampling).
- Paper: Pennington, Socher, and Manning, [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/) (2014; embeddings fitted to global co-occurrence counts; the project page has the paper and pretrained vectors).
- Article: Andrej Karpathy, [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) (2015; character-level RNN language models, with generated samples and an analysis of what individual neurons track).
- Article: Christopher Olah, [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) (2015; step-by-step diagrams of the LSTM gates and of the GRU variant).
- Paper: Sutskever, Vinyals, and Le, [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) and Bahdanau, Cho, and Bengio, [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) (2014; the encoder-decoder model, and the attention mechanism that the Transformer generalizes).
- Article: Jay Alammar, [Visualizing A Neural Machine Translation Model](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) (2018; seq2seq with attention, animated step by step; a companion to the two papers above).
- Paper: Peters et al., [Deep contextualized word representations](https://arxiv.org/abs/1802.05365) (ELMo, 2018; optional: word vectors that depend on the sentence, computed by a pretrained bidirectional LSTM language model; the intermediate step between static embeddings and pretrained Transformers).

**Milestone.** Implement skip-gram with negative sampling in PyTorch, train it on a small corpus such as text8, and verify that related words cluster, using nearest-neighbor lists for ten probe words and a two-dimensional principal component analysis (PCA) plot. Then train a small LSTM language model on the same corpus and report its held-out perplexity next to that of a bigram baseline.

**Time.** 6–8 weeks.

## Chapter 12 — The Transformer Architecture, Deep Dive

**Goals.** This chapter describes the Transformer, the neural network architecture used by almost all current large language models (LLMs), at the level of detail needed to implement it. It covers scaled dot-product attention, causal masking, multi-head attention, positional encodings, residual connections, layer normalization, and the feed-forward layer, first in the encoder-decoder model of the original paper and then in the decoder-only stack used by GPT-style models. It also summarizes the changes that current open models make to the 2017 design — rotary position embeddings (RoPE), root mean square normalization (RMSNorm), gated feed-forward layers (SwiGLU), grouped-query attention (GQA), and mixture-of-experts (MoE) layers — because the model reports read from Chapter 15 onward assume them. The milestone implements multi-head attention; the tokenizer follows in Chapter 13, and the full model with its training loop in Chapter 14.

**Topics.** The limits of recurrence and the design goals of "Attention Is All You Need"; scaled dot-product attention: queries, keys, values, and the scaling factor; self-attention, cross-attention, and causal masking; multi-head attention and its tensor shapes; positional encodings: sinusoidal, learned absolute, and rotary (RoPE); the Transformer block: residual connections, normalization (LayerNorm and RMSNorm, pre-norm and post-norm placement), and the feed-forward layer; encoder-only (BERT), encoder-decoder (T5), and decoder-only (GPT) models, and the reasons LLMs use the decoder-only design; parameter counts, and the quadratic time and memory cost of attention in the sequence length; changes in current open LLMs: SwiGLU, grouped-query attention (GQA), and mixture-of-experts (MoE) layers (overview).

**Resources.**

**Suggested path.** Start with the two 3Blue1Brown videos and *The Illustrated Transformer*, then watch CS224N 2024 lecture 8 for the derivation. Next, watch Karpathy's "Let's build GPT" while typing the code, and work through Raschka Chapter 3 (or Prince Chapter 12, which is free) for a second implementation of attention. Read "Attention Is All You Need" after that, with *Formal Algorithms for Transformers* open as the reference for shapes and pseudocode. CS336 lectures 3 and 4 cover the changes made by current models and fit at the end; when time is short, omit CS25, *The Annotated Transformer*, and the optional papers.

### University courses

- Stanford — [CS224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/) by Diyi Yang and Yejin Choi (free; Winter 2026 slides, notes, and assignments + [Spring 2024 lecture videos](https://www.youtube.com/playlist?list=PLoROMvodv4rOaMFbaqxPDoLWjDaRAdP9D) by Christopher Manning; start here: 2024 lecture 8 "Transformers" derives self-attention from the limits of the recurrent neural networks (RNNs) of Chapter 11, and lecture 9 "Pretraining" compares encoder, encoder-decoder, and decoder models; Assignment 3 covers self-attention and Transformers).
- Stanford — [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) by Tatsunori Hashimoto and Percy Liang (free; Spring 2026 [lecture videos](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV) and assignments; advanced: lecture 3 "Architectures, hyperparameters" and lecture 4 "Attention alternatives and mixture of experts" survey the design choices of current LLMs; Assignment 1 is scheduled in Chapters 13 and 14).
- Stanford — [CS25: Transformers United](https://web.stanford.edu/class/cs25/) (free seminar; optional; sixth edition in Spring 2026; the [recordings on YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM) include the overview lectures, one of them by Andrej Karpathy, followed by guest talks on current research).

### Online courses (MOOCs)

- Andrej Karpathy — [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) (free; 1h56m; Lecture 7 of [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html), after the six lectures used in Chapter 10; start here: builds a character-level, decoder-only Transformer on Tiny Shakespeare in about 200 lines of PyTorch; Chapter 14 reuses this code and does not require a second viewing).
- DeepLearning.AI — [How Transformer LLMs Work](https://www.deeplearning.ai/short-courses/how-transformer-llms-work/) by Jay Alammar and Maarten Grootendorst (free; 1h44m; the lessons from "Architectural Overview" to "Mixture of Experts" cover the Transformer block, self-attention, the key-value cache, grouped-query attention, and MoE; the tokenizer lessons belong to Chapter 13).
- Hugging Face — [LLM Course, Chapter 1: Transformer models](https://huggingface.co/learn/llm-course/chapter1/4) (free; about 2 hours; sections 4–6 describe the encoder-only, decoder-only, and encoder-decoder families and the tasks each is used for).

### Books

- Book: Sebastian Raschka, *Build a Large Language Model (From Scratch)* (Manning, 2024) — [official page](https://www.manning.com/books/build-a-large-language-model-from-scratch) (paid; free code at [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) and free [companion videos](https://www.youtube.com/playlist?list=PLTKMiZHVd_2IIEsoJrWACkIxLRdfMlw11); start here: Chapter 3 "Coding Attention Mechanisms", from simplified self-attention to causal multi-head attention; Chapter 4 is used in Chapter 14).
- Book: Simon J.D. Prince, *Understanding Deep Learning* (MIT Press, 2023) — [official page](https://udlbook.github.io/udlbook/) (free PDF; Chapter 12 "Transformers", with notebooks on self-attention, multi-head attention, and tokenization).

### Lectures, papers and articles

- Article: Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) (2018; start here: an illustrated walkthrough of the original encoder-decoder model; the sequel [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) covers the decoder-only stack and masked self-attention).
- Video: 3Blue1Brown, [Transformers, the tech behind LLMs](https://www.3blue1brown.com/lessons/gpt) and [Attention in transformers, step-by-step](https://www.3blue1brown.com/lessons/attention) (2024; Chapters 5 and 6 of the Deep Learning series; about 50 minutes in total; animated explanations of embeddings, queries, keys, values, and masking).
- Paper: Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017; the original paper; Section 3 defines the architecture).
- Article: Sasha Rush et al., Harvard NLP, [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) (2018, updated 2022; the paper re-implemented section by section in PyTorch).
- Paper: Phuong and Hutter, [Formal Algorithms for Transformers](https://arxiv.org/abs/2207.09238) (2022; pseudocode with explicit tensor shapes for every component; a reference to keep open while coding).
- Paper: Devlin et al., [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) (2018; the encoder-only design, for comparison with the decoder-only GPT-2 report read in Chapter 14).
- Paper: Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (2021; RoPE, the positional encoding used by most current open LLMs).
- Paper: Xiong et al., [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) (2020) and Zhang and Sennrich, [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) (2019; optional: the reasons current models use pre-norm placement and RMSNorm).

**Milestone.** Implement multi-head self-attention with a causal mask from scratch in PyTorch, copy its weights into `nn.MultiheadAttention`, and verify that both modules produce the same output to within 1e-5. Verify causality by changing the last input token and checking that all earlier output positions are unchanged. Measure the size of the attention matrix at sequence lengths 512 and 2,048, and explain in one paragraph why time and memory grow as O(n²).

**Time.** 4–6 weeks.

## Chapter 13 — Tokens and Embeddings: How Text Becomes Numbers

**Goals.** This chapter covers the two steps that convert text into the input of a Transformer: tokenization, which splits text into units from a fixed vocabulary, and embedding, which maps each unit to a vector. It describes the byte pair encoding (BPE), WordPiece, and Unigram algorithms, the byte-level BPE variant used by GPT-style models, and the effect of vocabulary size and training corpus on sequence length, cost, and model quality. Several known failures of large language models (LLMs) — in spelling, arithmetic, and non-English text — originate in the tokenizer, and the chapter shows how to diagnose them. It ends with the embedding table, its relation to the output layer, and the data loader that converts a tokenized corpus into the training batches used in Chapter 14.

**Topics.** Unicode, UTF-8 bytes, and text normalization; word, character, byte, and subword tokenization, and their trade-offs; byte pair encoding (BPE): training, encoding, and decoding; byte-level BPE with a pre-tokenization regular expression (GPT-2, tiktoken); WordPiece, the Unigram language model, and the SentencePiece library; vocabulary size, special tokens, and compression ratio; the dependence of a tokenizer on its training corpus, and the resulting cost differences between languages; model failures caused by tokenization (spelling, arithmetic, trailing whitespace, undertrained tokens); the token embedding table, its relation to the positional embeddings of Chapter 12, and weight tying with the output layer; tokenized datasets: sliding-window input-target pairs and batches.

**Resources.**

**Suggested path.** Start with Karpathy's "Let's build the GPT Tokenizer" with Tiktokenizer open in a browser, then complete the minbpe exercise, on which the milestone is based. Read Raschka Chapter 2 for the embedding table and the data loader, and Sennrich et al. and Section 2.2 of the GPT-2 report for the original sources. Work through Chapter 6 of the Hugging Face LLM Course in parallel for WordPiece, Unigram, and the production libraries. CS336 lecture 1 summarizes the chapter in one lecture; the tokenizer part of its Assignment 1 is an alternative specification of the milestone with stricter tests. When time is short, omit that assignment and all papers marked optional.

### University courses

- Stanford — [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) by Tatsunori Hashimoto and Percy Liang (free; Spring 2026 [lecture videos](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV) and assignments; start here: lecture 1 "Overview, tokenization"; the first part of [Assignment 1 (Basics)](https://github.com/stanford-cs336/assignment1-basics) specifies a byte-level BPE tokenizer with unit tests and speed targets; the Transformer part of the assignment is scheduled in Chapter 14).

### Online courses (MOOCs)

- Andrej Karpathy — [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) (free; 2h13m; Lecture 8 of [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html); start here: builds a byte-level BPE tokenizer from scratch, reproduces the GPT-4 tokenizer, and traces a list of LLM failures to tokenization; the code and a five-step [exercise](https://github.com/karpathy/minbpe/blob/master/exercise.md) are in [minbpe](https://github.com/karpathy/minbpe)).
- Hugging Face — [LLM Course, Chapter 6: The Tokenizers library](https://huggingface.co/learn/llm-course/chapter6/1) (free; a few hours; trains a new tokenizer from an existing one, and implements BPE, WordPiece, and Unigram step by step).

### Books

- Book: Sebastian Raschka, *Build a Large Language Model (From Scratch)* (Manning, 2024) — [official page](https://www.manning.com/books/build-a-large-language-model-from-scratch) (paid; free code at [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) and free [companion videos](https://www.youtube.com/playlist?list=PLTKMiZHVd_2IIEsoJrWACkIxLRdfMlw11); start here: Chapter 2 "Working with Text Data" covers tokenization, special tokens, sliding-window data loaders, and token and positional embeddings; the repository adds a BPE tokenizer from scratch as bonus material).
- Book: Dan Jurafsky and James H. Martin, *Speech and Language Processing* (3rd ed. draft, August 2026 release) — [official page](https://web.stanford.edu/~jurafsky/slp3/) (free PDF; Chapter 2 "Words and Tokens" for Unicode, BPE, and text normalization; Chapter 5 "Embeddings", read in Chapter 11, for the meaning of the vectors).
- Book: Jay Alammar and Maarten Grootendorst, *Hands-On Large Language Models* (O'Reilly, 2024) — [official page](https://llm-book.com/) (paid; free notebooks on [GitHub](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models); Chapter 2 "Tokens and Embeddings" compares the tokenizers of released models and relates token, word, and text embeddings; the tokenizer lessons of the authors' short course listed in Chapter 12 cover the same material on video).

### Lectures, papers and articles

- Paper: Sennrich, Haddow, and Birch, [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) (2015; start here: the paper that adapted BPE, a data-compression algorithm, to the segmentation of words into subword units for machine translation).
- Paper: Radford et al., [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2, 2019; Section 2.2 "Input Representation" describes byte-level BPE and the rule that prevents merges across character categories; the released code implements that rule as a pre-tokenization regular expression, and later GPT tokenizers use variants of it).
- Paper: Kudo, [Subword Regularization](https://arxiv.org/abs/1804.10959) (2018) and Kudo and Richardson, [SentencePiece](https://arxiv.org/abs/1808.06226) (2018; the Unigram language model tokenizer, and the language-independent trainer that works on raw text, used by T5, Llama 2, and Gemma).
- Paper: Wu et al., [Google's Neural Machine Translation System](https://arxiv.org/abs/1609.08144) (2016; optional: Section 4.1 defines the WordPiece model that BERT uses).
- Paper: Press and Wolf, [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859) (2016; optional: weight tying between the embedding table and the output layer).
- Paper: Petrov et al., [Language Model Tokenizers Introduce Unfairness Between Languages](https://arxiv.org/abs/2305.15425) (2023; optional: measures how many more tokens, and therefore how much more cost, the same sentence requires in languages that are rare in the tokenizer's training corpus).
- Paper: Xue et al., [ByT5: Towards a token-free future with pre-trained byte-to-byte models](https://arxiv.org/abs/2105.13626) (2021; optional: a model that reads raw bytes without a tokenizer, and the sequence-length cost of doing so).

### Tools and hands-on

- Tool: [Tiktokenizer](https://tiktokenizer.vercel.app/) (free web page; start here: shows how production tokenizers such as GPT-4o's split any pasted text; the library behind it is OpenAI's [tiktoken](https://github.com/openai/tiktoken)).
- Code: Hugging Face, [tokenizers](https://github.com/huggingface/tokenizers) and Google, [SentencePiece](https://github.com/google/sentencepiece) (the two libraries with which most open models train their tokenizers; used for comparison after the from-scratch implementation).

**Milestone.** Implement a byte-level BPE tokenizer from scratch (the minbpe exercise is a suitable specification), train it with a vocabulary of about 8,000 tokens on a public corpus, and verify that encoding followed by decoding returns byte-identical text, including emoji and non-English samples. Report the compression ratio in bytes per token on held-out English text and on text in one other language, next to the ratios of the GPT-2 tokenizer from tiktoken, and explain the differences.

**Time.** 3–4 weeks.

## Chapter 14 — Build an LLM from Scratch

**Goals.** This chapter assembles the components of Chapters 12 and 13 into a complete GPT-style large language model (LLM) and trains it. It covers the model definition (embedding tables, a stack of Transformer blocks, and the output layer that predicts the next token), the cross-entropy training objective, and a training loop with the AdamW optimizer, learning-rate warmup and cosine decay, gradient clipping, and periodic evaluation on held-out data. It then covers text generation with greedy decoding, temperature, and top-k and top-p sampling. The implementation is verified by loading the released GPT-2 weights into it, and the trained model is saved and published on the Hugging Face Hub. Chapters 15 and 16 scale the same procedure up in data, model size, and hardware.

**Topics.** The GPT model: token and positional embeddings, Transformer blocks, final normalization, and the output layer; parameter counting, weight tying between the embedding table and the output layer, and weight initialization; the next-token prediction objective, the cross-entropy loss, and the training/validation split; batches of token sequences: context length and batch size; the training loop: AdamW, weight decay, learning-rate warmup and cosine decay, and gradient clipping; monitoring of training and validation loss, overfitting, and generated samples; text generation: the autoregressive loop, greedy decoding, temperature, and top-k and top-p sampling; saving and loading checkpoints (model and optimizer state); loading the released GPT-2 weights into the implementation as a correctness check; publishing a model on the Hugging Face Hub.

**Resources.**

**Suggested path.** Raschka Chapters 4 and 5 with Appendix D are the main text of this chapter: type and run the code chapter by chapter, with the companion videos as a second explanation. Karpathy's "Let's build GPT" was watched in Chapter 12 and is not repeated here; continue with the first section of "Let's reproduce GPT-2 (124M)" for the model definition and the weight-loading check, and with its optimizer section for AdamW, gradient clipping, and the learning-rate schedule, with nanoGPT open as a reference implementation. Read Holtzman et al. and "How to generate text" before writing the sampling code, the TinyStories paper before choosing the milestone corpus, and Chapter 4 of the Hugging Face LLM Course before publishing the checkpoint. CS336 lectures 2–3 and the Transformer part of Assignment 1 are a stricter, test-driven route to the same result; when time is short, omit them together with *Hands-On Large Language Models*, the GPT-3 paper, nanochat, and llm.c.

### University courses

- Stanford — [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) by Tatsunori Hashimoto and Percy Liang (free; Spring 2026 [lecture videos](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV) and assignments; start here: lecture 2 "PyTorch (einops), resource accounting" and lecture 3 "Architectures, hyperparameters", first assigned in Chapter 12; in [Assignment 1 (Basics)](https://github.com/stanford-cs336/assignment1-basics) the Transformer language model, cross-entropy loss, AdamW, training loop, and decoding are written without the ready-made `torch.nn` layers and checked by unit tests; the tokenizer part was covered in Chapter 13).

### Online courses (MOOCs)

- Andrej Karpathy — [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) (free; 4 hours; the sequel to Lecture 7 of [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html), with step-by-step code in [build-nanogpt](https://github.com/karpathy/build-nanogpt); start here: section 1, about the first 80 minutes, defines the GPT-2 model, loads the released weights, and writes the training loop; the section on AdamW, gradient clipping, and the learning-rate schedule starts at about 2h15m; the sections on speed and multi-GPU training belong to Chapter 16).
- Hugging Face — [LLM Course, Chapter 4: Sharing models and tokenizers](https://huggingface.co/learn/llm-course/chapter4/1) (free; about 2 hours; the Hugging Face Hub, uploading a checkpoint, and writing a model card, as required by the milestone; Chapter 2 "Using Transformers" covers loading and saving models).

### Books

- Book: Sebastian Raschka, *Build a Large Language Model (From Scratch)* (Manning, 2024) — [official page](https://www.manning.com/books/build-a-large-language-model-from-scratch) (paid; free code at [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) and free [companion videos](https://www.youtube.com/playlist?list=PLTKMiZHVd_2IIEsoJrWACkIxLRdfMlw11); start here: Chapter 4 "Implementing a GPT Model from Scratch to Generate Text", Chapter 5 "Pretraining on Unlabeled Data", which includes decoding strategies and loading the GPT-2 weights, and Appendix D on warmup, cosine decay, and gradient clipping; Chapters 2 and 3 were used in Chapters 13 and 12).
- Book: Jay Alammar and Maarten Grootendorst, *Hands-On Large Language Models* (O'Reilly, 2024) — [official page](https://llm-book.com/) (paid; optional; free notebooks on [GitHub](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models); Chapter 3 "Looking Inside Transformer LLMs" follows one forward pass of a released model, including the decoding strategy and the key-value cache).

### Lectures, papers and articles

- Paper: Radford et al., [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2, 2019; start here: the report of the model reproduced in this chapter; Section 2.3 lists its changes to the original Transformer).
- Paper: Brown et al., [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (GPT-3, 2020; optional: the same design scaled to 175B parameters; Table 2.1 lists model sizes, batch sizes, and learning rates that later work reuses, and Appendix B gives the optimizer settings).
- Paper: Holtzman et al., [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) (2019; shows that greedy and beam search produce repetitive text, and introduces top-p (nucleus) sampling).
- Article: Patrick von Platen, [How to generate text: using different decoding methods for language generation with Transformers](https://huggingface.co/blog/how-to-generate) (Hugging Face blog, 2020; greedy search, beam search, and top-k and top-p sampling compared in code).
- Paper: Eldan and Li, [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759) (2023; a synthetic corpus of short stories on which models below 10M parameters produce fluent text; the [dataset](https://huggingface.co/datasets/roneneldan/TinyStories) makes the milestone feasible on a single GPU).

### Tools and hands-on

- Code: Andrej Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT) (start here: a GPT model definition of about 300 lines and a training loop of about 300 lines; the README now points to nanochat as the successor, but the code remains a compact reference for this chapter).
- Code: Andrej Karpathy, [nanochat](https://github.com/karpathy/nanochat) (2025; advanced: tokenizer, pretraining, fine-tuning, evaluation, and inference for a small chat model, designed to run on one node of eight H100 GPUs).
- Code: Andrej Karpathy, [llm.c](https://github.com/karpathy/llm.c) (2024; optional: GPT-2 training written in plain C and CUDA without PyTorch; shows every forward and backward kernel explicitly).

**Milestone.** Load the released GPT-2 (124M) weights into the from-scratch implementation and verify that its logits match those of the Hugging Face `GPT2LMHeadModel` on the same input. Then train a decoder-only model of 10–50M parameters on a public corpus such as TinyStories, plot the training and validation loss, and compare samples generated with greedy decoding, temperature, top-k sampling, and top-p sampling. Publish the checkpoint on the Hugging Face Hub with a model card that states the data, the hyperparameters, and the final validation loss.

**Time.** 6–8 weeks.

## Chapter 15 — Pretraining at Scale: Data, Scaling Laws, and Training Runs

**Goals.** This chapter covers the differences between the small model of Chapter 14 and a released open large language model (LLM): the data, the model size, and the length and cost of the training run. It describes how a pretraining corpus is built from web crawls and curated sources by text extraction, filtering, deduplication, and mixing, and how small ablation models are used to choose among data recipes. It presents the scaling laws of Kaplan et al. and of Hoffmann et al. (Chinchilla), which relate the loss to parameters, tokens, and compute, and the estimate C ≈ 6ND, which converts a parameter count N and a token count D into floating-point operations (FLOPs), GPU-hours, and cost. It also covers the configuration of a long training run — batch size, learning-rate schedule, checkpointing, and recovery from loss spikes — as documented in the reports of open models such as OLMo 2, Llama 3, and SmolLM3.

**Topics.** Sources of pretraining text: Common Crawl, code, books, papers, and curated corpora; legal and privacy constraints on training data: licenses, copyright, and removal of personal information; corpus construction: text extraction, language identification, quality filtering, deduplication, and removal of benchmark data (decontamination); open corpora: The Pile, RefinedWeb, Dolma, FineWeb, and FineWeb-Edu; data mixtures, synthetic data, and ablations on small models to choose among them; training compute: C ≈ 6ND, FLOPs, GPU-hours, model FLOPs utilization (MFU), and cost; scaling laws (Kaplan, Chinchilla) and compute-optimal training; token budgets in practice: training beyond the compute-optimal point to reduce inference cost; hyperparameters at scale: batch size, learning rate, and schedules (cosine, warmup-stable-decay); training stability: loss spikes, their mitigations, checkpointing, and restarts; the effect of data quality on downstream evaluation results.

**Resources.**

**Suggested path.** Start with *The Smol Training Playbook*, which follows one open training run from the decision to pretrain through ablations, data mixing, and failure recovery; it is book-length, so read it over several weeks alongside the other material. In parallel, watch CS336 lectures 9 and 11 with the Kaplan and Chinchilla papers, then lectures 13 and 14 with the FineWeb paper; *Transformer Math 101* supplies the compute formulas for the milestone, and the final part of Karpathy's "Let's reproduce GPT-2 (124M)", listed in Chapter 14, shows a complete run on 10B tokens of FineWeb-Edu. Of the model reports, read at least the pretraining sections of OLMo 2 and Llama 3. When time is short, omit CS324, the DeepLearning.AI short course, Raschka's bonus material, The Pile, RefinedWeb, Dolma, and DeepSeek-V3.

### University courses

- Stanford — [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) by Tatsunori Hashimoto and Percy Liang (free; Spring 2026 [lecture videos](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV) and assignments; start here: lectures 9 and 11 "Scaling laws", lecture 13 "Data (sources, datasets)", and lecture 14 "Data (filtering, deduplication, mixing, synthetic data)"; lecture 2, first assigned in Chapter 14, supplies the FLOPs accounting; [Assignment 3 (Scaling)](https://github.com/stanford-cs336/assignment3-scaling) has instructions for non-students, because its hosted training API is reserved for enrolled students; the filtering and deduplication parts of [Assignment 4 (Data)](https://github.com/stanford-cs336/assignment4-data) run offline, and its final 8-GPU training run can be omitted).
- Stanford — [CS324: Large Language Models](https://stanford-cs324.github.io/winter2022/) by Percy Liang, Tatsunori Hashimoto, and Christopher Ré (free; Winter 2022 lecture notes; optional: the notes on Data, Training, and Scaling laws are a short written summary of the same topics, without the developments after 2022).

### Online courses (MOOCs)

- DeepLearning.AI and Upstage — [Pretraining LLMs](https://www.deeplearning.ai/short-courses/pretraining-llms/) by Sung Kim and Lucy Park (free; 1h29m; optional: notebooks on data preparation, packing of token sequences, model initialization, a short training run, and evaluation).

### Books

- Book: Sebastian Raschka, *Build a Large Language Model (From Scratch)* (Manning, 2024) — [official page](https://www.manning.com/books/build-a-large-language-model-from-scratch) (paid; free code at [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) and free [companion videos](https://www.youtube.com/playlist?list=PLTKMiZHVd_2IIEsoJrWACkIxLRdfMlw11); optional: the pretraining chapter of the book was completed in Chapter 14; the repository's bonus material [Pretraining GPT on the Project Gutenberg Dataset](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/03_bonus_pretraining_on_gutenberg) extends the Chapter 5 training loop to a multi-gigabyte corpus).

### Lectures, papers and articles

- Article: Loubna Ben Allal et al. (Hugging Face), [The Smol Training Playbook: The Secrets to Building World-Class LLMs](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) (2025; start here: a book-length account of training SmolLM3, a 3B-parameter model, on 11T tokens: whether to pretrain at all, ablations, architecture, data mixing, loss spikes, and infrastructure failures).
- Article: Quentin Anthony, Stella Biderman, and Hailey Schoelkopf, [Transformer Math 101](https://blog.eleuther.ai/transformer-math/) (EleutherAI, 2023; first used in Chapter 7; the sections "Compute Requirements" and "Parameter vs Dataset Tradeoffs" give C ≈ 6PD, in which P is the parameter count, and its use for cost estimates; the sections on training memory and distributed training are used in Chapter 16).
- Paper: Kaplan et al., [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (2020; power laws relating the loss of Transformer language models to parameters, data, and compute).
- Paper: Hoffmann et al., [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Chinchilla, 2022; revises Kaplan et al.: parameters and tokens are scaled together, at roughly 20 tokens per parameter; Section 3.3 gives the parametric loss fit used in the milestone).
- Paper: Gao et al., [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027) (2020; an open corpus assembled from 22 curated sources, on which the GPT-NeoX and Pythia models were trained).
- Paper: Lee et al., [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) (2021; exact-substring and near-duplicate (MinHash) deduplication, and their effect on memorization and evaluation).
- Paper: Penedo et al., [The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116) (2023; reports that models trained on filtered and deduplicated web data alone match models trained on curated corpora).
- Paper: Penedo et al., [The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale](https://arxiv.org/abs/2406.17557) (2024; the pipeline, with an ablation for each step, behind the open [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) corpus and its classifier-filtered subset FineWeb-Edu).
- Paper: Soldaini et al., [Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research](https://arxiv.org/abs/2402.00159) (2024; a documented data pipeline, including the removal of personal information, released together with its tools).
- Paper: Groeneveld et al., [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838) (2024) and Team OLMo, [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656) (2024; open models released with data, code, logs, and checkpoints; the OLMo 2 report has a section on training stability).
- Paper: Meta, [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (2024; Section 3 "Pre-Training" describes the data pipeline, the scaling-law experiments, the 15T-token training recipe, and the MFU achieved on 16,000 GPUs).
- Paper: DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) (2024; advanced: a 671B-parameter mixture-of-experts (MoE) model trained in 8-bit floating point (FP8), with its pretraining data, hyperparameters, and compute cost in GPU-hours).

**Milestone.** Write a two-page training plan for a 1B-parameter open model: data sources and mixture, token budget (one compute-optimal and one over-trained variant), compute in FLOPs from C ≈ 6ND, GPU-hours and cost on a named GPU at a stated MFU, and the loss predicted by the Chinchilla parametric fit. Validate the arithmetic against published figures: reproduce the 3.8 × 10²⁵ FLOPs that the Llama 3 report states for its 405B model, and compute the MFU implied by the GPU count and training duration reported for SmolLM3.

**Time.** 6–8 weeks.

## Chapter 16 — Distributed Training and LLM Systems

**Goals.** This chapter describes how a large language model (LLM) that does not fit in the memory of one graphics processing unit (GPU), or whose training on one GPU would take years, is trained on many GPUs. It first accounts for the memory of one training step — weights, gradients, optimizer states, and activations — and covers the single-GPU techniques that reduce that memory or raise the speed: mixed precision, gradient accumulation, activation recomputation, kernel fusion, and FlashAttention. It then covers the collective communication operations (all-reduce, all-gather, and reduce-scatter) and the forms of parallelism built on them: data parallelism, sharded data parallelism with the Zero Redundancy Optimizer (ZeRO) and PyTorch Fully Sharded Data Parallel (FSDP), and tensor, pipeline, sequence, context, and expert parallelism. For each method the chapter states what is divided among the GPUs, what is communicated, and how the result is measured as throughput and as model FLOPs utilization (MFU), the fraction of the peak floating-point operations per second of the GPUs that the training run achieves. The model reports of Chapter 15 describe their training infrastructure in these terms, and Chapters 17 and 21 reuse the memory accounting and the attention kernels.

**Topics.** The memory of a training step: weights, gradients, optimizer states, and activations; mixed precision training: float16 (fp16) with loss scaling, bfloat16 (bf16), and float32 (fp32) master weights; gradient accumulation and activation recomputation (gradient checkpointing); GPU performance: the memory hierarchy, kernel fusion, `torch.compile`, and profiling; attention kernels that minimize GPU memory traffic (FlashAttention); collective communication: all-reduce, all-gather, and reduce-scatter, the NVIDIA Collective Communications Library (NCCL), and interconnect bandwidth; data parallelism with PyTorch DistributedDataParallel (DDP); sharded data parallelism: ZeRO stages 1–3 and FSDP; tensor parallelism and pipeline parallelism; sequence, context, and expert parallelism, and the conditions that require each; combined parallelism: choosing a configuration for a given model size and GPU count; measurement: tokens per second, scaling efficiency, and MFU; the functions automated by DeepSpeed, PyTorch FSDP, and Megatron-LM.

**Resources.**

**Suggested path.** Read the article by Lilian Weng for an overview and reread the article by Horace He from Chapter 7, then work through sections 2 and 3 of Karpathy's "Let's reproduce GPT-2 (124M)" with the code open, from mixed precision to DDP; the milestone repeats those steps. *The Ultra-Scale Playbook* is the main text: read it in order over several weeks, with CS336 lectures 5–8 in parallel and the PyTorch DDP and FSDP tutorials when the corresponding section is reached; its appendix A0 explains the collective operations used in the milestone. Read each paper after the Playbook section on the same method; the abstract, the method section, and the main results table are sufficient on a first pass. When time is short, omit CMU 11-868, *How to Scale Your Model*, CS336 Assignment 2, and the FSDP, FlashAttention-2, and FlashAttention-3 papers.

### University courses

- Stanford — [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) by Tatsunori Hashimoto and Percy Liang (free; Spring 2026 [lecture videos](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV) and assignments; start here: lecture 5 "GPUs, TPUs", lecture 6 "Kernels, Triton", and lectures 7 and 8 "Parallelism"; in [Assignment 2 (Systems)](https://github.com/stanford-cs336/assignment2-systems) the Assignment 1 model is profiled, FlashAttention-2 is written in Triton, and data-parallel training with a sharded optimizer state is implemented; the assignment is advanced and optional).
- CMU — [11-868: Large Language Model Systems](https://llmsystem.github.io/llmsystem2026spring/) by Lei Li (free; Spring 2026 slides and homework; optional: GPU programming, three lectures on distributed training, memory optimization, mixture-of-experts (MoE) models, and attention kernels; a second presentation of the topics of this chapter).

### Online courses (MOOCs)

- Andrej Karpathy — [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) (free; 4 hours; the sequel to Lecture 7 of [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html), with step-by-step code in [build-nanogpt](https://github.com/karpathy/build-nanogpt); start here: section 2, from about 1h22m to 2h15m, adds TF32 and bf16 mixed precision, `torch.compile`, and FlashAttention one change at a time and measures the speedup of each; gradient accumulation starts at about 2h34m and DDP on 8 GPUs at about 2h47m; section 1 was used in Chapter 14).

### Books

- Book: Nouamane Tazi, Ferdinand Mom et al. (Hugging Face), *The Ultra-Scale Playbook: Training LLMs on GPU Clusters* (2025) — [free online edition](https://huggingface.co/spaces/nanotron/ultrascale-playbook) (free online; start here: the main text of this chapter, first used in Chapter 7; covers training memory on one GPU, data parallelism and ZeRO, tensor, context, pipeline, and expert parallelism, the choice of a configuration, and GPU kernels, each with benchmark measurements; appendix A0 covers the collective operations).
- Book: Stas Bekman, *Machine Learning Engineering Open Book* (continuously updated) — [official page](https://github.com/stas00/ml-engineering) (free; a practitioner's reference, first used in Chapter 7; the Training chapters "Model parallelism", "Performance", and "Fault tolerance" and the Network chapter cover the choice of a parallelism method, throughput tuning, and interconnects).
- Book: Jacob Austin et al. (Google DeepMind), *How to Scale Your Model: A Systems View of LLMs on TPUs* (2025) — [free online edition](https://jax-ml.github.io/scaling-book/) (free online; advanced; optional: the sections on rooflines, Transformer math, and training parallelism derive the compute and communication cost of each method, followed by a worked example for Llama 3; written for Google's Tensor Processing Units (TPUs) and JAX, with a later section on GPUs).

### Lectures, papers and articles

- Article: Lilian Weng, [How to Train Really Large Models on Many GPUs?](https://lilianweng.github.io/posts/2021-09-25-train-large/) (2021; start here: a summary of data, pipeline, and tensor parallelism, MoE models, activation recomputation, and mixed precision, with figures from the source papers).
- Article: Horace He, [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) (2022; a second reading of the article from Chapter 7: compute, memory bandwidth, and overhead as the three bottlenecks of GPU work, and operator fusion as the method that reduces memory traffic).
- Article: Quentin Anthony, Stella Biderman, and Hailey Schoelkopf, [Transformer Math 101](https://blog.eleuther.ai/transformer-math/) (EleutherAI, 2023; the section "Memory Requirements" gives formulas for the memory of weights, gradients, optimizer states, and activations, and the section "Distributed Training" shows how ZeRO and combined parallelism divide that memory among GPUs; the compute sections were used in Chapter 15).
- Docs: PyTorch, [PyTorch Distributed Overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html) (current docs; lists the distributed APIs and the use case of each), then the [DDP tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html) and [Getting Started with FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) (runnable examples; FSDP1 is deprecated, and the tutorial uses its replacement, `fully_shard`).
- Paper: Micikevicius et al., [Mixed Precision Training](https://arxiv.org/abs/1710.03740) (2017; fp16 weights and activations with an fp32 master copy of the weights and loss scaling; the abstract was read in Chapter 7, the full paper is read here).
- Paper: Rajbhandari et al., [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) (2019; partitions optimizer states, gradients, and parameters among the data-parallel GPUs in three stages; the DeepSpeed [ZeRO tutorial](https://www.deepspeed.ai/tutorials/zero/) shows the three stages and offload to CPU memory, configured in a JSON file).
- Paper: Zhao et al., [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277) (2023; optional: the design of the PyTorch implementation of ZeRO stage 3 sharding).
- Paper: Shoeybi et al., [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) (2019; tensor parallelism inside each Transformer layer), Narayanan et al., [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) (2021; pipeline schedules and the combination of data, tensor, and pipeline parallelism), and Korthikanti et al., [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) (2022; sequence parallelism and selective activation recomputation; reports MFU).
- Paper: Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (2022; exact attention computed in blocks that fit in on-chip memory, so that the full attention matrix is never stored; the follow-ups [FlashAttention-2](https://arxiv.org/abs/2307.08691) (Dao, 2023) and [FlashAttention-3](https://arxiv.org/abs/2407.08608) (Shah et al., 2024) are advanced and optional).

### Tools and hands-on

- Code: Andrej Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT) (start here: a training loop of about 300 lines with mixed precision, `torch.compile`, gradient accumulation, and DDP launched with `torchrun`; its README reports a GPT-2 (124M) reproduction on one node of eight A100 GPUs in about 4 days, and now points to nanochat as the successor; the code base for the milestone, first listed in Chapter 14).

**Milestone.** Train nanoGPT or the Chapter 14 model on one GPU and record tokens per second and peak memory as mixed precision, `torch.compile`, and the fused attention kernel of `scaled_dot_product_attention` are enabled one at a time. Run the same configuration with DDP on 2–4 GPUs — the two T4 GPUs of a free Kaggle notebook are sufficient — and report tokens per second, scaling efficiency relative to one GPU, and MFU; with a single GPU, use gradient accumulation instead and verify that the loss curve matches that of the equivalent larger batch. Compute the data volume that one ring all-reduce of the gradients sends from each GPU, 2(N − 1)/N times the gradient size for N GPUs, estimate its duration at the bandwidth of the interconnect used, and compare it with the measured step time.

**Time.** 6–8 weeks.

---

# Part V — Post-Training: Fine-Tuning, Alignment, and Reasoning

## Chapter 17 — Fine-Tuning and Parameter-Efficient Methods

**Goals.** This chapter covers the adaptation of a pretrained open large language model (LLM) to follow instructions or to perform a single task. It treats supervised fine-tuning (SFT): training on prompts paired with reference responses, formatted with the chat template of the model, and the evidence that a few thousand curated examples can match a much larger unfiltered set. It explains why full fine-tuning requires graphics processing unit (GPU) memory equal to several times the size of the model, and how low-rank adaptation (LoRA), which trains small low-rank matrices added to frozen weights, and quantized LoRA (QLoRA), which applies LoRA to a base model quantized to 4 bits, reduce that requirement to the memory of one consumer GPU. It also covers the criteria for choosing fine-tuning over prompting or retrieval, and the measurement of both the task performance gained and the general capability lost. Alignment (Chapter 18), reasoning (Chapter 19), and the capstone (Chapter 23) start from a model fine-tuned with these methods.

**Topics.** What fine-tuning changes: continued pretraining, SFT, and instruction tuning; choosing between fine-tuning, prompting, and retrieval-augmented generation (RAG, Chapter 22); instruction datasets: Alpaca and chat-message formats, synthetic data, and data quality versus dataset size; chat templates and special tokens; the SFT loss: next-token prediction computed on the response tokens only, and sequence packing; full fine-tuning and its memory cost (weights, gradients, optimizer states); parameter-efficient fine-tuning (PEFT): freezing layers, adapters, and prefix and prompt tuning; LoRA: low-rank updates, the choice of layers to adapt, and merging the adapter into the base weights; LoRA hyperparameters (rank, alpha, learning rate, dropout); QLoRA: LoRA on a 4-bit quantized base model (quantization is treated in full in Chapter 21); training on consumer hardware: mixed precision, gradient accumulation, and gradient checkpointing; overfitting, catastrophic forgetting, and evaluation before and after fine-tuning.

**Resources.**

**Suggested path.** Start with chapter 11 of the Hugging Face LLM Course (a few evenings), then implement the same components from scratch with chapters 6–7 and appendix E of *Build a Large Language Model (From Scratch)*, the main text of this chapter. Read the LoRA and QLoRA papers alongside appendix E, and "LoRA Without Regret" before choosing hyperparameters for the milestone; CS336 lecture 15 fits at any point and covers instruction data and SFT practice at research-lab scale. When time is short, skip chapter 6 of *Build a Large Language Model (From Scratch)*, the FLAN and LIMA papers, and the other three books, and complete the milestone with the Hugging Face Transformer Reinforcement Learning (TRL) library or Unsloth, starting from the LLM Course notebooks.

### University courses

- Stanford — [CS336 Language Modeling from Scratch (Spring 2026)](https://cs336.stanford.edu/) by Tatsunori Hashimoto and Percy Liang (free; [videos](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV) + assignments; start here: lecture 15, "Mid/post-training (SFT/RLHF)", first half, on instruction data and SFT practice; advanced: the SFT part of [assignment 5](https://github.com/stanford-cs336/assignment5-alignment), which fine-tunes a model on math problems and is sized for a data-center GPU).

### Online courses (MOOCs)

- Hugging Face — [LLM Course, Chapter 11: Supervised Fine-Tuning](https://huggingface.co/learn/llm-course/chapter11/1) (free; a few hours; start here: four sections on chat templates, SFT with the TRL `SFTTrainer`, LoRA, and evaluation, each with a notebook).
- Hugging Face — [smol course](https://huggingface.co/learn/smol-course/unit0/1) (free; v2, 2025; self-paced, based on SmolLM3 and TRL; unit 1, "Instruction Tuning", provides further practice with chat templates and SFT, with graded exercises and a free certificate).

### Books

- Book: Sebastian Raschka, *Build a Large Language Model (From Scratch)* (Manning, 2024) — [official page](https://www.manning.com/books/build-a-large-language-model-from-scratch) (paid; free code at [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch); start here: chapter 6, "Fine-tuning for classification", chapter 7, "Fine-tuning to follow instructions", and appendix E, which implements LoRA from scratch, all on the GPT-2 model built in Chapter 14).
- Book: Jay Alammar and Maarten Grootendorst, *Hands-On Large Language Models* (O'Reilly, 2024) — [official page](https://llm-book.com/) (paid; free notebooks on [GitHub](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models); chapter 12, "Fine-Tuning Generation Models", demonstrates QLoRA instruction tuning of TinyLlama with the Hugging Face libraries, followed by preference tuning, which belongs to Chapter 18).
- Book: Chip Huyen, *AI Engineering* (O'Reilly, 2025) — [book repo](https://github.com/chiphuyen/aie-book) (paid; free chapter summaries; chapter 7, "Finetuning", covers the choice between fine-tuning, prompting, and RAG, the memory arithmetic, PEFT, and model merging; chapter 8, "Dataset Engineering", covers curating and synthesizing training data).
- Book: Nathan Lambert, *Reinforcement Learning from Human Feedback* (Manning, 2026) — [official page](https://rlhfbook.com/) (free online, paid print; chapter 4, "Instruction Fine-Tuning", is a short account of chat templates, prompt masking, and SFT practice in current post-training pipelines; the rest of the book belongs to Chapter 18).

### Lectures, papers and articles

- Paper: Edward J. Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (2021; start here: defines the low-rank update used by most fine-tunes of open models; read sections 1–4).
- Paper: Tim Dettmers et al., [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (2023; LoRA on a base model quantized to the 4-bit NormalFloat (NF4) format, with paged optimizers; reduces the memory needed to fine-tune a 7-billion-parameter model to that of one consumer GPU).
- Article: John Schulman and Thinking Machines Lab, [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) (2025; the conditions under which LoRA matches full fine-tuning: adapters on all layers, including the multilayer perceptron (MLP) layers, a rank sufficient for the dataset size, and a learning rate about 10 times that of full fine-tuning).
- Paper: Dan Biderman et al., [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673) (2024; measures the target-domain performance gained by LoRA and by full fine-tuning, and the base-model capability that each loses).
- Paper: Jason Wei et al., [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652) (2021, FLAN; shows that instruction tuning on many tasks improves zero-shot performance on unseen tasks).
- Paper: Chunting Zhou et al., [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) (2023; fine-tunes a 65-billion-parameter base model on 1,000 curated examples and obtains assistant-style answers of competitive quality; the standard reference for data quality versus dataset size).

### Tools and hands-on

- Hugging Face — [PEFT](https://huggingface.co/docs/peft/index) and [TRL](https://huggingface.co/docs/trl/index) documentation (free, open source; start here: the libraries most widely used for LoRA and QLoRA adapters (PEFT) and for SFT (TRL); the PEFT conceptual guides also cover adapters, prefix and prompt tuning, and LoRA variants such as weight-decomposed low-rank adaptation (DoRA); the [`SFTTrainer`](https://huggingface.co/docs/trl/sft_trainer) page documents the trainer used in the milestone).
- [Unsloth](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide), [Axolotl](https://github.com/axolotl-ai-cloud/axolotl), and [LlamaFactory](https://github.com/hiyouga/LlamaFactory) (free, open source; optional: Unsloth provides LoRA and QLoRA training with reduced memory use and training time, and ready-made Colab notebooks; Axolotl and LlamaFactory configure a complete fine-tuning run from one configuration file).
- Hugging Face — [SmolLM](https://github.com/huggingface/smollm) (free, open weights; the SmolLM2 models with 135M, 360M, and 1.7B parameters and the 3B SmolLM3 model can be fine-tuned on a free-tier GPU; the training recipes are published).

**Milestone.** Fine-tune a small base model with LoRA (SmolLM2-360M, or SmolLM2-135M on a slow GPU; both fit a free Colab or Kaggle T4 GPU) on a few thousand examples of an open instruction dataset, formatted with a chat template. Report, before and after fine-tuning: the loss on a held-out slice of the dataset, a blind pass/fail grading by hand of the answers to 30 held-out prompts, and, as a check on catastrophic forgetting, the perplexity on plain text such as WikiText-2. Repeat the run once with a different rank or learning rate and explain the difference.

**Time.** 4–6 weeks.

## Chapter 18 — Alignment: RLHF, DPO, and Getting Models to Follow Instructions

**Goals.** A pretrained large language model (LLM) continues text, and the supervised fine-tuning (SFT) of Chapter 17 teaches it the format of an assistant answer but provides no signal about which of two answers people prefer. This chapter covers how that preference is trained into a model: the collection of human or AI judgments between pairs of answers, the reward model that converts those judgments into a scalar score, and reinforcement learning from human feedback (RLHF), which uses the reinforcement learning (RL) algorithm proximal policy optimization (PPO) to increase that score while a Kullback–Leibler (KL) divergence penalty keeps the model close to its starting point. It then covers direct preference optimization (DPO), which optimizes the same objective with a classification loss on the preference pairs and no RL loop, and which the milestone applies to a small model. The chapter closes with reward hacking and over-optimization, in which the reward score rises while answer quality falls, and with the evaluation of whether a tuned model is more helpful and no less safe.

**Topics.** The difference between a base model and an assistant model, and the post-training pipeline (SFT, preference tuning, RL); human preference data: pairwise comparisons, annotation guidelines, and open datasets (the Anthropic Helpful and Harmless dataset (HH-RLHF), UltraFeedback); reward models and the Bradley-Terry preference model; RLHF with PPO: policy, reward, and advantage at a conceptual level (the RL foundations are in Chapter 19); the KL penalty that keeps the policy close to the reference model; rejection sampling and best-of-N sampling as baselines; DPO and its variants: identity preference optimization (IPO), Kahneman-Tversky optimization (KTO), odds ratio preference optimization (ORPO), and simple preference optimization (SimPO); offline vs online preference optimization; RL from AI feedback (RLAIF) and Constitutional AI; reward hacking and over-optimization; evaluation of helpfulness and harmlessness: win rates, LLM judges, refusals, and sycophancy; open end-to-end recipes (Tülu 3).

**Resources.**

**Suggested path.** Start with the article "Illustrating RLHF" and the lecture "State of GPT" for an overview of the pipeline, then use *Reinforcement Learning from Human Feedback* by Nathan Lambert and its companion lectures as the main text of this chapter, reading the InstructGPT and DPO papers when the book reaches those topics. Work through unit 3 of the smol course in parallel as preparation for the milestone. This chapter requires PPO only at a conceptual level: the RL foundations (Sutton and Barto, Berkeley CS 285, David Silver's course, Spinning Up) are listed in Chapter 19, which derives the algorithms. When time is short, skip the university courses other than CS336 lecture 15, the PPO paper, and the two Anthropic papers, and read only the summary sections of the Tülu 3 paper.

### University courses

- Stanford — [CS336 Language Modeling from Scratch (Spring 2026)](https://cs336.stanford.edu/) by Tatsunori Hashimoto and Percy Liang (free; [videos](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV) + assignments; start here: lecture 15, "Mid/post-training (SFT/RLHF)", second half, on preference data, PPO, and DPO; advanced: the optional safety-alignment part of [assignment 5](https://github.com/stanford-cs336/assignment5-alignment) implements DPO).
- Stanford — [CS234 Reinforcement Learning](https://web.stanford.edu/class/cs234/) by Emma Brunskill (free; Winter 2026 slides on the course site and [Spring 2024 videos](https://www.youtube.com/playlist?list=PLoROMvodv4rN4wG6Nk6sNpTEbuOSosZdX); optional: in the 2024 videos, lecture 9 is a guest lecture on DPO by its authors, and lectures 5–7, "Policy Search", derive policy gradients and PPO, the theory that Chapter 19 covers).
- Carnegie Mellon — [11-667 Large Language Models: Methods and Applications (Fall 2025)](https://2025.cmu-llms.org/) by Jaromir Savelka and Yubin Kim (free slides, no videos; optional: the lectures on task-oriented fine-tuning and on RLHF place this chapter and Chapter 17 in the context of the full LLM pipeline).

### Online courses (MOOCs)

- Nathan Lambert — [The Post-Training Course](https://rlhfbook.com/course) (free; [YouTube lectures](https://www.youtube.com/playlist?list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y) with slides that follow *Reinforcement Learning from Human Feedback* chapter by chapter; start here: lectures 1–4 on the overview, instruction tuning and reward models, and RL, then lecture 6 on DPO and lectures 8–9 on preference data and over-optimization).
- Hugging Face — [smol course](https://huggingface.co/learn/smol-course/unit0/1) (free; v2, 2025; self-paced, based on SmolLM3 and the Hugging Face Transformer Reinforcement Learning (TRL) library; unit 3, "Preference Alignment", trains a small model with DPO and prepares for the milestone).

### Books

- Book: Nathan Lambert, *Reinforcement Learning from Human Feedback* (Manning, 2026) — [official page](https://rlhfbook.com/) (free online, paid print; start here: the main text of this chapter; read chapters 3–6 and 8, which cover the training overview, instruction tuning, reward modeling, RL, and direct alignment algorithms, then chapters 11 and 14 on preference data and over-optimization).

### Lectures, papers and articles

- Article: Nathan Lambert, Louis Castricato, Leandro von Werra, and Alex Havrilla, [Illustrating Reinforcement Learning from Human Feedback (RLHF)](https://huggingface.co/blog/rlhf) (2022, Hugging Face blog; start here: a short description of the three RLHF stages and the KL penalty).
- Lecture: Andrej Karpathy, [State of GPT](https://www.youtube.com/watch?v=bZQun8Y4L2A) (2023, Microsoft Build; 40 minutes; the full assistant pipeline from pretraining through SFT, reward modeling, and RLHF, and what each stage contributes).
- Paper: Long Ouyang et al., [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (2022, InstructGPT; the three-stage recipe of SFT, reward modeling, and PPO on which ChatGPT was based; read sections 1–3).
- Paper: Rafael Rafailov et al., [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) (2023; rewrites the KL-constrained RLHF objective as a classification loss on preference pairs, with no separate reward model and no RL loop).
- Paper: Yuntao Bai et al., [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862) (2022, Anthropic; released the HH-RLHF preference dataset and measured the trade-off between helpfulness and harmlessness).
- Paper: Yuntao Bai et al., [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) (2022, Anthropic; replaces human harmlessness labels with AI feedback guided by written principles; the paper that introduced RLAIF).
- Paper: Nathan Lambert et al., [Tülu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124) (2024, Ai2; an open recipe of SFT, DPO, and RL with verifiable rewards (RLVR, Chapter 19), published with data, code, and evaluations).
- Article: Lilian Weng, [Reward Hacking in Reinforcement Learning](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) (2024; how policies exploit imperfect reward models, with a long section on reward hacking in the RLHF of language models).
- Paper: John Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (2017; advanced: the clipped policy-gradient algorithm used in the original RLHF recipes; assumes the RL foundations of Chapter 19).

### Tools and hands-on

- Hugging Face — [TRL](https://huggingface.co/docs/trl/index) documentation (free, open source; start here: the [`DPOTrainer`](https://huggingface.co/docs/trl/dpo_trainer) page documents the trainer used in the milestone, including the logged reward accuracy and reward margins; the [`RewardTrainer`](https://huggingface.co/docs/trl/reward_trainer) page documents the training of a Bradley-Terry reward model on the same pairs).

**Milestone.** Start from a small SFT model — the model from the Chapter 17 milestone or SmolLM2-360M-Instruct; with low-rank adaptation (LoRA) the run fits a free Colab or Kaggle T4 graphics processing unit (GPU) — and run DPO with TRL on a few thousand pairs from an open preference dataset such as UltraFeedback, once with a small value of the DPO coefficient beta and once with a large one. Report the preference accuracy on held-out pairs (the fraction of pairs for which the tuned model assigns the chosen answer a higher implicit reward than the rejected one) and a blind comparison by hand of the SFT and DPO answers to 30 held-out prompts. For each value of beta, also estimate the KL divergence from the reference model (the mean log-probability ratio between the two models on answers sampled from the tuned one), and explain how beta trades the gain in preference accuracy against drift from the reference model.

**Time.** 5–7 weeks.

## Chapter 19 — Reasoning Models and Reinforcement Learning

**Goals.** Reasoning models such as OpenAI o1 and DeepSeek-R1 solve difficult mathematics and coding problems by generating a long chain of thought before the final answer, and this chapter covers how such models are built. It begins with the methods that require no training: prompting a large language model (LLM) for step-by-step answers, and inference-time scaling, which spends more compute per question by sampling many answers and selecting one by majority vote or with a verifier. It then develops the reinforcement learning (RL) foundations that Chapter 18 used at a conceptual level — Markov decision processes (MDPs), policy gradients, and proximal policy optimization (PPO) — and applies them to language models with rewards that a program can verify, such as a correct final answer or passing unit tests, using group relative policy optimization (GRPO), the algorithm used to train DeepSeek-R1. It also covers the distillation of a large reasoning model into a small one. The milestone trains a small model with RL on a single graphics processing unit (GPU) and measures the change in its reasoning accuracy.

**Topics.** Chain-of-thought prompting and the effect of intermediate steps on accuracy; evaluating reasoning: answer extraction, exact-match checkers, and pass@k (the probability that at least one of k sampled answers is correct); inference-time scaling (self-consistency, best-of-N sampling, verifier-guided search); outcome and process reward models as verifiers; RL foundations: MDPs, returns, value functions, and advantages; policy gradients: REINFORCE, baselines, actor-critic, and the clipped objective of PPO; RL with verifiable rewards (RLVR) and its differences from reinforcement learning from human feedback (RLHF, Chapter 18); reward design for math and code (answer and format rewards, unit tests) and reward hacking; GRPO: group-relative advantages without a value network; reasoning models: DeepSeek-R1-Zero, DeepSeek-R1, and o1-style models; distillation of reasoning traces into small models.

**Resources.**

**Suggested path.** Read "Understanding Reasoning LLMs" by Sebastian Raschka for an overview, then divide each week between two tracks. The first track is RL foundations: the three-part introduction of Spinning Up, chapters 3 and 13 of Sutton and Barto, and the policy-gradient lectures of CS 285 (or David Silver's course, which proceeds at a slower pace). The second track is reasoning models: *Build a Reasoning Model (From Scratch)* is the main text of this chapter; read the chain-of-thought and self-consistency papers alongside its chapters 4–5, and the DeepSeekMath and DeepSeek-R1 papers, CS336 lecture 16, and chapter 12 of the Hugging Face LLM Course alongside its chapters 6–7. When time is short, skip David Silver's course, the remaining CS 285 lectures, and the process-reward and test-time-compute papers, and complete the milestone with the Hugging Face Transformer Reinforcement Learning (TRL) library or Unsloth instead of a hand-written training loop.

### University courses

- Stanford — [CS336 Language Modeling from Scratch (Spring 2026)](https://cs336.stanford.edu/) by Tatsunori Hashimoto and Percy Liang (free; [videos](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV) + assignments; start here: lecture 16, "Post-training - RLVR", on policy gradients for language models, GRPO, and the DeepSeek-R1 recipe; advanced: [assignment 5](https://github.com/stanford-cs336/assignment5-alignment) implements supervised fine-tuning (SFT), expert iteration, and GRPO for math reasoning and is sized for a data-center GPU).
- UC Berkeley — [CS 185/285 Deep Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse/) by Sergey Levine (free; Spring 2026 slides and homework on the course site and [Fall 2023 videos](https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps), whose numbering differs; the relevant lectures in the 2026 schedule are 4–6 on RL basics, policy gradients, and actor-critic, 9–10 on advanced policy gradients, and 14, "LLM RL").
- UCL and DeepMind — [Reinforcement Learning course (2015)](https://www.davidsilver.uk/teaching/) by David Silver (free; [videos](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ) + slides; optional: a ten-lecture introduction at a slower pace than CS 285; the relevant lectures are 1–3 on MDPs and value functions and 7 on policy gradients).

### Online courses (MOOCs)

- Hugging Face — [LLM Course, Chapter 12: Open R1 for Students](https://huggingface.co/learn/llm-course/chapter12/1) (free; a few hours; start here: covers RL for language models, a reading of the DeepSeek-R1 paper, GRPO in TRL, and a GRPO training exercise).
- OpenAI — [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/) (free; 2018; a concise introduction to RL terminology and policy optimization, with reference implementations of vanilla policy gradient and PPO; read "Introduction to RL" parts 1–3).

### Books

- Book: Sebastian Raschka, *Build a Reasoning Model (From Scratch)* (Manning, 2026) — [official hub](https://sebastianraschka.com/reasoning-from-scratch/) (paid; free code in [the GitHub repo](https://github.com/rasbt/reasoning-from-scratch); start here: chapter 3 on evaluating with verifiers, chapters 4–5 on inference-time scaling, chapters 6–7 on RLVR and GRPO, and chapter 8 on distillation, all implemented in PyTorch on a small Qwen3 model).
- Book: Richard S. Sutton and Andrew G. Barto, *Reinforcement Learning: An Introduction* (2nd ed., MIT Press, 2018) — [official page](http://incompleteideas.net/book/the-book-2nd.html) (free PDF; the standard RL textbook; chapter 3, "Finite Markov Decision Processes", and chapter 13, "Policy Gradient Methods", provide the terminology and the mathematics underlying PPO and GRPO).
- Book: Nathan Lambert, *Reinforcement Learning from Human Feedback* (Manning, 2026) — [official page](https://rlhfbook.com/) (free online, paid print; chapter 6, "Reinforcement Learning", derives the policy-gradient algorithms from REINFORCE to PPO and GRPO as they are applied to language models, and chapter 7, "Reasoning and Inference-Time Scaling", surveys RLVR and the reasoning-model recipes; lecture 5 of the companion [Post-Training Course](https://rlhfbook.com/course) covers chapter 7).

### Lectures, papers and articles

- Article: Sebastian Raschka, [Understanding Reasoning LLMs](https://magazine.sebastianraschka.com/p/understanding-reasoning-llms) (2025; start here: describes four approaches to building a reasoning model — inference-time scaling, pure RL, SFT followed by RL, and distillation — with DeepSeek-R1 as the case study).
- Paper: Jason Wei et al., [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) (2022; shows that prompting a large model to write out intermediate steps improves its accuracy on multi-step arithmetic, commonsense, and symbolic problems).
- Paper: Xuezhi Wang et al., [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171) (2022; samples many chains of thought and takes a majority vote over the final answers; a basic form of inference-time scaling).
- Paper: Zhihong Shao et al., [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) (2024; introduces GRPO, a PPO variant that omits the value network and computes advantages relative to a group of sampled answers; read section 4).
- Paper: DeepSeek-AI, [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) (2025; DeepSeek-R1-Zero acquires long chains of thought from RL alone with rule-based rewards; DeepSeek-R1 adds a cold-start SFT stage and is distilled into smaller models).
- Paper: Hunter Lightman et al., [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) (2023, OpenAI; process reward models, which grade each reasoning step, select correct solutions to problems of the MATH benchmark more often than outcome reward models).
- Paper: Charlie Snell et al., [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314) (2024; advanced: compares the allocation of inference compute between verifier-guided search and sequential revisions as a function of question difficulty).

### Tools and hands-on

- Hugging Face — [TRL](https://huggingface.co/docs/trl/index) documentation (free, open source; start here: the [`GRPOTrainer`](https://huggingface.co/docs/trl/grpo_trainer) page documents a widely used open-source GRPO implementation, its custom reward functions, and the logged reward and completion length that the milestone plots; optional: the [Unsloth](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide) RL guide wraps the same trainer, explains reward-function design, and links free Colab notebooks that train models of up to 1.5B parameters with low-rank adaptation (LoRA) in about 5 GB of GPU memory).

**Milestone.** Train a small instruction-tuned model (Qwen2.5-0.5B-Instruct or Qwen3-0.6B; with LoRA the run fits a free Colab or Kaggle T4 GPU, and a 1.5B-parameter model fits a 24 GB GPU) with GRPO — the TRL `GRPOTrainer` or a hand-written loop based on chapters 6–7 of *Build a Reasoning Model (From Scratch)* — on the training set of GSM8K, a dataset of grade-school math word problems, with a rule-based reward for the answer format and for an exact-match final answer. Plot the mean reward, the mean completion length, and the accuracy on a fixed 200-question slice of the GSM8K test set against training steps. Compare the final accuracy with that of the model before RL training under greedy decoding and under self-consistency with 8 samples, and state which of the two gains required more compute.

**Time.** 5–7 weeks.

---

# Part VI — Evaluation, Deployment, and the Open-Source Ecosystem

## Chapter 20 — Evaluation and Benchmarks

**Goals.** This chapter covers the methods used to measure the quality of a language model and to compare two models. It describes intrinsic metrics (perplexity and bits per byte), the standard benchmarks and the two ways in which they are scored (log-likelihood of answer choices and scoring of generated text), human-preference rankings, and the use of a strong model as a judge. It also covers the sources of error in reported scores — prompt format, few-shot settings, sampling noise, and benchmark contamination — and the statistics needed to state whether a difference between two models is significant. Later chapters apply these methods: Chapter 21 measures the accuracy cost of quantization, Chapter 22 measures the effect of fine-tuning and retrieval, and the Chapter 23 capstone requires an evaluation report.

**Topics.** The purposes of evaluation: model selection, regression testing, and reporting; perplexity and bits per byte as intrinsic metrics; standard benchmarks by capability: knowledge (MMLU), commonsense and science reasoning (HellaSwag, ARC), mathematics (GSM8K), code (HumanEval), graduate-level science (GPQA), and instruction following (IFEval); scoring methods: log-likelihood of answer choices, exact match on generated answers, and unit tests with pass@k for code; evaluation settings that change scores: few-shot examples, prompt and chat templates, answer normalization, and decoding parameters; statistical uncertainty: standard errors, confidence intervals, and paired comparisons; benchmark contamination and saturation; human evaluation and preference arenas: pairwise votes and Elo-style (Bradley–Terry) ratings; use of a large language model (LLM) as a judge, and its position, verbosity, and self-preference biases; agentic, multi-step benchmarks (SWE-bench); task-specific evaluation: building a test set and a custom harness task; reproducible reporting of evaluation results.

**Resources.**

**Suggested path.** Start with the LLM Evaluation Guidebook and Unit 2 of the smol course, and run lm-evaluation-harness on a small model during the first week; Biderman et al. explains the settings that change its scores. Read chapters 3 and 4 of *AI Engineering* and watch CS336 lecture 12 in parallel. Then read Miller for the statistics, Zheng et al. and Chiang et al. for judges and arenas, and Oren et al. for contamination. When time is short, read the benchmark papers for the task format only, and omit HELM, Inspect, and the model reports.

### University courses

- Stanford — [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) by Percy Liang and Tatsunori Hashimoto (free; Spring 2026 lecture videos on a [YouTube playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV) and public assignments; start here: lecture 12, "Evaluation", covers perplexity, knowledge, instruction-following, and agent benchmarks, and validity problems such as contamination).

### Online courses (MOOCs)

- Hugging Face — [smol course, Unit 2: Evaluation](https://huggingface.co/learn/smol-course/unit2/1) (free; v2, 2025; about one week; start here: running standard benchmarks with lighteval and building a custom evaluation for a specific domain).

### Books

- Book: Chip Huyen, *AI Engineering* (O'Reilly, 2025) — [book repo](https://github.com/chiphuyen/aie-book) (paid book, free chapter summaries; start here: chapter 3, "Evaluation Methodology", on perplexity, exact and similarity-based scoring, and AI judges; chapter 4, "Evaluate AI Systems", on evaluation criteria, model selection, and the design of an evaluation pipeline).

### Lectures, papers and articles

- Article: Clémentine Fourrier et al., [The LLM Evaluation Guidebook](https://huggingface.co/spaces/OpenEvals/evaluation-guidebook) (Hugging Face; free; start here: automatic benchmarks, human evaluation, LLM-as-a-judge, and a troubleshooting section on reproducibility, by the team that maintained the Open LLM Leaderboard).
- Paper: Biderman et al., [Lessons from the Trenches on Reproducible Evaluation of Language Models](https://arxiv.org/abs/2405.14782) (2024; the companion paper of lm-evaluation-harness; the reasons the same benchmark yields different numbers across implementations, and recommendations for reporting results).
- Paper: Evan Miller, [Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations](https://arxiv.org/abs/2411.00640) (2024; standard errors, clustered questions, paired differences, and sample-size calculation, using the statistics of Chapter 4).
- Papers: the original benchmark papers — Hendrycks et al., [MMLU](https://arxiv.org/abs/2009.03300) (2020); Clark et al., [ARC](https://arxiv.org/abs/1803.05457) (2018); Zellers et al., [HellaSwag](https://arxiv.org/abs/1905.07830) (2019); Cobbe et al., [GSM8K](https://arxiv.org/abs/2110.14168) (2021); Chen et al., [HumanEval](https://arxiv.org/abs/2107.03374) (2021; introduced in the Codex paper together with pass@k); Rein et al., [GPQA](https://arxiv.org/abs/2311.12022) (2023); Zhou et al., [IFEval](https://arxiv.org/abs/2311.07911) (2023); and Jimenez et al., [SWE-bench](https://arxiv.org/abs/2310.06770) (2023; tasks built from real GitHub issues; the design followed by later agent benchmarks). Read each for the task format and the scoring rule.
- Paper: Zheng et al., [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) (2023; measures the agreement between a strong model and human judges, and documents position, verbosity, and self-enhancement bias).
- Paper: Chiang et al., [Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://arxiv.org/abs/2403.04132) (2024; the pairwise-comparison statistics behind the public leaderboard now published at [Arena](https://arena.ai/), formerly LMArena).
- Paper: Oren et al., [Proving Test Set Contamination in Black Box Language Models](https://arxiv.org/abs/2310.17623) (2023; a statistical test that detects whether a benchmark was present in the training data, using only model outputs).
- Paper: Liang et al., [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) (2022; optional; the case for reporting several metrics — accuracy, calibration, robustness, fairness, and efficiency — over many scenarios; the [HELM framework and leaderboards](https://github.com/stanford-crfm/helm) have been in maintenance mode since June 2026).
- Article: Hamel Husain, [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/) (2024; evaluation of an application at three levels: unit tests, human and model review of logged outputs, and A/B tests; used again in Chapter 22).
- Papers: [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (2024) and [DeepSeek-R1](https://arxiv.org/abs/2501.12948) (2025) (optional; read only the evaluation sections, as examples of how model developers report benchmark settings and baselines).

### Tools and hands-on

- Tool: EleutherAI, [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (free, open source; start here: the harness used by most open-model reports and by this chapter's milestone; tasks are defined in YAML files, and results include standard errors).
- Tool: UK AI Security Institute, [Inspect](https://inspect.aisi.org.uk/) (free, open source; optional; a framework built from datasets, solvers, and scorers, suited to model-graded and multi-step agent evaluations).

**Milestone.** On a free Colab or Kaggle T4 GPU, run lm-evaluation-harness on two open base models of 1–2 billion parameters (for example Qwen2.5-1.5B and SmolLM2-1.7B) on ARC-Easy, HellaSwag, and a 250-question subset of GSM8K, and compare each score and its standard error with the value published by the model's authors. Add one custom task of at least 50 questions, defined in a harness YAML task file. Write a two-page report that states the evaluation settings (few-shot count, prompt format, answer normalization) and explains one score difference, either between the two models or between a measured and a published value.

**Time.** 3–4 weeks.

## Chapter 21 — Inference Optimization and Serving

**Goals.** This chapter covers the methods used to run a trained open large language model (LLM) at low cost and low latency, and the software that serves it to users. It describes the two phases of text generation — prefill, which processes the prompt in parallel and is limited by compute, and decode, which produces one token at a time and is limited by memory bandwidth — and the key-value (KV) cache, which avoids recomputing attention over earlier tokens at the cost of memory that grows with batch size and context length. It then presents the main optimizations: continuous batching, paged attention, FlashAttention, quantization of the weights to 8 or 4 bits, and speculative decoding, each with its effect on memory, speed, and accuracy. It introduces the serving engines vLLM, SGLang, and llama.cpp, and the metrics used to benchmark them: time to first token (TTFT), time per output token (TPOT), and throughput. The accuracy cost of quantization is measured with the methods of Chapter 20, and the serving setup is used again in Chapters 22 and 23.

**Topics.** The two phases of generation: prefill (compute-bound) and decode (memory-bound), in the workload classes of Chapter 7; inference arithmetic: memory for the weights and the KV cache, floating-point operations (FLOPs) per token, and the memory-bandwidth limit on decode speed; the KV cache and the methods that reduce its size: multi-query and grouped-query attention (Chapter 12), and KV-cache quantization; serving metrics: time to first token (TTFT), time per output token (TPOT), throughput in tokens per second, and the trade-off between latency and throughput; batching of requests: static, dynamic, and continuous batching; paged attention and prefix caching; FlashAttention (Chapter 16) in prefill and decode; post-training quantization of weights: LLM.int8() and 4-bit NormalFloat (NF4) in bitsandbytes, GPTQ, AWQ, and the GGUF formats of llama.cpp; quantization of activations and the outlier problem (SmoothQuant); measurement of the accuracy cost of quantization with perplexity and benchmark scores (Chapter 20); speculative decoding with a draft model; serving engines: vLLM, SGLang, llama.cpp, and the OpenAI-compatible HTTP interface; inference on several GPUs: tensor parallelism (Chapter 16) and disaggregated prefill and decode (overview).

**Resources.**

**Suggested path.** Start with the DeepLearning.AI short course Efficiently Serving LLMs and CS336 lecture 10 for the vocabulary; then read chapter 9 of *AI Engineering* and kipply's article for the inference arithmetic, and implement a KV cache with Raschka's article. Install llama.cpp and vLLM during the first week and run a small model on each, because the milestone depends on them. Then study quantization (the two DeepLearning.AI quantization courses, MIT 6.5940 lectures 5–6, and the GPTQ and AWQ papers), followed by the PagedAttention, FlashAttention, and speculative decoding papers. When time is short, read only the abstract and the method section of each paper, and omit Pope et al., CMU 11-868, and SGLang.

### University courses

- Stanford — [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) by Percy Liang and Tatsunori Hashimoto (free; Spring 2026 lecture videos on a [YouTube playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV) + public assignments; start here: lecture 10, "Inference", covers the prefill and decode workloads, the arithmetic of the KV cache, architectural and quantization methods that reduce it, speculative decoding, and continuous batching with paged attention).
- MIT — [6.5940: TinyML and Efficient Deep Learning Computing](https://hanlab.mit.edu/courses/2024-fall-65940) by Song Han (free; Fall 2024 slides, labs, and a public [lecture playlist](https://www.youtube.com/playlist?list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB); lectures 5–6 cover quantization, lecture 13 efficient LLM deployment, and lecture 15 long-context LLMs; the labs include compression of an LLM and deployment of Llama-2-7B on a laptop).
- CMU — [11-868: Large Language Model Systems](https://llmsystem.github.io/llmsystem2026spring/) by Lei Li (free; Spring 2026 slides; optional; lectures on paged attention and vLLM, KV-cache reuse, quantization, speculative decoding, and disaggregated prefill and decode).

### Online courses (MOOCs)

- DeepLearning.AI — [Efficiently Serving LLMs](https://www.deeplearning.ai/courses/efficiently-serving-llms) by Travis Addair of Predibase (free; 2 hours; start here: KV caching, continuous batching, quantization, and the serving of many low-rank adaptation (LoRA) adapters (Chapter 17) on one base model, each implemented in code).
- DeepLearning.AI — [Quantization Fundamentals with Hugging Face](https://www.deeplearning.ai/courses/quantization-fundamentals) by Younes Belkada and Marc Sun (free; 1 hour 15 minutes; data types, downcasting, and loading of quantized models with `transformers` and Quanto), followed by [Quantization in Depth](https://www.deeplearning.ai/courses/quantization-in-depth) by the same instructors (free; 2 hours 20 minutes; optional: implementation of a linear quantizer down to 2-bit weights, with measurement of the quantization error).

### Books

- Book: Chip Huyen, *AI Engineering* (O'Reilly, 2025) — [book repo](https://github.com/chiphuyen/aie-book) (paid book, free chapter summaries; start here: chapter 9, "Inference Optimization", covers the bottlenecks of inference, the performance metrics, and optimization at the model level and at the service level; its "AI Accelerators" section was assigned in Chapter 7).

### Lectures, papers and articles

- Article: kipply, [Transformer Inference Arithmetic](https://kipp.ly/p/transformer-inference-arithmetic) (2022; start here: derives the size of the KV cache, the memory-bound and compute-bound regimes, and latency estimates from the parameter count, the FLOPs, and the memory bandwidth of the accelerator).
- Article: Sebastian Raschka, [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) (2025; adds a KV cache to the GPT model of Chapter 14; reports a speedup of about 5x on a 124-million-parameter model and states the memory cost).
- Article: Patrick von Platen, [Optimizing your LLM in Production](https://huggingface.co/blog/optimize-llm) (Hugging Face, 2023; measured GPU memory and latency for 8-bit and 4-bit weights, FlashAttention, and multi-query and grouped-query attention).
- Paper: Pope et al., [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) (2022; advanced: a cost model of memory-bound decoding, the partitioning of a model over many accelerators, and the trade-off between latency and throughput).
- Papers: Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (2022) and Tri Dao, [FlashAttention-2](https://arxiv.org/abs/2307.08691) (2023; exact attention computed in blocks that fit in on-chip memory, which reduces memory traffic without approximation; introduced in Chapter 16).
- Papers: post-training quantization — Dettmers et al., [LLM.int8()](https://arxiv.org/abs/2208.07339) (2022; the 8-bit method of bitsandbytes); Frantar et al., [GPTQ](https://arxiv.org/abs/2210.17323) (2022); Lin et al., [AWQ](https://arxiv.org/abs/2306.00978) (2023); and Xiao et al., [SmoothQuant](https://arxiv.org/abs/2211.10438) (2022; the activation outliers that make 8-bit activations difficult, and a rescaling that transfers the difficulty to the weights).
- Papers: speculative decoding — Leviathan et al., [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (2022) and Chen et al., [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) (2023; two independent derivations of one method: a small draft model proposes several tokens, the large model verifies them in one pass, and the output distribution is proved unchanged).

### Tools and hands-on

- Tool: ggml-org, [llama.cpp](https://github.com/ggml-org/llama.cpp) (free, open source; start here: CPU and GPU inference in C/C++ with GGUF quantization from 8 bits to about 2 bits; provides `llama-server`, `llama-bench`, and `llama-perplexity`, which the milestone uses; [Ollama](https://ollama.com/) packages the same engine with a model downloader).
- Tool: [vLLM](https://docs.vllm.ai/en/latest/) (free, open source; a GPU serving engine with continuous batching, prefix caching, quantized models, and an OpenAI-compatible server; used again in Chapter 22), with its paper, Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (2023; KV-cache memory allocated in fixed-size blocks, as in the virtual memory of an operating system, which reduces fragmentation and permits larger batches).
- Tool: [SGLang](https://docs.sglang.io/) (free, open source; optional: a GPU serving engine with RadixAttention for prefix reuse, described in Zheng et al., [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104), 2023). NVIDIA [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) (advanced) runs on NVIDIA GPUs only, and Hugging Face [TGI](https://huggingface.co/docs/text-generation-inference) is in maintenance mode; its documentation refers users to vLLM and SGLang.

**Milestone.** Run an open model of 7–8 billion parameters with llama.cpp on a single machine at three GGUF quantization levels, for example Q8_0, Q4_K_M, and Q2_K; a free Colab or Kaggle T4 GPU is sufficient, and a 3-billion-parameter model can be substituted on a machine with less than 16 GB of memory. For each level, measure prefill and decode speed in tokens per second with `llama-bench` at prompt lengths of 512 and 4,096 tokens, measure perplexity on the WikiText-2 test set with `llama-perplexity`, and plot perplexity against decode speed. Compare the measured decode speed with the estimate of memory bandwidth divided by model size in bytes, then start `llama-server`, send one request through its OpenAI-compatible interface, and record the time to first token.

**Time.** 4–5 weeks.

## Chapter 22 — Operating and Improving an Open-Source LLM

**Goals.** This chapter combines the methods of Chapters 17–21 into the operating cycle of an application built on an open large language model (LLM): select a base model, adapt it to a task, serve it, evaluate it, and improve it from observed failures. It describes the open-model ecosystem — the main model families, the Hugging Face Hub, model cards, and licenses — and the difference between a model with open weights and a fully open model, for which the training data and code are also published. It presents the adaptation methods in order of increasing cost: prompt design; retrieval-augmented generation (RAG), which inserts retrieved documents into the prompt at inference time; fine-tuning; and continued pretraining. It introduces agents, in which the model calls external tools over several steps. It also covers the operations work: versioning, regression evaluations in continuous integration (CI), guardrails, monitoring, and the conversion of logged failures into evaluation cases and training data. The Chapter 23 capstone applies the same cycle to a model trained from scratch.

**Topics.** The open-model ecosystem: the Llama, Mistral, Qwen, DeepSeek, Gemma, and OLMo families; degrees of openness: open weights, open training data, and open training code, and the Open Source AI Definition; model hubs (Hugging Face Hub), model cards, and licenses; selection of a base model: size, license, context length, benchmark results (Chapter 20), and serving cost (Chapter 21); choice of an adaptation method: prompting, retrieval, fine-tuning (Chapter 17), or continued pretraining; prompt design: system prompts, few-shot examples, and structured output; retrieval-augmented generation (RAG): chunking, text-embedding models, vector search, hybrid search, and reranking; evaluation of a RAG system: retrieval metrics and the faithfulness of the answer to the retrieved text; agents: tool calling, the distinction between workflows and agents, and the evaluation of multi-step tasks; guardrails: validation of inputs and outputs, and prompt injection; versioning of data, prompts, adapters, and checkpoints, with regression evaluations in continuous integration (CI); monitoring in production: logging and tracing, latency and cost, output quality, and distribution shift; the improvement cycle: user feedback, error analysis, and the conversion of failures into evaluation cases and training data.

**Resources.**

**Suggested path.** Start with Yan's "Patterns" article and the LLM Bootcamp for an overview, then read chapters 5, 6, and 10 of *AI Engineering* as the main text. Select the task and the base model of the milestone during the first two weeks, using the Hub documentation and the Open Source AI Definition to check the license and the degree of openness; then build the retrieval path with the Open-Source AI Cookbook and chapter 8 of *Hands-On Large Language Models*. Follow the AI Agents Course in parallel, and read Husain, the Yan et al. report, and the monitoring chapters of *Designing Machine Learning Systems* while the system is evaluated and improved; the CS329S lecture notes are a free substitute for that book. When time is short, omit the DeepLearning.AI RAG course, Made With ML, and the survey by Gao et al.

### University courses

- Stanford — [CS329S: Machine Learning Systems Design](https://stanford-cs329s.github.io/) by Chip Huyen (free; Winter 2022 slides, lecture notes, and assignments; start here: the notes on deployment, on data distribution shifts and monitoring, and on continual learning; the course predates LLM applications and treats these subjects for machine learning systems in general; *Designing Machine Learning Systems*, listed under Books, is based on it).

### Online courses (MOOCs)

- Full Stack Deep Learning — [LLM Bootcamp](https://fullstackdeeplearning.com/llm-bootcamp/) by Charles Frye, Sergey Karayev, and Josh Tobin (free; recorded in Spring 2023; start here: lectures on prompt engineering, LLMOps, augmented language models, and user interfaces for language models, and a walkthrough of a deployed question-answering application; the tools shown are those of 2023).
- Hugging Face — [AI Agents Course](https://huggingface.co/learn/agents-course) (free; 2025; agents built with smolagents, LlamaIndex, and LangGraph; bonus units on fine-tuning a model for function calling and on the observability and evaluation of agents).
- DeepLearning.AI — [Retrieval Augmented Generation (RAG)](https://www.deeplearning.ai/courses/retrieval-augmented-generation) by Zain Hasan (free to audit, paid certificate; about 26 hours; optional: keyword, semantic, and hybrid search, vector databases, and the evaluation and production monitoring of a RAG system).
- Goku Mohandas — [Made With ML](https://madewithml.com/) (free; optional: a course in machine learning operations (MLOps) that covers the testing of code, data, and models, continuous integration and deployment, serving, and monitoring, built around the fine-tuning of an LLM for a text-classification task).

### Books

- Book: Chip Huyen, *AI Engineering* (O'Reilly, 2025) — [book repo](https://github.com/chiphuyen/aie-book) (paid book, free chapter summaries; start here: chapter 5, "Prompt Engineering"; chapter 6, "RAG and Agents"; and chapter 10, "AI Engineering Architecture and User Feedback"; chapter 7, "Finetuning", and chapter 8, "Dataset Engineering", were assigned in Chapter 17).
- Book: Chip Huyen, *Designing Machine Learning Systems* (O'Reilly, 2022) — [book repo](https://github.com/chiphuyen/dmls-book) (paid book, free chapter summaries and a list of MLOps tools; chapter 8, "Data Distribution Shifts and Monitoring", and chapter 9, "Continual Learning and Test in Production", cover subjects that *AI Engineering* treats only briefly).
- Book: Jay Alammar and Maarten Grootendorst, *Hands-On Large Language Models* (O'Reilly, 2024) — [official page](https://llm-book.com/) (paid; free notebooks on [GitHub](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models); chapter 6, "Prompt Engineering"; chapter 7, "Advanced Text Generation Techniques and Tools"; and chapter 8, "Semantic Search and Retrieval-Augmented Generation", each with runnable code).

### Lectures, papers and articles

- Article: Eugene Yan, [Patterns for Building LLM-based Systems & Products](https://eugeneyan.com/writing/llm-patterns/) (2023; start here: seven patterns — evaluations, RAG, fine-tuning, caching, guardrails, defensive user-interface design, and collection of user feedback — each with the conditions under which it applies).
- Article: Eugene Yan, Bryan Bischof, Charles Frye, Hamel Husain, Jason Liu, and Shreya Shankar, [What We've Learned From a Year of Building with LLMs](https://applied-llms.org/) (2024; practices reported by six practitioners, grouped as tactical — prompting, RAG, and evaluation — operational, and strategic).
- Article: Hamel Husain, [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/) (2024; introduced in Chapter 20; describes the improvement cycle of this chapter: logged traces are reviewed, failures become test cases, and the tests run on every change).
- Papers: Lewis et al., [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) (2020; the paper that introduced the term, with a retriever and a generator trained jointly) and Gao et al., [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) (2023; the retrieval, augmentation, and generation methods of later systems, and the metrics used to evaluate them).
- Article: Erik Schluntz and Barry Zhang, [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) (Anthropic, 2024; defines workflows and agents, describes five workflow patterns, and states the conditions under which an agent is justified), with the paper Yao et al., [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) (2022; the loop of reasoning step, tool call, and observation that agent frameworks implement).
- Article: Open Source Initiative, [The Open Source AI Definition 1.0](https://opensource.org/ai/open-source-ai-definition) (2024; the requirements on data information, code, and parameters for a model to be called open source), with an example of a fully open model: Ai2, [OLMo](https://allenai.org/olmo) (weights, training data, code, intermediate checkpoints, and evaluation tools are published), described in Groeneveld et al., [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838) (2024).

### Tools and hands-on

- Hands-on: Hugging Face, [Open-Source AI Cookbook](https://huggingface.co/learn/cookbook) (free; start here: notebooks on RAG with open models, RAG evaluation, LLM-as-a-judge, agents, and deployment; the RAG notebooks serve as the basis of the retrieval path of the milestone).
- Docs: Hugging Face, [Hub documentation](https://huggingface.co/docs/hub/index) (free; repositories, revisions, and Spaces), with its [model cards guide](https://huggingface.co/docs/hub/model-cards) and the paper that proposed model cards, Mitchell et al., [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) (2018; the intended use, evaluation data, and limitations that the publisher of a model documents).

**Milestone.** Build a question-answering system over a document collection of at least 100 pages, on an open instruction-tuned model of 1–3 billion parameters (for example Qwen2.5-1.5B-Instruct) that fits a free Colab or Kaggle T4 GPU. Write a held-out test set of at least 100 questions with reference answers, and measure four configurations — the base model, the base model with retrieval, a low-rank adaptation (LoRA) fine-tune (Chapter 17), and the fine-tune with retrieval — with the model served by vLLM (Chapter 21) and scored by one method of Chapter 20, either a custom lm-evaluation-harness task or a judge model. Publish the adapter on the Hugging Face Hub with a model card, and report the score, its standard error, and the mean latency of each configuration, together with an error analysis of 20 failed answers and the change that each failure suggests.

**Time.** 6–8 weeks.

## Chapter 23 — Capstone: Train, Ship, and Improve Your Own Model

**Goals.** This chapter is a capstone project: a small chat model is trained, evaluated, served, and published through the same sequence of stages that produces a released open large language model (LLM), within a fixed budget of about $15 to $100 of rented graphics processing unit (GPU) time. The project applies the earlier chapters in order: tokenizer training (Chapter 13), pretraining (Chapters 14–16), supervised fine-tuning (SFT, Chapter 17), an optional stage of Direct Preference Optimization (DPO) or reinforcement learning (RL) (Chapters 18 and 19), evaluation (Chapter 20), serving (Chapter 21), and publication with a model card (Chapter 22). It adds the project-level work that the single-topic chapters do not require: a written plan with a compute and cost estimate, a rehearsal run at reduced scale, experiment tracking, a cost record, and one controlled change to the pipeline whose effect is measured by re-evaluation. The result is a public repository on the Hugging Face Hub that contains the checkpoints, a model card, and a technical write-up.

**Topics.** Project plan: model size, token budget, compute estimated in floating-point operations (FLOPs) from C ≈ 6ND, GPU-hours, and a cost ceiling (Chapter 15); choice of compute: a rented node of eight GPUs, a single rented GPU, and spot instances (Chapters 7 and 16); pipeline design: the stages, the artifact that each stage produces, and a rehearsal run at reduced scale before the full run; experiment tracking and reproducibility: configurations, random seeds, logs, and a record of cost; tokenizer training and measurement of its compression ratio (Chapter 13); pretraining: data download, monitoring of loss and throughput, checkpointing, and restarts (Chapters 14–16); mid-training (a short stage on curated or task-specific data between pretraining and SFT) and SFT with a chat template (Chapter 17); an optional stage of DPO or of RL with verifiable rewards (Chapters 18 and 19); evaluation of the base and chat checkpoints against a published baseline (Chapter 20); serving: an inference engine with a key-value cache, a chat interface, and measurement of generation throughput (Chapter 21); iteration: one controlled change to the pipeline, followed by re-evaluation (Chapter 22); publication: checkpoints, model card, license, and technical write-up (Chapters 14 and 22).

**Resources.**

**Suggested path.** Watch "Intro to Large Language Models" as a review of the whole pipeline, then read the nanochat README, its `runs/speedrun.sh` script, and the guide "Beating GPT-2 for <<$100", and write the project plan. Run the pipeline first at reduced scale — `runs/runcpu.sh` on a laptop, or a 12-layer model on one rented GPU — so that every stage and every evaluation script has completed once before the eight-GPU node is rented; then run the full pipeline, with the sections of *The Smol Training Playbook* on ablations and loss spikes as the reference for monitoring. In parallel with the runs, work through the smol course units on instruction tuning and preference alignment, and use the SmolLM3 article and the model cards guide as templates for the write-up. CS336 and *Build a Large Language Model (From Scratch)* support a project written from scratch in place of nanochat; when time is short, omit them together with "Deep Dive into LLMs like ChatGPT", the DataComp-LM paper, and modded-nanogpt.

### University courses

- Stanford — [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) by Tatsunori Hashimoto and Percy Liang (free; Spring 2026 [lecture videos](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV) and assignments; optional: the five assignments — basics, systems, scaling, data, and alignment — cover the same stages as this capstone with unit tests, and were assigned in Chapters 13–19; they form an alternative project when the model is written from scratch).

### Online courses (MOOCs)

- Hugging Face — [smol course](https://huggingface.co/learn/smol-course/unit0/1) (free; v2, 2025; self-paced, based on SmolLM3 and TRL; start here: the units on instruction tuning, preference alignment, and evaluation rehearse the post-training half of the capstone on a small model).
- Andrej Karpathy — [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) (free; 4 hours; step-by-step code in [build-nanogpt](https://github.com/karpathy/build-nanogpt); first assigned in Chapter 14; the final section, a complete pretraining run on 10B tokens of FineWeb-Edu with a HellaSwag evaluation, is the part relevant here).

### Books

- Book: Sebastian Raschka, *Build a Large Language Model (From Scratch)* (Manning, 2024) — [official page](https://www.manning.com/books/build-a-large-language-model-from-scratch) (paid; free code at [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch); optional: the book was completed in Chapters 12–17; the repository's bonus material on the key-value cache, DPO, and the Llama 3 and Qwen3 architectures is a reference for a project written from scratch).

### Lectures, papers and articles

- Video: Andrej Karpathy, [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) (2023; 1 hour; start here: an overview of pretraining, fine-tuning, and deployment, used as a review before the project plan is written), and [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) (2025; 3.5 hours; optional: the same pipeline from pretraining data to post-training and RL in more detail).
- Article: Loubna Ben Allal et al. (Hugging Face), [The Smol Training Playbook: The Secrets to Building World-Class LLMs](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) (2025; first read in Chapter 15; the sections on whether to train at all, ablations, data mixtures, loss spikes, and restarts apply directly to the project plan and to monitoring the run).
- Article: Hugging Face, [SmolLM3: smol, multilingual, long-context reasoner](https://huggingface.co/blog/smollm3) (2025; a published recipe with architecture, a three-stage data mixture, mid-training, post-training, and evaluation tables; a template for the structure of the write-up).
- Article: Sam Dobson, [I trained a ChatGPT clone for £75](https://samdobson.uk/posts/training-a-chatgpt-clone-for-cheap/) (2025; an independent account of the original nanochat run on a rented 8xH100 node — 4 hours, about $100 — with the resulting benchmark scores; the current speedrun is shorter).
- Paper: Li et al., [DataComp-LM: In search of the next generation of training sets for language models](https://arxiv.org/abs/2406.11794) (2024; optional: defines the CORE score, the mean centered accuracy over 22 tasks, which nanochat reports for its base model and compares with GPT-2).
- Paper: Mitchell et al., [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) (2018; the sections of a model card: intended use, training data, evaluation data, metrics, and limitations), with the Hugging Face [model cards guide](https://huggingface.co/docs/hub/model-cards) (the metadata format of a `README.md` on the Hub) and [Spaces documentation](https://huggingface.co/docs/hub/spaces) (hosting of a demonstration application).

### Tools and hands-on

- Code: Andrej Karpathy, [nanochat](https://github.com/karpathy/nanochat) (free, open source, MIT license; start here: one codebase for tokenizer training, pretraining, SFT, RL, evaluation, and inference with a key-value cache and a command-line chat; as of 2026 `runs/speedrun.sh` trains a model with the CORE score of GPT-2 in about 2 hours on an 8xH100 node, about $48 on demand or about $15 on spot instances; the same code runs on one GPU in about eight times the time; the guide [Beating GPT-2 for <<$100: the nanochat journey](https://github.com/karpathy/nanochat/discussions/481) documents the current recipe).
- Code: Andrej Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT) (free; pretraining only; the README points to nanochat as the successor; the training loop of about 300 lines, used in Chapters 14 and 16, is a smaller base for a project that adds the other stages by hand).
- Code: Keller Jordan et al., [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) (advanced; a public speedrun of GPT-2 (124M) training derived from Karpathy's reproduction; its record log lists each optimizer and architecture change with the measured training time, and nanochat adopts several of them).
- Tool: EleutherAI, [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (free, open source; the harness of Chapter 20; applies to checkpoints in Hugging Face format; nanochat uses its own scripts for the CORE score, ARC, MMLU, GSM8K, and HumanEval).
- Tool: [Weights & Biases](https://docs.wandb.ai/) documentation (free tier for personal projects; experiment tracking; nanochat logs loss, throughput, and model FLOPs utilization (MFU) to it when `WANDB_RUN` is set).
- Hands-on: GPUs rented by the hour from [Lambda](https://lambda.ai/), [RunPod](https://www.runpod.io/), or [Vast.ai](https://vast.ai/) (paid; billed whether or not the GPU is in use; copy the checkpoints and logs off the instance and shut it down after each session).

**Milestone.** Write a one-page plan that states the model size, the token budget, the compute in FLOPs from C ≈ 6ND, and a cost ceiling between $15 and $100; then, within that ceiling, train a tokenizer, pretrain a model, and fine-tune it with SFT (a DPO or RL stage is optional). Evaluate the base checkpoint on one benchmark that has a published GPT-2 baseline, such as the CORE score or HellaSwag, and the chat checkpoint on at least two tasks, such as ARC and GSM8K, and report each accuracy with its standard error. Serve the final checkpoint through a chat interface and measure its generation throughput in tokens per second. Publish the checkpoints, a model card, and a write-up on the Hugging Face Hub; the write-up states the data, the hyperparameters, the loss curves, the actual cost, the evaluation results, and one change made after the first evaluation together with its measured effect.

**Time.** 4–8 weeks, flexible based on budget.

## Chapter 24 — Keeping Up: The Research Reading Loop

**Goals.** This chapter describes a routine for following research on large language models (LLMs) after the course ends: selecting papers, reading them in several passes, reproducing a result, and recording notes. The arXiv category Computation and Language (cs.CL) alone lists more than one hundred new papers per working day, so the routine begins with filtering: curated lists, newsletters, citation alerts, and the release notes of open models reduce the daily volume to a few papers per week. The chapter covers the three-pass reading method of Keshav, the parts of a machine-learning paper that carry the evidence (experimental setup, baselines, ablations, and the evaluation settings treated in Chapter 20), and the reproduction of a single quantitative claim on small hardware with released code and weights. It also covers the tools that keep the results retrievable: a reference manager, an annotated bibliography under version control, and citation alerts.

**Topics.** Sources of new work: arXiv listings (cs.CL and Machine Learning, cs.LG), Hugging Face Daily Papers, conference proceedings, and the technical reports of open models; filtering: newsletters, curated lists, citation graphs, and alerts; the three-pass method: the general idea from title, abstract, introduction, headings, and conclusions (first pass), the content of figures, tables, and experiments (second pass), and a reconstruction of the work (third pass); structure of a machine-learning paper: claims, method, experimental setup, baselines, ablations, and limitations; critical reading: evaluation settings, benchmark contamination, statistical uncertainty, and comparability of compute (Chapter 20); literature survey of an unfamiliar area from a few seed papers and their citations; open-source releases: model cards, technical reports, and the release notes of training and inference libraries (Chapters 21 and 22); reproduction of one claim: choice of the claim, use of released code and weights, reduction of scale, and a record of deviations from the paper; notes and reference library: a one-paragraph summary per paper, an annotated bibliography in a Git repository, and a reference manager.

**Resources.**

**Suggested path.** Read Keshav's paper (3 pages) and watch the first half of Andrew Ng's lecture, which covers the reading of research papers; together they take under two hours and define the method used in the milestone. Set up the tools next — Zotero, a Git repository for the bibliography, and Semantic Scholar alerts for three to five papers from Chapters 12–21 — and subscribe to at most two newsletters. From then on, reserve a fixed weekly slot of about three hours: scan Hugging Face Daily Papers and one newsletter to select papers, apply the first pass to each, carry one paper per week through the second pass, and every two weeks reproduce one claim with the released code, the Open-Source AI Cookbook, and the evaluation methods of Chapter 20. Read Pineau et al. and Lipton and Steinhardt once, before the first reproduction; the arXiv listings, CS25, the video channels, the article by Huyen, and the remaining newsletters are optional sources.

### University courses

- Stanford — [CS230, Lecture 8: Career Advice / Reading Research Papers](https://www.youtube.com/watch?v=733m6qBH-jI) by Andrew Ng (free; Autumn 2018; about 1 hour; start here: the first half describes a reading routine — a list of papers, several passes over each, and an estimate of the number of papers needed to understand an area; the second half, on careers, is optional).
- Stanford — [CS25: Transformers United](https://web.stanford.edu/class/cs25/) (free seminar; optional; sixth edition in Spring 2026; the [recordings on YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM) consist of guest talks in which researchers present current work; first listed in Chapter 12).

### Lectures, papers and articles

- Paper: S. Keshav, [How to Read a Paper](https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf) (ACM SIGCOMM Computer Communication Review, 2007; 3 pages; start here: the three-pass method, the time each pass takes, and its use for a literature survey).
- Paper: Pineau et al., [Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)](https://arxiv.org/abs/2003.12206) (2020; defines reproducibility, replicability, and robustness, and includes the Machine Learning Reproducibility Checklist, which serves as a list of the details to look for in a paper before a reproduction).
- Paper: Zachary C. Lipton and Jacob Steinhardt, [Troubling Trends in Machine Learning Scholarship](https://arxiv.org/abs/1807.03341) (2018; four recurring flaws — speculation presented as explanation, unidentified sources of empirical gains, mathematics that obscures, and misuse of language — with examples; a checklist for the second pass).
- Newsletter: Sebastian Raschka, [Ahead of AI](https://magazine.sebastianraschka.com/) (free tier; roughly monthly; long articles that explain recent LLM papers and architectures, often with code; the recommended first subscription, because the author's books are used in Chapters 12–19).
- Blog: Lilian Weng, [Lil'Log](https://lilianweng.github.io/) (free; infrequent survey articles, each covering one research area with a full reference list; the articles on large-model training and inference optimization are cited in Chapters 16 and 21).
- Newsletters: [Import AI](https://importai.substack.com/) by Jack Clark (weekly; research summaries and policy), [The Batch](https://www.deeplearning.ai/the-batch) by Andrew Ng and DeepLearning.AI (weekly; industry news and paper summaries), [Interconnects](https://www.interconnects.ai/) by Nathan Lambert (open models and post-training; the author's book is used in Chapters 17–19), and [Latent Space](https://www.latent.space/) (newsletter and podcast on AI engineering practice).
- Video: [Yannic Kilcher](https://www.youtube.com/@YannicKilcher) (optional; recorded walkthroughs of individual papers, section by section, with criticism of the experiments) and [Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy) (optional; infrequent long lectures on LLM training and use, several of which are assigned in Chapters 10–14).
- Article: Chip Huyen, [Open challenges in LLM research](https://huyenchip.com/2023/08/16/llm-research-open-challenges.html) (2023; optional: ten research directions, from hallucination to GPU alternatives; an exercise is to find one 2025–2026 paper for each direction and state what has changed).

### Tools and hands-on

- Hugging Face — [Daily Papers](https://huggingface.co/papers) (free; start here: a daily list of new papers selected by curators and ranked by community votes, with links to the code, models, and datasets on the Hub).
- [arXiv cs.CL](https://arxiv.org/list/cs.CL/recent) and [cs.LG](https://arxiv.org/list/cs.LG/recent) recent listings (free; optional: the unfiltered source; scanning the titles of one day shows the volume and the share that is relevant to a given interest).
- [Semantic Scholar](https://www.semanticscholar.org/) (free; citation graphs and email alerts for new citations of a chosen paper), [alphaXiv](https://www.alphaxiv.org/) (free; public comment threads on arXiv papers), and [OpenReview](https://openreview.net/) (free; the peer reviews and author responses of ICLR and NeurIPS submissions).
- Hugging Face — [Open-Source AI Cookbook](https://huggingface.co/learn/cookbook) (free; notebooks that implement published techniques with open models and libraries; a starting point for a reproduction).
- [Zotero](https://www.zotero.org/) (free, open source; a reference manager with browser capture, PDF annotation, and BibTeX export, used for the annotated bibliography of the milestone).

**Milestone.** Create a public Git repository with an annotated bibliography and a directory of reproductions, and for four weeks read three papers per week with the three-pass method, adding one entry per paper: the citation, the main claim, the evidence, the limitations, and the pass reached. Every two weeks, reproduce one quantitative claim from one of the papers on the available hardware — for example, a benchmark score of a released small model, or one row of an ablation table — and record the published value, the measured value, the evaluation settings, and an explanation of any difference. The routine is established when the repository holds twelve entries and two reproductions; continue at the same rate.

**Time.** Ongoing.

---

# Appendix A — The Core Book Shelf

The books below are referenced in several chapters. Free titles are marked.

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

# Appendix B — The Course Shelf

| Course | Provider | Chapter | Access |
| --- | --- | --- | --- |
| College Algebra / Pre-calculus | Khan Academy | 1 | Free, khanacademy.org |
| 18.01 / 18.02 Calculus | MIT OCW | 2 | Free, ocw.mit.edu |
| 18.06 Linear Algebra | MIT OCW | 3 | Free, ocw.mit.edu |
| Statistics 110 | Harvard | 4 | Free, YouTube |
| 18.065 Matrix Methods in Data Analysis, Signal Processing, and ML | MIT OCW | 5 | Free, ocw.mit.edu |
| CS50P / Python for Everybody | Harvard / PY4E | 6 | Free |
| CS229 Machine Learning | Stanford | 8 | Free |
| 6.S191 Introduction to Deep Learning | MIT | 9 | Free |
| Practical Deep Learning for Coders | fast.ai | 9 | Free (optional) |
| CS231n Deep Learning for Computer Vision | Stanford | 10 | Free |
| Zero to Hero | Karpathy | 10, 12, 13, 14, 16 | Free |
| CS224N NLP with Deep Learning | Stanford | 11, 12 | Free (videos) |
| CS25 Transformers United | Stanford | 12 | Free (videos, optional) |
| CS336 Language Modeling from Scratch | Stanford | 12–21, 23 | Free |
| The LLM Course | Hugging Face | 12, 13, 14, 17, 19 | Free, huggingface.co/learn |
| CS324 Foundation Models | Stanford | 15 | Free (optional) |
| 11-868 Large Language Model Systems | CMU | 16, 21 | Free, syllabus online |
| 11-667 LLM Methods and Applications | CMU | 18 | Free, syllabus online |
| The Post-Training Course | Nathan Lambert | 18, 19 | Free |
| smol course | Hugging Face | 17, 18, 20, 23 | Free |
| Efficiently Serving LLMs | DeepLearning.AI | 21 | Free |
| LLM Bootcamp | Full Stack Deep Learning | 22 | Free |
| CS329S Machine Learning Systems Design | Stanford | 22 | Free |

# Appendix C — Suggested First Week

1. Watch Andrej Karpathy's [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) (1 hour) for an overview of the subject.
2. Start Khan Academy College Algebra (Chapter 1): 30–45 minutes per day.
3. Start CS50P (Chapter 6): 45–60 minutes per day.
4. At the end of the week, re-read this plan and set a weekly time budget. A fixed weekly schedule is easier to sustain over a multi-year course than irregular long sessions.
