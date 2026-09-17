# Chapter 3 — Linear Algebra

> Part I — Mathematical Foundations · 8–12 weeks

## What you will learn

This chapter covers vectors, matrices, and the matrix factorizations built on them,
in both their algebraic and their geometric interpretation. A large language model
(LLM) represents each token (a unit of text) as a vector called an embedding,
measures the relevance of one token to another with dot products (the attention
scores of Chapter 12), and applies its weights by matrix multiplication, so every
later chapter uses this material. Orthogonality, projections, and least squares
underlie the regression methods of Chapter 8, and eigenvalues and positive definite
matrices describe the curvature of the loss functions studied in Chapter 5. The
singular value decomposition (SVD) and low-rank approximation are the basis of
principal component analysis (PCA) and of the low-rank fine-tuning methods of Chapter
17.

## Topics

- Vectors, dot products, norms, and angles (cosine similarity)
- Linear combinations, span, and linear independence
- Systems of linear equations and elimination (the LU factorization)
- Matrices as linear transformations
- Matrix multiplication and inverses
- Vector spaces, bases, dimension, and rank; the four fundamental subspaces
- Orthogonality, projections, and least squares
- Gram-Schmidt and the QR factorization
- Determinants
- Eigenvalues, eigenvectors, and diagonalization
- Symmetric and positive definite matrices
- The singular value decomposition
- Low-rank approximation and principal component analysis

## Resources

**Suggested path.** Watch Essence of Linear Algebra first, then follow 18.06SC from
start to finish, with Strang's *Introduction to Linear Algebra* or the free text by
Boyd and Vandenberghe as the written reference; the Khan Academy videos give slower
derivations of individual topics where needed. Read chapters 2–4 and 10 of
*Mathematics for Machine Learning* near the end, before the milestone. When time is
short, skip the 18.06SC sessions on differential equations, Markov matrices and
Fourier series, and complex matrices and the fast Fourier transform, and leave
18.065, Axler, and the other optional and advanced items for a second pass.

### University courses

- MIT OpenCourseWare — [18.06SC Linear
  Algebra](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/) by Gilbert
  Strang (Fall 2011; free; lecture and recitation videos, problem sets and exams with
  solutions; start here: the OpenCourseWare Scholar version of 18.06, arranged for
  independent study; elimination, the four fundamental subspaces, orthogonality,
  determinants, eigenvalues, and the SVD; the original course page is [18.06 Spring
  2010](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)).
- MIT OpenCourseWare — [18.065 Matrix Methods in Data Analysis, Signal Processing,
  and Machine
  Learning](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/)
  by Gilbert Strang (Spring 2018; free; lecture videos + problem sets; advanced; the
  follow-on course to 18.06: the SVD, low-rank approximation, and PCA, followed by
  optimization and neural networks; Chapter 5 uses lectures 21–25).
- MIT OpenCourseWare — [RES.18-010 A Vision of Linear
  Algebra](https://ocw.mit.edu/courses/res-18-010-a-2020-vision-of-linear-algebra-spring-2020/)
  by Gilbert Strang (Spring 2020, with later additions; free; optional; short videos
  that summarize the subject through the factorization A = CR, the four fundamental
  subspaces, and the SVD; a review after 18.06SC).

### Online courses (MOOCs)

- Khan Academy — [Linear Algebra](https://www.khanacademy.org/math/linear-algebra)
  (free; videos; start here: step-by-step derivations of span, null space, column
  space, and change of basis, as a supplement to the 18.06SC lectures).
- Imperial College London (Coursera) — [Mathematics for Machine Learning: Linear
  Algebra](https://www.coursera.org/learn/linear-algebra-machine-learning), and
  DeepLearning.AI (Coursera) — [Linear Algebra for Machine Learning and Data
  Science](https://www.coursera.org/learn/machine-learning-linear-algebra) (free to
  audit, paid certificates; optional here; the linear algebra courses of the two
  mathematics-for-machine-learning specializations that Chapter 5 describes and uses
  for consolidation).
- fast.ai — [Computational Linear Algebra for
  Coders](https://github.com/fastai/numerical-linear-algebra) by Rachel Thomas (2017;
  free; optional; Jupyter notebooks + a [lecture video
  playlist](https://www.youtube.com/playlist?list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY);
  the SVD, PCA, non-negative matrix factorization, and QR implemented in NumPy on
  applied problems).

### Books

- Book: Gilbert Strang, *Introduction to Linear Algebra* (6th ed.,
  Wellesley-Cambridge Press, 2023) — [official
  page](https://math.mit.edu/~gs/linearalgebra/ila6/indexila6.html) (paid; sample
  sections and selected solutions free; start here: the textbook of 18.06; chapters
  1–7 cover every topic of this chapter, through the SVD).
- Book: Stephen Boyd and Lieven Vandenberghe, *Introduction to Applied Linear
  Algebra: Vectors, Matrices, and Least Squares* (Cambridge University Press, 2018) —
  [official page](https://web.stanford.edu/~boyd/vmls/) (free PDF, with Julia and
  Python companions, slides, and lecture videos from Stanford ENGR108; an applied,
  data-oriented introduction to norms, clustering, and least squares; does not cover
  eigenvalues or the SVD).
- Book: Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong, *Mathematics for
  Machine Learning* (Cambridge University Press, 2020) — [official
  page](https://mml-book.github.io) (free PDF; chapters 2, "Linear Algebra", 3,
  "Analytic Geometry", and 4, "Matrix Decompositions", then chapter 10 on PCA, the
  subject of the milestone).
- Book: David C. Lay, Steven R. Lay, and Judi J. McDonald, *Linear Algebra and Its
  Applications* (6th ed., Pearson, 2020) — [publisher
  page](https://www.pearson.com/en-us/subject-catalog/p/linear-algebra-and-its-applications/P200000006235/9780136880929)
  (paid; optional; a widely assigned first-course textbook in US universities;
  computational, with many applications).
- Book: Jim Hefferon, *Linear Algebra* (4th ed., 2020) — [official
  page](https://hefferon.net/linearalgebra/) (free PDF, with a free book of worked
  answers to all exercises; optional; a proof-oriented first course that prepares for
  Axler).
- Book: Sheldon Axler, *Linear Algebra Done Right* (4th ed., Springer, 2024) —
  [official page](https://linear.axler.net) (free PDF, open access under CC BY-NC;
  advanced; a standard second course, which develops the theory through linear maps
  and introduces determinants only near the end).
- Book: Gilbert Strang, *Linear Algebra and Learning from Data* (Wellesley-Cambridge
  Press, 2019) — [official page](https://math.mit.edu/~gs/learningfromdata/) (paid;
  advanced; the textbook of 18.065: the SVD, low-rank approximation, optimization,
  and neural networks).

### Lectures, papers and articles

- Lecture series: Grant Sanderson (3Blue1Brown), [Essence of Linear
  Algebra](https://www.3blue1brown.com/lessons/eola-preview) (2016; free; 16 lessons;
  start here: matrices as transformations of space, determinants as scaling factors
  of area, and eigenvectors, all shown geometrically; watch before or alongside
  18.06SC).

## Milestone

Implement matrix multiplication and the transpose with plain Python loops and verify
them against the NumPy operators `@` and `.T`. Compute the dominant eigenvector of a
small symmetric matrix by power iteration and verify it against `numpy.linalg.eigh`.
Then perform PCA on a small dataset such as Iris in two ways — from the eigenvectors
of the covariance matrix and from the SVD of the centered data — and show that both
give the same two-dimensional projection up to sign.

## Estimated time

8–12 weeks.
