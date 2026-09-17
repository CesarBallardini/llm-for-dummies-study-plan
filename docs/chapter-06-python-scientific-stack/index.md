# Chapter 6 — Python and the Scientific Computing Stack

> Part II — Programming and Tools · 8–12 weeks, overlapping with math

This chapter runs in parallel with Part I and can start in the first week, alongside
Chapter 1. It does not depend on the later mathematics chapters, and implementing
each mathematical idea in code as it is studied provides a second way to check it.

## What you will learn

This chapter covers programming in Python, the language in which nearly all large
language model (LLM) tools are written, together with the standard working
environment: the command line, virtual environments, Git, and Jupyter notebooks. It
covers functions and classes, file input and output, and the testing and debugging
of code. It then introduces NumPy, which applies mathematical operations to whole
arrays at once, and Matplotlib, which plots the results; both are used to implement
the mathematics of Part I. The chapter closes with an introduction to PyTorch, the
library used from Chapter 7 onward, whose tensors are NumPy-style arrays that can
run on a graphics processing unit (GPU) and compute derivatives automatically.

## Topics

- The command line, a code editor, and running Python scripts
- Python syntax and built-in data structures (lists, tuples, dictionaries, sets)
- Functions, modules, and classes
- File I/O (text, CSV, and JSON)
- Errors, debugging, and unit tests (tracebacks, pytest)
- Virtual environments and packages (venv, pip, uv)
- Version control with Git (commit, branch, merge, push)
- Jupyter notebooks as a working environment (locally and on Google Colab)
- NumPy arrays: shapes, dtypes, indexing, broadcasting, and vectorized math
- Plotting with Matplotlib
- An introduction to PyTorch tensors and automatic differentiation (autograd)

## Resources

**Suggested path.** Start in the first week, alongside Chapter 1. Use CS50P as the
single main course (or Python for Everybody for a slower pace) at 45–60 minutes
per day next to the Part I mathematics, and read the matching chapter of *Think
Python* when a lecture needs a second explanation. In parallel, watch the shell and
Git lectures of the Missing Semester, read *Pro Git* chapters 1–3, and keep the
problem sets in a Git repository from the first week. After the course, work through
the NumPy beginners' page and the CS231n tutorial, finish with PyTorch's Learn the
Basics or Raschka's one-hour tutorial, and build the milestone. When time is short,
omit 6.100L, *Automate the Boring Stuff*, *Fluent Python*, and the official Python
Tutorial.

### University courses

- Harvard — [CS50P: Introduction to Programming with Python](https://cs50.harvard.edu/python/)
  by David J. Malan (free; videos + problem sets + final project; start here: ten
  weeks from variables to classes, including libraries and pip, unit tests with
  pytest, and file I/O; assumes no programming experience).
- MIT — [The Missing Semester of Your CS Education](https://missing.csail.mit.edu/)
  by Anish Athalye, Jon Gjengset, and Jose Javier Gonzalez Ortiz (free; 2026
  edition, videos + notes + exercises; the lectures on the shell, the command-line
  environment, debugging, Git, and packaging cover tools that introductory Python
  courses omit).
- MIT OpenCourseWare — [6.100L Introduction to CS and Programming Using Python (Fall 2022)](https://ocw.mit.edu/courses/6-100l-introduction-to-cs-and-programming-using-python-fall-2022/)
  by Ana Bell (free; videos + notes + problem sets; optional: an alternative first
  course with more computer science content: algorithms, recursion, and
  complexity).

### Online courses (MOOCs)

- University of Michigan (Coursera) — [Python for Everybody Specialization](https://www.coursera.org/specializations/python)
  by Charles Severance (free to audit, paid certificate; 5 courses; start here as
  the alternative to CS50P: a slower pace, with less on classes and nothing on
  testing; the same lectures, book, and autograded exercises are free at
  [py4e.com](https://www.py4e.com)).

### Books

- Book: Allen B. Downey, *Think Python* (3rd ed., 2024) — [official page](https://allendowney.github.io/ThinkPython/)
  (free online as Jupyter notebooks that run on Colab; start here: the companion
  text to the main course; each chapter is a notebook, so reading it also gives
  practice with Jupyter).
- Book: Al Sweigart, *Automate the Boring Stuff with Python* (3rd ed., 2025) —
  [official page](https://automatetheboringstuff.com) (free online; optional: the
  file, CSV, and JSON chapters add practice with file I/O on realistic tasks).
- Book: Jake VanderPlas, *Python Data Science Handbook* (1st ed. free online; 2nd
  ed., 2022, in print) — [official page](https://jakevdp.github.io/PythonDataScienceHandbook/)
  (free; chapter 1 on IPython and Jupyter, chapter 2, "Introduction to NumPy", and
  chapter 4, "Visualization with Matplotlib").
- Book: Scott Chacon and Ben Straub, *Pro Git* (2nd ed., 2014) —
  [official page](https://git-scm.com/book/en/v2) (free; chapters 1–3 cover all the
  Git this course requires: commits, branches, merges, and remotes).
- Book: Luciano Ramalho, *Fluent Python* (2nd ed., 2022) —
  [official page](https://www.fluentpython.com/) (paid; advanced: Part I, "Data
  Structures", explains how idiomatic Python uses the data model, sequences, and
  dictionaries; intended for a second pass after the main course).

### Lectures, papers and articles

- Tutorial: NumPy developers, [NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html)
  (current docs; start here for NumPy: arrays, shapes, indexing, broadcasting, and
  aggregation on one page).
- Tutorial: Justin Johnson (Stanford CS231n), [Python Numpy Tutorial (with Jupyter and Colab)](https://cs231n.github.io/python-numpy-tutorial/)
  (a compact Python, NumPy, and Matplotlib refresher written for deep-learning
  students; usable as a self-test after the main course).
- Tutorial: PyTorch, [Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
  (official; runnable in Colab; the Tensors and Autograd pages belong to this
  chapter, and the Autograd page requires derivatives from Chapter 2; the
  model-building pages belong to Chapter 9).
- Article: Sebastian Raschka, [PyTorch in One Hour: From Tensors to Training Neural Networks on Multiple GPUs](https://sebastianraschka.com/teaching/pytorch-1h/)
  (2025; covers the same ground as Appendix A of his *Build a Large Language
  Model (From Scratch)*; sections 1–4 on tensors and autograd belong to this
  chapter, the training-loop and GPU sections to Chapters 7 and 9).
- Tutorial: Python Software Foundation, [The Python Tutorial](https://docs.python.org/3/tutorial/)
  (current Python 3 docs; optional: the official tour of the language, written for
  readers who can already program; suited to a second pass).

### Tools and hands-on

- Docs: Astral, [uv: First steps](https://docs.astral.sh/uv/getting-started/first-steps/)
  (free; uv is a package and project manager that replaces venv and pip; together
  with the Projects guide on the same site, the page shows how to create a project,
  add NumPy and pytest, and run tests in an isolated environment).

## Milestone

In a dedicated Git repository and virtual environment, build a small NumPy library
with loop-free functions for the dot product, the matrix–vector product, mean
squared error, and a softmax that operates on each row of a 2-D array and does not
overflow on large inputs (subtract the row maximum before exponentiating). Write
pytest unit tests that pass, including one test per function that verifies the
result against the PyTorch equivalent with `np.allclose`.

## Estimated time

8–12 weeks, overlapping with math.
