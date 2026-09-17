# Chapter 20 — Evaluation and Benchmarks

> Part VI — Evaluation, Deployment, and the Ecosystem · 3–4 weeks

## What you will learn

This chapter covers the methods used to measure the quality of a language model and
to compare two models. It describes intrinsic metrics (perplexity and bits per
byte), the standard benchmarks and the two ways in which they are scored
(log-likelihood of answer choices and scoring of generated text), human-preference
rankings, and the use of a strong model as a judge. It also covers the sources of
error in reported scores — prompt format, few-shot settings, sampling noise, and
benchmark contamination — and the statistics needed to state whether a difference
between two models is significant. Later chapters apply these methods: Chapter 21
measures the accuracy cost of quantization, Chapter 22 measures the effect of
fine-tuning and retrieval, and the Chapter 23 capstone requires an evaluation
report.

## Topics

- The purposes of evaluation: model selection, regression testing, and reporting
- Perplexity and bits per byte as intrinsic metrics
- Standard benchmarks by capability: knowledge (MMLU), commonsense and science
  reasoning (HellaSwag, ARC), mathematics (GSM8K), code (HumanEval), graduate-level
  science (GPQA), and instruction following (IFEval)
- Scoring methods: log-likelihood of answer choices, exact match on generated
  answers, and unit tests with pass@k for code
- Evaluation settings that change scores: few-shot examples, prompt and chat
  templates, answer normalization, and decoding parameters
- Statistical uncertainty: standard errors, confidence intervals, and paired
  comparisons
- Benchmark contamination and saturation
- Human evaluation and preference arenas: pairwise votes and Elo-style
  (Bradley–Terry) ratings
- Use of a large language model (LLM) as a judge, and its position, verbosity, and
  self-preference biases
- Agentic, multi-step benchmarks (SWE-bench)
- Task-specific evaluation: building a test set and a custom harness task
- Reproducible reporting of evaluation results

## Resources

**Suggested path.** Start with the LLM Evaluation Guidebook and Unit 2 of the smol
course, and run lm-evaluation-harness on a small model during the first week;
Biderman et al. explains the settings that change its scores. Read chapters 3 and 4
of *AI Engineering* and watch CS336 lecture 12 in parallel. Then read Miller for the
statistics, Zheng et al. and Chiang et al. for judges and arenas, and Oren et al. for
contamination. When time is short, read the benchmark papers for the task format
only, and omit HELM, Inspect, and the model reports.

### University courses

- Stanford — [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
  by Percy Liang and Tatsunori Hashimoto (free; Spring 2026 lecture videos on a
  [YouTube playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)
  and public assignments; start here: lecture 12, "Evaluation", covers perplexity,
  knowledge, instruction-following, and agent benchmarks, and validity problems such
  as contamination).

### Online courses (MOOCs)

- Hugging Face — [smol course, Unit 2: Evaluation](https://huggingface.co/learn/smol-course/unit2/1)
  (free; v2, 2025; about one week; start here: running standard benchmarks with
  lighteval and building a custom evaluation for a specific domain).

### Books

- Book: Chip Huyen, *AI Engineering* (O'Reilly, 2025) — [book repo](https://github.com/chiphuyen/aie-book)
  (paid book, free chapter summaries; start here: chapter 3, "Evaluation
  Methodology", on perplexity, exact and similarity-based scoring, and AI judges;
  chapter 4, "Evaluate AI Systems", on evaluation criteria, model selection, and the
  design of an evaluation pipeline).

### Lectures, papers and articles

- Article: Clémentine Fourrier et al., [The LLM Evaluation Guidebook](https://huggingface.co/spaces/OpenEvals/evaluation-guidebook)
  (Hugging Face; free; start here: automatic benchmarks, human evaluation,
  LLM-as-a-judge, and a troubleshooting section on reproducibility, by the team that
  maintained the Open LLM Leaderboard).
- Paper: Biderman et al., [Lessons from the Trenches on Reproducible Evaluation of
  Language Models](https://arxiv.org/abs/2405.14782) (2024; the companion paper of
  lm-evaluation-harness; the reasons the same benchmark yields different numbers
  across implementations, and recommendations for reporting results).
- Paper: Evan Miller, [Adding Error Bars to Evals: A Statistical Approach to Language
  Model Evaluations](https://arxiv.org/abs/2411.00640) (2024; standard errors,
  clustered questions, paired differences, and sample-size calculation, using the
  statistics of Chapter 4).
- Papers: the original benchmark papers — Hendrycks et al., [MMLU](https://arxiv.org/abs/2009.03300)
  (2020); Clark et al., [ARC](https://arxiv.org/abs/1803.05457) (2018); Zellers et
  al., [HellaSwag](https://arxiv.org/abs/1905.07830) (2019); Cobbe et al.,
  [GSM8K](https://arxiv.org/abs/2110.14168) (2021); Chen et al.,
  [HumanEval](https://arxiv.org/abs/2107.03374) (2021; introduced in the Codex paper
  together with pass@k); Rein et al., [GPQA](https://arxiv.org/abs/2311.12022)
  (2023); Zhou et al., [IFEval](https://arxiv.org/abs/2311.07911) (2023); and Jimenez
  et al., [SWE-bench](https://arxiv.org/abs/2310.06770) (2023; tasks built from real
  GitHub issues; the design followed by later agent benchmarks). Read each for the
  task format and the scoring rule.
- Paper: Zheng et al., [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
  (2023; measures the agreement between a strong model and human judges, and
  documents position, verbosity, and self-enhancement bias).
- Paper: Chiang et al., [Chatbot Arena: An Open Platform for Evaluating LLMs by Human
  Preference](https://arxiv.org/abs/2403.04132) (2024; the pairwise-comparison
  statistics behind the public leaderboard now published at [Arena](https://arena.ai/),
  formerly LMArena).
- Paper: Oren et al., [Proving Test Set Contamination in Black Box Language
  Models](https://arxiv.org/abs/2310.17623) (2023; a statistical test that detects
  whether a benchmark was present in the training data, using only model outputs).
- Paper: Liang et al., [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110)
  (2022; optional; the case for reporting several metrics — accuracy, calibration,
  robustness, fairness, and efficiency — over many scenarios; the
  [HELM framework and leaderboards](https://github.com/stanford-crfm/helm) have been
  in maintenance mode since June 2026).
- Article: Hamel Husain, [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/)
  (2024; evaluation of an application at three levels: unit tests, human and model
  review of logged outputs, and A/B tests; used again in Chapter 22).
- Papers: [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (2024) and
  [DeepSeek-R1](https://arxiv.org/abs/2501.12948) (2025) (optional; read only the
  evaluation sections, as examples of how model developers report benchmark
  settings and baselines).

### Tools and hands-on

- Tool: EleutherAI, [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
  (free, open source; start here: the harness used by most open-model reports and by
  this chapter's milestone; tasks are defined in YAML files, and results include
  standard errors).
- Tool: UK AI Security Institute, [Inspect](https://inspect.aisi.org.uk/) (free,
  open source; optional; a framework built from datasets, solvers, and scorers,
  suited to model-graded and multi-step agent evaluations).

## Milestone

On a free Colab or Kaggle T4 GPU, run lm-evaluation-harness on two open base models
of 1–2 billion parameters (for example Qwen2.5-1.5B and SmolLM2-1.7B) on ARC-Easy,
HellaSwag, and a 250-question subset of GSM8K, and compare each score and its
standard error with the value published by the model's authors. Add one custom task
of at least 50 questions, defined in a harness YAML task file. Write a two-page
report that states the evaluation settings (few-shot count, prompt format, answer
normalization) and explains one score difference, either between the two models or
between a measured and a published value.

## Estimated time

3–4 weeks.
