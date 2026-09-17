# Chapter 24 — Keeping Up: The Research Reading Loop

> Part VI — Evaluation, Deployment, and the Ecosystem · Ongoing

## What you will learn

This chapter describes a routine for following research on large language models
(LLMs) after the course ends: selecting papers, reading them in several passes,
reproducing a result, and recording notes. The arXiv category Computation and
Language (cs.CL) alone lists more than one hundred new papers per working day, so the
routine begins with filtering: curated lists, newsletters, citation alerts, and the
release notes of open models reduce the daily volume to a few papers per week. The
chapter covers the three-pass reading method of Keshav, the parts of a
machine-learning paper that carry the evidence (experimental setup, baselines,
ablations, and the evaluation settings treated in Chapter 20), and the reproduction
of a single quantitative claim on small hardware with released code and weights. It
also covers the tools that keep the results retrievable: a reference manager, an
annotated bibliography under version control, and citation alerts.

## Topics

- Sources of new work: arXiv listings (cs.CL and Machine Learning, cs.LG), Hugging
  Face Daily Papers, conference proceedings, and the technical reports of open
  models
- Filtering: newsletters, curated lists, citation graphs, and alerts
- The three-pass method: the general idea from title, abstract, introduction,
  headings, and conclusions (first pass), the content of figures, tables, and
  experiments (second pass), and a reconstruction of the work (third pass)
- Structure of a machine-learning paper: claims, method, experimental setup,
  baselines, ablations, and limitations
- Critical reading: evaluation settings, benchmark contamination, statistical
  uncertainty, and comparability of compute (Chapter 20)
- Literature survey of an unfamiliar area from a few seed papers and their citations
- Open-source releases: model cards, technical reports, and the release notes of
  training and inference libraries (Chapters 21 and 22)
- Reproduction of one claim: choice of the claim, use of released code and weights,
  reduction of scale, and a record of deviations from the paper
- Notes and reference library: a one-paragraph summary per paper, an annotated
  bibliography in a Git repository, and a reference manager

## Resources

**Suggested path.** Read Keshav's paper (3 pages) and watch the first half of Andrew
Ng's lecture, which covers the reading of research papers; together they take under
two hours and define the method used in the milestone. Set up the tools next —
Zotero, a Git repository for the bibliography, and Semantic Scholar alerts for three
to five papers from Chapters 12–21 — and subscribe to at most two newsletters. From
then on, reserve a fixed weekly slot of about three hours: scan Hugging Face Daily
Papers and one newsletter to select papers, apply the first pass to each, carry one
paper per week through the second pass, and every two weeks reproduce one claim with
the released code, the Open-Source AI Cookbook, and the evaluation methods of
Chapter 20. Read Pineau et al. and Lipton and Steinhardt once, before the first
reproduction; the arXiv listings, CS25, the video channels, the article by Huyen,
and the remaining newsletters are optional sources.

### University courses

- Stanford — [CS230, Lecture 8: Career Advice / Reading Research Papers](https://www.youtube.com/watch?v=733m6qBH-jI)
  by Andrew Ng (free; Autumn 2018; about 1 hour; start here: the first half
  describes a reading routine — a list of papers, several passes over each, and an
  estimate of the number of papers needed to understand an area; the second half,
  on careers, is optional).
- Stanford — [CS25: Transformers United](https://web.stanford.edu/class/cs25/)
  (free seminar; optional; sixth edition in Spring 2026; the
  [recordings on YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM)
  consist of guest talks in which researchers present current work; first listed
  in Chapter 12).

### Lectures, papers and articles

- Paper: S. Keshav, [How to Read a Paper](https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf)
  (ACM SIGCOMM Computer Communication Review, 2007; 3 pages; start here: the
  three-pass method, the time each pass takes, and its use for a literature
  survey).
- Paper: Pineau et al., [Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)](https://arxiv.org/abs/2003.12206)
  (2020; defines reproducibility, replicability, and robustness, and includes the
  Machine Learning Reproducibility Checklist, which serves as a list of the details
  to look for in a paper before a reproduction).
- Paper: Zachary C. Lipton and Jacob Steinhardt, [Troubling Trends in Machine Learning Scholarship](https://arxiv.org/abs/1807.03341)
  (2018; four recurring flaws — speculation presented as explanation, unidentified
  sources of empirical gains, mathematics that obscures, and misuse of language —
  with examples; a checklist for the second pass).
- Newsletter: Sebastian Raschka, [Ahead of AI](https://magazine.sebastianraschka.com/)
  (free tier; roughly monthly; long articles that explain recent LLM papers and
  architectures, often with code; the recommended first subscription, because the
  author's books are used in Chapters 12–19).
- Blog: Lilian Weng, [Lil'Log](https://lilianweng.github.io/) (free; infrequent
  survey articles, each covering one research area with a full reference list; the
  articles on large-model training and inference optimization are cited in
  Chapters 16 and 21).
- Newsletters: [Import AI](https://importai.substack.com/) by Jack Clark (weekly;
  research summaries and policy), [The Batch](https://www.deeplearning.ai/the-batch)
  by Andrew Ng and DeepLearning.AI (weekly; industry news and paper summaries),
  [Interconnects](https://www.interconnects.ai/) by Nathan Lambert (open models and
  post-training; the author's book is used in Chapters 17–19), and
  [Latent Space](https://www.latent.space/) (newsletter and podcast on AI
  engineering practice).
- Video: [Yannic Kilcher](https://www.youtube.com/@YannicKilcher) (optional;
  recorded walkthroughs of individual papers, section by section, with criticism of
  the experiments) and [Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy)
  (optional; infrequent long lectures on LLM training and use, several of which are
  assigned in Chapters 10–14).
- Article: Chip Huyen, [Open challenges in LLM research](https://huyenchip.com/2023/08/16/llm-research-open-challenges.html)
  (2023; optional: ten research directions, from hallucination to GPU alternatives;
  an exercise is to find one 2025–2026 paper for each direction and state what has
  changed).

### Tools and hands-on

- Hugging Face — [Daily Papers](https://huggingface.co/papers) (free; start here: a
  daily list of new papers selected by curators and ranked by community votes, with
  links to the code, models, and datasets on the Hub).
- [arXiv cs.CL](https://arxiv.org/list/cs.CL/recent) and
  [cs.LG](https://arxiv.org/list/cs.LG/recent) recent listings (free; optional: the
  unfiltered source; scanning the titles of one day shows the volume and the share
  that is relevant to a given interest).
- [Semantic Scholar](https://www.semanticscholar.org/) (free; citation graphs and
  email alerts for new citations of a chosen paper), [alphaXiv](https://www.alphaxiv.org/)
  (free; public comment threads on arXiv papers), and [OpenReview](https://openreview.net/)
  (free; the peer reviews and author responses of ICLR and NeurIPS submissions).
- Hugging Face — [Open-Source AI Cookbook](https://huggingface.co/learn/cookbook)
  (free; notebooks that implement published techniques with open models and
  libraries; a starting point for a reproduction).
- [Zotero](https://www.zotero.org/) (free, open source; a reference manager with
  browser capture, PDF annotation, and BibTeX export, used for the annotated
  bibliography of the milestone).

## Milestone

Create a public Git repository with an annotated bibliography and a directory of
reproductions, and for four weeks read three papers per week with the three-pass
method, adding one entry per paper: the citation, the main claim, the evidence, the
limitations, and the pass reached. Every two weeks, reproduce one quantitative claim
from one of the papers on the available hardware — for example, a benchmark score of
a released small model, or one row of an ablation table — and record the published
value, the measured value, the evaluation settings, and an explanation of any
difference. The routine is established when the repository holds twelve entries and
two reproductions; continue at the same rate.

## Estimated time

Ongoing.
