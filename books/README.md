# The local library (`books/`)

Local copies of the core bookshelf. Titles are selected for clarity and precision on
their topic; price is secondary. Free titles that have no stable official PDF are cited
by URL instead of being downloaded (see the second table below).

## Downloaded (PDFs in this folder)

| File | Book | Edition / Version | Source (official) | License | Used for |
| --- | --- | --- | --- | --- | --- |
| `mml-book.pdf` | Deisenroth, Faisal, Ong — *Mathematics for Machine Learning* | 2020 | github.io [mml-book.github.io/book/mml-book.pdf](https://mml-book.github.io/book/mml-book.pdf) | CC BY-NC-SA 4.0 | Chapter 5 (optimization) prerequisites, linear algebra review |
| `linear_algebra_done_right_4e.pdf` | Axler — *Linear Algebra Done Right* | 4th ed., 2023 | linear.axler.net [LADR4e.pdf](https://linear.axler.net/LADR4e.pdf) | CC BY-NC 4.0 (Open Access) | Chapter 3 deeper rigor |
| `an_introduction_to_statistical_learning_python.pdf` | James, Witten, Hastie, Tibshirani — *ISL with Python* (ISLP) | 2023 (corr.) | hastie.su.domains [download page](https://hastie.su.domains/ISLP/ISLP_website.pdf.download.html) | Freely downloadable from authors (Springer copyright) | Chapter 8 (machine learning); preferred over the R edition because the course code is Python |
| `an_introduction_to_statistical_learning_r_2nd.pdf` | James et al. — *ISLR* 2nd ed. | 2023 corrected | hastie.su.domains [download page](https://hastie.su.domains/ISLR2/ISLRv2_corrected_June_2023.pdf.download.html) | Freely downloadable from authors | Chapter 8, when R wording matters |
| `introduction_to_statistical_learning_r_1st.pdf` | James et al. — *ISLR* 1st ed. | 7th printing | statlearning.com [ISLR-Seventh-Printing.pdf](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf) | Freely downloadable from authors | Historical/extra examples |
| `speech_and_language_processing_3e.pdf` | Jurafsky & Martin — *Speech and Language Processing* | 3rd ed. draft | stanford.edu [ed3book.pdf](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf) | Free draft (publisher release RR 2025) | Chapter 11 (NLP foundations) |
| `understanding_deep_learning.pdf` | Prince — *Understanding Deep Learning* | 09 Feb 2026 release | github [release v5.0.3](https://github.com/udlbook/udlbook/releases/download/v5.0.3/UnderstandingDeepLearning_02_09_26_C.pdf) | CC BY-NC-ND 4.0 (MIT Press Open Access) | Parts I, III, IV; the deep-learning reference text of this course |
| `udl_answer_booklet.pdf` | Prince — answers to selected questions | — | github [raw](https://github.com/udlbook/udlbook/raw/main/UDL_Answer_Booklet_Students.pdf) | CC BY-NC-ND 4.0 | Self-check on UDL exercises |
| `udl_errata.pdf` | Prince — errata for the PDF | — | github [raw](https://github.com/udlbook/udlbook/raw/main/UDL_Errata.pdf) | — | Keep next to the book |

> Sizes: 16.8M / 2.7M / 19.1M / 24M / 10.7M / 25.3M / 21.3M / 1.7M / 1.2M bytes. All verify as `%PDF`.
> The `403/404` notes from earlier research are why two ISL download pages exist: the
> official `hastie.su.domains` "download" pages are meta-refresh redirects to the authors'
> public Google Drive IDs, which `curl -L` follows automatically.

## Free to read online (no stable official PDF — cite, do not download)

| Book | Where | Note |
| --- | --- | --- |
| Blitzstein & Hwang — *Introduction to Probability* (2nd ed.) | [probabilitybook.net](https://probabilitybook.net) | Official free 2nd edition (Stat 110). The "Google Drive" and library PDFs in circulation are not officially distributed; use the site. Used in Chapter 4. |
| Goodfellow, Bengio, Courville — *Deep Learning* | [deeplearningbook.org](https://www.deeplearningbook.org) | Official free online edition (HTML per-chapter; `front_matter.pdf`). Cite chapters 1–12 for Part I/III rigor. The `janishar/mit-deep-learning-book-pdf` repo is a community render of uneven quality; cite the canonical HTML. |

## Paid (buy or borrow)

| Book | Role in this course | Get it | Free companion material (cite) |
| --- | --- | --- | --- |
| Strang — *Introduction to Linear Algebra* (6th ed.) | The textbook used by MIT 18.06; the applied text for Chapter 3 | [Borrow 4th ed. on archive.org](https://archive.org/details/introductiontoli0000stra) or [order from Wellesley-Cambridge Press](https://math.mit.edu/~gs/weborder.php) | MIT OCW 18.06 videos + [math.mit.edu/linearalgebra](https://math.mit.edu/linearalgebra) (viewer + site) |
| Raschka — *Build a Large Language Model (From Scratch)* | Main text for Chapters 13–15; builds a GPT-style model step by step in PyTorch | Buy (Manning/O'Reilly), read on liveBook | Free hub + videos: [sebastianraschka.com/llms-from-scratch](https://sebastianraschka.com/llms-from-scratch); code: [github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) |
| Raschka — *Build a Reasoning Model (From Scratch)* | Chapter 19; implements the training methods of reasoning models from scratch | Buy (Manning) | Free hub + videos: [sebastianraschka.com/reasoning-from-scratch](https://sebastianraschka.com/reasoning-from-scratch); code: [github.com/rasbt/reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch) |
| Alammar & Grootendorst — *Hands-On Large Language Models* | Illustrated explanations of Transformer and LLM internals (Chapters 13, 14, 17, 22) | Buy (O'Reilly) | Free site + code: [llm-book.com](https://llm-book.com/); blog: [jalammar.github.io](https://jalammar.github.io) |
| Kamath et al. — *Large Language Models: A Deep Dive* | Reference text for the research-reading routine (Chapter 24) | [Springer](https://link.springer.com/book/10.1007/978-3-031-65647-7) (open chapter 1, sample PDFs) | Springer preview chapters |
| Huyen — *AI Engineering* | Hardware vocabulary (Chapter 7), fine-tuning and dataset engineering (Chapter 17), and evaluation, serving, and operation (Chapters 20–22) | Buy (O'Reilly) | Free chapter summaries: [github.com/chiphuyen/aie-book](https://github.com/chiphuyen/aie-book) |

> Do **not** download "z-library"/mirror copies found via search (ISLP's official page is
> the only sanctioned Google Drive; the Blitzstein/archive.org PDFs and the Alammar
> `archive.org/download/...` item are unauthorized uploads).

## How to refresh or verify

```powershell
# Re-download the authors' canonical copy when the PDF updates (recommended for UDL):
curl -L -o books/understanding_deep_learning.pdf `
  https://github.com/udlbook/udlbook/releases/latest/download/UnderstandingDeepLearning_latest.pdf
# Check no file is corrupt:
Get-ChildItem books -Filter *.pdf | ForEach-Object {
  $b = [System.IO.File]::ReadAllBytes($_.FullName)
  [PSCustomObject]@{ File = $_.Name; Bytes = $b.Length; Magic = [Text.Encoding]::ASCII.GetString($b[0..3]) }
}
```

The PDFs are gitignored (`books/*.pdf`) — this manifest is the only tracked artifact.
