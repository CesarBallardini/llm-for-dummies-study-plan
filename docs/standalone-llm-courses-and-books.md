# Standalone courses and books

Compiled on 2026-10-11 from the resources this book cites.

The study plan is built from many single-topic resources, one set per chapter. This page lists the
resources it cites that are **already designed as a complete course or book** on large language models
(LLMs), or on applications built on LLMs. Each one could be followed on its own, as an alternative to the
plan or as a spine to read it against.

Scopes come from how the plan describes each resource. I did not re-open the course sites or books to
confirm them, so check a syllabus before committing to one.

## How to read the tables

- Course names link to the official page the study plan cites for them.
- **Scope** says which stages of the LLM pipeline it covers: architecture, tokenization, building a model,
  pretraining, scaling and systems, fine-tuning, alignment, reasoning, evaluation, serving, applications.
- **Covers** rates how much of the LLM stack the resource covers:
  - **Full** means from the Transformer through training, post-training and use.
  - **Broad** means most of it, with a few stages missing or thin.
  - **Stage** means one part of the stack, treated in depth.
- **Needs** names the background it assumes, in the plan's chapter numbers.
- **Plan ch.** lists the chapters of this plan that already use it.
- **Free?** means the full course or text can be followed at no cost. "Partial" means the code, notebooks
  or videos are free but the text is paid.

---

## 1. Courses

### 1.1 Whole-stack LLM courses

| Course | Provider and instructors | Edition cited | Scope | Covers | Needs | Plan ch. | Free? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) | Stanford (Percy Liang, Tatsunori Hashimoto) | Spring 2026 | Tokenizer, Transformer, GPU kernels, parallelism, scaling laws, data pipelines, inference, evaluation, SFT, RLHF and RLVR. Five graded assignments: basics, systems, scaling, data, alignment. | Full | Python, PyTorch, Chapters 1–10 | 7, 12–21, 23 | Yes (videos and public assignments) |
| [CME295: Transformers & Large Language Models](https://cme295.stanford.edu/) | Stanford (Afshine and Shervine Amidi) | Fall 2026 syllabus, Autumn 2025 recordings | Nine lectures of about 1h45m: Transformer, MoE and RoPE, pretraining, SFT, LoRA, RLHF and DPO, RL for reasoning, systems, agents, evaluation. No assignments. | Broad (survey, no hands-on) | Light | 12, 19, 20, 22 | Yes |
| [CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) | Stanford (Diyi Yang, Yejin Choi; Christopher Manning for the 2024 videos) | Winter 2026 materials, Spring 2024 videos | Word vectors, RNNs, seq2seq, Transformers, pretraining, then more NLP. Assignments cover Transformers. | Broad (strong on foundations, lighter on post-training and serving) | Python, PyTorch, Chapters 1–9 | 11, 12 | Yes (videos and assignments) |
| [11-667: Large Language Models: Methods and Applications](https://2025.cmu-llms.org/) | CMU (Jaromir Savelka, Yubin Kim) | Fall 2025 | Task fine-tuning, RLHF and applications, placed in the context of the full LLM pipeline. | Broad | Light | 18 | Yes (slides only, no videos) |
| [11-711: Advanced NLP](https://cmu-l3.github.io/anlp-spring2026/) | CMU (Graham Neubig; Sean Welleck for 2026) | Fall 2024 lectures, Spring 2026 page | Research-oriented treatment of word representation, language modeling and sequence models. | Stage (NLP foundations) | NLP basics | 11 | Yes |
| [CS324: Large Language Models](https://stanford-cs324.github.io/winter2022/) | Stanford (Percy Liang, Tatsunori Hashimoto, Christopher Ré) | Winter 2022 | Lecture notes on data, training and scaling laws. | Stage, and dated (no post-2022 developments) | Light | 15 | Yes |
| [CS25: Transformers United](https://web.stanford.edu/class/cs25/) | Stanford | Sixth edition, Spring 2026 | Seminar: overview lectures, then guest talks on current research. | Stage (research survey) | Light | 12, 24 | Yes |

### 1.2 Build-it-yourself video courses

| Course | Provider | Scope | Covers | Needs | Plan ch. | Free? |
| --- | --- | --- | --- | --- | --- | --- |
| [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) | Andrej Karpathy | Eight lectures: autograd, character-level models, hand-written backward pass, a small GPT, a BPE tokenizer. The GPT-2 reproduction lecture is its sequel. | Broad on building and pretraining; no post-training or serving | Python, basic calculus | 10, 12–14, 16, 23 | Yes |
| [HF LLM Course](https://huggingface.co/learn/llm-course) | Hugging Face | Transformers and tokenizers, sharing models, SFT and LoRA, then Open R1 for reasoning. | Broad on using and adapting models; light on from-scratch training | Python | 12–14, 17, 19 | Yes |
| [smol course (v2)](https://huggingface.co/learn/smol-course/unit0/1) | Hugging Face | Instruction tuning, preference alignment, evaluation, based on SmolLM3 and TRL. | Stage (post-training and evaluation) | Python, PyTorch | 17, 18, 20, 23 | Yes |
| [The Post-Training Course](https://rlhfbook.com/course) | Nathan Lambert | Lectures that follow the RLHF book: instruction tuning, reward models, RL, DPO, preference data, over-optimization. | Stage (post-training) | Chapters 8–9 | 18, 19 | Yes |

### 1.3 Courses on building applications on LLMs

| Course | Provider | Edition cited | Scope | Covers | Plan ch. | Free? |
| --- | --- | --- | --- | --- | --- | --- |
| [LLM Bootcamp](https://fullstackdeeplearning.com/llm-bootcamp/) | Full Stack Deep Learning (Charles Frye, Sergey Karayev, Josh Tobin) | Spring 2023 | Prompt engineering, LLMOps, augmented language models, user interfaces, a deployed question-answering app. | Applications; the tools shown are from 2023 | 22 | Yes |
| [AI Agents Course](https://huggingface.co/learn/agents-course) | Hugging Face | 2025 | Agents with smolagents, LlamaIndex and LangGraph; bonus units on function calling and agent observability. | Applications (agents) | 22 | Yes |
| [Retrieval Augmented Generation (RAG)](https://www.deeplearning.ai/courses/retrieval-augmented-generation) | DeepLearning.AI (Zain Hasan) | Not stated in the plan | Keyword, semantic and hybrid search, vector databases, RAG evaluation and monitoring. About 26 hours. | Applications (RAG) | 22 | Audit free, certificate paid |
| [CS329S: Machine Learning Systems Design](https://stanford-cs329s.github.io/) | Stanford (Chip Huyen) | Winter 2022 | Deployment, monitoring, distribution shift, continual learning. Predates LLM applications. | Stage (ML operations) | 22 | Yes |
| [Made With ML](https://madewithml.com/) | Goku Mohandas | Not stated in the plan | MLOps: testing, CI/CD, serving, monitoring, built around fine-tuning an LLM for classification. | Stage (operations) | 22 | Yes |

### 1.4 Short courses that cover one stage

Not full-sized, but they are packaged courses a reader can finish in a day or two.

| Course | Provider | Scope | Plan ch. |
| --- | --- | --- | --- |
| [How Transformer LLMs Work](https://www.deeplearning.ai/short-courses/how-transformer-llms-work/) | DeepLearning.AI (Jay Alammar, Maarten Grootendorst) | Architecture, KV cache, GQA, MoE, tokenizers. 1h44m. | 12, 13 |
| [Pretraining LLMs](https://www.deeplearning.ai/short-courses/pretraining-llms/) | DeepLearning.AI and Upstage | Data preparation, packing, initialization, a short training run. 1h29m. | 15 |
| [Efficiently Serving LLMs](https://www.deeplearning.ai/courses/efficiently-serving-llms) | DeepLearning.AI (Travis Addair) | KV caching, continuous batching, quantization, LoRA serving. 2 hours. | 21 |
| [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) | DeepLearning.AI (Andrew Ng et al.) | General deep learning, not LLM-specific. | 5, 9 |
| [6.S191: Introduction to Deep Learning](https://introtodeeplearning.com/) | MIT (Alexander and Ava Amini) | General deep learning; lab 3 fine-tunes an LLM. | 9 |
| [ARENA](https://github.com/callummcdougall/ARENA_3.0) | Callum McDougall et al. | Mechanistic interpretability of Transformers. | 14 |

---

## 2. Books

### 2.1 Whole-stack LLM books

| Book | Authors | Year and publisher | Scope | Covers | Needs | Plan ch. | Free? | Read online |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| *Build a Large Language Model (From Scratch)* | Sebastian Raschka | 2024, Manning | Text data, attention, a GPT model, pretraining, classification fine-tuning, instruction fine-tuning, LoRA in an appendix. | Broad (no alignment, serving or applications) | Python, PyTorch | 12–15, 17, 23 | Partial (code and videos free) | [Manning page](https://www.manning.com/books/build-a-large-language-model-from-scratch) (paid text) · [free code](https://github.com/rasbt/LLMs-from-scratch) · [free videos](https://www.youtube.com/playlist?list=PLTKMiZHVd_2IIEsoJrWACkIxLRdfMlw11) |
| *Hands-On Large Language Models* | Jay Alammar, Maarten Grootendorst | 2024, O'Reilly | Tokens and embeddings, inside a Transformer, prompting, semantic search and RAG, fine-tuning generation models, preference tuning. | Broad, with an applied slant | Python | 13, 14, 17, 22 | Partial (notebooks free) | [llm-book.com](https://llm-book.com/) (paid text) · [free notebooks](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models) |
| *Speech and Language Processing* (3rd ed. draft) | Dan Jurafsky, James H. Martin | August 2026 draft, Stanford | Textbook covering words and tokens, language models, embeddings, Transformers and pretraining, post-training, retrieval and RAG, interpretability. | Broad (a reference textbook, not a project course) | Light | 11–14, 17, 18, 22 | Yes | [Free draft chapters and PDF](https://web.stanford.edu/~jurafsky/slp3/) |
| *Large Language Models: A Deep Dive* | Uday Kamath, Kevin Keenan, Garrett Somers, Sarah Sorenson | 2024, Springer | Reference text on LLMs. The plan lists it as a reference only. | Broad (not checked) | Not checked | A | Paid | [Springer page](https://link.springer.com/book/10.1007/978-3-031-65647-7) (paid; I found no open-access sign) |
| *Large Language Models: Concepts, Techniques and Applications* | John Atkinson-Abutridy | 2024, CRC Press (Taylor & Francis); some listings say 2025 | A 185-page introduction to the science and applications of LLMs. It is in `books/` but not cited in the plan. | Broad (short overview) | Light | none | Paid | No publisher page verified |
| *Super Study Guide: Transformers & Large Language Models* | Afshine Amidi, Shervine Amidi | 2024, self-published | The written companion to CME295. The plan says it is not required. | Broad (study guide) | Light | 12 | Paid | No free text; [free companion cheatsheet](https://github.com/afshinea/stanford-cme-295-transformers-large-language-models) |

### 2.2 Books on building applications

| Book | Authors | Year and publisher | Scope | Plan ch. | Free? | Read online |
| --- | --- | --- | --- | --- | --- | --- |
| *AI Engineering* | Chip Huyen | 2025, O'Reilly | Evaluation, prompt engineering, RAG and agents, fine-tuning, dataset engineering, inference optimization, system architecture and user feedback. | 7, 17, 20–22 | Partial (chapter summaries free) | [Free summaries and notes](https://github.com/chiphuyen/aie-book) (the text itself is paid) |
| *Designing Machine Learning Systems* | Chip Huyen | 2022, O'Reilly | Deployment, distribution shift, monitoring and continual learning for ML systems in general. | 22 | Partial (summaries free) | [Free summaries](https://github.com/chiphuyen/dmls-book) (the text itself is paid) |

### 2.3 Books on one stage in depth

| Book | Authors | Year | Stage | Plan ch. | Free? | Read online |
| --- | --- | --- | --- | --- | --- | --- |
| *The Smol Training Playbook* | Loubna Ben Allal et al. (Hugging Face) | 2025 | Pretraining a small model end to end | 15, 23 | Yes | [Read online](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) |
| *The Ultra-Scale Playbook* | Nouamane Tazi, Ferdinand Mom et al. (Hugging Face) | 2025 | Distributed training on GPU clusters | 7, 16 | Yes | [Read online](https://huggingface.co/spaces/nanotron/ultrascale-playbook) |
| *Reinforcement Learning from Human Feedback* | Nathan Lambert | 2026, Manning | Post-training: reward models, RLHF, DPO, reasoning | 17–19 | Yes (print paid) | [Read online](https://rlhfbook.com/) |
| *Build a Reasoning Model (From Scratch)* | Sebastian Raschka | 2026, Manning | Reasoning with verifiers and RLVR | 19 | Partial (code free) | [Book hub](https://sebastianraschka.com/reasoning-from-scratch/) (paid text) · [free code](https://github.com/rasbt/reasoning-from-scratch) |
| *Machine Learning Engineering Open Book* | Stas Bekman | continuously updated | Practitioner's reference for training and operations | 7, 16 | Yes | [Read online](https://github.com/stas00/ml-engineering) |
| *How to Scale Your Model* | Jacob Austin et al. (Google DeepMind) | 2025 | Scaling and systems on TPUs | 16 | Yes | [Read online](https://jax-ml.github.io/scaling-book/) |

---

## 3. Using them as stand-alone routes

These are four ways to follow only the resources above, each with a different goal. They are suggestions,
not routes the plan already defines.

| Route | Resources, in order | Best for | Cost | Rough time (my estimate) |
| --- | --- | --- | --- | --- |
| **A. Academic, whole stack** | CS336 lectures and assignments, with CS224N for the NLP background and J&M for reading | A reader who wants research-level depth and has the math | Free; assignments need GPUs | One to two academic quarters |
| **B. Builder** | Zero to Hero, then Raschka's book, then the HF LLM Course and the smol course | A programmer who wants to build and fine-tune a model | Free apart from the book | About 3 months at 10 h/week |
| **C. Applications** | *AI Engineering*, LLM Bootcamp, AI Agents Course, DeepLearning.AI RAG, [HF Open-Source AI Cookbook](https://huggingface.co/learn/cookbook) | A reader who ships products on existing models | Free apart from the book | About 2 months at 10 h/week |
| **D. Survey first** | CME295 recordings and its cheatsheet, then *Hands-On Large Language Models* | A reader who wants the whole picture before choosing depth | Free apart from the book | About 1 to 1.5 months |

## 4. What these alternatives do not cover

Compared with this plan, none of the full-sized resources above:
- teaches the mathematics from algebra upward (Chapters 1–5). All of them assume it, apart from the
  general-purpose books such as Prince's *Understanding Deep Learning*;
- covers GPUs and the training environment as a starting topic (Chapter 7);
- includes a research reading routine (Chapter 24);
- offers a single coherent sequence from zero to a trained, served, published model. Only CS336 and the
  capstone of this plan, built around nanochat, come close.

## 5. Caveats

1. Years for the Zero to Hero lectures, the HF LLM Course, the DeepLearning.AI RAG course and Made With ML.
   The plan does not state them.
2. Kamath et al.: I found no sign it is open access, but the Springer page needs a login redirect, so I could not open it. Its scope is also unchecked.
3. The Atkinson-Abutridy year: retailers say October 2024 and one PDF says 2025. CRC Press is from search results.
4. The time estimates in section 3 are my own, not from the sources.
