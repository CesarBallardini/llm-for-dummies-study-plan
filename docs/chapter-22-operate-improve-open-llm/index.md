# Chapter 22 — Operating and Improving an Open-Source LLM

> Part VI — Evaluation, Deployment, and the Ecosystem · 6–8 weeks

## What you will learn

This chapter combines the methods of Chapters 17–21 into the operating cycle of an
application built on an open large language model (LLM): select a base model, adapt
it to a task, serve it, evaluate it, and improve it from observed failures. It
describes the open-model ecosystem — the main model families, the Hugging Face Hub,
model cards, and licenses — and the difference between a model with open weights and
a fully open model, for which the training data and code are also published. It
presents the adaptation methods in order of increasing cost: prompt design;
retrieval-augmented generation (RAG), which inserts retrieved documents into the
prompt at inference time; fine-tuning; and continued pretraining. It introduces
agents, in which the model calls external tools over several steps. It also covers
the operations work: versioning, regression evaluations in continuous integration
(CI), guardrails, monitoring, and the conversion of logged failures into evaluation
cases and training data. The Chapter 23 capstone applies the same cycle to a model
trained from scratch.

## Topics

- The open-model ecosystem: the Llama, Mistral, Qwen, DeepSeek, Gemma, and OLMo
  families
- Degrees of openness: open weights, open training data, and open training code, and
  the Open Source AI Definition
- Model hubs (Hugging Face Hub), model cards, and licenses
- Selection of a base model: size, license, context length, benchmark results
  (Chapter 20), and serving cost (Chapter 21)
- Choice of an adaptation method: prompting, retrieval, fine-tuning (Chapter 17), or
  continued pretraining
- Prompt design: system prompts, few-shot examples, and structured output
- Retrieval-augmented generation (RAG): chunking, text-embedding models, vector
  search, hybrid search, and reranking
- Evaluation of a RAG system: retrieval metrics and the faithfulness of the answer to
  the retrieved text
- Agents: tool calling, the distinction between workflows and agents, and the
  evaluation of multi-step tasks
- Guardrails: validation of inputs and outputs, and prompt injection
- Versioning of data, prompts, adapters, and checkpoints, with regression evaluations
  in continuous integration (CI)
- Monitoring in production: logging and tracing, latency and cost, output quality,
  and distribution shift
- The improvement cycle: user feedback, error analysis, and the conversion of
  failures into evaluation cases and training data

## Resources

**Suggested path.** Start with Yan's "Patterns" article and the LLM Bootcamp for an
overview, then read chapters 5, 6, and 10 of *AI Engineering* as the main text.
Select the task and the base model of the milestone during the first two weeks, using
the Hub documentation and the Open Source AI Definition to check the license and the
degree of openness; then build the retrieval path with the Open-Source AI Cookbook
and chapter 8 of *Hands-On Large Language Models*. Follow the AI Agents Course in
parallel, and read Husain, the Yan et al. report, and the monitoring chapters of
*Designing Machine Learning Systems* while the system is evaluated and improved; the
CS329S lecture notes are a free substitute for that book. When time is short, omit
the DeepLearning.AI RAG course, Made With ML, and the survey by Gao et al. Read the
OWASP list and the context-engineering article in the agents and guardrails weeks;
Langfuse is one choice of tracing tool for the monitoring weeks.

### University courses

- Stanford —
  [CS329S: Machine Learning Systems Design](https://stanford-cs329s.github.io/) by
  Chip Huyen (free; Winter 2022 slides, lecture notes, and assignments; start here:
  the notes on deployment, on data distribution shifts and monitoring, and on
  continual learning; the course predates LLM applications and treats these subjects
  for machine learning systems in general; *Designing Machine Learning Systems*,
  listed under Books, is based on it).
- Stanford — [CME295: Transformers & Large Language
  Models](https://cme295.stanford.edu/) by Afshine Amidi and Shervine Amidi (free;
  optional; first listed in Chapter 12; in the [Fall 2026
  syllabus](https://cme295.stanford.edu/syllabus/), lecture 6 "AI Agents" covers tool
  calling, the Model Context Protocol (MCP), memory and retrieval, context
  compaction, and coding agents, and is more current than the 2023 LLM Bootcamp
  listed under MOOCs; lecture 7 "Agentic LLMs" of the [Autumn 2025
  recordings](https://www.youtube.com/watch?v=h-7S6HNq0Vg) covers RAG, function
  calling, and ReAct).

### Online courses (MOOCs)

- Full Stack Deep Learning —
  [LLM Bootcamp](https://fullstackdeeplearning.com/llm-bootcamp/) by Charles Frye,
  Sergey Karayev, and Josh Tobin (free; recorded in Spring 2023; start here: lectures
  on prompt engineering, LLMOps, augmented language models, and user interfaces for
  language models, and a walkthrough of a deployed question-answering application;
  the tools shown are those of 2023).
- Hugging Face — [AI Agents Course](https://huggingface.co/learn/agents-course)
  (free; 2025; agents built with smolagents, LlamaIndex, and LangGraph; bonus units
  on fine-tuning a model for function calling and on the observability and evaluation
  of agents).
- DeepLearning.AI —
  [Retrieval Augmented Generation (RAG)](https://www.deeplearning.ai/courses/retrieval-augmented-generation)
  by Zain Hasan (free to audit, paid certificate; about 26 hours; optional: keyword,
  semantic, and hybrid search, vector databases, and the evaluation and production
  monitoring of a RAG system).
- Goku Mohandas — [Made With ML](https://madewithml.com/) (free; optional: a course
  in machine learning operations (MLOps) that covers the testing of code, data, and
  models, continuous integration and deployment, serving, and monitoring, built
  around the fine-tuning of an LLM for a text-classification task).

### Books

- Book: Chip Huyen, *AI Engineering* (O'Reilly, 2025) —
  [book repo](https://github.com/chiphuyen/aie-book) (paid book, free chapter
  summaries; start here: chapter 5, "Prompt Engineering"; chapter 6, "RAG and
  Agents"; and chapter 10, "AI Engineering Architecture and User Feedback"; chapter
  7, "Finetuning", and chapter 8, "Dataset Engineering", were assigned in Chapter
  17).
- Book: Chip Huyen, *Designing Machine Learning Systems* (O'Reilly, 2022) —
  [book repo](https://github.com/chiphuyen/dmls-book) (paid book, free chapter
  summaries and a list of MLOps tools; chapter 8, "Data Distribution Shifts and
  Monitoring", and chapter 9, "Continual Learning and Test in Production", cover
  subjects that *AI Engineering* treats only briefly).
- Book: Jay Alammar and Maarten Grootendorst, *Hands-On Large Language Models*
  (O'Reilly, 2024) — [official page](https://llm-book.com/) (paid; free notebooks on
  [GitHub](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models); chapter 6,
  "Prompt Engineering"; chapter 7, "Advanced Text Generation Techniques and Tools";
  and chapter 8, "Semantic Search and Retrieval-Augmented Generation", each with
  runnable code).
- Book: Dan Jurafsky and James H. Martin, *Speech and Language Processing* (3rd ed.
  draft, August 2026 release) — [official
  page](https://web.stanford.edu/~jurafsky/slp3/) (free PDF; Chapter 11 "Information
  Retrieval and RAG": classical and dense retrieval, the evaluation of retrieval
  systems, retrieval-augmented generation, and the evaluation of question answering;
  the textbook counterpart of the retrieval path of the milestone).

### Lectures, papers and articles

- Article: Eugene Yan,
  [Patterns for Building LLM-based Systems & Products](https://eugeneyan.com/writing/llm-patterns/)
  (2023; start here: seven patterns — evaluations, RAG, fine-tuning, caching,
  guardrails, defensive user-interface design, and collection of user feedback — each
  with the conditions under which it applies).
- Article: Eugene Yan, Bryan Bischof, Charles Frye, Hamel Husain, Jason Liu, and
  Shreya Shankar,
  [What We've Learned From a Year of Building with LLMs](https://applied-llms.org/)
  (2024; practices reported by six practitioners, grouped as tactical — prompting,
  RAG, and evaluation — operational, and strategic).
- Article: Hamel Husain,
  [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/) (2024;
  introduced in Chapter 20; describes the improvement cycle of this chapter: logged
  traces are reviewed, failures become test cases, and the tests run on every
  change).
- Papers: Lewis et al.,
  [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
  (2020; the paper that introduced the term, with a retriever and a generator trained
  jointly) and Gao et al.,
  [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997)
  (2023; the retrieval, augmentation, and generation methods of later systems, and
  the metrics used to evaluate them).
- Article: Erik Schluntz and Barry Zhang,
  [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
  (Anthropic, 2024; defines workflows and agents, describes five workflow patterns,
  and states the conditions under which an agent is justified), with the paper Yao et
  al.,
  [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
  (2022; the loop of reasoning step, tool call, and observation that agent frameworks
  implement).
- Article: Open Source Initiative,
  [The Open Source AI Definition 1.0](https://opensource.org/ai/open-source-ai-definition)
  (2024; the requirements on data information, code, and parameters for a model to be
  called open source), with an example of a fully open model: Ai2,
  [OLMo](https://allenai.org/olmo) (weights, training data, code, intermediate
  checkpoints, and evaluation tools are published), described in Groeneveld et al.,
  [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838)
  (2024).
- Docs: [Model Context Protocol](https://modelcontextprotocol.io/) (free, open
  standard, introduced by Anthropic in 2024; the specification and documentation of
  the protocol through which an agent discovers and calls the tools, resources, and
  prompts exposed by a server; the interface behind the tool-calling topic in current
  agent frameworks).
- Article: Anthropic Applied AI team, [Effective context engineering for AI
  agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
  (2025; the sequel to Building Effective Agents: the context window as the scarce
  resource of an agent, and compaction, structured note-taking, sub-agents, and
  just-in-time retrieval as the methods that manage it).
- Article: OWASP, [Top 10 for LLM Applications
  2025](https://genai.owasp.org/llm-top-10/) (free; the ten risk classes of an LLM
  application, from prompt injection and sensitive information disclosure to
  excessive agency and unbounded consumption; a checklist for the guardrails topic
  and the milestone), with Simon Willison's [prompt injection
  series](https://simonwillison.net/series/prompt-injection/) (2022 onward; the
  running record of the attack class, why input filtering does not solve it, and the
  design rules that limit the damage when an agent reads untrusted text).

### Tools and hands-on

- Hands-on: Hugging Face,
  [Open-Source AI Cookbook](https://huggingface.co/learn/cookbook) (free; start here:
  notebooks on RAG with open models, RAG evaluation, LLM-as-a-judge, agents, and
  deployment; the RAG notebooks serve as the basis of the retrieval path of the
  milestone).
- Docs: Hugging Face, [Hub documentation](https://huggingface.co/docs/hub/index)
  (free; repositories, revisions, and Spaces), with its
  [model cards guide](https://huggingface.co/docs/hub/model-cards) and the paper that
  proposed model cards, Mitchell et al.,
  [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) (2018; the
  intended use, evaluation data, and limitations that the publisher of a model
  documents).
- Tool: [Langfuse](https://langfuse.com/docs) (open source, self-hostable; tracing of
  prompts, retrieval calls, and model outputs, prompt versioning, and scoring of
  logged traces; one implementation of the logging, tracing, and feedback topics,
  usable to monitor the milestone system).

## Milestone

Build a question-answering system over a document collection of at least 100 pages,
on an open instruction-tuned model of 1–3 billion parameters (for example
Qwen2.5-1.5B-Instruct) that fits a free Colab or Kaggle T4 GPU. Write a held-out test
set of at least 100 questions with reference answers, and measure four configurations
— the base model, the base model with retrieval, a low-rank adaptation (LoRA)
fine-tune (Chapter 17), and the fine-tune with retrieval — with the model served by
vLLM (Chapter 21) and scored by one method of Chapter 20, either a custom
lm-evaluation-harness task or a judge model. Publish the adapter on the Hugging Face
Hub with a model card, and report the score, its standard error, and the mean latency
of each configuration, together with an error analysis of 20 failed answers and the
change that each failure suggests.

## Estimated time

6–8 weeks.
