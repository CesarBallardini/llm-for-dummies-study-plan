# Chapter 13 — Tokens and Embeddings: How Text Becomes Numbers

> Part IV — Transformers and LLMs · 3–4 weeks

## What you will learn

This chapter covers the two steps that convert text into the input of a
Transformer: tokenization, which splits text into units from a fixed vocabulary, and
embedding, which maps each unit to a vector. It describes the byte pair encoding
(BPE), WordPiece, and Unigram algorithms, the byte-level BPE variant used by
GPT-style models, and the effect of vocabulary size and training corpus on sequence
length, cost, and model quality. Several known failures of large language models
(LLMs) — in spelling, arithmetic, and non-English text — originate in the tokenizer,
and the chapter shows how to diagnose them. It ends with the embedding table, its
relation to the output layer, and the data loader that converts a tokenized corpus
into the training batches used in Chapter 14.

## Topics

- Unicode, UTF-8 bytes, and text normalization
- Word, character, byte, and subword tokenization, and their trade-offs
- Byte pair encoding (BPE): training, encoding, and decoding
- Byte-level BPE with a pre-tokenization regular expression (GPT-2, tiktoken)
- WordPiece, the Unigram language model, and the SentencePiece library
- Vocabulary size, special tokens, and compression ratio
- The dependence of a tokenizer on its training corpus, and the resulting cost
  differences between languages
- Model failures caused by tokenization (spelling, arithmetic, trailing whitespace,
  undertrained tokens)
- The token embedding table, its relation to the positional embeddings of Chapter
  12, and weight tying with the output layer
- Tokenized datasets: sliding-window input-target pairs and batches

## Resources

**Suggested path.** Start with Karpathy's "Let's build the GPT Tokenizer" with
Tiktokenizer open in a browser, then complete the minbpe exercise, on which the
milestone is based. Read Raschka Chapter 2 for the embedding table and the data
loader, and Sennrich et al. and Section 2.2 of the GPT-2 report for the original
sources. Work through Chapter 6 of the Hugging Face LLM Course in parallel for
WordPiece, Unigram, and the production libraries. CS336 lecture 1 summarizes the
chapter in one lecture; the tokenizer part of its Assignment 1 is an alternative
specification of the milestone with stricter tests. When time is short, omit that
assignment and all papers marked optional.

### University courses

- Stanford — [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
  by Tatsunori Hashimoto and Percy Liang (free; Spring 2026
  [lecture videos](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)
  and assignments; start here: lecture 1 "Overview, tokenization"; the first part of
  [Assignment 1 (Basics)](https://github.com/stanford-cs336/assignment1-basics)
  specifies a byte-level BPE tokenizer with unit tests and speed targets; the
  Transformer part of the assignment is scheduled in Chapter 14).

### Online courses (MOOCs)

- Andrej Karpathy — [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
  (free; 2h13m; Lecture 8 of [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html);
  start here: builds a byte-level BPE tokenizer from scratch, reproduces the GPT-4
  tokenizer, and traces a list of LLM failures to tokenization; the code and a
  five-step [exercise](https://github.com/karpathy/minbpe/blob/master/exercise.md)
  are in [minbpe](https://github.com/karpathy/minbpe)).
- Hugging Face — [LLM Course, Chapter 6: The Tokenizers library](https://huggingface.co/learn/llm-course/chapter6/1)
  (free; a few hours; trains a new tokenizer from an existing one, and implements
  BPE, WordPiece, and Unigram step by step).

### Books

- Book: Sebastian Raschka, *Build a Large Language Model (From Scratch)* (Manning,
  2024) — [official page](https://www.manning.com/books/build-a-large-language-model-from-scratch)
  (paid; free code at [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
  and free [companion videos](https://www.youtube.com/playlist?list=PLTKMiZHVd_2IIEsoJrWACkIxLRdfMlw11);
  start here: Chapter 2 "Working with Text Data" covers tokenization, special
  tokens, sliding-window data loaders, and token and positional embeddings; the
  repository adds a BPE tokenizer from scratch as bonus material).
- Book: Dan Jurafsky and James H. Martin, *Speech and Language Processing* (3rd ed.
  draft, August 2026 release) — [official page](https://web.stanford.edu/~jurafsky/slp3/)
  (free PDF; Chapter 2 "Words and Tokens" for Unicode, BPE, and text normalization;
  Chapter 5 "Embeddings", read in Chapter 11, for the meaning of the vectors).
- Book: Jay Alammar and Maarten Grootendorst, *Hands-On Large Language Models*
  (O'Reilly, 2024) — [official page](https://llm-book.com/) (paid; free notebooks on
  [GitHub](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models); Chapter 2
  "Tokens and Embeddings" compares the tokenizers of released models and relates
  token, word, and text embeddings; the tokenizer lessons of the authors' short
  course listed in Chapter 12 cover the same material on video).

### Lectures, papers and articles

- Paper: Sennrich, Haddow, and Birch, [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
  (2015; start here: the paper that adapted BPE, a data-compression algorithm, to
  the segmentation of words into subword units for machine translation).
- Paper: Radford et al., [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
  (GPT-2, 2019; Section 2.2 "Input Representation" describes byte-level BPE and the
  rule that prevents merges across character categories; the released code
  implements that rule as a pre-tokenization regular expression, and later GPT
  tokenizers use variants of it).
- Paper: Kudo, [Subword Regularization](https://arxiv.org/abs/1804.10959) (2018)
  and Kudo and Richardson, [SentencePiece](https://arxiv.org/abs/1808.06226)
  (2018; the Unigram language model tokenizer, and the language-independent
  trainer that works on raw text, used by T5, Llama 2, and Gemma).
- Paper: Wu et al., [Google's Neural Machine Translation System](https://arxiv.org/abs/1609.08144)
  (2016; optional: Section 4.1 defines the WordPiece model that BERT uses).
- Paper: Press and Wolf, [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859)
  (2016; optional: weight tying between the embedding table and the output layer).
- Paper: Petrov et al., [Language Model Tokenizers Introduce Unfairness Between Languages](https://arxiv.org/abs/2305.15425)
  (2023; optional: measures how many more tokens, and therefore how much more
  cost, the same sentence requires in languages that are rare in the tokenizer's
  training corpus).
- Paper: Xue et al., [ByT5: Towards a token-free future with pre-trained byte-to-byte models](https://arxiv.org/abs/2105.13626)
  (2021; optional: a model that reads raw bytes without a tokenizer, and the
  sequence-length cost of doing so).

### Tools and hands-on

- Tool: [Tiktokenizer](https://tiktokenizer.vercel.app/) (free web page; start
  here: shows how production tokenizers such as GPT-4o's split any pasted text;
  the library behind it is OpenAI's [tiktoken](https://github.com/openai/tiktoken)).
- Code: Hugging Face, [tokenizers](https://github.com/huggingface/tokenizers) and
  Google, [SentencePiece](https://github.com/google/sentencepiece) (the two
  libraries with which most open models train their tokenizers; used for
  comparison after the from-scratch implementation).

## Milestone

Implement a byte-level BPE tokenizer from scratch (the minbpe exercise is a suitable
specification), train it with a vocabulary of about 8,000 tokens on a public corpus,
and verify that encoding followed by decoding returns byte-identical text, including
emoji and non-English samples. Report the compression ratio in bytes per token on
held-out English text and on text in one other language, next to the ratios of the
GPT-2 tokenizer from tiktoken, and explain the differences.

## Estimated time

3–4 weeks.
