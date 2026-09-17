# Chapter 11 — NLP Foundations and Word Representations

> Part IV — Transformers and LLMs · 6–8 weeks

## What you will learn

This chapter covers the methods that natural language processing (NLP) used before
the Transformer. Several of them — next-word prediction, perplexity, word
embeddings, and attention — remain in use in large language models (LLMs). The
chapter defines language modeling as next-word prediction, with cross-entropy and
perplexity as the measures of quality, and follows language models from n-gram
counts to neural networks. It describes how word2vec and GloVe represent words as
dense vectors, how recurrent neural networks (RNNs) and long short-term memory
networks (LSTMs) process a sequence one step at a time, and how
sequence-to-sequence (seq2seq) encoder-decoder models map one sequence to another.
The chapter ends with attention, which was introduced to remove the fixed-size
bottleneck of encoder-decoder models, and with the limits of recurrence that the
Transformer of Chapter 12 removes.

## Topics

- Language modeling as next-word prediction, measured by cross-entropy and
  perplexity
- n-gram language models, smoothing, and data sparsity
- The distributional hypothesis and count-based word vectors
- Word embeddings: word2vec, with its skip-gram and continuous bag-of-words (CBOW)
  models and negative sampling, and GloVe
- Evaluation of embeddings (similarity, analogies) and the social biases they
  encode
- Neural language models, from fixed-window networks to RNNs
- Backpropagation through time, vanishing gradients, LSTMs, and gated recurrent
  units (GRUs)
- Seq2seq (encoder-decoder) models, teacher forcing, and greedy and beam-search
  decoding
- Attention, introduced to remove the fixed-size bottleneck of the encoder
- Contextual embeddings (ELMo) and the limits of recurrence that motivate the
  Transformer

## Resources

**Suggested path.** Start with Lena Voita's NLP Course (word embeddings, language
modeling, then seq2seq and attention) and read the matching chapters of Jurafsky and
Martin in parallel. Continue with CS224N Spring 2024 lectures 1, 2, 5, and 6, using
the two Alammar articles and Olah's LSTM article as visual companions to those
lectures. Implement the milestone with the code of *Dive into Deep Learning* as a
reference. When time is short, omit the DeepLearning.AI specialization, CMU 11-711,
Eisenstein, and all original papers except word2vec and Bahdanau et al.

### University courses

- Stanford — [CS224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/)
  by Diyi Yang and Yejin Choi (free; Winter 2026 slides, notes, and assignments +
  [Spring 2024 lecture videos](https://www.youtube.com/playlist?list=PLoROMvodv4rOaMFbaqxPDoLWjDaRAdP9D)
  by Christopher Manning; start here: 2024 lectures 1–2 on word vectors and
  language models, 5 on RNNs, and 6 on sequence-to-sequence models and machine
  translation; Assignment 1 explores count-based and GloVe word vectors).
- CMU — [11-711 Advanced NLP, Fall 2024 lectures](https://www.youtube.com/playlist?list=PL8PYTP1V4I8D4BeyjwWczukWq9d8PNyZp)
  by Graham Neubig (free; videos; optional: a second, research-oriented treatment
  of word representation, language modeling, and sequence models; the
  [Spring 2026 course page](https://cmu-l3.github.io/anlp-spring2026/) by Sean
  Welleck has current slides and readings).

### Online courses (MOOCs)

- Lena Voita — [NLP Course | For You](https://lena-voita.github.io/nlp_course.html)
  (free; self-paced; start here: illustrated lectures with exercises on word
  embeddings, language modeling, and seq2seq with attention, in the same order as
  this chapter's topics).
- DeepLearning.AI on Coursera — [Natural Language Processing Specialization](https://www.coursera.org/specializations/natural-language-processing)
  by Younes Bensouda Mourri and Łukasz Kaiser (paid, financial aid available;
  optional; 4 courses, about 3 months at 10 hours a week; an alternative to CS224N
  with graded labs on word vectors, n-gram autocomplete, RNNs, LSTMs, and
  attention).

### Books

- Book: Dan Jurafsky and James H. Martin, *Speech and Language Processing* (3rd ed.
  draft, August 2026 release) — [official page](https://web.stanford.edu/~jurafsky/slp3/)
  (free PDF; start here: Chapters 2 "Words and Tokens", 3 "N-gram Language Models",
  5 "Embeddings", 6 "Neural Networks", 13 "Machine Translation", and 14 "RNNs
  and LSTMs").
- Book: Aston Zhang, Zachary C. Lipton, Mu Li, and Alexander J. Smola, *Dive into
  Deep Learning* (Cambridge University Press, 2023) — [official
  page](https://d2l.ai/) (free online with runnable PyTorch code; Chapters 9–11 on
  RNNs, LSTMs, GRUs, encoder-decoder models, beam search, and attention, and Chapter
  15 on word2vec and GloVe; the code reference for the milestone).
- Book: Jacob Eisenstein, *Introduction to Natural Language Processing* (MIT
  Press, 2019) — [free draft PDF](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/notes/eisenstein-nlp-notes.pdf)
  (free; optional: a more mathematical treatment; Chapter 6 "Language models",
  Chapter 14 "Distributional and distributed semantics", and Chapter 18 "Machine
  translation").

### Lectures, papers and articles

- Article: Jay Alammar, [The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/)
  (2019; start here: an illustrated explanation of embeddings, skip-gram, and
  negative sampling).
- Paper: Bengio et al., [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/v3/bengio03a.html)
  (2003; a feed-forward language model that learns word embeddings jointly with
  the network weights; the fixed-window model that RNNs replaced).
- Paper: Mikolov et al., [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
  and [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)
  (2013; word2vec: skip-gram, CBOW, and negative sampling).
- Paper: Pennington, Socher, and Manning, [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
  (2014; embeddings fitted to global co-occurrence counts; the project page has
  the paper and pretrained vectors).
- Article: Andrej Karpathy, [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
  (2015; character-level RNN language models, with generated samples and an
  analysis of what individual neurons track).
- Article: Christopher Olah, [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
  (2015; step-by-step diagrams of the LSTM gates and of the GRU variant).
- Paper: Sutskever, Vinyals, and Le, [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
  and Bahdanau, Cho, and Bengio, [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
  (2014; the encoder-decoder model, and the attention mechanism that the
  Transformer generalizes).
- Article: Jay Alammar, [Visualizing A Neural Machine Translation Model](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
  (2018; seq2seq with attention, animated step by step; a companion to the two
  papers above).
- Paper: Peters et al., [Deep contextualized word representations](https://arxiv.org/abs/1802.05365)
  (ELMo, 2018; optional: word vectors that depend on the sentence, computed by a
  pretrained bidirectional LSTM language model; the intermediate step between
  static embeddings and pretrained Transformers).

## Milestone

Implement skip-gram with negative sampling in PyTorch, train it on a small corpus
such as text8, and verify that related words cluster, using nearest-neighbor lists
for ten probe words and a two-dimensional principal component analysis (PCA) plot.
Then train a small LSTM language model on the same corpus and report its held-out
perplexity next to that of a bigram baseline.

## Estimated time

6–8 weeks.
