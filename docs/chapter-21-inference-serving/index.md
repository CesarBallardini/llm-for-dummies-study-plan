# Chapter 21 — Inference Optimization and Serving

> Part VI — Evaluation, Deployment, and the Ecosystem · 4–5 weeks

## What you will learn

This chapter covers the methods used to run a trained open large language model (LLM)
at low cost and low latency, and the software that serves it to users. It describes
the two phases of text generation — prefill, which processes the prompt in parallel
and is limited by compute, and decode, which produces one token at a time and is
limited by memory bandwidth — and the key-value (KV) cache, which avoids recomputing
attention over earlier tokens at the cost of memory that grows with batch size and
context length. It then presents the main optimizations: continuous batching, paged
attention, FlashAttention, quantization of the weights to 8 or 4 bits, and
speculative decoding, each with its effect on memory, speed, and accuracy. It
introduces the serving engines vLLM, SGLang, and llama.cpp, and the metrics used to
benchmark them: time to first token (TTFT), time per output token (TPOT), and
throughput. The accuracy cost of quantization is measured with the methods of Chapter
20, and the serving setup is used again in Chapters 22 and 23.

## Topics

- The two phases of generation: prefill (compute-bound) and decode (memory-bound), in
  the workload classes of Chapter 7
- Inference arithmetic: memory for the weights and the KV cache, floating-point
  operations (FLOPs) per token, and the memory-bandwidth limit on decode speed
- The KV cache and the methods that reduce its size: multi-query and grouped-query
  attention (Chapter 12), and KV-cache quantization
- Serving metrics: time to first token (TTFT), time per output token (TPOT),
  throughput in tokens per second, and the trade-off between latency and throughput
- Batching of requests: static, dynamic, and continuous batching
- Paged attention and prefix caching
- FlashAttention (Chapter 16) in prefill and decode
- Post-training quantization of weights: LLM.int8() and 4-bit NormalFloat (NF4) in
  bitsandbytes, GPTQ, AWQ, and the GGUF formats of llama.cpp
- Quantization of activations and the outlier problem (SmoothQuant)
- Measurement of the accuracy cost of quantization with perplexity and benchmark
  scores (Chapter 20)
- Speculative decoding with a draft model
- Serving engines: vLLM, SGLang, llama.cpp, and the OpenAI-compatible HTTP interface
- Inference on several GPUs: tensor parallelism (Chapter 16) and disaggregated
  prefill and decode (overview)

## Resources

**Suggested path.** Start with the DeepLearning.AI short course Efficiently Serving
LLMs and CS336 lecture 10 for the vocabulary; then read chapter 9 of *AI Engineering*
and kipply's article for the inference arithmetic, and implement a KV cache with
Raschka's article. Install llama.cpp and vLLM during the first week and run a small
model on each, because the milestone depends on them. Then study quantization (the
two DeepLearning.AI quantization courses, MIT 6.5940 lectures 5–6, and the GPTQ and
AWQ papers), followed by the PagedAttention, FlashAttention, and speculative decoding
papers. When time is short, read only the abstract and the method section of each
paper, and omit Pope et al., CMU 11-868, and SGLang.

### University courses

- Stanford — [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) by
  Percy Liang and Tatsunori Hashimoto (free; Spring 2026 lecture videos on a
  [YouTube playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV) +
  public assignments; start here: lecture 10, "Inference", covers the prefill and
  decode workloads, the arithmetic of the KV cache, architectural and quantization
  methods that reduce it, speculative decoding, and continuous batching with paged
  attention).
- MIT —
  [6.5940: TinyML and Efficient Deep Learning Computing](https://hanlab.mit.edu/courses/2024-fall-65940)
  by Song Han (free; Fall 2024 slides, labs, and a public
  [lecture playlist](https://www.youtube.com/playlist?list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB);
  lectures 5–6 cover quantization, lecture 13 efficient LLM deployment, and lecture
  15 long-context LLMs; the labs include compression of an LLM and deployment of
  Llama-2-7B on a laptop).
- CMU —
  [11-868: Large Language Model Systems](https://llmsystem.github.io/llmsystem2026spring/)
  by Lei Li (free; Spring 2026 slides; optional; lectures on paged attention and
  vLLM, KV-cache reuse, quantization, speculative decoding, and disaggregated prefill
  and decode).

### Online courses (MOOCs)

- DeepLearning.AI —
  [Efficiently Serving LLMs](https://www.deeplearning.ai/courses/efficiently-serving-llms)
  by Travis Addair of Predibase (free; 2 hours; start here: KV caching, continuous
  batching, quantization, and the serving of many low-rank adaptation (LoRA) adapters
  (Chapter 17) on one base model, each implemented in code).
- DeepLearning.AI —
  [Quantization Fundamentals with Hugging Face](https://www.deeplearning.ai/courses/quantization-fundamentals)
  by Younes Belkada and Marc Sun (free; 1 hour 15 minutes; data types, downcasting,
  and loading of quantized models with `transformers` and Quanto), followed by
  [Quantization in Depth](https://www.deeplearning.ai/courses/quantization-in-depth)
  by the same instructors (free; 2 hours 20 minutes; optional: implementation of a
  linear quantizer down to 2-bit weights, with measurement of the quantization
  error).

### Books

- Book: Chip Huyen, *AI Engineering* (O'Reilly, 2025) —
  [book repo](https://github.com/chiphuyen/aie-book) (paid book, free chapter
  summaries; start here: chapter 9, "Inference Optimization", covers the bottlenecks
  of inference, the performance metrics, and optimization at the model level and at
  the service level; its "AI Accelerators" section was assigned in Chapter 7).

### Lectures, papers and articles

- Article: kipply,
  [Transformer Inference Arithmetic](https://kipp.ly/p/transformer-inference-arithmetic)
  (2022; start here: derives the size of the KV cache, the memory-bound and
  compute-bound regimes, and latency estimates from the parameter count, the FLOPs,
  and the memory bandwidth of the accelerator).
- Article: Sebastian Raschka,
  [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms)
  (2025; adds a KV cache to the GPT model of Chapter 14; reports a speedup of about
  5x on a 124-million-parameter model and states the memory cost).
- Article: Patrick von Platen,
  [Optimizing your LLM in Production](https://huggingface.co/blog/optimize-llm)
  (Hugging Face, 2023; measured GPU memory and latency for 8-bit and 4-bit weights,
  FlashAttention, and multi-query and grouped-query attention).
- Paper: Pope et al.,
  [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)
  (2022; advanced: a cost model of memory-bound decoding, the partitioning of a model
  over many accelerators, and the trade-off between latency and throughput).
- Papers: Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with
  IO-Awareness](https://arxiv.org/abs/2205.14135) (2022) and Tri Dao,
  [FlashAttention-2](https://arxiv.org/abs/2307.08691) (2023; exact attention
  computed in blocks that fit in on-chip memory, which reduces memory traffic without
  approximation; introduced in Chapter 16).
- Papers: post-training quantization — Dettmers et al.,
  [LLM.int8()](https://arxiv.org/abs/2208.07339) (2022; the 8-bit method of
  bitsandbytes); Frantar et al., [GPTQ](https://arxiv.org/abs/2210.17323) (2022); Lin
  et al., [AWQ](https://arxiv.org/abs/2306.00978) (2023); and Xiao et al.,
  [SmoothQuant](https://arxiv.org/abs/2211.10438) (2022; the activation outliers that
  make 8-bit activations difficult, and a rescaling that transfers the difficulty to
  the weights).
- Papers: speculative decoding — Leviathan et al.,
  [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
  (2022) and Chen et al.,
  [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
  (2023; two independent derivations of one method: a small draft model proposes
  several tokens, the large model verifies them in one pass, and the output
  distribution is proved unchanged).

### Tools and hands-on

- Tool: ggml-org, [llama.cpp](https://github.com/ggml-org/llama.cpp) (free, open
  source; start here: CPU and GPU inference in C/C++ with GGUF quantization from 8
  bits to about 2 bits; provides `llama-server`, `llama-bench`, and
  `llama-perplexity`, which the milestone uses; [Ollama](https://ollama.com/)
  packages the same engine with a model downloader).
- Tool: [vLLM](https://docs.vllm.ai/en/latest/) (free, open source; a GPU serving
  engine with continuous batching, prefix caching, quantized models, and an
  OpenAI-compatible server; used again in Chapter 22), with its paper, Kwon et al.,
  [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
  (2023; KV-cache memory allocated in fixed-size blocks, as in the virtual memory of
  an operating system, which reduces fragmentation and permits larger batches).
- Tool: [SGLang](https://docs.sglang.io/) (free, open source; optional: a GPU serving
  engine with RadixAttention for prefix reuse, described in Zheng et al.,
  [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104),
  2023). NVIDIA [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) (advanced)
  runs on NVIDIA GPUs only, and Hugging Face
  [TGI](https://huggingface.co/docs/text-generation-inference) is in maintenance
  mode; its documentation refers users to vLLM and SGLang.

## Milestone

Run an open model of 7–8 billion parameters with llama.cpp on a single machine at
three GGUF quantization levels, for example Q8_0, Q4_K_M, and Q2_K; a free Colab or
Kaggle T4 GPU is sufficient, and a 3-billion-parameter model can be substituted on a
machine with less than 16 GB of memory. For each level, measure prefill and decode
speed in tokens per second with `llama-bench` at prompt lengths of 512 and 4,096
tokens, measure perplexity on the WikiText-2 test set with `llama-perplexity`, and
plot perplexity against decode speed. Compare the measured decode speed with the
estimate of memory bandwidth divided by model size in bytes, then start
`llama-server`, send one request through its OpenAI-compatible interface, and record
the time to first token.

## Estimated time

4–5 weeks.
