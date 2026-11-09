+++
title = "Concepts"
description = "Community articles on the ideas behind deep learning systems: math, hardware, algorithms and trade-offs."
sort_by = "weight"
weight = 2
+++

The book follows kael one layer at a time. This section steps back and explains the ideas those layers rely on. Articles here are not tied to the order of the book, and many of them apply to any framework.

This section is written by the community. If a topic is missing, [request it](https://github.com/JetBerri/kael.cpp-web/issues/new/choose) or [write it](@/contributing.md).

## Wanted articles

These topics have no article yet. Each one is a good first contribution. Open an issue to claim one so two people do not write the same page.

| Topic | What the article should explain |
|-------|---------------------------------|
| The memory hierarchy | Registers, caches, DRAM and HBM, and why data movement dominates kernel time |
| Numerical stability | log-sum-exp, stable softmax, catastrophic cancellation, epsilon terms |
| Initialization | Why weight scale matters, Xavier and Kaiming initialization derived |
| Normalization layers | Batch norm, layer norm and RMS norm, what each normalizes and why |
| Optimizers in depth | Momentum, Adam and AdamW derived, and what weight decay really does |
| Tokenization | Characters, words, BPE and why tokenizers shape model behavior |
| Attention variants | Multi-query, grouped-query and sliding window attention, and their memory costs |
| Quantization | Int8 and int4 weights, scales and zero points, and the accuracy trade-off |
| Mixed precision training | Master weights, loss scaling and which ops must stay in float32 |
| Distributed training | Data, tensor and pipeline parallelism, and the communication each one needs |
| Reading a paper into code | A method to turn equations in a paper into a tested implementation |

## Articles
