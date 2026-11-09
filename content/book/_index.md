+++
title = "The book"
description = "Rebuild a deep learning framework from a flat buffer of floats to GPT-2, one layer per chapter."
sort_by = "weight"
weight = 1
+++

This book explains how kael works and how to rebuild it. It starts with a flat block of memory and a list of strides and ends with a GPT-2 checkpoint that loads, tokenizes text and generates it. Each chapter adds one layer on top of the previous ones, walks through the real code and ends with steps to write that layer yourself.

## Who it is for

Engineers who write C++ and want to understand what a deep learning framework does internally. The book assumes C++20 and basic linear algebra: vectors, matrices and matrix multiplication. It does not assume prior knowledge of PyTorch internals, automatic differentiation, CUDA or transformers. Each of those is introduced from first principles.

## How to read it

Read the chapters in order. Each one depends on the layers before it.

Most chapters match a git tag in the library repository. Open the tag in a separate worktree to see the code exactly as it was when that chapter was complete, and read the chapter next to it:

```bash
git clone https://github.com/JetBerri/kael.cpp
cd kael.cpp
git worktree add ../kael-views chapter-04-views
cmake -S ../kael-views -B ../kael-views/build
cmake --build ../kael-views/build
ctest --test-dir ../kael-views/build
```

Tags are numbered by project phase, so `chapter-02-shapes` holds the code for chapter 01 and `chapter-03-core` the code for chapter 02. Chapter 00 has no tag. Chapter 12 starts at `chapter-12-transformer` and is complete at `chapter-13-polish`, which also marks `v1.0.0`.

## How each chapter is organized

| Section | What it gives you |
|---------|-------------------|
| Why this layer exists | The problem the layer solves and a mental model to hold on to |
| Concepts | The idea from first principles, with worked numeric examples and diagrams |
| Implementation | The real types and functions, in the order you would write them |
| Edge cases and pitfalls | Failure modes the code guards against, and how |
| How it is tested | The test suites and golden data that prove the layer correct |
| Build it yourself | Ordered steps to reimplement the layer, each with a checkpoint |
| Check your understanding | Short questions with hidden answers |

## Conventions

- Paths are relative to the root of the library repository.
- Code excerpts are trimmed for reading and name the file they come from. When an excerpt and the code disagree, the code wins.
- Shapes are written `[2, 3]` and strides `(3, 1)`. `numel` is the number of elements.

## Chapters
