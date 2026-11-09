+++
title = "kael.cpp"
sort_by = "weight"

[extra]
tagline = "a deep learning framework built from scratch in C++, explained one layer at a time"
after_book = """
## a growing knowledge base

The book explains how kael works. The [concepts](@/concepts/_index.md) section explains the ideas around it: the math behind gradients, how memory hierarchies shape kernels, why attention works. It is written by the community and it is meant to keep growing.

Every page on this site is a Markdown file in a public repository. Fix a sentence, add an exercise, or write the article you wish you had read when you started. Read [how to contribute](@/contributing.md), or [request a topic](https://github.com/JetBerri/kael.cpp-web/issues/new/choose) you want explained.
"""
+++

## introduction

kael is a deep learning framework written from zero in modern C++. It has tensors with shapes and strides, views that share memory, a loop engine with broadcasting and type promotion, reverse-mode autograd, neural network modules and optimizers, a multithreaded CPU backend, a CUDA backend, Python bindings, and a GPT-2 that loads real weights and generates text.

None of it is a wrapper. Every layer is written by hand, tested against PyTorch, and explained in a book that rebuilds it step by step.

```cpp
#include <iostream>

#include "kael/kael.h"

int main() {
    kael::Tensor x = kael::randn({64, 784});
    kael::Tensor w = kael::randn({784, 10}).requires_grad_();

    kael::Tensor loss = kael::relu(kael::matmul(x, w)).sum();
    loss.backward();

    std::cout << *w.grad() << "\n";   // dloss/dw, shape=[784, 10]
}
```

## motivation

- Why does `reshape` sometimes copy and sometimes not?
- How does `a + b` work when `a` is `[3, 1]` and `b` is `[4]`?
- What does `loss.backward()` actually walk, and in which order?
- Why is a naive matrix multiply fifty times slower than a good one?
- What happens between a `.safetensors` file on disk and a sentence of generated text?

Frameworks such as PyTorch answer these questions in millions of lines of code. kael answers them in a codebase small enough to read in a few weeks, with every decision written down.

## quick start

kael needs CMake 3.24 or newer and a C++20 compiler.

```bash
git clone https://github.com/JetBerri/kael.cpp
cd kael.cpp
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Then start with the [overview](@/book/00-overview.md).
