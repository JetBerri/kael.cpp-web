+++
title = "Overview"
description = "What kael is, how its layers depend on each other, where each piece lives in the repository, and how to build and test it."
weight = 0
slug = "overview"

[extra]
chapter = "00"
kicker = "Chapter 00"
builds = "The architecture, the repository map and the build"
source = "docs/00-overview.md"
+++

kael is a deep learning framework written from scratch in C++20. It starts with the arithmetic of shapes and strides and ends with a GPT-2 model that loads published weights, tokenizes text and generates tokens, from C++, from Python and over HTTP. This book explains every layer in the order it was built, so that an engineer who knows C++ but has never looked inside a tensor library can rebuild kael from an empty directory.

This chapter is the map. Read it first and come back to it whenever a later chapter mentions a file or a layer you cannot place. It states the goals, follows one line of user code through every layer, draws how the layers depend on each other, lists where each piece lives in the repository, gives the build and test commands, and ties every chapter to a git tag that holds the code at the end of that stage.

## Why this book exists

Using PyTorch or NumPy teaches you what a tensor does. It does not teach you what a tensor is. The interesting parts of a framework are hidden behind a Python call: how `x.T` returns instantly without moving data, how `a + b` works when `a` is `[2, 3]` and `b` is `[3]`, how `loss.backward()` knows which operations produced `loss`, and why the second training step allocates no memory. Reading the PyTorch source answers those questions in principle, but the answer is spread over millions of lines, code generators and several languages.

kael answers them in a code base small enough to read end to end. Every mechanism exists once, in plain C++, with tests that pin down its behavior against NumPy or PyTorch. The mental model to hold on to for the whole book:

```text
a tensor = a flat buffer of bytes
         + metadata that says how to read it (shape, strides, offset, dtype, device)
         + optionally, a link to the operation that produced it (for autograd)
```

Every chapter adds machinery around one of those three lines. Chapters 01 to 04 are about the metadata. Chapters 05 to 07 are about computing with it and differentiating through it. Chapters 08 to 10 change where and how fast the buffer is processed. Chapters 11 to 13 build a real model and ship it.

## Goals and non-goals

kael is didactic. Each design choice favors code that a reader can follow over the last bit of speed, but the library still has to be correct and usable.

- **Understand by building.** Every mechanism that PyTorch hides (strides, views, type promotion, the autograd graph, the caching allocator, the KV cache) exists in kael as a small, readable implementation.
- **Correct before fast.** Every operation is tested against NumPy or PyTorch, with golden data for the numeric edge cases (NaN, infinities, signed zeros, empty tensors). Golden data here means reference outputs computed once by PyTorch, saved as `.npy` files and committed, so the C++ tests can compare against them without Python installed.
- **Real end goal.** Version 1.0 loads the original 124M-parameter GPT-2 weights from a safetensors file. The test `test_real_gpt2_logits_match_huggingface` in `python/tests/test_gpt.py` runs the prompt `"Hello, my name is"` through kael and through the Hugging Face `transformers` reference, checks that the model has 124,439,808 parameters, and requires the largest absolute difference between the two sets of logits at the last position to stay below `1e-3`. The test is skipped when the weights are not downloaded.
- **Few dependencies.** The library needs a C++20 compiler, the system threads library and nlohmann_json. GoogleTest is fetched by default for the tests (`KAEL_BUILD_TESTS=ON`). Google Benchmark, a BLAS library, the CUDA toolkit and nanobind are optional and only enter through CMake options.

kael is not a production framework, and some limits are deliberate:

- It has four element types (`float32`, `float64`, `int64`, `bool`), declared as `enum class DType { F32, F64, I64, Bool }` in `include/kael/dtype.h`. There is no `float16` or `bfloat16`.
- It has no negative strides. `Tensor::slice` rejects a step that is not positive, so a reversed view such as NumPy's `a[::-1]` does not exist.
- CI has no GPU. It tests the CUDA backend with a host emulation and a compile-only build. `scripts/cuda/validate.sh` validates it end to end on an NVIDIA GPU ([CUDA backend](@/book/09-cuda-backend.md#validation-on-a-gpu)).

## A tour through one line of code

The fastest way to see how the layers fit is to follow one expression down the stack and back up. Take two float tensors that require gradients and add them:

```cpp
kael::Tensor a = kael::randn({2, 3}).requires_grad_();
kael::Tensor b = kael::randn({3}).requires_grad_();
kael::Tensor c = a + b;
c.sum().backward();
```

This is what each layer does for `c = a + b`, in call order:

```text
operator+(a, b)                     ops.h, src/ops/binary.cpp            chapter 05
  resolve the operands and pick the result dtype (type promotion)
  run the kernel with autograd recording switched off
    binary_kernel                   kernels.h                            chapter 04
      TensorIterator::build         iterator.h, src/iterator.cpp         chapter 04
        broadcast [2, 3] with [3]  -> [2, 3]      (broadcast_shapes)     chapter 01
        strides of b read as [2, 3] -> (0, 1)     (broadcast_strides)    chapter 01
        allocate the output        empty() -> Storage -> allocator       chapters 02, 08
        coalesce dimensions into the fewest loops
      run the inner loop on the CPU, or hand the layout to the device    chapters 08, 09
  autograd::record_add(out, a, b)   src/autograd/                        chapter 06
    attach a graph node to c that remembers how to send gradients to a and b
```

And for `c.sum().backward()`:

```text
Tensor::backward                    tensor.h, src/autograd/engine.cpp    chapter 06
  order the graph nodes topologically, starting from the root
  call each node's gradient formula; the formulas are themselves ops    chapter 05
  reduce the gradient of b from [2, 3] back to [3] (sum over dim 0)      chapters 01, 06
  accumulate into a.grad() and b.grad()
```

Three facts fall out of this tour and explain most of the architecture:

1. Shape arithmetic from chapter 01 is called at almost every step. It is the bottom of the stack because everything else needs it.
2. Operations do not loop over elements themselves. They describe the computation and let `TensorIterator` decide how to walk memory. That is why the CPU performance work in chapter 08 and the CUDA backend in chapter 09 could change how loops run without touching the operations.
3. Autograd sits on both sides of the operations. Each operation calls a small recording hook after it computes its result, and the gradient formulas that the engine runs are written with the same operations. The hooks are declared in the private header `src/autograd/record.h`, so the public operation headers never include the engine. The kernel runs with recording switched off (`autograd::without_grad` in `add`, `NoGradGuard` in the kernel templates), and only the public operation records a node, so one `a + b` adds exactly one node to the graph.

## Architecture

The library is a stack of layers. Read the diagram from the bottom up, in the same order as the book. An arrow points from a layer to the layer that uses it: `^` points up to the layer built on the one below, and `<--` points from a side component into the layer that includes it. The user includes the provider's headers and calls its functions, and the provider does not know the user exists.

```text
                       USER-FACING SURFACES
  +-------------------+   +--------------------------+   +------------------------+
  | 13 Server, tools  |   | 10 Python bindings       |   | examples/              |
  | server/app.py     |<--| python/src (nanobind)    |   | mnist_mlp, char_gpt,   |
  | debug.h, fuzz/    |   | python/kael (pure Python)|   | gpt2_generate          |
  +-------------------+   +--------------------------+   +------------------------+
            ^                         ^                             ^
            +-------------------------+-----------------------------+
                                      |
                       MODELS AND LOADING
  +----------------------------------------------------------------------------+
  | 11, 12 Transformer and loading                                             |
  | nn/transformer.h (GPT, KV cache), io/safetensors.h, text/bpe.h             |
  +----------------------------------------------------------------------------+
                                      ^
                       TRAINING
  +----------------------------------------------------------------------------+
  | 07 nn, optim, data                                                         |
  | nn/module.h, nn/linear.h, ..., optim/adam.h, ..., data/dataloader.h, ...   |
  +----------------------------------------------------------------------------+
                                      ^
                       DIFFERENTIATION
  +----------------------------------------------------------------------------+
  | 06 Autograd                                                                |
  | autograd/node.h, engine.h, functions.h, gradcheck.h, src/autograd/         |
  +----------------------------------------------------------------------------+
                                ^          | record hooks called by every op
                       COMPUTATION         v
  +----------------------------------------------------------------------------+
  | 05 Operations                                                              |
  | ops.h, operand.h, scalar.h, src/ops/                                       |
  +----------------------------------------------------------------------------+
                                      ^
  +----------------------------------------------------------------------------+   +---------------------+
  | 04 Iterator and kernels                                                    |<--| 09 CUDA backend     |
  | iterator.h, kernels.h                                                      |   | device_kernels.h    |
  +----------------------------------------------------------------------------+   | src/cuda/           |
                                      ^                                            +---------------------+
                       DATA LAYOUT                                                 | 08 CPU performance  |
  +----------------------------------------------------------------------------+   | parallel.h,         |
  | 03 Views                                                                   |   | matmul_versions.h,  |
  | src/view.cpp, src/copy.cpp (methods declared in tensor.h)                  |   | allocator.h         |
  +----------------------------------------------------------------------------+   +---------------------+
                                      ^                                                     |
  +----------------------------------------------------------------------------+            |
  | 02 Storage and tensor                                                      |<-----------+
  | dtype.h, storage.h, tensor_impl.h, tensor.h, factory.h, random.h, error.h  |  allocator under Storage
  +----------------------------------------------------------------------------+
                                      ^
  +----------------------------------------------------------------------------+
  | 01 Shape math                                                              |
  | shape.h, src/shape.cpp, src/internal/checked_math.h                        |
  +----------------------------------------------------------------------------+
```

The two boxes on the right are drawn beside the main stack on purpose. CPU performance and CUDA do not add a layer on top. They replace the internals of lower layers:

- The caching allocator (`allocator.h`) sits under `Storage`, so every tensor allocation goes through it without `Storage`'s callers noticing.
- The thread pool (`parallel.h`) splits the loops that `TensorIterator` runs, and `iterator.h` includes it.
- The device backend (`device_kernels.h`) receives the memory layout that `TensorIterator` computed and runs the element-wise loop on the GPU instead of the CPU. `iterator.h` includes it as well.

Code above them does not change. An operation written in chapter 05 runs in parallel and on the GPU without being edited.

The layers in one sentence each:

| Layer | Role | Chapter |
|-------|------|---------|
| Shape math | Free functions on `Shape` and `Strides`: element count, contiguous strides, offsets, broadcasting | [01](@/book/01-shapes-and-strides.md) |
| Storage and tensor | `Storage` owns bytes, `TensorImpl` interprets them, `Tensor` is a shared handle to an impl | [02](@/book/02-memory-model.md) |
| Views | New metadata over the same storage: `transpose`, `slice`, `expand`, `view`, `reshape` | [03](@/book/03-views.md) |
| Iterator and kernels | `TensorIterator` computes the broadcast shape, the result dtype and a coalesced loop for any element-wise operation | [04](@/book/04-tensor-iterator.md) |
| Operations | Arithmetic, reductions, matrix products, indexing, softmax and sequence operations | [05](@/book/05-operations.md) |
| Autograd | Reverse-mode differentiation: graph nodes recorded by every operation, a topological engine, gradcheck | [06](@/book/06-autograd.md) |
| nn, optim, data | Modules, losses, optimizers, schedulers, datasets and a MNIST training loop | [07](@/book/07-neural-networks.md) |
| CPU performance | Thread pool, packed SIMD matrix product, caching allocator, fused kernels | [08](@/book/08-cpu-performance.md) |
| CUDA | A device backend behind `device_kernels.h`, with a host emulation for testing | [09](@/book/09-cuda-backend.md) |
| Python | nanobind module `kael`, NumPy interop and the wheel build | [10](@/book/10-python-bindings.md) |
| Transformer and loading | GPT blocks with a KV cache, safetensors, GPT-2 weight mapping and the BPE tokenizer | [11](@/book/11-transformer.md), [12](@/book/12-model-loading.md) |
| Server and tooling | FastAPI server, debugging helpers, fuzzers, sanitizers and CI | [13](@/book/13-serving-and-tooling.md) |

### Two decisions that shape every layer

Five design decisions affect every file above shape math, and each has its own record:

- [ADR 0001](@/adr/0001-runtime-dtype.md) makes the element type a run-time value. A `Tensor` is not a template `Tensor<float>`. It carries a `DType` field, and kernels dispatch on it at run time. This is the PyTorch model. It keeps tensors of every dtype in one autograd graph and lets the Python bindings expose a single `Tensor` type.
- [ADR 0002](@/adr/0002-tensor-as-handle.md) makes `Tensor` a shared handle. Copying a `Tensor` copies a pointer, not data, so `Tensor b = a;` gives two names for the same tensor, as in Python. Views and autograd both depend on this: a view must keep its storage alive, and a graph node must keep its inputs alive.
- [ADR 0003](@/adr/0003-type-promotion.md) fixes the type promotion rules. Mixed-dtype operations follow PyTorch, so results match what a PyTorch user expects.
- [ADR 0004](@/adr/0004-pytorch-shaped-api.md) shapes the API after PyTorch, with names, argument order and typed exceptions a PyTorch user recognizes.
- [ADR 0005](@/adr/0005-dynamic-autograd-graph.md) builds the autograd graph while operations run (define-by-run), so ordinary C++ control flow works with gradients.

### Why layers and not one big class

A framework could put everything into one `Tensor` class with methods for shapes, storage, views, math and gradients. kael splits them for two reasons. First, each layer can be tested alone: the shape functions of chapter 01 have their own suite and need no tensors at all. Second, the book can follow the build order. At the tag for each chapter, the code above that layer does not exist yet, and the code below it is finished and tested. The price is that a reader must know which layer a function lives in, which is what the map below is for.

## Repository map

The repository, from `git ls-files`, without the contents of `tests/golden/` (71 per-operation folders and a `manifest.txt`). The right column says which chapter explains each part.

```text
path                       contents                                                   chapter
-------------------------  ---------------------------------------------------------  -------
CMakeLists.txt             library target, options and optional subprojects            00
pyproject.toml             Python package kael-tensor (scikit-build-core, nanobind)    10
.clang-format, .clang-tidy formatting and static analysis rules                        13
.github/workflows/         ci.yml (build, sanitizers, CUDA compile, Python, fuzz,      00, 13
                           coverage, clang-tidy), wheels.yml (cibuildwheel)

include/kael/              public headers, included as "kael/<name>.h"
  shape.h                  Shape, Strides and the index arithmetic                      01
  error.h                  kael::Error, IndexError, ValueError, TypeError, KAEL_CHECK_*  01, 02
  dtype.h                  DType, Device, element sizes, dispatch                       02
  storage.h                Storage: owns the bytes                                      02
  tensor_impl.h            TensorImpl: shape, strides, offset, dtype over a Storage     02
  tensor.h                 Tensor: the shared handle, views, element access             02, 03
  factory.h                empty, zeros, ones, full, arange, linspace, eye, from_vector 02
  random.h                 rand, randn, randint, randperm, bernoulli, multinomial       02
  scalar.h, operand.h      Scalar and Operand, so ops accept tensors and numbers        05
  iterator.h, kernels.h    TensorIterator and the element-wise kernel templates         04
  ops.h                    every public operation                                       05
  autograd/                engine.h, functions.h, grad_mode.h, gradcheck.h, meta.h,     06
                           node.h
  nn/                      module.h, linear.h, layernorm.h, embedding.h, activation.h,  07, 11
                           dropout.h, sequential.h, functional.h, loss.h, init.h,
                           transformer.h
  optim/                   optimizer.h, sgd.h, adam.h, scheduler.h, clip.h              07
  data/                    dataset.h, dataloader.h, mnist.h                             07
  parallel.h               thread pool and parallel_for                                 08
  allocator.h              caching allocator under Storage                              08
  matmul_versions.h        matrix product variants, from v1_ijk to v6_threaded          08
  device.h                 parse_device, cuda_is_available, cuda_synchronize            09
  device_kernels.h         the interface the CUDA backend implements                    09
  io/safetensors.h         safetensors load (memory-mapped) and save                    12
  text/bpe.h               GPT-2 byte pair encoding tokenizer                           12
  debug.h                  explain, autograd_graph_dot, backward trace, anomaly mode    13
  version.h                kael::version()                                              00
  kael.h                   umbrella header

src/                       implementation, one .cpp per public header plus folders
  shape.cpp                                                                             01
  storage.cpp, tensor_impl.cpp, tensor.cpp, factory.cpp, random.cpp, print.cpp          02
  view.cpp, copy.cpp                                                                    03
  iterator.cpp                                                                          04
  ops/                     binary, unary, inplace, reduce, linalg, indexing, nn,        05
                           sequence
  autograd/                engine, node, recording, record.h hooks and the gradient     06
                           formulas (functions_math, _reduce, _view, _indexing, _linalg)
  nn/, optim/, data/       module, layers, losses, init, functional, transformer;       07, 11
                           optimizer.cpp; dataset.cpp, mnist.cpp
  parallel.cpp, allocator.cpp, linalg/matmul_versions.cpp                               08
  device.cpp, cuda/        backend.cu, kernels.inl, backend_fake.cpp, backend_none.cpp  09
  io/, text/               safetensors.cpp; bpe.cpp, unicode_classes.inc                12
  debug.cpp, version.cpp                                                                13
  internal/                private helpers: checked_math.h, arithmetic.h, copy.h,       01 onward
                           element_functions.h, index_checks.h, nan_aware.h, operands.h

tests/                     GoogleTest suites, one test_<component>.cpp per component    every
  generated/               NumPy-generated broadcast_cases.inc and view_cases.inc       01, 03
  golden/                  PyTorch-generated .npy fixtures, one folder per operation    05
  support/                 allclose.h, npy.h, npy.cpp, values.h                         05

python/                    Python package                                               10
  src/                     nanobind bindings (tensor, ops, autograd, interop, models)
  kael/                    pure Python layer: nn, optim, data
  tests/                   pytest suite, including test_gpt.py and test_server.py

examples/                  mnist_mlp.cpp, char_gpt.cpp, gpt2_generate.cpp,              07, 11, 12
                           gpt2_generate.py, mnist.py, quickstart.ipynb
benchmarks/                Google Benchmark suites (target kael_bench) and compare.py   08, 09
experiments/               micrograd.cpp, a scalar autograd warm-up                     06
fuzz/                      libFuzzer targets fuzz_safetensors and fuzz_idx, with seed   13
                           corpora
scripts/                   run_tests.sh                                                 00
  download/                common.sh, mnist.sh, shakespeare.sh, gpt2.sh, with SHA-256   07, 11, 12
                           checks
  generate/                broadcast_cases.py, view_cases.py, golden.py,                01, 03, 05, 12
                           unicode_classes.py
  cuda/                    validate.sh, run_remote.sh, validate_on_colab.ipynb          09
server/                    app.py (FastAPI), Dockerfile, requirements.txt               13
docs/                      the original book and the ADRs
```

### Public and private headers

Public headers live in `include/kael/` and are included as `"kael/tensor.h"`. Private headers live in `src/` (for example `src/internal/checked_math.h` or `src/autograd/record.h`) and are only visible to the library, because `CMakeLists.txt` declares:

```cmake
target_include_directories(kael PUBLIC include PRIVATE src)
```

`PUBLIC include` means that any target that links `kael` also sees `include/`. `PRIVATE src` means that only the library's own sources see `src/`. A program that tries to include `internal/checked_math.h` fails to compile, which keeps implementation details out of the API.

The umbrella header `include/kael/kael.h` includes the core (`shape.h`, `dtype.h`, `storage.h`, `tensor_impl.h`, `tensor.h`, `factory.h`, `random.h`, `error.h`), the operations and kernels, autograd, the device, parallel and allocator headers, `version.h` and the debugging helpers. The `nn`, `optim`, `data`, `io` and `text` headers are included one by one, so a program that only needs tensors does not compile the transformer.

## Build and test

Requirements: CMake 3.24 or newer and a C++20 compiler. Configuring downloads nlohmann_json 3.11.3 and GoogleTest 1.15.2 with `FetchContent`, so the first configure needs network access. Every archive is pinned with `URL_HASH SHA256=...`, so a download whose content changed fails the configure instead of entering the build.

Build the library and the tests, then run them:

```bash
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The first command configures: `-S .` is the source directory and `-B build` the build directory, so all generated files stay out of the source tree. The second compiles. The third runs every registered test and prints the output of the ones that fail.

`scripts/run_tests.sh [BUILD_DIR]` configures a Debug build in `BUILD_DIR` (default `build`) if it has none, then runs the last two commands, with `ctest --parallel` set to the number of CPUs. All C++ tests compile into a single executable, `build/tests/kael_tests`, and `gtest_discover_tests` registers each test with CTest. One executable keeps link time low and lets you run any subset without CTest. Run one group of suites directly with a GoogleTest filter:

```bash
build/tests/kael_tests --gtest_filter='BroadcastShapes.*'
build/tests/kael_tests --gtest_filter='Numel.*:ContiguousStrides.*'
```

A filter is a `:`-separated list of `Suite.Test` patterns, and `*` matches any text.

### The CI matrix and its local equivalents

The CI job `build-test` in `.github/workflows/ci.yml` configures a Debug build with warnings as errors and runs the tests in six variants:

| Variant | Compiler | Extra options |
|---------|----------|---------------|
| `linux-gcc` | GCC | none |
| `linux-clang` | Clang | none |
| `macos-clang` | Apple Clang | none |
| `linux-clang-sanitizers` | Clang | `KAEL_ENABLE_SANITIZERS=ON` |
| `linux-clang-tsan` | Clang | `KAEL_ENABLE_TSAN=ON` |
| `linux-clang-fake-cuda` | Clang | `KAEL_ENABLE_SANITIZERS=ON`, `KAEL_FAKE_CUDA=ON` |

Reproduce the three strictest variants locally with:

```bash
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DKAEL_WARNINGS_AS_ERRORS=ON -DKAEL_ENABLE_SANITIZERS=ON
cmake -S . -B build-tsan -DCMAKE_BUILD_TYPE=Debug -DKAEL_WARNINGS_AS_ERRORS=ON -DKAEL_ENABLE_TSAN=ON
cmake -S . -B build-fake-cuda -DCMAKE_BUILD_TYPE=Debug -DKAEL_WARNINGS_AS_ERRORS=ON -DKAEL_ENABLE_SANITIZERS=ON -DKAEL_FAKE_CUDA=ON
```

Build each directory with `cmake --build <dir> --parallel` and test it with `ctest --test-dir <dir> --output-on-failure`. Keep one build directory per configuration. CMake caches options per directory, so switching options inside one directory leads to confusing rebuilds.

What each variant catches:

- **AddressSanitizer and UndefinedBehaviorSanitizer** (ASan and UBSan) instrument every memory access and every arithmetic operation. They report reads past the end of a buffer, use after free, and signed integer overflow, which is undefined behavior in C++. Chapter 01 depends on UBSan to prove that its overflow checks run before the multiplication.
- **ThreadSanitizer** (TSan) reports data races in the thread pool of chapter 08.
- **Fake CUDA** runs the CUDA kernels on the host, so the device path of chapter 09 is tested under sanitizers on a machine without a GPU.

> [!NOTE]
> Both sanitizer options also define `KAEL_NO_CACHING_ALLOCATOR`, which compiles the caching allocator out. A caching allocator reuses freed blocks without returning them to the system, which would hide a use after free from AddressSanitizer.

The other CI jobs:

| Job | What it does |
|-----|--------------|
| `cuda-build` (shown as `cuda-compile-only`) | Configures a Release build with `-DKAEL_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86` in the `nvidia/cuda:12.6.2-devel-ubuntu22.04` container and builds. No tests run, because GitHub-hosted runners have no GPU. Execution on the device is validated with `scripts/cuda/validate.sh` ([CUDA backend](@/book/09-cuda-backend.md#validation-on-a-gpu)). |
| `python` | `pip install ".[test]"` and `pytest python/tests -q` on Linux and macOS with Python 3.12. |
| `fuzz` | Builds `fuzz_safetensors` and `fuzz_idx` with sanitizers and runs each for 60 seconds on its corpus, with `-malloc_limit_mb=2048`. |
| `coverage` | Builds with `KAEL_FAKE_CUDA=ON`, runs `kael_tests` with Clang source-based coverage and prints an `llvm-cov report` for `src` and `include/kael`. It calls `llvm-profdata-<version>` and `llvm-cov-<version>`, with the major version taken from `clang -dumpversion`, so the tools match the Clang that built the tests. |
| `clang-tidy` | Runs `run-clang-tidy` over every `.cpp` file under `src/` with the repository `.clang-tidy`. |

A second workflow, `.github/workflows/wheels.yml`, is started by hand. It builds wheels with cibuildwheel on Linux x86-64, Linux ARM and two macOS runners, builds a source distribution, and can publish to TestPyPI or PyPI.

### Python package

Build and test the Python package:

```bash
pip install ".[test]"
pytest python/tests
```

`pyproject.toml` builds the extension through scikit-build-core with `KAEL_BUILD_PYTHON=ON` and `KAEL_BUILD_TESTS=OFF`. The package name on the index is `kael-tensor`, and the import name is `kael`. Chapter 10 explains the bindings.

### Examples and benchmarks

Build the examples and the benchmarks with their options. Benchmarks need a Release build, because a Debug build measures the debug checks instead of the code:

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DKAEL_BUILD_EXAMPLES=ON -DKAEL_BUILD_BENCHMARKS=ON
cmake --build build-release --parallel
build-release/benchmarks/kael_bench
```

To see the end goal working, download the GPT-2 weights and generate text. `scripts/download/gpt2.sh` fetches `model.safetensors`, `vocab.json` and `merges.txt` into `data/gpt2`, checks the SHA-256 of each file, and keeps files that already match. `gpt2_generate` reads that directory by default:

```bash
scripts/download/gpt2.sh
build-release/examples/gpt2_generate "Hello, my name is" --tokens 20
```

Its options are `--data DIR`, `--device cpu|cuda`, `--tokens N`, `--temperature T`, `--top-k K` and `--seed S`. Run it with `--help` to print them.

## CMake options

All options are declared at the top of `CMakeLists.txt`.

| Option | Default | Effect |
|--------|---------|--------|
| `KAEL_WARNINGS_AS_ERRORS` | `OFF` | Adds `-Werror` to every kael target. The warning set is always `-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion`. |
| `KAEL_ENABLE_SANITIZERS` | `OFF` | Builds with `-fsanitize=address,undefined -fno-omit-frame-pointer`. Also compiles the caching allocator out (`KAEL_NO_CACHING_ALLOCATOR`) so that AddressSanitizer sees every allocation. |
| `KAEL_ENABLE_TSAN` | `OFF` | Builds with `-fsanitize=thread`. Configuring fails if it is combined with `KAEL_ENABLE_SANITIZERS`. Also compiles the caching allocator out. |
| `KAEL_BUILD_BENCHMARKS` | `OFF` | Fetches Google Benchmark 1.9.0 and builds `benchmarks/` (target `kael_bench`). Configuring fails if `KAEL_ENABLE_SANITIZERS` is on, because sanitizers distort timings. |
| `KAEL_BUILD_EXAMPLES` | `OFF` | Builds `mnist_mlp`, `char_gpt` and `gpt2_generate` from `examples/`. |
| `KAEL_USE_BLAS` | `OFF` | Multiplies matrices with the system BLAS: Accelerate on macOS, `find_package(BLAS)` elsewhere. |
| `KAEL_ENABLE_CUDA` | `OFF` | Enables the CUDA language, links `CUDA::cudart` and compiles `src/cuda/backend.cu`. |
| `KAEL_FAKE_CUDA` | `OFF` | Compiles `src/cuda/backend_fake.cpp`, which runs the CUDA kernels on the host to test the device path without a GPU. Configuring fails if it is combined with `KAEL_ENABLE_CUDA`. |
| `KAEL_USE_CUBLAS` | `OFF` | With `KAEL_ENABLE_CUDA`, links cuBLAS and uses it for device matrix products. |
| `KAEL_BUILD_TESTS` | `ON` | Fetches GoogleTest and builds `tests/` (`kael_tests`) and the `micrograd` experiment. |
| `KAEL_BUILD_FUZZERS` | `OFF` | Builds the libFuzzer targets in `fuzz/`. Needs Clang, because the targets use `-fsanitize=fuzzer`. |
| `KAEL_BUILD_PYTHON` | `OFF` | Builds the nanobind extension in `python/` and compiles the library as position independent code, which a shared Python module requires. |
| `KAEL_NATIVE` | `OFF` | Adds `-march=native`, or `-mcpu=native` where only that flag exists, to the library. The binary then only runs on CPUs with the same instruction set as the build machine. |

The three CUDA-related choices select exactly one device backend source file:

```text
KAEL_ENABLE_CUDA=ON   ->  src/cuda/backend.cu        real GPU kernels
KAEL_FAKE_CUDA=ON     ->  src/cuda/backend_fake.cpp  the same kernels, run on the host
neither               ->  src/cuda/backend_none.cpp  every device function throws
```

When neither CUDA option is on, every device operation throws `kael was built without CUDA; configure with KAEL_ENABLE_CUDA=ON`. The rest of the library compiles and runs unchanged, because it only sees the declarations in `device_kernels.h`.

The project version comes from `project(kael VERSION 1.0.0 LANGUAGES CXX)`. CMake passes it to the library as the `KAEL_VERSION` definition, and `kael::version()` in `include/kael/version.h` returns it.

## Phases, chapters and tags

kael was built in thirteen phases. Phase 1 set up the repository (license, `.gitignore`, `.clang-format`, `.editorconfig`). Every later phase has a chapter, and from phase 2 on, a git tag marks the last commit of the phase. Milestone tags `v0.1.0` to `v1.0.0` point at the same commits as the phase that completed each milestone.

| Phase | Chapter | Phase tag | Milestone tag | Last commit of the phase |
|-------|---------|-----------|---------------|--------------------------|
| 2. Shapes and strides | [01](@/book/01-shapes-and-strides.md) | `chapter-02-shapes` | | `87abce8 docs(readme): add README` |
| 3. Core tensor | [02](@/book/02-memory-model.md) | `chapter-03-core` | | `d90fbfb docs(chapters): add chapter 02 memory model` |
| 4. Views | [03](@/book/03-views.md) | `chapter-04-views` | | `d41fcb4 test(view): add NumPy-generated view chains` |
| 5. Iteration | [04](@/book/04-tensor-iterator.md) | `chapter-05-iteration` | | `5075617 test(bench): add elementwise benchmarks` |
| 6. Operations | [05](@/book/05-operations.md) | `chapter-06-ops` | | `c786049 fix(ops): align edge cases with PyTorch` |
| 7. Autograd | [06](@/book/06-autograd.md) | `chapter-07-autograd` | | `8933c72 feat(autograd): add reverse-mode engine, formulas and gradcheck` |
| 8. Training | [07](@/book/07-neural-networks.md) | `chapter-08-training` | `v0.1.0` | `09baa43 feat(examples): add MNIST training` |
| 9. CPU performance | [08](@/book/08-cpu-performance.md) | `chapter-09-cpu-performance` | `v0.2.0` | `9a30568 fix(storage): make the block deleter default constructible on GCC` |
| 10. CUDA | [09](@/book/09-cuda-backend.md) | `chapter-10-cuda` | `v0.3.0` | `1fd592c chore(release): bump version to 0.3.0` |
| 11. Python | [10](@/book/10-python-bindings.md) | `chapter-11-python` | `v0.4.0` | `6472c9d ci(github): build Python wheels` |
| 12. Transformer | [11](@/book/11-transformer.md), [12](@/book/12-model-loading.md) | `chapter-12-transformer` | | `cb0f4be test(nn): skip the full GPT-2 construction under sanitizers` |
| 13. Polish | [12](@/book/12-model-loading.md), [13](@/book/13-serving-and-tooling.md) | `chapter-13-polish` | `v1.0.0` | the commit tagged `v1.0.0` |

Three details of the tags matter when you use them:

- Tag numbers follow phases, not chapters. `chapter-02-shapes` holds the code of chapter 01, `chapter-03-core` the code of chapter 02, `chapter-04-views` the code of chapter 03, and so on up to `chapter-12-transformer`. Subtract one from the tag number to get the chapter number.
- `chapter-12-transformer` ends the transformer phase: sequence operations, safetensors, the GPT model with its KV cache, and the first debugging helpers, fuzzers and clang-tidy setup.
- Phase 13 runs from there to `v1.0.0`: the memory-mapped safetensors loader, atomic saves, the GPT-2 BPE tokenizer, `examples/gpt2_generate.cpp`, the HTTP server and its Docker image, the lock around the global random generator, the fused cross entropy, the correctness fixes, the book and the GPU validation. `chapter-13-polish` and `v1.0.0` mark its last commit, so chapters 12 and 13 are complete there.

## How to read this book

Read the chapters in order. Each one lists its prerequisites and files, explains why the layer exists, works through the idea with small numeric examples, walks through the real code, lists the edge cases, names the tests that prove the behavior, and ends with a numbered plan to rebuild that stage with a test checkpoint for every step.

The chapters describe the code at `v1.0.0`. To see the code as it was at the end of a phase, check out its tag in a separate worktree, so that your main checkout stays on the branch:

```bash
git worktree add ../kael-core chapter-03-core
cmake -S ../kael-core -B ../kael-core/build
cmake --build ../kael-core/build --parallel
ctest --test-dir ../kael-core/build --output-on-failure
```

A worktree is a second working directory backed by the same repository. It lets you build an old tag next to the current code without stashing or switching branches. Remove it with `git worktree remove ../kael-core` when you are done.

A plain `git checkout chapter-03-core` also builds, but the early tags carry older drafts of some chapters, so a separate worktree keeps this book in view. The early tags are useful because they are small: at `chapter-03-core` the library is shapes, storage, tensors, factories, random numbers and printing, with no views, operations or autograd. Later phases changed some of that code. For example, the caching allocator under `Storage` arrived in phase 9, and the typed error classes arrived in phase 10. Each chapter notes such changes where they matter.

To see what a phase added, diff two consecutive tags:

```bash
git diff --stat chapter-03-core chapter-04-views
git log --oneline chapter-03-core..chapter-04-views
```

Commits follow Conventional Commits with a scope (`feat(view): ...`, `fix(ops): ...`), so `git log --oneline` reads as a changelog of each phase.

> [!TIP]
> When you rebuild kael yourself, use the test suites as the specification. Each chapter names the GoogleTest suites in `tests/` that prove its behavior. Copy a suite into your own project before writing the code it tests, and work until it passes.

## Prerequisites for the reader

- C++17 or later: templates, lambdas, `std::shared_ptr`, `std::unique_ptr`, move semantics. kael uses C++20 features (`std::span`, concepts, templated lambdas, `std::ranges`), and the chapters explain them where they appear.
- CMake basics: configuring, building and running CTest.
- Linear algebra at the level of matrix products, and calculus at the level of the chain rule, from chapter 06 on.
- Python and NumPy from chapter 10 on, and for the scripts that generate test fixtures.

No prior knowledge of PyTorch internals is assumed. kael follows PyTorch semantics where it can (broadcasting, type promotion, view rules, autograd behavior), so knowing PyTorch as a user helps to check the results. The [glossary](@/concepts/glossary.md) defines the terms the book uses.

## Check your understanding

<details>
<summary>Which layer would you change to make every element-wise operation run on four threads, and why do the operations themselves not change?</summary>

The iterator layer. Operations in chapter 05 describe the computation and hand it to `TensorIterator`, which owns the loop over memory. Chapter 08 makes that loop split its range across the thread pool in `parallel.h`. Because no operation contains its own element loop, none of them has to be edited.

</details>

<details>
<summary>Why are CPU performance and CUDA drawn beside the stack instead of on top of it?</summary>

They do not add new behavior that higher layers call. They replace the internals of lower layers: the caching allocator sits under `Storage`, the thread pool runs the iterator's loops, and the device backend receives the iterator's memory layout. Code written against the layers above them keeps working unchanged.

</details>

<details>
<summary>You check out the tag <code>chapter-05-iteration</code>. Which chapter's code does it hold, and is <code>TensorIterator</code> there?</summary>

Tag numbers follow phases, and phase 5 matches chapter 04. So the tag holds the code at the end of chapter 04, and `TensorIterator` is the newest piece in it. The operations of chapter 05 are not there yet.

</details>

<details>
<summary>A program outside the library includes <code>"internal/checked_math.h"</code>. What happens, and why is that intended?</summary>

It fails to compile. `CMakeLists.txt` adds `src/` as a `PRIVATE` include directory of `kael`, so only the library's own sources can see it. Headers under `src/` are implementation details, and keeping them out of reach means they can change without breaking users.

</details>

<details>
<summary>Why do the sanitizer builds compile the caching allocator out?</summary>

A caching allocator keeps freed blocks and hands them out again instead of returning them to the system. AddressSanitizer only sees a use after free when the memory is really freed, so the cache would hide exactly the bugs the sanitizer is meant to find. The `KAEL_NO_CACHING_ALLOCATOR` definition makes every allocation go straight to the system allocator.

</details>

<details>
<summary>What does <code>build/tests/kael_tests --gtest_filter='Numel.*'</code> run, and why is there one test executable instead of one per file?</summary>

It runs every test in the `Numel` suite and nothing else. All C++ tests link into the single `kael_tests` executable and `gtest_discover_tests` registers each one with CTest. One executable links once, and the GoogleTest filter selects any subset without rebuilding.

</details>

## Further reading

- Edward Z. Yang, [PyTorch internals](http://blog.ezyang.com/2019/05/pytorch-internals/). The best single description of strides, storage, dispatch and autograd in a production framework; most of kael's layers have a counterpart in it.
- [NumPy: Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) and [the N-dimensional array](https://numpy.org/doc/stable/reference/arrays.ndarray.html). The reference semantics that kael's shape and view layers follow.
- Andrej Karpathy, [micrograd](https://github.com/karpathy/micrograd). A scalar autograd engine in about a hundred lines of Python. `experiments/micrograd.cpp` is the same idea in C++, written as a warm-up before chapter 06.
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017). The transformer architecture that chapter 11 implements.
- Radford et al., [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (2019). The GPT-2 paper, which describes the model whose weights kael loads.
- The kael [CMakeLists.txt](https://github.com/JetBerri/kael.cpp/blob/main/CMakeLists.txt) and [ci.yml](https://github.com/JetBerri/kael.cpp/blob/main/.github/workflows/ci.yml). Short, and the ground truth for every option and CI job in this chapter.

Next: [01. Shapes and strides](@/book/01-shapes-and-strides.md)
