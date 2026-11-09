+++
title = "Glossary"
description = "Definitions of the terms used across the kael.cpp book, each linked to the chapter that covers it."
weight = 1

[extra]
kicker = "Reference"
+++

This page defines the terms the book uses, in alphabetical order. Each term has its own heading, so other pages can link to it with an anchor such as `@/concepts/glossary.md#stride`. Each definition ends with a link to the chapter that explains the term in depth.

To add a term, place it under its letter in alphabetical order, keep the definition to one to three sentences, and link the chapter that covers it.

## A

### AccumulateGrad

The autograd node that sits at a leaf and adds incoming gradients into the leaf's `grad` field. Every operation that reads the same leaf shares one `AccumulateGrad` node, which the leaf references weakly through `AutogradMeta::accumulator`. See [Autograd](@/book/06-autograd.md).

### Adam

An optimizer that keeps two moving averages per parameter: the first moment of the gradient and the second moment (the mean of squared gradients). Both averages start at zero, so each update divides them by `1 - beta^t` to remove the bias towards zero. AdamW is the variant that applies weight decay directly to the weights instead of adding it to the gradient. See [Neural networks and training](@/book/07-neural-networks.md).

### Allclose

The comparison the tests use for floating point results: a value passes when `|actual - expected| <= atol + rtol * |expected|`. In kael it is `kael::test::allclose` in `tests/support/allclose.h`, which also requires equal shapes and dtypes, exact infinities, and NaN only when `equal_nan` is set. See [Operations](@/book/05-operations.md).

### Anomaly detection

A debugging mode, enabled with `set_anomaly_detection(true)`, in which the autograd engine checks the outputs of every node right after it runs. It reports the node that first produced a non-finite gradient, before the NaN spreads to every parameter. See [Serving and tooling](@/book/13-serving-and-tooling.md).

### Attention

The operation in which each position of a sequence computes a weighted average of the values of other positions. The weights are a softmax over the dot products of the position's query with every key. See [Transformer](@/book/11-transformer.md).

### Autograd

Automatic differentiation: the part of the framework that records the operations of the forward pass and computes gradients by applying the chain rule backwards from an output. kael implements reverse mode over a graph built dynamically as operations run. See [Autograd](@/book/06-autograd.md).

## B

### Backend

The code that executes tensor operations on one device. kael has the CPU path and a CUDA backend behind the interface in `include/kael/device_kernels.h`, with three implementations: the real CUDA backend, a host emulation, and a stub that throws when CUDA is not built. See [CUDA backend](@/book/09-cuda-backend.md).

### Backward pass

The second phase of a training step, in which the autograd engine visits the recorded graph from the output towards the leaves and computes the gradient of every input. It starts from a seed gradient, which is one for a scalar output. See [Autograd](@/book/06-autograd.md).

### BLAS

Basic Linear Algebra Subprograms, the standard interface for optimized vector and matrix routines such as matrix multiplication. kael can dispatch matmul to a BLAS library or to Accelerate on macOS, and compares its own matmul versions against it. See [CPU performance](@/book/08-cpu-performance.md).

### Broadcasting

The rule that lets an operation combine tensors of different shapes without copying. Shapes are aligned from the right, missing leading dimensions count as size 1, and two sizes are compatible when they are equal or one of them is 1. See [Shapes and strides](@/book/01-shapes-and-strides.md).

### Byte pair encoding

A tokenization method, abbreviated BPE, that starts from single bytes and applies an ordered list of learned merges of adjacent symbols. At each step it merges the adjacent pair with the lowest rank, and it stops when no pair in the word appears in the list. See [Loading GPT-2](@/book/12-model-loading.md).

### Byte-to-unicode mapping

The table GPT-2 uses to give each of the 256 byte values a printable Unicode character, so the vocabulary and merges files can be stored as text. The 188 printable bytes map to themselves and the other 68 map to code points from 256 upwards. See [Loading GPT-2](@/book/12-model-loading.md).

## C

### Caching allocator

An allocator that keeps freed memory blocks and returns them for later requests of the same size class, instead of calling the system allocator every time. It avoids repeated page mapping and page faults in training loops that allocate the same sizes on every step. See [CPU performance](@/book/08-cpu-performance.md).

### Causal mask

A boolean matrix that marks, for each query position, the key positions that come after it. Attention sets the scores of those pairs to minus infinity before the softmax, so a position cannot read the future tokens it is trained to predict. See [Transformer](@/book/11-transformer.md).

### Chain rule

The calculus rule that the derivative of a composition is the product of the local derivatives: if `y = f(u)` and `u = g(x)`, then `dy/dx = dy/du * du/dx`. Reverse-mode autograd applies it one node at a time, from the output backwards. See [Autograd](@/book/06-autograd.md).

### Coalescing

An iterator optimization that merges two adjacent dimensions into one when, for every operand, `stride[outer] == stride[inner] * shape[inner]`. A fully contiguous tensor collapses to a single dimension, so the inner loop runs over all elements at once. See [TensorIterator](@/book/04-tensor-iterator.md).

### Contiguous

A tensor whose elements are laid out in row-major order with no gaps, so its strides equal the contiguous strides of its shape. kael skips size-1 dimensions in this check and treats empty tensors as contiguous. See [Memory model](@/book/02-memory-model.md) and [Views](@/book/03-views.md).

### Coverage-guided fuzzing

A testing technique that runs a function on generated inputs, measures which code each input reaches, and keeps mutating the inputs that reach new code. kael fuzzes its safetensors and MNIST IDX parsers with libFuzzer. See [Serving and tooling](@/book/13-serving-and-tooling.md).

### Cross entropy

The loss for classification: the negative log of the probability the model assigns to the correct class, computed from logits through log-softmax. kael computes it with a fused row kernel on contiguous CPU inputs. See [Neural networks and training](@/book/07-neural-networks.md).

## D

### DataLoader

A range over a dataset that yields batches of examples, with options for batch size, shuffling and dropping the last incomplete batch. It holds a reference to its dataset, so the dataset must outlive it. See [Neural networks and training](@/book/07-neural-networks.md).

### Detach

An operation that returns a view on the same storage with no autograd metadata, so gradients stop flowing through it. The detached tensor shares the version counter with the original, and `detach_()` does the same to the tensor itself. See [Autograd](@/book/06-autograd.md).

### Dimension wrapping

The conversion of a negative dimension argument into a position, where `-1` means the last dimension. `wrap_dim(-1, 3)` returns 2, and an argument outside `[-ndim, ndim - 1]` raises `IndexError`. See [Shapes and strides](@/book/01-shapes-and-strides.md).

### Dispatch

The step that turns a runtime dtype into a C++ type. `dispatch(dtype, f)` is a `switch` that calls the templated lambda `f` with `float`, `double`, `std::int64_t` or `std::uint8_t`, and `dispatch_floating` and `dispatch_no_bool` restrict the set. See [Memory model](@/book/02-memory-model.md) and [Runtime dtype](@/adr/0001-runtime-dtype.md).

### DLPack

A protocol for exchanging tensors between libraries without copying memory. kael's Python `Tensor` implements `__dlpack__`, so `torch.from_dlpack(t)` returns a zero-copy view of a kael CPU tensor. See [Python bindings](@/book/10-python-bindings.md).

### Dropout

A regularization layer that zeroes each element with probability `p` during training. kael implements inverted dropout, which divides the kept elements by `1 - p` at training time, so evaluation needs no rescaling. See [Neural networks and training](@/book/07-neural-networks.md).

### Dtype

The element type of a tensor. kael has four: `float32`, `float64`, `int64` and `bool`, stored as the runtime value `enum class DType { F32, F64, I64, Bool }` instead of a template parameter. See [Memory model](@/book/02-memory-model.md) and [Runtime dtype](@/adr/0001-runtime-dtype.md).

## E

### Edge

A link in the autograd graph from a node to the node that receives the gradient of one of its inputs. In kael an edge is a `std::shared_ptr<Node>`, so holding the output keeps the whole graph alive. See [Autograd](@/book/06-autograd.md).

### Embedding

A layer that maps integer indices to rows of a learned table of shape `[count, dimension]`. GPT uses one embedding for tokens and one for positions. See [Neural networks and training](@/book/07-neural-networks.md).

### Expand

A view that makes a tensor look larger than its storage by giving each broadcast dimension a stride of 0. Every step along such a dimension reads the same element, which is how broadcasting is materialized without a copy. See [Views](@/book/03-views.md).

### External storage

A `Storage` that wraps memory kael did not allocate, created with `Storage::external`. An owner object keeps the memory alive, for example a memory-mapped safetensors file or a NumPy array. See [Memory model](@/book/02-memory-model.md).

## F

### Factory

A function that creates a new tensor, such as `zeros`, `ones`, `full`, `arange`, `empty` or `randn`. `empty` returns uninitialized memory, and the other factories decide what to write into it. See [Memory model](@/book/02-memory-model.md).

### Finite differences

A numerical estimate of a derivative from function values: the central form is `(f(x + eps) - f(x - eps)) / (2 * eps)`. Gradcheck compares it with the analytic gradient. See [Autograd](@/book/06-autograd.md).

### Fork safety

The property that a library keeps working, or fails cleanly, in a child process created with `fork()`. The child has only the thread that called `fork`, so kael registers a `pthread_atfork` handler that keeps the child from using or destroying the parent's thread pool. See [CPU performance](@/book/08-cpu-performance.md).

### Forward pass

The first phase of a training or inference step, in which operations compute outputs from inputs. When gradients are enabled, each operation also records the node needed for the backward pass. See [Autograd](@/book/06-autograd.md).

### Fused row kernel

A kernel that computes a row-wise operation such as softmax, log-softmax, cross entropy or layer normalization in a few passes over each row, instead of composing several full-tensor operations with temporaries. kael uses one when the reduced dimension is the last dimension of a contiguous CPU tensor. See [CPU performance](@/book/08-cpu-performance.md).

## G

### GIL

The Global Interpreter Lock of CPython, which lets only one thread run Python bytecode at a time. kael's bindings release it during heavy C++ work that touches no Python objects, so other Python threads keep running. See [Python bindings](@/book/10-python-bindings.md).

### Golden test

A test that compares kael's output against reference outputs produced by another implementation. `scripts/generate/golden.py` generates inputs and expected outputs with PyTorch, and `Golden.AllCasesMatchPyTorch` checks every case. See [Operations](@/book/05-operations.md).

### Grad function

The node that produced a tensor, stored in `AutogradMeta::grad_fn` and returned by `Tensor::grad_fn()`. A tensor with a null grad function is a leaf. See [Autograd](@/book/06-autograd.md).

### Grad mode

A per-thread flag that decides whether operations record autograd history. `NoGradGuard` turns recording off for a scope, `EnableGradGuard` turns it on, and the thread pool copies the caller's mode into each job. See [Autograd](@/book/06-autograd.md).

### Gradcheck

A test utility that compares analytic gradients with central finite differences. kael's `autograd::gradcheck` requires `float64` leaf inputs that require grad, because the finite-difference estimate is too noisy in `float32`. See [Autograd](@/book/06-autograd.md).

### Gradient accumulation

The rule that gradients add up instead of overwriting each other. A value used in several places receives the sum of the contributions, and a leaf's `grad` keeps growing across `backward()` calls until `zero_grad()` resets it. See [Autograd](@/book/06-autograd.md).

### Gradient clipping

Scaling gradients down when they are too large. `clip_grad_norm_(parameters, max_norm)` treats all gradients as one vector and rescales them in place when its L2 norm exceeds `max_norm`. See [Neural networks and training](@/book/07-neural-networks.md).

### Grain

The smallest range of work worth giving to another thread in `parallel_for`. Below it, waking a thread costs more than the work saves; the default for elementwise work is `kElementGrain = 524288` elements. See [CPU performance](@/book/08-cpu-performance.md).

## H

### Handle

An object that refers to shared state instead of owning a copy of it. A kael `Tensor` is a handle: it holds one `std::shared_ptr<TensorImpl>`, so copying a tensor shares the data and `clone()` copies it. See [Memory model](@/book/02-memory-model.md) and [Tensor as a shared handle](@/adr/0002-tensor-as-handle.md).

### Host emulation

A build of the CUDA backend interface that runs the device kernels on the CPU, selected with `-DKAEL_FAKE_CUDA=ON`. It visits elements in reverse order, so a kernel that depends on element order fails in tests without a GPU. See [CUDA backend](@/book/09-cuda-backend.md).

## I

### In-place operation

An operation that writes its result into an existing tensor instead of allocating a new one, named with a trailing underscore such as `add_` or `copy_`. Every in-place operation bumps the storage's version counter. See [Operations](@/book/05-operations.md) and [Views](@/book/03-views.md).

### Inner loop

The function a `TensorIterator` calls for one run of elements along the innermost dimension. It receives one byte pointer and one byte stride per operand, output first, and an element count. See [TensorIterator](@/book/04-tensor-iterator.md).

## K

### Kahn's algorithm

A topological ordering algorithm that repeatedly processes nodes whose count of unprocessed incoming edges is zero. The autograd engine uses it to run a node only after every consumer has delivered its gradient. See [Autograd](@/book/06-autograd.md).

### KV cache

The per-layer store of the keys and values of every position already processed during generation. With it, each generation step runs the model over the new token only, instead of recomputing every earlier position. kael stores it in `LayerCache`. See [Transformer](@/book/11-transformer.md).

## L

### Layer normalization

A normalization that rescales the trailing dimensions of each input to zero mean and unit variance, then applies a learned per-element scale and shift: `(x - mean) / sqrt(var + eps) * weight + bias`. See [Neural networks and training](@/book/07-neural-networks.md).

### Leaf

A tensor with no grad function, created by the user or a factory instead of by a recorded operation. Leaves that require grad, such as parameters, are where the backward pass stores gradients. See [Autograd](@/book/06-autograd.md).

### Learning rate scheduler

An object that changes the optimizer's learning rate over training, for example `StepLR`, `CosineAnnealingLR`, `LinearWarmup` or `WarmupCosineLR`. It keeps a step counter and computes each rate from the base rate recorded at construction. See [Neural networks and training](@/book/07-neural-networks.md).

### Logits

The unnormalized scores a model outputs before softmax. Losses such as cross entropy and binary cross entropy with logits take logits directly, because computing probabilities first loses precision. See [Neural networks and training](@/book/07-neural-networks.md).

### Logsumexp

The function `log(sum(exp(x)))`, computed stably as `max(x) + log(sum(exp(x - max(x))))`. kael replaces an infinite maximum with 0 before the shift, so all-minus-infinity and plus-infinity inputs give the same results as PyTorch. See [Operations](@/book/05-operations.md).

## M

### Matmul

Matrix multiplication, with batching and broadcasting over leading dimensions. The book builds it in six versions, from a naive triple loop to a packed, SIMD and threaded kernel. See [Operations](@/book/05-operations.md) and [CPU performance](@/book/08-cpu-performance.md).

### Memory mapping

Mapping a file into the address space, so that pages are read from disk only when touched and read-only pages are shared with the operating system's file cache. kael's `LoadMode::Mapped` builds tensors whose storage points into a mapped safetensors file. See [Loading GPT-2](@/book/12-model-loading.md).

### Micro-kernel

The innermost routine of a high-performance matmul. It computes a small block of the output entirely in registers over a range of the inner dimension and stores it once. See [CPU performance](@/book/08-cpu-performance.md).

### Module

A unit of a neural network that owns named parameters and named child modules, and defines a `forward` function. Tree operations such as listing parameters, moving to a device or switching to evaluation mode recurse through the children. See [Neural networks and training](@/book/07-neural-networks.md).

## N

### NaN

Not a Number, the floating point value that results from undefined operations such as `0 / 0` or `inf - inf`. It compares unequal to everything, including itself, and propagates through most arithmetic. See [Operations](@/book/05-operations.md) and [Floating point for deep learning](@/concepts/floating-point.md).

### Node

One backward function in the autograd graph. It maps the gradient of its output to the gradients of its inputs and sends each one along an edge. See [Autograd](@/book/06-autograd.md).

## O

### Offset

The position of a tensor's first element inside its storage, counted in elements. Views such as `select` and `slice` change the offset instead of copying data. See [Shapes and strides](@/book/01-shapes-and-strides.md) and [Views](@/book/03-views.md).

### Optimizer

An object that updates parameters from their gradients, such as SGD with momentum, Adam or AdamW. It also clears gradients with `zero_grad()` before the next step. See [Neural networks and training](@/book/07-neural-networks.md).

## P

### Packing

Copying blocks of matmul operands into the exact order the micro-kernel reads them, so every load in the inner loop has unit stride. The copy costs less than the cache misses it removes. See [CPU performance](@/book/08-cpu-performance.md).

### Parameter

A leaf tensor that requires grad and that a module registers by name. Optimizers update parameters, and modules list them recursively. See [Neural networks and training](@/book/07-neural-networks.md).

### Pre-layer-norm

The transformer block layout that applies layer normalization before attention and before the MLP, inside the residual branch: `x + attention(ln1(x))`, then `x + mlp(ln2(x))`. GPT-2 uses it, with one more layer norm after the last block. See [Transformer](@/book/11-transformer.md).

### Pretokenization

Splitting text into words with a regular expression before BPE runs, so merges never cross word boundaries. GPT-2's pattern separates contractions, letters, numbers, punctuation and whitespace. See [Loading GPT-2](@/book/12-model-loading.md).

## R

### Reduction

An operation that combines the elements along one or more dimensions into fewer values, such as `sum`, `mean`, `max` or `argmax`. With `keepdim` the reduced dimensions stay as size 1, which lets the result broadcast against the input. See [Operations](@/book/05-operations.md).

### Reshape

An operation that reinterprets a tensor's elements in row-major order under a new shape. In kael, `reshape` is `contiguous().view(shape)`: it returns a view when the input is contiguous and a copy otherwise. See [Views](@/book/03-views.md).

### Residual connection

Adding a block's input to its output, `x + f(x)`, so gradients have a direct path through deep networks. Each transformer block has two, one around attention and one around the MLP. See [Transformer](@/book/11-transformer.md).

### Retain grad

The call `retain_grad()`, which asks a non-leaf tensor to store its gradient during the backward pass. Intermediate tensors do not store gradients otherwise. See [Autograd](@/book/06-autograd.md).

### Row-major order

The memory layout that stores the last dimension contiguously: a matrix is stored one row after another. It is the default in C, NumPy, PyTorch and kael. See [Shapes and strides](@/book/01-shapes-and-strides.md).

## S

### Safetensors

A file format for named tensors: an 8-byte header length, a JSON header with dtype, shape and byte offsets for each tensor, then the raw data. It contains no code, so loading an untrusted file cannot execute anything. kael reads and writes `F32`, `F64`, `I64` and `BOOL`. See [Loading GPT-2](@/book/12-model-loading.md).

### Sampling

Choosing the next token from the model's output distribution during generation. kael's `sample_next` divides logits by a temperature, optionally keeps only the top k, then draws from the softmax with `multinomial`; temperature 0 selects the argmax. See [Transformer](@/book/11-transformer.md).

### Sanitizer

Compiler instrumentation that checks a program's behavior at run time and aborts with a report at the first error. kael builds with AddressSanitizer, UndefinedBehaviorSanitizer and ThreadSanitizer through CMake options. See [Serving and tooling](@/book/13-serving-and-tooling.md).

### Saved tensor

A tensor that a node keeps from the forward pass because its backward formula needs it. `autograd::SavedTensor` records the version counter when it saves the tensor and throws if an in-place write changed it before backward. See [Autograd](@/book/06-autograd.md).

### Scalar

A plain number passed to a factory or an operation, such as the `2.5` in `t + 2.5`. kael's `Scalar` stores it with a dtype category, and a scalar never raises the precision of a tensor within the same category. See [Memory model](@/book/02-memory-model.md) and [TensorIterator](@/book/04-tensor-iterator.md).

### Server-sent events

A one-way HTTP streaming format, abbreviated SSE, with content type `text/event-stream`, in which each event is one or more `data:` lines followed by a blank line. The generation server streams decoded text with it. See [Serving and tooling](@/book/13-serving-and-tooling.md).

### Shape

The number of elements along each dimension of a tensor. A matrix with 2 rows and 3 columns has shape `[2, 3]`, and the product of the sizes is the element count. See [Shapes and strides](@/book/01-shapes-and-strides.md).

### Softmax

The function that turns a vector of scores into probabilities: `exp(x_i) / sum_j exp(x_j)`. kael subtracts the maximum before exponentiating, which leaves the result unchanged and prevents overflow. See [Operations](@/book/05-operations.md).

### Storage

The object that owns a block of bytes on one device. It has no shape and no dtype; several `TensorImpl` objects can share one storage through a `std::shared_ptr`. See [Memory model](@/book/02-memory-model.md).

### Stride

For one dimension, the number of elements to skip in storage to move one step along that dimension. A contiguous `[2, 3]` tensor has strides `[3, 1]`, and the address of element `[i, j]` is `offset + i * 3 + j`. See [Shapes and strides](@/book/01-shapes-and-strides.md).

### Sum to shape

The reduction that undoes broadcasting in the backward pass. A gradient that arrives in the broadcast shape is summed over the dimensions that were added or stretched, so it gets the shape of the original input. See [Autograd](@/book/06-autograd.md).

## T

### Temperature

The divisor applied to logits before sampling. Values below 1 sharpen the distribution, values above 1 flatten it, and 0 means greedy selection of the argmax. See [Transformer](@/book/11-transformer.md).

### TensorImpl

The struct that pairs a storage with its interpretation: shape, strides, offset, dtype, device and optional autograd metadata. Strides and offset count elements, not bytes. See [Memory model](@/book/02-memory-model.md).

### TensorIterator

The component that prepares an elementwise operation: it computes the result dtype and shape, broadcasts the operands, coalesces dimensions, and runs a two-level loop that calls an inner loop for each run of elements. See [TensorIterator](@/book/04-tensor-iterator.md).

### Thread pool

A fixed set of worker threads that execute parallel work. kael's `parallel_for` splits a range into chunks that workers and the calling thread claim from an atomic cursor. See [CPU performance](@/book/08-cpu-performance.md).

### Top-k sampling

A sampling rule that keeps only the k largest logits and masks the rest to minus infinity before the softmax. It removes the long tail of unlikely tokens. See [Transformer](@/book/11-transformer.md).

### Type promotion

The rule that picks the result dtype when operands of different dtypes meet. kael's `promote_types` follows the order `Bool < I64 < F32 < F64`, and scalars and 0-d tensors only win when they belong to a higher category. See [TensorIterator](@/book/04-tensor-iterator.md) and [Memory model](@/book/02-memory-model.md).

## V

### Vector-Jacobian product

The product of an output gradient with the Jacobian of a function, which gives the gradient of each input. Backward functions compute it directly, without forming the Jacobian: for `y = a * b` the gradient of `a` is `g * b`, an elementwise product. See [Autograd](@/book/06-autograd.md).

### Version counter

A counter on a storage that every in-place operation increments. Because it lives on the storage, every view of a buffer reports the same version, and autograd uses it to detect that a saved tensor was modified. See [Views](@/book/03-views.md).

### View

A tensor that shares the storage of another tensor and differs only in metadata: shape, strides and offset. Transpose, slice, select, expand and view create views without copying data, and a write through a view reaches the original. See [Views](@/book/03-views.md).

## W

### Weight initialization

The choice of the initial scale of the weights so that activations and gradients keep a stable magnitude through depth. Xavier and Kaiming initialization derive the scale from the fan-in and fan-out of each weight. See [Neural networks and training](@/book/07-neural-networks.md).

### Weight tying

Using the token embedding matrix both to embed input tokens and to project the final hidden states to vocabulary logits. GPT-2 does this, and kael computes the logits with `token_embedding_->weight()`. See [Transformer](@/book/11-transformer.md).

## Z

### Zero-dimensional tensor

A tensor with an empty shape `[]`, which holds exactly one element. In type promotion it behaves like a scalar: it decides the result dtype only when its category is higher than that of the tensors with dimensions. See [TensorIterator](@/book/04-tensor-iterator.md).
