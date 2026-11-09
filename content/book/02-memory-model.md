+++
title = "Memory model"
description = "How kael turns shape arithmetic into real tensors: element types, storage, TensorImpl, the Tensor handle, factories, random numbers, printing and errors."
weight = 2
slug = "memory-model"

[extra]
chapter = "02"
kicker = "Chapter 02"
builds = "Data types, storage, TensorImpl, the Tensor handle, factories and random numbers"
source = "docs/02-memory-model.md"
tag = "chapter-03-core"
+++

This chapter turns the shape arithmetic of chapter 01 into real tensors. It maps to phase 3 of the project (tag `chapter-03-core`). By the end there is a `Tensor` you can create, read, write, clone, convert between element types, fill with random values and print. Later phases added a caching allocator under `Storage`, a CUDA device, autograd metadata and typed errors. This chapter describes the current code and marks those additions where they appear, so you can tell the phase 3 core from what came later.

| File | Role |
|------|------|
| [`include/kael/error.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/error.h) | `kael::Error`, `IndexError`, `ValueError`, `TypeError` and the `KAEL_CHECK_*` macros |
| [`include/kael/dtype.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/dtype.h) | `DType`, `Device`, element sizes, type traits, `convert_value`, `promote_types` and the `dispatch` family |
| [`include/kael/scalar.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/scalar.h) | `Scalar`, a number with a dtype category, for factories and operations with a constant |
| [`include/kael/device.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/device.h), [`src/device.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/device.cpp) | `parse_device`, `cuda_is_available` and `cuda_synchronize` |
| [`include/kael/storage.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/storage.h), [`src/storage.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/storage.cpp) | `Storage`: an owned or borrowed block of bytes on one device |
| [`include/kael/allocator.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/allocator.h), [`src/allocator.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/allocator.cpp) | The caching block allocator behind `Storage` (phase 9, covered in depth in chapter 08) |
| [`include/kael/tensor_impl.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/tensor_impl.h), [`src/tensor_impl.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/tensor_impl.cpp) | `TensorImpl`: storage plus shape, strides, offset, dtype, device; metadata validation |
| [`include/kael/tensor.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/tensor.h), [`src/tensor.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/tensor.cpp) | `Tensor`: the shared handle, element access, `to`, `clone`, `contiguous` |
| [`include/kael/factory.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/factory.h), [`src/factory.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/factory.cpp) | `empty`, `zeros`, `ones`, `full`, `arange`, `linspace`, `eye`, `scalar`, `from_vector` |
| [`include/kael/random.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/random.h), [`src/random.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/random.cpp) | The global generator and `rand`, `randn`, `randint`, `randperm`, `bernoulli`, `multinomial` |
| [`src/print.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/print.cpp) | `operator<<` for `Tensor` |
| [`include/kael/kael.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/kael.h) | Umbrella header |
| [`include/kael/version.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/version.h), [`src/version.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/version.cpp) | `kael::version()` |

## Prerequisites

- [01. Shapes and strides](@/book/01-shapes-and-strides.md): `Shape`, `Strides`, `numel`, `contiguous_strides`, `offset_from_index`, `index_from_linear`, `wrap_dim`, `dim_position` and the checked arithmetic helpers `detail::checked_mul` and `detail::checked_add`.
- The design records [ADR 0001](@/adr/0001-runtime-dtype.md) (runtime dtype), [ADR 0002](@/adr/0002-tensor-as-handle.md) (tensor as a handle) and [ADR 0004](@/adr/0004-pytorch-shaped-api.md) (a PyTorch-shaped API with typed errors).
- Basic C++20: `std::shared_ptr`, `std::unique_ptr` with a custom deleter, `if constexpr`, concepts, and lambdas with a template parameter list.

## Why this layer exists

Chapter 01 ended with pure arithmetic. Given a shape and strides, it can say that element `[1, 2]` of a `[2, 3]` row-major tensor lives at position 5. Nothing yet holds a position 5. There is no memory, no element type and no object a user can pass around.

A deep learning framework needs that object to settle five things for every tensor:

1. **Location.** Some block of memory, on the CPU or on a GPU, has to exist and be freed exactly once.
2. **Meaning.** The same 24 bytes can be six `float` values, three `double` values or 24 bools. The shape and strides say which bytes form which element.
3. **Element type.** A function such as `sum` has to work for every element type without the user picking a template instantiation.
4. **Sharing.** A transpose or a slice should not copy, so several tensors must be able to read the same memory, and the memory must live as long as any of them.
5. **Safety.** A wrong stride would read memory that belongs to someone else. Something has to refuse such a tensor before any element is read.

Without a deliberate answer to each point, the failure modes are concrete. Two objects that both own a buffer free it twice. A tensor whose strides reach past the end of its buffer reads garbage or crashes, and in C++ it is undefined behavior, so the crash may happen far away from the bug. A templated `Tensor<float>` cannot sit in the same container as a `Tensor<double>`, so the autograd graph and the Python bindings each need their own type erasure. A copy on every transpose makes `x.T()` cost as much memory as `x`.

kael settles all five with three objects stacked on top of each other. Hold this mental model through the rest of the book:

```text
Tensor       the handle you pass around     "which tensor"
TensorImpl   the lens over the bytes        "how to read the bytes"
Storage      the bytes                      "where the bytes are"
```

`Storage` is a dumb block of bytes on one device. `TensorImpl` is a lens: shape, strides, offset and element type that say how to read the block. `Tensor` is a cheap pointer to one lens. Copying a `Tensor` copies the pointer. Making a view (chapter 03) makes a new lens over the same block. Cloning makes a new block.

PyTorch uses the same split: `at::Tensor` holds an intrusive pointer to `TensorImpl`, which holds a `Storage`. NumPy folds the lens and the handle into one `ndarray` object, and the `base` attribute points to the array that owns the memory. The vocabulary in this chapter maps directly to both.

## Concepts

### Three layers

A tensor is three objects stacked on each other:

```text
Tensor              a handle: one std::shared_ptr<TensorImpl>
  |
  v
TensorImpl          the interpretation: shape, strides, offset, dtype, device, autograd metadata
  |
  v
Storage             the memory: nbytes bytes on one device
  |
  v
[ raw bytes ... ]   a block from the caching allocator, or external memory
```

Each layer knows only what it needs. `Storage` has no shape and no dtype: it is bytes. `TensorImpl` does not own the bytes, it shares them through a `std::shared_ptr<Storage>`. `Tensor` holds no data and no metadata of its own, only a pointer to an impl, which is why passing it by value is cheap.

The **dtype** (data type) is the element type of a tensor, such as `float32`. The **device** is where the bytes live: host memory (`cpu`) or GPU memory (`cuda`). The **offset** is the position, in elements, of the first element inside the storage. A **view** is a tensor that shares its storage with another tensor but has its own shape, strides or offset. See the [glossary](@/concepts/glossary.md) for these and later terms.

### Identity versus sharing

Once views exist (chapter 03), the ownership graph has two kinds of edges. Take `a = arange(6).view({2, 3})`, `b = a` and `t = a.transpose(0, 1)`:

```text
 Tensor a ---------+
                   +--> TensorImpl  shape [2, 3]  strides [3, 1] --+
 Tensor b = a -----+                                               |
                                                                   +--> Storage (24 bytes)
 Tensor t = a.transpose(0, 1) --> TensorImpl  shape [3, 2]         |
                                              strides [1, 3] ------+
```

`a` and `b` are the same tensor: one impl, two handles. `t` is a different tensor, with its own impl and its own metadata, over the same bytes. Both relations matter, and `Tensor` has a query for each:

- `is_same(other)` compares impl pointers. It answers "are these handles to one tensor".
- `shares_storage(other)` compares storage pointers. It answers "can a write through one change the other".

For the graph above, `a.is_same(b)` is true, `a.is_same(t)` is false, and `a.shares_storage(t)` is true. PyTorch has no public `is_same`, but `a is b` in Python and `a.data_ptr() == t.data_ptr()` are close equivalents.

The distinction becomes important in two places later in the book. In-place operations (chapter 05) change every tensor that shares storage, not only the handles of one tensor. Autograd (chapter 06) attaches gradient metadata to an impl, so `b.requires_grad_()` also changes `a` but not `t`.

### Element types are runtime values

`kael::Tensor` is one class. Its dtype is a field in the impl, not a template parameter. The rejected alternative is `Tensor<float>`, `Tensor<double>` and so on, and it fails in three places:

- A `std::vector<Tensor>` of mixed types needs a common base class or a `std::variant`, which is a runtime dtype by another name.
- The autograd graph stores tensors of every type in its nodes. With a templated tensor, each node type would be a template too, and a graph that mixes `float` and `int64` indices would not fit in one container.
- Python has one `Tensor` type. The bindings would have to erase the template parameter anyway.

The cost of the runtime choice is that code which touches elements needs the C++ type back. `dispatch` provides it with one `switch`, described in [Dispatch](#dispatch-from-a-runtime-dtype-to-a-c-type) below. PyTorch makes the same trade with its `AT_DISPATCH_*` macros. [ADR 0001](@/adr/0001-runtime-dtype.md) records the alternatives in full.

### Addresses count elements, not bytes

Strides and offset in kael count elements. The byte address of an element is:

```text
byte address = storage.data()
             + (offset + sum over i of index[i] * strides[i]) * element_size(dtype)
```

A worked example: element `[1, 2]` of a contiguous `[2, 3]` float32 tensor, strides `[3, 1]`, offset 0.

```text
element position = 0 + 1 * 3 + 2 * 1 = 5
byte position    = 5 * 4             = 20
storage bytes:   [ 0..3 | 4..7 | 8..11 | 12..15 | 16..19 | 20..23 ]
element:            0      1       2       3        4        5
                                                             ^ [1, 2]
```

Element units keep the stride arithmetic of chapter 01 independent of the dtype. Converting the same tensor to float64 with `to(DType::F64)` produces the same strides `[3, 1]`. Only the last multiplication changes, from 4 to 8. PyTorch also counts strides in elements. NumPy counts them in bytes, so `np.zeros((2, 3), np.float32).strides` is `(12, 4)`. The TensorIterator in chapter 04 converts to byte strides internally, at the last moment, because a loop over several operands of different dtypes needs bytes.

### Converting values between element types

Every value that enters a tensor of a given dtype passes through one conversion function. The rules come from two hazards in C++:

- A Bool element is one byte. A plain `static_cast<std::uint8_t>(2.5)` yields 2, which is neither true nor false.
- `static_cast<std::int64_t>(x)` is undefined behavior when `x` is NaN or outside the `int64` range. The compiler is allowed to assume it never happens.

The rules, traced on real values (checked by running the code):

| From | To | Input | Result | Rule |
|------|----|-------|--------|------|
| `double` | Bool | `2.5` | `1` | any non-zero value is true |
| `double` | Bool | NaN | `1` | `NaN != 0` is true, as in PyTorch |
| `double` | `I64` | `-2.7` | `-2` | in range: truncate toward zero |
| `double` | `I64` | `1e30` | `9223372036854775807` | above range: saturate to `INT64_MAX` |
| `double` | `I64` | `-inf` | `INT64_MIN` | below range: saturate to `INT64_MIN` |
| `double` | `I64` | NaN | `0` | NaN maps to zero, as Rust's `as` does |
| `I64` | `F32` | any | nearest float | plain `static_cast` |

### Type promotion

When two dtypes meet in an operation, the result type follows a fixed order:

```text
Bool  <  I64  <  F32  <  F64
```

`promote_types(a, b)` returns the higher of the two. `I64` with `F32` gives `F32`, as in PyTorch. NumPy differs here: `int64 + float32` gives `float64` in NumPy, because NumPy wants every int64 value to survive the conversion. PyTorch and kael prefer to keep the default float type.

A plain number, a **scalar**, follows a softer rule. A scalar has a category (bool, integer or floating), and it only changes the result dtype when its category is higher than the tensor's:

| Tensor dtype | Scalar | Result | Why |
|--------------|--------|--------|-----|
| `F32` | `2.5` | `F32` | same category, the tensor wins |
| `F64` | `2.5` | `F64` | same category, the tensor wins |
| `I64` | `2.5` | `F32` | floating beats integer, default float is `F32` |
| `Bool` | `3` | `I64` | integer beats bool |
| `I64` | `true` | `I64` | bool does not beat integer |

This is why `t * 0.5` on a float32 tensor does not silently produce float64. Chapter 04 builds the full promotion rules of the iterator on these two functions.

### Contiguity

A tensor is **contiguous**, in kael's definition, when its elements visited in row-major order sit at consecutive storage positions with no gaps. In terms of metadata: its strides equal `contiguous_strides(shape)`, with two relaxations explained in the implementation section. Many kernels have a fast path for contiguous inputs, `clone` of a contiguous tensor is one `memcpy`, and `view` in chapter 03 requires contiguous input.

A worked check of the transposed tensor `t` from the graph above, shape `[3, 2]`, strides `[1, 3]`:

```text
walk from the last dimension:
  dim 1: size 2, stride 3, expected 1   -> mismatch, not contiguous
```

And of `a`, shape `[2, 3]`, strides `[3, 1]`:

```text
  dim 1: size 3, stride 1, expected 1   -> ok, expected becomes 1 * 3 = 3
  dim 0: size 2, stride 3, expected 3   -> ok, expected becomes 3 * 2 = 6
  contiguous
```

### The reachable extent of a layout

The safety check of this chapter rests on one observation. Because strides are non-negative, the furthest element a tensor can reach has every index at its maximum:

```text
last_element = offset + sum over i of (shape[i] - 1) * strides[i]
bytes_needed = (last_element + 1) * element_size(dtype)
must hold:     bytes_needed <= storage.nbytes()
```

Three worked cases:

```text
contiguous [2, 3] float32, strides [3, 1], offset 0, 24-byte storage
  last_element = 0 + (2 - 1) * 3 + (3 - 1) * 1 = 5
  bytes_needed = (5 + 1) * 4 = 24          24 <= 24   valid

strided view: 10 float32 values (40 bytes), shape [3], stride [2], offset 4
  reads elements 4, 6 and 8
  last_element = 4 + (3 - 1) * 2 = 8
  bytes_needed = (8 + 1) * 4 = 36          36 <= 40   valid

same [2, 3] layout as the first case, but offset 1
  last_element = 1 + 5 = 6
  bytes_needed = (6 + 1) * 4 = 28          28 >  24   rejected
```

The third case is what a view with an off-by-one in its offset computation would produce. kael rejects it when the tensor is built, with this message (from a real run, path shortened):

```text
kael error at src/tensor_impl.cpp:31: check 'bytes_needed <= impl.storage->nbytes()' failed:
tensor with shape [2, 3], strides [3, 1], offset 1 and dtype float32 needs 28 bytes but storage has 24
```

## Implementation

The sections below follow the order in which you would write the code: errors first, because every other file throws them, then element types, storage, the impl, the handle, and the functions that create tensors.

### Errors and checks

Every kael error is a `kael::Error`, derived from `std::runtime_error`. Three subclasses classify the cause, and the Python bindings translate each to the matching Python exception.

| Class | Used for | Python exception |
|-------|----------|------------------|
| `IndexError` | Element index, dimension or position out of range | `IndexError` |
| `ValueError` | Bad shapes, sizes or argument values | `ValueError` |
| `TypeError` | Unsupported or mismatched dtype | `TypeError` |
| `Error` | Everything else: internal invariants, device problems | `RuntimeError` |

The subclasses arrived in phase 10 (`chapter-10-cuda`), one phase before the Python bindings, because Python users expect `t[5]` on a 3-element tensor to raise `IndexError` and not a generic runtime error. Before that every check threw `kael::Error`, which is still the base, so code that catches `kael::Error` catches all of them.

From `include/kael/error.h`:

```cpp
// `msg` is not parenthesized so that stream chains like "index " << i work.
// NOLINTBEGIN(bugprone-macro-parentheses)
#define KAEL_CHECK_AS(exception, cond, msg)                                                        \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            ::std::ostringstream kael_check_stream_;                                               \
            kael_check_stream_ << msg;                                                             \
            ::kael::detail::fail<exception>(__FILE__, __LINE__, #cond, kael_check_stream_.str());  \
        }                                                                                          \
    } while (0)
// NOLINTEND(bugprone-macro-parentheses)

#define KAEL_CHECK(cond, msg) KAEL_CHECK_AS(::kael::Error, cond, msg)
#define KAEL_CHECK_INDEX(cond, msg) KAEL_CHECK_AS(::kael::IndexError, cond, msg)
#define KAEL_CHECK_VALUE(cond, msg) KAEL_CHECK_AS(::kael::ValueError, cond, msg)
#define KAEL_CHECK_TYPE(cond, msg) KAEL_CHECK_AS(::kael::TypeError, cond, msg)
```

How the macro works, piece by piece:

- The condition is evaluated once, and the message is built only when the condition is false. A passing check costs one branch, so checks can stay on in release builds.
- `msg` is pasted into `kael_check_stream_ << msg`, so a call site can write a stream chain: `KAEL_CHECK_VALUE(n >= 0, "size must be non-negative, got " << n)`. That is why `msg` is not wrapped in parentheses, and why the clang-tidy warning about it is silenced.
- `#cond` turns the condition into text, so the message shows the exact expression that failed.
- `detail::fail<Exception>` formats `kael error at <file>:<line>: check '<condition>' failed: <message>` and throws. It is marked `[[noreturn]]`, so the compiler knows that code after a failed check is unreachable and does not warn about missing return values.
- The `do { ... } while (0)` wrapper makes the macro behave as one statement after an `if` without braces, and it requires the trailing semicolon at the call site.

A macro is used instead of a function because only a macro can capture the condition text, the file and the line at the call site, and avoid building the message when the check passes. C++20 `std::source_location` could replace `__FILE__` and `__LINE__`, but not `#cond` or the lazy message.

A real message, from asking a float32 tensor for a `double*` (path shortened):

```text
kael error at include/kael/tensor.h:168: check 'dtype_of_v<T> == dtype()' failed: expected dtype float64 but tensor has float32
```

### The element type enum and its tables

From `include/kael/dtype.h`:

```cpp
// Bool is stored as std::uint8_t: sizeof(bool) is not fixed by the standard and
// std::vector<bool> cannot hand out element pointers.
enum class DType { F32, F64, I64, Bool };

enum class Device { CPU, CUDA };
```

Four dtypes cover everything the book needs: `F32` for parameters and activations, `F64` for gradient checks and reference computations, `I64` for indices and token ids, and `Bool` for masks and comparison results. PyTorch has many more (float16, bfloat16, int8, complex). Each new dtype multiplies the number of kernel instantiations, so kael keeps the set small.

Each dtype has a size, a name and a C++ type:

| `DType` | `to_string` | `element_size` | C++ type (`cpp_type_t`) |
|---------|-------------|----------------|-------------------------|
| `F32` | `float32` | 4 | `float` |
| `F64` | `float64` | 8 | `double` |
| `I64` | `int64` | 8 | `std::int64_t` |
| `Bool` | `bool` | 1 | `std::uint8_t` |

The mapping exists in both directions as compile-time traits. `dtype_of_v<float>` is `DType::F32`, and `cpp_type_t<DType::F32>` is `float`. `dtype_of` has no primary definition, only specializations:

```cpp
template <typename T> struct dtype_of;

template <> struct dtype_of<float> {
    static constexpr DType value = DType::F32;
};
// ... double, std::int64_t, std::uint8_t

template <typename T> inline constexpr DType dtype_of_v = dtype_of<T>::value;
```

So `dtype_of_v<int>` does not compile. That is deliberate: `int` is not a kael element type, and a silent conversion to `I64` would hide a mistake. The same header also provides `is_floating_point(dtype)` and an `operator<<` that prints the name, so `std::cout << DType::I64` prints `int64`.

The `switch` statements over `DType` (`element_size`, `to_string`, `dispatch`) have no `default` branch:

```cpp
constexpr std::int64_t element_size(DType dtype) {
    switch (dtype) {
    case DType::F32:
        return 4;
    case DType::F64:
    case DType::I64:
        return 8;
    case DType::Bool:
        return 1;
    }
    detail::unreachable();
}
```

With `-Wall`, the compiler warns about every switch that misses an enumerator, so adding a fifth dtype points at every place that needs a new case. A `default` branch would silence that warning and turn a compile-time reminder into a runtime bug. After the switch, `detail::unreachable()` throws a `kael::Error` with the message `kael internal error: reached unreachable code`. That covers an enum value that is out of range (for example from a corrupted cast) without undefined behavior, which `std::unreachable()` would be.

### Devices

`Device` sits next to `DType` in the same header because both are plain enums used by every layer. `include/kael/device.h` declares three functions:

- `parse_device(name)` maps `"cpu"` to `Device::CPU`, and `"cuda"` or `"cuda:0"` to `Device::CUDA`. Anything else throws `ValueError` with `unknown device 'gpu', expected 'cpu' or 'cuda'`.
- `cuda_is_available()` and `cuda_synchronize()` forward to the device backend described in chapter 09.

The Python bindings and the C++ examples (`examples/mnist_mlp.cpp`, `examples/char_gpt.cpp`, `examples/gpt2_generate.cpp`) share `parse_device`, so both accept the same device names. kael has one CUDA device, which is why `cuda:1` is rejected rather than ignored.

### Dispatch: from a runtime dtype to a C++ type

Code that touches elements needs the C++ type back. `dispatch` is a `switch` that calls a templated lambda with the right type.

From `include/kael/dtype.h`:

```cpp
template <typename F> decltype(auto) dispatch(DType dtype, F&& f) {
    switch (dtype) {
    case DType::F32:
        return f.template operator()<float>();
    case DType::F64:
        return f.template operator()<double>();
    case DType::I64:
        return f.template operator()<std::int64_t>();
    case DType::Bool:
        return f.template operator()<std::uint8_t>();
    }
    detail::unreachable();
}
```

The lambda uses the C++20 template parameter list syntax, `[&]<typename T>() { ... }`. Its call operator is a template that cannot deduce `T` from arguments (there are none), so `dispatch` names the type explicitly with `f.template operator()<float>()`. The `template` keyword tells the parser that `operator()` is a template, because `F` is a dependent type. An operation is written once and the compiler instantiates it for each dtype.

A real user, from `src/factory.cpp`:

```cpp
Tensor full(const Shape& shape, Scalar value, DType dtype, Device device) {
    Tensor t = empty(shape, dtype, device);
    dispatch(dtype, [&]<typename T>() {
        const T converted = value.as<T>();
        if (device == Device::CPU) {
            std::fill_n(t.data_ptr<T>(), t.numel(), converted);
        } else {
            device::fill(dtype, t.device_data(), t.numel(), &converted);
        }
    });
    return t;
}
```

Trace `full({2}, 2.5, DType::Bool)`:

```text
dispatch(Bool, lambda)          -> case DType::Bool
lambda.operator()<uint8_t>()    -> T = std::uint8_t
value.as<uint8_t>()             -> convert_value<uint8_t>(2.5) = 1
std::fill_n(ptr, 2, 1)          -> storage bytes [01 01]
printed                         -> tensor([true, true], shape=[2], dtype=bool)
```

The return type is `decltype(auto)`, so `dispatch` returns whatever the lambda returns, including `void`. All four instantiations must return the same type, otherwise the function has inconsistent return statements and does not compile. `Tensor::item()` uses this to return a `double` from any dtype.

Two restricted variants serve operations that do not make sense for every dtype:

- `dispatch_floating(dtype, op, f)` instantiates only `float` and `double`, and throws `TypeError` with `"<op> only supports floating point dtypes, got <dtype>"` for the others. `linspace` and the random samplers use it. `rand({2}, DType::I64)` fails with `rand only supports floating point dtypes, got int64`.
- `dispatch_no_bool(dtype, op, f)` instantiates `float`, `double` and `std::int64_t`, and throws `TypeError` with `"<op> does not support dtype bool"` for `Bool`.

The restricted variants also cut compile time and binary size: a kernel that only makes sense for floats is not instantiated for integers. With four dtypes and binary operations that dispatch on two dtypes, the full product is 16 instantiations per kernel, so the savings add up.

### Why Bool is a `std::uint8_t`

Bool elements are stored as one byte holding 0 or 1, for the two reasons in the comment above the enum:

- `sizeof(bool)` is implementation defined. It is 1 on every mainstream compiler, but the standard does not promise it, so the storage size of a Bool tensor would not be portable.
- `std::vector<bool>` is a specialization that stores packed bits. It cannot return a `bool*` to its elements, so kael could not copy from it with a pointer or `std::memcpy`.

`from_vector` therefore has a separate non-template overload for `std::vector<bool>` and `std::initializer_list<bool>`, which copies element by element with `std::copy`. Each `bool` converts to 0 or 1 on the way.

The byte representation has one trap. `static_cast<std::uint8_t>(2.5)` is 2, which is not a valid bool. A Bool tensor holding the byte 2 would print `true` but compare unequal to another true element holding 1. `convert_value` closes the trap.

From `include/kael/dtype.h`:

```cpp
// A plain static_cast would turn 2.5 into the invalid bool value 2.
template <typename To, typename From> constexpr To convert_value(From value) {
    if constexpr (std::is_same_v<To, std::uint8_t>) {
        return value != From{0} ? To{1} : To{0};
    } else if constexpr (std::is_integral_v<To> && std::is_floating_point_v<From>) {
        return saturating_cast<To>(value);
    } else {
        return static_cast<To>(value);
    }
}
```

`if constexpr` selects one branch at compile time for each pair of types, so the function compiles down to a comparison, a saturating cast or a plain cast, with no runtime type test.

Every conversion into a tensor goes through it: `full(shape, 2.5, DType::Bool)` fills ones, `to(DType::Bool)` maps every non-zero to 1, and `from_vector(std::vector<std::uint8_t>{2, 0})` stores `{1, 0}`. NaN converts to true, because `NaN != 0` is true. That matches PyTorch.

### Saturating float to integer conversion

Conversions from floating point to `I64` go through `saturating_cast`. A `static_cast` of NaN or of a value outside the integer range is undefined behavior, so `saturating_cast` checks first:

```cpp
// Casting an out-of-range float to an integer is undefined behavior. Saturate instead, and map
// NaN to zero, as Rust's `as` does.
template <typename To, typename From> constexpr To saturating_cast(From value) {
    if (value != value) {
        return To{0};
    }
    constexpr auto lowest = static_cast<From>(std::numeric_limits<To>::min());
    if (value <= lowest) {
        return std::numeric_limits<To>::min();
    }
    if (value >= -lowest) {
        return std::numeric_limits<To>::max();
    }
    return static_cast<To>(value);
}
```

Line by line:

- `value != value` is true only for NaN. It avoids `std::isnan`, which is not `constexpr` before C++23.
- `lowest` is `INT64_MIN` converted to the float type: `-2^63`, which is exact in both `float` and `double` because it is a power of two.
- `-lowest` is `2^63`, also exact. The obvious bound `INT64_MAX` is `2^63 - 1`, which is not representable in `float` or `double` and rounds up to `2^63`. Comparing against it would let `2^63` through to the cast, which overflows. Comparing with `>= 2^63` is exact.
- In range, the cast truncates toward zero: `-2.7` becomes `-2`, as `TensorTo.FloatToIntTruncatesTowardsZero` pins down.

Out of range, `1e30` becomes `INT64_MAX`, `-inf` becomes `INT64_MIN`, and NaN becomes 0 (`FloatToInteger.SaturatesOutOfRangeValuesAndMapsNanToZero` and `FloatToInteger.KeepsTheExactLowerBound` in `tests/test_integer_overflow.cpp`). The CUDA copy kernel calls a device copy of the same function from `src/internal/element_functions.h`, which cannot use `std::numeric_limits` in device code.

In PyTorch, converting NaN or an out-of-range float to an integer gives a platform-dependent result. kael picks one defined answer, which also keeps UndefinedBehaviorSanitizer quiet in the sanitizer builds.

### Promotion in code

From `include/kael/dtype.h`:

```cpp
constexpr DType promote_types(DType a, DType b) {
    constexpr auto rank = [](DType dtype) {
        switch (dtype) {
        case DType::Bool:
            return 0;
        case DType::I64:
            return 1;
        case DType::F32:
            return 2;
        case DType::F64:
            return 3;
        }
        detail::unreachable();
    };
    return rank(a) >= rank(b) ? a : b;
}
```

The rank is a local lambda rather than the enum's numeric value. The enum order (`F32, F64, I64, Bool`) is chosen for readability, not for promotion, and coupling the two would make reordering the enum a silent behavior change. The same header defines `dtype_category(dtype)`, which returns 0 for Bool, 1 for `I64` and 2 for both floating types. `Scalar` uses it next.

### Scalar

Factories and operations accept plain numbers: `full({2}, 2.5)`, `t + 1`. `Scalar` carries such a number with a dtype category. It arrived in phase 5 with type promotion, and `full` uses it.

From `include/kael/scalar.h`:

```cpp
class Scalar {
public:
    constexpr Scalar(bool value) : value_(std::int64_t{value ? 1 : 0}), dtype_(DType::Bool) {}

    template <std::integral T>
    constexpr Scalar(T value) : value_(static_cast<std::int64_t>(value)), dtype_(DType::I64) {}

    template <std::floating_point T>
    constexpr Scalar(T value) : value_(static_cast<double>(value)), dtype_(DType::F32) {}

    [[nodiscard]] constexpr DType dtype() const noexcept { return dtype_; }

    template <typename T> [[nodiscard]] constexpr T as() const {
        return std::visit([](auto value) { return convert_value<T>(value); }, value_);
    }

private:
    std::variant<std::int64_t, double> value_;
    DType dtype_;
};
```

Design points:

- The constructors are implicit on purpose, so a literal converts at the call site: `full({2}, 2.5)` needs no `Scalar(2.5)`.
- The `bool` overload is a non-template. For `true` and `false`, overload resolution prefers a non-template over a template with an equally good match, so `bool` does not fall into the `std::integral` template (`bool` satisfies `std::integral`).
- Integers are stored exactly as `std::int64_t` and floating values as `double`. Storing everything as `double` would lose integers above `2^53`, such as large token ids. `as<T>()` converts once, through `convert_value`, at the point of use.

A floating scalar reports `F32`, not `F64`. The dtype of a scalar is a category with the default type of that category. `promote_with_scalar` uses the category:

```cpp
// A scalar never raises the precision of a tensor within the same category:
// F32 + 2.5 stays F32, while I64 + 2.5 moves to the default float, F32.
constexpr DType promote_with_scalar(DType tensor_dtype, Scalar scalar) {
    return dtype_category(scalar.dtype()) <= dtype_category(tensor_dtype) ? tensor_dtype
                                                                          : scalar.dtype();
}
```

The table in [Type promotion](#type-promotion) lists the results. PyTorch's type promotion documentation describes the same category rule for Python numbers and zero-dimensional tensors.

### Storage

`Storage` owns a block of bytes on one device.

From `include/kael/storage.h`:

```cpp
class Storage {
public:
    static constexpr std::size_t kAlignment = 64;

    explicit Storage(std::int64_t nbytes, Device device = Device::CPU);
    [[nodiscard]] static Storage external(std::byte* data, std::int64_t nbytes,
                                          std::shared_ptr<void> owner);

    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;
    Storage(Storage&&) noexcept = default;
    Storage& operator=(Storage&&) noexcept = default;
    ~Storage() = default;

    [[nodiscard]] std::byte* data() noexcept { return data_.get(); }
    [[nodiscard]] const std::byte* data() const noexcept { return data_.get(); }
    [[nodiscard]] std::int64_t nbytes() const noexcept { return nbytes_; }
    [[nodiscard]] Device device() const noexcept { return device_; }
    [[nodiscard]] std::int64_t version() const noexcept { return version_; }
    void bump_version() noexcept { ++version_; }

private:
    struct BlockDeleter {
        std::int64_t capacity;
        Device device;
        void operator()(std::byte* ptr) const noexcept;
    };

    std::unique_ptr<std::byte[], BlockDeleter> data_;
    std::shared_ptr<void> owner_;
    std::int64_t nbytes_;
    Device device_;
    std::int64_t version_ = 0;
};
```

The design rules, with the reason for each:

- **Not copyable.** Two owners of one block would free it twice. Sharing happens one level up, through `std::shared_ptr<Storage>` in `TensorImpl`. Deleting the copy operations turns an accidental copy into a compile error.
- **Movable.** A move transfers the `std::unique_ptr`, so the moved-from object owns nothing and its destructor frees nothing. `Storage::external` returns by value and relies on this.
- **Zero bytes means a null pointer.** The constructor allocates only when `nbytes > 0`. Allocating zero bytes has implementation-defined results (a null pointer or a unique non-null pointer), and an empty tensor never reads its storage.
- **Uninitialized memory.** `empty()` returns whatever the block held. `zeros` and the other factories decide what to write. Zeroing every block would waste a pass over memory for the many tensors that are written immediately, such as the output of a kernel.
- **64-byte alignment.** `kAlignment` is the size of a cache line on common CPUs and a multiple of the 16-, 32- and 64-byte SIMD vector widths (SSE, AVX2 and AVX-512), so a buffer never starts in the middle of a cache line or a vector. `Storage.AlignedTo64Bytes` checks sizes 1, 3, 64, 100 and 4097.
- **A version counter.** `version()` and `bump_version()` count in-place writes to the storage. Autograd uses it from phase 7 on to detect that a tensor saved for the backward pass was modified (chapter 06). It lives on the storage, not on the impl, so every view of the same bytes sees the same count (chapter 03).
- **Signed sizes.** `nbytes` is `std::int64_t`, like every size in kael. A negative request is caught by a check instead of wrapping to a huge `std::size_t`.

### The allocation path

The constructor does not call `operator new` itself. It asks the caching allocator for a block.

From `src/storage.cpp`:

```cpp
Storage::Storage(std::int64_t nbytes, Device device) : nbytes_(nbytes), device_(device) {
    KAEL_CHECK_VALUE(nbytes >= 0, "storage size must be non-negative, got " << nbytes << " bytes");

    if (nbytes > 0) {
        const std::int64_t capacity = detail::block_capacity(nbytes);
        data_ = {detail::allocate_block(capacity, device), BlockDeleter{capacity, device}};
    }
}
```

`block_capacity` rounds the request up to a bucket size: a multiple of 512 bytes up to 64 KiB, and the next power of two above that. Requests above `2^62` bytes throw `cannot reserve <n> bytes`.

```text
request      capacity    tensor that asks for it
1            512         a 1-element bool tensor
48           512         a [3, 4] float32 tensor
512          512
513          1024
40000        40448       a [100, 100] float32 tensor (79 * 512)
65536        65536
65537        131072
200000       262144
262144       262144      a [256, 256] float32 tensor
262145       524288
```

Buckets turn "any size" into a small set of sizes, so a freed block is likely to fit the next request of a similar size. The cost is internal waste: a 65537-byte request takes a 131072-byte block. PyTorch's CUDA caching allocator also rounds small requests to multiples of 512 bytes.

The deleter remembers the capacity and the device, so the block goes back to the right bucket of the right cache. `nbytes()` still reports the requested size, not the capacity: the rounding is invisible to `TensorImpl`, and the bounds check in `validate` uses the requested size, so a view cannot reach into the rounding slack.

```text
empty(shape, dtype, device)
  |
  v
Storage(nbytes, device)
  |
  v
detail::block_capacity(nbytes)          round up to a bucket
  |
  v
detail::allocate_block(capacity, device)
  |
  +-- free block of this capacity cached? -- yes --> reuse it (hit)
  |
  no (miss)
  |
  +-- CPU  --> operator new(capacity, std::align_val_t{64})
  +-- CUDA --> device::allocate(capacity)
```

When a `Storage` dies, `BlockDeleter` calls `detail::release_block`. From `src/allocator.cpp`, the release step of the cache:

```cpp
void release(std::byte* block, std::int64_t capacity) noexcept {
    {
        const std::lock_guard lock(mutex_);
        stats_.in_use_bytes -= capacity;
        if (enabled_ && stats_.cached_bytes + capacity <= kMaxCachedBytes &&
            try_cache(block, capacity)) {
            stats_.cached_bytes += capacity;
            return;
        }
        stats_.reserved_bytes -= capacity;
    }
    system_free_(block);
}
```

The block goes back in the free list of its capacity if the cache is enabled and the cached total stays at or below 1 GiB (`kMaxCachedBytes`). Otherwise it is freed. It is also freed when the free list itself cannot grow: `release` runs inside a `noexcept` deleter, so a `std::bad_alloc` from `push_back` would terminate the process. `try_cache` catches it and reports failure instead.

Host blocks are allocated and freed with the aligned forms of `operator new` and `operator delete`:

```cpp
std::byte* host_allocate(std::int64_t capacity) {
    return static_cast<std::byte*>(
        ::operator new(static_cast<std::size_t>(capacity), std::align_val_t{Storage::kAlignment}));
}

void host_free(std::byte* block) noexcept {
    ::operator delete(block, std::align_val_t{Storage::kAlignment});
}
```

The two forms must always be paired: freeing aligned memory with the plain `operator delete` is undefined behavior. The `std::unique_ptr` with a custom deleter makes the right pairing the only possible one. A default `std::unique_ptr<std::byte[]>` would call `delete[]`, which is the wrong function for this memory.

Why a cache at all: training loops allocate tensors of the same sizes on every step. Reusing blocks avoids a system allocation per intermediate result, and on a GPU it avoids repeated `cudaMalloc` and `cudaFree` calls, which are expensive and can synchronize the device. Phase 3 allocated directly with aligned `operator new`. Phase 9 added the cache. Chapter 08 covers the bucket policy, the statistics in `allocator_stats()`, `empty_cache()`, fork safety and thread safety.

The cache can be turned off three ways:

| When | How | Typical reason |
|------|-----|----------------|
| Build time | `KAEL_NO_CACHING_ALLOCATOR` compile definition | CMake sets it when `KAEL_ENABLE_SANITIZERS` or `KAEL_ENABLE_TSAN` is on, so the sanitizers see each allocation and each free |
| Startup | `KAEL_NO_CACHING_ALLOCATOR` environment variable | Debug a memory problem in an existing binary |
| Run time | `set_caching_allocator_enabled(false)` | Tests that need to observe real frees |

> [!NOTE]
> A cache hides use-after-free bugs from AddressSanitizer: a freed block that stays in the free list is still valid memory as far as the sanitizer knows. That is why sanitizer builds compile the cache out instead of relying on a runtime switch.

Without a CUDA backend, `device::allocate` throws `kael was built without CUDA; configure with KAEL_ENABLE_CUDA=ON`, so `Storage(16, Device::CUDA)` fails cleanly. `Storage.CudaThrows` covers it.

### External storage

`Storage::external` wraps memory that kael did not allocate.

From `src/storage.cpp`:

```cpp
Storage Storage::external(std::byte* data, std::int64_t nbytes, std::shared_ptr<void> owner) {
    Storage storage(0);
    storage.data_ = {data, BlockDeleter{0, Device::CPU}};
    storage.nbytes_ = nbytes;
    storage.owner_ = std::move(owner);
    return storage;
}

void Storage::BlockDeleter::operator()(std::byte* ptr) const noexcept {
    if (capacity > 0) {
        detail::release_block(ptr, capacity, device);
    }
}
```

A capacity of 0 tells `BlockDeleter` not to release anything. The `owner` keeps the real memory alive. `std::shared_ptr<void>` can hold any object with any deleter, so `Storage` needs no knowledge of what the owner is:

- The memory-mapped safetensors loader passes the mapping (chapter 12). When the owner dies, its deleter unmaps the file.
- The NumPy interop passes a reference to the source array (chapter 10). When the owner dies, it drops the array reference.

When the last tensor over the storage dies, `owner_` is released and its own deleter runs. This was added in phase 11. `Storage.ExternalMemoryIsAdoptedAndReleasedWithItsOwner` and `Storage.ExternalMemoryDoesNotTouchTheAllocator` cover both halves of the contract. PyTorch's `torch.from_numpy` and `torch.frombuffer` work the same way: the tensor keeps a reference that keeps the foreign buffer alive.

### TensorImpl

`TensorImpl` is a plain struct that pairs a storage with its interpretation.

From `include/kael/tensor_impl.h`:

```cpp
// Strides and offset count elements, not bytes.
struct TensorImpl {
    TensorImpl();
    TensorImpl(TensorImpl&&) noexcept;
    TensorImpl& operator=(TensorImpl&&) noexcept;
    ~TensorImpl();

    std::shared_ptr<Storage> storage;
    Shape shape;
    Strides strides;
    std::int64_t offset = 0;
    DType dtype = DType::F32;
    Device device = Device::CPU;
    std::unique_ptr<autograd::AutogradMeta> autograd;

    [[nodiscard]] std::int64_t numel() const { return kael::numel(shape); }
    [[nodiscard]] std::int64_t dim() const { return static_cast<std::int64_t>(shape.size()); }
    [[nodiscard]] bool is_contiguous() const;
    void validate() const;
};
```

Strides and offset count elements, not bytes, as described in [Addresses count elements, not bytes](#addresses-count-elements-not-bytes).

The members are public. `TensorImpl` is the internal representation, and views, operations and autograd all build impls directly. Accessors would add ceremony without protecting anything, because an impl under construction is temporarily invalid anyway (the shape is set before the strides). Invariants are enforced at one point, when an impl is wrapped in a `Tensor`, not by accessors.

`device` is stored on both the impl and the storage. The duplication lets `Tensor::device()` answer without following a second pointer, and `validate` checks that the two agree.

`autograd` holds the per-tensor autograd state (`requires_grad`, `grad`, `grad_fn`) from phase 7 on. It is a `std::unique_ptr` to a forward-declared type, so `tensor_impl.h` does not include the autograd headers, and tensors that never take part in autograd pay for one null pointer. A `unique_ptr` to an incomplete type needs the destructor and move operations to be defined where the type is complete. That is why the special members are declared in the header and defaulted in `src/tensor_impl.cpp`, which includes `kael/autograd/meta.h`:

```cpp
TensorImpl::TensorImpl() = default;
TensorImpl::TensorImpl(TensorImpl&&) noexcept = default;
TensorImpl& TensorImpl::operator=(TensorImpl&&) noexcept = default;
TensorImpl::~TensorImpl() = default;
```

If the destructor were defaulted in the header, every file that destroys a `TensorImpl` would need the full definition of `AutogradMeta`, and the compiler would reject the `unique_ptr` deleter on an incomplete type.

### Validation

A wrong stride or offset would read memory that belongs to another allocation. `validate()` checks every impl before it becomes a `Tensor`.

From `src/tensor_impl.cpp`:

```cpp
void TensorImpl::validate() const {
    KAEL_CHECK(storage != nullptr, "tensor has no storage");
    validate_shape(shape);
    KAEL_CHECK_VALUE(shape.size() == strides.size(),
                     "shape " << to_string(shape) << " and strides " << to_string(strides)
                              << " have different number of dimensions");
    for (std::size_t i = 0; i < strides.size(); ++i) {
        KAEL_CHECK_VALUE(strides[i] >= 0, "negative stride " << strides[i] << " at dimension " << i
                                                             << " in strides "
                                                             << to_string(strides));
    }
    KAEL_CHECK_VALUE(offset >= 0, "negative storage offset " << offset);
    KAEL_CHECK(device == storage->device(), "tensor device " << device
                                                             << " does not match storage device "
                                                             << storage->device());
    if (numel() > 0) {
        check_fits_in_storage(*this);
    }
}
```

The rules, in order:

1. The storage pointer is not null.
2. The shape has no negative dimension (`validate_shape` from chapter 01).
3. Shape and strides have the same length.
4. No stride is negative.
5. The offset is not negative.
6. The impl's device equals the storage's device.
7. If the tensor has elements, its last reachable element fits inside the storage.

The order matters. Rules 1 to 6 establish the preconditions of rule 7: the extent formula reads `storage->nbytes()`, indexes `strides[i]` for every `i` in the shape, and assumes non-negative terms.

The last rule is the formula from [The reachable extent of a layout](#the-reachable-extent-of-a-layout):

```cpp
void check_fits_in_storage(const TensorImpl& impl) {
    std::int64_t last_element = impl.offset;
    for (std::size_t i = 0; i < impl.shape.size(); ++i) {
        const std::int64_t extent =
            detail::checked_mul(impl.shape[i] - 1, impl.strides[i], "tensor extent");
        last_element = detail::checked_add(last_element, extent, "tensor extent");
    }
    const std::int64_t bytes_needed =
        detail::checked_mul(last_element + 1, element_size(impl.dtype), "tensor extent in bytes");

    KAEL_CHECK(bytes_needed <= impl.storage->nbytes(),
               "tensor with shape " << to_string(impl.shape) << ", strides "
                                    << to_string(impl.strides) << ", offset " << impl.offset
                                    << " and dtype " << impl.dtype << " needs " << bytes_needed
                                    << " bytes but storage has " << impl.storage->nbytes());
}
```

Every step is overflow-checked. Without that, absurd strides could wrap around to a small product that passes the comparison and then read far outside the buffer. For example, shape `[3]` with stride `2^62` gives an extent of `2 * 2^62 = 2^63`, which wraps to `INT64_MIN` in plain arithmetic. `TensorImplValidate.HugeStridesOverflowThrows` covers this case.

The check only runs when `numel() > 0`. An empty tensor reads nothing, so it needs no bytes, and `shape[i] - 1` would be `-1` for a zero dimension, which `checked_mul` rejects because it only accepts non-negative operands. The `TensorImplValidate.EmptyTensorNeedsNoBytes` test covers this case, and `TensorImplValidate.ExactlyAtTheLimit` and `OneByteShortThrows` pin the boundary on both sides, where `<` and `<=` get confused.

This check is the safety net for chapter 03. A view that computes its metadata wrong throws here instead of reading out of bounds. It costs one loop over the dimensions per new tensor, which is negligible next to the allocation of the impl itself.

kael rejects negative strides. NumPy allows them, for example to reverse an axis without copying: `a[::-1]` has stride `-itemsize`. Supporting them would change the bound above (the first reachable element would no longer be at `offset`, so the check would need a minimum as well as a maximum), and every loop and kernel would have to handle backward steps. PyTorch does not support negative strides either, and `torch.flip` copies.

### Contiguity in code

From `src/tensor_impl.cpp`:

```cpp
bool TensorImpl::is_contiguous() const {
    if (numel() == 0) {
        return true;
    }
    std::int64_t expected_stride = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        // The stride of a size-1 dimension is never used to step, so any value is fine.
        if (shape[i] == 1) {
            continue;
        }
        if (strides[i] != expected_stride) {
            return false;
        }
        expected_stride *= shape[i];
    }
    return true;
}
```

The loop `for (std::size_t i = shape.size(); i-- > 0;)` counts down with an unsigned index without wrapping: the test `i-- > 0` runs before the body, so the body sees `size - 1` down to `0`.

Two relaxations compared with a plain `strides == contiguous_strides(shape)`:

- Size-1 dimensions are skipped. `[2, 1, 3]` with strides `[3, 99, 1]` reads the same memory as with `[3, 3, 1]`, because the only valid index in the middle dimension is 0 and `0 * 99 = 0`. Views such as `unsqueeze` and `select` produce such strides.
- Empty tensors are always contiguous: there is nothing to lay out.

The offset is not part of the test. A contiguous block in the middle of a larger storage is still contiguous. PyTorch's `is_contiguous()` applies the same two relaxations. NumPy's `C_CONTIGUOUS` flag also ignores strides of size-1 dimensions in current versions.

### Tensor

`Tensor` holds one `std::shared_ptr<TensorImpl>` and nothing else.

From `include/kael/tensor.h`:

```cpp
class Tensor {
public:
    static Tensor from_impl(std::shared_ptr<TensorImpl> impl);

    [[nodiscard]] const Shape& shape() const noexcept { return impl_->shape; }
    [[nodiscard]] const Strides& strides() const noexcept { return impl_->strides; }
    [[nodiscard]] std::int64_t offset() const noexcept { return impl_->offset; }
    [[nodiscard]] DType dtype() const noexcept { return impl_->dtype; }
    [[nodiscard]] Device device() const noexcept { return impl_->device; }
    // ... element access, conversions, views, operations, autograd
private:
    explicit Tensor(std::shared_ptr<TensorImpl> impl);
    std::shared_ptr<TensorImpl> impl_;
};
```

`sizeof(Tensor)` is the size of one `std::shared_ptr`: two pointers, 16 bytes on a 64-bit platform. Passing a tensor by value copies those two pointers and increments an atomic reference count.

#### Copying shares, `clone` copies

`Tensor` follows the rule of zero: no hand-written copy constructor, assignment or destructor. `std::shared_ptr` already does the right thing. Copying a tensor copies the pointer and increments a reference count, and the data is not touched.

```cpp
kael::Tensor a = kael::zeros({2, 3});
kael::Tensor b = a;          // same tensor
b.at<float>({0, 0}) = 7.0f;
a.at<float>({0, 0});         // 7.0

kael::Tensor c = a.clone();  // independent copy
c.at<float>({0, 0}) = 1.0f;
a.at<float>({0, 0});         // still 7.0
```

The resulting objects:

```text
a ----+
      +--> impl #1 --> storage #1   [7, 0, 0, 0, 0, 0]
b ----+

c -------> impl #2 --> storage #2   [1, 0, 0, 0, 0, 0]
```

This is the PyTorch and NumPy model: `b = a` in Python binds a second name to the same object. [ADR 0002](@/adr/0002-tensor-as-handle.md) explains why value semantics would not work for views and autograd. In short: a view must see writes to its base, and a gradient accumulated on a parameter must be visible through every handle that refers to it, including the handles held by an optimizer.

> [!WARNING]
> In C++, `Tensor b = a;` looks like a copy of a value, as it would be for `std::vector`. It is not. Writes through `b` change `a`. Call `clone()` when you need independent data.

#### Construction always validates

The constructor is private, and the only public way in is `from_impl`, which calls it. The constructor rejects a null impl and calls `impl_->validate()`.

From `src/tensor.cpp`:

```cpp
Tensor::Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {
    KAEL_CHECK(impl_ != nullptr, "cannot create a tensor from a null impl");
    impl_->validate();
}

Tensor Tensor::from_impl(std::shared_ptr<TensorImpl> impl) {
    return Tensor(std::move(impl));
}
```

There is no default constructor. A `Tensor` always refers to a valid impl, so no member function checks for an empty handle. Code that needs an optional tensor uses `std::optional<Tensor>`, as `grad()` does. PyTorch's `at::Tensor` does have an undefined state, and its C++ API is full of `defined()` checks as a result. kael pays a little convenience (no `Tensor t;` followed by assignment later) for never having that state.

#### Queries

| Member | Returns |
|--------|---------|
| `shape()`, `strides()` | references to the impl's vectors |
| `offset()`, `dtype()`, `device()` | the impl's fields |
| `size(dim)`, `stride(dim)` | one entry, with negative `dim` counting from the end |
| `numel()`, `dim()`, `is_contiguous()` | forwarded to the impl |
| `nbytes()` | `numel() * element_size(dtype())` |
| `is_same(other)`, `shares_storage(other)` | identity and sharing, as described above |
| `impl()` | the `std::shared_ptr<TensorImpl>`, for internal code |

`size(dim)` and `stride(dim)` accept negative dimensions through `dim_position` from chapter 01, so `t.size(-1)` is the last dimension. `nbytes()` is the number of bytes the tensor's elements occupy, not the size of the storage, which can be larger for a view. A 2-element slice of `arange(1000000)` has `nbytes()` of 8, while its storage holds 4000000 bytes.

#### Element access

There are four ways to reach elements, from raw to checked:

| Member | Returns | Checks |
|--------|---------|--------|
| `data_ptr<T>()` | `T*` to the first element | dtype matches `T`, tensor on CPU |
| `data_ptr_raw()` | `std::byte*` to the first element | tensor on CPU |
| `at<T>(index)` | `T&` to one element | dtype, CPU, index count, each index in range |
| `item<T>()` | the single value as `T` | dtype matches `T`, exactly one element |
| `item()` | the single value as `double` | exactly one element |

The pointer to the first element must include the offset. Forgetting it is a classic bug that stays invisible while every offset is 0, which is the case until chapter 03 creates views. `test_tensor.cpp` builds an impl by hand with a non-zero offset to catch it before views exist.

From `src/tensor.cpp`:

```cpp
std::byte* Tensor::first_element() const {
    std::byte* base = impl_->storage->data();
    // An empty view can have an offset past the end of its storage, or a null storage.
    return numel() == 0 ? base : base + offset() * element_size(dtype());
}
```

For an empty tensor the base is returned unchanged. The storage of `zeros({3, 0, 2})` has a null pointer, and an empty view (for example `slice(0, 5, 5)` of a 5-element tensor) can have an offset at or past the end of its storage. Adding a non-zero offset to a null pointer, or moving a pointer more than one past the end of its allocation, is undefined behavior, and UndefinedBehaviorSanitizer reports it.

`data_ptr<T>()` checks that `T` matches the dtype. Asking a `float32` tensor for `data_ptr<double>()` throws `TypeError` with `expected dtype float64 but tensor has float32`. This is the runtime replacement for the compile-time check that a templated `Tensor<T>` would give. There is no implicit conversion: reading a float32 tensor as `double` must go through `to(DType::F64)` first.

`at` computes the element position through `element_offset`:

```cpp
std::int64_t Tensor::element_offset(std::span<const std::int64_t> index) const {
    KAEL_CHECK_VALUE(static_cast<std::int64_t>(index.size()) == dim(),
                     "got " << index.size() << " indices for a tensor with " << dim()
                            << " dimensions");
    std::vector<std::int64_t> wrapped(index.begin(), index.end());
    for (std::size_t i = 0; i < wrapped.size(); ++i) {
        if (wrapped[i] < 0) {
            wrapped[i] += shape()[i];
        }
    }
    return offset_from_index(shape(), strides(), offset(), wrapped);
}
```

Negative indices are wrapped once by adding the dimension size, so `t.at<float>({-1, -1})` on a `[2, 3]` tensor reads `[1, 2]`. Then `offset_from_index` from chapter 01 rejects anything still out of range with `IndexError`, so `-4` on a dimension of size 3 becomes `-1` and fails. The base offset is passed in, so `at` respects the offset as well. `at` has overloads for `std::span<const std::int64_t>` and `std::initializer_list<std::int64_t>`, so both `t.at<float>({1, 2})` and `t.at<float>(some_shape)` work.

A trace of `t.at<float>({-1, 0})` on `t = a.transpose(0, 1)`, shape `[3, 2]`, strides `[1, 3]`, over the storage `[0 1 2 3 4 5]`:

```text
index count 2 == dim 2         ok
wrap: -1 + 3 = 2   ->  [2, 0]
offset_from_index: 0 + 2 * 1 + 0 * 3 = 2
value: storage[2] = 2
```

`item<T>()` and `item()` work on any shape with one element, including `[]`, `[1]` and `[1, 1]`. On a CUDA tensor they copy to the host first. `data_ptr` and `at` throw on a CUDA tensor instead, with a message that tells you to call `cpu()`, because their pointer cannot be dereferenced on the host. `device_data()` returns the address without that check, for code that passes it to device kernels.

#### Conversions and copies

From `src/tensor.cpp`:

```cpp
Tensor Tensor::to(DType target) const {
    if (target == dtype()) {
        return *this;
    }
    Tensor out = empty(shape(), target, device());
    detail::convert_strided(out, *this);
    autograd::record_to(out, *this);
    return out;
}
```

When the dtype already matches, `to` returns the same tensor, not a copy. PyTorch does the same, and the `TensorTo.SameDtypeReturnsSameImpl` test pins it. Code that needs an independent copy must call `clone()`. Otherwise the result is a new contiguous tensor. `detail::convert_strided` (chapter 03) walks the source through its strides and converts each element with `convert_value`, so non-contiguous sources work.

The other members in this group:

- `to(Device)`, `cpu()` and `cuda()` copy between host and device through the backend's copy functions, after making the source contiguous. Same-device calls return `*this`.
- `move_to_(Device)` moves a tensor in place: it replaces the storage, strides and offset of the existing impl, so every handle to this tensor sees the new device. `nn::Module` uses it to move parameters without breaking references held by optimizers.
- `clone()` allocates a new contiguous tensor. For a contiguous source it copies the bytes with `std::memcpy` (or `device::copy_within_device`). For a non-contiguous source it uses `convert_strided`. The result always has offset 0 and a storage sized for its elements, even when the source was a small view into a large storage.
- `contiguous()` returns `*this` when the tensor is already contiguous, and `clone()` otherwise.

From `src/tensor.cpp`:

```cpp
Tensor Tensor::clone() const {
    Tensor out = empty(shape(), dtype(), device());
    if (!is_contiguous()) {
        detail::convert_strided(out, *this);
    } else if (nbytes() > 0) {
        if (device() == Device::CPU) {
            std::memcpy(out.data_ptr_raw(), data_ptr_raw(), static_cast<std::size_t>(nbytes()));
        } else {
            device::copy_within_device(out.device_data(), device_data(), nbytes());
        }
    }
    autograd::record_clone(out, *this);
    return out;
}
```

The `nbytes() > 0` guard keeps `std::memcpy` away from null pointers, which it does not accept even for a size of 0.

`move_to_` deserves a closer look, because it is the one member that mutates an impl's layout after construction:

```cpp
Tensor& Tensor::move_to_(Device target) {
    if (target == device()) {
        return *this;
    }
    const Tensor moved = detach().to(target);
    impl_->storage = moved.impl()->storage;
    impl_->strides = moved.strides();
    impl_->offset = moved.offset();
    impl_->device = target;
    return *this;
}
```

The new storage, strides and offset come from a freshly built and already validated tensor, so the impl stays valid even though it is changed in place. Other tensors that were views of the old storage keep the old storage and stay on the old device.

`to(DType)`, `to(Device)` and `clone()` call an `autograd::record_*` function. Those do nothing until chapter 06, where they record the graph node that sends gradients back through the conversion. `move_to_` records nothing.

| Call | Same dtype and device | Otherwise |
|------|-----------------------|-----------|
| `to(DType)` | returns `*this` | new contiguous tensor, converted |
| `to(Device)` | returns `*this` | new contiguous tensor on the target device |
| `contiguous()` | returns `*this` if contiguous | `clone()` |
| `clone()` | always a new tensor | always a new tensor |

### Factories

All factories end in `empty`, which computes the byte count with overflow checks, allocates the storage and builds a contiguous impl.

From `src/factory.cpp`:

```cpp
Tensor empty(const Shape& shape, DType dtype, Device device) {
    const std::int64_t bytes =
        detail::checked_mul(numel(shape), element_size(dtype), "tensor size in bytes");

    auto impl = std::make_shared<TensorImpl>();
    impl->storage = std::make_shared<Storage>(bytes, device);
    impl->device = device;
    impl->shape = shape;
    impl->strides = contiguous_strides(shape);
    impl->dtype = dtype;
    return Tensor::from_impl(std::move(impl));
}
```

`numel` can be within range while `numel * element_size` overflows. Shape `[2^61, 2]` in float64 has `2^62` elements, which fits in `int64`, but needs `2^65` bytes. `checked_mul` catches that:

```text
tensor size in bytes overflows int64 (4611686018427387904 * 8)
```

`Factory.BytesOverflowThrows` tests it. A shape whose element count itself overflows fails one step earlier, inside `numel`.

| Factory | Behavior |
|---------|----------|
| `empty(shape, dtype, device)` | Uninitialized. Defaults: `F32`, `CPU` |
| `zeros`, `ones` | `full` with 0 or 1 |
| `full(shape, value, dtype, device)` | Every element set to `value.as<T>()`. On a device, `device::fill` |
| `empty_like`, `zeros_like`, `ones_like` | Same shape, dtype and device as the argument |
| `arange(end, dtype)`, `arange(start, end, step, dtype)` | `ceil((end - start) / step)` values. Rejects `Bool`, non-finite bounds, zero step, and a step that points away from `end` |
| `linspace(start, end, steps, dtype)` | Floating only. With `steps > 1`, the last value is set to exactly `end` |
| `eye(n, dtype)` | `n x n` identity |
| `scalar(value, dtype)` | A 0-d tensor, shape `[]` |
| `from_vector(data, shape)`, `from_vector(data)` | Copies a `std::vector` or an initializer list. The dtype comes from the element type |

Worked examples, all from real runs:

```text
arange(0, 5, 2)                    ceil(5 / 2) = 3 values
  tensor([0.0000, 2.0000, 4.0000], shape=[3], dtype=float32)

arange(0.5, 3.5, 1.5, DType::I64)  ceil(3.0 / 1.5) = 2 values
  start truncates to 0, step truncates to 1
  tensor([0, 1], shape=[2], dtype=int64)

linspace(0, 1, 5)                  delta = (1 - 0) / 4 = 0.25
  tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000], shape=[5], dtype=float32)

linspace(3, 7, 1)                  one step: only start
  tensor([3.0000], shape=[1], dtype=float32)

eye(2)
  tensor([1.0000, 0.0000, 0.0000, 1.0000], shape=[2, 2], dtype=float32)
```

Details behind those results:

- `arange` with an integer dtype truncates `start` and `step` before it computes the values, as PyTorch does. The count still comes from the untruncated values, which is why `arange(0.5, 3.5, 1.5, DType::I64)` has 2 elements and stops at 1 instead of reaching 3.
- `arange` computes each value as `start + i * step` instead of accumulating `value += step`. Accumulation would add one rounding error per step, so the 1000th value of `arange(0, 100, 0.1)` would drift.
- `linspace` computes `start + i * delta` and then overwrites the last value with `end`, because the product can land one rounding step away from it. `Factory.LinspaceLastIsExactlyEnd` pins this.
- `from_vector` deduces the dtype from the element type through `dtype_of_v`, so the literals decide: `{1.f, 2.f}` is `F32`, `{1.0, 2.0}` is `F64`, `{std::int64_t{1}}` is `I64`. A list of plain `int` literals does not compile, because `dtype_of<int>` has no definition. `from_vector` also checks that the number of values equals `numel(shape)`, and passes each value through `convert_value<T>`, which normalizes Bool bytes (`Factory.FromVectorNormalizesBoolBytes`).

The equivalents in PyTorch are `torch.empty`, `torch.zeros`, `torch.full`, `torch.arange`, `torch.linspace`, `torch.eye` and `torch.tensor`. One difference: `torch.tensor([1, 2])` infers int64 from Python integers, while kael rejects C++ `int` and asks you to say which width you mean.

### Random numbers

All random factories draw from one global `std::mt19937_64`, the 64-bit Mersenne Twister engine.

From `src/random.cpp`:

```cpp
constexpr std::uint64_t kDefaultSeed = 0;

std::mt19937_64& global_generator() {
    static std::mt19937_64 generator(kDefaultSeed);
    return generator;
}
```

The generator is a function-local static, created on first use, so it does not depend on the initialization order of globals in other translation units (the static initialization order problem). Its default seed is fixed, so a program that never calls `manual_seed` still produces the same numbers on every run on the same platform. PyTorch also starts from a fixed default seed.

| Function | Distribution | Dtypes |
|----------|--------------|--------|
| `manual_seed(seed)` | Reseeds the generator | |
| `rand(shape, dtype)` | Uniform in `[0, 1)` | `F32`, `F64` |
| `randn(shape, dtype)` | Standard normal | `F32`, `F64` |
| `randint(low, high, shape)` | Uniform integers in `[low, high)` | `I64` |
| `randperm(n)` | A permutation of `0 .. n-1` | `I64` |
| `bernoulli(shape, p, dtype)` | 1 with probability `p` | any |
| `multinomial(weights, samples, replacement)` | Indices drawn with the given weights, per row of a `[K]` or `[B, K]` tensor | `I64` |

The floating samplers share one helper:

```cpp
template <typename Distribution>
Tensor sample_floating(const Shape& shape, DType dtype, const char* op, Distribution distribution) {
    Tensor t = empty(shape, dtype);
    dispatch_floating(dtype, op, [&]<typename T>() {
        const auto lock = lock_generator();
        std::generate_n(t.data_ptr<T>(), t.numel(),
                        [&] { return static_cast<T>(distribution(global_generator())); });
    });
    return t;
}
```

Floating samples are drawn as `double` and converted, so the same seed gives the same sequence for `F32` and `F64`: every float32 sample equals the float64 sample of the same position cast to `float` (`Random.Float32AndFloat64SeeTheSameSequence`). The conversion has one trap: a `double` slightly below 1 can round up to `1.0f`, which is outside `[0, 1)`. `rand` replaces any `1.0f` with `std::nextafter(1.0f, 0.0f)`, the largest float below 1.

`randint` rejects `low >= high`. `randperm` builds `arange(0, n)` as `I64` and shuffles it with `std::shuffle`. `bernoulli(shape, p, dtype)` is `(rand(shape, DType::F64) < p).to(dtype)`: a comparison produces a Bool tensor, and `to` converts it, which is why it accepts any dtype. `p` must be in `[0, 1]`.

`multinomial` draws, for each row, an index with probability proportional to its weight. It computes the cumulative sum of the weights, draws a uniform point in `[0, total)`, and finds the first cumulative value above it with `std::upper_bound`. Without replacement, the picked weight is set to 0 before the next draw. A worked draw with weights `[1, 3]`:

```text
cumulative = [1, 4], total = 4
point = uniform * 4, say 2.6
upper_bound(cumulative, 2.6) -> index 1   (1 <= 2.6 < 4)
index 1 is picked with probability 3/4
```

The weights must be finite and non-negative, at least one must be positive, and without replacement `samples` cannot exceed the number of positive weights. `multinomial` arrived in phase 12 for text generation (chapter 11).

The generator engine is fully specified by the C++ standard, but `std::normal_distribution` and `std::uniform_real_distribution` are not. libstdc++ and libc++ can produce different numbers from the same engine state. A seed reproduces results on one platform, not across platforms. The tests therefore check properties (same seed gives the same values, the mean and standard deviation of `randn` are close to 0 and 1, `randint` stays in bounds) and never compare against hard-coded numbers.

In phase 13, a mutex was added around the generator, with `pthread_atfork` handlers that hold it across `fork()`, so a child process never inherits a locked mutex:

```cpp
// A thread sampling during fork would leave the generator locked forever in the child.
void install_fork_handlers() {
#if !defined(_WIN32)
    static const int installed =
        pthread_atfork([] { generator_mutex().lock(); }, [] { generator_mutex().unlock(); },
                       [] { generator_mutex().unlock(); });
    (void)installed;
#endif
}
```

The three handlers run before the fork (lock), in the parent after the fork (unlock) and in the child after the fork (unlock). The static `installed` makes registration happen once. `RandomThreads.ConcurrentSamplingDoesNotRace` samples from several threads under ThreadSanitizer.

### Printing

`operator<<` prints the values as a flat list, then the shape, the dtype and, for non-CPU tensors, the device:

```text
tensor([0.0000, 1.0000, 2.0000, 3.0000, 4.0000, 5.0000], shape=[2, 3], dtype=float32)
tensor([true, true, true], shape=[3], dtype=bool)
tensor([3.5000], shape=[], dtype=float32)
tensor([], shape=[2, 0], dtype=float32)
tensor([0.0000, 1.0000, 2.0000, ..., 1497.0000, 1498.0000, 1499.0000], shape=[1500], dtype=float32)
```

From `src/print.cpp`:

```cpp
std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    std::ostringstream text;
    text << std::fixed << std::setprecision(kFloatPrecision) << "tensor([";
    const Tensor host = t.cpu();
    dispatch(t.dtype(), [&]<typename T>() { print_elements<T>(text, host); });
    text << "], shape=" << to_string(t.shape()) << ", dtype=" << t.dtype();
    if (t.device() != Device::CPU) {
        text << ", device=" << t.device();
    }
    text << ")";
    return os << text.str();
}
```

Design points:

- The text is built in a separate `std::ostringstream`, so `std::fixed` and the precision do not leak into the caller's stream (`TensorPrint.LeavesStreamFlagsUntouched`). Stream flags are sticky, and a print that changed them would alter every later `double` the caller prints.
- `t.cpu()` returns `t` itself for a CPU tensor and a host copy otherwise, so printing works on any device.
- Elements are read with `at<T>(index_from_linear(i, shape))`, the slow general path from chapter 01, which handles any strides. A transposed tensor prints in its logical order: `a.T()` of `[[0, 1, 2], [3, 4, 5]]` prints `0, 3, 1, 4, 2, 5`.
- Tensors with more than 1000 elements (`kSummarizeThreshold`) print the first 3 and the last 3 (`kSummarizeEdge`) with `...` between them.
- `F32` and `F64` print with 4 decimals, `I64` as integers, and `Bool` as `true` and `false`.

The flat list is a deliberate simplification. PyTorch and NumPy print nested brackets that show the shape. kael prints the shape as a separate field, which keeps the printer short and the output simple to compare in tests.

### Umbrella header and version

`include/kael/kael.h` includes the core headers, the operations, autograd, the parallel runtime, the allocator and the debugging helpers. The `nn`, `optim`, `data`, `io` and `text` headers are not included and must be included individually. Keeping them out keeps the compile time of a program that only needs tensors low.

`kael::version()` returns the `KAEL_VERSION` compile definition, which `CMakeLists.txt` sets from the project version (`1.0.0`):

```cmake
project(kael VERSION 1.0.0 LANGUAGES CXX)
target_compile_definitions(kael PRIVATE KAEL_VERSION="${PROJECT_VERSION}")
```

The test `Version.MatchesTheProjectVersion` compares it with `KAEL_EXPECTED_VERSION`, the same CMake value passed to the tests, so a version bump that forgets one side fails the tests.

## Edge cases and pitfalls

- **Offset in pointers.** `data_ptr`, `data_ptr_raw` and `at` all add the offset. A pointer computed from `storage->data()` alone is wrong for every view. `Tensor.DataPtrRespectsOffset` and `TensorAt.RespectsOffsetAndStrides` cover it.
- **Empty tensors.** Storage of 0 bytes has a null pointer. `validate` skips the fit check, `first_element` avoids pointer arithmetic on null, `clone` skips `memcpy`, and `print` writes `tensor([], ...)`. `Factory.ZeroSizedDimensionAllocatesNothing` checks that nothing is allocated.
- **Byte count overflow.** `empty` checks `numel * element_size`, and `validate` checks every term of the extent, so a huge shape or stride throws `ValueError` instead of allocating a wrapped-around size.
- **Bool bytes.** Any path that writes a Bool element goes through `convert_value`, so a Bool tensor never holds a byte other than 0 or 1.
- **Float to integer.** NaN and out-of-range floats would be undefined behavior in a `static_cast`. `convert_value` saturates them to `INT64_MIN` or `INT64_MAX` and maps NaN to 0.
- **`to` may alias.** `to(dtype())` and `to(device())` return the same tensor. Writing to the result writes to the source. The same holds for `contiguous()` on a contiguous tensor.
- **Copying a handle is not copying data.** `Tensor b = a;` shares. Use `clone()`.
- **Device memory on the host.** `data_ptr` and `at` throw for CUDA tensors. `device_data()` is the only way to get a device address, and its comment states that only device code may dereference it.
- **Platform-dependent random numbers.** The same seed gives different `randn` values on libstdc++ and libc++. Tests must not hard-code sampled values.
- **Static tensors.** The allocator caches are allocated with `new` and never destroyed, so a tensor with static storage duration can release its block after `main` returns without touching a destroyed cache.
- **Views keep storage alive.** A small view holds a reference to the whole storage. `clone()` it to release the rest.

## How it is tested

`-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion` are on for the library and the tests, and CI configures with `-DKAEL_WARNINGS_AS_ERRORS=ON`, which adds `-Werror`. One CI job runs everything under AddressSanitizer and UndefinedBehaviorSanitizer with the cache compiled out, and another runs under ThreadSanitizer.

| File | Suites | What they prove |
|------|--------|-----------------|
| `tests/test_dtype.cpp` | `DType`, `Version`, `Device` | Element sizes, names, stream output, traits in both directions, `dispatch` selects the matching type and supports `void` lambdas, the restricted variants reject the wrong dtypes, the version string, `parse_device` accepts `cpu`, `cuda` and `cuda:0` and rejects other names |
| `tests/test_storage.cpp` | `Storage` | Allocation, 64-byte alignment for several sizes, every byte writable, zero bytes is null, negative size throws, CUDA throws without a backend, moves, external memory released with its owner and never passed to the allocator |
| `tests/test_tensor_impl.cpp` | `TensorImpl`, `TensorImplValidate` | Contiguity for normal and transposed strides, 0-d impls, the exact byte limit and one byte short, offset past the end, negative strides and offsets, size mismatches, null storage, empty tensors, dtype size, stride overflow |
| `tests/test_tensor.cpp` | `Tensor`, `TensorItem`, `TensorAt`, `TensorPrint`, `TensorTo` | Copy shares and clone does not, clone of an offset tensor starts at 0, `data_ptr` type checks and offset, negative `size` and `stride` dims, `item` rules, negative indices in `at`, printing of every dtype, summaries and stream flags, conversion rounding, truncation and Bool rules |
| `tests/test_factory.cpp` | `Factory` | Defaults, `full` with Bool, `*_like`, `arange` examples and errors, integer truncation, `linspace` endpoints, `eye`, `scalar`, `from_vector` dtype deduction and Bool normalization, zero-sized shapes allocate nothing, byte overflow |
| `tests/test_random.cpp` | `Random`, `RandomThreads` | Seeds reproduce, different seeds differ, F32 and F64 see one sequence, ranges, moments, permutations, Bernoulli extremes and frequency, dtype rejection, concurrent sampling |
| `tests/test_allocator.cpp` | `CachingAllocator`, `Allocator` | Bucket sizes, reuse, statistics, `empty_cache`, the 1 GiB limit, disabling the cache (details in chapter 08) |
| `tests/test_errors.cpp` | `TypedErrors` | Each error kind derives from `kael::Error`, and representative failures throw the expected subclass |
| `tests/test_integer_overflow.cpp` | `FloatToInteger` (and `IntegerOverflow`, chapter 05) | Saturation, NaN to zero and the exact lower bound of float to integer conversion |

The `make_tensor` helper in `test_tensor.cpp` builds an impl by hand with a chosen storage size and offset. That is how the offset tests run before any view function exists. It is a useful pattern in general: when a bug can only appear with metadata that a later layer produces, build that metadata by hand in the test of the earlier layer.

Two kinds of boundary test recur in this chapter:

- **Both sides of a limit.** `ExactlyAtTheLimit` and `OneByteShortThrows` together prove the comparison is `<=`. Either test alone passes with the wrong operator.
- **Properties instead of values.** The random tests compare runs against each other and check statistics, because the exact values depend on the standard library.

## Build it yourself

Start from chapter 01: `shape.h`, `shape.cpp`, `checked_math.h` and `error.h` with their tests passing. Build with `cmake -S . -B build && cmake --build build`, and run a subset of tests with `build/tests/kael_tests --gtest_filter='Storage.*'`.

1. Write `include/kael/dtype.h`: `DType`, `Device`, `element_size`, `to_string`, `is_floating_point`, `operator<<`, the `dtype_of` and `cpp_type` traits, `convert_value` and `dispatch`. Checkpoint: `DType.DispatchSelectsMatchingType` and `DType.DispatchMatchesElementSize` pass, and `dtype_of_v<int>` fails to compile.
2. Write `Storage` with aligned `operator new` and a deleter that calls the aligned `operator delete`. Leave the caching allocator for chapter 08. Checkpoint: `Storage.AlignedTo64Bytes`, `ZeroBytesIsNull`, `NegativeBytesThrows` and `Movable` pass.
3. Write `TensorImpl` with `is_contiguous` and `validate`, including the checked extent computation. Checkpoint: all `TensorImplValidate` tests pass, in particular `ExactlyAtTheLimit`, `OneByteShortThrows` and `EmptyTensorNeedsNoBytes`.
4. Write `Tensor` with the private validating constructor, `from_impl`, the queries, `is_same` and `shares_storage`. Checkpoint: `Tensor.FromImplValidates` and `Tensor.Queries` pass.
5. Add `empty`, `full`, `zeros`, `ones` and the `*_like` variants. Checkpoint: `Tensor.CopySharesData` passes, and `Factory.BytesOverflowThrows` throws.
6. Add `first_element`, `data_ptr`, `data_ptr_raw`, `at` with negative index wrapping, and `item`. Checkpoint: `Tensor.DataPtrRespectsOffset`, `TensorAt.RespectsOffsetAndStrides` and the `TensorItem` tests pass.
7. Add `arange`, `linspace`, `eye`, `scalar` and `from_vector` with its Bool overloads. Checkpoint: the `Factory` suite passes.
8. Add `to(DType)` and `clone()`. Until chapter 03 provides `convert_strided`, a loop over `index_from_linear` with `at` works for any strides. Checkpoint: the `TensorTo` tests and `Tensor.CloneOfOffsetTensorStartsAtZero` pass.
9. Add the global generator, `manual_seed`, `rand`, `randn` and `randint`. Checkpoint: `Random.SameSeedSameValues`, `Float32AndFloat64SeeTheSameSequence` and `RandInUnitInterval` pass.
10. Add `operator<<` in `print.cpp`. Checkpoint: the `TensorPrint` tests pass, including `LeavesStreamFlagsUntouched`.
11. Add `kael.h` and `version()`. Checkpoint: the README example compiles against the umbrella header and prints `tensor([7.0000, 1.0000, 2.0000, 3.0000, 4.0000, 5.0000], shape=[6], dtype=float64)`.
12. Configure with `-DKAEL_ENABLE_SANITIZERS=ON` and run all tests. Checkpoint: no sanitizer reports.

The caching allocator, CUDA storage, external storage, `Scalar`, the typed error subclasses, `move_to_`, autograd metadata and the random mutex come later. Each chapter that needs one adds it.

The README example from step 11, for reference:

```cpp
#include <iostream>

#include "kael/kael.h"

int main() {
    kael::Tensor a = kael::arange(0, 6).to(kael::DType::F64);
    kael::Tensor b = a;          // shares data with a
    kael::Tensor c = a.clone();  // independent copy
    b.at<double>({0}) = 7.0;
    std::cout << a << "\n";      // tensor([7.0000, 1.0000, 2.0000, 3.0000, 4.0000, 5.0000], shape=[6], dtype=float64)
}
```

## Check your understanding

<details>
<summary>Why is Storage not copyable, while Tensor is?</summary>

`Storage` owns its block through a `std::unique_ptr` with a deleter that returns the block to the allocator. Two copies would both run that deleter and free the block twice. Sharing is expressed one level up: `TensorImpl` holds a `std::shared_ptr<Storage>`, and `Tensor` holds a `std::shared_ptr<TensorImpl>`. Copying a `Tensor` copies a shared pointer and increments a reference count, which is safe and cheap.

</details>

<details>
<summary>A tensor has shape [4], stride [3], offset 2 and dtype float32. What is the smallest storage that passes validation?</summary>

The last reachable element is `2 + (4 - 1) * 3 = 11`. The tensor needs `(11 + 1) * 4 = 48` bytes. A storage of 48 bytes passes, and 47 bytes throws. This is the `ExactlyAtTheLimit` and `OneByteShortThrows` pair applied to a strided layout.

</details>

<details>
<summary>Why does saturating_cast compare against -lowest instead of INT64_MAX?</summary>

`INT64_MAX` is `2^63 - 1`, which is not representable in `float` or `double`. Converted to the float type, it rounds up to `2^63`, so a comparison `value > INT64_MAX` would let `2^63` itself through to the `static_cast`, which is undefined behavior. `-lowest` is exactly `2^63` in both float types, so `value >= -lowest` catches every value that does not fit.

</details>

<details>
<summary>What does to(DType::F32) return for a float32 tensor, and why does it matter?</summary>

It returns `*this`, the same tensor with the same impl, not a copy. A write through the result changes the source. Code that needs independent data must call `clone()`. PyTorch behaves the same way, and `TensorTo.SameDtypeReturnsSameImpl` pins it.

</details>

<details>
<summary>Why is the fit check in validate skipped for empty tensors?</summary>

An empty tensor reads no elements, so it needs no bytes, and its storage may have a null pointer. The extent formula would also compute `shape[i] - 1 = -1` for a zero dimension, and `checked_mul` rejects negative operands. Skipping the check is both correct and necessary. `TensorImplValidate.EmptyTensorNeedsNoBytes` covers it.

</details>

<details>
<summary>What is the result dtype of an int64 tensor times 0.5, and of a float64 tensor times 0.5?</summary>

`0.5` is a floating scalar, category 2, reported as `F32`. For the int64 tensor (category 1) the scalar's category is higher, so the result is `F32`. For the float64 tensor (category 2) the categories are equal, so the tensor wins and the result stays `F64`. A scalar never raises precision within its own category.

</details>

## Further reading

- [Edward Z. Yang, "PyTorch internals"](http://blog.ezyang.com/2019/05/pytorch-internals/): the tensor, storage, strides and dispatch model this chapter follows, explained by one of PyTorch's maintainers.
- [NumPy, "The N-dimensional array (ndarray)"](https://numpy.org/doc/stable/reference/arrays.ndarray.html): the section on internal memory layout shows the same ideas with byte strides and negative strides.
- [PyTorch, "Type promotion"](https://pytorch.org/docs/stable/tensor_attributes.html): the `torch.dtype` section documents the promotion categories that kael's `Scalar` rules mirror.
- [cppreference, `operator new`](https://en.cppreference.com/w/cpp/memory/new/operator_new): the aligned overloads with `std::align_val_t` and why they must pair with the aligned `operator delete`.
- [cppreference, `std::mersenne_twister_engine`](https://en.cppreference.com/w/cpp/numeric/random/mersenne_twister_engine): the engine behind `global_generator`, and the note that distributions are implementation defined.
- [kael `src/tensor_impl.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/tensor_impl.cpp): the whole validation layer in under 80 lines, worth reading in full.

Next: [03. Views](@/book/03-views.md)
