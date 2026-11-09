+++
title = "Runtime data type"
description = "Why the element type of a kael tensor is a runtime value checked by a switch, not a template parameter of the Tensor class."
weight = 1
slug = "runtime-dtype"

[extra]
kicker = "ADR 0001"
source = "docs/adr/0001-runtime-dtype.md"
+++

Every tensor library has to answer one question early: is the element type part of the C++ type of a tensor, or a value the tensor carries? The answer shapes the public API, the kernels, the autograd graph and the Python bindings. kael carries the element type as a runtime value. This record explains the forces behind that choice, the alternatives it rejected, and what the choice costs.

A **dtype** (data type) is the type of the elements stored in a tensor: `float32`, `float64`, `int64` or `bool` in kael. **Dispatch** is the step that turns a runtime dtype value into a call to code compiled for the matching C++ type.

## Status

Accepted in phase 3 (tag `chapter-03-core`, which matches [Memory model](@/book/02-memory-model.md) in the book). Still in force at `v1.0.0` and at the current head of the repository.

## Context

A tensor in kael holds elements of one of four types: `float32`, `float64`, `int64` or `bool`. There are two ways to represent that in C++.

- **Compile-time dtype.** The element type is a template parameter: `Tensor<float>`, `Tensor<double>`. The compiler knows the type at every call site.
- **Runtime dtype.** There is one `Tensor` class. Each tensor stores a small enum value that says what its bytes mean. The type is known only when the program runs.

A templated `Tensor<T>` is attractive. The compiler rejects `Tensor<float> + Tensor<std::int64_t>` unless someone wrote that overload, reading an element needs no check, and no `switch` runs before a kernel. Libraries such as Eigen use this model with success.

It conflicts with four other parts of kael's design.

- **Heterogeneous containers.** A model's `state_dict` maps names to tensors. A safetensors file ([Loading GPT-2](@/book/12-model-loading.md)) holds `F32`, `I64` and `BOOL` tensors side by side. With `Tensor<T>`, a `std::map<std::string, ???>` needs a type-erasure layer on top of the templates, and that layer is a runtime dtype in disguise.
- **The autograd graph.** A graph node ([Autograd](@/book/06-autograd.md)) stores its inputs and receives gradients. One graph mixes dtypes: a comparison produces `bool`, an index tensor is `int64`, the loss is `float32`. Nodes would need a common, type-erased tensor handle.
- **Python.** The bindings ([Python bindings](@/book/10-python-bindings.md)) expose one `kael.Tensor` type, as PyTorch exposes one `torch.Tensor`. Python has no templates. With `Tensor<T>`, every bound function needs one overload per dtype, and the binding layer must pick one at run time anyway.
- **Mixed-dtype arithmetic.** `int64 + float32` is legal and returns `float32`. With templates, every binary operation needs an overload for each of the 16 ordered pairs of dtypes, and type promotion becomes a compile-time computation over every combination.

The pattern is the same in all four cases. Somewhere, the program must decide the type at run time. The question is only where that decision lives.

```text
compile-time dtype                     runtime dtype

Tensor<float>  Tensor<int64_t>         Tensor { impl -> dtype = F32 }
      \             /                  Tensor { impl -> dtype = I64 }
   AnyTensor (type erasure)                       |
         |                                  dispatch(dtype, f)
   switch on a tag                                |
         |                             f<float>()  f<int64_t>() ...
   Tensor<T> again
```

The left column ends up with a runtime tag anyway, plus a second type for every tensor. The right column has one type and one place where the tag is read.

## Decision

The element type is a runtime value. From [`include/kael/dtype.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/dtype.h):

```cpp
// Bool is stored as std::uint8_t: sizeof(bool) is not fixed by the standard and
// std::vector<bool> cannot hand out element pointers.
enum class DType { F32, F64, I64, Bool };
```

The value lives in `TensorImpl::dtype` ([`include/kael/tensor_impl.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/tensor_impl.h)), next to the shape, strides and offset. There is one C++ class, `kael::Tensor`.

### Dispatch

Code that needs the concrete element type calls `dispatch(dtype, f)`. It is a `switch` that calls the templated lambda `f` with the matching C++ type. From `include/kael/dtype.h`:

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

The lambda is a C++20 templated lambda, written `[&]<typename T>() { ... }`. Inside it, `T` is a real compile-time type, so the loop body is ordinary typed code that the compiler can optimize and vectorize. The `switch` runs once per operation, not once per element.

Two restricted variants exist for kernels that do not make sense for every dtype:

| Function | Types it instantiates | Rejects |
|----------|----------------------|---------|
| `dispatch(dtype, f)` | `float`, `double`, `std::int64_t`, `std::uint8_t` | nothing |
| `dispatch_floating(dtype, op, f)` | `float`, `double` | `int64` and `bool`, with `TypeError` |
| `dispatch_no_bool(dtype, op, f)` | `float`, `double`, `std::int64_t` | `bool`, with `TypeError` |

The `op` argument names the operation in the error message, for example `softmax only supports floating point dtypes, got int64`.

### Traits between the two worlds

Two traits map between the enum and C++ types at compile time:

| Trait | Direction | Example |
|-------|-----------|---------|
| `dtype_of_v<T>` | C++ type to enum | `dtype_of_v<double>` is `DType::F64` |
| `cpp_type_t<D>` | enum to C++ type | `cpp_type_t<DType::I64>` is `std::int64_t` |

`Tensor::data_ptr<T>()` uses `dtype_of_v<T>` to check that the caller asked for the type the tensor holds.

### Bool storage

`Bool` is stored as `std::uint8_t` holding 0 or 1. Two facts force this:

- `sizeof(bool)` is implementation defined. A kernel that computes byte offsets from `sizeof(bool)` would be non-portable, and the safetensors `BOOL` format fixes one byte per element.
- `std::vector<bool>` is a packed bit vector. It cannot hand out a `bool*` to an element, so it cannot back a strided view.

A plain `static_cast<std::uint8_t>(2.5)` produces 2, which is not a valid boolean. `convert_value` handles the target type explicitly:

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

Any non-zero value becomes 1 when the target is `Bool`. A floating-point value converted to `int64` saturates and NaN becomes 0, because an out-of-range float-to-integer cast is undefined behavior in C++.

### A worked example

Trace `a + b` with `a` of dtype `int64` and shape `[2]`, and `b` of dtype `float32` and shape `[2]`.

```text
a: int64   [2]  storage bytes: 16
b: float32 [2]  storage bytes:  8

1. result_dtype({&a, &b})       promote_types(I64, F32) -> F32
2. inputs cast to F32           a.to(F32) allocates a new float tensor
                                b.to(F32) returns b itself, no copy
3. TensorIterator built         every operand is float32
4. dispatch(F32, ...)           calls the lambda with T = float
5. inner loop                   out[i] = convert_value<float>(x[i] + y[i])
```

The dtype is read twice: once to compute the result type, once in `dispatch`. Each read is a branch on a four-value enum. The loop over elements contains no dtype checks.

## Alternatives considered

### A templated Tensor<T>

The tensor type is `Tensor<T>`, as in Eigen's `Matrix<T, ...>` or xtensor's `xarray<T>`.

- Gains: dtype errors at compile time, no dispatch branch, no runtime dtype checks in accessors.
- Costs: a second, type-erased type for containers, the autograd graph and Python; one overload or template per dtype pair for mixed arithmetic; promotion rules written as template metaprograms; error messages from template instantiation instead of a sentence.
- Rejected because every major consumer of tensors in kael (graph nodes, state dicts, safetensors, Python) needs the type-erased form. The templated form would be the exception, not the rule.

### Templates plus a variant

Keep `Tensor<T>` and add `using AnyTensor = std::variant<Tensor<float>, Tensor<double>, Tensor<std::int64_t>, Tensor<std::uint8_t>>`. `std::visit` plays the role of `dispatch`.

- Gains: the typed API stays available for code that knows its dtype.
- Costs: two ways to do everything. Every operation needs a typed version and a variant version. A binary `std::visit` over two variants instantiates 16 combinations, most of which are invalid or only differ by a cast.
- Rejected as more code for the same runtime behavior.

### Runtime dtype with virtual dispatch

Make storage polymorphic: `FloatStorage`, `Int64Storage` and so on, with virtual methods for each operation.

- Gains: no `switch` statements.
- Costs: one virtual method per operation per class, so adding an operation touches every subclass. A virtual call per element would destroy performance; a virtual call per loop is the same as `dispatch` with more boilerplate. Binary operations need double dispatch.
- Rejected because the set of dtypes is small and closed, while the set of operations is large and grows. A `switch` over a closed enum fits that shape. Virtual functions fit the opposite shape.

### What PyTorch and NumPy do

Both chose a runtime dtype.

- PyTorch's `at::Tensor` carries a `ScalarType` value. Kernels use macros in the `AT_DISPATCH_*` family, which expand to a `switch` that defines `scalar_t` in each branch. kael's `dispatch` is the same idea written with C++20 templated lambdas instead of macros.
- NumPy's `ndarray` carries a `dtype` object. A universal function (ufunc) holds a table of inner loops, one per supported type signature, and selects a loop at call time.

kael follows the same model so that a reader who knows `tensor.dtype` in PyTorch or `array.dtype` in NumPy finds the same concept.

## Consequences

### Every element access goes through dispatch

Every operation that touches elements goes through `dispatch` or one of its variants. A kernel is written once, as a generic lambda, and instantiated for each dtype it supports. The rest of the library never names `float` directly in a kernel that should work for every dtype.

### Dtype mismatches are runtime errors

A mismatch that templates would catch at compile time now throws. `data_ptr<double>()` on a `float32` tensor throws `TypeError` with the message `expected dtype float64 but tensor has float32`, instead of failing to compile. The test `Tensor.DataPtrWrongTypeMessage` pins this message.

The trade is deliberate. Tests catch these errors early, the message names both types, and Python users get the same message as a Python exception.

### Promotion is a runtime function

Type promotion is `promote_types`, a `constexpr` function with the order `Bool < I64 < F32 < F64`. It returns whichever input ranks higher.

| a | b | `promote_types(a, b)` |
|---|---|----------------------|
| `Bool` | `I64` | `I64` |
| `I64` | `F32` | `F32` |
| `F32` | `F64` | `F64` |
| `F64` | `Bool` | `F64` |

Because it is `constexpr`, the test `Promotion.Table` checks the table with `static_assert`. The element-wise kernels apply promotion once per operation, before they build the iterator ([TensorIterator](@/book/04-tensor-iterator.md)). `result_dtype` in [`include/kael/kernels.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/kernels.h) refines the rule for 0-dimensional tensors: a 0-d tensor changes the result only when its dtype is in a higher category (bool, integer, floating point). So `int64 [3] + float64 0-d` gives `float64`, while `float32 [3] + float64 0-d` stays `float32`. Scalar arguments follow the same idea through `promote_with_scalar` in `include/kael/scalar.h`: `F32 + 2.5` stays `F32`, and `I64 + 2.5` becomes `F32`.

### The compiler finds every switch that needs a new dtype

`switch` statements over `DType` have no `default` branch. Control reaches `detail::unreachable()` after the `switch` only for an invalid enum value. If a dtype is added, `-Wall` (through `-Wswitch`) flags every `switch` that does not handle the new enumerator, and `KAEL_WARNINGS_AS_ERRORS=ON` in CI turns those warnings into build failures. Adding `float16` would be a guided edit, not a search.

### Code size grows with the dtype set

Each kernel passed to `dispatch` is compiled four times. A float-only kernel such as `softmax` would be compiled for `int64` and `bool` too, and those copies would be dead code or would not compile at all (`std::exp` on a `std::uint8_t`). The restricted dispatchers keep such kernels out of the integer instantiations:

```text
dispatch             4 instantiations  float double int64_t uint8_t
dispatch_no_bool     3 instantiations  float double int64_t
dispatch_floating    2 instantiations  float double
```

### Mixed-dtype kernels still need care

The element-wise kernels cast every input to the result dtype before the iterator runs, so the iterator sees one dtype. `TensorIterator::build` rejects inputs of different dtypes with `iterator inputs must share a dtype`. The cast costs an extra allocation for inputs that are not already of the result type. Comparison kernels are the exception: they read inputs of type `T` and write `std::uint8_t`, which is why `run_typed` takes separate `Out` and `In` type parameters.

## Where it shows up in the code

| File | What to look for |
|------|------------------|
| [`include/kael/dtype.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/dtype.h) | `DType`, `element_size`, `promote_types`, `dtype_category`, `dtype_of_v`, `cpp_type_t`, `convert_value`, `saturating_cast`, `dispatch`, `dispatch_floating`, `dispatch_no_bool` |
| [`include/kael/tensor_impl.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/tensor_impl.h) | `TensorImpl::dtype` |
| [`include/kael/tensor.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/tensor.h) | `data_ptr<T>` and `check_dtype<T>` |
| [`include/kael/kernels.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/kernels.h) | `result_dtype`, `same_dtype_kernel`, `run_typed` |
| [`include/kael/scalar.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/scalar.h) | `promote_with_scalar` |
| [`src/ops/nn.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/ops/nn.cpp), [`src/ops/reduce.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/ops/reduce.cpp) | `dispatch_floating` for softmax, cross entropy and floating reductions |
| [`src/ops/linalg.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/ops/linalg.cpp) | `dispatch_no_bool` for `matmul` |
| [`tests/test_dtype.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_dtype.cpp) | `DType.DispatchSelectsMatchingType`, `DType.DispatchFloatingRejectsIntegers`, `DType.DispatchNoBoolRejectsBool` |
| [`tests/test_iterator.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_iterator.cpp) | `Promotion.Table`, `Promotion.Scalars` |

Chapters that depend on this decision:

- [Memory model](@/book/02-memory-model.md): `TensorImpl`, `Storage` and `element_size`.
- [TensorIterator](@/book/04-tensor-iterator.md): byte strides computed from the dtype, one dtype per iterator.
- [Operations](@/book/05-operations.md): every kernel is a lambda passed to a dispatcher.
- [Autograd](@/book/06-autograd.md): nodes check that each gradient has the dtype of its input.
- [CUDA backend](@/book/09-cuda-backend.md): device kernels receive the dtype as a value too.
- [Python bindings](@/book/10-python-bindings.md): one `kael.Tensor` type with a `dtype` property.
- [Loading GPT-2](@/book/12-model-loading.md): safetensors dtype strings map to `DType` values.

## Check your understanding

<details>
<summary>Where does kael decide the C++ element type of a kernel, and how often does that decision run?</summary>

In `dispatch` or one of its restricted variants. It runs once per operation, as one `switch` over the dtype. The loop over elements inside the templated lambda has a fixed compile-time type and contains no dtype checks.

</details>

<details>
<summary>Why is a bool tensor stored as std::uint8_t and not as bool?</summary>

`sizeof(bool)` is implementation defined, so byte offsets computed from it would not be portable, and file formats such as safetensors fix one byte per element. `std::vector<bool>` is a packed bit vector that cannot return element pointers, so it cannot back a strided view. `convert_value` then keeps the stored values at 0 or 1.

</details>

<details>
<summary>What is the result dtype of an int64 tensor of shape [3] plus a float32 tensor of shape [3], and what happens to each input?</summary>

`promote_types(I64, F32)` is `F32`. The `int64` input is converted with `to(DType::F32)`, which allocates a new tensor. The `float32` input is returned as is, because `to` returns the same tensor when the dtype already matches. The iterator then runs with three `float32` operands.

</details>

<details>
<summary>A contributor adds DType::F16. How does the build help them find the code to update?</summary>

Every `switch` over `DType` has no `default` branch. The compiler's `-Wswitch` warning, enabled by `-Wall`, reports each `switch` that does not handle `F16`. With `KAEL_WARNINGS_AS_ERRORS=ON`, as in CI, those warnings fail the build.

</details>

<details>
<summary>Why does softmax use dispatch_floating instead of dispatch?</summary>

Softmax needs `std::exp` and division with fractional results, which make no sense for `int64` or `bool`. `dispatch` would instantiate the kernel for all four types, producing dead or invalid code. `dispatch_floating` instantiates only `float` and `double` and throws `TypeError` with the operation name for other dtypes.

</details>

<details>
<summary>What does a runtime dtype cost compared with Tensor&lt;T&gt;?</summary>

Mismatches become runtime errors instead of compile errors, for example `data_ptr<double>()` on a `float32` tensor throws `TypeError`. Each operation pays one `switch`. Each kernel is compiled once per supported dtype, which grows the binary. In return there is one tensor type for containers, the autograd graph and Python, and promotion is an ordinary function.

</details>
