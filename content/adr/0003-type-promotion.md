+++
title = "Type promotion"
description = "Why kael picks the result dtype of mixed operations with PyTorch's category-based rules, and where it deliberately departs from them."
weight = 3
slug = "type-promotion"

[extra]
kicker = "ADR 0003"
source = "docs/adr/0003-type-promotion.md"
+++

Once the dtype is a runtime value ([ADR 0001](@/adr/0001-runtime-dtype.md)), every operation with two inputs has to answer a second question: when the inputs have different dtypes, what is the dtype of the result? The same question arises when one input is a plain C++ or Python number, as in `x * 0.5`. kael answers it with PyTorch's category-based rules. This record explains the families of rules that exist, why kael chose this one, and the few places where kael departs from PyTorch on purpose.

**Type promotion** is the rule that picks the result dtype from the input dtypes. A **category** is a coarser grouping of dtypes: Bool, integer and floating point. A **0-d tensor** is a tensor with shape `[]`: one element and no dimensions, such as a loss value or the result of `sum(t)`.

## Status

Accepted in phase 5 (tag `chapter-05-iteration`, which matches [TensorIterator](@/book/04-tensor-iterator.md) in the book). That tag introduces `promote_types` and `promote_with_scalar`. The 0-d rule in `result_dtype` and the per-operation overrides arrive at `chapter-06-ops` ([Operations](@/book/05-operations.md)). Still in force at `v1.0.0` and at the current head of the repository.

## Context

Mixed dtypes are common in ordinary code:

- `int64 + float32`: an index or a count combined with an activation.
- `x * 0.5`: a float32 tensor scaled by a literal.
- `loss * scale`: a float32 activation combined with a 0-d tensor that may be float64.
- `mask + mask`: two Bool tensors combined arithmetically.

Each case needs a defined result dtype. Three families of rules are in use.

- **Strict typing.** Mixed inputs are an error and the caller casts explicitly. JAX offers this as its strict promotion mode. The rule is easy to state, but every loss scaled by a literal and every integer index added to a float needs a cast.
- **Value-based promotion.** The dtype of a scalar depends on its value. NumPy used this rule before version 2.0: whether a Python integer fit into the array's integer type changed the result type. The result dtype then depends on the data, not only on the types, which makes kernels and tests harder to reason about.
- **Category-based promotion.** Dtypes are ranked. A Python number or a 0-d tensor changes the result only when it belongs to a higher category than the tensors with dimensions. PyTorch uses this rule.

kael's main use is reproducing PyTorch results. Its golden tests ([Operations](@/book/05-operations.md)) run the same operation in kael and in PyTorch and compare the outputs, including the dtype. A rule that differs from PyTorch would make every mixed case a special case in the test generator.

```text
expression                 strict      value-based         category-based
float32 [3] * 0.5          error       float32             float32
int64 [3] + 2.5            error       float64 (NumPy 1)   float32
float32 [3] + float64 0-d  error       depends on value    float32
int64 [3] + float32 [3]    error       float64 (NumPy 1)   float32
```

## Decision

kael follows PyTorch's category-based rules. The rule has four parts, each a small function.

### The full order

`promote_types` in [`include/kael/dtype.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/dtype.h) ranks the dtypes `Bool < I64 < F32 < F64` and returns the higher one. It applies when every input is a tensor of the same kind, all with dimensions or all 0-d.

| `promote_types` | Bool | I64 | F32 | F64 |
|-----------------|------|-----|-----|-----|
| Bool | Bool | I64 | F32 | F64 |
| I64 | I64 | I64 | F32 | F64 |
| F32 | F32 | F32 | F32 | F64 |
| F64 | F64 | F64 | F64 | F64 |

### Categories

`dtype_category` maps each dtype to its category:

```cpp
constexpr int dtype_category(DType dtype) {
    return dtype == DType::Bool ? 0 : (dtype == DType::I64 ? 1 : 2);
}
```

F32 and F64 share category 2. That is the property the next two rules use: a scalar or a 0-d tensor can move the result from integer to floating point, but it cannot move it from float32 to float64.

### Scalars

A number passed to an operation becomes a `Scalar` ([`include/kael/scalar.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/scalar.h)). Its dtype comes from its C++ type, not from its value: `true` is Bool, `2` is I64, and both `2.5` and `2.5f` are F32. The value is stored as `std::int64_t` or `double`, so no precision is lost before the final conversion. `promote_with_scalar` decides the result:

```cpp
// A scalar never raises the precision of a tensor within the same category:
// F32 + 2.5 stays F32, while I64 + 2.5 moves to the default float, F32.
constexpr DType promote_with_scalar(DType tensor_dtype, Scalar scalar) {
    return dtype_category(scalar.dtype()) <= dtype_category(tensor_dtype) ? tensor_dtype
                                                                          : scalar.dtype();
}
```

| Tensor | Scalar | Result | Why |
|--------|--------|--------|-----|
| F32 | `2.5` | F32 | same category, the tensor wins |
| F64 | `2.5f` | F64 | same category, the tensor wins |
| I64 | `2.5` | F32 | higher category, the scalar's dtype wins |
| Bool | `2` | I64 | higher category |
| F32 | `2` | F32 | lower category |
| Bool | `true` | Bool | same category |

When the scalar wins, the result takes the scalar's dtype. A floating scalar is always F32, the default float, so `I64 + 2.5` gives F32 and not F64.

### 0-d tensors

`result_dtype` in [`include/kael/kernels.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/kernels.h) applies the same idea to 0-d tensors. It splits the inputs into two groups, promotes each group with `promote_types`, and lets the 0-d group win only with a higher category:

```cpp
// Tensors with dimensions decide the dtype; a 0-d tensor only wins with a higher category.
inline DType result_dtype(std::initializer_list<const Tensor*> tensors) {
    std::optional<DType> dimensioned;
    std::optional<DType> zero_dimensional;
    for (const Tensor* t : tensors) {
        std::optional<DType>& group = t->dim() == 0 ? zero_dimensional : dimensioned;
        group = group ? promote_types(*group, t->dtype()) : t->dtype();
    }
    if (!dimensioned || !zero_dimensional) {
        return dimensioned ? *dimensioned : *zero_dimensional;
    }
    return dtype_category(*zero_dimensional) > dtype_category(*dimensioned) ? *zero_dimensional
                                                                            : *dimensioned;
}
```

A scalar operand reaches `result_dtype` as a 0-d tensor. `resolve_operands` in [`src/ops/binary.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/ops/binary.cpp) materializes it with the dtype from `promote_with_scalar`, so scalars and 0-d tensors follow one path from there on.

### Per-operation overrides

`result_dtype` is the default. A few operations adjust it in [`src/internal/arithmetic.h`](https://github.com/JetBerri/kael.cpp/blob/main/src/internal/arithmetic.h) and [`src/ops/inplace.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/ops/inplace.cpp):

| Operation | Rule | Function | Same as PyTorch |
|-----------|------|----------|-----------------|
| `add` | Bool + Bool gives I64 | `add_dtype` | no, PyTorch gives Bool |
| `div` | an integer or Bool result becomes F32 | `div_dtype` | yes |
| `sub`, `pow`, `floor_divide` | Bool inputs throw `TypeError` | `check_no_bool` | PyTorch also rejects Bool `sub` |
| comparisons | compare in `result_dtype`, output Bool | `comparison_kernel` | yes |
| in-place ops | the result category must not exceed the category of `self` | `check_result_fits` | yes |

### Where promotion runs

Promotion runs once per operation, in the kernel layer, before the loop. `binary_kernel` computes `result_dtype`, and `same_dtype_kernel` converts each input with `to(dtype)` before it adds the input to the `TensorIterator`. `to` returns the same tensor when the dtype already matches, and allocates a converted copy otherwise. `TensorIterator::build` then rejects inputs of different dtypes with `iterator inputs must share a dtype`, so the inner loop always runs on one C++ type.

### A worked example

Trace `kael::ones({3}, DType::I64) * 0.5`.

```text
a: int64 [3]      0.5: Scalar, dtype F32, value stored as double

1. Operand(0.5)                 holds a Scalar of dtype F32
2. resolve_operands             promote_with_scalar(I64, 0.5): category 2 > 1 -> F32
                                0.5 becomes a 0-d float32 tensor
3. result_dtype({&a, &half})    dimensioned group I64, 0-d group F32
                                category 2 > 1 -> F32
4. same_dtype_kernel            a.to(F32) allocates [1, 1, 1] as float32
                                the 0-d tensor is already F32, no copy
5. iterator, dispatch(F32)      out[i] = 1.0f * 0.5f
6. result                       float32 [3] = [0.5, 0.5, 0.5]
```

Now replace `a` with a float64 tensor. Step 2 gives F64, because the scalar's category (2) is not higher than the tensor's (2). The result is float64. Replace it with a float32 tensor and a float64 0-d tensor `scalar(0.5, DType::F64)`. Step 3 sees F32 and F64 in the same category, so the tensor with dimensions wins and the result stays float32.

## Alternatives considered

### Strict typing

Every mixed operation throws, and the caller writes `x.to(DType::F32) * y`.

- Gains: no hidden conversions. The result dtype is always the dtype of the inputs.
- Costs: `x * 0.5` on an integer tensor, `loss / batch_size` and every mask arithmetic need a cast. Ports from PyTorch need edits on lines that PyTorch accepts. The golden tests could not cover mixed cases at all.
- Rejected because it makes the common case noisy and breaks line-by-line comparison with PyTorch ([ADR 0004](@/adr/0004-pytorch-shaped-api.md)).

### Value-based promotion

The dtype of a scalar depends on whether its value fits a smaller type, as in NumPy before 2.0.

- Gains: small literals never widen a tensor.
- Costs: the result dtype depends on runtime values. A test that passes with `x + 1` can fail with `x + 300` on a narrower integer type. The rule needs a range check for each scalar and each dtype.
- Rejected because kael has a single integer type, so the main benefit of value-based rules, keeping narrow integer types narrow, does not apply. A rule based on types alone is also easier to test. NumPy itself moved away from value-based promotion in version 2.0.

### Full order for everything

Apply `promote_types` to every input, scalars and 0-d tensors included.

- Gains: one rule instead of three.
- Costs: a float64 0-d tensor, such as a loss accumulated in float64 or `scalar(0.5, DType::F64)`, would push a float32 activation to float64 and double its memory. The result dtype of a large computation would depend on the dtype of one element.
- Rejected. The category rules exist to stop exactly this silent upcast.

### Promote int64 + float32 to float64

`int64 + float32` would give float64, as in classic NumPy, so every int64 value up to 2^53 stays exact.

- Gains: fewer precision surprises with large integers.
- Costs: every mixed operation doubles its memory and bandwidth, and kael would disagree with PyTorch on one of the most common mixed cases.
- Rejected. kael follows PyTorch and documents the precision loss instead.

## Consequences

### Literals and loss values do not upcast

`x * 0.5` on a float32 tensor stays float32. A float64 0-d tensor does not upcast a large float32 activation. Integer and Bool tensors still become floating point when a floating scalar or 0-d tensor touches them, because that change crosses a category.

### Large integers lose precision in float32

`int64 + float32` gives float32, as in PyTorch. Float32 has a 24-bit significand, so integers above 2^24 are not all representable: `16777217` becomes `16777216`. Code that needs exact large integers keeps them in int64 or converts to float64 explicitly.

### Bool + Bool counts

`add` turns Bool + Bool into I64, so `true + true` is `2` (`Arithmetic.BoolPlusBoolIsInt64`). PyTorch returns Bool and behaves as a logical or. kael counts instead, and `logical_or` gives the PyTorch meaning. The golden generator skips the Bool pattern for `add` (`BOOL_PATTERNS[1:]` in [`scripts/generate/golden.py`](https://github.com/JetBerri/kael.cpp/blob/main/scripts/generate/golden.py)). The difference is documented in [TensorIterator](@/book/04-tensor-iterator.md), the chapter that introduces it. `mul` of two Bool tensors stays Bool (`Arithmetic.MulOfBoolsIsBool`).

### In-place operations cannot change dtype

An in-place operation writes into the existing storage of `self`, so it cannot change the dtype of `self`. `check_result_fits` allows any result whose category is not higher than the category of `self`, and converts the result on store:

```text
float32 t; t.add_(float64 tensor)   F64 is category 2, t is category 2: allowed, rounds to float32
float32 t; t.add_(int64 tensor)     F32, category 2: allowed
int64 t;   t.add_(1)                I64, category 1: allowed
int64 t;   t.add_(0.5)              F32, category 2 > 1: TypeError
int64 t;   t.div_(2)                div_dtype gives F32: TypeError
bool t;    t.add_(t)                add_dtype gives I64, category 1 > 0: TypeError
```

The error names the operation and both dtypes, for example `add_ result of dtype float32 cannot be stored in a tensor of dtype int64` (`InPlace.LosingCategoryThrows`). The check uses categories, not the full order, so the float32 and float64 case works, as it does in PyTorch.

### Mixed inputs cost a conversion

Because inputs are converted with `to(dtype)` before the iterator runs, an input that does not already have the result dtype is copied once into a new tensor. The kernels stay simple: each one is compiled for a single element type ([ADR 0001](@/adr/0001-runtime-dtype.md)). Comparison kernels are the one place where input and output types differ: they read the promoted type and write `std::uint8_t`.

### The rules are testable at compile time

`promote_types`, `dtype_category` and `promote_with_scalar` are `constexpr`. `Promotion.Table` and `Promotion.Scalars` in [`tests/test_iterator.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_iterator.cpp) check them with `static_assert`, so a wrong rule fails the build. The tests in [`tests/test_ops_binary.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_ops_binary.cpp) check the same rules at run time through real operations, and the golden cases `zero_dim_wider`, `zero_dim_higher_category` and `zero_dim_lower_category` check them against PyTorch.

## Where it shows up in the code

| File | What to look for |
|------|------------------|
| [`include/kael/dtype.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/dtype.h) | `promote_types`, `dtype_category`, `is_floating_point` |
| [`include/kael/scalar.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/scalar.h) | `Scalar` constructors, `promote_with_scalar` |
| [`include/kael/kernels.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/kernels.h) | `result_dtype`, `binary_kernel`, `same_dtype_kernel`, `comparison_kernel` |
| [`src/ops/binary.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/ops/binary.cpp) | `materialize`, `resolve_operands`, `add`, `div` |
| [`src/internal/arithmetic.h`](https://github.com/JetBerri/kael.cpp/blob/main/src/internal/arithmetic.h) | `add_dtype`, `div_dtype`, `check_no_bool` |
| [`src/ops/inplace.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/ops/inplace.cpp) | `check_result_fits`, `binary_inplace` |
| [`src/iterator.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/iterator.cpp) | the `iterator inputs must share a dtype` check in `build` |
| [`tests/test_iterator.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_iterator.cpp) | `Promotion.Table`, `Promotion.Scalars` |
| [`tests/test_ops_binary.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_ops_binary.cpp) | `Promotion.TensorPairs`, `Promotion.ScalarsDoNotRaisePrecision`, `Promotion.ZeroDimensionalTensorsOnlyWinWithHigherCategory`, `Arithmetic.BoolPlusBoolIsInt64` |
| [`tests/test_ops_inplace.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_ops_inplace.cpp) | `InPlace.DtypeNeverChanges`, `InPlace.LosingCategoryThrows` |

Chapters that depend on this decision:

- [TensorIterator](@/book/04-tensor-iterator.md): promotion tables, scalars, 0-d tensors and the per-operation overrides.
- [Operations](@/book/05-operations.md): in-place rules, reductions with their own dtype rules, and the golden tests against PyTorch.
- [Autograd](@/book/06-autograd.md): a gradient is converted back to the dtype of its input, so a mixed-dtype operation returns each gradient in the dtype of the tensor that requires it.
- [Python bindings](@/book/10-python-bindings.md): Python `bool`, `int` and `float` arguments become `Scalar` values with dtype Bool, I64 and F32, so the same rules apply.

## Check your understanding

<details>
<summary>What is the dtype of ones({3}, DType::I64) + 2.5, and why not float64?</summary>

`float32`. The scalar `2.5` has dtype F32 because every floating scalar takes the default float. Its category (2) is higher than the tensor's (1), so `promote_with_scalar` returns the scalar's dtype, F32.

</details>

<details>
<summary>A float32 activation of shape [1024, 1024] is multiplied by a float64 0-d tensor. What is the result dtype, and what would a full-order rule have cost?</summary>

`float32`. F32 and F64 share category 2, so the 0-d group does not win in `result_dtype`. A full-order rule would return float64 and allocate a result twice the size, 8 MB instead of 4 MB, from a single scalar value.

</details>

<details>
<summary>Why does int64_tensor.add_(0.5) throw while float32_tensor.add_(float64_tensor) does not?</summary>

The first result is F32, category 2, which is higher than the int64 category 1, so `check_result_fits` throws `TypeError`. The second result is F64, category 2, the same category as float32, so the check passes and the result is rounded to float32 on store.

</details>

<details>
<summary>Where is the result dtype computed, and does the TensorIterator ever see mixed dtypes?</summary>

In the kernel layer: `binary_kernel` calls `result_dtype`, and `same_dtype_kernel` converts each input with `to(dtype)` before adding it to the iterator. The iterator never sees mixed inputs. If it did, `build` would throw `iterator inputs must share a dtype`.

</details>

<details>
<summary>Why does the golden generator skip the bools pattern for add but not for mul?</summary>

kael's `add` turns Bool + Bool into I64 and counts, while PyTorch returns Bool. The outputs would never match, so `add` uses `BOOL_PATTERNS[1:]`. `mul` of two Bool tensors stays Bool in both libraries, so it runs every Bool pattern.

</details>

<details>
<summary>Which test fails first if someone changes promote_types to return F64 for I64 and F32?</summary>

`Promotion.Table` in `tests/test_iterator.cpp`. Its checks are `static_assert`s, so the test file stops compiling before any test runs. `Promotion.TensorPairs` and the golden tests would fail at run time after that.

</details>
