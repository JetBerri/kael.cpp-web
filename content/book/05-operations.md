+++
title = "Operations"
description = "Arithmetic, math functions, in-place ops, reductions, matmul, indexing, sequence ops and the softmax family, checked case by case against PyTorch."
weight = 5
slug = "operations"

[extra]
chapter = "05"
kicker = "Chapter 05"
builds = "Arithmetic, math functions, reductions, matmul, indexing and softmax, checked against PyTorch"
source = "docs/05-operations.md"
tag = "chapter-06-ops"
math = true
+++

This chapter builds the operator library on top of the iterator: elementwise arithmetic and math, in-place variants, reductions, matrix multiplication, indexing, the softmax family and sequence operations. It also builds the golden test pipeline that compares 783 cases across 67 operations against PyTorch. It corresponds to the operations phase of the project.

| File | Role |
|------|------|
| [`include/kael/ops.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/ops.h) | the public declarations of every free-function op and the operator overloads |
| [`src/ops/binary.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/ops/binary.cpp) | `add`, `sub`, `mul`, `div`, `floor_divide`, `pow`, `maximum`, `minimum`, comparisons, logical ops, `equal`, operators, `detail::resolve_operands` |
| [`src/ops/unary.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/ops/unary.cpp) | `neg`, `abs`, `sign`, `exp`, `log`, `log1p`, `expm1`, `sqrt`, `rsqrt`, `sin`, `cos`, `tanh`, `sigmoid`, `relu`, `gelu`, `clamp` |
| [`src/ops/inplace.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/ops/inplace.cpp) | `add_`, `sub_`, `mul_`, `div_`, `pow_`, `clamp_`, `copy_`, `fill_`, `zero_`, `masked_fill_`, `scatter_add_`, `index_add_`, compound assignment |
| [`src/ops/reduce.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/ops/reduce.cpp) | `sum`, `prod`, `mean`, `max`, `min`, `argmax`, `argmin`, `var`, `std`, `all`, `any`, `norm` |
| [`src/ops/linalg.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/ops/linalg.cpp) | `matmul`, `dot`, `outer`, `mm`, `bmm` |
| [`src/ops/indexing.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/ops/indexing.cpp) | `cat`, `stack`, `where`, `masked_fill`, `index_select`, `gather` |
| [`src/ops/nn.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/ops/nn.cpp) | `logsumexp`, `softmax`, `log_softmax`, `cross_entropy_losses`, `standardize` |
| [`src/ops/sequence.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/ops/sequence.cpp) | `cumsum`, `sort`, `topk`, `tril`, `triu` |
| [`src/internal/`](https://github.com/JetBerri/kael.cpp/tree/main/src/internal) `arithmetic.h`, `index_checks.h`, `nan_aware.h`, `operands.h` | shared helpers for dtype rules, index validation, NaN handling and operand resolution |
| [`scripts/generate/golden.py`](https://github.com/JetBerri/kael.cpp/blob/main/scripts/generate/golden.py), [`tests/golden/`](https://github.com/JetBerri/kael.cpp/tree/main/tests/golden), [`tests/support/`](https://github.com/JetBerri/kael.cpp/tree/main/tests/support), [`tests/test_golden.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_golden.cpp) | the golden test pipeline |

## Prerequisites

- [03. Views](@/book/03-views.md): `expand`, `narrow`, `permute`, `contiguous`, `copy_`, aliasing rules.
- [04. TensorIterator](@/book/04-tensor-iterator.md): `binary_kernel` and friends, type promotion, `dispatch`, element functions.

## Why this layer exists

### The problem

The iterator from [04. TensorIterator](@/book/04-tensor-iterator.md) can apply a functor to aligned positions of several tensors. A user does not want a functor. A user wants `x + 1`, `sum(x, {1})`, `a.matmul(b)` and `softmax(logits, -1)`, with the same semantics as PyTorch or NumPy, because that is what the papers, the tutorials and the reference implementations assume.

Matching those semantics is most of the work. Each op has rules that are simple to state and subtle to get exactly right:

| Expression | Naive C++ answer | PyTorch answer |
|------------|------------------|----------------|
| `int_tensor / 2` | integer division | float32 true division |
| `floor_divide(-7, 2)` | `-3`, truncation toward zero | `-4`, as Python's `//` |
| maximum of `[1, NaN, 3]` | depends on where the NaN sits | NaN |
| `sum` of an empty tensor | undefined until someone decides | 0 |
| `max` of an empty tensor | undefined until someone decides | an error |
| shape of `[3] @ [5, 3, 2]` | not a valid matrix product | `[5, 2]` |
| `softmax([1000, 1000])` in float32 | `inf / inf = NaN` | `[0.5, 0.5]` |

Each wrong answer is a silent numerical bug in a model built on top. When a model trains in PyTorch and diverges in kael, one of these rules is a likely cause.

### What breaks without it

Without a uniform operator layer, every op would reinvent promotion, autograd recording, error messages and edge cases. Composite ops like softmax would show up in the autograd graph as five nodes with five saved tensors instead of one. And without a systematic comparison against PyTorch, the only tested cases would be the ones the author thought of.

### The mental model

Every op in this chapter is **a thin, checked wrapper around one of three engines**:

```text
 elementwise ops ---------> TensorIterator + functor          (chapter 04)
 reductions --------------> plan_reduction + reduce_chunks    (this chapter)
 matmul ------------------> multiply_batches + matrix kernel  (this chapter, chapter 08)
 indexing, sequence ops --> views + copy_ + small loops       (this chapter)
```

The wrapper validates inputs, chooses the result dtype, runs the engine without recording autograd history, and records one autograd node at the end. The golden tests then hold the whole layer to PyTorch's behavior, dtype included.

## Concepts

### The shape of an op

Every public op in kael follows the same four-step pattern. `mul` in `src/ops/binary.cpp` is the shortest example:

```cpp
Tensor mul(const Operand& a, const Operand& b) {
    const auto [x, y] = detail::resolve_operands(a, b);
    Tensor out = binary_kernel(x, y, detail::kTimes);
    autograd::record_mul(out, x, y);
    return out;
}
```

1. **Normalize the inputs.** `resolve_operands` turns a `Scalar` operand into a 0-d tensor whose dtype follows `promote_with_scalar`, so `t * 2` and `t * scalar(2)` behave the same.
2. **Validate and choose the result dtype.** `mul` uses the default `result_dtype` inside `binary_kernel`; other ops add checks.
3. **Compute without recording autograd history.** The kernel templates open a `NoGradGuard` internally; ops that compose several kernels wrap the body in `autograd::without_grad`.
4. **Record one autograd node** for the whole op with an `autograd::record_*` function. [06. Autograd](@/book/06-autograd.md) describes what each one saves.

Separating step 3 from step 4 means a composite op such as `softmax`, built from `max`, `sub`, `exp`, `sum` and `div`, appears in the graph as one `SoftmaxBackward` node with one hand-written gradient, not five nodes. The hand-written gradient is also cheaper and more accurate than chaining five generic ones.

`Operand` is implicitly constructible from a `Tensor` and from any type convertible to `Scalar`, so `include/kael/ops.h` declares the arithmetic operators once for all combinations:

```cpp
Tensor operator+(const Operand& a, const Operand& b);
Tensor operator*(const Operand& a, const Operand& b);
Tensor& operator+=(Tensor& t, const Operand& other);
```

`x + 1`, `1 + x` and `x + y` all resolve to `add`. `x += y` calls `x.add_(y)`. Comparison operators (`==`, `<`, ...) follow the same pattern and return Bool tensors, as in PyTorch and NumPy. That is why `equal(a, b)` exists as a separate function returning a C++ `bool`.

### Keepdim and reduction shapes

A **reduction** combines many values into one: a sum, a maximum, a mean. A reduction over some dimensions of a tensor produces one value per position of the remaining, **kept**, dimensions. The `keepdim` flag decides what happens to the reduced dimensions in the output shape:

```text
t: shape [2, 3, 4]

sum(t, {1})         -> [2, 4]        reduced dim removed
sum(t, {1}, true)   -> [2, 1, 4]     reduced dim kept with size 1
sum(t)              -> []            every dim reduced
sum(t, {}, true)    -> [1, 1, 1]
```

`keepdim = true` exists for broadcasting. `t - mean(t, {1}, true)` subtracts a `[2, 1, 4]` tensor from a `[2, 3, 4]` tensor, and broadcasting lines the mean up with every row. Without `keepdim`, the `[2, 4]` mean would broadcast against the last two dimensions `[3, 4]` and fail. Softmax, layer normalization and variance all rely on this.

### Numerical stability

Floating-point arithmetic rounds after every operation. Three effects shape the design of reductions and softmax in this chapter:

- **Overflow.** float32 holds values up to about `3.4e38`. `exp(89)` is already larger and becomes `inf`.
- **Accumulated rounding.** Adding a small value to a large running total loses the low bits of the small value. A float32 running sum stops growing once the total is about 2^24 times the term.
- **Catastrophic cancellation.** Subtracting two nearly equal large numbers leaves only their rounding error. `E[x^2] - E[x]^2` does this when the mean is large compared to the spread.

Each section below names the effect it guards against and the test that proves it.

## Elementwise binary ops

| Op | Dtype rule | Notes |
|----|------------|-------|
| `add` | `add_dtype`: Bool + Bool gives I64, else `result_dtype` | inputs cast before the kernel |
| `sub` | `result_dtype` | Bool throws |
| `mul` | `result_dtype` | Bool * Bool stays Bool |
| `div` | `div_dtype`: integer results become F32 | true division, never integer division |
| `floor_divide` | `result_dtype` | Bool throws; integer division by zero throws |
| `pow` | `result_dtype` | Bool throws; integer base with a negative integer scalar exponent throws |
| `maximum`, `minimum` | `result_dtype` | NaN propagates |
| `eq`, `ne`, `lt`, `le`, `gt`, `ge` | compute in `result_dtype`, output Bool | `comparison_kernel` |
| `logical_and`, `logical_or`, `logical_xor` | inputs converted to Bool, output Bool | |
| `logical_not` | output Bool | `IsZero` on the Bool input |

`add` and `div` compute their dtype first, cast both inputs to it, and then call `binary_kernel` inside `autograd::without_grad`, so the casts do not record autograd nodes. `div` uses true division: `ones({3}, DType::I64) / 2` gives float32 `[0.5, 0.5, 0.5]`, as in PyTorch and Python 3. Floating division by zero follows IEEE 754: `1 / 0` is `inf`, `0 / 0` is NaN (`Arithmetic.FloatDivisionByZeroFollowsIeee`).

Comparisons compute in the promoted dtype. `ones({3}, DType::I64) < 1.5` promotes to F32 first, so `1 < 1.5` is true, rather than truncating `1.5` to `1`. NaN compares unequal to everything, itself included, so only `ne` returns true for a NaN (`Comparisons.NanIsOnlyNotEqual`).

`equal(a, b)` returns a C++ `bool`: false if the shapes differ, otherwise true when every element of `eq(a, b)` is set. It copies to the host first, so it works for CUDA tensors.

### floor_divide

**Floor division** rounds the quotient toward negative infinity, as Python's `//` does. C++ `/` on integers truncates toward zero. The two agree when the signs match and differ by one when they do not and the division is inexact. The integer version corrects the truncated quotient in exactly that case. From `src/ops/binary.cpp`:

```cpp
template <typename T> T integer_floor_division(T a, T b) {
    KAEL_CHECK(b != T{0}, "floor_divide by zero with integer dtype");
    if (b == static_cast<T>(-1)) {
        using Wrapping = std::make_unsigned_t<T>;
        return static_cast<T>(Wrapping{0} - static_cast<Wrapping>(a));
    }
    const T quotient = static_cast<T>(a / b);
    const bool rounded_towards_zero = (a % b != 0) && ((a < 0) != (b < 0));
    return rounded_towards_zero ? static_cast<T>(quotient - 1) : quotient;
}
```

The `b == -1` branch exists because `INT64_MIN / -1` overflows: the true answer `2^63` does not fit in `int64`. That is undefined behavior in C++ and a `SIGFPE` on x86. Negating through the unsigned type wraps instead, so `INT64_MIN // -1` is `INT64_MIN` (`Arithmetic.FloorDivideOfMinimumByMinusOneWraps`). Division by zero throws `kael::Error` instead of crashing.

Trace for `floor_divide([7, -7, 7, -7], [2, 2, -2, -2])`:

| `a` | `b` | `a / b` in C++ | `a % b` | Signs differ | Result |
|-----|-----|----------------|---------|--------------|--------|
| 7 | 2 | 3 | 1 | no | 3 |
| -7 | 2 | -3 | -1 | yes | -4 |
| 7 | -2 | -3 | 1 | yes | -4 |
| -7 | -2 | 3 | -1 | no | 3 |

The compiled check gives `[3, -4, -4, 3]`.

The floating version, `floating_floor_division`, starts from `std::fmod`. It computes `remainder = fmod(a, b)` and `quotient = (a - remainder) / b`, subtracts 1 when the remainder is nonzero and its sign differs from the sign of `b`, returns a zero with the sign of `a / b` when the quotient is zero, and finally rounds the quotient to the nearest integer to remove the rounding error of the division. This mirrors the algorithm PyTorch and CPython use, so signed zeros and exact multiples come out the same. Division by zero falls through to `a / b`, giving `inf` or NaN. The result for floats is checked against PyTorch, including the 100 special-value pairs in the golden `special` case.

`floor_divide` is built from a lambda, not a tagged functor, so on CUDA it runs through the host fallback of the kernel templates.

### pow

Floating types call `std::pow`. Integer types use `functions::integer_power`: exponentiation by squaring in the unsigned type ([04. TensorIterator](@/book/04-tensor-iterator.md)), so overflow wraps instead of being undefined. A negative integer exponent has an integer answer only for bases 1 and -1, so kael returns 1 for base 1, 1 or -1 for base -1 depending on the parity of the exponent, and 0 for every other base, which is the truncated value of `1 / base^n`.

Compiled check: `pow([2, 1, -1, -1, 0], [-1, -1, -1, -2, -1])` gives `[0, 1, -1, 1, 0]`. The golden case `i64_negative_exponent` checks the same values against PyTorch.

When the exponent is a C++ integer scalar and the base is an integer tensor, `detail::check_integer_exponent` rejects a negative value up front with `kael::ValueError` "integers to negative integer powers are not allowed", which matches PyTorch's behavior for scalar exponents. A tensor exponent is not checked, because PyTorch does not check it either.

## Elementwise unary ops

`src/ops/unary.cpp` is a list of one-line wrappers. Two templates choose the dtype:

- `unary_kernel` keeps the input dtype: `neg` (Bool throws), `abs`, `sign`, `relu`.
- `floating_unary_kernel` keeps floating dtypes and turns integer and Bool inputs into F32: `exp`, `log`, `log1p`, `expm1`, `sqrt`, `rsqrt`, `sin`, `cos`, `tanh`, `sigmoid`, `gelu`.

`exp(ones({3}, DType::I64))` therefore returns float32 `[2.7183, 2.7183, 2.7183]`, as in PyTorch. Returning I64 would truncate to 2.

`log1p(x)` and `expm1(x)` compute `log(1 + x)` and `exp(x) - 1` accurately for small `x`. For `x = 1e-10` in float32, `1 + x` rounds to exactly 1, so `log(1 + x)` returns 0, while `log1p(x)` returns `1e-10`.

`gelu(t, approximate)` picks `functions::GeluExact` (the `erf` form) or `functions::GeluTanh`. The golden tests register them as `gelu` and `gelu_tanh`.

`clamp(t, min, max)` promotes the dtype with both scalars, so clamping an I64 tensor to `0.5` returns F32, then runs `functions::Clamp<T>` through `detail::clamp_into`, which builds an iterator with the given output and dispatches on its dtype. `Clamp` returns NaN when either bound is NaN.

## In-place ops

An **in-place op** writes its result into the existing storage of `self` instead of allocating a new tensor. In-place ops write into the existing layout of `self`, whatever its strides. `binary_inplace` in `src/ops/inplace.cpp` runs every check before writing:

```cpp
template <typename F>
Tensor& binary_inplace(Tensor& self, const Tensor& other, DType result, const char* op, F f) {
    check_grad_rules(self, &other, op);
    check_writable(self, op);
    check_result_fits(result, self.dtype(), op);
    binary_kernel_out(self, self, without_partial_aliasing(self, other), f);
    self.impl()->storage->bump_version();
    return self;
}
```

| Check | Rule | Example that throws |
|-------|------|---------------------|
| `check_grad_rules` | with grad mode on, `self` must not be a leaf that requires grad, must not have history, and the source must not require grad | `w.add_(1)` on a parameter outside `NoGradGuard` |
| `check_writable` | `self` must not have internal overlap | `x.expand({2, 3}).add_(1)` |
| `check_result_fits` | the category of the result dtype must not exceed the category of `self` | `int_tensor.add_(0.5)`, because I64 + F32 scalar gives F32 |
| `prepare_output` (iterator) | `self` does not broadcast | `zeros({3}).add_(ones({2, 3}))` |

**Internal overlap** means two positions of `self` share one memory element, as after `expand`. Writing `x + 1` to such a view would add 1 twice to the shared element, or once, depending on the order. kael refuses with a message that suggests `clone()`.

The dtype of `self` never changes. A result in a lower or equal category is converted on store: `float_tensor.add_(int_tensor)` is fine, and so is `float32_tensor.add_(float64_tensor)`, which rounds to float32 (`InPlace.DtypeNeverChanges`). `div_` on an I64 tensor fails `check_result_fits` because `div_dtype` gives F32. The check uses categories rather than the full order so that the float32 and float64 case works, as it does in PyTorch.

`without_partial_aliasing` clones `other` when it shares storage with `self` in a different layout ([03. Views](@/book/03-views.md)). `x.add_(x)` has the same layout and needs no clone. `x.add_(x.T())` on a square matrix does: the loop would otherwise read elements it has already overwritten. `bump_version` increments a counter on the storage, so autograd can detect that a tensor saved for backward was modified afterwards.

The other in-place ops follow the same pattern:

- `clamp_(min, max)` runs the same three checks and calls `detail::clamp_into(*this, *this, min, max)`.
- `copy_(source)` moves the source to `self`'s device, removes partial aliasing, broadcasts it with `expand(shape())` and calls `convert_strided`. It converts dtypes as part of the copy.
- `fill_(value)` is `copy_(scalar(value, dtype()))`; `copy_` broadcasts the 0-d source with `expand`. `zero_()` is `fill_(0)`.
- `masked_fill_(mask, value)` requires a Bool mask and runs `functions::MaskedFill<T>` over `self` and the mask converted to `self`'s dtype.
- `scatter_add_` and `index_add_` are covered under indexing below.
- `+=`, `-=`, `*=` and `/=` call `add_`, `sub_`, `mul_` and `div_` (`CompoundOperators.CallTheInPlaceVersion`).

## Reductions

### Semantics

All reductions except `argmax` and `argmin` take `const Dims& dims = {}` and `bool keepdim = false`:

- An empty `dims` reduces every dimension.
- Negative dims wrap: `-1` is the last dimension.
- A repeated dim throws, after wrapping: `sum(t, {0, 0})` and `sum(t, {0, -2})` on a 2-d tensor both fail with "dimension ... appears more than once in reduction".
- A 0-d tensor accepts dim `0` and `-1`, matching PyTorch (`Reduce.ScalarsAcceptDimZero`).
- With `keepdim = false`, reduced dimensions are removed. With `keepdim = true`, they stay as size 1.

`argmax` and `argmin` reduce one dimension (`argmax(t, dim, keepdim)`) or the flattened tensor (`argmax(t)`); they do not take a list.

Result dtypes, empty-input behavior and NaN handling, as implemented:

| Op | Input dtypes | Result dtype | Empty reduced dimension | NaN in input |
|----|--------------|--------------|-------------------------|--------------|
| `sum` | all | float keeps its dtype; I64 and Bool give I64 | 0 | NaN |
| `prod` | all | same as `sum` | 1 | NaN |
| `mean` | floating only | input dtype | NaN (0 / 0) | NaN |
| `var`, `std` | floating only | input dtype | NaN | NaN |
| `norm` | floating only | input dtype | 0 | NaN |
| `max`, `min` | all | input dtype | throws | NaN propagates |
| `argmax`, `argmin` | all | I64 | throws | index of the first NaN |
| `all`, `any` | all | Bool | `all` gives true, `any` gives false | NaN counts as nonzero |

The empty results are the identities of each operation: adding zero terms gives 0, multiplying zero factors gives 1, "every element is true" holds vacuously. `max` has no identity in the finite numbers, so it throws instead of inventing `-inf`. `mean` of an integer tensor throws `kael::TypeError` (`Reduce.MeanOnlyFloats`), as in PyTorch; call `.to(DType::F32)` first.

The NaN rules match PyTorch. `max` and `min` use `detail::nan_aware_max` and `detail::nan_aware_min` from `src/internal/nan_aware.h`: if either argument is NaN, return it. A plain `std::max(a, b)` returns `a` when `a < b` is false, so the result would depend on where the NaN sits:

```text
std::max fold over [1, NaN, 3]:  max(1, NaN) = 1,   max(1, 3) = 3      -> 3
std::max fold over [NaN, 1, 3]:  max(NaN, 1) = NaN, max(NaN, 3) = NaN  -> NaN
nan_aware_max fold, any order:                                         -> NaN
```

`argmax` uses `best_position`, which jumps to the first NaN and stops searching:

```cpp
template <typename T, typename Chunk, typename Better>
std::int64_t best_position(Chunk chunk, std::int64_t size, Better better) {
    std::int64_t best = 0;
    for (std::int64_t i = 1; i < size; ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            if (std::isnan(chunk[best])) {
                break;
            }
            if (std::isnan(chunk[i])) {
                best = i;
                continue;
            }
        }
        if (better(chunk[i], chunk[best])) {
            best = i;
        }
    }
    return best;
}
```

`better` is `std::greater` for `argmax` and `std::less` for `argmin`. Strict comparison means ties keep the first index: `argmax([1, 3, 3])` is 1 (`Reduce.ArgmaxTiesTakeTheFirst`). For `[1, NaN, 3, NaN]`, the loop sets `best = 1` at `i = 1`, then breaks at `i = 2` because `chunk[1]` is NaN: the result is 1. `Chunk` is a template parameter so the same function works on a raw pointer and on the `StridedChunk` described below.

`max` of a dimension with size 0 throws, for example "max of an empty dimension in shape [0] has no result" for `max(zeros({0}))`. Reducing a non-empty dimension of a tensor that is empty elsewhere is fine: `max(zeros({0, 3}), {1})` has shape `[0]`. There are no outputs, so there is nothing to throw about.

### Variance

`var(t, dims, keepdim, correction)` computes

$$
\operatorname{var}(x) = \frac{1}{\max(N - c, 0)} \sum_{i=1}^{N} (x_i - \bar{x})^2
$$

where $N$ is the number of reduced elements, $\bar{x}$ their mean and $c$ the correction. The default `correction = 1` gives the unbiased estimator (Bessel's correction); `correction = 0` gives the population variance. With one element and correction 1 the divisor is 0 and the result is NaN, as in PyTorch (`Reduce.VarWithTooFewElementsIsNan`). `std` is `sqrt(var(...))`.

For `[1, 2, 3, 4]`: the mean is 2.5, the squared deviations are `2.25, 0.25, 0.25, 2.25`, their sum is 5. With correction 1 the variance is `5 / 3 = 1.6667`; with correction 0 it is `5 / 4 = 1.25`.

### Planning a reduction

Every CPU reduction starts by turning "reduce dims D of tensor t" into "for each of `output_count` outputs, reduce a chunk of `chunk_size` values". `plan_reduction` in `src/ops/reduce.cpp` does this with a permute:

1. Mark the reduced dimensions (`reduced_positions`), checking for repeats.
2. Permute the tensor so kept dimensions come first and reduced dimensions come last.
3. If the permuted tensor is contiguous, each output's chunk is a contiguous run. If it is not and exactly one dimension is reduced on the CPU, read the chunk with a stride (`StridedChunk`) instead of copying. Otherwise call `contiguous()`, which copies.

The result is a `Reduction` struct: the data tensor, `chunk_size`, `output_count`, `output_shape`, and for the strided plan `chunk_stride`, `kept_sizes` and `kept_strides`.

#### Worked example: `sum(t, {1})` on `[2, 3, 4]`

`t` is contiguous with strides `[12, 4, 1]`.

```text
kept_first     [0, 2]   reduced_last [1]
permute        [0, 2, 1]  -> shape [2, 4, 3], strides [12, 1, 4]   (not contiguous)
one reduced dim, CPU      -> strided plan, no copy
  chunk_size   3          (the reduced size)
  chunk_stride 4          (stride of the reduced dim)
  output_count 8          (2 * 4)
  kept_sizes   [2, 4]   kept_strides [12, 1]
output shape   [2, 4]   (or [2, 1, 4] with keepdim)
```

Output 5 sits at kept index `[1, 1]` (`5 = 1 * 4 + 1`), so `kept_offset` computes its chunk start as `1 * 12 + 1 * 1 = 13`. The chunk reads elements 13, 17 and 21, which are `t[1, 0, 1]`, `t[1, 1, 1]` and `t[1, 2, 1]`:

```text
storage index:  0  1  2  3 | 4  5  6  7 | 8  9 10 11 || 12 13 14 15 | 16 17 18 19 | 20 21 22 23
                  t[0, 0, :]   t[0, 1, :]   t[0, 2, :]     t[1, 0, :]    t[1, 1, :]    t[1, 2, :]
output 5 reads:                                                ^             ^             ^
                                                              13            17            21
```

Two other cases on the same tensor:

| Reduced dims | Permutation | Contiguous after permute | Plan |
|--------------|-------------|--------------------------|------|
| `{2}` | `[0, 1, 2]` | yes | contiguous rows of 4, no copy |
| `{1}` | `[0, 2, 1]` | no | strided, one reduced dim, no copy |
| `{0, 2}` | `[1, 0, 2]` | no | two reduced dims: `contiguous()` copies |

The strided path exists because reducing one non-last dimension, for example `sum(x, {0})` over a batch, is common and a copy would double memory traffic. `ReduceStrided.SingleDimOfNonContiguousViewsMatchesACopy` checks that the strided path gives exactly the result of reducing a contiguous copy. The device path never takes the strided plan, because `device::reduce` expects contiguous chunks.

### Accumulation and reproducibility

`reduce_chunks` runs one reduction per output, in parallel over outputs with `parallel_for`. Within one chunk, the sum-like reductions (`sum`, `prod`, `mean`, `var`, `norm`) use two more levels: lanes inside a block, and blocks inside a chunk.

From `src/ops/reduce.cpp`:

```cpp
// Independent accumulators let the compiler vectorize without reordering a single accumulator.
template <typename Value, typename Term, typename Combine>
Value fold_lanes(std::int64_t first, std::int64_t last, Value identity, Term term,
                 Combine combine) {
    std::array<Value, kLanes> lanes;
    lanes.fill(identity);
    std::int64_t i = first;
    for (; i + kLanes <= last; i += kLanes) {
        for (std::int64_t lane = 0; lane < kLanes; ++lane) {
            auto& accumulator = lanes[static_cast<std::size_t>(lane)];
            accumulator = combine(accumulator, term(i + lane));
        }
    }
    Value total = identity;
    for (const Value lane : lanes) {
        total = combine(total, lane);
    }
    for (; i < last; ++i) {
        total = combine(total, term(i));
    }
    return total;
}
```

`fold_lanes` keeps `kLanes = 8` independent accumulators. Floating-point addition is not associative, so the compiler may not reorder a single running sum into SIMD lanes on its own: that would change the result. Eight explicit lanes give it eight independent chains that map onto vector registers. For a chunk of 20 elements:

```text
elements 0..15   lane k accumulates elements k and k + 8      (two full rounds of 8)
lanes combined   total = lane0 + lane1 + ... + lane7
tail 16..19      total += x16, x17, x18, x19
```

`reduce_in_blocks` splits chunks longer than `kReductionBlock = 16384` elements into fixed blocks, reduces blocks in parallel, then combines the partial results pairwise in a tree. The comment in the code states the reason: "The blocks do not depend on the thread count, so floating-point results are reproducible." For a chunk of 40000 elements:

```text
blocks     [0, 16384)   [16384, 32768)   [32768, 40000)
partials       p0            p1               p2
tree       width 3:  p0 + p1 -> q0,  p2 carried -> q1
           width 2:  q0 + q1 -> total
```

Which thread computes which block changes with the thread count. The block boundaries and the tree do not, so the same input gives bit-identical sums on 1 or 16 threads (`ParallelReduce.ResultsDoNotDependOnThreadCount`). Pairwise combination also bounds the rounding error better than a long sequential sum.

Floating inputs accumulate in `double` and integer inputs in `std::int64_t` (the `Accumulator<In>` alias). Float32 results are cast back at the end. Summing ten million copies of `0.1f` stays within 1 of `1e6` (`Reduce.SumOfManyFloatsDoesNotDrift`); a running float32 sum would drift far more, because once the total passes a few hundred thousand, each `0.1` added is rounded to a multiple of the spacing between neighboring float32 values. Integer sums and products go through `wrapping_add` and `wrapping_multiply`, so overflow wraps (`IntegerOverflow.SumAndProductWrapAround`).

`var` is two-pass: compute the mean in `double`, then sum squared deviations from it, both with `sum_in_blocks`. The one-pass formula `E[x^2] - E[x]^2` cancels catastrophically when the mean is large compared to the spread. For `[1e8 + 1, 1e8 - 1]`, `E[x^2]` is about `1e16`, where the spacing between neighboring doubles is 2, so the true variance of 1 is lost in rounding. The two-pass form subtracts the mean first and squares `1` and `-1`. `Reduce.VarDoesNotLosePrecisionWithLargeMean` checks `var([1e8 + 1, 1e8 - 1], correction 0)` equals 1 to within `1e-9`.

`max`, `min`, `argmax`, `argmin`, `all` and `any` are order-independent and use a plain loop per chunk. `all` and `any` stop at the first element that decides the answer.

On CUDA, `sum`, `max` and `min` call `device::reduce` on the contiguous plan; `mean`, `var` and `norm` are composed from `device_sum` and elementwise ops; `prod`, `argmax`, `argmin`, `all` and `any` copy to the host (`on_host`) and copy the result back. [09. CUDA backend](@/book/09-cuda-backend.md) covers the device kernels.

## Matrix multiplication

### Shape rules

`matmul(a, b)` follows NumPy and PyTorch:

1. A 1-d `a` is treated as a row `[1, n]`; a 1-d `b` as a column `[n, 1]`. The added dimension is removed from the result.
2. The last dimension of `a` must equal the second-to-last of `b`.
3. All dimensions before the last two are **batch dimensions** and broadcast with `broadcast_shapes`.

Every row below is a golden case in `MATMUL_PATTERNS`:

| `a` | `b` | Result |
|-----|-----|--------|
| `[4]` | `[4]` | `[]` |
| `[2, 3]` | `[3, 4]` | `[2, 4]` |
| `[3]` | `[3, 2]` | `[2]` |
| `[2, 3]` | `[3]` | `[2]` |
| `[5, 2, 3]` | `[5, 3, 2]` | `[5, 2, 2]` |
| `[2, 1, 3, 4]` | `[5, 4, 2]` | `[2, 5, 3, 2]` |
| `[3]` | `[5, 3, 2]` | `[5, 2]` |
| `[5, 2, 3]` | `[3]` | `[5, 2]` |
| `[2, 0]` | `[0, 3]` | `[2, 3]` of zeros |
| `[0, 2, 3]` | `[3, 2]` | `[0, 2, 2]` |

The `[2, 0] @ [0, 3]` row is worth a second look: an empty inner dimension gives a non-empty output, because each output element is a sum of zero products, which is 0. That is why the output starts as `zeros` and not `empty`.

From `matmul_forward` in `src/ops/linalg.cpp`:

```cpp
const bool a_is_vector = a.dim() == 1;
const bool b_is_vector = b.dim() == 1;
const Tensor left = a_is_vector ? a.unsqueeze(0) : a;
const Tensor right = b_is_vector ? b.unsqueeze(1) : b;

KAEL_CHECK_VALUE(left.size(-1) == right.size(-2),
                 "matmul shapes " << to_string(a.shape()) << " and " << to_string(b.shape())
                                  << " do not align");

const Shape batch = broadcast_shapes(batch_shape_of(left), batch_shape_of(right));
Tensor out =
    multiply_batches(left, right, batch, {left.size(-2), left.size(-1), right.size(-1)});
if (b_is_vector) {
    out = out.squeeze(-1);
}
return a_is_vector ? out.squeeze(b_is_vector ? -1 : -2) : out;
```

Before this, `matmul_forward` requires both inputs to have at least one dimension and to be on the same device. The public `matmul` wraps it in `autograd::without_grad` and records one `matmul` node.

Trace for `a` of shape `[3]` and `b` of shape `[5, 3, 2]`: `left` is `[1, 3]`, `right` is `[5, 3, 2]`, the batch is `broadcast_shapes([], [5]) = [5]`, the product is `[5, 1, 2]`, and `squeeze(-2)` gives `[5, 2]`. For `[4] @ [4]`, the product is `[1, 1]`, the first `squeeze(-1)` gives `[1]` and the second gives `[]`.

### Batching

`multiply_batches` reduces every case to "multiply `B` pairs of contiguous row-major matrices":

```cpp
Tensor as_batch_of_matrices(const Tensor& matrix, const Shape& batch, std::int64_t rows,
                            std::int64_t columns) {
    return matrix.expand(with_matrix_dims(batch, rows, columns))
        .reshape({numel(batch), rows, columns});
}
```

`expand` broadcasts the batch dimensions with stride 0 ([03. Views](@/book/03-views.md)). `reshape` then flattens the batch into one dimension. A contiguous input passes through without a copy. A transposed, sliced or broadcast input is copied into a contiguous buffer by `reshape`, because no single stride can describe it after flattening.

Trace for `[2, 1, 3, 4] @ [5, 4, 2]`:

```text
batch      broadcast_shapes([2, 1], [5]) = [2, 5]      B = 10
a          [2, 1, 3, 4] --expand--> [2, 5, 3, 4]   (stride 0 on dim 1)
           --reshape--> [10, 3, 4]                  copy: 10 matrices of 3x4
b          [5, 4, 2]    --expand--> [2, 5, 4, 2]   (stride 0 on dim 0)
           --reshape--> [10, 4, 2]                  copy: 10 matrices of 4x2
out        zeros [2, 5, 3, 2]; matrix i of the batch writes out[i / 5, i % 5]
```

This trades memory for one simple kernel that only handles packed row-major matrices. A production library passes strides to BLAS or uses a batched GEMM with stride 0 instead. Both inputs are first converted to `promote_types(a.dtype(), b.dtype())`; Bool throws "matmul does not support dtype bool".

The output starts as `zeros`. The CPU path then runs `parallel_for` over the batch with a grain chosen so each task does at least `kWorkPerParallelTask` (2^21) multiply-adds: the grain is `max(1, 2^21 / (rows * inner * columns))`. Each matrix product calls:

- `matmul_versions::multiply` for `float` and `double`, which picks BLAS when available and otherwise one of the hand-written kernels by problem size ([08. CPU performance](@/book/08-cpu-performance.md));
- `multiply_exact` for `std::int64_t`: a plain i-k-j loop with `wrapping_add` and `wrapping_multiply`, so overflow wraps around instead of being undefined. Integer products must be exact, and BLAS has no integer routines.

On CUDA, `multiply_batches` calls `device::matmul` with the batch count and the three sizes.

### dot, outer, mm and bmm

`dot`, `mm` and `bmm` check the ranks (1 and 1, 2 and 2, 3 and 3) with `check_ranks` and call `matmul`. They exist for code ported from PyTorch, where they reject shapes `matmul` would accept (`StrictAliases`). `outer(a, b)` checks that both are 1-d and computes `a.unsqueeze(1) * b.unsqueeze(0)`: a broadcasting multiply of `[n, 1]` by `[1, m]`, not a matrix product.

## Indexing ops

| Op | Implementation |
|----|----------------|
| `cat(tensors, dim)` | checks every shape matches outside `dim`, promotes all dtypes with `promote_types`, allocates the output and writes each input into `out.narrow(dim, start, length)` with `copy_` |
| `stack(tensors, dim)` | checks all shapes are equal, `unsqueeze(dim)` each input, then `cat` |
| `where(condition, a, b)` | requires a Bool condition; `a` and `b` are `Operand`s; runs `functions::Where` through `ternary_kernel`, so all three broadcast |
| `masked_fill(t, mask, value)` | `t.expand(broadcast_shapes(t.shape(), mask.shape())).clone()`, then `masked_fill_` on the clone |
| `index_select(t, dim, indices)` | 1-d I64 indices; picks whole slices along `dim` |
| `gather(t, dim, indices)` | `out[i][j] = t[indices[i][j]][j]` for `dim = 0`, and so on |
| `scatter_add_(dim, indices, source)` | the inverse of `gather`: `self[indices[i][j]][j] += source[i][j]` for `dim = 0` |
| `index_add_(dim, indices, source)` | reshapes the 1-d indices to broadcast along `dim`, expands them to `source.shape()`, and calls `scatter_add_` |

`cat` reuses views and `copy_` instead of computing output offsets by hand. Each `narrow` is a window into the output, and `copy_` handles any input layout and dtype conversion:

```text
cat([a: [2, 3], b: [1, 3]], 0)

out: [3, 3]
  out.narrow(0, 0, 2)  <-- copy_(a)     rows 0..1
  out.narrow(0, 2, 1)  <-- copy_(b)     row 2
```

`stack([a, b], 0)` on two `[2, 3]` tensors unsqueezes each to `[1, 2, 3]` and concatenates to `[2, 2, 3]`. `stack` wraps `dim` against `dim() + 1`, because the new dimension can be placed after the last one.

`where` has one subtlety: `ternary_kernel` casts all inputs, including the Bool condition, to the result dtype of the three. `functions::Where` then tests `keep != T{0}`. Because `result_dtype` includes the condition, which is Bool and therefore the lowest category, the condition never changes the result dtype. `where([true, false, true], int64 [1, 2, 3], 0.5)` gives float32 `[1, 0.5, 3]`: the dimensioned group is I64, the scalar `0.5` became a 0-d F32 tensor, and category 2 beats category 1.

`masked_fill` clones an expanded copy so the output has the broadcast shape even when the mask is larger than `t` (the golden case `wider_mask`).

`index_select` has two paths. When `dim == 0`, both tensors are contiguous and `t` is not empty, on the CPU, each selected row is one `std::memcpy` of `nbytes() / shape[0]` bytes. Otherwise it copies `t.narrow(dim, source, 1)` into `out.narrow(dim, i, 1)` for each index. On CUDA it reshapes the indices and calls `gather`.

### gather and scatter_add_

`gather(t, dim, indices)` produces an output with the shape of `indices`. Each output position takes the value of `t` at the same position, except along `dim`, where the coordinate comes from `indices`. Worked example with `dim = 1`:

```text
t = [[0, 1, 2],          indices = [[2, 0],
     [3, 4, 5]]                     [1, 1]]

out[0][0] = t[0][indices[0][0]] = t[0][2] = 2
out[0][1] = t[0][indices[0][1]] = t[0][0] = 0
out[1][0] = t[1][indices[1][0]] = t[1][1] = 4
out[1][1] = t[1][indices[1][1]] = t[1][1] = 4

out = [[2, 0], [4, 4]]
```

`gather` and `scatter_add_` walk the index tensor with `detail::IndexWalker` from `src/internal/index_checks.h`. The walker is an odometer over the index shape, like the one in the iterator, that tracks three offsets at once: into the index tensor, into the source (for `scatter_add_`), and into the target, where the target offset skips the indexed axis. `target_offset(index)` then adds `index * target_strides[axis]`. For the example above, `t` has strides `[3, 1]` and the axis is 1:

| Index position | Target offset without axis | Index value | `target_offset` | Value |
|----------------|----------------------------|-------------|-----------------|-------|
| `[0, 0]` | 0 | 2 | 0 + 2 * 1 = 2 | 2 |
| `[0, 1]` | 0 | 0 | 0 | 0 |
| `[1, 0]` | 3 | 1 | 3 + 1 = 4 | 4 |
| `[1, 1]` | 3 | 1 | 4 | 4 |

Every index goes through `detail::checked_index`, which throws `kael::IndexError` (through `KAEL_CHECK_INDEX`) for values outside `[0, size)`. kael does not accept negative indices in these ops; PyTorch's `gather` does not either. The index shape may be smaller than `t` in the non-indexed dimensions (`check_gather_shapes`, golden case `smaller_index`).

Compiled checks:

```text
gather(arange(6).reshape({2, 3}), 1, [[2, 0], [1, 1]])  -> [[2, 0], [4, 4]]
zeros({3}).scatter_add_(0, [0, 2, 0, 0], [1, 2, 3, 4])   -> [8, 0, 2]
index_select(arange(6).reshape({3, 2}), 0, [2, 0, 2])    -> [[4, 5], [0, 1], [4, 5]]
```

`scatter_add_` sums repeated indices: `0` appears three times above, so position 0 receives `1 + 3 + 4 = 8`, and position 2 receives 2. This is what makes it the backward of `gather`: a value read twice in the forward pass receives the sum of both gradients. `scatter_add_` refuses a source or index tensor that shares storage with `self`, because the loop reads and writes in one pass. It also requires `source` to have the dtype of `self`.

`index_add_(dim, indices, source)` turns 1-d indices into a `scatter_add_` index tensor. For `self` of shape `[3, 2]`, `dim = 0`, 4 indices and a `[4, 2]` source, the indices are reshaped to `[4, 1]` and expanded to `[4, 2]`, so every element of source row `i` goes to row `indices[i]` of `self`.

## Sequence ops

All sequence ops move the target dimension last on the host, run a per-row loop on contiguous data, and move the result back:

```cpp
Tensor moved_to_last_on_host(const Tensor& t, std::int64_t dim) {
    return t.detach().cpu().transpose(dim, -1).contiguous();
}
```

`moved_back` transposes again, makes the result contiguous and moves it to the original device. The cost is two copies; the benefit is that every loop below works on contiguous rows of the last dimension.

| Op | Rule |
|----|------|
| `cumsum(t, dim)` | running sum along `dim`; integer and Bool inputs give I64 |
| `sort(t, dim, descending)` | `std::stable_sort` on index arrays; returns `SortResult{values, indices}` |
| `topk(t, k, dim, largest)` | `std::partial_sort` of the first `k` positions when `k` is smaller than the size, `std::stable_sort` when it equals the size; `k` must be in `[0, size]` |
| `tril(t, diagonal)`, `triu(t, diagonal)` | build a Bool mask for the last two dims and call `where(mask, t, zeros_like(t))` |

`cumsum([1, 2, 3, 4])` is `[1, 3, 6, 10]`. Integer running sums go through `wrapping_add` (`IntegerOverflow.CumsumWrapsAround`).

`sort` and `topk` compute only the index tensor, then fill `values` with `gather(t, dim, indices)`. Gradients therefore flow through the existing `gather` backward without a dedicated sort backward. The index arrays are sorted with a comparator that reads the row values, so the values themselves are never moved.

NaN ordering follows PyTorch: NaN is greater than every number. `ordered_before(a, b)` returns true for `a` finite and `b` NaN. Ascending order puts NaN last and descending order puts it first. Compiled check: `sort([3, NaN, 1, 2])` gives values `[1, 2, 3, NaN]` and indices `[2, 3, 0, 1]`.

`stable_sort` keeps equal elements in input order for `sort`. `topk` with `k` smaller than the size uses `partial_sort`, which is not stable, so the order of tied values in its output is not specified.

`tril(t, 0)` keeps positions where `j - i <= 0`, the lower triangle including the diagonal. For a `[3, 3]` input:

```text
tril mask (diagonal 0)    triu mask (diagonal 1)
1 0 0                     0 1 1
1 1 0                     0 0 1
1 1 1                     0 0 0
```

`triu(t, 1)` is the causal attention mask pattern used in [11. Transformer](@/book/11-transformer.md): it marks the positions a token must not attend to.

## Softmax, log-softmax and logsumexp

### The overflow problem

Softmax turns a vector of scores into probabilities:

$$
\operatorname{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

Evaluated literally in float32, `exp(89)` already overflows to `inf`, and `inf / inf` is NaN. The fix uses the identity

$$
\operatorname{softmax}(x) = \operatorname{softmax}(x - c)
$$

for any constant $c$, which holds because $e^{-c}$ cancels between numerator and denominator. Choosing $c = \max(x)$ makes the largest exponent `exp(0) = 1` and every other exponent at most 1, so nothing overflows and the denominator is at least 1.

The same shift gives the two related functions:

$$
\log \operatorname{softmax}(x)_i = (x_i - c) - \log \sum_j e^{x_j - c}
\qquad
\operatorname{logsumexp}(x) = c + \log \sum_j e^{x_j - c}
$$

Worked example, `x = [1, 2, 3]`:

```text
max                3
x - max            [-2, -1, 0]
exp                [0.1353, 0.3679, 1.0000]
sum                1.5032
softmax            [0.0900, 0.2447, 0.6652]
log(sum)           0.4076
log_softmax        x - max - log(sum) = [-2.4076, -1.4076, -0.4076]
logsumexp          max + log(sum)     = 3.4076
```

With `x = [1000, 1000, -1000, 999]`, kael returns softmax `[0.4223, 0.4223, 0.0000, 0.1554]` and logsumexp `1000.8620`. Both values were printed by a compiled check; the naive formula returns NaN. The shift makes the exponents `[1, 1, 0, 0.3679]`, which sum to `2.3679`, and `1000 + log(2.3679) = 1000.8620`.

`log_softmax` is computed as `shifted - log(sum(exp(shifted)))`, not as `log(softmax(x))`. For a very negative entry, `softmax` underflows to 0 and `log(0)` is `-inf`, while the shifted form keeps the exact value. `LogSoftmax.IsExactWhereLogOfSoftmaxIsNot` uses `x = [-1000, 0]`:

```text
log(softmax(x))[0]   softmax[0] = exp(-1000) / (exp(-1000) + 1) = 0 in float32  -> log(0) = -inf
log_softmax(x)[0]    (-1000 - 0) - log(exp(-1000) + exp(0)) = -1000 - log(1)    -> -1000
```

Cross-entropy losses need exactly this value, which is why classifiers use `log_softmax` and not `log` of `softmax`.

### Two implementations

`softmax` and `log_softmax` in `src/ops/nn.cpp` have a fused row kernel and a composed fallback:

```cpp
Tensor softmax(const Tensor& t, std::int64_t dim) {
    Tensor out = autograd::without_grad([&] {
        const Tensor x = as_floating(t);
        if (x.numel() == 0) {
            return x.clone();
        }
        if (is_last_dim_of_contiguous(x, dim)) {
            return dispatch_floating(x.dtype(), "softmax", [&]<typename T>() {
                return map_rows<T>(x, x.size(-1), kExponentialCost, softmax_row<T>);
            });
        }
        const Tensor exponentials = exp(shifted_by_max(x, dim));
        return exponentials / sum(exponentials, {dim}, true);
    });
    autograd::record_softmax(out, t, dim);
    return out;
}
```

- `as_floating` turns integer and Bool inputs into F32 (`Softmax.IntegersBecomeFloat32`).
- The fused path runs when the input is a contiguous CPU tensor with at least one dimension and `dim` is the last dimension, which is the case for attention scores and classifier logits. `softmax_row` makes one pass for the maximum (with `nan_aware_max`), one pass that writes `exp(x - max)` and accumulates the total in `double`, and one pass that divides. `map_rows` parallelizes over rows, with a grain that accounts for the cost of `exp` (`kExponentialCost = 16`). [08. CPU performance](@/book/08-cpu-performance.md) measures the gain.
- The composed path handles every other layout with `max(..., keepdim = true)`, broadcasting subtraction, `exp`, `sum(..., keepdim = true)` and division. `keepdim` is what makes the broadcast line up.

The composed path allocates four intermediate tensors and reads the input several times. The fused path reads each row three times from cache and allocates only the output. Both paths compute the same formula, and `FusedKernels.LastDimensionMatchesTheComposedPath` and `FusedKernels.HandlesNonFiniteValuesLikeTheComposedPath` check that they agree. These two tests are parameterized over float32 and float64, so gtest lists them as `Floating/FusedKernels.*`.

### Non-finite inputs

`logsumexp` uses a variant of the shift:

```cpp
const Tensor largest = finite_or_zero(max(x, {dim}, true));
const Tensor shifted_total = log(sum(exp(x - largest), {dim}, true)) + largest;
```

If every entry is `-inf`, the maximum is `-inf` and `x - max` would be `-inf - (-inf) = NaN`. If an entry is `+inf`, `x - max` would be `inf - inf = NaN` too. `finite_or_zero` replaces an infinite maximum with 0, through `where(abs(t) == infinity, 0, t)`, so the shift is skipped exactly when it would create NaN:

| Input | Result | Matches PyTorch |
|-------|--------|-----------------|
| `[-inf, -inf]` | `-inf` | yes (`all_minus_infinity`) |
| `[inf, 1]` | `inf` | yes (`plus_infinity`) |
| `[-inf, 0, 1]` | `1.3133` | yes (`some_minus_infinity`) |
| empty | `log(sum(empty)) = log(0) = -inf` | yes |

For `[-inf, -inf]`: the shift is 0, `exp(-inf) = 0`, the sum is 0, and `log(0) + 0 = -inf`. For `[-inf, 0, 1]`: the maximum 1 is finite, so the shift applies and the result is `log(exp(0) + exp(1)) = log(3.7183) = 1.3133`.

`softmax` and `log_softmax` do not use `finite_or_zero`. For `[inf, 1]` both return `[NaN, NaN]`, which is also what PyTorch returns (`LogSoftmax.InfinityGivesNanLikePyTorch` and the golden `plus_infinity` cases). An all `-inf` row gives NaN for the same reason. Matching PyTorch here is deliberate: a NaN tells the user the row has no valid probability distribution, which is more useful than a made-up uniform answer.

`cross_entropy_losses(logits, targets)` and `standardize(t, normalized_dims, eps)` live in the same file. They are fused row kernels used by the loss and layer norm modules and are covered in [08. CPU performance](@/book/08-cpu-performance.md).

## The golden test pipeline

Hand-written expected values catch the cases the author thought of. The golden pipeline checks kael against PyTorch on a systematic grid of shapes, layouts, dtypes and special values. A **golden file** is a stored expected output produced by a trusted reference, here PyTorch.

```text
scripts/generate/golden.py  (PyTorch + NumPy)
   |
   |-- np.save base inputs and the contiguous output --> tests/golden/OP/CASE_*.npy
   |-- one line per case ------------------------------> tests/golden/manifest.txt
                                                                |
tests/test_golden.cpp  Golden.AllCasesMatchPyTorch  <-----------+
   |  reads the inputs with tests/support/npy.cpp (read_npy)
   |  applies the views in C++, calls the kael op
   v
tests/support/allclose.h  compares shape, dtype and values
```

### Generating cases

`scripts/generate/golden.py` describes each input with `spec(shape, dtype, kind, view, values)`. `kind` selects the distribution in `make_base`: `normal` uses `torch.randn`, `positive` uses `rand * 3 + 0.01` (for `log`, `sqrt` and `pow` bases), I64 uses `randint(-5, 6)`, Bool uses `rand > 0.5`, and explicit `values` override all of them. `view` is a string applied after creation: `transpose:0:1`, `permute:1:2:0`, `slice:dim:start:end:step` or `expand:3:4`.

Cases are grouped into reusable pattern lists, and `add_cases(op, compute, patterns)` attaches a list to an op:

- `BROADCAST_PATTERNS` covers same shape, rank broadcasting, size-1 broadcasting, a transposed input, a stepped slice, an expanded input, a permuted broadcast, scalar and tensor, scalar and scalar, an empty input and float64.
- `PROMOTION_PATTERNS` covers the 0-d promotion rules and mixed dtypes.
- `SPECIAL_PATTERN` pairs every value in `SPECIAL_VALUES` (NaN, `inf`, `-inf`, `0`, `-0`, `1`, `-1`, `1e-10`, `1000`, `-1000`) with every other, 100 pairs in total.
- `BOOL_PATTERNS` adds Bool and Bool, and Bool and float32.
- Unary ops use `UNARY_PATTERNS`, or `POSITIVE_PATTERNS` for functions with a restricted domain, plus an integer input.
- In-place ops use `INPLACE_PATTERNS` (transposed and stepped `self`, broadcast `other`, mixed dtypes).
- Reductions use `REDUCTION_PATTERNS` (every dim, negative dims, two dims, keepdim, views, 0-d, empty, NaN), and `argmax` and `argmin` use `ARG_PATTERNS` with ties and NaN.
- The softmax family uses `NN_PATTERNS` plus the infinity cases `ALL_MINUS_INFINITY` and `PLUS_INFINITY`.

`write_case` does the work for one case:

```python
def write_case(manifest, op, name, inputs, args, compute):
    torch.manual_seed(zlib.crc32(f"{op}/{name}".encode()))
    directory = GOLDEN_DIR / op
    directory.mkdir(parents=True, exist_ok=True)

    viewed = []
    for input_name, input_spec in inputs.items():
        base = make_base(input_spec)
        np.save(directory / f"{name}_{input_name}.npy", base.numpy())
        viewed.append(apply_view(base, input_spec["view"]))

    expected = compute(*viewed, **args)
    np.save(directory / f"{name}_out.npy", expected.contiguous().numpy())
    # ... append the manifest line
```

Three choices make the data reproducible and portable:

- The seed is the CRC32 of `op/name`, so adding or removing a case does not change the random data of any other case. A global seed would shift every later case whenever one is inserted, and the diff of the golden directory would be unreadable.
- The file stores the base tensor before the view, as a plain contiguous `.npy`. The view is recorded in the manifest and applied again in C++. The C++ side therefore exercises its own `transpose`, `slice`, `permute` and `expand` on exactly the same data.
- The output is saved after `.contiguous()`, so the reader needs no stride support.

Each case becomes one manifest line:

```text
op=add case=transposed inputs=a,b view_a=transpose:0:1 view_b=none args=
op=sum case=keepdim_two inputs=x view_x=none args=dims:1/2,keepdim:1
op=clamp case=matrix inputs=x view_x=none args=min:-0.5,max:0.5
```

`encode_args` writes lists with `/` and booleans as `0` or `1`. For in-place ops the `compute` lambda calls the PyTorch in-place method on the view (`lambda a, b: a.add_(b)`), so the expected output is the mutated view.

Regenerate the data with the project virtual environment, which has PyTorch installed through the `test` extra in `pyproject.toml`:

```bash
venv/bin/python scripts/generate/golden.py
```

The script deletes `tests/golden/` first and rewrites everything, including the fixtures in `tests/golden/npy/`, `mlp/`, `nn/` and `optim/` that other test files read. Commit the regenerated directory together with the code change that needed it. CI does not run the generator; it reads the committed files.

> [!WARNING]
> Regenerating with a different PyTorch version can change expected values in the last bits, or change behavior on special values. Review the diff of `tests/golden/` before committing it, and do not regenerate as a way to make a failing case pass.

### Reading .npy files

The `.npy` format is a small header followed by raw array bytes:

```text
offset 0   \x93NUMPY                         6-byte magic
offset 6   0x01 0x00                         format version 1.0
offset 8   header length, 2 bytes LE         for example 0x76 0x00 = 118
offset 10  {'descr': '<f4', 'fortran_order': False, 'shape': (3, 4), }   padded with spaces
then       3 * 4 * 4 = 48 bytes of float32 data in C order
```

`tests/support/npy.cpp` implements `kael::test::read_npy(path)`, a minimal reader for that format:

1. Check the 6-byte magic `\x93NUMPY` and require format version 1.
2. Read the 2-byte little-endian header length and the header text.
3. Extract `descr`, `fortran_order` and `shape` from the header with string search. `'<f4'`, `'<f8'`, `'<i8'` and `'|b1'` map to F32, F64, I64 and Bool. Fortran order and any other dtype throw.
4. Allocate `empty(shape, dtype)` and read `nbytes()` raw bytes into it. A short read throws "truncated npy file", except for empty arrays.

It relies on little-endian hosts, which covers every platform kael targets. The Bool mapping works because NumPy stores `bool` as one byte with value 0 or 1, the same as kael's `std::uint8_t`. A 0-d array has shape `()`, which `parse_shape` turns into an empty `Shape`.

### Comparing

`tests/support/allclose.h` provides `kael::test::allclose(actual, expected, options)`, a GoogleTest `AssertionResult`:

- Shapes must be equal and dtypes must be equal. A promotion bug fails even when the values match.
- Floating values pass when $|a - e| \le \text{atol} + \text{rtol} \cdot |e|$. `default_tolerance` gives `rtol = 1e-5, atol = 1e-6` for F32 and `rtol = 1e-12, atol = 1e-14` for F64. I64 and Bool must match exactly. A test can pass its own `Tolerance` in the options.
- Infinities must match exactly, including sign.
- NaN matches NaN only when `equal_nan` is set.
- The failure message names the first mismatching element by index and prints both values and the difference.

The relative term scales the tolerance with the magnitude of the expected value: for `e = 1000` in float32, the allowed difference is `1e-6 + 1e-5 * 1000 = 0.010001`. The absolute term keeps values near zero from requiring an impossible relative precision.

`tests/support/values.h` has `values<T>(t)`, which reads any tensor into a `std::vector<T>` in logical order. The hand-written unit tests use it.

### Running the cases

`tests/test_golden.cpp` contains one test, `Golden.AllCasesMatchPyTorch`. It reads the manifest line by line, parses the `key=value` fields, looks up the op name in a `std::map<std::string, GoldenOp>`, loads the inputs, applies the views, calls the op and compares:

```cpp
void run_case(const GoldenCase& golden) {
    const auto op = golden_ops().find(golden.op);
    ASSERT_NE(op, golden_ops().end()) << "no golden op registered for " << golden.op;

    const std::filesystem::path directory = kGoldenDir / golden.op;
    Inputs inputs;
    for (const std::string& input : golden.input_names) {
        const Tensor base = kael::test::read_npy(directory / (golden.name + "_" + input + ".npy"));
        inputs.push_back(apply_view(base, golden.fields.at("view_" + input)));
    }
    const Tensor expected = kael::test::read_npy(directory / (golden.name + "_out.npy"));
    const Tensor actual = op->second(inputs, Args(golden.fields.at("args")));
    EXPECT_TRUE(kael::test::allclose(actual, expected, {.equal_nan = true}));
}
```

`GoldenOp` is `std::function<Tensor(const Inputs&, const Args&)>`. Small adapters (`binary`, `unary`, `inplace`, `reduction`, `reduction_with_correction`, `arg_reduction`, `tensor_pair`) wrap kael functions so each registry entry is one line, for example `{"sum", reduction(kael::sum)}`. `Args` parses `dims:1/2,keepdim:1` into typed accessors such as `ints("dims")` and `flag("keepdim")`.

Each case runs under `SCOPED_TRACE(op + "/" + name)` and inside `try`/`catch`, so one throwing case reports its name with `ADD_FAILURE` and the loop continues. A failure reads like `add/transposed` followed by the first mismatching element. `KAEL_GOLDEN_DIR` is set by `tests/CMakeLists.txt` to the source tree path, so the test runs from any working directory.

To add a golden op:

1. Add an `add_cases(...)` call in `scripts/generate/golden.py`.
2. Regenerate the data.
3. Add one entry to `golden_ops()`.

The `ASSERT_NE` on the registry lookup fails loudly if step 3 is forgotten.

The same idea appears at the Python level: `python/tests/test_ops.py` compares the bindings against PyTorch directly with pytest (`test_binary_operators_match_torch_with_broadcast`, `test_reductions_match_torch`, `test_matmul_matches_torch`, `test_softmax_family_matches_torch` and others).

## Edge cases and pitfalls

- **Integer division.** `div` always returns a floating result, so integer `/` never truncates and never divides by zero in integer arithmetic. `floor_divide` throws on an integer zero divisor and handles `INT64_MIN // -1` by wrapping.
- **Integer overflow in `pow`.** `integer_power` multiplies in the unsigned type, so overflow wraps instead of being undefined.
- **Signed integer overflow elsewhere.** `add`, `sub`, `mul`, `neg`, `abs`, `sum`, `prod`, `matmul`, `cumsum` and `scatter_add_` go through `wrapping_add`, `wrapping_subtract` and `wrapping_multiply` from `src/internal/element_functions.h`, so `INT64_MAX + 1` is `INT64_MIN` in two's complement, as in PyTorch. On a real GPU, device `scatter_add_` wraps through an unsigned `atomicAdd`; the host emulation uses `wrapping_add` (`Cuda.IntegerScatterAddWrapsAround`).
- **Float to integer conversion.** `to(DType::I64)` saturates out-of-range values to `INT64_MIN` or `INT64_MAX` and maps NaN to 0 ([02. Memory model](@/book/02-memory-model.md)).
- **In-place dtype.** In-place ops never change `self.dtype()` and throw when the result needs a higher category (`InPlace.LosingCategoryThrows`).
- **NaN in reductions.** `max`, `min` and the softmax maximum use NaN-aware comparisons; `argmax` returns the first NaN; `all` and `any` treat NaN as true.
- **Empty reductions.** `sum` gives 0, `prod` gives 1, `mean` gives NaN, `max`, `min`, `argmax` and `argmin` throw for an empty reduced dimension.
- **Variance of one element.** Divides by 0 with the default correction and returns NaN.
- **Floating-point reproducibility.** Reductions split long chunks into fixed 16384-element blocks and combine them in a fixed tree, so results do not depend on the thread count. They can still differ in the last bits from PyTorch, which uses a different summation order; the golden tolerance absorbs that.
- **Overflow in softmax.** Always shifted by the maximum. `logsumexp` also guards infinite maxima with `finite_or_zero`.
- **Hidden copies in matmul.** Broadcast batch dimensions and non-contiguous inputs are materialized by `reshape` before the kernel. A `[1000, 64, 64] @ [64, 64]` product copies the right operand 1000 times.
- **Hidden copies in sequence ops.** `cumsum`, `sort` and `topk` move the target dimension last with a contiguous copy, unless it already is the last dimension of a contiguous CPU tensor.
- **Index validation.** `gather`, `scatter_add_`, `index_select` and `cross_entropy_losses` check every index and throw `kael::IndexError`. Negative indices are rejected, not wrapped.
- **Aliasing in `scatter_add_`.** A source or index tensor that shares storage with `self` throws.
- **Unstable `topk`.** Tied values may appear in any order when `k` is smaller than the size.

> [!NOTE]
> Several of these rules differ between PyTorch and NumPy, for example the dtype of `int64 + float32` and of integer true division, which NumPy makes float64 and PyTorch float32. kael follows PyTorch everywhere except Bool + Bool in `add`, and the golden tests encode that choice.

## How it is tested

| File | Suites | What they prove |
|------|--------|-----------------|
| `tests/test_ops_binary.cpp` | `Arithmetic`, `Binary`, `Comparisons`, `Equal`, `Logical`, `Operators`, `Promotion`, `Scalars` | results, dtype rules, scalar operands on both sides, operator overloads |
| `tests/test_ops_unary.cpp` | `Unary`, `Clamp` | values, integer to F32, NaN bounds in `clamp` |
| `tests/test_ops_inplace.cpp` | `InPlace`, `Copy`, `Fill`, `CompoundOperators`, `OpsInplace` | storage kept, views see writes, category check, overlap check, aliasing, `index_add_` |
| `tests/test_ops_reduce.cpp` | `Reduce`, `ReduceStrided` | keepdim, dims validation, dtypes, empty inputs, NaN propagation, ties, `var` precision, strided plan equals a copy |
| `tests/test_ops_linalg.cpp` | `Matmul`, `Dot`, `Outer`, `StrictAliases` | shape rules, empty inner dimension, promotion, views, rank checks |
| `tests/test_ops_indexing.cpp` | `Cat`, `Stack`, `Where`, `MaskedFill`, `IndexSelect`, `Gather`, `ScatterAdd` | shapes, promotion, broadcasting, index errors |
| `tests/test_ops_nn.cpp` | `LogSumExp`, `Softmax`, `LogSoftmax`, `NnFunctions`, `FusedKernels` | no overflow, all `-inf`, integer inputs, fused path equals composed path |
| `tests/test_sequence_ops.cpp` | `Cumsum`, `Sort`, `Topk`, `Triangular`, `IndexAdd`, `FastIndexing`, `Multinomial` | sequence semantics and NaN ordering |
| `tests/test_integer_overflow.cpp` | `IntegerOverflow`, `FloatToInteger` | signed `add`, `sub`, `mul`, `neg`, `abs`, `sum`, `prod`, `matmul`, `cumsum` and `scatter_add_` wrap around; float to `I64` saturates, maps NaN to 0 and keeps the exact lower bound |
| `tests/test_parallel_ops.cpp` | `ParallelReduce` | thread-count independence and accuracy of long sums |
| `tests/test_golden.cpp` | `Golden.AllCasesMatchPyTorch` | 783 cases over 67 ops against PyTorch, including dtype |
| `tests/test_golden_support.cpp` | `Allclose`, `Npy` | the comparison and the reader themselves: tolerance per dtype, NaN option, infinities, shape and dtype checks, failure message, every dtype, scalar and empty arrays, rejected files (missing, Fortran order, float16, not an npy file) |

The unit tests and the golden test complement each other. The unit tests pin down decisions with names (`Reduce.ArgmaxTiesTakeTheFirst`, `Arithmetic.FloorDivideOfMinimumByMinusOneWraps`), so a regression points at the rule it broke. The golden test covers the grid of layouts and special values that nobody would write by hand. `test_golden_support.cpp` tests the test tools, because a comparison that always passes would make the 783 cases worthless.

Run the op tests and the golden test:

```bash
cmake --build build
build/tests/kael_tests --gtest_filter='Arithmetic.*:Reduce*.*:Matmul.*:Softmax.*:LogSoftmax.*:LogSumExp.*:Golden.*'
```

## Build it yourself

1. Write `detail::resolve_operands` and `add`, `sub`, `mul`, `div` with their dtype rules in `src/internal/arithmetic.h`. Leave out the autograd `record_*` calls until [06. Autograd](@/book/06-autograd.md). Checkpoint: `Arithmetic.*` and `Promotion.*` in `tests/test_ops_binary.cpp`.
2. Add `floor_divide`, `pow`, `maximum`, `minimum`, comparisons, logical ops, `equal` and the operator overloads. Checkpoint: the rest of `tests/test_ops_binary.cpp`.
3. Write the unary ops on `unary_kernel` and `floating_unary_kernel`. Checkpoint: `Unary.*`, `Clamp.*`.
4. Write `binary_inplace` with its checks, then `add_` to `pow_`, `clamp_`, `fill_`, `zero_`, `masked_fill_` and the compound operators. Checkpoint: `InPlace.*`, `Fill.*`, `CompoundOperators.*`.
5. Write `tests/support/npy.cpp`, `allclose.h` and `values.h`. Checkpoint: `Allclose.*` and `Npy.*`.
6. Write `tests/test_golden.cpp` with the binary and unary ops registered, and `scripts/generate/golden.py` with the matching pattern lists. Run `venv/bin/python scripts/generate/golden.py`. Checkpoint: `Golden.AllCasesMatchPyTorch` passes for those ops.
7. Write `plan_reduction` with only the contiguous path, `reduce_chunks`, and `sum` with a plain `double` loop. Checkpoint: `Reduce.SumAllAndAlongDims`, `Reduce.KeepdimLeavesSizeOne`, `Reduce.RepeatedOrInvalidDimsThrow`.
8. Add `fold_lanes` and `reduce_in_blocks`, then `prod`, `mean`, `var`, `std`, `norm`, `max`, `min`, `argmax`, `argmin`, `all`, `any`. Checkpoint: all `Reduce.*`, then the reduction golden cases.
9. Add the strided plan for a single reduced dimension. Checkpoint: `ReduceStrided.*`.
10. Write `matmul_forward` and `multiply_batches` with `multiply_exact` for every dtype first, then switch floats to `matmul_versions::multiply` in [08. CPU performance](@/book/08-cpu-performance.md). Add `dot`, `outer`, `mm`, `bmm`. Checkpoint: `Matmul.*` and the matmul golden cases.
11. Write `cat`, `stack`, `where`, `masked_fill`, `index_select`, `gather` and `scatter_add_` with `IndexWalker`. Checkpoint: `tests/test_ops_indexing.cpp` and the indexing golden cases.
12. Write the composed `logsumexp`, `softmax` and `log_softmax`. Checkpoint: `LogSumExp.*`, `Softmax.*`, `LogSoftmax.*`, and the golden `large_values`, `all_minus_infinity` and `plus_infinity` cases.
13. Add the fused row kernels. Checkpoint: `*FusedKernels.*`, which includes the parameterized `Floating/FusedKernels.*` tests.
14. Write `cumsum`, `sort`, `topk`, `tril`, `triu`. Checkpoint: `Cumsum.*`, `Sort.*`, `Topk.*`, `Triangular.*`.

## Check your understanding

<details>
<summary>Why does softmax record one autograd node instead of letting max, sub, exp, sum and div record their own?</summary>

The body runs inside `autograd::without_grad`, so the inner ops record nothing, and `autograd::record_softmax` adds one `SoftmaxBackward` node at the end. One node saves one output tensor and applies one hand-written gradient formula. Five nodes would save five intermediate tensors and chain five generic gradients, which costs more memory and time and accumulates more rounding error.

</details>

<details>
<summary>Why is int_tensor.add_(0.5) an error while float32_tensor.add_(float64_tensor) is allowed?</summary>

`check_result_fits` compares categories, not full dtypes. I64 plus a floating scalar gives F32, category 2, which does not fit in an I64 tensor (category 1): storing it would silently drop the fraction. float32 plus float64 gives F64, also category 2, the same category as `self`, so the result is rounded to float32 on store. The dtype of `self` never changes in either case.

</details>

<details>
<summary>sum(t, {1}) on a contiguous [2, 3, 4] tensor does not copy. Which plan does it use, and where does the chunk of output 2 start?</summary>

The permutation `[0, 2, 1]` is not contiguous and exactly one dimension is reduced, so the CPU uses the strided plan: `chunk_size` 3, `chunk_stride` 4, `kept_sizes` `[2, 4]`, `kept_strides` `[12, 1]`. Output 2 is kept index `[0, 2]`, so its chunk starts at `0 * 12 + 2 * 1 = 2` and reads elements 2, 6 and 10, which are `t[0, 0, 2]`, `t[0, 1, 2]` and `t[0, 2, 2]`.

</details>

<details>
<summary>Why do long sums give the same bits on 1 and 16 threads?</summary>

`reduce_in_blocks` splits a chunk into fixed blocks of 16384 elements and combines the block results in a fixed pairwise tree. Threads only change who computes each block, not where the blocks start or in which order they are combined. Within a block, `fold_lanes` always uses the same 8 lanes. The sequence of floating-point additions is therefore identical for any thread count.

</details>

<details>
<summary>Why does logsumexp replace an infinite maximum with 0, and why do softmax and log_softmax not do the same?</summary>

Shifting by an infinite maximum computes `inf - inf` or `-inf - (-inf)`, both NaN, even though the mathematical answer is well defined: `logsumexp([-inf, -inf])` is `-inf` and `logsumexp([inf, 1])` is `inf`. With a shift of 0 the formula produces those values. Softmax of a row containing `inf`, or of a row of only `-inf`, has no meaningful probability distribution, and PyTorch returns NaN for it. kael matches PyTorch, and the golden `plus_infinity` and `all_minus_infinity` cases check both behaviors.

</details>

<details>
<summary>Why does the golden generator store the base tensor and the view string instead of the viewed tensor?</summary>

`.npy` files are contiguous, so saving the viewed tensor would lose its layout, and kael would only ever run on contiguous inputs. Storing the base and re-applying `transpose`, `slice`, `permute` or `expand` in C++ makes kael's own view code produce the non-contiguous layout, on exactly the data PyTorch used. The test then covers the view code and the op's handling of strides at the same time.

</details>

## Further reading

- PyTorch documentation for [`torch.matmul`](https://pytorch.org/docs/stable/generated/torch.matmul.html), [`torch.sort`](https://pytorch.org/docs/stable/generated/torch.sort.html), [`torch.var`](https://pytorch.org/docs/stable/generated/torch.var.html) and [`torch.logsumexp`](https://pytorch.org/docs/stable/generated/torch.logsumexp.html): the reference semantics kael's golden tests check against.
- NumPy documentation, ["Broadcasting"](https://numpy.org/doc/stable/user/basics.broadcasting.html) and the [`.npy` format specification](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html): the shape rules of `matmul` batches and the byte layout `read_npy` parses.
- Nicholas J. Higham, "Accuracy and Stability of Numerical Algorithms" (SIAM): summation error, pairwise summation and the analysis behind two-pass variance.
- David Goldberg, "What Every Computer Scientist Should Know About Floating-Point Arithmetic" (ACM Computing Surveys, 1991): rounding, cancellation and why reordering a sum changes its result.
- [`src/ops/reduce.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/ops/reduce.cpp): the whole reduction engine in one file, short enough to read end to end.

Next: [06. Autograd](@/book/06-autograd.md)
