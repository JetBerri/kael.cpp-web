+++
title = "Floating point for deep learning"
description = "How float32, float16 and bfloat16 store numbers, where rounding, overflow and summation order change results, and how kael guards against each."
weight = 10
authors = ["kael.cpp contributors"]

[extra]
kicker = "Concept"
math = true
+++

Every number a neural network computes is a floating point number. Most of the time the format is invisible. It becomes visible when a softmax returns NaN, when a loss stops decreasing because updates round away, when the same sum gives a different answer on 8 threads than on 1, or when a test fails by `3e-7`. This article explains the formats from the bit level up, shows where each failure comes from, and points to the places in kael that deal with them.

Prerequisites: binary integers and scientific notation. The kael sections assume you have read [Memory model](@/book/02-memory-model.md) and [Operations](@/book/05-operations.md), but the first half of the article stands on its own.

## The problem

A computer has a fixed number of bits per value. A 32-bit integer represents every whole number in about plus or minus two billion, with a gap of exactly 1 between neighbors. That is the wrong shape for deep learning. Activations, weights and gradients span many orders of magnitude: a gradient of `1e-8` and an attention logit of `1e3` appear in the same training step. What matters for these values is the relative error, not the absolute error.

Floating point solves this by storing a number as a significand times a power of two, like scientific notation in base 2. The spacing between neighboring values grows with their magnitude, so the relative precision stays roughly constant across the whole range.

The price is that floating point arithmetic is not real arithmetic:

- Most decimal values, including `0.1`, have no exact representation.
- Every operation rounds its result, so `(a + b) + c` and `a + (b + c)` can differ.
- Values outside the range become infinity or zero, and undefined operations produce NaN.

A framework that ignores these facts produces code that works on small tests and fails on real data. A softmax over logits of `[1000, 1000]` returns NaN. A float32 sum of ten million copies of `0.1` returns `1087937` instead of `1000000`. A training run in float16 sees small gradients turn into exact zeros. Each of these has a standard fix, and the fixes only make sense once the format is clear.

## The idea

### The IEEE 754 layout

IEEE 754 defines a binary floating point number as three bit fields:

```text
| sign | exponent | fraction |
  1 bit  E bits     F bits
```

For a normal number, the value is

$$
x = (-1)^{s} \times 2^{e - \text{bias}} \times (1.f)_2
$$

where $s$ is the sign bit, $e$ is the exponent field read as an unsigned integer, $\text{bias} = 2^{E-1} - 1$, and $(1.f)_2$ is the binary number formed by a leading 1 followed by the fraction bits. The leading 1 is not stored. Every normal number has it, so storing it would waste a bit. This is called the implicit or hidden bit, and it gives the format $F + 1$ bits of precision.

The four formats that matter for deep learning:

| Format | Sign | Exponent | Fraction | Bias | Precision bits | kael dtype |
|--------|------|----------|----------|------|----------------|------------|
| float64 | 1 | 11 | 52 | 1023 | 53 | `DType::F64` |
| float32 | 1 | 8 | 23 | 127 | 24 | `DType::F32` |
| float16 | 1 | 5 | 10 | 15 | 11 | none |
| bfloat16 | 1 | 8 | 7 | 127 | 8 | none |

float16 and bfloat16 both use 16 bits but split them differently. float16 spends more bits on the fraction and has more precision but a narrow range. bfloat16 keeps the 8 exponent bits of float32, so it covers the same range as float32 with much less precision. A bfloat16 value is the upper 16 bits of a float32 value with the same sign, exponent and leading fraction bits.

```text
float32   s eeeeeeee fffffffffffffffffffffff
bfloat16  s eeeeeeee fffffff                 (same exponent, fraction cut to 7 bits)
float16   s eeeee    ffffffffff              (smaller exponent, 10-bit fraction)
```

Two exponent values are reserved. An exponent field of all ones encodes infinity when the fraction is zero and NaN otherwise. An exponent field of all zeros encodes zero when the fraction is zero and a subnormal number otherwise. Subnormals drop the hidden bit and fill the gap between the smallest normal number and zero, at the cost of precision.

### A worked example: -6.25 in three formats

Write the number in binary scientific notation first.

```text
6.25        = 110.01 in binary
            = 1.1001 x 2^2
sign        = 1 (negative)
exponent    = 2
significand = 1.1001, so the stored fraction starts with 1001
```

Now encode it in each format.

```text
float32   bias 127, exponent field = 2 + 127 = 129 = 10000001
          1 10000001 10010000000000000000000
          = 1100 0000 1100 1000 0000 0000 0000 0000 = 0xC0C80000

float16   bias 15, exponent field = 2 + 15 = 17 = 10001
          1 10001 1001000000
          = 1100 0110 0100 0000 = 0xC640

bfloat16  bias 127, exponent field = 129 = 10000001
          1 10000001 1001000
          = 1100 0000 1100 1000 = 0xC0C8
```

`-6.25` has only four significant binary digits, so it is exact in all three formats. Note that `0xC0C8` is the upper half of `0xC0C80000`: converting float32 to bfloat16 is a matter of keeping the top 16 bits and rounding the rest.

### A worked example: 0.1 does not exist

`0.1` in binary is `0.000110011001100...`, with the block `1100` repeating forever. In scientific notation it is $1.1001100110011\ldots_2 \times 2^{-4}$. Every format has to cut the infinite fraction somewhere and round.

```text
format     bits                                   hex          stored value
float32    0 01111011 10011001100110011001101     0x3DCCCCCD   0.100000001490116...
float16    0 01011 1001100110                     0x2E66       0.0999755859375
bfloat16   0 01111011 1001101                     0x3DCD       0.10009765625
```

The float32 value is off by about $1.5 \times 10^{-9}$, a relative error of $1.5 \times 10^{-8}$. The bfloat16 value is off by about $10^{-4}$, a relative error of $10^{-3}$. The last fraction bit of the float32 and bfloat16 encodings is a 1 because the discarded bits were more than half a unit, so the value rounded up. Truncating float32 to bfloat16 instead of rounding would give `0x3DCC`, which is `0.099609375`.

This is why `0.1 + 0.2 == 0.3` is false in float64: the rounding errors of `0.1`, `0.2`, the sum and `0.3` do not cancel. In float32 the same comparison is true: the rounded sum lands on the same float as the rounded `0.3`. Whether such a comparison holds depends on the format and the values, which is why tests compare computed floats with a tolerance.

### Machine epsilon and the spacing of floats

Machine epsilon, written $\varepsilon$, is the distance from 1 to the next representable number. With $p$ bits of precision it is $\varepsilon = 2^{1-p}$.

| Format | $\varepsilon$ | Largest finite | Smallest normal | Smallest subnormal |
|--------|---------------|----------------|-----------------|--------------------|
| float64 | $2^{-52} \approx 2.22 \times 10^{-16}$ | $\approx 1.80 \times 10^{308}$ | $2^{-1022}$ | $2^{-1074}$ |
| float32 | $2^{-23} \approx 1.19 \times 10^{-7}$ | $\approx 3.40 \times 10^{38}$ | $2^{-126} \approx 1.18 \times 10^{-38}$ | $2^{-149} \approx 1.40 \times 10^{-45}$ |
| float16 | $2^{-10} \approx 9.77 \times 10^{-4}$ | $65504$ | $2^{-14} \approx 6.10 \times 10^{-5}$ | $2^{-24} \approx 5.96 \times 10^{-8}$ |
| bfloat16 | $2^{-7} = 0.0078125$ | $\approx 3.39 \times 10^{38}$ | $2^{-126}$ | $2^{-133}$ |

The spacing between neighbors scales with the magnitude. Between $2^k$ and $2^{k+1}$ the spacing is $\varepsilon \times 2^k$. In float32, the spacing near 1 is about $1.2 \times 10^{-7}$, near $10^8$ it is 8, and above $2^{24} = 16777216$ it is at least 2. Two consequences follow directly:

- In float32, `16777216.0f + 1.0f == 16777216.0f`. Integers above $2^{24}$ are not all representable.
- In float32, `1e8f + 1.0f == 1e8f`. An update smaller than half the spacing disappears completely.

The second fact is the one that matters for training. A weight of `1.0` in bfloat16 has neighbors at `1 - 2^-8` and `1 + 2^-7`. An update of `1e-3` rounds back to `1.0`, and the weight never moves.

### Rounding

IEEE 754 requires that the basic operations (addition, subtraction, multiplication, division and square root) return the exact result rounded to the format. The default rounding mode is round to nearest, ties to even: pick the nearest representable value, and when the exact result lies exactly halfway, pick the one whose last fraction bit is 0. Ties to even avoids the upward bias that always rounding halves up would create over many operations.

Under round to nearest, each operation has a relative error bounded by the unit roundoff $u = \varepsilon / 2$:

$$
\text{fl}(a \circ b) = (a \circ b)(1 + \delta), \qquad |\delta| \le u
$$

This bound is the basis of all error analysis. It says one operation is accurate. It says nothing about a chain of operations, where the errors compound.

### Overflow, underflow and special values

A result larger than the largest finite value overflows to infinity. A result smaller in magnitude than the smallest subnormal underflows to zero. Between the smallest normal and the smallest subnormal, values are subnormal and lose precision gradually.

`exp` is where deep learning meets both limits. In float32, $\ln(3.40 \times 10^{38}) \approx 88.72$, so:

```text
expf(88)    = 1.65e38       finite
expf(89)    = inf           overflow
expf(-88)   = 6.05e-39      subnormal, already below the smallest normal 1.18e-38
expf(-104)  = 0             underflow, below the smallest subnormal 1.40e-45
```

In float16 the window is much narrower: `exp(x)` overflows for $x > \ln(65504) \approx 11.09$. A logit of 12 is enough.

Infinity and NaN then propagate by fixed rules:

```text
inf + 1      = inf
inf - inf    = NaN
inf / inf    = NaN
0 * inf      = NaN
1 / 0        = inf
0 / 0        = NaN
NaN == NaN   is false
```

Once a NaN appears, nearly every operation that reads it returns NaN. One NaN in a gradient reaches every parameter after the next optimizer step.

### Why accumulation order changes results

Floating point addition is commutative but not associative. In float32:

```text
(1e8 + 1) - 1e8  = 1e8 - 1e8  = 0       the 1 was lost when added to 1e8
(1e8 - 1e8) + 1  = 0 + 1      = 1       exact
```

Both orders compute the same real number, and they give different floats. Any change of order changes the rounding: a different thread count, a different block size, SIMD lanes, or a different matmul algorithm.

Order also changes the size of the error, not only its value. A running total `total += x[i]` adds each term to one value that keeps growing. Once the total is large, each new small term loses most of its bits. The worst-case error of a running sum of $n$ terms grows like $n u$. Summing in a tree, where each addition combines two partial sums of similar size, has a worst-case error that grows like $u \log_2 n$.

A concrete case: add `0.1f` to a float32 total ten million times.

```text
running float32 total             1087937.0       8.8% too high
running total in float64, cast    1000000.0
```

Once the float32 total passes $2^{19} = 524288$, the spacing between neighbors is $2^{19} \times 2^{-23} = 0.0625$. The exact sum `total + 0.1` lies 1.6 spacings above `total`, so it rounds to 2 spacings: every addition adds `0.125` instead of `0.1`. Above $2^{20}$ the spacing is `0.125` and the result is the same. The error has the same sign at every step, so it accumulates instead of averaging out. A float64 accumulator has a spacing near $10^6$ of about $10^{-10}$, so the same loop stays close to the true value, and the final cast to float32 rounds once.

A related failure is catastrophic cancellation: subtracting two nearly equal numbers removes their common leading digits and leaves only the rounding errors. The one-pass variance formula $E[x^2] - E[x]^2$ fails this way when the mean is large compared with the spread.

### Why loss scaling exists

Training in float16 runs into its narrow range at the bottom, not at the top. Gradients of a deep network are often smaller than $2^{-24} \approx 6 \times 10^{-8}$, the smallest float16 subnormal. They underflow to zero, and the weights they belong to stop learning, without any error.

Loss scaling shifts the gradients up into the representable range. Multiply the loss by a factor $S$ before the backward pass. By the chain rule, every gradient is multiplied by $S$ too. Before the optimizer step, divide the gradients by $S$ in float32.

```text
true gradient          1e-8
in float16             0                    below 5.96e-8, underflows
scaled by S = 65536    6.55e-4              a normal float16 value
unscaled in float32    1e-8                 recovered
```

If $S$ is too large, some gradients overflow to infinity instead. Dynamic loss scaling handles this: when any gradient is not finite, skip the step and lower $S$; after a run of good steps, raise $S$ again. Mixed precision training combines loss scaling with a float32 master copy of the weights, so that small updates are not lost to the rounding described in the epsilon section.

bfloat16 has the exponent range of float32, so its gradients rarely underflow, and training in bfloat16 usually runs without loss scaling. Its problem is precision, not range, which is why accumulations and weight updates still run in float32.

## In practice

### The dtypes kael has

kael has two floating point dtypes, `float32` and `float64`, next to `int64` and `bool`. From [`include/kael/dtype.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/dtype.h):

```cpp
enum class DType { F32, F64, I64, Bool };
```

`DType::F32` maps to `float` and `DType::F64` to `double` through `cpp_type_t`. There is no float16 and no bfloat16. [Runtime dtype](@/adr/0001-runtime-dtype.md) records the decision to make the dtype a runtime value, and every kernel dispatches over these four types. The file loaders reject what kael cannot represent instead of converting silently. The safetensors reader in [`src/io/safetensors.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/io/safetensors.cpp) throws on any other dtype string:

```cpp
throw TypeError("safetensors dtype '" + name +
                "' is not supported: kael has F32, F64, I64 and BOOL");
```

The NumPy reader used by the tests does the same, and `Npy.RejectsWhatItCannotRead` checks that a float16 `.npy` file raises an error. The GPT-2 checkpoint that [Loading GPT-2](@/book/12-model-loading.md) downloads stores its weights as `F32`, so it loads without conversion.

Because kael has no half-precision types, it has no loss scaling and no mixed precision training. The rest of this section is about the problems that exist in float32 and float64 too: overflow in `exp`, summation error, and comparing results that are correct only up to rounding.

`float32` is the default. Factories such as `zeros` and `full` take `DType dtype = DType::F32`, and a floating point scalar reports `F32` as its dtype category. A scalar never raises the precision of a tensor within the same category, so `F32 + 2.5` stays `F32` and `I64 + 2.5` becomes `F32` ([Memory model](@/book/02-memory-model.md) covers `Scalar` and `promote_with_scalar`).

Conversions from floating point to integers are a hazard of their own: in C++, casting an out-of-range float to an integer is undefined behavior. kael's `saturating_cast` in `include/kael/dtype.h` clamps to the integer range and maps NaN to 0:

```cpp
// Casting an out-of-range float to an integer is undefined behavior. Saturate instead, and map
// NaN to zero, as Rust's `as` does.
template <typename To, typename From> constexpr To saturating_cast(From value) {
    if (value != value) {
        return To{0};
    }
    // ... clamp to std::numeric_limits<To>::min() and max()
}
```

The test `value != value` is true only for NaN, which is the one value that does not compare equal to itself.

### Stable softmax and log-sum-exp

The literal softmax $e^{x_i} / \sum_j e^{x_j}$ overflows in float32 as soon as one logit exceeds about 88.7: the numerator and denominator both become infinity, and `inf / inf` is NaN. The fix uses an identity. For any constant $c$,

$$
\frac{e^{x_i}}{\sum_j e^{x_j}} = \frac{e^{x_i - c}}{\sum_j e^{x_j - c}}
$$

Choosing $c = \max_j x_j$ makes the largest exponent $e^0 = 1$ and every other exponent at most 1. Nothing overflows, and the denominator is at least 1, so it cannot underflow to zero either.

kael's fused row kernel in [`src/ops/nn.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/ops/nn.cpp) applies the shift and accumulates the denominator in `double`:

```cpp
template <typename T> void softmax_row(const T* row, T* out, std::int64_t size) {
    const T largest = row_maximum(row, size);
    double total = 0.0;
    for (std::int64_t i = 0; i < size; ++i) {
        out[i] = std::exp(row[i] - largest);
        total += static_cast<double>(out[i]);
    }
    const auto denominator = static_cast<T>(total);
    for (std::int64_t i = 0; i < size; ++i) {
        out[i] /= denominator;
    }
}
```

`row_maximum` uses `nan_aware_max`, so a NaN in the row becomes the maximum and propagates instead of being skipped. When the softmax dimension is not the last dimension of a contiguous CPU tensor, `softmax` falls back to the composed form `exp(x - max(x, {dim}, true)) / sum(...)`, which applies the same shift. `Softmax.LargeValuesDoNotOverflow` checks that `softmax([1000, 1000])` in float32 returns exactly `[0.5, 0.5]`.

Trace the shift for `x = [1000, 1000, -1000, 999]`:

```text
max                      1000
x - max                  [0, 0, -2000, -1]
exp                      [1, 1, 0, 0.3679]       exp(-2000) underflows to 0, harmlessly
sum                      2.3679
softmax                  [0.4223, 0.4223, 0, 0.1554]
```

The underflow of `exp(-2000)` to zero is correct here: that probability is far below anything float32 can represent.

`log_softmax` is not computed as `log(softmax(x))`. A probability that underflows to 0 would give `log(0) = -inf`, while the exact log probability is a finite number. kael computes `x - max - log(sum(exp(x - max)))` directly:

```cpp
template <typename T> void log_softmax_row(const T* row, T* out, std::int64_t size) {
    const T largest = row_maximum(row, size);
    double total = 0.0;
    for (std::int64_t i = 0; i < size; ++i) {
        out[i] = row[i] - largest;
        total += static_cast<double>(std::exp(out[i]));
    }
    const T log_total = std::log(static_cast<T>(total));
    for (std::int64_t i = 0; i < size; ++i) {
        out[i] -= log_total;
    }
}
```

`LogSoftmax.IsExactWhereLogOfSoftmaxIsNot` checks the difference on `x = [-1000, 0]` in float32: `log_softmax` gives about `-1000` for the first entry, and `log(softmax(x))` gives `-inf`.

`logsumexp` needs one more guard. If the maximum is itself infinite, the shift computes `inf - inf` or `-inf - (-inf)`, both NaN. kael replaces an infinite maximum with 0 before shifting:

```cpp
const Tensor largest = finite_or_zero(max(x, {dim}, true));
const Tensor shifted_total = log(sum(exp(x - largest), {dim}, true)) + largest;
```

With this guard, `logsumexp([-inf, -inf])` is `-inf` (`LogSumExp.AllMinusInfinityIsMinusInfinity`) and `logsumexp([1000, 1000])` is `1000 + log(2)` (`LogSumExp.DoesNotOverflow`). `softmax` and `log_softmax` do not use the guard: for `[inf, 1]` they return `[NaN, NaN]`, which is what PyTorch returns (`LogSoftmax.InfinityGivesNanLikePyTorch`).

The same idea appears in binary cross entropy with logits. Computing `sigmoid(x)` and then `log` overflows or returns `log(0)` for large `|x|`. The loss in [`src/nn/loss.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/nn/loss.cpp) is rearranged so that `exp` only ever sees a non-positive argument:

```cpp
const Tensor losses =
    maximum(logits, 0) - logits * targets + log1p(kael::exp(-kael::abs(logits)));
```

`log1p(y)` computes $\ln(1 + y)$ accurately for small $y$, where forming `1 + y` first would round away most of the bits of $y$.

### Accumulation in reductions

Sums are where accumulation order matters most, and kael's reductions in [`src/ops/reduce.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/ops/reduce.cpp) address both the size of the error and its reproducibility.

First, floating point inputs accumulate in `double`, whatever their dtype, and the result is cast back at the end:

```cpp
template <typename In>
using Accumulator = std::conditional_t<std::is_floating_point_v<In>, double, std::int64_t>;
```

`Reduce.SumOfManyFloatsDoesNotDrift` sums ten million float32 copies of `0.1` and requires the result to be within 1 of `1e6`. A running float32 total gives `1087937`.

Second, the order is fixed by the program, not by the hardware or the thread count. `fold_lanes` keeps eight independent accumulators, and `reduce_in_blocks` splits long inputs into blocks of `kReductionBlock = 16384` elements whose partial sums combine in a tree:

```cpp
// The blocks do not depend on the thread count, so floating-point results are reproducible.
```

Eight lanes give the compiler independent dependency chains to map onto SIMD registers, without the reordering that `-ffast-math` would allow. Because the block boundaries depend only on the input size, `ParallelReduce.ResultsDoNotDependOnThreadCount` can require bit-identical results for any thread count. The tree also shortens the error growth from linear to roughly logarithmic in the input size: `ParallelReduce.LongSumIsMoreAccurateThanRunningTotal` sums $2^{24}$ copies of `0.1` in float64 and checks that the result is closer to the exact value than a running total.

Third, variance is computed in two passes: the mean first, then the sum of squared deviations from it. This avoids the cancellation of $E[x^2] - E[x]^2$. `Reduce.VarDoesNotLosePrecisionWithLargeMean` checks that the variance of `[1e8 + 1, 1e8 - 1]` with correction 0 is 1 to within `1e-9`. The fused layer normalization kernel `standardize_row` follows the same pattern, with the mean, the squared deviations and the scale all computed in `double`.

Matmul is the exception. The matmul versions in [`src/linalg/matmul_versions.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/linalg/matmul_versions.cpp) accumulate in the element type `T`, and each version adds the products in a different order: the naive `v1_ijk` uses one running total per output, while the packed micro-kernels keep a block of accumulators in registers. Their float32 results agree only up to rounding, which the tests below account for.

### Comparing against PyTorch with tolerances

Two correct implementations of the same function rarely produce bit-identical floats, because they round in different places. Tests compare with a tolerance instead of `==`. kael's `kael::test::allclose` in [`tests/support/allclose.h`](https://github.com/JetBerri/kael.cpp/blob/main/tests/support/allclose.h) uses the same rule as NumPy and PyTorch:

$$
|a - e| \le \text{atol} + \text{rtol} \times |e|
$$

The relative term scales with the expected value, which matches how floating point error behaves. The absolute term handles expected values near zero, where any relative tolerance becomes zero. The defaults depend on the dtype:

```cpp
inline Tolerance default_tolerance(DType dtype) {
    switch (dtype) {
    case DType::F32:
        return {1e-5, 1e-6};
    case DType::F64:
        return {1e-12, 1e-14};
    case DType::I64:
    case DType::Bool:
        return {0.0, 0.0};
    }
    detail::unreachable();
}
```

The float32 relative tolerance of `1e-5` is about 84 times float32's $\varepsilon$, which leaves room for a few dozen roundings. The float64 value of `1e-12` is about 4500 times float64's $\varepsilon$. Integers and booleans must match exactly. Infinities must match exactly, including the sign, and NaN matches NaN only when `equal_nan` is set.

`Golden.AllCasesMatchPyTorch` in [`tests/test_golden.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_golden.cpp) runs every case generated by [`scripts/generate/golden.py`](https://github.com/JetBerri/kael.cpp/blob/main/scripts/generate/golden.py) through `allclose` with the default tolerances and `equal_nan = true`, so NaN outputs must appear in the same places as in PyTorch. Other tests widen or tighten the tolerance where the arithmetic justifies it:

| Test | float32 (rtol, atol) | float64 (rtol, atol) | Why |
|------|----------------------|----------------------|-----|
| Golden cases | `1e-5, 1e-6` | `1e-12, 1e-14` | Defaults |
| Matmul versions, `tests/test_matmul_versions.cpp` | `1e-3, 1e-3` | `1e-11, 1e-11` | Long float32 dot products summed in different orders |
| CPU against CUDA, `tests/test_cuda.cpp` | `1e-4, 1e-4` | `1e-10, 1e-10` | Different kernels and summation order on the device |
| Fused against composed cross entropy gradient, `tests/test_nn_layers.cpp` | `1e-5, 1e-5` | `1e-13, 1e-13` | Same math, different operation order |
| Layer and loss goldens, `tests/test_nn_layers.cpp` | not used | `1e-10, 1e-12` | Inputs generated in float64 |

Gradcheck goes further and refuses float32 entirely. `autograd::gradcheck` estimates each derivative with a central difference, `(f(x + eps) - f(x - eps)) / (2 * eps)`, with `eps = 1e-6` by default. The subtraction in the numerator cancels: `f(x + eps)` and `f(x - eps)` agree in most of their digits. Each of them carries a rounding error of about $u |f|$, and dividing by $2 \varepsilon_{\text{step}}$ amplifies it. In float32, $u \approx 6 \times 10^{-8}$ divided by $10^{-6}$ gives an error near 0.06 for $|f| \approx 1$, which would hide any real bug. In float64 the same error is about $10^{-10}$. The check in [`src/autograd/gradcheck.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/autograd/gradcheck.cpp) enforces it:

```cpp
KAEL_CHECK_TYPE(input.dtype() == DType::F64,
                "gradcheck input " << position << " must be F64, got " << input.dtype());
```

[Autograd](@/book/06-autograd.md) covers the rest of gradcheck.

## Pitfalls

- **Exponentiating without a shift.** `exp(x) / sum(exp(x))` works on unit tests with small logits and returns NaN on real logits. The same applies to any `log(sum(exp(...)))` written by hand. Use `softmax`, `log_softmax` and `logsumexp`.
- **Taking the log of a probability.** `log(softmax(x))` turns underflowed probabilities into `-inf`. Use `log_softmax`, and pass logits, not probabilities, to the loss functions.
- **Accumulating in the input precision.** A float32 running total over millions of elements drifts by percent, not by ulps. Accumulate in a wider type, or sum in a tree.
- **Expecting bit-identical results across implementations.** A different thread count, a SIMD path, a GPU kernel or a BLAS library changes the summation order. kael fixes the order for its own reductions, but its float32 matmul versions still differ from each other. Compare with `allclose` and a tolerance chosen for the dtype.
- **Comparing with `==` to detect NaN.** `x == NaN` is always false, and so is `NaN == NaN`. Use `std::isnan`, or `value != value` where a `constexpr` context needs it.
- **Choosing a tolerance by trial and error.** A tolerance that is too loose hides bugs, and one that is too tight fails on correct code. Start from the unit roundoff of the dtype and the number of operations that round, as the tables above do.
- **Casting a float to an integer without a range check.** In C++ an out-of-range or NaN value makes the cast undefined behavior. kael's `convert_value` routes such casts through `saturating_cast`.
- **Updates smaller than the spacing.** In a low-precision format, `w + lr * g` can round back to `w` for every step. Mixed precision training keeps a float32 copy of the weights for this reason.
- **Underflowing gradients in float16.** Small gradients become exact zeros with no error or warning. Loss scaling exists to prevent this; bfloat16 avoids it with a wider exponent range.

## Check your understanding

<details>
<summary>Why does bfloat16 rarely need loss scaling while float16 does?</summary>

bfloat16 has 8 exponent bits, the same as float32, so its smallest normal value is $2^{-126}$, the same as float32's. Gradients of a deep network almost never fall below that. float16 has 5 exponent bits, its smallest subnormal is $2^{-24} \approx 6 \times 10^{-8}$, and real gradients often fall below it and underflow to zero. Loss scaling moves gradients up into the representable range of float16. bfloat16 already covers that range and pays for it with fewer fraction bits.

</details>

<details>
<summary>In float32, what is 1e8 + 1, and why?</summary>

It is `1e8`. $10^8$ lies between $2^{26}$ and $2^{27}$, where the spacing between neighboring float32 values is $2^{26} \times 2^{-23} = 8$. The exact sum $100000001$ is closer to $100000000$ than to $100000008$, so round to nearest returns $10^8$. Any addend smaller than half the spacing, here 4, disappears.

</details>

<details>
<summary>Subtracting the maximum before exp changes every exponent. Why does softmax return the same values?</summary>

$e^{x_i - c} = e^{x_i} e^{-c}$. The factor $e^{-c}$ appears in every term of the numerator and the denominator and cancels. With $c = \max_j x_j$, the largest term is $e^0 = 1$, so nothing overflows, and the denominator is at least 1, so the division is safe. In exact arithmetic the result is identical; in floating point it is the version that stays in range.

</details>

<details>
<summary>kael's sum gives bit-identical results on 1 and 16 threads. What design choice makes that possible?</summary>

The summation order depends only on the input size. `reduce_in_blocks` splits the input into fixed blocks of 16384 elements, `fold_lanes` sums each block with eight fixed lanes, and the partial sums combine in a fixed tree. Threads decide which blocks they compute, not how the blocks are combined. If the blocks were sized by thread count, a different thread count would change the order of additions, and floating point addition is not associative.

</details>

<details>
<summary>Why does gradcheck require float64 inputs?</summary>

A central difference subtracts two nearly equal values, `f(x + eps)` and `f(x - eps)`, and divides by `2 * eps`. The rounding error of each value, about $u |f|$, is amplified by $1 / (2 \varepsilon_{\text{step}})$. With float32's $u \approx 6 \times 10^{-8}$ and `eps = 1e-6`, the estimate is off by around $0.03$ to $0.06$ for values near 1, far above the default tolerances of `1e-5` absolute and `1e-3` relative. In float64 the same error is around $10^{-10}$.

</details>

<details>
<summary>The golden tests use rtol 1e-5 for float32 but the matmul version tests use 1e-3. Why the difference?</summary>

Most golden cases are elementwise operations or reductions that kael accumulates in `double`, so a handful of roundings separate kael from PyTorch. The matmul versions accumulate in float32 and add the products of each dot product in different orders, from a single running total in `v1_ijk` to blocks of register accumulators in the packed kernels. Each product and each addition rounds, the errors add up along the inner dimension, and different orders accumulate different errors. The wider tolerance reflects that error budget.

</details>

## Further reading

- David Goldberg, "What Every Computer Scientist Should Know About Floating-Point Arithmetic", ACM Computing Surveys, 1991. The classic introduction to rounding error, guard digits and cancellation.
- Nicholas J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd edition, SIAM, 2002. The reference for error bounds of summation, dot products and matrix multiplication, including pairwise summation.
- IEEE Std 754-2019, *IEEE Standard for Floating-Point Arithmetic*. The definition of the formats, rounding modes and special values.
- Paulius Micikevicius et al., "Mixed Precision Training", ICLR 2018. The paper that introduced float32 master weights and loss scaling for float16 training.
- Dhiraj Kalamkar et al., "A Study of BFLOAT16 for Deep Learning Training", 2019. Reports bfloat16 training results across many workloads and explains the role of its float32-sized exponent.
- [`src/ops/nn.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/ops/nn.cpp) and [`src/ops/reduce.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/ops/reduce.cpp). Short, readable implementations of a stable softmax, log-sum-exp and reproducible float sums.
- [`tests/support/allclose.h`](https://github.com/JetBerri/kael.cpp/blob/main/tests/support/allclose.h). A complete tolerance-based comparison in 85 lines, with the special cases for NaN and infinity.
