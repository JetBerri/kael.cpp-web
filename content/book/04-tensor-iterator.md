+++
title = "TensorIterator"
description = "The loop engine behind every elementwise op: type promotion, dtype dispatch, broadcasting with byte strides, dimension coalescing and a parallel two-level loop."
weight = 4
slug = "tensor-iterator"

[extra]
chapter = "04"
kicker = "Chapter 04"
builds = "The loop engine behind every elementwise op: broadcasting, type promotion and dimension coalescing"
source = "docs/04-tensor-iterator.md"
tag = "chapter-05-iteration"
+++

This chapter builds the engine behind every elementwise operation: a loop that walks one output and any number of inputs with arbitrary shapes, strides and offsets, applies broadcasting, merges dimensions that can be walked as one, and hands contiguous runs of elements to a typed inner loop. On top of it sit the type promotion rules and the dtype dispatch that turn a generic lambda into a call on `float`, `double`, `std::int64_t` or `std::uint8_t`. It corresponds to the iterator phase of the project.

| File | Role |
|------|------|
| [`include/kael/iterator.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/iterator.h) | `LoopDims`, `coalesce_dims`, `check_same_device`, the `TensorIterator` class and its templated `run` and `run_range` |
| [`src/iterator.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/iterator.cpp) | building the iterator: input checks, common shape, output allocation, byte strides, coalescing, the device layout |
| [`include/kael/dtype.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/dtype.h) | `promote_types`, `dtype_category`, `saturating_cast`, `convert_value`, `dispatch`, `dispatch_floating`, `dispatch_no_bool` |
| [`include/kael/scalar.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/scalar.h) | `Scalar` and `promote_with_scalar` |
| [`include/kael/operand.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/operand.h) | `Operand`, a tensor or a scalar |
| [`include/kael/kernels.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/kernels.h) | `result_dtype`, `run_typed`, `typed_inner_loop` and the public kernel templates (`unary_kernel`, `binary_kernel`, `comparison_kernel`, ...) |
| [`src/internal/element_functions.h`](https://github.com/JetBerri/kael.cpp/blob/main/src/internal/element_functions.h) | the functor for every elementwise operation, shared by CPU and CUDA |

## Prerequisites

- [01. Shapes and strides](@/book/01-shapes-and-strides.md): `broadcast_shapes`, `broadcast_strides`, `numel`, `index_from_linear`.
- [02. Memory model](@/book/02-memory-model.md): `DType`, `Scalar`, `Tensor::device_data`, factories such as `empty` and `scalar`.
- [03. Views](@/book/03-views.md): strides, stride 0 from `expand`, contiguity, `convert_strided`.
- [ADR 0001: runtime dtype](@/adr/0001-runtime-dtype.md) explains why the dtype is a runtime value and not a template parameter of `Tensor`.
- [ADR 0003: type promotion](@/adr/0003-type-promotion.md) explains why the promotion rules follow PyTorch.

## Why this layer exists

### The problem

A binary operation such as `a + b` looks like one line of arithmetic. In a tensor library it must handle every combination of four independent variables:

- **Shapes** that broadcast against each other: `[2, 3, 4]` plus `[4]`, or `[3, 1]` plus `[1, 4]`.
- **Layouts** that are not contiguous: a transpose, a stepped slice, a stride-0 `expand`, a view with a nonzero offset.
- **Dtypes** that differ: `int64` plus `float32`, or a float32 tensor times a C++ `double` literal.
- **Outputs** that are either allocated fresh or are an existing view, as in `a.add_(b)` or `copy_`.

The library has around forty elementwise operations (arithmetic, comparisons, logical ops, `exp`, `sigmoid`, `clamp`, `where`, `masked_fill`, dtype conversion, ...). Writing one loop per operation per case gives thousands of loops, each with its own chance of a stride bug. A bug in the loop for "transposed float64 input with a broadcast int64 input" would only show up for that exact combination.

### What breaks without it

Without a shared engine, each op would need to answer the same questions on its own: what is the output shape, which dtype wins, how to find element `i` of a non-contiguous input, how to split the work across threads, how to keep a Bool tensor holding only 0 and 1. Some ops would answer differently from others. `a + b` and `a * b` could disagree on the result dtype, or one could forget that a given output must not broadcast.

### The mental model

Hold on to this picture: **an elementwise op is a functor applied to aligned positions of N+1 strided arrays.** The iterator aligns the arrays. The kernel layer picks the C++ type. The functor does the arithmetic on two numbers and knows nothing else.

```text
            functor:  Plus{}(x, y)            knows values only
                 ^
                 |  one call per element
   typed inner loop<T>                         knows the element type, one stride per operand
                 ^
                 |  (pointers, innermost byte strides, count)
   TensorIterator::run                         knows shapes, byte strides, threads
                 ^
                 |  one base pointer and one byte-stride vector per operand
   TensorIterator::build                       broadcast, allocate output, coalesce
```

The iterator never learns `T`. It only moves `std::byte*` pointers by byte strides. The kernel layer never sees a stride except the innermost one. This separation is the same one PyTorch uses in its own `TensorIterator`, and the same one NumPy uses between `nditer` and the typed inner loops of a ufunc.

## Concepts

### The path of one call

Before reading any code, follow `kael::add(a, b)` from the public function down to the arithmetic:

```text
kael::add(a, b)                        src/ops/binary.cpp
  |
  v
detail::resolve_operands               scalars become 0-d tensors
  |
  v
result dtype                           result_dtype / detail::add_dtype
  |
  v
binary_kernel                          include/kael/kernels.h
  |
  v
detail::same_dtype_kernel              cast inputs with to(dtype)
  |
  v
TensorIterator::build                  broadcast, byte strides, coalesce
  |
  v
dispatch(dtype)                        pick T at runtime
  |
  v
detail::run_typed<T, T, 2>
  |
  v
TensorIterator::run                    parallel_for over elements
  |
  v
detail::typed_inner_loop               contiguous or strided branch
  |
  v
functions::Plus{}(x, y)
```

Each box has one job. The iterator knows nothing about dtypes beyond byte sizes. The kernel layer knows nothing about strides. The functor knows only how to combine two values.

### Type promotion

**Type promotion** is the rule that picks the dtype of a result when the inputs have different dtypes. kael has four dtypes. `promote_types` in `include/kael/dtype.h` ranks them `Bool < I64 < F32 < F64` and returns the higher one:

| `promote_types` | Bool | I64 | F32 | F64 |
|-----------------|------|-----|-----|-----|
| Bool | Bool | I64 | F32 | F64 |
| I64 | I64 | I64 | F32 | F64 |
| F32 | F32 | F32 | F32 | F64 |
| F64 | F64 | F64 | F64 | F64 |

`int64 + float32` gives `float32`, as in PyTorch, even though large integers lose precision: float32 has a 24-bit significand, so `16777217` becomes `16777216`. The alternative, promoting `int64 + float32` to `float64`, would be exact for more values but would double the memory of every mixed operation. PyTorch and kael choose the smaller type. NumPy's classic rules choose `float64`. The table is checked at compile time by `static_assert`s in `Promotion.Table` in `tests/test_iterator.cpp`.

A coarser grouping, the **category**, is defined by `dtype_category`: Bool is 0, I64 is 1, and both floating types are 2. F32 and F64 share a category. Categories drive the two rules that follow, for scalars and for 0-d tensors.

### Scalars

A C++ number passed to an op becomes a `Scalar`. Its dtype comes from its C++ type, not its value: `true` is Bool, `2` is I64, `2.5` and `2.5f` are both F32. A floating scalar is F32 even when written as a `double` literal. It still stores the value as `double`, so no precision is lost before the cast to the final dtype.

The rule is: **a scalar never raises the precision of a tensor within the same category; it only moves the result to a higher category.** `promote_with_scalar` implements it:

| Tensor dtype | Scalar | Category of tensor, scalar | Result |
|--------------|--------|----------------------------|--------|
| F32 | `2.5` | 2, 2 | F32 |
| F64 | `2.5f` | 2, 2 | F64 |
| I64 | `2.5` | 1, 2 | F32 |
| Bool | `2.5` | 0, 2 | F32 |
| I64 | `2` | 1, 1 | I64 |
| F32 | `2` | 2, 1 | F32 |
| Bool | `2` | 0, 1 | I64 |
| Bool | `true` | 0, 0 | Bool |
| F64 | `true` | 2, 0 | F64 |

Every row is a `static_assert` in `Promotion.Scalars`. When the scalar wins, the result takes the scalar's dtype, which for a floating scalar is the default float, F32. That is why `I64 + 2.5` gives F32 and not F64.

### 0-d tensors

A **0-d tensor** is a tensor with shape `[]`: one element and no dimensions. A loss value is a 0-d tensor, and so is the result of `sum(t)`. kael gives 0-d tensors the same treatment as scalars: tensors with at least one dimension decide the dtype, and a 0-d tensor only wins when its category is higher.

`result_dtype` splits the inputs into two groups, the dimensioned ones and the 0-d ones, promotes each group with the table, and then compares the groups by category. Results of a compiled check:

| Expression | Dimensioned group | 0-d group | Result |
|------------|-------------------|-----------|--------|
| `randn({3}) + scalar(1.0, DType::F64)` | F32 | F64 (category 2) | float32 |
| `ones({3}, DType::I64) + scalar(1.0)` | I64 | F32 (category 2) | float32 |
| `ones({3}, DType::I64) + 2.5` | I64 | F32 (from the scalar) | float32 |
| `ones({3}) + ones({3}, DType::F64)` | F64 | none | float64 |
| `ones({3}, DType::I64) * true` | I64 | I64 (from the scalar) | int64 |

The reason for this rule is the common case `x * 0.5` on a float32 tensor. Without it, the literal, or a 0-d float64 tensor holding a loss value, would upcast a large activation tensor to float64 and double its memory. PyTorch uses the same rule, and the golden cases `zero_dim_wider`, `zero_dim_higher_category` and `zero_dim_lower_category` in `scripts/generate/golden.py` check kael against it.

#### Worked example: `ones({3}, DType::I64) + 2.5`

1. `2.5` converts to `Operand`, which holds a `Scalar` of dtype F32 and value `2.5` stored as `double`.
2. `detail::resolve_operands` sees one tensor and one scalar. It calls `promote_with_scalar(I64, 2.5)`. The scalar's category (2) is higher than the tensor's (1), so the 0-d tensor gets dtype F32.
3. `detail::add_dtype` is not Bool + Bool, so it calls `result_dtype({&x, &y})`. The dimensioned group is I64, the 0-d group is F32. Category 2 beats category 1: the result is F32.
4. `detail::same_dtype_kernel` calls `x.to(F32)`, which allocates a new F32 tensor `[1, 1, 1]`, and `y.to(F32)`, which returns `y` itself because it already is F32.
5. The iterator now sees two F32 inputs and an F32 output. The functor runs as `Plus{}(float, float)` and the result is `[3.5, 3.5, 3.5]`.

Promotion is decided once, before the loop. The loop itself runs on one type.

### Op-specific overrides

`result_dtype` is the default. Some ops adjust it, in `src/internal/arithmetic.h` and in the op files:

| Op | Rule | Source |
|----|------|--------|
| `add` | Bool + Bool gives I64 | `detail::add_dtype` |
| `div` | integer or Bool results become F32 | `detail::div_dtype` |
| `sub`, `pow`, `floor_divide` | Bool inputs throw | `detail::check_no_bool` |
| comparisons | compute in `result_dtype`, output Bool | `comparison_kernel` |
| `logical_*` | inputs converted to Bool first | `logical` in `src/ops/binary.cpp` |
| `exp`, `log`, `sin`, ... | integer and Bool inputs become F32 | `floating_unary_kernel` |
| in-place ops | result category must not exceed the category of `self` | `check_result_fits` in `src/ops/inplace.cpp` |

`add_dtype` differs from PyTorch, where Bool + Bool stays Bool and behaves as a logical or. kael counts instead: `true + true` is `2`. For that reason the `add` golden cases skip the `bools` pattern (`BOOL_PATTERNS[1:]` in `scripts/generate/golden.py`). Reductions, matmul and `cat` have their own dtype rules, covered in [05. Operations](@/book/05-operations.md).

### Dtype dispatch

Tensors carry their dtype as a runtime `enum class DType`. Kernels are templates on the C++ element type. **Dtype dispatch** is the bridge: a `switch` on the runtime value that calls a template instantiated for the matching C++ type.

| `DType` | C++ type | Element size |
|---------|----------|--------------|
| F32 | `float` | 4 |
| F64 | `double` | 8 |
| I64 | `std::int64_t` | 8 |
| Bool | `std::uint8_t` | 1 |

The caller writes the body once as a C++20 template lambda. The compiler instantiates it four times, once per dtype, and the `switch` picks one at runtime. PyTorch does the same with its `AT_DISPATCH_*` macros; kael uses a template lambda instead of a macro because C++20 allows `[&]<typename T>() { ... }`.

Bool is stored as `std::uint8_t`. The comment in `dtype.h` gives the reason: `sizeof(bool)` is not fixed by the standard, and `std::vector<bool>` cannot hand out element pointers. A Bool tensor must hold only 0 or 1. Any other byte would break `logical_*` and `equal`, which compare the bytes. So every store goes through `convert_value`, which maps any nonzero input to 1:

| From | Value | `convert_value<std::uint8_t>` | `convert_value<std::int64_t>` |
|------|-------|-------------------------------|-------------------------------|
| `double` | `2.5` | 1 | 2 |
| `double` | `0.0` | 0 | 0 |
| `double` | NaN | 1 (NaN is nonzero) | 0 |
| `double` | `1e30` | 1 | `INT64_MAX` |
| `double` | `-1e30` | 1 | `INT64_MIN` |

The last column comes from `saturating_cast`. A plain `static_cast` of an out-of-range float or of NaN to an integer is undefined behavior in C++, so kael clamps to the integer range and maps NaN to 0, as Rust's `as` does. [02. Memory model](@/book/02-memory-model.md) introduces it.

### Byte strides and broadcasting

The iterator receives an output, given or to be allocated, and one or more inputs. It reduces each operand to two things: a **base pointer**, the address of its first element, and a **stride vector** over a shared shape, the **common shape** of all inputs.

Broadcasting uses the same stride-0 trick as `expand` from [03. Views](@/book/03-views.md). An input of shape `[4]` against a common shape `[2, 3, 4]` gets strides `[0, 0, 1]`: moving along the first two dimensions does not move in memory. The loop then treats every operand the same way. No operand is special because it was broadcast.

The strides are then converted from elements to **bytes**: each stride is multiplied by the element size of its operand. This is what allows operands of different dtypes in one loop. When `to(DType::F32)` converts an I64 tensor, the iterator has an F32 output and an I64 input:

```text
convert_strided: F32 output, I64 input, shape [3]

element strides   output [1]    input [1]
element size      4             8
byte strides      output [4]    input [8]

memory   output: |o0  |o1  |o2  |              4 bytes per step
         input:  |i0      |i1      |i2      |  8 bytes per step
```

With byte strides, one `std::byte*` per operand and one stride per operand per dimension describe everything. The iterator never needs to know `T`, and the same compiled iterator code serves all 16 dtype pairs of a conversion.

### Dimension coalescing

Walking a `[2, 3, 4]` tensor as three nested loops costs a counter update at every row boundary and keeps rows short. When the memory layout allows it, two adjacent dimensions can be merged into one longer dimension. **Coalescing** is that merge.

The condition, for every operand, is that stepping once along the outer dimension lands where stepping `size` times along the inner dimension would:

```text
stride[outer] == stride[inner] * shape[inner]
```

For a contiguous `[2, 3, 4]` float32 tensor with byte strides `[48, 16, 4]`, `16 == 4 * 4` and `48 == 16 * 3`, so the whole tensor is one dimension of 24 elements with stride 4. A transposed tensor fails the test, because its strides grow in the wrong direction.

The condition must hold for **all operands**, because they share one loop. One operand that cannot merge blocks the merge for all of them.

#### Worked example: `[2, 3, 4] + [4]`

`x` is a contiguous float32 tensor of shape `[2, 3, 4]`. `y` is a contiguous float32 tensor of shape `[4]`. The output is allocated.

Before coalescing, in bytes (element size 4):

```text
common shape          [ 2,  3, 4]
operand 0 (output)    [48, 16, 4]
operand 1 (x)         [48, 16, 4]
operand 2 (y)         [ 0,  0, 4]   broadcast_strides([4], [1], [2,3,4]) = [0,0,1], times 4
```

Pass 1, `inner = 2`, `outer = 1`:

```text
output: 16 == 4 * 4   yes
x:      16 == 4 * 4   yes
y:       0 == 4 * 4   no    -> no merge
```

Pass 2, `inner = 1`, `outer = 0`:

```text
output: 48 == 16 * 3  yes
x:      48 == 16 * 3  yes
y:       0 ==  0 * 3  yes   -> merge: shape [6, 4], each operand keeps the inner stride
```

Result:

```text
loop shape            [ 6, 4]
operand 0 (output)    [16, 4]
operand 1 (x)         [16, 4]
operand 2 (y)         [ 0, 4]
```

The loop runs 6 rows of 4 contiguous elements. `y` restarts at its base for every row because its outer stride is 0. A compiled check of `TensorIterator::dims()` on this input prints exactly `[6 4]`, `[16 4]`, `[16 4]`, `[0 4]`.

The broadcast operand blocked the inner merge, but not the outer one: `y` is constant along both outer dimensions, and two stride-0 dimensions always merge because `0 == 0 * n`.

#### Worked example: a permuted input

`Iterator.ThreeDimensionalOdometer` adds `a = arange(0, 24).reshape({2, 3, 4}).permute({2, 0, 1})` to a contiguous `zeros({4, 2, 3})`. `a` has shape `[4, 2, 3]` and element strides `[1, 12, 4]`.

```text
common shape          [ 4,  2,  3]
operand 0 (output)    [24, 12,  4]
operand 1 (a)         [ 4, 48, 16]
operand 2 (zeros)     [24, 12,  4]

pass 1 (outer 1, inner 2):  output 12 == 4*3 yes, a 48 == 16*3 yes, zeros yes   -> merge
                            shape [4, 6], strides [24, 4], [4, 16], [24, 4]
pass 2 (outer 0, inner 1):  output 24 == 4*6 yes, a 4 == 16*6 no                -> stop

loop shape [4, 6]
```

A permuted tensor can still merge some of its dimensions. Here the last two dimensions of `a` came from dimensions 0 and 1 of the original contiguous tensor, which were adjacent and in order. The inner stride of `a` is 16 bytes, not 4, so the typed inner loop takes its strided branch for this operand.

#### More cases

| Inputs | Loop shape | Why |
|--------|------------|-----|
| `[2, 3, 4]` contiguous with `[2, 3, 4]` contiguous | `[24]` | every pair merges |
| `[2, 3, 4]` with `[2, 1, 4]` | `[2, 3, 4]` | `y` has byte strides `[16, 0, 4]`; neither `0 == 4 * 4` nor `16 == 0 * 3` holds |
| a transposed `[2, 3]` with strides `[1, 2]` | `[2, 3]` | `1 != 2 * 3` (`Coalesce.TransposedMergesNothing`) |
| `[2, 3]` with strides `[0, 0]` | `[6]` | `0 == 0 * 3`: a fully broadcast scalar merges (`Coalesce.StrideZeroDimsMerge`) |
| `[2, 1, 3, 1]` with strides `[3, 99, 1, 7]` | `[6]` | size-1 dims are dropped first, their strides are never used (`Coalesce.SizeOneDimsDisappear`) |

The iterator does **not** reorder dimensions. PyTorch's `TensorIterator` sorts dimensions by stride so a transposed input can still be walked in memory order. kael walks every operand in the logical order of the common shape. For a transposed output that means writes follow the output's logical order (`Iterator.GivenOutputIsWrittenInItsOrder`). Sorting would make transposed inputs faster at the cost of a permutation step and harder-to-follow code. It is not implemented.

### The two-level loop

After coalescing, the loop shape has an **innermost dimension**, the last one, and zero or more **outer dimensions**. The iterator splits the work in two levels:

- The outer level is an **odometer**: a counter per outer dimension that increments the last digit and carries into the previous one on overflow, like a car's mileage counter. Each increment adds one stride to the byte offset of every operand.
- The inner level is a call to the inner loop with one pointer per operand, the innermost byte stride of each, and a count of elements along the innermost dimension.

The odometer is the same idea as `index_from_linear` and `offset_from_index` from [01. Shapes and strides](@/book/01-shapes-and-strides.md), but incremental. Computing the full offset from a linear index costs a division and a modulo per dimension per element. The odometer costs one addition per operand per row.

Work is split across threads by **linear element index**, not by row. A thread may receive `[10, 24)` on a loop of rows of 4, which starts in the middle of row 2. The setup step converts `first` into an odometer position and an offset inside the row; the loop then finishes the partial row, walks whole rows, and stops at `last`, possibly in the middle of a row.

Splitting by element instead of by row keeps the load even when the loop has few, long rows (a `[2, 1000000]` tensor) or many short ones.

## Implementation

The code follows the order you would write it: the value types, the promotion functions, dispatch, the iterator, the loop, the typed layer and finally the functors.

### Scalar and Operand

From `include/kael/scalar.h`:

```cpp
constexpr Scalar(bool value) : value_(std::int64_t{value ? 1 : 0}), dtype_(DType::Bool) {}

template <std::integral T>
constexpr Scalar(T value) : value_(static_cast<std::int64_t>(value)), dtype_(DType::I64) {}

template <std::floating_point T>
constexpr Scalar(T value) : value_(static_cast<double>(value)), dtype_(DType::F32) {}
```

The value lives in a `std::variant<std::int64_t, double>`, so an integer scalar keeps all 64 bits and a floating scalar keeps the full `double`. `Scalar::as<T>()` reads it back through `convert_value<T>`. The non-template `bool` constructor matters: without it, `true` would match `std::integral` and become I64.

The scalar rule from the Concepts section is one comparison of categories:

```cpp
// A scalar never raises the precision of a tensor within the same category:
// F32 + 2.5 stays F32, while I64 + 2.5 moves to the default float, F32.
constexpr DType promote_with_scalar(DType tensor_dtype, Scalar scalar) {
    return dtype_category(scalar.dtype()) <= dtype_category(tensor_dtype) ? tensor_dtype
                                                                          : scalar.dtype();
}
```

An `Operand` (`include/kael/operand.h`) holds either a `Tensor` or a `Scalar` in a `std::variant`. It is implicitly constructible from a `Tensor` and from any type convertible to `Scalar`, so one signature `add(const Operand&, const Operand&)` accepts `add(t, 2)`, `add(2, t)` and `add(t, u)`.

Before any kernel runs, `detail::resolve_operands` in `src/ops/binary.cpp` turns a scalar operand into a 0-d tensor of dtype `promote_with_scalar(other.dtype(), value)`. If both operands are scalars, each keeps its own scalar dtype. If the tensor lives on CUDA, the 0-d tensor is created there with `full({}, value, dtype, device)`. From that point on, every operand is a tensor.

### result_dtype

0-d tensors get the scalar treatment in `result_dtype`, which picks the dtype of an op with several tensor inputs. From `include/kael/kernels.h`:

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

The inputs split into two groups, each promoted with the table. If only one group exists, it decides alone: two 0-d tensors promote normally. If both groups exist, the 0-d group wins only when its category is strictly higher. Because a scalar operand became a 0-d tensor in `resolve_operands`, this one function covers scalars too.

### dispatch and convert_value

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

The caller writes a C++20 template lambda:

```cpp
dispatch(dtype, [&]<typename T>() { run_typed<T, T, 2>(iterator, f); });
```

`f.template operator()<float>()` is the syntax for calling a template lambda with an explicit template argument, since `T` cannot be deduced from an empty argument list. `decltype(auto)` lets `dispatch` return whatever the lambda returns, so `dispatch` can produce a `Tensor` as well as `void`. `detail::unreachable()` throws: the compiler cannot prove that the `switch` is exhaustive, and an out-of-range enum value is a bug worth reporting.

Two variants restrict the set:

- `dispatch_floating(dtype, op, f)` instantiates only `float` and `double`. For other dtypes it throws `kael::TypeError` with a message naming `op`: "`op` only supports floating point dtypes, got int64".
- `dispatch_no_bool(dtype, op, f)` excludes `std::uint8_t` and throws `kael::TypeError` with "`op` does not support dtype bool".

Restricting instantiation matters for functors that do not compile, or do not make sense, for every type, such as `std::exp` on `std::int64_t` or `-x` on a Bool. With plain `dispatch`, all four bodies must compile even if one is never called at runtime.

`convert_value` performs every store:

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

The branches are chosen at compile time with `if constexpr`, so the common case, same type in and out, compiles to a plain assignment. The second branch handles float to I64 through `saturating_cast`, which checks NaN with `value != value`, compares against `std::numeric_limits<To>::min()` converted to the float type, and clamps.

Conversion between two runtime dtypes, as in `to(DType)`, needs a double dispatch. `convert_strided` in `src/copy.cpp` nests one `dispatch` on the destination dtype inside one on the source dtype, producing 16 instantiations of the copy loop ([03. Views](@/book/03-views.md)):

```cpp
auto identity = [](auto value) { return value; };
dispatch(src.dtype(), [&]<typename From>() {
    dispatch(dst.dtype(), [&]<typename To>() { run_typed<To, From, 1>(iterator, identity); });
});
```

The functor is the identity. The conversion happens in `convert_value<To>` inside the typed inner loop.

### Building the iterator

`TensorIterator` is a builder. The caller adds an output or an output dtype, adds inputs, and calls `build`:

```cpp
TensorIterator iterator;
iterator.allocate_output(DType::F32).add_input(a).add_input(b).build();
```

`add_output(Tensor)` gives an existing tensor to write into. `allocate_output(DType)` asks the iterator to allocate one with the common shape. `build` runs four steps. From `src/iterator.cpp`:

```cpp
TensorIterator& TensorIterator::build() {
    check_inputs();
    const Shape common_shape = common_input_shape();
    prepare_output(common_shape);
    prepare_loop(common_shape);
    return *this;
}
```

`check_inputs` requires at least one input, an output or an output dtype, the same dtype on every input, and the same device on every operand. A mixed-dtype call throws `kael::TypeError` with "iterator inputs must share a dtype, got float32 and float64". The iterator does not convert dtypes; the kernel layer casts inputs before adding them. Keeping the inputs single-dtype keeps every typed inner loop down to one input type `In`.

`common_input_shape` folds `broadcast_shapes` over all input shapes. The output is not part of broadcasting.

`prepare_output` either allocates `empty(common_shape, dtype, device)` on the device of the first input, or checks that the given output has exactly the common shape. An in-place op cannot grow `self`: `a.add_(b)` with `a` of shape `[3]` and `b` of shape `[2, 3]` throws "output shape [3] does not match the shape [2, 3] of the inputs". The reverse direction works: `zeros({2, 3}).add_(ones({3}))` broadcasts the input to the output (`InPlace.OtherBroadcastsButSelfDoesNot`).

`prepare_loop` turns every operand into a base pointer and a byte-stride vector over the common shape:

```cpp
void TensorIterator::prepare_loop(const Shape& common_shape) {
    base_pointers_.clear();
    LoopDims full;
    full.shape = common_shape;
    full.operand_strides.push_back(to_bytes(output_->strides(), output_->dtype()));
    base_pointers_.push_back(output_->device_data());
    for (Tensor& input : inputs_) {
        full.operand_strides.push_back(to_bytes(
            broadcast_strides(input.shape(), input.strides(), common_shape), input.dtype()));
        base_pointers_.push_back(input.device_data());
    }

    element_count_ = numel(common_shape);
    dims_ = coalesce_dims(full);
    if (dims_.shape.empty()) {
        dims_.shape.push_back(1);
        for (Strides& strides : dims_.operand_strides) {
            strides.push_back(0);
        }
    }
}
```

`LoopDims` is a plain struct: a `Shape` and one `Strides` vector per operand, output first. Three decisions are visible here:

1. Broadcasting is the stride-0 trick from `expand`. `broadcast_strides` returns stride 0 for every dimension the input lacks or has with size 1.
2. Strides are converted to bytes by `to_bytes`, which multiplies each stride by `element_size(dtype)`. Operand 0 and operand 1 can have different element sizes.
3. The base pointer is `device_data()`, which already includes the tensor's offset. The loop only adds stride multiples.

If every dimension has size 1, or the tensor is 0-d, coalescing leaves no dimensions. `prepare_loop` then adds one dimension of size 1 with stride 0 for every operand, so the loop body always has an innermost dimension and `run_range` needs no special case.

`base_pointers_.clear()` at the start makes `build` safe to call twice (`Iterator.BuildingTwiceIsSafe`).

### coalesce_dims

From `src/iterator.cpp`:

```cpp
LoopDims coalesce_dims(const LoopDims& dims) {
    LoopDims merged = without_size_one_dims(dims);

    for (std::size_t inner = merged.shape.size(); inner-- > 1;) {
        const std::size_t outer = inner - 1;
        if (!can_merge_with_next(merged, outer)) {
            continue;
        }
        merged.shape[outer] *= merged.shape[inner];
        merged.shape.erase(merged.shape.begin() + static_cast<std::ptrdiff_t>(inner));
        for (Strides& strides : merged.operand_strides) {
            strides[outer] = strides[inner];
            strides.erase(strides.begin() + static_cast<std::ptrdiff_t>(inner));
        }
    }
    return merged;
}
```

First `without_size_one_dims` drops every size-1 dimension, because its stride is never used: index 0 times any stride is 0. Dropping them first also removes the arbitrary strides that size-1 dimensions often carry, which would otherwise block merges.

Then the loop goes from the innermost pair outward. `inner-- > 1` is the idiom for a reverse loop over unsigned indices that stops at 1. When a pair merges, the outer size becomes the product and the merged dimension keeps the inner stride, because the inner stride is the step between consecutive elements of the merged dimension. The next iteration compares the merged dimension with the one before it, so a fully contiguous tensor collapses to a single dimension in one pass.

`can_merge_with_next` applies the merge condition to all operands:

```cpp
bool can_merge_with_next(const LoopDims& dims, std::size_t outer) {
    return std::ranges::all_of(dims.operand_strides, [&](const Strides& strides) {
        return strides[outer] == strides[outer + 1] * dims.shape[outer + 1];
    });
}
```

`coalesce_dims` is a free function, not a member, so the tests can call it directly on hand-written `LoopDims` without building tensors.

### run and run_range

`TensorIterator::run(inner_loop)` executes the operation. The inner loop is any callable with this signature:

```cpp
void inner_loop(std::byte* const* pointers, const std::int64_t* strides, std::int64_t count);
```

It receives one pointer per operand (output first), the innermost byte stride of each operand, and the number of elements to process along the innermost dimension. The iterator handles everything else.

From `include/kael/iterator.h`:

```cpp
template <typename InnerLoop> void TensorIterator::run(InnerLoop&& inner_loop) const {
    check_runs_on_host();
    if (element_count_ == 0) {
        return;
    }
    parallel_for(0, element_count_, kElementGrain, [&](std::int64_t first, std::int64_t last) {
        run_range(first, last, inner_loop);
    });
}
```

`check_runs_on_host` throws if the output is on CUDA: the host loop cannot dereference device pointers. `run` is a template in the header, so the inner loop is inlined into `run_range` and the compiler sees the whole loop.

`parallel_for` ([08. CPU performance](@/book/08-cpu-performance.md)) splits `[0, element_count_)` into chunks of at least `kElementGrain` (524288) elements, the last one possibly smaller. A range no longer than the grain runs on the calling thread, so small tensors pay no threading cost. A chunk can start and end in the middle of a row, which is why `run_range` has a setup part and a main loop.

Setup decomposes `first` into an outer counter and a position inside the row:

```cpp
std::int64_t outer_position = first / inner_size;
for (std::size_t dim = outer_dims; dim-- > 0;) {
    counter[dim] = outer_position % dims_.shape[dim];
    outer_position /= dims_.shape[dim];
    for (std::size_t op = 0; op < operand_count; ++op) {
        byte_offsets[op] += counter[dim] * dims_.operand_strides[op][dim];
    }
}
std::int64_t skipped_in_row = first % inner_size;
```

`outer_dims` is the number of dimensions except the last, and `inner_size` is the size of the last one. This is `index_from_linear` applied to the row number, done once per chunk.

The main loop calls the inner loop on the rest of the current row, then advances the odometer:

```cpp
for (std::int64_t remaining = last - first; remaining > 0;) {
    const std::int64_t count = std::min(inner_size - skipped_in_row, remaining);
    for (std::size_t op = 0; op < operand_count; ++op) {
        pointers[op] =
            base_pointers_[op] + byte_offsets[op] + skipped_in_row * inner_strides[op];
    }
    inner_loop(pointers.data(), inner_strides.data(), count);
    remaining -= count;
    skipped_in_row = 0;

    for (std::size_t dim = outer_dims; dim-- > 0;) {
        ++counter[dim];
        for (std::size_t op = 0; op < operand_count; ++op) {
            byte_offsets[op] += dims_.operand_strides[op][dim];
        }
        if (counter[dim] < dims_.shape[dim]) {
            break;
        }
        for (std::size_t op = 0; op < operand_count; ++op) {
            byte_offsets[op] -= dims_.operand_strides[op][dim] * dims_.shape[dim];
        }
        counter[dim] = 0;
    }
}
```

The odometer increments the last outer counter and adds its stride. On overflow it subtracts `stride * size`, which returns that dimension to 0, resets the counter and carries into the next dimension. After the first call, `skipped_in_row` is 0, so every later call starts at the beginning of a row.

#### Trace

Take the coalesced worked example: loop shape `[6, 4]`, output strides `[16, 4]`, `y` strides `[0, 4]`. Assume a worker receives `first = 10`, `last = 24`. `inner_size` is 4, so setup computes row `10 / 4 = 2` and `skipped_in_row = 10 % 4 = 2`.

| Step | Counter | Output byte offset | `y` byte offset | `count` | Elements |
|------|---------|--------------------|-----------------|---------|----------|
| setup | `[2]` | 2 * 16 = 32 | 2 * 0 = 0 | | |
| call 1 | `[2]` | 32 + 2 * 4 = 40 | 0 + 2 * 4 = 8 | 2 | 10, 11 |
| call 2 | `[3]` | 48 | 0 | 4 | 12 to 15 |
| call 3 | `[4]` | 64 | 0 | 4 | 16 to 19 |
| call 4 | `[5]` | 80 | 0 | 4 | 20 to 23 |

Byte offset 40 in the output is element 10, and byte offset 8 in `y` is `y[2]`, which is the value element 10 needs (`10 % 4 == 2`). After call 4, `remaining` is 0 and the loop exits. Coalescing reduced the per-row bookkeeping from two counters to one.

A 0-d input traces the same way. `add(scalar(2), scalar(3))` has an empty common shape, so `prepare_loop` produced loop shape `[1]` with strides `[0]` for all operands. `element_count_` is `numel([]) = 1`. `run_range(0, 1)` has `outer_dims = 0`, `inner_size = 1`, and makes one call with `count = 1`.

### typed_inner_loop and run_typed

The kernel layer turns a value-level functor `f(x, y)` into an inner loop. From `include/kael/kernels.h`:

```cpp
template <typename Out, typename In, std::size_t... I, typename F>
void typed_inner_loop(std::byte* const* pointers, const std::int64_t* strides, std::int64_t count,
                      F& f, std::index_sequence<I...>) {
    const bool contiguous = strides[0] == static_cast<std::int64_t>(sizeof(Out)) &&
                            ((strides[I + 1] == static_cast<std::int64_t>(sizeof(In))) && ...);
    if (contiguous) {
        Out* out = reinterpret_cast<Out*>(pointers[0]);
        const In* const inputs[] = {reinterpret_cast<const In*>(pointers[I + 1])...};
        for (std::int64_t i = 0; i < count; ++i) {
            out[i] = convert_value<Out>(f(inputs[I][i]...));
        }
        return;
    }
    for (std::int64_t i = 0; i < count; ++i) {
        *reinterpret_cast<Out*>(pointers[0] + i * strides[0]) = convert_value<Out>(
            f(*reinterpret_cast<const In*>(pointers[I + 1] + i * strides[I + 1])...));
    }
}
```

The `std::index_sequence<I...>` expands the input pack. For a binary functor, `I...` is `0, 1`, so `f(inputs[I][i]...)` becomes `f(inputs[0][i], inputs[1][i])` and the fold expression checks `strides[1]` and `strides[2]`. One template handles unary (`InputCount = 1`), binary (2) and ternary (3) functors. All inputs share the type `In`; only the output can differ: `Out = std::uint8_t` for comparisons, any `To` for `convert_strided`.

The contiguous branch matters for speed. With typed pointers and unit stride, the compiler can vectorize the loop: it loads several elements into one SIMD register and applies the functor to all of them at once. The strided branch is the general case and handles stride 0 (broadcast) and any other step. After coalescing, most real workloads hit the contiguous branch for long rows.

| Case | Innermost byte strides (out, in...) | Branch |
|------|-------------------------------------|--------|
| `[2, 3, 4] + [2, 3, 4]` float32 | 4, 4, 4 | contiguous, 24 elements in one call |
| `[2, 3, 4] + [4]` float32 | 4, 4, 4 | contiguous, rows of 4 |
| `[3, 2]` transposed + `[3, 2]` float32 | 4, 12, 4 | strided |
| float32 `x + scalar(1.0)` | 4, 4, 0 | strided, because of the stride-0 scalar |

The last row shows a cost of the design: adding a broadcast scalar takes the strided branch even though the tensor operand is contiguous. The strided loop still runs one multiply-add of addresses per element, and the stride-0 load hits the same cache line every time.

`run_typed` connects the typed loop to the iterator and to the device backend:

```cpp
template <typename Out, typename In, std::size_t InputCount, typename F>
void run_typed(const TensorIterator& iterator, F& f) {
    if constexpr (DeviceFunction<F>) {
        if (iterator.on_device()) {
            device::elementwise(F::kOp, dtype_of_v<Out>, dtype_of_v<In>, iterator.device_layout(),
                                f.parameters());
            return;
        }
    }
    iterator.run([&](std::byte* const* pointers, const std::int64_t* strides, std::int64_t count) {
        typed_inner_loop<Out, In>(pointers, strides, count, f,
                                  std::make_index_sequence<InputCount>{});
    });
}
```

A functor that satisfies the `DeviceFunction` concept, meaning it has a static `kOp` convertible to `device::ElementOp` and a `parameters()` method, can run on CUDA. The iterator exports its coalesced layout as a `device::ElementLayout` with `device_layout()`, limited to `device::kMaxDims` (8) dimensions and `device::kMaxOperands` (4) operands, and the backend runs the same functor on the GPU. [09. CUDA backend](@/book/09-cuda-backend.md) covers that path. A plain lambda is not a `DeviceFunction`. Kernels built from one copy device inputs to the host, run there, and copy the result back.

### The kernel templates

The public templates in `include/kael/kernels.h` are thin wrappers over two helpers:

- `detail::same_dtype_kernel(dtype, f, inputs...)` opens a `NoGradGuard`, allocates the output with `allocate_output(dtype)`, adds each input as `input.to(dtype)`, builds, and dispatches `run_typed<T, T, N>`.
- `detail::same_dtype_kernel_out(out, f, inputs...)` does the same with a given output, casting inputs to `out.dtype()`.

| Template | Output dtype | Use |
|----------|--------------|-----|
| `unary_kernel(t, f)` | `t.dtype()` | `neg`, `abs`, `sign`, `relu` |
| `floating_unary_kernel(t, f)` | `t.dtype()` if floating, else F32 | `exp`, `log`, `tanh`, `sigmoid`, `gelu` |
| `unary_kernel_out(out, t, f)` | `out.dtype()` | writes into a view |
| `binary_kernel(a, b, f)` | `result_dtype({&a, &b})` | arithmetic |
| `binary_kernel_out(out, a, b, f)` | `out.dtype()` | in-place arithmetic |
| `ternary_kernel(a, b, c, f)` | `result_dtype({&a, &b, &c})` | `where` |
| `comparison_kernel(a, b, f)` | Bool | comparisons and logical ops |

`floating_unary_kernel` uses `dispatch_floating`, so its functor never sees an integer type. `comparison_kernel` computes in `result_dtype({&a, &b})` and dispatches `run_typed<std::uint8_t, T, 2>`: the inputs are `T`, the output is Bool.

Casting with `to(dtype)` before the iterator keeps the iterator single-dtype and keeps every functor monomorphic in `T`. The cost is a temporary copy of any input whose dtype differs from the result. `to` returns the tensor itself when the dtype already matches, so the common case pays nothing. The alternative, letting each operand keep its own dtype and converting inside the inner loop, is what PyTorch's CUDA elementwise kernels do. It avoids the temporary but adds a conversion per element and per operand, and the loop must handle every combination of input dtypes.

The `NoGradGuard` exists because these kernels are building blocks. The public op records one autograd node for the whole operation ([06. Autograd](@/book/06-autograd.md)), and the internal `to` calls must not record their own.

### Element functions

`src/internal/element_functions.h` defines one functor per elementwise operation in namespace `kael::functions`. Functors without runtime arguments inherit from `Tagged<ElementOp::...>`, which provides `kOp` and an empty `parameters()`. Functors with arguments, such as `Clamp` and `MaskedFill`, declare both themselves:

```cpp
template <ElementOp Op> struct Tagged {
    static constexpr ElementOp kOp = Op;
    [[nodiscard]] KAEL_HD ElementParameters parameters() const { return {}; }
};

struct Plus : Tagged<ElementOp::Plus> {
    template <typename T> KAEL_HD auto operator()(T a, T b) const {
        if constexpr (kWraps<T>) {
            return wrapping_add(a, b);
        } else {
            return a + b;
        }
    }
};
```

`KAEL_HD` expands to `__host__ __device__` when `__CUDACC__` is defined (under `nvcc`) and to nothing otherwise, so the CPU loop and the CUDA kernel compile the same arithmetic. Functors are structs instead of lambdas for that reason and because the `kOp` tag gives the device backend a value it can switch on. A lambda has no name the CUDA side could refer to.

Functors with runtime arguments store them as members and return them from `parameters()`. `Clamp<T>` holds `low` and `high`, and `MaskedFill<T>` holds `replacement`. `ElementParameters` keeps integer arguments in an `std::int64_t` array apart from the `double` array for real ones, because a `double` cannot hold every `int64` exactly: above 2^53, consecutive integers are no longer representable ([09. CUDA backend](@/book/09-cuda-backend.md)).

Several functors encode numerical or semantic decisions:

- `Sigmoid` uses `1 / (1 + exp(-x))` for `x >= 0` and `exp(x) / (1 + exp(x))` for `x < 0`, so `exp` never receives a large positive argument. For `x = -1000` the naive form computes `1 / (1 + exp(1000)) = 1 / inf = 0`, which happens to be right, but for `x = 1000` the second form would compute `inf / inf = NaN`. The split always evaluates the side that cannot overflow.
- `Maximum` and `Minimum` call `nan_aware_max` and `nan_aware_min`: if either value is NaN, the result is NaN. A plain `a < b ? b : a` would return the second argument whenever the first is NaN, because every comparison with NaN is false, making the result depend on argument order. `src/internal/nan_aware.h` has a host-only copy of the same functions, in `kael::detail`, used by reductions and softmax.
- `Plus`, `Minus`, `Times` and `Negate` compute signed integers through `wrapping_add`, `wrapping_subtract` and `wrapping_multiply`, which cast to the unsigned type, operate there and cast back. Signed overflow is undefined behavior in C++; unsigned arithmetic is defined modulo 2^64, so the unsigned detour makes it wrap around in two's complement, as PyTorch does. `Absolute` negates the same way, so `abs(INT64_MIN)` is `INT64_MIN`. The reductions, `matmul`, `cumsum` and `scatter_add_` (through `Plus` on the CPU, an unsigned `atomicAdd` on CUDA and `wrapping_add` in the host emulation) call the same helpers.
- `Power` uses `std::pow` for floating types and `integer_power` for integers: exponentiation by squaring in the unsigned type, so overflow wraps instead of being undefined. A negative integer exponent gives `1` for base 1, `1` or `-1` for base -1 depending on the parity of the exponent, and `0` otherwise.
- `Absolute` and `Sign` return their input unchanged for `std::uint8_t`, because Bool values are already 0 or 1.
- `Clamp` returns NaN if either bound is NaN.
- `GeluExact` uses `std::erf`; `GeluTanh` uses the tanh approximation with constant `0.044715`.

Exponentiation by squaring computes `3^5` in three steps instead of four multiplications:

```text
exponent 5 = 0b101    result 1, factor 3
bit 1: result = 1 * 3 = 3       exponent 2, factor 9
bit 0: result unchanged         exponent 1, factor 81
bit 1: result = 3 * 81 = 243    exponent 0, done
```

The number of multiplications is proportional to the number of bits of the exponent, not to its value.

## Edge cases and pitfalls

- **Empty tensors.** `element_count_` is 0, so `run` returns before calling `parallel_for`. The output still has the broadcast shape (`Kernels.EmptyAndScalarInputs`, `Iterator.ZeroSizedRunsNothing`).
- **0-d tensors.** Coalescing leaves no dimensions; `prepare_loop` inserts one dimension of size 1 and the inner loop runs once (`Iterator.ScalarsRunOneElement`).
- **Output aliasing an input.** Same layout is safe because each output element depends only on the input element at the same position, and that element is read before it is written (`Kernels.OutputCanBeOneOfTheInputs`). Different layouts over the same storage are not safe: writing `out[0, 1]` could overwrite an input element that a later position still needs. The in-place ops clone such inputs first ([03. Views](@/book/03-views.md)).
- **Output shape.** A given output never broadcasts. `prepare_output` throws if its shape differs from the common input shape (`Kernels.GivenOutputWithWrongShapeThrows`).
- **Mixed dtypes reaching the iterator.** `check_inputs` throws "iterator inputs must share a dtype". Callers cast first; the kernel templates do this.
- **Bool stores.** Every store goes through `convert_value`, so no Bool tensor ever holds a value other than 0 or 1. Code that writes into a Bool tensor through a raw `std::uint8_t*` bypasses that guarantee.
- **Float to integer stores.** NaN becomes 0 and out-of-range values saturate. The result is defined, but it is not an error: `to(DType::I64)` of a NaN tensor silently gives zeros.
- **Integer division by zero under threads.** `floor_divide` throws from inside the functor. The exception crosses the `parallel_for` boundary and reaches the caller (`ParallelIterator.IntegerDivisionByZeroStillThrows`).
- **Determinism.** Elementwise results do not depend on the thread count, because each element is computed independently (`ParallelIterator.ResultsDoNotDependOnThreadCount`).
- **Transposed operands are slower.** The iterator does not sort dimensions, so a transposed operand keeps a non-unit inner stride and forces the strided branch.
- **Device limits.** A layout with more than 8 dimensions after coalescing, or an op with more than 4 operands, throws in `device_layout()`. The CPU path has no such limit.

> [!WARNING]
> Mixed-dtype ops allocate a converted copy of each input whose dtype differs from the result. `int_tensor + float_tensor` on large tensors therefore needs memory for a float copy of `int_tensor` while the op runs.

## How it is tested

[`tests/test_iterator.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_iterator.cpp):

- `Coalesce.*` call `coalesce_dims` directly on hand-written `LoopDims`. `FullyContiguousBecomesOneLoop` turns `[2, 3, 4]` into `[24]`. `BroadcastOperandStopsTheMerge` turns `[2, 3, 4]` with a `[0, 4, 1]` operand into `[2, 12]`. `TransposedMergesNothing`, `SizeOneDimsDisappear`, `AllSizeOneLeavesNoDims` and `StrideZeroDimsMerge` cover the other rows of the table above.
- `Iterator.*` drive `TensorIterator` with a hand-written float inner loop: contiguous input is one dimension (`ContiguousIsOneDimension`), strides are in bytes per operand (`StridesAreInBytesPerOperand`, a Bool output with a float64 input gives strides 1 and 8), contiguous, broadcast and transposed adds, a three-dimensional odometer, zero-sized and scalar inputs, a scalar with a larger tensor, a transposed given output, building twice, and the error cases in `Iterator.Errors` (wrong output shape, mixed dtypes, shapes that do not broadcast, no input).
- `Promotion.Table` and `Promotion.Scalars` are `static_assert`s on `promote_types` and `promote_with_scalar`. They fail at compile time, not at test time.

[`tests/test_kernels.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_kernels.cpp):

- `Kernels.RandomCasesMatchTheNaiveReference` generates 500 random pairs of broadcastable shapes (0 to 4 dims, sizes 0 to 5) with random layouts (transposed, stepped, expanded or plain) and compares `binary_kernel` against a reference that indexes every element with `index_from_linear` and `at<double>`. The generator is seeded with 2024, so failures reproduce. Every combination of broadcasting, stride and coalescing path is exercised against code that has no fast path at all.
- The other `Kernels.*` tests cover promotion, ternary broadcasting, Bool output, writing through views, output aliasing and the output-shape check. `StridedCopy.ConvertsThroughIterator` checks `to` and `clone` on a transpose.

[`tests/test_parallel_ops.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_parallel_ops.cpp) has `ParallelIterator.*`: large ranges are split, small ranges stay in one call, every layout visits every element in its place under 4 threads, results do not depend on the thread count, and exceptions propagate.

The golden tests of [05. Operations](@/book/05-operations.md) check promotion against PyTorch for every binary op: the comparison in `tests/test_golden.cpp` requires the same dtype, not only the same values.

> [!TIP]
> The random reference test is the pattern to copy when you add a fast path. Write the slowest correct version you can, then compare the fast path against it on randomly generated layouts.

## Build it yourself

1. Write `promote_types`, `dtype_category`, `saturating_cast`, `convert_value` and `dispatch` in `dtype.h`. Checkpoint: `Promotion.Table` compiles.
2. Write `Scalar` and `promote_with_scalar`. Checkpoint: `Promotion.Scalars` compiles.
3. Write `LoopDims` and `coalesce_dims` with the helpers `without_size_one_dims` and `can_merge_with_next`. Checkpoint: all six `Coalesce.*` tests pass.
4. Write `TensorIterator` with `add_output`, `allocate_output`, `add_input`, `build` and the private steps `check_inputs`, `common_input_shape`, `prepare_output`, `prepare_loop`. Checkpoint: `Iterator.ContiguousIsOneDimension` and `Iterator.StridesAreInBytesPerOperand`.
5. Write `run` and `run_range` without threads first: call `run_range(0, element_count_, inner_loop)`. Checkpoint: the remaining `Iterator.*` tests pass, including `ThreeDimensionalOdometer` and `GivenOutputIsWrittenInItsOrder`.
6. Write `typed_inner_loop` and `run_typed` (host branch only), then `same_dtype_kernel`, `same_dtype_kernel_out`, `result_dtype` and the public templates. Checkpoint: `Kernels.*` pass, including `RandomCasesMatchTheNaiveReference`.
7. Replace the temporary loop in `convert_strided` ([03. Views](@/book/03-views.md)) with the iterator. Checkpoint: `StridedCopy.ConvertsThroughIterator` and `ViewChain.GeneratedNumPyCases`.
8. Add `src/internal/element_functions.h` with `Plus`, `Minus`, `Times`, `Divides` and the comparison functors. Checkpoint: `binary_kernel(a, b, functions::Plus{})` gives the same result as the lambda version.
9. When [08. CPU performance](@/book/08-cpu-performance.md) adds `parallel_for`, wrap `run_range` in it. Checkpoint: `ParallelIterator.*`.

```bash
cmake --build build
build/tests/kael_tests --gtest_filter='Coalesce.*:Iterator.*:Promotion.*:Kernels.*:StridedCopy.*'
```

## Check your understanding

<details>
<summary>Why does the iterator store strides in bytes instead of elements?</summary>

Operands of one loop can have different element sizes. When `to(DType::F32)` converts an I64 tensor, the output steps 4 bytes per element and the input steps 8. With byte strides, each operand is a `std::byte*` plus a stride vector, and the iterator can advance every pointer without knowing any C++ type. The typed inner loop is the only place that reinterprets the bytes.

</details>

<details>
<summary>A contiguous [2, 3, 4] tensor is added to a contiguous [4] tensor. Why does the loop shape become [6, 4] and not [24]?</summary>

The `[4]` operand has byte strides `[0, 0, 4]` after broadcasting. Merging the last two dimensions needs `0 == 4 * 4` for that operand, which is false, so the innermost dimension stays separate. The first two dimensions merge because `0 == 0 * 3` holds for the broadcast operand and `48 == 16 * 3` holds for the others. All operands share one loop, so one operand that cannot merge blocks the merge for all.

</details>

<details>
<summary>What dtype does ones({3}, DType::I64) + scalar(1.0, DType::F64) produce, and why?</summary>

float64. The dimensioned group is I64 (category 1) and the 0-d group is F64 (category 2). A 0-d tensor wins when its category is strictly higher, and it then contributes its own dtype, F64. Compare `randn({3}) + scalar(1.0, DType::F64)`: both are category 2, so the dimensioned F32 tensor decides and the result stays float32.

</details>

<details>
<summary>Why does run split the work by linear element index, and what does that force run_range to handle?</summary>

Splitting by element keeps the load even whatever the loop shape: a `[2, 1000000]` loop has only two rows, which would leave most threads idle if work were split by row. The price is that a chunk can start and end in the middle of a row. `run_range` therefore decomposes `first` into an odometer position plus `skipped_in_row`, calls the inner loop on the partial first row, and clips the last call with `std::min(..., remaining)`.

</details>

<details>
<summary>Why does convert_value not use static_cast when the target is Bool?</summary>

Bool is stored as `std::uint8_t`. `static_cast<std::uint8_t>(2.5)` gives 2, which is not a valid Bool value, and `logical_*` and `equal` would then treat two "true" values as different bytes. `convert_value` maps any nonzero value, including NaN, to 1 and zero to 0.

</details>

<details>
<summary>Why are the element functions structs with a kOp tag instead of lambdas?</summary>

The same arithmetic must compile for the CPU loop and for a CUDA kernel. A struct with `KAEL_HD` members compiles under both compilers, and its static `kOp` gives the device backend an `ElementOp` value it can switch on to pick the matching GPU kernel. Arguments such as clamp bounds travel through `parameters()`. A lambda has no name or tag the backend could refer to, so kernels built from lambdas run on the host only.

</details>

## Further reading

- Edward Z. Yang, ["PyTorch internals"](http://blog.ezyang.com/2019/05/pytorch-internals/): the section on TensorIterator and kernel dispatch describes the design kael simplifies here.
- PyTorch documentation, ["Tensor Attributes"](https://pytorch.org/docs/stable/tensor_attributes.html): the type promotion section states the category and 0-d rules that kael follows.
- PyTorch source, [`aten/src/ATen/TensorIterator.h`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/TensorIterator.h): the production version, including dimension reordering and per-operand dtypes that kael leaves out.
- NumPy documentation, ["Broadcasting"](https://numpy.org/doc/stable/user/basics.broadcasting.html): the shape rules the iterator implements.
- NumPy documentation, ["Iterating over arrays"](https://numpy.org/doc/stable/reference/arrays.nditer.html): `nditer`, NumPy's equivalent of the iterator, with its own coalescing and external inner loop.
- [ADR 0001: runtime dtype](@/adr/0001-runtime-dtype.md): why dispatch happens at runtime in the first place.

Next: [05. Operations](@/book/05-operations.md)
