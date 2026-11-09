+++
title = "Shapes and strides"
description = "The index arithmetic that turns a flat buffer into an n-dimensional array: element counts, strides, offsets, broadcasting and dimension wrapping."
weight = 1
slug = "shapes-and-strides"

[extra]
chapter = "01"
kicker = "Chapter 01"
builds = "The index arithmetic that turns a flat buffer into an n-dimensional array, and broadcasting"
source = "docs/01-shapes-and-strides.md"
tag = "chapter-02-shapes"
+++

This chapter builds the metadata arithmetic that turns a flat block of memory into an n-dimensional tensor. It maps to phase 2 of the project and the tag `chapter-02-shapes`. There is no `Tensor` class yet: everything here is a set of free functions over two vector types, and every later layer (views, the iterator, broadcasting in operations, autograd's gradient reduction) calls them.

| File | Role |
|------|------|
| [`include/kael/shape.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/shape.h) | `Shape` and `Strides` type aliases and the declarations of every function in this chapter |
| [`src/shape.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/shape.cpp) | Implementation: validation, element count, contiguous strides, offsets, broadcasting, dimension wrapping, shape inference |
| [`src/internal/checked_math.h`](https://github.com/JetBerri/kael.cpp/blob/main/src/internal/checked_math.h) | `detail::checked_mul` and `detail::checked_add`, overflow-checked arithmetic on `std::int64_t` |
| [`include/kael/error.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/error.h) | `kael::Error`, its subclasses and the `KAEL_CHECK_*` macros used to report bad input |
| [`scripts/generate/broadcast_cases.py`](https://github.com/JetBerri/kael.cpp/blob/main/scripts/generate/broadcast_cases.py) | Generates random broadcasting cases with NumPy |
| [`tests/generated/broadcast_cases.inc`](https://github.com/JetBerri/kael.cpp/blob/main/tests/generated/broadcast_cases.inc) | The generated cases, included by the test |
| [`tests/test_shape.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_shape.cpp) | GoogleTest suites for every function except `infer_shape` and `dim_position` |

Three functions in `shape.h` arrived in later phases but belong to the same family: `wrap_dim` came with the `Tensor` handle in phase 3, `infer_shape` came with `view` and `reshape` in phase 4, and `dim_position` came with a refactor in phase 6. This chapter covers them so that `shape.h` is complete.

## Prerequisites

- [00. Overview](@/book/00-overview.md) for the build commands and the layout of the repository.
- Familiarity with C++ `std::vector`, `std::span` and exceptions. No tensor knowledge is assumed.

## Why this layer exists

Memory is one-dimensional. A computer gives you a range of addresses, and the only thing you can do with an address is add an integer to it. A matrix, an image batch with shape `[batch, channels, height, width]`, or the attention scores of a transformer are all n-dimensional. Something has to translate "row 1, column 2" into "the sixth number in this buffer". That translation is the subject of this chapter.

The naive answer is to hard-code it: a matrix class that stores `rows` and `cols` and computes `row * cols + col`. That works for one layout of one rank, and breaks the moment you want anything else:

- A transpose has to copy every element into a new buffer, because the formula only describes row-major order.
- A slice such as "column 2 of this matrix" has to copy too, because its elements are not adjacent.
- Adding a bias vector to every row of a matrix has to materialize the vector once per row first.
- Every rank needs its own formula, so a 3-d and a 4-d tensor need different code.

The fix is to describe the layout with numbers instead of code. A tensor carries a **shape** (how many elements along each dimension) and **strides** (how far to move in memory for one step along each dimension). A single formula turns any index into a memory position for any rank and any layout. Transpose, slice and broadcast then become operations on those numbers, with no data movement. NumPy, PyTorch and every serious array library work this way.

The mental model to hold on to:

```text
data:      a flat buffer, never reshaped, rarely copied
metadata:  shape + strides + offset, cheap to create, describes how to read the data

offset of element [i0, i1, ..., in] = offset + i0*s0 + i1*s1 + ... + in*sn
```

Everything in this chapter is arithmetic on the metadata. If that arithmetic is wrong, every later chapter reads the wrong element, and it does so silently. That is why this layer validates every input, checks every multiplication for overflow and is tested against NumPy.

## Concepts

### A tensor is a buffer plus metadata

In memory, a tensor is a flat array of numbers plus a few values that say how to read it:

- **Shape**: the number of elements along each dimension. A matrix with 2 rows and 3 columns has shape `[2, 3]`. The number of dimensions is the **rank**, or `ndim`.
- **Strides**: for each dimension, how many elements to skip in the flat array to move one step along that dimension.
- **Offset**: the position of the first element inside the buffer.

```text
buffer:  [ 1.0  2.0  3.0  4.0  5.0  6.0 ]

shape:   [2, 3]
strides: [3, 1]
offset:  0
```

The buffer stores the data. The metadata stores the interpretation. Keeping the two apart is what lets later chapters transpose, slice and broadcast a tensor without copying any data: those operations only produce new metadata over the same buffer.

kael measures strides in **elements**, like PyTorch. NumPy measures them in **bytes**. The same float32 matrix reports different numbers in each:

```text
                               shape     strides
kael     zeros({2, 3})         [2, 3]    [3, 1]      elements
PyTorch  torch.zeros(2, 3)     (2, 3)    (3, 1)      elements
NumPy    np.zeros((2, 3), np.float32)    (12, 4)     bytes
```

Element strides keep this chapter independent of the element type. The conversion to bytes happens once, in the iterator of [chapter 04](@/book/04-tensor-iterator.md), which multiplies each stride by the element size before running a loop.

### Shape and element count

The element count, `numel`, is the product of the dimensions.

```text
shape        numel
[2, 3, 4]    24
[5]          5
[]           1
[3, 0, 2]    0
```

Two rows deserve attention.

The shape `[]` has no dimensions. The product of an empty list is 1, so it holds exactly one element. This is a **0-d tensor**, also called a scalar tensor. It is what `sum()` over all dimensions returns in PyTorch. It is not a special case in the code: the loop in `numel` runs zero times and returns its initial value.

The shape `[3, 0, 2]` has a dimension of size 0, so it holds no elements. Empty tensors are legal. A batch with no samples or a slice with `start == end` produces one, and NumPy and PyTorch both allow them. Only negative dimensions are an error.

Do not confuse `[]` with `[0]`. The first has rank 0 and one element. The second has rank 1 and no elements.

### Row-major layout

A matrix of 2 rows and 3 columns is stored one row after the other. This is **row-major order**, the default in C, NumPy and PyTorch. The alternative, column-major order, stores one column after the other and is the default in Fortran, MATLAB and most BLAS libraries.

```text
logical view (shape [2, 3]):

        col 0   col 1   col 2
row 0 [  a       b       c   ]
row 1 [  d       e       f   ]

flat memory, row-major:

position:  0    1    2    3    4    5
value:   [ a    b    c    d    e    f ]
```

To move one column to the right, skip 1 element. To move one row down, skip 3 elements, a whole row. Those distances are the strides: `[3, 1]`. A tensor whose strides are exactly the row-major ones is called **contiguous**.

The strides of a contiguous layout come from a running product, built from the last dimension to the first:

```text
shape [2, 3, 4]

start with stride = 1
dim 2 (size 4): strides[2] = 1    then stride = 1 * 4  = 4
dim 1 (size 3): strides[1] = 4    then stride = 4 * 3  = 12
dim 0 (size 2): strides[0] = 12   then stride = 12 * 2 = 24

strides = [12, 4, 1]
```

The last dimension always has stride 1 because its elements are adjacent. Each dimension to its left skips everything to its right: one step along dimension 0 skips a whole `[3, 4]` block of 12 elements. The final value of the running product, 24, is the element count. It is computed but not stored.

A size-1 dimension does not change the running product. For shape `[3, 1, 2]` the strides are `[2, 2, 1]`: dimension 1 has stride 2, but since its only valid index is 0, that stride never moves the position.

### From index to offset

Given a multidimensional index, the position of the element in the flat buffer is:

```text
offset = base_offset + sum over i of index[i] * strides[i]
```

Worked example. Shape `[2, 3, 4]`, strides `[12, 4, 1]`, base offset 0, index `[1, 0, 1]`:

```text
dimension 0:  1 * 12 = 12
dimension 1:  0 *  4 =  0
dimension 2:  1 *  1 =  1
                     -----
offset             =  13
```

The same formula works for any strides, not only contiguous ones. That is the property the rest of the library relies on: views change the strides and the base offset, and element access keeps using this formula.

A second example shows what the base offset is for. Take the `[2, 3]` matrix above and describe its middle column as a 1-d tensor, without copying:

```text
buffer:    [ a  b  c  d  e  f ]
             0  1  2  3  4  5

column 1:  shape [2], strides [3], base offset 1

index [0]:  1 + 0 * 3 = 1   -> b
index [1]:  1 + 1 * 3 = 4   -> e
```

The base offset moves the start to `b`, and the stride of 3 jumps one row at a time. This is what `t.slice(1, 1, 2)` and NumPy's `a[:, 1]` produce. [Chapter 03](@/book/03-views.md) builds those views.

### From linear position to index

The inverse problem: given a position `linear` in row-major order, recover the multidimensional index. Walk the shape from right to left. The remainder of the division by the dimension size is the coordinate in that dimension. The quotient is what remains for the dimensions to the left. This is the same procedure as writing a number in a mixed-radix base, where each digit has its own base: seconds, minutes and hours are a mixed-radix system with bases 60, 60 and 24.

Linear position 13 in shape `[2, 3, 4]`:

```text
dimension 2 (size 4):  13 % 4 = 1    remaining 13 / 4 = 3
dimension 1 (size 3):   3 % 3 = 0    remaining  3 / 3 = 1
dimension 0 (size 2):   1 % 2 = 1    remaining  1 / 2 = 0

index = [1, 0, 1]
```

That is the index from the previous section, which confirms that the two functions are inverses for contiguous strides. A second trace with a size-1 dimension, position 5 in shape `[3, 1, 2]`:

```text
dimension 2 (size 2):  5 % 2 = 1    remaining 5 / 2 = 2
dimension 1 (size 1):  2 % 1 = 0    remaining 2 / 1 = 2
dimension 0 (size 3):  2 % 3 = 2    remaining 2 / 3 = 0

index = [2, 0, 1]
```

The size-1 dimension always yields coordinate 0 and passes the quotient through unchanged.

### Transposing without copying

Keeping data and metadata apart pays off here. Take the same six values:

```text
buffer: [ a  b  c  d  e  f ]
```

Read with shape `[2, 3]` and strides `[3, 1]`:

```text
[ a  b  c ]
[ d  e  f ]
```

Read with shape `[3, 2]` and strides `[1, 3]`:

```text
[ a  d ]
[ b  e ]
[ c  f ]
```

The second reading is the transpose, and no value moved. Check with the offset formula: element `[2, 1]` of the transposed reading is at `2 * 1 + 1 * 3 = 5`, which holds `f`. In the original, `f` is at `[1, 2]`, which is `[2, 1]` with the coordinates swapped.

Swapping two entries of the shape and the same two entries of the strides is the whole transpose. The result is no longer contiguous: its strides `[1, 3]` are not the row-major strides `[2, 1]` of shape `[3, 2]`. Walking it in index order jumps around in memory (positions 0, 3, 1, 4, 2, 5), which is why some operations later call `contiguous()` to copy such a tensor into row-major order first. [Chapter 03](@/book/03-views.md) builds `transpose`, `permute`, `slice` and the other views on this idea.

### Broadcasting shapes

**Broadcasting** lets an operation combine tensors of different shapes, for example adding a `[3]` bias to every row of a `[2, 3]` matrix, without first copying the bias into two rows. It is a rule that combines two shapes into one result shape.

The rules, identical to NumPy's and PyTorch's:

1. Align the shapes from the right. If one shape has fewer dimensions, treat the missing leading dimensions as size 1.
2. Two sizes are compatible if they are equal or if one of them is 1.
3. The result size is the larger of the two, or the common value when they are equal.

If any pair fails rule 2, the shapes are not broadcastable.

```text
a:       [5, 1, 4]
b:          [3, 1]      missing leading dimension counts as 1

aligned:
a:        5   1   4
b:        1   3   1
          |   |   |
result:   5   3   4
```

Alignment from the right is a convention with a reason: the last dimensions usually hold the features (the columns of a matrix, the channels of a pixel), and a smaller tensor usually describes something per feature. A `[3]` bias lines up with the 3 columns of `[2, 3]`, not with its 2 rows.

Size 0 follows the same rules, because the rule says "one of them is 1", not "the smaller one". `[0]` with `[1]` gives `[0]`, but `[0]` with `[3]` fails. NumPy behaves the same way.

| a | b | result |
|---|---|--------|
| `[3]` | `[2, 3]` | `[2, 3]` |
| `[2, 1]` | `[1, 3]` | `[2, 3]` |
| `[5, 1, 4]` | `[3, 1]` | `[5, 3, 4]` |
| `[]` | `[4]` | `[4]` |
| `[]` | `[]` | `[]` |
| `[0]` | `[1]` | `[0]` |
| `[0]` | `[3]` | error |
| `[2, 3]` | `[3, 2]` | error |

The `[2, 1]` with `[1, 3]` row shows that broadcasting can stretch both inputs at once. The result is an outer combination: every row of `a` meets every column of `b`.

### Broadcasting strides

Knowing the result shape is half of broadcasting. The other half is reading the smaller tensor as if it had the larger shape. The key is a **stride of 0**.

Take a vector `v` with shape `[3]` and strides `[1]`, and read it as `[2, 3]` with strides `[0, 1]`:

```text
buffer of v:  [ 10  20  30 ]

broadcast strides: [0, 1]

           j=0   j=1   j=2
row i=0  [ 10    20    30 ]     offset = 0*i + 1*j = j
row i=1  [ 10    20    30 ]     same offsets, same data
```

Moving down a row skips 0 elements, so both rows read the same three values. The result looks like a `[2, 3]` tensor and still owns three values.

A column works the same way. Shape `[2, 1]` with strides `[1, 1]` read as `[2, 4]` gets strides `[1, 0]`: moving along a row skips 0 elements, so every column repeats the single value of its row.

```text
buffer of col:  [ 7  9 ]

broadcast strides: [1, 0]

           j=0  j=1  j=2  j=3
row i=0  [  7    7    7    7 ]    offset = 1*i + 0*j = 0
row i=1  [  9    9    9    9 ]    offset = 1
```

The rule, per dimension and aligned from the right:

| Input size | Target size | Result stride |
|------------|-------------|---------------|
| missing (new leading dimension) | any | 0 |
| equal to target | n | original stride |
| 1 | `n != 1` | 0 |
| anything else | n | error |

A full worked example puts both halves together. Compute `m + v` with `m` of shape `[2, 3]` holding `1 2 3 / 4 5 6`, and `v` of shape `[3]` holding `10 20 30`:

```text
result shape:   broadcast_shapes([2, 3], [3]) = [2, 3]
m strides:      broadcast_strides([2, 3], [3, 1], [2, 3]) = [3, 1]
v strides:      broadcast_strides([3],    [1],    [2, 3]) = [0, 1]

index    offset in m       offset in v       m + v
[0, 0]   0*3 + 0*1 = 0     0*0 + 0*1 = 0     1 + 10 = 11
[0, 1]   0*3 + 1*1 = 1     0*0 + 1*1 = 1     2 + 20 = 22
[0, 2]                2                  2   3 + 30 = 33
[1, 0]   1*3 + 0*1 = 3     1*0 + 0*1 = 0     4 + 10 = 14
[1, 1]                4                  1   5 + 20 = 25
[1, 2]                5                  2   6 + 30 = 36
```

No copy of `v` exists at any point. The iterator of [chapter 04](@/book/04-tensor-iterator.md) runs exactly this loop, with faster bookkeeping.

A tensor with stride 0 in a dimension larger than 1 has **internal overlap**: several logical positions map to one memory location. Reading through it is safe. Writing through it is not, because two writes land on the same element and the result depends on which one runs last.

### Dimension arguments

Many operations take a dimension argument: `sum(dims)`, `transpose(dim0, dim1)`, `size(dim)`. kael follows Python and PyTorch and accepts negative dimensions, where `-1` means the last dimension and `-ndim` the first.

```text
ndim = 3
dimension argument:   -3  -2  -1   0   1   2
position in shape:     0   1   2   0   1   2
```

Anything outside `[-ndim, ndim - 1]` is an error. A 0-d tensor has no dimensions, so every dimension argument is an error for it.

### Shape inference

`view` and `reshape` accept one `-1` in the requested shape, meaning "whatever size makes the element count match". `view({-1, 4})` on a 12-element tensor becomes `[3, 4]`, because the known dimensions multiply to 4 and `12 / 4 = 3`. If the known product does not divide the element count, there is no valid answer and the request fails.

## Implementation

The code follows the order of the concepts: types, error reporting, element count with its overflow check, strides, the two index conversions, broadcasting, and dimension handling.

### The types

kael represents shapes and strides as plain vectors of signed 64-bit integers.

From `include/kael/shape.h`:

```cpp
using Shape = std::vector<std::int64_t>;
using Strides = std::vector<std::int64_t>;
```

The two aliases name the same type, so the compiler does not stop you from passing strides where a shape is expected. The aliases exist for the reader. A strong type per concept would catch that mix-up, but it would also force conversions at every call site that does arithmetic between the two, and the tests catch the mistake anyway. PyTorch makes the same trade: its `IntArrayRef` serves for sizes and strides alike.

A `std::vector` allocates on the heap. Inline storage would avoid that, and PyTorch keeps up to five sizes and strides inline in each `TensorImpl` (`c10::impl::SizesAndStrides`). kael keeps the standard type because shape metadata is never in an inner loop: it is computed once per operation, and the element loops of chapter 04 run on precomputed integers.

The element type is `std::int64_t` for three reasons:

- Products of dimensions grow fast. A `[50000, 50000]` float tensor has 2.5 billion elements, which overflows a 32-bit `int` (maximum about 2.1 billion).
- Strides, offsets and indices take part in the same arithmetic. One signed width for all of them avoids mixed-width conversions, which `-Wconversion -Wsign-conversion` would flag on every line.
- Signed values let the validation code detect a negative dimension instead of seeing a huge unsigned number. A `std::size_t` dimension of `-1` would silently become 18446744073709551615.

The full public API of this chapter, from `include/kael/shape.h`:

```cpp
std::string to_string(const Shape& shape);
void validate_shape(const Shape& shape);
std::int64_t numel(const Shape& shape);
Strides contiguous_strides(const Shape& shape);

std::int64_t offset_from_index(std::span<const std::int64_t> shape,
                               std::span<const std::int64_t> strides, std::int64_t base_offset,
                               std::span<const std::int64_t> index);
Shape index_from_linear(std::int64_t linear, const Shape& shape);

Shape broadcast_shapes(const Shape& a, const Shape& b);
Strides broadcast_strides(const Shape& shape, const Strides& strides, const Shape& target);

std::int64_t wrap_dim(std::int64_t dim, std::int64_t ndim);
std::size_t dim_position(std::int64_t dim, std::int64_t ndim);
Shape infer_shape(const Shape& requested, std::int64_t element_count);
```

Every function is free and stateless. None of them knows about storage, element types or devices, which is why they can be tested without a single tensor.

### Error reporting

Every check in this chapter goes through a macro from `include/kael/error.h`. The exception hierarchy is small:

```text
std::runtime_error
  kael::Error
    kael::IndexError     an index or dimension is out of range
    kael::ValueError     a shape, size or argument is invalid
    kael::TypeError      a dtype is not supported
```

The macros, from `include/kael/error.h`:

```cpp
#define KAEL_CHECK_AS(exception, cond, msg)                                                        \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            ::std::ostringstream kael_check_stream_;                                               \
            kael_check_stream_ << msg;                                                             \
            ::kael::detail::fail<exception>(__FILE__, __LINE__, #cond, kael_check_stream_.str());  \
        }                                                                                          \
    } while (0)

#define KAEL_CHECK(cond, msg) KAEL_CHECK_AS(::kael::Error, cond, msg)
#define KAEL_CHECK_INDEX(cond, msg) KAEL_CHECK_AS(::kael::IndexError, cond, msg)
#define KAEL_CHECK_VALUE(cond, msg) KAEL_CHECK_AS(::kael::ValueError, cond, msg)
#define KAEL_CHECK_TYPE(cond, msg) KAEL_CHECK_AS(::kael::TypeError, cond, msg)
```

Three details make the macro worth its weight:

- `msg` is pasted into a stream expression, so a call can write `"index " << i << " out of range"`. The message is only built when the check fails, so a passing check costs one comparison.
- `#cond` turns the condition into a string, and `__FILE__` and `__LINE__` locate it. The message says what failed and where, without a debugger.
- The `do { ... } while (0)` wrapper makes the macro a single statement, so it behaves correctly inside an `if` without braces.

`detail::fail` formats `kael error at <file>:<line>: check '<cond>' failed: <message>` and throws the requested exception type. It is marked `[[noreturn]]`, so the compiler knows the code after a failed check is unreachable. [Chapter 02](@/book/02-memory-model.md) uses the same macros for storage and tensor checks.

### Validation and `to_string`

`validate_shape` rejects negative dimensions and nothing else.

From `src/shape.cpp`:

```cpp
void validate_shape(const Shape& shape) {
    for (std::size_t i = 0; i < shape.size(); ++i) {
        KAEL_CHECK_VALUE(shape[i] >= 0,
                         "negative dimension: " << shape[i] << " at position " << i
                                                << " in shape: " << to_string(shape));
    }
}
```

`KAEL_CHECK_VALUE` throws `kael::ValueError`. For a shape `[2, -2]`, the message reads:

```text
kael error at <path>/src/shape.cpp:<line>: check 'shape[i] >= 0' failed: negative dimension: -2 at position 1 in shape: [2, -2]
```

`to_string(const Shape&)` formats a shape as `[2, 3]`, and `[]` for a 0-d shape. It delegates to a `join` helper in an anonymous namespace that takes a `std::span<const std::int64_t>`, so the same formatting serves shapes, strides and indices. Every error message in kael uses it, so shapes always print the same way and a test can search a message for `"[2, -2]"`.

### `numel` and the overflow check

`numel` multiplies the dimensions, and the product can exceed the range of `std::int64_t`. Signed overflow in C++ is **undefined behavior**: the standard places no requirement on what the program does. In practice the optimizer assumes it never happens, so the code cannot multiply first and check the result afterwards. A check such as `if (total * dim < total)` can be deleted by the compiler, because under that assumption the condition is always false.

The check must run before each multiplication. If `total * dim` would exceed the maximum `max`, then `total > max / dim`, and that division cannot overflow. Integer division rounds down, which keeps the test exact: `total <= max / dim` holds if and only if `total * dim <= max` for positive values.

From `src/shape.cpp`:

```cpp
std::int64_t numel(const Shape& shape) {
    validate_shape(shape);

    if (std::ranges::find(shape, 0) != shape.end()) {
        return 0;
    }

    std::int64_t total = 1;
    for (const std::int64_t dim : shape) {
        KAEL_CHECK_VALUE(total <= std::numeric_limits<std::int64_t>::max() / dim,
                         "number of elements overflows int64 for shape " << to_string(shape));
        total *= dim;
    }
    return total;
}
```

The early return for a zero dimension does two jobs. It gives the right answer (any product with a 0 is 0), and it guarantees that `dim` is at least 1 inside the loop, so `max / dim` never divides by zero.

The order also matters for correctness. Without the early return, `numel({1LL << 40, 1LL << 40, 0})` would throw on the first two dimensions even though the tensor has no elements.

A worked boundary, from the `Numel.Overflow` test. `max` is `2^63 - 1`, which is 9223372036854775807:

```text
numel({max})         = max                (1 <= max / max = 1, fits exactly)
numel({max, 2})      -> throws            (max <= max / 2 is false)
numel({2^31, 2^31})  = 2^62               (2^31 <= max / 2^31 = 2^32 - 1)
numel({2^40, 2^40})  -> throws            (2^40 <= max / 2^40 = 2^23 - 1 is false)
```

### Checked arithmetic helpers

The same pattern repeats in several places, so `src/internal/checked_math.h` packages it for non-negative operands.

From `src/internal/checked_math.h`:

```cpp
inline std::int64_t checked_mul(std::int64_t a, std::int64_t b, const char* quantity) {
    KAEL_CHECK(a >= 0 && b >= 0, "checked_mul expects non-negative operands for " << quantity);
    if (a == 0 || b == 0) {
        return 0;
    }
    KAEL_CHECK_VALUE(a <= std::numeric_limits<std::int64_t>::max() / b,
                     quantity << " overflows int64 (" << a << " * " << b << ")");
    return a * b;
}

inline std::int64_t checked_add(std::int64_t a, std::int64_t b, const char* quantity) {
    KAEL_CHECK(a >= 0 && b >= 0, "checked_add expects non-negative operands for " << quantity);
    KAEL_CHECK_VALUE(a <= std::numeric_limits<std::int64_t>::max() - b,
                     quantity << " overflows int64 (" << a << " + " << b << ")");
    return a + b;
}
```

The two checks fail with different types on purpose. A negative operand is a bug in kael itself, so it throws the base `kael::Error`. An overflow is a consequence of the user's input, so it throws `ValueError`. The `quantity` argument names what is being computed (`"tensor stride"`, `"tensor size in bytes"`, `"slice offset"`), so the error message tells you which calculation overflowed.

The header is private: it lives under `src/`, not `include/`, because it is an implementation detail and not part of the public API. Chapter 02 uses it to check that a tensor's extent fits in its storage, and chapter 03 uses it for slice offsets and strides.

C++ compilers offer builtins such as `__builtin_mul_overflow` that do the same job in one instruction. kael uses the portable division form because it is standard C++, reads the same on every compiler, and the cost does not matter outside inner loops.

### `contiguous_strides`

From `src/shape.cpp`:

```cpp
Strides contiguous_strides(const Shape& shape) {
    validate_shape(shape);

    Strides strides(shape.size());
    std::int64_t stride = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride = detail::checked_mul(stride, std::max<std::int64_t>(shape[i], 1), "tensor stride");
    }
    return strides;
}
```

The loop header `for (std::size_t i = shape.size(); i-- > 0;)` is the idiom for counting down with an unsigned index. The test `i-- > 0` compares before decrementing, so the body sees `size - 1` down to `0` and the loop ends without wrapping around. For a 0-d shape, `i` starts at 0, the test fails immediately, and the result is the empty strides `[]`.

#### Why `max(dim, 1)`

The running product multiplies by `max(dim, 1)` instead of `dim`. A dimension of size 0 has no elements, so its stride is never used to read memory. But a plain product would set every stride to its left to 0, and the tensor would lose the layout of its other dimensions:

```text
shape [2, 0, 3]

with max(dim, 1):                 with a plain product:
dim 2: stride 1, then 1*3 = 3     dim 2: stride 1, then 1*3 = 3
dim 1: stride 3, then 3*1 = 3     dim 1: stride 3, then 3*0 = 0
dim 0: stride 3, then 3*2 = 6     dim 0: stride 0, then 0*2 = 0

strides [3, 3, 1]                 strides [0, 3, 1]
```

Well-formed strides matter in [chapter 03](@/book/03-views.md), where views of an empty tensor go through the same stride arithmetic as views of a full one, and where a stride of 0 would wrongly mark the tensor as overlapping. PyTorch produces the same strides: `torch.empty(2, 0, 3).stride()` is `(3, 3, 1)`. NumPy differs here and reports all-zero strides for empty arrays: `np.empty((2, 0, 3)).strides` is `(0, 0, 0)`.

The product is overflow-checked even though `numel` would catch most overflows, because `max(dim, 1)` changes the product. A shape like `[2^40, 0, 2^40]` has 0 elements, so `numel` returns 0, but its contiguous strides need the product `2^40 * 1 * 2^40 = 2^80`. The last multiplication of the loop, for dimension 0, overflows. `contiguous_strides` throws `ValueError` with `tensor stride overflows int64` for it, and so does `zeros` with that shape, because the factory calls `contiguous_strides`.

### `offset_from_index`

From `src/shape.cpp`:

```cpp
std::int64_t offset_from_index(std::span<const std::int64_t> shape,
                               std::span<const std::int64_t> strides, std::int64_t base_offset,
                               std::span<const std::int64_t> index) {
    KAEL_CHECK_VALUE(shape.size() == strides.size(),
                     "shape and strides have different number of dimensions: shape "
                         << join(shape) << " has " << shape.size() << ", strides " << join(strides)
                         << " has " << strides.size());
    KAEL_CHECK_VALUE(index.size() == shape.size(),
                     "index and shape have different number of dimensions: index "
                         << join(index) << " has " << index.size() << ", shape " << join(shape)
                         << " has " << shape.size());

    std::int64_t offset = base_offset;
    for (std::size_t i = 0; i < shape.size(); ++i) {
        KAEL_CHECK_INDEX(index[i] >= 0 && index[i] < shape[i],
                         "index " << index[i] << " out of range [0, " << shape[i]
                                  << ") at dimension " << i << " in shape " << join(shape));
        offset += index[i] * strides[i];
    }
    return offset;
}
```

Four design points:

- The parameters are `std::span<const std::int64_t>`, not `const Shape&`. A span is a non-owning view of a contiguous range: a pointer and a length. It accepts a `Shape`, a `std::array`, or a pointer and a length without copying, so callers that keep indices in other containers do not have to build a `Shape` first. `Tensor::at` passes an `std::initializer_list` this way.
- A rank mismatch between the three arguments is a `ValueError`, because it means the caller passed malformed metadata. An index outside `[0, shape[i])` throws `kael::IndexError`. The Python bindings in `python/src/bindings.cpp` map `IndexError` to Python's `IndexError`, so `t[5]` on a 3-element tensor raises the exception a Python user expects.
- The function accepts only non-negative indices. Python-style negative indices (`-1` for the last element) are resolved one layer up, in `Tensor::element_offset`, which `Tensor::at` calls. It adds the dimension size to each negative index before calling this function. Keeping this layer strict means a computed index that went negative by mistake fails here instead of silently reading another element.
- The products `index[i] * strides[i]` are not overflow-checked. The index is in range and the strides belong to a tensor whose extent was already validated against its storage ([chapter 02](@/book/02-memory-model.md)), so every partial sum is bounded by the size of the buffer.

For a 0-d shape, all three spans are empty, the loop does nothing, and the result is the base offset. That is the position of the single element.

### `index_from_linear`

From `src/shape.cpp`:

```cpp
Shape index_from_linear(std::int64_t linear, const Shape& shape) {
    const std::int64_t count = numel(shape);
    KAEL_CHECK_INDEX(linear >= 0 && linear < count, "linear index "
                                                        << linear << " out of range [0, " << count
                                                        << ") for shape " << to_string(shape));

    Shape index(shape.size());
    for (std::size_t i = shape.size(); i-- > 0;) {
        index[i] = linear % shape[i];
        linear /= shape[i];
    }
    return index;
}
```

The range check comes first and also protects the division. An empty shape has `count == 0`, so every `linear` fails the check and the loop never divides by a zero dimension. For a 0-d shape, `count == 1`, `linear` must be 0, and the result is the empty index `[]`. The call to `numel` also validates the shape, so `index_from_linear(0, {-1, 3})` throws `ValueError` before the range check runs.

`index_from_linear` is the slow, general way to visit every element: one division and one remainder per dimension per element, and a heap allocation for the result. The tensor printer in `src/print.cpp` and several tests use it. The iterator in [chapter 04](@/book/04-tensor-iterator.md) replaces it with incremental counters for the hot loops: instead of recomputing the index from scratch, it adds one to the last coordinate and carries into the next one when it reaches the dimension size, like an odometer.

### `broadcast_shapes`

From `src/shape.cpp`:

```cpp
Shape broadcast_shapes(const Shape& a, const Shape& b) {
    validate_shape(a);
    validate_shape(b);

    const std::size_t ndim = std::max(a.size(), b.size());
    Shape result(ndim);
    for (std::size_t from_right = 0; from_right < ndim; ++from_right) {
        const std::int64_t dim_a = from_right < a.size() ? a[a.size() - 1 - from_right] : 1;
        const std::int64_t dim_b = from_right < b.size() ? b[b.size() - 1 - from_right] : 1;
        KAEL_CHECK_VALUE(dim_a == dim_b || dim_a == 1 || dim_b == 1,
                         "shapes " << to_string(a) << " and " << to_string(b)
                                   << " are not broadcastable");
        result[ndim - 1 - from_right] = dim_a == 1 ? dim_b : dim_a;
    }
    return result;
}
```

The loop counts positions from the right (`from_right`), which handles the alignment and the implicit leading 1s in one place. An alternative is to left-pad the shorter shape with 1s into a new vector first. That costs an allocation and a second loop for the same result.

Trace for `a = [5, 1, 4]`, `b = [3, 1]`, `ndim = 3`:

```text
from_right  dim_a       dim_b            check        result[2 - from_right]
0           a[2] = 4    b[1] = 1         b is 1       4
1           a[1] = 1    b[0] = 3         a is 1       3
2           a[0] = 5    missing -> 1     b is 1       5

result = [5, 3, 4]
```

The result line `dim_a == 1 ? dim_b : dim_a` covers all three valid cases: equal sizes, `a` is 1, and `b` is 1. It also gives the right answer for `[1]` with `[0]`: `dim_a` is 1, so the result is `dim_b`, which is 0. Using `std::max(dim_a, dim_b)` instead would return 1 for that pair, which is wrong.

The function takes two shapes. `TensorIterator::common_input_shape` in [chapter 04](@/book/04-tensor-iterator.md) folds it over any number of inputs, starting from the first input's shape. Broadcasting is associative and commutative, so the fold order does not change the result.

### `broadcast_strides`

From `src/shape.cpp`:

```cpp
Strides broadcast_strides(const Shape& shape, const Strides& strides, const Shape& target) {
    validate_shape(shape);
    validate_shape(target);
    KAEL_CHECK_VALUE(shape.size() == strides.size(),
                     "shape " << to_string(shape) << " and strides " << to_string(strides)
                              << " do not match size " << shape.size() << ":" << strides.size());
    KAEL_CHECK_VALUE(shape.size() <= target.size(),
                     "shape size " << to_string(shape) << " is bigger than target shape"
                                   << to_string(target) << " " << shape.size() << ":"
                                   << target.size());

    const std::size_t new_dims = target.size() - shape.size();
    Strides result(target.size(), 0);
    for (std::size_t i = 0; i < shape.size(); ++i) {
        const std::int64_t target_dim = target[i + new_dims];
        KAEL_CHECK_VALUE(shape[i] == target_dim || shape[i] == 1, "incompatible shapes "
                                                                      << to_string(shape) << " and "
                                                                      << to_string(target));
        if (shape[i] == target_dim) {
            result[i + new_dims] = strides[i];
        }
    }
    return result;
}
```

The result starts filled with zeros, so new leading dimensions and size-1 dimensions that grow need no code: they keep the 0. Only matching dimensions copy their stride.

Before the loop, the function validates both shapes, checks that `shape` and `strides` have the same length, and checks that `shape` has no more dimensions than `target`. Broadcasting only adds dimensions. It never removes them.

A worked example with a new leading dimension. Shape `[1, 3]`, strides `[3, 1]`, target `[4, 2, 3]`:

```text
target:      4   2   3
input:           1   3
new_dims = 1

position 0: new leading dimension        -> 0
position 1: input 1, target 2            -> 0
position 2: input 3, target 3, stride 1  -> 1

strides = [0, 0, 1]
```

The loop runs over the input's dimensions, not the target's, and maps input position `i` to target position `i + new_dims`. That is the right-alignment of broadcasting written as an index shift.

`broadcast_strides` does not compute the target shape. The caller computes it with `broadcast_shapes` over all inputs first, then calls `broadcast_strides` once per input. [Chapter 04](@/book/04-tensor-iterator.md) shows the iterator doing that, and [chapter 03](@/book/03-views.md) uses it for `Tensor::expand`, which is the user-visible form of a broadcast: it returns the stride-0 view itself. Chapter 03 also adds `Tensor::has_internal_overlap`, which reports a dimension with size greater than 1 and stride 0, and the in-place operations of [chapter 05](@/book/05-operations.md) reject such tensors with a `ValueError`.

### `wrap_dim` and `dim_position`

From `src/shape.cpp`:

```cpp
std::int64_t wrap_dim(std::int64_t dim, std::int64_t ndim) {
    KAEL_CHECK_INDEX(ndim > 0, "dimension " << dim << " specified but tensor has no dimensions");
    KAEL_CHECK_INDEX(dim >= -ndim && dim < ndim,
                     "dimension " << dim << " out of range [" << -ndim << ", " << ndim - 1 << "]");
    return dim < 0 ? dim + ndim : dim;
}

std::size_t dim_position(std::int64_t dim, std::int64_t ndim) {
    return static_cast<std::size_t>(wrap_dim(dim, ndim));
}
```

For a 3-d tensor the valid range is `[-3, 2]`, and `wrap_dim(-1, 3)` returns 2. `wrap_dim(3, 2)` throws with `dimension 3 out of range [-2, 1]`, which the test `WrapDim.OutOfRangeThrowsWithRange` checks verbatim. A 0-d tensor has no dimensions at all, so every dimension argument is an error.

`dim_position` returns the wrapped value as a `std::size_t`, ready to index a `Shape`, which keeps the signed-to-unsigned cast in one place instead of at every call site. `Tensor::slice` in `src/view.cpp`, for example, calls `dim_position(dim, this->dim())` and uses the result to index both the shape and the strides.

PyTorch accepts one more case: it treats a 0-d tensor as if it had one dimension for wrapping, so `torch.tensor(5.0).sum(0)` works. kael is stricter and rejects it.

This differs from `offset_from_index` on purpose. Dimension arguments come from user code, where `-1` is idiomatic. Element indices passed to `offset_from_index` come from internal loops, where a negative value is a bug.

### `infer_shape`

From `src/shape.cpp`:

```cpp
Shape infer_shape(const Shape& requested, std::int64_t element_count) {
    KAEL_CHECK_VALUE(std::ranges::count(requested, kInferredDim) <= 1,
                     "only one dimension can be -1 in shape " << to_string(requested));

    Shape known;
    std::ranges::copy_if(requested, std::back_inserter(known),
                         [](std::int64_t dim) { return dim != kInferredDim; });
    const std::int64_t known_count = numel(known);

    const auto inferred = std::ranges::find(requested, kInferredDim);
    if (inferred == requested.end()) {
        KAEL_CHECK_VALUE(known_count == element_count,
                         "shape " << to_string(requested) << " has " << known_count
                                  << " elements but the tensor has " << element_count);
        return requested;
    }

    KAEL_CHECK_VALUE(known_count != 0, "cannot infer the -1 in shape "
                                           << to_string(requested)
                                           << ": a dimension of size 0 makes it ambiguous");
    KAEL_CHECK_VALUE(element_count % known_count == 0, "shape " << to_string(requested)
                                                                << " does not fit " << element_count
                                                                << " elements");

    Shape result = requested;
    result[static_cast<std::size_t>(std::distance(requested.begin(), inferred))] =
        element_count / known_count;
    return result;
}
```

`kInferredDim` is a constant equal to `-1` in an anonymous namespace of `src/shape.cpp`. The function proceeds in four steps:

1. Reject more than one `-1`. Two unknowns have no unique solution.
2. Copy every other dimension into `known` and take its `numel`. That call also rejects any other negative value, such as `-2`, through `validate_shape`.
3. With no `-1`, the requested shape must have exactly `element_count` elements.
4. With one `-1`, divide `element_count` by `known_count` and write the quotient into the `-1` position.

Traces:

```text
infer_shape({4, -1}, 12)    known [4],  known_count 4,  12 % 4 == 0   -> [4, 3]
infer_shape({-1}, 12)       known [],   known_count 1                 -> [12]
infer_shape({-1, 3}, 0)     known [3],  known_count 3,  0 % 3 == 0    -> [0, 3]
infer_shape({5, -1}, 12)    known [5],  12 % 5 == 2                   -> throws
infer_shape({-1, 0}, 0)     known [0],  known_count 0                 -> throws
```

The zero check guards an ambiguous case: `[-1, 0]` or `[0, -1]` for a tensor with 0 elements fits any value in place of the `-1`, so there is no single answer. PyTorch rejects the same case. It also prevents the division by zero that would follow.

### How this code evolved

The functions above are the current code, not the phase 2 snapshot. Comparing with the tag `chapter-02-shapes`, the last commit of phase 2, shows three changes:

- Phase 2 had a single `kael::Error` class and one `KAEL_CHECK` macro. The typed subclasses and `KAEL_CHECK_INDEX` and `KAEL_CHECK_VALUE` arrived in phase 10.
- `contiguous_strides` already used `max(dim, 1)`, but multiplied without a check. `src/internal/checked_math.h` arrived in phase 3 with `TensorImpl`, and the stride product moved to `checked_mul` in phase 6 (`c786049 fix(ops): align edge cases with PyTorch`).
- `wrap_dim`, `infer_shape` and `dim_position` did not exist yet, as listed at the top of this chapter.

## Edge cases and pitfalls

- **Signed overflow is undefined behavior.** Every product of dimensions or strides is checked before it happens: in `numel`, in `contiguous_strides` through `checked_mul`, and in chapter 02's storage-fit check. Checking after the multiplication is wrong, because the compiler may assume the overflow does not occur.
- **Empty tensors.** `numel` returns 0 before the overflow loop. `contiguous_strides` treats size 0 as 1 so strides stay meaningful. `index_from_linear` rejects every position for an empty shape before it divides. A shape with a 0 and very large other dimensions still fails in `contiguous_strides`, because its strides cannot be represented.
- **0-d tensors.** Shape `[]` has one element, strides `[]`, and the index `[]`. Every function handles it through the general path: empty loops, product of nothing equals 1. `wrap_dim` is the exception: a 0-d tensor has no dimension to name, so it throws.
- **`[]` is not `[0]`.** The first has one element, the second none. Code that tests `shape.empty()` to mean "no elements" is wrong. Test `numel(shape) == 0` instead.
- **Unsigned countdown loops.** The loops that walk from the last dimension use `for (std::size_t i = size; i-- > 0;)`. The more common `for (std::size_t i = size - 1; i >= 0; --i)` never ends, because an unsigned value is always `>= 0`, and it wraps for an empty shape.
- **Broadcasting with size 0.** A 0 only matches a 0 or a 1. Treating 0 as "smaller, so it stretches" would be wrong and would disagree with NumPy. For the same reason the result size is not `max(dim_a, dim_b)`.
- **Strides of size-1 dimensions.** `broadcast_strides` keeps the original stride when the input size equals the target size, including when both are 1. That stride is never used to step, so its value does not matter. Chapter 02's contiguity check skips size-1 dimensions for the same reason.
- **Writing through stride 0.** A broadcast view reads correctly but must never be written element by element. Two logical positions share one memory location, so the last write wins and the result depends on loop order.
- **Typed errors.** Index problems throw `IndexError` (`offset_from_index`, `index_from_linear`, `wrap_dim`). Shape and argument problems throw `ValueError`. A negative operand to `checked_mul` or `checked_add` is an internal bug and throws the base `Error`. Code that catches `kael::Error` catches all of them, because all derive from it.

> [!WARNING]
> Mixing up `Shape` and `Strides` compiles without a warning, because both are `std::vector<std::int64_t>`. Passing `(strides, shape)` instead of `(shape, strides)` to `offset_from_index` produces wrong offsets, not an error, whenever both vectors are in range. Name your variables after what they hold and check argument order in review.

## How it is tested

The tests for this chapter's functions are in `tests/test_shape.cpp` and run as part of `kael_tests`. There are 46 tests in these suites:

| Suite | Tests | What it proves |
|-------|-------|----------------|
| `ShapeString` | 5 | Formatting of normal, empty (`[]`), one-dimensional, zero-sized and large shapes |
| `ValidateShape` | 3 | Valid shapes pass, negative dimensions throw, and the message contains the shape and the position |
| `Numel` | 3 | Regular counts, negative dimensions, and the exact overflow boundary (`{max}` fits, `{max, 2}` throws) |
| `ContiguousStrides` | 5 | Row-major strides, 0-d shapes, size-1 dimensions, `[2, 0, 3]` gives `[3, 3, 1]`, negative shapes throw |
| `OffsetFromIndex` | 9 | Contiguous and non-contiguous strides, base offset, 0-d, out-of-range and negative indices, size mismatches, error message contents |
| `IndexFromLinear` | 4 | Row-major order, 0-d shapes, out-of-range positions, and the round trip with `offset_from_index` |
| `BroadcastShapes` | 6 | Valid and invalid pairs, symmetry, negative dimensions, error messages, and 300 NumPy-generated cases |
| `BroadcastStrides` | 8 | Stride results, same shape, extra dimensions, mismatches, error messages, and two semantic tests that read real data |
| `WrapDim` | 3 | Positive and negative dimensions, range errors with the exact message, and the 0-d error |

`infer_shape` is tested in `tests/test_view.cpp`, in the `InferShape` suite (`ResolvesMinusOne` and `Errors`), because it arrived with `view` and `reshape`. Those two tests cover the five traces shown above, `infer_shape({2, 3}, 6)`, `infer_shape({}, 1)`, and the rejection of two `-1`s, of `-2`, and of a shape whose count does not match.

Run the suites of this chapter alone:

```bash
build/tests/kael_tests --gtest_filter='ShapeString.*:ValidateShape.*:Numel.*:ContiguousStrides.*:OffsetFromIndex.*:IndexFromLinear.*:BroadcastShapes.*:BroadcastStrides.*:WrapDim.*:InferShape.*'
```

Three kinds of test carry most of the weight.

**The round trip** ties the two index functions together over every element of a 3-d shape.

From `tests/test_shape.cpp`:

```cpp
TEST(IndexFromLinear, RoundTripWithOffsetFromIndex) {
    kael::Shape shape = {2, 3, 4};
    kael::Strides strides = kael::contiguous_strides(shape);
    for (std::int64_t linear = 0; linear < kael::numel(shape); linear++) {
        kael::Shape index = kael::index_from_linear(linear, shape);
        EXPECT_EQ(kael::offset_from_index(shape, strides, 0, index), linear);
    }
}
```

With contiguous strides, the linear position and the memory offset are the same number. If either function walked the dimensions in the wrong order or had an off-by-one error, some element would map to the wrong offset. The test also exercises `contiguous_strides`, so a bug in any of the three functions shows up here.

**The semantic broadcast tests** (`BroadcastStrides.SemanticBroadcastRows` and `SemanticBroadcastColumns`) put real values in a buffer, broadcast a `[3]` row to `[2, 3]` and a `[2, 1]` column to `[2, 4]`, then read every element of the target through `index_from_linear` and `offset_from_index`. Each value must equal the one the broadcast is supposed to repeat.

From `tests/test_shape.cpp`:

```cpp
TEST(BroadcastStrides, SemanticBroadcastRows) {
    kael::Shape shape = {3};
    kael::Strides strides = {1};
    kael::Shape target = {2, 3};

    std::vector<float> data = {0.0f, 1.0f, 2.0f};

    kael::Strides broadcasted = kael::broadcast_strides(shape, strides, target);

    for (std::int64_t linear = 0; linear < kael::numel(target); linear++) {
        kael::Shape index = kael::index_from_linear(linear, target);

        std::int64_t offset = kael::offset_from_index(target, broadcasted, 0, index);

        EXPECT_FLOAT_EQ(data.at(static_cast<std::size_t>(offset)),
                        data.at(static_cast<std::size_t>(index[1])));
    }
}
```

They check that stride 0 does what the diagrams in this chapter claim, not only that the stride vectors have the expected numbers. `data.at` also bounds-checks every offset, so a stride that pointed outside the buffer would throw.

**The generated cases** compare kael with NumPy. `scripts/generate/broadcast_cases.py` seeds Python's `random` with 42, draws 300 pairs of shapes with 0 to 4 dimensions and sizes from `{0, 1, 2, 3}` with weights `2, 3, 2, 3` (so 1 and 3 are drawn more often and compatible pairs are common), and asks `np.broadcast_shapes` for each answer. Of the 300 cases, 199 are valid and 101 are not. It writes one initializer per pair to `tests/generated/broadcast_cases.inc`:

```text
    {{1, 0}, {}, true, {1, 0}},
    {{0, 1, 0, 1}, {2, 2, 3, 3}, false, {}},
```

Each line is `{a, b, valid, expected}`. `BroadcastShapes.GeneratedCases` includes the file into an array of `BroadcastCase` structs and expects either the same result or a `kael::Error` for every case. A `SCOPED_TRACE` prints the case number and both shapes when one fails. Regenerate the file with:

```bash
python3 scripts/generate/broadcast_cases.py
```

The output is deterministic, so a regenerated file only changes if the script changes. The generated file is committed, which means the C++ tests do not need Python or NumPy to run. Random cases find combinations a person would not think to write, such as two empty shapes of different rank.

> [!TIP]
> Build with `-DKAEL_ENABLE_SANITIZERS=ON` and run these suites. UndefinedBehaviorSanitizer reports signed overflow at run time, so a clean run is evidence that no multiplication escaped its check.

## Build it yourself

Start from an empty CMake project with a library target, GoogleTest through `FetchContent`, and the warning flags from the [overview](@/book/00-overview.md).

1. Create `include/kael/error.h` with `kael::Error` (derived from `std::runtime_error`), the subclasses `IndexError`, `ValueError` and `TypeError`, and the `KAEL_CHECK_AS` macro with the `KAEL_CHECK`, `KAEL_CHECK_INDEX` and `KAEL_CHECK_VALUE` shortcuts. Checkpoint: a test that `KAEL_CHECK_VALUE(false, "x " << 1)` throws `kael::ValueError` and the message contains `x 1`.
2. Create `include/kael/shape.h` with the `Shape` and `Strides` aliases and `to_string`. Checkpoint: the `ShapeString` tests pass, including `[]` for an empty shape.
3. Add `validate_shape` and `numel` with the zero early return and the division-based overflow check. Checkpoint: `ValidateShape.*` and `Numel.*` pass, including `numel({max})` and the throw for `numel({max, 2})`.
4. Add `src/internal/checked_math.h` with `checked_mul` and `checked_add`, make `src/` a private include directory, then write `contiguous_strides` with the `max(dim, 1)` product. Checkpoint: `ContiguousStrides.*` pass, in particular `[2, 0, 3]` gives `[3, 3, 1]` and `[3, 1, 2]` gives `[2, 2, 1]`.
5. Add `offset_from_index` with `std::span` parameters, the size checks and the per-dimension range check. Checkpoint: `OffsetFromIndex.*` pass, including the non-contiguous case with strides `[1, 3]`.
6. Add `index_from_linear` with the range check before the division loop. Checkpoint: `IndexFromLinear.RoundTripWithOffsetFromIndex` passes for all 24 positions.
7. Add `broadcast_shapes`, counting positions from the right. Checkpoint: `BroadcastShapes.ValidCases`, `Symmetry` and `IncompatibleThrows` pass.
8. Add `scripts/generate/broadcast_cases.py`, run it, and add `BroadcastShapes.GeneratedCases`. Checkpoint: all 300 cases agree with NumPy.
9. Add `broadcast_strides` with the zero-filled result. Checkpoint: `BroadcastStrides.SemanticBroadcastRows` and `SemanticBroadcastColumns` pass.
10. Add `wrap_dim`, `dim_position` and `infer_shape`. They are only needed from chapter 02 and 03 on, but they belong in this file. Checkpoint: `WrapDim.*` pass, and `infer_shape({-1, 4}, 12)` returns `[3, 4]` while `infer_shape({0, -1}, 0)` throws. Copy `InferShape.ResolvesMinusOne` and `InferShape.Errors` from `tests/test_view.cpp` to cover the rest.
11. Build with `-DKAEL_ENABLE_SANITIZERS=ON` and run the suite again. Checkpoint: no AddressSanitizer or UndefinedBehaviorSanitizer reports. UBSan reports signed overflow, so this run confirms that no overflow check was skipped.

## Check your understanding

<details>
<summary>Why does a transpose not copy data?</summary>

A transpose only swaps two entries of the shape and the same two entries of the strides. The offset formula `base_offset + sum(index[i] * strides[i])` then maps each transposed index to the memory position of the original element. For a `[2, 3]` tensor with strides `[3, 1]`, the transpose has shape `[3, 2]` and strides `[1, 3]`, and its element `[2, 1]` is at `2 * 1 + 1 * 3 = 5`, the same position as element `[1, 2]` of the original.

</details>

<details>
<summary>What are the contiguous strides of shape <code>[4, 0, 5]</code>, and why not <code>[0, 5, 1]</code>?</summary>

`[5, 5, 1]`. The running product multiplies by `max(dim, 1)`, so the 0 counts as 1: dimension 2 gets 1, dimension 1 gets 5, and dimension 0 gets `5 * 1 = 5`. A plain product would give `[0, 5, 1]`, which loses the layout of dimension 0 and would make the tensor look as if it had internal overlap. PyTorch reports `(5, 5, 1)` for the same shape.

</details>

<details>
<summary>Why does <code>numel</code> check <code>total &lt;= max / dim</code> instead of multiplying and checking whether the result went negative?</summary>

Signed overflow is undefined behavior in C++. The compiler is allowed to assume it never happens, so a check on the result of an overflowing multiplication can be optimized away or can see any value. Dividing the maximum by `dim` cannot overflow, and the test `total <= max / dim` is true exactly when `total * dim` fits. The early return for a zero dimension guarantees `dim >= 1`, so the division is safe.

</details>

<details>
<summary>What strides does <code>broadcast_strides({2, 1}, {1, 1}, {3, 2, 4})</code> return, step by step?</summary>

`[0, 1, 0]`. The target has 3 dimensions and the input 2, so `new_dims = 1` and position 0 is a new leading dimension with stride 0. Input dimension 0 has size 2 and target size 2, so it keeps stride 1 at position 1. Input dimension 1 has size 1 and target size 4, so it stays 0 at position 2. Reading through these strides repeats each of the two values 4 times along the last dimension and the whole `[2, 4]` block 3 times along the first.

</details>

<details>
<summary>Why does <code>offset_from_index</code> reject <code>-1</code> while <code>wrap_dim</code> accepts it?</summary>

They serve different callers. Dimension arguments come from user code, where `-1` for the last dimension is idiomatic Python and PyTorch. Element indices reach `offset_from_index` from internal loops and from `Tensor::element_offset`, which has already converted negative user indices. A negative value at that point is a bug, and rejecting it with `IndexError` stops the code from silently reading a different element.

</details>

<details>
<summary>Are the shapes <code>[0, 3]</code> and <code>[2, 1]</code> broadcastable? What is the result?</summary>

No. Aligned from the right, the last dimensions are 3 and 1, which is fine and gives 3. The first dimensions are 0 and 2: they are not equal and neither is 1, so rule 2 fails and `broadcast_shapes` throws `ValueError`. A 0 is not a wildcard. It only matches a 0 or a 1.

</details>

## Further reading

- [NumPy: Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html). The reference definition of the rules in this chapter, with diagrams of stretched dimensions.
- [NumPy: Internal memory layout of an ndarray](https://numpy.org/doc/stable/reference/arrays.ndarray.html). Defines strides in bytes and the offset formula, and explains C and Fortran order.
- Edward Z. Yang, [PyTorch internals](http://blog.ezyang.com/2019/05/pytorch-internals/). The section on strides and views shows the same metadata model in PyTorch, with element strides as in kael.
- [PyTorch: `torch.Tensor.stride`](https://pytorch.org/docs/stable/generated/torch.Tensor.stride.html). Useful to check by hand that kael's strides match PyTorch for any shape.
- [Clang: UndefinedBehaviorSanitizer](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html). Documents `-fsanitize=signed-integer-overflow`, the check that proves the overflow guards in this chapter run in time.
- [cppreference: Arithmetic operators, overflows](https://en.cppreference.com/w/cpp/language/operator_arithmetic). States that signed overflow is undefined and unsigned arithmetic wraps, the two facts behind the overflow checks and the countdown loop idiom.
- [`src/shape.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/shape.cpp). Under 200 lines; read it in full after this chapter.

Next: [02. Memory model](@/book/02-memory-model.md)
