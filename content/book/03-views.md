+++
title = "Views"
description = "How kael reshapes, transposes, slices and broadcasts tensors by rewriting metadata over shared storage, and how it copies and writes safely when layouts alias."
weight = 3
slug = "views"

[extra]
chapter = "03"
kicker = "Chapter 03"
builds = "Reshape, transpose, slicing and expand without copying data"
source = "docs/03-views.md"
tag = "chapter-04-views"
+++

This chapter builds the view layer: the operations that produce a new tensor by changing only its shape, strides and offset, while the bytes stay in the same `Storage`. It also builds the one place that moves bytes between two arbitrary layouts, the strided copy, and the write rules that keep in-place operations correct when several tensors share memory. Together they correspond to the views phase of the project (tag `chapter-04-views`) and give every later chapter cheap transposes, slices and broadcasts.

| File | Role |
|------|------|
| [`src/view.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/view.cpp) | `make_view` and every view method of `Tensor`: `view`, `reshape`, `transpose`, `permute`, `T`, `mT`, `squeeze`, `unsqueeze`, `expand`, `slice`, `select`, `narrow`, `flatten`, `unflatten`, `split`, `chunk`, `detach`, `has_internal_overlap` |
| [`src/tensor_impl.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/tensor_impl.cpp) | `TensorImpl::is_contiguous` and `TensorImpl::validate`, which every view passes through |
| [`src/tensor.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/tensor.cpp) | `clone`, `contiguous`, `to(DType)`: the operations that materialize a layout |
| [`src/copy.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/copy.cpp), [`src/internal/copy.h`](https://github.com/JetBerri/kael.cpp/blob/main/src/internal/copy.h) | `detail::convert_strided`, the strided copy with dtype conversion |
| [`src/ops/inplace.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/ops/inplace.cpp) | `copy_` and the in-place write rules that protect aliased memory |
| [`include/kael/storage.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/storage.h) | the version counter shared by every view of one buffer |
| [`scripts/generate/view_cases.py`](https://github.com/JetBerri/kael.cpp/blob/main/scripts/generate/view_cases.py) | generates `tests/generated/view_cases.inc` from NumPy |

## Prerequisites

- [01. Shapes and strides](@/book/01-shapes-and-strides.md): `contiguous_strides`, `broadcast_strides`, `infer_shape`, `wrap_dim`, `dim_position`, `offset_from_index`.
- [02. Memory model](@/book/02-memory-model.md): `Storage`, `TensorImpl`, the `Tensor` handle, `DType`, `dispatch`, `TensorImpl::validate` and `is_contiguous`.
- [ADR 0002: tensor as handle](@/adr/0002-tensor-as-handle.md) explains why a `Tensor` is a `std::shared_ptr<TensorImpl>`.

## Why this layer exists

Deep learning code rearranges tensors constantly. A linear layer multiplies by the transpose of its weight. Attention splits a `[batch, seq, hidden]` activation into heads with a reshape and a permute. A training loop takes a batch as a slice of a larger dataset. Broadcasting adds a `[hidden]` bias to every row of a `[batch, hidden]` matrix.

If each of those operations copied the data, three things would break:

- **Cost.** `x.T()` on a 4096 by 4096 float32 matrix would allocate and write 64 MiB. A model that transposes weights on every forward pass would spend more time copying than multiplying.
- **Semantics.** Users of NumPy and PyTorch expect `x[0] = 5` or `x.narrow(0, 1, 2).zero_()` to change `x`. A copying slice would silently write to a temporary.
- **Broadcasting.** Adding a bias to every row would require materializing a full matrix of repeated bias values before the addition.

The fix rests on the observation from chapter 01: the layout of a tensor is fully described by its shape, strides and offset. Changing those numbers changes which bytes a tensor reads and in which order, without touching the bytes. A **view** is a new tensor that shares the storage of an existing tensor and differs only in that metadata.

The mental model for this chapter:

```text
one Storage, many lenses

  Storage   [ 0 | 1 | 2 | 3 | 4 | 5 ]        bytes, written once

  a         shape [2, 3]  strides [3, 1]  offset 0     rows of 3
  a.T()     shape [3, 2]  strides [1, 3]  offset 0     columns of a
  a[:, 2]   shape [2]     strides [3]     offset 2     last column
```

Views are cheap and share writes. They also bring two new hazards that the rest of the chapter has to handle: a view whose metadata is wrong reads someone else's memory, and a write through one view can change the input of an operation that reads through another view. kael answers the first with the validation from chapter 02 and the second with explicit write rules.

## Concepts

### What a view is

A tensor in kael is four pieces of metadata over a byte buffer:

```text
TensorImpl
  storage  -> shared_ptr<Storage>   (bytes, shared)
  shape    -> [2, 3]                (logical sizes)
  strides  -> [3, 1]                (elements to skip per step in each dim)
  offset   -> 0                     (element index of [0, 0] inside storage)
```

The address of element `[i, j]` is `offset + i * strides[0] + j * strides[1]`, measured in elements. Two `TensorImpl` objects that hold the same `storage` pointer read and write the same bytes, even when their shapes, strides and offsets differ.

**Views** are new `TensorImpl` objects that share the storage of an existing tensor and change only the metadata. Creating a view costs a few small heap allocations (the impl and its shape and stride vectors) and never touches the data.

Consider `a = arange(6).view({2, 3})`:

```text
storage (float32):  [ 0 | 1 | 2 | 3 | 4 | 5 ]
                      0   1   2   3   4   5    <- element index

a          shape [2, 3]  strides [3, 1]  offset 0
           [[0, 1, 2],
            [3, 4, 5]]

a.T()      shape [3, 2]  strides [1, 3]  offset 0
           [[0, 3],
            [1, 4],
            [2, 5]]

a.select(1, 2)
           shape [2]     strides [3]     offset 2
           [2, 5]

arange(10).slice(0, 1, 8, 3)
           shape [3]     strides [3]     offset 1
           [1, 4, 7]
```

All four results were checked by running the code against the built library. A trace of one element through the transpose, `a.T()[2, 1]`:

```text
position = offset + 2 * strides[0] + 1 * strides[1]
         = 0      + 2 * 1          + 1 * 3
         = 5
value    = storage[5] = 5        which is a[1, 2], as a transpose requires
```

The sharing relations, as objects:

```text
a                 [2, 3] / [3, 1] / 0  ----+
a.T()             [3, 2] / [1, 3] / 0  ----+---->  Storage (24 bytes)
a.select(1, 2)    [2]    / [3]    / 2  ----+

a.T().reshape({6}) [6]   / [1]    / 0  --------->  new Storage (24 bytes)
```

The last line shows the one case in this chapter that does not share: `reshape` of a non-contiguous tensor allocates new storage. [view versus reshape](#view-versus-reshape) explains why.

PyTorch uses the same model, and its documentation has a page on tensor views listing which operations return them. NumPy also returns views for basic slicing and transposes, and exposes the original array through the `base` attribute.

### Contiguity

**Contiguous** tensors are those whose elements, visited in row-major logical order, sit at consecutive storage positions. For a shape `[d0, d1, ..., dn]` that means the strides equal `contiguous_strides(shape)`, with two relaxations that `TensorImpl::is_contiguous` implements:

1. An empty tensor is always contiguous. There is nothing to visit.
2. The stride of a size-1 dimension is ignored. The loop never steps along it, so its value cannot change any address.

The offset does not matter: a contiguous block in the middle of a buffer (`a.select(0, 1)` with offset 3) is still contiguous.

Worked checks, each confirmed by running the code:

| Tensor | Shape | Strides | Contiguous | Reason |
|--------|-------|---------|------------|--------|
| `a` | `[2, 3]` | `[3, 1]` | yes | matches `[3, 1]` |
| `a.T()` | `[3, 2]` | `[1, 3]` | no | last stride is 3, expected 1 |
| `a.unsqueeze(1)` | `[2, 1, 3]` | `[3, 3, 1]` | yes | the size-1 stride is skipped |
| `a.select(0, 1)` | `[3]` | `[1]` | yes | offset 3 does not matter |
| `a.select(1, 2)` | `[2]` | `[3]` | no | gap of 2 elements between values |
| `arange(10).slice(0, 0, 10, 2)` | `[5]` | `[2]` | no | gap between elements |
| `from_vector({1.f, 2.f, 3.f}).expand({2, 3})` | `[2, 3]` | `[0, 1]` | no | stride 0 repeats a row |
| `zeros({0, 3}).T()` | `[3, 0]` | `[1, 3]` | yes | empty |

Contiguity matters for three reasons. `view` requires it. Kernels have a fast path when every operand advances by exactly one element ([04. TensorIterator](@/book/04-tensor-iterator.md)). And `clone` of a contiguous tensor is one `memcpy`.

### Stride 0 and internal overlap

A stride of 0 means "stepping along this dimension does not move". Every index along it reads the same element. That is how a 3-element vector can look like a `[2, 3]` matrix without allocating the second row:

```text
v = from_vector({1.f, 2.f, 3.f})       shape [3]     strides [1]
v.expand({2, 3})                       shape [2, 3]  strides [0, 1]

position of [i, j] = 0 + i * 0 + j * 1 = j

           j=0  j=1  j=2
  i = 0     1    2    3       positions 0 1 2
  i = 1     1    2    3       positions 0 1 2   (same bytes again)
```

When several logical positions map to one storage element, the tensor has **internal overlap**. Reading such a tensor is safe and useful. Writing to it is not: `expanded.add_(1)` would add 1 to storage element 0 twice, once for `[0, 0]` and once for `[1, 0]`, and the result would depend on how many times and in which order the loop visits each element. Parallel loops would also race on the same address. kael refuses such writes.

### Aliasing between operands

Internal overlap is about one tensor. **Aliasing** is about two: a source and a destination that share storage. Three cases exist:

```text
1. no sharing         dst and src use different storages       safe
2. same layout        dst and src are the same elements        safe for elementwise ops
3. partial overlap    dst and src share some elements,         unsafe without a copy
                      at different positions
```

Case 2 is safe because an elementwise operation reads element `k` and writes element `k`, so every read happens before the write to the same address. Case 3 breaks that. With `t = [0, 1, 2, 3]`, consider writing `t[0:3]` into `t[1:4]`:

```text
intended:     t[1], t[2], t[3] = old t[0], old t[1], old t[2]   ->  [0, 0, 1, 2]

element by element, in place:
  step 0: t[1] = t[0] = 0     t = [0, 0, 2, 3]
  step 1: t[2] = t[1] = 0     t = [0, 0, 0, 3]   t[1] was already overwritten
  step 2: t[3] = t[2] = 0     t = [0, 0, 0, 0]   wrong
```

The fix in kael is to clone the source first whenever case 3 applies. NumPy handles the same problem in `np.copyto` and ufuncs by detecting overlap and buffering. PyTorch detects some partial overlap cases and raises an error instead.

### When a new shape can share storage

`view` reinterprets the same elements under a new shape, in row-major order. That is only free when row-major logical order and storage order agree. For a contiguous tensor they do: logical element `k` is at storage position `offset + k`, so any new shape with contiguous strides reads the same sequence.

For a non-contiguous tensor they disagree. `a.T()` has logical order `0, 3, 1, 4, 2, 5`. A 1-d view of it would need a single stride that steps `0 -> 3 -> 1 -> 4`, which does not exist. Producing that order requires a copy.

PyTorch's `view` is smarter than this. It accepts some non-contiguous inputs, as long as the requested shape only splits or merges dimensions in ways that the existing strides can express. kael takes the simpler rule, described in [view versus reshape](#view-versus-reshape).

## Implementation

### The view constructor

Every view goes through one helper in an anonymous namespace. From `src/view.cpp`:

```cpp
Tensor make_view(const Tensor& base, Shape shape, Strides strides, std::int64_t offset) {
    auto impl = std::make_shared<TensorImpl>();
    impl->storage = base.impl()->storage;
    impl->shape = std::move(shape);
    impl->strides = std::move(strides);
    impl->offset = offset;
    impl->dtype = base.dtype();
    impl->device = base.device();
    return Tensor::from_impl(std::move(impl));
}
```

Three design points follow from this function:

- The new impl copies the `shared_ptr<Storage>`, so the storage lives as long as any view of it. A slice of a large tensor keeps the whole buffer alive. That is the same trade-off PyTorch and NumPy make.
- `Tensor::from_impl` calls the private constructor, which calls `impl_->validate()`. Validation rejects negative strides, negative offsets, a device mismatch, and, through `check_fits_in_storage`, any layout whose last element would fall outside the storage. The extent arithmetic uses `detail::checked_mul` and `detail::checked_add`, so a malicious or buggy layout throws instead of overflowing `int64`. No view method can produce an out-of-bounds tensor, even if its own arithmetic has a bug.
- The autograd metadata is not copied. Each public view method calls an `autograd::record_*` function after `make_view` (for example `record_transpose`) so the backward pass knows how to route gradients through the view. [06. Autograd](@/book/06-autograd.md) covers those functions.

The first user is the simplest view of all:

```cpp
Tensor Tensor::detach() const {
    return make_view(*this, shape(), strides(), offset());
}
```

`detach` is the one view that records nothing: it returns a tensor with the same storage and metadata and no history. `a.detach().shares_storage(a)` is true, and `a.detach().is_same(a)` is false. It is the equivalent of `torch.Tensor.detach`.

> [!TIP]
> Every view method in this section reduces to three questions: what is the new shape, what are the new strides, and what is the new offset. Answer them by hand on a small example before reading the code.

### transpose and permute

`transpose(dim0, dim1)` swaps two entries of shape and strides. `permute(dims)` reorders all of them. Both accept negative dims through `dim_position`.

From `src/view.cpp`:

```cpp
Tensor Tensor::transpose(std::int64_t dim0, std::int64_t dim1) const {
    if (dim() == 0) {
        wrap_dim(dim0, 1);
        wrap_dim(dim1, 1);
        return *this;
    }
    const std::size_t first = dim_position(dim0, dim());
    const std::size_t second = dim_position(dim1, dim());

    Shape new_shape = shape();
    Strides new_strides = strides();
    std::swap(new_shape[first], new_shape[second]);
    std::swap(new_strides[first], new_strides[second]);
    Tensor out = make_view(*this, std::move(new_shape), std::move(new_strides), offset());
    autograd::record_transpose(out, *this, dim0, dim1);
    return out;
}
```

A 0-d tensor accepts `dim0` and `dim1` in `[-1, 0]`, the same rule PyTorch uses. `wrap_dim(dim, 1)` validates the index as if the tensor had one dimension, and the function returns the tensor itself. `squeeze(dim)` follows the same rule.

Swapping shape and strides together is the whole trick. Element `[i, j]` of the transpose must be element `[j, i]` of the original, and swapping the strides makes `i * s1 + j * s0` equal to `j * s0 + i * s1`.

`permute` generalizes the swap. It checks that `dims` has one entry per dimension and that no dimension repeats, using a `std::vector<bool> seen`:

```cpp
Tensor Tensor::permute(const Dims& dims) const {
    KAEL_CHECK_VALUE(static_cast<std::int64_t>(dims.size()) == dim(),
                     "permute got " << dims.size() << " dimensions for a tensor with " << dim());

    std::vector<bool> seen(dims.size(), false);
    Shape new_shape(dims.size());
    Strides new_strides(dims.size());
    for (std::size_t i = 0; i < dims.size(); ++i) {
        const std::size_t source = dim_position(dims[i], dim());
        KAEL_CHECK_VALUE(!seen[source], "permute got dimension " << dims[i] << " more than once");
        seen[source] = true;
        new_shape[i] = shape()[source];
        new_strides[i] = strides()[source];
    }
    Tensor out = make_view(*this, std::move(new_shape), std::move(new_strides), offset());
    autograd::record_permute(out, *this, dims);
    return out;
}
```

Output dimension `i` takes the size and stride of input dimension `dims[i]`. A worked example on `x = arange(24).view({2, 3, 4})`, strides `[12, 4, 1]`:

```text
x.permute({1, 2, 0})
  out dim 0 <- in dim 1: size 3, stride 4
  out dim 1 <- in dim 2: size 4, stride 1
  out dim 2 <- in dim 0: size 2, stride 12
  shape [3, 4, 2]  strides [4, 1, 12]

element [2, 1, 1] = 2 * 4 + 1 * 1 + 1 * 12 = 21   (= x[1, 2, 1])
```

The check for repeats matters: `a.permute({0, 0})` would otherwise produce shape `[2, 2]` with strides `[3, 3]`, a tensor with internal overlap that drops a dimension. It throws `permute got dimension 0 more than once`.

`T()` is `transpose(0, 1)` for a 2-d tensor and the identity for 0-d and 1-d tensors. It throws for more than 2 dimensions and points to `mT()`:

```text
T needs at most 2 dimensions, got shape [2, 3, 4]; use transpose or mT instead
```

`mT()` is `transpose(-2, -1)`, the transpose of the last two dimensions, and requires at least 2 dimensions. It is the batched matrix transpose: `mT()` of shape `[2, 3, 4]` has shape `[2, 4, 3]`. PyTorch deprecated `T` on tensors with more than 2 dimensions for the same reason: reversing all dimensions is rarely what a batched computation wants.

### squeeze and unsqueeze

`squeeze()` drops every size-1 dimension together with its stride. `squeeze(dim)` drops one dimension if its size is 1 and returns `*this` otherwise, as in PyTorch. Neither changes any address, because a size-1 dimension only ever has index 0.

```text
zeros({2, 1, 3, 1}).squeeze()     shape [2, 3]     strides [3, 1]
zeros({2, 1, 3}).squeeze(0)       shape [2, 1, 3]  unchanged: size 2 is not 1
```

`unsqueeze(dim)` inserts a size-1 dimension. `dim` may range over `[-(dim()+1), dim()]`, because the new dimension can go at the end. The inserted stride never steps, so any value is legal, but kael picks one that keeps contiguity obvious:

```cpp
const std::size_t position = dim_position(dim, this->dim() + 1);
const std::int64_t new_stride =
    position < shape().size() ? shape()[position] * strides()[position] : 1;
```

The stride is the extent of the dimension it is inserted before, or 1 at the end. For `a` with shape `[2, 3]` and strides `[3, 1]`:

| Call | Shape | Strides |
|------|-------|---------|
| `a.unsqueeze(0)` | `[1, 2, 3]` | `[6, 3, 1]` |
| `a.unsqueeze(1)` | `[2, 1, 3]` | `[3, 3, 1]` |
| `a.unsqueeze(2)` | `[2, 3, 1]` | `[3, 1, 1]` |
| `a.unsqueeze(-1)` | `[2, 3, 1]` | `[3, 1, 1]` |

The result equals `contiguous_strides` of the new shape when the input is contiguous, so code that compares strides directly (and tools that print them) see a familiar layout. PyTorch computes the inserted stride the same way. On a 0-d tensor, `unsqueeze(0)` gives shape `[1]` with stride 1.

### expand and stride 0

`expand(shape)` makes a tensor look larger than its storage by repeating elements. It is how broadcasting is materialized as a view. The rule is the broadcasting rule from [01. Shapes and strides](@/book/01-shapes-and-strides.md): align shapes from the right; a size-1 dimension can grow to any size; new leading dimensions can be added; any other mismatch is an error.

A dimension that grows from 1 to `n` gets stride 0. New leading dimensions also get stride 0. `broadcast_strides` in `src/shape.cpp` computes exactly that:

```cpp
const std::size_t new_dims = target.size() - shape.size();
Strides result(target.size(), 0);
for (std::size_t i = 0; i < shape.size(); ++i) {
    const std::int64_t target_dim = target[i + new_dims];
    KAEL_CHECK_VALUE(shape[i] == target_dim || shape[i] == 1, /* ... */);
    if (shape[i] == target_dim) {
        result[i + new_dims] = strides[i];
    }
}
```

The result starts as all zeros, so new leading dimensions keep stride 0 without a special case. A dimension that already has the target size keeps its stride. A size-1 dimension that grows keeps the initial 0. A size-1 dimension that stays size 1 takes the branch `shape[i] == target_dim` and keeps its stride, which is harmless because it never steps.

`Tensor::expand` first resolves `-1` entries with `resolve_expand_shape`, then calls `broadcast_strides` and `make_view`:

```cpp
Tensor Tensor::expand(const Shape& requested) const {
    Shape target = resolve_expand_shape(shape(), requested);
    Strides target_strides = broadcast_strides(shape(), strides(), target);
    Tensor out = make_view(*this, std::move(target), std::move(target_strides), offset());
    autograd::record_expand(out, *this);
    return out;
}
```

`-1` keeps the current size of that dimension. It is not allowed for a new leading dimension, because there is no current size to keep.

Worked examples, all printed by a compiled check:

```text
v = from_vector({1.f, 2.f, 3.f})        shape [3], strides [1]
v.expand({2, 3})                        shape [2, 3]     strides [0, 1]
  row 0: offset 0 + 0*0 + j*1 -> 1 2 3
  row 1: offset 0 + 1*0 + j*1 -> 1 2 3

c = from_vector({1.f, 2.f, 3.f}, {3, 1})
c.expand({3, 4})                        shape [3, 4]     strides [1, 0]
  [[1, 1, 1, 1],
   [2, 2, 2, 2],
   [3, 3, 3, 3]]

c.expand({2, -1, 4})                    shape [2, 3, 4]  strides [0, 1, 0]
  -1 keeps size 3; new leading dim and the grown dim get stride 0
```

The `[3, 4]` view reads 12 logical elements from a 3-element buffer. Errors, also from real runs:

```text
v.expand({2, 2})    incompatible shapes [3] and [2, 2]
v.expand({-1, 3})   expand cannot use -1 for new dimension 0 in shape [-1, 3]
```

This is the same operation as `torch.Tensor.expand` and `numpy.broadcast_to`. NumPy marks the result read-only. PyTorch leaves it writable but rejects most in-place writes to it. kael does the same as PyTorch, through the check described next.

### Detecting internal overlap

`Tensor::has_internal_overlap` detects the case that kael's own views can create, a stride of 0 on a dimension larger than 1:

```cpp
bool Tensor::has_internal_overlap() const {
    for (std::size_t i = 0; i < shape().size(); ++i) {
        if (shape()[i] > 1 && strides()[i] == 0) {
            return true;
        }
    }
    return false;
}
```

A size-1 dimension with stride 0 is not overlap, because it has only one index. Every in-place operation calls `check_writable` in `src/ops/inplace.cpp`:

```cpp
void check_writable(const Tensor& self, const char* op) {
    KAEL_CHECK_VALUE(!self.has_internal_overlap(),
                     op << " cannot write to shape " << to_string(self.shape()) << " with strides "
                        << to_string(self.strides())
                        << " because several positions share one element; call clone() first");
}
```

The message for `from_vector({1.f, 2.f, 3.f}).expand({2, 3}).add_(1)`:

```text
add_ cannot write to shape [2, 3] with strides [0, 1] because several positions share one element; call clone() first
```

The check looks only for stride 0. That is the only way kael's own view methods create overlap: `expand` is the only one that sets a stride to 0, strides are never negative (validation rejects them), `permute` rejects repeated dimensions, and `slice` only multiplies strides by a positive step. A general overlap test for arbitrary strides, such as shape `[3, 2]` with strides `[1, 1]` built by hand, is not implemented. That layout cannot come out of the public view API.

### slice, select and narrow

`slice(dim, start, end, step)` keeps every `step`-th index in `[start, end)` of one dimension. It is the C++ form of Python's `t[start:end:step]` on one dimension.

From `src/view.cpp`:

```cpp
Tensor Tensor::slice(std::int64_t dim, std::int64_t start, std::int64_t end,
                     std::int64_t step) const {
    KAEL_CHECK_VALUE(step > 0, "slice step must be positive, got " << step);

    const std::size_t position = dim_position(dim, this->dim());
    const std::int64_t first = clamp_slice_bound(start, shape()[position]);
    const std::int64_t last = clamp_slice_bound(end, shape()[position]);
    const std::int64_t stride = strides()[position];
    const std::int64_t new_offset = detail::checked_add(
        offset(), detail::checked_mul(first, stride, "slice offset"), "slice offset");

    Shape new_shape = shape();
    Strides new_strides = strides();
    new_shape[position] = last > first ? (last - first - 1) / step + 1 : 0;
    new_strides[position] = detail::checked_mul(stride, step, "slice stride");
    Tensor out = make_view(*this, std::move(new_shape), std::move(new_strides), new_offset);
    autograd::record_slice(out, *this, dim, start, end, step);
    return out;
}
```

Bounds follow Python slicing: negative values count from the end, then both are clamped to `[0, size]`:

```cpp
std::int64_t clamp_slice_bound(std::int64_t bound, std::int64_t size) {
    return std::clamp(bound < 0 ? bound + size : bound, std::int64_t{0}, size);
}
```

The new metadata, in formula form:

```text
offset'      = offset + first * stride
shape'[dim]  = last > first ? (last - first - 1) / step + 1 : 0
stride'[dim] = stride * step
```

The size formula counts the indices `first, first + step, first + 2 * step, ...` that are below `last`. It is a ceiling division of `last - first` by `step`, written without floating point.

Worked traces on `r = arange(10)`, each confirmed by a run:

```text
r.slice(0, 1, 8, 3)
  first = 1, last = 8
  size   = (8 - 1 - 1) / 3 + 1 = 3
  stride = 1 * 3 = 3
  offset = 0 + 1 * 1 = 1
  reads storage 1, 4, 7

r.slice(0, -3, 100)
  first = clamp(-3 + 10) = 7, last = clamp(100) = 10
  size   = (10 - 7 - 1) / 1 + 1 = 3
  reads storage 7, 8, 9

r.slice(0, 5, 5)
  first = 5, last = 5, size 0
  shape [0], offset 5: an empty view whose offset is at its own end
```

On a multi-dimensional tensor only one dimension changes. `x = arange(24).view({2, 3, 4})` sliced with `x.slice(2, 1, 4, 2)` keeps columns 1 and 3 of every row: shape `[2, 3, 2]`, strides `[12, 4, 2]`, offset 1.

The offset and stride products use `checked_mul` and `checked_add`, so a huge step throws `ValueError` (a `kael::Error`) with the quantity name in the message, such as `slice stride overflows int64`, instead of wrapping. `step <= 0` throws `slice step must be positive, got 0`.

`select(dim, index)` fixes one index and removes the dimension:

```text
offset' = offset + index * stride
then erase shape[dim] and strides[dim]
```

Negative indices wrap once. An index outside `[-size, size)` throws through `KAEL_CHECK_INDEX`, so the error is an `IndexError`: `a.select(0, 2)` on shape `[2, 3]` fails with `select index 2 out of range for size 2 at dimension 0`. `a.select(1, 2)` gives shape `[2]`, stride `[3]`, offset 2: the last column. `select` is what Python's `t[i]` maps to in the bindings. Unlike `slice`, it does not clamp, because a single out-of-range index has no sensible clamped meaning.

`narrow(dim, start, length)` is a strict `slice(dim, start, start + length)`:

```cpp
Tensor Tensor::narrow(std::int64_t dim, std::int64_t start, std::int64_t length) const {
    const std::int64_t size = shape()[dim_position(dim, this->dim())];
    const std::int64_t first = start < 0 ? start + size : start;
    KAEL_CHECK_INDEX(first >= 0 && first <= size, /* ... */);
    KAEL_CHECK_VALUE(length >= 0 && length <= size - first, /* ... */);
    return slice(dim, first, first + length);
}
```

Unlike `slice`, it does not clamp. It throws when `start` or `length` falls outside the dimension: `arange(10).narrow(0, 8, 5)` fails with `narrow length 5 does not fit size 10 from start 8`. Use `narrow` where an out-of-range request is a bug, such as `cat` writing each input into its part of the output. PyTorch's `narrow` has the same strictness.

### split and chunk

Both return `std::vector<Tensor>` of `narrow` views along one dimension.

```cpp
std::vector<Tensor> Tensor::split(std::int64_t chunk_size, std::int64_t dim) const {
    KAEL_CHECK_VALUE(chunk_size > 0, "split size must be positive, got " << chunk_size);

    const std::int64_t total = size(dim);
    std::vector<Tensor> pieces;
    std::int64_t start = 0;
    do {
        pieces.push_back(narrow(dim, start, std::min(chunk_size, total - start)));
        start += chunk_size;
    } while (start < total);
    return pieces;
}
```

- `split(chunk_size, dim)` produces pieces of `chunk_size`, with a smaller last piece. `arange(5).split(2)` gives sizes 2, 2, 1 at offsets 0, 2, 4. The `do ... while` loop guarantees at least one piece, so splitting a size-0 dimension returns one empty view.
- `chunk(chunks, dim)` computes `chunk_size = ceil(total / chunks)` as `(total + chunks - 1) / chunks` and calls `split`. Like PyTorch, it can return fewer pieces than requested: `arange(6).chunk(4)` uses chunk size 2 and gives 3 pieces of size 2, and `arange(5).chunk(3)` gives sizes 2, 2, 1. If the dimension is empty, it returns `chunks` handles to one empty view, so `zeros({0}).chunk(3)` has 3 elements and `zeros({0}).split(2)` has 1.

```text
arange(6).chunk(4)
  chunk_size = (6 + 4 - 1) / 4 = 2
  pieces: [0, 1]  [2, 3]  [4, 5]          3 pieces, not 4

arange(5).chunk(3)
  chunk_size = (5 + 3 - 1) / 3 = 2
  pieces: [0, 1]  [2, 3]  [4]
```

Every piece is a view, so writing to a piece writes to the original. Attention code uses `chunk` to split a fused QKV projection into three tensors without copying.

### view versus reshape

`view` and `reshape` both reinterpret the elements in row-major order under a new shape. Both accept one `-1`, resolved by `infer_shape` from chapter 01. They differ in what they promise.

From `src/view.cpp`:

```cpp
Tensor Tensor::view(const Shape& requested) const {
    KAEL_CHECK_VALUE(is_contiguous(), "view needs a contiguous tensor, but shape "
                                          << to_string(shape()) << " has strides "
                                          << to_string(strides()) << "; use reshape instead");
    Shape target = infer_shape(requested, numel());
    Strides target_strides = contiguous_strides(target);
    Tensor out = make_view(*this, std::move(target), std::move(target_strides), offset());
    autograd::record_view(out, *this);
    return out;
}

Tensor Tensor::reshape(const Shape& requested) const {
    return contiguous().view(requested);
}
```

`view` never copies. It requires a contiguous input, because only then is "row-major order of the old shape" the same as "consecutive storage positions", which is what `contiguous_strides(target)` assumes. It keeps the offset, so a view of a contiguous slice still starts in the right place: `a.select(0, 1).view({3, 1})` has offset 3 and reads `3, 4, 5`.

`reshape` always succeeds when the element counts match. `contiguous()` returns `*this` when the tensor is already contiguous and `clone()` otherwise. So:

| Input | `view` | `reshape` |
|-------|--------|-----------|
| contiguous | shares storage | shares storage |
| non-contiguous | throws `ValueError` | copies into new storage, then views it |

Worked example with `t = a.T()` (shape `[3, 2]`, strides `[1, 3]`):

```text
t.view({6})       -> error: view needs a contiguous tensor, but shape [3, 2]
                     has strides [1, 3]; use reshape instead
t.reshape({6})    -> [0, 3, 1, 4, 2, 5], new storage
a.reshape({3, 2}) -> shares storage with a
```

The flattened order `0, 3, 1, 4, 2, 5` is the logical row-major order of `t`, not the storage order of `a`. No choice of strides on the original 6-element buffer can express that sequence as a single 1-d view, so a copy is unavoidable here.

kael's rule is deliberately simpler than PyTorch's. PyTorch's `reshape` returns a view whenever the requested shape is compatible with the existing strides, even for some non-contiguous inputs. An example where the two differ:

```text
x  = arange(24).view({2, 3, 4})        strides [12, 4, 1]
xt = x.transpose(0, 1)                 shape [3, 2, 4]  strides [4, 12, 1]
xt.reshape({3, 2, 2, 2})               splits only the last dimension

PyTorch: a view, strides [4, 12, 2, 1]
kael:    a copy (shares_storage is false), because xt is not contiguous
```

The cost is an occasional unnecessary copy; the gain is one line of code and a rule a reader can state in a sentence. The header comment in `include/kael/tensor.h` states it: "Shares memory with this tensor only if it is contiguous; otherwise it copies."

> [!WARNING]
> Because `reshape` may copy, writes to its result reach the original only when the input was contiguous. Code that relies on sharing must use `view`, which throws instead of copying silently.

`flatten(start_dim, end_dim)` and `unflatten(dim, sizes)` compute the target shape and call `reshape`, so they inherit the same rule:

```text
x.flatten(1)                    shape [2, 12]    shares storage (x is contiguous)
x.transpose(1, 2).flatten(1)    shape [2, 12]    new storage, values 0, 4, 8, 1, 5, 9, ...
arange(6).unflatten(0, {2, -1}) shape [2, 3]     -1 resolved against the size of dim 0
```

Two special cases in `flatten`: a 0-d tensor becomes shape `[1]`, and `start_dim == end_dim` returns `*this`. `unflatten` resolves `-1` with `infer_shape` against the size of the one dimension it replaces, not against the whole tensor.

### Materializing a layout

Three operations produce a fresh contiguous tensor from any layout, as described in [02. Memory model](@/book/02-memory-model.md):

- `clone()` allocates `empty(shape(), dtype(), device())`. For a contiguous source it copies `nbytes()` with `std::memcpy` (or `device::copy_within_device` on CUDA). For a non-contiguous source it calls `detail::convert_strided`.
- `contiguous()` is `is_contiguous() ? *this : clone()`.
- `to(DType)` allocates the target dtype and calls `convert_strided`, which converts each element.

The clone of a stride-0 view is a full, writable tensor: `from_vector({1.f, 2.f, 3.f}).expand({2, 3}).clone()` has strides `[3, 1]` and its own 24-byte storage. That is the remedy the overlap error message suggests.

`convert_strided` is the strided copy. It does not loop over indices itself. It builds a `TensorIterator` ([04. TensorIterator](@/book/04-tensor-iterator.md)) with the destination as output and the source as input, then runs an identity function through a double dtype dispatch. From `src/copy.cpp`:

```cpp
void convert_strided(Tensor& dst, const Tensor& src) {
    TensorIterator iterator;
    iterator.add_output(dst).add_input(src).build();
    if (iterator.on_device()) {
        device::convert(dst.dtype(), src.dtype(), iterator.device_layout());
        return;
    }
    auto identity = [](auto value) { return value; };
    dispatch(src.dtype(), [&]<typename From>() {
        dispatch(dst.dtype(), [&]<typename To>() { run_typed<To, From, 1>(iterator, identity); });
    });
}
```

The outer dispatch picks the source type and the inner one picks the destination type, so all 16 pairs of the four dtypes are instantiated. `run_typed<To, From, 1>` writes `convert_value<To>(identity(x))` for every element, so a float `2.5` becomes bool `1`, not the invalid byte `2` that a plain `static_cast` would produce.

Routing the copy through the iterator means the copy inherits what the iterator does: dimension coalescing, the contiguous fast path of the inner loop, and parallel execution with `parallel_for`. The iterator requires the destination shape to equal the source shape. It does not broadcast the source to the destination by itself. `convert_strided(zeros({2, 2}), from_vector({1.f, 2.f}))` throws `output shape [2, 2] does not match the shape [2] of the inputs`. Callers that want broadcasting expand the source first, as `copy_` does below.

A trace of `t.reshape({6})` for `t = a.T()`:

```text
reshape -> contiguous() -> is_contiguous() is false -> clone()
clone:   out = empty({3, 2}), strides [3, 1]
         convert_strided(out, t)
           iterator visits t in logical order:
             t[0,0]=0  t[0,1]=3  t[1,0]=1  t[1,1]=4  t[2,0]=2  t[2,1]=5
           writes them to out positions 0..5
then:    out.view({6})  -> [0, 3, 1, 4, 2, 5]
```

### copy_ and aliasing

`Tensor::copy_(source)` writes `source` into the existing layout of `*this`. It is the write half of every view-based algorithm: `cat` writes each input into `out.narrow(...)`, and `index_select` uses it in its general path.

From `src/ops/inplace.cpp`:

```cpp
Tensor& Tensor::copy_(const Tensor& source) {
    check_grad_rules(*this, &source, "copy_");
    check_writable(*this, "copy_");
    const autograd::NoGradGuard guard;
    const Tensor same_device = source.to(device());
    detail::convert_strided(*this, without_partial_aliasing(*this, same_device).expand(shape()));
    impl_->storage->bump_version();
    return *this;
}
```

Reading the steps in order:

1. `check_grad_rules` rejects writes that would corrupt the autograd graph ([06. Autograd](@/book/06-autograd.md)).
2. `check_writable` rejects a destination with internal overlap.
3. The source moves to the destination device. If it is already there, `to` returns it unchanged.
4. `without_partial_aliasing` clones the source if it shares storage with the destination but has a different layout.
5. `expand(shape())` broadcasts the source to the destination shape, so `copy_` from a `[2]` row into a `[2, 2]` matrix fills both rows: `zeros({2, 2}).copy_(from_vector({1.f, 2.f}))` gives `[[1, 2], [1, 2]]`.
6. The storage version is bumped.

The guard in step 4 is two small functions:

```cpp
bool same_layout(const Tensor& a, const Tensor& b) {
    return a.shape() == b.shape() && a.strides() == b.strides() && a.offset() == b.offset();
}

Tensor without_partial_aliasing(const Tensor& self, const Tensor& input) {
    return input.shares_storage(self) && !same_layout(input, self) ? input.clone() : input;
}
```

They implement the three cases from [Aliasing between operands](#aliasing-between-operands). No sharing returns the input. Same layout returns the input. Sharing with a different layout clones it. Consider `t = arange(0, 4)` and `t.slice(0, 1, 4).copy_(t.slice(0, 0, 3))`. The source reads elements 0, 1, 2 and the destination writes elements 1, 2, 3. Run element by element in place, the result would be `[0, 0, 0, 0]`, as traced earlier. Cloning the source first gives the intended `[0, 0, 1, 2]`. The test `Copy.OverlappingWithItself` covers this case.

The same guard runs inside every in-place binary operation. For `t.slice(0, 1, 4).add_(t.slice(0, 0, 3))` on `t = [0, 1, 2, 3]`:

```text
with the clone (kael):   t[1..3] += old [0, 1, 2]       ->  [0, 1, 3, 5]
without it, in order:    t[1] = 1 + 0 = 1
                         t[2] = 2 + t[1] = 3
                         t[3] = 3 + t[2] = 6            ->  [0, 1, 3, 6]   wrong
```

`InPlace.PartiallyOverlappingOperandIsReadBeforeWriting` covers this pattern for `add_`.

When the source is exactly the destination, as in `t.mul_(t)`, each output element depends only on the input element at the same address, so no clone is needed and none is made. `t = [0, 1, 2, 3]` becomes `[0, 1, 4, 9]` (`InPlace.SelfAsOperand`).

The guard is conservative. Two views can share storage and differ in layout without touching the same elements (for example the even and odd elements of one vector), and the guard still clones. A precise test would need to intersect two strided index sets, which is expensive in general. A spare copy in a rare case is cheaper than that complexity.

### The version counter

Views make it possible to change a tensor's values through a different handle. Autograd must notice that, because a backward formula that saved `x` would otherwise compute gradients from the new values. kael stores a counter on the shared `Storage`. From `include/kael/storage.h`:

```cpp
[[nodiscard]] std::int64_t version() const noexcept { return version_; }
void bump_version() noexcept { ++version_; }
```

`Tensor::version()` returns `impl_->storage->version()`. Because the counter lives on the storage and not on the `TensorImpl`, every view of a buffer reports the same version, and a write through any of them is visible to all. Every in-place operation in `src/ops/inplace.cpp` calls `bump_version()` after writing: `binary_inplace` (used by `add_`, `sub_`, `mul_`, `div_`, `pow_`), `clamp_`, `copy_`, `masked_fill_` and `scatter_add_`. `fill_` and `zero_` delegate to `copy_`, and `index_add_` delegates to `scatter_add_`. Creating a view does not bump the counter.

Compiled check:

```cpp
Tensor a = arange(6).view({2, 3});
// a.version() == 0
Tensor v = a.view({6});
v.add_(1);
// a.version() == 1, a == [[1, 2, 3], [4, 5, 6]]
```

A copy made by `reshape` has its own storage and its own counter. For `c = a.T().reshape({6})`, `c.add_(1)` leaves `a.version()` at 0 and moves `c.version()` to 1.

`autograd::SavedTensor` in `src/autograd/recording.h` records `t.version()` when a tensor is saved for backward and checks it again in `get()`:

```cpp
explicit SavedTensor(const Tensor& t) : data_(t.detach()), version_(t.version()) {}

[[nodiscard]] const Tensor& get() const {
    KAEL_CHECK(data_.version() == version_,
               "one of the variables needed for gradient computation has been modified by an "
               "inplace operation: expected version "
                   << version_ << ", got " << data_.version());
    return data_;
}
```

A mismatch throws "one of the variables needed for gradient computation has been modified by an inplace operation". PyTorch uses the same mechanism and the same message prefix, and exposes the counter as `tensor._version`. [06. Autograd](@/book/06-autograd.md) shows where tensors are saved.

## Edge cases and pitfalls

- **Writing through an expanded view.** `expand` returns stride 0. The header comment says "do not write to it", and `check_writable` enforces it for every in-place op. Call `clone()` first to get a writable copy (`InPlace.ExpandedTensorThrows`).
- **Partial aliasing in `copy_` and in-place binary ops.** A source that shares storage with a different layout is cloned before the write. Without that, the result depends on traversal order.
- **`reshape` hides a copy.** After `t.reshape(...)` on a non-contiguous `t`, writes to the result do not reach `t`. Use `view` when sharing is required: it throws instead of silently copying. `flatten` and `unflatten` share this behavior.
- **`convert_strided` does not broadcast.** Its source must already have the destination shape. Expand the source first, as `copy_` does.
- **Empty tensors.** An empty tensor is contiguous. A slice with `start >= end` has size 0 and keeps a valid offset. `Tensor::first_element` returns the storage base for an empty tensor, because an empty view can carry an offset past the end of its storage. `chunk` of an empty dimension returns `chunks` empty views; `split` returns one.
- **0-d tensors.** `transpose` and `squeeze(dim)` accept dims in `[-1, 0]` and return the tensor itself. `flatten` turns a 0-d tensor into shape `[1]`. `unsqueeze(0)` on a 0-d tensor gives shape `[1]` with stride 1. `ZeroDimensional.TransposeAndSqueezeAcceptDimZero` checks that `transpose(0, 1)` and `squeeze(1)` throw.
- **Overflow in view arithmetic.** `slice` uses `checked_mul` and `checked_add` for the offset and stride. `validate` recomputes the extent with checked math for every new impl, so even a layout that slipped through cannot point outside the storage.
- **Negative steps.** Not supported. `slice` throws for `step <= 0` and `validate` rejects negative strides. The Python bindings also reject `t[::-1]` with `ValueError` (`test_negative_step_is_rejected` in `python/tests/test_indexing.py`). Reverse an axis with an index operation, which copies.
- **Lifetime.** A view holds the storage. A small slice of a large tensor keeps the whole buffer alive until the slice is destroyed. A 2-element slice of `arange(1000000)` keeps 4000000 bytes alive. `clone()` the slice to release it: the clone's storage holds 8 bytes.
- **`T` on batched tensors.** `T()` throws for more than 2 dimensions. Use `mT()` for the last two dimensions or `permute` for anything else.

## How it is tested

`tests/test_view.cpp` holds one GoogleTest suite per operation:

| Suite | What it proves |
|-------|----------------|
| `IsContiguous` | size-1 dimensions do not matter, transposes and gaps are not contiguous, empty tensors always are |
| `Contiguous`, `Clone`, `To` | `contiguous()` returns the same impl when possible; `clone()` of a view does not share storage and is contiguous, including stride-0 views and views with an offset; `to` works on non-contiguous input |
| `InferShape` | `-1` resolution and its errors |
| `View` | shares storage, writes reach the original, keeps the offset, works on 0-d, rejects non-contiguous input |
| `Reshape` | `ContiguousShares` and `NonContiguousCopiesInLogicalOrder` |
| `Transpose`, `Permute`, `MatrixTranspose` | stride swaps, inverses, negative dims, `T` and `mT` rules |
| `Squeeze`, `Unsqueeze` | removal and insertion of size-1 dims and round trips |
| `Expand` | stride 0 for new and size-1 dims, `-1`, errors, `HasInternalOverlap` |
| `Slice`, `Select`, `Narrow` | step arithmetic, clamping, negative bounds, empty results, errors |
| `Flatten`, `Unflatten`, `Split`, `Chunk` | dimension ranges and PyTorch-compatible piece sizes |
| `ViewChain` | chains of views, plus `GeneratedNumPyCases` |
| `ZeroDimensional` | dim 0 and -1 accepted on scalars |

`ViewChain.GeneratedNumPyCases` is the differential test: it checks kael against an independent implementation, NumPy, on inputs neither was tuned for. `scripts/generate/view_cases.py` (seed 7, 60 cases) works like this:

```text
for each case:
  shape    = 1 to 4 dimensions, each of size 1 to 4
  array    = numpy.arange(prod(shape)).reshape(shape)
  chain    = 1 to 4 random operations from:
             transpose, permute, slice, select, unsqueeze, squeeze, expand, reshape(-1)
  record   = initial shape, the chain, final shape, final values (raveled)
write all cases to tests/generated/view_cases.inc
```

Slices use random bounds in `[-size - 1, size + 1]` and steps 1 to 3, so clamping and negative bounds are exercised. A generated line looks like this:

```text
{{2, 4, 1}, {{ViewOp::Permute, {2, 0, 1}}}, {1, 2, 4}, {0, 1, 2, 3, 4, 5, 6, 7}},
```

The test replays each chain in kael on an iota tensor (values equal to their own indices) and compares the shape, the values read through the view, and the values of `clone()` of the view. Because the input values are their own indices, any wrong stride or offset shows up as a wrong number, and the failing case number is printed through `SCOPED_TRACE`. Comparing `clone()` as well checks `convert_strided` on the same layouts.

`tests/test_ops_inplace.cpp` covers the write side: `InPlace.ExpandedTensorThrows`, `InPlace.PartiallyOverlappingOperandIsReadBeforeWriting`, `InPlace.SelfAsOperand`, `InPlace.ViewsSeeTheChange`, `Copy.BetweenDifferentStrides`, `Copy.ConvertsAndBroadcasts` and `Copy.OverlappingWithItself`. `tests/test_kernels.cpp` has `StridedCopy.ConvertsThroughIterator`. The golden tests in [05. Operations](@/book/05-operations.md) apply `transpose`, `permute`, `slice` and `expand` views to inputs before every operation, so every op is also checked on non-contiguous layouts.

From Python, `test_views_share_memory` in `python/tests/test_tensor.py` and `test_getitem_is_a_view` in `python/tests/test_indexing.py` check sharing through the bindings. The first writes through `view`, `T`, `transpose` and `flatten` of a contiguous tensor and reads the changes back from the original.

## Build it yourself

1. Write `TensorImpl::is_contiguous` with the two relaxations (empty, size-1). Checkpoint: the four `IsContiguous` tests pass.
2. Write `make_view` and make sure it goes through `Tensor::from_impl` so `validate` runs. Write `detach` as the first user. Checkpoint: a detached tensor `shares_storage` with its source.
3. Write `clone` with the contiguous `memcpy` path only, and `contiguous`. Checkpoint: `Contiguous.ContiguousReturnsSameImpl`.
4. Write `transpose`, `permute`, `T`, `mT`. Checkpoint: `Transpose.*`, `Permute.*`, `MatrixTranspose.*`.
5. Write `detail::convert_strided` on top of the `TensorIterator` from [04. TensorIterator](@/book/04-tensor-iterator.md) (or, if you build in book order, a temporary odometer loop that walks `index_from_linear` and `offset_from_index`, replaced later). Use it in `clone` for non-contiguous input and in `to(DType)`. Checkpoint: `Contiguous.TransposedValuesInLogicalOrder`, `Clone.OfViewDoesNotShareAndIsContiguous`, `To.WorksOnNonContiguous`.
6. Write `view` and `reshape`. Checkpoint: `View.*` and `Reshape.*`; `a.T().view({6})` throws and `a.T().reshape({6})` returns `[0, 3, 1, 4, 2, 5]`.
7. Write `squeeze` (both overloads) and `unsqueeze` with the stride rule above. Checkpoint: `Squeeze.*`, `Unsqueeze.*`.
8. Write `expand` on top of `broadcast_strides`, and `has_internal_overlap`. Checkpoint: `Expand.*`; `from_vector({1.f, 2.f, 3.f}).expand({2, 3}).strides()` is `[0, 1]`.
9. Write `slice` with clamped bounds and checked arithmetic, then `select` and `narrow`. Checkpoint: `Slice.*`, `Select.*`, `Narrow.IsAStrictSlice`.
10. Write `flatten`, `unflatten`, `split`, `chunk`. Checkpoint: `Flatten.*`, `Unflatten.*`, `Split.*`, `Chunk.*`.
11. Run `python3 scripts/generate/view_cases.py` (needs NumPy) and include the generated file in a `ViewChain` test. Checkpoint: `ViewChain.GeneratedNumPyCases` passes.
12. Add `Storage::version` and `bump_version`, then `copy_` with `check_writable` and `without_partial_aliasing`. Checkpoint: `Copy.*` and `InPlace.ExpandedTensorThrows`.

The temporary odometer loop of step 5 can be this small, before the iterator exists:

```cpp
// Temporary: replaced by the TensorIterator version in chapter 04.
void convert_strided(Tensor& dst, const Tensor& src) {
    dispatch(src.dtype(), [&]<typename From>() {
        dispatch(dst.dtype(), [&]<typename To>() {
            for (std::int64_t i = 0; i < src.numel(); ++i) {
                const Shape index = index_from_linear(i, src.shape());
                dst.at<To>(index) = convert_value<To>(src.at<From>(index));
            }
        });
    });
}
```

It is slow, because it recomputes every index and offset from scratch, but it is correct for any strides and gives the later iterator version a reference to test against.

Run a subset with the built test binary:

```bash
cmake --build build
build/tests/kael_tests --gtest_filter='IsContiguous.*:View.*:Reshape.*:Expand.*:Slice.*:ViewChain.*:Copy.*'
```

## Check your understanding

<details>
<summary>Why does transpose not copy data?</summary>

A transpose only changes which storage position each logical index maps to. Swapping the two entries of the shape and of the strides makes element `[i, j]` of the result land at the same address as element `[j, i]` of the input. The bytes are already in the right place; only the metadata changes, so `make_view` builds a new impl over the same storage.

</details>

<details>
<summary>What are the shape, strides and offset of arange(12).view({3, 4}).slice(1, 1, 4, 2)?</summary>

The input has strides `[4, 1]`. On dimension 1, `first = 1` and `last = 4`, so the size is `(4 - 1 - 1) / 2 + 1 = 2`, the stride is `1 * 2 = 2`, and the offset is `0 + 1 * 1 = 1`. The result has shape `[3, 2]`, strides `[4, 2]` and offset 1. It reads storage elements 1, 3, 5, 7, 9 and 11.

</details>

<details>
<summary>Why does view refuse a transposed tensor while reshape accepts it?</summary>

`view` computes contiguous strides for the new shape and keeps the old offset, which is only correct when logical row-major order equals storage order. For a transposed tensor it does not: the logical order of `a.T()` is `0, 3, 1, 4, 2, 5`, which no single stride can express over the original buffer. `view` throws rather than return wrong values. `reshape` calls `contiguous()` first, which clones the tensor into a contiguous layout, and then views the copy.

</details>

<details>
<summary>Why is writing to an expanded tensor rejected, while reading from it is fine?</summary>

`expand` sets the stride of every grown dimension to 0, so several logical positions refer to one storage element. Reading the same element several times is harmless. Writing would update that element once per logical position, so the result would depend on the number and order of visits, and a parallel loop would race. `check_writable` detects the stride-0 pattern through `has_internal_overlap` and throws before any write.

</details>

<details>
<summary>In t.slice(0, 1, 4).copy_(t.slice(0, 0, 3)), why does kael clone the source, and why does it not clone in t.mul_(t)?</summary>

In the first case the source and destination share storage at shifted positions. An in-place, element-by-element copy would overwrite element 1 before it is read as the source of element 2, producing `[0, 0, 0, 0]` instead of `[0, 0, 1, 2]`. `without_partial_aliasing` sees shared storage with a different layout and clones the source. In `t.mul_(t)` the layouts are identical, so every element is read and then written at the same address, and the result is correct without a copy.

</details>

<details>
<summary>Why does the version counter live on Storage and not on TensorImpl?</summary>

Views of one buffer have different impls but share the storage. A write through any view changes the bytes that every other view reads. If the counter lived on the impl, a tensor saved for backward would not notice a write made through a different view. On the storage, every view reports the same version, so `SavedTensor::get()` detects the modification no matter which handle wrote it.

</details>

## Further reading

- [Edward Z. Yang, "PyTorch internals"](http://blog.ezyang.com/2019/05/pytorch-internals/): the storage, sizes, strides and offset model this chapter follows, with diagrams of views over one storage.
- [PyTorch, "Tensor Views"](https://pytorch.org/docs/stable/tensor_view.html): the list of PyTorch operations that return views, and the note that `reshape` may or may not return one.
- [NumPy, "The N-dimensional array (ndarray)"](https://numpy.org/doc/stable/reference/arrays.ndarray.html): the section on internal memory layout and strides, including negative strides that kael rejects.
- [NumPy, `numpy.lib.stride_tricks.as_strided`](https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.as_strided.html): what goes wrong when overlapping views are written, from the library that allows arbitrary strides.
- [NumPy, "Broadcasting"](https://numpy.org/doc/stable/user/basics.broadcasting.html): the rule that `expand` turns into stride-0 views.
- [kael `src/view.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/view.cpp): every view in under 300 lines, each a few lines of shape, stride and offset arithmetic.

Next: [04. TensorIterator](@/book/04-tensor-iterator.md)
