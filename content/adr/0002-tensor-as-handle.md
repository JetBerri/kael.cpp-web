+++
title = "Tensor as a shared handle"
description = "Why copying a kael Tensor shares its data, how the three layers Tensor, TensorImpl and Storage divide the work, and what that sharing costs."
weight = 2
slug = "tensor-as-handle"

[extra]
kicker = "ADR 0002"
source = "docs/adr/0002-tensor-as-handle.md"
+++

The line `Tensor b = a;` looks trivial. Its meaning decides how views work, how autograd finds the tensors a user holds, what it costs to pass a tensor to a function, and which bugs readers will hit. kael makes `Tensor` a shared handle: copying it copies a pointer, and both copies refer to the same tensor. This record explains why, which alternatives were rejected, and which guards the library adds to make shared mutable state safe.

Two terms are used throughout. **Value semantics** means a copy is an independent object, as with `std::vector` or `int`. **Handle semantics** means a copy is another name for the same object, as with `std::shared_ptr` or a Python variable.

## Status

Accepted in phase 3 (tag `chapter-03-core`, which matches [Memory model](@/book/02-memory-model.md) in the book). Still in force at `v1.0.0` and at the current head of the repository.

## Context

C++ programmers expect value semantics by default. Copy a `std::vector<float>` and the copy owns its own buffer:

```text
value semantics                      handle semantics

a: [1 2 3]                           a ----+
b = a                                      +--> [1 2 3]
b: [1 2 3]   (separate buffer)       b ----+
b[0] = 9                             b[0] = 9
a: [1 2 3]   b: [9 2 3]              a: [9 2 3]   b: [9 2 3]
```

PyTorch and NumPy use handle semantics. In Python, `b = a` binds a second name to the same array, and `b[0] = 9` is visible through `a`.

Value semantics conflicts with three other parts of kael.

- **Views.** `transpose`, `slice`, `expand`, `view` and `detach` ([Views](@/book/03-views.md)) return tensors that read and write the memory of their base. A transpose of a 1,000 by 1,000 matrix must not copy 4 MB. Several tensor objects must therefore share one buffer, which is already handle semantics for the buffer.
- **Autograd.** Graph nodes ([Autograd](@/book/06-autograd.md)) keep references to forward tensors, and `AccumulateGrad` writes gradients into the leaf tensors the user holds. If the graph held copies, `w.grad()` on the user's `w` would stay empty while the gradient landed in a copy nobody can reach.
- **Passing and storing.** Functions take tensors, return tensors and store them in containers such as a module's parameter list. With value semantics each of those steps copies the whole buffer unless every signature is written with references or moves, and a single missed `&` turns into a silent multi-megabyte copy.

## Decision

`Tensor` holds one `std::shared_ptr<TensorImpl>` and nothing else. Copying a `Tensor` copies the pointer and increments a reference count. The design has three layers:

```text
Tensor a ---------+
                  +--> TensorImpl #1 ----------+
Tensor b = a -----+    shape    [2, 3]         |
                       strides  [3, 1]         |
                       offset   0              |
                       dtype    float32        +--> Storage
                       device   cpu            |    24 bytes
                       autograd (grad, grad_fn)|    version 0
                                               |
Tensor t = a.T() ----> TensorImpl #2 ----------+
                       shape    [3, 2]
                       strides  [1, 3]
                       offset   0
```

| Layer | Holds | Shared by |
|-------|-------|-----------|
| `Tensor` | one `std::shared_ptr<TensorImpl>` | nothing; it is the handle |
| `TensorImpl` | shape, strides, offset, dtype, device, autograd state, `std::shared_ptr<Storage>` | every copy of the same `Tensor` |
| `Storage` | the bytes, their size, the device, a version counter | every view of the same data |

From [`include/kael/tensor_impl.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/tensor_impl.h):

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
    // ...
};
```

`b = a` shares the `TensorImpl`, so `a` and `b` are the same tensor: same shape, same `grad`, same `grad_fn`. A view such as `a.T()` gets a new `TensorImpl` with its own shape and strides but the same `Storage`. That split is what lets a view have different metadata while sharing the data.

`Storage` is not copyable: its copy constructor and copy assignment are deleted. It owns its bytes through a `std::unique_ptr` with a custom deleter, or, for memory created with `Storage::external` such as a memory-mapped safetensors file ([Loading GPT-2](@/book/12-model-loading.md)), it keeps the external owner alive. Either way, the only way to share a buffer is to share the `Storage` object through its `shared_ptr`.

### Deep copy is explicit

`clone()` is the deep copy. It returns a new contiguous tensor with its own storage, whatever the layout of the source. Cloning a 3-element `int64` tensor that starts at offset 2 of a 5-element storage returns a tensor with offset 0 and a 24-byte storage, sized for its 3 elements only (`Tensor.CloneOfOffsetTensorStartsAtZero`).

### No empty handles

`Tensor` has no default constructor. Its only constructor is private and validates the impl. From [`src/tensor.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/tensor.cpp):

```cpp
Tensor::Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {
    KAEL_CHECK(impl_ != nullptr, "cannot create a tensor from a null impl");
    impl_->validate();
}

Tensor Tensor::from_impl(std::shared_ptr<TensorImpl> impl) {
    return Tensor(std::move(impl));
}
```

`Tensor::from_impl` is the only way to wrap an impl. `validate` checks that the impl has a storage, that shape and strides have the same rank, that no stride or offset is negative, that the device matches the storage device, and that the last reachable element stays inside the storage. An impl that would read past its buffer never becomes a `Tensor` (`Tensor.FromImplValidates`).

### A worked example

Trace four lines and watch the reference counts.

```text
Tensor a = kael::zeros({2, 3});   impl #1 (refs 1), storage S (refs 1), 24 bytes
Tensor b = a;                     impl #1 (refs 2), storage S (refs 1)
Tensor t = a.T();                 impl #2 (refs 1), storage S (refs 2)
Tensor c = a.clone();             impl #3 (refs 1), storage S2 (refs 1), 24 bytes

a.is_same(b)         true    same impl
a.is_same(t)         false   different impl
a.shares_storage(t)  true    same storage
a.shares_storage(c)  false   clone has its own storage
```

Write `b.data_ptr<float>()[4] = 42` and read `a.data_ptr<float>()[4]`: it is 42 (`Tensor.CopySharesData`). Read the element of `t` at row 1, column 1: its storage index is `0 + 1*1 + 1*3 = 4`, so it is 42 too. Read `c`: it still holds 0.

`kael::explain` ([Serving and tooling](@/book/13-serving-and-tooling.md)) prints the storage reference count as `references: N`, which makes this sharing visible while debugging.

## Alternatives considered

### Value semantics with separate view types

`Tensor` behaves like `std::vector`, and views are distinct types, as with `std::span` over a vector or Eigen's `Map` and `Block` over a matrix.

- Gains: no aliasing surprises; `b = a` is a real copy, as C++ readers expect.
- Costs: two or more types for every tensor concept. Every operation must accept both owning tensors and views. A view must not outlive its owner, which the type system does not check, so lifetimes become the user's problem. Autograd must hold either copies (wrong object for gradients) or raw references (dangling when the user's tensor goes out of scope).
- Rejected because views and autograd both need several long-lived objects to refer to one buffer.

### Copy-on-write

`b = a` shares the buffer until either side writes, then the writer copies. Qt's implicitly shared containers and Swift's arrays work this way.

- Gains: cheap copies with the appearance of value semantics.
- Costs: a view that writes through to its base is exactly what copy-on-write forbids, so `x.slice(0, 0, 1).fill_(0)` would stop modifying `x`. Every write needs a reference-count check, and the copy happens at an unpredictable point, sometimes inside a kernel.
- Rejected because in-place writes through views are a feature of the library, not an accident.

### A nullable handle

Handle semantics, but with a default constructor that creates an empty handle. PyTorch's `at::Tensor` works this way and exposes `defined()` to test it.

- Gains: tensors can be declared before they are assigned, and members of type `Tensor` need no initializer.
- Costs: every member function either checks for an empty handle or has undefined behavior on one. The empty state leaks into every API as a possible input.
- Rejected. kael represents a possibly missing tensor as `std::optional<Tensor>`, which makes absence visible in the type. `Tensor::grad()` returns `std::optional<Tensor>`, and graph nodes pass gradients as `std::vector<std::optional<Tensor>>`.

### An intrusive reference count

PyTorch's `TensorImpl` stores its own reference count and is managed by `c10::intrusive_ptr`, which saves the separate control block of `std::shared_ptr`.

- Gains: one allocation per impl instead of up to two, and a pointer half the size of a `shared_ptr`.
- Costs: custom smart-pointer code that a reader must learn before reading the tensor class.
- Rejected for clarity. kael is a teaching framework, and `std::shared_ptr` is the standard tool every C++ reader knows. `std::make_shared` already folds the control block into the same allocation.

## Consequences

### Assignment aliases

`b = a` does not copy. A write through `b` is visible through `a`. This matches PyTorch and NumPy and surprises readers who expect C++ value semantics. Use `clone()` when an independent copy is needed. The same holds for function arguments: a function that takes `Tensor t` by value and calls `t.add_(1)` modifies the caller's tensor.

### The rule of zero

`Tensor` follows the rule of zero: it declares no copy constructor, move constructor, assignment operators or destructor, and the compiler-generated ones are correct because the only member is a `std::shared_ptr`. Copying increments an atomic reference count. Moving transfers the pointer without touching the count.

### No null checks in member functions

Every `Tensor` refers to a validated impl, so member functions such as `shape()` and `dtype()` dereference `impl_` directly and can be `noexcept`. Optional tensors use `std::optional<Tensor>`.

### Identity is testable

Two identity queries make aliasing testable. From [`include/kael/tensor.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/tensor.h):

```cpp
[[nodiscard]] bool is_same(const Tensor& other) const noexcept { return impl_ == other.impl_; }

[[nodiscard]] bool shares_storage(const Tensor& other) const noexcept {
    return impl_->storage == other.impl_->storage;
}
```

`is_same` asks whether two handles are the same tensor. `shares_storage` asks whether writes through one can be visible through the other. View tests assert `shares_storage`, and `clone` tests assert its negation. `nn::Module` uses the same identity, keyed on the impl pointer, to list a parameter registered under two names only once (`NnModule.SharedParameterIsListedOnce`).

### Autograd writes into the user's tensor

`AccumulateGrad` holds a `std::shared_ptr<TensorImpl>` to its leaf ([`include/kael/autograd/node.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/autograd/node.h)). After `backward`, the gradient sits in that impl's `AutogradMeta`, and every handle to the leaf, including the one the optimizer holds, sees it through `grad()`.

### Shared mutable state needs guards

Sharing makes mutation visible in places that did not perform it. Two guards keep that from producing silently wrong results.

- **The version counter.** `Storage` carries a counter that every in-place operation increments with `bump_version`. When autograd saves a tensor for the backward pass, it records the version. At backward time, a different version means someone modified the saved data, and the engine throws `one of the variables needed for gradient computation has been modified by an inplace operation` ([Autograd](@/book/06-autograd.md)). The counter lives on the storage, not on the impl, so a write through any view or through `detach()` is caught (`AutogradVersion.ModifyingSavedOutputThroughDetachThrows`).
- **The overlap check.** In-place operations reject tensors in which several positions map to one element, such as the output of `expand`, which has stride 0 ([Views](@/book/03-views.md)). Writing `add_(1)` to such a tensor would add 1 to the same element several times. The error says `call clone() first` (`InPlace.ExpandedTensorThrows`).

A worked example of the version counter, from `AutogradVersion.ModifyingSavedInputThrows`:

```text
x = leaf([1, 2])               requires grad
constant = [3, 4]              storage version 0
y = (x * constant).sum()       MulBackward saves constant at version 0
constant.add_(1)               constant is now [4, 5], version 1
y.backward()                   MulBackward needs constant: version 1 != 0, throws
```

Without the check, the gradient of `x` would be computed as `[4, 5]` instead of the correct `[3, 4]`.

When an in-place operation reads an operand that partially overlaps its own output, kael clones that operand first (`without_partial_aliasing` in [`src/ops/inplace.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/ops/inplace.cpp)), so the result does not depend on the order in which elements are written.

### Device moves are visible to every handle

`move_to_` changes the device of an existing impl in place: it replaces the impl's storage, strides, offset and device. Every handle to that impl sees the move. `nn::Module::to` calls `move_to_` on each parameter, so an optimizer that captured the parameters before the move keeps working after it. A view created before the move has its own impl and keeps referring to the old storage on the old device.

## Where it shows up in the code

| File | What to look for |
|------|------------------|
| [`include/kael/tensor.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/tensor.h) | the single `impl_` member, `from_impl`, the private constructor, `is_same`, `shares_storage`, `clone`, `move_to_` |
| [`include/kael/tensor_impl.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/tensor_impl.h) | `TensorImpl` and `validate` |
| [`include/kael/storage.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/storage.h) | deleted copy operations, `Storage::external`, `version` and `bump_version` |
| [`src/tensor.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/tensor.cpp) | constructor validation, `clone`, `move_to_` |
| [`src/view.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/view.cpp) | `make_view`, which builds a new impl around the base's storage, and `has_internal_overlap` |
| [`src/autograd/recording.h`](https://github.com/JetBerri/kael.cpp/blob/main/src/autograd/recording.h) | `SavedTensor` and its version check |
| [`src/ops/inplace.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/ops/inplace.cpp) | `check_writable`, `without_partial_aliasing`, `bump_version` calls |
| [`src/nn/module.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/nn/module.cpp) | parameter deduplication by impl and `Module::to` |
| [`tests/test_tensor.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_tensor.cpp) | `Tensor.CopySharesData`, `Tensor.CloneDoesNotShareData`, `Tensor.FromImplValidates` |
| [`tests/test_autograd_engine.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_autograd_engine.cpp) | the `AutogradVersion` tests |

Chapters that depend on this decision:

- [Memory model](@/book/02-memory-model.md): the three layers and their ownership.
- [Views](@/book/03-views.md): every view is a new impl over shared storage.
- [Autograd](@/book/06-autograd.md): saved tensors, version checks and `AccumulateGrad`.
- [Neural networks and training](@/book/07-neural-networks.md): parameters, optimizers and `Module::to`.
- [Python bindings](@/book/10-python-bindings.md): a Python `kael.Tensor` object wraps a handle, so Python assignment and C++ copy mean the same thing.

## Check your understanding

<details>
<summary>After Tensor b = a; b.fill_(1), what does a hold, and which query tells you so without reading elements?</summary>

`a` holds ones too, because `a` and `b` share one `TensorImpl` and therefore one `Storage`. `a.is_same(b)` returns `true`, which implies `a.shares_storage(b)`.

</details>

<details>
<summary>Why does a.T() create a new TensorImpl instead of sharing a's impl?</summary>

The transpose has different shape and strides from `a`, and those live in the impl. Sharing the impl would change `a` itself. A new impl with swapped strides that points at the same `Storage` gives a different view of the same bytes.

</details>

<details>
<summary>What would go wrong in autograd if Tensor had value semantics?</summary>

The graph would hold copies of the leaves. `AccumulateGrad` would write the gradient into its copy, and the user's tensor and the optimizer's reference would never see it. Saved tensors would also be full copies of every forward input, which multiplies memory use.

</details>

<details>
<summary>Why does kael use std::optional&lt;Tensor&gt; instead of a default-constructed empty Tensor?</summary>

An empty handle forces every member function to check for it or risk undefined behavior. With no default constructor and a validating private constructor, every `Tensor` refers to a valid impl, so members dereference `impl_` directly. `std::optional<Tensor>` makes the possibility of absence explicit in the type, for example in `grad()`.

</details>

<details>
<summary>The version counter is stored on Storage, not on TensorImpl. Why does that matter?</summary>

Writes can arrive through any view or through `detach()`, each with its own impl. Only the storage is common to all of them. A counter on the impl would miss a write through another view of the same data, and autograd would compute gradients from modified values without an error.

</details>

<details>
<summary>A model calls module.to(Device::CUDA) after the optimizer was built. Why does the optimizer still update the right tensors?</summary>

`Module::to` calls `move_to_` on each parameter, which replaces the storage and device inside the existing impl rather than creating a new tensor. The optimizer holds handles to those same impls, so it sees the parameters on the new device.

</details>
