+++
title = "Python bindings"
description = "A nanobind extension and a pure-Python layer that give kael a PyTorch-style API, zero-copy NumPy interop and GIL-free compute."
weight = 10
slug = "python-bindings"

[extra]
chapter = "10"
kicker = "Chapter 10"
builds = "A Python package on top of the C++ library, with NumPy interop"
source = "docs/10-python-bindings.md"
tag = "chapter-11-python"
+++

This chapter exposes kael to Python as the package `kael`, distributed on PyPI as `kael-tensor`. A compiled extension module `kael._C`, written with nanobind, binds `Tensor`, the operations, autograd entry points, NumPy interop, safetensors I/O, the BPE tokenizer and the C++ `GPT` model. A pure-Python layer on top adds PyTorch-style factories, `no_grad`, `nn` modules, optimizers and a data loader. The chapter maps to milestone v0.4 and to the git tag `chapter-11-python`.

| File | Role |
|------|------|
| [`pyproject.toml`](https://github.com/JetBerri/kael.cpp/blob/main/pyproject.toml) | Package metadata, scikit-build-core backend, pytest and cibuildwheel settings |
| [`python/CMakeLists.txt`](https://github.com/JetBerri/kael.cpp/blob/main/python/CMakeLists.txt) | `nanobind_add_module(_C NB_STATIC ...)`, stub generation, install rules |
| [`python/src/bind.h`](https://github.com/JetBerri/kael.cpp/blob/main/python/src/bind.h) | Shared declarations: conversion helpers and the five `bind_*` functions |
| [`python/src/bindings.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/python/src/bindings.cpp) | `NB_MODULE(_C, m)`, exception translation, argument and operand parsing |
| [`python/src/tensor_bind.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/python/src/tensor_bind.cpp) | `DType`, the `Tensor` class, properties, operators, in-place methods, views |
| [`python/src/ops_bind.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/python/src/ops_bind.cpp) | Unary, binary and reduction functions, factories, runtime controls |
| [`python/src/autograd_bind.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/python/src/autograd_bind.cpp) | `is_grad_enabled`, `set_grad_enabled`, `gradcheck` |
| [`python/src/interop_bind.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/python/src/interop_bind.cpp) | Indexing, `tensor()`, `from_numpy`, `numpy()`, `__array__`, `__dlpack__` |
| [`python/src/models_bind.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/python/src/models_bind.cpp) | safetensors, `BpeTokenizer`, sequence ops, debug tools, `GPT` |
| [`python/kael/__init__.py`](https://github.com/JetBerri/kael.cpp/blob/main/python/kael/__init__.py) | Public namespace, dtype aliases, factories, `no_grad`, `enable_grad` |
| [`python/kael/nn/`](https://github.com/JetBerri/kael.cpp/tree/main/python/kael/nn), [`python/kael/optim/`](https://github.com/JetBerri/kael.cpp/tree/main/python/kael/optim), [`python/kael/data/`](https://github.com/JetBerri/kael.cpp/tree/main/python/kael/data) | Pure-Python `Module`, layers, losses, init, optimizers, schedulers, `DataLoader` |
| [`python/tests/`](https://github.com/JetBerri/kael.cpp/tree/main/python/tests) | pytest suite |
| [`.github/workflows/wheels.yml`](https://github.com/JetBerri/kael.cpp/blob/main/.github/workflows/wheels.yml) | Wheel builds with cibuildwheel |

## Prerequisites

- [02. Memory model](@/book/02-memory-model.md): `Storage::external` and the shared ownership of tensor memory.
- [03. Views](@/book/03-views.md): strides, which NumPy interop maps one to one.
- [06. Autograd](@/book/06-autograd.md): grad mode and `backward`.
- [07. Neural networks and training](@/book/07-neural-networks.md): the C++ `nn` and `optim` APIs that the Python layer mirrors.
- [08. CPU performance](@/book/08-cpu-performance.md): the thread pool, whose work runs while the GIL is released, and its fork handling.
- [09. CUDA backend](@/book/09-cuda-backend.md): device names and transfers.
- [ADR 0002: tensor as handle](@/adr/0002-tensor-as-handle.md): why a Python `Tensor` can alias another.

## Why this layer exists

Deep learning research and most production training happen in Python. A C++ framework without Python bindings can only be used by people willing to write, compile and debug C++ for every experiment. PyTorch, TensorFlow and JAX all follow the same pattern: a compiled core does the numeric work, and Python drives it.

The boundary between the two languages is where most of the hard problems live:

1. **Two memory managers.** Python frees objects by reference counting. kael frees storage when the last `shared_ptr` dies. A NumPy array that views kael memory, or a kael tensor that views NumPy memory, must keep the other side alive without leaking it.
2. **One interpreter lock.** CPython runs Python bytecode on one thread at a time, guarded by the Global Interpreter Lock (GIL). A C++ call that holds the GIL for a long matrix product freezes every other Python thread, including a web server.
3. **Two error models.** C++ throws exceptions, Python raises them. A `kael::IndexError` must arrive as a Python `IndexError` so that ordinary `except` clauses work.
4. **Two object models.** Python expects `x += 1` to keep `x` the same object, `True` to be a `bool` and not an `int`, and `a + "x"` to raise `TypeError`. The bindings must honor each of those protocols.

Without careful bindings, the result works in a demo and fails under real use: a NumPy array that reads freed memory, a server that stops answering during inference, an optimizer that updates temporary objects.

The mental model to hold on to:

```text
user code: import kael
     |                         \
kael (__init__.py)              kael.nn, kael.optim, kael.data
factories, dtype aliases,       pure Python, subclassable
no_grad                                 |
     |                                  |
     +---------------+------------------+
                     |
          kael._C  (nanobind extension, one shared object)
          Tensor, ops, autograd, interop, GPT
                     |
          libkael  (C++20 static library, chapters 01 to 09)
```

The compiled module is a thin translation layer. It owns no numeric logic. Every operation it exposes is a function from chapters 01 to 09. The Python layer above it owns everything that is orchestration.

## Concepts

### Two layers and where the line goes

The compiled module binds what must be fast or must touch memory: the tensor type, every operation, autograd, NumPy and DLPack exchange, file formats, the tokenizer and the C++ transformer. Everything that is plain orchestration is Python: module registration, optimizer update rules, schedulers and batching.

The Python `nn.Module` is a separate implementation from `kael::nn::Module`. Users subclass it and write `forward` in Python, as in PyTorch. Only `GPT` is exposed as a C++ module.

The split keeps the binding code small. A Python `Linear` is a short class on top of `kael.matmul` and `kael.nn.init`. Binding the C++ `nn::Linear` instead would need two extra mechanisms:

- **Trampolines** for the virtual `forward`, so that a Python subclass can override a C++ virtual function. nanobind supports them with `NB_TRAMPOLINE`, but each overridable method adds boilerplate.
- **Lifetime rules** for submodules owned by C++. A Python object that holds a C++ submodule must not outlive the C++ parent that owns it.

The cost of the split is speed in orchestration code. An optimizer step in Python issues one bound call per in-place operation per parameter. For a model with a few hundred parameter tensors that overhead is small next to the matrix products. PyTorch makes the same trade: `torch.optim` is Python, and only the kernels are compiled.

### nanobind in one paragraph

nanobind is a C++17 library that generates CPython extension modules, a smaller and faster successor of pybind11 by the same author. A binding declares a module with `NB_MODULE(name, m)`, adds functions with `m.def(name, function)`, and wraps C++ classes with `nb::class_<T>`. nanobind converts arguments and return values with type casters: Python `int` to `std::int64_t`, `list` to `std::vector`, a bound class to a reference to the C++ object inside it. `nb::handle` is a borrowed, non-owning `PyObject*`. `nb::object` owns a reference. `nb::borrow(h)` turns a handle into a new owning reference.

### The GIL

The Global Interpreter Lock is a mutex in CPython that a thread must hold to touch Python objects or run bytecode. C++ code called from Python starts with the GIL held. Releasing it lets other Python threads run while C++ works, but from that point the C++ code must not touch any Python object until it reacquires the lock.

```text
GIL held for the whole call:
main thread    [py][  kael.matmul (C++, GIL held)  ][py]
worker thread  ......waiting for GIL..................[py]

GIL released after argument conversion:
main thread    [py][conv][ matmul (C++, no GIL) ][py]
worker thread  [py][py][py][py][py][py][py][py]...
```

### Ownership across the boundary

kael storage is owned by `std::shared_ptr<Storage>`. A Python object is owned by its reference count. Zero-copy exchange means one side holds a strong reference to the other side's owner:

```text
from_numpy:   kael Tensor -> Storage::external -> shared_ptr<void> owner -> nb::object -> numpy array
numpy():      numpy array -> capsule -> shared_ptr<Storage> -> bytes
```

The rule in both directions: the memory is freed only when nobody on either side can reach it.

## Implementation

### Build integration

From `python/CMakeLists.txt`:

```cmake
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_C NB_STATIC
    src/bindings.cpp
    src/tensor_bind.cpp
    src/ops_bind.cpp
    src/autograd_bind.cpp
    src/interop_bind.cpp
    src/models_bind.cpp
)
target_link_libraries(_C PRIVATE kael::kael)
target_compile_features(_C PRIVATE cxx_std_20)
install(TARGETS _C LIBRARY DESTINATION kael)
```

`NB_STATIC` links nanobind's support library into `_C` itself, so the wheel contains one shared object and no separate nanobind library. `kael` is a static library, so the root `CMakeLists.txt` sets `POSITION_INDEPENDENT_CODE ON` on it when `KAEL_BUILD_PYTHON` is on. A static library linked into a shared object must be compiled as position-independent code, otherwise the linker fails with a relocation error on Linux.

`nanobind_add_stub` runs after the build and writes `_C.pyi`, a type stub generated from the bound signatures and docstrings. It is installed next to `_C` so editors and type checkers see the C++ API:

```cmake
nanobind_add_stub(_C_stub
    MODULE _C
    OUTPUT _C.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:_C>
    DEPENDS _C
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/_C.pyi DESTINATION kael)
```

The installed layout inside the wheel:

```text
kael/
  __init__.py          from python/kael
  nn/  optim/  data/   from python/kael
  _C.<platform>.so     the extension, built by CMake
  _C.pyi               the generated stub
```

### Module entry point

From `python/src/bindings.cpp`:

```cpp
NB_MODULE(_C, m) {
    namespace kp = kael::python;
    namespace nb = nanobind;

    nb::register_exception_translator([](const std::exception_ptr& error, void*) {
        // ... see "Error translation"
    });

    m.attr("__version__") = std::string(kael::version());

    kp::bind_tensor(m);
    kp::bind_ops(m);
    kp::bind_autograd(m);
    kp::bind_interop(m);
    kp::bind_models(m);
}
```

The order matters. `bind_tensor` creates the `Tensor` class. The later functions fetch it with `nb::borrow<nb::class_<Tensor>>(m.attr("Tensor"))` and add methods to it. Each topic therefore keeps both its free functions and its methods in one file. `sum` as `kael.sum(t)` and as `t.sum()` are both declared in `ops_bind.cpp`, and `t.numpy()` is declared in `interop_bind.cpp`.

### Conversion helpers

`python/src/bind.h` declares the helpers every binding file uses:

| Helper | Converts |
|--------|----------|
| `integers_from_args(args)` | `view(2, 3)` and `view((2, 3))` to one `std::vector<std::int64_t>` |
| `dims_from(dim)` | `None` (all dimensions), an `int`, or an iterable of ints to `Dims` |
| `maybe_operand(value)` | A `Tensor`, `bool`, `int`, `float`, an object with `__index__`, or a scalar with `__float__` and no `__len__`, to `Operand`; anything else to `std::nullopt` |
| `operand_from`, `scalar_from` | Like `maybe_operand`, but raise `TypeError`; `scalar_from` also rejects tensors |
| `device_name(device)` | `Device` to the string `"cpu"` or `"cuda"` |

Device names go through `kael::parse_device` from `include/kael/device.h` ([Memory model](@/book/02-memory-model.md)). `"cpu"` gives `Device::CPU`, `"cuda"` or `"cuda:0"` gives `Device::CUDA`, and anything else raises `ValueError`. The C++ examples call the same function, so both accept the same names.

`maybe_operand` is the core of every mixed tensor and number expression:

```cpp
std::optional<Operand> maybe_operand(nb::handle value) {
    if (nb::isinstance<Tensor>(value)) {
        return Operand(nb::cast<const Tensor&>(value));
    }
    if (PyBool_Check(value.ptr())) {
        return Operand(value.ptr() == Py_True);
    }
    if (PyLong_Check(value.ptr())) {
        return Operand(nb::cast<std::int64_t>(value));
    }
    if (PyFloat_Check(value.ptr())) {
        return Operand(nb::cast<double>(value));
    }
    if (PyIndex_Check(value.ptr())) {
        return Operand(static_cast<std::int64_t>(PyNumber_AsSsize_t(value.ptr(), nullptr)));
    }
    if (PyObject_HasAttrString(value.ptr(), "__float__") != 0 &&
        PyObject_HasAttrString(value.ptr(), "__len__") == 0) {
        return Operand(nb::cast<double>(nb::float_(value)));
    }
    return std::nullopt;
}
```

The order of checks matters. `bool` is tested before `int` because Python's `True` is also an `int`, and kael keeps `bool` as its own dtype. The `__float__` without `__len__` rule accepts NumPy scalars such as `np.float32(2)` but rejects NumPy arrays, which also define `__float__` for size-1 arrays. Trace a few values through it:

| Value | First matching check | Operand |
|-------|---------------------|---------|
| `True` | `PyBool_Check` | `bool` scalar |
| `3` | `PyLong_Check` | `int64` scalar |
| `2.5` | `PyFloat_Check` | `double` scalar |
| `np.float64(2.5)` | `PyFloat_Check`, because `np.float64` subclasses `float` | `double` scalar |
| `np.int64(3)` | `PyIndex_Check`, because `np.int64` is not an `int` subclass but defines `__index__` | `int64` scalar |
| `np.float32(2)` | `__float__` and no `__len__` | `double` scalar |
| `np.array([1.0])` | none: it has `__len__` | `std::nullopt` |
| `"x"`, `[1, 2]` | none | `std::nullopt` |

### Binding Tensor

`kael::Tensor` is a handle to a shared `TensorImpl` ([ADR 0002](@/adr/0002-tensor-as-handle.md)). nanobind stores a `Tensor` by value inside each Python object, so a Python tensor owns one handle. Copying the handle shares the data:

```python
x = kael.ones(2)
y = kael.Tensor(x)  # nb::init<const Tensor&>: a second handle to the same TensorImpl
y.add_(1)
x.tolist()  # [2.0, 2.0]
```

```text
Python object x --> [ Tensor handle ] --\
                                         >--> TensorImpl --> Storage
Python object y --> [ Tensor handle ] --/
```

The handle design makes binding cheap. nanobind does not need a holder type or custom reference counting, because `Tensor` is already a small value type with shared semantics.

#### Properties and methods

`bind_tensor` in `python/src/tensor_bind.cpp` exposes the PyTorch spellings:

| Python | C++ |
|--------|-----|
| `shape`, `size()`, `size(dim)` | `shape()` as a tuple, `size(dim)` |
| `dtype`, `device`, `ndim` | `dtype()`, device name string, `dim()` |
| `grad`, `grad_fn`, `is_leaf`, `requires_grad` | autograd fields; `grad_fn` is the node name or `None`; `requires_grad` is writable |
| `T`, `mT` | `T()`, `mT()` |
| `stride()`, `numel()`, `is_contiguous()`, `contiguous()` | same names; `stride()` returns a tuple |
| `item()`, `tolist()` | `float`, `int` or `bool` according to dtype; nested lists |
| `to(...)`, `cuda()`, `cpu()` | `to(DType)` and `to(Device)` |
| `view`, `reshape`, `expand`, `permute`, `transpose`, `squeeze`, `unsqueeze`, `narrow`, `select`, `flatten`, `unflatten`, `split`, `chunk` | view functions from [Views](@/book/03-views.md) |
| `backward(gradient=None, retain_graph=False)` | `backward` |
| `clone()`, `detach()`, `detach_()`, `zero_grad()`, `retain_grad()` | same names |

`item()` maps the dtype to the matching Python type: `f32` and `f64` give `float`, `i64` gives `int` with all 64 bits, `bool` gives `bool`. `tolist()` recurses with `select(0, i)` until it reaches 0-d tensors.

`to` accepts any mix of a `DType`, a device name and a tensor, positionally or as `device=` and `dtype=` keywords. A tensor argument means "same device and dtype as this one". `to_anything` applies the positional arguments in order, then `device`, then `dtype`:

```python
t.to(kael.float64)              # dtype only
t.to("cuda")                    # device only
t.to("cuda", kael.float64)      # both, in order
t.to(device="cpu", dtype=kael.int64)
t.to(other)                     # other.device, then other.dtype
```

Any other argument type raises `TypeError` with "to() expects a dtype, a device name or a tensor".

#### Operators and NotImplemented

Arithmetic and comparison operators go through `binary_or_not_implemented`:

```cpp
template <typename Compute>
nb::object binary_or_not_implemented(nb::handle other, Compute compute) {
    std::optional<Operand> operand = maybe_operand(other);
    if (!operand) {
        return nb::borrow(Py_NotImplemented);
    }
    return nb::cast(compute(*operand));
}
```

Returning `NotImplemented` instead of raising lets Python try the other operand's reflected method. That is the protocol for mixed-type arithmetic. If both sides decline, Python raises its own `TypeError`. Trace `t + [1, 2]`:

```text
1. Python calls Tensor.__add__(t, [1, 2])
2. maybe_operand([1, 2]) -> nullopt -> return NotImplemented
3. Python calls list.__radd__ -> list has no __radd__
4. Python raises TypeError: unsupported operand type(s) for +
```

Reflected operators (`__radd__`, `__rsub__`, `__rmul__`, `__rtruediv__`, `__rpow__`) call the C++ function with the operands swapped, so `1 - x` computes `sub(1, x)`. `__floordiv__` maps to `floor_divide` and `@` to `matmul`.

`&`, `|` and `^` map to `logical_and`, `logical_or` and `logical_xor`, and `~` to `logical_not`. These are logical, not bitwise, operations: kael has no bitwise integer kernels.

#### In-place methods return self

`add_`, `mul_`, `copy_`, `fill_`, `zero_`, `clamp_`, `requires_grad_`, `detach_` and the other in-place methods are bound as lambdas over `nb::handle self` that return `nb::borrow(self)`:

```cpp
.def("add_",
     [](nb::handle self, nb::handle other) {
         nb::cast<Tensor&>(self).add_(operand_from(other));
         return nb::borrow(self);
     })
```

Returning the C++ `Tensor&` would make nanobind wrap a new Python object around a copy of the handle. The data would be shared, but `x.add_(1) is x` would be false. Chaining such as `velocity.mul_(m).add_(g)` in `kael.optim.SGD` would then operate on temporary wrappers. `__iadd__` and the other augmented assignments use the same pattern through the `inplace<&Tensor::add_>` template, so `x += 1` keeps the identity of `x` (`test_in_place_operators_keep_identity`).

Why identity matters for `+=`: Python compiles `x += 1` to `x = x.__iadd__(1)`. Whatever `__iadd__` returns is bound to `x`. If it returned a new wrapper, any other reference to the old wrapper, such as an entry in an optimizer's parameter list, would no longer be `x`.

#### Python protocol details

- `__hash__` returns the object identity. `__eq__` is elementwise and returns a tensor, so a value-based hash would be meaningless. Identity hashing lets tensors live in sets and serve as dict keys.
- `__bool__` raises unless the tensor has exactly one element, with PyTorch's message "Boolean value of Tensor with more than one value is ambiguous".
- `__int__` returns `item<std::int64_t>()` for `i64` tensors, keeping every bit. For floating tensors it range-checks the value against `2^63` before converting.
- `__float__` returns `item()`.
- `__index__` accepts only `i64` tensors, so an integer tensor can index a Python list.
- `__len__` and `__iter__` raise `TypeError` on 0-d tensors. `__iter__` yields `select(0, i)` views.
- `__repr__` uses the C++ `operator<<`.

### Indexing

`__getitem__` and `__setitem__` call `index_tensor` and `assign_index` in `python/src/interop_bind.cpp`. Both start from `parse_key`, which turns a single key or a tuple into a list of items and checks that at most one is `...`. `apply_key` then walks the items left to right with a running dimension counter:

| Key item | Effect | Dimension counter |
|----------|--------|-------------------|
| `int` or object with `__index__` | `select(dim, i)`, dimension removed | unchanged |
| `slice` with positive step | `slice(dim, start, stop, step)` after `PySlice_AdjustIndices`; an empty slice becomes `narrow(dim, 0, 0)` | `+1` |
| `None` | `unsqueeze(dim)` | `+1` |
| `...` | Skips `dim() - consumed` dimensions | `+ dim() - consumed` |
| 0-d `i64` tensor | `select` with its value | unchanged |
| 1-d `i64` tensor | `index_select(dim, indices)`, a copy | `+1` |
| `bool` tensor as the whole key | Boolean mask over all elements, a copy | returns immediately |

`consumed` counts the items that address an existing dimension: everything except `None` and `...`.

Worked example: `t[0, ..., None, 1:3]` with `t` of shape `[2, 3, 4]`. The items are `0`, `...`, `None`, `1:3`, and `consumed = 2`.

```text
item    action               view shape    dim after
0       select(0, 0)         [3, 4]        0
...     skip 3 - 2 = 1 dims  [3, 4]        1
None    unsqueeze(1)         [3, 1, 4]     2
1:3     slice(2, 1, 3, 1)    [3, 1, 2]     3
```

The result has shape `[3, 1, 2]`, the same as NumPy and PyTorch give for that key. Every step is a view, so the result shares memory with `t`.

The boolean mask path copies the mask to the host, collects the positions of the true elements, and calls `index_select` on the flattened tensor. The result is always 1-d, as in NumPy.

Negative slice steps, bare Python `bool` keys and 2-d index tensors raise `ValueError` or `IndexError`. Basic indexing returns views, so `t[0, 1:] = 5` writes into `t`. Assignment through an integer tensor index is refused with `TypeError`, because `index_select` produces a copy and the write would be lost. Assignment through a bool mask calls `masked_fill_` and accepts only a number. Any other assignment builds the view with `apply_key`, then calls `copy_` for a tensor value or `fill_` for a number.

### Operations and factories

`bind_ops` in `python/src/ops_bind.cpp` binds most operations twice, as a module function and as a `Tensor` method, with templates that remove the repetition:

```cpp
template <UnaryFunction F>
void add_unary(nb::module_& m, nb::class_<Tensor>& tensor, const char* name, const char* doc) {
    m.def(name, F, nb::arg("input"), doc, nb::call_guard<nb::gil_scoped_release>());
    tensor.def(name, F, doc, nb::call_guard<nb::gil_scoped_release>());
}
```

`add_reduction` gives every reduction the signature `(input, dim=None, keepdim=False)`:

```cpp
const auto wrapped = [function](const Tensor& t, const nb::object& dim, bool keepdim) {
    const Dims dims = dims_from(dim);
    const nb::gil_scoped_release release;
    return function(t, dims, keepdim);
};
```

`var` and `std` add `correction=1`. `argmax` and `argmin` take `dim=None` and `keepdim=False`. Binary functions such as `kael.add(a, b)` accept tensors or numbers on either side through `operand_from`, so `kael.add(1, t)` works.

The C++ factories take a shape vector, a `DType` and a device name. `python/kael/__init__.py` wraps them in PyTorch-style signatures:

```python
def zeros(*shape, dtype=_DEFAULT_DTYPE, device="cpu"):
    return _C.zeros(_shape(shape), dtype, device)
```

`_shape` accepts `zeros(2, 3)` and `zeros((2, 3))`: one non-`int` argument is treated as the shape itself. `arange` infers `int64` when all of `start`, `end` and `step` are Python `int`, and `float32` otherwise, so `arange(5)` is `int64` and `arange(0, 1, 0.25)` is `float32`. Random factories (`rand`, `randn`, `randint`, `randperm`, `bernoulli`) generate on the host and move the result with `.to(device)` in the binding. One seed therefore gives the same values on both devices.

The runtime controls are bound as plain functions: `set_num_threads`, `get_num_threads`, `cuda_is_available`, `cuda_synchronize`, `empty_cache`, `allocator_stats(device="cpu")` and `manual_seed`. `allocator_stats` returns a `dict` with the keys `reserved_bytes`, `in_use_bytes`, `cached_bytes`, `hits` and `misses`. `set_num_threads` raises `ValueError` for a count outside `1..1024`. The first call that starts the thread runtime, such as `get_num_threads()`, raises `ValueError` when `KAEL_NUM_THREADS` is not an integer in that range ([CPU performance](@/book/08-cpu-performance.md)).

`bind_models` in `python/src/models_bind.cpp` adds `save_safetensors`, `load_safetensors(path, mmap=False)`, `read_safetensors_metadata`, the `BpeTokenizer` class, the sequence operations `cumsum`, `sort`, `topk`, `tril`, `triu` and `multinomial`, the debug tools `explain`, `autograd_graph` and `set_anomaly_detection`, and the `GPT` class. Chapters 11 to 13 describe them.

### The kael namespace

`python/kael/__init__.py` re-exports the `_C` functions by name, defines the dtype aliases `float32`, `float64`, `int64` and `bool`, and lists the public API in `__all__`. The alias `bool` shadows the builtin inside the module. That is why the file imports `builtins as _builtins` and calls `_builtins.all(...)` in `arange`: `all` is also shadowed, by `kael.all`.

`no_grad` and `enable_grad` share `_GradMode`, which works both as a context manager and as a decorator:

```python
class _GradMode:
    _enabled = None

    def __init__(self):
        self._previous = []

    def __enter__(self):
        self._previous.append(is_grad_enabled())
        set_grad_enabled(self._enabled)
        return self

    def __exit__(self, *_):
        set_grad_enabled(self._previous.pop())
        return False

    def __call__(self, function):
        @_functools.wraps(function)
        def wrapper(*args, **kwargs):
            with type(self)():
                return function(*args, **kwargs)

        return wrapper
```

The stack of previous states makes a single instance re-entrant. Trace one `no_grad` instance used twice, nested:

```text
guard = kael.no_grad()          grad mode: True
with guard:                     push True,  set False
    with guard:                 push False, set False
        ...
                                pop False, set False
                                pop True,  set True
```

A single saved value would be overwritten by the inner `__enter__`, and the outer `__exit__` would restore `False`. The decorator form creates a fresh instance per call, so recursive or concurrent calls of a decorated function do not share a stack. `__exit__` returns `False`, so exceptions propagate after the mode is restored (`test_no_grad_restores_after_exception`).

The C++ grad mode is a `thread_local` flag in `src/autograd/grad_mode.cpp` ([Autograd](@/book/06-autograd.md)). Each Python thread runs on its own OS thread, so `no_grad` in one Python thread does not affect another. PyTorch has the same per-thread semantics.

### The pure-Python layer

#### nn

`kael.nn.Parameter` subclasses the bound `Tensor`:

```python
class Parameter(kael.Tensor):
    def __init__(self, data, requires_grad=True):
        super().__init__(data.detach())
        self.requires_grad = requires_grad
```

nanobind classes accept Python subclasses. The parameter shares the storage of `data` and is a leaf, because `detach()` cuts it from any graph.

`Module` registers attributes in `__setattr__`:

```python
def __setattr__(self, name, value):
    if "_parameters" not in self.__dict__:
        raise AttributeError("call super().__init__() before assigning attributes")
    for table in (self._parameters, self._modules, self._buffers):
        table.pop(name, None)
    if isinstance(value, Parameter):
        self._parameters[name] = value
    elif isinstance(value, Module):
        self._modules[name] = value
    object.__setattr__(self, name, value)
```

A `Parameter` goes into `_parameters`, a `Module` into `_modules`, and `register_buffer` adds non-trainable state to `_buffers`. Reassigning an attribute first removes it from every table, so replacing a parameter with a plain tensor unregisters it. Assigning an attribute before `super().__init__()` raises `AttributeError` with a hint, which catches the most common subclassing mistake.

`state_dict` includes the buffers, and `load_state_dict` validates and restores them together with the parameters (`test_load_state_dict_restores_buffers`). A missing key raises `KeyError` and a shape mismatch raises `ValueError`. `named_parameters` deduplicates by `id()`, so a weight shared by two layers appears once (`test_shared_parameters_appear_once`). Without deduplication an optimizer would update a tied weight twice per step.

`Module.to(device)` keeps parameter identity:

```python
def to(self, device):
    with kael.no_grad():
        for parameter in self.parameters():
            parameter._move_to_(device)
            parameter._set_grad_device(device)
        for module in self.modules():
            for name, buffer in module._buffers.items():
                module.register_buffer(name, buffer.to(device))
    return self
```

`_move_to_` calls `Tensor::move_to_`, and `_set_grad_device` moves an existing gradient. Parameters keep their identity, so the optimizer's references stay valid after a move ([CUDA backend](@/book/09-cuda-backend.md)). Buffers are not held by optimizers, so they are replaced with moved copies.

The layers are `Linear`, `ReLU`, `GELU`, `Sigmoid`, `Tanh`, `Softmax`, `Flatten`, `Dropout`, `LayerNorm`, `Embedding`, `Sequential`, `ModuleList`, `CrossEntropyLoss` and `MSELoss`. A few details connect them to the C++ layers:

- `Linear` initializes its weight with `kaiming_uniform_(weight, a=math.sqrt(5))` and its bias uniformly in `[-1/sqrt(in_features), 1/sqrt(in_features)]`, the same scheme as PyTorch, and computes `matmul(input, weight.mT) + bias`.
- `LayerNorm.forward` calls `kael.standardize`, and `F.cross_entropy` calls `kael.cross_entropy_losses`, so Python models get the fused C++ kernels from [CPU performance](@/book/08-cpu-performance.md).
- `Embedding.forward` calls `index_select` on the flattened indices and reshapes the result.
- `F.binary_cross_entropy_with_logits` computes `max(x, 0) - x * t + log1p(exp(-|x|))`. It writes `max(x, 0)` as `kael.maximum(logits, kael.zeros_like(logits))` instead of `kael.relu`.

The last choice deserves a trace. At `x = 0` with target `t`, the analytic gradient of the loss is `sigmoid(0) - t = 0.5 - t`. The three terms contribute:

```text
term                 gradient at x = 0
max(x, 0)            maximum splits a tie: 0.5        relu would give 0
-x * t               -t
log1p(exp(-|x|))     0  (the abs gradient at 0 is 0)
total with maximum   0.5 - t    = sigmoid(0) - t, as in the C++ loss
total with relu      -t         wrong by 0.5
```

`test_binary_cross_entropy_gradient_at_zero_is_sigmoid_minus_target` checks it. `kael.nn.init` provides `uniform_`, `normal_`, `zeros_`, `ones_`, `xavier_uniform_`, `kaiming_uniform_` and `kaiming_normal_`, all under `no_grad`.

#### optim and data

`kael.optim` defines `SGD` (momentum, weight decay), `Adam`, `AdamW`, `clip_grad_norm_`, and the schedulers `StepLR`, `CosineAnnealingLR` and `WarmupCosineLR`. `clip_grad_norm_` raises `ValueError` on a NaN or infinite total norm, like the C++ version in [Neural networks and training](@/book/07-neural-networks.md).

`Optimizer.step` is decorated with `@kael.no_grad()` and updates parameters with in-place methods (`sub_`, `mul_`, `add_`). In-place methods bump the storage version, as [Autograd](@/book/06-autograd.md) requires, so a graph that saved a parameter before the step fails loudly instead of computing a wrong gradient. The SGD update with momentum:

```python
velocity.mul_(self.momentum).add_(direction)
parameter.sub_(direction * self.lr)
```

`AdamW` applies decoupled weight decay with `parameter.mul_(1 - lr * weight_decay)` before the Adam step. `Adam` adds `weight_decay * parameter` to the gradient instead. Optimizer state is keyed by the parameter's position in the list passed to the constructor.

`kael.data` defines `Dataset`, `TensorDataset`, `DataLoader` and `load_mnist`. For a `TensorDataset`, `DataLoader` builds each batch with one `index_select` per tensor instead of stacking per-sample tensors. For a batch of 64 MNIST images, that is one bound call instead of 64 `__getitem__` calls followed by a `stack` of 64 tensors. Other datasets fall back to `_collate`, which stacks samples column by column. `shuffle=True` draws the order with `kael.randperm`, so `manual_seed` makes it reproducible. `load_mnist` calls the C++ IDX reader through `_C._load_mnist`.

### NumPy interop

kael exchanges memory with NumPy in both directions without copying when the layout allows it. Both directions solve the same problem: two libraries with independent memory management must agree on who frees the buffer.

```text
from_numpy
  numpy array <--- strong ref --- nb::object (heap) <--- shared_ptr<void> owner
                                                              ^
                                       Storage::external -----+
                                              ^
                                       kael Tensor (and its views)

numpy()
  kael Tensor ---> shared_ptr<Storage> ---> bytes
                         ^
                   copy held in a capsule <--- numpy array (nb::ndarray)
```

#### from_numpy: a tensor over NumPy memory

`from_numpy` reads the `data` entry of `__array_interface__`, which gives the data address and a read-only flag, and the `shape`, `strides` and `dtype` attributes of the array:

```cpp
std::shared_ptr<void> owner(new nb::object(array), [](void* held) {
    numpy_owner_release(static_cast<nb::object*>(held));
});
auto impl = std::make_shared<TensorImpl>();
impl->storage = std::make_shared<Storage>(
    Storage::external(first, empty_array ? 0 : last_byte, std::move(owner)));
impl->shape = std::move(shape);
impl->strides = std::move(strides);
impl->dtype = dtype;
Tensor result = Tensor::from_impl(std::move(impl));
return read_only ? result.clone() : result;
```

`Storage::external` ([Memory model](@/book/02-memory-model.md)) wraps memory kael did not allocate. It does not free the bytes. Instead the storage holds `owner`, a `shared_ptr<void>` that keeps a strong Python reference to the array. When the last tensor or view over that storage dies, the owner's deleter drops the reference:

```cpp
void numpy_owner_release(nb::object* array) {
    if (Py_IsInitialized() != 0) {
        const nb::gil_scoped_acquire acquire;
        delete array;
    }
}
```

The deleter can run on any thread, including a thread that released the GIL inside a kael call, so it acquires the GIL before touching the Python object. During interpreter shutdown the reference is leaked instead, because acquiring the GIL is no longer possible. Leaking a few bytes at exit is harmless. Touching a finalized interpreter crashes.

The conversion validates everything kael cannot represent:

| NumPy property | Handling |
|----------------|----------|
| `float32`, `float64`, `int64`, `bool` | Mapped to `F32`, `F64`, `I64`, `Bool` by `dtype_from_numpy_kind` |
| Any other dtype, for example `int32` | `TypeError` naming the kind and size |
| Non-native byte order | `TypeError` |
| Negative stride | `ValueError`: "call .copy() first" |
| Stride not a multiple of the item size | `ValueError` |
| Address not aligned to the item size | `kael::Error`, raised as `RuntimeError` |
| Read-only buffer | Copied with `clone()`, because kael tensors are always writable |
| Zero-size array | Storage of 0 bytes |

NumPy strides are in bytes and kael strides in elements, so each stride is divided by the item size. Worked example with a transposed view:

```text
a = np.arange(6, dtype=np.float32).reshape(2, 3)    shape (2, 3), strides (12, 4) bytes
a.T                                                  shape (3, 2), strides (4, 12) bytes
kael.from_numpy(a.T)                                 shape (3, 2), strides (1, 3) elements
```

The storage size is the byte extent of the view: `item_size + sum((size - 1) * stride)` over the dimensions. Here that is `4 + 2 * 4 + 1 * 12 = 24` bytes, the whole buffer.

A second example with a step slice:

```text
a[:, ::2]               shape (2, 2), strides (12, 8) bytes, first element a[0][0]
kael strides            (3, 2) elements
extent                  4 + 1 * 12 + 1 * 8 = 24 bytes
elements reached        offsets 0, 2, 3, 5  ->  values 0, 2, 3, 5
```

A stride of 6 bytes, as in a structured array with mixed field sizes, fails the divisibility check, because no integer element stride reaches every element.

`kael.tensor(data)` always copies, also for NumPy input: it calls `from_numpy(...).clone()`. Use `from_numpy` when sharing is the intent. PyTorch draws the same line between `torch.tensor` and `torch.from_numpy`.

#### numpy(): a NumPy array over tensor memory

```cpp
template <typename Framework> nb::ndarray<Framework> as_ndarray(const Tensor& t) {
    KAEL_CHECK(t.device() == Device::CPU, "cannot share a tensor on device "
                                              << t.device() << " with numpy; call .cpu() first");
    KAEL_CHECK(!t.requires_grad(),
               "cannot share a tensor that requires grad; call .detach() first");
    const nb::capsule owner(
        new std::shared_ptr<Storage>(t.impl()->storage),
        [](void* held) noexcept { delete static_cast<std::shared_ptr<Storage>*>(held); });
    std::vector<std::size_t> shape(t.shape().begin(), t.shape().end());
    std::vector<std::int64_t> strides(t.strides().begin(), t.strides().end());
    return nb::ndarray<Framework>(t.device_data(), shape.size(), shape.data(), owner,
                                  strides.data(), dlpack_dtype(t.dtype()));
}
```

The capsule owns a copy of the `shared_ptr<Storage>`. The NumPy array keeps the storage alive after the tensor is gone, and the storage block is not returned to the caching allocator until the array dies. `test_numpy_keeps_the_tensor_memory_alive` proves it: it deletes the tensor, calls `empty_cache()`, allocates a zero tensor of the same size, and checks that the array still reads its original values.

Holding the storage, not the `TensorImpl`, also means that `Module.to("cuda")` replacing a parameter's storage leaves an earlier NumPy view valid and unchanged (`test_numpy_view_survives_moving_the_tensor`). The array points at the old host storage, which stays alive through the capsule.

Strides pass through unchanged, because `nb::ndarray` takes strides in elements, like kael. `t.T.numpy()` and `t[:, ::2].numpy()` are strided NumPy views, not copies.

Tensors that require grad are refused. Writes through NumPy would bypass the version counter and corrupt saved tensors without an error. Calling `detach()` first states the intent. CUDA tensors are refused because NumPy can only address host memory.

`__array__(dtype=None, copy=None)` implements the NumPy protocol, so `np.asarray(t)` shares memory and `np.array(t, copy=True)` copies. A `dtype` argument converts with `astype`, which always copies. `from_numpy(t.numpy())` round-trips to the same memory (`test_round_trip_through_numpy_needs_no_copy`).

#### DLPack

DLPack is a small C struct for describing a tensor in memory, shared by PyTorch, JAX, CuPy, NumPy and others. `__dlpack__` delegates to the NumPy view's own `__dlpack__`, and `__dlpack_device__` returns `(1, 0)`, the DLPack code for CPU memory with device id 0. Any consumer of the protocol, for example `torch.from_dlpack(t)`, gets a zero-copy view with the same lifetime rules (`test_dlpack_with_torch_shares_memory`, `test_dlpack_keeps_memory_alive`). kael has no `from_dlpack`: import goes through `from_numpy`.

### Releasing the GIL

A C++ call that keeps the GIL while it multiplies two large matrices blocks every other Python thread, including one that serves HTTP requests in `server/app.py` ([Serving and tooling](@/book/13-serving-and-tooling.md)). Calls that do heavy work and touch no Python objects release it.

Two mechanisms are used:

- `nb::call_guard<nb::gil_scoped_release>()` on a `def` releases the GIL for the whole call, after nanobind has converted the arguments.
- A `const nb::gil_scoped_release release;` local inside the lambda, placed after the code that still needs Python. `add_reduction` converts `dim` with `dims_from` first, then releases. `backward` uses the same local release.

| Released for | Binding |
|--------------|---------|
| Unary functions and methods (`exp`, `relu`, ...) | `add_unary` |
| Reductions (`sum`, `mean`, `max`, `min`, `prod`, `norm`, `all`, `any`) | `add_reduction` |
| `matmul`, `Tensor.matmul`, `@` | `call_guard` |
| `clone` | `call_guard` |
| `backward` | local release |
| `BpeTokenizer.encode`, `GPT.__call__`, `GPT.generate` | `call_guard` |

Binary operators, indexing, `softmax`, `var`, `std`, `cat` and factories keep the GIL. Their argument conversion runs on Python objects, and most of them are short.

Releasing the GIL also lets kael's own thread pool do its job. With the GIL released, a `matmul` call runs on the calling thread plus the pool workers from [CPU performance](@/book/08-cpu-performance.md), while other Python threads keep running bytecode.

`GPT.generate` takes an optional Python `on_token` callback. The C++ generation loop runs without the GIL, so the callback wrapper reacquires it before calling into Python:

```cpp
options.on_token = [&on_token](const Tensor& token) {
    const nb::gil_scoped_acquire gil;
    const nb::object answer = (*on_token)(token);
    if (answer.is_none()) {
        return true;
    }
    const int truth = PyObject_IsTrue(answer.ptr());
    if (truth < 0) {
        nb::raise_python_error();
    }
    return truth == 1;
};
```

Returning `False` from the callback stops generation. Returning `None` continues, so a callback that only prints tokens needs no `return`. An exception in the callback propagates through C++ back to the caller as the original Python exception.

```text
Python thread        C++ generate loop (no GIL)          callback
generate(...)  --->  release GIL
                     token 1 ----------------------->    acquire GIL, call on_token, release
                     token 2 ----------------------->    acquire GIL, call on_token, release
                     ...
               <---  reacquire GIL, return tensor
```

Releasing the GIL is safe only if the released code never touches Python state. In kael that holds for every released call, with one indirect path: dropping the last reference to a tensor created by `from_numpy` runs `numpy_owner_release`, which reacquires the GIL itself.

`test_gil_is_released_during_matmul` and `test_gil_is_released_during_backward` in `python/tests/test_interop.py` start the work in a background thread and count iterations of a busy loop in the main thread. The matmul test multiplies two `1500 x 1500` matrices three times and requires more than 100 iterations. With the GIL held, the loop could not run while the C++ code works.

### Error translation

kael throws a small hierarchy from `include/kael/error.h`: `kael::Error` derives from `std::runtime_error`, and `IndexError`, `ValueError` and `TypeError` derive from `Error`. The module registers a translator that maps them to the matching Python exceptions:

```cpp
nb::register_exception_translator([](const std::exception_ptr& error, void*) {
    try {
        std::rethrow_exception(error);
    } catch (const kael::IndexError& e) {
        PyErr_SetString(PyExc_IndexError, e.what());
    } catch (const kael::ValueError& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
    } catch (const kael::TypeError& e) {
        PyErr_SetString(PyExc_TypeError, e.what());
    } catch (const kael::Error& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
});
```

The most derived classes come first, since a `catch` of `kael::Error` would also match them. Exceptions outside the hierarchy are not caught here and fall through to nanobind's built-in translators, so a `std::bad_alloc` still becomes `MemoryError`.

The `KAEL_CHECK_INDEX`, `KAEL_CHECK_VALUE` and `KAEL_CHECK_TYPE` macros choose the class at the throw site, and `KAEL_CHECK` throws the base `kael::Error`. Python code can write `except IndexError` around `t[5]`, as with a list. Messages keep the C++ prefix `kael error at <file>:<line>: check '<condition>' failed:` followed by the explanation:

| Python expression | Python exception | Message contains |
|-------------------|------------------|------------------|
| `kael.zeros(2)[5]` | `IndexError` | "out of range" |
| `kael.zeros(2) + kael.zeros(3)` | `ValueError` | "broadcastable" |
| `kael.zeros(2).backward()` | `RuntimeError` | the backward precondition |
| `kael.zeros(2).cuda()` without CUDA | `RuntimeError` | "CUDA" |

`test_errors.py` checks each of these. `test_only_kael_errors_are_translated` checks that a `FileNotFoundError` raised by Python code is not touched by the translator.

### Building the wheel

`pyproject.toml` selects scikit-build-core as the build backend, with nanobind as a build requirement:

```toml
[build-system]
requires = ["scikit-build-core>=0.10", "nanobind>=2.4"]
build-backend = "scikit_build_core.build"

[tool.scikit-build]
wheel.packages = ["python/kael"]
cmake.version = ">=3.24"
cmake.build-type = "Release"

[tool.scikit-build.cmake.define]
KAEL_BUILD_PYTHON = "ON"
KAEL_BUILD_TESTS = "OFF"
```

What `pip install .` does, step by step:

```text
1. pip creates an isolated environment and installs scikit-build-core and nanobind
2. scikit-build-core runs CMake on the root CMakeLists.txt with
   KAEL_BUILD_PYTHON=ON, KAEL_BUILD_TESTS=OFF, CMAKE_BUILD_TYPE=Release
   and makes nanobind's CMake package findable
3. CMake builds libkael (position independent) and _C, then generates _C.pyi
4. CMake installs _C and _C.pyi into the kael/ directory of the wheel
5. scikit-build-core adds python/kael and writes the wheel metadata
6. pip installs the wheel and numpy, the only runtime dependency
```

The `[project]` table declares `license = { text = "Apache-2.0" }`, so the wheel metadata names the license, and `requires-python = ">=3.9"`.

Install from a checkout, with the test extras:

```bash
python3 -m venv venv
venv/bin/pip install ".[test]"
```

The `test` extra installs `pytest`, `hypothesis`, `torch`, `transformers`, `safetensors`, `tiktoken`, `fastapi` and `httpx`. Pass CMake options through `--config-settings`, for example for CUDA:

```bash
venv/bin/pip install --config-settings=cmake.define.KAEL_ENABLE_CUDA=ON .
```

As [CUDA backend](@/book/09-cuda-backend.md#validation-on-a-gpu) explains, `scripts/cuda/validate.sh` builds the package this way and runs the Python suite and examples on the GPU.

#### Release wheels

`.github/workflows/wheels.yml` runs on manual dispatch. It builds wheels with `pypa/cibuildwheel` on `ubuntu-latest`, `ubuntu-24.04-arm`, `macos-latest` and `macos-15-intel`, builds an sdist with `pipx run build --sdist`, and optionally publishes to TestPyPI or PyPI with trusted publishing (an `id-token: write` permission instead of a stored API token).

The `[tool.cibuildwheel]` table in `pyproject.toml` controls the matrix:

```toml
[tool.cibuildwheel]
build = "cp39-* cp310-* cp311-* cp312-* cp313-*"
skip = "*-musllinux* pp*"
archs = ["native"]
test-requires = ["pytest", "hypothesis", "fastapi", "httpx"]
test-command = "pytest {project}/python/tests -q"

# Aligned operator new needs macOS 10.13 and std::filesystem needs 10.15.
[tool.cibuildwheel.macos]
environment = { MACOSX_DEPLOYMENT_TARGET = "10.15" }

# NumPy no longer ships manylinux2014 wheels, so tests in that image would build it from source.
[tool.cibuildwheel.linux]
manylinux-x86_64-image = "manylinux_2_28"
manylinux-aarch64-image = "manylinux_2_28"
```

It builds CPython 3.9 to 3.13, skips musllinux and PyPy, and runs the test suite against every wheel with `pytest`, `hypothesis`, `fastapi` and `httpx` installed. Linux builds use `manylinux_2_28` images, because NumPy no longer publishes `manylinux2014` wheels for the test environment. macOS builds set `MACOSX_DEPLOYMENT_TARGET` to 10.15, because aligned `operator new` needs 10.13 and `std::filesystem` needs 10.15.

`archs = ["native"]` builds only the architecture of each runner. `ubuntu-latest` produces the x86_64 Linux wheels, `ubuntu-24.04-arm` the aarch64 ones, `macos-latest` the arm64 macOS wheels and `macos-15-intel` the x86_64 ones. Every wheel is tested on the machine that built it. Listing both architectures per platform would make each Linux runner need QEMU for the foreign one, and each macOS runner would produce cross-compiled duplicates that no test runs.

### Running the tests

```bash
venv/bin/pytest python/tests -q
```

`pyproject.toml` sets `testpaths = ["python/tests"]`, so a bare `pytest` from the repository root finds the suite. `python/tests/conftest.py` seeds both kael and NumPy with 1234 before every test through an autouse fixture.

Tests that need optional packages skip themselves with `pytest.importorskip`:

| Optional package | Tests that need it |
|------------------|--------------------|
| `torch` | `test_autograd.py`, `test_ops.py`, `test_nn.py`, `test_gpt.py`, the DLPack tests in `test_interop.py` |
| `transformers`, `safetensors.torch` | `test_gpt.py` |
| `tiktoken` | tokenizer comparisons in `test_gpt.py` and `test_tokenizer.py` |
| `fastapi`, `httpx` | `test_server.py` |

`test_cuda.py` skips without a device. The cibuildwheel run installs `fastapi` and `httpx`, so the server tests run against every wheel. It leaves PyTorch out: PyTorch publishes no Intel macOS wheels after 2.2, and the default Linux wheel pulls in the CUDA libraries for every Python version. The PyTorch comparisons run in the `python` job of `.github/workflows/ci.yml` instead, which installs `".[test]"`, builds from source and runs `pytest python/tests -q` with every optional package present.

## Edge cases and pitfalls

- **Lifetime across the boundary.** A tensor from `from_numpy` holds a reference to the array, and an array from `numpy()` holds the storage. Deleting either side never leaves a dangling pointer (`test_from_numpy_keeps_the_array_alive`, `test_view_of_from_numpy_keeps_the_array_alive`, `test_numpy_of_view_keeps_memory_alive`).
- **GIL in deleters.** The NumPy owner may be released on a thread without the GIL. `numpy_owner_release` acquires it, and skips the release after interpreter shutdown.
- **Read-only arrays.** Sharing them would let kael write into memory NumPy declared immutable. `from_numpy` copies them (`test_from_numpy_copies_read_only_arrays`).
- **Gradients and shared memory.** `numpy()` refuses tensors that require grad.
- **Bool before int.** `maybe_operand` checks `PyBool_Check` before `PyLong_Check`. Otherwise `True` would become an `int64` scalar and change the result dtype of operations on `bool` tensors.
- **Equality with foreign objects.** `t == "x"` returns `False`, not a tensor and not an error. Both `__eq__` methods return `NotImplemented`, and Python then falls back to identity comparison. `test_foreign_operands_give_type_errors` pins this behavior next to the `TypeError` for `t + "x"`.
- **Nested sequences.** `tensor()` validates ragged lists ("nested sequences have inconsistent sizes"), mixed depth, nesting deeper than 32 levels (`kMaxNesting`), and self-referencing lists, which hit the nesting limit. Integer lists stay `int64` with every bit. Any float makes the result `float32` unless `dtype` is given.
- **Integer conversion.** `int(t)` on a float tensor rejects values outside `[-2^63, 2^63)` instead of wrapping (`test_int_rejects_a_float_at_the_upper_int64_bound`).
- **Identity of in-place results.** In-place methods return the same Python object, which optimizers and `x += 1` rely on.
- **Negative slice steps.** Not supported, because kael strides are non-negative. The error says so.
- **Assignment through an index tensor.** `t[idx] = v` with a 1-d `i64` tensor raises `TypeError`. Use `index_add_` or `scatter_add_` for writes through indices.
- **Device batches.** `DataLoader` builds its index tensor on the CPU with `randperm` or `arange`. Keep a `TensorDataset` on the CPU and move each batch to the device inside the training loop, because `index_select` rejects a CPU index tensor for a device tensor.
- **Fork.** `multiprocessing` with the `fork` start method forks a process whose thread pool already started. The `pthread_atfork` handlers from [CPU performance](@/book/08-cpu-performance.md) make the child usable (`test_fork_keeps_parallel_work_running`).

> [!WARNING]
> `kael.from_numpy(a)` aliases `a`. Writing `a[0] = 99` changes the tensor, and an in-place kael operation changes the array. No version counter tracks writes made through NumPy, so a graph that saved the tensor for backward can silently compute a wrong gradient. Use `kael.tensor(a)` when the data must not be shared.

## How it is tested

| File | Test functions | What it proves |
|------|-------|----------------|
| `test_tensor.py` | 37 | Factories, `tensor()` inference and validation, properties, `item`, `tolist`, conversions, hashing, operators with scalars on either side, in-place identity, views |
| `test_ops.py` | 12 | Operations against PyTorch |
| `test_autograd.py` | 15 | Gradients against PyTorch, `retain_graph`, `no_grad` and `enable_grad` as context managers and decorators, `gradcheck` with a Python callback, in-place and version checks |
| `test_indexing.py` | 18 | Every key form of `__getitem__` and `__setitem__` |
| `test_interop.py` | 23 | `from_numpy`, `numpy()`, `__array__`, DLPack, lifetimes, GIL release, fork |
| `test_errors.py` | 7 | `IndexError`, `ValueError`, `TypeError`, `RuntimeError` translation and messages, an invalid `KAEL_NUM_THREADS` |
| `test_nn.py` | 31 | `Module` registration, buffers in `load_state_dict`, layers, losses, the BCE gradient at zero, optimizers and schedulers against PyTorch, clipping a non-finite norm, training XOR |
| `test_data.py` | 8 | `Dataset`, `TensorDataset`, `DataLoader` |
| `test_cuda.py` | 10 | Device factories, transfers, `Module.to`, training on the device |
| `test_gpt.py`, `test_tokenizer.py`, `test_server.py` | 9, 3, 26 | Covered in chapters 11 to 13 |

Three testing strategies run through the suite:

- **PyTorch as the oracle.** `test_ops.py`, `test_autograd.py` and `test_nn.py` compute the same expression in kael and in PyTorch and compare values and gradients. The bindings are thin, so these tests check the C++ library and the binding together.
- **Behavior, not values.** Lifetime tests delete one side and check that the other still reads correct data. GIL tests measure that another thread made progress. Identity tests use `is`.
- **Error contracts.** `test_errors.py` checks the Python exception class and a fragment of the message, because Python callers write `except` clauses against both.

`test_cuda.py` skips without a device. The `python` CI job builds the package without a CUDA backend, so the file skips there. `scripts/cuda/validate.sh` runs it on a GPU and fails if any of its tests skips. In the [reference run](@/book/09-cuda-backend.md#reference-run) on an NVIDIA T4, its 10 tests and the full suite of 346 tests passed on a CUDA build.

## Build it yourself

1. Add `pyproject.toml` with scikit-build-core and nanobind, `python/CMakeLists.txt` with `nanobind_add_module(_C NB_STATIC ...)`, and an empty `NB_MODULE`. Set `POSITION_INDEPENDENT_CODE` on `kael`. Checkpoint: `pip install .` succeeds and `import kael._C` works.
2. Bind `DType` and `Tensor` with `shape`, `dtype`, `item`, `tolist` and `__repr__`, and the factories. Checkpoint: `kael._C.zeros([2, 3], kael._C.DType.F32, "cpu").shape == (2, 3)`.
3. Register the exception translator. Checkpoint: `test_errors.py` passes.
4. Write `maybe_operand` and the operators with `NotImplemented`. Checkpoint: `test_scalars_mix_on_either_side` and `test_foreign_operands_give_type_errors`.
5. Bind in-place methods returning `self`, views, `to`, `backward`. Checkpoint: `test_in_place_operators_keep_identity` and `test_views_share_memory`.
6. Bind unary functions, reductions and `matmul` with GIL release. Checkpoint: `test_gil_is_released_during_matmul`.
7. Write `apply_key` and `assign_index`. Checkpoint: `test_indexing.py`.
8. Write `from_numpy` with `Storage::external` and the GIL-acquiring owner, then `as_ndarray`, `__array__` and `__dlpack__`. Checkpoint: `test_interop.py`.
9. Write `python/kael/__init__.py` with factories and `no_grad`. Checkpoint: `test_tensor.py`.
10. Write `kael.nn`, `kael.optim` and `kael.data`. Checkpoint: `test_nn.py` and `test_data.py`.
11. Bind safetensors, `BpeTokenizer` and `GPT`, with the `on_token` callback reacquiring the GIL. Checkpoint: the full `pytest python/tests -q` run passes.
12. Add the cibuildwheel settings and `wheels.yml`. Checkpoint: a manual workflow run produces wheels for every platform.

## Check your understanding

<details>
<summary>Why do in-place methods return nb::borrow(self) instead of the C++ Tensor&?</summary>

nanobind would wrap the returned `Tensor&` in a new Python object holding a copy of the handle. The data would be shared, but the object would differ, so `x.add_(1) is x` would be false. Python binds the result of `__iadd__` to the name, so `x += 1` would rebind `x` to a new wrapper, and chained updates in optimizers would operate on temporaries. Returning `self` keeps one Python object per tensor.

</details>

<details>
<summary>Why is bool checked before int in maybe_operand?</summary>

In Python, `bool` is a subclass of `int`, so `PyLong_Check(True)` is true. If the `int` check ran first, `True` would become an `int64` scalar, and type promotion would treat an expression such as `bool_tensor & True` as a mix of `bool` and integer operands. The result dtype of operations on `bool` tensors would change. kael keeps `bool` as its own dtype, so `True` must stay a `bool` scalar.

</details>

<details>
<summary>What keeps the memory alive in a = t.numpy(); del t, and what keeps it alive in t = kael.from_numpy(a); del a?</summary>

In the first case the NumPy array holds a capsule that owns a copy of the tensor's `shared_ptr<Storage>`, so the storage block stays allocated and is not returned to the caching allocator. In the second case the tensor's `Storage::external` holds a `shared_ptr<void>` whose object is an `nb::object` with a strong reference to the array, so NumPy does not free the buffer. In both cases the memory is freed only when the last owner on either side dies.

</details>

<details>
<summary>Why does numpy_owner_release acquire the GIL, and why does it skip the release during shutdown?</summary>

The last kael reference to the storage can die on any thread, including inside a call that released the GIL or on a thread pool worker. Decrementing a Python reference count without the GIL is a data race. During interpreter shutdown the GIL can no longer be acquired safely, so the reference is leaked instead. A few leaked bytes at exit are harmless, a crash in a finalized interpreter is not.

</details>

<details>
<summary>Which dimensions does t[..., 0] select for t of shape [2, 3, 4], and what is the result shape?</summary>

The items are `...` and `0`, so `consumed = 1`. The ellipsis advances the dimension counter by `3 - 1 = 2`, and `0` then calls `select(2, 0)`. The result is a view of shape `[2, 3]` with the first element of every last-dimension row.

</details>

<details>
<summary>Why does F.binary_cross_entropy_with_logits use maximum(x, 0) instead of relu(x)?</summary>

Both give the same forward value. Their gradients differ at `x = 0`. `relu` gives 0 there, while `maximum` splits the gradient of a tie and gives 0.5. With the other terms, the total gradient at 0 becomes `0.5 - t`, which equals the analytic `sigmoid(0) - t` and the C++ loss. With `relu` it would be `-t`.

</details>

## Further reading

- [nanobind documentation](https://nanobind.readthedocs.io/): the sections on exceptions, `ndarray`, GIL management and stub generation cover every mechanism in this chapter.
- [NumPy `__array_interface__` protocol](https://numpy.org/doc/stable/reference/arrays.interface.html): the fields `from_numpy` reads, including the read-only flag.
- [NumPy interoperability guide](https://numpy.org/doc/stable/user/basics.interoperability.html): `__array__`, the `copy` keyword and DLPack from NumPy's side.
- [DLPack specification](https://dmlc.github.io/dlpack/latest/): the struct and device codes behind `__dlpack__` and `__dlpack_device__`.
- [Python array API standard](https://data-apis.org/array-api/latest/): the section on `__dlpack__` defines the exchange protocol that `torch.from_dlpack` consumes.
- [scikit-build-core documentation](https://scikit-build-core.readthedocs.io/): configuration keys used in `pyproject.toml` and how CMake options pass through `--config-settings`.
- [Python data model: emulating numeric types](https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types): the `NotImplemented` and reflected operator protocol.
- [`python/src/interop_bind.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/python/src/interop_bind.cpp): indexing, `tensor()` and NumPy exchange in under 400 lines.

Next: [11. Transformer](@/book/11-transformer.md)
