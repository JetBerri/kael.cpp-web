+++
title = "A PyTorch-shaped API with typed errors"
description = "Why kael copies PyTorch's names and semantics, how numbers mix with tensors in C++, and why failures are typed exceptions that map to Python exceptions."
weight = 4
slug = "pytorch-shaped-api"

[extra]
kicker = "ADR 0004"
source = "docs/adr/0004-pytorch-shaped-api.md"
+++

A tensor library has two surfaces that every user touches: the names of its operations and the way it reports failure. kael copies PyTorch's names and semantics, and it reports failure with a small hierarchy of C++ exceptions that the Python module turns into Python exceptions of the same name. This record explains why kael did not invent its own vocabulary, why it did not use error codes, and what the choice costs.

Two terms are used throughout. An **in-place operation** writes its result into the storage of an existing tensor instead of allocating a new one. A **typed error** is an exception whose class says what kind of problem occurred, such as an index out of range or an unsupported dtype.

## Status

Accepted in phase 3 (tag `chapter-03-core`, which matches [Memory model](@/book/02-memory-model.md) in the book) and extended in every later phase. `Scalar` arrives at `chapter-05-iteration`, `Operand` and the arithmetic operators at `chapter-06-ops`, the typed error subclasses at `chapter-10-cuda`, and the Python exception translator at `chapter-11-python`. Still in force at `v1.0.0` and at the current head of the repository.

## Context

kael is didactic. A reader learns how a tensor library works by putting kael next to PyTorch and recognizing every call. Three constraints follow.

- **Recognition.** A reader who knows `torch.sum`, `view`, `masked_fill` or `optim.Adam` should find the same concept under the same name. Every renamed function is a translation step between the book and the reference implementation.
- **Two languages.** The API must read well in C++, which has no keyword arguments and no dynamic typing, and in Python, which has both.
- **Failure in both languages.** A shape mismatch deep inside an expression must surface in C++ as something the caller can catch, and in Python as an ordinary exception such as `IndexError` or `ValueError`.

Error handling in C++ has three common forms.

```text
exceptions                    error codes                    std::expected<Tensor, Error>

auto y = (a * b).sum();       Tensor p; Tensor y;            auto p = mul(a, b);
                              if (mul(a, b, &p) != OK)       if (!p) return p.error();
                                  return ...;                auto y = sum(*p);
                              if (sum(p, &y) != OK)          if (!y) return y.error();
                                  return ...;
```

Only the first form keeps arithmetic readable. The other two turn every operator into a statement and give Python nothing idiomatic to raise.

## Decision

### Names and semantics follow PyTorch

Operations use PyTorch's names and argument order: `sum`, `matmul`, `view`, `reshape`, `transpose`, `masked_fill`, `scatter_add_`, `nn::Linear`, `optim::Adam`, `backward`. A trailing underscore marks an in-place operation, as in PyTorch: `add_`, `mul_`, `fill_`, `copy_`. In C++, gradient recording is switched off with `autograd::NoGradGuard`. In Python it is `kael.no_grad()`, as in PyTorch.

### Free functions, methods and operators

Computational operations are free functions in `kael::`, declared in [`include/kael/ops.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/ops.h): `kael::exp(t)`, `kael::sum(t, {1})`, `kael::matmul(a, b)`. `Tensor` methods in [`include/kael/tensor.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/tensor.h) cover the rest: the views (`view`, `reshape`, `transpose`, `permute`, `slice`, ...), which exist only as methods, plus `to`, `clone`, `backward` and every in-place operation. `sum`, `mean` and `matmul` exist in both forms. Arithmetic and comparisons are operator overloads: `+`, `-`, `*`, `/`, `==`, `!=`, `<`, `<=`, `>`, `>=`, and the compound forms `+=`, `-=`, `*=`, `/=`, which call `add_`, `sub_`, `mul_` and `div_`.

### Numbers mix with tensors

Binary operations take two `Operand` values ([`include/kael/operand.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/operand.h)). An `Operand` holds either a `Tensor` or a `Scalar` ([`include/kael/scalar.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/scalar.h)), and it converts implicitly from both:

```cpp
class Operand {
public:
    Operand(const Tensor& tensor) : value_(tensor) {}

    template <std::convertible_to<Scalar> T> Operand(T scalar) : value_(Scalar(scalar)) {}
    // ...
private:
    std::variant<Tensor, Scalar> value_;
};
```

Because the operators are free functions of two `Operand` values, `t * 2`, `2 * t` and `t * other` are all valid, with one declaration per operator. The dtype that a number takes is decided by the promotion rules in [ADR 0003](@/adr/0003-type-promotion.md).

### `[[nodiscard]]` on queries

`Tensor` member functions that return a value, such as `shape()`, `to()`, `clone()`, `view()` and `sum()`, are `[[nodiscard]]`, so `t.reshape({6});` without an assignment is a compiler warning. The losses in [`include/kael/nn/loss.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/nn/loss.h) are `[[nodiscard]]` too. In-place methods return `Tensor&` for chaining and are not marked.

### Errors are typed exceptions

[`include/kael/error.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/error.h) defines one base class and three subclasses:

```cpp
class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class IndexError : public Error { /* ... */ };
class ValueError : public Error { /* ... */ };
class TypeError : public Error { /* ... */ };
```

| Class | Thrown for | Macro |
|-------|------------|-------|
| `IndexError` | an index, dimension or position out of range | `KAEL_CHECK_INDEX` |
| `ValueError` | bad shapes, sizes or argument values | `KAEL_CHECK_VALUE` |
| `TypeError` | an unsupported or mismatched dtype | `KAEL_CHECK_TYPE` |
| `Error` | everything else: internal invariants, autograd preconditions, devices | `KAEL_CHECK` |

Each macro checks a condition and, if it fails, throws the class with a message that names the operation and the offending values. The message is built with stream syntax, so values of any printable type can be inserted:

```cpp
KAEL_CHECK_VALUE(left.size(-1) == right.size(-2),
                 "matmul shapes " << to_string(a.shape()) << " and " << to_string(b.shape())
                                  << " do not align");
```

`detail::fail` prefixes the message with the source location and the failed condition:

```text
kael error at src/ops/linalg.cpp:<line>: check 'left.size(-1) == right.size(-2)' failed: matmul shapes [2, 3] and [2, 3] do not align
```

### Python sees Python exceptions

The Python module registers a translator in [`python/src/bindings.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/python/src/bindings.cpp):

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

The subclasses are caught before the base, because a `catch` of `kael::Error` would also match them. Exceptions outside the hierarchy fall through to nanobind's own translators.

### A worked example

The training step of the MNIST example, in kael C++ ([`examples/mnist_mlp.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/examples/mnist_mlp.cpp)) and kael Python ([`examples/mnist.py`](https://github.com/JetBerri/kael.cpp/blob/main/examples/mnist.py)):

```cpp
const Tensor loss = kael::nn::cross_entropy(model(images.to(device)), labels.to(device));
optimizer.zero_grad();
loss.backward();
optimizer.step();
```

```python
loss = F.cross_entropy(model(images.to(device)), labels.to(device))
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

The Python version is the PyTorch training step with `torch` replaced by `kael`. The C++ version differs only in syntax: `const Tensor` for the declaration and `kael::nn::` for the namespace.

Now trace a failure. `kael::matmul(a, b)` with `a` and `b` both of shape `[2, 3]`:

```text
1. matmul_forward            KAEL_CHECK_VALUE(left.size(-1) == right.size(-2)) fails: 3 != 2
2. detail::fail<ValueError>  builds "kael error at ...: matmul shapes [2, 3] and [2, 3] do not align"
3. C++ caller                catch (const kael::ValueError&) or catch (const kael::Error&)
4. Python caller             the translator raises ValueError with the same text
```

The test `TypedErrors.ShapeAndArgumentProblems` checks the C++ class for this call. `test_value_errors` in `python/tests/test_errors.py` checks that shape problems reach Python as `ValueError`.

## Alternatives considered

### A naming scheme of kael's own

Names chosen for C++ style or for internal consistency, such as `reduce_sum` or `Tensor::add_inplace`.

- Gains: freedom to fix names that PyTorch got wrong for historical reasons.
- Costs: every example in the book needs a mental translation to compare with PyTorch. The golden tests ([Operations](@/book/05-operations.md)) could no longer call the same name on both sides.
- Rejected because recognition is the point of a teaching library.

### Error codes

Each operation returns a status and writes its result through an output parameter.

- Gains: failure paths are explicit at every call site. No exception machinery.
- Costs: operators cannot return a status, so `a + b` would be impossible. Every expression becomes a sequence of statements. Python would still need a translation into exceptions.
- Rejected because it breaks the operator-based API.

### `std::expected<Tensor, Error>`

Each operation returns either a tensor or an error value, as in Rust's `Result`.

- Gains: failure is visible in the type, and there is no stack unwinding.
- Costs: `(a * b).sum()` needs an unwrap after each step, or a monadic chain with `and_then`. Operator overloads on `std::expected` are not standard and would have to be written for every operator. Python still needs exceptions.
- Rejected for the same readability reason. Errors in kael are mostly programming errors, such as wrong shapes, not conditions the caller is expected to recover from on the success path.

### One exception class with an error code

A single `kael::Error` that carries an enum field for the kind of failure.

- Gains: one class to catch.
- Costs: C++ callers would `catch` the base and then `switch` on the code. The Python translator would need the same `switch`. A `catch` clause cannot select on a field.
- Rejected. Subclasses let both languages select with their native `catch` and `except`, and `kael::Error` remains the single base for code that catches everything.

## Consequences

### PyTorch code ports almost line by line

A model written in PyTorch maps to kael with the same structure, the same names and the same order of arguments. The golden tests call the same operation on both sides and compare outputs ([Operations](@/book/05-operations.md)), and `AutogradGolden.TwoLayerPerceptronMatchesPyTorch` compares the gradients of a small network with PyTorch's.

### Error messages are part of the behavior

Tests match on messages, and the book quotes them. `Tensor.DataPtrWrongTypeMessage` pins `expected dtype float64 but tensor has float32`, and `AutogradEngine.SecondBackwardWithoutRetainThrows` checks for `second time`. A change to a message is a change to the behavior and needs the matching test and chapter updated.

### Exceptions keep expressions readable

Exceptions cost nothing on the success path with the table-based unwinding used by mainstream compilers, and they keep expressions such as `(a * b).sum()` readable. The price is that failure paths are implicit in C++ code. A reader cannot see from `a + b` that it can throw `ValueError` for shapes that do not broadcast.

### `[[nodiscard]]` is not universal

The free functions in `include/kael/ops.h` and `include/kael/factory.h`, such as `kael::add` and `kael::zeros`, are not marked `[[nodiscard]]`, apart from `kael::equal`. A discarded `kael::exp(t);` compiles without a warning. The method form `t.sum();` warns.

### Deliberate differences are documented where they appear

Where kael differs from PyTorch, the chapter that introduces the operation says so:

| Difference | PyTorch | kael | Documented in |
|------------|---------|------|---------------|
| Bool + Bool | Bool, a logical or | I64, a count | [TensorIterator](@/book/04-tensor-iterator.md), [ADR 0003](@/adr/0003-type-promotion.md) |
| `reshape` of a non-contiguous tensor | a view when the strides allow it | always a copy, `contiguous().view(shape)` | [Views](@/book/03-views.md) |
| `view` of a non-contiguous tensor | accepted when strides allow it | rejected, with `use reshape instead` | [Views](@/book/03-views.md) |
| in-place op on a tensor that requires grad | allowed in many cases | rejected with grad mode on | [Autograd](@/book/06-autograd.md), [ADR 0005](@/adr/0005-dynamic-autograd-graph.md) |

### The C++ API has fewer methods than PyTorch

PyTorch exposes most operations as both `torch.exp(t)` and `t.exp()`. In kael C++, most computational operations exist only as free functions, so `t.exp()` does not compile and the call is `kael::exp(t)`. A port from PyTorch replaces such method calls with function calls.

## Where it shows up in the code

| File | What to look for |
|------|------------------|
| [`include/kael/ops.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/ops.h) | the free functions and the operator overloads |
| [`include/kael/tensor.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/tensor.h) | `[[nodiscard]]` methods, in-place methods with a trailing underscore |
| [`include/kael/scalar.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/scalar.h), [`include/kael/operand.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/operand.h) | `Scalar` and `Operand` |
| [`include/kael/error.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/error.h) | `Error`, `IndexError`, `ValueError`, `TypeError`, `detail::fail`, the `KAEL_CHECK_*` macros |
| [`python/src/bindings.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/python/src/bindings.cpp) | the exception translator, `operand_from` |
| [`include/kael/nn/loss.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/nn/loss.h) | `nn::cross_entropy` and the other losses |
| [`tests/test_errors.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_errors.cpp) | `TypedErrors.IndexProblems`, `TypedErrors.ShapeAndArgumentProblems`, `TypedErrors.DTypeProblems`, `TypedErrors.EverythingElseStaysPlain` |
| [`python/tests/test_errors.py`](https://github.com/JetBerri/kael.cpp/blob/main/python/tests/test_errors.py) | the Python exception classes |

Chapters that depend on this decision:

- [Memory model](@/book/02-memory-model.md): the error hierarchy and the `KAEL_CHECK_*` macros.
- [TensorIterator](@/book/04-tensor-iterator.md): `Scalar` and `Operand`.
- [Operations](@/book/05-operations.md): PyTorch names and the golden tests that compare them.
- [Neural networks and training](@/book/07-neural-networks.md): `nn::Linear`, `optim::Adam` and the training step.
- [Python bindings](@/book/10-python-bindings.md): the Python names and the exception translator.

## Check your understanding

<details>
<summary>Why are t * 2, 2 * t and t * other all valid with a single declaration of operator*?</summary>

`operator*` takes two `Operand` values, and `Operand` converts implicitly from a `Tensor` and from any type convertible to `Scalar`. Either side can be a number or a tensor, so one free function covers all three forms.

</details>

<details>
<summary>A C++ caller wants to handle a bad index differently from any other kael failure. What does it catch, and in which order?</summary>

`catch (const kael::IndexError&)` first, then `catch (const kael::Error&)`. The subclass must come first, because the base class would also match an `IndexError`.

</details>

<details>
<summary>Which Python exception does kael.zeros(2).backward() raise, and why not ValueError?</summary>

`RuntimeError`. The check `backward needs a tensor that requires grad` uses `KAEL_CHECK`, which throws the base `kael::Error`. The translator maps any `kael::Error` that is not one of the three subclasses to `RuntimeError`.

</details>

<details>
<summary>Why did kael not use std::expected&lt;Tensor, Error&gt; for its operations?</summary>

Every step of an expression such as `(a * b).sum()` would need an unwrap, and operators on `std::expected` are not standard. Python would still need exceptions. Most kael errors are programming errors such as mismatched shapes, which a caller does not handle on the success path.

</details>

<details>
<summary>Does kael::exp(t); without an assignment produce a compiler warning? Does t.sum(); ?</summary>

`kael::exp(t);` does not, because the free functions in `ops.h` are not marked `[[nodiscard]]`. `t.sum();` does, because `Tensor::sum` is `[[nodiscard]]`.

</details>

<details>
<summary>A pull request changes the wording of a dtype error message. What else must change?</summary>

The tests that match on that message and the chapters that quote it. Error messages are part of the behavior in kael: tests such as `Tensor.DataPtrWrongTypeMessage` pin them, and Python users see the same text in their exceptions.

</details>
