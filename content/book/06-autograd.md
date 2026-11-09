+++
title = "Autograd"
description = "Reverse-mode automatic differentiation over a dynamic graph: how kael records every operation and walks the graph backward to compute gradients."
weight = 6
slug = "autograd"

[extra]
chapter = "06"
kicker = "Chapter 06"
builds = "Reverse-mode automatic differentiation over a dynamic graph"
source = "docs/06-autograd.md"
tag = "chapter-07-autograd"
math = true
+++

This chapter builds reverse-mode automatic differentiation. After it, any tensor expression built from kael operations can call `backward()`, and every leaf that requires a gradient receives the derivative of the result with respect to itself. The chapter starts from a scalar engine of about 130 lines in `experiments/micrograd.cpp`. It then walks through the tensor engine that the rest of the library uses: the graph data structures, how operations record history, the engine that walks the graph, the derivative formulas, and the finite-difference checker that tests them. It maps to the autograd phase of the project: commit `feat(experiments): add scalar micrograd`, followed by `feat(autograd): add reverse-mode engine, formulas and gradcheck`, and the later fix `fix(autograd): skip gradients of tensors detached in place`.

| File | Role |
|------|------|
| [`experiments/micrograd.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/experiments/micrograd.cpp) | Scalar warm-up: a `Value` type with `+`, `*`, `tanh`, `exp` and `backward()` |
| [`include/kael/autograd/meta.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/autograd/meta.h) | `AutogradMeta`: the per-tensor autograd state stored on `TensorImpl` |
| `include/kael/autograd/node.h`, [`src/autograd/node.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/autograd/node.cpp) | `Node`, `NodeInput`, `Edge`, `AccumulateGrad`, `FunctionNode`, `gradient_edge`, `make_input` |
| `include/kael/autograd/engine.h`, [`src/autograd/engine.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/autograd/engine.cpp) | `backward()`: seeding, dependency counting, gradient buffers, graph release, trace and anomaly hooks |
| `include/kael/autograd/grad_mode.h`, `src/autograd/grad_mode.cpp` | Thread-local grad mode, `GradModeGuard`, `NoGradGuard`, `EnableGradGuard`, `without_grad` |
| `src/autograd/recording.h`, [`src/autograd/recording.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/autograd/recording.cpp) | `SavedTensor`, `should_record`, `set_history`, `sum_to_shape` |
| [`src/autograd/record.h`](https://github.com/JetBerri/kael.cpp/blob/main/src/autograd/record.h) | One `record_*` declaration per differentiable operation |
| `src/autograd/functions_math.cpp` | Binary and unary formulas (add, sub, mul, div, pow, maximum, minimum, neg, abs, exp, log, log1p, expm1, sqrt, rsqrt, sin, cos, tanh, sigmoid, relu, gelu, clamp) |
| `src/autograd/functions_reduce.cpp` | Reductions, softmax, log-softmax, cross entropy, standardize, cumsum |
| `src/autograd/functions_view.cpp` | View and conversion formulas (view, transpose, permute, squeeze, unsqueeze, expand, slice, select, clone, to) |
| `src/autograd/functions_indexing.cpp` | cat, where, masked_fill, index_select, gather |
| `src/autograd/functions_linalg.cpp` | matmul, including the 1-D and batched cases |
| `src/autograd/tensor_autograd.cpp` | The `Tensor` member functions: `requires_grad_`, `grad`, `zero_grad`, `backward`, `retain_grad`, `detach_`, `version` |
| `include/kael/autograd/gradcheck.h`, [`src/autograd/gradcheck.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/autograd/gradcheck.cpp) | Finite-difference gradient checker |
| `include/kael/autograd/functions.h` | Public `sum_to_shape` |

The whole directory is at [src/autograd](https://github.com/JetBerri/kael.cpp/tree/main/src/autograd), and the public headers are at [include/kael/autograd](https://github.com/JetBerri/kael.cpp/tree/main/include/kael/autograd).

## Prerequisites

- [02. Memory model](@/book/02-memory-model.md): `TensorImpl`, `Storage`, the `Tensor` handle.
- [03. Views](@/book/03-views.md): views share storage, and the storage version counter.
- [05. Operations](@/book/05-operations.md): every operation whose derivative this chapter defines.
- [ADR 0005: define-by-run autograd graph](@/adr/0005-dynamic-autograd-graph.md) explains why the graph is built while operations run.
- Calculus at the level of the chain rule and partial derivatives. The chapter restates what it needs.

## Why this layer exists

Training a model means changing its parameters so that a loss gets smaller. Gradient descent needs, for every parameter $\theta_k$, the partial derivative $\partial L / \partial \theta_k$ of the scalar loss $L$. A small transformer has millions of parameters. The loss is the end of a long composition of matrix multiplications, nonlinearities, normalizations and reductions.

There are four ways to get those derivatives, and three of them fail at this scale.

| Approach | How it works | Why it fails for training |
|----------|--------------|---------------------------|
| By hand | Derive $\partial L / \partial \theta$ on paper and code it | Every architecture change means a new derivation. One sign error silently ruins training. |
| Symbolic | A computer algebra system differentiates the formula | Expressions grow exponentially with depth, and it cannot follow loops and branches in C++ code. |
| Numerical | Perturb each parameter and evaluate the loss twice | Two forward passes per parameter. With $10^6$ parameters that is $2 \cdot 10^6$ forward passes per step, and rounding limits the accuracy. |
| Automatic, reverse mode | Record the operations during the forward pass, then apply the chain rule backward through the record | One backward pass, at a cost close to one forward pass, gives the exact derivative with respect to every parameter. |

Without this layer, [chapter 07](@/book/07-neural-networks.md) cannot train anything: `Linear`, `LayerNorm`, the optimizers and the training loop all assume that `loss.backward()` fills `param.grad()`. Numerical differentiation survives in this chapter only as a test tool, in `gradcheck`.

### The mental model

Hold on to this picture for the rest of the chapter.

```text
forward pass (values flow up, a graph is recorded)

    x ──┐
        ├─ mul ── u ──┐
    w ──┘             ├─ add ── v ── tanh ── t ── sum ── L
    x ────────────────┘

backward pass (gradients flow down the same edges, in reverse)

    dL/dx <──┐
             ├── MulBackward <── dL/du ──┐
    dL/dw <──┘                           ├── AddBackward <── dL/dv <── TanhBackward <── SumBackward <── 1
    dL/dx <──────────────────────────────┘
```

1. Every operation that runs on a tensor that requires a gradient leaves behind a small object, a node, that knows one thing: given the gradient of the loss with respect to the operation's output, compute the gradient with respect to each of its inputs.
2. Nodes point at the nodes that produced their inputs. Together they form a directed acyclic graph rooted at the loss.
3. `backward()` walks that graph from the loss to the leaves, passing a gradient tensor along each edge. When several edges reach the same node, their gradients are added before the node runs.

The graph is built while the program runs, one operation at a time. It is dynamic: a C++ `if` or `for` in the forward pass changes the graph for that call only. This is the define-by-run design of PyTorch and Chainer, as opposed to the define-then-run design of early TensorFlow, where the graph was declared first and executed later.

## Concepts

### The chain rule

If $y = f(u)$ and $u = g(x)$, the derivative of the composition is the product of the local derivatives:

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}
$$

When $x$ influences $y$ through several intermediate values $u_1, \dots, u_m$, the contributions of every path add up. This is the multivariable chain rule:

$$
\frac{\partial y}{\partial x} = \sum_{k=1}^{m} \frac{\partial y}{\partial u_k} \cdot \frac{\partial u_k}{\partial x}
$$

The sum is the reason a value used twice receives two gradient contributions. Everything the engine does follows from these two formulas: multiply along a path, add across paths.

A small numeric check. Let $u = 3x$ and $y = u^2$, at $x = 2$. Then $u = 6$, $dy/du = 2u = 12$ and $du/dx = 3$, so $dy/dx = 36$. Expanding directly, $y = 9x^2$ and $dy/dx = 18x = 36$.

### Terms

These terms appear throughout the chapter. The [glossary](@/concepts/glossary.md) collects them with the rest of the book.

- **Leaf**: a tensor created by the user, not by a differentiable operation. Parameters and inputs are leaves. A leaf that requires a gradient receives one in `grad()`.
- **Non-leaf** or **intermediate**: a tensor produced by an operation that recorded history. It has a `grad_fn`.
- **`requires_grad`**: a flag that says the tensor participates in the graph. It is set by the user on leaves and propagated automatically to results.
- **Node**: one backward function in the graph. In kael, `FunctionNode` for operations and `AccumulateGrad` for leaves.
- **Edge**: a pointer from a node to the node that receives the gradient of one of its inputs.
- **Upstream gradient**: the gradient that arrives at a node from the loss side, written $g$. It has the shape of the node's output.
- **Local derivative**: the derivative of one operation's output with respect to one of its inputs, evaluated at the forward values.
- **Seed**: the gradient the engine starts with at the root. For a scalar loss it is $1$, because $\partial L / \partial L = 1$.

### Forward mode and reverse mode

A program maps $n$ inputs to $m$ outputs, $f: \mathbb{R}^n \to \mathbb{R}^m$. Its full derivative is the Jacobian $J$, an $m \times n$ matrix with $J_{ij} = \partial f_i / \partial x_j$. Automatic differentiation can traverse the chain rule in two directions.

- **Forward mode** carries a derivative alongside each value, from inputs to outputs. One pass computes $J v$ for one direction $v$, which is one column of $J$ when $v$ is a unit vector. Getting the full gradient of a scalar loss with respect to $n$ parameters needs $n$ passes.
- **Reverse mode** first runs the program forward and records it, then carries derivatives from the outputs back to the inputs. One pass computes $v^\top J$ for one output weighting $v$, which is one row of $J$. For a scalar loss, $m = 1$, so a single pass gives the gradient with respect to all $n$ inputs.

Training has $m = 1$ and $n$ in the millions. Reverse mode wins by a factor of $n$. It pays in memory: the forward values needed by the backward formulas must be kept until the backward pass uses them. That cost explains several design choices below, such as releasing saved tensors as soon as a node has run.

### Vector-Jacobian products, not Jacobians

The engine never builds a Jacobian. Each backward function computes a vector-Jacobian product (VJP): it maps the upstream gradient $g = \partial L / \partial y$, with the shape of the output $y$, to $\partial L / \partial x$, with the shape of the input $x$:

$$
\frac{\partial L}{\partial x_j} = \sum_i g_i \frac{\partial y_i}{\partial x_j}
$$

For most operations this product is far cheaper than the Jacobian. Take elementwise multiplication $y = a \odot b$ with $n$ elements. The Jacobian $\partial y / \partial a$ is an $n \times n$ diagonal matrix with $b$ on the diagonal. Its VJP is the elementwise product $g \odot b$: $n$ multiplications and no matrix. With $a = [1, 2, 3]$, $b = [4, 5, 6]$ and $g = [1, 1, 1]$, the gradient for $a$ is $[4, 5, 6]$ and the gradient for $b$ is $[1, 2, 3]$.

The same contract holds for every formula in kael: the input of a backward function has the output's shape, and its result has the input's shape and dtype. The engine checks it after every call.

### Two rules every engine must keep

1. **Accumulate.** A value used in several places receives a gradient from each use, and the engine must add them. This is the sum in the multivariable chain rule.
2. **Wait.** A node must not propagate its gradient until every contribution has arrived. If it ran after the first contribution, the inputs would receive a partial gradient, and the later contributions would have nowhere to go.

The second rule is an ordering problem. The engine must visit nodes in an order where every node runs after all the nodes that consume its output. That is a topological order of the reversed graph. micrograd gets it from a depth-first sort; the tensor engine gets it from dependency counting.

### Worked example: tracing a small graph by hand

Take two leaves that require gradients, `x = [1, 2]` and `w = [0.5, -1]`, and this program:

```cpp
Tensor u = x * w;          // [0.5, -2]
Tensor v = u + x;          // [1.5, 0]
Tensor t = kael::tanh(v);  // [0.905148, 0]
Tensor L = t.sum();        // 0.905148
L.backward();
```

`x` is used twice: once in the product and once in the addition. That makes this graph a small diamond, and the trace exercises both rules.

**Forward pass.** Each line runs the operation and records a node.

| Value | Formula | Result | Node recorded | Saves |
|-------|---------|--------|---------------|-------|
| `u` | $x \odot w$ | `[0.5, -2]` | `MulBackward` | `x` and `w` |
| `v` | $u + x$ | `[1.5, 0]` | `AddBackward` | nothing |
| `t` | $\tanh(v)$ | `[0.905148, 0]` | `TanhBackward` | its output `t` |
| `L` | $\sum_i t_i$ | `0.905148` | `SumBackward` | the input shape |

The recorded graph, with edges pointing from each node to the node that receives its input gradients:

```text
SumBackward
  └── TanhBackward
        └── AddBackward
              ├── MulBackward
              │     ├── AccumulateGrad (x)
              │     └── AccumulateGrad (w)
              └── AccumulateGrad (x)        same node as above
```

**Dependency counts.** Before running anything, the engine counts how many edges reach each node from the root.

```text
TanhBackward      1   (from Sum)
AddBackward       1   (from Tanh)
MulBackward       1   (from Add)
AccumulateGrad x  2   (from Add and from Mul)
AccumulateGrad w  1   (from Mul)
```

**Backward pass.** The seed is $\partial L / \partial L = 1$. The local derivative of each node, with $g$ the upstream gradient:

- Sum: $\partial L / \partial t_i = g$ for every $i$, so the scalar is broadcast back to the shape `[2]`.
- Tanh: $\frac{d}{dv}\tanh(v) = 1 - \tanh^2(v) = 1 - t^2$, so the gradient is $g \odot (1 - t^2)$.
- Add: $\partial v / \partial u = 1$ and $\partial v / \partial x = 1$, so both inputs receive $g$ unchanged.
- Mul: $\partial u / \partial x = w$ and $\partial u / \partial w = x$, so the inputs receive $g \odot w$ and $g \odot x$.

Step by step, with the ready stack the engine uses:

| Step | Node runs | Incoming $g$ | Sends | Counts after |
|------|-----------|--------------|-------|--------------|
| 1 | `SumBackward` | `1` | `[1, 1]` to Tanh | Tanh 0, ready |
| 2 | `TanhBackward` | `[1, 1]` | $[1, 1] \odot [1 - 0.819293, 1 - 0] =$ `[0.180707, 1]` to Add | Add 0, ready |
| 3 | `AddBackward` | `[0.180707, 1]` | `[0.180707, 1]` to Mul and to x | Mul 0, ready; x 1, waiting |
| 4 | `MulBackward` | `[0.180707, 1]` | $g \odot w =$ `[0.090353, -1]` to x; $g \odot x =$ `[0.180707, 2]` to w | x 0, ready; w 0, ready |
| 5 | `AccumulateGrad` (w) | `[0.180707, 2]` | `w.grad = [0.180707, 2]` | |
| 6 | `AccumulateGrad` (x) | `[0.180707, 1] + [0.090353, -1] = [0.271060, 0]` | `x.grad = [0.271060, 0]` | |

At step 3, `x` has received one contribution and waits. Only after step 4 delivers the second contribution does its count reach zero. `AccumulateGrad` for `x` runs once, with the sum.

**Check the result by hand.** Elementwise, $L = \sum_i \tanh(x_i w_i + x_i) = \sum_i \tanh(x_i (w_i + 1))$. Differentiating directly:

$$
\frac{\partial L}{\partial x_i} = (1 - t_i^2)(w_i + 1), \qquad \frac{\partial L}{\partial w_i} = (1 - t_i^2) x_i
$$

For $i = 0$: $0.180707 \cdot 1.5 = 0.271060$ and $0.180707 \cdot 1 = 0.180707$. For $i = 1$: $1 \cdot 0 = 0$ and $1 \cdot 2 = 2$. Both match the trace.

**Check it on the real engine.** Setting `KAEL_TRACE_BACKWARD` in the environment, or calling `kael::set_backward_trace(&std::cout)` from `kael/debug.h`, prints one line per node with the incoming gradient's shape and L2 norm. For this program the output is:

```text
backward SumBackward shape=[] norm=1
backward TanhBackward shape=[2] norm=1.41421
backward AddBackward shape=[2] norm=1.0162
backward MulBackward shape=[2] norm=1.0162
backward AccumulateGrad shape=[2] norm=2.00815
backward AccumulateGrad shape=[2] norm=0.27106
```

The norms match the table: $\lVert [1, 1] \rVert = 1.41421$, $\lVert [0.180707, 1] \rVert = 1.0162$, $\lVert [0.180707, 2] \rVert = 2.00815$ for `w`, and $\lVert [0.271060, 0] \rVert = 0.27106$ for `x`. `w` runs before `x` because the engine pops the most recently pushed ready node first. The order between nodes whose counts are both zero does not affect the result.

> [!TIP]
> `kael::autograd_graph_dot(L)` from `kael/debug.h` prints the recorded graph in Graphviz DOT format, with the gradient shape on each edge. Pipe it into `dot -Tsvg` to see the graph of any expression.

### Warm-up: scalar micrograd

[`experiments/micrograd.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/experiments/micrograd.cpp) is a scalar engine in the style of Andrej Karpathy's micrograd. It is built as the `micrograd` executable and registered as a CTest test in `tests/CMakeLists.txt`. Read it first: every concept of the tensor engine appears here in miniature, on single `double` values.

From `experiments/micrograd.cpp`:

```cpp
struct Node {
    double data = 0.0;
    double grad = 0.0;
    std::vector<std::shared_ptr<Node>> parents;
    std::function<void(Node&)> propagate;
};
```

A `Value` is a handle to a shared `Node`, the same handle-and-body split that `Tensor` and `TensorImpl` use (see [ADR 0002](@/adr/0002-tensor-as-handle.md)). Each operator computes the result and attaches a `propagate` closure that pushes the output's gradient into the parents.

From `experiments/micrograd.cpp`:

```cpp
friend Value operator*(const Value& a, const Value& b) {
    return Value::derived(a.data() * b.data(), {a.node_, b.node_}, [](Node& out) {
        out.parents[0]->grad += out.parents[1]->data * out.grad;
        out.parents[1]->grad += out.parents[0]->data * out.grad;
    });
}
```

The closure is the VJP of multiplication: the gradient of `a` is `b * out.grad`, and the gradient of `b` is `a * out.grad`. The `+=` is the accumulation rule. Writing `=` instead breaks every graph where a value is used twice.

`backward()` performs a depth-first topological sort, seeds the root with `1.0` and calls `propagate` in reverse topological order, so every node runs after all the nodes that consume it.

From `experiments/micrograd.cpp`:

```cpp
std::function<void(Node*)> visit = [&](Node* node) {
    if (!visited.insert(node).second) {
        return;
    }
    for (const auto& parent : node->parents) {
        visit(parent.get());
    }
    order.push_back(node);
};
visit(node_.get());

node_->grad = 1.0;
std::for_each(order.rbegin(), order.rend(), [](Node* node) {
    if (node->propagate) {
        node->propagate(*node);
    }
});
```

`visit` appends a node only after all its parents, so `order` lists inputs before outputs. Walking it in reverse visits the root first and every node after all its consumers. Leaves have no `propagate` and are skipped.

The test `neuron_matches_hand_derivation` evaluates one neuron $y = \tanh(w_1 x_1 + w_2 x_2 + b)$ with `w1 = 0.5`, `x1 = 2.0`, `w2 = -1.5`, `x2 = 0.5`, `b = 0.25`. Writing $s$ for the slope of tanh at the activation, the chain rule gives $\partial y / \partial w_1 = s \cdot x_1$ and so on.

```text
activation = 0.5*2.0 + (-1.5)*0.5 + 0.25 = 0.5
y          = tanh(0.5)                 = 0.462117
slope      = 1 - tanh(0.5)^2           = 0.786448

dy/dw1 = slope * x1 =  1.572895
dy/dx1 = slope * w1 =  0.393224
dy/dw2 = slope * x2 =  0.393224
dy/dx2 = slope * w2 = -1.179672
dy/db  = slope      =  0.786448
```

The other three checks pin down the rules the tensor engine must keep:

- `repeated_input_accumulates`: `a * a` with `a = 3` gives `6`, because both parents of the product are the same node and both `+=` land on it. Mathematically, $\frac{d}{da} a^2 = 2a = 6$.
- `diamond_waits_for_both_paths`: `x*2 + x*3` with `x = 2` gives `5`. The topological order guarantees that `x` is final only after both products have added into it.
- `exp_derivative_is_itself`: `exp` reuses its own output, `out.data`, in the backward closure, because $\frac{d}{dx} e^x = e^x$. Saving outputs instead of inputs is a pattern the tensor engine reuses for `exp`, `expm1`, `tanh`, `sigmoid`, `sqrt` and `rsqrt`.

When all four pass, the executable prints `micrograd: gradients match`.

Three limitations of the scalar design motivate the tensor engine:

| micrograd | Problem at scale | Tensor engine answer |
|-----------|------------------|----------------------|
| Recursive DFS in `backward()` | Stack overflow on long chains | Iterative traversal with explicit stacks |
| `grad` lives on every node | Intermediates hold gradients nobody reads | Only leaves (and `retain_grad` tensors) store `grad` |
| Closures read parent `data` at backward time | Mutating a parent after the forward pass silently corrupts gradients | `SavedTensor` checks a version counter |

There is a fourth, less visible one. micrograd's closures keep the whole graph alive as long as the root `Value` lives, and destroying a long chain recurses through `shared_ptr` destructors. The tensor engine frees saved values after each node runs and destroys nodes iteratively.

## Implementation

The implementation has three parts, presented in the order you would write them: the data structures that make up the graph, the recording step that every operation performs, and the engine that walks the graph. The formulas and the checker follow in their own sections.

### Autograd metadata on the tensor

Every `TensorImpl` carries an optional pointer to autograd state. Tensors that never take part in differentiation, which is most tensors during inference, pay one null pointer.

From `include/kael/tensor_impl.h`:

```cpp
std::unique_ptr<autograd::AutogradMeta> autograd;
```

From `include/kael/autograd/meta.h`:

```cpp
struct AutogradMeta {
    bool requires_grad = false;
    std::optional<Tensor> grad;
    std::shared_ptr<Node> grad_fn;
    std::weak_ptr<Node> accumulator;
};
```

- `requires_grad`: the tensor participates in the graph.
- `grad`: the accumulated gradient. Only leaves and tensors marked with `retain_grad()` fill it.
- `grad_fn`: the node that produced this tensor. A tensor with a null `grad_fn` is a leaf.
- `accumulator`: a weak reference to the leaf's `AccumulateGrad` node, so every operation that reads the same leaf shares one accumulator.

The metadata lives on `TensorImpl`, not on `Storage`. Two views of the same buffer are different tensors for autograd: a view has its own history, recorded by its view operation, and its own `requires_grad`. `AutogradFields.ViewDoesNotShareRequiresGradWithoutHistory` checks this.

The `Tensor` accessors are thin reads of this struct.

From `src/autograd/tensor_autograd.cpp`:

```cpp
bool Tensor::requires_grad() const noexcept {
    return impl_->autograd && impl_->autograd->requires_grad;
}

bool Tensor::is_leaf() const noexcept {
    return grad_fn() == nullptr;
}
```

`requires_grad_(bool required = true)` creates the metadata lazily and refuses two things. It refuses to change the flag on a non-leaf ("you can only change requires_grad flags of leaf tensors"), because a non-leaf's flag is a consequence of its history. It refuses to request gradients for a non-floating dtype ("only floating point tensors can require gradients"), because the derivative of an integer-valued function is zero or undefined almost everywhere. `zero_grad()` resets the `grad` optional to empty; it does not fill it with zeros.

The PyTorch equivalents are the same names: `x.requires_grad_()`, `x.is_leaf`, `x.grad`, `x.grad_fn`.

### Nodes and edges

A node is one backward function. An edge points from a node to the node that receives the gradient of one of its inputs. In kael an edge is a `std::shared_ptr<Node>`, so the graph is owned from the output towards the leaves. Holding the loss keeps the whole graph alive, and dropping it frees everything. This matches the order of use: the backward pass starts at the loss, so the loss is the natural owner.

From `include/kael/autograd/node.h`:

```cpp
using Edge = std::shared_ptr<Node>;
using GradientList = std::vector<std::optional<Tensor>>;
using GradientFn = std::function<Tensor(const Tensor&)>;

struct NodeInput {
    Edge edge;
    Shape shape;
    DType dtype;
};
```

`NodeInput` stores the shape and dtype of the input as it was at record time. The engine uses them to validate every gradient a node returns. Storing them separately from the input tensor means the node does not keep the input alive only to remember its shape.

`Node` is an abstract class with `name()`, `apply(gradient)` and `release_variables()`. Two concrete classes exist.

- `FunctionNode` holds a name such as `"MulBackward"` and one `GradientFn` per input. `apply` calls each non-null function and returns the list. A null function means "this input does not need a gradient" and produces `std::nullopt`.
- `AccumulateGrad` holds a strong reference to a leaf's `TensorImpl` and adds the incoming gradient into `leaf.autograd->grad`. It has no inputs, so the traversal stops there.

Using one generic `FunctionNode` with a vector of closures, instead of one class per operation as PyTorch generates, keeps each formula next to its recording call and avoids about fifty class definitions, one per `record_*` function. The cost is a `std::function` call per input per backward, which is negligible next to the tensor work inside it.

The ownership structure:

```text
Tensor ──shared──> TensorImpl ──unique──> AutogradMeta
                                            │  requires_grad
                                            │  grad          (optional<Tensor>)
                                            │  grad_fn ──────shared──────> Node (FunctionNode)
                                            └  accumulator ··weak········> Node (AccumulateGrad)

Node
  inputs_            vector<NodeInput>: each holds an Edge (shared_ptr<Node>), shape, dtype
  retained_output_   weak_ptr<TensorImpl>, set by retain_grad

FunctionNode : Node   name_, gradients_ (one GradientFn per input), released_
AccumulateGrad : Node leaf_ ──shared──> TensorImpl of the leaf
```

The weak references break cycles. A leaf owns its metadata, the metadata must find the accumulator, and the accumulator must reach the leaf. If both directions were strong, neither would ever be freed. The same reasoning applies to `retained_output_`, described under `retain_grad` below.

`gradient_edge` returns the edge a new node should point at for one of its inputs: the input's `grad_fn` if it has one, a shared accumulator if it is a leaf that requires grad, and null otherwise.

From `src/autograd/node.cpp`:

```cpp
Edge gradient_edge(const Tensor& t) {
    if (!t.requires_grad()) {
        return nullptr;
    }
    if (Edge function = t.grad_fn()) {
        return function;
    }
    AutogradMeta& meta = *t.impl()->autograd;
    Edge accumulator = meta.accumulator.lock();
    if (!accumulator) {
        accumulator = std::make_shared<AccumulateGrad>(t.impl());
        meta.accumulator = accumulator;
    }
    return accumulator;
}
```

The accumulator is created on first use and found again through the weak pointer on every later use. Without the sharing, `x * w + x` would create two accumulators for `x`, each with a dependency count of 1, and the engine could not tell that both feed the same leaf. With it, the engine sees one node with two incoming edges and sums before accumulating. When every graph that used the leaf is gone, the accumulator dies and the weak pointer expires; the next operation creates a new one.

`make_input(t)` bundles `gradient_edge(t)` with `t.shape()` and `t.dtype()` into a `NodeInput`.

For `y = (a * b).sum()` with `a` of shape `[2, 3]` and `b` of shape `[3]`, both leaves, the recorded graph is:

```text
SumBackward
  MulBackward
    AccumulateGrad   (a)
    AccumulateGrad   (b)
```

### Recording history

#### Who records

Each differentiable operation computes its result first and then calls a `record_*` function from `src/autograd/record.h`. The computation runs with gradients disabled, so internal helpers do not record their own nodes. The single `record_*` call adds the one node that represents the whole operation.

From `src/ops/binary.cpp`:

```cpp
Tensor add(const Operand& a, const Operand& b) {
    const auto [x, y] = detail::resolve_operands(a, b);
    const DType dtype = detail::add_dtype(x, y);
    Tensor out = autograd::without_grad(
        [&] { return binary_kernel(x.to(dtype), y.to(dtype), detail::kPlus); });
    autograd::record_add(out, x, y);
    return out;
}
```

`without_grad` matters here. `x.to(dtype)` would otherwise record a `ToBackward` node, and the graph of a mixed-precision addition would contain a conversion node plus an addition node. With `without_grad`, it contains one `AddBackward` whose formula casts the gradient back to each input's dtype. Fused operations (`softmax`, `log_softmax`, `cross_entropy_losses`, `standardize` in `src/ops/nn.cpp`) follow the same pattern with larger bodies.

This split between computing and recording is the main design choice of the recording layer. The kernels from [chapter 04](@/book/04-tensor-iterator.md) and [chapter 05](@/book/05-operations.md) know nothing about autograd. The `record_*` functions know nothing about kernels. Each operation's public function is the only place where the two meet.

#### When to record

From `src/autograd/recording.cpp`:

```cpp
bool can_record(const Tensor& output) {
    return GradMode::is_enabled() && is_floating_point(output.dtype());
}

bool should_record(const Tensor& output, std::initializer_list<const Tensor*> inputs) {
    return can_record(output) && std::any_of(inputs.begin(), inputs.end(), [](const Tensor* input) {
               return input->requires_grad();
           });
}
```

An operation records when three conditions hold: grad mode is on, the output is floating point, and at least one input requires grad. A second overload takes a `std::span<const Tensor>` for `cat`, whose inputs are a vector.

Comparisons, `argmax`, `sign`, `floor_divide`, `all` and other operations with integral or boolean outputs never record, because their outputs are not floating point. `AutogradFormulas.NonDifferentiableOpsHaveNoHistory` checks a list of them. `sign` has no `record_*` function at all: its derivative is zero wherever it exists, so the graph treats its result as a constant.

#### Attaching the node

`set_history` builds the `FunctionNode` and installs fresh metadata on the output.

From `src/autograd/recording.cpp`:

```cpp
void set_history(const Tensor& output, std::string name, const std::vector<Tensor>& inputs,
                 std::vector<GradientFn> gradients) {
    std::vector<NodeInput> node_inputs;
    node_inputs.reserve(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        node_inputs.push_back(make_input(inputs[i]));
        if (!inputs[i].requires_grad()) {
            gradients[i] = nullptr;
        }
    }
    auto meta = std::make_unique<AutogradMeta>();
    meta->requires_grad = true;
    meta->grad_fn = std::make_shared<FunctionNode>(std::move(name), std::move(node_inputs),
                                                   std::move(gradients));
    output.impl()->autograd = std::move(meta);
}
```

Nulling the gradient function of an input that does not require grad does two things. The engine skips that computation, and the closure, with any tensors it captured, is destroyed immediately. In `Linear`, `matmul(input, weight.T())` records a closure for the input gradient that captures the weight, and a closure for the weight gradient that captures the input. When the input is raw data, its closure dies at once and the graph does not keep an extra reference to the weight.

Note that the output's metadata is replaced, not merged. The output was created a few lines earlier by the kernel and has no metadata of its own.

#### Saved tensors and version counters

Most formulas need forward values: `mul` needs the other operand, `exp` needs its output. They capture a `SavedTensor`.

From `src/autograd/recording.h`:

```cpp
class SavedTensor {
public:
    explicit SavedTensor(const Tensor& t) : data_(t.detach()), version_(t.version()) {}

    [[nodiscard]] const Tensor& get() const {
        KAEL_CHECK(data_.version() == version_,
                   "one of the variables needed for gradient computation has been modified by an "
                   "inplace operation: expected version "
                       << version_ << ", got " << data_.version());
        return data_;
    }

private:
    Tensor data_;
    std::int64_t version_;
};
```

Two choices are deliberate.

1. It stores `t.detach()`, a view without autograd metadata. Storing the tensor itself would create a cycle when an operation saves its own output: the output impl owns its metadata, the metadata owns `grad_fn`, and `grad_fn` owns the saved output impl. `AutogradEngine.DetachedSavedOutputDoesNotLeak` checks that the node of `exp(x)` dies with its output.
2. It records the storage version. Every in-place operation calls `storage->bump_version()` (see [03. Views](@/book/03-views.md)), and views share the counter because it lives on `Storage`. Any write through any alias after the forward pass is detected at backward time.

Saving by `detach()` is also cheap: a view shares the buffer, so saving a tensor copies no data. Activation memory in training is exactly this: the buffers that saved tensors keep alive between the forward and the backward pass.

Example from `AutogradVersion.ModifyingSavedInputThrows`:

```cpp
const Tensor x = leaf({1, 2});
Tensor constant = doubles({3, 4});
const Tensor y = (x * constant).sum();
constant.add_(1);
y.backward();
```

`MulBackward` saved `constant` at version 0. `add_` moves it to version 1. Backward throws:

```text
one of the variables needed for gradient computation has been modified by an inplace operation: expected version 0, got 1
```

Without the check, `x.grad` would silently be `[4, 5]` instead of `[3, 4]`: the formula $g \odot c$ would read the new values of $c$, not the ones the forward pass used. The message is the same one PyTorch uses for the same situation, so the error is searchable.

A version counter does not catch every alias problem. It only detects writes that go through kael's in-place operations, which all bump the counter. Writing through a raw `data_ptr<T>()` bypasses it.

#### In-place operations and autograd

kael does not rebase history for in-place operations. `src/ops/inplace.cpp` rejects every in-place operation that would need it while grad mode is on.

From `src/ops/inplace.cpp`:

```cpp
void check_grad_rules(const Tensor& self, const Tensor* source, const char* op) {
    if (!autograd::GradMode::is_enabled()) {
        return;
    }
    KAEL_CHECK(!self.requires_grad() || !self.is_leaf(),
               "a leaf tensor that requires grad is being used in the in-place operation " << op);
    KAEL_CHECK(
        !self.requires_grad(),
        op << " is not supported on a tensor with gradient history; use the out-of-place op");
    KAEL_CHECK(source == nullptr || !source->requires_grad(),
               op << " is not supported with a source that requires grad");
}
```

| Target of `add_` | Grad mode on | Grad mode off |
|------------------|--------------|---------------|
| Plain tensor | allowed, bumps version | allowed, bumps version |
| Leaf with `requires_grad` | throws | allowed (optimizer updates) |
| Non-leaf with history | throws | allowed |
| Source requires grad | throws | allowed |

The three checks cover three distinct problems. Writing into a leaf that requires grad would destroy the value the graph was recorded against. Writing into a non-leaf would require its `grad_fn` to change to describe the composed operation. Writing a source that requires grad into a plain tensor would make the plain tensor depend on the graph without any record of it.

This keeps the engine small. PyTorch supports in-place operations on graph tensors by rewriting `grad_fn` to a node that composes the old history with the in-place operation; that requires per-view base tracking that kael does not have. The cost is that user code must write `y = y + 1` instead of `y.add_(1)` inside a differentiable computation. Optimizers run under `NoGradGuard` and are not affected.

## The engine

`kael::autograd::backward(root, gradient, retain_graph)` in `include/kael/autograd/engine.h` is the only entry point. `Tensor::backward(retain_graph)` and `Tensor::backward(gradient, retain_graph)` forward to it. The function first checks that the root requires grad ("backward needs a tensor that requires grad"), then installs a `NoGradGuard` for its whole body.

The guard has an important consequence: the formula computations do not record history, so gradients are plain tensors. kael computes first-order gradients only. There is no equivalent of PyTorch's `create_graph=True`, and higher-order derivatives such as Hessian-vector products are out of scope.

### Seeding

From `src/autograd/engine.cpp`:

```cpp
Tensor seed_gradient(const Tensor& root, const std::optional<Tensor>& gradient) {
    if (!gradient) {
        KAEL_CHECK_VALUE(root.numel() == 1,
                         "grad can be implicitly created only for scalar outputs, got "
                         "shape "
                             << to_string(root.shape()));
        return ones_like(root);
    }
    KAEL_CHECK_VALUE(gradient->shape() == root.shape(), "gradient shape "
                                                            << to_string(gradient->shape())
                                                            << " does not match output shape "
                                                            << to_string(root.shape()));
    return gradient->to(root.dtype());
}
```

A scalar root gets an implicit gradient of one. A non-scalar root needs an explicit gradient of the same shape. That gradient is the vector $v$ in the vector-Jacobian product $v^\top J$: passing $v$ to `y.backward(v)` computes the gradient of the scalar $\sum_i v_i y_i$.

`AutogradFormulas.MeanExampleFromHand` passes `[3, 6]` to `mean(x, {1})` for `x` of shape `[2, 3]` and checks that row 0 receives $3/3 = 1$ and row 1 receives $6/3 = 2$ per element.

The root does not need a `grad_fn`. The engine starts from `gradient_edge(root)`, so a leaf root starts at its own `AccumulateGrad`: `AutogradEngine.LeafAsRootReceivesSeed` calls `backward()` on a one-element leaf and gets `grad = [1]`.

### Ordering by dependency count

The engine needs a node to run only after every consumer has delivered its gradient. Instead of a DFS topological sort, it counts, for each node, how many edges reach it from the root's subgraph, then runs Kahn's algorithm: a node becomes ready when its count drops to zero.

From `src/autograd/engine.cpp`:

```cpp
std::unordered_map<Node*, std::int64_t> count_dependencies(Node* root) {
    std::unordered_map<Node*, std::int64_t> dependencies;
    std::unordered_set<Node*> visited{root};
    std::vector<Node*> pending{root};
    while (!pending.empty()) {
        const Node* node = pending.back();
        pending.pop_back();
        for (const NodeInput& input : node->inputs()) {
            Node* next = input.edge.get();
            if (next == nullptr) {
                continue;
            }
            ++dependencies[next];
            if (visited.insert(next).second) {
                pending.push_back(next);
            }
        }
    }
    return dependencies;
}
```

Every edge increments its target's count, even when the target was already visited. Only unvisited targets are pushed for exploration. The result is the in-degree of each node inside the subgraph reachable from the root.

The traversal uses an explicit stack, so depth is bounded by heap memory, not by the call stack. `AutogradEngine.DeepChainBackwardAndDestruction` runs backward through 10000 chained additions.

Why Kahn and not a DFS order as in micrograd: the counts are needed anyway to know when a gradient buffer is complete, and they let the engine run a node the moment it is ready instead of after a full sort. Counting only the subgraph reachable from the root matters when graphs share nodes. If `a` feeds both `loss1` and `loss2`, calling `loss1.backward()` must not wait for a contribution from `loss2` that will never come.

### The main loop

From `src/autograd/engine.cpp`:

```cpp
while (!ready.empty()) {
    Node* node = ready.back();
    ready.pop_back();

    GradientList gradients;
    if (const auto buffered = buffers.find(node); buffered != buffers.end()) {
        const Tensor incoming = std::move(buffered->second);
        buffers.erase(buffered);
        node->store_retained_gradient(incoming);
        if (std::ostream* output = detail::trace_output().load()) {
            trace_node(*output, *node, incoming);
        }
        gradients = node->apply(incoming);
        if (detail::anomaly_detection().load()) {
            check_anomaly(*node, gradients);
        }
    }
    if (!retain_graph) {
        node->release_variables();
    }

    const std::vector<NodeInput>& inputs = node->inputs();
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        Node* next = inputs[i].edge.get();
        if (next == nullptr) {
            continue;
        }
        if (i < gradients.size() && gradients[i]) {
            check_gradient(*node, i, *gradients[i]);
            add_to_buffer(buffers, next, *gradients[i]);
        }
        if (--dependencies[next] == 0) {
            ready.push_back(next);
        }
    }
}
```

Before the loop, the engine stores the seed in `buffers` under the start node and pushes the start node onto `ready`. Per node, in order:

1. Take the summed gradient out of `buffers`. A node that received no gradient, because every path to it returned `nullopt`, skips `apply` but still releases its inputs' dependency counts, so the loop always terminates.
2. Store it for `retain_grad` if requested.
3. If tracing is on, print the node name, shape and norm. Call `apply` to get one optional gradient per input. If anomaly detection is on, check the results for NaN.
4. Unless `retain_graph`, call `release_variables()`. For a `FunctionNode` this clears the `GradientFn` list and with it every `SavedTensor`. This frees forward activations as soon as they are no longer needed, and makes a second `backward()` through the same node fail with `Trying to backward through the graph a second time (...); specify retain_graph=true`.
5. For each input edge, validate the gradient with `check_gradient`, add it into the target's buffer, decrement the target's count, and push the target when the count reaches zero.

`check_gradient` compares the returned gradient with the recorded `NodeInput`:

```text
Function MulBackward returned an invalid gradient at index 1 - got [2, 3] but expected shape [3]
```

A message of this form is what a formula with a missing `sum_to_shape` produces. It names the node and the input index, which points directly at the faulty formula.

Buffer accumulation is out of place.

From `src/autograd/engine.cpp`:

```cpp
void add_to_buffer(std::unordered_map<Node*, Tensor>& buffers, Node* node, const Tensor& gradient) {
    const auto [slot, inserted] = buffers.try_emplace(node, gradient);
    if (!inserted) {
        slot->second = slot->second + gradient;
    }
}
```

The first gradient is stored by handle, without a copy. It may alias a tensor that a formula returned unchanged. In the worked example above, `AddBackward` returns `g` itself to both inputs: the buffer for `MulBackward` and the buffer for `AccumulateGrad (x)` hold the same tensor. An in-place `add_` into x's buffer would change the tensor that `MulBackward` is about to read. The sum allocates a new tensor instead.

The ready list is a stack, so the traversal is depth-first among ready nodes. Any order that respects the counts gives the same gradients up to floating-point rounding, because addition of the contributions is the only order-dependent step. PyTorch's engine also counts dependencies, but it orders ready nodes by a sequence number assigned at creation, so the most recently created node runs first, and it can run nodes on several device threads.

### Worked trace: the diamond

`y = (x*2 + x*3).sum()` with `x = [1, 2]`:

```text
SumBackward
  └── AddBackward
        ├── MulBackward (x*2) ──┐
        └── MulBackward (x*3) ──┴── AccumulateGrad (x)
```

Dependency counts: `AddBackward` 1, each `MulBackward` 1, `AccumulateGrad` 2. The scalar operands `2` and `3` do not require grad, so each `MulBackward` has a null edge for its second input and a null gradient function for it.

```text
step  pop             incoming   outputs                      counts after
1     SumBackward     1          [1, 1] -> Add                Add 0  -> push
2     AddBackward     [1, 1]     [1, 1] -> M1, [1, 1] -> M2   M1 0, M2 0 -> push both
3     MulBackward x*3 [1, 1]     [3, 3] -> X                  X 1
4     MulBackward x*2 [1, 1]     [2, 2] -> X (buffer 5, 5)    X 0 -> push
5     AccumulateGrad  [5, 5]     x.grad = [5, 5]
```

`AddBackward` pushes `M1` first and `M2` second, so `M2` (the `x*3` branch) pops first. `AccumulateGrad` runs once, with the sum. This is `AutogradEngine.DiamondWaitsForBothPaths` in `tests/test_autograd_engine.cpp`. The trace output of the real engine shows the five nodes in this order, ending with `AccumulateGrad shape=[2] norm=7.07107`, which is $\lVert [5, 5] \rVert$.

### Accumulating into leaves

From `src/autograd/node.cpp`:

```cpp
void accumulate(TensorImpl& target, const Tensor& gradient) {
    if (!target.autograd) {
        return;
    }
    std::optional<Tensor>& stored = target.autograd->grad;
    if (stored) {
        stored->add_(gradient);
    } else {
        stored = gradient.clone();
    }
}
```

The first gradient is cloned, so `x.grad` never aliases an engine buffer or a saved tensor. `AutogradEngine.GradIsACopyOfThePropagatedGradient` checks the values and that the stored gradient does not require grad. Later gradients are added in place, because the stored gradient is owned by the leaf and nobody else holds it. `add_` is allowed here because the engine runs under `NoGradGuard` and the stored gradient is a plain tensor.

This is why gradients accumulate across `backward()` calls until `zero_grad()` resets the optional. `AutogradEngine.LeafSharedByTwoGraphsAccumulates` runs `(x*2).sum()` and then `(x*3).sum()` and leaves `x.grad = [5, 5]`. Accumulation across calls is a feature: it lets a training loop sum the gradients of several micro-batches before one optimizer step. PyTorch behaves the same way. The early return handles tensors detached in place, described below.

### Gradient mode

From `src/autograd/grad_mode.cpp`:

```cpp
thread_local bool grad_enabled = true;
```

The flag is per thread. `GradModeGuard` saves the previous value and restores it in its destructor, so guards nest and survive exceptions (`GradMode.NestedGuardsRestoreInOrder`, `GradMode.ExceptionRestoresState`). `NoGradGuard` and `EnableGradGuard` are the two fixed-value subclasses, and `without_grad(f)` runs `f` under a `NoGradGuard` and returns its result.

From `include/kael/autograd/grad_mode.h`:

```cpp
template <typename F> auto without_grad(F compute) {
    const NoGradGuard guard;
    return compute();
}
```

`NoGradGuard` is the C++ form of PyTorch's `with torch.no_grad():`, and `EnableGradGuard` corresponds to `torch.enable_grad()`. Use the first for inference and optimizer updates: no nodes are created, and no tensors are saved, so memory stays flat. The Python bindings expose the flag as `is_grad_enabled` and `set_grad_enabled` (see [chapter 10](@/book/10-python-bindings.md)).

A thread-local flag interacts with the thread pool from [08. CPU performance](@/book/08-cpu-performance.md). A kernel launched from a `NoGradGuard` scope runs its chunks on worker threads whose own flag is still `true`. `src/parallel.cpp` copies the caller's mode into each job.

From `src/parallel.cpp`:

```cpp
Job(std::int64_t begin, std::int64_t end, std::int64_t chunk_size, const ParallelBody& body)
    : begin_(begin), end_(end), chunk_size_(chunk_size), body_(body),
      grad_enabled_(autograd::GradMode::is_enabled()) {}

void work() noexcept {
    const autograd::GradModeGuard grad_mode(grad_enabled_);
    ...
```

Why thread-local and not global: two threads can run independent work, one training and one evaluating, and a global flag would let the evaluating thread disable recording for the training thread.

### `retain_grad`

Intermediate tensors do not store gradients. That saves memory: in a deep network, storing the gradient of every activation would double the memory of the backward pass. When you need one for debugging, `retain_grad()` asks the tensor's `grad_fn` to remember the output with a weak reference.

From `src/autograd/tensor_autograd.cpp`:

```cpp
void Tensor::retain_grad() {
    KAEL_CHECK(requires_grad(), "retain_grad needs a tensor that requires grad");
    if (const auto function = grad_fn()) {
        function->retain_output(impl_);
    }
}
```

When the engine pops that node it calls `store_retained_gradient(incoming)` before `apply`, which runs the same `accumulate` used by leaves. On a leaf, `retain_grad` does nothing because leaves already store their gradient. The reference is weak so a retained tensor that goes out of scope is not resurrected by its own graph.

`AutogradEngine.RetainGradStoresIntermediateGradient`: for `middle = x * 3` and loss `(middle * 2).sum()`, `middle.grad = [2, 2]` and `x.grad = [6, 6]`. The incoming gradient at `MulBackward` for `middle` is the gradient of the loss with respect to `middle`, which is exactly what `retain_grad` stores.

### `detach` and `detach_`

`detach()` returns a new view on the same storage with no autograd metadata. `make_view` in `src/view.cpp` copies storage, shape, strides, offset, dtype and device, nothing else. Gradients stop at it: `(x.detach() * x).sum()` gives `x.grad = x`, not `2x` (`AutogradDetach.CutsTheGradient`). With `x = [1, 2]`, the result is `[1, 2]`, because only the second factor carries history and its local derivative is the detached first factor.

It shares the version counter, so writing through a detached alias of a saved tensor still trips the version check (`AutogradVersion.ModifyingSavedOutputThroughDetachThrows`).

`detach_()` resets the tensor's own metadata.

From `src/autograd/tensor_autograd.cpp`:

```cpp
Tensor& Tensor::detach_() {
    impl_->autograd.reset();
    return *this;
}
```

The tensor becomes a leaf that does not require grad. Graphs that were already recorded still reference its `TensorImpl`: an `AccumulateGrad` holds it strongly, and a `retain_grad` node holds it weakly. Commit `fix(autograd): skip gradients of tensors detached in place` made `accumulate` return early when `target.autograd` is null. Before that fix, running backward through such a graph dereferenced a null `unique_ptr`.

The regression tests are `AutogradDetach.InPlaceOnALeafInTheGraphSkipsItsGradient` and `AutogradDetach.InPlaceOnARetainedOutputSkipsItsGradient`. In the second one, `y = x * 2` is retained, then detached in place. Backward of `(y * 3).sum()` still reaches `x` (`x.grad = [6, 6]`) because the graph edges are untouched, but `y.grad` stays empty.

### Destroying long graphs

A chain of 50000 additions creates 50000 nodes, each owning the previous one through a `shared_ptr`. The default destructor of the last node would destroy its input, whose destructor destroys its input, and so on: one set of stack frames per node, enough to overflow the stack. `Node::~Node` flattens that recursion.

From `src/autograd/node.cpp`:

```cpp
// A long chain would otherwise overflow the stack by destroying each node from the previous one.
Node::~Node() {
    std::vector<Edge> pending;
    move_edges_into(pending);
    while (!pending.empty()) {
        const Edge edge = std::move(pending.back());
        pending.pop_back();
        if (edge && edge.use_count() == 1) {
            edge->move_edges_into(pending);
        }
    }
}
```

When `edge` is the last owner of a node (`use_count() == 1`), the loop steals that node's edges before the node dies. The node's own destructor then finds `inputs_` empty and returns at once. Nodes still owned elsewhere, such as a shared leaf accumulator or a node whose output tensor the user still holds, are left alone.

Trace it on a chain of three nodes `C -> B -> A`, where the user drops `C`:

```text
~C: pending = [B]                         (C's edge moved out)
pop B, use_count 1: pending = [A]         (B's edges moved out, B dies with no edges)
pop A, use_count 1: pending = []          (A's edges moved out, A dies with no edges)
```

The stack depth stays at one destructor call regardless of chain length. `AutogradEngine.DeepChainDestructionWithoutBackward` builds 50000 nodes and drops them without calling `backward`, which is the case where `release_variables` never ran and every node is still fully populated.

## Derivative formulas

Every formula is a lambda `Tensor(const Tensor& g)`, where `g` has the output's shape. It must return a tensor of the input's shape and dtype, or the engine rejects it with `Function <name> returned an invalid gradient at index <i>`. This section derives each formula from the VJP definition and shows the kael code.

### Broadcasting and `sum_to_shape`

If `a` has shape `[2, 3]` and `b` has shape `[3]`, `a * b` broadcasts `b` to `[2, 3]` (see [01. Shapes and strides](@/book/01-shapes-and-strides.md)). The gradient for `b` comes back as `[2, 3]` and must be reduced. Each element of `b` was used twice, once per row, so by the multivariable chain rule its gradient is the sum over the broadcast dimension.

From `src/autograd/recording.cpp`:

```cpp
Tensor sum_to_shape(const Tensor& gradient, const Shape& target) {
    KAEL_CHECK_VALUE(gradient.dim() >= static_cast<std::int64_t>(target.size()),
                     "cannot reduce gradient of shape " << to_string(gradient.shape()) << " to "
                                                        << to_string(target));
    const auto extra = static_cast<std::size_t>(gradient.dim()) - target.size();
    Tensor reduced = gradient;
    if (extra > 0) {
        Dims leading(extra);
        std::iota(leading.begin(), leading.end(), std::int64_t{0});
        reduced = kael::sum(reduced, leading);
    }
    Dims stretched;
    for (std::size_t i = 0; i < target.size(); ++i) {
        if (target[i] == 1 && reduced.shape()[i] != 1) {
            stretched.push_back(static_cast<std::int64_t>(i));
        }
    }
    return stretched.empty() ? reduced : kael::sum(reduced, stretched, true);
}
```

Broadcasting does two things: it adds leading dimensions and it stretches size-1 dimensions. `sum_to_shape` undoes both. It sums away the leading ones, then sums the stretched ones with `keepdim`. With `g = [[1, 2, 3], [4, 5, 6]]`:

| Target | Steps | Result |
|--------|-------|--------|
| `[2, 3]` | none | `g` |
| `[3]` | sum dim 0 | `[5, 7, 9]` |
| `[1, 3]` | dim 0 is stretched, sum keepdim | `[[5, 7, 9]]` |
| `[2, 1]` | dim 1 is stretched, sum keepdim | `[[6], [15]]` |
| `[]` | sum dims 0 and 1 | `21` |

The duality is worth remembering: broadcasting in the forward pass becomes summation in the backward pass, and summation in the forward pass becomes broadcasting in the backward pass, as `SumBackward` shows below. When nothing was broadcast, `sum_to_shape` returns its argument by handle and costs nothing.

Every binary formula goes through `matching_input`, which applies `sum_to_shape` and casts back to the input's dtype. The cast handles type promotion: `F32 * F64` computes in `F64`, but the `F32` input must receive an `F32` gradient (`AutogradFormulas.MixedPrecisionInputKeepsItsDtype`).

From `src/autograd/functions_math.cpp`:

```cpp
GradientFn matching_input(const Tensor& input, GradientFn gradient) {
    return [shape = input.shape(), dtype = input.dtype(), gradient = std::move(gradient)](
               const Tensor& g) { return sum_to_shape(gradient(g), shape).to(dtype); };
}
```

The wrapper captures only the input's shape and dtype, never the input itself. `binary_history` wraps both gradient functions of a binary operation this way, and `unary_history` passes a unary formula through unchanged, because a unary elementwise operation neither broadcasts nor promotes.

### Multiplication

For $y = a \odot b$ elementwise, $\partial y_i / \partial a_i = b_i$ and every off-diagonal entry of the Jacobian is zero, so the VJP is $g \odot b$. Symmetrically, the gradient for $b$ is $g \odot a$.

From `src/autograd/functions_math.cpp`:

```cpp
void record_mul(const Tensor& out, const Tensor& a, const Tensor& b) {
    if (should_record(out, {&a, &b})) {
        binary_history(
            out, "MulBackward", a, b,
            [saved = SavedTensor(b)](const Tensor& g) { return g * saved.get(); },
            [saved = SavedTensor(a)](const Tensor& g) { return g * saved.get(); });
    }
}
```

The gradient for `a` saves `b`, and the gradient for `b` saves `a`. If only `a` requires grad, `set_history` destroys the second closure and with it the saved `a`.

Worked example with `a = [[1, 2, 3], [4, 5, 6]]`, `b = [10, 20, 30]`, loss `(a * b).sum()`. `SumBackward` expands the seed `1` to `g = ones([2, 3])`.

```text
a.grad = sum_to_shape(g * b, [2, 3]) = [[10, 20, 30], [10, 20, 30]]
b.grad = sum_to_shape(g * a, [3])    = [1+4, 2+5, 3+6] = [5, 7, 9]
```

`g * b` broadcasts `b` again, so the gradient for `a` already has `a`'s shape and passes through `sum_to_shape` untouched. `g * a` has shape `[2, 3]` and is summed over dim 0 to reach `b`'s shape.

### Elementwise formulas

The other elementwise formulas follow the same template. Each row gives the local derivative, the resulting backward expression, and what the closure saves.

| Op | Local derivative | Backward | Saves |
|----|------------------|----------|-------|
| `add` | $1$, $1$ | `g`, `g` | nothing |
| `sub` | $1$, $-1$ | `g`, `-g` | nothing |
| `div` | $1/b$, $-a/b^2$ | `g / b`, `-g * a / b^2` | `a`, `b` |
| `pow` | $p x^{p-1}$, $y \ln x$ | `g * p * x^(p-1)` (0 where `p == 0`), `g * y * log(x)` (0 where `x == 0`) | base, exponent, output |
| `maximum`, `minimum` | 1 for the winner, 0 for the loser | `g` to the winner, `g/2` to each on a tie | both inputs |
| `neg` | $-1$ | `-g` | nothing |
| `abs` | $\mathrm{sign}(x)$ | `g * sign(x)` | input |
| `exp` | $e^x = y$ | `g * y` | output |
| `expm1` | $e^x = y + 1$ | `g * (y + 1)` | output |
| `log` | $1/x$ | `g / x` | input |
| `log1p` | $1/(1 + x)$ | `g / (x + 1)` | input |
| `sqrt` | $1/(2\sqrt{x}) = 1/(2y)$ | `g / (2y)` | output |
| `rsqrt` | $-\frac{1}{2} x^{-3/2} = -\frac{1}{2} y / x$ | `g * y / x * -0.5` | output, input |
| `sin` | $\cos x$ | `g * cos(x)` | input |
| `cos` | $-\sin x$ | `-g * sin(x)` | input |
| `tanh` | $1 - y^2$ | `g * (1 - y^2)` | output |
| `sigmoid` | $y (1 - y)$ | `g * y * (1 - y)` | output |
| `relu` | $1$ if $x > 0$, else $0$ | `g * (x > 0)` | input |
| `gelu` | exact or tanh-approximation slope | `g * slope(x)` | input |
| `clamp` | $1$ inside the range, $0$ outside | `g * (min <= x <= max)` | input |
| `clone` | $1$ | `g` | nothing |

Several choices in this table are worth explaining.

- **Saving the output.** For `exp`, `expm1`, `sqrt`, `tanh` and `sigmoid` the derivative is cheaper to express through the output than through the input. Saving `y` avoids recomputing $e^x$ or $\tanh x$ in the backward pass.
- **`pow` guards.** At $p = 0$, the term $p \cdot x^{p-1}$ becomes $0 \cdot x^{-1}$, which is NaN at $x = 0$. The formula forces the result to 0 there. At $x = 0$, $\ln x$ is $-\infty$, and the exponent's gradient is forced to 0. Both guards use `where`.
- **Ties in `maximum` and `minimum`.** On a tie the function is not differentiable. Giving `g/2` to each side splits the gradient evenly, so the total that flows back still equals `g`. `extreme_weight` builds the weight tensor with `where(ties, 0.5, wins)`.
- **`gelu`.** The slope is computed by `floating_unary_kernel` with a per-element function: $\Phi(x) + x \varphi(x)$ for the exact form, where $\Phi$ is the standard normal CDF and $\varphi$ its density, or the derivative of the tanh approximation when `approximate` is true.

Conventions at non-differentiable points match PyTorch: $\mathrm{relu}'(0) = 0$ and $\mathrm{abs}'(0) = \mathrm{sign}(0) = 0$ (`AutogradFormulas.ConventionsAtNonDifferentiablePoints`). Any value in the subgradient would be mathematically defensible; matching PyTorch makes golden tests possible.

A worked example for `tanh`: with $x = 0.5$, $y = 0.462117$ and $g = 2$, the gradient is $2 \cdot (1 - 0.462117^2) = 2 \cdot 0.786448 = 1.572896$.

### Matrix multiplication

For $C = AB$ with $A$ of shape $[m, k]$ and $B$ of shape $[k, n]$, each entry is $C_{ij} = \sum_p A_{ip} B_{pj}$. The entry $A_{ip}$ appears in every $C_{ij}$ of row $i$, with local derivative $\partial C_{ij} / \partial A_{ip} = B_{pj}$. Summing over the outputs it touches, with upstream gradient $G$:

$$
\frac{\partial L}{\partial A_{ip}} = \sum_j G_{ij} B_{pj} = (G B^\top)_{ip}
$$

$$
\frac{\partial L}{\partial B_{pj}} = \sum_i G_{ij} A_{ip} = (A^\top G)_{pj}
$$

A shape check confirms the formulas: $G B^\top$ is $[m, n] \times [n, k] = [m, k]$, the shape of $A$, and $A^\top G$ is $[k, m] \times [m, n] = [k, n]$, the shape of $B$. When you forget which side the transpose goes on, the shapes tell you.

Worked example with `A = [[1, 2], [3, 4]]`, `B = [[5, 6], [7, 8]]`, loss `(A B).sum()`, so `G = ones([2, 2])`:

```text
B^T      = [[5, 7], [6, 8]]
G B^T    = [[11, 15], [11, 15]]   = A.grad
A^T      = [[1, 3], [2, 4]]
A^T G    = [[4, 4], [6, 6]]       = B.grad
```

Read the first row of `A.grad`: $A_{00}$ multiplies $B_{00} = 5$ in $C_{00}$ and $B_{01} = 6$ in $C_{01}$, so its gradient is $5 + 6 = 11$.

`record_matmul` also handles 1-D operands and batching. `matmul` treats a 1-D left operand as a row and a 1-D right operand as a column, then removes that dimension from the result (see [05. Operations](@/book/05-operations.md)). `MatmulLayout` reinserts the dimension in `g` and in the saved operand, applies the two matrix formulas, and squeezes it out again. For batched inputs, `mT()` transposes the last two dimensions and `sum_to_shape` folds broadcast batch dimensions back.

From `src/autograd/functions_linalg.cpp`:

```cpp
[layout, right = SavedTensor(layout.right_matrix(b)), shape = a.shape(),
 dtype = a.dtype()](const Tensor& g) {
    const Tensor gradient = matmul(layout.matrix_gradient(g), right.get().mT());
    const Tensor reduced =
        sum_to_shape(gradient, layout.left_is_vector ? Shape{1, shape[0]} : shape);
    return (layout.left_is_vector ? reduced.squeeze(0) : reduced).to(dtype);
},
```

The gradient for `A` saves `B`, and the gradient for `B` saves `A`. If only `B` requires grad, `set_history` drops the closure that captured `B`. The backward of one matmul is two matmuls of the same size, which is why the backward pass of a network dominated by matrix products costs about twice its forward pass.

A 1-D example: for a vector $v$ of shape $[k]$ and a matrix $B$ of shape $[k, n]$, `matmul(v, B)` has shape $[n]$. In the backward pass `g` of shape $[n]$ becomes $[1, n]$, the left gradient $g B^\top$ has shape $[1, k]$, and `squeeze(0)` returns it to $[k]$.

### Reductions

The gradient of a reduction is the incoming gradient broadcast back over the reduced dimensions, times a local factor. Two helpers in `src/autograd/functions_reduce.cpp` do the broadcasting: `restore_reduced_dims` reinserts the dimensions that `keepdim = false` removed, and `expand_reduced` broadcasts to the input shape.

| Reduction | Local factor | Notes |
|-----------|--------------|-------|
| `sum` | 1 | |
| `mean` | `1 / count` | `count` is the product of the reduced sizes |
| `prod` | product of the other elements | three cases: no zero, one zero, several zeros |
| `max`, `min` | `1 / ties` on the arg-extremes, 0 elsewhere | `AutogradFormulas.MaxSplitsTiesEvenly` |
| `var` | `2 (x - mean) / (count - correction)` | `correction` 0 or 1 |
| `norm` | `x / norm` | 0 where `norm == 0` |
| `logsumexp` | `softmax(x)` | |

The derivations are short. For a sum, $\partial (\sum_j x_j) / \partial x_i = 1$. For a mean over $n$ elements, $1/n$. For the L2 norm $\lVert x \rVert = (\sum_j x_j^2)^{1/2}$, $\partial \lVert x \rVert / \partial x_i = x_i / \lVert x \rVert$. For the variance with correction $c$, $\sigma^2 = \frac{1}{n - c} \sum_j (x_j - \mu)^2$, and the terms from $\mu$'s own dependence on $x_i$ sum to zero, leaving $2 (x_i - \mu) / (n - c)$. For $\mathrm{logsumexp}(x) = \ln \sum_j e^{x_j}$, the derivative is $e^{x_i} / \sum_j e^{x_j}$, which is the softmax.

Worked example for `mean`: `x` has shape `[2, 3]`, `y = mean(x, {1})` has shape `[2]`, and the upstream gradient is `[3, 6]`.

```text
g                      [3, 6]                    shape [2]
restore_reduced_dims   [[3], [6]]                shape [2, 1]
expand                 [[3, 3, 3], [6, 6, 6]]    shape [2, 3]
divide by count = 3    [[1, 1, 1], [2, 2, 2]]
```

`prod` avoids dividing by zero. With no zero in the slice, the gradient of element $i$ is $\prod_{j \ne i} x_j = \mathrm{prod} / x_i$. With exactly one zero, only that position gets a non-zero gradient: the product of the others. With two or more zeros, every product of "all others" contains a zero, so every gradient is zero. The implementation computes `others = prod(where(x == 0, 1, x))`, counts zeros per slice, and selects among the three cases with `where` (`AutogradFormulas.ProdWithZeros`).

```text
x = [2, 0, 3]   one zero    others = 2*1*3 = 6   grad = [0, 6, 0]
x = [2, 0, 0]   two zeros                        grad = [0, 0, 0]
x = [2, 4, 3]   no zero     prod = 24            grad = [24/2, 24/4, 24/3] = [12, 6, 8]
```

For `max` and `min`, the formula compares every input element with the saved result. Elements equal to the extreme share the gradient. `AutogradFormulas.MaxSplitsTiesEvenly` takes `x = [3, 3, 1, 3]`: three elements tie at the maximum, each receives $1/3$, and the `1` receives $0$. PyTorch's full reduction `torch.max(x)` also splits ties evenly, while its `max(x, dim)` with indices sends the gradient to one index only.

`cumsum` along a dimension has the reverse cumulative sum as its backward: element $i$ contributes to outputs $i, i+1, \dots$, so its gradient is $\sum_{j \ge i} g_j$. The formula computes it as `sum(g) - cumsum(g) + g`. With `g = [1, 10, 100, 1000]` the gradient is `[1111, 1110, 1100, 1000]`.

### Softmax and log-softmax

For $y = \mathrm{softmax}(x)$ along a dimension, $y_i = e^{x_i} / \sum_k e^{x_k}$, and the Jacobian is $\partial y_i / \partial x_j = y_i (\delta_{ij} - y_j)$, where $\delta_{ij}$ is 1 when $i = j$ and 0 otherwise. Multiply by $g_i$ and sum over $i$:

$$
\frac{\partial L}{\partial x_j} = \sum_i g_i y_i (\delta_{ij} - y_j) = y_j \left( g_j - \sum_i g_i y_i \right)
$$

From `src/autograd/functions_reduce.cpp`:

```cpp
void record_softmax(const Tensor& out, const Tensor& x, std::int64_t dim) {
    if (should_record(out, {&x})) {
        set_history(out, "SoftmaxBackward", {x}, {[y = SavedTensor(out), dim](const Tensor& g) {
                        return y.get() * (g - kael::sum(g * y.get(), {dim}, true));
                    }});
    }
}
```

The formula never forms the $n \times n$ Jacobian. It costs one elementwise product, one reduction and one more elementwise product. Worked example with `x = [1, 2, 3]` and `g = [1, 0, 0]`:

```text
y          = [0.0900, 0.2447, 0.6652]
sum(g * y) = 0.0900
g - 0.0900 = [0.9100, -0.0900, -0.0900]
x.grad     = y * (g - 0.0900) = [0.0819, -0.0220, -0.0599]
```

The gradient sums to zero, as it must: adding a constant to every logit does not change a softmax, so moving all logits together cannot change the loss.

For $y = \log \mathrm{softmax}(x)$, $y_i = x_i - \ln \sum_k e^{x_k}$, so $\partial y_i / \partial x_j = \delta_{ij} - \mathrm{softmax}_j$ and:

$$
\frac{\partial L}{\partial x_j} = g_j - \mathrm{softmax}_j \sum_i g_i
$$

The code is `g - exp(y) * sum(g, {dim}, true)`. `exp(y)` recovers the softmax from the saved output, so the backward does not need the input.

### Cross entropy

`cross_entropy_losses(logits, targets)` in `src/ops/nn.cpp` computes per-row losses $-\ln \mathrm{softmax}(x)_t$, where $t$ is the target class of the row, in one fused kernel, and records one `CrossEntropyBackward` node. Combining the log-softmax derivative above with $g = -\mathrm{onehot}(t)$ gives the gradient with respect to one row's logits:

$$
\frac{\partial \ell}{\partial x_j} = \mathrm{softmax}(x)_j - [j = t]
$$

scaled by the upstream gradient of that row. `kael::detail::cross_entropy_gradient` computes this directly in one pass per row, without materializing a one-hot tensor.

From `src/ops/nn.cpp`:

```cpp
softmax_row(in + row * row_size, result, row_size);
for (std::int64_t i = 0; i < row_size; ++i) {
    result[i] *= weight[row];
}
result[target[row]] -= weight[row];
```

Here `weight[row]` is the upstream gradient for that row's loss. When the total loss is the mean over $N$ rows, it is $1/N$ for every row.

Worked example from `NnLoss.CrossEntropyWorkedExample` in `tests/test_nn_layers.cpp`, logits `[2, 1, 0.1]`, target 0:

```text
softmax   = [0.6590, 0.2424, 0.0986]
loss      = -log(0.6590) = 0.4170
gradient  = [0.6590 - 1, 0.2424, 0.0986] = [-0.3410, 0.2424, 0.0986]
```

Only `logits` gets an edge; `targets` are integer class indices and are saved for the formula, not differentiated. The fused node records one node and saves only the logits and targets. Composing `log_softmax`, `gather` and `mean` would record three nodes and save an extra tensor of the logits' size.

### Layer normalization core

`standardize(x, normalized_dims, eps)` computes $y = (x - \mu) / \sigma$ over the trailing `normalized_dims` dimensions, where $\mu$ is the mean, $\sigma = \sqrt{\mathrm{var} + \epsilon}$, and the variance uses no correction. Its backward is the classic layer norm formula, where the means run over the normalized dimensions:

$$
\frac{\partial L}{\partial x} = \frac{1}{\sigma} \left( g - \mathrm{mean}(g) - y \odot \mathrm{mean}(g \odot y) \right)
$$

The three terms have a reading. $g$ is the direct path through $x$. $\mathrm{mean}(g)$ removes the part that flows through $\mu$: shifting every element by the same amount does not change $y$. $y \odot \mathrm{mean}(g \odot y)$ removes the part that flows through $\sigma$: scaling the deviations does not change $y$ either.

From `src/autograd/functions_reduce.cpp`:

```cpp
const Tensor scale = kael::sqrt(kael::var(saved.get(), dims, true, 0) + eps);
return (g - kael::mean(g, dims, true) -
        y.get() * kael::mean(g * y.get(), dims, true)) /
       scale;
```

The closure saves both the input and the output. It recomputes $\sigma$ from the saved input at backward time instead of saving it, trading a small reduction for one less saved tensor. `LayerNorm` in [07. Neural networks](@/book/07-neural-networks.md) multiplies the result by `weight` and adds `bias`; those get their gradients from the ordinary `MulBackward` and `AddBackward` nodes.

### Views and indexing

View formulas reshape the gradient back to the input's geometry. Indexing formulas scatter the gradient back to the positions the forward pass read.

| Op | Backward |
|----|----------|
| `view`, `reshape`, `squeeze()` | `g.reshape(input_shape)` |
| `transpose(d0, d1)` | `g.transpose(d0, d1)` |
| `permute(dims)` | `g.permute(inverse(dims))` |
| `unsqueeze(d)` / `squeeze(d)` | `g.squeeze(d)` / `g.unsqueeze(d)` |
| `expand` | `sum_to_shape(g, input_shape)` |
| `slice`, `select` | zeros of the input shape, copy `g` into the slice |
| `cat` | `g.narrow(dim, start, length)` per input |
| `index_select` | zeros, then `index_add_(dim, indices, g)` |
| `gather` | zeros, then `scatter_add_(dim, indices, g)` |
| `where` | `where(mask, g, 0)` and `where(mask, 0, g)`, both reduced with `sum_to_shape` |
| `masked_fill` | `masked_fill(g, mask, 0)`, reduced with `sum_to_shape` |
| `to(dtype)`, `to(device)` | `g.to(input dtype)`, `g.to(input device)` |

Each row follows from asking where each output element came from.

- A view moves no data, so its backward only has to undo the change of geometry. A transpose is its own inverse. A permutation's inverse is computed once at record time: for `permute({2, 0, 1})` the inverse is `{1, 2, 0}`.
- `expand` is broadcasting made explicit, so its backward is `sum_to_shape`.
- `slice` and `select` read a subset. Positions outside the subset did not influence the output, so they get zero. For `x` of shape `[4]` and `x.slice(0, 1, 3)` with `g = [7, 9]`, the gradient is `[0, 7, 9, 0]`.
- `cat` places each input in a contiguous range of the output, so each input's gradient is the matching range of `g`. Splitting a tensor and concatenating the pieces in a different order, as `AutogradFormulas.SplitAccumulatesPieces` does, sends each piece's gradient back to the right range of the original through `SliceBackward` nodes.
- `masked_fill` replaced masked positions with a constant, so those positions get zero gradient.

`index_select` and `gather` must add, not assign: an index that appears twice receives both gradients. `AutogradFormulas.IndexSelectAddsRepeatedIndices` selects indices `[2, 0, 2]` from `x = [1, 2, 3]`, sums the result, and checks that `x.grad = [1, 0, 2]`: index 2 was read twice, index 1 never. This is what makes `Embedding`, which is built on `index_select`, correct when a batch repeats a token.

## Gradcheck

Analytic formulas often go subtly wrong: a missing transpose, a sign, a factor of two. The shape check in the engine catches some of these, but not a wrong value of the right shape. `gradcheck` compares the analytic gradient against central finite differences.

From `include/kael/autograd/gradcheck.h`:

```cpp
struct GradcheckOptions {
    double epsilon = 1e-6;
    double absolute_tolerance = 1e-5;
    double relative_tolerance = 1e-3;
    std::uint64_t projection_seed = 0;
};

struct GradcheckResult {
    bool passed = true;
    std::string report;
};

using GradcheckFunction = std::function<Tensor(const std::vector<Tensor>&)>;

[[nodiscard]] GradcheckResult gradcheck(const GradcheckFunction& function,
                                        const std::vector<Tensor>& inputs,
                                        const GradcheckOptions& options = {});
```

### Random projection

The function may return a tensor of any shape. Checking the full Jacobian would need one backward per output element. Instead, `gradcheck` draws a fixed random tensor $v$ of the output's shape, with standard normal entries from `std::mt19937_64` seeded by `projection_seed`, and checks the scalar

$$
L(x) = \sum_i f(x)_i v_i
$$

One backward gives $\partial L / \partial x = v^\top J$ for every input element at once. A wrong Jacobian entry changes $v^\top J$ with probability one for a random $v$, so the projection loses no detection power in practice. PyTorch's `torch.autograd.gradcheck` checks the full Jacobian by default and offers a similar projection as `fast_mode=True`.

Before the analytic pass, `gradcheck` clears each input's gradient with `zero_grad()`, so earlier backward calls do not pollute the comparison.

### Central differences

The numeric side perturbs one element at a time in place, evaluates $L$ under `NoGradGuard` at $x + \epsilon$ and $x - \epsilon$, and restores the value.

From `src/autograd/gradcheck.cpp`:

```cpp
double numeric_derivative(const GradcheckFunction& function, const std::vector<Tensor>& inputs,
                          double& element, const Tensor& weights, double epsilon) {
    const double original = element;
    element = original + epsilon;
    const double above = projected_value(function, inputs, weights);
    element = original - epsilon;
    const double below = projected_value(function, inputs, weights);
    element = original;
    return (above - below) / (2 * epsilon);
}
```

The estimate is

$$
\frac{\partial L}{\partial x_k} \approx \frac{L(x + \epsilon e_k) - L(x - \epsilon e_k)}{2 \epsilon}
$$

where $e_k$ is the unit vector for element $k$. A Taylor expansion shows why the central form is used: the even-order terms cancel, so the error is $O(\epsilon^2)$, against $O(\epsilon)$ for the forward difference $(L(x + \epsilon e_k) - L(x)) / \epsilon$. A small example: for $f(x) = x^3$ at $x = 1$ with $\epsilon = 0.1$, the forward difference gives $(1.331 - 1) / 0.1 = 3.31$, and the central difference gives $(1.331 - 0.729) / 0.2 = 3.01$. The true value is $3$.

A pair passes when $|a - n| \le \mathrm{atol} + \mathrm{rtol} \cdot |n|$, with $a$ the analytic value, $n$ the numeric value, and the tolerances from `GradcheckOptions`. Failures are collected into `GradcheckResult::report`, one line per element, in the form `input <i> element [<index>]: analytic <a> != numeric <n>`. If an input received no gradient at all, the analytic side is taken as 0.

### Input requirements

Inputs must be `F64`, require grad and be leaves. Each requirement has a reason.

- `F64`: `F32` has about 7 significant decimal digits. With $\epsilon = 10^{-6}$, the difference $f(x + \epsilon) - f(x - \epsilon)$ is about $10^{-6}$ times smaller than $f$ itself, so it would lose six of the seven digits to rounding. `F64` has about 16 digits and keeps ten.
- Leaves that require grad: the checker reads `input.grad()` and writes elements through `input.at<double>(index)`. An intermediate has no `grad`, and writing into it would not change the function's real input.

`Gradcheck.RequiresF64Input` and `Gradcheck.RequiresInputWithGrad` check the error paths, and `Gradcheck.RestoresInputValues` checks that every perturbed element is put back.

`Gradcheck.DetectsAWrongGradient` proves the checker can fail. The function returns `x * 2` when grad mode is on, so backward sees slope 2, and `x * 3` under `NoGradGuard`, so finite differences see slope 3. The report must contain `input 0 element [0]`. A test framework that never fails proves nothing, so this test is as important as the formula tests.

## Edge cases and pitfalls

- **Non-scalar roots.** `backward()` without a gradient on a tensor with more than one element throws `grad can be implicitly created only for scalar outputs, got shape [2]`. Pass a gradient of the root's shape, or reduce first. An explicit gradient of the wrong shape throws `gradient shape ... does not match output shape ...`.
- **Roots without history.** `backward()` on a tensor that does not require grad throws `backward needs a tensor that requires grad`. The usual cause is a forward pass that ran under `NoGradGuard`, or a loss computed from inputs where no parameter had `requires_grad_()`.
- **Second backward.** Without `retain_graph = true`, nodes release their closures after the first pass. A second pass throws `Trying to backward through the graph a second time (SumBackward); specify retain_graph=true` instead of silently computing with freed data (`AutogradEngine.SecondBackwardWithoutRetainThrows`). With `retain_graph`, leaves accumulate both passes (`AutogradEngine.RetainGraphAccumulatesInLeaves`).
- **Stale saved tensors.** Any in-place write after the forward pass to a tensor a node saved, through any view or `detach()` alias, throws at backward time. A common way to hit this is calling `optimizer.step()` between a forward pass and its `backward()`. The optimizer updates parameters in place, and a second linear layer saved its weight for the input gradient.
- **In-place on graph tensors.** `add_` and the other in-place operations on a leaf that requires grad, or on any tensor with history, throw while grad mode is on. Wrap parameter updates in `NoGradGuard`.
- **Gradient accumulation.** Leaf gradients add up across calls. Call `zero_grad()` on each parameter, or `Module::zero_grad` or `Optimizer::zero_grad`, before each step.
- **Detached-in-place tensors inside a live graph.** `detach_()` removes the tensor's metadata but not the graph's references to it. The engine skips the gradient instead of crashing.
- **Dtype and shape mismatches from a formula.** `check_gradient` rejects them with the node name and input index. `AutogradEngine.InvalidGradientShapeNamesTheNode` injects a `BadNode` that returns a `[2, 3]` gradient for a `[3]` input to prove the message.
- **Deep graphs.** Traversal, dependency counting and destruction are iterative. Recursion depth does not grow with graph depth.
- **Integer outputs.** Operations with integral or boolean outputs never record, even if an input requires grad. A mask computed with `x > 0` is a constant for autograd.
- **Higher-order gradients.** Not supported. The engine runs under `NoGradGuard`, so gradients have no history.
- **Non-differentiable points.** `relu`, `abs`, `clamp`, `max` and `maximum` use fixed conventions at kinks. A finite-difference check that straddles a kink disagrees with them, which is why the tests keep inputs away from those points.
- **Debug hooks.** Setting `KAEL_TRACE_BACKWARD` in the environment prints each node, its gradient shape and its L2 norm to stderr. `set_backward_trace` redirects the trace to any stream. `set_anomaly_detection(true)` from `kael/debug.h` makes the engine check every returned gradient for NaN and throw `Function <name> returned NaN in its gradient`. Both are covered in [13. Serving and tooling](@/book/13-serving-and-tooling.md).

> [!WARNING]
> Forgetting `zero_grad()` produces no error. The gradients of the second step include those of the first, the effective learning rate grows with every step, and training diverges or stalls without an obvious cause.

## How it is tested

| Test file | Suites | What it proves |
|-----------|--------|----------------|
| `experiments/micrograd.cpp` (CTest `micrograd`) | four boolean checks | The scalar engine matches a hand derivation, accumulates, waits on diamonds, and reuses the output for `exp` |
| `tests/test_autograd_fields.cpp` | `AutogradFields` | Leaf defaults, `requires_grad_` rules, only floating point can require grad, views do not inherit `requires_grad` without history, storage version starts at 0 and is shared by views, every in-place op bumps it, in-place grad rules |
| `tests/test_autograd_grad_mode.cpp` | `GradMode` | Default on, guards restore on scope exit, nesting, exceptions |
| `tests/test_autograd_engine.cpp` | `AutogradEngine`, `AutogradDetach`, `AutogradVersion` | Recording, `NoGradGuard` records nothing, accumulation (repeated input, triple add, diamond), seeding rules, leaf as root, `retain_graph`, `zero_grad`, gradient copy semantics, invalid gradients, no leak through saved outputs, 10000-node backward and 50000-node destruction, `retain_grad`, `detach`, `detach_` regression tests, version errors |
| `tests/test_autograd_sum_to_shape.cpp` | `SumToShape` | Leading and stretched dimensions, identity, scalar target, too few dimensions |
| `tests/test_autograd_gradcheck.cpp` | `Gradcheck`, `AutogradFormulas` | The checker detects wrong gradients, validates inputs and restores them; every formula passes gradcheck on seeded random `F64` inputs (`manual_seed(1234)`), plus hand examples for `mean`, ties in `max`, zeros in `prod`, repeated indices, mixed precision and non-differentiable points |
| `tests/test_autograd_mlp.cpp` | `AutogradGolden` | A two-layer tanh perceptron with log-softmax and gather matches PyTorch's loss to `1e-12` and the gradients of `w1`, `b1`, `w2`, `b2` |
| `tests/test_nn_layers.cpp` | `NnLoss` | Includes the cross entropy worked example above |

The tests sit at three levels, and each catches a different kind of error.

1. **Engine tests** use the simplest formulas (`add`, `mul`, `sum`) and check the mechanics: ordering, accumulation, memory and errors. A failure here is a bug in the engine, not in a formula.
2. **Formula tests** run `gradcheck` on each operation. They prove that each formula matches the forward function's numeric derivative, independently of any external reference.
3. **The golden test** replays a full model against PyTorch. It catches errors that both the formula and its finite difference would share, such as a forward function that disagrees with PyTorch's definition.

The golden fixture comes from `write_mlp_fixture` in `scripts/generate/golden.py`. It seeds PyTorch with `20260`, builds a 6x4 input, a 5-unit tanh hidden layer and 3 classes in `float64`, runs `loss.backward()` and saves inputs, targets, loss, parameters and gradients as `.npy` files under `tests/golden/mlp/`. The C++ test loads them with `kael::test::read_npy` and replays the same computation. The loss must match to `1e-12`, and each gradient is compared with `allclose` at a relative tolerance of `1e-6` and an absolute tolerance of `1e-12`.

The `AutogradFormulas` fixture uses helpers that keep inputs in the differentiable region: `positive_input` for `log`, `sqrt` and `pow`, which draws from `[0.5, 2.5)`; and `away_from_zero` for `abs`, `relu` and `prod`, which pushes every value at least `0.2` away from zero, so a finite difference never straddles a kink.

Run the autograd tests from a configured build directory:

```bash
ctest --test-dir build -R "Autograd|GradMode|Gradcheck|SumToShape|micrograd" --output-on-failure
```

## Build it yourself

1. Write the scalar engine: `Node` with `data`, `grad`, `parents`, `propagate`; `Value` with `+`, `*`, `tanh`, `exp`; DFS `backward()`. Checkpoint: the four checks in `experiments/micrograd.cpp` print `micrograd: gradients match`.
2. Add `AutogradMeta` and the `std::unique_ptr<autograd::AutogradMeta> autograd` field on `TensorImpl`. Implement `requires_grad`, `requires_grad_`, `is_leaf`, `grad`, `zero_grad`, `version`. Checkpoint: `AutogradFields.*` pass.
3. Implement `GradMode` with a `thread_local` flag and the guard classes. Checkpoint: `GradMode.*` pass.
4. Implement `Node`, `NodeInput`, `FunctionNode`, `AccumulateGrad`, `gradient_edge`, `make_input`. Start with a recursive destructor; the default one is fine for now.
5. Implement `should_record`, `set_history` and `SavedTensor` without the version check. Add `record_add` and `record_mul` and call them from `add` and `mul`, computing the result under `without_grad`. Checkpoint: `AutogradEngine.ResultOfOpHasHistory` and `AutogradEngine.OpWithoutGradInputsHasNoHistory`.
6. Implement the engine: seed, `count_dependencies`, the ready stack, gradient buffers, `check_gradient`, `release_variables`. Checkpoint: `RepeatedInputAccumulates`, `TripleAddAccumulates`, `DiamondWaitsForBothPaths`, `SecondBackwardWithoutRetainThrows`. Trace the worked example of this chapter by hand and compare it with `KAEL_TRACE_BACKWARD`.
7. Implement `sum_to_shape` and route binary formulas through `matching_input`. Checkpoint: `SumToShape.*` and `AutogradFormulas.AddBroadcast`.
8. Write `gradcheck`. Checkpoint: `Gradcheck.*` pass, including `DetectsAWrongGradient`.
9. Add formulas file by file: math, reduce, view, indexing, linalg. Add one `AutogradFormulas` test per op before its formula. Checkpoint: the full `AutogradFormulas` suite passes.
10. Add the version check in `SavedTensor::get` and `check_grad_rules` in the in-place ops. Checkpoint: `AutogradVersion.*` and `AutogradFields.InplaceOnLeafRequiringGradThrows`.
11. Add `retain_grad`, `detach`, `detach_` and the null check in `accumulate`. Checkpoint: `AutogradEngine.RetainGrad*` and `AutogradDetach.*`.
12. Replace the recursive node destructor with the iterative one. Checkpoint: `DeepChainDestructionWithoutBackward` passes; with the recursive version it crashes.
13. Generate the MLP fixture with `scripts/generate/golden.py` and replay it. Checkpoint: `AutogradGolden.TwoLayerPerceptronMatchesPyTorch`.

## Check your understanding

<details>
<summary>Why does reverse mode, not forward mode, suit training?</summary>

Training differentiates one scalar loss with respect to millions of parameters. Forward mode computes one column of the Jacobian per pass, so it needs one pass per parameter. Reverse mode computes one row per pass, and a scalar loss has a single row, so one backward pass gives every parameter's gradient. The price is memory: reverse mode must keep the forward values its formulas need until the backward pass reaches them.

</details>

<details>
<summary>In the worked example, why does AccumulateGrad for x not run right after AddBackward?</summary>

Its dependency count is 2: one edge from `AddBackward` and one from `MulBackward`. After `AddBackward` delivers `[0.180707, 1]`, the count drops to 1 and the node waits. Only when `MulBackward` adds `[0.090353, -1]` does the count reach 0. The node then runs once with the sum `[0.271060, 0]`. Running early would store a partial gradient and lose the second contribution.

</details>

<details>
<summary>Why does SavedTensor store t.detach() instead of t?</summary>

Saving `t` itself would keep `t`'s autograd metadata alive through the node. When an operation saves its own output, as `exp` does, that creates a cycle: the output's metadata owns the node through `grad_fn`, and the node owns the output. Reference counting never frees a cycle. A detached view shares the buffer and the version counter, which is all the formula needs, and carries no metadata.

</details>

<details>
<summary>For a of shape [4, 1] and b of shape [3], what shapes do the gradients of a * b have, and how are they reduced?</summary>

The product broadcasts to `[4, 3]`, so both raw gradients arrive as `[4, 3]`. For `a`, the target `[4, 1]` has the same rank and dim 1 was stretched, so `sum_to_shape` sums dim 1 with `keepdim` and returns `[4, 1]`. For `b`, the target `[3]` has one fewer dimension, so `sum_to_shape` sums the leading dim 0 and returns `[3]`. `check_gradient` would reject either result if the reduction were missing.

</details>

<details>
<summary>Why does a second backward() through the same graph throw, and what does retain_graph change?</summary>

After a node runs, the engine calls `release_variables()`, which clears its gradient functions and the saved tensors they captured, and marks the node as released. That frees activation memory during the backward pass. A second pass reaches a released node and `FunctionNode::apply` throws instead of computing with missing data. `retain_graph = true` skips the release, so the closures survive and a second pass works; leaf gradients then accumulate both passes.

</details>

<details>
<summary>Why must gradcheck inputs be F64?</summary>

The central difference subtracts two nearly equal values, $f(x + \epsilon)$ and $f(x - \epsilon)$, with $\epsilon = 10^{-6}$. The difference is about a millionth of $f$. In `F32`, with about 7 significant digits, almost every digit of that difference is rounding noise, and the comparison with the analytic gradient would fail or pass at random. `F64` keeps about ten significant digits of the difference.

</details>

## Further reading

- Andrej Karpathy, [micrograd](https://github.com/karpathy/micrograd), and the lecture "The spelled-out intro to neural networks and backpropagation: building micrograd". The model for `experiments/micrograd.cpp`, explained step by step.
- Atilim Gunes Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, Jeffrey Mark Siskind, "Automatic Differentiation in Machine Learning: a Survey", JMLR 2018. A precise account of forward and reverse mode, and how they differ from symbolic and numerical differentiation.
- Andreas Griewank and Andrea Walther, "Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation". The reference text on the theory and complexity of automatic differentiation.
- Adam Paszke et al., "Automatic differentiation in PyTorch", NIPS 2017 Autodiff Workshop. The design of the define-by-run engine that kael follows.
- PyTorch documentation, [Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html). How PyTorch handles saved tensors, in-place operations, version counters and `no_grad`, the same problems this chapter solves in a smaller form.
- Edward Z. Yang, [PyTorch internals](http://blog.ezyang.com/2019/05/pytorch-internals/). A tour of the tensor and autograd layers of a production framework, useful to compare with kael's design.
- kael source, [src/autograd](https://github.com/JetBerri/kael.cpp/tree/main/src/autograd) and [tests/test_autograd_engine.cpp](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_autograd_engine.cpp). The engine is under 200 lines; the engine tests read as a specification of its behavior.

Next: [07. Neural networks and training](@/book/07-neural-networks.md)
