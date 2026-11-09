+++
title = "Define-by-run autograd graph"
description = "Why kael records the autograd graph while operations run, how nodes and edges own each other, and which restrictions keep the backward pass correct."
weight = 5
slug = "dynamic-autograd-graph"

[extra]
kicker = "ADR 0005"
source = "docs/adr/0005-dynamic-autograd-graph.md"
+++

Reverse-mode automatic differentiation needs a record of the operations that produced a value, so that it can walk them backwards and apply the chain rule. How that record is built and stored decides whether models can use ordinary loops and branches, when memory is freed, and what in-place writes are allowed. kael builds the record while operations run, as PyTorch does. This record explains the alternatives, the structure kael uses, and the restrictions that come with it.

Three terms are used throughout. A **node** is one recorded operation in the backward graph. An **edge** connects a node to the node that produced one of its inputs. A **leaf** is a tensor created by the user rather than by a recorded operation, such as a model parameter; its gradient is stored in `grad()` after the backward pass.

## Status

Accepted in phase 7 (tag `chapter-07-autograd`, which matches [Autograd](@/book/06-autograd.md) in the book). Still in force at `v1.0.0` and at the current head of the repository.

## Context

Three designs for the record are common.

- **A static graph built before execution.** The program first describes the computation as a graph, then runs it, as in TensorFlow 1. The whole graph is known in advance, which allows global optimization. Control flow must be expressed with graph operations instead of `if` and `for`, so model code and host-language code are separate worlds.
- **A global tape.** Every operation appends an entry to one list, and the backward pass replays the list in reverse. The design is simple. Every tensor shares one tape, so freeing the record of one computation while keeping another, or differentiating two unrelated losses, needs explicit tape management.
- **A define-by-run graph.** Each result points to the node that produced it, and each node points to the nodes of its inputs. The graph is built as a side effect of running ordinary code and lives as long as some tensor references it. PyTorch uses this design.

kael targets readers who know PyTorch. Its models ([Neural networks and training](@/book/07-neural-networks.md), [Transformer](@/book/11-transformer.md)) are plain C++ with loops over layers, early returns and branches on configuration values. The record has to follow that control flow without a special API.

```text
static graph                     global tape                     define-by-run

graph = build(x, w)              tape: [mul, add, sum]           y = x * w      y.grad_fn -> MulBackward
session.run(graph, feed)         backward: replay tape            z = y + b      z.grad_fn -> AddBackward
                                 in reverse                       loss = z.sum() loss.grad_fn -> SumBackward
control flow: graph ops          one tape for everything          control flow: C++ if and for
```

## Decision

kael builds the graph while operations run. The design has six parts.

### Autograd state lives on the tensor

Each `TensorImpl` holds a `std::unique_ptr<autograd::AutogradMeta>`, defined in [`include/kael/autograd/meta.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/autograd/meta.h):

```cpp
struct AutogradMeta {
    bool requires_grad = false;
    std::optional<Tensor> grad;
    std::shared_ptr<Node> grad_fn;
    std::weak_ptr<Node> accumulator;
};
```

| Field | Meaning |
|-------|---------|
| `requires_grad` | the tensor takes part in differentiation |
| `grad` | the accumulated gradient, for leaves and for tensors that called `retain_grad` |
| `grad_fn` | the node that produced this tensor; empty for leaves |
| `accumulator` | a weak reference to the leaf's `AccumulateGrad` node, so every use of a leaf shares one node |

A tensor that never takes part in differentiation has a null `autograd` pointer. The cost of autograd for such tensors is one pointer per `TensorImpl`. Only floating-point tensors can require gradients: `requires_grad_` on an `int64` tensor throws `only floating point tensors can require gradients, got dtype int64`.

### Nodes compute vector-Jacobian products

A `Node` ([`include/kael/autograd/node.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/autograd/node.h)) holds its input edges and maps the gradient of its output to the gradients of its inputs:

```cpp
using Edge = std::shared_ptr<Node>;
using GradientList = std::vector<std::optional<Tensor>>;

struct NodeInput {
    Edge edge;
    Shape shape;
    DType dtype;
};

class Node {
public:
    virtual GradientList apply(const Tensor& gradient) = 0;
    // ...
private:
    std::vector<NodeInput> inputs_;
};
```

`apply` computes a **vector-Jacobian product**: the gradient of the output multiplied by the Jacobian of the operation, without ever forming the Jacobian. For `y = a * b`, the node returns `g * b` and `g * a`, two tensors of the input shapes, not a matrix of size `numel(y)` by `numel(a)`. The derivations are in [Autograd](@/book/06-autograd.md).

There are two kinds of node. `FunctionNode` holds a name such as `MulBackward` and one gradient function per input. `AccumulateGrad` sits at a leaf and adds the incoming gradient into the leaf's `grad`.

### History is recorded only when needed

Every differentiable operation calls a `record_*` function after computing its result. The recording checks two conditions in [`src/autograd/recording.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/autograd/recording.cpp):

- grad mode is on and the output dtype is floating point;
- at least one input requires gradients.

If both hold, `set_history` creates a `FunctionNode` with one edge per input and stores it as the output's `grad_fn`. An input that does not require gradients gets a null edge and no gradient function.

Grad mode is a `thread_local bool`, `true` by default. `NoGradGuard` sets it to `false` for the lifetime of the guard and restores the previous value in its destructor, also when an exception unwinds through it (`GradMode.ExceptionRestoresState`). Optimizers update parameters inside a `NoGradGuard`, and the Python `kael.no_grad()` context manager wraps the same flag.

### The engine orders nodes by dependency count

`autograd::backward` in [`src/autograd/engine.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/autograd/engine.cpp) runs in three steps:

1. **Seed.** A root with one element gets the gradient `ones_like(root)`. Any other root needs an explicit gradient of the same shape, or the call throws `grad can be implicitly created only for scalar outputs`.
2. **Count.** `count_dependencies` walks every node reachable from the root and counts, for each node, how many edges point to it.
3. **Process.** A node is ready when its count reaches zero, which means every contribution to its gradient has arrived. The engine applies the ready node, adds each returned gradient into the buffer of the target node, and decrements the target's count.

A value used several times therefore receives the sum of all its contributions before its node runs. The engine also checks that each returned gradient has the shape and dtype recorded in `NodeInput`, and names the node in the error if not.

The whole backward pass runs inside a `NoGradGuard`. Gradient functions are ordinary tensor operations, but they record no history, so the backward pass computes first-order gradients only.

### Saved tensors check the version counter

A gradient function that needs a forward value captures it as a `SavedTensor` ([`src/autograd/recording.h`](https://github.com/JetBerri/kael.cpp/blob/main/src/autograd/recording.h)). It stores a detached handle and the storage version at the time of the forward pass:

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
    // ...
};
```

Every in-place operation increments the version of its storage. If a saved tensor was modified between forward and backward, `get` throws instead of returning wrong values. The counter lives on the storage, so a write through any view or through `detach()` is caught ([ADR 0002](@/adr/0002-tensor-as-handle.md)).

### In-place operations do not rewrite history

With grad mode on, `check_grad_rules` in [`src/ops/inplace.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/ops/inplace.cpp) rejects three cases:

| Case | Message |
|------|---------|
| `self` is a leaf that requires grad | `a leaf tensor that requires grad is being used in the in-place operation add_` |
| `self` has gradient history | `add_ is not supported on a tensor with gradient history; use the out-of-place op` |
| the source requires grad | `add_ is not supported with a source that requires grad` |

PyTorch allows in-place operations on non-leaf tensors and rewrites the `grad_fn` of the modified tensor. kael refuses, so a recorded node never changes after it is created.

### A worked example

Trace the diamond from `AutogradEngine.DiamondWaitsForBothPaths`:

```cpp
const Tensor x = leaf({1, 2});   // float64, requires grad
const Tensor a = x * 2;
const Tensor b = x * 3;
(a + b).sum().backward();
```

The forward pass builds this graph. Arrows point from a node to the nodes of its inputs:

```text
SumBackward
    |
AddBackward
    |         \
MulBackward(a) MulBackward(b)       the constants 2 and 3 have null edges
    |         /
AccumulateGrad(x)
```

`count_dependencies` gives `AddBackward` 1, each `MulBackward` 1 and `AccumulateGrad` 2. Then:

```text
step  node             incoming gradient   result
1     SumBackward      1 (seed, shape [])  [1, 1] to AddBackward
2     AddBackward      [1, 1]              [1, 1] to each MulBackward
3     MulBackward(b)   [1, 1]              [3, 3] buffered for AccumulateGrad, count 2 -> 1
4     MulBackward(a)   [1, 1]              [2, 2] added to the buffer: [5, 5], count 1 -> 0
5     AccumulateGrad   [5, 5]              x.grad() = [5, 5]
```

`AccumulateGrad` runs once, with the complete sum. Had it run at step 3, `x.grad()` would hold `[3, 3]` and the second contribution would arrive at a node that had already finished. The ready list is a stack, so `MulBackward(b)`, pushed last, runs first. The order of steps 3 and 4 does not affect the result. The count guarantees that step 5 comes last.

## Alternatives considered

### A static graph

Build the graph first, with placeholders for inputs, and execute it separately.

- Gains: the whole graph is visible before execution, so it can be optimized, fused or compiled.
- Costs: control flow needs graph-level constructs. A loop over a variable number of layers or an early exit becomes a graph operation instead of C++. Debugging means inspecting a graph instead of stepping through code. Readers who know PyTorch would need a second mental model.
- Rejected because kael's models and examples are ordinary C++ and Python code.

### A global tape

Append every recorded operation to one thread-local list. `backward` walks the list in reverse.

- Gains: one data structure, no reference counting between nodes, and a simple reverse loop instead of dependency counting.
- Costs: the tape grows until someone clears it, so the memory of a computation is not tied to the tensors that use it. Two independent losses share one tape. A tensor that outlives one training step keeps the whole tape alive or is invalidated by a clear.
- Rejected because ownership through references makes memory follow the code without an explicit reset.

### Gradients of gradients

Record the backward pass itself as a graph, as PyTorch does with `create_graph=True`, so that higher-order derivatives are available.

- Gains: Hessian-vector products, gradient penalties and meta-learning.
- Costs: every gradient formula must be written in differentiable operations and tested twice. Saved tensors and released graphs interact in more ways.
- Rejected for scope. None of kael's models need second derivatives, and first-order backward is easier to explain and test.

## Consequences

### Models are ordinary code

Loops, early returns and data-dependent branches need no special API. The graph that `backward` sees is exactly the sequence of operations the program executed on this call.

### Memory follows references

A tensor's `grad_fn` holds its node, and each node holds the nodes of its inputs through `shared_ptr` edges. When the last tensor that references a graph is destroyed, the nodes go with it. No explicit clear is needed.

A long chain such as `y = y + 1` repeated 50,000 times would overflow the stack if each node destroyed the next one from its own destructor. `Node::~Node` moves the edges into a local list and releases them in a loop, so destruction is iterative (`AutogradEngine.DeepChainDestructionWithoutBackward`). The engine itself walks the graph with explicit stacks, not recursion, so `backward` on a deep chain is safe too (`AutogradEngine.DeepChainBackwardAndDestruction`).

### A graph is used once by default

After a node runs, the engine calls `release_variables`, which drops the gradient functions and the tensors they saved. A second `backward` through the same graph throws `Trying to backward through the graph a second time (...); specify retain_graph=true` (`AutogradEngine.SecondBackwardWithoutRetainThrows`). Passing `retain_graph = true` keeps the saved tensors, and a second call then accumulates into the leaves again (`AutogradEngine.RetainGraphAccumulatesInLeaves`).

### In-place operations are more restricted than in PyTorch

An in-place operation on a tensor that requires grad throws while grad mode is on. Optimizers and initializers write parameters inside `NoGradGuard`, which is the only common case that needs in-place writes on such tensors. The rules stay simple to state and test: a recorded node never changes, and a saved tensor is either unmodified or the backward pass throws.

### Higher-order derivatives are out of scope

`backward` runs in `NoGradGuard`, so the gradients it produces have no `grad_fn`. Differentiating a gradient is not possible.

### Every formula is checked numerically

`autograd::gradcheck` ([`include/kael/autograd/gradcheck.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/autograd/gradcheck.h)) compares the analytic gradient with central finite differences on float64 inputs. The `AutogradFormulas` tests in [`tests/test_autograd_gradcheck.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_autograd_gradcheck.cpp) run it for the recorded operations, and [Autograd](@/book/06-autograd.md) derives the formulas.

## Where it shows up in the code

| File | What to look for |
|------|------------------|
| [`include/kael/autograd/meta.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/autograd/meta.h) | `AutogradMeta` |
| [`include/kael/autograd/node.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/autograd/node.h) | `Node`, `NodeInput`, `AccumulateGrad`, `FunctionNode`, `gradient_edge` |
| [`include/kael/autograd/grad_mode.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/autograd/grad_mode.h) | `GradMode`, `NoGradGuard`, `EnableGradGuard`, `without_grad` |
| [`src/autograd/node.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/autograd/node.cpp) | the iterative `Node` destructor, `accumulate`, `release_variables` |
| [`src/autograd/engine.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/autograd/engine.cpp) | `seed_gradient`, `count_dependencies`, `check_gradient`, `backward` |
| [`src/autograd/recording.h`](https://github.com/JetBerri/kael.cpp/blob/main/src/autograd/recording.h), [`src/autograd/recording.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/autograd/recording.cpp) | `SavedTensor`, `should_record`, `set_history` |
| [`src/autograd/tensor_autograd.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/autograd/tensor_autograd.cpp) | `requires_grad_`, `retain_grad`, `detach_`, `Tensor::backward` |
| [`src/ops/inplace.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/ops/inplace.cpp) | `check_grad_rules` |
| [`tests/test_autograd_engine.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_autograd_engine.cpp) | the `AutogradEngine`, `AutogradDetach` and `AutogradVersion` tests |
| [`tests/test_autograd_gradcheck.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_autograd_gradcheck.cpp) | `Gradcheck` and `AutogradFormulas` |

Chapters that depend on this decision:

- [Autograd](@/book/06-autograd.md): the metadata, the engine, the formulas and gradcheck.
- [Neural networks and training](@/book/07-neural-networks.md): parameters as leaves, optimizers that update inside `NoGradGuard`.
- [Transformer](@/book/11-transformer.md): a model whose forward pass loops over layers in plain C++.
- [Python bindings](@/book/10-python-bindings.md): `grad`, `grad_fn`, `backward` and `kael.no_grad()` in Python.

## Check your understanding

<details>
<summary>x * 2 is computed from a leaf x that requires grad. Which node does the result point to, and which edges does that node hold?</summary>

The result's `grad_fn` is a `MulBackward` `FunctionNode`. It has two inputs: an edge to the `AccumulateGrad` node of `x`, and a null edge for the constant `2`, which became a 0-d tensor that does not require grad. Only the first input has a gradient function.

</details>

<details>
<summary>In (x * x).sum().backward(), why does x.grad() hold 2x and not x?</summary>

`MulBackward` has two edges, both to the `AccumulateGrad` node of `x`, so that node's dependency count is 2. It runs only after both contributions, `g * x` and `g * x`, have been added in its buffer. `AutogradEngine.RepeatedInputAccumulates` checks this.

</details>

<details>
<summary>What does a tensor that never takes part in differentiation pay for autograd?</summary>

One null `std::unique_ptr<AutogradMeta>` in its `TensorImpl`. The meta object is created only when `requires_grad_` is called or when an operation records history for the tensor.

</details>

<details>
<summary>y = kael::exp(x) with x requiring grad, then y.add_(1). What happens and why?</summary>

`add_` throws immediately: `y` has gradient history, and `check_grad_rules` rejects in-place operations on such tensors while grad mode is on. PyTorch would allow the call and rewrite `y.grad_fn`. kael keeps every recorded node unchanged instead.

</details>

<details>
<summary>Why does a 50,000-step chain of y = y + 1 not overflow the stack when y goes out of scope?</summary>

`Node::~Node` does not let each node destroy the next one recursively. It moves its edges into a local vector and, in a loop, moves the edges of every node whose last reference is in that vector. Destruction is iterative, so stack depth stays constant.

</details>

<details>
<summary>Why can kael not compute a gradient of a gradient?</summary>

`backward` runs inside a `NoGradGuard`, so the operations in gradient functions record no history and the resulting gradients have no `grad_fn`. Higher-order derivatives would need the backward pass to record its own graph, which kael leaves out of scope.

</details>
