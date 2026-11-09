+++
title = "Neural networks and training"
description = "Modules, layers, initialization, losses, optimizers, schedulers, data loading and an MNIST classifier that trains end to end on a CPU."
weight = 7
slug = "neural-networks"

[extra]
chapter = "07"
kicker = "Chapter 07"
builds = "Modules, losses, optimizers, data loading and an MNIST classifier"
source = "docs/07-neural-networks.md"
tag = "chapter-08-training"
math = true
+++

This chapter builds everything between a differentiable tensor and a trained classifier: modules with named parameters, layers, weight initialization, loss functions, optimizers, learning rate schedules, gradient clipping, datasets, a batching data loader and an MNIST reader. It ends with [`examples/mnist_mlp.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/examples/mnist_mlp.cpp), a two-layer perceptron that reaches about 97% test accuracy on MNIST in five epochs on a CPU. It maps to milestone v0.1 in the README: "train a model on MNIST on CPU, end to end".

| File | Role |
|------|------|
| [`include/kael/nn/module.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/nn/module.h), [`src/nn/module.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/nn/module.cpp) | `Module`, `UnaryModule`, parameter registry, `state_dict` |
| [`include/kael/nn/linear.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/nn/linear.h) | `Linear` |
| [`include/kael/nn/activation.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/nn/activation.h) | `ReLU`, `GELU`, `Sigmoid`, `Tanh` |
| [`include/kael/nn/dropout.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/nn/dropout.h) | `Dropout` |
| [`include/kael/nn/embedding.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/nn/embedding.h) | `Embedding` |
| [`include/kael/nn/layernorm.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/nn/layernorm.h) | `LayerNorm` |
| [`include/kael/nn/sequential.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/nn/sequential.h) | `Sequential`, `Flatten` |
| [`src/nn/layers.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/nn/layers.cpp) | Constructors and `forward` of every layer above |
| [`include/kael/nn/functional.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/nn/functional.h), [`src/nn/functional.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/nn/functional.cpp) | Stateless functions: `relu`, `gelu`, `sigmoid`, `tanh`, `softmax`, `log_softmax`, `dropout` |
| [`include/kael/nn/init.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/nn/init.h), [`src/nn/init.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/nn/init.cpp) | `fans_of`, uniform, normal, constant, Xavier and Kaiming initializers |
| [`include/kael/nn/loss.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/nn/loss.h), [`src/nn/loss.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/nn/loss.cpp) | `Reduction`, `mse_loss`, `nll_loss`, `cross_entropy`, `binary_cross_entropy_with_logits` |
| [`include/kael/optim/optimizer.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/optim/optimizer.h) | `Optimizer` base class |
| [`include/kael/optim/sgd.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/optim/sgd.h), [`include/kael/optim/adam.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/optim/adam.h) | `SGD`, `Adam`, `AdamW` |
| [`include/kael/optim/scheduler.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/optim/scheduler.h) | `StepLR`, `CosineAnnealingLR`, `LinearWarmup`, `WarmupCosineLR` |
| [`include/kael/optim/clip.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/optim/clip.h) | `clip_grad_norm_` |
| [`src/optim/optimizer.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/optim/optimizer.cpp) | All optimizer, scheduler and clipping code |
| [`include/kael/data/dataset.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/data/dataset.h), [`include/kael/data/dataloader.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/data/dataloader.h), [`src/data/dataset.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/data/dataset.cpp) | `Example`, `Dataset`, `TensorDataset`, `DataLoader` |
| [`include/kael/data/mnist.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/data/mnist.h), [`src/data/mnist.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/data/mnist.cpp) | IDX parser and `load_mnist` |
| [`examples/mnist_mlp.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/examples/mnist_mlp.cpp) | End-to-end MNIST training |
| [`tests/test_training.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_training.cpp) | Small end-to-end training runs that must converge |

## Prerequisites

- [05. Operations](@/book/05-operations.md): `matmul`, `softmax`, `log_softmax`, `gather`, `index_select`, the fused `cross_entropy_losses` and `standardize`.
- [06. Autograd](@/book/06-autograd.md): `requires_grad_`, `backward`, `zero_grad`, `NoGradGuard`, the in-place grad rules and the version check.

## Why this layer exists

After [chapter 06](@/book/06-autograd.md) kael can differentiate any scalar computed from tensors. That is the hard part of deep learning, and it is still not enough to train a model. Write a two-layer classifier with autograd alone and the training loop fills up with bookkeeping:

- Every weight and bias is a loose variable. You call `requires_grad_()` on each, pass each to the update code, clear each gradient, and save each to disk. Add a layer and you edit five places.
- Every weight needs an initial value. Draw them from `randn` with no scaling and a deep network outputs either huge numbers or zeros before it has seen a single example.
- The objective is a formula you write by hand each time, and the obvious way to write cross entropy overflows for large scores.
- The update rule `w -= lr * grad` sits inline in the loop. Switching to momentum or Adam means new per-weight state that you have to allocate and keep in sync with the weights.
- The data arrives as one big tensor. Shuffling it, cutting it into batches and handling the short last batch is more loop code.

Without this layer every model is a one-off script. With it, the same loop trains any model: the model is a tree of modules that can list its own parameters, the loss is a function call, the optimizer owns the update rule and its state, and the data loader yields batches.

The mental model to hold on to is a training step as a cycle with five stations:

```text
            +-------------------------------------------------------------+
            |                                                             |
            v                                                             |
  DataLoader yields          model(images)         cross_entropy(...)     |
  (images, labels)  ----->   forward pass   ----->  scalar loss            |
                                                         |                |
                                                         v                |
                                              optimizer.zero_grad()       |
                                              clear old gradients         |
                                                         |                |
                                                         v                |
                                              loss.backward()             |
                                              fill parameter.grad()       |
                                                         |                |
                                                         v                |
                                              optimizer.step()            |
                                              update weights in place     |
                                              under NoGradGuard  ---------+
```

Autograd already provides the gradients. This chapter adds the bookkeeping around them: a way to collect every trainable tensor of a model (`Module`), a set of layers with sensible defaults (`Linear`, `LayerNorm`, and others), scalar objectives (losses), update rules that read `grad()` and write parameters in place (optimizers), and a way to feed batches (`DataLoader`). If you know PyTorch, the names are deliberately the same: `torch.nn.Module`, `torch.nn.functional`, `torch.optim` and `torch.utils.data` map to `kael::nn`, `kael::nn::functional`, `kael::optim` and `kael::data`.

## Concepts

### Parameters, modules and the parameter tree

A **parameter** is a leaf tensor that requires grad and that a module registers by name. A **leaf** is a tensor that no recorded operation produced, so autograd accumulates a gradient into it instead of passing the gradient further back (see [06. Autograd](@/book/06-autograd.md)). A **module** is an object that owns a list of named parameters and a list of named child modules. Modules nest, so a model is a tree, and the full name of a parameter is the path from the root joined with dots.

The MNIST model of this chapter is a `Sequential` with four children. Only the two `Linear` layers own parameters:

```text
Sequential (root)
  "0" Flatten            no parameters
  "1" Linear(784, 128)   weight [128, 784]   bias [128]
  "2" ReLU               no parameters
  "3" Linear(128, 10)    weight [10, 128]    bias [10]

named_parameters():
  1.weight  [128, 784]   100352 elements
  1.bias    [128]           128 elements
  3.weight  [10, 128]      1280 elements
  3.bias    [10]             10 elements
                         ------
  num_parameters()        101770
```

That list is what an optimizer receives, what `zero_grad` clears, and what `state_dict` saves. The tree structure exists only to produce stable names; the optimizer sees a flat vector.

### Training mode and evaluation mode

Some layers behave differently while training. Dropout is the example in this chapter: it zeroes random elements during training and does nothing during evaluation. A module carries a `training` flag, and setting it on the root sets it on the whole tree. The flag changes behavior only. It does not disable gradient recording; that is the job of `NoGradGuard`. PyTorch makes the same split between `model.eval()` and `torch.no_grad()`, and confusing the two is a classic bug in both frameworks.

### What a loss is

A **loss** is a function that maps the model output and the target to a single number that is small when the prediction is good. Training minimizes it. The loss must be a scalar because `backward()` starts from one value with gradient 1. Losses here compute a per-element or per-sample value and then apply a **reduction**: the mean (default), the sum, or none.

### What an optimizer is

An **optimizer** reads the gradient of each parameter and writes a new value into the parameter. Plain gradient descent moves each weight a small step against its gradient:

$$
w \leftarrow w - \eta \, g
$$

where $\eta$ is the **learning rate** and $g = \partial L / \partial w$. Momentum and Adam keep extra state per parameter (a running average of past gradients) and use it to choose a better step. A **learning rate scheduler** changes $\eta$ over the course of training. **Gradient clipping** rescales gradients that are too large before the step.

### Batches, epochs and the data loader

A **batch** is a group of examples processed in one forward pass; the loss is averaged over it, so each step follows the average gradient of the batch instead of the gradient of one example. An **epoch** is one pass over the whole training set. A **data loader** turns a dataset into a sequence of batches, optionally in a new random order each epoch.

```text
dataset of 10 examples, batch_size 4, shuffle off

  indices   0 1 2 3 | 4 5 6 7 | 8 9
  batches   batch 0 | batch 1 | batch 2 (short: 2 examples)

  num_batches = ceil(10 / 4) = 3
  with drop_last: 10 / 4 = 2, the short batch is discarded
```

## Implementation

The implementation follows the order you would write it in: the module tree first, then the layers that live in it, the initializers those layers call, the losses, the optimizers, the data pipeline, and finally the example that uses all of them.

### Modules

#### Parameters and children

From [`include/kael/nn/module.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/nn/module.h):

```cpp
class Module {
public:
    Module() = default;
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;
    virtual ~Module() = default;

    Tensor register_parameter(std::string name, Tensor parameter);

    template <typename M>
    std::shared_ptr<M> register_module(std::string name, std::shared_ptr<M> m) {
        add_child(std::move(name), m);
        return m;
    }

    [[nodiscard]] std::vector<Tensor> parameters() const;
    [[nodiscard]] std::vector<NamedTensor> named_parameters() const;
    [[nodiscard]] std::int64_t num_parameters() const;
    ...
private:
    void add_child(std::string name, std::shared_ptr<Module> child);
    void collect_parameters(const std::string& prefix, std::vector<NamedTensor>& out) const;

    std::vector<NamedTensor> parameters_;
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> children_;
    bool training_ = true;
};
```

`NamedTensor` is `std::pair<std::string, Tensor>`. `register_parameter` calls `requires_grad_()` on the tensor, stores the handle and returns it. Because `Tensor` is a handle (see [ADR 0002](@/adr/0002-tensor-as-handle.md)), the copy in `parameters_` and the copy in the layer's own member (`weight_`) share one `TensorImpl`. An optimizer that updates the copy it received from `parameters()` updates the layer. This is the reason kael needs no `Parameter` class: in PyTorch, `nn.Parameter` is a tensor subclass that marks a tensor for registration through attribute assignment. C++ has no attribute hook, so registration is an explicit call, and the handle semantics already give the sharing PyTorch gets from Python references.

`register_module` is a template so it can return the child with its concrete type. A layer that writes `linear_ = register_module("proj", std::make_shared<Linear>(4, 2));` keeps a typed pointer to call `linear_->weight()`, while the base class stores it as `std::shared_ptr<Module>`. `add_child` rejects a null pointer with "cannot register a null module named ..." (`NnModule.RegisterNullModuleThrows`).

`collect_parameters` walks the tree depth first, own parameters before children, and builds dotted names:

```cpp
void Module::collect_parameters(const std::string& prefix, std::vector<NamedTensor>& out) const {
    for (const auto& [name, parameter] : parameters_) {
        out.emplace_back(prefix + name, parameter);
    }
    for (const auto& [name, child] : children_) {
        child->collect_parameters(prefix + name + ".", out);
    }
}
```

A `Linear` registered as `"0"` inside a `Sequential` contributes `0.weight` and `0.bias`, in registration order (`NnModule.NamedParametersUseDottedNamesInOrder`). Order matters: an optimizer stores its per-parameter state in vectors indexed by position, and a stable order makes runs reproducible.

`named_parameters()` then removes duplicates by `TensorImpl` address:

```cpp
std::vector<NamedTensor> Module::named_parameters() const {
    std::vector<NamedTensor> all;
    collect_parameters("", all);

    std::unordered_set<const TensorImpl*> seen;
    std::vector<NamedTensor> unique;
    for (auto& entry : all) {
        if (seen.insert(entry.second.impl().get()).second) {
            unique.push_back(std::move(entry));
        }
    }
    return unique;
}
```

A parameter shared between two places (tied weights, such as an embedding matrix reused as the output projection of a language model) appears once, under the first name that reaches it, and the optimizer updates it once (`NnModule.SharedParameterIsListedOnce`). Without deduplication, Adam would keep two sets of moments for one tensor and apply two updates per step. `parameters()` is `named_parameters()` without the names.

Modules are not copyable. Copying one would copy handles, and two "independent" models would share every weight. Deleting the copy constructor turns that bug into a compile error. To duplicate a model, build a second one and copy the weights with `state_dict` and `load_state_dict`.

`UnaryModule` adds the single-input interface used by every layer in this chapter:

```cpp
class UnaryModule : public Module {
public:
    virtual Tensor forward(const Tensor& input) = 0;
    Tensor operator()(const Tensor& input) { return forward(input); }
};
```

The split between `Module` and `UnaryModule` is deliberate. `Module` knows nothing about `forward`, because a transformer block or an attention layer takes several inputs ([11. Transformer](@/book/11-transformer.md)). `UnaryModule` is the common case of one tensor in and one out, which is what `Sequential` needs to chain layers.

#### Training mode

`train(bool)` sets `training_` and recurses into children; `eval()` is `train(false)`. `is_training()` reads the flag. Only `Dropout` reads the flag in this chapter (`NnModule.TrainingFlagPropagates`).

#### Other tree operations

| Member | Behavior |
|--------|----------|
| `zero_grad()` | `zero_grad()` on every unique parameter |
| `num_parameters()` | Sum of `numel()` over unique parameters |
| `to(Device)` | Under `NoGradGuard`, moves each parameter's storage with `move_to_` and moves its gradient too. The `TensorImpl` stays the same, so optimizers created earlier still see the parameters |
| `state_dict()` | `std::map<std::string, Tensor>` (aliased as `StateDict`) of `parameter.detach()` for every registered name, duplicates included |
| `load_state_dict(state)` | Validates every parameter name and shape first, then copies under `NoGradGuard` |

From [`src/nn/module.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/nn/module.cpp):

```cpp
void Module::to(Device device) {
    const autograd::NoGradGuard guard;
    for (Tensor& parameter : parameters()) {
        parameter.move_to_(device);
        if (const std::optional<Tensor> gradient = parameter.grad()) {
            parameter.impl()->autograd->grad = gradient->to(device);
        }
    }
}
```

`move_to_` replaces the storage inside the existing `TensorImpl` ([02. Memory model](@/book/02-memory-model.md)). The alternative, `parameter = parameter.to(device)`, would create a new `TensorImpl`, and every other handle (the layer's `weight_`, an optimizer's list) would keep pointing at the old CPU tensor. PyTorch's `Module.to` solves the same problem by swapping `param.data` in place.

`state_dict` entries are detached views: they share storage with the parameters (`NnModule.StateDictEntriesAreDetached`) but carry no autograd metadata, so saving them does not keep a graph alive. Because they share storage, a state dict is a snapshot only until the next optimizer step. Clone the entries if you need a frozen copy.

```cpp
void Module::load_state_dict(const StateDict& state) {
    std::vector<NamedTensor> all;
    collect_parameters("", all);
    for (const auto& [name, parameter] : all) {
        const auto found = state.find(name);
        KAEL_CHECK(found != state.end(), "state dict is missing parameter " << name);
        KAEL_CHECK_VALUE(found->second.shape() == parameter.shape(), ...);
    }
    const autograd::NoGradGuard guard;
    for (auto& [name, parameter] : all) {
        parameter.copy_(state.at(name));
    }
}
```

`load_state_dict` checks all names and shapes before it writes anything, so a mismatch leaves the model untouched (`NnModule.LoadStateDictRejectsWrongShape`, `NnModule.LoadStateDictRejectsMissingName`). A loader that copied as it validated would leave a half-loaded model after an error, with no way to tell which layers changed. Keys in the state dict that match no parameter are ignored, which is the behavior of `strict=False` in PyTorch for unexpected keys, combined with `strict=True` for missing ones.

kael has no buffer registry. Modules hold only parameters, and non-trainable configuration (the `eps` of `LayerNorm`, the `p` of `Dropout`) lives in ordinary members. No layer in this chapter has running statistics (PyTorch's `BatchNorm` stores its running mean in a buffer), so nothing else needs to be saved.

### Layers

#### Linear

`Linear(in_features, out_features, bias = true, dtype = DType::F32)` stores `weight` with shape `[out, in]` and an optional `bias` with shape `[out]`. The layout matches PyTorch, which makes loading PyTorch weights a direct copy. `weight()` and `bias()` return them; `bias()` is a `std::optional<Tensor>` and is empty for `Linear(4, 2, false)` (`NnLinear.WithoutBiasRegistersOneParameter`).

From [`src/nn/layers.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/nn/layers.cpp):

```cpp
Tensor Linear::forward(const Tensor& input) {
    const Tensor product = matmul(input, weight_.T());
    return bias_ ? product + *bias_ : product;
}
```

For an input `[N, in]`, `matmul` with `weight.T()` (`[in, out]`) gives `[N, out]` and the bias broadcasts over rows. `weight_.T()` is a view with swapped strides ([03. Views](@/book/03-views.md)), so the transpose copies nothing. Worked example with `in = 2`, `out = 3`:

```text
x      = [[1, 2]]                  shape [1, 2]
weight = [[1, 0],                  shape [3, 2]
          [0, 1],
          [1, 1]]
bias   = [0.5, 0.5, 0.5]           shape [3]

x @ weight.T = [[1*1 + 2*0, 1*0 + 2*1, 1*1 + 2*1]] = [[1, 2, 3]]
+ bias       = [[1.5, 2.5, 3.5]]   shape [1, 3]
```

`matmul` also accepts a 1-D input `[in]` and returns `[out]`, and a batched input `[B, N, in]` returns `[B, N, out]` (`NnLinear.OutputShapes`). The backward pass needs no layer-specific code: `MatmulBackward`, `TransposeBackward` and `AddBackward` produce the gradients, and `sum_to_shape` sums the bias gradient over the batch.

The default initialization reproduces PyTorch's:

```cpp
Tensor default_linear_weight(std::int64_t in_features, std::int64_t out_features, DType dtype) {
    Tensor weight = empty({out_features, in_features}, dtype);
    init::kaiming_uniform_(weight, std::sqrt(5.0));
    return weight;
}

Tensor default_linear_bias(std::int64_t in_features, std::int64_t out_features, DType dtype) {
    Tensor bias = empty({out_features}, dtype);
    const double limit = in_features > 0 ? 1.0 / std::sqrt(static_cast<double>(in_features)) : 0.0;
    return init::uniform_(bias, -limit, limit);
}
```

`kaiming_uniform_` with `a = sqrt(5)` has gain `sqrt(2 / (1 + 5)) = sqrt(1/3)` and bound `gain * sqrt(3 / fan_in) = 1 / sqrt(fan_in)`. For the MNIST layer `Linear(784, 128)` both weight and bias are uniform in `[-1/28, 1/28]`, about `[-0.0357, 0.0357]` (`NnLinear.DefaultInitBoundsMatchPyTorch`). The `a = sqrt(5)` constant has no deep meaning; it is the historical PyTorch default, kept so that a kael model and a PyTorch model with the same seed logic start from the same distribution. The `in_features > 0` guard avoids a division by zero for an empty layer. The [weight initialization](#weight-initialization) section below explains the formulas.

#### Activations

`ReLU`, `GELU(approximate = false)`, `Sigmoid` and `Tanh` are stateless `UnaryModule` wrappers around `kael::nn::functional`, which forwards to the ops of [05. Operations](@/book/05-operations.md):

```cpp
class GELU : public UnaryModule {
public:
    explicit GELU(bool approximate = false) : approximate_(approximate) {}
    Tensor forward(const Tensor& input) override { return functional::gelu(input, approximate_); }

private:
    bool approximate_;
};
```

They exist as modules so they can sit inside a `Sequential`. `GELU(true)` selects the tanh approximation used by GPT-2. The functional forms (`functional::relu(x)` and the others) are for code that writes `forward` by hand, as PyTorch's `F.relu` is. `NnActivation.ModulesCallTheirFunctions` checks that each module and its function agree.

#### Dropout

**Dropout** zeroes each element with probability `p` during training. It prevents units from relying on specific other units being present, which reduces overfitting. kael implements inverted dropout: the kept elements are divided by `1 - p` at training time, so the expected value of each output equals its input and evaluation needs no rescaling.

From [`src/nn/functional.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/nn/functional.cpp):

```cpp
Tensor dropout(const Tensor& input, double p, bool training) {
    KAEL_CHECK_VALUE(p >= 0.0 && p < 1.0, "dropout probability must be in [0, 1), got " << p);
    if (!training || p == 0.0) {
        return input;
    }
    const double keep = 1.0 - p;
    return input * bernoulli(input.shape(), keep, input.dtype()).to(input.device()) / keep;
}
```

The `Dropout` module stores `p` (default `0.5`), validates it in its constructor with the same check, and calls `functional::dropout(input, p_, is_training())`.

Worked example with `p = 0.2` on six ones:

```text
input   [1    1    1    1    1    1   ]
mask    [1    0    1    1    1    1   ]   bernoulli(keep = 0.8)
output  [1.25 0    1.25 1.25 1.25 1.25]   input * mask / 0.8
```

Over many elements about 80% of the outputs are `1.25` and the rest are `0`; the mean stays `1.0` (`NnDropout.TrainKeepsProportionAndExpectedValue` checks this on 100000 elements). The alternative, scaling by `keep` at evaluation time (the original formulation), would make evaluation depend on `p` and force every inference path to know about dropout. `p = 1` is rejected because `1 / keep` would divide by zero (`NnDropout.InvalidProbabilityThrows`). In eval mode and for `p = 0` the function returns the input handle itself (`NnDropout.EvalIsIdentity`, `NnDropout.ZeroProbabilityIsIdentity`).

The mask is drawn on the CPU from the global generator and moved to the input's device, so the same seed gives the same mask on any device. The gradient flows through `MulBackward` and `DivBackward`, so dropped elements get a zero gradient without extra code.

#### Embedding

`Embedding(count, dimension)` is a lookup table: `weight` has shape `[count, dimension]`, initialized from `N(0, 1)` like PyTorch. Row `i` is the learned vector for token `i`.

From [`src/nn/layers.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/nn/layers.cpp):

```cpp
Tensor Embedding::forward(const Tensor& indices) {
    Shape output_shape = indices.shape();
    output_shape.push_back(weight_.size(1));
    return index_select(weight_, 0, indices.flatten()).reshape(output_shape);
}
```

Indices of shape `[2, 3]` produce `[2, 3, dimension]`: flatten to 6 indices, select 6 rows, reshape. The lookup is mathematically a one-hot vector times the weight matrix, but `index_select` reads only the selected rows, so the cost is proportional to the number of tokens instead of the vocabulary size.

The backward of `index_select` is `index_add_` into a zero tensor, so a token that appears twice in a batch receives the sum of both gradients:

```text
weight [3, 2], indices [1, 1, 2], loss = sum(embedding(indices))

gradient of weight:
  row 0  [0, 0]     index 0 never used
  row 1  [2, 2]     index 1 used twice, 1 + 1
  row 2  [1, 1]     index 2 used once
```

That is `NnEmbedding.BackwardAddsRepeatedIndices`. An index outside `[0, count)` throws (`NnEmbedding.OutOfRangeIndexThrows`).

#### LayerNorm

`LayerNorm(normalized_shape, eps = 1e-5)` normalizes the trailing dimensions to zero mean and unit variance, then applies a learned per-element scale and shift:

$$
y = \frac{x - \operatorname{mean}(x)}{\sqrt{\operatorname{var}(x) + \epsilon}} \cdot \text{weight} + \text{bias}
$$

`mean` and the biased `var` (divide by `n`, not `n - 1`) run over the last `normalized_shape.size()` dimensions. `weight` starts at ones and `bias` at zeros, so a fresh layer is a pure standardization. The `eps` keeps the division finite for a constant row, whose variance is zero.

From [`src/nn/layers.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/nn/layers.cpp):

```cpp
Tensor LayerNorm::forward(const Tensor& input) {
    const std::size_t count = normalized_shape_.size();
    KAEL_CHECK_VALUE(static_cast<std::size_t>(input.dim()) >= count &&
                         Shape(input.shape().end() - static_cast<std::ptrdiff_t>(count),
                               input.shape().end()) == normalized_shape_,
                     "LayerNorm over " << to_string(normalized_shape_) << " got input shape "
                                       << to_string(input.shape()));
    return kael::standardize(input, static_cast<std::int64_t>(count), eps_) * weight_ + bias_;
}
```

The check compares the trailing dimensions of the input with `normalized_shape`, so `LayerNorm({4})` on a `[2, 3]` input throws with both shapes in the message (`NnLayerNorm.RejectsWrongTrailingShape`). The normalization itself is the fused `standardize` op with its own `StandardizeBackward` formula, derived in [06. Autograd](@/book/06-autograd.md). The scale and shift use ordinary broadcasting ops, so their gradients come from `MulBackward` and `AddBackward` (`NnLayerNorm.BackwardReachesParameters`).

Worked example over one row `x = [1, 2, 3, 4]`:

```text
mean     = (1 + 2 + 3 + 4) / 4                     = 2.5
deviation = [-1.5, -0.5, 0.5, 1.5]
var      = (2.25 + 0.25 + 0.25 + 2.25) / 4         = 1.25
sigma    = sqrt(1.25 + 1e-5)                        = 1.11804
output   = deviation / sigma = [-1.3416, -0.4472, 0.4472, 1.3416]
```

Why a fused op instead of composing `mean`, `sub`, `var` and `div`: the composition records five or six nodes and saves several intermediate tensors for backward. `standardize` is one node with a closed-form gradient, and on the CPU it runs as one pass per row.

#### Sequential and Flatten

`Sequential` registers each layer under its position (`"0"`, `"1"`, ...) and applies them in order. It has a constructor from an initializer list and an `add` member.

From [`src/nn/layers.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/nn/layers.cpp):

```cpp
void Sequential::add(std::shared_ptr<UnaryModule> layer) {
    layers_.push_back(register_module(std::to_string(layers_.size()), std::move(layer)));
}

Tensor Sequential::forward(const Tensor& input) {
    Tensor output = input;
    for (const auto& layer : layers_) {
        output = layer->forward(output);
    }
    return output;
}
```

`Sequential` keeps its own `layers_` vector of `UnaryModule` pointers next to the base class `children_`, because `children_` stores `Module` pointers that have no `forward`. Positional names match PyTorch's `nn.Sequential`, so a PyTorch state dict for the same stack has the same keys (`NnSequential.NamesLayersByPosition`, `NnSequential.AppliesLayersInOrder`).

`Flatten(start_dim = 1, end_dim = -1)` calls `input.flatten(start_dim, end_dim)`, which turns MNIST images `[N, 1, 28, 28]` into `[N, 784]` and keeps the batch dimension (`NnFlatten.KeepsBatchDimension`). The default `start_dim = 1` is the reason: flattening from 0 would merge the batch into the features.

### Weight initialization

The scale of the initial weights decides whether activations and gradients keep a stable magnitude through depth. Consider one linear layer `y = W x` with `fan_in` inputs, independent zero-mean weights of variance $\sigma_w^2$ and inputs of variance $\sigma_x^2$. Each output is a sum of `fan_in` products, so

$$
\operatorname{Var}(y) = \text{fan\_in} \cdot \sigma_w^2 \cdot \sigma_x^2 .
$$

The layer multiplies the variance by $c = \text{fan\_in} \cdot \sigma_w^2$, and a network of depth $L$ multiplies it by $c^L$:

```text
c = 2.0, 10 layers:  2^10   = 1024       activations explode
c = 0.5, 10 layers:  0.5^10 = 0.00098    activations vanish
c = 1.0, 10 layers:  1                   stable
```

The standard schemes choose $\sigma_w^2$ so that $c$ is close to 1. The same argument applied to the backward pass gives a condition on `fan_out` instead, because gradients flow through the transposed weight.

`fans_of(weight)` computes the two quantities the schemes need. For a weight of shape `[out, in, k1, k2, ...]` (a convolution kernel, for example) the receptive field is `k1 * k2 * ...`.

From [`src/nn/init.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/nn/init.cpp):

```cpp
Fans fans_of(const Tensor& weight) {
    KAEL_CHECK_VALUE(weight.dim() >= 2, "fans need a tensor with at least 2 dimensions, got shape "
                                            << to_string(weight.shape()));
    std::int64_t receptive_field = 1;
    for (std::int64_t dim = 2; dim < weight.dim(); ++dim) {
        receptive_field *= weight.size(dim);
    }
    return {weight.size(1) * receptive_field, weight.size(0) * receptive_field};
}
```

`Fans` is a struct with `in` and `out`. The product runs over the trailing dimensions instead of dividing `numel()` by `size(0) * size(1)`, because that division fails for a weight with an empty leading dimension. A `[0, 3, 2]` weight has fan in `6` and fan out `0` (`NnInit.FansOfAnEmptyWeightAreZero`). A 1-D tensor has no defined fans and throws (`NnInit.FansNeedTwoDimensions`).

| Function | Distribution | Parameter |
|----------|--------------|-----------|
| `uniform_(t, low, high)` | `U(low, high)` | |
| `normal_(t, mean, std)` | `N(mean, std^2)` | |
| `zeros_`, `ones_`, `constant_(t, value)` | constant | |
| `xavier_uniform_(t, gain = 1)` | `U(-b, b)` | `b = gain * sqrt(6 / (fan_in + fan_out))` |
| `kaiming_uniform_(t, a = 0)` | `U(-b, b)` | `b = gain(a) * sqrt(3 / fan_in)` |
| `kaiming_normal_(t, a = 0)` | `N(0, s^2)` | `s = gain(a) / sqrt(fan_in)` |

`gain(a) = sqrt(2 / (1 + a^2))` is the leaky ReLU gain (the function `leaky_relu_gain` in `src/nn/init.cpp`); `a = 0` gives `sqrt(2)`, the ReLU gain. A ReLU zeroes half of a symmetric input, which halves the variance, and the factor 2 in the gain restores it.

A uniform distribution on `[-b, b]` has variance `b^2 / 3`. Substituting the Kaiming bound gives `gain^2 * 3 / fan_in / 3 = gain^2 / fan_in`, the same variance as the normal variant. Xavier balances forward and backward variance with `2 / (fan_in + fan_out)`, a compromise between the two conditions above. Kaiming keeps the forward variance for ReLU networks, where half the units are zero.

Worked numbers from `NnInit.XavierAndKaimingLimits` for a weight `[100, 200]` (`fan_in = 200`, `fan_out = 100`):

```text
xavier bound  = sqrt(6 / (200 + 100))        = sqrt(0.02)  = 0.1414
kaiming bound = sqrt(2) * sqrt(3 / 200)      = 1.4142 * 0.1225 = 0.1732
```

Every initializer runs under `NoGradGuard` and writes in place with `copy_`, `zero_` or `fill_`:

```cpp
Tensor& uniform_(Tensor& t, double low, double high) {
    const autograd::NoGradGuard guard;
    return t.copy_(rand(t.shape(), t.dtype()) * (high - low) + low);
}
```

Without the guard, the in-place rules of [06. Autograd](@/book/06-autograd.md) would reject writes to a parameter that requires grad (`NnInit.InitWorksOnParameterRequiringGrad`). Writing in place also keeps the `TensorImpl`, so reinitializing a registered parameter updates the module. The random values come from `rand` and `randn`, so `manual_seed` makes initialization reproducible. The trailing underscore follows the PyTorch convention for functions that modify their argument (`torch.nn.init.kaiming_uniform_`).

### Loss functions

Every loss takes a `Reduction`, an enum with `Mean` (default), `Sum` and `None` (per-element or per-sample losses):

```cpp
Tensor reduce(const Tensor& losses, Reduction reduction) {
    switch (reduction) {
    case Reduction::Mean:
        return kael::mean(losses);
    case Reduction::Sum:
        return kael::sum(losses);
    case Reduction::None:
        return losses;
    }
    detail::unreachable();
}
```

The mean is the default because it makes the gradient scale independent of the batch size: doubling the batch with `Sum` doubles every gradient and changes the effective learning rate.

#### Mean squared error

`mse_loss(prediction, target)` is `reduce((prediction - target)^2)`. Its gradient with mean reduction is `2 (prediction - target) / n`; autograd derives it from `SubBackward`, `MulBackward` and `MeanBackward`. Worked example: prediction `[1, 3]`, target `[0, 0]`, loss `(1 + 9) / 2 = 5`, gradient `[2 * 1 / 2, 2 * 3 / 2] = [1, 3]`.

#### Negative log-likelihood and cross entropy

`nll_loss(log_probabilities, targets)` expects `[N, C]` log-probabilities and `[N]` class indices of dtype `I64`, picks `log_probabilities[i, targets[i]]` with `gather` and negates it:

```cpp
const Tensor picked = gather(log_probabilities, 1, targets.unsqueeze(1)).squeeze(1);
return reduce(-picked, reduction);
```

`cross_entropy(logits, targets)` is `nll_loss(log_softmax(logits, 1), targets)` mathematically, but it calls the fused `cross_entropy_losses` op. For one row with logits $x$ and target $t$:

$$
\text{loss} = -\log \operatorname{softmax}(x)_t = \log \sum_j e^{x_j} - x_t
$$

The fused op in [`src/ops/nn.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/ops/nn.cpp) computes this per row after subtracting the row maximum, so large logits do not overflow `exp`:

```cpp
template <typename T> T cross_entropy_row(const T* row, std::int64_t target, std::int64_t size) {
    const T largest = row_maximum(row, size);
    double total = 0.0;
    for (std::int64_t i = 0; i < size; ++i) {
        total += static_cast<double>(std::exp(row[i] - largest));
    }
    return std::log(static_cast<T>(total)) - (row[target] - largest);
}
```

It records a single `CrossEntropyBackward` node whose gradient is `softmax(x) - onehot(t)` per row (`CrossEntropyFusedGraph.RecordsItsOwnNode`). On devices other than the CPU it falls back to `-gather(log_softmax(logits, 1), ...)`. Rows run in parallel through `parallel_for` ([08. CPU performance](@/book/08-cpu-performance.md)).

Worked example (`NnLoss.CrossEntropyWorkedExample`), logits `[2, 1, 0.1]`, target 0:

```text
shift by max 2:        [0, -1, -1.9]
exp:                   [1, 0.3679, 0.1496]          sum 1.5175
logsumexp - x[t]:      log(1.5175) - 0 = 0.4170      loss

softmax:               [0.6590, 0.2424, 0.0986]
gradient = softmax - onehot(0):
                       [-0.3410, 0.2424, 0.0986]
```

The gradient reads directly: push the correct class up by its missing probability, push the others down by the probability they took. With uniform logits over `C` classes every softmax entry is `1/C` and the loss is `log(C)`, which is why a fresh classifier on 10 classes starts near `2.30` (`NnLoss.UniformPredictionGivesLogOfClasses`).

Both losses validate targets: shape `[N]`, dtype `I64`, every value in `[0, C)`. The error names the offending value ("target 10 is out of bounds for 10 classes"; `NnLoss.TargetsOutOfRangeThrowWithClearMessage`, `NnLoss.TargetsMustMatchBatch`). Without the check an out-of-range target would read outside the row. The check costs a `max` and a `min` over the targets, which is negligible next to the forward pass.

#### Binary cross entropy with logits

For a logit $x$ and a target $t$ in `[0, 1]`, binary cross entropy on $p = \sigma(x)$ is $-(t \log p + (1 - t) \log(1 - p))$. Computing `sigmoid` first and then `log` overflows or returns `log(0)` for large `|x|`: in float32 `sigmoid(20)` rounds to exactly `1`, and `log(1 - 1)` is `-inf`. Substituting $\log \sigma(x) = -\log(1 + e^{-x})$ and rearranging gives a form that never exponentiates a positive number:

$$
\text{loss} = \max(x, 0) - x t + \log(1 + e^{-|x|})
$$

From [`src/nn/loss.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/nn/loss.cpp):

```cpp
Tensor binary_cross_entropy_with_logits(const Tensor& logits, const Tensor& targets,
                                        Reduction reduction) {
    const Tensor losses =
        maximum(logits, 0) - logits * targets + log1p(kael::exp(-kael::abs(logits)));
    return reduce(losses, reduction);
}
```

`log1p(y)` computes `log(1 + y)` accurately for small `y`, which is the regime of `exp(-|x|)` for large `|x|`. Worked values with `Reduction::None`:

| `x` | `t` | loss |
|-----|-----|------|
| 0 | 1 | `log 2 = 0.6931` |
| 2 | 1 | `2 - 2 + log1p(e^-2) = 0.1269` |
| -2 | 0 | `0 - 0 + log1p(e^-2) = 0.1269` |
| 1000 | 1 | `0` |

The last row is `NnLoss.BinaryCrossEntropyIsStableForLargeLogits`. The gradient with respect to the logit is `sigmoid(x) - t`; at `x = 0`, `t = 1` that is `-0.5` (`NnLoss.BinaryCrossEntropyGradientAtZeroIsSigmoidMinusTarget`). kael has no `binary_cross_entropy` on probabilities; pass logits.

### Optimizers

#### The base class

From [`include/kael/optim/optimizer.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/optim/optimizer.h):

```cpp
class Optimizer {
public:
    Optimizer(std::vector<Tensor> parameters, double learning_rate);
    Optimizer(const Optimizer&) = delete;
    Optimizer& operator=(const Optimizer&) = delete;
    virtual ~Optimizer() = default;

    void step();
    void zero_grad();

    [[nodiscard]] double learning_rate() const noexcept { return learning_rate_; }
    void set_learning_rate(double learning_rate);

protected:
    [[nodiscard]] std::vector<Tensor>& parameters() noexcept { return parameters_; }

private:
    virtual void update() = 0;

    std::vector<Tensor> parameters_;
    double learning_rate_;
};
```

The constructor rejects a negative learning rate and any parameter that is not a leaf requiring grad (`Optimizer.InvalidConfigurationThrows`). A non-leaf would never receive a gradient, so accepting it would hide a wiring bug.

`step()` is a template method, the pattern where the base class fixes the skeleton and subclasses fill one step:

```cpp
void Optimizer::step() {
    const autograd::NoGradGuard guard;
    update();
}
```

The guard is what allows `parameter.sub_(...)` on a leaf that requires grad, and it keeps the update from recording a graph. Putting it in the base class means no subclass can forget it. The update is in place, so the parameter keeps its `TensorImpl` and storage, stays a leaf and stays registered in its module (`Optimizer.StepCreatesNoGraphAndKeepsTheSameTensor`). Parameters with no gradient are skipped (`Optimizer.ParametersWithoutGradAreSkipped`), which matters for layers a forward pass did not use. `Optimizer::zero_grad()` clears the gradient of each parameter it holds (`Optimizer.ZeroGradClearsGradients`); it is equivalent to `Module::zero_grad()` when the optimizer holds all the model's parameters.

kael has one learning rate per optimizer. PyTorch's parameter groups (a different learning rate for different layers) are not implemented; build two optimizers if you need that.

#### SGD with momentum

`SgdOptions` has `learning_rate` (default `1e-3`), `momentum` (default `0`), `weight_decay` (default `0`) and `nesterov` (default `false`). `SGD` takes the options explicitly: `SGD optimizer(model.parameters(), {.learning_rate = 0.1});`.

**Momentum** keeps a velocity per parameter, a running sum of past gradients that decays by `mu` each step. Directions where the gradient is consistent build up speed; directions where it oscillates cancel out. **Weight decay** adds `lambda * w` to the gradient, which pulls weights towards zero and acts as L2 regularization.

The update for gradient `g`, parameter `w`, momentum `mu`, weight decay `lambda` and learning rate `lr`:

```text
g = g + lambda * w                    (if weight_decay > 0)
v = g                                 (first step)
v = mu * v + g                        (later steps)
d = g + mu * v  if nesterov, else v
w = w - lr * d
```

From [`src/optim/optimizer.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/optim/optimizer.cpp):

```cpp
void SGD::update() {
    for (std::size_t i = 0; i < parameters().size(); ++i) {
        Tensor& parameter = parameters()[i];
        const std::optional<Tensor> stored = parameter.grad();
        if (!stored) {
            continue;
        }
        Tensor gradient =
            options_.weight_decay > 0.0 ? *stored + parameter * options_.weight_decay : *stored;
        if (options_.momentum > 0.0) {
            std::optional<Tensor>& velocity = velocities_[i];
            if (velocity) {
                velocity->mul_(options_.momentum).add_(gradient);
            } else {
                velocity = gradient.clone();
            }
            gradient = options_.nesterov ? gradient + *velocity * options_.momentum : *velocity;
        }
        parameter.sub_(gradient * learning_rate());
    }
}
```

`velocities_` is a vector of `std::optional<Tensor>` indexed like the parameters. The velocity is allocated lazily on the first step with a gradient, by cloning that gradient, so a parameter that never gets a gradient costs no memory. The clone matters: without it, the velocity and the gradient would share storage, and the next `zero_grad` or the `mul_` would corrupt one through the other.

This is PyTorch's formulation: the velocity accumulates raw gradients and the learning rate multiplies at the end, so changing the learning rate mid-training does not rescale the stored velocity. The textbook alternative, `v = mu * v + lr * g`, bakes the old learning rate into `v` and makes a scheduler's change take effect only gradually.

Worked example: `w = 1`, loss `2w` (so `g = 2` every step), `lr = 0.1`, `mu = 0.9`.

```text
step 1: v = 2                     w = 1    - 0.1*2    = 0.8
step 2: v = 0.9*2    + 2 = 3.8    w = 0.8  - 0.1*3.8  = 0.42
step 3: v = 0.9*3.8  + 2 = 5.42   w = 0.42 - 0.1*5.42 = -0.122
```

The velocity grows towards the fixed point of `v = mu * v + g`, which is `g / (1 - mu) = 20`, so momentum `0.9` amplifies a constant gradient up to ten times. Nesterov without positive momentum is rejected with "nesterov needs a positive momentum".

#### Adam and AdamW

**Adam** adapts the step size per element. It keeps two moving averages per parameter: the first moment `m` (mean of gradients) and the second moment `v` (mean of squared gradients). Dividing `m` by `sqrt(v)` normalizes the step: an element with large gradients gets a smaller learning rate, one with small gradients a larger one. Both averages start at zero, so early averages are biased towards zero; dividing by `1 - beta^t` corrects that.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\hat m_t &= m_t / (1 - \beta_1^t), \qquad \hat v_t = v_t / (1 - \beta_2^t) \\
w_t &= w_{t-1} - \eta \, \hat m_t / (\sqrt{\hat v_t} + \epsilon)
\end{aligned}
$$

From [`src/optim/optimizer.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/optim/optimizer.cpp):

```cpp
const auto step = static_cast<double>(++steps_[i]);
const double first_correction = 1.0 - std::pow(options_.beta1, step);
const double second_correction = 1.0 - std::pow(options_.beta2, step);
...
first.mul_(options_.beta1).add_(gradient * (1.0 - options_.beta1));
second.mul_(options_.beta2).add_(gradient * gradient * (1.0 - options_.beta2));
const Tensor corrected_first = first / first_correction;
const Tensor corrected_second = second / second_correction;
parameter.sub_(corrected_first / (kael::sqrt(corrected_second) + options_.epsilon) *
               learning_rate());
```

On the first step `m_hat = g` and `v_hat = g^2`, so each element moves by `lr * g / (|g| + eps)`, which is `lr` in the direction of `-sign(g)` regardless of the gradient's scale (`Optimizer.AdamFirstStepMovesByLearningRate`). This scale invariance is why Adam's default learning rate `1e-3` works across very different models, while SGD's learning rate must be tuned to the gradient magnitude.

Trace for `w = 1`, loss `w^2` (so `g = 2w`), `lr = 0.1`, default betas:

```text
t  g        m        v          m_hat    v_hat    w
1  2.0000   0.2000   0.004000   2.0000   4.0000   0.900000
2  1.8000   0.3600   0.007236   1.8947   3.6198   0.800412
3  1.6008   0.4841   0.009791   1.7863   3.2671   0.701586
```

Each step moves by about `0.1` even though the gradient shrinks, because `m_hat / sqrt(v_hat)` stays close to 1 while the gradient keeps its sign.

`AdamOptions` defaults: `learning_rate = 1e-3`, `beta1 = 0.9`, `beta2 = 0.999`, `epsilon = 1e-8`, `weight_decay = 0`. `Adam` and `AdamW` share `AdamBase` and differ only in `WeightDecayMode`:

```cpp
if (options_.weight_decay > 0.0) {
    if (mode_ == WeightDecayMode::Coupled) {
        gradient = gradient + parameter * options_.weight_decay;
    } else {
        parameter.mul_(1.0 - learning_rate() * options_.weight_decay);
    }
}
```

- **Coupled** decay (`Adam`) adds `lambda * w` to the gradient. The decay term then passes through the adaptive scaling, so weights with large gradient history decay less.
- **Decoupled** decay (`AdamW`) shrinks the weight directly by `lr * lambda` and leaves the moments untouched. `AdamWOptions` sets `weight_decay = 0.01` by default.

The difference is large. Take `w = 1`, a zero gradient, `lr = 1e-3`, `lambda = 0.01`, one step:

```text
AdamW:  w = 1 * (1 - 1e-3 * 0.01)                       = 0.99999
Adam:   g = 0 + 0.01 * 1 = 0.01
        m_hat = 0.01, v_hat = 0.0001
        w = 1 - 1e-3 * 0.01 / (0.01 + 1e-8)             = 0.999001
```

With coupled decay the adaptive normalization turns a tiny decay gradient into a full `lr`-sized step, a hundred times the decoupled one. Loshchilov and Hutter showed that coupling makes weight decay interact badly with the adaptive scaling, and AdamW is the standard choice for transformers.

`steps_` holds one step count per parameter, and a parameter's count increments only on the steps where it has a gradient, as in PyTorch. A parameter that receives its first gradient late still gets the bias correction of a first step, so it moves by `lr` (`Optimizer.AdamCorrectsTheFirstStepOfALateParameter`). A single shared counter would apply the corrections of a later step to moments that hold only one gradient, and shrink that first update. At a shared `t = 2` with the default betas, `m_hat = 0.1 g / 0.19 = 0.526 g` and `v_hat = 0.001 g^2 / 0.001999 = 0.500 g^2`, so the late parameter would move by `0.526 / 0.707 = 0.74` times `lr` instead of `lr`.

The moment buffers are allocated in the `AdamBase` constructor with `zeros_like(parameter)`, on the parameter's device at that moment. That has a consequence for the MNIST example, below.

#### Learning rate schedulers

A **scheduler** owns a reference to an optimizer and rewrites its learning rate. A large rate early makes fast progress; a small rate late lets the weights settle into a minimum instead of bouncing around it. The base class records the optimizer's learning rate at construction as `base_lr` and keeps a step counter.

From [`src/optim/optimizer.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/optim/optimizer.cpp):

```cpp
void LRScheduler::step() {
    ++step_count_;
    apply_schedule();
}

void LRScheduler::apply_schedule() {
    optimizer_.set_learning_rate(lr_at(step_count_));
}
```

Each subclass implements `lr_at(step)` as a pure function of the step and calls `apply_schedule()` at the end of its constructor, so the learning rate for step 0 is in place before the first `optimizer.step()`. `get_lr()` reads the current rate. Call `scheduler.step()` after each `optimizer.step()`. A closed-form `lr_at` (instead of multiplying the previous rate by a factor, which PyTorch does internally for some schedulers) means the rate at any step can be computed without replaying history, and rounding errors do not accumulate.

| Scheduler | `lr_at(step)` |
|-----------|---------------|
| `StepLR(step_size, gamma)` | `base * gamma^(floor(step / step_size))` |
| `CosineAnnealingLR(total, min_lr = 0)` | `min + (base - min) * (1 + cos(pi * min(step, total) / total)) / 2` |
| `LinearWarmup(warmup)` | `base * min(1, (step + 1) / warmup)` |
| `WarmupCosineLR(warmup, total, min_lr = 0)` | `base * (step + 1) / warmup` while `step < warmup`, then cosine from `base` to `min_lr` over the remaining `total - warmup` steps |

Worked values:

```text
StepLR, base 0.1, step_size 3, gamma 0.5, steps 0..9:
  0.1  0.1  0.1  0.05  0.05  0.05  0.025  0.025  0.025  0.0125

CosineAnnealingLR, base 0.001, total 10, after 5 steps:
  0.5 * 0.001 * (1 + cos(pi/2)) = 0.0005

LinearWarmup, base 0.1, warmup 4, steps 0..5:
  0.025, 0.05, 0.075, 0.1, 0.1, 0.1

WarmupCosineLR, base 0.1, warmup 2, total 6, steps 0..7:
  step   0      1     2     3        4      5        6    7
  lr     0.05   0.1   0.1   0.08536  0.05   0.01464  0    0
         warmup       cosine, progress (step - 2) / 4     clamped
```

**Warmup** exists because Adam's second-moment estimate is noisy in the first steps, and a full learning rate on a freshly initialized transformer can push it into a bad region it never leaves. The warmup schedules count `step + 1`, so the rate set at construction is `base / warmup` (`base` for a `WarmupCosineLR` without warmup), not `0`, and the first optimizer step already updates the parameters. The rate reaches `base` at step `warmup - 1`. The cosine schedules clamp the step at `total`, so the rate stays at `min_lr` afterwards. Invalid lengths (a non-positive step size or total, a warmup that does not fit in the total) throw (`Schedulers.InvalidLengthsThrow`).

#### Gradient clipping

`clip_grad_norm_(parameters, max_norm)` treats all gradients as one vector, computes its L2 norm, and scales every gradient in place when the norm exceeds `max_norm`. It protects training from a single batch with an exploding gradient, which would otherwise move the weights so far that the loss never recovers.

From [`src/optim/optimizer.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/optim/optimizer.cpp):

```cpp
double clip_grad_norm_(const std::vector<Tensor>& parameters, double max_norm) {
    const autograd::NoGradGuard guard;
    double squared_total = 0.0;
    for (const Tensor& parameter : parameters) {
        if (const auto gradient = parameter.grad()) {
            const double norm = kael::norm(*gradient).item();
            squared_total += norm * norm;
        }
    }
    const double total = std::sqrt(squared_total);
    KAEL_CHECK_VALUE(std::isfinite(total),
                     "the total gradient norm is " << total << ", so it cannot be clipped");
    const double coefficient = max_norm / (total + 1e-6);
    if (coefficient < 1.0) {
        for (const Tensor& parameter : parameters) {
            if (auto gradient = parameter.grad()) {
                gradient->mul_(coefficient);
            }
        }
    }
    return total;
}
```

`grad()` returns a handle to the stored gradient, so `mul_` on it updates the parameter's gradient. The function returns the norm before clipping, which is the value worth logging. The `1e-6` keeps the division finite when every gradient is zero.

Worked example (`ClipGradNorm.ScalesWhenAboveTheLimit`), two gradients `[3]` and `[4]`, `max_norm = 1`:

```text
total       = sqrt(3^2 + 4^2) = 5
coefficient = 1 / (5 + 1e-6)  = 0.2 (approximately)
gradients   = [0.6], [0.8]    new norm 1, same direction
returned    = 5
```

A gradient below the limit is left alone (`ClipGradNorm.LeavesSmallGradientsAlone`). Clipping preserves the direction of the full gradient, unlike per-element clamping, which would change `[3, 4]` clamped to `1` into `[1, 1]`, a different direction. Call it between `backward()` and `optimizer.step()`.

A NaN or infinite gradient makes the total norm non-finite, and the coefficient would then be 0 or NaN. Scaling by it would hide the problem or spread NaN to every gradient, so `clip_grad_norm_` throws `ValueError` instead (`ClipGradNorm.RejectsANonFiniteNorm`). The Python `kael.optim.clip_grad_norm_` raises Python's `ValueError` with the same message. PyTorch returns the non-finite norm by default and only raises when you pass `error_if_nonfinite=True`; kael chooses the loud behavior.

### Data

#### Dataset and TensorDataset

From [`include/kael/data/dataset.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/data/dataset.h):

```cpp
struct Example {
    Tensor input;
    Tensor target;
};

class Dataset {
public:
    [[nodiscard]] virtual std::int64_t size() const = 0;
    [[nodiscard]] virtual Example get(std::int64_t index) const = 0;
    [[nodiscard]] virtual Example get_batch(const Tensor& indices) const;
};
```

The default `get_batch` calls `get` for each index and `stack`s the results, which works for any dataset (`Dataset.DefaultBatchStacksExamples`):

```cpp
Example Dataset::get_batch(const Tensor& indices) const {
    std::vector<Tensor> inputs;
    std::vector<Tensor> targets;
    for (std::int64_t i = 0; i < indices.numel(); ++i) {
        Example example = get(indices.at<std::int64_t>({i}));
        inputs.push_back(std::move(example.input));
        targets.push_back(std::move(example.target));
    }
    return {stack(inputs), stack(targets)};
}
```

`TensorDataset(inputs, targets)` holds two tensors whose first dimension indexes examples. `get` returns `select(0, index)` views. `get_batch` is overridden with two `index_select` calls along dimension 0, one copy per batch instead of one per example plus a stack. Its constructor requires equal first dimensions (`TensorDataset.RequiresSameFirstDimension`). `Dataset` is not copyable, for the same reason as `Module`.

PyTorch splits this differently: a `Dataset` returns single examples and a `collate_fn` in the `DataLoader` stacks them. kael puts batching in the dataset because the dataset knows its storage: a tensor-backed dataset can gather a batch in one call, and a file-backed dataset can read a contiguous range.

#### DataLoader

`DataLoader(dataset, {.batch_size, .shuffle, .drop_last})` is a range of `Example` batches. `DataLoaderOptions` defaults to `batch_size = 1`, `shuffle = false`, `drop_last = false`; a non-positive batch size throws (`DataLoader.InvalidBatchSizeThrows`). The loader holds a `const Dataset&`; the dataset must outlive it.

From [`src/data/dataset.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/data/dataset.cpp):

```cpp
DataLoader::Iterator DataLoader::begin() const {
    const std::int64_t size = dataset_.size();
    Tensor order =
        options_.shuffle ? randperm(size) : arange(0, static_cast<double>(size), 1.0, DType::I64);
    return Iterator(this, std::move(order), 0);
}

Example& DataLoader::Iterator::operator*() {
    if (!current_) {
        const std::int64_t batch_size = loader_->options_.batch_size;
        const std::int64_t start = batch_ * batch_size;
        const std::int64_t length = std::min(batch_size, order_->numel() - start);
        current_ = loader_->dataset_.get_batch(order_->narrow(0, start, length));
    }
    return *current_;
}
```

Each call to `begin()` draws a fresh permutation from the global generator, so every epoch of a range-`for` sees a new order, and `manual_seed` makes the sequence of orders reproducible (`DataLoader.ShuffleIsReproducibleWithSeedAndChangesBetweenEpochs`). Every index appears exactly once per epoch (`DataLoader.ShuffleVisitsEachIndexOncePerEpoch`). The iterator compares by batch index only; `end()` is the iterator at `num_batches()` with no order tensor. Dereferencing builds the batch on first access and caches it in `current_` until `++`, so `*it` twice does not gather twice. The batch is a slice of the permutation (`narrow`), so the order tensor is created once per epoch.

Trace with 10 examples, batch size 4 and shuffle on:

```text
begin():  order = randperm(10) = [7 2 9 0 5 1 8 3 6 4]   (example values)

batch 0   narrow(0, 0, 4)  -> [7 2 9 0]   get_batch -> inputs [4, ...]
batch 1   narrow(0, 4, 4)  -> [5 1 8 3]
batch 2   narrow(0, 8, 2)  -> [6 4]       length = min(4, 10 - 8) = 2
end       batch index 3 == num_batches()
```

The batch count:

```text
num_batches = drop_last ? size / batch_size : ceil(size / batch_size)

60000 examples, batch 64:  938 batches, the last one has 60000 - 937*64 = 32 examples
                           937 batches with drop_last
```

The loader is single-threaded and loads synchronously. For MNIST, which fits in memory as one tensor, a batch is one `index_select`, and a background prefetch thread would not pay for itself.

#### MNIST and the IDX format

MNIST is a dataset of 70000 grayscale handwritten digits, 28 by 28 pixels, split into 60000 training and 10000 test images. It ships as four IDX files. [`scripts/download/mnist.sh`](https://github.com/JetBerri/kael.cpp/blob/main/scripts/download/mnist.sh) fetches them into `data/mnist` (or the directory given as its argument), decompresses them and verifies the SHA-256 of each decompressed file. Files already present with the right checksum are kept. An IDX file is a big-endian header followed by raw bytes.

```text
offset  bytes  content
0       4      magic: 0x00000803 (images, 3 dims) or 0x00000801 (labels, 1 dim)
4       4      size of dim 0 (count)
8       4      size of dim 1 (rows)      images only
12      4      size of dim 2 (columns)   images only
16      ...    one unsigned byte per pixel or label, row-major
```

The magic encodes the element type in its third byte (`0x08`, unsigned byte) and the number of dimensions in its fourth. The first 16 bytes of `train-images-idx3-ubyte` and the first 8 bytes of `train-labels-idx1-ubyte`:

```text
images:  00 00 08 03   00 00 EA 60   00 00 00 1C   00 00 00 1C
         magic         60000         28 rows       28 columns
labels:  00 00 08 01   00 00 EA 60
         magic         60000
```

From [`src/data/mnist.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/data/mnist.cpp):

```cpp
std::uint32_t read_big_endian(std::ifstream& file, const std::filesystem::path& path) {
    std::array<unsigned char, 4> bytes{};
    file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    KAEL_CHECK(file.good(), "IDX file " << path << " is truncated in its header");
    return (std::uint32_t{bytes[0]} << 24U) | (std::uint32_t{bytes[1]} << 16U) |
           (std::uint32_t{bytes[2]} << 8U) | std::uint32_t{bytes[3]};
}
```

Assembling the integer from bytes works on any host byte order, unlike reading a `uint32_t` and swapping: the swap is correct on a little-endian host and wrong on a big-endian one, and the shift form needs no `#ifdef` (`Idx.BigEndianSizes`). `00 00 EA 60` becomes `0xEA << 8 | 0x60 = 60000`.

`read_header` checks the magic, then reads one size per dimension. `read_payload` compares the number of bytes left in the file with `numel(shape)` before allocating:

```cpp
const std::int64_t expected = numel(shape);
const std::streampos start = file.tellg();
file.seekg(0, std::ios::end);
const std::int64_t available = file.tellg() - start;
KAEL_CHECK(available == expected,
           "IDX file " << path << " has " << available << " data bytes, expected " << expected);
```

A truncated file and trailing garbage throw before the payload is allocated (`Idx.TruncatedDataThrows`, `Idx.TrailingDataThrows`). A header that declares `0xFFFFFFFF` in every dimension throws even earlier: `numel` detects that the product overflows `int64` (`Idx.HugeDeclaredSizeThrowsBeforeAllocating`). Both checks defend against a corrupted or hostile file making the reader allocate gigabytes. A wrong magic, a short header and a missing file also throw, with the path in the message (`Idx.WrongMagicThrows`, `Idx.TruncatedHeaderThrows`, `Idx.MissingFileThrows`).

`read_idx_images` returns `F32` of shape `[N, 1, 28, 28]` with pixels scaled to `[0, 1]` (byte divided by 255; `Idx.ReadsImagesAsNormalizedFloats`). The extra dimension of size 1 is the channel, which keeps the layout compatible with convolutional models that expect `[N, C, H, W]`. `read_idx_labels` returns `I64` of shape `[N]`, the dtype the class losses require. The 60000 training images take `60000 * 784 * 4 = 188160000` bytes, about 188 MB, as float32.

`load_mnist(directory, MnistPart::Train)` reads `train-images-idx3-ubyte` and `train-labels-idx1-ubyte` (`t10k-*` for `MnistPart::Test`), checks that the counts match, and returns an `MnistSplit` with the images standardized by the conventional MNIST statistics `mean = 0.1307`, `std = 0.3081`:

```text
black pixel: (0 - 0.1307) / 0.3081 = -0.4242
white pixel: (1 - 0.1307) / 0.3081 =  2.8215
```

Standardized inputs have roughly zero mean and unit variance, which is the input distribution the initialization formulas assume.

### The MNIST training loop

[`examples/mnist_mlp.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/examples/mnist_mlp.cpp) puts the chapter together. The model is `784 -> hidden -> 10` with a ReLU, trained with `AdamW` at its defaults (`lr = 1e-3`, `weight_decay = 0.01`) on batches of 64 for 5 epochs.

```cpp
kael::nn::Sequential model{
    std::make_shared<kael::nn::Flatten>(), std::make_shared<Linear>(784, options.hidden),
    std::make_shared<kael::nn::ReLU>(), std::make_shared<Linear>(options.hidden, 10)};
model.to(options.device);
kael::optim::AdamW optimizer(model.parameters());
const kael::data::DataLoader loader(train_set, {.batch_size = kBatchSize, .shuffle = true});
```

The shapes through one training batch, with the default `hidden = 128`:

```text
images        [64, 1, 28, 28]   F32, standardized
Flatten       [64, 784]
Linear 1      [64, 128]         [64, 784] @ [784, 128] + [128]
ReLU          [64, 128]
Linear 3      [64, 10]          logits
cross_entropy []                scalar, mean over 64 rows
labels        [64]              I64
```

`model.to` runs before the optimizer is built. `to` moves storage inside the same `TensorImpl`, so the handles the optimizer holds stay valid either way, but `AdamBase` allocates its moment buffers with `zeros_like(parameter)` in its constructor. Moving the model afterwards would leave those buffers on the old device. PyTorch has the same rule for the same reason.

One training epoch, from `train_epoch`:

```cpp
model.train();
std::int64_t step = 0;
for (const auto& [images, labels] : loader) {
    const Tensor loss = kael::nn::cross_entropy(model(images.to(device)), labels.to(device));
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
    if (step % kReportEvery == 0) {
        std::cout << "epoch " << epoch << " step " << step << " loss " << loss.item() << '\n';
    }
    ++step;
}
```

`zero_grad` sits between the forward pass and `backward`, which is the latest point it can run: gradients accumulate, so the previous step's gradients must be cleared before this step's arrive. Placing it after the forward pass is harmless because the forward pass reads parameters, not gradients. The loss is printed every 100 steps (`kReportEvery`).

Evaluation, from `accuracy`, switches to eval mode, disables recording, and counts correct predictions in batches of 1000:

```cpp
model.eval();
const kael::autograd::NoGradGuard guard;
std::int64_t correct = 0;
for (const auto& [images, labels] : kael::data::DataLoader(dataset, {.batch_size = 1000})) {
    const Tensor predictions = kael::argmax(model(images.to(device)), 1);
    correct += kael::sum(predictions == labels.to(device)).item<std::int64_t>();
}
```

`NoGradGuard` saves memory and time: no nodes and no saved activations. `eval()` does not change this model's output (it has no dropout) but is the correct habit. `argmax` over the logits gives the predicted class; the softmax is unnecessary because it preserves the order of the logits. The temporary `DataLoader` in the range-`for` is safe because it refers to `dataset`, a parameter that outlives the loop, and range-`for` extends the lifetime of the loader itself.

Arguments: a positional data directory (default `data/mnist`), `--device cpu|cuda` and `--hidden N` (default 128). `--help` or `-h` prints the usage line. An unknown option, a missing value or a device name other than `cpu`, `cuda` or `cuda:0` ends the program with an error instead of being ignored; the device goes through `kael::parse_device` ([02. Memory model](@/book/02-memory-model.md)). The seed is fixed at `42` (`kSeed`).

```bash
scripts/download/mnist.sh
cmake -S . -B build -DKAEL_BUILD_EXAMPLES=ON
cmake --build build --target mnist_mlp
build/examples/mnist_mlp data/mnist
```

A run of the Debug build in `build/` on an Apple M5 Pro printed these test accuracies:

```text
epoch 1 test accuracy 96% in 7.30869 s
epoch 2 test accuracy 96.73% in 7.14235 s
epoch 3 test accuracy 97.35% in 7.33984 s
epoch 4 test accuracy 97.35% in 7.39241 s
epoch 5 test accuracy 97.42% in 8.22106 s
```

The first logged loss is `2.40`, close to `log(10) = 2.30` as expected from a fresh classifier. No unit test runs the example, because CI does not download MNIST. [`scripts/cuda/validate.sh`](https://github.com/JetBerri/kael.cpp/blob/main/scripts/cuda/validate.sh) builds kael with CUDA on a machine with an NVIDIA GPU. When `data/mnist` exists, it runs the example on CPU and on CUDA (hidden 128 and 1024) and fails a step if the final accuracy is below `minimum_mnist_accuracy="97"`. Without the data it records the MNIST steps as skipped, and a run with skipped steps exits with status 3, not 0. Timings depend on the machine and build type; [08. CPU performance](@/book/08-cpu-performance.md) is about making them smaller.

[`examples/mnist.py`](https://github.com/JetBerri/kael.cpp/blob/main/examples/mnist.py) is the same program through the Python bindings, described in [10. Python bindings](@/book/10-python-bindings.md). Its options are `--data`, `--device` (`cpu` or `cuda`, enforced by `argparse`), `--epochs` (default 5) and `--hidden` (default 128), and `--help` describes them.

## Edge cases and pitfalls

- **Forgetting `zero_grad`.** Gradients accumulate across `backward()` calls. Skipping `zero_grad` sums the gradients of every past batch, and the effective step grows each iteration until training diverges.
- **Stepping between forward and backward.** `optimizer.step()` updates parameters in place and bumps their version. A graph built before the step that saved a parameter (the second `Linear` of a stack saves its weight for the input gradient) fails at `backward()` with the in-place version error.
- **Writing to parameters outside an optimizer.** In-place writes to a parameter throw while grad mode is on. Use `NoGradGuard`, as `init`, `load_state_dict` and `Module::to` do.
- **Moving the model after building the optimizer.** Adam's moment buffers stay on the device the parameters had when the optimizer was constructed. Call `model.to(device)` first.
- **Dropout probability.** `p` must be in `[0, 1)`. `Dropout(1.0)` and `Dropout(-0.1)` throw.
- **Targets.** Class targets must be `I64`, 1-D, as long as the batch, and inside `[0, C)`. A float label tensor is a type error, not a silent cast.
- **Large logits.** Use `cross_entropy` and `binary_cross_entropy_with_logits` on raw scores. Both are written to avoid overflow; composing `softmax` then `log` is not.
- **Softmax before `cross_entropy`.** `cross_entropy` applies `log_softmax` itself. Passing probabilities instead of logits applies softmax twice, which trains, but slowly and to a worse result, with no error.
- **Warmup schedulers start below the base rate.** The first update after constructing `LinearWarmup` or `WarmupCosineLR` uses `base / warmup`, not `0`.
- **Non-finite gradients.** `clip_grad_norm_` throws `ValueError` when the total norm is NaN or infinite. Catch it to skip the batch, or enable anomaly detection with `set_anomaly_detection` ([13. Serving and tooling](@/book/13-serving-and-tooling.md)) to find the node that produced it.
- **DataLoader lifetime.** `DataLoader` stores `const Dataset&`. Constructing a named loader from a temporary dataset leaves a dangling reference.
- **State dict aliasing.** `state_dict()` entries share storage with the parameters. An optimizer step after taking the state dict changes the saved values too. Clone the entries for a frozen snapshot.
- **Tied parameters in `state_dict`.** A tensor registered under two names appears once in `parameters()` but twice in `state_dict()`. `load_state_dict` copies both entries into the same storage, so the second wins.
- **Empty batches.** `cross_entropy` on zero rows skips the target range check and its gradient is an empty tensor (`CrossEntropyFusedGraph.GradientOfAnEmptyBatchIsEmpty`).
- **Malformed IDX files.** Wrong magic, short header, short or long payload and missing files all throw `kael::Error` with the path in the message.

> [!WARNING]
> `model.eval()` does not disable gradient recording, and `NoGradGuard` does not switch dropout off. Evaluation needs both. Forgetting the guard wastes memory on a graph nobody uses; forgetting `eval()` silently evaluates a model with dropout active.

## How it is tested

| Test file | Suites | What it proves |
|-----------|--------|----------------|
| [`tests/test_nn_module.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_nn_module.cpp) | `NnModule`, `NnSequential` | Recursive collection, dotted names in registration order, registered parameters require grad, training flag propagation, `zero_grad`, `state_dict` round trip between two models, detached entries, shape and name checks in `load_state_dict`, tied parameters listed once, null child rejected, `Sequential` naming and order |
| [`tests/test_nn_layers.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_nn_layers.cpp) | `NnInit`, `NnLinear`, `NnActivation`, `NnDropout`, `NnLayerNorm`, `NnEmbedding`, `NnFlatten`, `NnLoss`, `CrossEntropyFusedGraph` | Fans, including an empty weight, initializer ranges and moments, PyTorch default bounds for `Linear`, layer outputs against PyTorch golden data, gradcheck for `Linear`, `LayerNorm`, `Embedding` and every loss, dropout statistics, stable BCE and its gradient `sigmoid(x) - t` at `x = 0`, target validation, the fused cross entropy node on contiguous and non-contiguous logits |
| [`tests/test_optim.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_optim.cpp) | `OptimGolden`, `Optimizer`, `ClipGradNorm`, `Schedulers` | SGD (plain, momentum, Nesterov, weight decay) and Adam, Adam with decay, AdamW trajectories match PyTorch after steps 1 and 5; Adam's first step size, also for a parameter whose first gradient arrives late; in-place update keeps the tensor; skipping parameters without grad; invalid configuration; clipping above and below the limit and rejecting a non-finite norm; every scheduler against PyTorch or hand values |
| [`tests/test_data.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_data.cpp) | `Idx`, `TensorDataset`, `Dataset`, `DataLoader` | IDX parsing of synthetic files written by the test (including big-endian sizes and every malformed case), dataset indexing, default `get_batch` stacking, batch counts, order, `drop_last`, shuffle coverage and reproducibility, invalid batch size |
| [`tests/test_training.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_training.cpp) | `Training` | A `20 -> 64 -> 4` MLP overfits 32 random examples (initial loss near `log 4`, final loss below `0.02` after 200 Adam steps); `Linear(1, 1)` with SGD recovers `y = 3x + 2` within `0.05`; a `2 -> 8 -> 2` tanh MLP solves XOR; the same seed gives bit-identical losses and a different seed does not |

The tests use three kinds of evidence. **Golden data** compares kael with PyTorch on fixed inputs. **Gradcheck** compares autograd gradients with finite differences ([06. Autograd](@/book/06-autograd.md)). **Training tests** check that the pieces work together: a model that cannot overfit 32 examples has a bug somewhere, even when every unit test passes.

Golden data comes from [`scripts/generate/golden.py`](https://github.com/JetBerri/kael.cpp/blob/main/scripts/generate/golden.py). `write_nn_fixtures` (seed `20261`) saves inputs and outputs of `torch.nn.functional.linear`, `layer_norm` over `(3, 4)`, `embedding`, and the four losses under every reduction into `tests/golden/nn/`. `write_optim_fixtures` (seed `20262`) runs each optimizer for five steps on the loss `sum(w * c) + 0.5 * sum(w^2)` and saves the parameter after steps 1 and 5, and records the learning rate sequence of `StepLR(step_size=3, gamma=0.5)` and `CosineAnnealingLR(T_max=10, eta_min=0.001)`, into `tests/golden/optim/`. The optimizer configurations are plain SGD, SGD with momentum `0.9`, Nesterov, SGD with momentum `0.5` and weight decay `0.01`, Adam, Adam with weight decay `0.1`, and AdamW. Step 5 is the interesting snapshot: by then momentum and the bias corrections have had time to diverge if a formula is subtly wrong.

> [!TIP]
> When a training run misbehaves, reproduce `Training.OverfitsASingleBatch` on your own model: one fixed batch, a few hundred steps, and the loss must go near zero. If it does not, the bug is in the model or the loss, not in the data.

## Build it yourself

1. Write `Module` with `register_parameter`, `register_module`, `named_parameters` with dotted names and deduplication, `parameters`, `train`/`eval`, `zero_grad`. Checkpoint: `NnModule.ParametersAreCollectedRecursively`, `NnModule.NamedParametersUseDottedNamesInOrder`, `NnModule.SharedParameterIsListedOnce`.
2. Add `state_dict` and `load_state_dict` with validation before writing. Checkpoint: `NnModule.StateDict*` and `NnModule.LoadStateDict*`.
3. Write `init`: `fans_of`, `uniform_`, `normal_`, the constant fills, Xavier and Kaiming, all under `NoGradGuard`. Checkpoint: `NnInit.*`.
4. Write `Linear` with the PyTorch default initialization. Checkpoint: `NnLinear.*`, including `NnLinear.MatchesPyTorch` and `NnLinear.Gradcheck`.
5. Write the activation modules, `Dropout`, `Flatten` and `Sequential`. Checkpoint: `NnActivation.*`, `NnDropout.*`, `NnFlatten.*`, `NnSequential.*`.
6. Write `LayerNorm` on top of `standardize` and `Embedding` on top of `index_select`. Checkpoint: `NnLayerNorm.*`, `NnEmbedding.*`.
7. Write the losses with `Reduction`. Checkpoint: `NnLoss.*` and `CrossEntropyFusedGraph.*`.
8. Write `Optimizer` with `step()` under `NoGradGuard`, then `SGD`. Checkpoint: `Training.LinearRegressionConverges` and `OptimGolden.SgdVariantsMatchPyTorch`.
9. Write `AdamBase`, `Adam` and `AdamW` with per-parameter step counts. Checkpoint: `OptimGolden.AdamVariantsMatchPyTorch`, `Optimizer.AdamFirstStepMovesByLearningRate`, `Optimizer.AdamCorrectsTheFirstStepOfALateParameter`, then all of `Training.*`.
10. Write the schedulers and `clip_grad_norm_`. Checkpoint: `Schedulers.*`, `ClipGradNorm.*`.
11. Write `Dataset`, `TensorDataset` and `DataLoader`. Checkpoint: `TensorDataset.*`, `Dataset.*`, `DataLoader.*`.
12. Write the IDX reader and `load_mnist`. Checkpoint: `Idx.*`.
13. Run `scripts/download/mnist.sh`, configure with `-DKAEL_BUILD_EXAMPLES=ON`, build `mnist_mlp` and run it. Checkpoint: the last line reports at least 97% test accuracy.

To run one checkpoint, filter the test binary with GoogleTest's `--gtest_filter`, for example `--gtest_filter='NnModule.*'`.

## Check your understanding

<details>
<summary>Why does kael not need a separate Parameter type to let an optimizer update a layer's weights?</summary>

`Tensor` is a handle to a shared `TensorImpl`. `register_parameter` stores a copy of the handle in the module's registry, and the layer keeps another copy in its member. Both point at the same storage, so when the optimizer calls `sub_` on the handle it got from `parameters()`, the layer sees the new values. The only marking a parameter needs is `requires_grad_()`, which `register_parameter` applies.

</details>

<details>
<summary>A model ties its embedding matrix to its output projection. What goes wrong if named_parameters does not deduplicate?</summary>

The optimizer receives the same tensor twice. SGD subtracts the gradient twice per step, doubling the effective learning rate for that tensor. Adam keeps two independent pairs of moments for the same storage and applies two updates computed from them. Deduplication by `TensorImpl` address lists the tensor once, so it gets one update and one set of state.

</details>

<details>
<summary>On the first step, Adam moves every element by about the learning rate, whatever the gradient's size. Why?</summary>

After one step, `m = (1 - beta1) g` and `v = (1 - beta2) g^2`. The bias corrections divide by `1 - beta1` and `1 - beta2`, which gives `m_hat = g` and `v_hat = g^2`. The update is `lr * g / (sqrt(g^2) + eps) = lr * sign(g)` for any `g` much larger than `eps`.

</details>

<details>
<summary>With w = 1, a zero gradient, lr = 1e-3 and weight decay 0.01, why does Adam move w a hundred times further than AdamW in one step?</summary>

`Adam` adds the decay to the gradient, so the gradient becomes `0.01`. The adaptive normalization divides it by its own magnitude, which turns it into a step of about `lr = 1e-3`, giving `0.999001`. `AdamW` multiplies the weight by `1 - lr * lambda = 1 - 1e-5`, giving `0.99999`. Coupled decay is rescaled by the adaptive denominator; decoupled decay is not.

</details>

<details>
<summary>Why does the MNIST example call model.to(device) before constructing the optimizer?</summary>

`AdamBase` allocates its first and second moment buffers with `zeros_like(parameter)` in its constructor, on the parameter's current device. `Module::to` keeps the same `TensorImpl`, so the optimizer's parameter handles would stay valid after a later move, but the moment buffers would remain on the CPU while the parameters and gradients live on the GPU.

</details>

<details>
<summary>Why does binary_cross_entropy_with_logits use max(x, 0) - x t + log1p(exp(-|x|)) instead of applying sigmoid then log?</summary>

For large positive `x`, `sigmoid(x)` rounds to `1` and `log(1 - p)` becomes `log(0) = -inf`. For large negative `x`, `exp(-x)` overflows. The rearranged form only ever computes `exp` of a non-positive number, which lies in `(0, 1]`, and `log1p` keeps precision when that number is tiny. At `x = 1000`, `t = 1` it returns exactly `0`.

</details>

## Further reading

- [`src/optim/optimizer.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/optim/optimizer.cpp): every optimizer, scheduler and the clipping function in about 210 lines, the best single file to read after this chapter.
- Glorot and Bengio, "Understanding the difficulty of training deep feedforward neural networks" (2010): the variance argument behind Xavier initialization.
- He, Zhang, Ren and Sun, "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification" (2015): introduces Kaiming initialization and the ReLU gain of `sqrt(2)`.
- Kingma and Ba, "Adam: A Method for Stochastic Optimization" (2014): the algorithm and the derivation of the bias correction.
- Loshchilov and Hutter, "Decoupled Weight Decay Regularization": why AdamW separates decay from the adaptive step.
- Loshchilov and Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts": the origin of cosine learning rate schedules.
- Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (2014): the method and the reasoning behind the expected-value scaling.
- Ba, Kiros and Hinton, "Layer Normalization" (2016): the normalization used by every transformer in [11. Transformer](@/book/11-transformer.md).
- LeCun, Cortes and Burges, the MNIST database page: documents the IDX format byte by byte.
- The PyTorch documentation for `torch.nn.Module`, `torch.optim` and `torch.utils.data`: the API kael mirrors, useful to compare design choices.
- The [glossary](@/concepts/glossary.md) for short definitions of the terms used here.

Next: [08. CPU performance](@/book/08-cpu-performance.md)
