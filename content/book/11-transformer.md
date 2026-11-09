+++
title = "Transformer"
description = "Causal self-attention, a GPT-2 style model with tied weights, a key/value cache and a sampling loop, built from the modules of chapter 07."
weight = 11
slug = "transformer"

[extra]
chapter = "11"
kicker = "Chapter 11"
builds = "Attention, a GPT model, a KV cache and sampling"
source = "docs/11-transformer.md"
tag = "chapter-12-transformer"
math = true
+++

This chapter builds a GPT-2 style decoder-only transformer on top of the modules from [chapter 07](@/book/07-neural-networks.md): token and position embeddings, causal self-attention with several heads, a GELU MLP, pre-layer-norm residual blocks, the GPT model with tied output weights, a key/value cache for fast generation, and a sampling loop with temperature and top-k. It ends with `examples/char_gpt.cpp`, which trains a character-level model on Tiny Shakespeare. The chapter maps to the v1.0 milestone (run GPT-2) and the `chapter-12-transformer` tag. [Chapter 12](@/book/12-model-loading.md) then loads the real GPT-2 weights into the same classes. The `on_token` callback and the training-mode restore in `generate` landed in phase 13, so the code quoted here is complete at `chapter-13-polish`, the same commit as `v1.0.0`.

| File | Role |
|------|------|
| [`include/kael/nn/transformer.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/nn/transformer.h) | `GptConfig`, `LayerCache`, `KvCache`, `CausalSelfAttention`, `TransformerMlp`, `TransformerBlock`, `GPT`, `GenerateOptions`, `generate`, `load_gpt2_weights` |
| [`src/nn/transformer.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/nn/transformer.cpp) | Implementation of all of the above, the sampler `sample_next` and the GPT-2 name mapping |
| [`include/kael/nn/activation.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/nn/activation.h) | `GELU` module with the exact and the tanh variant |
| [`src/internal/element_functions.h`](https://github.com/JetBerri/kael.cpp/blob/main/src/internal/element_functions.h) | `GeluExact` and `GeluTanh` element functions |
| [`src/random.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/random.cpp) | `multinomial`, the sampler behind `generate` |
| [`examples/char_gpt.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/examples/char_gpt.cpp) | Character-level GPT trained on Tiny Shakespeare |
| [`scripts/download/shakespeare.sh`](https://github.com/JetBerri/kael.cpp/blob/main/scripts/download/shakespeare.sh) | Downloads `input.txt` and checks its SHA-256 |
| [`tests/test_transformer.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_transformer.cpp) | GoogleTest suites `Attention`, `Block`, `Gpt`, `Generate`, `GptGenerate`, `Gpt2Weights` |
| [`python/tests/test_gpt.py`](https://github.com/JetBerri/kael.cpp/blob/main/python/tests/test_gpt.py) | pytest comparison against PyTorch and Hugging Face `transformers` |

## Prerequisites

- [01. Shapes and strides](@/book/01-shapes-and-strides.md): broadcasting, used by the mask, the position embeddings and the bias.
- [03. Views](@/book/03-views.md): `view`, `select`, `transpose`, `slice`, `narrow`, `mT`, and when `contiguous` is needed.
- [05. Operations](@/book/05-operations.md): `matmul` with batch dimensions, `softmax`, `masked_fill`, `triu`, `topk`, `cat`, `argmax`.
- [06. Autograd](@/book/06-autograd.md): `NoGradGuard` and gradcheck.
- [07. Neural networks and training](@/book/07-neural-networks.md): `Module`, `Linear`, `Embedding`, `LayerNorm`, `Dropout`, `cross_entropy`, `AdamW`, schedulers and gradient clipping.

## Why this layer exists

Chapters 01 to 10 built a general tensor library: strided views, operations, autograd, modules, optimizers, a CUDA backend and Python bindings. None of it knows what a language model is. This chapter is the first place where the library is used to build a specific architecture, and the architecture it builds is the one behind GPT-2 and most text generators since.

The problem a language model solves is next-token prediction. Given a sequence of token ids, produce a probability distribution over the vocabulary for the token that comes next. Two older designs show what goes wrong without the ideas in this chapter:

- A **fixed-window MLP** concatenates the embeddings of the last `k` tokens and feeds them to a feed-forward network. It cannot look further back than `k`, and its weights for position 1 and position 2 are unrelated, so it learns every pattern once per offset.
- A **recurrent network** reads one token at a time and carries a hidden state forward. It can in principle remember anything, but in practice the information has to survive many multiplications, and training is sequential: step `t` cannot start before step `t - 1` ends.

A transformer replaces both with **attention**: every position looks at all earlier positions directly, with weights computed from the content of the tokens, in one batched matrix multiply. Nothing has to survive a chain of steps, and all positions of a sequence are computed in parallel during training.

Without the pieces in this chapter, the library could train an MLP on MNIST (chapter 07) but could not run GPT-2. Without the causal mask, training would let each position read the answer. Without the KV cache, generating text would recompute the whole prefix for every new token. Without a sampler, the model would produce distributions but no text.

The mental model to hold on to for the rest of the chapter:

```text
token ids  ->  vectors  ->  L x [ mix across positions (attention) ]  ->  scores per vocabulary entry
                                [ transform each position (MLP)    ]
```

A **residual stream** of shape `[B, T, C]` flows from the embeddings to the output. Each block reads a normalized copy of it, computes something, and adds the result back. Attention is the only place where information moves between positions. The MLP works on each position alone. The final layer turns each position's vector into one score per vocabulary entry.

## Concepts

### Notation

The chapter uses one letter per dimension. Every shape below is written with these letters.

| Letter | Meaning | GPT-2 small | `char_gpt` |
|--------|---------|-------------|------------|
| `B` | batch size | 1 when generating | 32 |
| `T` | number of new positions in this call | prompt length, then 1 | 64 |
| `P` | positions already in the KV cache | 0 to 1023 | 0 |
| `C` | model width, `GptConfig::dimension` | 768 | 128 |
| `H` | attention heads, `GptConfig::heads` | 12 | 4 |
| `D` | head dimension, `C / H` | 64 | 32 |
| `V` | vocabulary size, `GptConfig::vocab_size` | 50257 | 65 |
| `L` | number of blocks, `GptConfig::layers` | 12 | 4 |
| `N` | context length, `GptConfig::max_positions` | 1024 | 64 |

A **token** is one unit of text with an integer id in `[0, V)`. For `char_gpt` a token is one byte of the training text. For GPT-2 it is a piece of a word produced by the BPE tokenizer of [chapter 12](@/book/12-model-loading.md). A **logit** is an unnormalized score: softmax turns a row of logits into probabilities.

### The model in one picture

A GPT maps a batch of token ids `[B, T]` to a batch of logits `[B, T, V]`: for every position, one score per vocabulary entry for the token that comes next.

```text
indices [B, T] int64                 positions arange(P, P + T)
        |                                    |
        v                                    v
token_embedding  [B, T, C]           position_embedding  [T, C]
        |                                    |
        +-------------- sum, broadcast over B --------------+
                                |
                                v
                         dropout [B, T, C]
                                |
                                v
                     blocks.0, blocks.1, ..., blocks.L-1
                                |
                                v
                       ln_final  [B, T, C]
                                |
                                v
          matmul with token_embedding weight  ->  logits [B, T, V]
```

Each block is **pre-layer-norm**: the residual stream `x` passes through unchanged, and each sublayer reads a normalized copy and adds its result back.

```text
x [B, T, C] ----------------------------+
   |                                    |
   +--> ln1 --> attn ------------------(+)--> a [B, T, C] ---------------+
                                              |                          |
                                              +--> ln2 --> mlp ---------(+)--> out [B, T, C]
```

The two `+` nodes are the only path from input to output that has no weights on it. That is the property that lets 12 or 48 of these blocks be stacked and still train.

### Attention from first principles: one head, one sequence

Attention lets each position build its output as a weighted average of information from other positions. Every position produces three vectors of length `D`:

- a **query**: what this position looks for,
- a **key**: what this position offers to others,
- a **value**: what this position hands over when it is chosen.

For a sequence of `T` positions, stack them into matrices $Q$, $K$, $V$ of shape `[T, D]`. Scaled dot-product attention is

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{Q K^\top}{\sqrt{D}}\right) V
$$

with softmax applied to each row. For a row of scores $s$, softmax gives the weights

$$
p_j = \frac{e^{s_j}}{\sum_{k} e^{s_k}}
$$

The same thing written as shapes:

```text
scores  = Q K^T / sqrt(D)          [T, D] x [D, T] = [T, T]
weights = softmax(scores, dim=-1)  [T, T], each row sums to 1
output  = weights V                [T, T] x [T, D] = [T, D]
```

Row `i` of `scores` holds the dot products of query `i` with every key. Softmax turns each row into a probability distribution over positions, and row `i` of `output` is the matching mixture of value vectors. A query that points in the same direction as a key gives a large dot product, so that key's value gets a large share of the mix.

A worked example with `T = 3` and `D = 2`, so $\sqrt{D} = 1.414$:

```text
Q = [ 1 0 ]    K = [ 1 0 ]    V = [ 1 0 ]
    [ 0 1 ]        [ 1 1 ]        [ 0 1 ]
    [ 1 1 ]        [ 0 2 ]        [ 2 2 ]

Q K^T        = [ 1 1 0 ]      / 1.414  =  [ 0.707 0.707 0.000 ]
               [ 0 1 2 ]                  [ 0.000 0.707 1.414 ]
               [ 1 2 2 ]                  [ 0.707 1.414 1.414 ]
```

Row 2 (query `[1, 1]`) gives softmax of `[0.707, 1.414, 1.414]`, which is `[0.198, 0.401, 0.401]`. Its output is `0.198 * [1, 0] + 0.401 * [0, 1] + 0.401 * [2, 2] = [1.000, 1.203]`. The other two rows follow the same steps; the next section shows how the mask changes them.

#### Why divide by the square root of D

The division by $\sqrt{D}$ keeps the scores in a range where softmax still has gradient. If the entries of $q$ and $k$ are independent with zero mean and unit variance, each product $q_i k_i$ has variance 1, and the dot product is a sum of $D$ of them:

$$
\operatorname{Var}(q \cdot k) = \sum_{i=1}^{D} \operatorname{Var}(q_i k_i) = D
$$

Dividing by $\sqrt{D}$ brings the variance back to 1. Without it, for $D = 64$ the scores have a standard deviation of 8 (a quick NumPy draw of 100,000 pairs gives 7.999). Softmax of numbers that far apart is nearly one-hot: one position gets almost all the weight, the derivative of softmax at the losing positions is close to zero, and those positions stop learning.

### The causal mask

A language model predicts token `t + 1` from tokens `0..t`. If position `t` could attend to position `t + 1`, training would let it copy the answer, and the model would learn nothing useful for generation, where the future does not exist yet.

**Causal masking** forbids every query from looking at a later key. It sets those scores to minus infinity before softmax, so their weight is exactly $e^{-\infty} = 0$. The boolean matrix that marks the forbidden pairs is the **causal mask**.

kael builds the mask once, as a boolean matrix where `true` means "future, hide it". From [`src/nn/transformer.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/nn/transformer.cpp):

```cpp
Tensor future_positions_mask(std::int64_t size) {
    return triu(ones({size, size}, DType::Bool), 1);
}
```

`triu(x, 1)` keeps the entries strictly above the main diagonal and zeroes the rest. For `T = 3`:

```text
         key 0  key 1  key 2
query 0  false  true   true
query 1  false  false  true
query 2  false  false  false
```

A worked example with made-up scores (already divided by $\sqrt{D}$):

```text
scores                masked                       softmax
[ 2.0  1.0  0.0 ]     [ 2.0  -inf -inf ]           [ 1.000 0.000 0.000 ]
[ 1.0  1.0  5.0 ]  -> [ 1.0   1.0 -inf ]       ->  [ 0.500 0.500 0.000 ]
[ 0.0  0.0  0.0 ]     [ 0.0   0.0  0.0 ]           [ 0.333 0.333 0.333 ]
```

Query 0 only sees itself. The large score 5.0 in row 1 has no effect because it points to the future. The diagonal is never masked, so every row has at least one finite entry and softmax never divides by zero.

Applied to the `Q`, `K`, `V` example above, the masked weights are:

```text
row 0  [ 1.000 0.000 0.000 ]   output [ 1.000 0.000 ]
row 1  [ 0.330 0.670 0.000 ]   output [ 0.330 0.670 ]
row 2  [ 0.198 0.401 0.401 ]   output [ 1.000 1.203 ]
```

Row 2 is unchanged because the last position may see everything. Row 1 loses the 1.414 score of key 2 and splits its weight between keys 0 and 1.

The mask is also what makes training efficient. A single forward pass over a sequence of 64 tokens produces 64 valid predictions at once: position 0 predicts token 1 from token 0, position 1 predicts token 2 from tokens 0 and 1, and so on. A recurrent network would need 64 sequential steps for the same work.

### Many heads

One head computes one weighting of positions. A model often needs several at once: one head may track the previous token, another the subject of the sentence, a third the matching open bracket.

**Multi-head attention** splits the width `C` into `H` slices of `D = C / H` and runs `H` independent attentions, one per slice, in parallel. The outputs of the heads are concatenated back to width `C` and mixed by an output projection. The total cost is the same as one head of width `C`, because each head works on `D = C / H` columns.

The split costs nothing in memory. A tensor `[B, T, C]` is reinterpreted as `[B, T, H, D]` with `view`, then the head dimension moves next to the batch with `transpose(1, 2)`, giving `[B, H, T, D]`. From then on `matmul` treats `B` and `H` as batch dimensions and multiplies the last two, which is exactly "one attention per head per sequence". The merge is the reverse: transpose back to `[B, T, H, D]`, make it contiguous, view as `[B, T, C]`.

```text
[B, T, C]  --view-->  [B, T, H, D]  --transpose(1, 2)-->  [B, H, T, D]
                                                              |
                                          attention per (b, h): [T, D] -> [T, D]
                                                              |
[B, T, C]  <--view--  [B, T, H, D]  <--transpose + contiguous-- [B, H, T, D]
```

PyTorch users know this pattern from nanoGPT and from `torch.nn.functional.scaled_dot_product_attention`, which takes `[B, H, T, D]` inputs for the same reason.

### Shapes at every step

`CausalSelfAttention::forward` with `B = 2`, `T = 5`, `C = 8`, `H = 2`, `D = 4` and no cache (`P = 0`):

| Step | Expression | Shape |
|------|------------|-------|
| input | `input` | `[2, 5, 8]` |
| fused projection | `query_key_value_->forward(input)` | `[2, 5, 24]` |
| split into q, k, v and heads | `.view({batch, length, 3, heads_, head_dimension_})` | `[2, 5, 3, 2, 4]` |
| queries | `parts.select(2, 0).transpose(1, 2)` | `[2, 2, 5, 4]` |
| keys, values | `parts.select(2, 1)`, `select(2, 2)`, transposed | `[2, 2, 5, 4]` each |
| keys transposed | `keys.mT()` | `[2, 2, 4, 5]` |
| scores | `matmul(queries, keys.mT()) / sqrt(4)` | `[2, 2, 5, 5]` |
| mask slice | `future_mask_.slice(0, 0, 5).slice(1, 0, 5)` | `[5, 5]`, broadcast |
| weights | `softmax(masked, -1)`, then dropout | `[2, 2, 5, 5]` |
| mixed | `matmul(weights, values)` | `[2, 2, 5, 4]` |
| merge heads | `.transpose(1, 2).contiguous()` | `[2, 5, 2, 4]` |
| concatenate heads | `mixed.view({batch, length, dimension_})` | `[2, 5, 8]` |
| output projection | `projection_->forward(...)`, then dropout | `[2, 5, 8]` |

The fused `qkv` layer is a single `Linear(C, 3C)`. Its output columns are laid out `[q | k | v]`, each block of `C` columns subdivided into `H` heads of `D`:

```text
column  0 .. 3   4 .. 7   8 .. 11  12 .. 15  16 .. 19  20 .. 23
        q head0  q head1  k head0  k head1   v head0   v head1
        \___ q (C) ___/   \___ k (C) ___/    \___ v (C) ___/
```

That is why the view is `[B, T, 3, H, D]` and not `[B, T, H, 3, D]`: the 3 is the slower-varying index. GPT-2's `c_attn` uses the same layout, which is what lets chapter 12 load its weights without reshuffling columns.

### The MLP and GELU

After attention mixes information across positions, the **MLP** (multi-layer perceptron) transforms each position on its own. It widens to `4C`, applies GELU, and projects back. Attention decides which information to gather; the MLP decides what to compute from it. Most of the parameters of a transformer live in the MLPs: `8C^2` of the `12C^2` weights per block.

**GELU** (Gaussian Error Linear Unit) weights its input by the probability that a standard normal variable is below it:

$$
\operatorname{GELU}(x) = x \Phi(x) = \frac{x}{2}\left(1 + \operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right)
$$

where $\Phi$ is the standard normal cumulative distribution function. GPT-2 used a cheaper approximation:

$$
\operatorname{GELU}(x) \approx \frac{x}{2}\left(1 + \tanh\left(\sqrt{2/\pi}\left(x + 0.044715 x^3\right)\right)\right)
$$

Unlike ReLU, GELU is smooth at zero and lets small negative values through. Some values:

| `x` | -3 | -1 | -0.5 | 0 | 0.5 | 1 | 3 |
|-----|----|----|------|---|-----|---|---|
| exact | -0.0040 | -0.1587 | -0.1543 | 0 | 0.3457 | 0.8413 | 2.9960 |
| tanh | -0.0036 | -0.1588 | -0.1543 | 0 | 0.3457 | 0.8412 | 2.9964 |
| ReLU | 0 | 0 | 0 | 0 | 0.5 | 1 | 3 |

The largest difference between the two GELU forms is about 0.00047, near `x = -2.7`.

### The residual stream and pre-layer-norm

The **residual stream** is the tensor `x` that flows from the embeddings to the final layer norm. Each sublayer reads `LayerNorm(x)` and adds its output to `x`. Two properties follow:

- The gradient of the loss with respect to `x` reaches every block through the `+` without passing through any weights. The derivative of $x + f(x)$ is $1 + f'(x)$, and the 1 survives no matter what $f$ does. Deep stacks train without the gradient shrinking layer by layer.
- Layer norm sits before each sublayer (**pre-LN**, as in GPT-2), not after the addition (**post-LN**, as in the original transformer paper). In post-LN, the residual path itself passes through a normalization at every layer, and deep post-LN stacks are known to need a careful learning-rate warmup to train stably. Pre-LN keeps the identity path clean. The price is that the stream is never normalized inside the stack, which is why `GPT` applies one more layer norm, `ln_final`, before the output head.

```text
post-LN (2017 paper):   x -> LN(x + f(x)) -> LN(. + g(.)) -> ...   normalization on the main path
pre-LN  (GPT-2):        x -> x + f(LN(x)) -> . + g(LN(.)) -> ... -> LN -> head
```

### Weight tying

**Weight tying** reuses the token embedding matrix `[V, C]` as the output projection. Row `v` of the matrix is both the input vector of token `v` and the direction whose dot product with the final hidden state gives the logit of `v`. Tokens that appear in similar contexts get similar rows, and the same similarity then makes them similarly likely as outputs. GPT-2 was trained this way, so the checkpoint has no separate head to load, and the model saves `V * C` parameters: 38,597,376 for GPT-2 small, almost a third of the model.

### Generation and the KV cache

**Autoregressive generation** appends one token at a time: run the model, read the logits of the last position, choose a token, append it, repeat. Without a cache, step `n` runs the whole model over all `n` tokens so far, only to read the logits of the last position. Everything computed for positions `0..n-2` was already computed in step `n-1`, and causal masking guarantees it has not changed: position `i` never attends to anything after `i`.

The only per-position state that later positions need is the key and value of every earlier position in every layer. Queries are needed only for the position being predicted. A **KV cache** stores those keys and values per layer and turns each step into a forward pass over one token.

```text
step without cache           step with cache
tokens 0 1 2 3 4 [5]         tokens [5]
q: all 6 positions           q: position 5 only
k, v: all 6 recomputed       k, v: position 5 computed, 0..4 read from the cache
logits: 6 rows, 5 discarded  logits: 1 row
```

Approximate cost per layer to generate token `n`, in multiply-adds:

| | Linear layers (qkv, proj, fc1, fc2) | Attention scores and mixing |
|---|---|---|
| without cache | `n * 12C^2` | `n^2 * C` |
| with cache | `12C^2` | `n * C` |

Over `M` generated tokens, the uncached loop costs $O(M^2 C^2 + M^3 C)$ and the cached loop $O(M C^2 + M^2 C)$. For GPT-2 small, `12C^2` is 7,077,888 multiply-adds per layer per token, about 85 million over 12 layers. The output head adds `V * C` = 38.6 million per position, so without a cache the head alone repeats that cost for every position of the prefix. The `--benchmark` flag of `examples/gpt2_generate.py` measures both rates side by side on your machine.

### Sampling

The model produces a distribution, not a token. A **sampler** turns the last row of logits into one token id. kael has three controls:

- **Temperature** $\tau$ divides the logits before softmax: $p_j \propto e^{s_j / \tau}$. Below 1 it sharpens the distribution toward the top token, above 1 it flattens it. $\tau = 0$ is the limit case: always pick the largest logit, called **greedy decoding**.
- **Top-k** keeps only the `k` largest logits and gives the rest probability zero.
- **Multinomial sampling** draws one index with probability proportional to its weight.

The effect of temperature on the logits `[2.0, 1.0, 0.5, -1.0]`:

| Temperature | Probabilities |
|-------------|---------------|
| 2.0 | `[0.434, 0.263, 0.205, 0.097]` |
| 1.0 | `[0.609, 0.224, 0.136, 0.030]` |
| 0.5 | `[0.842, 0.114, 0.042, 0.002]` |
| 0 (greedy) | `[1, 0, 0, 0]` |

Top-k exists because of the long tail. GPT-2 has 50,257 tokens. Each of the 50,000 least likely ones may have a tiny probability, but together they can hold a noticeable share of the mass, so pure sampling picks a nonsense token every so often. Removing the tail keeps the text on topic while still allowing variety among the plausible choices.

## Implementation

### The attention module

The constructor validates the head split, builds the mask for the full context once and registers two linear layers. From [`src/nn/transformer.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/nn/transformer.cpp):

```cpp
CausalSelfAttention::CausalSelfAttention(std::int64_t dimension, std::int64_t heads,
                                         std::int64_t max_positions, double dropout, DType dtype)
    : dimension_(dimension), heads_(heads),
      head_dimension_(checked_head_dimension(dimension, heads)), max_positions_(max_positions),
      dropout_(dropout), future_mask_(future_positions_mask(max_positions)),
      query_key_value_(
          register_module("qkv", std::make_shared<Linear>(dimension, 3 * dimension, true, dtype))),
      projection_(
          register_module("proj", std::make_shared<Linear>(dimension, dimension, true, dtype))) {}
```

`checked_head_dimension` throws `ValueError` unless `heads > 0` and `dimension % heads == 0`. Calling it in the member initializer list means a bad configuration fails before any weight is allocated. The mask is a plain member, not a parameter, so it does not appear in `state_dict()` and is never trained. Its size is `[N, N]` booleans: 1 MiB for GPT-2's `N = 1024`, and each of the 12 attention layers holds its own copy.

`CausalSelfAttention` is a `UnaryModule`, so `attention(x)` works like any other module. The one-argument `forward(input)` forwards to `forward(input, nullptr)`, the uncached path.

The forward pass first validates the input and reads `P`, the number of cached positions:

```cpp
KAEL_CHECK_VALUE(input.dim() == 3 && input.size(2) == dimension_,
                 "attention needs an input of shape [B, T, " << dimension_ << "], got "
                                                             << to_string(input.shape()));
const std::int64_t batch = input.size(0);
const std::int64_t length = input.size(1);
const std::int64_t past = cache != nullptr && cache->keys ? cache->keys->size(2) : 0;
KAEL_CHECK_VALUE(past + length <= max_positions_,
                 "sequence of " << past + length << " positions does not fit a context of "
                                << max_positions_);
```

Then the computation:

```cpp
const Tensor parts =
    query_key_value_->forward(input).view({batch, length, 3, heads_, head_dimension_});
const Tensor queries = parts.select(2, 0).transpose(1, 2);
Tensor keys = parts.select(2, 1).transpose(1, 2);
Tensor values = parts.select(2, 2).transpose(1, 2);
if (cache != nullptr) {
    if (cache->keys) {
        keys = cat({*cache->keys, keys}, 2);
        values = cat({*cache->values, values}, 2);
    }
    cache->keys = keys;
    cache->values = values;
}

if (future_mask_.device() != input.device()) {
    future_mask_ = future_mask_.to(input.device());
}
const Tensor mask = future_mask_.slice(0, past, past + length).slice(1, 0, past + length);
const Tensor scores =
    matmul(queries, keys.mT()) / std::sqrt(static_cast<double>(head_dimension_));
const Tensor masked = masked_fill(scores, mask, -std::numeric_limits<double>::infinity());
const Tensor weights = functional::dropout(softmax(masked, -1), dropout_, is_training());
const Tensor mixed = matmul(weights, values).transpose(1, 2).contiguous();
const Tensor output = projection_->forward(mixed.view({batch, length, dimension_}));
return functional::dropout(output, dropout_, is_training());
```

Design choices and the alternatives they reject:

- **One fused `Linear(C, 3C)` instead of three `Linear(C, C)`.** One matrix multiply with three times the columns makes better use of the matmul kernel than three small ones, and it matches the GPT-2 checkpoint. The split into q, k, v is a `select`, which is a view and copies nothing.
- **Views until the last moment.** `select` plus `transpose` produce non-contiguous views, and the code passes them to `matmul` as they are. The only explicit copy is `contiguous()` before the final `view`: after `transpose(1, 2)` the heads of one position are no longer adjacent in memory, and kael's `view` only accepts contiguous tensors (it throws `ValueError` and suggests `reshape` otherwise). `matmul` itself is not copy-free for these inputs: it flattens the batch dimensions with `reshape` (in `as_batch_of_matrices` in [`src/ops/linalg.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/ops/linalg.cpp)), and `reshape` is `contiguous().view()`, so a strided query or key tensor is copied into a packed buffer inside the call. The copies are of size `B * H * T * D`, small next to the weight matrices.
- **The mask moves lazily.** It lives on the CPU until the first input arrives on another device. The forward pass then moves it once and keeps the moved copy, so a model moved with `to(Device::CUDA)` pays the transfer once. The alternative, registering the mask as a buffer that `Module::to` moves, would need a buffer concept that kael's `Module` does not have.
- **`masked_fill` with minus infinity, not a large negative number.** With `-inf`, the masked weights are exactly zero in every dtype, independent of the scale of the scores. A finite value such as `-1e9` only underflows to zero when it is far enough below the other scores, which turns a correctness property into an assumption about magnitudes.
- **Dropout twice.** Dropout is applied to the attention weights and to the projected output. It is a no-op when `dropout == 0.0` or the module is in eval mode (see `functional::dropout` in chapter 07). GPT-2 small and `char_gpt` both use `dropout = 0.0`.

The input check rejects anything other than `[B, T, C]`, and a second check rejects `P + T > N`: the mask, and the position embedding table, have no rows past `N`.

#### The mask slice with a cache

The slice `rows P..P+T, columns 0..P+T` is what makes the same mask serve both a full sequence and a cached step. With `N = 4`, `P = 2` and `T = 2`:

```text
full mask [4, 4]                   slice rows 2..4, columns 0..4
     k0 k1 k2 k3
q0 [ F  T  T  T ]
q1 [ F  F  T  T ]
q2 [ F  F  F  T ]   -> new query 0   [ F  F  F  T ]
q3 [ F  F  F  F ]   -> new query 1   [ F  F  F  F ]
```

New query 0 sits at absolute position 2 and may see keys 0, 1 and 2. New query 1 sits at position 3 and sees everything. For a single new token (`T = 1`) the slice is one row with no `true` entries: the newest token may see everything before it.

### The MLP

From [`src/nn/transformer.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/nn/transformer.cpp):

```cpp
Tensor TransformerMlp::forward(const Tensor& input) {
    return dropout_->forward(fc2_->forward(activation_->forward(fc1_->forward(input))));
}
```

Shapes: `[B, T, C]` to `[B, T, 4C]` (`fc1`), GELU elementwise, `[B, T, 4C]` to `[B, T, C]` (`fc2`), dropout. The constructor registers `fc1`, `activation`, `fc2` and `dropout`. Only `fc1` and `fc2` have parameters, so only they appear in `state_dict()`.

The two GELU forms are element functions in [`src/internal/element_functions.h`](https://github.com/JetBerri/kael.cpp/blob/main/src/internal/element_functions.h), run by the tensor iterator of chapter 04 on the CPU and by the CUDA kernels of chapter 09 (`KAEL_HD` marks them host and device):

```cpp
struct GeluExact : Tagged<ElementOp::GeluExact> {
    template <typename T> KAEL_HD T operator()(T x) const {
        return T(0.5) * x * (1 + std::erf(x / std::numbers::sqrt2_v<T>));
    }
};
struct GeluTanh : Tagged<ElementOp::GeluTanh> {
    template <typename T> KAEL_HD T operator()(T x) const {
        const T scale = std::sqrt(2 / std::numbers::pi_v<T>);
        return T(0.5) * x * (1 + std::tanh(scale * (x + T(0.044715) * x * x * x)));
    }
};
```

The `GELU` module in [`include/kael/nn/activation.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/nn/activation.h) takes a flag, `GELU(bool approximate = false)`, and calls `functional::gelu(input, approximate_)`. `TransformerMlp` passes `GptConfig::tanh_gelu` as that flag.

GPT-2 was trained with the tanh approximation, so `GptConfig` defaults to `tanh_gelu = true`. Using the exact form with GPT-2 weights shifts the logits slightly. `char_gpt` trains from scratch and sets `tanh_gelu = false`. The two curves differ by less than 0.0005 everywhere, so either works for a new model, but a pretrained model must use the form it was trained with. PyTorch exposes the same choice as `torch.nn.GELU(approximate="tanh")`, and Hugging Face calls the GPT-2 variant `gelu_new`.

### The residual block

From [`src/nn/transformer.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/nn/transformer.cpp):

```cpp
Tensor TransformerBlock::forward(const Tensor& input, LayerCache* cache) {
    const Tensor attended = input + attention_->forward(ln1_->forward(input), cache);
    return attended + mlp_->forward(ln2_->forward(attended));
}
```

Two lines implement the pre-LN picture from the Concepts section. The block passes its `LayerCache` to the attention layer and to nothing else: the MLP and the layer norms work per position and need no state from earlier steps.

All three kinds of layer norm in the model (`ln1`, `ln2`, `ln_final`) use `eps = 1e-5`, the GPT-2 value. The constructor registers them under the names `ln1`, `attn`, `ln2` and `mlp`.

### The GPT configuration

From [`include/kael/nn/transformer.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/nn/transformer.h):

```cpp
struct GptConfig {
    std::int64_t vocab_size = 50257;
    std::int64_t max_positions = 1024;
    std::int64_t layers = 12;
    std::int64_t heads = 12;
    std::int64_t dimension = 768;
    double dropout = 0.0;
    bool tanh_gelu = true;
    DType dtype = DType::F32;

    [[nodiscard]] static GptConfig gpt2_small() { return {}; }
};
```

The defaults are GPT-2 small, so `gpt2_small()` returns a default-constructed config. A plain struct with default member values lets callers write `GptConfig config; config.layers = 4;` and change only what differs, the same style as Hugging Face's `GPT2Config`. `dtype` is passed to every layer, so a model built with `DType::F64` runs gradcheck in double precision (the tests do this).

The `GPT` constructor runs the config through `checked_config`, which throws `ValueError` when `layers <= 0`, then registers the submodules under fixed names. These names are the keys of `state_dict()` and are what chapter 12 maps GPT-2 checkpoint names onto:

```text
token_embedding.weight          [V, C]
position_embedding.weight       [N, C]
blocks.{i}.ln1.weight / .bias   [C]
blocks.{i}.attn.qkv.weight      [3C, C]   bias [3C]
blocks.{i}.attn.proj.weight     [C, C]    bias [C]
blocks.{i}.ln2.weight / .bias   [C]
blocks.{i}.mlp.fc1.weight       [4C, C]   bias [4C]
blocks.{i}.mlp.fc2.weight       [C, 4C]   bias [C]
ln_final.weight / .bias         [C]
```

`Linear` stores its weight as `[out_features, in_features]` and computes `input @ weight.T() + bias` (chapter 07), the same convention as PyTorch's `nn.Linear`. There is no separate output head parameter, because of weight tying.

#### Parameter count

The `ParameterCountFormula` test states the count directly. Per block:

```text
qkv   3C*C + 3C
proj   C*C +  C
fc1   4C*C + 4C
fc2   4C*C +  C
ln1, ln2   2 * 2C
total per block = 12C^2 + 13C
```

The whole model is

$$
V C + N C + L (12 C^2 + 13 C) + 2 C
$$

For GPT-2 small, term by term:

```text
V*C                      = 50257 * 768             =  38,597,376
N*C                      =  1024 * 768             =     786,432
L*(12C^2 + 13C)          = 12 * (7,077,888 + 9,984) =  85,054,464
2C                       =                                  1,536
total                                              = 124,439,808
```

`Gpt2SmallHasTheKnownParameterCount` checks this number. For `char_gpt` (`V = 65`, `N = 64`, `L = 4`, `C = 128`) the formula gives `8,320 + 8,192 + 4 * 198,272 + 256 = 809,856`, which is what the example prints.

#### Weight initialization

From [`src/nn/transformer.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/nn/transformer.cpp):

```cpp
void GPT::initialize_weights() {
    const double residual_std = kInitialStd / std::sqrt(2.0 * static_cast<double>(config_.layers));
    Tensor token_weight = token_embedding_->weight();
    Tensor position_weight = position_embedding_->weight();
    init::normal_(token_weight, 0.0, kInitialStd);
    init::normal_(position_weight, 0.0, kInitialStd);
    const auto named = named_parameters();
    for (const auto& [name, parameter] : named) {
        if (name.rfind("blocks.", 0) != 0 || name.find(".ln") != std::string::npos) {
            continue;
        }
        Tensor target = parameter;
        if (name.ends_with(".bias")) {
            init::zeros_(target);
        } else {
            const bool residual =
                name.ends_with(".attn.proj.weight") || name.ends_with(".mlp.fc2.weight");
            init::normal_(target, 0.0, residual ? residual_std : kInitialStd);
        }
    }
}
```

`checked_config` has already guaranteed `layers > 0`, so the square root never sees zero. The rules, with `kInitialStd = 0.02`:

| Parameter | Initial value |
|-----------|---------------|
| embeddings | normal, std 0.02 |
| linear biases in blocks | zero |
| `qkv.weight`, `fc1.weight` | normal, std 0.02 |
| `attn.proj.weight`, `mlp.fc2.weight` | normal, std `0.02 / sqrt(2L)` |
| every layer norm | weight 1, bias 0 (the `LayerNorm` default, skipped by the loop) |

The loop selects parameters by name. That is less type-safe than walking the module tree, but it uses exactly the names that chapter 12 relies on, so a renamed parameter breaks a test instead of silently keeping its default initialization.

The two scaled weights are the last layer of each sublayer, the ones that write into the residual stream. The stream receives `2L` additions. If each addition is independent with variance $s^2$, the stream ends with variance

$$
\operatorname{Var}(x_{\text{final}}) \approx 2L \cdot s^2
$$

Scaling $s$ by $1 / \sqrt{2L}$ keeps the total near $0.02^2$ regardless of depth. For `L = 8` the scaled std is `0.02 / 4 = 0.005`, which `ResidualProjectionsStartSmaller` checks. For GPT-2 small, `L = 12` gives `0.02 / sqrt(24) = 0.0041`.

A small std overall makes the initial logits close to zero, so the first prediction is close to uniform. The cross-entropy of a uniform prediction over `V` classes is

$$
-\ln \frac{1}{V} = \ln V
$$

so the first loss should be close to $\ln V$. `InitialLossMatchesUniformPrediction` checks this for `V = 65`: $\ln 65 = 4.17$. A first loss far above $\ln V$ means the initialization is too large and the model starts out confidently wrong.

### The forward pass and weight tying

`GPT::forward` validates its input, prepares the cache, and runs the stack. The checks and the cache setup:

```cpp
KAEL_CHECK_VALUE(indices.dim() == 2,
                 "GPT needs indices of shape [B, T], got " << to_string(indices.shape()));
KAEL_CHECK_TYPE(indices.dtype() == DType::I64,
                "GPT needs int64 indices, got " << indices.dtype());
if (cache != nullptr && cache->empty()) {
    cache->resize(blocks_.size());
}
KAEL_CHECK_VALUE(cache == nullptr || cache->size() == blocks_.size(),
                 "cache has " << cache->size() << " layers but the model has "
                              << blocks_.size());
const std::int64_t past =
    cache != nullptr && cache->front().keys ? cache->front().keys->size(2) : 0;
```

Then the computation:

```cpp
const Tensor positions =
    arange(static_cast<double>(past), static_cast<double>(past + length), 1.0, DType::I64)
        .to(indices.device());
Tensor x = dropout_->forward(token_embedding_->forward(indices) +
                             position_embedding_->forward(positions));
for (std::size_t i = 0; i < blocks_.size(); ++i) {
    x = blocks_[i]->forward(x, cache != nullptr ? &(*cache)[i] : nullptr);
}
// Multiplying by the weight transposed would copy the whole embedding on every step.
return matmul(token_embedding_->weight(), ln_final_->forward(x).mT()).mT().contiguous();
```

Shapes:

| Step | Shape |
|------|-------|
| `indices` | `[B, T]`, must be `I64` |
| `token_embedding_->forward(indices)` | `[B, T, C]` |
| `positions` | `[T]` with values `P, P+1, ..., P+T-1` |
| `position_embedding_->forward(positions)` | `[T, C]`, broadcast over `B` |
| after the blocks | `[B, T, C]` |
| `ln_final_->forward(x).mT()` | `[B, C, T]` |
| `matmul(weight [V, C], [B, C, T])` | `[B, V, T]` (the weight broadcasts over `B`) |
| `.mT().contiguous()` | `[B, T, V]` |

Positions start at `P`, the number of cached positions, so a token processed alone in step `k` of generation gets the same position embedding it would get inside the full sequence. GPT-2 uses learned **absolute position embeddings**: a table with one trainable row per position. Later models replaced this with rotary or relative schemes; GPT-2 weights require the table.

The comment explains the operand order. The direct form `matmul(h, weight.T())` multiplies by a transposed `[C, V]` view. Because `matmul` packs its operands with `reshape` (see the attention design notes), it would make a contiguous copy of the whole embedding, 154 MB in F32 for GPT-2, on every forward call. Computing $(W h^\top)^\top$ keeps the large operand in its stored row-major layout and transposes only the hidden states and the result. When `B = 1`, the weight broadcast to `[1, V, C]` is still contiguous, because kael's `is_contiguous` ignores the stride of size-1 dimensions, so no copy of the weight happens. With `B > 1` the broadcast weight has a stride-0 batch dimension and `matmul` packs `B` copies of it; that is acceptable for training batches of a small model such as `char_gpt` and is the price of the simple batched kernel. Generation in kael runs with `B = 1` in the examples and the server.

`WeightTyingKeepsASingleEmbeddingParameter` checks that exactly one registered parameter shares storage with `token_embedding_weight()`.

### The KV cache

#### Layout

From [`include/kael/nn/transformer.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/nn/transformer.h):

```cpp
struct LayerCache {
    std::optional<Tensor> keys;
    std::optional<Tensor> values;
};

using KvCache = std::vector<LayerCache>;
```

One `LayerCache` per block. `std::optional` encodes the empty state: before the first call, a layer has no keys yet, and `P` is 0. `keys` and `values` have shape `[B, H, P, D]`, the same layout attention uses internally, so appending is a `cat` along dimension 2 and no transposition is needed when reading:

```text
KvCache (L entries)
  [0] keys   [B, H, P, D]   values [B, H, P, D]
  [1] keys   [B, H, P, D]   values [B, H, P, D]
  ...
  [L-1] keys [B, H, P, D]   values [B, H, P, D]
```

Memory: `2 * L * B * P * C` elements (`H * D = C`). For GPT-2 small at a full context (`P = 1024`, `B = 1`, F32), that is `2 * 12 * 1024 * 768 * 4 = 75,497,472` bytes, 72 MiB.

#### Protocol

As implemented in `GPT::forward` and `CausalSelfAttention::forward`:

1. The caller passes an empty `KvCache`. `GPT::forward` resizes it to `L` entries.
2. A cache of another size throws `ValueError` (`RejectsACacheOfAnotherModel`).
3. `P` is read from `cache->front().keys->size(2)`, or 0 when empty.
4. Each attention layer concatenates its new keys and values to the cached ones, stores the result back, and attends over all `P + T` keys.
5. The mask slice `rows P..P+T, columns 0..P+T` lets new query `i` see keys `0..P+i`.

A trace with `L = 2`, `B = 1`, a 4-token prefix and then single tokens, as in `CacheGivesTheSameLogitsAsTheFullSequence`:

```text
call                         T   P before   cache[i].keys after   logits
forward(tokens 0..3, &c)     4   0          [1, H, 4, D]          [1, 4, V]
forward(token 4, &c)         1   4          [1, H, 5, D]          [1, 1, V]
forward(token 5, &c)         1   5          [1, H, 6, D]          [1, 1, V]
...
forward(token 8, &c)         1   8          [1, H, 9, D]          [1, 1, V]
```

The test runs a 4-token prefix, then feeds 5 more tokens one at a time, and checks that each step's logits equal the corresponding row of a single full-sequence forward. It ends by checking that the cache has 2 entries and 9 cached positions.

PyTorch's `nn.MultiheadAttention` takes no cache argument. Hugging Face `transformers` passes `past_key_values` through the model in the same way as kael, one entry per layer.

### Generation

#### The loop

From [`src/nn/transformer.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/nn/transformer.cpp):

```cpp
for (std::int64_t step = 0; step < options.max_new_tokens; ++step) {
    const std::int64_t total = sequence.size(1);
    const std::int64_t window = std::min(total, context);
    const std::int64_t cached =
        !cache.empty() && cache.front().keys ? cache.front().keys->size(2) : 0;
    const auto run_model = [&] {
        if (!options.use_cache) {
            return model(sequence.narrow(1, total - window, window));
        }
        if (cached > 0 && cached < context) {
            return model.forward(sequence.narrow(1, total - 1, 1), &cache);
        }
        cache.clear();
        return model.forward(sequence.narrow(1, total - window, window), &cache);
    };
    const Tensor logits = run_model();
    const Tensor next = sample_next(logits.select(1, logits.size(1) - 1), options);
    sequence = cat({sequence, next}, 1);
    if (options.on_token && !options.on_token(next)) {
        break;
    }
}
```

Three paths per step:

- Without cache: run the last `window = min(total, N)` tokens. Older tokens fall out of the window.
- With a cache that has room: run only the newest token.
- With an empty or full cache: clear it and run the last `window` tokens to refill it. The first step always takes this path and processes the whole prompt at once.

A trace with `N = 4`, a 3-token prompt and `max_new_tokens = 4`, cache enabled:

```text
step  total  cached  path            tokens run       positions   cached after
0     3      0       refill          0 1 2            0 1 2       3
1     4      3       newest token    3                3           4
2     5      4       refill (full)   1 2 3 4          0 1 2 3     4
3     6      4       refill (full)   2 3 4 5          0 1 2 3     4
```

Past the context length, the full-cache path runs on every step: each new token shifts the window by one, every position embedding changes, and the cached keys are no longer valid. Generation keeps working (`KeepsGoingPastTheContextLength`), but at the uncached cost. A model with relative positions could keep its cache across the shift; GPT-2's absolute positions cannot.

#### Setup around the loop

Before the loop, `generate` validates its options and the prompt:

```cpp
KAEL_CHECK_VALUE(options.max_new_tokens >= 0,
                 "max_new_tokens must not be negative, got " << options.max_new_tokens);
KAEL_CHECK_VALUE(options.temperature >= 0.0,
                 "temperature must not be negative, got " << options.temperature);
KAEL_CHECK_VALUE(options.top_k >= 0, "top_k must not be negative, got " << options.top_k);
KAEL_CHECK_VALUE(prompt.dim() == 2 && prompt.size(1) > 0,
                 "generate needs a prompt of shape [B, T], got " << to_string(prompt.shape()));

const autograd::NoGradGuard guard;
const TrainingModeRestorer restorer(model);
model.eval();
```

`NoGradGuard` disables gradient recording: generation never calls `backward`, and recording a graph for 500 steps would keep every intermediate tensor alive. `model.eval()` turns dropout off. `TrainingModeRestorer` is a small RAII class that records `model.is_training()` in its constructor and calls `model.train(was_training_)` in its destructor, so the previous mode comes back when `generate` returns or throws (`AThrowingReceiverRestoresTheTrainingMode`). The alternative, asking callers to restore the mode themselves, fails silently: a training loop that samples text halfway through would continue with dropout off.

`GenerateOptions`, from [`include/kael/nn/transformer.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/nn/transformer.h):

| Field | Default | Meaning |
|-------|---------|---------|
| `max_new_tokens` | 16 | Tokens to append. Negative throws. |
| `temperature` | 1.0 | 0 means greedy. Negative throws. |
| `top_k` | 0 | 0 disables top-k. Negative throws. |
| `use_cache` | `true` | Use the KV cache. |
| `on_token` | empty | Called with each new `[B, 1]` token tensor. Returning `false` stops the loop. |

`GenerateOptions` is an aggregate, so callers use C++20 designated initializers: `generate(model, prompt, {.max_new_tokens = 6, .temperature = 0.0})`. The callback runs after the token is appended, so the token that triggered a stop is part of the returned sequence. [Chapter 13](@/book/13-serving-and-tooling.md) uses this callback to stream tokens and cancel generations in the HTTP server.

### Temperature, top-k and multinomial

From [`src/nn/transformer.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/nn/transformer.cpp):

```cpp
Tensor sample_next(const Tensor& last_logits, const GenerateOptions& options) {
    if (options.temperature == 0.0) {
        return argmax(last_logits, -1, true);
    }
    Tensor logits = last_logits / options.temperature;
    if (options.top_k > 0 && options.top_k < logits.size(-1)) {
        const Tensor smallest_kept =
            topk(logits, options.top_k).values.select(-1, options.top_k - 1);
        logits = masked_fill(logits, logits < smallest_kept.unsqueeze(-1),
                             -std::numeric_limits<double>::infinity());
    }
    return multinomial(softmax(logits, -1), 1, true);
}
```

`last_logits` has shape `[B, V]`. All three branches return `[B, 1]` int64, which `cat` appends to the `[B, total]` sequence. `argmax(..., -1, true)` keeps the reduced dimension for that reason.

- **Temperature 0** is handled as `argmax`, which avoids a division by zero and makes greedy decoding deterministic.
- **Top-k** finds the `k`-th largest value per row with `topk(...).values.select(-1, k - 1)`, then masks everything strictly below it. Values equal to the threshold are kept, so ties can let more than `k` tokens through. A `top_k` of `V` or more is a no-op, which is why the server can accept a `top_k` up to 1,000,000.
- **Multinomial** draws one index per row with probability proportional to its weight.

A worked example with `V = 4`, logits `[2.0, 1.0, 0.5, -1.0]`, temperature 0.5, `top_k = 2`:

```text
divide by 0.5          [ 4.0  2.0  1.0  -2.0 ]
2nd largest            2.0
mask values < 2.0      [ 4.0  2.0  -inf -inf ]
softmax                [ 0.881 0.119 0.000 0.000 ]
```

`multinomial` in [`src/random.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/random.cpp) works on the CPU in F64:

1. Copy the weights with `weights.detach().cpu().to(DType::F64).contiguous()`.
2. For each row, check that every weight is finite and non-negative and that at least one is positive. Otherwise throw `ValueError`.
3. Build the cumulative sum with `std::partial_sum`.
4. Draw `u` uniform in `[0, 1)` from the global generator, under its lock, and scale it by the last cumulative value.
5. Return the first index whose cumulative sum exceeds `u` (`std::upper_bound`). If rounding puts `u` at the very end, take the last index, and step back past any zero-weight entries so a masked token is never returned.
6. Move the result back to the input's device.

With the cumulative sums `[0.881, 1.0, 1.0, 1.0]`, any `u` below 0.881 picks token 0 and the rest pick token 1. Because sampling uses the global generator, `manual_seed` makes generation reproducible, and the cached and uncached paths produce the same tokens for the same seed (`SamplingMatchesWithAndWithoutCacheForTheSameSeed`). PyTorch's `torch.multinomial` has the same signature, `multinomial(input, num_samples, replacement)`.

## Training a character-level model

[`examples/char_gpt.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/examples/char_gpt.cpp) trains a small GPT on Tiny Shakespeare, about 1 MB of plays, with one token per byte. It is the smallest end-to-end test of everything above: the model, the loss, the optimizer, the checkpoint format of chapter 12 and the sampler.

### Data

`scripts/download/shakespeare.sh` downloads `input.txt` from Andrej Karpathy's `char-rnn` repository into `data/shakespeare` (or the directory given as its first argument) through `fetch` from `scripts/download/common.sh`. `fetch` keeps a file that already has the expected SHA-256, downloads to `input.txt.partial`, and moves the file into place only after its checksum matches. The `Vocabulary` class marks which of the 256 byte values occur in the text, then numbers the present ones from 0 in byte order. Tiny Shakespeare has 65 distinct characters, so `V = 65`. The encoded text is split 90 percent for training and 10 percent for validation.

`sample_batch` draws 32 random start offsets with `randint`, copies 64 tokens from each as inputs and the same 64 tokens shifted by one as targets:

```text
text     F  i  r  s  t     C  i  t  ...
inputs   F  i  r  s  t     C  i        (64 tokens)
targets  i  r  s  t     C  i  t        (64 tokens, shifted by one)
```

Every position in every row is a training example: position `t` predicts token `t + 1` from tokens `0..t`. The causal mask is what makes all 64 predictions in a row valid at once. Inputs have shape `[32, 64]`; targets are stored flat as `[2048]`, ready for the loss.

### Model and loss

```cpp
kael::nn::GptConfig config;
config.vocab_size = vocabulary.size();
config.max_positions = kContext;
config.layers = 4;
config.heads = 4;
config.dimension = 128;
config.tanh_gelu = false;
```

That is `C = 128`, `H = 4`, `D = 32`, `N = 64`, 809,856 parameters. The loss reshapes the logits from `[32, 64, 65]` to `[2048, 65]` and calls `cross_entropy` with the `[2048]` targets. From [`examples/char_gpt.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/examples/char_gpt.cpp):

```cpp
Tensor batch_loss(kael::nn::GPT& model, const Batch& batch, std::int64_t vocabulary) {
    return kael::nn::cross_entropy(model(batch.inputs).view({kBatchSize * kContext, vocabulary}),
                                   batch.targets);
}
```

The `view` works because `GPT::forward` ends with `.contiguous()`.

### Training loop

| Constant | Value |
|----------|-------|
| `kBatchSize` | 32 |
| `kContext` | 64 |
| `kLearningRate` | `1e-3` |
| `kWarmupSteps` | 200 |
| `kMaxGradientNorm` | 1.0 |
| `kEvaluateEvery` | 250 steps |
| `kEvaluationBatches` | 20 |
| `kSampleLength` | 500 characters |
| default `--steps` | 5000 |

Each step:

1. Sample a batch from the training split and compute the loss.
2. Zero the gradients and run backward.
3. Clip the global gradient norm to 1.0 with `clip_grad_norm_`.
4. Step `AdamW` (weight decay 0.01, the `AdamWOptions` default).
5. Step a `WarmupCosineLR` schedule: linear warmup over 200 steps, then a cosine decay to the end.

Every 250 steps, including step 0, it averages the loss of 20 validation batches under `NoGradGuard` and in eval mode, then switches back to training mode. When the validation loss improves it saves the `state_dict()` to `best.safetensors` in the data directory ([chapter 12](@/book/12-model-loading.md) explains the format). At the end it reloads the best checkpoint and generates 500 characters from the prompt `"ROMEO:\n"` with temperature 0.8 and `top_k = 40`.

The arguments are `--steps N`, `--device cpu|cuda` and an optional data directory, and `--help` prints the usage line. An unknown option or device name is an error. The seed is fixed with `manual_seed(1337)`.

## Edge cases and pitfalls

- **Head split.** `C` must divide by `H`. `CausalSelfAttention(7, 2, 16)` and `CausalSelfAttention(8, 0, 16)` throw `ValueError`.
- **Context overflow.** `P + T > N` throws `ValueError` in both `GPT::forward` and the attention layer. Without the check, `position_embedding_` would read past its table and the mask slice would be too short. `generate` avoids the error by windowing.
- **Index validation.** Indices must be 2-D (`ValueError` otherwise) and `I64` (`TypeError` otherwise). An id outside `[0, V)` throws `IndexError` from the embedding lookup (`RejectsSequencesThatDoNotFit`).
- **At least one layer.** `GptConfig::layers <= 0` throws `ValueError` in `checked_config`, before anything is allocated (`NeedsAtLeastOneLayer`).
- **Empty prompt.** `generate` needs a prompt with at least one token. A prompt of shape `[1, 0]`, for example the encoding of an empty string, throws `ValueError`.
- **Non-contiguous merge.** Removing `contiguous()` before the final `view` in attention makes `view` throw `ValueError`, because the transposed tensor is not contiguous. `reshape` would also work, at the same copy cost.
- **Masked rows.** Every row of the causal mask keeps its diagonal, so softmax never sees a row of only minus infinity. A custom mask that hides a whole row would produce NaN.
- **Cache growth by `cat`.** Each cached step allocates new key and value tensors of size `P + 1` and copies the old ones. The copy is `O(P * C)` per layer, the same order as the attention itself, but it allocates on every step. A preallocated `[B, H, N, D]` buffer written in place would avoid that at the cost of a fixed 72 MiB per sequence for GPT-2.
- **The first cached call stores views.** On the first call the cache holds `select` and `transpose` views of the `qkv` output, so the whole `[B, T, 3C]` projection buffer stays alive until the next `cat` replaces them.
- **A cache across models or prompts.** A `KvCache` is valid only for the model and the token prefix that filled it. Reusing it with a different prompt silently attends to stale keys. `generate` creates a fresh cache per call.
- **Full logits for the prompt.** `GPT::forward` returns logits for every input position, and `generate` reads only the last row. On the first step of a long prompt that is a `[1, T, V]` tensor: for `T = 1024` and GPT-2's vocabulary, about 206 MB in F32, most of it discarded.
- **Temperature near zero.** A positive temperature small enough to overflow the scaled logits to infinity makes softmax return NaN, which `multinomial` rejects with `ValueError`. The server in chapter 13 accepts only 0 or at least 0.001.
- **Warmup longer than the run.** `WarmupCosineLR` requires `warmup_steps < total_steps`. `char_gpt` checks `--steps` against its 200 warmup steps while parsing the arguments, so `char_gpt --steps 200` or fewer stops with "--steps must exceed the 200 warmup steps" before it loads any data.
- **Training mode.** `generate` turns dropout off and restores the previous mode afterwards, even on an exception. Calling `model(...)` directly for evaluation still needs `model.eval()`.

> [!WARNING]
> Loading pretrained GPT-2 weights into a model with `tanh_gelu = false` succeeds without any error. The logits are close to the reference but not equal, and generated text degrades in ways that are hard to attribute. Keep the default `tanh_gelu = true` for GPT-2 checkpoints.

## How it is tested

[`tests/test_transformer.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_transformer.cpp) (GoogleTest) uses a tiny F64 config from `tiny_config()`: `V = 11`, `N = 16`, `L = 2`, `H = 2`, `C = 8`. F64 makes gradcheck meaningful and keeps the cache comparisons tight.

| Test | What it proves |
|------|----------------|
| `Attention.OutputKeepsTheInputShape` | `[B, T, C]` in and out, wrong ranks and context overflow rejected |
| `Attention.DimensionMustDivideByHeads` | head split validation |
| `Attention.LaterPositionsDoNotAffectEarlierOnes` | adding 100 to position 5 leaves outputs 0 to 4 equal within the F64 tolerance of `allclose` and changes output 5: the mask works |
| `Attention.Gradcheck`, `Block.GradcheckThroughTheWholeBlock` | analytic gradients of attention and a full block match finite differences |
| `Gpt.LogitsHaveTheExpectedShape`, `Gpt.RejectsSequencesThatDoNotFit` | output shape and input validation |
| `Gpt.ParameterNamesFollowTheGuide` | the `state_dict` names listed above |
| `Gpt.WeightTyingKeepsASingleEmbeddingParameter` | no separate output head |
| `Gpt.ParameterCountFormula`, `Gpt.Gpt2SmallHasTheKnownParameterCount` | `12C^2 + 13C` per block and 124,439,808 for GPT-2 small (only the GPT-2 small test skips under sanitizers, where the caching allocator is off) |
| `Gpt.InitialLossMatchesUniformPrediction` | initial loss within 0.1 of `ln(65)` |
| `Gpt.ResidualProjectionsStartSmaller` | std 0.02 and `0.02 / sqrt(2L)`, zero biases |
| `Gpt.OverfitsASmallSequence` | 300 AdamW steps drive the loss on one sequence below 0.05 |
| `Gpt.NeedsAtLeastOneLayer` | `layers = 0` is rejected |
| `Gpt.CacheGivesTheSameLogitsAsTheFullSequence`, `Gpt.CacheCannotOutgrowTheContext`, `Gpt.RejectsACacheOfAnotherModel` | KV cache correctness and limits |
| `Generate.GreedyMatchesWithAndWithoutCache`, `Generate.SamplingMatchesWithAndWithoutCacheForTheSameSeed` | the cache does not change results |
| `Generate.KeepsGoingPastTheContextLength` | windowing past `N`: 40 new tokens on a 16-position model, with and without the cache |
| `Generate.TopKOneIsGreedy` | `top_k = 1` equals argmax |
| `Generate.RestoresTrainingModeAndHandlesBatches`, `Generate.ValidatesOptions` | mode restore, `B > 1`, option checks |
| `GptGenerate.ReportsEveryNewTokenInOrder`, `GptGenerate.AReceiverCanStopTheGeneration`, `GptGenerate.AThrowingReceiverRestoresTheTrainingMode` | the `on_token` callback |

`OverfitsASmallSequence` is the cheapest whole-model test that catches a broken backward pass. A model that cannot memorize nine tokens in 300 steps has a bug in some gradient, even if every individual gradcheck passes.

[`python/tests/test_gpt.py`](https://github.com/JetBerri/kael.cpp/blob/main/python/tests/test_gpt.py) compares against PyTorch when `torch`, `transformers` and `safetensors` are installed:

- `test_attention_matches_torch_scaled_dot_product` builds the masked attention from kael ops, using the input as query, key and value, and compares it with `torch.nn.functional.scaled_dot_product_attention(..., is_causal=True)` to `1e-10`.
- `test_logits_match_a_small_huggingface_model` creates a random 3-layer `GPT2LMHeadModel`, saves its weights, loads them into a kael `GPT` and compares logits (`atol=2e-5`, `rtol=2e-4`).
- `test_greedy_generation_matches_huggingface` compares 10 greedy tokens with Hugging Face `generate`, with and without the cache.
- `test_model_from_python_trains_with_python_optimizer` trains a one-layer model built from Python with `kael.optim.AdamW` for 200 steps and expects the loss below 0.2.

## Build it yourself

1. Write `CausalSelfAttention` without a cache: fused `qkv`, the `[B, T, 3, H, D]` view, the scaled scores, the `triu` mask and the output projection. Checkpoint: `Attention.OutputKeepsTheInputShape` and `Attention.LaterPositionsDoNotAffectEarlierOnes` pass.
2. Run gradcheck on the attention module in F64. Checkpoint: `Attention.Gradcheck` passes. A failure here usually means a missing `contiguous()` or a wrong transpose.
3. Add `GELU` with both formulas and `TransformerMlp`. Checkpoint: GELU of 0 is 0, GELU of a large positive value is close to the value, and the two variants differ by less than `5e-4`.
4. Write `TransformerBlock` with pre-layer-norm residuals. Checkpoint: `Block.GradcheckThroughTheWholeBlock` passes.
5. Write `GptConfig` and `GPT` with the registered names from this chapter, tied output weights and `initialize_weights`. Checkpoint: `ParameterNamesFollowTheGuide`, `WeightTyingKeepsASingleEmbeddingParameter`, `ParameterCountFormula`, `ResidualProjectionsStartSmaller` and `InitialLossMatchesUniformPrediction` pass.
6. Train the tiny model on one sequence. Checkpoint: `Gpt.OverfitsASmallSequence` passes.
7. Add `LayerCache` and `KvCache`, the `past` offset in `GPT::forward` and in the mask slice. Checkpoint: `CacheGivesTheSameLogitsAsTheFullSequence` passes.
8. Write `sample_next` and `generate` with the three-path loop, `NoGradGuard` and `TrainingModeRestorer`. Checkpoint: all `Generate` and `GptGenerate` tests pass.
9. Build and run the example:

   ```bash
   scripts/download/shakespeare.sh
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKAEL_BUILD_EXAMPLES=ON
   cmake --build build --target char_gpt
   build/examples/char_gpt --steps 5000
   ```

   Checkpoint: the first line reports `vocabulary 65, parameters 809856`, the validation loss at step 0 is near `ln(65) = 4.17` and falls as training proceeds (one run printed 4.18 at step 0 and 2.32 at step 250), and the final sample has the shape of a play: speaker names in capitals followed by lines of text.

> [!TIP]
> If the loss at step 0 is far from 4.17, check the initialization before anything else. If it starts right but does not fall, check the targets: an off-by-one in `sample_batch` that makes the targets equal to the inputs lets every position read its own answer, and the loss collapses toward zero on both splits while the samples stay meaningless.

## Check your understanding

<details>
<summary>Why does the attention view use [B, T, 3, H, D] and not [B, T, H, 3, D]?</summary>

The fused `qkv` layer lays its `3C` output columns out as all of `q`, then all of `k`, then all of `v`, each subdivided into `H` heads of `D` columns. In a row-major view the last index varies fastest, so the index that selects q, k or v must come before the head index. `[B, T, H, 3, D]` would interpret columns 0 to 3 as q, k and v of head 0, which mixes the three projections. The same layout is used by GPT-2's `c_attn`, so chapter 12 can load the weights without reordering columns.

</details>

<details>
<summary>The first row of the causal mask hides every key but one. Why can no row ever hide all keys?</summary>

The mask is `triu(ones, 1)`, which is `true` only strictly above the diagonal. Every query `i` can always see key `i`, so each row of scores has at least one finite value. Softmax of a row that is entirely minus infinity would compute 0 / 0 and return NaN. With a cache, the mask slice starts at row `P`, and row `P + i` keeps columns `0..P+i` visible, so the property still holds.

</details>

<details>
<summary>Why are only attn.proj.weight and mlp.fc2.weight initialized with a smaller standard deviation?</summary>

They are the last linear layer of each sublayer, so their outputs are what gets added to the residual stream. The stream receives `2L` such additions, and independent additions add their variances. Scaling these two weights by `1 / sqrt(2L)` keeps the variance of the final stream near what one addition would give, independent of depth. The other weights (`qkv`, `fc1`) feed into nonlinear computations inside the sublayer and do not accumulate in the stream.

</details>

<details>
<summary>Generation with the cache enabled passes the context length N. What happens on each later step, and why can the cache not be kept?</summary>

Once the cache holds `N` positions, `generate` takes the refill path on every step: it clears the cache and runs the last `N` tokens. Each new token shifts the window by one, so every token in the window moves to a new absolute position and receives a different position embedding. Keys and values computed with the old positions are wrong for the new window. The output stays correct, but each step costs a full forward pass over `N` tokens.

</details>

<details>
<summary>With top_k = 2 and logits [3.0, 1.0, 1.0, 0.0], how many tokens can be sampled?</summary>

Three. The threshold is the second largest value, 1.0, and the mask hides only values strictly below it. Both entries equal to 1.0 survive, so tokens 0, 1 and 2 keep a nonzero probability and token 3 is masked. Ties at the threshold let more than `k` tokens through.

</details>

<details>
<summary>Why does GPT::forward compute matmul(weight, h.mT()).mT() instead of matmul(h, weight.T())?</summary>

`matmul` packs its operands into contiguous batched matrices with `reshape`. `weight.T()` is a transposed view of the `[V, C]` embedding, so the direct form would copy the whole table, 154 MB for GPT-2 in F32, on every call. In the chosen form the weight is used in its stored layout; for `B = 1` its broadcast to `[1, V, C]` is still contiguous and is not copied. Only the hidden states and the `[B, V, T]` result are transposed, and they are much smaller during generation.

</details>

## Further reading

- [Vaswani et al., "Attention Is All You Need" (2017)](https://arxiv.org/abs/1706.03762): the original transformer, with scaled dot-product attention, multi-head attention and the post-LN layout this chapter compares against.
- [Radford et al., "Language Models are Unsupervised Multitask Learners" (2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf): the GPT-2 paper, including pre-layer-norm, the final layer norm and the residual initialization scale.
- [Xiong et al., "On Layer Normalization in the Transformer Architecture" (2020)](https://arxiv.org/abs/2002.04745): why post-LN needs warmup and pre-LN trains stably.
- [Hendrycks and Gimpel, "Gaussian Error Linear Units (GELUs)" (2016)](https://arxiv.org/abs/1606.08415): the definition of GELU and the tanh approximation.
- [Press and Wolf, "Using the Output Embedding to Improve Language Models" (2017)](https://arxiv.org/abs/1608.05859): the argument and the evidence for weight tying.
- [Fan, Lewis and Dauphin, "Hierarchical Neural Story Generation" (2018)](https://arxiv.org/abs/1805.04833): the paper that popularized top-k sampling for text generation.
- [Holtzman et al., "The Curious Case of Neural Text Degeneration" (2019)](https://arxiv.org/abs/1904.09751): why pure sampling and greedy decoding both fail, and nucleus sampling as an alternative to top-k.
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT): a compact PyTorch GPT-2 with the same Tiny Shakespeare setup, useful to compare line by line with `src/nn/transformer.cpp`.
- [`src/nn/transformer.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/nn/transformer.cpp): this chapter's code and the GPT-2 loader of chapter 12 in 310 lines.

Next: [12. Loading GPT-2](@/book/12-model-loading.md)
