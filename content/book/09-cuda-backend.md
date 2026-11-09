+++
title = "CUDA backend"
description = "Device memory, device kernels shared between a CUDA backend and a host emulation, and the rules that keep tensors on one device."
weight = 9
slug = "cuda-backend"

[extra]
chapter = "09"
kicker = "Chapter 09"
builds = "Device memory, device kernels and a host emulation backend for testing"
source = "docs/09-cuda-backend.md"
tag = "chapter-10-cuda"
+++

This chapter adds a second device. A tensor can live in GPU memory, move between host and device, and run elementwise kernels, reductions, matmul, gather and scatter there. Autograd and `nn` modules work on the device without changes. The backend is a small C-style interface of about a dozen functions. Three translation units implement it: the real CUDA backend, a host emulation that runs the same kernel code on the CPU, and a stub for builds without CUDA. The chapter maps to milestone v0.3 and to the git tag `chapter-10-cuda`.

> [!NOTE]
> The backend is validated end to end on a real GPU. `scripts/cuda/validate.sh` builds kael with CUDA, runs the C++ and Python suites on the device with and without cuBLAS, checks device memory accesses with `compute-sanitizer`, and trains and runs the example models on the GPU. The reference run on an NVIDIA Tesla T4 is in [Validation on a GPU](@/book/09-cuda-backend.md#validation-on-a-gpu). On every push, CI also runs the device test suite against the host emulation (`KAEL_FAKE_CUDA`) and compiles the real backend with `nvcc`.

| File | Role |
|------|------|
| [`include/kael/dtype.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/dtype.h) | `enum class Device { CPU, CUDA }` |
| [`include/kael/device.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/device.h) | Public `parse_device()`, `cuda_is_available()` and `cuda_synchronize()` |
| [`include/kael/device_kernels.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/device_kernels.h) | The backend interface in `kael::device`: memory, copies, kernels, layouts |
| [`src/device.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/device.cpp) | Device name parsing; forwards the public functions to `kael::device` |
| [`src/cuda/kernels.inl`](https://github.com/JetBerri/kael.cpp/blob/main/src/cuda/kernels.inl) | Kernel bodies shared by the real and the emulated backend |
| [`src/cuda/backend.cu`](https://github.com/JetBerri/kael.cpp/blob/main/src/cuda/backend.cu) | Real backend: CUDA runtime, kernel launcher, optional cuBLAS |
| [`src/cuda/backend_fake.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/cuda/backend_fake.cpp) | Host emulation of the device, for tests without a GPU |
| [`src/cuda/backend_none.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/cuda/backend_none.cpp) | Stub that reports CUDA as unavailable and throws on use |
| [`src/internal/element_functions.h`](https://github.com/JetBerri/kael.cpp/blob/main/src/internal/element_functions.h) | Elementwise functors usable from host and device code (`KAEL_HD`) |
| [`src/allocator.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/allocator.cpp) | The device instance of the caching allocator |
| [`scripts/cuda/validate.sh`](https://github.com/JetBerri/kael.cpp/blob/main/scripts/cuda/validate.sh), [`scripts/cuda/run_remote.sh`](https://github.com/JetBerri/kael.cpp/blob/main/scripts/cuda/run_remote.sh), [`scripts/cuda/validate_on_colab.ipynb`](https://github.com/JetBerri/kael.cpp/blob/main/scripts/cuda/validate_on_colab.ipynb) | GPU validation run, remote launcher and Colab notebook |

## Prerequisites

- [02. Memory model](@/book/02-memory-model.md): `Storage`, `TensorImpl`, and the `Device` field.
- [04. TensorIterator](@/book/04-tensor-iterator.md): dimension coalescing, which produces the layouts sent to the device.
- [05. Operations](@/book/05-operations.md): the host reference for every kernel in this chapter.
- [06. Autograd](@/book/06-autograd.md): backward functions, which must keep gradients on the right device.
- [08. CPU performance](@/book/08-cpu-performance.md): the caching allocator reused for device memory.

## Why this layer exists

Training a transformer is dominated by matrix products and long elementwise passes over large buffers. A GPU runs thousands of threads at once and has memory bandwidth several times higher than a desktop CPU. Every framework that trains real models runs that work on an accelerator. A didactic framework that stops at the CPU hides the part of the design that shapes PyTorch the most: data that lives somewhere the host cannot read.

Adding a device breaks three assumptions that chapters 01 to 08 relied on:

1. **Memory is addressable.** On the CPU, `data_ptr<float>()[3]` reads an element. A pointer into GPU memory is a valid 64-bit number on the host, but dereferencing it there is undefined behavior. Usually it crashes, sometimes it reads garbage.
2. **Kernels are C++ functions.** A CPU kernel is a loop the compiler inlines. A GPU kernel is a function compiled by `nvcc` for a different instruction set and started with a launch, which returns before the work is done.
3. **Errors happen at the call site.** A fault inside a GPU kernel is reported by some later CUDA call, often far from the operation that caused it.

Without a device layer, every operation would need its own GPU code path, every file would need the CUDA compiler, and nobody without an NVIDIA card could build or test the project. kael answers each problem with one decision:

- A tensor records its device in its storage, and host accessors refuse device tensors.
- Operations name their kernel with an enum and pass a flat layout struct through a C-style interface. Only one file is CUDA code.
- The same kernel bodies compile twice: once for the GPU, once for a host emulation that runs everywhere and is tested in CI.

The mental model to hold on to:

```text
          kael operations (plain C++20)
                     |
        device::elementwise(op, dtype, layout, ...)     <- the only door
                     |
   +-----------------+-----------------+
   |                 |                 |
backend.cu     backend_fake.cpp   backend_none.cpp
GPU threads    host loop          throws "built without CUDA"
   \                 /
    kernels.inl (same task bodies)
```

Everything above the door is ordinary C++ that never touches device memory. Everything below it is selected at configure time, one backend per build.

## Concepts

### The device model

A GPU has its own memory, called device memory. Data reaches it only through explicit copies, and kernels run on the device only through launches. kael keeps the model minimal:

- There is one device type besides the host: `Device::CUDA`. There is no device index. `parse_device` in `src/device.cpp` accepts `"cuda"` and `"cuda:0"` as the same device, and `"cpu"` for the host. Any other name raises `ValueError`.
- A tensor's `Storage` records its `Device`. Every byte of one storage lives on one device.
- Views share storage, so a view of a CUDA tensor is a CUDA tensor with the same stride and offset rules as on the host ([Views](@/book/03-views.md)).
- Operations require all operands on the same device. There is no implicit transfer.

```text
        Host                                   Device
+------------------------+            +------------------------+
| Tensor (CPU)           |            | Tensor (CUDA)          |
|   Storage, Device::CPU |  to(CUDA)  |   Storage, Device::CUDA|
|   host_cache()         | ---------> |   device_cache()       |
|                        |  copy_to_  |                        |
|                        |  device    |                        |
|                        | <--------- |                        |
|                        |  to(CPU)   |                        |
|                        |  copy_to_  |                        |
|                        |  host      |                        |
+------------------------+            +------------------------+
```

PyTorch has the same rules: `torch.device("cuda:0")`, `.to()`, `.cuda()`, `.cpu()`, and an error when operands live on different devices. PyTorch adds device indices, streams, pinned host memory and peer copies between GPUs. kael leaves all of those out, because none of them changes how a kernel addresses its operands.

### The backend interface

[`include/kael/device_kernels.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/device_kernels.h) declares everything the rest of kael may ask of a device. It uses raw `std::byte*` pointers, `DType` tags and plain structs, so it compiles in both a C++ and a CUDA translation unit without templates crossing the boundary.

```cpp
[[nodiscard]] bool available();
void synchronize();

[[nodiscard]] std::byte* allocate(std::int64_t bytes);
void release(std::byte* block) noexcept;
void copy_to_device(std::byte* target, const std::byte* source, std::int64_t bytes);
void copy_to_host(std::byte* target, const std::byte* source, std::int64_t bytes);
void copy_within_device(std::byte* target, const std::byte* source, std::int64_t bytes);

void fill(DType dtype, std::byte* target, std::int64_t count, const void* value);
void convert(DType out, DType in, const ElementLayout& layout);
void elementwise(ElementOp op, DType out, DType in, const ElementLayout& layout,
                 ElementParameters parameters);
void reduce(ReduceOp op, DType out, DType in, std::byte* results, const std::byte* data,
            std::int64_t chunk_size, std::int64_t output_count);
void matmul(DType dtype, const std::byte* a, const std::byte* b, std::byte* out,
            const MatmulSizes& sizes);
void gather(DType dtype, std::byte* out, const std::byte* data, const std::int64_t* indices,
            const IndexLayout& layout);
void scatter_add(DType dtype, std::byte* data, const std::int64_t* indices, const std::byte* source,
                 const IndexLayout& layout);
```

The functions fall into three groups:

| Group | Functions | Purpose |
|-------|-----------|---------|
| Runtime | `available`, `synchronize` | Is there a device; wait for all queued work |
| Memory | `allocate`, `release`, three copies | Raw bytes on the device and transfers in each direction |
| Kernels | `fill`, `convert`, `elementwise`, `reduce`, `matmul`, `gather`, `scatter_add` | All the compute kael runs on a device |

Operations are named by enums. `ElementOp` lists 36 elementwise operations, from `Plus` to `MaskedFill`. `ReduceOp` has `Sum`, `Max` and `Min`.

An alternative design passes C++ functors to the device, the way the CPU kernels of [Operations](@/book/05-operations.md) take a lambda. That would force every operation file to be compiled by `nvcc`, because a kernel template must be instantiated in a CUDA translation unit. With enums, only `backend.cu` needs the CUDA compiler, and the rest of kael stays plain C++20. The cost is a `switch` over `ElementOp` in the backend, written once.

### Element parameters and exact integers

Most elementwise operations take only tensors. Two take constants: `Clamp` (low and high) and `MaskedFill` (the fill value). `ElementParameters` carries up to two constants:

```cpp
// Integer arguments travel apart from real ones: a double cannot hold every int64 exactly.
struct ElementParameters {
    std::array<double, 2> reals{};
    std::array<std::int64_t, 2> integers{};

    template <typename T> [[nodiscard]] T get(std::size_t index) const {
        if constexpr (std::is_integral_v<T>) {
            return static_cast<T>(integers[index]);
        } else {
            return static_cast<T>(reals[index]);
        }
    }
};
```

`Clamp<T>::parameters()` and `MaskedFill<T>::parameters()` fill `integers` for integer `T` and `reals` otherwise. The backend reads them back with `parameters.get<T>(0)`.

A single `double` field would be simpler and wrong. A `double` has a 53-bit significand, so it represents every integer up to `2^53` and only every second integer above it. Trace a clamp bound through a `double`:

```text
bound as int64           9007199254740993     (2^53 + 1)
static_cast<double>      9007199254740992.0   (rounded to the nearest representable value)
back to int64            9007199254740992     low bit lost
```

An `i64` clamp with that upper bound would change values that should pass through unchanged. `Cuda.IntegerClampAndMaskedFillKeepEveryBit` checks that bounds and fill values above `2^53` stay exact.

### Layouts

The host already computes everything a kernel needs to address strided operands. `TensorIterator` merges dimensions where possible and computes byte strides for each operand ([TensorIterator](@/book/04-tensor-iterator.md)). `ElementLayout` copies that result into a fixed-size struct that can be passed by value to a kernel:

```cpp
inline constexpr int kMaxDims = 8;
inline constexpr int kMaxOperands = 4;

struct ElementLayout {
    int dims;
    int operands;
    std::int64_t count;
    std::int64_t shape[kMaxDims];
    std::int64_t strides[kMaxOperands][kMaxDims];
    std::byte* base[kMaxOperands];
};
```

Fixed arrays are required because kernel arguments are copied by value into the launch. A `std::vector` member would carry a pointer to host memory, which the device cannot read. `TensorIterator::device_layout()` builds the struct and rejects iteration spaces with more than 8 dimensions after merging, or more than 4 operands. Four operands is one output plus up to three inputs, enough for `where`. Coalescing makes the limit generous: a contiguous tensor of any rank collapses to one dimension.

Worked example: `a + b` with `a` of shape `[2, 3]` (contiguous, `float`) and `b` of shape `[3]`, broadcast. Dimensions 0 and 1 cannot merge, because `b` has stride 0 in dimension 0 while merging needs `stride[0] == shape[1] * stride[1]` for every operand. The layout is:

```text
dims = 2, operands = 3, count = 6
shape          = [2, 3]
strides[0] out = [12, 4]    bytes
strides[1] a   = [12, 4]
strides[2] b   = [ 0, 4]    broadcast dimension has stride 0
base           = [out, a, b] device addresses
```

Compare with a case that does merge. For `a + c` with both operands contiguous `[2, 3]`, every operand satisfies `12 == 3 * 4`, and the layout collapses to `dims = 1`, `shape = [6]`, every stride `[4]`.

`IndexLayout` plays the same role for `gather` and `scatter_add`. It holds the index tensor's shape, the axis, and three stride arrays (`data_strides`, `index_strides`, `source_strides`) in elements, not bytes. `MatmulSizes` holds four integers: `batches`, `rows`, `inner` and `columns`.

### Synchronous and asynchronous work

A CUDA kernel launch queues work and returns. A host-to-device or device-to-host `cudaMemcpy` waits for earlier work on the same stream and blocks the host until the copy finishes. kael uses only the default stream, so the timeline looks like this:

```text
host:    launch add  launch sum  cudaMemcpy D->H ......... returns, value ready
device:  [   add    ][   sum   ][ copy ]
                                 ^ the copy waits for sum, which waits for add
```

Two consequences follow. First, kernels run in issue order, so no explicit synchronization is needed between operations. Second, timing a kernel with a host clock measures only the launch unless the code calls `cuda_synchronize()` before stopping the clock.

## Implementation

### Moving tensors between devices

From [`src/tensor.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/tensor.cpp):

```cpp
Tensor Tensor::to(Device target) const {
    if (target == device()) {
        return *this;
    }
    const Tensor source = autograd::without_grad([&] { return detach().contiguous(); });
    Tensor out = empty(shape(), dtype(), target);
    if (nbytes() > 0) {
        if (target == Device::CPU) {
            device::copy_to_host(out.device_data(), source.device_data(), nbytes());
        } else {
            device::copy_to_device(out.device_data(), source.device_data(), nbytes());
        }
    }
    autograd::record_to_device(out, *this);
    return out;
}
```

The steps:

1. Same device returns the same tensor handle, not a copy (`Cuda.SameDeviceReturnsTheSameTensor`). PyTorch behaves the same way: `t.to(t.device) is t`.
2. The source is made contiguous on its own device first. A transfer is one block copy, and a strided view would need one copy per element. `contiguous()` on a CUDA tensor runs the device `convert` kernel, so the gathering happens on the device, not over the bus.
3. The destination is allocated on the target device with `empty`, and one bulk copy moves the bytes.
4. `record_to_device` adds a `ToDeviceBackward` node whose backward moves the gradient back with `g.to(device)`, using the source device. A CPU leaf used on the device therefore receives a CPU gradient (`Cuda.GradientsCrossDevicesBackToTheOrigin`).

Trace `x.T.cuda()` for a contiguous `x` of shape `[2, 3]` with values `0..5`:

```text
x            shape [2, 3], strides (3, 1), storage [0 1 2 3 4 5] on cpu
x.T          shape [3, 2], strides (1, 3), same storage
contiguous   shape [3, 2], strides (2, 1), new storage [0 3 1 4 2 5] on cpu
empty        shape [3, 2] on cuda, 24 bytes from device_cache()
copy         24 bytes host -> device
result       shape [3, 2], strides (2, 1), cuda
```

`cuda()` and `cpu()` are shorthands for `to(Device::CUDA)` and `to(Device::CPU)`.

`device_data()` returns the address of the first element on any device. Its comment in `include/kael/tensor.h` warns that on a CUDA tensor only device code can dereference it. `data_ptr<T>()` and the other host accessors call `check_on_host()` and throw "this operation reads tensor memory directly, which is not possible on device cuda; call cpu() first" (`Cuda.HostOnlyAccessFailsClearly`). `item()` on a CUDA tensor copies to the host first (`Cuda.ReadingValuesCopiesToHost`).

### Moving in place

`to()` returns a new tensor. An optimizer that already holds handles to the parameters would keep updating the old CPU tensors after the model moved. `Tensor::move_to_(Device)` solves this. It replaces the storage, strides, offset and device of an existing `TensorImpl` with those of a moved copy, so every handle to that tensor sees the new device:

```cpp
Tensor& Tensor::move_to_(Device target) {
    if (target == device()) {
        return *this;
    }
    const Tensor moved = detach().to(target);
    impl_->storage = moved.impl()->storage;
    impl_->strides = moved.strides();
    impl_->offset = moved.offset();
    impl_->device = target;
    return *this;
}
```

This works because a `Tensor` is a handle to a shared `TensorImpl` ([ADR 0002](@/adr/0002-tensor-as-handle.md)). `nn::Module::to(Device)` in `src/nn/module.cpp` uses it:

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

```text
before:  optimizer.params[0] --\
                                >-- TensorImpl --> Storage (cpu)
         model.weight ---------/

after:   optimizer.params[0] --\
                                >-- TensorImpl --> Storage (cuda)   same TensorImpl, new storage
         model.weight ---------/
```

`Cuda.ModuleToKeepsParameterIdentityAndValues` and `Cuda.ModuleToMovesExistingGradients` check both properties. PyTorch gets the same effect differently: `Module.to` replaces `param.data` in place.

### Factories and scalars

`empty`, `zeros`, `ones` and `full` take a `Device`. `full` on the device calls `device::fill` with the converted value. Random factories (`rand`, `randn`, `randint`) have no device parameter in C++. Generate on the host and call `cuda()`. The Python wrappers do the same with `.to(device)` ([Python bindings](@/book/10-python-bindings.md)). Either way one seed produces the same values on both devices, because the generator always runs on the host. PyTorch differs here: its CUDA generator is a separate Philox-based generator, so `torch.manual_seed(0)` gives different values for `torch.randn(3)` and `torch.randn(3, device="cuda")`.

A Python or C++ scalar operand follows the tensor's device (`Cuda.ScalarsFollowTheTensorDevice`). `x.cuda() + 1` works without the caller building a device scalar. The scalar becomes a 0-d tensor on the same device as `x` inside `src/ops/binary.cpp`.

### How operations reach the device

Each operation family has a device branch.

| Family | Device path |
|--------|-------------|
| Elementwise with a tagged functor | `run_typed` in `include/kael/kernels.h` calls `device::elementwise(F::kOp, ...)` |
| Dtype conversion, `contiguous`, `clone` of a strided view | `convert_strided` in `src/copy.cpp` calls `device::convert` |
| `sum`, `mean`, `max`, `min`, `var`, `std`, `norm` | `device_reduce` in `src/ops/reduce.cpp` calls `device::reduce` |
| `matmul`, `mm`, `bmm` | `multiply_batches` in `src/ops/linalg.cpp` calls `device::matmul` with batch sizes |
| `gather`, `index_select` | `device::gather` |
| `scatter_add_`, `index_add_` | `device::scatter_add` |
| `full`, `zeros`, `ones` | `device::fill` |
| `prod`, `argmax`, `argmin`, `all`, `any`, `sort`, `topk`, `cumsum` | Copy to host, compute, copy back |

The last row is a deliberate shortcut. `on_host` in `src/ops/reduce.cpp` and `moved_to_last_on_host` in `src/ops/sequence.cpp` copy the input to the CPU, run the host kernel and move the result back. The result is correct and on the right device. The cost is two transfers, which matters little for operations that are rare in a training step.

#### Tagged functors

The elementwise branch keys off a compile-time tag. Functors in `src/internal/element_functions.h` derive from `Tagged<ElementOp::...>`:

```cpp
template <ElementOp Op> struct Tagged {
    static constexpr ElementOp kOp = Op;
    [[nodiscard]] KAEL_HD ElementParameters parameters() const { return {}; }
};
```

The concept `detail::DeviceFunction<F>` in `include/kael/kernels.h` detects `F::kOp` and `f.parameters()`:

```cpp
template <typename F>
concept DeviceFunction = requires(const F& f) {
    { F::kOp } -> std::convertible_to<device::ElementOp>;
    f.parameters();
};
```

`run_typed` uses it:

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

A tagged functor on a device tensor becomes one `device::elementwise` call. An untagged functor, such as a lambda passed to `unary_kernel`, has no name the device knows. `same_dtype_kernel` therefore copies its inputs to the host, runs there, and copies the result back:

```cpp
if constexpr (!DeviceFunction<F>) {
    if (any_on_device(inputs...)) {
        check_inputs_share_device(inputs...);
        const Device device = std::get<0>(std::tie(inputs...)).device();
        return same_dtype_kernel(dtype, f, inputs.cpu()...).to(device);
    }
}
```

The fallback keeps every operation correct on the device, at the cost of two transfers. Adding a device kernel for an operation means adding an `ElementOp`, a `case` in the backend switch, and the tag on the functor. No operation file changes.

#### One definition for host and device

The functors are the same objects for host and device. `KAEL_HD` expands to `__host__ __device__` under `nvcc` and to nothing otherwise:

```cpp
#if defined(__CUDACC__)
#define KAEL_HD __host__ __device__
#else
#define KAEL_HD
#endif
```

One definition of `GeluTanh` or `nan_aware_max` serves the CPU kernels, the CUDA kernels and the emulation. The three cannot disagree on semantics such as NaN propagation or integer overflow. PyTorch reaches a similar goal with its `C10_HOST_DEVICE` macro on shared math helpers.

Fused CPU kernels from [CPU performance](@/book/08-cpu-performance.md) check `x.device() == Device::CPU` and take the composed path on the device. Softmax and log softmax take the fused row kernel only for a contiguous CPU input reduced over its last dimension (`is_last_dim_of_contiguous` in `src/ops/nn.cpp`). `standardize` on the device computes `(x - mean) / sqrt(var + eps)` from device reductions. Cross entropy on the device computes `-gather(log_softmax(logits, 1), 1, targets.unsqueeze(1)).squeeze(1)`. Each composed path is built from elementwise and reduce kernels that already exist.

### The shared kernel set

[`src/cuda/kernels.inl`](https://github.com/JetBerri/kael.cpp/blob/main/src/cuda/kernels.inl) is included by both `backend.cu` and `backend_fake.cpp`. It is written as a set of **tasks**: function objects whose `operator()(std::int64_t i)` processes one work item. A task knows nothing about threads or blocks. The including file supplies two things before the include:

- `for_each(count, task)`: run `task(i)` for every `i` in `[0, count)`.
- `tasks::atomic_add(target, value)`: add to memory that several work items may hit.

```text
                     kernels.inl
             tasks + dispatch (switch on ElementOp)
                 /                         \
          backend.cu                   backend_fake.cpp
  for_each   = grid-stride CUDA kernel  for_each   = backwards host loop
  atomic_add = atomicAdd                atomic_add = wrapping add, no atomics
```

This split is the central trick of the chapter. The work item is the unit both backends agree on. How work items map to hardware is the only thing that differs.

#### Elementwise

`tasks::Element<Out, In, Inputs, F>` processes one element. `locate` decomposes the linear index into coordinates, innermost first, and accumulates byte offsets for every operand:

```cpp
KAEL_HD inline void locate(const ElementLayout& layout, std::int64_t element, std::byte** pointers) {
    std::int64_t offsets[kMaxOperands] = {};
    std::int64_t remaining = element;
    for (int dim = layout.dims - 1; dim >= 0; --dim) {
        const std::int64_t position = remaining % layout.shape[dim];
        remaining /= layout.shape[dim];
        for (int op = 0; op < layout.operands; ++op) {
            offsets[op] += position * layout.strides[op][dim];
        }
    }
    for (int op = 0; op < layout.operands; ++op) {
        pointers[op] = layout.base[op] + offsets[op];
    }
}
```

On the CPU, [TensorIterator](@/book/04-tensor-iterator.md) avoids this per-element division with a counter that carries from one element to the next. On a GPU each thread handles independent elements and has no previous element to carry from, so the decomposition is per element. Coalescing keeps `dims` small, usually 1 or 2, so the loop runs once or twice.

With the layout from the `a + b` example, trace every element:

```text
element  dim 1: pos = e % 3   dim 0: pos = (e / 3) % 2   out   a    b    addresses
0        0                    0                          0     0    0    out[0][0] = a[0][0] + b[0]
1        1                    0                          4     4    4    out[0][1] = a[0][1] + b[1]
2        2                    0                          8     8    8    out[0][2] = a[0][2] + b[2]
3        0                    1                          12    12   0    out[1][0] = a[1][0] + b[0]
4        1                    1                          16    16   4    out[1][1] = a[1][1] + b[1]
5        2                    1                          20    20   8    out[1][2] = a[1][2] + b[2]
```

For element 4: `position = 4 % 3 = 1` in dimension 1, then `remaining = 4 / 3 = 1` and `1 % 2 = 1` in dimension 0. The offsets are `out: 1*12 + 1*4 = 16`, `a: 16`, `b: 1*0 + 1*4 = 4` bytes. The stride 0 of `b` in dimension 0 is what makes rows 0 and 1 read the same three values.

`elementwise` switches on `ElementOp` and picks a typed runner:

| Runner | Output type | Input types | Used for |
|--------|-------------|-------------|----------|
| `run_same<N>` | Same as input | All four dtypes | Arithmetic, `Relu`, `Sign`, `Where` |
| `run_floating` | Same as input | `float` or `double` only | `Exp`, `Log`, `Tanh`, `GeluExact`, other transcendental functions |
| `run_compare` | `std::uint8_t`, kael's `bool` | All four dtypes | Comparisons and logical operations |

`with_type` maps a runtime `DType` to a C++ type with a `switch`. `cast_value` converts the result. It maps any non-zero value to 1 for `bool` outputs and calls `functions::saturating_cast` for float to integer conversion, the device copy of the saturating conversion of [Memory model](@/book/02-memory-model.md). `convert` is the same `Element` task with an `Identity` functor and different input and output types.

#### Reductions

`device::reduce` receives contiguous rows: `output_count` rows of `chunk_size` elements each, as planned by `plan_reduction` on the host. `sum(x, dim=1)` on a `[4, 10]` tensor arrives as 4 rows of 10. A reduction over a non-last dimension arrives after the host permutes and copies the reduced dimensions to the end. Two strategies are chosen by row length:

- Rows of at most `kBlock = 4096` elements: one work item per row (`ReduceRows`).
- Longer rows: for each row, one work item per 4096-element block (`ReduceBlocks`) writes a partial result. Then `MergePairs` halves the partial array repeatedly, ping-ponging between two scratch buffers, until one value remains. `StoreResult` copies it into the output.

```text
row of 20000 floats, kBlock = 4096, block_count = ceil(20000 / 4096) = 5
blocks:   [0..4095] [4096..8191] [8192..12287] [12288..16383] [16384..19999]
partial:     p0         p1           p2            p3             p4         width 5
merge 1:   p0+p1      p2+p3          p4                                      width 3
merge 2: (p0+p1)+(p2+p3)  p4                                                 width 2
merge 3: ((p0+p1)+(p2+p3))+p4                                                width 1
```

`MergePairs` writes `target[i] = source[2i] + source[2i + 1]` when both exist, and copies `source[2i]` when the width is odd and `i` is the last item. The next width is `(width + 1) / 2`. For this row the host issues five launches: one `ReduceBlocks`, three `MergePairs` and one `StoreResult`.

The order of additions depends only on the block size, not on the number of GPU threads or their scheduling. Device sums are therefore deterministic run to run (`Cuda.LongSumsMatchCpuAndAreDeterministic`). A design that has every thread `atomicAdd` into one accumulator would be faster to write and would give a different rounding on every run.

Sums accumulate in `double` for floating-point inputs and in `std::int64_t` for integers, matching the CPU. Integer sums use `functions::wrapping_add`, so overflow wraps around as on the CPU: summing `[INT64_MAX, 1]` gives `INT64_MIN` (`Cuda.IntegerSumWrapsAndFloatConversionSaturates`). `Max` and `Min` use `nan_aware_max` and `nan_aware_min`, so a NaN anywhere in the row makes the result NaN.

The scratch buffers come from `allocate` and go back through `release`, bypassing the caching allocator. Rows are processed one after another on the host side, with three or more launches per long row. This is simple and correct. A production backend would reduce all rows in one launch with shared-memory trees, as described by Mark Harris in the reading list below.

#### Matmul

Without cuBLAS, `MatmulTiles<T>` gives each work item a 4 x 4 tile of one output matrix (`kMatmulTile = 4`) and loops over the full inner dimension:

```cpp
T accumulators[kMatmulTile][kMatmulTile] = {};
for (std::int64_t k = 0; k < sizes.inner; ++k) {
    const T* b_row = batch_b + k * sizes.columns + first_column;
    for (std::int64_t r = 0; r < row_count; ++r) {
        const T a_value = batch_a[(first_row + r) * sizes.inner + k];
        for (std::int64_t c = 0; c < column_count; ++c) {
            accumulators[r][c] = functions::wrapping_add(
                accumulators[r][c], functions::wrapping_multiply(a_value, b_row[c]));
        }
    }
}
```

The tile index encodes the batch, the tile row and the tile column:

```text
tile_rows    = ceil(rows / 4)
tile_columns = ceil(columns / 4)
per_batch    = tile_rows * tile_columns
batch        = tile / per_batch
first_row    = ((tile % per_batch) / tile_columns) * 4
first_column = ((tile % per_batch) % tile_columns) * 4
```

Worked example with `rows = 5`, `inner = 3`, `columns = 9`, one batch. `tile_rows = 2`, `tile_columns = 3`, so there are 6 tiles:

```text
            columns 0..3   columns 4..7   column 8
rows 0..3   tile 0 (4x4)   tile 1 (4x4)   tile 2 (4x1)
row  4      tile 3 (1x4)   tile 4 (1x4)   tile 5 (1x1)
```

Tile 4 has `first_row = (4 / 3) * 4 = 4` and `first_column = (4 % 3) * 4 = 4`. Edge tiles clip the counts: `row_count = min(5 - 4, 4) = 1` and `column_count = min(9 - 4, 4) = 4`. Tile 4 therefore computes `out[4][4..7]`.

The result is written with `=`, not `+=`. The CPU matmul ladder of [CPU performance](@/book/08-cpu-performance.md) accumulates into a zeroed output, but the device kernel owns its output tile completely. This kernel favors clarity. It reads every operand from global memory, with no shared-memory staging, so on real hardware it is expected to be far slower than cuBLAS. On the T4 of the [reference run](@/book/09-cuda-backend.md#reference-run) it reaches 368 to 427 GFLOP/s for square `f32` matrices from 512 to 2048. It also handles `i64` matrices, which cuBLAS does not. `wrapping_add` and `wrapping_multiply` make their overflow wrap around as on the CPU. `multiply_batches` rejects `bool` matmul with `TypeError` before calling the device.

#### Gather and scatter

`Gather<T>` and `ScatterAdd<T>` use `locate_index` to compute three positions from one element of the index tensor. It walks the coordinates innermost first, like `locate`, and accumulates:

- `index`: where to read the index value.
- `source`: where in the source (or output) to read or write.
- `data`: where in the data tensor the element points. The coordinate on `axis` is skipped, then replaced by the index value.

Worked example: `gather(data, 1, indices)` with `data` of shape `[2, 3]` (contiguous, strides `(3, 1)` in elements) and `indices = [[2, 0], [1, 1]]` (contiguous, strides `(2, 1)`):

```text
element  coords  index pos  indices[...]  data pos                 out
0        (0, 0)  0          2             0*3 + 2*1 = 2            out[0][0] = data[0][2]
1        (0, 1)  1          0             0*3 + 0*1 = 0            out[0][1] = data[0][0]
2        (1, 0)  2          1             1*3 + 1*1 = 4            out[1][0] = data[1][1]
3        (1, 1)  3          1             1*3 + 1*1 = 4            out[1][1] = data[1][1]
```

`ScatterAdd` runs the same walk in reverse direction of data flow: it adds `source[position.source]` into `data[position.data]`. Elements 2 and 3 above point at the same destination. In a scatter, two GPU threads would update that address at the same time, so `ScatterAdd` uses `atomic_add`. On the real device this is `atomicAdd`. The `std::int64_t` specialization reinterprets the target as `unsigned long long`, because CUDA has no signed 64-bit `atomicAdd`. Two's complement addition gives the same bits for signed and unsigned operands, so the result wraps like the CPU (`Cuda.IntegerScatterAddWrapsAround`).

Index bounds are checked on the host before the launch. `check_device_indices` in `src/internal/index_checks.h` computes `max` and `min` of the index tensor with device reductions and validates both with `checked_index`. Out-of-range indices throw `IndexError` instead of reading or writing outside the allocation (`Cuda.OutOfRangeIndicesStillThrow`). The price is two reductions and a device-to-host copy of two scalars per call. PyTorch instead checks inside the kernel with a device-side assert, which aborts the CUDA context and reports the error asynchronously.

### The real backend

[`src/cuda/backend.cu`](https://github.com/JetBerri/kael.cpp/blob/main/src/cuda/backend.cu) provides the launcher:

```cpp
inline constexpr int kThreadsPerBlock = 256;
inline constexpr std::int64_t kMaxBlocks = 65535;

template <typename Task> __global__ void run_task(std::int64_t count, Task task) {
    std::int64_t element = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (; element < count; element += stride) {
        task(element);
    }
}

template <typename Task> void for_each(std::int64_t count, const Task& task) {
    if (count == 0) {
        return;
    }
    const auto blocks =
        static_cast<int>(std::min((count + kThreadsPerBlock - 1) / kThreadsPerBlock, kMaxBlocks));
    run_task<Task><<<blocks, kThreadsPerBlock>>>(count, task);
    KAEL_CUDA_CHECK(cudaGetLastError());
}
```

A CUDA launch starts a **grid** of **blocks**, each a group of threads (here 256) that run on one streaming multiprocessor. `run_task` is a **grid-stride loop**: each thread starts at its global index and advances by the total number of threads in the grid. Two worked launches:

```text
count = 1000
blocks  = min(ceil(1000 / 256), 65535) = 4
threads = 4 * 256 = 1024
thread t handles element t; threads 1000..1023 find element >= count and exit

count = 20 000 000
blocks  = min(ceil(20 000 000 / 256), 65535) = min(78125, 65535) = 65535
threads = 65535 * 256 = 16 776 960
thread 0 handles elements 0 and 16 776 960
threads 0 .. 3 223 039 run two iterations, the rest run one
```

The cap of 65535 blocks keeps the launch valid on every GPU generation, and large counts are handled by more loop iterations per thread. The index arithmetic is in `std::int64_t` because `count` can exceed `2^31`. `blockIdx.x * blockDim.x` in plain `unsigned int` would wrap silently. The task is passed by value, so its layout struct travels with the launch.

Every CUDA runtime call goes through `KAEL_CUDA_CHECK`. On failure it calls `fail_call`, which throws `kael::Error` with the message `CUDA error at <file>:<line>: <call> failed with <cudaGetErrorString>`. `cudaGetLastError` after a launch catches configuration errors such as an invalid grid size. Errors that occur while the kernel runs surface at the next synchronizing call. `scripts/cuda/validate.sh` runs the device tests with `CUDA_LAUNCH_BLOCKING=1`, which makes every launch synchronous so an error points at the kernel that caused it.

Memory goes through `cudaMalloc` and `cudaFree`. Copies use `cudaMemcpy`, which is synchronous with respect to the host. All launches use the default stream, so kernels run in issue order and a copy to the host waits for the kernels before it. `cuda_synchronize()` calls `cudaDeviceSynchronize`. Benchmarks need it to time kernel execution instead of launch latency: `run_on_device` in `benchmarks/bench_cuda.cpp` wraps `BM_DeviceAdd`, `BM_DeviceSum` and `BM_DeviceMatmul`.

`available()` returns true when `cudaGetDeviceCount` succeeds with at least one device. A binary built with CUDA still runs on a machine without a GPU: `cuda_is_available()` returns false and the device tests skip.

#### cuBLAS

With `-DKAEL_USE_CUBLAS=ON` (and `KAEL_ENABLE_CUDA=ON`), `matmul` sends `f32` and `f64` products to `cublasSgemm` and `cublasDgemm`. Other dtypes still use `MatmulTiles`.

```cpp
// cuBLAS is column-major: for row-major matrices it computes the transpose, Cᵀ = Bᵀ Aᵀ.
template <typename T, typename Gemm>
void cublas_multiply(Gemm gemm, const T* a, const T* b, T* out, const MatmulSizes& sizes) {
    const T one{1};
    const T zero{0};
    for (std::int64_t batch = 0; batch < sizes.batches; ++batch) {
        const cublasStatus_t status = gemm(
            cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(sizes.columns),
            static_cast<int>(sizes.rows), static_cast<int>(sizes.inner), &one,
            b + batch * sizes.inner * sizes.columns, static_cast<int>(sizes.columns),
            a + batch * sizes.rows * sizes.inner, static_cast<int>(sizes.inner), &zero,
            out + batch * sizes.rows * sizes.columns, static_cast<int>(sizes.columns));
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw Error("cuBLAS gemm failed");
        }
    }
}
```

cuBLAS follows the Fortran BLAS convention and reads matrices in column-major order. kael stores them row-major. The transpose trick avoids any copy. A row-major `M x N` matrix has the same bytes as a column-major `N x M` matrix, which is its transpose:

```text
A (row-major 2 x 3)        bytes: 1 2 3 4 5 6
| 1 2 3 |
| 4 5 6 |

same bytes read column-major as 3 x 2:
| 1 4 |
| 2 5 |   = A^T
| 3 6 |
```

The code asks cuBLAS for `B^T A^T`, passing B first with dimensions `columns x inner` and A second with `inner x rows`. The product `B^T A^T = (A B)^T` is written in column-major order, and those bytes are `A B` in row-major order. The leading dimension arguments are the row lengths of the row-major matrices. The handle is created once in a function-local static. Batches loop on the host with one GEMM call each. `cublasSgemmStridedBatched` would issue them in one call, at the cost of a less obvious excerpt. `matmul` returns early when any of `rows`, `inner` or `columns` is 0, because cuBLAS is not called with empty matrices.

### The host emulation

[`src/cuda/backend_fake.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/cuda/backend_fake.cpp) implements the same interface on the CPU. `-DKAEL_FAKE_CUDA=ON` selects it. CMake rejects `KAEL_FAKE_CUDA` together with `KAEL_ENABLE_CUDA` with "KAEL_ENABLE_CUDA and KAEL_FAKE_CUDA are mutually exclusive".

```cpp
// Backwards, so that a kernel relying on the order of its elements fails here too.
template <typename Task> void for_each(std::int64_t count, const Task& task) {
    for (std::int64_t i = count; i-- > 0;) {
        task(i);
    }
}

template <typename T> void atomic_add(T* target, T value) {
    *target = static_cast<T>(functions::wrapping_add(*target, value));
}
```

The emulated `atomic_add` needs no atomic instruction because the loop is single-threaded. It uses `wrapping_add` so that integer overflow wraps the same way as `atomicAdd` on unsigned bits.

Device memory in the emulation is `::operator new`, copies are `std::memcpy`, `synchronize()` does nothing, and `available()` returns true. A tensor on `Device::CUDA` in this build is still a separate allocation reached only through the backend interface. Every rule of the device model applies: host accessors throw, mixed devices are rejected, and every transfer goes through `copy_to_device` and `copy_to_host`.

The emulation exists because the kernels and all the dispatch around them can then be tested on any machine:

- The task bodies in `kernels.inl` are the same source that `nvcc` compiles. A wrong stride, a wrong offset or a wrong merge step fails in the emulation as it would on a GPU.
- Running work items in reverse order catches tasks that assume item `i - 1` finished before item `i`. GPU threads give no such guarantee.
- Every device branch in the operations (`on_device()`, `device_layout()`, host fallbacks, autograd across devices, `Module::to`) runs in CI.
- Coverage and clang-tidy jobs build with `KAEL_FAKE_CUDA=ON`, so the device branches count toward coverage and are linted.

Why backwards and not shuffled: a reverse loop is deterministic, so a failure reproduces on every run, and it breaks the most common hidden assumption, that work items run in increasing order. Consider a buggy prefix-sum task that reads `out[i - 1]`. A forward loop would compute it correctly. The backward loop reads `out[i - 1]` before it is written and fails.

The CI job for the emulation also enables AddressSanitizer and UndefinedBehaviorSanitizer, so an out-of-bounds access in a task body is reported as a host memory error. Sanitizer builds define `KAEL_NO_CACHING_ALLOCATOR`, so `Cuda.DeviceBlocksAreCached` skips itself in that job.

What the emulation cannot catch:

| Problem | Why the emulation misses it |
|---------|-----------------------------|
| Launch configuration limits | No launch happens |
| Device math differences | Host `std::exp` replaces the device `exp` |
| CUDA runtime behavior | No CUDA runtime is linked |
| Performance | Nothing runs on GPU hardware |
| Code that `nvcc` alone compiles | `backend.cu` is not part of the build |

Those need a real GPU. `scripts/cuda/validate.sh` runs the device tests on one under `CUDA_LAUNCH_BLOCKING=1`, runs two of them under the `compute-sanitizer` memory checker (the default `memcheck` tool), which reports out-of-bounds and misaligned device accesses, and runs the device benchmarks.

Races between GPU threads are excluded by construction. Every task writes only its own output elements, and the only shared writes, the scatter and index additions, go through `atomic_add`. The kernels use no shared memory, so `compute-sanitizer --tool racecheck`, which checks shared-memory hazards, has nothing to inspect. `backend_fake.cpp` defines its own `matmul`, which always uses the tile kernel, because the cuBLAS branch in `backend.cu` does not apply.

### The stub backend

Without either option, [`src/cuda/backend_none.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/cuda/backend_none.cpp) is linked. `available()` returns false, `synchronize()` and `release()` do nothing, and every other function throws:

```cpp
[[noreturn]] void fail_without_cuda() {
    throw Error("kael was built without CUDA; configure with KAEL_ENABLE_CUDA=ON");
}
```

`release()` must not throw because it runs in `noexcept` destructors. It never receives a real block, since `allocate()` already threw. `CudaAbsent.MovingToADeviceThatIsMissingFailsClearly` checks that `cuda()` and `zeros(..., Device::CUDA)` throw `kael::Error` in such a build.

The three backends compared:

| Function | `backend.cu` | `backend_fake.cpp` | `backend_none.cpp` |
|----------|--------------|--------------------|--------------------|
| `available()` | `cudaGetDeviceCount` > 0 | `true` | `false` |
| `allocate()` | `cudaMalloc` | `::operator new` | throws |
| copies | `cudaMemcpy` | `std::memcpy` | throw |
| `for_each` | grid-stride kernel | backwards loop | not present |
| `matmul` | cuBLAS or tiles | tiles | throws |

### The device caching allocator

`Storage(nbytes, Device::CUDA)` goes through the same `detail::allocate_block` as host storage. `cache_of(device)` returns `device_cache()`, a second `BlockCache` instance whose system functions are `device::allocate` and `device::release`:

```cpp
BlockCache& device_cache() {
    static auto* cache = new BlockCache(device::allocate, device::release);
    return *cache;
}
```

The cache is created with `new` and never destroyed. The comment above `host_cache()` gives the reason: tensors with static storage can release their blocks after `main` returns, and a destroyed cache would receive those blocks after its own destructor ran.

Size classes, the 1 GiB cap, statistics and `empty_cache()` behave as described in [CPU performance](@/book/08-cpu-performance.md). Caching matters more on the device. `cudaMalloc` and `cudaFree` are slow calls, and `cudaFree` synchronizes the device, so freeing per tensor would serialize every training step. PyTorch's CUDA caching allocator exists for the same reason. `allocator_stats(Device::CUDA)` reports the device numbers. `Cuda.DeviceBlocksAreCached` checks that a second allocation of the same size is a hit and that `empty_cache()` drops the cached bytes to zero.

Reduce scratch buffers bypass the cache. They call `device::allocate` and `device::release` directly.

### Build options

| Option | Default | Effect |
|--------|---------|--------|
| `KAEL_ENABLE_CUDA` | `OFF` | Enables the CUDA language, requires `CUDAToolkit`, compiles `src/cuda/backend.cu` as C++20 CUDA, links `CUDA::cudart` |
| `KAEL_USE_CUBLAS` | `OFF` | With `KAEL_ENABLE_CUDA`, links `CUDA::cublas` and defines `KAEL_USE_CUBLAS` |
| `KAEL_FAKE_CUDA` | `OFF` | Compiles `src/cuda/backend_fake.cpp` instead; no CUDA toolkit needed |
| `CMAKE_CUDA_ARCHITECTURES` | CMake default | Target GPU architectures, for example `86` or `native` |

Build the real backend:

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DKAEL_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build-cuda --parallel
```

Build and test the emulation on any machine:

```bash
cmake -S . -B build-fake -DKAEL_FAKE_CUDA=ON
cmake --build build-fake --parallel
build-fake/tests/kael_tests --gtest_filter='Cuda*'
```

Build the Python package with CUDA:

```bash
pip install --config-settings=cmake.define.KAEL_ENABLE_CUDA=ON \
    --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=native .
```

### Continuous integration

The CI workflow `.github/workflows/ci.yml` covers the device code in three ways:

| Job | What runs |
|-----|-----------|
| `linux-clang-fake-cuda` (build matrix entry) | Full `ctest` with `KAEL_FAKE_CUDA=ON` and AddressSanitizer plus UndefinedBehaviorSanitizer: every `Cuda.*` test executes against the emulation |
| `cuda-compile-only` | Configures and builds with `KAEL_ENABLE_CUDA=ON` and `CMAKE_CUDA_ARCHITECTURES=86` in the `nvidia/cuda:12.6.2-devel-ubuntu22.04` container; nothing runs |
| `coverage`, `clang-tidy` | Build with `KAEL_FAKE_CUDA=ON` |

GitHub-hosted runners have no GPU, so no CI job executes the real backend. The compile-only job proves that `backend.cu` and the `KAEL_HD` functors are valid CUDA C++. It says nothing about whether the kernels produce the right numbers on hardware. Execution on hardware is validated with `scripts/cuda/validate.sh`, described in the next section.

### Validation on a GPU

#### scripts/cuda/validate.sh

Run it on a machine with an NVIDIA GPU, `nvcc` and CMake 3.24 or newer:

```bash
bash scripts/cuda/validate.sh [architecture] [build-directory] [report-directory]
```

The defaults are `native`, `build-cuda` and `cuda-report`, both directories relative to the repository root, and `-h` or `--help` prints the usage. The script runs named steps, logs each one to `<report>/<step>.log`, and writes a PASS, FAIL or SKIP line per step to `<report>/summary.txt`:

1. `environment`, `cmake-version`, `gpu`: record `uname`, CMake, `nvcc`, compiler and `nvidia-smi` output; check that CMake is recent enough and `nvcc` is on `PATH`.
2. `configure`, `build`: Release build with `KAEL_ENABLE_CUDA=ON`, examples and benchmarks.
3. `ctest`: the whole C++ suite.
4. `device-tests`: `kael_tests --gtest_filter='Cuda*:*CudaMatmul*'` with `CUDA_LAUNCH_BLOCKING=1`. The step fails if the output contains "no CUDA device", because a skipped device test must not count as a pass.
5. `compute-sanitizer`: `Cuda.BackwardMatchesCpuAndStaysOnTheDevice` and `Cuda.IndexingMatchesCpu` under NVIDIA's memory checker (the default `memcheck` tool), if it is installed.
6. `benchmarks`: `kael_bench --benchmark_filter='BM_Device'`.
7. `configure-cublas`, `build-cublas`, `device-tests-cublas`: a second build in `<build>-cublas` with `KAEL_USE_CUBLAS=ON`, limited to the `kael_tests` target, and the same device tests, so the cuBLAS matmul path runs too.
8. `mnist-cpu`, `mnist-cuda`, `mnist-cuda-1024`: the `mnist_mlp` example (`examples/mnist_mlp.cpp`) must reach at least 97 percent test accuracy, if `data/mnist` exists.
9. `char-gpt-cuda`, `gpt2-cpp-cuda`: the transformer examples on the device, if their data exists.
10. `python-build` and the Python steps: a virtual environment with the package built for CUDA, `kael.cuda_is_available()`, `python/tests/test_cuda.py` with no skips, the full pytest suite, and the Python examples on the device. The environment comes from `python3 -m venv`, with `python3 -m virtualenv` as a fallback, because some images such as Google Colab ship `python3` without `ensurepip`.

A failure in `configure` or `build` stops the run and prints the summary. A failure in `configure-cublas` or `build-cublas` drops the cuBLAS device tests, and a failure in `python-build` drops the Python steps. Every other step runs even when an earlier one fails. The exit status is 0 when every step ran and passed, 1 when a step failed, and 3 when nothing failed but some steps were skipped. Exit status 3 is not a complete pass.

The data steps skip when their files are missing. Fetch them with `scripts/download/mnist.sh`, `scripts/download/shakespeare.sh` and `scripts/download/gpt2.sh`. Each download script checks the SHA-256 of every file and keeps files that already match.

#### scripts/cuda/validate_on_colab.ipynb

[`scripts/cuda/validate_on_colab.ipynb`](https://github.com/JetBerri/kael.cpp/blob/main/scripts/cuda/validate_on_colab.ipynb) runs the same validation on Google Colab, for readers without a local NVIDIA GPU. Open it with a GPU runtime, such as a T4, and run every cell. The notebook says the run takes 40 to 60 minutes on a T4, most of it in `nvcc`. The cells do the following:

1. Stop early if no GPU is visible, then record the GPU, driver, CUDA compiler, CMake and C++ compiler.
2. Clone or update the repository at the configured revision (`main` by default).
3. Download the datasets with `scripts/download/*.sh`.
4. Install `virtualenv` and run `scripts/cuda/validate.sh`, streaming its output.
5. Show a table of steps and the key metrics.
6. Pack the logs as `kael-cuda-report-<commit>-<gpu>.tar.gz` and download the archive.

#### scripts/cuda/run_remote.sh

```bash
scripts/cuda/run_remote.sh user@host [--arch native] [--without-weights] [--no-run]
```

The script packs the current commit with `git archive`. Uncommitted changes are not sent, and the script prints a note when there are any. It adds `data/mnist`, `data/shakespeare` and, unless `--without-weights`, `data/gpt2`, copies both archives to `kael.cpp-<commit>` on the host over `scp`, and unpacks them there. It then runs `scripts/cuda/validate.sh` on the host and copies the `cuda-report` directory back as `cuda-report-<commit>-<timestamp>`. `--no-run` stops after the copy. Without a host, or when the first argument starts with `-` (such as `--help`), the script prints its usage and exits with status 2. The exit status of a full run is the exit status of `validate.sh` on the host.

#### Reference run

The reference validation ran on Google Colab with `scripts/cuda/validate_on_colab.ipynb` at commit `06f7dad`, from a clean clone with the datasets fetched by `scripts/download/*.sh`. The commits between it and `v1.0.0` change documentation only.

| Environment | Value |
|-------------|-------|
| GPU | NVIDIA Tesla T4, compute capability 7.5, 15 GB |
| Driver | 580.82.07 |
| CUDA compiler | `nvcc` 13.0.88 |
| Host compiler | GCC 13.3.0 |
| CMake | 3.31.10 |
| System | Ubuntu 24.04, Linux 6.6 x86_64 |

All 23 steps passed:

| Step | Result |
|------|--------|
| `ctest` on the CUDA build | 838 tests, 0 failures |
| `device-tests` | 42 device tests passed with the tile matmul kernel |
| `device-tests-cublas` | 42 device tests passed with `KAEL_USE_CUBLAS=ON` |
| `compute-sanitizer` | 0 errors |
| `mnist-cpu` | 97.18% test accuracy, 14.81 s per epoch |
| `mnist-cuda` | 97.18% test accuracy, 1.52 s per epoch |
| `mnist-cuda-1024` | 97.97% test accuracy with 1024 hidden units, 4.44 s per epoch |
| `char-gpt-cuda` | validation loss from 4.18 to 2.34 in 300 steps |
| `gpt2-cpp-cuda` | GPT-2 small generates text at 14.00 tokens per second |
| `python-device-tests` | 10 tests passed |
| `python-tests` | 346 tests passed |
| `python-mnist-cuda` | 97.18% test accuracy, 1.84 s per epoch |
| `gpt2-cuda` | GPT-2 small generates the same text from Python |

The 42 device tests are the 35 tests of the `Cuda` fixture and the 7 shapes of `CudaMatmul.MatchesCpu`. The MNIST examples in C++ and Python both seed the generator with 42, and the CPU, CUDA and Python CUDA runs report the same 97.18%. The device run is about 9 times faster per epoch than the CPU run.

`kael_bench --benchmark_filter=BM_Device` on the same GPU, with the tile matmul kernel:

| Benchmark | Throughput |
|-----------|------------|
| `BM_DeviceAdd/16777216` | 5.49 G elements/s |
| `BM_DeviceSum/16777216` | 19.87 G elements/s |
| `BM_DeviceMatmul/512` | 368 GFLOP/s |
| `BM_DeviceMatmul/1024` | 401 GFLOP/s |
| `BM_DeviceMatmul/2048` | 427 GFLOP/s |

## Edge cases and pitfalls

- **Host access to device memory.** Every host accessor checks the device. Only `device_data()` returns a device address, and its comment in `include/kael/tensor.h` warns that only device code can dereference it.
- **Mixed devices.** `check_same_device` throws "expected all tensors to be on the same device, but found cpu and cuda" (`Cuda.MixedDevicesAreRejected`). Implicit transfers would hide slow host round trips inside innocent-looking expressions.
- **Non-contiguous transfers.** `to(Device)` makes the source contiguous first (`Cuda.NonContiguousTensorsTransfer`).
- **Empty tensors.** `to` skips the copy when `nbytes()` is 0. `for_each` returns before launching when `count` is 0, since a launch with zero blocks is an error. `Cuda.EmptyTensorsWork` checks elementwise ops and reductions on empty tensors. The cuBLAS branch of `device::matmul` returns early on a zero dimension, before it calls `cublasSgemm` or `cublasDgemm`.
- **Too many dimensions or operands.** `device_layout()` throws when the layout exceeds `kMaxDims` or `kMaxOperands`, and `index_layout()` throws when the index tensor exceeds `kMaxDims`, instead of overflowing the fixed arrays.
- **Out-of-range indices.** Checked on the host before launching gather and scatter.
- **64-bit indexing.** Kernel index arithmetic uses `std::int64_t`. A kernel with 32-bit indices would wrap silently above `2^31` elements.
- **Order assumptions.** The emulation runs work items backwards to expose them.
- **Concurrent scatter.** `ScatterAdd` uses atomics. `bool` scatter has no device atomic, so `scatter_add_` on a `bool` device tensor runs on the host.
- **Non-determinism.** Device sums are deterministic. `scatter_add` with repeated indices on a real GPU is not: atomic float additions land in an unspecified order, and float addition is not associative.
- **Asynchronous errors.** A fault inside a kernel is reported by a later CUDA call, not the launch. Use `CUDA_LAUNCH_BLOCKING=1` to locate it.
- **Unchecked frees.** `release` calls `cudaFree` without checking the result, because it runs in `noexcept` destructors.

> [!TIP]
> When a device test fails under `KAEL_FAKE_CUDA`, debug it like any CPU bug. Set a breakpoint in the task's `operator()`. The emulation runs the task on the host thread, so a debugger, AddressSanitizer and `printf` all work.

## How it is tested

[`tests/test_cuda.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_cuda.cpp) defines the `Cuda` fixture, which calls `GTEST_SKIP()` with "no CUDA device" when `cuda_is_available()` is false and seeds the generator with 2024 otherwise. With `KAEL_FAKE_CUDA=ON` every test runs. Most tests compute the same expression on the CPU and on the device and compare with `expect_matches_cpu`. It checks that the device result is on `Device::CUDA`, has the same shape and dtype, and matches within `1e-4` for `f32` or `1e-10` for `f64`, with NaN equal to NaN.

The CPU is the oracle. Every kernel in this chapter has a host reference from [Operations](@/book/05-operations.md) that is already tested against PyTorch golden values, so the device tests only need to prove agreement with the host.

| Test | What it proves |
|------|----------------|
| `TransfersRoundTrip`, `SameDeviceReturnsTheSameTensor`, `NonContiguousTensorsTransfer`, `CopyBetweenDevicesWorks` | Transfers preserve values and layout semantics |
| `FactoriesCreateOnTheDevice`, `ScalarsFollowTheTensorDevice` | Factories and scalars respect the device |
| `HostOnlyAccessFailsClearly`, `ReadingValuesCopiesToHost`, `MixedDevicesAreRejected` | Device boundary rules |
| `UnaryOpsMatchCpuOnEveryLayout`, `BinaryOpsMatchCpuWithBroadcast`, `DoublePrecisionMatchesCpu`, `IntegerOpsMatchCpu`, `BoolReductionsMatchCpu`, `NonFiniteValuesMatchCpu` | 43 unary and reduction ops on contiguous, transposed and strided inputs, and 19 binary ops with same-shape, row, column and 0-d broadcasts, match the CPU in `f32` and `f64` |
| `LongSumsMatchCpuAndAreDeterministic`, `EmptyTensorsWork` | The block reduction and empty shapes |
| `IntegerClampAndMaskedFillKeepEveryBit`, `IntegerSumWrapsAndFloatConversionSaturates`, `IntegerScatterAddWrapsAround` | `i64` clamp bounds and fill values above `2^53` stay exact; integer sums and scatter adds wrap and float to `i64` saturates as on the CPU |
| `CudaMatmul.MatchesCpu`, `MatmulBatchesVectorsAndIntegersMatchCpu` | Matmul shapes, batches, vectors and `i64` |
| `IndexingMatchesCpu`, `OutOfRangeIndicesStillThrow`, `SteppedIndicesWorkOnTheDevice` | Gather, scatter, index checks |
| `BackwardMatchesCpuAndStaysOnTheDevice`, `GradientsCrossDevicesBackToTheOrigin`, `ViewAndIndexingGradientsMatchCpu`, `GradcheckRunsOnTheDevice` | Autograd on the device and across devices |
| `ModuleToKeepsParameterIdentityAndValues`, `ModuleToMovesExistingGradients`, `LayersMatchCpu`, `TrainingMatchesCpuStepByStep` | `nn` and `optim` on the device |
| `DeviceBlocksAreCached` | The device allocator cache |
| `SequenceOpsMatchCpu`, `GptMatchesCpuIncludingTheCacheAndGeneration` | Host-fallback ops and the full GPT model with KV cache |

`CudaMatmul.MatchesCpu` is a parameterized test over the shapes `1x1x1`, `4x4x4`, `5x3x9`, `8x16x7`, `33x20x65`, `64x1x64` and `1x64x1` (rows, inner, columns). Shapes that are not multiples of 4 exercise the clipped edge tiles.

`tests/test_cuda_absent.cpp` (`CudaAbsent`) runs in builds without a device and skips itself when one is present. `python/tests/test_cuda.py` repeats the main checks from Python and skips without a device.

CI runs these tests against the host emulation on every push. Passing them there shows that the kernel logic and the dispatch are correct. It does not show that the CUDA build behaves the same on a GPU. `scripts/cuda/validate.sh` runs the same tests on a GPU, with the tile kernel and with cuBLAS, and the [reference run](@/book/09-cuda-backend.md#reference-run) passed all of them on a T4.

## Build it yourself

1. Add `Device` to `Storage` and `TensorImpl`, and write `include/kael/device_kernels.h` with only `available`, `allocate`, `release` and the three copy functions. Checkpoint: the project builds with `backend_none.cpp`.
2. Write `backend_fake.cpp` with `memcpy` copies and `operator new`, and add the `KAEL_FAKE_CUDA` CMake option. Implement `Tensor::to(Device)`, `cuda()`, `cpu()` and `check_on_host`. Checkpoint: `Cuda.TransfersRoundTrip` and `Cuda.HostOnlyAccessFailsClearly` pass with `-DKAEL_FAKE_CUDA=ON`.
3. Route device storage through a second `BlockCache`. Checkpoint: `Cuda.DeviceBlocksAreCached`.
4. Create `kernels.inl` with `Fill` and `device::fill`, and the backwards `for_each`. Checkpoint: `Cuda.FactoriesCreateOnTheDevice`.
5. Add `ElementLayout`, `TensorIterator::device_layout()`, `locate`, `Element`, `convert`, and the `Tagged` functors with `KAEL_HD`. Branch in `run_typed` and `convert_strided`, and add the host fallback for untagged functors. Checkpoint: `Cuda.UnaryOpsMatchCpuOnEveryLayout` and `Cuda.BinaryOpsMatchCpuWithBroadcast`.
6. Add `ReduceRows`, `ReduceBlocks`, `MergePairs`, `StoreResult` and the device branches in `src/ops/reduce.cpp`. Checkpoint: `Cuda.LongSumsMatchCpuAndAreDeterministic`.
7. Add `MatmulTiles` and the device branch of `multiply_batches`. Checkpoint: `CudaMatmul.MatchesCpu`.
8. Add `IndexLayout`, `Gather`, `ScatterAdd` with `atomic_add`, and host-side index checks. Checkpoint: `Cuda.IndexingMatchesCpu` and `Cuda.OutOfRangeIndicesStillThrow`.
9. Add `record_to_device` and `Module::to` with `move_to_`. Checkpoint: `Cuda.TrainingMatchesCpuStepByStep` and `Cuda.GptMatchesCpuIncludingTheCacheAndGeneration`.
10. Write `backend.cu` with the grid-stride `run_task`, `KAEL_CUDA_CHECK` and the CUDA runtime calls, then the optional cuBLAS path. Checkpoint: `cmake -DKAEL_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86` builds in the `nvidia/cuda` container.
11. On a GPU machine, run `scripts/cuda/validate.sh`. Checkpoint: exit status 0. Without a local GPU, run `scripts/cuda/validate_on_colab.ipynb` on Colab instead.

## Check your understanding

<details>
<summary>Why does kael name device operations with an enum instead of passing the functor to the backend?</summary>

A functor passed to the backend would have to be instantiated inside a kernel template, which requires the CUDA compiler for every file that calls it. With `ElementOp`, the operation files stay plain C++20 and only `backend.cu` is compiled by `nvcc`. The backend maps each enum value back to the same `KAEL_HD` functor in one `switch`, so the semantics still come from a single definition.

</details>

<details>
<summary>A device kernel for element 5 of the a + b example reads which bytes of b?</summary>

Element 5 decomposes into position `5 % 3 = 2` in dimension 1 and `(5 / 3) % 2 = 1` in dimension 0. The offset into `b` is `1 * 0 + 2 * 4 = 8` bytes, so it reads `b[2]`. The output and `a` offsets are `1 * 12 + 2 * 4 = 20` bytes, which is `[1][2]`.

</details>

<details>
<summary>Why are device sums deterministic, and why is scatter_add with repeated indices not?</summary>

The block reduction fixes the order of additions by the block size: each 4096-element block is summed in order by one work item, and `MergePairs` combines partials in a fixed tree. Thread scheduling never changes which numbers are added together. `scatter_add` resolves collisions with `atomicAdd`, and the order in which threads reach the atomic is not defined. Float addition is not associative, so different orders can give different results.

</details>

<details>
<summary>What does running work items backwards in the emulation catch, and what can it never catch?</summary>

It catches tasks that silently depend on item `i - 1` having finished before item `i`, which GPU threads never guarantee. Because the loop is deterministic, the failure reproduces on every run. It cannot catch races between threads, since one host thread runs every item, nor launch limits, device math differences or CUDA runtime behavior.

</details>

<details>
<summary>Why does nn::Module::to call move_to_ on each parameter instead of assigning parameter.to(device)?</summary>

`to` returns a new tensor with a new `TensorImpl`. An optimizer created before the move holds handles to the old `TensorImpl`s and would keep updating CPU tensors that the model no longer uses. `move_to_` replaces the storage inside the existing `TensorImpl`, so every handle, including the optimizer's, sees the device tensor.

</details>

<details>
<summary>The cuda-compile-only job is green. What has been proven about the GPU backend?</summary>

Only that `backend.cu`, `kernels.inl` and the `KAEL_HD` functors compile with `nvcc` for architecture 86 against CUDA 12.6. No kernel runs in that job. Inside CI, the correctness evidence comes from the emulation, which runs the same task bodies on the host. Behavior on hardware is verified only when `scripts/cuda/validate.sh` exits with status 0 on a GPU, which CI does not do.

</details>

## Further reading

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/): the programming model, memory hierarchy, streams and error checking that this chapter simplifies.
- [CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/): the launch pattern used by `run_task`, with its trade-offs.
- Mark Harris, "Optimizing Parallel Reduction in CUDA": the shared-memory reduction trees a production backend would use instead of `MergePairs`.
- [cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/): column-major storage, leading dimensions and `cublasSgemm`, which explain the transpose trick.
- [Compute Sanitizer documentation](https://docs.nvidia.com/compute-sanitizer/): the memory checker that `scripts/cuda/validate.sh` runs on real hardware, and the race checker for shared-memory hazards, which kael's kernels cannot have because they use no shared memory.
- [PyTorch CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html): asynchronous execution, the caching allocator and device selection in a production framework.
- [`src/cuda/kernels.inl`](https://github.com/JetBerri/kael.cpp/blob/main/src/cuda/kernels.inl): under 500 lines, every device task in one file.

Next: [10. Python bindings](@/book/10-python-bindings.md)
