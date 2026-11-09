+++
title = "CPU performance"
description = "A thread pool, a six-step matmul ladder, a caching allocator and fused row kernels that make kael fast on a CPU without changing a single result."
weight = 8
slug = "cpu-performance"

[extra]
chapter = "08"
kicker = "Chapter 08"
builds = "A thread pool, a fast matmul, a caching allocator and fused kernels"
source = "docs/08-cpu-performance.md"
tag = "chapter-09-cpu-performance"
+++

The previous chapters produce a correct framework that trains an MLP on MNIST. This chapter makes it fast on a CPU without changing any result. It adds a thread pool with a `parallel_for` primitive, splits elementwise kernels and reductions across threads, builds matrix multiplication as a ladder of six versions from a textbook triple loop to a packed, register-blocked, multithreaded kernel, adds a caching block allocator, and fuses softmax, log softmax, layer normalization and cross entropy into single-kernel row loops. It maps to milestone v0.2 (performance) and to the git tag `chapter-09-cpu-performance`.

| File | Role |
|------|------|
| [`include/kael/parallel.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/parallel.h) | `parallel_for`, `set_num_threads`, `get_num_threads`, `kElementGrain` |
| [`src/parallel.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/parallel.cpp) | `Job`, `ThreadPool`, `ParallelRuntime`, thread count detection, fork handling |
| [`include/kael/matmul_versions.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/matmul_versions.h) | The six matmul versions, `blas`, `multiply` dispatch |
| [`src/linalg/matmul_versions.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/linalg/matmul_versions.cpp) | Loop orders, tiling, packing, micro-kernels, threading, BLAS calls |
| [`src/ops/linalg.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/ops/linalg.cpp) | `matmul` front end: batching, dtype dispatch, exact integer path |
| [`src/ops/reduce.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/ops/reduce.cpp) | Blocked, lane-split, thread-count-independent reductions |
| [`src/ops/nn.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/ops/nn.cpp) | Fused row kernels for softmax, log softmax, `standardize` and cross entropy |
| [`include/kael/allocator.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/allocator.h), [`src/allocator.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/allocator.cpp) | Caching block allocator and its statistics |
| [`include/kael/iterator.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/iterator.h) | `TensorIterator::run`, which splits elementwise work |
| [`benchmarks/`](https://github.com/JetBerri/kael.cpp/tree/main/benchmarks) | Google Benchmark suites and `compare.py` |

## Prerequisites

- [02. Memory model](@/book/02-memory-model.md): `Storage`, alignment, and how a tensor owns its bytes.
- [04. TensorIterator](@/book/04-tensor-iterator.md): the two-level loop that this chapter parallelizes.
- [05. Operations](@/book/05-operations.md): reductions, `matmul`, softmax and their reference semantics.
- [06. Autograd](@/book/06-autograd.md): grad mode, which worker threads must inherit.
- [07. Neural networks and training](@/book/07-neural-networks.md): `LayerNorm` and `cross_entropy`, which use the fused kernels.

## Why this layer exists

At the end of chapter 07, kael is correct but slow. Every kernel runs on one core. Matrix multiplication is a loop that a compiler cannot turn into fast code by itself. Every temporary tensor goes through the system allocator. Softmax walks memory five times where one walk would do. A training step for a small MLP spends most of its time in `matmul`, and a modern CPU has many cores and wide vector units that sit idle.

Without this layer, three things break in practice:

- **Training time.** The MNIST example of chapter 07 takes much longer per epoch than the same model in PyTorch, which makes experiments painful and the comparison with real frameworks meaningless.
- **Reproducibility under naive parallelism.** The obvious way to parallelize a sum (each thread sums its share, then add the shares) gives results that change in the last bits when the thread count changes. A test that passes on a laptop then fails on a CI machine with a different core count.
- **Process safety.** A thread pool that ignores `fork()` deadlocks Python programs that use `multiprocessing`, which includes many data loaders.

The mental model for this chapter is a single sentence: **performance work moves data less, not computes less.** Every technique here has the same goal. Threads add more cores pulling on the memory bus. Packing and register blocking reuse a value many times once it is loaded. The caching allocator avoids touching fresh pages. Fused kernels read a row once instead of five times. None of them changes the arithmetic that defines a result, and the tests check that the bits stay the same.

A second rule follows from the first: **a result must not depend on how work is split.** kael picks block boundaries from the problem size, never from the thread count. That choice costs a little flexibility and buys bit-identical output for 1 thread and for 16.

## Concepts

### Where the time goes

Two numbers describe most CPU kernels.

- **Memory traffic**: the bytes a kernel reads from and writes to main memory.
- **Arithmetic intensity**: floating-point operations (flops) per byte of memory traffic.

A kernel with low arithmetic intensity is **memory bound**: the arithmetic units wait for data. Adding threads helps only until the memory bus saturates, and SIMD helps little. A kernel with high arithmetic intensity is **compute bound**: the arithmetic units are the limit, and both SIMD and threads pay off. The **roofline model** (see Further reading) draws this as a graph: attainable flops per second rise linearly with intensity until they hit the machine's peak, then stay flat.

```text
flops/s
  ^
  |                 _________________________  peak compute
  |                /
  |               /   compute bound region
  |              /
  |             /   slope = memory bandwidth
  |            /
  |  memory   /
  |  bound   /
  |         /
  +--------+---------------------------------> flops per byte
         ridge point
```

Compare elementwise addition and matrix multiplication on `float` data (4 bytes per element).

| Kernel | Bytes moved | Flops | Intensity |
|--------|-------------|-------|-----------|
| `out = a + b`, n elements | 12n (two reads, one write) | n | 1/12 flop per byte |
| `C += A B`, all n x n | at least 12n^2 | 2n^3 | n/6 flops per byte |

Trace the matmul row for n = 1024:

```text
A, B, C each hold 1024 * 1024 = 1 048 576 floats = 4 MiB
distinct data: A and B read, C written       = 12 MiB
multiply-adds: 1024^3                        = 1 073 741 824
flops (one multiply + one add each)          = 2 147 483 648
intensity: 2 147 483 648 / 12 582 912 bytes  = about 171 flops per byte
```

Each element of A and B takes part in n = 1024 multiply-adds. If the kernel reloads operands from main memory for every use, it behaves like the addition and wastes that reuse. The whole matmul ladder below is about turning the theoretical reuse into reuse that happens in registers and caches.

Addition cannot be improved that way: each element is used once. If memory delivers B bytes per second, the addition does at most B / 12 additions per second, whatever the CPU. For elementwise work the remaining levers are threads (more cores pulling on the memory bus), avoiding per-element index arithmetic ([chapter 04](@/book/04-tensor-iterator.md)), and not allocating.

### The memory hierarchy in one picture

A CPU core does not read main memory directly. It reads from a stack of caches, each smaller and faster than the one below. Data moves between levels in **cache lines** of 64 bytes on common CPUs, which is 16 `float` values.

```text
                size (typical order)   what kael keeps there
registers       hundreds of bytes      the 8 x 12 output tile of the micro-kernel
L1 cache        tens of KiB            one packed B panel (24 KiB for float)
L2 cache        hundreds of KiB to MiB one packed A block
L3 / shared     several MiB            one packed B block
main memory     GiB                    the tensors
```

The exact sizes vary by CPU. The ordering does not, and the block sizes in the matmul ladder are chosen so that each operand sits at the level where it is reused most.

Two access patterns matter:

- **Unit stride**: consecutive elements in memory. One cache line load serves 16 floats, and the hardware **prefetcher** (a unit that predicts the next lines and loads them early) keeps up.
- **Large stride**: each access lands in a new cache line and uses 4 of its 64 bytes. The other 60 bytes are wasted bandwidth.

### Threads, grains and chunks

A **thread pool** is a set of threads created once and reused for many jobs. Creating a thread costs a system call and a new stack; waking a sleeping one costs far less. kael's pool sleeps between jobs and wakes for each `parallel_for`.

A **grain** is the smallest range of work worth handing to another thread. Below it, the cost of waking a thread and synchronizing exceeds the work. A **chunk** is the range a thread actually claims at once. kael makes chunks at least one grain long, and cuts the range into a few chunks per thread so that a fast thread can take extra chunks while a slow one finishes. That is **dynamic load balancing**.

PyTorch has the same primitive. `at::parallel_for(begin, end, grain_size, f)` in ATen has the same shape as `kael::parallel_for`, and `torch.set_num_threads` plays the role of `kael::set_num_threads`.

### Floating-point addition is not associative

In exact arithmetic, `(a + b) + c == a + (b + c)`. In floating point it is not. A worked example in `float`, where values near 10^8 are spaced 8 apart:

```text
(1e8 + 1) - 1e8   ->  1e8 + 1 rounds to 1e8   ->  0
(1e8 - 1e8) + 1   ->  0 + 1                   ->  1
```

Any change in the order of additions can change the last bits of a sum. A parallel sum whose shares follow thread boundaries uses a different order for each thread count. kael fixes the order by fixing the blocks: their boundaries depend on the input size only.

The same fact explains why compilers do not vectorize `total += x[i]` on their own. Vectorizing means keeping several partial sums and adding them at the end, which is a different order. A compiler may do that only with `-ffast-math`, which gives up exact IEEE semantics. kael writes the partial sums into the source code, so the vector order is the program's order.

### SIMD and the register budget

**SIMD** (single instruction, multiple data) instructions operate on a vector register that holds several values. A 128-bit NEON register on ARM holds 4 floats or 2 doubles. An **FMA** (fused multiply-add) computes `c + a * b` in one instruction with one rounding.

A fast matmul keeps a block of the output in registers for the whole inner loop and only loads the inputs. The size of that block is limited by the number of vector registers. On AArch64 there are 32 of them. Spending them well is the job of the micro-kernel in v5.

### Fusion

**Kernel fusion** merges several operations into one loop so that intermediate values stay in registers or cache instead of going to memory. The composed softmax `exp(x - max(x)) / sum(exp(...))` runs five kernels and allocates three full-size temporaries. A fused softmax reads a row, keeps it in cache across three short passes, and writes the result once. The math is the same; only the memory traffic changes.

## Implementation

The pieces are built in dependency order: the thread pool first, because everything else uses it; then elementwise kernels and reductions, which only need `parallel_for`; then matmul, the allocator and the fused kernels.

### The thread pool

#### Design

`parallel_for(begin, end, grain, body)` calls `body(first, last)` on disjoint sub-ranges whose union is `[begin, end)`. The caller passes a grain. From `include/kael/parallel.h`:

```cpp
void set_num_threads(std::int64_t count);
[[nodiscard]] std::int64_t get_num_threads();

inline constexpr std::int64_t kElementGrain = 524288;

using ParallelBody = std::function<void(std::int64_t, std::int64_t)>;

void parallel_for(std::int64_t begin, std::int64_t end, std::int64_t grain,
                  const ParallelBody& body);
```

`kElementGrain` (2^19) is the grain for one cheap operation per element. Kernels with more work per index divide it by their cost, as the reductions and fused kernels below show.

The body takes a range, not a single index. The call goes through `std::function`, which is an indirect call. Paying it once per chunk is negligible; paying it once per element would dominate a cheap kernel. That is why the inner loop stays inside the body.

The runtime has three pieces in `src/parallel.cpp`:

```text
ParallelRuntime  (one per process, a function-local static)
  mutex_                       guards the pool; taken with try_to_lock
  thread_count_                atomic, read by get_num_threads
  pool_ : unique_ptr<ThreadPool>   created lazily on first use
  try_run(Job&) -> bool
  set_thread_count(count)          resets pool_
        |
        | creates lazily, rebuilds after set_thread_count
        v
ThreadPool
  workers_ : vector<thread>        thread_count - 1 workers
  generation_                      bumped once per job
  busy_workers_                    workers still inside the job
  run(Job&)
        |
        | runs one Job at a time
        v
Job  (lives on the caller's stack)
  next_ : atomic<int64>            cursor of the next unclaimed index
  chunk_size_
  error_ : exception_ptr           first exception thrown by the body
  grad_enabled_                    caller's grad mode
  work()
  rethrow_error()
```

#### The parallel_for algorithm

From `src/parallel.cpp`:

```cpp
void parallel_for(std::int64_t begin, std::int64_t end, std::int64_t grain,
                  const ParallelBody& body) {
    KAEL_CHECK_VALUE(grain > 0, "parallel_for grain must be positive, got " << grain);
    const std::int64_t length = end - begin;
    if (length <= 0) {
        return;
    }
    const std::int64_t threads = inside_parallel_region ? 1 : get_num_threads();
    if (length <= grain || threads == 1) {
        body(begin, end);
        return;
    }
    const std::int64_t chunks = threads * kChunksPerThread;
    const std::int64_t chunk_size = std::max(grain, (length + chunks - 1) / chunks);
    Job job(begin, end, chunk_size, body);
    job.start();
    if (ParallelRuntime::instance().try_run(job)) {
        job.rethrow_error();
    } else {
        body(begin, end);
    }
}
```

The range is cut into about `threads * kChunksPerThread` chunks (`kChunksPerThread = 4`), never smaller than the grain. Four chunks per thread give dynamic load balancing: a thread that finishes early takes another chunk instead of waiting for a slow one. The grain floor keeps chunks big enough to amortize scheduling.

Worked example with 8 threads and `grain = kElementGrain = 524288`:

| Range length | Chunks wanted | `chunk_size` | Chunks run |
|--------------|---------------|--------------|------------|
| 100 | none | none | 1 inline call (`length <= grain`) |
| 1 000 000 | 32 | `max(524288, 31250) = 524288` | 2 |
| 4 000 000 | 32 | `max(524288, 125000) = 524288` | 8 |

Adding two 1000 x 1000 tensors therefore uses two chunks. The grain says that splitting a memory-bound add further does not pay.

A second trace, with 4 threads and a length of 3 000 000:

```text
chunks wanted = 4 * 4                         = 16
even share    = ceil(3 000 000 / 16)          = 187 500
chunk_size    = max(524 288, 187 500)         = 524 288
chunks run    = ceil(3 000 000 / 524 288)     = 6

chunk 0: [0,         524 288)
chunk 1: [524 288,   1 048 576)
chunk 2: [1 048 576, 1 572 864)
chunk 3: [1 572 864, 2 097 152)
chunk 4: [2 097 152, 2 621 440)
chunk 5: [2 621 440, 3 000 000)    378 560 elements
```

Six chunks over four threads means two threads take a second chunk. The grain wins over the per-thread share here, so balance is coarse. That is a deliberate trade: for a memory-bound kernel, finer chunks would not move more bytes per second.

When the grain is small, the per-thread share wins instead. With 4 threads, a length of 10 and a grain of 1, `chunk_size = max(1, ceil(10 / 16)) = 1`, so ten chunks of one index each are handed out.

#### Chunk claiming

Each `Job` holds an atomic cursor. Every participating thread, including the caller, loops on `fetch_add`:

```cpp
void work() noexcept {
    const autograd::GradModeGuard grad_mode(grad_enabled_);
    const bool was_inside = std::exchange(inside_parallel_region, true);
    for (;;) {
        const std::int64_t start = next_.fetch_add(chunk_size_, std::memory_order_relaxed);
        if (start >= end_) {
            break;
        }
        try {
            body_(start, std::min(start + chunk_size_, end_));
        } catch (...) {
            keep_first_error(std::current_exception());
            next_.store(end_, std::memory_order_relaxed);
        }
    }
    inside_parallel_region = was_inside;
}
```

A timeline of the 3 000 000 element job above, with one possible interleaving:

```text
cursor   caller        worker 1      worker 2      worker 3
0        claim 0
524288                 claim 524288
1048576                              claim 1048576
1572864                                            claim 1572864
2097152  claim 2097152                (still busy)
2621440                claim 2621440
3145728  fetch_add returns 3145728 >= end, stop
         ...every thread sees start >= end_ on its next claim and leaves
```

Three details matter:

1. **Relaxed ordering is enough for the cursor.** `fetch_add` is atomic whatever the memory order, so each start value is handed out once. The data the body writes is published to the caller by the mutex and condition variable in `ThreadPool::run`, not by the cursor.
2. **The first exception is kept** and rethrown on the calling thread by `rethrow_error`. Storing `end_` into the cursor stops the other threads from claiming new chunks. Without the capture, an error in one chunk would terminate the process, because `work` is `noexcept` and runs on a worker. `keep_first_error` takes a small mutex so that two failing chunks do not race on `error_`.
3. **Grad mode is thread-local** ([chapter 06](@/book/06-autograd.md)). The `Job` constructor captures the caller's mode with `autograd::GradMode::is_enabled()` and `work` installs it on every participant with `GradModeGuard`. Otherwise a kernel running under `NoGradGuard` would record graph nodes on the workers.

#### The pool

`ThreadPool` owns `thread_count - 1` workers. The calling thread is the last participant, so `set_num_threads(4)` means four threads do the work, not five. The caller does useful work instead of blocking while others compute.

If starting a worker throws, for example `std::system_error` when the system refuses a new thread, the constructor stops and joins the workers already started before it rethrows. Otherwise the destructors of joinable `std::thread` objects would call `std::terminate`. The `kMaxThreads` limit of 1024 keeps a mistyped count from asking the system for millions of threads.

From `src/parallel.cpp`:

```cpp
void run(Job& job) {
    {
        const std::lock_guard lock(mutex_);
        job_ = &job;
        ++generation_;
        busy_workers_ = workers_.size();
    }
    wake_workers_.notify_all();
    job.work();
    std::unique_lock lock(mutex_);
    workers_done_.wait(lock, [this] { return busy_workers_ == 0; });
    job_ = nullptr;
}
```

The worker side:

```cpp
void worker_loop() {
    std::uint64_t seen_generation = 0;
    std::unique_lock lock(mutex_);
    for (;;) {
        wake_workers_.wait(lock, [&] { return stopping_ || generation_ != seen_generation; });
        if (stopping_) {
            return;
        }
        seen_generation = generation_;
        Job* job = job_;
        lock.unlock();
        job->work();
        lock.lock();
        if (--busy_workers_ == 0) {
            workers_done_.notify_one();
        }
    }
}
```

Workers sleep on `wake_workers_` and wait for `generation_` to change. A generation counter instead of a boolean flag avoids two classic bugs:

- **Missed wake-up.** With a flag, a worker that was not yet waiting when the flag went up and down again would sleep through a job. The counter has changed, so the predicate is true whenever the worker checks it.
- **Doubled wake-up.** With a flag that stays set, a fast worker could finish, loop, and run the same job again. Each worker remembers the generation it ran, so it runs each generation exactly once.

The caller waits until every worker has left `job.work()`, because the `Job` lives on the caller's stack. Returning earlier would leave a worker reading a destroyed object.

#### One job at a time

`ParallelRuntime::try_run` takes the runtime mutex with `std::try_to_lock`:

```cpp
bool try_run(Job& job) {
    const std::unique_lock lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
        return false;
    }
    if (!pool_ && thread_count() > 1) {
        pool_ = std::make_unique<ThreadPool>(thread_count() - 1);
    }
    if (!pool_) {
        return false;
    }
    pool_->run(job);
    return true;
}
```

If another thread already runs a job, the second caller does not queue. It runs its whole range inline on its own thread. Two Python threads calling `matmul` at once therefore both make progress: one uses the pool and the other runs serially. The test `ParallelThreads.SecondCallerDoesNotWaitForTheActiveJob` checks this.

The alternative, a work queue shared by all callers, would let both callers use the pool. It would also need a scheduler, fairness rules and a way to stop one caller's chunks from starving the other's. kael keeps one job at a time because it is short to write, cannot deadlock, and never puts more threads to work than the configured count.

The pool is created on first use. `set_thread_count` resets it, and the next `try_run` builds a new one with the new size.

#### Nested parallel regions

`inside_parallel_region` is a `thread_local` flag set by `Job::work`. A `parallel_for` called from inside a body sees it and runs inline:

```cpp
const std::int64_t threads = inside_parallel_region ? 1 : get_num_threads();
```

Nesting happens in practice. Batched `matmul` in `src/ops/linalg.cpp` splits batches with `parallel_for`, and each batch calls `matmul_versions::multiply`, which also uses `parallel_for`.

```text
matmul([8, 512, 512] x [8, 512, 512]), 8 threads

outer parallel_for over 8 batches, grain 1
  thread 0: batch 0 -> multiply -> parallel_for (flag set) -> runs inline
  thread 1: batch 1 -> multiply -> parallel_for (flag set) -> runs inline
  ...
  thread 7: batch 7 -> multiply -> parallel_for (flag set) -> runs inline

matmul([2, 32, 32] x [2, 32, 32]), 8 threads

outer parallel_for over 2 batches, grain 64 -> length <= grain -> inline, flag unset
  caller: batch 0 -> multiply -> v2_ikj (small problem, no threads)
  caller: batch 1 -> multiply -> v2_ikj
```

When the batches are split, each matrix runs single-threaded inside its chunk. When the batch grain covers the whole batch, the outer call runs inline on the caller with the flag unset, and the inner matmul may use all threads. Either way the machine is used once, not **oversubscribed** (more runnable threads than cores, which wastes time on context switches).

`set_num_threads` refuses to run inside a region (`ParallelThreads.ChangingThreadCountInsideBodyThrows`), because it resets the pool that is running the body. Reading the count inside a body is allowed (`ParallelThreads.ThreadCountIsReadableInsideBody`).

#### Choosing the thread count

From `src/parallel.cpp`:

```cpp
constexpr std::int64_t kMaxThreads = 1024;

std::int64_t default_thread_count() {
    const char* configured = std::getenv("KAEL_NUM_THREADS");
    if (configured == nullptr) {
        return hardware_thread_count();
    }
    char* end = nullptr;
    const std::int64_t parsed = std::strtoll(configured, &end, 10);
    KAEL_CHECK_VALUE(end != configured && *end == '\0' && parsed > 0 && parsed <= kMaxThreads,
                     "KAEL_NUM_THREADS must be an integer between 1 and "
                         << kMaxThreads << ", got '" << configured << "'");
    return parsed;
}
```

The order of precedence is:

1. `set_num_threads(count)` at runtime. It rejects a count outside `1..kMaxThreads` (1024) with `ValueError`. Python exposes it as `kael.set_num_threads`.
2. `KAEL_NUM_THREADS`, read once when the runtime is first used. Values such as `abc`, `0`, `-2`, `4x` or `100000` throw `ValueError` instead of being ignored, so a typo does not fall back to the hardware count unnoticed. `strtoll` stops at the first non-digit; the `*end == '\0'` check is what rejects `4x`.
3. `hardware_thread_count()`. On Apple platforms it reads `hw.perflevel0.physicalcpu`, the number of performance cores. Efficiency cores slow matmul and elementwise kernels down: the threaded matmul gives every thread one row block of the same size, so the slowest core sets the finish time. Elsewhere it uses `std::thread::hardware_concurrency()`, at least 1.

```bash
KAEL_NUM_THREADS=1 build/tests/kael_tests --gtest_filter='*MatmulVersions*'
```

The leading `*` in the filter matters: the parameterized suite is registered as `Shapes/MatmulVersions`, so `MatmulVersions*` alone would skip it.

#### Fork safety

After `fork()`, the child has a copy of the parent's memory but only the thread that called `fork`. The pool's worker threads do not exist in the child.

```text
parent before fork          child after fork
caller thread    -------->  caller thread (only survivor)
worker 1 (sleeping)         gone, but ThreadPool still lists it
worker 2 (sleeping)         gone
mutex_ (maybe held)         copied in whatever state it had
```

Calling into the pool would wait forever for workers that never answer. Destroying it would try to join threads that do not exist. A mutex held by a parent thread at the moment of the fork would stay locked forever.

`ParallelRuntime` registers a `pthread_atfork` child handler on POSIX systems in its constructor:

```cpp
// The workers do not exist in a forked child, so the pool can be neither used nor destroyed.
void forget_threads_after_fork() {
    new (&mutex_) std::mutex;
    [[maybe_unused]] const auto* forgotten = pool_.release();
    inside_parallel_region = false;
}
```

The child rebuilds the mutex in place, leaks the pool object on purpose, and clears the region flag. The next `parallel_for` in the child creates a fresh pool. This matters for Python: `multiprocessing` with the `fork` start method, and data loaders built on it, fork a process that already ran kael kernels.

> [!WARNING]
> Leaking the pool is the only safe choice. `pool_.reset()` would run `~ThreadPool`, which joins workers that do not exist in the child, and the child would hang or crash.

### Splitting elementwise work

`TensorIterator::run` in `include/kael/iterator.h` is the single place where elementwise kernels go parallel:

```cpp
template <typename InnerLoop> void TensorIterator::run(InnerLoop&& inner_loop) const {
    check_runs_on_host();
    if (element_count_ == 0) {
        return;
    }
    parallel_for(0, element_count_, kElementGrain, [&](std::int64_t first, std::int64_t last) {
        run_range(first, last, inner_loop);
    });
}
```

The split is over the linear element index of the coalesced iteration space ([chapter 04](@/book/04-tensor-iterator.md)). `run_range` converts `first` into a multi-dimensional counter once, then walks the two-level loop until `last`. Each chunk writes a disjoint part of the output, so no synchronization is needed.

```text
output [4, 6], 3 chunks of 8 elements (grain lowered for the picture)

linear:  0  1  2  3  4  5 | 6  7  8  9 10 11 | 12 ...
chunk A: [0, 8)   starts at counter (0, 0), ends inside row 1
chunk B: [8, 16)  starts at counter (1, 2)
chunk C: [16, 24) starts at counter (2, 4)
```

A chunk can start in the middle of a row. That is why `run_range` computes the start counter from the linear index instead of assuming that chunks align with rows.

Every binary, unary, comparison, `where`, `clamp` and dtype conversion kernel inherits threading from this one call. Every element is computed by the same expression whatever the chunk boundaries are, so elementwise results do not depend on the thread count. `ParallelIterator.ResultsDoNotDependOnThreadCount` checks it for a broadcast add, an `exp` of a transposed tensor and a transpose copy.

### Splitting reductions

Reductions have two shapes of parallelism, and `src/ops/reduce.cpp` uses both.

```text
sum(x, {1}) on [1000, 1000]            sum(x) on [1 000 000]

1000 outputs, 1000 inputs each          1 output, 1 000 000 inputs
-> split the outputs across threads     -> split the input into fixed blocks
   (reduce_chunks)                         and combine partials (reduce_in_blocks)
```

#### Many outputs

`sum(x, {1})` on a `[1000, 1000]` tensor produces 1000 independent outputs. `reduce_chunks` splits the output index:

```cpp
const std::int64_t grain =
    std::max<std::int64_t>(1, kElementGrain / std::max<std::int64_t>(1, size));
parallel_for(0, reduction.output_count, grain, [&](std::int64_t first, std::int64_t last) {
```

The grain is counted in outputs, so it is `kElementGrain` divided by the number of elements each output reads (`size`). With `size = 1000` the grain is 524 outputs, so the 1000 outputs run as two chunks. With `size = 1 000 000` and 4 outputs, the grain is 1, and each output can go to its own thread.

Before the loop, `plan_reduction` permutes the reduced dimensions to the end. When the permuted tensor is not contiguous, only one dimension is reduced and the tensor is on the CPU, it keeps the original layout and reads each output's inputs through `StridedChunk`, a pointer and a stride. Otherwise it copies to contiguous memory once so every chunk is a unit-stride run.

#### One long output

`sum(x)` over 1 000 000 elements has one output. Splitting the output index gives nothing to split. `reduce_in_blocks` cuts the input into fixed blocks of `kReductionBlock = 16384` elements, reduces each block, and combines the partial results with a pairwise tree:

```cpp
// The blocks do not depend on the thread count, so floating-point results are reproducible.
template <typename Value, typename Block, typename Combine>
Value reduce_in_blocks(std::int64_t size, Value initial, Block block, Combine combine) {
    if (size <= kReductionBlock) {
        return combine(initial, block(0, size));
    }
    const std::int64_t block_count = (size + kReductionBlock - 1) / kReductionBlock;
    std::vector<Value> partial(static_cast<std::size_t>(block_count));
    parallel_for(0, block_count, kElementGrain / kReductionBlock,
                 [&](std::int64_t first, std::int64_t last) {
                     for (std::int64_t i = first; i < last; ++i) {
                         const std::int64_t begin = i * kReductionBlock;
                         partial[static_cast<std::size_t>(i)] =
                             block(begin, std::min(begin + kReductionBlock, size));
                     }
                 });
    for (std::size_t width = partial.size(); width > 1; width = (width + 1) / 2) {
        for (std::size_t i = 0; i < width / 2; ++i) {
            partial[i] = combine(partial[2 * i], partial[2 * i + 1]);
        }
        if (width % 2 == 1) {
            partial[width / 2] = partial[width - 1];
        }
    }
    return combine(initial, partial.front());
}
```

Floating-point addition is not associative. If the partial sums followed thread boundaries, `sum(x)` with 4 threads would differ in the last bits from `sum(x)` with 8 threads. Here the block boundaries depend only on the size, and the tree combines partials in a fixed order, so the result is bit-identical for any thread count. `ParallelReduce.ResultsDoNotDependOnThreadCount` tests that for `sum`, `mean`, `var`, `norm`, `prod`, `max` and `argmax`.

For 1 000 000 elements there are 62 blocks (61 full blocks and one of 576 elements). The grain is `524288 / 16384 = 32` blocks, so at most two chunks run.

The tree, traced for five partials `p0..p4`:

```text
width 5:  p0 = p0+p1   p1 = p2+p3   p2 = p4 (odd one carried)
width 3:  p0 = (p0+p1)+(p2+p3)      p1 = p4
width 2:  p0 = ((p0+p1)+(p2+p3)) + p4
width 1:  done
```

The tree runs on the calling thread after `parallel_for` returns. It touches `block_count` values, which is tiny next to the input, so it does not need threads.

#### Vectorizable accumulation

Inside a block, `fold_lanes` keeps eight independent accumulators:

```cpp
// Independent accumulators let the compiler vectorize without reordering a single accumulator.
template <typename Value, typename Term, typename Combine>
Value fold_lanes(std::int64_t first, std::int64_t last, Value identity, Term term,
                 Combine combine) {
    std::array<Value, kLanes> lanes;
    lanes.fill(identity);
    std::int64_t i = first;
    for (; i + kLanes <= last; i += kLanes) {
        for (std::int64_t lane = 0; lane < kLanes; ++lane) {
            auto& accumulator = lanes[static_cast<std::size_t>(lane)];
            accumulator = combine(accumulator, term(i + lane));
        }
    }
    Value total = identity;
    for (const Value lane : lanes) {
        total = combine(total, lane);
    }
    for (; i < last; ++i) {
        total = combine(total, term(i));
    }
    return total;
}
```

Traced for 19 elements `x0..x18`:

```text
lane:      0     1     2     3     4     5     6     7
pass 1:    x0    x1    x2    x3    x4    x5    x6    x7
pass 2:   +x8   +x9   +x10  +x11  +x12  +x13  +x14  +x15
combine:  total = ((((((0 + L0) + L1) + L2) + L3) + L4) + L5) + L6) + L7
tail:     total = ((total + x16) + x17) + x18
```

A loop `total += x[i]` has a dependency chain through `total`: each addition waits for the previous one. The compiler cannot vectorize it without `-ffast-math`, because reordering changes the rounding. Eight explicit lanes make the order part of the program, so the compiler can map lanes onto SIMD registers and stay exact with respect to the source. Floating-point sums accumulate in `double` (`Accumulator<In>`), and integer sums in `std::int64_t` with wrapping addition.

The pairwise tree also improves accuracy. A running total adds every term to one ever-growing value, so rounding error builds up over all n additions. Here each partial covers at most 16384 terms split over eight lanes, and the partials meet in a tree of depth log2 of the block count. `ParallelReduce.LongSumIsMoreAccurateThanRunningTotal` sums 2^24 copies of 0.1 in `double` and checks that the result is closer to the exact value than a running total. NumPy makes the same choice: its floating-point `sum` uses pairwise summation.

`max`, `min`, `argmax` and `argmin` do not use lanes or blocks. They are exact in any order, so they only use the output split of `reduce_chunks` and a plain loop per output.

### The matmul ladder

`src/linalg/matmul_versions.cpp` keeps every step of the optimization as a separate function, all with one signature. From `include/kael/matmul_versions.h`:

```cpp
struct Sizes {
    std::int64_t rows;
    std::int64_t inner;
    std::int64_t columns;
};

template <typename T> void v1_ijk(const T* a, const T* b, T* out, const Sizes& sizes);
template <typename T> void v2_ikj(const T* a, const T* b, T* out, const Sizes& sizes);
template <typename T> void v3_tiled(const T* a, const T* b, T* out, const Sizes& sizes);
template <typename T> void v4_packed(const T* a, const T* b, T* out, const Sizes& sizes);
template <typename T> void v5_simd(const T* a, const T* b, T* out, const Sizes& sizes);
template <typename T> void v6_threaded(const T* a, const T* b, T* out, const Sizes& sizes);
```

`a` is `rows x inner`, `b` is `inner x columns`, both row-major and contiguous. Every version computes `out += a * b`: it accumulates into `out`. The caller zero-fills the output first (`multiply_batches` uses `zeros`). Accumulation lets the same function serve as a building block for blocked loops and matches BLAS with `beta = 1`.

All versions are instantiated for `float` and `double` and tested against `v1_ijk` as the reference. Keeping the whole ladder in the library, instead of only the final kernel, makes each optimization measurable on its own with `BM_MatmulVersion`.

```text
version   change                                  what it fixes
v1_ijk    textbook loop                           baseline
v2_ikj    swap the two inner loops                strided reads of b
v3_tiled  cache blocks of 64 x 256 x 2048         b streamed once per row
v4_packed pack panels, 4 x 4 register kernel      out loaded once per k
v5_simd   8 x 12 NEON kernel (or 6 x 8 generic)   scalar arithmetic
v6        threads over row blocks                 one core in use
```

Measured with `kael_bench --benchmark_filter='BM_MatmulVersion.*/1024/'` on an Apple M5 Pro, `f32`, 1024 x 1024, with 15 threads for v6:

| Version | GFLOP/s | Speedup over v1 |
|---------|---------|-----------------|
| `v1_ijk` | 1.8 | 1x |
| `v2_ikj` | 35 | 20x |
| `v3_tiled` | 36 | 20x |
| `v4_packed` | 37 | 21x |
| `v5_simd` | 124 | 69x |
| `v6_threaded` | 557 | 311x |

PyTorch 2.15 on the same machine calls Apple Accelerate and reaches 1820 GFLOP/s, so v6 runs at 31% of the vendor library. v3 and v4 gain little on their own at this size on this CPU. Their value is the packed layout that the v5 micro-kernel reads.

#### v1: ijk

```cpp
for (std::int64_t i = 0; i < rows; ++i) {
    for (std::int64_t j = 0; j < columns; ++j) {
        T total{0};
        for (std::int64_t k = 0; k < inner; ++k) {
            total += a[i * inner + k] * b[k * columns + j];
        }
        out[i * columns + j] += total;
    }
}
```

The inner loop walks `b` down a column with stride `columns`. Each access touches a new cache line and uses one `float` of its 16.

```text
b, row-major, columns = 1024, float

address of b[k][j] = (k * 1024 + j) * 4 bytes
k = 0:  byte      0    line 0
k = 1:  byte   4096    line 64
k = 2:  byte   8192    line 128
...     each step skips 4096 bytes and loads a new 64-byte line for 4 useful bytes
```

For n = 1024 the stride is 4096 bytes, which also maps every access in the column onto few cache sets, so lines evict each other early. This version is the baseline.

#### v2: ikj

Swapping the two inner loops makes every inner access unit-stride:

```cpp
for (std::int64_t i = 0; i < rows; ++i) {
    T* out_row = out + i * columns;
    for (std::int64_t k = 0; k < inner; ++k) {
        const T a_ik = a[i * inner + k];
        const T* b_row = b + k * columns;
        for (std::int64_t j = 0; j < columns; ++j) {
            out_row[j] += a_ik * b_row[j];
        }
    }
}
```

The inner loop is now `out_row[j] += scalar * b_row[j]`, an **AXPY** (a times x plus y, the BLAS name for this operation) that the compiler vectorizes. The weakness is reuse: for each `i`, the whole of `b` streams through the cache. Once `b` is larger than the cache, every row of the output reloads `b` from memory.

`multiply` uses v2 for small problems, where packing overhead would dominate.

#### v3: tiled

**Cache blocking** (tiling) restricts the loops to tiles that fit in cache, so a tile of `b` is reused for many rows before it is evicted. `kTileBlocks{64, 256, 2048}` gives 64 rows, 256 inner and 2048 columns per tile:

```cpp
for (std::int64_t jc = 0; jc < columns; jc += kTileBlocks.columns) {
    const std::int64_t j_end = std::min(columns, jc + kTileBlocks.columns);
    for (std::int64_t pc = 0; pc < inner; pc += kTileBlocks.inner) {
        const std::int64_t k_end = std::min(inner, pc + kTileBlocks.inner);
        for (std::int64_t ic = 0; ic < rows; ic += kTileBlocks.rows) {
            const std::int64_t i_end = std::min(rows, ic + kTileBlocks.rows);
            for (std::int64_t i = ic; i < i_end; ++i) {
                T* out_row = out + i * columns;
                for (std::int64_t k = pc; k < k_end; ++k) {
                    const T a_ik = a[i * inner + k];
                    const T* b_row = b + k * columns;
                    for (std::int64_t j = jc; j < j_end; ++j) {
                        out_row[j] += a_ik * b_row[j];
                    }
                }
            }
        }
    }
}
```

For `float`, a 256 x 2048 tile of `b` is 2 MiB, read once per 64-row band. The inner loop is still the v2 AXPY, so `out` is read and written once per `k`. The output row segment of 2048 floats is 8 KiB, small enough to stay in L1 across the 256 `k` steps of a tile.

The remaining cost is that each `out` element is loaded and stored `inner` times. The arithmetic units wait on loads and stores, not on multiplies.

#### v4: packed with a register micro-kernel

The fix is the structure used by GotoBLAS and BLIS. Compute a small `kRows x kColumns` block of the output entirely in registers, across the whole `k` range of a block, and store it once. To feed that **micro-kernel** with unit-stride loads, copy (**pack**) the operands into the order the kernel reads them.

The loop nest of `multiply_packed`:

```text
for jc in column blocks of B          nc <= 4096 (rounded up to kColumns)
  for pc in inner blocks              kc <= 512
    pack B[pc:pc+kc, jc:jc+nc] into kColumns-wide panels      (shared, read-only)
    for ic in row blocks of A         mc <= 384 (rounded up to kRows)
      pack A[ic:ic+mc, pc:pc+kc] into kRows-tall panels       (private per thread)
      for each B panel (kColumns columns)
        for each A panel (kRows rows)
          Kernel::run over kc        -> kRows x kColumns result in registers
          add the valid part of the result into out
```

The block sizes are `kPackedBlocks{384, 512, 4096}`, rounded up to multiples of the micro-kernel shape. With the 8 x 12 NEON kernel the column block becomes 4104, the next multiple of 12.

The packed layout of a B panel puts the `kColumns` values of row `k` next to each other, then row `k + 1`, and so on. The packed A panel stores the `kRows` values of column `k`, then column `k + 1`. From `pack_rows`:

```cpp
for (std::int64_t top = 0; top < rows; top += Kernel::kRows) {
    const std::int64_t height = std::min(Kernel::kRows, rows - top);
    for (std::int64_t k = 0; k < inner; ++k) {
        for (std::int64_t i = 0; i < height; ++i) {
            packed[i] = a[(top + i) * stride + k];
        }
        std::fill(packed + height, packed + Kernel::kRows, T{0});
        packed += Kernel::kRows;
    }
}
```

Packing a 10 x 3 block of A with `kRows = 4`:

```text
A rows 0..9, columns k = 0..2

panel 0 (rows 0..3):  a00 a10 a20 a30 | a01 a11 a21 a31 | a02 a12 a22 a32
panel 1 (rows 4..7):  a40 a50 a60 a70 | a41 a51 a61 a71 | a42 a52 a62 a72
panel 2 (rows 8..9):  a80 a90  0   0  | a81 a91  0   0  | a82 a92  0   0
```

`pack_column_panels` does the same for B. Packing a 3 x 6 block of B with `kColumns = 4`:

```text
B rows k = 0..2, columns 0..5

panel 0 (cols 0..3):  b00 b01 b02 b03 | b10 b11 b12 b13 | b20 b21 b22 b23
panel 1 (cols 4..5):  b04 b05  0   0  | b14 b15  0   0  | b24 b25  0   0
```

Edges are zero-padded, so the micro-kernel always runs a full `kRows x kColumns` shape with no bounds checks. A zero in the padding adds `0 * x` to an accumulator that is thrown away. `multiply_row_block` copies only the valid `height x width` part of the result back into `out`.

Packing costs one extra read and write of each operand block. In exchange:

- The micro-kernel reads both panels with unit stride, which is the best case for the hardware prefetcher.
- A panel of B (`kc * kColumns` elements, 24 KiB for `float` with `kc = 512` and `kColumns = 12`) is reused for every A panel in the row block.
- Each output element is loaded and stored once per `kc` block instead of once per `k`.

The micro-kernel for v4 is `GenericKernel<T, 4, 4>`: a 4 x 4 accumulator array in plain C++.

```cpp
static void run(const T* a, const T* b, std::int64_t inner, T* result) {
    std::array<T, static_cast<std::size_t>(Rows * Columns)> accumulators{};
    for (std::int64_t k = 0; k < inner; ++k) {
        for (std::int64_t i = 0; i < Rows; ++i) {
            const T a_value = a[i];
            T* row = accumulators.data() + i * Columns;
            for (std::int64_t j = 0; j < Columns; ++j) {
                row[j] += a_value * b[j];
            }
        }
        a += Rows;
        b += Columns;
    }
    std::copy(accumulators.begin(), accumulators.end(), result);
}
```

Each `k` step is an **outer product**: a column of 4 A values times a row of 4 B values, added into the 4 x 4 block.

```text
         b[0] b[1] b[2] b[3]
a[0]  [  +    +    +    +  ]
a[1]  [  +    +    +    +  ]     16 multiply-adds from 8 loads
a[2]  [  +    +    +    +  ]
a[3]  [  +    +    +    +  ]
```

Per `k` step it loads 4 + 4 values and does 16 multiply-adds: an intensity of 2 per load, against 1/2 for the AXPY of v2 and v3 (one multiply-add per load of `b` and per load of `out`, plus a store).

Pack buffers come from `allocate_pack`, which aligns them to `kPackAlignment = 64` bytes, the cache line size of common CPUs. An aligned panel never splits a vector load across two lines.

#### v5: SIMD micro-kernel

v5 runs the same packed algorithm with a larger kernel chosen by `SimdKernelFor<T>`. On ARM with NEON (`__ARM_NEON`), `float` uses `NeonFloatKernel`, an 8 x 12 kernel:

```cpp
for (std::int64_t k = 0; k < inner; ++k) {
    const float32x4_t a_low = vld1q_f32(a);
    const float32x4_t a_high = vld1q_f32(a + 4);
    const float32x4_t b0 = vld1q_f32(b);
    const float32x4_t b1 = vld1q_f32(b + 4);
    const float32x4_t b2 = vld1q_f32(b + 8);
    KAEL_FMA_COLUMN(0, b0, 0);
    // ... columns 1 to 11
```

`KAEL_FMA_COLUMN(column, source, lane)` expands to two `vfmaq_laneq_f32` calls, one for the low four rows and one for the high four rows of that output column.

The register budget explains the shape. AArch64 has 32 vector registers of 128 bits.

```text
register use per k step (8 x 12 float kernel)

accumulators  c[12][2]   12 columns x 2 halves of 4 rows    24 registers
A column      a_low, a_high                                  2 registers
B row         b0, b1, b2                                     3 registers
                                                    total   29 of 32, no spills
```

Each `k` step does 5 loads and 24 vector FMAs (96 scalar multiply-adds), using `vfmaq_laneq_f32` to multiply a vector of A by one lane of B without a separate broadcast. That is 96 / 20 scalar values loaded, almost 5 multiply-adds per value, against 2 for the 4 x 4 scalar kernel.

`double` uses `NeonDoubleKernel`, 6 x 8 with 2-wide vectors: 24 accumulator registers (`c[8][3]`) plus three for A, with B broadcast by `vdupq_n_f64`.

Without NEON, `SimdKernelFor<T>` is `GenericKernel<T, 6, 8>`. The fixed-size accumulator loop is written so that the compiler can vectorize it. Build with `-DKAEL_NATIVE=ON` to compile `kael` with `-march=native` (or `-mcpu=native` when the compiler does not accept the first) and let the compiler use AVX2 or AVX-512 on x86.

#### v6: threaded

v6 is v5 with `Threaded = true`. Two loops run through `run_blocks`, which calls `parallel_for` when threaded and calls the body directly otherwise:

```cpp
template <bool Threaded, typename Body>
void run_blocks(std::int64_t count, std::int64_t grain, Body body) {
    if constexpr (Threaded) {
        parallel_for(0, count, grain, body);
    } else {
        body(0, count);
    }
}
```

1. Packing B is split by panel, with grain `kElementGrain / panel_elements`.
2. The row blocks of A are split across threads.

The row block size adapts to the thread count so that every thread gets a block:

```cpp
template <typename Kernel, bool Threaded>
std::int64_t row_block_for(std::int64_t rows, std::int64_t largest) {
    if constexpr (Threaded) {
        const std::int64_t threads = get_num_threads();
        return std::min(largest, rounded_up<Kernel::kRows>((rows + threads - 1) / threads));
    } else {
        return largest;
    }
}
```

For 1024 rows, 8 threads and `kRows = 8`, the row block is `min(384, rounded_up<8>(128)) = 128`, giving 8 blocks. With the full 384 rows, only 3 threads would have work.

The same formula for 1000 rows and 7 threads, one of the shapes in `MatmulThreadPartition`:

```text
ceil(1000 / 7)        = 143
rounded_up<8>(143)    = 144
row block             = min(384, 144) = 144
blocks                = ceil(1000 / 144) = 7
rows per block        = 144 144 144 144 144 144 136
```

Each body invocation allocates its own packed A buffer, so threads never share writable scratch memory. The packed B buffer is shared and read-only during the multiply phase. The two `run_blocks` calls are separate `parallel_for` calls, and the first returns only when every panel is packed, so no thread reads a half-packed B.

Small blocks do not go parallel:

```cpp
const bool worthwhile = rows * kc * nc >= kThreadedProblem;
// ...
run_blocks<Threaded>(block_count, worthwhile ? 1 : block_count, ...);
```

`kThreadedProblem` is 128^3. Below it, the grain equals the block count, so `parallel_for` runs everything inline.

Threading does not change results. Each output element is accumulated over `k` in the same order, by the same micro-kernel, whatever row block contains it. The thread count changes which rows a block holds, never the `kc` blocks or the order inside one. `MatmulThreadPartition.MatchesTheSingleThreadResultExactly` compares v6 with 1 thread and with 2 to 16 threads using `kael::equal`, on shapes chosen to produce ragged edges (for example 387 x 513 x 4100 with 4 threads).

#### BLAS and Accelerate

**BLAS** (Basic Linear Algebra Subprograms) is the standard interface for vendor-tuned linear algebra. `-DKAEL_USE_BLAS=ON` links a system BLAS. On Apple platforms CMake links the Accelerate framework and defines `KAEL_USE_ACCELERATE`. Elsewhere it uses `find_package(BLAS)` and defines `KAEL_USE_CBLAS`. Both paths call the CBLAS interface:

```cpp
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, columns, inner, 1.0f, a, inner,
            b, columns, 1.0f, out, columns);
```

`beta = 1.0f` keeps the `out +=` contract. `double` calls `cblas_dgemm` with the same arguments. The sizes are cast to `int`, the CBLAS index type. Without BLAS, calling `matmul_versions::blas` throws "kael was built without BLAS; configure with KAEL_USE_BLAS=ON".

This is what NumPy does for `np.matmul` on floating-point arrays, and what PyTorch does on the CPU through MKL or OpenBLAS. kael keeps its own ladder as the default so that the chapter's code is the code that runs.

#### Dispatch

From `src/linalg/matmul_versions.cpp`:

```cpp
template <typename T> void multiply(const T* a, const T* b, T* out, const Sizes& sizes) {
    if (blas_available()) {
        blas(a, b, out, sizes);
    } else if (sizes.rows * sizes.inner * sizes.columns <= kSmallProblem) {
        v2_ikj(a, b, out, sizes);
    } else {
        v6_threaded(a, b, out, sizes);
    }
}
```

`kSmallProblem` is 32^3. `matmul` in `src/ops/linalg.cpp` sends only floating-point types to `multiply`. Integer matrices go to `multiply_exact`, an ikj loop in the element type with wrapping addition and multiplication, so that large `i64` values are never rounded through a floating-point kernel (`MatmulDispatch.LargeIntegerMatrixStillUsesExactArithmetic`). `bool` matrices are rejected.

```text
matmul(a, b)
  |
  +-- device != CPU ------------------------> device::matmul (chapter 09)
  |
  +-- integer dtype -------------------------> multiply_exact (ikj, wrapping)
  |
  +-- float / double -> multiply
                          |
                          +-- BLAS built in --> cblas_sgemm / cblas_dgemm
                          +-- m*k*n <= 32^3 --> v2_ikj
                          +-- otherwise ------> v6_threaded
```

Batched matmul broadcasts the batch dimensions, flattens them, and splits the batch across threads with grain `kWorkPerParallelTask / work`, where `kWorkPerParallelTask` is 2^21 multiply-adds and `work = rows * inner * columns`. Batches of tiny matrices are grouped into one chunk. Batches of large matrices run one at a time with the threaded kernel, as described under nested regions.

Two traces of that grain:

```text
64 matrices of 32 x 32 x 32:  work = 32 768     grain = 2 097 152 / 32 768 = 64
                              64 <= 64 -> whole batch inline on the caller

8 matrices of 512^3:          work = 134 217 728  grain = max(1, 0) = 1
                              8 threads -> one matrix per thread, each single-threaded
```

### The caching block allocator

A training step allocates and frees the same set of tensor sizes on every iteration. A system allocator call for a multi-megabyte block can map fresh pages from the operating system, and the first touch of each page costs a **page fault** (a trap into the kernel to back the page with physical memory). The caching allocator keeps freed blocks and hands them back for the next request of the same size class.

```text
step 1: allocate 4 MiB  -> miss, system allocator, page faults on first write
        free 4 MiB      -> block goes to the free list for 4 MiB
step 2: allocate 4 MiB  -> hit, same block, pages already mapped
        free 4 MiB      -> back to the free list
```

PyTorch does the same on CUDA, where device allocation is expensive, and exposes `torch.cuda.empty_cache` and `torch.cuda.memory_stats`. kael uses one implementation for both the host and the device.

#### Size classes

From `src/allocator.cpp`:

```cpp
std::int64_t block_capacity(std::int64_t nbytes) {
    if (nbytes <= kLargestSteppedSize) {
        return (nbytes + kBucketStep - 1) / kBucketStep * kBucketStep;
    }
    KAEL_CHECK(nbytes <= kLargestBucket, "cannot reserve " << nbytes << " bytes");
    return next_power_of_two(nbytes);
}
```

Requests up to 64 KiB (`kLargestSteppedSize`) round up to a multiple of 512 bytes (`kBucketStep`). Larger requests round up to a power of two. Values printed by a snippet linked against `build/libkael.a`:

| Requested bytes | Capacity |
|-----------------|----------|
| 1 | 512 |
| 24 | 512 |
| 513 | 1024 |
| 65 536 | 65 536 |
| 65 537 | 131 072 |
| 1 000 000 | 1 048 576 |

Rounding trades memory for hit rate. A `[1000, 1000]` `float` tensor (4 000 000 bytes) takes a 4 MiB block (4 194 304 bytes) and wastes 194 304 bytes, about 5 percent. Exact-size buckets would waste nothing, but tensors whose sizes differ by a few bytes would never share blocks. Requests above 2^62 bytes (`kLargestBucket`) are rejected before rounding could overflow a signed 64-bit integer.

A zero-byte `Storage` never reaches the allocator: its pointer stays null (`Allocator.ZeroBytesStayNull`).

#### The cache

`BlockCache` keeps one free list per capacity in an `unordered_map<std::int64_t, std::vector<std::byte*>>`, all guarded by one mutex.

```text
free_blocks_
  512      -> [ p1, p2 ]
  1024     -> [ ]
  4194304  -> [ p3 ]

allocate(4194304): pop p3 (hit), in_use += 4 MiB, cached -= 4 MiB
allocate(2048):    list empty -> system allocate (miss), reserved += 2048
release(p3, 4 MiB): push p3, in_use -= 4 MiB, cached += 4 MiB
```

`allocate` pops a cached block (a hit) or calls the system allocator (a miss). `release` pushes the block back, unless caching is disabled or the cache would exceed `kMaxCachedBytes` (1 GiB), in which case the block goes back to the system. `try_cache` catches the `std::bad_alloc` that growing a free list can throw and frees the block instead: caching is optional, and `release` runs from a `noexcept` deleter.

```cpp
// Caching is optional: when the free list cannot grow, the block goes back to the system.
bool try_cache(std::byte* block, std::int64_t capacity) noexcept {
    try {
        free_blocks_[capacity].push_back(block);
        return true;
    } catch (const std::bad_alloc&) {
        return false;
    }
}
```

The statistics keep one invariant: `reserved_bytes == in_use_bytes + cached_bytes`. `Allocator.ConcurrentUseKeepsTheBooksBalanced` checks that the books balance after many threads allocate and free at once.

`Storage` holds the capacity in its deleter (`BlockDeleter`), so freeing a tensor releases its block to the right bucket without looking it up. Host blocks come from `::operator new` with `std::align_val_t{Storage::kAlignment}`, 64 bytes, so cached blocks keep the alignment guarantee of [chapter 02](@/book/02-memory-model.md) (`CachingAllocator.ReusedBlocksStayAligned`).

The public API in `include/kael/allocator.h`:

| Function | Effect |
|----------|--------|
| `allocator_stats(Device)` | `reserved_bytes`, `in_use_bytes`, `cached_bytes`, `hits`, `misses` |
| `empty_cache()` | Frees every cached block of both the host and the device cache |
| `set_caching_allocator_enabled(bool)` | Turns caching off (dropping cached blocks) or on, for both caches |
| `caching_allocator_enabled()` | Current state |

Python exposes `kael.allocator_stats` and `kael.empty_cache`.

Caching is on by default. Set `KAEL_NO_CACHING_ALLOCATOR` in the environment to start with it off; the code checks only that the variable exists, so any value, including `0`, turns caching off. Builds with `KAEL_ENABLE_SANITIZERS` or `KAEL_ENABLE_TSAN` compile it out with the `KAEL_NO_CACHING_ALLOCATOR` definition, because reused blocks hide use-after-free bugs from AddressSanitizer.

Two lifetime details:

- The caches are allocated with `new` and never destroyed. A tensor with static storage duration can free its block after `main` returns, and the cache must still exist then.
- `install_fork_handlers` registers `pthread_atfork` handlers that lock both cache mutexes before `fork` and release them in the parent and the child. Without them, a thread holding the cache mutex during the fork would leave it locked forever in the child.

The thread pool and the allocator solve the same fork problem two ways. The pool cannot be made consistent in the child, so the child forgets it. The allocator's state is plain data, so holding the lock across the fork guarantees the child sees a consistent copy.

The same `BlockCache` class backs the device cache in [chapter 09](@/book/09-cuda-backend.md), with `device::allocate` and `device::release` as the system functions.

### Fused row kernels

The composed softmax in [chapter 05](@/book/05-operations.md) is `exp(x - max(x)) / sum(exp(...))`. On a `[N, C]` tensor that is five passes over memory and three temporaries of size N x C.

```text
composed softmax on [N, C]                    fused softmax_row

1. m = max(x, dim)          read  N*C         for each row (stays in cache):
2. s = x - m                read  N*C,          pass 1: max
                            write N*C           pass 2: out = exp(x - max), total += out
3. e = exp(s)               read  N*C,          pass 3: out /= total
                            write N*C
4. t = sum(e, dim)          read  N*C         memory: read N*C, write N*C
5. y = e / t                read  N*C,
                            write N*C
```

When the reduced dimension is the last one of a contiguous CPU tensor, `src/ops/nn.cpp` uses one kernel per row instead.

#### map_rows

```cpp
template <typename T, typename RowKernel>
Tensor map_rows(const Tensor& x, std::int64_t row_size, std::int64_t cost, RowKernel kernel) {
    Tensor out = empty(x.shape(), x.dtype());
    const T* in = x.data_ptr<T>();
    T* results = out.data_ptr<T>();
    const std::int64_t grain = std::max<std::int64_t>(1, kElementGrain / (cost * row_size));
    parallel_for(0, x.numel() / row_size, grain, [&](std::int64_t first, std::int64_t last) {
        for (std::int64_t row = first; row < last; ++row) {
            kernel(in + row * row_size, results + row * row_size, row_size);
        }
    });
    return out;
}
```

`cost` scales the grain to the work per element. `kExponentialCost = 16` reflects that `std::exp` costs far more than an add. `kStandardizeCost = 4` covers the three passes of layer normalization. For softmax over `[1000, 1000]`, the grain is `524288 / (16 * 1000) = 32` rows, so the 1000 rows split into up to 32 chunks.

Softmax and log softmax select the fast path with `is_last_dim_of_contiguous`: CPU device, at least one dimension, contiguous, and `dim` naming the last dimension. Any other case falls back to the composed expression, which is correct for every layout. `standardize` and `cross_entropy_losses` make their input contiguous first and always use the row kernel on the CPU.

#### Softmax and log softmax

```cpp
template <typename T> void softmax_row(const T* row, T* out, std::int64_t size) {
    const T largest = row_maximum(row, size);
    double total = 0.0;
    for (std::int64_t i = 0; i < size; ++i) {
        out[i] = std::exp(row[i] - largest);
        total += static_cast<double>(out[i]);
    }
    const auto denominator = static_cast<T>(total);
    for (std::int64_t i = 0; i < size; ++i) {
        out[i] /= denominator;
    }
}
```

Three passes over one row, which stays in cache between passes. The max shift keeps `exp` from overflowing. The sum accumulates in `double`. `row_maximum` uses `nan_aware_max`, so a NaN in the row propagates the same way as in the composed path (`FusedKernels.HandlesNonFiniteValuesLikeTheComposedPath`).

Trace `softmax_row` on the row `[1, 2, 3]`:

```text
pass 1: largest = 3
pass 2: out = [exp(-2), exp(-1), exp(0)] = [0.135335, 0.367879, 1.000000]
        total = 1.503215
pass 3: out = [0.090031, 0.244728, 0.665241]          (sums to 1)
```

Without the shift, a row `[1000, 1001]` in `float` overflows: `exp(1000)` is far above the largest finite `float`, about 3.4e38. Shifted, the row becomes `[-1, 0]`, and the result is the same mathematically.

`log_softmax_row` computes `x - max - log(sum(exp(x - max)))` without ever forming `log(softmax)`:

```cpp
template <typename T> void log_softmax_row(const T* row, T* out, std::int64_t size) {
    const T largest = row_maximum(row, size);
    double total = 0.0;
    for (std::int64_t i = 0; i < size; ++i) {
        out[i] = row[i] - largest;
        total += static_cast<double>(std::exp(out[i]));
    }
    const T log_total = std::log(static_cast<T>(total));
    for (std::int64_t i = 0; i < size; ++i) {
        out[i] -= log_total;
    }
}
```

On `[1, 2, 3]`: `log(1.503215) = 0.407606`, so the output is `[-2.407606, -1.407606, -0.407606]`. Taking `log` of the softmax would give `-inf` wherever the softmax underflows to zero, for example for an entry 200 below the row maximum in `float`. The direct formula gives the finite value `-200 - log_total` (`LogSoftmax.IsExactWhereLogOfSoftmaxIsNot`).

#### Layer normalization

`standardize(t, normalized_dims, eps)` computes `(x - mean) / sqrt(var + eps)` over the trailing `normalized_dims` dimensions. `nn::LayerNorm::forward` calls it and then applies the affine weight and bias. `standardize_row` makes two `double` passes for the mean and the centered sum of squares, then one pass to write the output:

```cpp
template <typename T> void standardize_row(const T* row, T* out, std::int64_t size, double eps) {
    double total = 0.0;
    for (std::int64_t i = 0; i < size; ++i) {
        total += static_cast<double>(row[i]);
    }
    const double center = total / static_cast<double>(size);
    double squares = 0.0;
    for (std::int64_t i = 0; i < size; ++i) {
        const double deviation = static_cast<double>(row[i]) - center;
        squares += deviation * deviation;
    }
    const double scale = std::sqrt(squares / static_cast<double>(size) + eps);
    for (std::int64_t i = 0; i < size; ++i) {
        out[i] = static_cast<T>((static_cast<double>(row[i]) - center) / scale);
    }
}
```

Trace on `[1, 2, 3, 4]` with `eps = 1e-5`:

```text
pass 1: center  = 10 / 4                      = 2.5
pass 2: squares = 2.25 + 0.25 + 0.25 + 2.25   = 5.0
        scale   = sqrt(5.0 / 4 + 1e-5)        = 1.1180385
pass 3: out     = [-1.341635, -0.447212, 0.447212, 1.341635]
```

The two-pass variance avoids the cancellation of the one-pass formula `E[x^2] - E[x]^2`. With values near 10 000 in `float`, `x^2` is near 10^8, where neighbouring `float` values are 8 apart. A variance below 8 then disappears in the subtraction. Centering first keeps the squared terms small. The backward pass is recorded separately as `StandardizeBackward` ([chapter 06](@/book/06-autograd.md)).

On a CUDA tensor, `standardize` uses the composed `mean` and `var` expression.

#### Cross entropy

`cross_entropy_losses(logits, targets)` computes one loss per row on the CPU without materializing `log_softmax`:

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

On logits `[1, 2, 3]` with target 0: `log(1.503215) - (1 - 3) = 2.407606`, which is minus the first entry of the log softmax above. The loss row writes nothing but one scalar.

The gradient is fused too. `record_cross_entropy` stores a `CrossEntropyBackward` node that calls `detail::cross_entropy_gradient`. For each row it runs `softmax_row` into the output, scales it by `g[row]`, and subtracts `g[row]` at the target column:

```text
row [1, 2, 3], target 0, g = 1

softmax         [ 0.090031,  0.244728,  0.665241]
times g         [ 0.090031,  0.244728,  0.665241]
minus g at 0    [-0.909969,  0.244728,  0.665241]   = softmax - one_hot(target)
```

The composed path (`nll_loss(log_softmax(x), t)`) would build a `log_softmax` node, a `gather` node and a scatter in the backward pass. The benchmark pair `BM_CrossEntropyFused` and `BM_CrossEntropyComposed` measures the difference for `[64, 10]`, `[4096, 65]` and `[1024, 50257]` logits, the last being a GPT-2 vocabulary.

On a CUDA tensor, `cross_entropy_losses` uses the composed path.

## Measuring performance

A performance change without a measurement is a guess. kael keeps its benchmarks in the repository so that every claim in this chapter can be checked on the reader's machine.

### Building

Benchmarks use Google Benchmark v1.9.0, fetched by CMake. Build them in Release mode in a separate build directory:

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DKAEL_BUILD_BENCHMARKS=ON -DKAEL_BUILD_EXAMPLES=ON
cmake --build build-release --parallel
```

Configuring with both `KAEL_BUILD_BENCHMARKS` and `KAEL_ENABLE_SANITIZERS` fails on purpose: sanitizers distort timings. Add `-DKAEL_USE_BLAS=ON` to also benchmark the BLAS path. It defines `KAEL_BENCH_BLAS` for the benchmark target, which registers `BM_MatmulVersion<blas<float>>`.

### Suites

All suites link into one executable, `build-release/benchmarks/kael_bench`.

| File | Benchmarks | Measures |
|------|------------|----------|
| `bench_matmul.cpp` | `BM_Matmul`, `BM_MatmulVersion<v1_ijk<float>>` to `<v6_threaded<float>>`, `BM_MatmulThreads` | `matmul` at 128, 256 and 512; each ladder step at 256, 512, 1024 and 2048; v6 at 1024 with 1, 2, 4, 5, 8 and 15 threads; reports a `flops` rate |
| `bench_elementwise.cpp` | `BM_NaiveAdd*`, `BM_KernelAdd*`, `BM_ContiguousFromTransposed`, `BM_AddThreads` | The naive index-per-element add in `naive_add.h` against `binary_kernel`, for contiguous, broadcast and transposed inputs; add scaling from 2^13 to 2^23 elements with 1, 4 and 15 threads |
| `bench_compare.cpp` | `BM_Compare*`, `BM_CrossEntropyFused`, `BM_CrossEntropyComposed`, `BM_LinearWithBias`, `BM_LinearWithoutBias` | The operations compared with NumPy and PyTorch; fused against composed cross entropy; a `Linear` training step |
| `bench_cuda.cpp` | `BM_Device*` | Device kernels; they report an error and skip without a CUDA device |

The `flops` counter is `2 * n^3` per iteration as a rate, the same count as the intensity table at the start of the chapter. Compare it with the peak of the CPU to see how far a version is from the roof.

Run a subset with a filter:

```bash
build-release/benchmarks/kael_bench --benchmark_filter='BM_MatmulVersion'
KAEL_NUM_THREADS=4 build-release/benchmarks/kael_bench --benchmark_filter='BM_Compare'
```

`BM_MatmulThreads` and `BM_AddThreads` call `set_num_threads` themselves. The other benchmarks use the default thread count or `KAEL_NUM_THREADS`.

> [!TIP]
> Read `BM_AddThreads` and `BM_MatmulThreads` side by side. The add is memory bound and stops scaling after a few threads. The matmul is compute bound and keeps scaling until the performance cores run out. That contrast is the roofline model measured on your machine.

### Comparing with NumPy and PyTorch

`benchmarks/compare.py` runs `kael_bench --benchmark_filter=BM_Compare --benchmark_format=json`, times the same operations with NumPy and PyTorch through `timeit` (best of 7 repeats of 5 calls), and times one MNIST epoch with `examples/mnist_mlp` and with an equivalent PyTorch model. It prints a Markdown table. It needs `numpy` and `torch` installed and the MNIST files in `data/mnist`.

```bash
scripts/download/mnist.sh
python benchmarks/compare.py --threads 8
```

| Option | Default |
|--------|---------|
| `--bench` | `build-release/benchmarks/kael_bench` |
| `--mnist` | `build-release/examples/mnist_mlp` |
| `--data` | `data/mnist` |
| `--threads` | `torch.get_num_threads()` |

`--threads` sets the PyTorch thread count and exports `KAEL_NUM_THREADS` with the same value before it starts `kael_bench` and `mnist_mlp`, so both libraries use the same number of threads. NumPy follows its BLAS environment variables. `python benchmarks/compare.py --help` describes every option. The repository does not store benchmark results: numbers depend on the machine, so generate them locally.

## Edge cases and pitfalls

- **Empty ranges and empty matrices.** `parallel_for` returns at once when `end <= begin`. `multiply_packed` and `blas` return when any of `rows`, `inner`, `columns` is zero, leaving `out` untouched (`MatmulVersionsEmpty.ZeroSizesLeaveOutputUntouched`). Empty reductions keep their meaning: `sum` of an empty tensor is 0, `prod` is 1 and `mean` is NaN (`ParallelReduce.EmptyReductionsKeepTheirMeaning`). `max`, `min`, `argmax` and `argmin` of an empty dimension throw, because `plan_nonempty_reduction` rejects them.
- **Non-positive grain.** `parallel_for` rejects `grain <= 0` with `ValueError`. Grain formulas such as `kElementGrain / (cost * row_size)` wrap the result in `std::max<std::int64_t>(1, ...)` because the division reaches zero for wide rows. A `[1024, 50257]` softmax gives `524288 / (16 * 50257) = 0` before the clamp.
- **Exceptions in workers.** A body that throws on a worker would call `std::terminate` without the capture in `Job::work`. Integer division by zero inside a parallel elementwise kernel still reaches the caller as an exception (`ParallelIterator.IntegerDivisionByZeroStillThrows`), and the pool keeps working afterwards (`ParallelThreads.PoolKeepsWorkingAfterAnException`). Chunks already running when the error happens finish; their partial output is discarded with the exception.
- **Thread-count-dependent rounding.** Any reduction whose partial sums follow chunk boundaries gives different bits for different thread counts. Reductions use fixed `kReductionBlock` blocks, and matmul keeps a fixed `k` order per element. A new kernel that splits a sum by `parallel_for` chunk breaks this rule silently.
- **Lost grad mode.** A kernel that runs autograd-aware code on a worker would see the default grad mode. `Job` copies the caller's mode.
- **Oversubscription.** Nested `parallel_for` calls run inline. A second concurrent caller runs serially instead of creating more threads. Inside a nested region, `get_num_threads` still returns the configured count, so v6 sizes its row blocks for that count even though they then run inline. The result is the same; only the block size differs.
- **Fork.** The child process cannot use the parent's pool or locked mutexes. Both the parallel runtime and the allocator install `pthread_atfork` handlers. The C++ fork test is compiled out under ThreadSanitizer.
- **Edge tiles.** Packing zero-pads partial panels, and `multiply_row_block` copies only `height x width` results back. Shapes that are not multiples of the micro-kernel are tested explicitly.
- **Integer overflow in BLAS sizes.** CBLAS takes `int` dimensions. kael casts with `static_cast<int>`, so a single matrix dimension above 2^31 - 1 is out of reach of the BLAS path.
- **Allocator cap.** The cache never holds more than 1 GiB per device. Blocks released beyond that go back to the system (`CachingAllocator.HugeBlocksAreNotKeptWhenOverTheLimit`). Call `empty_cache()` to return cached memory explicitly.
- **Memory that looks leaked.** With caching on, freed tensors keep their blocks in `cached_bytes`, so process memory does not drop after a large tensor goes away. Read `allocator_stats` before concluding there is a leak.
- **Benchmarking a debug build.** A Debug build disables the optimizations every version of the ladder relies on. Compare timings only from a Release build.

## How it is tested

| Suite | File | What it proves |
|-------|------|----------------|
| `ParallelThreads` | `tests/test_parallel.cpp` | Every index visited once, non-zero `begin`, grain respected, small ranges inline, exceptions propagate, grad mode copied, nested calls inline, concurrent callers complete, thread count readable but not writable inside a body, a second caller does not wait |
| `ParallelConfig` | `tests/test_parallel.cpp` | `set_num_threads` validation, including a count above `kMaxThreads`; one thread runs the whole range in one call |
| `ParallelFork` | `tests/test_parallel.cpp` | A forked child runs `parallel_for` and allocates `Storage` without hanging, and the parent keeps working |
| `ParallelIterator` | `tests/test_parallel_ops.cpp` | Large ranges split, small ranges do not, every layout writes every element in place, results independent of thread count, errors reach the caller |
| `ParallelReduce` | `tests/test_parallel_ops.cpp` | Results independent of thread count, many outputs match hand-computed slices, long sums beat a running total in accuracy, empty reductions |
| `Shapes/MatmulVersions` | `tests/test_matmul_versions.cpp` | v2 to v6 and `multiply` match `v1_ijk` in `float` and `double` over shapes that exercise edge tiles |
| `MatmulVersionsEmpty` | `tests/test_matmul_versions.cpp` | Zero sizes leave the output untouched |
| `MatmulVersionsThreads`, `Partitions/MatmulThreadPartition` | `tests/test_matmul_versions.cpp` | Bitwise identical results for any thread count and partition |
| `MatmulDispatch` | `tests/test_matmul_versions.cpp` | Integer matmul stays exact |
| `CachingAllocator`, `Allocator` | `tests/test_allocator.cpp` | Bucket sharing, hits and misses, alignment, statistics, `empty_cache`, the 1 GiB cap, disabling mid-life, concurrent use, zero-byte storage, failed reservations, unrepresentable sizes |
| `FusedKernels`, `Softmax`, `LogSoftmax` | `tests/test_ops_nn.cpp` | Fused kernels match the composed path including non-finite values, strided and wide inputs, `standardize` gradcheck |

`python/tests/test_interop.py::test_fork_keeps_parallel_work_running` repeats the fork check from Python with `multiprocessing` and the `fork` start method.

Three testing techniques recur in this chapter and are worth copying:

- **Compare against the slow version.** Every matmul version is checked against `v1_ijk`, which is short enough to verify by reading. The fused kernels are checked against the composed expressions of chapter 05.
- **Compare across thread counts with exact equality.** `kael::equal` instead of a tolerance turns "the result does not depend on the thread count" into a check that cannot pass by accident.
- **Run under ThreadSanitizer.** `-DKAEL_ENABLE_TSAN=ON` reports data races that a normal run hides. The caching allocator is compiled out in that build, so the sanitizer sees every allocation.

```bash
cmake -S . -B build-tsan -DKAEL_ENABLE_TSAN=ON
cmake --build build-tsan --parallel
build-tsan/tests/kael_tests --gtest_filter='Parallel*:*MatmulVersions*:*MatmulThreadPartition*'
```

## Build it yourself

1. Write `parallel_for` as a serial loop that calls `body(begin, end)` and validates the grain. Checkpoint: a test that records every visited index and checks each appears once.
2. Add `ThreadPool` with a generation counter, `Job` with an atomic cursor, and the `try_to_lock` runtime. Checkpoint: `ParallelThreads.VisitsEveryIndexOnce` and `ConcurrentCallersAllComplete` pass under `-DKAEL_ENABLE_TSAN=ON`.
3. Capture exceptions and grad mode in `Job::work`, and add the `thread_local` region flag. Checkpoint: `ExceptionReachesCaller`, `GradModeIsCopiedToWorkers` and `NestedCallRunsInline`.
4. Read `KAEL_NUM_THREADS`, detect performance cores, and add the `pthread_atfork` child handler. Checkpoint: `ParallelFork.ChildCanStillRunParallelWork`.
5. Call `parallel_for` from `TensorIterator::run`. Checkpoint: `ParallelIterator.ResultsDoNotDependOnThreadCount`.
6. Rewrite reductions with `fold_lanes` and `reduce_in_blocks`. Checkpoint: `ParallelReduce.ResultsDoNotDependOnThreadCount` and `LongSumIsMoreAccurateThanRunningTotal`.
7. Write `v1_ijk` and a parameterized test that uses it as the reference, then add `v2_ikj` and `v3_tiled` to the same test. Checkpoint: `MatmulVersions` passes for odd shapes such as 1 x 1 x 1, 7 x 1 x 13 and 65 x 257 x 31.
8. Write `pack_rows`, `pack_column_panels`, `multiply_row_block` and `GenericKernel`, and wire them as `v4_packed`. Checkpoint: `MatmulVersions` with v4, and `BM_MatmulVersion<v4_packed<float>>` faster than v3 at 1024.
9. Add the NEON kernels (or rely on `GenericKernel<T, 6, 8>` with `KAEL_NATIVE`) as `v5_simd`. Checkpoint: same tests, higher `flops` counter.
10. Thread the packed loop as `v6_threaded` and add `multiply` dispatch. Checkpoint: `MatmulThreadPartition` and `BM_MatmulThreads`.
11. Add the BLAS path behind `KAEL_USE_BLAS`. Checkpoint: configure with `-DKAEL_USE_BLAS=ON` and run `MatmulVersions`.
12. Write `BlockCache`, `block_capacity` and the fork handlers, and route `Storage` through `allocate_block` and `release_block`. Checkpoint: `CachingAllocator.*`, and `ctest` with `-DKAEL_ENABLE_SANITIZERS=ON`.
13. Add `map_rows` with the softmax, log softmax and standardize row kernels, then the fused cross entropy forward and gradient. Checkpoint: `FusedKernels.*` and `BM_CrossEntropyFused` faster than `BM_CrossEntropyComposed`.
14. Build the benchmarks and run `benchmarks/compare.py`. Checkpoint: the table prints with a row for every operation.

## Check your understanding

<details>
<summary>Why does adding two 1000 x 1000 float tensors use only two chunks on an 8-thread machine?</summary>

The range has 1 000 000 elements and the grain is `kElementGrain = 524288`. `parallel_for` wants 32 chunks, but a chunk is never smaller than the grain, so `chunk_size = 524288` and the range splits into two chunks. Addition is memory bound: more threads would compete for the same memory bandwidth and add wake-up cost without moving more bytes per second.

</details>

<details>
<summary>Why is `sum(x)` bit-identical for 1 thread and for 8 threads, when floating-point addition is not associative?</summary>

`reduce_in_blocks` cuts the input into blocks of `kReductionBlock = 16384` elements whose boundaries depend only on the input size. Inside each block, `fold_lanes` adds in a fixed lane order, and the partial results meet in a pairwise tree whose shape depends only on the block count. The thread count decides which thread computes which block, never the order of any addition.

</details>

<details>
<summary>What does packing buy in v4, given that it adds an extra read and write of each operand block?</summary>

It lets the micro-kernel read both operands with unit stride and keep a `kRows x kColumns` output block in registers for a whole `kc` range. Each output element is then loaded and stored once per `kc` block instead of once per `k`, and each packed B panel is reused for every A panel of the row block. The extra copy is linear in the block size, while the reuse it enables is proportional to the block size times `kc`.

</details>

<details>
<summary>Why is the NEON float kernel 8 x 12 and not, for example, 16 x 16?</summary>

The output tile must fit in registers with room for the operands. AArch64 has 32 vector registers of 4 floats. An 8 x 12 tile is 96 floats, 24 registers, plus 2 for the A column and 3 for the B row: 29 registers. A 16 x 16 tile alone would need 64 registers, so the compiler would spill accumulators to memory on every `k` step and lose the point of register blocking.

</details>

<details>
<summary>What goes wrong if a forked child calls `pool_.reset()` instead of `pool_.release()`?</summary>

`reset` runs the `ThreadPool` destructor, which signals and joins its worker threads. Those threads do not exist in the child, because `fork` copies only the calling thread. Joining them is undefined behaviour and in practice hangs or crashes the child. `release` drops the pointer without running the destructor, leaking a small object on purpose so that the next `parallel_for` can build a fresh pool.

</details>

<details>
<summary>Why does `log_softmax` compute `x - max - log(sum(exp(x - max)))` instead of `log(softmax(x))`?</summary>

When an entry is far below the row maximum, its softmax underflows to zero, and `log(0)` is `-inf`. The direct formula never forms that tiny probability: it subtracts a finite `log` of the row total from a finite shifted value, so the output stays finite and exact to rounding. Cross entropy relies on the same formula, which is why `cross_entropy_row` never materializes the softmax.

</details>

## Further reading

- Kazushige Goto and Robert van de Geijn, "Anatomy of High-Performance Matrix Multiplication", ACM Transactions on Mathematical Software, 2008. The packed-panel loop structure behind v4 to v6, with the reasoning for which operand lives in which cache level.
- Field Van Zee and Robert van de Geijn, "BLIS: A Framework for Rapidly Instantiating BLAS Functionality", ACM Transactions on Mathematical Software, 2015. Splits the Goto algorithm into a portable loop nest and a small micro-kernel, the same split kael uses between `multiply_packed` and the `Kernel` types.
- Samuel Williams, Andrew Waterman and David Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures", Communications of the ACM, 2009. The model behind the arithmetic intensity table and the reason elementwise kernels stop scaling.
- Nicholas Higham, "The Accuracy of Floating Point Summation", SIAM Journal on Scientific Computing, 1993. Error bounds for running and pairwise summation, which explain `LongSumIsMoreAccurateThanRunningTotal`.
- Ulrich Drepper, "What Every Programmer Should Know About Memory", 2007. A long, detailed account of caches, cache lines, prefetching and NUMA from the software side.
- The [Google Benchmark user guide](https://github.com/google/benchmark/blob/main/docs/user_guide.md). Counters, argument products and filters, as used in `benchmarks/`.
- [`src/parallel.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/parallel.cpp) and [`src/linalg/matmul_versions.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/linalg/matmul_versions.cpp). The two files are short enough to read in one sitting and contain every mechanism in this chapter.
- The [glossary](@/concepts/glossary.md) for short definitions of the terms used here.

Next: [09. CUDA backend](@/book/09-cuda-backend.md)
