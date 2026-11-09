+++
title = "Serving and tooling"
description = "An HTTP server for GPT-2 generation, debugging helpers for autograd, fuzzing, sanitizers, static analysis and the CI that runs all of it."
weight = 13
slug = "serving-and-tooling"

[extra]
chapter = "13"
kicker = "Chapter 13"
builds = "An HTTP server, debugging tools, fuzzing, sanitizers and CI"
source = "docs/13-serving-and-tooling.md"
tag = "chapter-13-polish"
+++

This last chapter covers what surrounds the library: an HTTP server that exposes GPT-2 generation, its Docker image, the debugging helpers in `debug.h`, fuzzing with libFuzzer, the sanitizer builds, clang-tidy and clang-format, and the CI jobs that run all of it. It maps to the `chapter-13-polish` tag, the same commit as `v1.0.0`.

| File | Role |
|------|------|
| [`server/app.py`](https://github.com/JetBerri/kael.cpp/blob/main/server/app.py) | FastAPI application: request validation, one generation slot, streaming, cancellation, metrics |
| [`server/requirements.txt`](https://github.com/JetBerri/kael.cpp/blob/main/server/requirements.txt) | Runtime dependencies of the server: `fastapi`, `pydantic`, `uvicorn` |
| [`server/Dockerfile`](https://github.com/JetBerri/kael.cpp/blob/main/server/Dockerfile) | Two-stage image: build the wheel, then run the server as a non-root user |
| [`.dockerignore`](https://github.com/JetBerri/kael.cpp/blob/main/.dockerignore) | Keeps `build*`, `data`, `venv` and `.git` out of the Docker build context |
| [`include/kael/debug.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/debug.h), [`src/debug.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/debug.cpp) | `explain`, `autograd_graph_dot`, `set_backward_trace`, `set_anomaly_detection` |
| [`src/autograd/engine.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/autograd/engine.cpp) | Where tracing and anomaly checks run during backward |
| [`fuzz/fuzz_safetensors.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/fuzz/fuzz_safetensors.cpp), [`fuzz/fuzz_idx.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/fuzz/fuzz_idx.cpp), [`fuzz/CMakeLists.txt`](https://github.com/JetBerri/kael.cpp/blob/main/fuzz/CMakeLists.txt) | libFuzzer targets and their build |
| [`fuzz/corpus/`](https://github.com/JetBerri/kael.cpp/tree/main/fuzz/corpus) | Seed inputs for each fuzzer |
| [`CMakeLists.txt`](https://github.com/JetBerri/kael.cpp/blob/main/CMakeLists.txt) | `KAEL_ENABLE_SANITIZERS`, `KAEL_ENABLE_TSAN`, `KAEL_BUILD_FUZZERS`, `kael_configure_target` |
| [`.clang-tidy`](https://github.com/JetBerri/kael.cpp/blob/main/.clang-tidy), [`.clang-format`](https://github.com/JetBerri/kael.cpp/blob/main/.clang-format) | Static analysis and formatting rules |
| [`.github/workflows/ci.yml`](https://github.com/JetBerri/kael.cpp/blob/main/.github/workflows/ci.yml), [`.github/workflows/wheels.yml`](https://github.com/JetBerri/kael.cpp/blob/main/.github/workflows/wheels.yml) | CI and wheel release workflows |
| [`python/tests/test_server.py`](https://github.com/JetBerri/kael.cpp/blob/main/python/tests/test_server.py), [`tests/test_debug.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_debug.cpp) | Tests for the server and the debug helpers |

## Prerequisites

- [06. Autograd](@/book/06-autograd.md): graph nodes, edges, `AccumulateGrad`, the engine.
- [08. CPU performance](@/book/08-cpu-performance.md): the caching allocator, which sanitizer builds disable.
- [10. Python bindings](@/book/10-python-bindings.md): GIL release and reacquisition around C++ calls.
- [11. Transformer](@/book/11-transformer.md): `generate` and its `on_token` callback.
- [12. Loading GPT-2](@/book/12-model-loading.md): `load_gpt2_weights`, `BpeTokenizer`, the safetensors reader that the fuzzer attacks.

## Why this layer exists

At the end of chapter 12 the library can load GPT-2 and generate text from a Python script. That is enough for a notebook. It is not enough for three other situations, and each one needs a different kind of tool.

**Someone else wants to use the model.** A script runs one prompt for one person. A service answers many clients over the network, and it must keep answering while a generation runs, refuse work it cannot do instead of queuing it forever, stop computing when a client leaves, and report what it is doing. None of that is the business of a tensor library, so it lives in a separate program, `server/app.py`.

**Something is wrong and the cause is hidden.** A view writes into memory you did not expect. A gradient is NaN and the loss is NaN, and nothing says which of 200 graph nodes produced it first. The core library reports results, not reasons. `debug.h` adds four functions that expose the internal state that the earlier chapters described: strides, storage sharing, the graph, and the gradients flowing through it.

**The code must stay correct as it changes.** The unit tests check the cases someone thought of. They do not check a safetensors header crafted to make the parser read past the end of a buffer, a data race that shows up one run in a thousand, or a use after free that happens to read valid-looking memory. Fuzzers, sanitizers and static analysis search for exactly those bugs, and CI runs them on every push so a regression is caught before it is merged.

```text
                +------------------------------------------+
                |  exposure     server/app.py, Dockerfile  |
                |  +------------------------------------+  |
                |  | observability   debug.h            |  |
                |  |  +------------------------------+  |  |
                |  |  |  kael core (chapters 01-12)  |  |  |
                |  |  +------------------------------+  |  |
                |  +------------------------------------+  |
                +------------------------------------------+
     verification around all of it: fuzzers, ASan, UBSan, TSan,
     clang-tidy, clang-format, CI on every push
```

Hold on to this mental model: the core computes, the debug helpers explain, the server exposes, and the verification tools keep the other three honest. None of the tools in this chapter change what the library computes.

## Concepts

### Blocking work on an event loop

FastAPI runs request handlers on an **asyncio event loop**: one thread that switches between many requests whenever one of them waits for I/O. That model works when every handler gives control back quickly. A handler that runs for two seconds without awaiting anything freezes every other request for those two seconds.

A GPT-2 forward pass is that kind of work. Consider a hypothetical generation that takes 2 seconds, started at time 0, and a health probe that arrives at 0.1 seconds:

```text
generation on the loop thread          generation on a worker thread

t=0.0  /generate starts computing      t=0.0  /generate hands work to a thread
t=0.1  /health arrives, waits          t=0.1  /health arrives, loop answers
t=2.0  /generate finishes              t=0.1  /health answered
t=2.0  /health answered (1.9 s late)   t=2.0  /generate finishes on its thread
```

A load balancer that sees `/health` take two seconds marks the instance as dead. The fix is to run the computation on another thread and let the loop keep serving. For that to help, the computation must not hold Python's **global interpreter lock** (GIL), the lock that allows only one thread at a time to run Python code. The kael bindings release the GIL around `generate` ([Python bindings](@/book/10-python-bindings.md)), so the C++ loop runs in parallel with the event loop.

### Admission control and backpressure

One GPT model object has one set of layers and runs one generation at a time. When a second request arrives during a generation, the server has three choices: run both and share the model (incorrect for this design), queue the second one, or refuse it.

An unbounded queue looks friendly and fails under load. Each queued request holds a connection and memory, latency grows without limit, and clients time out after the server has already spent work on them. **Admission control** means deciding at the door whether a request can be served now. **Backpressure** means telling the client to slow down. HTTP has a status for both: `429 Too Many Requests`, with a `Retry-After` header that says how many seconds to wait. A load balancer in front of several replicas can send the retry to another instance.

### Server-sent events

A plain HTTP response arrives when it is complete. For text generation, the client then waits for all tokens before seeing the first one. **Server-sent events** (SSE) is a one-way streaming format over an ordinary HTTP response: the content type is `text/event-stream`, and each event is one or more `data: ...` lines followed by a blank line. The connection stays open until the server ends the stream.

```text
HTTP/1.1 200 OK
content-type: text/event-stream

data: {"text": " not"}

data: {"text": " only"}

data: [DONE]

```

SSE needs no protocol upgrade, unlike WebSockets, and it is enough when only the server sends after the request.

### Cooperative cancellation

A thread cannot be safely killed from outside in Python or C++: it may hold a lock or be halfway through updating shared state. **Cooperative cancellation** means the worker checks a flag at safe points and stops itself. In kael the safe point is the token boundary: after each sampled token, `generate` calls `on_token`, and a falsy return value ends the loop. The cost is latency: a cancellation takes effect only after the forward pass in progress finishes.

### Metrics: counters, gauges and histograms

Prometheus is a monitoring system that periodically fetches a text page of metrics from each service. Three metric types matter here.

| Type | Behavior | Example question |
|------|----------|-----------------|
| Counter | only goes up, resets on restart | how many requests failed since start |
| Gauge | goes up and down | how many generations run right now |
| Histogram | a set of counters, one per upper bound, plus a sum and a count | what fraction of generations take under 2 seconds |

Prometheus histograms are **cumulative**: the bucket labeled `le="2"` (less than or equal to 2) counts every observation at most 2 seconds, including those already counted in `le="1"`. Trace three generations that take 0.4, 1.5 and 7 seconds through kael's buckets:

```text
bound   0.1  0.5  1  2  5  10  30  60  120  +Inf
0.4 s    -    x   x  x  x   x   x   x    x    x
1.5 s    -    -   -  x  x   x   x   x    x    x
7.0 s    -    -   -  -  -   x   x   x    x    x
count    0    1   1  2  2   3   3   3    3    3

kael_request_seconds_sum    8.9
kael_request_seconds_count  3
```

From these lines Prometheus can estimate percentiles over any time window, which a single "last duration" gauge cannot.

### Incremental UTF-8 decoding

GPT-2's tokenizer is byte-level ([Loading GPT-2](@/book/12-model-loading.md)): a token is a sequence of bytes, and nothing forces a token boundary to fall on a character boundary. In UTF-8, `ñ` is the two bytes `C3 B1` and `€` is the three bytes `E2 82 AC`. If tokens carry one byte each and each token is decoded on its own, the first byte `C3` is an incomplete sequence, and a decoder in `replace` mode emits the replacement character `U+FFFD`.

An **incremental decoder** keeps the bytes of an incomplete character between calls and emits the character when the last byte arrives:

```text
token  bytes  decoder buffer   emitted
1      C3     C3               ""
2      B1     (empty)          "ñ"
3      E2     E2               ""
4      82     E2 82            ""
5      AC     (empty)          "€"
end    -      (empty)          ""     decode(b"", final=True)
```

### Multi-stage container images

Building the kael wheel needs a C++ compiler, CMake and git. Running it needs none of them. A **multi-stage** Dockerfile has several `FROM` lines: an early stage builds artifacts, and the final stage starts from a clean base image and copies in only the artifacts. The toolchain stays behind in the discarded stage, which makes the final image smaller and removes tools an attacker could use inside it.

### Inspecting autograd from outside

Three questions come up repeatedly when training misbehaves, and each maps to one helper:

| Question | Helper |
|----------|--------|
| What memory does this tensor read, and who else shares it | `explain` |
| What graph will backward walk from this tensor | `autograd_graph_dot` |
| Where in backward does a gradient explode, vanish or become NaN | `set_backward_trace`, `set_anomaly_detection` |

The graph exporter writes **Graphviz DOT**, a plain-text graph language: `n0 -> n1` is an edge, and `[label="..."]` attaches text. The `dot` program renders it as an image.

### Coverage-guided fuzzing

**Coverage-guided fuzzing** runs a function on generated inputs, measures which code each input reaches, and keeps mutating the inputs that reach new code. The loop:

```text
  corpus (seed files)
        |
        v
  pick an input --> mutate bytes --> run target --> new coverage? --yes--> add to corpus
        ^                                |                  |
        |                                |                  no
        +--------------------------------+------------------+
                                         |
                              crash, sanitizer report or leak
                                         |
                                         v
                               save input as crash-<hash>
```

libFuzzer, part of LLVM, drives this loop inside the process, which is much faster than starting a program per input. Coverage alone finds crashes. Combined with sanitizers, it also finds silent memory errors, because the sanitizer turns an out-of-bounds read into a crash.

### Sanitizers

**Sanitizers** are compiler instrumentation that checks a program's behavior while it runs and aborts with a report at the first error. AddressSanitizer (ASan) surrounds each allocation with poisoned bytes called redzones and keeps a shadow map of which bytes are addressable; every load and store checks the map. UndefinedBehaviorSanitizer (UBSan) inserts checks for operations that the C++ standard leaves undefined, such as signed overflow. ThreadSanitizer (TSan) records which thread accessed which memory under which locks and reports two unsynchronized accesses where at least one is a write.

Sanitized code runs several times slower, so these builds are for tests, not for production.

### Static analysis and formatting

A **static analyzer** reads the source without running it and reports patterns that are likely bugs or needless costs. clang-tidy is LLVM's analyzer for C++; it needs a **compilation database** (`compile_commands.json`) that records the exact compiler flags for each file, because the analysis depends on macros and include paths. A **formatter** such as clang-format rewrites whitespace and layout to a fixed style, so code review discusses logic, not indentation.

### Continuous integration

**Continuous integration** (CI) runs the build and the tests on every push in fresh machines. Its value comes from the configuration matrix: one developer machine has one compiler and one OS, while CI builds with GCC and Clang, on Linux and macOS, with and without sanitizers, in parallel.

## Implementation

### The generation server

#### What it exposes

[`server/app.py`](https://github.com/JetBerri/kael.cpp/blob/main/server/app.py) loads GPT-2 small once at startup and serves three endpoints:

| Method and path | Response |
|-----------------|----------|
| `POST /generate` | JSON with the generated text, or a stream of server-sent events |
| `GET /health` | `{"status": "ok"}` |
| `GET /metrics` | Prometheus text format |

The request body is validated by a pydantic model. From `server/app.py`:

```python
class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARACTERS)
    max_tokens: int = Field(default=64, ge=1, le=MAX_NEW_TOKENS)
    temperature: float = Field(default=0.8, ge=0.0, le=10.0)
    top_k: int = Field(default=40, ge=0, le=MAX_TOP_K)
    stream: bool = False

    @field_validator("temperature")
    @classmethod
    def temperature_is_zero_or_usable(cls, value):
        if 0.0 < value < MIN_TEMPERATURE:
            raise ValueError(f"temperature must be 0 or at least {MIN_TEMPERATURE}")
        return value
```

| Field | Default | Allowed |
|-------|---------|---------|
| `prompt` | required | 1 to 8,192 characters (`MAX_PROMPT_CHARACTERS`) |
| `max_tokens` | 64 | 1 to 512 (`MAX_NEW_TOKENS`) |
| `temperature` | 0.8 | 0, or 0.001 (`MIN_TEMPERATURE`) to 10 |
| `top_k` | 40 | 0 to 1,000,000 (`MAX_TOP_K`); values at or above the vocabulary size disable top-k |
| `stream` | `false` | `true` for server-sent events |

The fields map onto the sampling options of `generate` ([Transformer](@/book/11-transformer.md)). A temperature of 0 selects the most likely token at every step (greedy decoding), so the output is deterministic. A `top_k` of 0 disables top-k filtering. In C++, `sample_next` applies top-k only when `top_k > 0 && top_k < vocabulary size`, which is why a value of 100,000 on a 50,257-token vocabulary behaves like 0. Note that the server's default `top_k` is 40 while the Python binding's default is 0.

A plain request and its response:

```bash
curl -s localhost:8000/generate -H 'content-type: application/json' \
  -d '{"prompt": "The meaning of life is", "max_tokens": 20, "temperature": 0}'
```

```json
{"text": "...", "tokens": 20, "seconds": 1.9}
```

`text` holds only the generated continuation, not the prompt. `tokens` is the number of generated tokens and `seconds` the wall time from the start of generation until the response is built. The values shown are illustrative.

#### Architecture

A GPT-2 forward pass takes tens of milliseconds to seconds and holds no Python lock, because the binding releases the GIL. As the concepts section showed, it must not run on the event loop thread. The server runs each generation on a dedicated worker thread and connects it to the handler through an `asyncio.Queue`.

```text
Client         Event loop                     Worker thread           kael (C++)
  |                |                               |                       |
  |-- POST ------->| read body (size cap)          |                       |
  |                | validate with pydantic        |                       |
  |                | reserve slot, or answer 429   |                       |
  |                | tokenize in a pool thread,    |                       |
  |                |   check the context           |                       |
  |                |-- start thread -------------->|                       |
  |                |                               |-- model.generate ---->|
  |                |                     each token|<-- on_token(token) ---|
  |                |<-- call_soon_threadsafe ------|                       |
  |<-- SSE event --|   (or buffer the token)       |-- continue or stop -->|
  |                |                               |         ...           |
  |                |<-- None, or the exception ----|                       |
  |                |                               | record metrics,       |
  |                |                               | release slot          |
  |<-- [DONE] -----|   (or the JSON body)          |                       |
```

Four pieces carry the design:

| Piece | Role |
|-------|------|
| `Generator` | Owns the model, the tokenizer, the device, the context length, the slot (`threading.Lock`), the `shutting_down` event and the `Metrics` |
| `Metrics` | Counters, gauges and the latency histogram behind one `threading.Lock` |
| `create_app(generator)` | Builds the FastAPI app and its handlers around a `Generator` |
| `load_app(data, device)` | Creates the real `Generator`: `kael.BpeTokenizer` from `vocab.json` and `merges.txt`, `kael.GPT()` with `load_gpt2`, the model moved to the device and switched to eval mode, and a context of 1024 |

`Generator` accepts any model with `generate` and any tokenizer with `encode` and `decode`. This is duck typing, and it is what makes the server testable: the tests replace the model and tokenizer with small fakes that misbehave on purpose.

#### The request flow

The `/generate` handler is four calls. From `server/app.py`:

```python
@app.post("/generate")
async def generate(http_request: Request):
    request = await parse_request(http_request)
    prompt = await prepare_prompt(request)
    started = time.perf_counter()
    queue, cancelled = start_generation(prompt, request)
    if request.stream:
        return StreamingResponse(
            stream_tokens(queue, cancelled), media_type="text/event-stream"
        )

    tokens = await collect_tokens(queue, cancelled, http_request)
    if tokens is None:
        return Response(status_code=499)
    generator.metrics.count_request("ok")
    return {
        "text": generator.tokenizer.decode(tokens).decode("utf-8", "replace"),
        "tokens": len(tokens),
        "seconds": time.perf_counter() - started,
    }
```

1. `parse_request` reads the body chunk by chunk and stops with 413 as soon as it exceeds `MAX_BODY_BYTES`. Only then does pydantic parse it, with `GenerateRequest.model_validate_json`. A `ValidationError` becomes 422 with the error list, built with `include_input=False` so the response never echoes the prompt back. Both count as `rejected`.
2. `prepare_prompt` reserves the generation slot, then tokenizes the prompt in a thread with `asyncio.to_thread`. Encoding 8,192 characters is not instant, and doing it on the loop would stall other requests (`test_the_event_loop_keeps_answering_while_a_prompt_is_encoded`). `prompt_tensor` rejects a prompt with no tokens (422) and a prompt whose token count plus `max_tokens` exceeds the context (413). Any failure here releases the slot.
3. `start_generation` creates the queue and a `threading.Event` named `cancelled`, and starts a daemon thread that calls `generator.run`.
4. Streaming requests return a `StreamingResponse` at once. Plain requests wait in `collect_tokens` and return JSON.

The failure handling in `prepare_prompt` is the part most likely to be written wrong. From `server/app.py`:

```python
async def prepare_prompt(request):
    generator.reserve()
    try:
        return await asyncio.to_thread(generator.prompt_tensor, request)
    except HTTPException:
        generator.metrics.count_request("rejected")
        generator.release()
        raise
    except Exception as error:
        generator.metrics.count_request("error")
        generator.release()
        raise HTTPException(500, "generation failed") from error
    except asyncio.CancelledError:
        generator.metrics.count_request("cancelled")
        generator.release()
        raise
```

`asyncio.CancelledError` reaches `prepare_prompt` when the request task is cancelled while the prompt is still being encoded. The test creates the handler task, waits until encoding has started, and cancels it. `CancelledError` derives from `BaseException`, not `Exception`, so the `except Exception` clause does not catch it and it needs its own clause. Without it, the slot would stay reserved and every later request would get 429 (`test_a_request_cancelled_while_encoding_frees_its_slot`).

Rejecting `len(ids) + max_tokens > context` up front means the server never relies on the sliding window that `generate` uses past the context length ([Transformer](@/book/11-transformer.md)). Past that point the KV cache is full, and every new token recomputes attention over the whole window, which is much slower. A worked example with the test context of 64: the byte-level test tokenizer turns `"hi"` into 2 tokens, so `max_tokens` 62 gives 64 and is accepted, while 63 gives 65 and is rejected with 413 (`test_a_request_that_would_outgrow_the_context_is_rejected`).

#### The concurrency slot

The slot is a `threading.Lock` acquired without blocking. From `server/app.py`:

```python
def reserve(self):
    if not self.slot.acquire(blocking=False):
        self.metrics.count_request("busy")
        raise HTTPException(429, "the model is busy", headers={"Retry-After": "1"})
    with self.metrics.lock:
        self.metrics.in_flight += 1
```

A second request while one is running gets `429 Too Many Requests` with `Retry-After: 1` immediately, instead of queuing without bound. Clients retry, and a load balancer in front can spread retries across replicas. A `threading.Lock` and not an `asyncio.Lock` is the right primitive, because the slot is released from the worker thread, outside the event loop.

The slot belongs to the worker thread, not to the HTTP handler. `Generator.run` releases it in a `finally` block after recording the metrics:

```python
def run(self, prompt, request, on_token):
    started = time.perf_counter()
    generated = 0

    def receive(token):
        nonlocal generated
        generated += 1
        return on_token(token.item()) and not self.shutting_down.is_set()

    try:
        self.model.generate(
            prompt,
            request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            on_token=receive,
        )
    finally:
        self.metrics.finish_generation(generated, time.perf_counter() - started)
        self.release()
```

The comment in `start_generation` states why the ownership matters:

```python
# The worker owns the slot: a client that never reads the body must not keep it.
threading.Thread(target=work, daemon=True).start()
```

If the slot were released when the streaming response finished, a client that opened a stream and never read it would hold the model forever. With the worker as owner, the slot is freed when generation ends, whatever the client does (`test_a_client_that_never_reads_the_stream_does_not_keep_its_slot`).

```text
time ->
handler   reserve |-- tokenize --| start worker | return StreamingResponse
worker                           |== generate tokens ==| metrics | release
client    request                                 never reads the body ...
slot      [held .......................................................]  free
```

#### Crossing from C++ to the event loop

`model.generate` is a C++ call with the GIL released. For each token, the C++ loop calls the Python callback, and the binding reacquires the GIL for that call only ([Python bindings](@/book/10-python-bindings.md)). The binding interprets the return value: `None` means continue, any other value is tested for truth with `PyObject_IsTrue`, so `False`, `0` and `""` stop generation while `True`, `1` and `"yes"` continue (`test_falsy_callback_answers_stop`, `test_truthy_or_missing_callback_answers_continue`).

The server's callback chain has two levels:

- `Generator.run` passes `receive`, which counts the token and returns `on_token(token.item()) and not self.shutting_down.is_set()`.
- `on_token`, defined in `start_generation`, hands the id to the event loop and reports whether to continue:

```python
def on_token(token):
    loop.call_soon_threadsafe(queue.put_nowait, token)
    return not cancelled.is_set()
```

`asyncio.Queue` is not thread-safe, so the worker never touches it directly. `call_soon_threadsafe` schedules `put_nowait` on the loop thread and wakes the loop. When `work` finishes, it enqueues `None` for success or the exception object for failure, which ends the consumer's loop:

```python
def work():
    try:
        generator.run(prompt, request, on_token)
        outcome = None
    except Exception as error:  # noqa: BLE001
        outcome = error
    loop.call_soon_threadsafe(queue.put_nowait, outcome)
```

The queue therefore carries three kinds of items: an `int` token id, `None` for the end, and an `Exception` for a failure. A callback that returns a falsy value stops generation in C++: `generate` breaks out of its loop after the current token. A callback that raises propagates the exception to the Python caller of `generate` (`test_python_callback_errors_reach_the_caller`).

#### Streaming with server-sent events

The server sends one event per decoded piece of text and a final `[DONE]` marker. From `server/app.py`:

```python
async def stream_tokens(queue, cancelled):
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    finished = False
    try:
        while (item := await queue.get()) is not None:
            if isinstance(item, Exception):
                finished = True
                generator.metrics.count_request("error")
                yield sse({"error": "generation failed"})
                return
            text = decoder.decode(generator.tokenizer.decode([item]))
            if text:
                yield sse({"text": text})
        tail = decoder.decode(b"", final=True)
        if tail:
            yield sse({"text": tail})
        finished = True
        generator.metrics.count_request("ok")
        yield "data: [DONE]\n\n"
    finally:
        cancelled.set()
        if not finished:
            generator.metrics.count_request("cancelled")
```

`sse(payload)` returns `f"data: {json.dumps(payload)}\n\n"`. Each payload is JSON, so a token containing a newline cannot break the event framing.

The incremental UTF-8 decoder handles the partial-character problem described in the concepts section. `BpeTokenizer.decode` returns `bytes`, and one GPT-2 token can hold half of a multi-byte character. `test_stream_does_not_split_multibyte_characters` feeds `"ñ€"` one byte per token, the five bytes `C3 B1 E2 82 AC`, and expects exactly the events `"ñ"` and `"€"`. Empty strings are not sent as events. The final `decode(b"", final=True)` flushes any incomplete sequence left at the end as a replacement character.

The `finished` flag decides how the request is counted. Every normal exit sets it: the end of the stream counts `ok`, and an exception item counts `error`. If the generator is closed early, the `finally` block runs with `finished` still false and counts `cancelled`. The `finally` block always sets `cancelled`, which is harmless after a normal end because the worker has already returned.

Errors are reported as `{"error": "generation failed"}`, never the exception text, so internal details do not leak to clients (`test_a_failed_generation_answers_500_and_frees_its_slot` checks that a `"secret detail"` message stays hidden).

#### Cancellation

A generation can run for many seconds. When the client goes away, the server stops computing tokens nobody will read. There are three cases.

- **Streaming.** When the client disconnects, Starlette stops iterating the generator and closes it. The `finally` block sets `cancelled`. The next `on_token` call returns `False`, and the C++ loop stops after at most one more token. The request is counted as `cancelled`.
- **Plain requests.** Nothing is being sent, so a disconnect is only visible by asking. `collect_tokens` waits on the queue with a timeout of `DISCONNECT_POLL_SECONDS = 0.25` and checks `http_request.is_disconnected()` at most every 0.25 seconds. On a disconnect it sets `cancelled`, counts `cancelled` and returns `None`, and the handler answers with status 499, the nginx convention for "client closed request". The `PENDING` sentinel marks a poll that timed out without a token.
- **Shutdown.** The FastAPI lifespan handler runs after the server stops accepting requests. It sets `shutting_down`, which makes `receive` return `False`, and then waits up to `SHUTDOWN_WAIT_SECONDS = 30` for the slot, so the process exits only after the worker has released it (`test_shutdown_stops_a_running_generation`).

From `server/app.py`, the polling loop for plain requests:

```python
async def collect_tokens(queue, cancelled, http_request):
    tokens = []
    next_check = time.monotonic() + DISCONNECT_POLL_SECONDS
    while True:
        try:
            item = await asyncio.wait_for(queue.get(), DISCONNECT_POLL_SECONDS)
        except asyncio.TimeoutError:
            item = PENDING
        if item is None:
            return tokens
        if isinstance(item, Exception):
            generator.metrics.count_request("error")
            raise HTTPException(500, "generation failed") from item
        if item is not PENDING:
            tokens.append(item)
        if time.monotonic() >= next_check:
            next_check = time.monotonic() + DISCONNECT_POLL_SECONDS
            if await http_request.is_disconnected():
                cancelled.set()
                generator.metrics.count_request("cancelled")
                return None
```

The timeout on `queue.get` guarantees the loop wakes up even when no token arrives, for example during the first forward pass over a long prompt. The `next_check` deadline keeps a fast stream of tokens from calling `is_disconnected` once per token.

The shutdown handler, from `server/app.py`:

```python
@asynccontextmanager
async def lifespan(_):
    yield
    generator.shutting_down.set()
    if await asyncio.to_thread(generator.slot.acquire, True, SHUTDOWN_WAIT_SECONDS):
        generator.slot.release()
```

The code after `yield` runs at shutdown. Acquiring the slot with a timeout is a way to wait for the worker: the acquire succeeds as soon as the worker releases the slot. The acquire runs in a thread because a blocking `acquire` on the loop would freeze the shutdown sequence.

#### Limits

| Limit | Value | Enforced in | Status |
|-------|-------|-------------|--------|
| Body size | `MAX_BODY_BYTES = 6 * 8192 + 1024 = 50,176` bytes | `parse_request`, while reading | 413 |
| Prompt length | 8,192 characters | pydantic | 422 |
| New tokens | 512 | pydantic | 422 |
| Prompt plus new tokens | 1,024 tokens (the context) | `prompt_tensor` | 413 |
| Concurrent generations | 1 | `reserve` | 429 |
| Temperature | 0 or 0.001 to 10 | pydantic | 422 |

`JSON_ESCAPE_FACTOR = 6` sizes the body cap: a character in the Basic Multilingual Plane can take six bytes in JSON as a `\uXXXX` escape. With the factor of 6, a prompt of 8,192 such characters takes at most 49,152 bytes, and the extra 1,024 bytes cover the field names and the other fields.

```text
"ñ" raw UTF-8          C3 B1               2 bytes
"ñ" JSON escape        ñ              6 bytes
8,192 x 6            = 49,152 bytes
+ 1,024 for the rest = 50,176 bytes = MAX_BODY_BYTES
```

Checking the size while reading the body means an attacker cannot make the server buffer a gigabyte before validation fails (`test_a_body_larger_than_the_limit_is_rejected_before_parsing`).

> [!NOTE]
> A character outside the Basic Multilingual Plane, such as an emoji, is one Python character but twelve bytes when a client escapes it as a surrogate pair (`😀`). A prompt of 8,192 such characters passes the pydantic length check but exceeds the body cap if the client escapes every character. Sent as raw UTF-8, the same character takes four bytes and fits.

The temperature floor exists because dividing the logits by a tiny positive number can overflow to infinity, and softmax of infinities is NaN ([Transformer](@/book/11-transformer.md)). The logits are `float32`, whose largest finite value is about `3.4e38`. A logit of 5 divided by `1e-40` gives `5e40`, which overflows to infinity, and softmax then computes infinity minus infinity. The floor of 0.001 keeps a logit of 100 at `1e5`, far from overflow. The test list includes `{"temperature": 1e-40}` among the rejected bodies.

A JSON string with a lone surrogate such as `"\ud800"` is not valid Unicode and is rejected with 422 (`test_a_lone_surrogate_is_a_rejected_request`). The same test list rejects a `top_k` of `2**70`. It exceeds `MAX_TOP_K`, so pydantic rejects it before it could overflow the C++ `std::int64_t` parameter of the binding.

#### Metrics

`Metrics` keeps counters behind a `threading.Lock`, because the worker thread and the event loop both update them. `GET /metrics` renders them in the Prometheus text format:

| Metric | Type | Meaning |
|--------|------|---------|
| `kael_requests_total{status="..."}` | counter | Requests by outcome: `ok`, `busy`, `rejected`, `error`, `cancelled` |
| `kael_generated_tokens_total` | counter | Tokens generated across all requests |
| `kael_tokens_per_second` | gauge | Rate of the most recent generation |
| `kael_requests_in_flight` | gauge | 0 or 1: generations holding the slot |
| `kael_request_seconds` | histogram | Generation time, buckets 0.1, 0.5, 1, 2, 5, 10, 30, 60, 120 seconds and `+Inf` |
| `kael_peak_memory_bytes` | gauge | Peak resident memory of the process |

The histogram is cumulative, as Prometheus requires. From `server/app.py`:

```python
def finish_generation(self, tokens, seconds):
    with self.lock:
        self.tokens += tokens
        self.tokens_per_second = tokens / seconds if seconds > 0 else 0.0
        self.latency_sum += seconds
        self.latency_total += 1
        for index, bound in enumerate(LATENCY_BUCKETS):
            if seconds <= bound:
                self.latency_counts[index] += 1
```

Every bucket whose bound is at least the observed time is incremented, and the `+Inf` bucket is rendered from `latency_total`. The duration covers the call to `model.generate`, not the request parsing or the tokenization.

A request outcome is counted exactly once, but not always in the same place: `ok` for a plain request is counted in the handler, `ok` for a stream in `stream_tokens`, `busy` in `reserve`, and `rejected` in `parse_request` or `prepare_prompt`.

`kael_peak_memory_bytes` comes from `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss`, which is in bytes on macOS and in kibibytes on Linux. `PEAK_RSS_UNIT` is 1 on `darwin` and 1024 elsewhere. On Linux, a reported `ru_maxrss` of 600,000 becomes `600,000 * 1024 = 614,400,000` bytes.

#### Running the server

```bash
scripts/download/gpt2.sh
pip install . -r server/requirements.txt
python server/app.py --data data/gpt2 --device cpu --host 127.0.0.1 --port 8000
```

The four flags above are all the options, and the values shown are their defaults. `--data` is the directory with `vocab.json`, `merges.txt` and `model.safetensors`. `--device` accepts only `cpu` or `cuda`, and `python server/app.py --help` describes each flag. `main` imports `uvicorn` inside the function, so importing `app.py` in tests does not require it.

### The Docker image

From [`server/Dockerfile`](https://github.com/JetBerri/kael.cpp/blob/main/server/Dockerfile):

```text
FROM python:3.12-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /source
COPY . .
RUN pip wheel --no-deps --wheel-dir /wheels .

FROM python:3.12-slim
RUN useradd --create-home --uid 1000 kael
COPY --from=build /wheels /wheels
COPY server/requirements.txt /app/requirements.txt
COPY server/app.py /app/app.py
RUN pip install --no-cache-dir /wheels/*.whl -r /app/requirements.txt && rm -rf /wheels
USER kael
WORKDIR /app
EXPOSE 8000
VOLUME /weights
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["python", "app.py", "--data", "/weights", "--host", "0.0.0.0"]
```

- The build stage has the compiler, CMake and git (CMake's `FetchContent` downloads nlohmann_json during configuration). It builds the `kael-tensor` wheel from the repository root. `--no-deps` builds only kael's own wheel.
- The runtime stage copies only the wheel and `app.py`. The compiler toolchain stays in the discarded build stage, which keeps the final image small.
- The server runs as user `kael` (uid 1000), not root. A vulnerability in the server then runs with the rights of an unprivileged user.
- Weights are not baked into the image. They are mounted at `/weights`, so one image serves any compatible checkpoint and image rebuilds do not copy 548 MB.
- The health check uses the Python standard library because the slim image has no `curl`. `--start-period=60s` covers the time to load the weights, so the container is not marked unhealthy while it starts.
- The server listens on `0.0.0.0` inside the container so that the published port reaches it. The default `127.0.0.1` would accept only connections from inside the container.
- The image runs on the CPU. It has no CUDA runtime.

The Dockerfile copies the whole build context, so build from the repository root:

```bash
docker build -f server/Dockerfile -t kael-server .
docker run --rm -p 8000:8000 -v "$PWD/data/gpt2:/weights:ro" kael-server
```

[`.dockerignore`](https://github.com/JetBerri/kael.cpp/blob/main/.dockerignore) excludes `.git`, `build`, `build-*`, `cuda-report*`, `data`, `venv`, `experiments`, `legacy`, `notes` and `**/__pycache__`. The context stays small, and local build directories, which hold object files compiled for the host, do not leak into the image and confuse the CMake build inside it.

### Debugging tools

[`include/kael/debug.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/debug.h) declares four functions:

```cpp
[[nodiscard]] std::string explain(const Tensor& t);
[[nodiscard]] std::string autograd_graph_dot(const Tensor& t);

void set_backward_trace(std::ostream* output);
void set_anomaly_detection(bool enabled);
```

Python exposes three of them as `kael.explain`, `kael.autograd_graph` and `kael.set_anomaly_detection`. `set_backward_trace` has no binding; from Python, use the environment variable described below.

#### explain

`explain` prints everything about a tensor's memory and autograd state, one field per line. For the view `x.T().slice(0, 1, 3)` of a 4x4 float tensor `x`:

```text
shape: [2, 4]
strides: [1, 4]
offset: 1
dtype: float32
device: cpu
contiguous: no
storage: 0x9b0c04000, 64 bytes, version 0, references: 2
requires_grad: no
leaf: yes
has grad: no
```

Trace where each number comes from ([Shapes and strides](@/book/01-shapes-and-strides.md), [Views](@/book/03-views.md)):

```text
x                 shape [4, 4]  strides [4, 1]  offset 0
x.T()             shape [4, 4]  strides [1, 4]  offset 0    strides swapped
.slice(0, 1, 3)   shape [2, 4]  strides [1, 4]  offset 0 + 1*1 = 1
storage           16 floats * 4 bytes = 64 bytes
```

Each line answers a question that comes up when a view behaves unexpectedly:

- `strides` and `offset` show how the view maps into storage. Element `[i, j]` of the view lives at storage index `1 + i*1 + j*4`.
- `contiguous: no` means a kernel cannot treat the elements as one flat run, and `reshape` will copy.
- `references: 2` is the `use_count` of the `shared_ptr<Storage>`: the base `x` and the view share the same 64 bytes, so writing through one changes the other. The intermediate `x.T()` is a temporary that no longer exists, so it does not count. See [Tensor as a shared handle](@/adr/0002-tensor-as-handle.md).
- `version` is the storage version counter, which autograd uses to detect in-place modification of saved tensors ([Autograd](@/book/06-autograd.md)).
- When the tensor has a `grad_fn`, a last line names it, for example `grad_fn: MulBackward`.

#### Graph export

`autograd_graph_dot` walks the graph from the tensor's gradient edge and prints it in Graphviz DOT format. From [`src/debug.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/debug.cpp):

```cpp
std::string autograd_graph_dot(const Tensor& t) {
    std::ostringstream dot;
    dot << "digraph autograd {\n";
    const autograd::Edge root = t.requires_grad() ? autograd::gradient_edge(t) : nullptr;
    if (root) {
        std::unordered_map<const autograd::Node*, std::size_t> ids{{root.get(), 0}};
        std::vector<const autograd::Node*> pending{root.get()};
        dot << "  n0 [label=\"" << root->name() << "\"];\n";
        while (!pending.empty()) {
            const autograd::Node* node = pending.back();
            pending.pop_back();
            const std::size_t from = ids.at(node);
            for (const autograd::NodeInput& input : node->inputs()) {
                const autograd::Node* next = input.edge.get();
                if (next == nullptr) {
                    continue;
                }
                const auto [slot, inserted] = ids.try_emplace(next, ids.size());
                if (inserted) {
                    dot << "  n" << slot->second << " [label=\"" << next->name() << "\"];\n";
                    pending.push_back(next);
                }
                dot << "  n" << from << " -> n" << slot->second << " [label=\""
                    << to_string(input.shape) << "\"];\n";
            }
        }
    }
    dot << "}\n";
    return dot.str();
}
```

The walk is a depth-first traversal with an explicit stack (`pending`). The `ids` map assigns each node a number the first time it is seen, keyed by its address, so a node reached through two paths is printed once. Every input edge is printed, labeled with the shape of the gradient it carries. Inputs without a node, such as tensors that do not require grad, have a null edge and are skipped.

For `sum(x * x)` with `x` of shape `[2]`:

```text
digraph autograd {
  n0 [label="SumBackward"];
  n1 [label="MulBackward"];
  n0 -> n1 [label="[2]"];
  n2 [label="AccumulateGrad"];
  n1 -> n2 [label="[2]"];
  n1 -> n2 [label="[2]"];
}
```

```text
SumBackward (n0)
     |  [2]
     v
MulBackward (n1)
     |  [2]   two edges, one per operand
     v  [2]
AccumulateGrad (n2)  -> x.grad
```

`MulBackward` has two edges to the same `AccumulateGrad` because `x` is both operands, which is why `x.grad` ends up as `2x`. Render the output with `dot -Tpng graph.dot -o graph.png`. A tensor that does not require grad gives an empty `digraph autograd {}`.

#### Backward tracing

`set_backward_trace(&stream)` makes the engine print one line per node as it runs, with the shape of the incoming gradient and its L2 norm. From [`src/autograd/engine.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/autograd/engine.cpp):

```cpp
void trace_node(std::ostream& output, const Node& node, const Tensor& gradient) {
    output << "backward " << node.name() << " shape=" << to_string(gradient.shape()) << " norm="
           << std::sqrt(kael::sum(gradient.to(DType::F64) * gradient.to(DType::F64)).item())
           << '\n';
}
```

The gradient is converted to `float64` before squaring, so the norm of a large `float32` gradient does not overflow while it is being computed. The engine calls `trace_node` right before `node->apply`, with the gradient the node is about to consume:

```cpp
if (std::ostream* output = detail::trace_output().load()) {
    trace_node(*output, *node, incoming);
}
gradients = node->apply(incoming);
if (detail::anomaly_detection().load()) {
    check_anomaly(*node, gradients);
}
```

For `sum(w * w).backward()` with `w = [3, 3]`:

```text
backward SumBackward shape=[] norm=1
backward MulBackward shape=[2] norm=1.41421
backward AccumulateGrad shape=[2] norm=8.48528
```

Trace the numbers:

```text
SumBackward      receives the seed gradient 1 (a scalar)       norm sqrt(1)       = 1
MulBackward      receives [1, 1], the sum spread to each input norm sqrt(1 + 1)   = 1.41421
                 returns w * [1, 1] = [3, 3] for each operand
AccumulateGrad   receives [3, 3] + [3, 3] = [6, 6]             norm sqrt(36 + 36) = 8.48528
```

The two gradients for `w` are summed in the engine's buffer before `AccumulateGrad` runs, so it appears once. The value labeled `norm` is the same quantity that `clip_grad_norm_` compares with `max_norm` ([Neural networks and training](@/book/07-neural-networks.md)). The lines appear in the engine's execution order, which makes it possible to see where a gradient grows or shrinks by orders of magnitude.

`set_backward_trace(nullptr)` turns tracing off. Setting the environment variable `KAEL_TRACE_BACKWARD` to any value turns it on at startup with output to standard error:

```cpp
std::atomic<std::ostream*>& trace_output() {
    static std::atomic<std::ostream*> output =
        std::getenv("KAEL_TRACE_BACKWARD") != nullptr ? &std::cerr : nullptr;
    return output;
}
```

The function-local `static` is initialized once, the first time the engine checks the trace stream, so the variable is read once. Tracing computes a reduction per node and is slow; it is a debugging aid only.

#### Anomaly detection

A NaN produced deep in backward spreads to every parameter, and by the time the loss is NaN the origin is lost. `set_anomaly_detection(true)` makes the engine check the outputs of every node right after it runs. From `src/autograd/engine.cpp`:

```cpp
void check_anomaly(const Node& node, const GradientList& gradients) {
    for (const std::optional<Tensor>& gradient : gradients) {
        if (gradient && is_floating_point(gradient->dtype())) {
            KAEL_CHECK(!kael::any(*gradient != *gradient).item<std::uint8_t>(),
                       "Function " << node.name() << " returned NaN in its gradient");
        }
    }
}
```

`x != x` is true only for NaN, because IEEE 754 defines NaN as unequal to everything, itself included. For `sum(sqrt(x)).backward()` with `x = -1`, the forward value `sqrt(-1)` is already NaN, and the gradient `0.5 / sqrt(x)` is NaN too. The error names the culprit:

```text
kael error at <file>:<line>: check '...' failed: Function SqrtBackward returned NaN in its gradient
```

Infinite gradients are not flagged, only NaN.

> [!WARNING]
> Both the trace stream and the anomaly flag are process-wide globals, not per-thread settings. Turn them on around the code under investigation and off again afterwards, including on the error path, as `AnomalyDetection.NamesTheNodeProducingNan` does in its `catch` block.

`autograd::detail::trace_output()` and `autograd::detail::anomaly_detection()` return references to a `std::atomic<std::ostream*>` and a `std::atomic<bool>`, so switching them from one thread while another runs backward is not a data race. The stream itself is not synchronized: two threads tracing into one stream interleave their lines. Anomaly detection adds a full pass over every gradient and is off by default (`IsOffByDefaultAndLeavesCleanGraphsAlone`).

PyTorch offers the same two tools under different names: `torch.autograd.set_detect_anomaly` for NaN checks, and hooks or `torchviz` for graph inspection.

### Fuzzing

kael fuzzes its two parsers of untrusted files: safetensors ([Loading GPT-2](@/book/12-model-loading.md)) and the MNIST IDX format ([Neural networks and training](@/book/07-neural-networks.md)). Both read sizes and offsets from the file and then index memory with them, which is exactly where a crafted file can cause an out-of-bounds read.

#### The targets

A libFuzzer target is one function, `LLVMFuzzerTestOneInput`, called once per generated input. From [`fuzz/fuzz_safetensors.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/fuzz/fuzz_safetensors.cpp):

```cpp
extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    static const std::filesystem::path path =
        std::filesystem::temp_directory_path() / ("kael_fuzz_st_" + std::to_string(getpid()));
    std::ofstream(path, std::ios::binary)
        .write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    for (const auto mode : {kael::io::LoadMode::Copy, kael::io::LoadMode::Mapped}) {
        try {
            const auto loaded = kael::io::read_safetensors(path, mode);
            for (const auto& [name, tensor] : loaded.tensors) {
                if (tensor.nbytes() > 0) {
                    volatile auto first = *tensor.data_ptr_raw();
                    volatile auto last = tensor.data_ptr_raw()[tensor.nbytes() - 1];
                    (void)first;
                    (void)last;
                }
            }
        } catch (const kael::Error&) {
        }
    }
    return 0;
}
```

- The reader takes a path, so the harness writes each input to a temporary file. The name includes the process id, so parallel fuzzing jobs (`-jobs=N`) do not overwrite each other's file.
- Both load modes run on every input. They share the validation but differ in how they reach the bytes: `Copy` reads into a new buffer, `Mapped` points tensors into a memory mapping of the file when the data is aligned for its dtype.
- After a successful load, the harness reads the first and last byte of every tensor through `volatile` variables, so the compiler cannot drop the reads. If the parser accepted offsets that point outside the file or the mapping, AddressSanitizer reports the out-of-bounds access here.
- Only `kael::Error` is caught. A rejected file is the expected outcome for most inputs. Any other exception (for example an `nlohmann` exception or `std::bad_alloc`), a crash or a sanitizer report counts as a finding.

[`fuzz/fuzz_idx.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/fuzz/fuzz_idx.cpp) follows the same pattern for `kael::data::read_idx_images` and `kael::data::read_idx_labels`, calling both on every input, each in its own `try` block.

#### The seed corpus

`fuzz/corpus/fuzz_safetensors/seed1` is a valid 247-byte safetensors file. Its layout:

```text
bytes 0-7      header length, little-endian u64 = 0xde = 222
bytes 8-229    JSON header (222 bytes):
               {"__metadata__": {"format": "pt"},
                "a": {"dtype": "F32",  "shape": [2],    "data_offsets": [0, 8]},
                "b": {"dtype": "BOOL", "shape": [1],    "data_offsets": [8, 9]},
                "c": {"dtype": "I64",  "shape": [1, 1], "data_offsets": [9, 17]}}
bytes 230-246  data (17 bytes)
total          8 + 222 + 17 = 247
```

The seed covers metadata, three dtypes, a multi-dimensional shape and tightly packed offsets. Tensor `c` starts at file byte `230 + 9 = 239`, which is not a multiple of 8, and tensor `a` starts at byte 230, which is not a multiple of 4. In `Mapped` mode the reader copies such misaligned tensors instead of pointing into the mapping, so the seed exercises both paths of the mapped loader in a single input.

`fuzz/corpus/fuzz_idx/` holds two files:

```text
images (24 bytes)   00 00 08 03   magic: unsigned bytes, 3 dimensions
                    00 00 00 02   2 images
                    00 00 00 02   2 rows
                    00 00 00 02   2 columns
                    8 pixel bytes = 2 * 2 * 2
labels (11 bytes)   00 00 08 01   magic: unsigned bytes, 1 dimension
                    00 00 00 03   3 labels
                    01 02 03
```

Starting from valid files lets the fuzzer reach the deeper checks quickly instead of spending its time failing on the first byte.

#### Building and running

From [`fuzz/CMakeLists.txt`](https://github.com/JetBerri/kael.cpp/blob/main/fuzz/CMakeLists.txt):

```cmake
foreach (target fuzz_safetensors fuzz_idx)
    add_executable(${target} ${target}.cpp)
    target_link_libraries(${target} PRIVATE kael::kael)
    target_compile_options(${target} PRIVATE -fsanitize=fuzzer)
    target_link_options(${target} PRIVATE -fsanitize=fuzzer)
    kael_configure_target(${target})
endforeach ()

target_compile_options(kael PRIVATE -fsanitize=fuzzer-no-link)
```

`-fsanitize=fuzzer` links libFuzzer's `main` into the targets. The library itself gets `-fsanitize=fuzzer-no-link`, which adds the coverage instrumentation without a second `main`. Without it, libFuzzer would only see coverage in the small harness files and could not guide mutations through the parser.

`KAEL_BUILD_FUZZERS` needs a Clang that ships libFuzzer. Apple's Clang does not; on macOS use Homebrew's LLVM.

```bash
cmake -S . -B build-fuzz -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_COMPILER=clang++ -DKAEL_BUILD_FUZZERS=ON -DKAEL_BUILD_TESTS=OFF \
  -DKAEL_ENABLE_SANITIZERS=ON
cmake --build build-fuzz --parallel
build-fuzz/fuzz/fuzz_safetensors -max_total_time=60 -malloc_limit_mb=2048 fuzz/corpus/fuzz_safetensors
```

| Flag | Effect |
|------|--------|
| `-max_total_time=60` | stop after 60 seconds |
| `-malloc_limit_mb=2048` | treat any single allocation above 2 GiB as a finding, which catches headers that trick the parser into a huge allocation |
| corpus directory argument | read seeds from it and write new interesting inputs into it |

libFuzzer writes new interesting inputs into the corpus directory it is given, so a local run grows `fuzz/corpus/`. Review those files before committing them.

To reproduce a crash, run the target on the saved input: `build-fuzz/fuzz/fuzz_safetensors crash-<hash>`. With one file argument, libFuzzer runs that input once instead of fuzzing.

### Sanitizer builds

| Sanitizer | Finds | CMake option |
|-----------|-------|--------------|
| AddressSanitizer (ASan) | out-of-bounds access, use after free, double free, leaks on Linux | `KAEL_ENABLE_SANITIZERS=ON` |
| UndefinedBehaviorSanitizer (UBSan) | signed overflow, misaligned pointers, invalid shifts, invalid `bool` or enum values | `KAEL_ENABLE_SANITIZERS=ON` (together with ASan) |
| ThreadSanitizer (TSan) | data races between threads | `KAEL_ENABLE_TSAN=ON` |

From `CMakeLists.txt`, inside `kael_configure_target`, which every target (library, tests, examples, fuzzers, benchmarks) calls:

```cmake
if (KAEL_ENABLE_SANITIZERS)
    set(san_flags -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_compile_options(${target} PRIVATE $<${gcc_like}:${san_flags}>)
    target_link_options(${target} PRIVATE $<${gcc_family}:${san_flags}>)
endif ()
if (KAEL_ENABLE_TSAN)
    set(tsan_flags -fsanitize=thread -fno-omit-frame-pointer)
    target_compile_options(${target} PRIVATE $<${gcc_like}:${tsan_flags}>)
    target_link_options(${target} PRIVATE $<${gcc_family}:${tsan_flags}>)
endif ()
```

`gcc_family` is a generator expression that is true for GCC, Clang and AppleClang; `gcc_like` further restricts compile options to C++ sources, so CUDA sources do not receive host-only flags.

- `-fno-omit-frame-pointer` keeps stack traces in the reports readable.
- TSan and ASan cannot run in one binary. Configuring with both options fails with `FATAL_ERROR` and the message `ThreadSanitizer cannot be combined with AddressSanitizer`.
- Sanitizers and benchmarks are also mutually exclusive, because instrumentation distorts timings: `KAEL_BUILD_BENCHMARKS` with `KAEL_ENABLE_SANITIZERS` fails at configuration time.
- Either sanitizer option defines `KAEL_NO_CACHING_ALLOCATOR` on the library.

The last point deserves a trace. The caching allocator ([CPU performance](@/book/08-cpu-performance.md)) keeps freed blocks and hands them out again. ASan only sees the blocks the system allocator gives out and takes back:

```text
with the caching allocator             without it (sanitizer builds)

p = alloc(256)   block B from cache     p = alloc(256)   malloc, ASan tracks B
free(p)          B back to cache        free(p)          free, ASan poisons B
q = alloc(256)   B reused for q         q = alloc(256)   new block C
read *p          reads q's data,        read *p          ASan: heap-use-after-free
                 no report
```

With caching compiled out, every free goes to the system allocator, where ASan sees it. The allocator can also be disabled at runtime in a normal build by setting the environment variable `KAEL_NO_CACHING_ALLOCATOR`.

Some tests adapt to the slower build. `Gpt.Gpt2SmallHasTheKnownParameterCount` checks `kael::caching_allocator_enabled()` and skips when it is off, with the reason "drawing 250 million samples is too slow under sanitizers".

Build and run the tests under each sanitizer:

```bash
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DKAEL_ENABLE_SANITIZERS=ON
cmake --build build-asan --parallel && ctest --test-dir build-asan --output-on-failure

cmake -S . -B build-tsan -DCMAKE_BUILD_TYPE=RelWithDebInfo -DKAEL_ENABLE_TSAN=ON
cmake --build build-tsan --parallel && ctest --test-dir build-tsan --output-on-failure
```

TSan matters for kael because of the thread pool and `parallel_for` ([CPU performance](@/book/08-cpu-performance.md)), the global random generator and its mutex, and the Python server's worker threads calling into C++.

### clang-tidy and clang-format

#### clang-tidy

[`.clang-tidy`](https://github.com/JetBerri/kael.cpp/blob/main/.clang-tidy) enables two check families and treats every finding as an error:

```text
Checks: >
  -*,
  bugprone-*,
  -bugprone-easily-swappable-parameters,
  -bugprone-narrowing-conversions,
  -bugprone-exception-escape,
  -bugprone-unchecked-optional-access,
  -bugprone-random-generator-seed,
  performance-*,
  -performance-enum-size,
  -performance-avoid-endl
HeaderFilterRegex: '(include/kael|src)/.*'
ExcludeHeaderFilterRegex: '_deps/.*'
WarningsAsErrors: '*'
```

- `-*` first disables everything, so only the families listed afterwards run.
- `bugprone-*` catches likely bugs: dangling references, suspicious string comparisons, use after move, integer division in floating point context and similar.
- `performance-*` catches needless copies, such as a `const std::string` parameter passed by value or a range-for loop that copies elements.
- The repository does not record why each check is disabled. Likely reasons: many kael functions take several `std::int64_t` dimensions in a row (`easily-swappable-parameters`), narrowing is already reported by the `-Wconversion` warning flag that `kael_configure_target` adds, and fixed seeds are deliberate in a library with `manual_seed`.
- `HeaderFilterRegex` restricts reports to kael's own headers and sources, and `ExcludeHeaderFilterRegex` skips dependencies fetched into `_deps`.

Some code needs an exception. `KAEL_CHECK_AS` in `include/kael/error.h` streams its `msg` argument without parentheses so that chains like `"index " << i` work, and wraps the macro in `NOLINTBEGIN(bugprone-macro-parentheses)` and `NOLINTEND`.

Run clang-tidy on the sources with a compilation database:

```bash
cmake -S . -B build-tidy -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DKAEL_FAKE_CUDA=ON -DKAEL_BUILD_TESTS=OFF
run-clang-tidy -p build-tidy -quiet "src/.*\\.cpp$"
```

`KAEL_FAKE_CUDA=ON` compiles the host emulation of the CUDA kernels ([CUDA backend](@/book/09-cuda-backend.md)), so clang-tidy also analyzes the kernel code paths without a CUDA toolkit.

#### clang-format

[`.clang-format`](https://github.com/JetBerri/kael.cpp/blob/main/.clang-format) is based on LLVM style with these changes:

| Setting | Value | Effect |
|---------|-------|--------|
| `IndentWidth` | 4 | 4-space indentation |
| `ColumnLimit` | 100 | lines wrap at 100 columns |
| `PointerAlignment` | `Left` | `int* p`, not `int *p` |
| `InsertBraces` | `true` | braces inserted on every control statement |
| `AllowShortIfStatementsOnASingleLine` | `Never` | no single-line `if` bodies |
| `AllowShortLoopsOnASingleLine` | `false` | no single-line loop bodies |
| `IncludeBlocks` | `Regroup` | includes sorted into groups by `IncludeCategories` |

The include groups are: standard and system headers (`<...>`, priority 1), then `gtest` and `benchmark` (priority 2), then `kael/` headers (priority 3), with any other quoted include in a final block. Format a file in place with `clang-format -i src/view.cpp`, or check without changing anything with `clang-format --dry-run --Werror <files>`. No CI job runs clang-format.

### Continuous integration

[`.github/workflows/ci.yml`](https://github.com/JetBerri/kael.cpp/blob/main/.github/workflows/ci.yml) runs on every push and pull request. All jobs start in parallel; none depends on another.

```text
push or pull request
   |
   +--> build-test         6 configurations
   +--> cuda-compile-only  nvidia/cuda container, build only
   +--> python             Linux and macOS
   +--> fuzz               both fuzzers, 60 s each
   +--> coverage           llvm-cov report
   +--> clang-tidy         every file under src/
```

| Job | What it does |
|-----|--------------|
| `build-test` | Configures with `CMAKE_BUILD_TYPE=Debug` and `KAEL_WARNINGS_AS_ERRORS=ON`, builds, runs `ctest`. Six configurations: `linux-gcc`, `linux-clang`, `macos-clang`, `linux-clang-sanitizers` (ASan and UBSan), `linux-clang-tsan`, and `linux-clang-fake-cuda` (host-emulated CUDA kernels under ASan and UBSan). `fail-fast: false` lets every configuration report |
| `cuda-compile-only` | In the `nvidia/cuda:12.6.2-devel-ubuntu22.04` container, configures a `Release` build with `KAEL_ENABLE_CUDA=ON` and `CMAKE_CUDA_ARCHITECTURES=86` and builds. It builds the real CUDA backend with `nvcc` on every push. GitHub runners have no GPU, so execution on the device is validated with `scripts/cuda/validate.sh` ([CUDA backend](@/book/09-cuda-backend.md)) |
| `python` | On Ubuntu and macOS with Python 3.12: `pip install ".[test]"` (pytest, hypothesis, torch, transformers, safetensors, tiktoken, fastapi, httpx) and `pytest python/tests -q`. Tests that need the downloaded GPT-2 files skip |
| `fuzz` | Builds both fuzzers with Clang, `RelWithDebInfo`, ASan and UBSan, then runs each for 60 seconds on its corpus with a 2 GiB allocation limit |
| `coverage` | Builds the tests with Clang's source-based coverage (`-fprofile-instr-generate -fcoverage-mapping`) and fake CUDA, runs `kael_tests`, and prints an `llvm-cov report` for `src` and `include/kael`. It calls the versioned tools `llvm-profdata-$version` and `llvm-cov-$version`, where `version` is the major version from `clang -dumpversion` |
| `clang-tidy` | Exports a compilation database with fake CUDA and runs `run-clang-tidy` on every file under `src/` |

The matrix in `build-test` is where each tool in this chapter meets the unit tests. Each configuration answers a different question:

| Configuration | Question it answers |
|---------------|--------------------|
| `linux-gcc`, `linux-clang` | does the code compile warning-free with both compilers |
| `macos-clang` | does it work on AppleClang and macOS's libc++ |
| `linux-clang-sanitizers` | do the tests trigger memory errors or undefined behavior |
| `linux-clang-tsan` | do the tests contain data races |
| `linux-clang-fake-cuda` | are the CUDA kernels correct, checked on the host under ASan and UBSan |

The coverage job calls versioned tool names because Ubuntu's LLVM packages install `llvm-cov-18` and similar names, and an unversioned `llvm-cov` may be missing or belong to another LLVM version than `clang`. A profile written by one version cannot always be read by another.

[`.github/workflows/wheels.yml`](https://github.com/JetBerri/kael.cpp/blob/main/.github/workflows/wheels.yml) runs only on manual dispatch (`workflow_dispatch`). It builds wheels with `cibuildwheel` on `ubuntu-latest`, `ubuntu-24.04-arm`, `macos-latest` and `macos-15-intel`, builds a source distribution, and, if the `publish` input is `testpypi` or `pypi`, uploads them with trusted publishing (`id-token: write`, no stored token). [Python bindings](@/book/10-python-bindings.md) describes the wheel configuration.

## Edge cases and pitfalls

Server:

- A slow client and the slot. The worker thread owns the slot and releases it in `finally`, so a client that never reads its stream cannot block the server.
- A client that disconnects during a plain request is detected only by polling, within 0.25 seconds plus the time to produce one token.
- The `cancelled` flag stops generation at a token boundary. A single long forward pass, such as the first step over a long prompt, runs to completion.
- Tokens per second is a gauge of the last generation only. Use the histogram and the token counter for rates over time.
- Exceptions inside the worker become 500 or an SSE error event with a fixed message. The original exception is chained with `from item` in the plain path, but no handler logs it, so the detail reaches neither the client nor the server log.
- A streaming response has already sent `200 OK` when generation fails, so the failure arrives as an `{"error": ...}` event, not as a status code. Clients must check each event.
- `HTTPException` details from pydantic exclude the input, so a rejected prompt is not echoed back.
- The body cap assumes at most six bytes per character. A client that escapes characters outside the Basic Multilingual Plane as surrogate pairs uses twelve, and a long prompt of such characters gets 413.
- `kael_peak_memory_bytes` is a peak, not the current usage. It never goes down.
- Do not mount weights for another configuration. `load_app` always builds `kael.GPT()` with GPT-2 small defaults, and other checkpoints fail to load.
- The image has no CUDA runtime. `--device cuda` inside the container fails.

Debugging tools:

- `explain` prints the storage address. It differs between runs; do not compare it in tests.
- `KAEL_TRACE_BACKWARD` is read once. Setting it after the first backward pass has no effect; use `set_backward_trace` from C++ instead.
- The stream passed to `set_backward_trace` must outlive every backward pass that runs while it is set. Reset it to `nullptr` before the stream is destroyed.
- Anomaly detection reports NaN, not infinity, and slows backward down. Enable it only while hunting a NaN.

Verification tools:

- The fuzzer harness writes to the system temporary directory and grows the corpus directory in place.
- ASan and TSan cannot be combined, and neither can sanitizers and benchmarks. Use separate build directories.
- Sanitizer builds disable the caching allocator, so their timings and memory use do not represent a normal build.
- clang-format is not enforced in CI. Run it before committing.

## How it is tested

[`python/tests/test_server.py`](https://github.com/JetBerri/kael.cpp/blob/main/python/tests/test_server.py) is skipped unless `fastapi` and `httpx` are installed (`pytest.importorskip`). It loads `server/app.py` as a module from its path, drives the app with FastAPI's `TestClient`, and uses a tiny `kael.GPT(vocab_size=256, max_positions=64, layers=1, heads=2, dimension=16)` with a byte-level tokenizer that has 256 single-byte tokens and no merges. With that tokenizer, the number of tokens equals the number of UTF-8 bytes, which makes context arithmetic in the tests exact.

Several tests replace parts of the system with fakes to force a behavior:

| Fake | Used for |
|------|----------|
| `FakeHttpRequest` | controls the body stream and the disconnected flag; tests call the endpoint function directly |
| `EndlessModel` | a model whose `generate` loops until `on_token` returns a falsy value, to test cancellation and shutdown |
| `Sequence` and `Bytes` | a model that emits the bytes of `"ñ€"` one per token and a tokenizer that decodes ids to raw bytes, for multi-byte streaming |
| `SlowTokenizer` | an encoder that sleeps, to show that the loop keeps answering and that cancellation during encoding frees the slot |
| `Broken` | a model whose `generate` raises `RuntimeError("secret detail")`, for the plain and streaming failure paths |
| `BrokenTokenizer` | an encoder that raises, for the 500 path during tokenization |

| Area | Tests |
|------|-------|
| Results | `test_generate_returns_the_requested_token_count`, `test_temperature_zero_is_deterministic`, `test_stream_sends_the_same_text_as_a_plain_request`, `test_stream_does_not_split_multibyte_characters` |
| Slot | `test_only_one_generation_runs_at_a_time`, `test_an_exhausted_server_answers_429_and_recovers`, `test_slots_are_released_after_a_stream`, `test_a_client_that_never_reads_the_stream_does_not_keep_its_slot` |
| Cancellation | `test_a_disconnected_plain_client_cancels_its_generation`, `test_shutdown_stops_a_running_generation` |
| Limits and validation | `test_a_body_larger_than_the_limit_is_rejected_before_parsing`, `test_invalid_requests_are_rejected` (10 bodies), `test_a_lone_surrogate_is_a_rejected_request`, `test_a_large_top_k_is_clamped_to_the_vocabulary`, `test_a_prompt_longer_than_the_context_is_rejected`, `test_a_request_that_would_outgrow_the_context_is_rejected` |
| Robustness | `test_the_event_loop_keeps_answering_while_a_prompt_is_encoded`, `test_a_failed_generation_answers_500_and_frees_its_slot`, `test_a_failed_stream_sends_an_error_event_and_frees_its_slot`, `test_an_encoding_failure_answers_500_and_frees_its_slot`, `test_a_request_cancelled_while_encoding_frees_its_slot` |
| Metrics | `test_health_and_prometheus_metrics` |
| Callback binding | `test_python_callback_errors_reach_the_caller`, `test_truthy_or_missing_callback_answers_continue`, `test_falsy_callback_answers_stop` |

The ten bodies of `test_invalid_requests_are_rejected` are: an empty object, an empty prompt, a prompt one character over the limit, `max_tokens` of 0, `max_tokens` one over the limit, a negative temperature, a negative `top_k`, a `top_k` of `2**70`, a temperature of `1e-40`, and the JSON string `"not json"`. Each must answer 422, leave `in_flight` at 0 and count one `rejected` request.

[`tests/test_debug.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_debug.cpp):

| Test | What it proves |
|------|----------------|
| `Explain.ShowsLayoutAndDtype` | shape, strides, offset, dtype, device, contiguity, byte count (`48 bytes` for a `[2, 3]` `float64` tensor) and reference count |
| `Explain.ShowsSharedStorageOfViews` | a view reports a non-zero offset, non-contiguity and two references |
| `Explain.ShowsAutogradState` | `requires_grad`, `leaf` and `grad_fn: MulBackward` |
| `GraphDot.HasOneNodePerOperationAndOneEdgePerInput` | four node types and exactly five edges for `sum(a * a + a)` |
| `GraphDot.ATensorWithoutGraphIsAnEmptyDigraph` | no edges without a graph |
| `BackwardTrace.PrintsEveryNodeWithShapeAndNorm`, `BackwardTrace.ReportsTheL2NormOfEachGradient`, `BackwardTrace.IsOffByDefault` | trace format, the norm is the L2 norm (`2.82843` for `[2, 2]`), and default |
| `AnomalyDetection.NamesTheNodeProducingNan`, `AnomalyDetection.IsOffByDefaultAndLeavesCleanGraphsAlone` | the error names `Sqrt` and `NaN`; off by default; no false positives |

The five edges of `sum(a * a + a)` are worth counting by hand: `SumBackward -> AddBackward` (1), `AddBackward -> MulBackward` (2), `AddBackward -> AccumulateGrad` for the `+ a` term (3), and two edges `MulBackward -> AccumulateGrad` for the two operands of `a * a` (4 and 5). In `ReportsTheL2NormOfEachGradient`, `sum(x * 2.0)` with `x = [3, 3]` sends `[2, 2]` to `AccumulateGrad`, and `sqrt(4 + 4) = 2.82843`.

The fuzzers and the sanitizer configurations are themselves tests: CI runs the whole GoogleTest suite under ASan with UBSan and under TSan, and fuzzes both parsers on every push.

## Build it yourself

1. Write `explain` and `autograd_graph_dot` in `src/debug.cpp`. Checkpoint: the `Explain` and `GraphDot` tests pass, and `dot -Tpng` renders the graph of a small expression.
2. Add the trace stream and the anomaly flag to the engine as function-local atomics, with `set_backward_trace` and `set_anomaly_detection`. Checkpoint: the `BackwardTrace` and `AnomalyDetection` tests pass.
3. Add `KAEL_ENABLE_SANITIZERS` and `KAEL_ENABLE_TSAN` to `kael_configure_target`, reject the combination, and define `KAEL_NO_CACHING_ALLOCATOR` when either is on. Checkpoint: `ctest` passes in an ASan build and in a TSan build.
4. Write `fuzz/fuzz_safetensors.cpp` and `fuzz/CMakeLists.txt`, and add a valid seed file. Checkpoint: `fuzz_safetensors -max_total_time=60` on the corpus finishes without a finding.
5. Add `.clang-tidy` and `.clang-format`. Checkpoint: `run-clang-tidy` on `src/` reports nothing.
6. Write `Generator`, `Metrics` and `create_app` with the plain JSON path only. Checkpoint: `test_generate_returns_the_requested_token_count`, the slot tests and `test_invalid_requests_are_rejected` pass.
7. Add streaming with the incremental decoder and the worker-owned slot. Checkpoint: `test_stream_sends_the_same_text_as_a_plain_request`, `test_stream_does_not_split_multibyte_characters` and `test_a_client_that_never_reads_the_stream_does_not_keep_its_slot` pass.
8. Add disconnect polling, the shutdown lifespan and the body size cap. Checkpoint: the whole `test_server.py` passes.
9. Write the Dockerfile and `.dockerignore`. Checkpoint: `docker build -f server/Dockerfile -t kael-server .` succeeds, the container reports healthy, and `curl` against `/generate` returns text.
10. Add the CI workflow. Checkpoint: every job is green on a pull request.

## Check your understanding

<details>
<summary>Why does the server run model.generate on a separate thread instead of awaiting it in the handler?</summary>

`generate` is a long C++ call that never yields to the event loop. Run on the loop thread, it would block every other request, including `/health`, until it finished. On a worker thread it runs in parallel, which works because the binding releases the GIL for the duration of the C++ call and reacquires it only for each `on_token` callback.

</details>

<details>
<summary>Why is the slot released by the worker thread and not when the HTTP response ends?</summary>

A streaming response ends only when the client reads it to the end or disconnects. A client that opens a stream and never reads it would keep the slot, and every other request would get 429 forever. The worker releases the slot in a `finally` block as soon as generation ends, whatever the client does.

</details>

<details>
<summary>The model emits the tokens C3, B1, E2, 82, AC one byte each. Which SSE events does the server send, and why not five?</summary>

It sends `{"text": "ñ"}`, `{"text": "€"}` and `[DONE]`. The incremental UTF-8 decoder buffers `C3` until `B1` completes `ñ`, and buffers `E2 82` until `AC` completes `€`. Calls that produce an empty string send no event. Decoding each byte separately would emit replacement characters instead.

</details>

<details>
<summary>Three generations take 0.3, 0.8 and 4 seconds. What are the values of the le="1" and le="5" buckets, and of kael_request_seconds_count?</summary>

The buckets are cumulative. `le="1"` counts 0.3 and 0.8, so it is 2. `le="5"` counts all three, so it is 3. `kael_request_seconds_count` is 3, the same as `le="+Inf"`.

</details>

<details>
<summary>sum(w * w).backward() with w = [3, 3] prints norm=8.48528 for AccumulateGrad. Where does that number come from?</summary>

`MulBackward` receives `[1, 1]` and returns `w * [1, 1] = [3, 3]` for each of its two operands. Both edges lead to the same `AccumulateGrad`, so the engine sums them into `[6, 6]` before running it. The L2 norm is `sqrt(36 + 36) = 8.48528`.

</details>

<details>
<summary>Why do sanitizer builds disable the caching allocator?</summary>

The caching allocator keeps freed blocks and reuses them without returning them to the system allocator. ASan tracks only system allocations, so a read through a dangling pointer would land in a block that ASan considers live and would not be reported. With `KAEL_NO_CACHING_ALLOCATOR`, every free reaches the system allocator, ASan poisons the block, and the next access through the old pointer is reported as a use after free.

</details>

## Further reading

- [libFuzzer documentation](https://llvm.org/docs/LibFuzzer.html): the fuzzing loop, corpus handling and every flag used in this chapter, such as `-max_total_time` and `-malloc_limit_mb`.
- [AddressSanitizer](https://clang.llvm.org/docs/AddressSanitizer.html), [UndefinedBehaviorSanitizer](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html) and [ThreadSanitizer](https://clang.llvm.org/docs/ThreadSanitizer.html) in the Clang manual: what each sanitizer detects, its overhead and how to read its reports.
- [clang-tidy check list](https://clang.llvm.org/extra/clang-tidy/checks/list.html): the description of every `bugprone-*` and `performance-*` check, including the ones kael disables.
- [Server-sent events in the HTML Living Standard](https://html.spec.whatwg.org/multipage/server-sent-events.html): the exact event stream format and how browsers consume it with `EventSource`.
- [Prometheus exposition formats](https://prometheus.io/docs/instrumenting/exposition_formats/) and [histograms and summaries](https://prometheus.io/docs/practices/histograms/): the text format that `/metrics` emits and how cumulative buckets become percentiles.
- [FastAPI lifespan events](https://fastapi.tiangolo.com/advanced/events/) and the Starlette documentation on `StreamingResponse`: the two framework features the server's shutdown and streaming rely on.
- [`server/app.py`](https://github.com/JetBerri/kael.cpp/blob/main/server/app.py): 340 lines that contain every mechanism described here, readable in one sitting.
- [Runtime data type](@/adr/0001-runtime-dtype.md) and [Tensor as a shared handle](@/adr/0002-tensor-as-handle.md): the two design decisions that `explain` makes visible, the dtype and the shared storage.

Back to the start: [00. Overview](@/book/00-overview.md)
