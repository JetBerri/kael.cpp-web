+++
title = "Loading GPT-2"
description = "The safetensors format with validated, atomic and memory-mapped I/O, the GPT-2 weight mapping, and a byte-level BPE tokenizer that matches tiktoken."
weight = 12
slug = "model-loading"

[extra]
chapter = "12"
kicker = "Chapter 12"
builds = "The safetensors format, the GPT-2 weight mapping and the BPE tokenizer"
source = "docs/12-model-loading.md"
tag = "chapter-13-polish"
+++

This chapter connects the transformer from [chapter 11](@/book/11-transformer.md) to the real GPT-2 small checkpoint published by OpenAI on Hugging Face. It covers three pieces: the safetensors file format and its reader and writer in `io/safetensors.cpp`, the translation of GPT-2 parameter names and layouts in `load_gpt2_weights`, and the byte-level BPE tokenizer in `text/bpe.cpp` that turns text into token ids and back. It ends with the end-to-end run: download the files, load them, generate text from C++ and from Python. This is the v1.0 milestone, "run GPT-2". The safetensors format and the GPT-2 weight mapping arrive with `chapter-12-transformer`. The BPE tokenizer, memory-mapped loading, atomic saves and `examples/gpt2_generate.cpp` arrive in phase 13, so the chapter is complete at `chapter-13-polish`, the same commit as `v1.0.0`.

| File | Role |
|------|------|
| [`include/kael/io/safetensors.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/io/safetensors.h) | `TensorMap`, `Metadata`, `LoadMode`, `SafetensorsFile`, `save_safetensors`, `read_safetensors`, `load_safetensors` |
| [`src/io/safetensors.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/io/safetensors.cpp) | Header parsing and validation, atomic save, memory-mapped load |
| [`src/nn/transformer.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/nn/transformer.cpp) | `translated_gpt2_name`, `stored_transposed`, `load_gpt2_weights` |
| [`include/kael/text/bpe.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/text/bpe.h) | `split_gpt2_words`, `BpeTokenizer` |
| [`src/text/bpe.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/text/bpe.cpp) | UTF-8 decoding, the pretokenizer, byte-to-unicode mapping, merges |
| [`src/text/unicode_classes.inc`](https://github.com/JetBerri/kael.cpp/blob/main/src/text/unicode_classes.inc) | Generated tables of Unicode letter and number ranges |
| [`scripts/generate/unicode_classes.py`](https://github.com/JetBerri/kael.cpp/blob/main/scripts/generate/unicode_classes.py) | Generates `unicode_classes.inc` from Python's `unicodedata` |
| [`scripts/download/gpt2.sh`](https://github.com/JetBerri/kael.cpp/blob/main/scripts/download/gpt2.sh) | Downloads the weights, vocabulary and merges and verifies their checksums |
| [`examples/gpt2_generate.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/examples/gpt2_generate.cpp) | Text generation from C++ |
| [`examples/gpt2_generate.py`](https://github.com/JetBerri/kael.cpp/blob/main/examples/gpt2_generate.py) | Text generation from Python, with a cache benchmark |
| [`fuzz/fuzz_safetensors.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/fuzz/fuzz_safetensors.cpp) | libFuzzer target for the reader ([chapter 13](@/book/13-serving-and-tooling.md)) |

## Prerequisites

- [02. Memory model](@/book/02-memory-model.md): `Storage`, `Storage::external`, `TensorImpl`, dtypes and element sizes.
- [03. Views](@/book/03-views.md): `T()` and `contiguous()`.
- [07. Neural networks and training](@/book/07-neural-networks.md): `Module::state_dict`, `load_state_dict`, `Linear` weight layout.
- [10. Python bindings](@/book/10-python-bindings.md): `kael.GPT`, `kael.BpeTokenizer` and `kael.load_safetensors` on the Python side.
- [11. Transformer](@/book/11-transformer.md): `GPT`, its parameter names and `generate`.

## Why this layer exists

Chapter 11 ends with a GPT that has the right architecture and random weights. Training GPT-2 small from scratch is far outside what a didactic library on one machine can do. OpenAI already did it, and the weights are public. Running GPT-2 needs three things the library does not have yet:

1. **A file format reader.** The weights are a 548 MB file of named tensors. kael needs to read it without trusting it, because a model file downloaded from the internet is untrusted input.
2. **A translation layer.** The checkpoint was written by a different codebase with different parameter names and, for some layers, a different matrix layout. Loading it into kael's `GPT` is a mapping problem.
3. **A tokenizer.** The model consumes integer ids, not text. The ids only mean something under the exact tokenizer GPT-2 was trained with. A tokenizer that is almost right produces ids the model has never seen in that order and degrades the output without any error.

What breaks without each piece:

- Without validation, a corrupt or malicious file can make the reader allocate exabytes, read past the end of a buffer, or construct an invalid `bool`. All of these are crashes or undefined behavior, not errors.
- Without atomic saves, a crash halfway through `save_safetensors` destroys the previous good checkpoint, and a process that has the old file memory-mapped can receive `SIGBUS`.
- Without the transpose, the shapes of four weights per block do not even match; with a wrong transpose on a square matrix (`attn.c_proj` is `[768, 768]`), the shapes match and the model silently computes garbage.
- Without a byte-exact tokenizer, the same prompt gives different ids than in Python, and every comparison with the reference implementation fails.

The mental model to hold on to:

```text
model.safetensors --read + validate--> TensorMap (GPT-2 names, GPT-2 layout)
                  --translate names, transpose Conv1D weights--> StateDict (kael names, kael layout)
                  --load_state_dict--> GPT parameters

"Hello, world!" --split into words--> "Hello" "," " world" "!"
                --bytes to symbols, apply merges--> ids [15496, 11, 995, 0]
                --GPT + generate--> more ids --decode--> bytes
```

Every step is deterministic and testable against a reference: the `safetensors` Python package for the file format, Hugging Face `transformers` for the logits, and `tiktoken` for the tokenizer.

## Concepts

### The safetensors format

**Safetensors** is a file format for named tensors designed by Hugging Face. It has no code in it, unlike Python pickle, which is what PyTorch's `torch.save` uses and which can run arbitrary code when loaded. Loading an untrusted safetensors file cannot execute anything, and its data section can be mapped into memory directly. A file has three parts:

```text
offset 0        8 bytes   N: header length, unsigned 64-bit little-endian
offset 8        N bytes   header: UTF-8 JSON object, may end with spaces
offset 8 + N    rest      data: raw tensor bytes, back to back
```

The JSON header maps each tensor name to a description:

```json
{
  "__metadata__": {"format": "pt"},
  "wte.weight": {"dtype": "F32", "shape": [50257, 768], "data_offsets": [287209472, 441598976]}
}
```

- `dtype` is a string. kael reads and writes `F32`, `F64`, `I64` and `BOOL`. Any other dtype string (`F16`, `BF16`, `I32` and so on) raises `TypeError`, because kael has no such dtype (see [ADR 0001](@/adr/0001-runtime-dtype.md)).
- `shape` is an array of non-negative integers. `[]` is a scalar.
- `data_offsets` is `[begin, end)`, in bytes, relative to the start of the data section, not to the start of the file.
- The optional `__metadata__` entry is a flat map from string to string. It is not a tensor.

Elements are stored in row-major (C) order, little-endian, with no padding between tensors. The format stores no strides: every tensor in the file is contiguous.

The `wte.weight` entry above is real. `[287209472, 441598976)` is 154,389,504 bytes, which is `50257 * 768 * 4`: the shape times the 4-byte element size. A reader checks exactly that equation for every entry.

### A file byte by byte

This program writes two small tensors:

```cpp
kael::io::save_safetensors("t.safetensors",
                           {{"b", kael::ones({2, 2})},
                            {"a", kael::arange(0, 3, 1, kael::DType::I64)}});
```

The resulting file, from `xxd t.safetensors`:

```text
00000000: 7000 0000 0000 0000 7b22 6122 3a7b 2264  p.......{"a":{"d
00000010: 6174 615f 6f66 6673 6574 7322 3a5b 302c  ata_offsets":[0,
00000020: 3234 5d2c 2264 7479 7065 223a 2249 3634  24],"dtype":"I64
00000030: 222c 2273 6861 7065 223a 5b33 5d7d 2c22  ","shape":[3]},"
00000040: 6222 3a7b 2264 6174 615f 6f66 6673 6574  b":{"data_offset
00000050: 7322 3a5b 3234 2c34 305d 2c22 6474 7970  s":[24,40],"dtyp
00000060: 6522 3a22 4633 3222 2c22 7368 6170 6522  e":"F32","shape"
00000070: 3a5b 322c 325d 7d7d 0000 0000 0000 0000  :[2,2]}}........
00000080: 0100 0000 0000 0000 0200 0000 0000 0000  ................
00000090: 0000 803f 0000 803f 0000 803f 0000 803f  ...?...?...?...?
```

Reading it:

| Bytes | Content |
|-------|---------|
| `0x00` to `0x07` | `70 00 00 00 00 00 00 00`: header length 0x70 = 112, little-endian |
| `0x08` to `0x77` | 112 bytes of JSON. Keys are sorted (`"a"` before `"b"`) |
| `0x78` to `0x8F` | tensor `a`: data offsets `[0, 24)`, three int64 values 0, 1, 2 |
| `0x90` to `0x9F` | tensor `b`: data offsets `[24, 40)`, four float32 values `0x3F800000` = 1.0 |

The data section starts at `8 + 112 = 120 = 0x78`. Tensor `a` occupies data bytes 0 to 24, so file bytes `0x78` to `0x8F`. The file is 160 bytes, `8 + 112 + 40`. The JSON happens to be exactly 112 bytes, a multiple of 8, so no padding spaces were needed. Little-endian means the least significant byte comes first: the int64 value 1 is `01 00 00 00 00 00 00 00`, and the float 1.0, `0x3F800000`, is stored as `00 00 80 3F`.

### Why some GPT-2 weights are transposed

GPT-2 was written in TensorFlow, and its Hugging Face port keeps the original layer type for its projections, `Conv1D`. Despite the name, `Conv1D` is a linear layer that stores its weight as `[in_features, out_features]` and computes `x @ W + b`. PyTorch's `nn.Linear` and kael's `Linear` store `[out_features, in_features]` and compute `x @ W^T + b`. Both compute the same function, but the stored matrices are transposes of each other.

A worked example with `in = 2`, `out = 3`:

```text
Conv1D stores W_hf [2, 3]         kael Linear stores W [3, 2] = W_hf^T
[ 1 2 3 ]                         [ 1 4 ]
[ 4 5 6 ]                         [ 2 5 ]
                                  [ 3 6 ]

x = [1, 1]
Conv1D:  x @ W_hf   = [1+4, 2+5, 3+6] = [5, 7, 9]
Linear:  x @ W^T    = [1+4, 2+5, 3+6] = [5, 7, 9]
```

The four `Conv1D` layers per block are `c_attn`, `attn.c_proj`, `c_fc` and `mlp.c_proj`. Only their `.weight` tensors need transposing. Biases are vectors, embeddings are `[rows, C]` tables in both libraries, and layer norm parameters are vectors.

### Byte pair encoding

GPT-2 does not see characters or words. It sees ids from a vocabulary of 50,257 tokens built by byte pair encoding over UTF-8 bytes. `BpeTokenizer` reproduces the GPT-2 tokenizer from the two files Hugging Face publishes, `vocab.json` and `merges.txt`.

**Byte pair encoding** (BPE) starts from single bytes and repeatedly merges the most frequent adjacent pair in a training corpus into a new symbol. The ordered list of merges is the learned model: merge 0 was the most frequent pair, merge 1 the next, and so on. To encode new text, apply the same merges in the same priority: at each step, among all adjacent pairs in the current word, merge the one with the lowest **rank** (its position in the list). Stop when no adjacent pair appears in the list.

The design sits between two extremes. A character or byte vocabulary never meets an unknown input, but sequences become long and the model must learn spelling. A word vocabulary gives short sequences but cannot represent a word it has not seen. BPE keeps frequent words as one token and splits rare words into known pieces, and because it starts from bytes, any input is representable.

Worked example with the merges `h e` (rank 0), `e l` (rank 1), `he l` (rank 2), from the `MergesFollowTheirRank` test:

```text
start       [h] [e] [l]
pairs       "h e" rank 0, "e l" rank 1      -> merge rank 0
            [he] [l]
pairs       "he l" rank 2                   -> merge rank 2
            [hel]
pairs       none                            -> stop, look up "hel"
```

With only `e l` (rank 0) and `h e` (rank 1), the result is `[h] [el]`: rank decides, not position (`ALowerRankWinsOverAnEarlierPosition`).

The same procedure on the real GPT-2 merges, for the word `" world"` (the leading space is part of the word, written `Ġ`, see below):

```text
[Ġ] [w] [o] [r] [l] [d]
rank    10  Ġ + w      -> [Ġw] [o] [r] [l] [d]
rank    17  o + r      -> [Ġw] [or] [l] [d]
rank    79  l + d      -> [Ġw] [or] [ld]
rank   220  Ġw + or    -> [Ġwor] [ld]
rank   739  Ġwor + ld  -> [Ġworld]                 id 995
```

And for `"tokenization"`, which ends as two tokens because no merge joins them:

```text
[t] [o] [k] [e] [n] [i] [z] [a] [t] [i] [o] [n]
rank     5  o + n        rank     9  a + t        rank    12  e + n
rank    39  i + on       rank    85  at + ion     rank   226  o + k
rank   272  i + z        rank  1378  iz + ation   rank  3977  ok + en
rank 29745  t + oken  -> [token] [ization]        ids 30001, 1634
```

The ranks come from running the merge loop of this chapter over `data/gpt2/merges.txt`. Early ranks are common letter pairs, later ranks are whole words and suffixes.

### Byte-to-unicode mapping

The vocabulary and merges files are text. Their tokens cannot contain raw bytes such as `0x00` or `0x0A`, and `merges.txt` uses a space as the separator between the two symbols of a pair, so a raw space cannot appear inside a symbol. GPT-2 solves this by assigning every byte a printable Unicode character:

- The 188 bytes that are already printable map to themselves: `0x21` to `0x7E`, `0xA1` to `0xAC`, `0xAE` to `0xFF`.
- The other 68 bytes (`0x00` to `0x20`, `0x7F` to `0xA0`, `0xAD`) map, in byte order, to code points 256, 257, and up.

```text
byte    0x00 ... 0x20 | 0x21 ... 0x7E | 0x7F ... 0xA0 | 0xA1 ... 0xAC | 0xAD | 0xAE ... 0xFF
symbol  256  ... 288  | 0x21 ... 0x7E | 289  ... 322  | 0xA1 ... 0xAC | 323  | 0xAE ... 0xFF
        33 bytes        94 bytes        34 bytes        12 bytes        1      82 bytes
```

Examples, checked against `BpeTokenizer::byte_symbols()`:

| Byte | Meaning | Symbol | Code point |
|------|---------|--------|------------|
| `0x00` | NUL | `Ā` | U+0100 (256) |
| `0x0A` | newline | `Ċ` | U+010A (266) |
| `0x20` | space | `Ġ` | U+0120 (288) |
| `0x21` | `!` | `!` | U+0021 |
| `0xAD` | soft hyphen | `Ń` | U+0143 (323) |
| `0xFF` | | `ÿ` | U+00FF |

That is why GPT-2 tokens with a leading space appear as `Ġthe` (id 262) in `vocab.json`, and a newline as `Ċ` (id 198).

### Pretokenization

BPE never merges across word boundaries. Before merging, GPT-2 splits text with this regular expression:

```text
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

The alternatives, tried in order at each position:

1. English contractions: `'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d` (lowercase only).
2. An optional single space followed by one or more letters (`\p{L}`).
3. An optional single space followed by one or more numbers (`\p{N}`).
4. An optional single space followed by one or more characters that are not whitespace, letters or numbers.
5. A run of whitespace not followed by a non-whitespace character. In practice: a whitespace run minus its last character when more text follows, so that the last space can start the next word.
6. Any remaining whitespace run.

The split serves two purposes. It keeps merges inside words, so `"dog."` and `"dog!"` share the token for `"dog"` instead of learning one token per punctuation variant. And it attaches a single leading space to the word that follows, so `" world"` is one token and the space between words costs nothing.

Examples from [`tests/test_bpe.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_bpe.cpp) and direct runs of the pattern:

```text
"Hello, world!"                  -> "Hello" "," " world" "!"
"don't we'll"                    -> "don" "'t" " we" "'ll"
"o'clock"                        -> "o" "'" "clock"
"abc123 45"                      -> "abc" "123" " 45"
"a  b"                           -> "a" " " " b"
"a\n\nb"                         -> "a" "\n" "\n" "b"
"a   "                           -> "a" "   "
" \n x"                          -> " \n" " x"
"Hello, world!  I'm here 2024"   -> "Hello" "," " world" "!" " " " I" "'m" " here" " 2024"
"año €5 日本語"                   -> "año" " €" "5" " 日本語"
```

Trace `"a  b"` (two spaces) through the alternatives at position 1:

```text
position 1: ' '   contraction? no   ' '+letter? next is ' ', no   ' '+number? no   ' '+other? no
                  whitespace run = 2 spaces, more text follows -> take 1: " "
position 2: ' '   ' '+letter? next is 'b', yes -> " b"
```

`"o'clock"` shows the contraction rule is lowercase and fixed: `'c` is not a contraction, so the apostrophe falls to the "other" alternative alone.

### Unicode classes

`\p{L}` and `\p{N}` are the Unicode general categories starting with L (letters: Lu, Ll, Lt, Lm, Lo) and N (numbers: Nd, Nl, No). `\s` is whitespace. Python's built-in `re` module does not support `\p{...}`; the reference implementation uses the third-party `regex` package. A C++ library has the same problem: `std::regex` has no Unicode property classes and no lookahead on Unicode code points.

## Implementation

### Writing a file

From [`src/io/safetensors.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/io/safetensors.cpp):

```cpp
std::int64_t position = 0;
std::vector<Tensor> payloads;
payloads.reserve(tensors.size());
for (const auto& [name, tensor] : tensors) {
    KAEL_CHECK_VALUE(name != kMetadataKey, "a tensor cannot be named " << kMetadataKey);
    payloads.push_back(tensor.detach().cpu().contiguous());
    const std::int64_t end = position + tensor.nbytes();
    header[name] = {{"dtype", dtype_name(tensor.dtype())},
                    {"shape", tensor.shape()},
                    {"data_offsets", {position, end}}};
    position = end;
}

std::string text = dump_header(header);
text.append((kHeaderAlignment - (text.size() % kHeaderAlignment)) % kHeaderAlignment, ' ');
const std::uint64_t header_size = text.size();
```

The public API, from [`include/kael/io/safetensors.h`](https://github.com/JetBerri/kael.cpp/blob/main/include/kael/io/safetensors.h):

```cpp
using TensorMap = std::map<std::string, Tensor>;
using Metadata = std::map<std::string, std::string>;

enum class LoadMode { Copy, Mapped };

struct SafetensorsFile {
    TensorMap tensors;
    Metadata metadata;
};

void save_safetensors(const std::filesystem::path& path, const TensorMap& tensors,
                      const Metadata& metadata = {});
[[nodiscard]] SafetensorsFile read_safetensors(const std::filesystem::path& path,
                                               LoadMode mode = LoadMode::Copy);
[[nodiscard]] TensorMap load_safetensors(const std::filesystem::path& path,
                                         LoadMode mode = LoadMode::Copy);
```

`load_safetensors` returns only the tensors; `read_safetensors` also returns the metadata. A `Module::state_dict()` is a `TensorMap`-compatible map, so `save_safetensors(path, model.state_dict())` is the whole checkpoint code of `char_gpt`.

- `TensorMap` is a `std::map<std::string, Tensor>`, so tensors are laid out in name order. The header is an `nlohmann::json` object, which also sorts its keys. The same input always produces the same bytes.
- Each tensor is detached, copied to the CPU and made contiguous. A view is saved by value: a transposed `[3, 2]` view of a `[2, 3]` tensor is written as a `[3, 2]` tensor in row-major order (`SavesViewsByValue`). The format has no strides, so this is the only possible choice.
- The JSON text is padded with spaces to a multiple of `kHeaderAlignment = 8`. Since the length prefix is also 8 bytes, the data section starts at a multiple of 8 in the file, and every tensor whose offset is a multiple of its element size is naturally aligned. That matters for mapped loading below. JSON allows trailing whitespace, so other readers accept the padding.
- A tensor named `__metadata__` is rejected, because it would collide with the metadata entry.
- The header is JSON, which must be valid UTF-8. `dump_header` catches the `nlohmann::json::type_error` that serializing an invalid string throws and raises `ValueError` ("tensor names and metadata must be valid UTF-8"), before any file is created (`SavingANameThatIsNotUtf8IsAValueError`).

All validation and the header serialization run before the first byte is written. A save that fails for a reason kael can detect in memory never touches the file system.

### Atomic save

Writing directly to the target file has two failure modes. A crash or a full disk halfway through leaves a truncated file where a good checkpoint used to be. And a process that has the old file memory-mapped sees its pages change or disappear when the file is truncated. kael avoids both by writing to a temporary file in the same directory and renaming it over the target. From [`src/io/safetensors.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/io/safetensors.cpp):

```cpp
// Mapped tensors may read from the file being replaced: truncating it would invalidate them.
std::filesystem::path temporary = path;
temporary +=
    ".tmp" + std::to_string(::getpid()) + "." + std::to_string(next_temporary_suffix());
TemporaryFileRemover remover{temporary};
std::ofstream file(temporary, std::ios::binary);
KAEL_CHECK(file.is_open(), "cannot write safetensors file " << temporary);
file.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
file.write(text.data(), static_cast<std::streamsize>(text.size()));
for (const Tensor& payload : payloads) {
    if (payload.nbytes() > 0) {
        file.write(reinterpret_cast<const char*>(payload.data_ptr_raw()),
                   static_cast<std::streamsize>(payload.nbytes()));
    }
}
file.close();
KAEL_CHECK(file.good(), "failed writing safetensors file " << temporary);
flush_to_disk(temporary);
std::filesystem::rename(temporary, path);
remover.path.clear();
flush_to_disk(path.parent_path().empty() ? std::filesystem::path(".") : path.parent_path());
```

The sequence as a diagram:

```text
write path.tmpPID.NNN
        |
        +-- any failure --> TemporaryFileRemover deletes path.tmpPID.NNN, path untouched
        |
        v
fsync path.tmpPID.NNN            data is on disk before the name changes
        |
        v
rename path.tmpPID.NNN -> path   atomic: readers see the old file or the new one
        |
        v
fsync the directory              the new directory entry survives a power loss
        |
        v
old inode lives on while any process still maps it
```

- The suffix is the process id followed by `next_temporary_suffix`: the steady clock count plus an atomic counter. The counter keeps two concurrent saves to the same path in one process apart, and the process id keeps saves from different processes apart.
- `TemporaryFileRemover` is an RAII guard. If anything throws before the rename, its destructor deletes the temporary file, ignoring errors. After a successful rename, clearing its path disarms it.
- On POSIX, `rename` replaces the directory entry atomically. A reader sees either the old file or the new one, never a mix. A process that mapped the old file keeps its pages, because the old inode is only freed when its last mapping goes away.
- `flush_to_disk` opens a path read-only and calls `fsync` on it. The temporary file is flushed before the rename, so a crash after the rename cannot leave a renamed but empty or partial file. The directory is flushed after the rename, so the new directory entry itself survives a power loss. A failed `fsync` throws `kael::Error`.
- The length prefix and the tensor bytes are written in host byte order. On the little-endian hosts kael targets (x86-64 and ARM64), that is the format's byte order.

The temporary file lives in the same directory as the target on purpose. `rename` is only atomic within one file system; a temporary file in `/tmp` could sit on a different file system, and `rename` would fail with `EXDEV`.

### Reading a file: validation

The reader treats the file as untrusted input. A corrupt or malicious file must produce a `kael::Error`, never a crash, an out-of-bounds read or a huge allocation. `read_safetensors` and its helpers check, in order:

| Check | Error |
|-------|-------|
| The file opens | `Error`: "cannot open safetensors file" |
| The file has at least 8 bytes | `ValueError`: "is too short" |
| `N <= 100,000,000` (`kMaxHeaderBytes`) and `N <= file_size - 8` | `ValueError`: "does not fit the file" |
| The header bytes can be read | `ValueError`: "is truncated" |
| The header parses as JSON and is an object | `ValueError`: "is not a JSON object" |
| `__metadata__`, if present, is an object of strings | `ValueError` |
| Each entry is an object with a string `dtype`, an array `shape` and an array `data_offsets` of exactly two elements | `ValueError` |
| `dtype` is one of `F32`, `F64`, `I64`, `BOOL` | `TypeError` |
| Every dimension and offset is a non-negative integer that fits in `int64` | `ValueError` |
| `begin <= end` | `ValueError`: "ends before it starts" |
| `end - begin == numel(shape) * element_size(dtype)`, computed with overflow checks | `ValueError` |
| Sorted by offset, the tensors tile the data section exactly: the first starts at 0, each starts where the previous ended, the last ends at the end of the file | `ValueError` |
| Every byte of a `BOOL` tensor is 0 or 1 | `ValueError` |

The framing part of `read_safetensors`:

```cpp
std::ifstream file(path, std::ios::binary | std::ios::ate);
KAEL_CHECK(file.is_open(), "cannot open safetensors file " << path);
const auto file_size = static_cast<std::uint64_t>(file.tellg());
KAEL_CHECK_VALUE(file_size >= sizeof(std::uint64_t),
                 "safetensors file " << path << " is too short");
file.seekg(0);

std::uint64_t header_size = 0;
file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
KAEL_CHECK_VALUE(header_size <= kMaxHeaderBytes &&
                     header_size <= file_size - sizeof(header_size),
                 "safetensors header of " << header_size << " bytes does not fit the file");
```

Notes on the less obvious checks:

- The header length is compared as `header_size <= file_size - sizeof(header_size)`, after checking that the file has at least 8 bytes. Writing it as `header_size + 8 <= file_size` would overflow for a length of `0xFFFFFFFFFFFFFFFF` and pass, because `0xFFFFFFFFFFFFFFFF + 8` wraps to 7 in unsigned 64-bit arithmetic (`HugeHeaderLengthDoesNotOverflow`).
- The 100 MB cap bounds the string allocated for the header before the JSON is parsed. The GPT-2 header is 14,283 bytes.
- JSON is parsed with `Json::parse(text, nullptr, false)`, which returns a discarded value instead of throwing `nlohmann` exceptions, so all failures surface as kael errors.
- `non_negative_integer` accepts both signed and unsigned JSON integers and rejects floats, strings, negatives and unsigned values above `INT64_MAX`.
- The size check uses `numel(entry.shape)` and `detail::checked_mul`, both of which throw on overflow. A shape such as `[9223372036854775807, 9223372036854775807]` is rejected instead of wrapping to a small number.
- `check_layout` sorts the entries by `(begin, end)` and walks them. This one pass rejects gaps, overlaps, offsets past the end and trailing bytes. Overlapping entries would let two tensors alias the same bytes; gaps would hide data. Only after this check does the reader allocate or seek.
- A `BOOL` byte other than 0 or 1 would be a valid `uint8_t` but an invalid `bool`, and reading it as `bool` is undefined behavior.

`check_layout` in full, because the single sorted walk is what carries most of the safety:

```cpp
void check_layout(std::vector<Entry>& entries, std::int64_t data_size) {
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        return a.begin != b.begin ? a.begin < b.begin : a.end < b.end;
    });
    std::int64_t position = 0;
    for (const Entry& entry : entries) {
        KAEL_CHECK_VALUE(entry.begin == position,
                         "safetensors entry " << entry.name << " starts at " << entry.begin
                                              << " but the previous one ended at " << position);
        position = entry.end;
    }
    KAEL_CHECK_VALUE(position == data_size, "safetensors tensors cover "
                                                << position << " bytes but the file has "
                                                << data_size);
}
```

Two headers from the `BadHeader` suite, traced through it, with a 16-byte data section:

```text
a [4, 8)  b [4, 16)          sorted: a [4, 8), b [4, 16)
                             a.begin 4 != position 0          -> gap at the start, rejected

a [0, 8)  b [4, 12)  c [12, 16)
                             a ok, position 8
                             b.begin 4 != position 8          -> overlap, rejected
```

The `BadHeader` parameterized suite in [`tests/test_safetensors.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_safetensors.cpp) feeds 26 corrupt headers, each followed by 16 data bytes, and expects `ValueError` for each.

The GPT-2 file shows why sorting is required. Its entries are not stored in name order: data offset 0 holds `h.3.attn.c_proj.bias`, followed by `h.2.mlp.c_fc.bias`. Only the offsets define the layout.

### The copying load

`LoadMode::Copy`, the default, allocates each tensor with `empty(shape, dtype)`, seeks to `8 + N + begin` and reads `end - begin` bytes. The resulting tensors own their memory and are independent of the file. Empty tensors are allocated but nothing is read for them.

### The memory-mapped load

`LoadMode::Mapped` maps the whole file into the address space and builds tensors whose storage points into the mapping. Nothing is read until a page is touched, and pages that are only read are shared with the operating system's file cache. Two processes that map the same checkpoint share one copy of it in RAM. From [`src/io/safetensors.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/io/safetensors.cpp):

```cpp
// Private pages: writing to a mapped tensor copies the page and never reaches the file.
std::shared_ptr<void> map_file(const std::filesystem::path& path, std::uint64_t expected_size) {
#if defined(_WIN32)
    (void)path;
    (void)expected_size;
    return nullptr;
#else
    const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    KAEL_CHECK(descriptor >= 0, "cannot open safetensors file " << path);
    struct stat status{};
    const bool same_size = ::fstat(descriptor, &status) == 0 &&
                           static_cast<std::uint64_t>(status.st_size) == expected_size;
    void* address = same_size ? ::mmap(nullptr, static_cast<std::size_t>(expected_size),
                                       PROT_READ | PROT_WRITE, MAP_PRIVATE, descriptor, 0)
                              : MAP_FAILED;
    ::close(descriptor);
    KAEL_CHECK(same_size, "safetensors file " << path << " changed while it was being read");
    if (address == MAP_FAILED) {
        return nullptr;
    }
    return {address, [expected_size](void* mapped) {
                ::munmap(mapped, static_cast<std::size_t>(expected_size));
            }};
#endif
}
```

- The header is validated first, through the ordinary stream, exactly as in copy mode. The mapping is created only for a file that passed every check.
- The file is reopened for the mapping. `fstat` confirms its size still matches the size used for validation; a file that changed in between is rejected.
- `MAP_PRIVATE` with `PROT_READ | PROT_WRITE` gives copy-on-write pages. Tensors are mutable in kael, so a mapped tensor must accept writes such as `fill_`. The first write to a page copies it privately, and the file on disk never changes (`WritingToAMappedTensorDoesNotChangeTheFile`). With `PROT_READ` alone, such a write would crash the process with a memory protection fault.
- The descriptor is closed right after `mmap`. A mapping stays valid without an open descriptor. `O_CLOEXEC` keeps the descriptor from leaking into a child process started in that window.
- If `mmap` fails, `map_file` returns `nullptr` and every tensor falls back to the copying path. On Windows, `map_file` returns `nullptr`, so the mapped mode always copies.

Each tensor is then built over the mapping. From [`src/io/safetensors.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/io/safetensors.cpp):

```cpp
Tensor tensor_over_mapping(const std::shared_ptr<void>& mapping, std::uint64_t position,
                           const Entry& entry) {
    auto impl = std::make_shared<TensorImpl>();
    auto* first = static_cast<std::byte*>(mapping.get()) + position;
    impl->storage =
        std::make_shared<Storage>(Storage::external(first, entry.end - entry.begin, mapping));
    impl->shape = entry.shape;
    impl->strides = contiguous_strides(entry.shape);
    impl->dtype = entry.dtype;
    return Tensor::from_impl(std::move(impl));
}
```

`Storage::external(std::byte* data, std::int64_t nbytes, std::shared_ptr<void> owner)` wraps memory that the storage does not allocate. Its third argument is an owner: a `shared_ptr` that the storage keeps alive. Every mapped tensor holds a copy of the same `shared_ptr<void>`, and its deleter calls `munmap`. This is the same mechanism [chapter 02](@/book/02-memory-model.md) uses for memory that comes from outside the allocator, and it is the C++ counterpart of NumPy's `np.memmap` or `torch.from_file`.

The decision per tensor, from `read_safetensors`:

```cpp
const auto position = static_cast<std::uint64_t>(data_start + entry.begin);
const auto alignment = static_cast<std::uint64_t>(element_size(entry.dtype));
const bool over_mapping = mapping && entry.end > entry.begin && position % alignment == 0;
```

A tensor is built over the mapping only when it is non-empty and its file position is a multiple of its element size. `mmap` returns a page-aligned address, so a file offset that is a multiple of 4 gives a 4-byte aligned `float*`. A misaligned tensor is copied instead (`MappedLoadCopiesMisalignedTensors`), because dereferencing a misaligned `double*` is undefined behavior and traps on some CPUs.

```text
mapping (one shared_ptr<void>, munmap on last release)
+--------+-----------------+----------------------+----------------------+
| N (8B) | header (N bytes)| tensor a bytes       | tensor b bytes       |
+--------+-----------------+----------------------+----------------------+
                           ^                      ^
                           Storage of a           Storage of b
                           (owner = mapping)      (owner = mapping)
```

A worked alignment check. For a file written by kael with `N = 112`, the data starts at 120. A float tensor at data offset 24 sits at file position 144: `144 % 4 == 0`, mapped. For the official GPT-2 file, `N = 14,283`, so the data starts at 14,291. The first tensor sits at file position 14,291: `14,291 % 4 == 3`, copied. Every other F32 tensor in that file has an offset that is a multiple of 4 from the data start, so all of them share the same misalignment, and the mapped load copies every tensor.

### Lifetime rules for mapped tensors

1. The mapping lives as long as any tensor, view or copy of a storage that came from it. Dropping the `TensorMap` does not unmap anything while one tensor is still referenced (`MappedTensorsOutliveTheMap`). That test even deletes the file and then reads the tensor.
2. Writes to mapped tensors are private to the process and are lost when the mapping goes away.
3. Do not truncate or rewrite the file in place while mapped tensors are alive. Touching a page past the new end of a truncated file raises `SIGBUS`. The Python docstring of `kael.load_safetensors(path, mmap=True)` repeats this rule.
4. Replacing the file with `save_safetensors` is safe: the rename leaves the old inode alive for the existing mapping (`SavingOverTheFileBehindAMappedTensorKeepsItValid`).

> [!WARNING]
> Rule 3 has no error path. A `SIGBUS` kills the process, and no `try` block catches it. Tools that overwrite files in place, such as `cp` over an existing file or a writer that opens with truncation, break every live mapping of that file.

### What is in the GPT-2 checkpoint

The Hugging Face `gpt2` checkpoint is a 548,105,171-byte safetensors file with 160 F32 tensors and the metadata `{"format": "pt"}`. The entries of the first block and the global ones:

```text
wte.weight                [50257, 768]
wpe.weight                [1024, 768]
h.0.ln_1.weight / .bias   [768]
h.0.attn.c_attn.weight    [768, 2304]     bias [2304]
h.0.attn.c_proj.weight    [768, 768]      bias [768]
h.0.attn.bias             [1, 1, 1024, 1024]
h.0.ln_2.weight / .bias   [768]
h.0.mlp.c_fc.weight       [768, 3072]     bias [3072]
h.0.mlp.c_proj.weight     [3072, 768]     bias [768]
ln_f.weight / .bias       [768]
```

The count: 13 entries per block times 12 blocks, plus `wte`, `wpe` and the two `ln_f` entries, gives 160. Without the 12 `attn.bias` buffers, 148 entries remain, which is exactly the number of parameters in kael's `state_dict()` for GPT-2 small. The `attn.bias` buffers are the causal mask stored as floats: 4 MiB each, 48 MiB of the file that kael ignores.

Compare `h.0.attn.c_attn.weight [768, 2304]` with kael's `blocks.0.attn.qkv.weight [2304, 768]`: same numbers, transposed.

### The name mapping

| GPT-2 name | kael name | Transposed |
|------------|-----------|------------|
| `wte.weight` | `token_embedding.weight` | no |
| `wpe.weight` | `position_embedding.weight` | no |
| `h.{i}.ln_1.{weight,bias}` | `blocks.{i}.ln1.{weight,bias}` | no |
| `h.{i}.attn.c_attn.weight` | `blocks.{i}.attn.qkv.weight` | yes |
| `h.{i}.attn.c_attn.bias` | `blocks.{i}.attn.qkv.bias` | no |
| `h.{i}.attn.c_proj.weight` | `blocks.{i}.attn.proj.weight` | yes |
| `h.{i}.attn.c_proj.bias` | `blocks.{i}.attn.proj.bias` | no |
| `h.{i}.ln_2.{weight,bias}` | `blocks.{i}.ln2.{weight,bias}` | no |
| `h.{i}.mlp.c_fc.weight` | `blocks.{i}.mlp.fc1.weight` | yes |
| `h.{i}.mlp.c_fc.bias` | `blocks.{i}.mlp.fc1.bias` | no |
| `h.{i}.mlp.c_proj.weight` | `blocks.{i}.mlp.fc2.weight` | yes |
| `h.{i}.mlp.c_proj.bias` | `blocks.{i}.mlp.fc2.bias` | no |
| `ln_f.{weight,bias}` | `ln_final.{weight,bias}` | no |
| `h.{i}.attn.bias` | none: the causal mask buffer, ignored | |
| `lm_head.weight` | none: tied to `wte`, ignored | |

A leading `transformer.` prefix, present in checkpoints saved from `GPT2LMHeadModel.state_dict()`, is stripped first. The translation is plain string replacement. From [`src/nn/transformer.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/nn/transformer.cpp):

```cpp
std::string translated_gpt2_name(std::string name) {
    const std::string prefix = "transformer.";
    if (name.rfind(prefix, 0) == 0) {
        name.erase(0, prefix.size());
    }
    const auto replace = [&](const std::string& from, const std::string& to) {
        const auto position = name.find(from);
        if (position != std::string::npos) {
            name.replace(position, from.size(), to);
        }
    };
    if (name.rfind("wte.", 0) == 0) {
        replace("wte.", "token_embedding.");
    } else if (name.rfind("wpe.", 0) == 0) {
        replace("wpe.", "position_embedding.");
    } else if (name.rfind("ln_f.", 0) == 0) {
        replace("ln_f.", "ln_final.");
    } else if (name.rfind("h.", 0) == 0) {
        name.replace(0, 2, "blocks.");
        replace(".ln_1.", ".ln1.");
        replace(".ln_2.", ".ln2.");
        replace(".attn.c_attn.", ".attn.qkv.");
        replace(".attn.c_proj.", ".attn.proj.");
        replace(".mlp.c_fc.", ".mlp.fc1.");
        replace(".mlp.c_proj.", ".mlp.fc2.");
    }
    return name;
}
```

`name.rfind(prefix, 0) == 0` is the C++17 idiom for "starts with" (C++20 also has `starts_with`). Two traces:

```text
transformer.h.7.mlp.c_proj.weight
  strip "transformer."       h.7.mlp.c_proj.weight
  "h." -> "blocks."          blocks.7.mlp.c_proj.weight
  ".mlp.c_proj." -> ...      blocks.7.mlp.fc2.weight          stored_transposed: yes

h.0.attn.bias
  "h." -> "blocks."          blocks.0.attn.bias               not a parameter: dropped
```

The block-level replacements match segments with dots on both sides, so `.attn.c_proj.` and `.mlp.c_proj.` cannot be confused.

`stored_transposed` decides on the translated name:

```cpp
bool stored_transposed(const std::string& name) {
    const bool conv1d = name.find(".attn.qkv.") != std::string::npos ||
                        name.find(".attn.proj.") != std::string::npos ||
                        name.find(".mlp.fc1.") != std::string::npos ||
                        name.find(".mlp.fc2.") != std::string::npos;
    return conv1d && name.ends_with(".weight");
}
```

### The loader

From [`src/nn/transformer.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/nn/transformer.cpp):

```cpp
void load_gpt2_weights(GPT& model, const std::filesystem::path& path) {
    const io::TensorMap stored = io::load_safetensors(path, io::LoadMode::Mapped);
    const StateDict expected = model.state_dict();
    StateDict translated;
    for (const auto& [stored_name, tensor] : stored) {
        const std::string name = translated_gpt2_name(stored_name);
        if (expected.contains(name)) {
            translated.emplace(name, stored_transposed(name) ? tensor.T().contiguous() : tensor);
        }
    }
    model.load_state_dict(translated);
}
```

- Entries whose translated name is not a model parameter are dropped. That removes `attn.bias`, `lm_head.weight` and anything else the model does not have, so checkpoints with extra buffers load without special cases.
- `tensor.T().contiguous()` materializes the transpose. The model's `Linear` could use the stored layout as a strided view, but `load_state_dict` copies into existing parameters anyway, and a contiguous source keeps that copy a straight memory copy.
- `Module::load_state_dict` (chapter 07) first checks that every model parameter is present with the same shape, then copies all of them under `NoGradGuard`. A missing name throws `Error` ("state dict is missing parameter ..."), a shape mismatch throws `ValueError`. Because all checks run before the first copy, a failed load leaves the model unchanged.
- The model's parameters receive copies. When `load_gpt2_weights` returns, `stored` and `translated` are destroyed, and so is the mapping. The model never depends on the file afterwards.
- The model's `GptConfig` must match the checkpoint. Loading GPT-2 small into a model with another width fails on the first shape check.

`load_state_dict` itself:

```cpp
void Module::load_state_dict(const StateDict& state) {
    std::vector<NamedTensor> all;
    collect_parameters("", all);
    for (const auto& [name, parameter] : all) {
        const auto found = state.find(name);
        KAEL_CHECK(found != state.end(), "state dict is missing parameter " << name);
        KAEL_CHECK_VALUE(found->second.shape() == parameter.shape(),
                         "parameter " << name << " has shape " << to_string(parameter.shape())
                                      << " but the state dict has "
                                      << to_string(found->second.shape()));
    }
    const autograd::NoGradGuard guard;
    for (auto& [name, parameter] : all) {
        parameter.copy_(state.at(name));
    }
}
```

The check-then-copy split is the same idea as a database transaction: validate everything, then commit. PyTorch's `load_state_dict(strict=True)` reports missing and unexpected keys; kael's version reports missing ones and ignores extras, which is why the filter in `load_gpt2_weights` is a choice of clarity rather than a requirement.

The official `gpt2` file has a 14,283-byte header, so its data section starts at byte 14,291. That offset is not a multiple of 4, so no F32 tensor in it is 4-byte aligned in the file, and the mapped load copies every tensor. Files written by `save_safetensors` pad the header to 8 bytes and are mapped without copies.

A rough memory budget for loading GPT-2 small in F32, computed from the shapes above:

```text
stored     every tensor copied from the misaligned file          ~548 MB
translated 48 transposed Conv1D weights (12 x 7,077,888 floats)  ~340 MB
model      124,439,808 parameters                                ~498 MB
```

The peak is the sum, around 1.4 GB, until `load_gpt2_weights` returns and the first two are released.

### The Unicode class tables

kael stores `\p{L}` and `\p{N}` as sorted tables of inclusive code point ranges, generated from Python's `unicodedata`. From [`scripts/generate/unicode_classes.py`](https://github.com/JetBerri/kael.cpp/blob/main/scripts/generate/unicode_classes.py):

```python
def ranges_of(category_prefix):
    ranges = []
    for code_point in range(sys.maxunicode + 1):
        if not unicodedata.category(chr(code_point)).startswith(category_prefix):
            continue
        if ranges and ranges[-1][1] == code_point - 1:
            ranges[-1][1] = code_point
        else:
            ranges.append([code_point, code_point])
    return ranges
```

The script writes `src/text/unicode_classes.inc` with two arrays, `kLetters` and `kNumbers`, and a first line recording the Unicode version of the Python that generated it:

```cpp
// Generated by scripts/generate/unicode_classes.py from Unicode 15.0.0.
inline constexpr CodePointRange kLetters[] = {
    {0x41, 0x5A},
    {0x61, 0x7A},
    {0xAA, 0xAA},
```

`bpe.cpp` includes the file inside its anonymous namespace, right after defining `CodePointRange`. Whitespace (`\s`) is a short hand-written table, `kWhitespace`, with the ten ranges of the Unicode `White_Space` property: `0x9` to `0xD`, `0x20`, `0x85`, `0xA0`, `0x1680`, `0x2000` to `0x200A`, `0x2028` to `0x2029`, `0x202F`, `0x205F` and `0x3000`.

Membership is a binary search. From [`src/text/bpe.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/text/bpe.cpp):

```cpp
bool contains(std::span<const CodePointRange> ranges, char32_t code_point) {
    const auto* found = std::upper_bound(
        ranges.data(), ranges.data() + ranges.size(), code_point,
        [](char32_t value, const CodePointRange& range) { return value < range.first; });
    return found != ranges.data() && code_point <= (found - 1)->last;
}
```

`upper_bound` finds the first range that starts after the code point. The only candidate is the range before it, which contains the code point if its `last` is not smaller. A trace with the first three letter ranges and the code point `0x62` (`b`):

```text
ranges   {0x41, 0x5A}  {0x61, 0x7A}  {0xAA, 0xAA}  ...
first range starting after 0x62: {0xAA, 0xAA}
candidate: the range before it, {0x61, 0x7A}; 0x62 <= 0x7A -> letter

code point 0x5F (_): first range starting after it is {0x61, 0x7A}
candidate {0x41, 0x5A}; 0x5F > 0x5A -> not a letter
```

Generating the table at build time instead of at runtime keeps the library free of ICU and gives the same result on every platform.

### UTF-8 decoding and the word splitter

`decode_utf8` reads one code point at a byte position and returns it with its length in bytes. It rejects overlong encodings, surrogates, values above U+10FFFF and truncated sequences by returning U+FFFD with a length of 1. A length of 1 for every invalid case means the decoder always makes progress, one byte at a time, through any garbage.

`WordSplitter` implements the pretokenization pattern by hand instead of pulling in a regex library with Unicode property support and lookahead. Its constructor decodes the whole text into code points and records the byte offset of each one. `word_end` tries the alternatives in the same order as the regex. From [`src/text/bpe.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/text/bpe.cpp):

```cpp
[[nodiscard]] std::size_t word_end(std::size_t begin) const {
    for (const std::size_t end :
         {contraction_end(begin), class_run_end(begin, is_letter),
          class_run_end(begin, is_number), class_run_end(begin, is_other)}) {
        if (end != begin) {
            return end;
        }
    }
    return whitespace_end(begin);
}
```

Each helper returns `begin` when its alternative does not match, so "first alternative that moved" is "first alternative that matched". `class_run_end` implements the optional leading space: it skips one `U' '` (only the ASCII space, as in the regex) and then requires at least one code point of the class.

`whitespace_end` implements alternatives 5 and 6 together:

```cpp
[[nodiscard]] std::size_t whitespace_end(std::size_t begin) const {
    std::size_t end = begin;
    while (end < points_.size() && is_space(points_[end])) {
        ++end;
    }
    if (end == points_.size() || end - begin == 1) {
        return end;
    }
    return end - 1;
}
```

Three cases: a run at the end of the text is taken whole (`"a   "`), a single whitespace character is taken alone (`"\n"` before `"b"`), and a longer run followed by text gives back its last character so the next word can start with it (`" \n x"` gives `" \n"` and `" x"`).

The splitter cuts words by byte offsets, not by re-encoding code points. U+FFFD is neither a letter, a number nor whitespace, so an invalid byte joins an "other" run, and because the cut is by offsets, the original invalid byte is kept, not replaced by the three-byte encoding of U+FFFD (`KeepsEveryByteOfInvalidUtf8`).

### The byte-to-unicode table in code

From [`src/text/bpe.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/text/bpe.cpp):

```cpp
std::array<char32_t, 256> make_symbol_of_byte() {
    std::array<char32_t, 256> symbols{};
    char32_t next_unprintable = kFirstUnprintableSymbol;
    for (unsigned byte = 0; byte < 256; ++byte) {
        const bool printable = (byte >= 0x21 && byte <= 0x7E) || (byte >= 0xA1 && byte <= 0xAC) ||
                               (byte >= 0xAE && byte <= 0xFF);
        symbols[byte] = printable ? static_cast<char32_t>(byte) : next_unprintable++;
    }
    return symbols;
}
```

`kFirstUnprintableSymbol` is 256. `BpeTokenizer::byte_symbols()` returns the 256 symbols UTF-8 encoded, for building the initial symbols of a word. An internal function, `byte_of_symbol()`, holds the reverse map from code point to byte; the constructor uses it once per vocabulary token. The highest symbol is U+0143, so `encode_utf8` only needs the one-, two- and three-byte forms.

### Loading the vocabulary and merges

The `BpeTokenizer` constructor takes the two paths and reads `vocab.json`, a JSON object from token string to id:

- Each id must be a non-negative integer, otherwise `ValueError`.
- `ids_` maps the token string to its id, for encoding.
- `bytes_of_id_` maps the id to the raw bytes the token stands for, computed by `bytes_of_token`: decode the token's UTF-8 code points and map each one back through `byte_of_symbol()`. A token containing a code point that is not one of the 256 byte symbols is not byte-level BPE and throws `ValueError`.

Then it reads `merges.txt`:

```cpp
std::istringstream lines(read_file(merges));
std::string line;
std::int64_t rank = 0;
while (std::getline(lines, line)) {
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    if (line.empty() || line.starts_with("#version")) {
        continue;
    }
    const std::size_t space = line.find(' ');
    KAEL_CHECK_VALUE(space != std::string::npos && space > 0 && space + 1 < line.size() &&
                         line.find(' ', space + 1) == std::string::npos,
                     "merges line '" << line << "' is not a pair of symbols");
    merge_ranks_.emplace(line, rank++);
}
```

Each non-empty line that does not start with `#version` must hold exactly two symbols separated by one space. A trailing `\r` is removed, so files with Windows line endings work. The line itself, such as `"Ġ t"`, is the key in `merge_ranks_`, and its rank is its position among the merge lines. Using the whole line as the key avoids a pair type and a custom hash: looking up a pair means building `left + " " + right` with `merge_key`. The GPT-2 file has 50,000 merges after its `#version: 0.2` line; the first five are `Ġ t`, `Ġ a`, `h e`, `i n` and `r e`.

Special tokens get no special treatment. `<|endoftext|>` (id 50256) is a vocabulary entry made of printable ASCII, so `decode({50256})` returns the text `<|endoftext|>`, and encoding that text produces ordinary tokens for `<`, `|` and so on. This matches `tiktoken`'s `encode_ordinary`.

### Encode and decode

`encode` splits the text into words with `split_gpt2_words` and runs BPE on each. From [`src/text/bpe.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/text/bpe.cpp), the merge loop of `append_word_ids`:

```cpp
while (symbols.size() > 1) {
    std::int64_t best_rank = std::numeric_limits<std::int64_t>::max();
    std::size_t best = symbols.size();
    for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
        const auto found = merge_ranks_.find(merge_key(symbols[i], symbols[i + 1]));
        if (found != merge_ranks_.end() && found->second < best_rank) {
            best_rank = found->second;
            best = i;
        }
    }
    if (best == symbols.size()) {
        break;
    }
    const std::string left = symbols[best];
    const std::string right = symbols[best + 1];
    std::vector<std::string> merged;
    merged.reserve(symbols.size());
    for (std::size_t i = 0; i < symbols.size(); ++i) {
        if (i + 1 < symbols.size() && symbols[i] == left && symbols[i + 1] == right) {
            merged.push_back(left + right);
            ++i;
        } else {
            merged.push_back(std::move(symbols[i]));
        }
    }
    symbols = std::move(merged);
}
```

- The initial symbols are the byte symbols of the word's bytes, so the space in `" world"` becomes `Ġ`.
- Each round finds the adjacent pair with the lowest rank and merges every non-overlapping occurrence of it, scanning from left to right. This matches the reference GPT-2 implementation. For `"aaa"` with a merge `a a`, the result is `[aa] [a]`, not `[a] [aa]`.
- When no pair is in the merge table, each remaining symbol is looked up in `ids_`. A symbol that is not in the vocabulary throws `ValueError` (`RejectsTextOutsideItsVocabulary`). With the real GPT-2 files this cannot happen, because every single byte symbol is a token.

The cost per word is quadratic in its length in the worst case, which is acceptable because pretokenization keeps words short. The reference implementation caches results per word; kael does not.

`decode` concatenates `bytes_of_id_` for each id and throws `IndexError` for an unknown id. The result is a byte string, not necessarily valid UTF-8: a multi-byte character can be split across two tokens, and decoding one token alone returns half a character. The real vocabulary shows it: `"日本"` encodes to four tokens whose bytes are `e6 97`, `a5`, `e6 9c` and `ac`. Neither character survives decoding its tokens one at a time. The Python binding returns `bytes` for this reason, and chapter 13 shows how the server decodes a stream incrementally.

The full round trip for a prompt used in the tests:

```text
"Hello, my name is"
split    "Hello"  ","  " my"  " name"  " is"
ids      15496    11   616    1438     318
decode   b"Hello" b"," b" my" b" name" b" is"   -> "Hello, my name is"
```

## End to end

### 1. Download

From [`scripts/download/gpt2.sh`](https://github.com/JetBerri/kael.cpp/blob/main/scripts/download/gpt2.sh):

```bash
directory="${1:-data/gpt2}"
base="https://huggingface.co/gpt2/resolve/main"

fetch "$base/model.safetensors" "$directory/model.safetensors" \
    248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707
fetch "$base/vocab.json" "$directory/vocab.json" \
    196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783
fetch "$base/merges.txt" "$directory/merges.txt" \
    1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5
```

`fetch` comes from [`scripts/download/common.sh`](https://github.com/JetBerri/kael.cpp/blob/main/scripts/download/common.sh). It keeps a file that already matches its SHA-256 checksum, so a second run downloads nothing. Otherwise it downloads to a `.partial` file with `curl --continue-at -`, which resumes an interrupted download of the 548 MB weights file. It moves the file into place only after its checksum matches. A mismatch deletes the partial file and fails, so a corrupt or truncated download never reaches the loaders. `--fail` turns an HTTP error into a nonzero exit, and `set -euo pipefail` at the top of the script stops it there. `scripts/download/gpt2.sh --help` prints the usage.

The script does not download the checkpoint's `config.json`. kael's loaders do not need it: the configuration comes from `GptConfig::gpt2_small()`. The Python test `test_real_gpt2_logits_match_huggingface` builds its Hugging Face reference model from `transformers.GPT2Config()`, whose defaults are GPT-2 small, so it does not need the file either.

### 2. Generate from C++

`examples/gpt2_generate.cpp` builds with `-DKAEL_BUILD_EXAMPLES=ON`. Its `run` function is the whole pipeline:

```cpp
const kael::text::BpeTokenizer tokenizer(options.directory / "vocab.json",
                                         options.directory / "merges.txt");
kael::nn::GPT model(kael::nn::GptConfig::gpt2_small());
kael::nn::load_gpt2_weights(model, options.directory / "model.safetensors");
model.to(options.device);
model.eval();

const std::vector<std::int64_t> ids = tokenizer.encode(options.prompt);
const auto length = static_cast<std::int64_t>(ids.size());
const kael::Tensor prompt =
    kael::from_vector<std::int64_t>(ids, {1, length}).to(options.device);

kael::manual_seed(options.seed);
const auto started = std::chrono::steady_clock::now();
const kael::Tensor output = kael::nn::generate(model, prompt,
                                               {.max_new_tokens = options.tokens,
                                                .temperature = options.temperature,
                                                .top_k = options.top_k});
```

The order matters. Weights are loaded on the CPU, then the whole model moves to the device once. The seed is set right before `generate`, so the random draws of weight initialization in the `GPT` constructor do not shift the sampling sequence. The example then decodes `output.select(0, 0)`, prints the text to standard output and the generation rate in tokens per second to standard error, so the text can be piped without the timing line.

| Argument | Default |
|----------|---------|
| positional prompt | `"The meaning of life is"` |
| `--tokens` | 60 |
| `--temperature` | 0.8 |
| `--top-k` | 40 |
| `--seed` | 42 |
| `--data` | `data/gpt2` |
| `--device` | `cpu` (`cuda` selects the CUDA device) |

`--help` prints the usage. An unknown option, an option without its value or an unknown device name is an error, not a silent fallback to the defaults. The device name goes through `kael::parse_device`.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKAEL_BUILD_EXAMPLES=ON
cmake --build build --target gpt2_generate
build/examples/gpt2_generate --tokens 20
```

```text
The meaning of life is not only about the meaning of life, but also the meaning of life itself." "It is a
```

That output came from one macOS run with the default seed. The exact text depends on the standard library's random distributions and the floating point summation order, so another platform may print different text with the same seed.

### 3. Generate from Python

`examples/gpt2_generate.py` uses the Python bindings for the model and `tiktoken` for the tokenizer:

```python
encoding = tiktoken.get_encoding("gpt2")
model = kael.GPT()
model.load_gpt2(arguments.weights)
model.to(arguments.device)
model.eval()
prompt = kael.tensor([encoding.encode(arguments.prompt)]).to(arguments.device)
```

`kael.GPT()` with no arguments is GPT-2 small, and `load_gpt2` calls `load_gpt2_weights`. Its arguments mirror the C++ example, except that `--weights` (default `data/gpt2/model.safetensors`) replaces `--data`. `argparse` restricts `--device` to `cpu` or `cuda`, and `--help` describes every option. With `--benchmark`, it runs greedy generation (temperature 0) twice, with and without the KV cache, calls `kael.cuda_synchronize()` before stopping the clock, and prints the rate of each:

```bash
python examples/gpt2_generate.py --benchmark --tokens 60
```

To use kael's own tokenizer from Python instead of `tiktoken`, create `kael.BpeTokenizer("data/gpt2/vocab.json", "data/gpt2/merges.txt")`. Its `encode` returns a list of ids, its `decode` returns `bytes`, and `vocabulary_size` is 50257.

## Edge cases and pitfalls

- **Header length overflow.** A length of `2^64 - 1` must not wrap. The comparison is ordered to avoid the addition (`HugeHeaderLengthDoesNotOverflow`).
- **Size overflow in shapes.** `numel` and `checked_mul` throw instead of wrapping, so a header cannot claim a huge tensor that fits in 16 bytes.
- **Overlapping or gapped offsets.** `check_layout` rejects both before any allocation.
- **Invalid booleans.** Bytes other than 0 and 1 in a `BOOL` tensor are rejected (`InvalidBoolBytesAreRejected`).
- **Unsupported dtypes.** `F16` and `BF16` checkpoints throw `TypeError`. Convert them to F32 first, for example with the `safetensors` and `torch` packages in Python.
- **Empty tensors.** A tensor with zero elements has `begin == end`. It is never placed over the mapping and no bytes are read or written for it.
- **Truncating a mapped file.** Causes `SIGBUS` on the next page fault. Replace files with `save_safetensors` or another write-and-rename tool.
- **Misaligned files.** Files from other writers may not pad their header to 8 bytes. Mapped loading then copies misaligned tensors, and for the official GPT-2 file that means every tensor.
- **Big-endian hosts.** The reader and writer do not swap bytes. Safetensors is little-endian, so kael's files are only portable between little-endian hosts.
- **Rename failures.** `std::filesystem::rename` reports failure by throwing `std::filesystem::filesystem_error`, not a `kael::Error`. The remover still deletes the temporary file, and the target is untouched.
- **Wrong GELU or config.** Loading GPT-2 weights into a model with `tanh_gelu = false` succeeds, but the logits drift from the reference. A different `dimension` or `layers` fails with a shape or missing-parameter error.
- **A square matrix loaded without the transpose.** `attn.c_proj.weight` is `[768, 768]`, so skipping its transpose passes every shape check. Only a comparison of logits against a reference catches it. That is the reason `Conv1DWeightsAreTransposed` compares every parameter bit for bit.
- **Text that is not UTF-8.** The tokenizer handles it: invalid bytes become their own byte symbols and decode back to the same bytes (`RoundTripsArbitraryBytes`).
- **Partial characters in decode.** Decoding a prefix of a token sequence can end in the middle of a UTF-8 character. Decode incrementally or decode the whole sequence.
- **Empty prompt.** `encode("")` returns no ids, and `generate` rejects a prompt with zero tokens with `ValueError`.
- **Unicode version drift.** The class tables are frozen at Unicode 15.0.0. Rerunning `scripts/generate/unicode_classes.py` with another Python can change the tables and therefore the tokenization of rare characters. Check the first line of the generated file after regenerating.

## How it is tested

[`tests/test_safetensors.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_safetensors.cpp), suite `Safetensors` plus the parameterized `Corrupt/BadHeader`:

| Test | What it proves |
|------|----------------|
| `RoundTripEveryDtype`, `MetadataRoundTrip`, `StateDictRoundTrip` | save then load preserves values, dtypes, shapes and metadata, including a scalar and a `[0, 3]` tensor |
| `SavesViewsByValue` | non-contiguous views are written in logical order |
| `HeaderFormatMatchesTheSpecification` | the header contains `"dtype":"F32"`, `"shape":[2,2]`, `"data_offsets":[0,16]` and `8 + N` is a multiple of 8 |
| `ReadsAFileWithPadding` | trailing spaces in the header are accepted |
| `MissingFile`, `TruncatedLengthPrefix`, `HeaderLongerThanFile`, `HugeHeaderLengthDoesNotOverflow` | framing errors |
| `BadHeader.IsRejected` | 26 malformed headers, each raises `ValueError` |
| `InvalidBoolBytesAreRejected`, `UnsupportedDtypeIsATypeError`, `NamesCannotBeTheMetadataKey`, `SavingANameThatIsNotUtf8IsAValueError` | content errors |
| `MappedLoadMatchesTheCopyingLoad`, `MappedLoadRejectsCorruptFilesLikeTheCopyingLoad` | the two modes agree |
| `MappedTensorsOutliveTheMap`, `WritingToAMappedTensorDoesNotChangeTheFile` | lifetime and copy-on-write |
| `MappedLoadSharesTheFilePagesWhenAligned`, `MappedLoadCopiesMisalignedTensors` | aligned tensors point into one mapping 4 floats apart; misaligned ones are copied and aligned |
| `SavingOverTheFileBehindAMappedTensorKeepsItValid`, `AFailedSaveLeavesTheExistingFileIntact`, `SavingLeavesNoTemporaryFilesBehind` | atomic save |

The 26 `BadHeader` cases cover, among others: text that is not JSON, JSON that is not an object, a missing `dtype`, `shape` or `data_offsets`, a numeric `dtype`, negative, fractional and string dimensions, one or three offsets, negative and reversed offsets, a size that does not match the shape, gaps, overlaps, a shape whose element count overflows, an end offset of `2^64 - 1`, and malformed metadata.

`AFailedSaveLeavesTheExistingFileIntact` triggers the failure with a tensor named `__metadata__`, which is rejected before the temporary file is created. It proves that a rejected save does not touch the target, but it does not exercise a failure after the temporary file is written.

[`tests/test_transformer.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_transformer.cpp), suite `Gpt2Weights`:

- `MissingEntriesAreReported`: a file with only `wte.weight` throws.
- `Conv1DWeightsAreTransposed`: builds a fake Hugging Face file from a kael model's own state dict, with `transformer.h.*` prefixes, transposed Conv1D weights, an `lm_head.weight` and an `attn.bias` buffer, loads it into a fresh model and checks every parameter is bit-identical to the source.

[`tests/test_bpe.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/tests/test_bpe.cpp), suites `Gpt2Split` and `Bpe`: the splitting examples above, merge ranking, a round trip of all 256 byte values, the `Ġ` leading space, special tokens decoded as text, and rejection of unknown ids, unknown symbols, corrupt files (an id that is a string, a negative id, a merges line with one symbol, a token outside the byte symbols) and missing files. The `Bpe` fixture writes tiny `vocab.json` and `merges.txt` files, so these tests need no download.

[`python/tests/test_tokenizer.py`](https://github.com/JetBerri/kael.cpp/blob/main/python/tests/test_tokenizer.py) (skipped unless `tiktoken` is installed and `data/gpt2` holds the vocabulary) compares `kael.BpeTokenizer` with `tiktoken.get_encoding("gpt2").encode_ordinary` on fixed texts with contractions, mixed whitespace, several scripts and emoji, and on 2,000 random strings generated by Hypothesis. For the fixed texts it also checks that decoding the ids gives back the original text.

[`python/tests/test_gpt.py`](https://github.com/JetBerri/kael.cpp/blob/main/python/tests/test_gpt.py):

- `test_safetensors_written_by_python_load_in_kael` and `test_safetensors_written_by_kael_load_in_python` check interoperability with the reference `safetensors` package in both directions, with F32, F64, I64 and bool tensors. The first also reads the metadata back with `kael.read_safetensors_metadata`; the second includes an empty tensor.
- `test_mapped_safetensors_match_and_stay_private` checks the mapped mode from Python.
- `test_real_gpt2_logits_match_huggingface` (skipped without the downloaded weights) loads the real checkpoint, checks 124,439,808 parameters, and compares the last-position logits for "Hello, my name is" with Hugging Face `transformers` to within `1e-3`.

[`fuzz/fuzz_safetensors.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/fuzz/fuzz_safetensors.cpp) writes each fuzzer input to a file, runs `read_safetensors` on it in both load modes, touches the first and last byte of every tensor, and accepts `kael::Error` as the only allowed failure (chapter 13).

## Build it yourself

1. Write `save_safetensors` without the temporary file: build the header, pad it, write the length, header and payloads. Checkpoint: `HeaderFormatMatchesTheSpecification` passes and `xxd` shows the layout from this chapter.
2. Write `read_safetensors` in copy mode with every check in the validation table. Checkpoint: all `BadHeader` cases, `HugeHeaderLengthDoesNotOverflow` and `RoundTripEveryDtype` pass.
3. Make the save atomic with the temporary file, the RAII remover and `rename`. Checkpoint: `AFailedSaveLeavesTheExistingFileIntact` and `SavingLeavesNoTemporaryFilesBehind` pass.
4. Add `LoadMode::Mapped` with `map_file`, `Storage::external` and the alignment fallback. Checkpoint: the six mapped tests pass, and `SavingOverTheFileBehindAMappedTensorKeepsItValid` still passes.
5. Write `translated_gpt2_name`, `stored_transposed` and `load_gpt2_weights`. Checkpoint: `Gpt2Weights.Conv1DWeightsAreTransposed` and `MissingEntriesAreReported` pass.
6. Write `decode_utf8`, `contains` and generate `unicode_classes.inc` with `python3 scripts/generate/unicode_classes.py`. Checkpoint: the library compiles with the generated file included.
7. Write `WordSplitter` and `split_gpt2_words`. Checkpoint: all `Gpt2Split` tests pass.
8. Write the byte-to-unicode mapping, the constructor and `encode`/`decode`. Checkpoint: all `Bpe` tests pass.
9. Run `scripts/download/gpt2.sh`, then `pytest python/tests/test_tokenizer.py python/tests/test_gpt.py`. Checkpoint: the tokenizer matches `tiktoken` and the real GPT-2 logits match Hugging Face.
10. Build and run `gpt2_generate`. Checkpoint: it prints a coherent English continuation of the prompt.

> [!TIP]
> Debug the tokenizer before the weights. If `encode("Hello, world!")` does not return `[15496, 11, 995, 0]`, every later comparison fails for a reason that has nothing to do with the model.

## Check your understanding

<details>
<summary>Why does the reader compare header_size with file_size - 8 instead of comparing header_size + 8 with file_size?</summary>

Both values are unsigned 64-bit integers. A malicious header length of `0xFFFFFFFFFFFFFFFF` plus 8 wraps around to 7, which is smaller than any real file size, so the addition form would accept it and the reader would then try to allocate a huge string. The subtraction cannot wrap because the reader has already checked that the file has at least 8 bytes. The 100 MB cap is a second, independent bound.

</details>

<details>
<summary>The official GPT-2 file is loaded with LoadMode::Mapped. Why does it end up copying every tensor?</summary>

Its header is 14,283 bytes, so the data section starts at file byte 14,291, which is 3 more than a multiple of 4. Every tensor offset inside the data section is a multiple of 4, so every F32 tensor sits at a file position that is not 4-byte aligned. A `float*` into the mapping would be misaligned, which is undefined behavior, so the reader copies each tensor into an aligned allocation. Files written by kael pad the header to 8 bytes and do not have this problem.

</details>

<details>
<summary>Why is the temporary file of an atomic save created in the same directory as the target, and why are there two fsync calls?</summary>

`rename` replaces a directory entry atomically only within one file system, and the target's directory is on the target's file system by definition. The first `fsync` flushes the temporary file's data before the rename, so a crash cannot leave the new name pointing at an empty or partial file. The second `fsync`, on the directory, makes the rename itself durable, so after a power loss the directory does not still point at the old file or at nothing.

</details>

<details>
<summary>attn.c_proj.weight is square. What happens if the loader forgets to transpose it?</summary>

The shape check in `load_state_dict` passes, because `[768, 768]` transposed is still `[768, 768]`. The model loads without any error and computes `x @ W` where it should compute `x @ W^T`. The logits are wrong, and generated text degrades. Only a numeric comparison catches it, which is why `Conv1DWeightsAreTransposed` checks every parameter bit for bit and the Python test compares logits with Hugging Face.

</details>

<details>
<summary>Why does BpeTokenizer::decode return a byte string instead of valid UTF-8 text?</summary>

Tokens are sequences of bytes, not characters. A character that takes several UTF-8 bytes can be split across tokens: `"日本"` encodes to four tokens with bytes `e6 97`, `a5`, `e6 9c` and `ac`. Decoding a prefix of the ids can end in the middle of a character. Returning bytes keeps decode lossless and lets the caller decide how to handle an incomplete tail, which is what the streaming server in chapter 13 does.

</details>

<details>
<summary>Why does GPT-2 map the space byte to Ġ instead of keeping it as a space?</summary>

`merges.txt` uses a space to separate the two symbols of a merge, so a symbol cannot contain a raw space without making the file ambiguous. The vocabulary and merges are also stored as text, which cannot hold bytes such as NUL. Mapping each of the 68 unprintable bytes, including the space, to a printable code point from U+0100 upward makes every token a printable string, and the mapping is a bijection, so decoding recovers the exact bytes.

</details>

## Further reading

- [The safetensors repository](https://github.com/huggingface/safetensors): the format specification in its README, and the reference reader and writer used by the interoperability tests.
- [Sennrich, Haddow and Birch, "Neural Machine Translation of Rare Words with Subword Units" (2016)](https://arxiv.org/abs/1508.07909): BPE applied to language, the origin of the merge procedure in this chapter.
- [OpenAI's GPT-2 `encoder.py`](https://github.com/openai/gpt-2/blob/master/src/encoder.py): the original byte-to-unicode table, the pretokenization regex and the merge loop that kael reproduces.
- [tiktoken](https://github.com/openai/tiktoken): the fast reference tokenizer the Python tests compare against, and the meaning of `encode_ordinary`.
- [Hugging Face `modeling_gpt2.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py): `GPT2Model`, the parameter names this chapter maps, and the use of `Conv1D`.
- [Pillai et al., "All File Systems Are Not Created Equal" (OSDI 2014)](https://www.usenix.org/conference/osdi14/technical-sessions/presentation/pillai): what can go wrong between `write`, `fsync` and `rename` on real file systems.
- The `mmap(2)`, `rename(2)` and `fsync(2)` manual pages: the exact guarantees behind mapped loading and atomic saves.
- [`src/io/safetensors.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/io/safetensors.cpp) and [`src/text/bpe.cpp`](https://github.com/JetBerri/kael.cpp/blob/main/src/text/bpe.cpp): the two files of this chapter, about 700 lines together.

Next: [13. Serving and tooling](@/book/13-serving-and-tooling.md)
