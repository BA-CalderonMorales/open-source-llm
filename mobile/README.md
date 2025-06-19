# Mobile Deployment Guide

This folder now contains a lightweight Rust library and binary for quantizing the demo transformer model from this repository. The resulting file can be loaded and executed by custom mobile runtimes.

## Build

```bash
cargo build --release --manifest-path mobile/Cargo.toml
```

## Usage

```bash
./target/release/mobile --out quantized.bin
```

The tool creates `quantized.bin` containing 8-bit versions of the embedding and output head. It serves as a minimal example of preparing weights for on-device inference.

### Loading with memory mapping

The quantized weights can be mapped directly into memory:

```rust
use std::path::PathBuf;
use mobile::QTransformer;

let model = QTransformer::load_mmap(&PathBuf::from("quantized.bin"))?;
```

This avoids reading the entire file at once and reduces peak memory usage.

Once loaded, you can run a forward pass or generate tokens:

```rust
let tokens = vec![1_usize, 2, 3];
let logits = model.forward(&tokens);
let generated = model.generate(&tokens, 4);
```

### Tokenization

The `Tokenizer` type handles simple whitespace tokenization. Create it from a
list of tokens and use `encode` and `decode` to convert between text and token
ids.

```rust
use mobile::Tokenizer;

let tokenizer = Tokenizer::new(vec!["<unk>".into(), "hello".into(), "world".into()]);
let ids = tokenizer.encode("hello world");
let text = tokenizer.decode(&ids);
```

### C FFI

The library exposes a small C-compatible API for integration with native mobile
applications. Build the crate as a `cdylib` to obtain a shared library.

```bash
cargo build --release --manifest-path mobile/Cargo.toml
```

Key functions include:

```c
QTransformer* mobile_model_load(const char* path);
void mobile_model_free(QTransformer* model);
TokenArray mobile_generate(QTransformer* model, const size_t* ids,
                           size_t len, size_t max_tokens);
void token_array_free(TokenArray arr);

Tokenizer* tokenizer_new(const char* const* tokens, size_t len);
void tokenizer_free(Tokenizer* t);
TokenArray tokenizer_encode(Tokenizer* t, const char* text);
char* tokenizer_decode(Tokenizer* t, const size_t* ids, size_t len);
void string_free(char* s);
```

The `TokenArray` struct holds a pointer and length for arrays returned by the
library. Call the corresponding `*_free` functions to release allocated memory.

## Limitations

This example quantizes only a toy model and does not load external checkpoints. Adapting it for real-world models like the DeepSeek distilled series will require significant additional work.
