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

## Limitations

This example quantizes only a toy model and does not load external checkpoints. Adapting it for real-world models like the DeepSeek distilled series will require significant additional work.
