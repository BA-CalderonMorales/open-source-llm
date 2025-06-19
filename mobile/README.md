# Mobile Deployment Guide

This folder now contains a lightweight Rust tool for quantizing the demo transformer model from this repository. The resulting file can be loaded by custom mobile runtimes.

## Build

```bash
cargo build --release --manifest-path mobile/Cargo.toml
```

## Usage

```bash
./target/release/mobile --out quantized.bin
```

The tool creates `quantized.bin` containing 8-bit versions of the embedding and output head. It serves as a minimal example of preparing weights for on-device inference.

## Limitations

This example quantizes only a toy model and does not load external checkpoints. Adapting it for real-world models like the DeepSeek distilled series will require significant additional work.
