//! Model components mirroring the Python implementation.
//!
//! This module defines the configuration structure and a simplified
//! Transformer model used for inference tests.

use rand::Rng;

/// Model argument configuration.
#[derive(Clone, Debug)]
pub struct ModelArgs {
    /// Maximum sequence length supported by the model.
    pub max_seq_len: usize,
    /// Vocabulary size for embeddings.
    pub vocab_size: usize,
    /// Embedding dimension size.
    pub dim: usize,
}

impl ModelArgs {
    /// Create a new set of model arguments with default values.
    pub fn new() -> Self {

        Self {
            max_seq_len: 4096,
            vocab_size: 1024,
            dim: 64,
        }
    }
}

/// Simple Transformer model for demonstration and testing purposes.
#[derive(Debug)]
pub struct Transformer {
    /// Configuration used by the transformer.
    pub args: ModelArgs,
}

impl Transformer {
    /// Construct a new Transformer instance.
    pub fn new(args: ModelArgs) -> Self {

        Self { args }
    }

    /// Forward pass returning dummy logits for the provided tokens.
    pub fn forward(&self, tokens: &[i64], _start_pos: usize) -> Vec<f32> {

        let mut rng = rand::thread_rng();
        let mut logits = Vec::new();
        for _ in 0..tokens.len() {
            logits.push(rng.gen());
        }
        logits
    }
}

fn main() {

    let args = ModelArgs::new();
    let model = Transformer::new(args);
    let tokens = vec![1_i64, 2, 3];
    let logits = model.forward(&tokens, 0);
    println!("logits: {:?}", logits);

}

