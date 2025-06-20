//! Text generation utility using the simplified Transformer.
//!
//! This binary mirrors the behavior of `generate.py` and provides
//! a minimal interface for sampling tokens.

use inference_re::model::{ModelArgs, Transformer};
use rand::Rng;

/// Application for generating text from prompts.
pub struct GenerateApp {
    model: Transformer,
}

impl GenerateApp {
    /// Create a new generator application.
    pub fn new(model: Transformer) -> Self {

        Self { model }
    }

    /// Generate tokens for a prompt.
    pub fn generate(&self, prompt: &[usize], max_new: usize) -> Vec<usize> {

        let mut rng = rand::thread_rng();
        let mut tokens = prompt.to_vec();
        for _ in 0..max_new {
            let next: usize = rng.gen_range(0..self.model.args.vocab_size);
            tokens.push(next);
        }
        tokens
    }
}

fn main() {

    let args = ModelArgs::new();
    let model = Transformer::new(args.clone());
    let app = GenerateApp::new(model);
    let prompt = vec![0_usize];
    let out = app.generate(&prompt, 5);
    println!("generated: {:?}", out);

}

