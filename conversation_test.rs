#!/usr/bin/env rust-script
//! Simple test to understand current generation capabilities
//! 
//! ```cargo
//! [dependencies]
//! inference-re = { path = "./inference-re" }
//! rand = "0.8"
//! ```

use inference_re::model::{ModelArgs, Transformer};

fn main() {
    println!("Testing current model generation...");
    
    // Create a small model for testing
    let args = ModelArgs {
        max_seq_len: 128,
        vocab_size: 1000,
        dim: 64,
        n_layers: 2,
        n_heads: 4,
        hidden_dim: 256,
    };
    
    let model = Transformer::new(args);
    
    // Test forward pass
    let tokens = vec![0, 1, 2];
    let logits = model.forward(&tokens);
    
    println!("Model created successfully!");
    println!("Input tokens: {:?}", tokens);
    println!("Output shape: {:?}", logits.shape());
    
    // Test a simple generation loop with actual sampling from logits
    let mut generated = tokens.clone();
    for i in 0..5 {
        let current_logits = model.forward(&generated);
        let last_logit_row = current_logits.row(current_logits.nrows() - 1);
        
        // Find the token with highest probability (argmax)
        let next_token = last_logit_row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
            
        generated.push(next_token);
        println!("Step {}: generated token {} (logit: {:.3})", 
                i, next_token, last_logit_row[next_token]);
    }
    
    println!("Final generated sequence: {:?}", generated);
}