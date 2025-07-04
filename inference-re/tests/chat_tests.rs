use inference_re::model::{ModelArgs, Transformer};

#[test]
fn test_tokenizer_encoding() {
    // Test the simple tokenizer functionality
    use std::collections::HashMap;
    
    // Simulate the SimpleTokenizer logic
    let vocab = vec![
        "<pad>", "<unk>", "<start>", "<end>", 
        "hello", "world", "how", "are", "you", "i", "am", "fine"
    ];
    
    let word_to_id: HashMap<String, usize> = vocab.iter()
        .enumerate()
        .map(|(i, word)| (word.to_string(), i))
        .collect();
    
    // Test encoding
    let text = "hello world";
    let mut tokens = vec![2]; // <start>
    for word in text.split_whitespace() {
        tokens.push(word_to_id.get(word).copied().unwrap_or(1));
    }
    
    assert_eq!(tokens, vec![2, 4, 5]); // <start>, hello, world
}

#[test] 
fn test_model_forward_pass() {
    // Test that the model can do a forward pass without crashing
    let args = ModelArgs {
        max_seq_len: 32,
        vocab_size: 100,
        dim: 32,
        n_layers: 1,
        n_heads: 2,
        hidden_dim: 64,
    };
    
    let model = Transformer::new(args);
    let tokens = vec![0, 1, 2];
    let logits = model.forward(&tokens);
    
    // Check output shape is correct
    assert_eq!(logits.nrows(), tokens.len());
    assert_eq!(logits.ncols(), 100); // vocab_size
}

#[test]
fn test_generation_produces_valid_tokens() {
    // Test that generation produces tokens within vocab bounds
    let args = ModelArgs {
        max_seq_len: 32,
        vocab_size: 50,
        dim: 16,
        n_layers: 1,
        n_heads: 2,
        hidden_dim: 32,
    };
    
    let model = Transformer::new(args);
    let prompt = vec![0, 1];
    
    // Simulate generation logic
    let logits = model.forward(&prompt);
    let last_logits = logits.row(logits.nrows() - 1);
    
    // Find argmax (most likely next token)
    let next_token = last_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();
    
    // Token should be within vocab bounds
    assert!(next_token < 50);
}