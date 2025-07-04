//! Interactive chat interface using the simplified Transformer.
//!
//! This provides a basic conversation loop that demonstrates the MVP 
//! architecture described in the PRD. It uses the existing Rust model
//! and adds proper text generation with sampling plus SQLite context storage.

use inference_re::model::{ModelArgs, Transformer};
use inference_re::context::{ContextStore, ContextStats};
use std::io::{self, Write};
use rand::prelude::*;
use std::path::PathBuf;

/// A simple tokenizer that maps words to indices for demo purposes.
/// In production, this would use a proper BPE tokenizer.
pub struct SimpleTokenizer {
    vocab: Vec<String>,
    word_to_id: std::collections::HashMap<String, usize>,
}

impl SimpleTokenizer {
    pub fn new() -> Self {
        // Create a small demo vocabulary with common words
        let vocab = vec![
            "<pad>", "<unk>", "<start>", "<end>", 
            "hello", "world", "how", "are", "you", "i", "am", "fine", "what", "is", "your", "name",
            "my", "name", "is", "assistant", "help", "can", "please", "thank", "yes", "no",
            "the", "and", "a", "to", "of", "in", "that", "have", "it", "for", "not", "on", 
            "with", "he", "as", "his", "they", "be", "at", "this", "from", "or", "had",
            "good", "great", "nice", "bad", "ok", "sure", "maybe", "think", "know", "see",
        ];
        
        let word_to_id = vocab.iter()
            .enumerate()
            .map(|(i, word)| (word.to_string(), i))
            .collect();
            
        Self { vocab: vocab.into_iter().map(String::from).collect(), word_to_id }
    }
    
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut tokens = vec![2]; // <start> token
        
        for word in text.to_lowercase().split_whitespace() {
            let token_id = self.word_to_id.get(word).copied().unwrap_or(1); // <unk> if not found
            tokens.push(token_id);
        }
        
        tokens
    }
    
    pub fn decode(&self, tokens: &[usize]) -> String {
        tokens.iter()
            .map(|&id| self.vocab.get(id).unwrap_or(&"<unk>".to_string()).clone())
            .filter(|word| word != "<start>" && word != "<pad>") // Filter out special tokens
            .collect::<Vec<_>>()
            .join(" ")
    }
    
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

/// Application for interactive conversation.
pub struct ChatApp {
    model: Transformer,
    tokenizer: SimpleTokenizer,
    context_store: ContextStore,
    max_context_length: usize,
    max_generation_length: usize,
}

impl ChatApp {
    /// Create a new chat application.
    pub fn new() -> Self {
        let tokenizer = SimpleTokenizer::new();
        
        // Create model with vocab size matching our tokenizer
        let args = ModelArgs {
            max_seq_len: 128,
            vocab_size: tokenizer.vocab_size(),
            dim: 64,
            n_layers: 2,
            n_heads: 4,
            hidden_dim: 256,
        };
        
        let model = Transformer::new(args);
        
        // Initialize SQLite context store
        let db_path = PathBuf::from("conversation_history.db");
        let context_store = ContextStore::new(&db_path)
            .expect("Failed to initialize conversation database");
        
        Self { 
            model, 
            tokenizer,
            context_store,
            max_context_length: 64,  // Limit context to avoid memory issues
            max_generation_length: 20, // Limit response length for demo
        }
    }
    
    /// Generate a response using context from conversation history.
    pub fn generate_response_with_context(&self, user_input: &str) -> String {
        // Get recent context from database (following PRD architecture)
        let recent_conversations = self.context_store.get_recent_conversations(3)
            .unwrap_or_else(|_| vec![]);
        
        // Build context-aware prompt
        let mut prompt_text = String::new();
        
        // Add recent conversation context
        for conv in &recent_conversations {
            prompt_text.push_str(&format!("Human: {}\nAssistant: {}\n", 
                                         conv.user_message, conv.assistant_response));
        }
        
        // Add current user input
        prompt_text.push_str(&format!("Human: {}\nAssistant: ", user_input));
        
        // Encode the full prompt
        let prompt_tokens = self.tokenizer.encode(&prompt_text);
        
        // Generate response using existing generation logic
        let response_tokens = self.generate_response(&prompt_tokens);
        let response_text = self.tokenizer.decode(&response_tokens);
        
        // Clean up the response
        let cleaned_response = response_text
            .replace("<end>", "")
            .trim()
            .to_string();
        
        // Save this conversation turn to the database
        let token_count = (prompt_tokens.len() + response_tokens.len()) as i32;
        if let Err(e) = self.context_store.save_conversation(user_input, &cleaned_response, token_count) {
            eprintln!("Warning: Failed to save conversation to database: {}", e);
        }
        
        cleaned_response
    }
    
    /// Generate a response to the given prompt tokens using proper sampling.
    fn generate_response(&self, prompt_tokens: &[usize]) -> Vec<usize> {
        let mut rng = thread_rng();
        let mut tokens = prompt_tokens.to_vec();
        
        // Limit context length for performance
        if tokens.len() > self.max_context_length {
            tokens = tokens[(tokens.len() - self.max_context_length)..].to_vec();
        }
        
        for _ in 0..self.max_generation_length {
            let logits = self.model.forward(&tokens);
            let last_logits = logits.row(logits.nrows() - 1);
            
            // Simple temperature sampling - use higher temperature for more randomness
            let temperature = 0.7;
            let probs: Vec<f32> = last_logits.iter()
                .map(|&logit| (logit / temperature).exp())
                .collect();
            
            let total: f32 = probs.iter().sum();
            let normalized_probs: Vec<f32> = probs.iter().map(|p| p / total).collect();
            
            // Sample from the probability distribution
            let mut cumsum = 0.0;
            let random_value: f32 = rng.gen();
            let mut next_token = 0;
            
            for (i, prob) in normalized_probs.iter().enumerate() {
                cumsum += prob;
                if random_value <= cumsum {
                    next_token = i;
                    break;
                }
            }
            
            tokens.push(next_token);
            
            // Stop if we generate an end token (index 3)
            if next_token == 3 {
                break;
            }
        }
        
        // Return only the newly generated tokens
        tokens[prompt_tokens.len()..].to_vec()
    }
    
    /// Run the interactive chat loop.
    pub fn run(&self) {
        println!("🤖 DeepSeek Local Assistant (MVP Demo)");
        println!("=======================================");
        
        // Show context store statistics
        if let Ok(stats) = self.context_store.get_stats() {
            println!("📊 Database: {} conversations, {} tokens stored", 
                    stats.total_conversations, stats.total_tokens);
        }
        
        println!("Commands: 'quit' to exit, '!stats' for statistics, '!clear' to clear history");
        println!("Note: This is a demo with limited vocabulary and untrained model.");
        println!();
        
        loop {
            print!("You: ");
            io::stdout().flush().unwrap();
            
            let mut input = String::new();
            match io::stdin().read_line(&mut input) {
                Ok(_) => {
                    let input = input.trim();
                    if input.is_empty() {
                        continue;
                    }
                    
                    // Handle special commands
                    match input.to_lowercase().as_str() {
                        "quit" => {
                            println!("Goodbye! 👋");
                            break;
                        }
                        "!stats" => {
                            self.show_stats();
                            continue;
                        }
                        "!clear" => {
                            if let Err(e) = self.context_store.clear_history() {
                                println!("❌ Error clearing history: {}", e);
                            } else {
                                println!("✅ Conversation history cleared.");
                            }
                            continue;
                        }
                        _ => {}
                    }
                    
                    // Warn about resource limits as per PRD
                    if input.len() > 200 {
                        println!("⚠️  Warning: Input is quite long. Processing with limited context.");
                    }
                    
                    print!("Assistant: ");
                    io::stdout().flush().unwrap();
                    
                    // Generate context-aware response
                    let response = self.generate_response_with_context(input);
                    
                    if response.trim().is_empty() || response.trim() == "unk" {
                        println!("I'm sorry, I'm having trouble generating a response.");
                    } else {
                        println!("{}", response);
                    }
                    
                    println!(); // Empty line for readability
                }
                Err(error) => {
                    eprintln!("Error reading input: {}", error);
                    break;
                }
            }
        }
    }
    
    /// Show database statistics and system status.
    fn show_stats(&self) {
        println!("\n📊 DeepSeek Local Assistant Statistics");
        println!("=====================================");
        
        match self.context_store.get_stats() {
            Ok(stats) => {
                println!("📝 Total conversations: {}", stats.total_conversations);
                println!("🔤 Total tokens processed: {}", stats.total_tokens);
                println!("💾 Database size: {} bytes ({:.1} KB)", 
                        stats.db_size_bytes, stats.db_size_bytes as f64 / 1024.0);
                
                // Show recent conversations
                if let Ok(recent) = self.context_store.get_recent_conversations(3) {
                    if !recent.is_empty() {
                        println!("\n🕒 Recent conversations:");
                        for (i, conv) in recent.iter().enumerate() {
                            let truncated_user = if conv.user_message.len() > 50 {
                                format!("{}...", &conv.user_message[..47])
                            } else {
                                conv.user_message.clone()
                            };
                            
                            let truncated_response = if conv.assistant_response.len() > 50 {
                                format!("{}...", &conv.assistant_response[..47])
                            } else {
                                conv.assistant_response.clone()
                            };
                            
                            println!("  {}. User: \"{}\"", i + 1, truncated_user);
                            println!("     Assistant: \"{}\"", truncated_response);
                        }
                    }
                }
            }
            Err(e) => {
                println!("❌ Error retrieving statistics: {}", e);
            }
        }
        
        println!("🧠 Model: {} vocab, {} layers, {} dim", 
                self.tokenizer.vocab_size(), self.model.args.n_layers, self.model.args.dim);
        println!();
    }
}

fn main() {
    let app = ChatApp::new();
    app.run();
}