use anyhow::Result;
use clap::Parser;
use rust_bert::pipelines::generation::{GenerateConfig, LanguageGenerator, GPT2Generator};
use std::fs;
use std::io::{self, Write};

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Path to model checkpoint (unused for GPT2)
    #[arg(long)]
    ckpt_path: Option<String>,

    /// Path to config json
    #[arg(long)]
    config: Option<String>,

    /// Input file with prompts (one per line)
    #[arg(long, default_value = "")]
    input_file: String,

    /// Run in interactive mode
    #[arg(long)]
    interactive: bool,

    /// Maximum new tokens
    #[arg(long, default_value_t = 100)]
    max_new_tokens: i64,

    /// Sampling temperature
    #[arg(long, default_value_t = 1.0)]
    temperature: f64,
}

fn load_prompts(path: &str) -> Result<Vec<String>> {
    let content = fs::read_to_string(path)?;
    Ok(content.lines().map(|s| s.to_owned()).collect())
}

fn main() -> Result<()> {
    let args = Args::parse();

    let generate_config = GenerateConfig {
        max_length: args.max_new_tokens,
        temperature: args.temperature,
        ..Default::default()
    };
    let mut generator = GPT2Generator::new(generate_config)?;

    if args.interactive {
        let mut input = String::new();
        let stdin = io::stdin();
        loop {
            print!(">>> ");
            io::stdout().flush()?;
            input.clear();
            stdin.read_line(&mut input)?;
            let trimmed = input.trim();
            if trimmed == "/exit" {
                break;
            }
            let output = generator.generate(&[trimmed], None);
            println!("{}", output[0]);
        }
    } else if !args.input_file.is_empty() {
        let prompts = load_prompts(&args.input_file)?;
        let outputs = generator.generate(&prompts, None);
        for (prompt, response) in prompts.iter().zip(outputs.iter()) {
            println!("Prompt: {}", prompt);
            println!("Completion: {}\n", response);
        }
    }
    Ok(())
}
