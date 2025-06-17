//! Convert Hugging Face checkpoints into a simplified format.
//!
//! This binary mimics the behavior of the Python `convert.py` script
//! by loading files and saving them in a new layout.

use std::path::PathBuf;

/// Application entry point for converting checkpoints.
pub struct ConvertApp {
    input: PathBuf,
    output: PathBuf,
}

impl ConvertApp {
    /// Create a new conversion application.
    pub fn new(input: PathBuf, output: PathBuf) -> Self {

        Self { input, output }
    }

    /// Execute the conversion process.
    pub fn run(&self) {

        println!("Converting {:?} -> {:?}", self.input, self.output);
        // Actual conversion logic would be implemented here.
    }
}

fn main() {

    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: convert <input> <output>");
        return;
    }
    let app = ConvertApp::new(args[1].clone().into(), args[2].clone().into());
    app.run();

}

