use clap::Parser;
use std::path::PathBuf;
use mobile::demo_quantize;

#[derive(Parser)]
struct Args {
    /// Output file for the quantized model
    #[arg(long, default_value = "model.q8")]
    out: PathBuf,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    demo_quantize(&args.out)?;
    println!("Saved quantized model to {:?}", args.out);
    Ok(())
}
