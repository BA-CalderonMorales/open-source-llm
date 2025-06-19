use clap::Parser;
use inference_re::model::{ModelArgs, Transformer, Linear, Embedding};
use ndarray::{Array1, Array2};
use std::{fs::File, io::Write, path::PathBuf};
use bytemuck::cast_slice;

/// Simple quantization of a tensor to 8-bit integers with a scale factor.
fn quantize_tensor(t: &Array2<f32>) -> (Vec<i8>, f32) {
    let max = t.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
    let scale = if max == 0.0 { 1.0 } else { 127.0 / max };
    let data = t.iter().map(|&v| (v * scale).round() as i8).collect();
    (data, 1.0 / scale)
}

/// Quantized linear layer.
struct QLinear {
    weight: Vec<i8>,
    bias: Option<Array1<f32>>,
    scale: f32,
    in_features: usize,
    out_features: usize,
}

impl QLinear {
    fn from_linear(l: &Linear) -> Self {
        let (weight, scale) = quantize_tensor(l.weight());
        Self {
            weight,
            bias: l.bias().cloned(),
            scale,
            in_features: l.weight().ncols(),
            out_features: l.weight().nrows(),
        }
    }
}

/// Quantized embedding layer.
struct QEmbedding {
    weight: Vec<i8>,
    scale: f32,
    vocab: usize,
    dim: usize,
}

impl QEmbedding {
    fn from_embedding(e: &Embedding) -> Self {
        let (weight, scale) = quantize_tensor(e.weight());
        Self { weight, scale, vocab: e.weight().nrows(), dim: e.weight().ncols() }
    }
}

/// Quantized transformer holding only embedding and head for demo purposes.
struct QTransformer {
    embed: QEmbedding,
    head: QLinear,
}

impl QTransformer {
    fn from_model(m: &Transformer) -> Self {
        Self {
            embed: QEmbedding::from_embedding(m.embed()),
            head: QLinear::from_linear(m.head()),
        }
    }

    fn save(&self, path: &PathBuf) -> std::io::Result<()> {
        let mut f = File::create(path)?;
        // write embedding
        f.write_all(&(self.embed.vocab as u32).to_le_bytes())?;
        f.write_all(&(self.embed.dim as u32).to_le_bytes())?;
        f.write_all(&self.embed.scale.to_le_bytes())?;
        f.write_all(cast_slice(&self.embed.weight))?;
        // write head
        f.write_all(&(self.head.out_features as u32).to_le_bytes())?;
        f.write_all(&(self.head.in_features as u32).to_le_bytes())?;
        f.write_all(&self.head.scale.to_le_bytes())?;
        f.write_all(cast_slice(&self.head.weight))?;
        if let Some(b) = &self.head.bias {
            f.write_all(&1u8.to_le_bytes())?;
            for v in b.iter() { f.write_all(&v.to_le_bytes())?; }
        } else {
            f.write_all(&0u8.to_le_bytes())?;
        }
        Ok(())
    }
}

#[derive(Parser)]
struct Args {
    /// Output file for the quantized model
    #[arg(long, default_value = "model.q8")]
    out: PathBuf,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();

    // create tiny transformer and quantize
    let model = Transformer::new(ModelArgs::new());
    let q = QTransformer::from_model(&model);
    q.save(&args.out)?;
    println!("Saved quantized model to {:?}", args.out);
    Ok(())
}
