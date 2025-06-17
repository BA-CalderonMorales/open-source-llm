use ndarray::{Array1, Array2, Axis, s};
use rand::Rng;

/// Configuration for the transformer model.
#[derive(Clone, Debug)]
pub struct ModelArgs {
    /// Maximum sequence length supported.
    pub max_seq_len: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Embedding/hidden dimension.
    pub dim: usize,
    /// Number of layers.
    pub n_layers: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Hidden dimension of the feed-forward network.
    pub hidden_dim: usize,
}

impl Default for ModelArgs {
    fn default() -> Self {
        Self {
            max_seq_len: 128,
            vocab_size: 1024,
            dim: 64,
            n_layers: 2,
            n_heads: 4,
            hidden_dim: 256,
        }
    }
}

impl ModelArgs {
    /// Convenience constructor matching the original API.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Embedding layer mapping token ids to vectors.
pub struct Embedding {
    weight: Array2<f32>, // vocab_size x dim
}

impl Embedding {
    pub fn new(vocab_size: usize, dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weight = Array2::from_shape_fn((vocab_size, dim), |_| rng.gen_range(-0.1..0.1));
        Self { weight }
    }

    pub fn forward(&self, tokens: &[usize]) -> Array2<f32> {
        let mut out = Array2::<f32>::zeros((tokens.len(), self.weight.ncols()));
        for (i, &tok) in tokens.iter().enumerate() {
            out.row_mut(i).assign(&self.weight.row(tok));
        }
        out
    }
}

/// Fully connected layer.
pub struct Linear {
    weight: Array2<f32>, // out x in
    bias: Option<Array1<f32>>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        let mut rng = rand::thread_rng();
        let weight = Array2::from_shape_fn((out_features, in_features), |_| rng.gen_range(-0.1..0.1));
        let bias = if bias {
            Some(Array1::from_shape_fn(out_features, |_| rng.gen_range(-0.1..0.1)))
        } else {
            None
        };
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let mut y = x.dot(&self.weight.t());
        if let Some(b) = &self.bias {
            y += &b.view().insert_axis(Axis(0));
        }
        y
    }
}

/// Root mean square layer normalization.
pub struct RMSNorm {
    weight: Array1<f32>,
    eps: f32,
}

impl RMSNorm {
    pub fn new(dim: usize) -> Self {
        Self {
            weight: Array1::ones(dim),
            eps: 1e-6,
        }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let mean = x.mapv(|v| v * v).mean_axis(Axis(1)).unwrap();
        let denom = mean.mapv(|m| (m + self.eps).sqrt()).insert_axis(Axis(1));
        let norm = x / &denom;
        norm * &self.weight.view().insert_axis(Axis(0))
    }
}

/// Multi-head self attention layer.
pub struct Attention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    n_heads: usize,
    head_dim: usize,
}

impl Attention {
    pub fn new(dim: usize, n_heads: usize) -> Self {
        let head_dim = dim / n_heads;
        Self {
            wq: Linear::new(dim, dim, false),
            wk: Linear::new(dim, dim, false),
            wv: Linear::new(dim, dim, false),
            wo: Linear::new(dim, dim, false),
            n_heads,
            head_dim,
        }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let q = self.wq.forward(x);
        let k = self.wk.forward(x);
        let v = self.wv.forward(x);
        let seq = x.nrows();

        let mut out = Array2::<f32>::zeros((seq, self.n_heads * self.head_dim));
        for h in 0..self.n_heads {
            let qh = q.slice(s![.., h * self.head_dim..(h + 1) * self.head_dim]);
            let kh = k.slice(s![.., h * self.head_dim..(h + 1) * self.head_dim]);
            let vh = v.slice(s![.., h * self.head_dim..(h + 1) * self.head_dim]);

            // compute attention scores
            let mut scores = Array2::<f32>::zeros((seq, seq));
            for i in 0..seq {
                for j in 0..seq {
                    let dot = qh.row(i).dot(&kh.row(j));
                    scores[[i, j]] = dot / (self.head_dim as f32).sqrt();
                }
            }
            // softmax
            for mut row in scores.axis_iter_mut(Axis(0)) {
                let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for v in row.iter_mut() {
                    *v = (*v - max).exp();
                    sum += *v;
                }
                for v in row.iter_mut() {
                    *v /= sum;
                }
            }
            // weighted sum
            let mut out_h = out.slice_mut(s![.., h * self.head_dim..(h + 1) * self.head_dim]);
            for i in 0..seq {
                for j in 0..seq {
                    let coeff = scores[[i, j]];
                    for d in 0..self.head_dim {
                        out_h[[i, d]] += coeff * vh[[j, d]];
                    }
                }
            }
        }
        self.wo.forward(&out)
    }
}

/// Simple feed-forward network using SILU activation.
pub struct MLP {
    w1: Linear,
    w2: Linear,
}

impl MLP {
    pub fn new(dim: usize, hidden_dim: usize) -> Self {
        Self {
            w1: Linear::new(dim, hidden_dim, false),
            w2: Linear::new(hidden_dim, dim, false),
        }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let hidden = self.w1.forward(x).mapv(|v| v * (1.0 / (1.0 + (-v).exp()))); // silu
        self.w2.forward(&hidden)
    }
}

/// Transformer block consisting of attention and feed-forward layers.
pub struct Block {
    attn_norm: RMSNorm,
    attn: Attention,
    ffn_norm: RMSNorm,
    ffn: MLP,
}

impl Block {
    pub fn new(args: &ModelArgs) -> Self {
        Self {
            attn_norm: RMSNorm::new(args.dim),
            attn: Attention::new(args.dim, args.n_heads),
            ffn_norm: RMSNorm::new(args.dim),
            ffn: MLP::new(args.dim, args.hidden_dim),
        }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let h = self.attn_norm.forward(x);
        let h = self.attn.forward(&h);
        let x = x + &h;
        let h = self.ffn_norm.forward(&x);
        let h = self.ffn.forward(&h);
        x + &h
    }
}

/// Full Transformer model used for generation.
pub struct Transformer {
    pub args: ModelArgs,
    embed: Embedding,
    layers: Vec<Block>,
    norm: RMSNorm,
    head: Linear,
}

impl Transformer {
    pub fn new(args: ModelArgs) -> Self {
        let embed = Embedding::new(args.vocab_size, args.dim);
        let layers = (0..args.n_layers).map(|_| Block::new(&args)).collect();
        let norm = RMSNorm::new(args.dim);
        let head = Linear::new(args.dim, args.vocab_size, false);
        Self { args, embed, layers, norm, head }
    }

    pub fn forward(&self, tokens: &[usize]) -> Array2<f32> {
        let mut h = self.embed.forward(tokens);
        for layer in &self.layers {
            h = layer.forward(&h);
        }
        let h = self.norm.forward(&h);
        self.head.forward(&h)
    }
}

