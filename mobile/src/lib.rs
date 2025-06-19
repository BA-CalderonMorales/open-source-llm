use inference_re::model::{ModelArgs, Transformer, Linear, Embedding};
use ndarray::{Array1, Array2, Axis};
use std::{collections::HashMap, fs::File, io::Write, path::PathBuf};
use bytemuck::cast_slice;
use memmap2::MmapOptions;

pub mod ffi;
pub use ffi::*;

/// Simple quantization of a tensor to 8-bit integers with a scale factor.
fn quantize_tensor(t: &Array2<f32>) -> (Vec<i8>, f32) {
    let max = t.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
    let scale = if max == 0.0 { 1.0 } else { 127.0 / max };
    let data = t.iter().map(|&v| (v * scale).round() as i8).collect();
    (data, 1.0 / scale)
}

/// Quantized linear layer.
#[derive(Clone)]
pub struct QLinear {
    weight: Vec<i8>,
    bias: Option<Array1<f32>>,
    scale: f32,
    in_features: usize,
    out_features: usize,
}

impl QLinear {
    pub fn from_linear(l: &Linear) -> Self {
        let (weight, scale) = quantize_tensor(l.weight());
        Self {
            weight,
            bias: l.bias().cloned(),
            scale,
            in_features: l.weight().ncols(),
            out_features: l.weight().nrows(),
        }
    }

    pub fn from_mmap(buf: &[u8]) -> (Self, usize) {
        let mut offset = 0;
        let out_features = u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let in_features = u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let scale = f32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap());
        offset += 4;
        let weight_len = out_features * in_features;
        let weight = buf[offset..offset + weight_len].iter().map(|&b| b as i8).collect();
        offset += weight_len;
        let has_bias = buf[offset] == 1;
        offset += 1;
        let bias = if has_bias {
            let mut b = Vec::with_capacity(out_features);
            for _ in 0..out_features {
                b.push(f32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap()));
                offset += 4;
            }
            Some(Array1::from(b))
        } else {
            None
        };
        (
            Self { weight, bias, scale, in_features, out_features },
            offset,
        )
    }

    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        let weight = Array2::from_shape_fn((self.out_features, self.in_features), |(i, j)| {
            self.weight[i * self.in_features + j] as f32 * self.scale
        });
        let mut out = input.dot(&weight.t());
        if let Some(b) = &self.bias {
            out += &b.view().insert_axis(Axis(0));
        }
        out
    }
}

/// Quantized embedding layer.
#[derive(Clone)]
pub struct QEmbedding {
    weight: Vec<i8>,
    scale: f32,
    vocab: usize,
    dim: usize,
}

impl QEmbedding {
    pub fn from_embedding(e: &Embedding) -> Self {
        let (weight, scale) = quantize_tensor(e.weight());
        Self { weight, scale, vocab: e.weight().nrows(), dim: e.weight().ncols() }
    }

    pub fn from_mmap(buf: &[u8]) -> (Self, usize) {
        let mut offset = 0;
        let vocab = u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let dim = u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let scale = f32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap());
        offset += 4;
        let weight_len = vocab * dim;
        let weight = buf[offset..offset + weight_len].iter().map(|&b| b as i8).collect();
        offset += weight_len;
        (
            Self { weight, scale, vocab, dim },
            offset,
        )
    }

    pub fn forward(&self, tokens: &[usize]) -> Array2<f32> {
        let mut out = Array2::<f32>::zeros((tokens.len(), self.dim));
        for (i, &tok) in tokens.iter().enumerate() {
            assert!(tok < self.vocab);
            for d in 0..self.dim {
                let idx = tok * self.dim + d;
                out[[i, d]] = self.weight[idx] as f32 * self.scale;
            }
        }
        out
    }
}

/// Quantized transformer holding only embedding and head for demo purposes.
#[derive(Clone)]
pub struct QTransformer {
    pub embed: QEmbedding,
    pub head: QLinear,
}

impl QTransformer {
    pub fn from_model(m: &Transformer) -> Self {
        Self {
            embed: QEmbedding::from_embedding(m.embed()),
            head: QLinear::from_linear(m.head()),
        }
    }

    pub fn save(&self, path: &PathBuf) -> std::io::Result<()> {
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

    pub fn load(path: &PathBuf) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let (embed, off) = QEmbedding::from_mmap(&bytes);
        let (head, _) = QLinear::from_mmap(&bytes[off..]);
        Ok(Self { embed, head })
    }

    pub fn load_mmap(path: &PathBuf) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let bytes = &mmap[..];
        let (embed, off) = QEmbedding::from_mmap(bytes);
        let (head, _) = QLinear::from_mmap(&bytes[off..]);
        Ok(Self { embed, head })
    }

    pub fn forward(&self, tokens: &[usize]) -> Array2<f32> {
        let h = self.embed.forward(tokens);
        self.head.forward(&h)
    }

    pub fn generate(&self, prompt: &[usize], max_tokens: usize) -> Vec<usize> {
        let mut tokens = prompt.to_vec();
        for _ in 0..max_tokens {
            let logits = self.forward(&tokens);
            let last = logits.row(logits.nrows() - 1);
            let next = argmax(&last.to_owned());
            tokens.push(next);
        }
        tokens
    }
}

fn argmax(v: &Array1<f32>) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

pub fn demo_quantize(path: &PathBuf) -> std::io::Result<()> {
    let model = Transformer::new(ModelArgs::new());
    let q = QTransformer::from_model(&model);
    q.save(path)
}

// === FFI ===
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

#[repr(C)]
pub struct TokenArray {
    pub ptr: *mut usize,
    pub len: usize,
}

#[no_mangle]
pub extern "C" fn token_array_free(arr: TokenArray) {
    if !arr.ptr.is_null() {
        unsafe {
            Vec::from_raw_parts(arr.ptr, arr.len, arr.len);
        }
    }
}

#[no_mangle]
pub extern "C" fn string_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            drop(CString::from_raw(ptr));
        }
    }
}

#[no_mangle]
pub extern "C" fn mobile_model_load(path: *const c_char) -> *mut QTransformer {
    if path.is_null() {
        return std::ptr::null_mut();
    }
    let cstr = unsafe { CStr::from_ptr(path) };
    let path = PathBuf::from(cstr.to_string_lossy().into_owned());
    match QTransformer::load(&path) {
        Ok(m) => Box::into_raw(Box::new(m)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn mobile_model_free(model: *mut QTransformer) {
    if !model.is_null() {
        unsafe {
            drop(Box::from_raw(model));
        }
    }
}

#[no_mangle]
pub extern "C" fn mobile_generate(
    model: *mut QTransformer,
    tokens: *const usize,
    len: usize,
    max_tokens: usize,
) -> TokenArray {
    if model.is_null() || tokens.is_null() {
        return TokenArray { ptr: std::ptr::null_mut(), len: 0 };
    }
    let model = unsafe { &mut *model };
    let input = unsafe { std::slice::from_raw_parts(tokens, len) };
    let out = model.generate(input, max_tokens);
    let len = out.len();
    let mut slice = out.into_boxed_slice();
    let ptr = slice.as_mut_ptr();
    std::mem::forget(slice);
    TokenArray { ptr, len }
}

#[no_mangle]
pub extern "C" fn tokenizer_new(tokens: *const *const c_char, len: usize) -> *mut Tokenizer {
    if tokens.is_null() {
        return std::ptr::null_mut();
    }
    let slice = unsafe { std::slice::from_raw_parts(tokens, len) };
    let mut vec = Vec::with_capacity(len);
    for &ptr in slice {
        if ptr.is_null() {
            return std::ptr::null_mut();
        }
        let s = unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() };
        vec.push(s);
    }
    Box::into_raw(Box::new(Tokenizer::new(vec)))
}

#[no_mangle]
pub extern "C" fn tokenizer_free(t: *mut Tokenizer) {
    if !t.is_null() {
        unsafe { drop(Box::from_raw(t)); }
    }
}

#[no_mangle]
pub extern "C" fn tokenizer_encode(
    tokenizer: *mut Tokenizer,
    text: *const c_char,
) -> TokenArray {
    if tokenizer.is_null() || text.is_null() {
        return TokenArray { ptr: std::ptr::null_mut(), len: 0 };
    }
    let tok = unsafe { &mut *tokenizer };
    let text = unsafe { CStr::from_ptr(text).to_string_lossy() };
    let out = tok.encode(&text);
    let len = out.len();
    let mut slice = out.into_boxed_slice();
    let ptr = slice.as_mut_ptr();
    std::mem::forget(slice);
    TokenArray { ptr, len }
}

#[no_mangle]
pub extern "C" fn tokenizer_decode(
    tokenizer: *mut Tokenizer,
    tokens: *const usize,
    len: usize,
) -> *mut c_char {
    if tokenizer.is_null() || tokens.is_null() {
        return std::ptr::null_mut();
    }
    let tok = unsafe { &mut *tokenizer };
    let slice = unsafe { std::slice::from_raw_parts(tokens, len) };
    let text = tok.decode(slice);
    CString::new(text).unwrap().into_raw()
}

/// Simple whitespace tokenizer backed by a fixed vocabulary.
pub struct Tokenizer {
    vocab: HashMap<String, usize>,
    inv_vocab: Vec<String>,
    unk_id: usize,
}

impl Tokenizer {
    /// Create a tokenizer from a list of tokens. The first entry is used as
    /// the unknown token.
    pub fn new(tokens: Vec<String>) -> Self {
        let mut vocab = HashMap::with_capacity(tokens.len());
        for (i, tok) in tokens.iter().enumerate() {
            vocab.insert(tok.clone(), i);
        }
        Self { vocab, inv_vocab: tokens, unk_id: 0 }
    }

    /// Encode a string into token ids using whitespace splitting. Unknown
    /// tokens map to the `unk` id.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.split_whitespace()
            .map(|s| self.vocab.get(s).cloned().unwrap_or(self.unk_id))
            .collect()
    }

    /// Decode token ids back into a space separated string.
    pub fn decode(&self, tokens: &[usize]) -> String {
        tokens
            .iter()
            .map(|&id| {
                self.inv_vocab
                    .get(id)
                    .map(|s| s.as_str())
                    .unwrap_or(&self.inv_vocab[self.unk_id])
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}
