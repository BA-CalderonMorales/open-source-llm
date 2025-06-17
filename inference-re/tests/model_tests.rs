use inference_re::model::{ModelArgs, Transformer};
use rand;

#[test]
fn test_forward_shapes() {
    let args = ModelArgs::new();
    let model = Transformer::new(args.clone());
    let tokens = vec![1_usize, 2, 3];
    let logits = model.forward(&tokens);
    assert_eq!(logits.nrows(), tokens.len());
    assert_eq!(logits.ncols(), args.vocab_size);
}

#[test]
fn test_generation_len() {
    let args = ModelArgs::new();
    let mut tokens = vec![0_usize];
    for _ in 0..5 {
        let next = rand::random::<usize>() % args.vocab_size;
        tokens.push(next);
    }
    assert_eq!(tokens.len(), 6);
}

