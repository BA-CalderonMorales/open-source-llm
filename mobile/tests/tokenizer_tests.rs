use mobile::Tokenizer;

#[test]
fn encode_decode_roundtrip() {
    let tokens = vec!["<unk>".to_string(), "hello".to_string(), "world".to_string()];
    let tokenizer = Tokenizer::new(tokens);
    let ids = tokenizer.encode("hello world");
    assert_eq!(ids, vec![1, 2]);
    let text = tokenizer.decode(&ids);
    assert_eq!(text, "hello world");
}

#[test]
fn unknown_token() {
    let tokens = vec!["<unk>".to_string(), "foo".to_string()];
    let tokenizer = Tokenizer::new(tokens);
    let ids = tokenizer.encode("bar");
    assert_eq!(ids, vec![0]);
    let text = tokenizer.decode(&ids);
    assert_eq!(text, "<unk>");
}
