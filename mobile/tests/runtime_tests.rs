use inference_re::model::{ModelArgs, Transformer};
use mobile::QTransformer;
use tempfile::NamedTempFile;

#[test]
fn test_save_and_load() -> std::io::Result<()> {
    let model = Transformer::new(ModelArgs::new());
    let q = QTransformer::from_model(&model);
    let file = NamedTempFile::new()?;
    q.save(&file.path().to_path_buf())?;
    let loaded = QTransformer::load(&file.path().to_path_buf())?;
    let tokens = vec![1usize, 2, 3];
    let out = loaded.forward(&tokens);
    assert_eq!(out.nrows(), tokens.len());
    Ok(())
}

#[test]
fn test_generate_length() -> std::io::Result<()> {
    let model = Transformer::new(ModelArgs::new());
    let q = QTransformer::from_model(&model);
    let prompt = vec![0usize];
    let out = q.generate(&prompt, 4);
    assert_eq!(out.len(), 5);
    Ok(())
}
