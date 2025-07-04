use inference_re::model::{ModelArgs, Transformer};
use mobile::*;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use tempfile::NamedTempFile;

#[test]
fn ffi_model_and_tokenizer() {
    let model = Transformer::new(ModelArgs::new());
    let q = QTransformer::from_model(&model);
    let tmp = NamedTempFile::new().unwrap();
    q.save(&tmp.path().to_path_buf()).unwrap();

    unsafe {
        let path = CString::new(tmp.path().to_str().unwrap()).unwrap();
        let m = mobile_model_load(path.as_ptr());
        assert!(!m.is_null());

        let vocab = [
            CString::new("<unk>").unwrap(),
            CString::new("hello").unwrap(),
            CString::new("world").unwrap(),
        ];
        let ptrs: Vec<*const c_char> = vocab.iter().map(|s| s.as_ptr()).collect();
        let tok = tokenizer_new(ptrs.as_ptr(), ptrs.len());
        assert!(!tok.is_null());

        let text = CString::new("hello").unwrap();
        let ids = tokenizer_encode(tok, text.as_ptr());
        assert_eq!(ids.len, 1);

        let generated = mobile_generate(m, ids.ptr, ids.len, 2);
        assert!(generated.len >= ids.len);

        let decoded_ptr = tokenizer_decode(tok, generated.ptr, generated.len);
        assert!(!decoded_ptr.is_null());
        let _decoded = CStr::from_ptr(decoded_ptr).to_str().unwrap();

        string_free(decoded_ptr);
        token_array_free(ids);
        token_array_free(generated);
        assert_eq!(tokenizer_vocab_size(tok), 3);
        tokenizer_free(tok);
        mobile_model_free(m);
    }
}

#[test]
fn ffi_tokenizer_vocab_size() {
    unsafe {
        let vocab = [
            CString::new("<unk>").unwrap(),
            CString::new("a").unwrap(),
            CString::new("b").unwrap(),
            CString::new("c").unwrap(),
        ];
        let ptrs: Vec<*const c_char> = vocab.iter().map(|s| s.as_ptr()).collect();
        let tok = tokenizer_new(ptrs.as_ptr(), ptrs.len());
        assert!(!tok.is_null());
        assert_eq!(tokenizer_vocab_size(tok), 4);
        tokenizer_free(tok);
    }
}

#[test]
fn ffi_tokenizer_contains() {
    unsafe {
        let vocab = [
            CString::new("<unk>").unwrap(),
            CString::new("hello").unwrap(),
        ];
        let ptrs: Vec<*const c_char> = vocab.iter().map(|s| s.as_ptr()).collect();
        let tok = tokenizer_new(ptrs.as_ptr(), ptrs.len());
        assert!(!tok.is_null());
        let ok = CString::new("hello").unwrap();
        assert!(tokenizer_contains(tok, ok.as_ptr()));
        let missing = CString::new("world").unwrap();
        assert!(!tokenizer_contains(tok, missing.as_ptr()));
        tokenizer_free(tok);
    }
}

#[test]
fn ffi_tokenizer_token_id() {
    unsafe {
        let vocab = [CString::new("<unk>").unwrap(), CString::new("x").unwrap()];
        let ptrs: Vec<*const c_char> = vocab.iter().map(|s| s.as_ptr()).collect();
        let tok = tokenizer_new(ptrs.as_ptr(), ptrs.len());
        assert!(!tok.is_null());
        let token = CString::new("x").unwrap();
        assert_eq!(tokenizer_token_id(tok, token.as_ptr()), 1);
        let missing = CString::new("y").unwrap();
        assert_eq!(tokenizer_token_id(tok, missing.as_ptr()), -1);
        tokenizer_free(tok);
    }
}
