use super::QTransformer;
use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::path::PathBuf;

#[repr(C)]
pub struct InferenceEngine {
    model: QTransformer,
}

#[no_mangle]
pub extern "C" fn inference_new(model_path: *const c_char) -> *mut InferenceEngine {
    if model_path.is_null() {
        return std::ptr::null_mut();
    }
    let c_str = unsafe { CStr::from_ptr(model_path) };
    let path = PathBuf::from(c_str.to_string_lossy().into_owned());
    match QTransformer::load_mmap(&path).or_else(|_| QTransformer::load(&path)) {
        Ok(model) => Box::into_raw(Box::new(InferenceEngine { model })),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn inference_run(
    engine: *mut InferenceEngine,
    input_ids: *const usize,
    len: usize,
    out_ptr: *mut usize,
    out_len: *mut usize,
) -> c_int {
    if engine.is_null() || input_ids.is_null() || out_ptr.is_null() || out_len.is_null() {
        return -1;
    }
    let engine = unsafe { &mut *engine };
    let input = unsafe { std::slice::from_raw_parts(input_ids, len) };
    let output = engine.model.generate(input, 1);
    unsafe {
        std::ptr::copy_nonoverlapping(output.as_ptr(), out_ptr, output.len());
        *out_len = output.len();
    }
    0
}

#[no_mangle]
pub extern "C" fn inference_free(engine: *mut InferenceEngine) {
    if !engine.is_null() {
        unsafe { drop(Box::from_raw(engine)); }
    }
}
