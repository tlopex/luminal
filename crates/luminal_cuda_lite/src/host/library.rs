//! Shared loading and automatic source builds for TVM FFI providers.
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow, ensure};

/// A prebuilt library path, or a `.json` recipe for `tvm_ffi.cpp.build`.
/// Recipe paths are relative to the recipe file; strings expand environment
/// variables. Source builds require Python's TVM FFI package and Ninja.
/// Compilation happens on first resolution, before CUDA graph capture.
/// Clones share the resolved artifact; SDK builds use its persistent cache.
/// Construct a new instance to resolve updated source/configuration files.
#[derive(Debug, Clone)]
pub struct TVMFFILibrary {
    path: PathBuf,
    built: Arc<Mutex<Option<PathBuf>>>,
}

impl TVMFFILibrary {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            built: Arc::default(),
        }
    }

    /// Resolve a prebuilt library directly, or automatically build a recipe.
    /// `LUMINAL_TVM_FFI_PYTHON` selects the Python executable; the SDK honors
    /// `CC`, `CXX`, `CUDA_HOME`, and `TVM_FFI_CACHE_DIR` for its toolchain/cache.
    pub fn resolve(&self) -> Result<PathBuf> {
        if self.path.extension().is_none_or(|ext| ext != "json") {
            return Ok(self.path.clone());
        }
        let mut built = self
            .built
            .lock()
            .map_err(|_| anyhow!("TVM FFI build lock poisoned"))?;
        if let Some(path) = built.as_ref() {
            return Ok(path.clone());
        }
        let python = std::env::var_os("LUMINAL_TVM_FFI_PYTHON")
            .unwrap_or_else(|| if cfg!(windows) { "python" } else { "python3" }.into());
        let output = std::process::Command::new(python)
            .args(["-c", include_str!("build.py")])
            .arg(&self.path)
            .output()
            .context("starting TVM FFI builder; install apache-tvm-ffi and ninja for the selected Python")?;
        ensure!(
            output.status.success(),
            "building TVM FFI recipe {} failed: {}",
            self.path.display(),
            String::from_utf8_lossy(&output.stderr)
        );
        let path = PathBuf::from(
            String::from_utf8(output.stdout)
                .context("TVM FFI build path must be UTF-8")?
                .trim(),
        );
        ensure!(
            path.is_file(),
            "TVM FFI builder returned no library: {}",
            path.display()
        );
        *built = Some(path.clone());
        Ok(path)
    }

    /// Load the resolved artifact; the SDK's functions retain their module.
    pub fn load(&self) -> Result<tvm_ffi::Module> {
        let path = self.resolve()?;
        tvm_ffi::Module::load_from_file(
            path.to_str()
                .context("TVM FFI library path must be UTF-8")?,
        )
        .map_err(|e| anyhow!("loading TVM FFI library {}: {e}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_relative_sources_caches_artifacts_and_loads_prebuilt_libraries() {
        struct TempDir(PathBuf);
        impl Drop for TempDir {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = TempDir(
            std::env::temp_dir().join(format!("luminal-ffi-build-{}-{id}", std::process::id())),
        );
        std::fs::create_dir_all(dir.0.join("include")).unwrap();
        std::fs::write(dir.0.join("include/value.h"), "#define FACTOR 7\n").unwrap();
        std::fs::write(
            dir.0.join("provider.cc"),
            r#"
#include <tvm/ffi/function.h>
#include "value.h"
int64_t Apply(int64_t a, int64_t b) { return a * FACTOR + b; }
TVM_FFI_DLL_EXPORT_TYPED_FUNC(apply, Apply);
"#,
        )
        .unwrap();
        let recipe = dir.0.join("provider.json");
        std::fs::write(&recipe, r#"{"name":"luminal_ffi_build_test","sources":["provider.cc"],"extra_include_paths":["include"]}"#).unwrap();
        let library = TVMFFILibrary::new(&recipe);
        let artifact = library.resolve().unwrap();
        let modified = std::fs::metadata(&artifact).unwrap().modified().unwrap();
        assert_eq!(library.clone().resolve().unwrap(), artifact);
        assert_eq!(TVMFFILibrary::new(&recipe).resolve().unwrap(), artifact);
        assert_eq!(
            std::fs::metadata(&artifact).unwrap().modified().unwrap(),
            modified
        );
        let module = TVMFFILibrary::new(&artifact).load().unwrap();
        let function = module.get_function("apply").unwrap();
        let (a, b) = (12i64, 3i64);
        let result = function
            .call_packed(&[tvm_ffi::AnyView::from(&a), tvm_ffi::AnyView::from(&b)])
            .unwrap();
        assert_eq!(i64::try_from(result).unwrap(), 87);
        let invalid = dir.0.join("invalid.json");
        std::fs::write(
            &invalid,
            r#"{"name":"invalid_provider","sources":["absent.cc"]}"#,
        )
        .unwrap();
        let error = TVMFFILibrary::new(invalid).resolve().unwrap_err();
        assert!(error.to_string().contains("absent.cc"), "{error:#}");
    }
}
