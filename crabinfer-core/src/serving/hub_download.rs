//! HuggingFace Hub download client.
//!
//! Provides utilities for detecting HF repo IDs, downloading model files to a
//! local cache, and verifying file integrity with SHA-256.

/// Returns `true` if `model_path` looks like a HuggingFace repo ID
/// (e.g. `"meta-llama/Llama-3.1-8B-Instruct"`), `false` if it's a local path.
///
/// Heuristic:
/// - Must contain exactly one `/`
/// - Must not start with `/` or `.`
/// - Must not contain backslashes
/// - Must not end with a file extension like `.gguf` or `.safetensors`
pub fn is_hf_repo_id(model_path: &str) -> bool {
    // Must not start with path separators or relative markers
    if model_path.starts_with('/') || model_path.starts_with('.') || model_path.starts_with('\\') {
        return false;
    }

    // Must not contain backslashes (Windows paths)
    if model_path.contains('\\') {
        return false;
    }

    // Must not look like a file (has a recognized model extension)
    let lower = model_path.to_lowercase();
    if lower.ends_with(".gguf")
        || lower.ends_with(".safetensors")
        || lower.ends_with(".bin")
        || lower.ends_with(".pt")
        || lower.ends_with(".pth")
    {
        return false;
    }

    // Count slashes — must be exactly one (org/model)
    let slash_count = model_path.chars().filter(|&c| c == '/').count();
    if slash_count != 1 {
        return false;
    }

    // Both org and model name must be non-empty
    let parts: Vec<&str> = model_path.splitn(2, '/').collect();
    if parts.len() != 2 || parts[0].is_empty() || parts[1].is_empty() {
        return false;
    }

    true
}

/// Returns `true` if the given filename should be downloaded from the Hub.
///
/// Downloads: model weights, config, tokenizer, and related JSON metadata.
/// Skips: documentation, git files, and other non-essential files.
pub fn should_download(filename: &str) -> bool {
    let lower = filename.to_lowercase();

    // Always download safetensors weight files
    if lower.ends_with(".safetensors") {
        return true;
    }

    // Essential JSON config/tokenizer files
    matches!(
        filename,
        "config.json"
            | "tokenizer.json"
            | "tokenizer_config.json"
            | "quantize_config.json"
            | "generation_config.json"
            | "model.safetensors.index.json"
            | "special_tokens_map.json"
    )
}

/// Download all relevant model files for `repo_id` and return the local cache directory.
///
/// Files are cached under `~/.cache/crabinfer/<repo_id>/`. Already-cached files
/// are not re-downloaded (the `hf-hub` crate handles this automatically via its
/// etag-based caching).
///
/// The `HF_TOKEN` environment variable is read automatically by `hf-hub` for
/// gated/private models.
#[cfg(feature = "providers")]
pub async fn ensure_model_cached(repo_id: &str) -> anyhow::Result<std::path::PathBuf> {
    use anyhow::Context;
    use hf_hub::api::tokio::ApiBuilder;

    let cache_dir = dirs::cache_dir()
        .context("Cannot determine user cache directory")?
        .join("crabinfer");

    tracing::info!(
        "Downloading model {} to {}",
        repo_id,
        cache_dir.display()
    );

    let api = ApiBuilder::new()
        .with_cache_dir(cache_dir.clone())
        .build()
        .context("Failed to build HuggingFace Hub API client")?;

    let repo = api.model(repo_id.to_string());

    // Fetch the repository file listing
    let files = repo
        .info()
        .await
        .context("Failed to fetch repository info from HuggingFace Hub")?;

    // Download each relevant file
    for sibling in &files.siblings {
        let filename = &sibling.rfilename;
        if !should_download(filename) {
            tracing::debug!("Skipping: {}", filename);
            continue;
        }
        tracing::info!("Downloading: {}", filename);
        let local_path = repo
            .get(filename)
            .await
            .with_context(|| format!("Failed to download {filename} from {repo_id}"))?;
        tracing::info!("Cached: {}", local_path.display());
    }

    // The local directory is the snapshot directory for the model
    // hf-hub stores files at: <cache_dir>/models--<org>--<model>/snapshots/<hash>/
    // We find the snapshot directory by looking for the resolved path of config.json
    let config_path = repo
        .get("config.json")
        .await
        .context("Failed to resolve config.json path for cache directory")?;

    let local_dir = config_path
        .parent()
        .context("Cannot determine parent directory of config.json")?
        .to_path_buf();

    tracing::info!("Model cached at: {}", local_dir.display());
    Ok(local_dir)
}

/// Verify that a file's SHA-256 digest matches `expected_hex`.
///
/// Reads the file in 64KB chunks to avoid loading large model files into memory.
/// Returns an error with a descriptive message if the digest does not match or
/// the file cannot be read.
pub fn verify_sha256(file_path: &std::path::Path, expected_hex: &str) -> anyhow::Result<()> {
    use anyhow::Context;
    use sha2::{Digest, Sha256};
    use std::io::Read;

    let mut file = std::fs::File::open(file_path)
        .with_context(|| format!("Cannot open file for SHA-256 verification: {}", file_path.display()))?;

    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 65536]; // 64KB chunks

    loop {
        let n = file
            .read(&mut buffer)
            .with_context(|| format!("Error reading file during SHA-256 computation: {}", file_path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }

    let digest = format!("{:x}", hasher.finalize());
    let expected_lower = expected_hex.to_lowercase();

    if digest != expected_lower {
        anyhow::bail!(
            "SHA-256 mismatch for {}: expected {}, got {}",
            file_path.display(),
            expected_lower,
            digest
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── is_hf_repo_id ──────────────────────────────────────────────────────────

    #[test]
    fn test_is_hf_repo_id_valid() {
        assert!(is_hf_repo_id("meta-llama/Llama-3.1-8B-Instruct"));
        assert!(is_hf_repo_id("org/model-name-GPTQ"));
        assert!(is_hf_repo_id("TheBloke/Mistral-7B-v0.1-GPTQ"));
        assert!(is_hf_repo_id("microsoft/phi-2"));
    }

    #[test]
    fn test_is_hf_repo_id_local_absolute() {
        assert!(!is_hf_repo_id("/local/path/to/model"));
        assert!(!is_hf_repo_id("/models/llama"));
    }

    #[test]
    fn test_is_hf_repo_id_local_relative() {
        assert!(!is_hf_repo_id("./relative/path"));
        assert!(!is_hf_repo_id("../models/llama"));
    }

    #[test]
    fn test_is_hf_repo_id_file_extensions() {
        assert!(!is_hf_repo_id("model.gguf"));
        assert!(!is_hf_repo_id("model.safetensors"));
        assert!(!is_hf_repo_id("model.bin"));
    }

    #[test]
    fn test_is_hf_repo_id_too_many_slashes() {
        assert!(!is_hf_repo_id("a/b/c"));
        assert!(!is_hf_repo_id("org/model/variant"));
    }

    #[test]
    fn test_is_hf_repo_id_no_slash() {
        assert!(!is_hf_repo_id("modelname"));
        assert!(!is_hf_repo_id("localmodel"));
    }

    // ── should_download ────────────────────────────────────────────────────────

    #[test]
    fn test_should_download_model_files() {
        assert!(should_download("model.safetensors"));
        assert!(should_download("model-00001-of-00002.safetensors"));
        assert!(should_download("config.json"));
        assert!(should_download("tokenizer.json"));
        assert!(should_download("tokenizer_config.json"));
        assert!(should_download("quantize_config.json"));
        assert!(should_download("generation_config.json"));
        assert!(should_download("model.safetensors.index.json"));
        assert!(should_download("special_tokens_map.json"));
    }

    #[test]
    fn test_should_download_skip_files() {
        assert!(!should_download(".gitattributes"));
        assert!(!should_download("README.md"));
        assert!(!should_download(".git"));
        assert!(!should_download("LICENSE"));
        assert!(!should_download("pytorch_model.bin")); // Old-format bin files are not downloaded
    }

    // ── verify_sha256 ──────────────────────────────────────────────────────────

    #[cfg(feature = "providers")]
    #[test]
    fn test_verify_sha256_correct_hash() {
        use std::io::Write;
        let dir = tempfile::tempdir().expect("tempdir");
        let file_path = dir.path().join("test.bin");
        let content = b"hello, crabinfer!";
        std::fs::File::create(&file_path)
            .unwrap()
            .write_all(content)
            .unwrap();

        // Compute the expected hash using sha2 directly
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(content);
        let expected = format!("{:x}", hasher.finalize());

        assert!(
            verify_sha256(&file_path, &expected).is_ok(),
            "Should accept file with correct hash"
        );
    }

    #[test]
    fn test_verify_sha256_wrong_hash() {
        use std::io::Write;
        let dir = tempfile::tempdir().expect("tempdir");
        let file_path = dir.path().join("test.bin");
        std::fs::File::create(&file_path)
            .unwrap()
            .write_all(b"hello, crabinfer!")
            .unwrap();

        let result = verify_sha256(
            &file_path,
            "0000000000000000000000000000000000000000000000000000000000000000",
        );
        assert!(result.is_err(), "Should reject file with wrong hash");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("SHA-256 mismatch"),
            "Error message should mention SHA-256 mismatch: {}",
            err_msg
        );
    }

    #[test]
    fn test_verify_sha256_missing_file() {
        let result = verify_sha256(std::path::Path::new("/nonexistent/file.bin"), "abc123");
        assert!(result.is_err(), "Should fail for missing file");
    }
}
