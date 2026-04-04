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

/// Parse LFS SHA-256 entries from a HuggingFace API JSON response.
///
/// Iterates `siblings` array and collects `rfilename -> lfs.sha256` for every
/// sibling that has an `lfs.sha256` field.  Returns an empty map when the JSON
/// structure is unexpected or when no siblings have LFS metadata.
///
/// This is a pure function so it can be unit-tested without making HTTP requests.
pub fn parse_lfs_sha256_map(json: &serde_json::Value) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();

    if let Some(siblings) = json["siblings"].as_array() {
        for sibling in siblings {
            if let (Some(name), Some(sha256)) = (
                sibling["rfilename"].as_str(),
                sibling["lfs"]["sha256"].as_str(),
            ) {
                if !sha256.is_empty() {
                    map.insert(name.to_string(), sha256.to_string());
                }
            }
        }
    }

    map
}

/// Fetch LFS SHA-256 checksums for all files in `repo_id` from the HF API.
///
/// Uses a blocking HTTP request.  Returns an empty map on any failure so that
/// callers can degrade gracefully rather than blocking the download.
///
/// This function is `#[cfg(feature = "providers")]` because it requires the
/// optional `reqwest` dependency.
#[cfg(feature = "providers")]
fn fetch_lfs_sha256_map_blocking(
    repo_id: &str,
) -> anyhow::Result<std::collections::HashMap<String, String>> {
    let client = reqwest::blocking::Client::new();
    let url = format!("https://huggingface.co/api/models/{repo_id}");
    let mut req = client.get(&url);
    if let Ok(token) = std::env::var("HF_TOKEN") {
        req = req.bearer_auth(token);
    }
    let resp = req.send()?.json::<serde_json::Value>()?;
    Ok(parse_lfs_sha256_map(&resp))
}

/// Download all relevant model files for `repo_id` and return the local cache directory.
///
/// Files are cached under `~/.cache/crabinfer/<repo_id>/`. Already-cached files
/// are not re-downloaded (the `hf-hub` crate handles this automatically via its
/// etag-based caching).
///
/// After each safetensors file is downloaded its SHA-256 digest is verified
/// against the LFS metadata from the HuggingFace API.  If the API is
/// unreachable a warning is logged and verification is skipped so that
/// air-gapped or rate-limited environments are not blocked.
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

    // Fetch LFS SHA-256 metadata before the download loop.
    // On failure we warn and continue with an empty map (graceful degradation).
    let repo_id_for_sha = repo_id.to_string();
    let lfs_sha_map = tokio::task::spawn_blocking(move || {
        fetch_lfs_sha256_map_blocking(&repo_id_for_sha)
    })
    .await
    .unwrap_or_else(|_| Ok(std::collections::HashMap::new()))
    .unwrap_or_else(|e| {
        tracing::warn!(
            "Could not fetch LFS checksums for {repo_id}, skipping SHA-256 verification: {e}"
        );
        std::collections::HashMap::new()
    });

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

        // Verify SHA-256 for safetensors files when we have an LFS checksum.
        let lower = filename.to_lowercase();
        if lower.ends_with(".safetensors") {
            if let Some(expected_sha) = lfs_sha_map.get(filename.as_str()) {
                verify_sha256(&local_path, expected_sha)
                    .with_context(|| format!("Integrity check failed for {filename}"))?;
                tracing::info!("SHA-256 verified: {}", filename);
            }
        }
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

    // ── parse_lfs_sha256_map ───────────────────────────────────────────────────

    /// Test 3: parse_lfs_sha256_map correctly extracts LFS sha256 entries from
    /// a HF API response with a mix of LFS and non-LFS files.
    #[test]
    fn test_parse_lfs_sha256_from_json() {
        let json_str = r#"{
            "siblings": [
                {
                    "rfilename": "model.safetensors",
                    "lfs": {
                        "sha256": "abc123def456abc123def456abc123def456abc123def456abc123def456abc1",
                        "size": 1234567890,
                        "pointer_size": 134
                    }
                },
                {
                    "rfilename": "config.json"
                },
                {
                    "rfilename": "pytorch_model.bin",
                    "lfs": {
                        "sha256": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
                        "size": 9876543210,
                        "pointer_size": 134
                    }
                }
            ]
        }"#;

        let json: serde_json::Value = serde_json::from_str(json_str).expect("valid JSON");
        let map = parse_lfs_sha256_map(&json);

        // safetensors file with LFS should be present
        assert_eq!(
            map.get("model.safetensors"),
            Some(&"abc123def456abc123def456abc123def456abc123def456abc123def456abc1".to_string()),
            "safetensors LFS sha256 should be in map"
        );

        // pytorch_model.bin with LFS should also be present
        assert_eq!(
            map.get("pytorch_model.bin"),
            Some(&"deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef".to_string()),
            "pytorch_model.bin LFS sha256 should be in map"
        );

        // config.json has no LFS metadata — must NOT be in map
        assert!(
            !map.contains_key("config.json"),
            "config.json (no LFS) should not be in map"
        );
    }

    /// Test 4: parse_lfs_sha256_map returns an empty map when no siblings have
    /// LFS metadata (tokenizer.json, config.json, etc.).
    #[test]
    fn test_parse_lfs_sha256_non_lfs_files_empty_map() {
        let json_str = r#"{
            "siblings": [
                { "rfilename": "config.json" },
                { "rfilename": "tokenizer.json" },
                { "rfilename": "tokenizer_config.json" },
                { "rfilename": "special_tokens_map.json" }
            ]
        }"#;

        let json: serde_json::Value = serde_json::from_str(json_str).expect("valid JSON");
        let map = parse_lfs_sha256_map(&json);

        assert!(
            map.is_empty(),
            "Non-LFS files should produce an empty sha256 map, got: {:?}",
            map
        );
    }

    /// Edge case: empty siblings array returns empty map.
    #[test]
    fn test_parse_lfs_sha256_empty_siblings() {
        let json: serde_json::Value = serde_json::from_str(r#"{"siblings": []}"#).unwrap();
        assert!(parse_lfs_sha256_map(&json).is_empty());
    }

    /// Edge case: missing siblings key returns empty map.
    #[test]
    fn test_parse_lfs_sha256_missing_siblings_key() {
        let json: serde_json::Value = serde_json::from_str(r#"{"sha": "abc"}"#).unwrap();
        assert!(parse_lfs_sha256_map(&json).is_empty());
    }
}
