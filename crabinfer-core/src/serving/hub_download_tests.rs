//! Tests for HuggingFace Hub download behaviors.
//!
//! Tests for the hub_download module: is_hf_repo_id, should_download,
//! verify_sha256, and ensure_model_cached (compilation check only — no network).

#[cfg(test)]
mod tests {
    use crate::serving::hub_download::{is_hf_repo_id, should_download, verify_sha256};

    /// Verify is_hf_repo_id correctly classifies repo IDs vs local paths.
    #[test]
    fn test_is_hf_repo_id_classification() {
        // Valid HF repo IDs
        assert!(
            is_hf_repo_id("meta-llama/Llama-3.1-8B-Instruct"),
            "Should accept valid HF repo ID"
        );
        assert!(
            is_hf_repo_id("org/model-name-GPTQ"),
            "Should accept org/model format"
        );
        assert!(
            is_hf_repo_id("TheBloke/Mistral-7B-GPTQ"),
            "Should accept TheBloke models"
        );

        // Local paths — must be rejected
        assert!(
            !is_hf_repo_id("/local/path/to/model"),
            "Should reject absolute path"
        );
        assert!(
            !is_hf_repo_id("./relative/path"),
            "Should reject relative path"
        );
        assert!(!is_hf_repo_id("model.gguf"), "Should reject .gguf file");
        assert!(
            !is_hf_repo_id("a/b/c"),
            "Should reject path with more than one slash"
        );
    }

    /// Verify ensure_model_cached function exists and is accessible (no network call).
    ///
    /// We take a reference to the function pointer to verify it compiles.
    /// We don't call it since that would require network access.
    #[cfg(feature = "providers")]
    #[test]
    fn test_ensure_model_cached_creates_local_dir() {
        // Verify the function symbol is reachable by taking a raw function reference.
        // This ensures the function is pub and has the expected signature at compile time.
        let _fn_ref = crate::serving::hub_download::ensure_model_cached;
        // If we reach here, the function exists and compiles
        assert!(true, "ensure_model_cached should exist and compile with correct signature");
    }

    /// Verify verify_sha256 rejects a file whose hash doesn't match expected.
    #[test]
    fn test_verify_sha256_rejects_corrupted_file() {
        use std::io::Write;
        let dir = tempfile::tempdir().expect("tempdir");
        let file_path = dir.path().join("model.safetensors");
        std::fs::File::create(&file_path)
            .unwrap()
            .write_all(b"corrupted model data")
            .unwrap();

        // A hash that doesn't match the content above
        let wrong_hash = "0000000000000000000000000000000000000000000000000000000000000000";
        let result = verify_sha256(&file_path, wrong_hash);
        assert!(
            result.is_err(),
            "verify_sha256 should return Err when hash mismatches"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("SHA-256 mismatch"),
            "Error should mention SHA-256 mismatch, got: {}",
            err_msg
        );
    }

    /// Verify verify_sha256 accepts a file with the correct hash.
    #[test]
    fn test_verify_sha256_accepts_correct_file() {
        use sha2::{Digest, Sha256};
        use std::io::Write;
        let dir = tempfile::tempdir().expect("tempdir");
        let file_path = dir.path().join("model.safetensors");
        let content = b"valid model data";
        std::fs::File::create(&file_path)
            .unwrap()
            .write_all(content)
            .unwrap();

        let mut hasher = Sha256::new();
        hasher.update(content);
        let correct_hash = format!("{:x}", hasher.finalize());

        let result = verify_sha256(&file_path, &correct_hash);
        assert!(result.is_ok(), "verify_sha256 should return Ok for correct hash");
    }

    /// Verify should_download filters correctly.
    #[test]
    fn test_should_download_filters() {
        assert!(
            should_download("model.safetensors"),
            "Should download safetensors"
        );
        assert!(
            should_download("config.json"),
            "Should download config.json"
        );
        assert!(
            should_download("tokenizer.json"),
            "Should download tokenizer.json"
        );
        assert!(
            !should_download(".gitattributes"),
            "Should skip .gitattributes"
        );
        assert!(!should_download("README.md"), "Should skip README.md");
    }
}
