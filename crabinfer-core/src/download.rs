//! Model download manager with resume, integrity checks, and progress reporting.
//!
//! Behind `#[cfg(feature = "providers")]` — shares the `reqwest` dependency
//! with cloud providers. Swift wraps blocking calls in `Task.detached`.

use crate::catalog::CatalogEntry;
use crate::CrabInferError;

use sha2::{Digest, Sha256};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Types (UniFFI-exported)
// ---------------------------------------------------------------------------

/// Status of a model download.
#[derive(Debug, Clone, PartialEq, Eq, uniffi::Enum)]
pub enum DownloadStatus {
    NotDownloaded,
    Downloading,
    Downloaded,
    Failed,
}

/// A successfully downloaded model with local paths.
#[derive(Debug, Clone, uniffi::Record)]
pub struct DownloadedModel {
    /// Catalog entry ID.
    pub id: String,
    /// Absolute path to the GGUF model file.
    pub model_path: String,
    /// Absolute path to tokenizer.json.
    pub tokenizer_path: String,
    /// File size in bytes.
    pub size_bytes: u64,
    /// ISO 8601 download timestamp.
    pub downloaded_at: String,
}

/// Callback interface for download progress (implemented in Swift).
#[uniffi::export(callback_interface)]
pub trait DownloadProgressListener: Send + Sync {
    fn on_progress(&self, bytes_downloaded: u64, bytes_total: u64, phase: String);
}

// ---------------------------------------------------------------------------
// Download manager
// ---------------------------------------------------------------------------

/// Manages model downloads, storage, and lifecycle.
#[derive(uniffi::Object)]
pub struct ModelDownloadManager {
    storage_dir: PathBuf,
}

#[uniffi::export]
impl ModelDownloadManager {
    /// Create a download manager with the default storage directory.
    ///
    /// macOS: `~/Library/Application Support/CrabInfer/models/`
    /// Other: `~/.crabinfer/models/`
    #[uniffi::constructor]
    pub fn new() -> Result<Self, CrabInferError> {
        let storage_dir = default_storage_dir()?;
        fs::create_dir_all(&storage_dir).map_err(|e| CrabInferError::StorageError {
            reason: format!("Failed to create storage directory: {e}"),
        })?;
        Ok(Self { storage_dir })
    }

    /// Create a download manager with a custom storage directory.
    #[uniffi::constructor(name = "with_directory")]
    pub fn with_directory(directory: String) -> Result<Self, CrabInferError> {
        let storage_dir = PathBuf::from(directory);
        fs::create_dir_all(&storage_dir).map_err(|e| CrabInferError::StorageError {
            reason: format!("Failed to create storage directory: {e}"),
        })?;
        Ok(Self { storage_dir })
    }

    /// Download a model from HuggingFace. Supports resume via HTTP Range.
    ///
    /// Downloads the GGUF file and tokenizer.json, validates integrity,
    /// and returns a `DownloadedModel` with local paths.
    pub fn download(
        &self,
        entry: CatalogEntry,
        listener: Option<Box<dyn DownloadProgressListener>>,
    ) -> Result<DownloadedModel, CrabInferError> {
        let model_dir = self.storage_dir.join(&entry.id);
        fs::create_dir_all(&model_dir).map_err(|e| CrabInferError::StorageError {
            reason: format!("Failed to create model directory: {e}"),
        })?;

        let gguf_path = model_dir.join(&entry.gguf_file);
        let tokenizer_path = model_dir.join("tokenizer.json");

        // Phase 1: Download GGUF file
        let gguf_url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            entry.hf_repo, entry.gguf_file
        );
        if let Some(ref l) = listener {
            l.on_progress(0, entry.size_bytes, "downloading_model".into());
        }
        download_file_with_resume(&gguf_url, &gguf_path, entry.size_bytes, listener.as_deref())?;

        // Validate GGUF magic bytes
        validate_gguf(&gguf_path)?;

        // SHA256 verification (if hash provided)
        if !entry.sha256.is_empty() {
            if let Some(ref l) = listener {
                l.on_progress(entry.size_bytes, entry.size_bytes, "verifying".into());
            }
            verify_sha256(&gguf_path, &entry.sha256)?;
        }

        // Phase 2: Download tokenizer.json
        let tokenizer_url = format!(
            "https://huggingface.co/{}/resolve/main/tokenizer.json",
            entry.tokenizer_repo
        );
        if let Some(ref l) = listener {
            l.on_progress(entry.size_bytes, entry.size_bytes, "downloading_tokenizer".into());
        }
        download_file_with_resume(&tokenizer_url, &tokenizer_path, 0, None)?;

        // Validate tokenizer JSON
        validate_json(&tokenizer_path)?;

        // Write metadata
        let now = chrono_now_iso8601();
        let meta_path = model_dir.join("metadata.json");
        let meta = format!(
            r#"{{"id":"{}","downloaded_at":"{}","size_bytes":{}}}"#,
            entry.id, now, entry.size_bytes
        );
        fs::write(&meta_path, meta).map_err(|e| CrabInferError::StorageError {
            reason: format!("Failed to write metadata: {e}"),
        })?;

        if let Some(ref l) = listener {
            l.on_progress(entry.size_bytes, entry.size_bytes, "complete".into());
        }

        Ok(DownloadedModel {
            id: entry.id,
            model_path: gguf_path.to_string_lossy().into_owned(),
            tokenizer_path: tokenizer_path.to_string_lossy().into_owned(),
            size_bytes: fs::metadata(&gguf_path)
                .map(|m| m.len())
                .unwrap_or(entry.size_bytes),
            downloaded_at: now,
        })
    }

    /// Check if a model is downloaded.
    pub fn is_downloaded(&self, id: String) -> bool {
        let model_dir = self.storage_dir.join(&id);
        model_dir.join("metadata.json").exists()
    }

    /// Get the GGUF model path for a downloaded model.
    pub fn model_path(&self, id: String) -> Option<String> {
        let model_dir = self.storage_dir.join(&id);
        if !model_dir.join("metadata.json").exists() {
            return None;
        }
        // Find the .gguf file in the directory
        if let Ok(entries) = fs::read_dir(&model_dir) {
            for entry in entries.flatten() {
                if entry
                    .path()
                    .extension()
                    .is_some_and(|ext| ext == "gguf")
                {
                    return Some(entry.path().to_string_lossy().into_owned());
                }
            }
        }
        None
    }

    /// Get the tokenizer.json path for a downloaded model.
    pub fn tokenizer_path(&self, id: String) -> Option<String> {
        let path = self.storage_dir.join(&id).join("tokenizer.json");
        if path.exists() {
            Some(path.to_string_lossy().into_owned())
        } else {
            None
        }
    }

    /// List all downloaded models.
    pub fn list_downloaded(&self) -> Vec<DownloadedModel> {
        let mut result = Vec::new();
        let Ok(entries) = fs::read_dir(&self.storage_dir) else {
            return result;
        };
        for dir_entry in entries.flatten() {
            let dir_path = dir_entry.path();
            if !dir_path.is_dir() {
                continue;
            }
            let meta_path = dir_path.join("metadata.json");
            if !meta_path.exists() {
                continue;
            }
            let id = dir_path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();

            let (downloaded_at, size_bytes) = parse_metadata(&meta_path);

            let model_path = self.model_path(id.clone()).unwrap_or_default();
            let tokenizer_path = self.tokenizer_path(id.clone()).unwrap_or_default();

            result.push(DownloadedModel {
                id,
                model_path,
                tokenizer_path,
                size_bytes,
                downloaded_at,
            });
        }
        result
    }

    /// Delete a downloaded model.
    pub fn delete(&self, id: String) -> Result<(), CrabInferError> {
        let model_dir = self.storage_dir.join(&id);
        if !model_dir.exists() {
            return Ok(());
        }
        fs::remove_dir_all(&model_dir).map_err(|e| CrabInferError::StorageError {
            reason: format!("Failed to delete model {id}: {e}"),
        })
    }

    /// Total disk usage of all downloaded models in bytes.
    pub fn total_disk_usage(&self) -> u64 {
        dir_size(&self.storage_dir)
    }

    /// The storage directory path.
    pub fn storage_directory(&self) -> String {
        self.storage_dir.to_string_lossy().into_owned()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Default storage directory for downloaded models.
fn default_storage_dir() -> Result<PathBuf, CrabInferError> {
    #[cfg(target_os = "macos")]
    {
        if let Some(home) = dirs_home() {
            return Ok(home
                .join("Library")
                .join("Application Support")
                .join("CrabInfer")
                .join("models"));
        }
    }

    #[cfg(target_os = "ios")]
    {
        // On iOS, use the app's Documents directory (sandbox)
        if let Some(home) = dirs_home() {
            return Ok(home.join("Documents").join("CrabInfer").join("models"));
        }
    }

    // Fallback for Linux / other
    if let Some(home) = dirs_home() {
        return Ok(home.join(".crabinfer").join("models"));
    }

    Err(CrabInferError::StorageError {
        reason: "Cannot determine home directory".into(),
    })
}

/// Get the user's home directory.
fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

/// Download a file with HTTP Range resume support.
fn download_file_with_resume(
    url: &str,
    dest: &Path,
    expected_size: u64,
    listener: Option<&dyn DownloadProgressListener>,
) -> Result<(), CrabInferError> {
    let partial_path = dest.with_extension("partial");

    // Check if we can resume from a partial download
    let existing_bytes = if partial_path.exists() {
        fs::metadata(&partial_path)
            .map(|m| m.len())
            .unwrap_or(0)
    } else {
        0
    };

    // If the final file already exists and looks complete, skip
    if dest.exists() {
        let file_size = fs::metadata(dest).map(|m| m.len()).unwrap_or(0);
        if expected_size == 0 || file_size >= expected_size {
            return Ok(());
        }
    }

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(3600)) // 1 hour for large models
        .build()
        .map_err(|e| CrabInferError::DownloadFailed {
            reason: format!("HTTP client error: {e}"),
        })?;

    let mut request = client.get(url);
    if existing_bytes > 0 {
        request = request.header("Range", format!("bytes={}-", existing_bytes));
    }

    let response = request.send().map_err(|e| CrabInferError::DownloadFailed {
        reason: format!("Download request failed: {e}"),
    })?;

    let status = response.status();
    if !status.is_success() && status.as_u16() != 206 {
        return Err(CrabInferError::DownloadFailed {
            reason: format!("HTTP {} from {url}", status),
        });
    }

    // Determine total size from Content-Length or Content-Range
    let content_length = response.content_length().unwrap_or(0);
    let total_size = if status.as_u16() == 206 {
        // Partial content — total is existing + remaining
        existing_bytes + content_length
    } else if content_length > 0 {
        content_length
    } else {
        expected_size
    };

    // If server didn't honor Range (sent 200 instead of 206), start fresh
    let resume_offset = if status.as_u16() == 206 {
        existing_bytes
    } else {
        0
    };

    let mut file = if resume_offset > 0 {
        fs::OpenOptions::new()
            .append(true)
            .open(&partial_path)
            .map_err(|e| CrabInferError::StorageError {
                reason: format!("Failed to open partial file: {e}"),
            })?
    } else {
        fs::File::create(&partial_path).map_err(|e| CrabInferError::StorageError {
            reason: format!("Failed to create file: {e}"),
        })?
    };

    // Stream the response body to disk
    let mut downloaded = resume_offset;
    let mut reader = response;
    let mut buf = vec![0u8; 256 * 1024]; // 256 KB buffer
    loop {
        let n = reader.read(&mut buf).map_err(|e| CrabInferError::DownloadFailed {
            reason: format!("Read error: {e}"),
        })?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .map_err(|e| CrabInferError::StorageError {
                reason: format!("Write error: {e}"),
            })?;
        downloaded += n as u64;

        if let Some(l) = listener {
            l.on_progress(downloaded, total_size, "downloading_model".into());
        }
    }

    // Rename partial to final
    fs::rename(&partial_path, dest).map_err(|e| CrabInferError::StorageError {
        reason: format!("Failed to rename partial file: {e}"),
    })?;

    Ok(())
}

/// Validate GGUF magic bytes (first 4 bytes should be "GGUF").
fn validate_gguf(path: &Path) -> Result<(), CrabInferError> {
    let mut file = fs::File::open(path).map_err(|e| CrabInferError::IntegrityCheckFailed {
        reason: format!("Cannot open GGUF file: {e}"),
    })?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)
        .map_err(|e| CrabInferError::IntegrityCheckFailed {
            reason: format!("Cannot read GGUF header: {e}"),
        })?;
    if &magic != b"GGUF" {
        return Err(CrabInferError::IntegrityCheckFailed {
            reason: format!(
                "Invalid GGUF magic bytes: expected 'GGUF', got '{}'",
                String::from_utf8_lossy(&magic)
            ),
        });
    }
    Ok(())
}

/// Validate that a file looks like JSON (starts with '{' or '[').
fn validate_json(path: &Path) -> Result<(), CrabInferError> {
    let mut file = fs::File::open(path).map_err(|e| CrabInferError::IntegrityCheckFailed {
        reason: format!("Cannot open tokenizer file: {e}"),
    })?;
    let mut first = [0u8; 1];
    file.read_exact(&mut first)
        .map_err(|e| CrabInferError::IntegrityCheckFailed {
            reason: format!("Cannot read tokenizer file: {e}"),
        })?;
    if first[0] != b'{' && first[0] != b'[' {
        return Err(CrabInferError::IntegrityCheckFailed {
            reason: "Tokenizer file does not look like valid JSON".into(),
        });
    }
    Ok(())
}

/// Verify SHA256 hash of a file.
fn verify_sha256(path: &Path, expected: &str) -> Result<(), CrabInferError> {
    let mut file = fs::File::open(path).map_err(|e| CrabInferError::IntegrityCheckFailed {
        reason: format!("Cannot open file for SHA256: {e}"),
    })?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024]; // 1 MB buffer
    loop {
        let n = file.read(&mut buf).map_err(|e| CrabInferError::IntegrityCheckFailed {
            reason: format!("Read error during SHA256: {e}"),
        })?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let hash = format!("{:x}", hasher.finalize());
    if hash != expected {
        return Err(CrabInferError::IntegrityCheckFailed {
            reason: format!("SHA256 mismatch: expected {expected}, got {hash}"),
        });
    }
    Ok(())
}

/// Parse metadata.json to extract downloaded_at and size_bytes.
fn parse_metadata(path: &Path) -> (String, u64) {
    let Ok(content) = fs::read_to_string(path) else {
        return (String::new(), 0);
    };
    let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) else {
        return (String::new(), 0);
    };
    let downloaded_at = val
        .get("downloaded_at")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let size_bytes = val
        .get("size_bytes")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    (downloaded_at, size_bytes)
}

/// Recursively compute directory size in bytes.
fn dir_size(path: &Path) -> u64 {
    let mut total = 0;
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                total += dir_size(&p);
            } else {
                total += fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
            }
        }
    }
    total
}

/// ISO 8601 timestamp without external chrono dependency.
fn chrono_now_iso8601() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    // Simple UTC timestamp: YYYY-MM-DDTHH:MM:SSZ
    let days = secs / 86400;
    let time_secs = secs % 86400;
    let hours = time_secs / 3600;
    let minutes = (time_secs % 3600) / 60;
    let seconds = time_secs % 60;

    // Convert days since epoch to date (simplified — good enough for metadata)
    let (year, month, day) = days_to_date(days);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hours, minutes, seconds
    )
}

/// Convert days since Unix epoch to (year, month, day).
fn days_to_date(mut days: u64) -> (u64, u64, u64) {
    let mut year = 1970;
    loop {
        let days_in_year = if is_leap(year) { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }
    let month_days: &[u64] = if is_leap(year) {
        &[31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        &[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut month = 1;
    for &md in month_days {
        if days < md {
            break;
        }
        days -= md;
        month += 1;
    }
    (year, month, days + 1)
}

fn is_leap(year: u64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_storage_dir() {
        let dir = default_storage_dir();
        assert!(dir.is_ok());
        let dir = dir.unwrap();
        let dir_str = dir.to_string_lossy();
        assert!(
            dir_str.contains("CrabInfer") || dir_str.contains("crabinfer"),
            "Storage dir should contain 'CrabInfer': {}",
            dir_str
        );
    }

    #[test]
    fn test_download_manager_custom_dir() {
        let tmp = std::env::temp_dir().join("crabinfer_test_dl");
        let _ = fs::remove_dir_all(&tmp);
        let mgr = ModelDownloadManager::with_directory(tmp.to_string_lossy().into_owned());
        assert!(mgr.is_ok());
        let mgr = mgr.unwrap();
        assert_eq!(mgr.storage_directory(), tmp.to_string_lossy());
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_is_downloaded_false() {
        let tmp = std::env::temp_dir().join("crabinfer_test_dl2");
        let _ = fs::remove_dir_all(&tmp);
        let mgr = ModelDownloadManager::with_directory(tmp.to_string_lossy().into_owned()).unwrap();
        assert!(!mgr.is_downloaded("nonexistent".into()));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_list_downloaded_empty() {
        let tmp = std::env::temp_dir().join("crabinfer_test_dl3");
        let _ = fs::remove_dir_all(&tmp);
        let mgr = ModelDownloadManager::with_directory(tmp.to_string_lossy().into_owned()).unwrap();
        assert!(mgr.list_downloaded().is_empty());
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_validate_gguf_valid() {
        let tmp = std::env::temp_dir().join("test_valid.gguf");
        let mut f = fs::File::create(&tmp).unwrap();
        f.write_all(b"GGUF\x03\x00\x00\x00").unwrap();
        assert!(validate_gguf(&tmp).is_ok());
        let _ = fs::remove_file(&tmp);
    }

    #[test]
    fn test_validate_gguf_invalid() {
        let tmp = std::env::temp_dir().join("test_invalid.gguf");
        let mut f = fs::File::create(&tmp).unwrap();
        f.write_all(b"BAAD\x00\x00\x00\x00").unwrap();
        assert!(validate_gguf(&tmp).is_err());
        let _ = fs::remove_file(&tmp);
    }

    #[test]
    fn test_validate_json_valid() {
        let tmp = std::env::temp_dir().join("test_valid.json");
        fs::write(&tmp, r#"{"version":"1.0"}"#).unwrap();
        assert!(validate_json(&tmp).is_ok());
        let _ = fs::remove_file(&tmp);
    }

    #[test]
    fn test_validate_json_invalid() {
        let tmp = std::env::temp_dir().join("test_invalid.json");
        fs::write(&tmp, "NOT JSON").unwrap();
        assert!(validate_json(&tmp).is_err());
        let _ = fs::remove_file(&tmp);
    }

    #[test]
    fn test_verify_sha256() {
        let tmp = std::env::temp_dir().join("test_sha256.bin");
        fs::write(&tmp, b"hello world").unwrap();
        // SHA256 of "hello world"
        let hash = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";
        assert!(verify_sha256(&tmp, hash).is_ok());
        assert!(verify_sha256(&tmp, "wrong_hash").is_err());
        let _ = fs::remove_file(&tmp);
    }

    #[test]
    fn test_chrono_now_iso8601() {
        let ts = chrono_now_iso8601();
        assert!(ts.starts_with("20"), "Timestamp should start with '20': {ts}");
        assert!(ts.ends_with('Z'), "Timestamp should end with 'Z': {ts}");
        assert_eq!(ts.len(), 20, "ISO 8601 length: {ts}");
    }

    #[test]
    fn test_delete_nonexistent_ok() {
        let tmp = std::env::temp_dir().join("crabinfer_test_dl4");
        let _ = fs::remove_dir_all(&tmp);
        let mgr = ModelDownloadManager::with_directory(tmp.to_string_lossy().into_owned()).unwrap();
        assert!(mgr.delete("nonexistent".into()).is_ok());
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_disk_usage_empty() {
        let tmp = std::env::temp_dir().join("crabinfer_test_dl5");
        let _ = fs::remove_dir_all(&tmp);
        let mgr = ModelDownloadManager::with_directory(tmp.to_string_lossy().into_owned()).unwrap();
        assert_eq!(mgr.total_disk_usage(), 0);
        let _ = fs::remove_dir_all(&tmp);
    }
}
