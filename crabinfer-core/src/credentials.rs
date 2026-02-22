//! In-memory credential store for cloud provider API keys.
//!
//! The CredentialManager is a global singleton that maps provider names
//! to API keys. Swift apps populate it at launch (e.g., from Keychain),
//! and providers fall back to it when their ProviderConfig has an empty key.

use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

use crate::CrabInferError;

// ---------------------------------------------------------------------------
// Global singleton
// ---------------------------------------------------------------------------

struct CredentialManager {
    keys: RwLock<HashMap<String, String>>,
}

static INSTANCE: OnceLock<CredentialManager> = OnceLock::new();

fn manager() -> &'static CredentialManager {
    INSTANCE.get_or_init(|| CredentialManager {
        keys: RwLock::new(HashMap::new()),
    })
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Store or update an API key for a provider.
///
/// Provider names: `"openai"`, `"anthropic"`, `"google"`, `"ollama"`.
pub fn set_api_key(provider: &str, key: &str) {
    manager()
        .keys
        .write()
        .unwrap()
        .insert(provider.to_string(), key.to_string());
}

/// Retrieve the stored API key for a provider.
pub fn get_api_key(provider: &str) -> Option<String> {
    manager().keys.read().unwrap().get(provider).cloned()
}

/// Remove a stored API key.
pub fn remove_api_key(provider: &str) {
    manager().keys.write().unwrap().remove(provider);
}

/// Check if a key is stored for the given provider.
pub fn has_api_key(provider: &str) -> bool {
    manager().keys.read().unwrap().contains_key(provider)
}

/// List provider names that have stored keys.
pub fn configured_providers() -> Vec<String> {
    manager().keys.read().unwrap().keys().cloned().collect()
}

/// Resolve the effective API key for a provider.
///
/// If `explicit_key` is non-empty, returns it (explicit always wins).
/// Otherwise, falls back to the credential manager.
pub fn resolve_api_key(provider: &str, explicit_key: &str) -> Option<String> {
    if !explicit_key.is_empty() {
        Some(explicit_key.to_string())
    } else {
        get_api_key(provider)
    }
}

/// Redact an API key for safe logging. Shows first 4 chars + "***".
pub fn redact_key(key: &str) -> String {
    if key.is_empty() {
        "<empty>".to_string()
    } else if key.len() <= 4 {
        "***".to_string()
    } else {
        format!("{}***", &key[..4])
    }
}

// ---------------------------------------------------------------------------
// Validation (makes lightweight API calls to verify keys)
// ---------------------------------------------------------------------------

/// Validate a stored API key by making a lightweight API call.
///
/// Returns `Ok(true)` if the key is valid, `Ok(false)` if authentication
/// fails, or `Err` for network/other errors.
pub fn validate_api_key(provider: &str) -> Result<bool, CrabInferError> {
    let key = get_api_key(provider).ok_or(CrabInferError::AuthenticationFailed {
        provider: provider.to_string(),
    })?;

    let client = crate::providers::http_utils::build_client(10)?;

    match provider {
        "openai" => validate_openai(&client, &key),
        "anthropic" => validate_anthropic(&client, &key),
        "google" => validate_google(&client, &key),
        "ollama" => validate_ollama(&client),
        "vllm" => validate_vllm(&client, &key),
        _ => Err(CrabInferError::InvalidConfig),
    }
}

fn validate_openai(
    client: &reqwest::blocking::Client,
    key: &str,
) -> Result<bool, CrabInferError> {
    // GET /v1/models — lightweight, only checks auth
    let resp = client
        .get("https://api.openai.com/v1/models")
        .header("Authorization", format!("Bearer {}", key))
        .send()
        .map_err(|e| CrabInferError::NetworkError {
            reason: e.to_string(),
        })?;
    match resp.status().as_u16() {
        200 => Ok(true),
        401 => Ok(false),
        _ => Ok(true), // Non-auth errors don't mean invalid key
    }
}

fn validate_anthropic(
    client: &reqwest::blocking::Client,
    key: &str,
) -> Result<bool, CrabInferError> {
    // POST /v1/messages with minimal payload
    let body = serde_json::json!({
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "hi"}],
    });
    let resp = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", key)
        .header("anthropic-version", "2023-06-01")
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .map_err(|e| CrabInferError::NetworkError {
            reason: e.to_string(),
        })?;
    match resp.status().as_u16() {
        200 => Ok(true),
        401 => Ok(false),
        _ => Ok(true),
    }
}

fn validate_google(
    client: &reqwest::blocking::Client,
    key: &str,
) -> Result<bool, CrabInferError> {
    // GET /v1/models — lists available models, checks auth
    let resp = client
        .get("https://generativelanguage.googleapis.com/v1/models")
        .header("x-goog-api-key", key)
        .send()
        .map_err(|e| CrabInferError::NetworkError {
            reason: e.to_string(),
        })?;
    match resp.status().as_u16() {
        200 => Ok(true),
        401 | 403 => Ok(false),
        _ => Ok(true),
    }
}

fn validate_ollama(client: &reqwest::blocking::Client) -> Result<bool, CrabInferError> {
    // Ollama has no auth — just check connectivity
    let resp = client
        .get("http://localhost:11434/api/tags")
        .send()
        .map_err(|e| CrabInferError::NetworkError {
            reason: e.to_string(),
        })?;
    Ok(resp.status().is_success())
}

fn validate_vllm(client: &reqwest::blocking::Client, key: &str) -> Result<bool, CrabInferError> {
    // vLLM — check health endpoint. The key is the base_url stored as the credential.
    // If the "key" looks like a URL, use it as base_url. Otherwise fall back to localhost.
    let base_url = if key.starts_with("http://") || key.starts_with("https://") {
        key.trim_end_matches('/')
    } else {
        "http://localhost:8000"
    };
    let resp = client
        .get(format!("{}/health", base_url))
        .send()
        .map_err(|e| CrabInferError::NetworkError {
            reason: e.to_string(),
        })?;
    Ok(resp.status().is_success())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_get_remove() {
        set_api_key("test_provider_sgr", "sk-test123");
        assert_eq!(
            get_api_key("test_provider_sgr"),
            Some("sk-test123".to_string())
        );
        assert!(has_api_key("test_provider_sgr"));
        remove_api_key("test_provider_sgr");
        assert!(!has_api_key("test_provider_sgr"));
        assert_eq!(get_api_key("test_provider_sgr"), None);
    }

    #[test]
    fn test_resolve_prefers_explicit() {
        set_api_key("resolve_test", "stored-key");
        assert_eq!(
            resolve_api_key("resolve_test", "explicit-key"),
            Some("explicit-key".to_string())
        );
        assert_eq!(
            resolve_api_key("resolve_test", ""),
            Some("stored-key".to_string())
        );
        assert_eq!(resolve_api_key("no_such_provider", ""), None);
        // Cleanup
        remove_api_key("resolve_test");
    }

    #[test]
    fn test_configured_providers_lists_keys() {
        set_api_key("cp_test_a", "key_a");
        set_api_key("cp_test_b", "key_b");
        let providers = configured_providers();
        assert!(providers.contains(&"cp_test_a".to_string()));
        assert!(providers.contains(&"cp_test_b".to_string()));
        // Cleanup
        remove_api_key("cp_test_a");
        remove_api_key("cp_test_b");
    }

    #[test]
    fn test_redact_key() {
        assert_eq!(redact_key(""), "<empty>");
        assert_eq!(redact_key("abc"), "***");
        assert_eq!(redact_key("sk-proj-abc123"), "sk-p***");
    }

    #[test]
    fn test_validate_unknown_provider() {
        set_api_key("unknown_xyz", "key");
        let result = validate_api_key("unknown_xyz");
        assert!(result.is_err());
        remove_api_key("unknown_xyz");
    }
}
