//! Request/response types for the guided completions endpoint.
//!
//! The `/v1/guided/completions` endpoint is a superset of `/v1/chat/completions`
//! that requires a `constraint` field for guided decoding. Response types are
//! reused from `types::openai`.

use super::common::ChatMessage;
use super::openai::{StopCondition, StreamOptions};
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Guided completion request
// ---------------------------------------------------------------------------

/// Request body for `POST /v1/guided/completions`.
///
/// Identical to `ChatCompletionRequest` with the addition of a required
/// `constraint` field and an optional `strict_constraints` flag.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct GuidedCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    /// Guided decoding constraint. Required — there is no fallback to
    /// unconstrained generation unless `strict_constraints` is `false`.
    pub constraint: GuidedConstraintSpec,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<StopCondition>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    pub top_logprobs: Option<u32>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Request priority (lower = higher priority, default 0). CrabInfer extension.
    #[serde(default)]
    pub priority: Option<i32>,
    /// If `false`, invalid constraints log a warning and generate unconstrained
    /// instead of returning HTTP 400. Default: `true`.
    #[serde(default = "default_strict")]
    pub strict_constraints: bool,
}

fn default_strict() -> bool {
    true
}

/// Guided constraint specification in the API request.
///
/// Uses serde's `tag = "type"` internally so the JSON looks like:
/// ```json
/// {"type": "regex", "pattern": "[0-9]+"}
/// {"type": "json_schema", "json_schema": { ... }}
/// ```
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum GuidedConstraintSpec {
    /// Regex pattern -- the output must match this pattern.
    #[serde(rename = "regex")]
    Regex { pattern: String },
    /// JSON Schema -- the output must be valid JSON conforming to this schema.
    #[serde(rename = "json_schema")]
    JsonSchema { json_schema: serde_json::Value },
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_regex_constraint() {
        let json = r#"{"type": "regex", "pattern": "[0-9]+"}"#;
        let spec: GuidedConstraintSpec = serde_json::from_str(json).expect("deserialize regex");
        match spec {
            GuidedConstraintSpec::Regex { pattern } => {
                assert_eq!(pattern, "[0-9]+");
            }
            _ => panic!("expected Regex variant"),
        }
    }

    #[test]
    fn test_deserialize_json_schema_constraint() {
        let json = r#"{"type": "json_schema", "json_schema": {"type": "object"}}"#;
        let spec: GuidedConstraintSpec = serde_json::from_str(json).expect("deserialize json_schema");
        match spec {
            GuidedConstraintSpec::JsonSchema { json_schema } => {
                assert_eq!(json_schema["type"], "object");
            }
            _ => panic!("expected JsonSchema variant"),
        }
    }

    #[test]
    fn test_strict_constraints_default_true() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "constraint": {"type": "regex", "pattern": "[0-9]+"}
        }"#;
        let req: GuidedCompletionRequest = serde_json::from_str(json).expect("deserialize request");
        assert!(req.strict_constraints, "strict_constraints should default to true");
    }

    #[test]
    fn test_strict_constraints_explicit_false() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "constraint": {"type": "regex", "pattern": "[0-9]+"},
            "strict_constraints": false
        }"#;
        let req: GuidedCompletionRequest = serde_json::from_str(json).expect("deserialize request");
        assert!(!req.strict_constraints, "strict_constraints should be false when set");
    }

    #[test]
    fn test_missing_constraint_fails_deserialization() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}]
        }"#;
        let result = serde_json::from_str::<GuidedCompletionRequest>(json);
        assert!(
            result.is_err(),
            "deserialization should fail when constraint is missing"
        );
    }
}
