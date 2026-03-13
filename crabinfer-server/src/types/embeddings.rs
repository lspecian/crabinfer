use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: EmbeddingInput,
    /// Encoding format: "float" (default) or "base64".
    #[serde(default)]
    pub encoding_format: Option<String>,
}

/// Input can be a single string or a batch of strings.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Batch(Vec<String>),
}

impl EmbeddingInput {
    /// Convert to a vec of strings regardless of variant.
    pub fn into_texts(self) -> Vec<String> {
        match self {
            EmbeddingInput::Single(s) => vec![s],
            EmbeddingInput::Batch(v) => v,
        }
    }
}

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingObject {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_single_input() {
        let json = r#"{"model": "text-embedding-3-small", "input": "Hello world"}"#;
        let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "text-embedding-3-small");
        match req.input {
            EmbeddingInput::Single(s) => assert_eq!(s, "Hello world"),
            _ => panic!("expected Single variant"),
        }
    }

    #[test]
    fn test_deserialize_batch_input() {
        let json = r#"{"model": "text-embedding-3-small", "input": ["Hello", "World"]}"#;
        let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "text-embedding-3-small");
        match req.input {
            EmbeddingInput::Batch(v) => {
                assert_eq!(v.len(), 2);
                assert_eq!(v[0], "Hello");
                assert_eq!(v[1], "World");
            }
            _ => panic!("expected Batch variant"),
        }
    }

    #[test]
    fn test_serialize_response() {
        let resp = EmbeddingResponse {
            object: "list".to_string(),
            data: vec![
                EmbeddingObject {
                    object: "embedding".to_string(),
                    embedding: vec![0.1, 0.2, 0.3],
                    index: 0,
                },
                EmbeddingObject {
                    object: "embedding".to_string(),
                    embedding: vec![0.4, 0.5, 0.6],
                    index: 1,
                },
            ],
            model: "text-embedding-3-small".to_string(),
            usage: EmbeddingUsage {
                prompt_tokens: 4,
                total_tokens: 4,
            },
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["object"], "list");
        assert_eq!(json["data"][0]["object"], "embedding");
        assert_eq!(json["data"][0]["index"], 0);
        assert_eq!(json["data"][1]["index"], 1);
        assert_eq!(json["usage"]["prompt_tokens"], 4);
        assert_eq!(json["usage"]["total_tokens"], 4);
        assert_eq!(json["model"], "text-embedding-3-small");
    }

    #[test]
    fn test_into_texts_single() {
        let input = EmbeddingInput::Single("hello".to_string());
        let texts = input.into_texts();
        assert_eq!(texts, vec!["hello"]);
    }

    #[test]
    fn test_into_texts_batch() {
        let input = EmbeddingInput::Batch(vec!["a".to_string(), "b".to_string()]);
        let texts = input.into_texts();
        assert_eq!(texts, vec!["a", "b"]);
    }
}
