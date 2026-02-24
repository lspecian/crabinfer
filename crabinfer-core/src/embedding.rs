//! Embedding Providers — generate vector embeddings for text.
//!
//! Supports two strategies:
//! - **TF-IDF**: Offline, no model needed. Uses term frequency–inverse document
//!   frequency to produce sparse-style embeddings. Good enough for many use cases.
//! - **OpenAI**: Cloud-based, uses `text-embedding-3-small` for high-quality
//!   dense embeddings. Requires an API key.

use crate::CrabInferError;
use std::collections::HashMap;

/// Trait for generating text embeddings.
pub trait EmbeddingProvider: Send + Sync {
    /// Generate embeddings for a batch of texts.
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, CrabInferError>;

    /// Get the embedding dimension.
    fn dimension(&self) -> usize;

    /// Provider name for logging.
    fn name(&self) -> &str;
}

// ─── TF-IDF Embedder ────────────────────────────────────────────────────────

/// Simple TF-IDF-based embedder that works offline without any model.
///
/// Produces fixed-dimension vectors by hashing terms into buckets.
/// Not as accurate as neural embeddings but works without any external
/// dependencies or API keys.
pub struct TfIdfEmbedder {
    dimension: usize,
    /// IDF weights learned from the corpus (term_hash → idf_weight).
    idf: HashMap<usize, f32>,
    /// Number of documents seen.
    doc_count: usize,
}

impl TfIdfEmbedder {
    /// Create a new TF-IDF embedder with the given dimension.
    /// Default dimension is 256.
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension: dimension.max(64),
            idf: HashMap::new(),
            doc_count: 0,
        }
    }

    /// Add documents to the IDF corpus (call before embedding for best results).
    /// This updates the inverse document frequency weights.
    pub fn fit(&mut self, documents: &[String]) {
        for doc in documents {
            self.doc_count += 1;
            let mut seen_buckets = std::collections::HashSet::new();
            for term in tokenize(doc) {
                let bucket = hash_term(&term, self.dimension);
                if seen_buckets.insert(bucket) {
                    *self.idf.entry(bucket).or_insert(0.0) += 1.0;
                }
            }
        }
    }

    fn embed_single(&self, text: &str) -> Vec<f32> {
        let mut vector = vec![0.0f32; self.dimension];
        let terms = tokenize(text);
        if terms.is_empty() {
            return vector;
        }

        // Term frequency
        let mut tf: HashMap<usize, f32> = HashMap::new();
        for term in &terms {
            let bucket = hash_term(term, self.dimension);
            *tf.entry(bucket).or_insert(0.0) += 1.0;
        }

        // TF-IDF scoring
        let n = (self.doc_count.max(1)) as f32;
        for (bucket, freq) in &tf {
            let tf_score = *freq / terms.len() as f32;
            let df = self.idf.get(bucket).copied().unwrap_or(1.0);
            let idf_score = (n / df).ln() + 1.0;
            vector[*bucket] = tf_score * idf_score;
        }

        // L2 normalize
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut vector {
                *v /= norm;
            }
        }

        vector
    }
}

impl EmbeddingProvider for TfIdfEmbedder {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, CrabInferError> {
        Ok(texts.iter().map(|t| self.embed_single(t)).collect())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        "tfidf"
    }
}

impl Default for TfIdfEmbedder {
    fn default() -> Self {
        Self::new(256)
    }
}

// ─── OpenAI Embedder ────────────────────────────────────────────────────────

/// OpenAI embedding provider using `text-embedding-3-small`.
///
/// Requires the `providers` feature and an OpenAI API key.
#[cfg(feature = "providers")]
pub struct OpenAIEmbedder {
    api_key: String,
    model: String,
    dimension: usize,
}

#[cfg(feature = "providers")]
impl OpenAIEmbedder {
    /// Create a new OpenAI embedder.
    /// Default model: text-embedding-3-small (1536 dimensions).
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            model: "text-embedding-3-small".to_string(),
            dimension: 1536,
        }
    }

    /// Use a custom model and dimension.
    pub fn with_model(mut self, model: &str, dimension: usize) -> Self {
        self.model = model.to_string();
        self.dimension = dimension;
        self
    }
}

#[cfg(feature = "providers")]
impl EmbeddingProvider for OpenAIEmbedder {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, CrabInferError> {
        let client = reqwest::blocking::Client::new();

        let body = serde_json::json!({
            "input": texts,
            "model": self.model,
        });

        let response = client
            .post("https://api.openai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .map_err(|e| CrabInferError::NetworkError {
                reason: format!("OpenAI embedding request failed: {}", e),
            })?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response.text().unwrap_or_default();
            return Err(CrabInferError::ApiError {
                provider: "openai".to_string(),
                status_code: status as u32,
                message: text,
            });
        }

        let json: serde_json::Value =
            response.json().map_err(|e| CrabInferError::NetworkError {
                reason: format!("Failed to parse embedding response: {}", e),
            })?;

        let data = json["data"]
            .as_array()
            .ok_or(CrabInferError::NetworkError {
                reason: "Missing 'data' in embedding response".to_string(),
            })?;

        let mut embeddings = Vec::with_capacity(data.len());
        for item in data {
            let embedding: Vec<f32> = item["embedding"]
                .as_array()
                .ok_or(CrabInferError::NetworkError {
                    reason: "Missing 'embedding' in response item".to_string(),
                })?
                .iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect();
            embeddings.push(embedding);
        }

        Ok(embeddings)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        "openai"
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Simple whitespace tokenizer with lowercasing and stop word removal.
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 2 && !is_stop_word(w))
        .map(|w| w.to_string())
        .collect()
}

/// Hash a term to a bucket index.
fn hash_term(term: &str, dimension: usize) -> usize {
    let mut hash: u64 = 5381;
    for byte in term.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
    }
    (hash as usize) % dimension
}

/// Common English stop words.
fn is_stop_word(word: &str) -> bool {
    matches!(
        word,
        "the" | "and" | "for" | "are" | "but" | "not" | "you" | "all"
            | "can" | "had" | "her" | "was" | "one" | "our" | "out"
            | "has" | "have" | "been" | "from" | "this" | "that"
            | "with" | "they" | "will" | "each" | "which" | "their"
            | "there" | "what" | "about" | "would" | "into" | "more"
            | "other" | "were" | "then" | "them" | "than" | "some"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tfidf_basic() {
        let mut embedder = TfIdfEmbedder::new(128);
        let docs = vec![
            "Rust is a systems programming language".to_string(),
            "Python is great for data science".to_string(),
        ];
        embedder.fit(&docs);

        let embeddings = embedder.embed(&docs).unwrap();
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 128);
        assert_eq!(embeddings[1].len(), 128);
    }

    #[test]
    fn test_tfidf_similarity() {
        let mut embedder = TfIdfEmbedder::new(256);
        let corpus = vec![
            "Rust programming language systems".to_string(),
            "Python data science machine learning".to_string(),
            "Rust memory safety ownership borrowing".to_string(),
        ];
        embedder.fit(&corpus);

        let embeddings = embedder.embed(&corpus).unwrap();

        // Cosine similarity between Rust docs should be higher than Rust vs Python
        let sim_rust_rust = cosine_similarity(&embeddings[0], &embeddings[2]);
        let sim_rust_python = cosine_similarity(&embeddings[0], &embeddings[1]);
        assert!(
            sim_rust_rust > sim_rust_python,
            "Rust-Rust similarity ({}) should be > Rust-Python ({})",
            sim_rust_rust,
            sim_rust_python
        );
    }

    #[test]
    fn test_tfidf_normalized() {
        let embedder = TfIdfEmbedder::new(128);
        let embeddings = embedder
            .embed(&["Hello world this is a test".to_string()])
            .unwrap();
        let norm: f32 = embeddings[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "Vector should be L2-normalized");
    }

    #[test]
    fn test_empty_text() {
        let embedder = TfIdfEmbedder::new(128);
        let embeddings = embedder.embed(&["".to_string()]).unwrap();
        assert_eq!(embeddings[0].len(), 128);
        // All zeros for empty text
        assert!(embeddings[0].iter().all(|&v| v == 0.0));
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
}
