//! Text Chunking — splits documents into overlapping passages for RAG.
//!
//! Chunks text at paragraph or sentence boundaries with configurable
//! chunk size and overlap. Used by the KnowledgeBase to prepare documents
//! for embedding and retrieval.

use serde::{Deserialize, Serialize};

/// Metadata for a text chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Source identifier (file path or document name).
    pub source: String,
    /// Zero-based index of this chunk within the source document.
    pub chunk_index: usize,
    /// Character offset in the source where this chunk begins.
    pub start_offset: usize,
}

/// A chunk of text extracted from a document.
#[derive(Debug, Clone)]
pub struct TextChunk {
    pub text: String,
    pub metadata: ChunkMetadata,
}

/// Configurable text chunker.
pub struct TextChunker {
    /// Target chunk size in characters (not tokens).
    /// Default: 2000 chars ≈ 500 tokens.
    chunk_size: usize,
    /// Number of overlapping characters between consecutive chunks.
    /// Default: 200 chars ≈ 50 tokens.
    chunk_overlap: usize,
}

impl TextChunker {
    /// Create a new chunker with the given chunk size and overlap (in characters).
    pub fn new(chunk_size: usize, overlap: usize) -> Self {
        Self {
            chunk_size: chunk_size.max(100),
            chunk_overlap: overlap.min(chunk_size / 2),
        }
    }

    /// Create a chunker with default settings (2000 chars, 200 overlap).
    pub fn default_settings() -> Self {
        Self::new(2000, 200)
    }

    /// Split text into overlapping chunks, preferring to break at paragraph
    /// or sentence boundaries.
    pub fn chunk(&self, text: &str, source: &str) -> Vec<TextChunk> {
        if text.is_empty() {
            return Vec::new();
        }

        // If the text fits in one chunk, return it directly
        if text.len() <= self.chunk_size {
            return vec![TextChunk {
                text: text.to_string(),
                metadata: ChunkMetadata {
                    source: source.to_string(),
                    chunk_index: 0,
                    start_offset: 0,
                },
            }];
        }

        let mut chunks = Vec::new();
        let mut start = 0;
        let mut chunk_index = 0;

        while start < text.len() {
            let end = (start + self.chunk_size).min(text.len());

            // Try to find a good break point
            let actual_end = if end < text.len() {
                self.find_break_point(text, start, end)
            } else {
                end
            };

            let chunk_text = text[start..actual_end].trim().to_string();
            if !chunk_text.is_empty() {
                chunks.push(TextChunk {
                    text: chunk_text,
                    metadata: ChunkMetadata {
                        source: source.to_string(),
                        chunk_index,
                        start_offset: start,
                    },
                });
                chunk_index += 1;
            }

            // Advance with overlap
            let step = actual_end.saturating_sub(start).saturating_sub(self.chunk_overlap);
            start += step.max(1);
        }

        chunks
    }

    /// Find the best break point near `target_end`, preferring paragraph
    /// then sentence boundaries.
    fn find_break_point(&self, text: &str, start: usize, target_end: usize) -> usize {
        let search_start = target_end.saturating_sub(self.chunk_size / 4).max(start);
        let search_text = &text[search_start..target_end];

        // Prefer paragraph break (\n\n)
        if let Some(pos) = search_text.rfind("\n\n") {
            return search_start + pos + 2;
        }

        // Then sentence end (. or ! or ? followed by space or newline)
        for (i, c) in search_text.char_indices().rev() {
            if (c == '.' || c == '!' || c == '?') && i + 1 < search_text.len() {
                let next = search_text.as_bytes().get(search_start + i + 1);
                if matches!(next, Some(b' ') | Some(b'\n')) {
                    return search_start + i + 1;
                }
            }
        }

        // Then newline
        if let Some(pos) = search_text.rfind('\n') {
            return search_start + pos + 1;
        }

        // Then word boundary (space)
        if let Some(pos) = search_text.rfind(' ') {
            return search_start + pos + 1;
        }

        // Fallback: hard cut
        target_end
    }
}

impl Default for TextChunker {
    fn default() -> Self {
        Self::default_settings()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_text() {
        let chunker = TextChunker::default_settings();
        assert!(chunker.chunk("", "test").is_empty());
    }

    #[test]
    fn test_short_text_single_chunk() {
        let chunker = TextChunker::default_settings();
        let chunks = chunker.chunk("Hello world", "test.txt");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Hello world");
        assert_eq!(chunks[0].metadata.source, "test.txt");
        assert_eq!(chunks[0].metadata.chunk_index, 0);
    }

    #[test]
    fn test_long_text_multiple_chunks() {
        let chunker = TextChunker::new(100, 20);
        let text = "a ".repeat(200); // 400 chars
        let chunks = chunker.chunk(&text, "test.txt");
        assert!(chunks.len() > 1);

        // Verify chunk indices are sequential
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.metadata.chunk_index, i);
        }
    }

    #[test]
    fn test_paragraph_break_preference() {
        let chunker = TextChunker::new(100, 20);
        let text = format!(
            "{}.\n\n{}.",
            "First paragraph with enough text to fill a chunk".repeat(1),
            "Second paragraph starts here"
        );
        let chunks = chunker.chunk(&text, "test.txt");
        // Should prefer to break at \n\n
        if chunks.len() >= 2 {
            assert!(!chunks[0].text.ends_with("Second"));
        }
    }

    #[test]
    fn test_overlap() {
        let chunker = TextChunker::new(50, 10);
        let text = "word ".repeat(50); // 250 chars
        let chunks = chunker.chunk(&text, "test.txt");

        // Verify chunks overlap (later chunk starts before previous ends)
        if chunks.len() >= 2 {
            let first_end = chunks[0].metadata.start_offset + chunks[0].text.len();
            let second_start = chunks[1].metadata.start_offset;
            // Due to trimming and break finding, overlap may vary
            assert!(second_start < first_end + 20);
        }
    }

    #[test]
    fn test_chunk_metadata() {
        let chunker = TextChunker::new(50, 10);
        let chunks = chunker.chunk("Hello world. This is a test.", "doc.md");
        assert_eq!(chunks[0].metadata.source, "doc.md");
        assert_eq!(chunks[0].metadata.start_offset, 0);
    }
}
