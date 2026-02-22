//! Shared HTTP utilities for cloud providers.
//!
//! SSE line parser, HTTP error mapping, and request helpers.

use crate::CrabInferError;
use std::io::{BufRead, BufReader, Read};
use std::time::Duration;

/// Create a reqwest blocking client with the given timeout.
pub fn build_client(timeout_seconds: u32) -> Result<reqwest::blocking::Client, CrabInferError> {
    let timeout = if timeout_seconds == 0 { 30 } else { timeout_seconds };
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(timeout as u64))
        .build()
        .map_err(|e| CrabInferError::NetworkError {
            reason: format!("failed to build HTTP client: {}", e),
        })
}

/// Map an HTTP response status to a CrabInferError.
pub fn check_response(
    provider: &str,
    resp: &reqwest::blocking::Response,
) -> Result<(), CrabInferError> {
    let status = resp.status();
    if status.is_success() {
        return Ok(());
    }
    match status.as_u16() {
        401 => Err(CrabInferError::AuthenticationFailed {
            provider: provider.to_string(),
        }),
        429 => {
            let retry_after = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(10);
            Err(CrabInferError::RateLimited {
                provider: provider.to_string(),
                retry_after_seconds: retry_after,
            })
        }
        code => Err(CrabInferError::ApiError {
            provider: provider.to_string(),
            status_code: code as u32,
            message: status.canonical_reason().unwrap_or("unknown").to_string(),
        }),
    }
}

/// An SSE event parsed from a stream.
#[derive(Debug)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

/// Iterator that reads SSE events from a byte stream.
pub struct SseReader<R: Read> {
    reader: BufReader<R>,
    done: bool,
}

impl<R: Read> SseReader<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader: BufReader::new(reader),
            done: false,
        }
    }
}

impl<R: Read> Iterator for SseReader<R> {
    type Item = Result<SseEvent, CrabInferError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let mut event_type: Option<String> = None;
        let mut data_lines: Vec<String> = Vec::new();

        loop {
            let mut line = String::new();
            match self.reader.read_line(&mut line) {
                Ok(0) => {
                    self.done = true;
                    if !data_lines.is_empty() {
                        return Some(Ok(SseEvent {
                            event: event_type,
                            data: data_lines.join("\n"),
                        }));
                    }
                    return None;
                }
                Ok(_) => {
                    let line = line.trim_end_matches('\n').trim_end_matches('\r');

                    if line.is_empty() {
                        // Empty line = end of event
                        if !data_lines.is_empty() {
                            return Some(Ok(SseEvent {
                                event: event_type,
                                data: data_lines.join("\n"),
                            }));
                        }
                        // No data accumulated yet, continue reading
                        continue;
                    }

                    if let Some(value) = line.strip_prefix("event:") {
                        event_type = Some(value.trim().to_string());
                    } else if let Some(value) = line.strip_prefix("data:") {
                        data_lines.push(value.trim().to_string());
                    }
                    // Ignore other fields (id:, retry:, comments)
                }
                Err(e) => {
                    self.done = true;
                    return Some(Err(CrabInferError::NetworkError {
                        reason: format!("SSE read error: {}", e),
                    }));
                }
            }
        }
    }
}

/// Iterator that reads line-delimited JSON from a byte stream (for Ollama).
pub struct JsonLineReader<R: Read> {
    reader: BufReader<R>,
    done: bool,
}

impl<R: Read> JsonLineReader<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader: BufReader::new(reader),
            done: false,
        }
    }
}

impl<R: Read> Iterator for JsonLineReader<R> {
    type Item = Result<String, CrabInferError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let mut line = String::new();
        match self.reader.read_line(&mut line) {
            Ok(0) => {
                self.done = true;
                None
            }
            Ok(_) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    // Skip empty lines, try next
                    self.next()
                } else {
                    Some(Ok(trimmed.to_string()))
                }
            }
            Err(e) => {
                self.done = true;
                Some(Err(CrabInferError::NetworkError {
                    reason: format!("read error: {}", e),
                }))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_sse_reader_basic() {
        let input = "data: hello\n\ndata: world\n\n";
        let reader = SseReader::new(Cursor::new(input));
        let events: Vec<_> = reader.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].data, "hello");
        assert_eq!(events[1].data, "world");
    }

    #[test]
    fn test_sse_reader_named_events() {
        let input = "event: message_start\ndata: {\"type\":\"message_start\"}\n\nevent: content_block_delta\ndata: {\"text\":\"Hi\"}\n\n";
        let reader = SseReader::new(Cursor::new(input));
        let events: Vec<_> = reader.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event.as_deref(), Some("message_start"));
        assert_eq!(events[1].event.as_deref(), Some("content_block_delta"));
    }

    #[test]
    fn test_sse_reader_done_marker() {
        let input = "data: {\"content\":\"Hi\"}\n\ndata: [DONE]\n\n";
        let reader = SseReader::new(Cursor::new(input));
        let events: Vec<_> = reader.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[1].data, "[DONE]");
    }

    #[test]
    fn test_json_line_reader() {
        let input = "{\"done\":false,\"message\":{\"content\":\"Hi\"}}\n{\"done\":true}\n";
        let reader = JsonLineReader::new(Cursor::new(input));
        let lines: Vec<_> = reader.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("\"done\":false"));
        assert!(lines[1].contains("\"done\":true"));
    }
}
