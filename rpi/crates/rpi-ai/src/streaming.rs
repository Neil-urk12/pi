//! SSE (Server-Sent Events) streaming utilities.
//!
//! Provides a parser for SSE streams from HTTP responses, commonly used by
//! LLM providers for streaming chat completions.

use std::time::Duration;

use futures::Stream;
use futures::StreamExt;

/// A parsed Server-Sent Event.
#[derive(Debug, Clone, Default)]
pub struct SseEvent {
    /// The event type field (e.g., "message_start", "content_block_delta").
    pub event: Option<String>,
    /// The data payload. Multi-line data fields are joined with newlines.
    pub data: String,
    /// The event ID.
    pub id: Option<String>,
}

impl SseEvent {
    /// Returns `true` if this is the `[DONE]` sentinel used by OpenAI-style APIs.
    pub fn is_done(&self) -> bool {
        self.data.trim() == "[DONE]"
    }

    /// Parse the event data as JSON.
    pub fn json<T: serde::de::DeserializeOwned>(&self) -> serde_json::Result<T> {
        serde_json::from_str(&self.data)
    }
}


/// Typed errors for SSE stream operations.
///
/// Enables programmatic error matching instead of fragile string parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SseError {
    /// The stream was idle for longer than the configured timeout.
    IdleTimeout {
        /// The timeout duration in milliseconds.
        timeout_ms: u64,
    },
    /// An error occurred while reading from the underlying byte stream.
    ReadError {
        /// The error message from the underlying IO error.
        message: String,
    },
}

impl SseError {
    /// Returns true if this error is retryable (connection can be retried).
    pub fn is_retryable(&self) -> bool {
        matches!(self, SseError::IdleTimeout { .. } | SseError::ReadError { .. })
    }
}

impl std::fmt::Display for SseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SseError::IdleTimeout { timeout_ms } => {
                write!(f, "SSE stream idle timeout after {}ms", timeout_ms)
            }
            SseError::ReadError { message } => {
                write!(f, "SSE stream read error: {}", message)
            }
        }
    }
}

impl std::error::Error for SseError {}

/// Convert a [`reqwest::Response`] into a stream of [`SseEvent`]s.
///
/// Handles the SSE wire protocol: line buffering, multi-line `data:` fields,
/// comment lines (`:`), and event/id/retry fields.
///
/// # Errors
///
/// Yields `Err` if the underlying byte stream encounters a read error.
pub fn sse_stream(
    response: reqwest::Response,
) -> impl Stream<Item = anyhow::Result<SseEvent>> + Send {
    sse_events_from_byte_stream(response.bytes_stream(), None)
}


/// Similar to [`sse_stream`], but with configurable idle timeout.
///
/// If no data arrives within the specified `idle_timeout`, the stream yields
/// an error and terminates.
///
/// # Errors
///
/// Yields `Err` if the underlying byte stream encounters a read error or if
/// `idle_timeout` expires.
pub fn sse_stream_with_idle_timeout(
    response: reqwest::Response,
    idle_timeout: Duration,
) -> impl Stream<Item = anyhow::Result<SseEvent>> + Send {
    sse_events_from_byte_stream(response.bytes_stream(), Some(idle_timeout))
}

fn sse_events_from_byte_stream<B, E>(
    byte_stream: impl Stream<Item = std::result::Result<B, E>> + Send + 'static,
    idle_timeout: Option<Duration>,
) -> impl Stream<Item = anyhow::Result<SseEvent>> + Send
where
    B: AsRef<[u8]> + Send + 'static,
    E: std::fmt::Display + Send + Sync + 'static,
{
    async_stream::stream! {
        futures::pin_mut!(byte_stream);
        let mut buffer: Vec<u8> = Vec::new();
        let mut current_event = SseEvent::default();

        loop {
            let next_chunk = if let Some(timeout) = idle_timeout {
                match tokio::time::timeout(timeout, byte_stream.next()).await {
                    Ok(next_chunk) => next_chunk,
                    Err(_) => {
                        yield Err(anyhow::Error::new(SseError::IdleTimeout {
                        timeout_ms: timeout.as_millis() as u64,
                    }));
                        return;
                    }
                }
            } else {
                byte_stream.next().await
            };

            let Some(chunk_result) = next_chunk else {
                break;
            };

            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    yield Err(anyhow::Error::new(SseError::ReadError {
                    message: e.to_string(),
                }));
                    return;
                }
            };

            buffer.extend_from_slice(chunk.as_ref());

            // Process all complete lines in the buffer.
            while let Some(newline_pos) = buffer.iter().position(|&b| b == b'\n') {
                let line = String::from_utf8_lossy(&buffer[..newline_pos]).into_owned();
                let line = line.trim_end_matches('\r');
                buffer.drain(..newline_pos + 1);

                if line.is_empty() {
                    // Empty line signals the end of an event.
                    if !current_event.data.is_empty() {
                        yield Ok(std::mem::take(&mut current_event));
                    }
                } else if line.starts_with(':') {
                    // Comment line — ignore.
                } else if let Some(value) = line.strip_prefix("data:") {
                    let value = value.strip_prefix(' ').unwrap_or(value);
                    if !current_event.data.is_empty() {
                        current_event.data.push('\n');
                    }
                    current_event.data.push_str(value);
                } else if let Some(value) = line.strip_prefix("event:") {
                    current_event.event =
                        Some(value.strip_prefix(' ').unwrap_or(value).to_string());
                } else if let Some(value) = line.strip_prefix("id:") {
                    current_event.id =
                        Some(value.strip_prefix(' ').unwrap_or(value).to_string());
                }
                // `retry:` and unknown fields are silently ignored.
            }
        }

        // Flush any trailing event that wasn't terminated by a blank line.
        if !current_event.data.is_empty() {
            yield Ok(current_event);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::time::Duration;

    use futures::stream;

    use super::*;

    async fn collect_sse_events(
        events: impl Stream<Item = anyhow::Result<SseEvent>>,
    ) -> Vec<anyhow::Result<SseEvent>> {
        futures::pin_mut!(events);
        let mut results = Vec::new();
        while let Some(event) = events.next().await {
            results.push(event);
        }
        results
    }

    #[tokio::test]
    async fn sse_events_from_byte_stream_preserves_event_id_and_multiline_data() {
        let chunks = stream::iter(vec![Ok::<_, io::Error>(
            b": comment\nretry: 1000\nevent: delta\nid: evt-1\ndata: {\"a\":1}\ndata: {\"b\":2}\n\n"
                .to_vec(),
        )]);

        let results = collect_sse_events(sse_events_from_byte_stream(chunks, None)).await;

        assert_eq!(results.len(), 1);
        let event = results[0].as_ref().expect("event should parse");
        assert_eq!(event.event.as_deref(), Some("delta"));
        assert_eq!(event.id.as_deref(), Some("evt-1"));
        assert_eq!(event.data, "{\"a\":1}\n{\"b\":2}");
    }

    #[tokio::test]
    async fn sse_events_from_byte_stream_reports_read_errors() {
        let chunks = stream::iter(vec![Err::<Vec<u8>, _>(io::Error::new(
            io::ErrorKind::ConnectionReset,
            "connection reset",
        ))]);

        let results = collect_sse_events(sse_events_from_byte_stream(chunks, None)).await;

        assert_eq!(results.len(), 1);
        let error = results[0]
            .as_ref()
            .expect_err("read error should propagate");
        assert!(error.to_string().contains("SSE stream read error"));
    }

    #[tokio::test]
    async fn sse_events_from_byte_stream_reports_idle_timeout() {
        let chunks = stream::pending::<Result<Vec<u8>, io::Error>>();

        let results = collect_sse_events(sse_events_from_byte_stream(
            chunks,
            Some(Duration::from_millis(1)),
        ))
        .await;

        assert_eq!(results.len(), 1);
        let error = results[0]
            .as_ref()
            .expect_err("idle timeout should fail the stream");
        assert!(error.to_string().contains("SSE stream idle timeout"));
    }

    #[tokio::test]
    async fn sse_events_from_byte_stream_handles_utf8_split_across_chunks() {
        // Euro sign '€' is 3 bytes: E2 82 AC.
        // Split so chunk 1 has first 2 bytes, chunk 2 has the last byte.
        let chunks = stream::iter(vec![
            Ok::<_, io::Error>(b"data: hello \xe2\x82".to_vec()),
            Ok::<_, io::Error>(b"\xac world\n\n".to_vec()),
        ]);

        let results = collect_sse_events(sse_events_from_byte_stream(chunks, None)).await;

        assert_eq!(results.len(), 1);
        let event = results[0].as_ref().expect("event should parse");
        assert_eq!(event.data, "hello \u{20ac} world");
        // Must NOT contain replacement character U+FFFD
        assert!(!event.data.contains('\u{fffd}'), "data contained U+FFFD: {:?}", event.data);
    }

    #[test]
    fn sse_stream_error_has_typed_variants() {
        // Test that we can match on error type instead of string content
        use super::SseError;

        // Test idle timeout variant
        let timeout_err = SseError::IdleTimeout { timeout_ms: 5000 };
        assert!(matches!(timeout_err, SseError::IdleTimeout { .. }));
        assert!(timeout_err.is_retryable());

        // Test read error variant
        let read_err = SseError::ReadError { message: "connection reset".to_string() };
        assert!(matches!(read_err, SseError::ReadError { .. }));
        assert!(read_err.is_retryable());
    }

    #[tokio::test]
    async fn sse_stream_yields_typed_timeout_error() {
        // Test that idle timeout produces SseError::IdleTimeout
        let chunks = stream::pending::<Result<Vec<u8>, std::io::Error>>();

        let results = collect_sse_events(sse_events_from_byte_stream(
            chunks,
            Some(Duration::from_millis(1)),
        )).await;

        assert_eq!(results.len(), 1);
        let error = results[0].as_ref().expect_err("should be error");

        // Downcast anyhow error to SseError
        let sse_error = error.downcast_ref::<SseError>().expect("should be SseError");
        assert!(matches!(sse_error, SseError::IdleTimeout { .. }));
    }

    #[tokio::test]
    async fn sse_stream_yields_typed_read_error() {
        // Test that read errors produce SseError::ReadError
        use std::io;

        let chunks = stream::iter(vec![Err::<Vec<u8>, _>(io::Error::new(
            io::ErrorKind::ConnectionReset,
            "connection reset",
        ))]);

        let results = collect_sse_events(sse_events_from_byte_stream(chunks, None)).await;

        assert_eq!(results.len(), 1);
        let error = results[0].as_ref().expect_err("should be error");

        // Downcast anyhow error to SseError
        let sse_error = error.downcast_ref::<SseError>().expect("should be SseError");
        assert!(matches!(sse_error, SseError::ReadError { .. }));
    }
}
