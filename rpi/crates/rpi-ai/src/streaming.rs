//! SSE (Server-Sent Events) streaming utilities.
//!
//! Provides a parser for SSE streams from HTTP responses, commonly used by
//! LLM providers for streaming chat completions.

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
    async_stream::stream! {
        let mut byte_stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut current_event = SseEvent::default();

        while let Some(chunk_result) = byte_stream.next().await {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    yield Err(anyhow::anyhow!("SSE stream read error: {e}"));
                    return;
                }
            };

            buffer.push_str(&String::from_utf8_lossy(&chunk));

            // Process all complete lines in the buffer.
            while let Some(newline_pos) = buffer.find('\n') {
                let line = buffer[..newline_pos].to_string();
                buffer = buffer[newline_pos + 1..].to_string();

                // Handle \r\n line endings.
                let line = line.trim_end_matches('\r');

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
