//! SSE reconnection with exponential backoff.
//!
//! Wraps the basic SSE stream with retry logic for resilient streaming
//! from LLM providers that may drop connections.

use std::time::Duration;
use rand::Rng;

use futures::Stream;
use futures::StreamExt;
use tracing::warn;

use super::streaming::{sse_stream, sse_stream_with_idle_timeout, SseError, SseEvent};

/// Configuration for stream reconnection behavior.
#[derive(Debug, Clone, PartialEq)]
pub struct ReconnectConfig {
    /// Maximum number of reconnection attempts. None = unlimited.
    pub max_retries: Option<u32>,
    /// Initial backoff duration.
    pub initial_backoff: Duration,
    /// Maximum backoff duration.
    pub max_backoff: Duration,
    /// Backoff multiplier (default 2.0).
    pub multiplier: f64,
    /// Idle timeout between chunks.
    pub idle_timeout: Option<Duration>,
}

impl Default for ReconnectConfig {
    fn default() -> Self {
        Self {
            max_retries: Some(3),
            initial_backoff: Duration::from_secs(1),
            max_backoff: Duration::from_secs(30),
            multiplier: 2.0,
            idle_timeout: Some(Duration::from_secs(300)),
        }
    }
}

/// Create a reconnecting SSE stream.
///
/// The `make_request` closure is called to create each new HTTP request.
/// On connection drop or idle timeout, the stream retries with exponential
/// backoff. Clean stream completion (including `[DONE]`) returns normally.
///
/// # Retry behavior
///
/// - Request creation errors: retry with backoff
/// - Idle timeout: retry with backoff
/// - Read errors (connection reset): retry with backoff
/// - Other stream errors: propagate immediately
/// - `[DONE]` sentinel: return immediately (no retry)
#[must_use]
pub fn reconnecting_sse_stream<F, Fut>(
    make_request: F,
    config: ReconnectConfig,
) -> impl Stream<Item = anyhow::Result<SseEvent>> + Send
where
    F: Fn() -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = anyhow::Result<reqwest::Response>> + Send + 'static,
{
    async_stream::stream! {
        let mut attempt = 0u32;
        let mut backoff = config.initial_backoff;

        loop {
            attempt += 1;

            // Check retry limit.
            if let Some(max) = config.max_retries {
                if attempt > max {
                    yield Err(anyhow::anyhow!(
                        "SSE stream failed after {max} reconnection attempts"
                    ));
                    return;
                }
            }

            // Make the request.
            let response = match make_request().await {
                Ok(r) => r,
                Err(e) => {
                    warn!("SSE request failed (attempt {attempt}): {e}");
                    tokio::time::sleep(backoff).await;
                    backoff = next_backoff(backoff, config.multiplier, config.max_backoff);
                    continue;
                }
            };

            // Stream events — box to unify types.
            let event_stream: std::pin::Pin<Box<dyn Stream<Item = anyhow::Result<SseEvent>> + Send>> =
                if let Some(timeout) = config.idle_timeout {
                    Box::pin(sse_stream_with_idle_timeout(response, timeout))
                } else {
                    Box::pin(sse_stream(response))
                };

            futures::pin_mut!(event_stream);

            let mut had_events = false;
            while let Some(event) = event_stream.next().await {
                match event {
                    Ok(sse_event) => {
                        had_events = true;
                        let is_done: bool = sse_event.is_done();
                        yield Ok(sse_event);
                        if is_done {
                            return;
                        }
                    }
                    Err(e) => {
                        // Classify errors using typed SseError instead of string matching.
                        let is_retryable = e.downcast_ref::<SseError>()
                            .is_some_and(|sse| sse.is_retryable());

                        if is_retryable {
                            warn!("SSE stream error (attempt {attempt}): {e}");
                            break;
                        }

                        // Non-retryable errors propagate.
                        yield Err(e);
                        return;
                    }
                }
            }

            // Stream ended cleanly after producing events — done.
            if had_events {
                return;
            }

            // No events received — backoff and retry.
            warn!("SSE stream ended with no events (attempt {attempt})");
            tokio::time::sleep(backoff).await;
            backoff = next_backoff(backoff, config.multiplier, config.max_backoff);
        }
    }
}

fn next_backoff(current: Duration, multiplier: f64, max: Duration) -> Duration {
    let next = Duration::from_secs_f64(current.as_secs_f64() * multiplier);
    let capped = if next > max { max } else { next };

    // Equal jitter: backoff/2 + random(0, backoff/2)
    // Range is [capped/2, capped] — prevents near-zero delays while still spreading load
    let mut rng = rand::rng();
    let half_nanos = u64::try_from(capped.as_nanos() / 2)
        .expect("max_backoff exceeds u64 nanosecond range");
    let jittered_nanos = half_nanos + rng.random_range(0..=half_nanos);
    Duration::from_nanos(jittered_nanos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_backoff_doubles_until_max() {
        // With jitter, backoff values are randomized but should be within expected ranges
        let config = ReconnectConfig::default();
        let mut backoff = config.initial_backoff;

        assert_eq!(backoff, Duration::from_secs(1));

        // After first call: expected deterministic = 2s, jittered should be in [1s, 2s]
        backoff = next_backoff(backoff, config.multiplier, config.max_backoff);
        assert!(backoff <= Duration::from_secs(2), "Backoff {:?} exceeded expected max of 2s", backoff);

        // After second call: expected deterministic = 4s, jittered should be in [2s, 4s]
        backoff = next_backoff(backoff, config.multiplier, config.max_backoff);
        assert!(backoff <= Duration::from_secs(4), "Backoff {:?} exceeded expected max of 4s", backoff);

        // After third call: expected deterministic = 8s, jittered should be in [4s, 8s]
        backoff = next_backoff(backoff, config.multiplier, config.max_backoff);
        assert!(backoff <= Duration::from_secs(8), "Backoff {:?} exceeded expected max of 8s", backoff);

        // After fourth call: expected deterministic = 16s, jittered should be in [8s, 16s]
        backoff = next_backoff(backoff, config.multiplier, config.max_backoff);
        assert!(backoff <= Duration::from_secs(16), "Backoff {:?} exceeded expected max of 16s", backoff);

        // After fifth call: expected deterministic = 30s (capped at max), jittered should be in [15s, 30s]
        backoff = next_backoff(backoff, config.multiplier, config.max_backoff);
        assert!(backoff <= Duration::from_secs(30), "Backoff {:?} exceeded max of 30s", backoff);

        // After sixth call: should still be capped at max
        backoff = next_backoff(backoff, config.multiplier, config.max_backoff);
        assert!(backoff <= Duration::from_secs(30), "Backoff {:?} exceeded max of 30s", backoff);
    }

    #[test]
    fn reconnect_config_default_values() {
        let config = ReconnectConfig::default();
        assert_eq!(config.max_retries, Some(3));
        assert_eq!(config.initial_backoff, Duration::from_secs(1));
        assert_eq!(config.max_backoff, Duration::from_secs(30));
        assert_eq!(config.multiplier, 2.0);
        assert_eq!(config.idle_timeout, Some(Duration::from_secs(300)));
    }

    #[test]
    fn reconnect_config_has_partial_eq() {
        // Test that ReconnectConfig implements PartialEq
        let config1 = ReconnectConfig::default();
        let config2 = ReconnectConfig::default();
        assert_eq!(config1, config2);

        // Test inequality
        let mut config3 = ReconnectConfig::default();
        config3.max_retries = Some(5);
        assert_ne!(config1, config3);
    }

    #[test]
    fn reconnect_config_can_be_compared_in_tests() {
        // Test that we can assert on full config values
        let expected = ReconnectConfig {
            max_retries: Some(3),
            initial_backoff: Duration::from_secs(1),
            max_backoff: Duration::from_secs(30),
            multiplier: 2.0,
            idle_timeout: Some(Duration::from_secs(300)),
        };
        assert_eq!(ReconnectConfig::default(), expected);
    }

    #[test]
    fn next_backoff_jitters_within_range() {
        // With jitter, repeated calls with same input should produce DIFFERENT values
        let current = Duration::from_secs(2);
        let multiplier = 2.0;
        let max = Duration::from_secs(30);

        let mut values = Vec::new();
        for _ in 0..10 {
            values.push(next_backoff(current, multiplier, max));
        }

        // Should have at least 2 different values (jitter working)
        let unique: std::collections::HashSet<_> = values.iter().collect();
        assert!(unique.len() > 1, "Expected jitter to produce varied values, got all same: {:?}", values[0]);
    }

    #[test]
    fn next_backoff_jitter_stays_within_bounds() {
        // Jittered backoff should be between [0, next_deterministic]
        let current = Duration::from_secs(10);
        let multiplier = 2.0;
        let max = Duration::from_secs(30);

        for _ in 0..100 {
            let backoff = next_backoff(current, multiplier, max);
            assert!(backoff <= max, "Backoff {:?} exceeded max {:?}", backoff, max);
            assert!(backoff <= Duration::from_secs(20), "Backoff {:?} exceeded deterministic next {:?}", backoff, Duration::from_secs(20));
        }
    }

    #[test]
    fn next_backoff_equal_jitter_minimum_bound() {
        // Equal jitter: backoff/2 + random(0, backoff/2) → minimum is backoff/2
        // Input: current=10s, multiplier=2.0, max=30s → capped=20s
        // Equal jitter range: [10s, 20s]
        // Assert every result >= 10s (half of capped) — fails with old 0..=capped impl
        let current = Duration::from_secs(10);
        let multiplier = 2.0;
        let max = Duration::from_secs(30);
        let minimum = Duration::from_secs(10);

        for _ in 0..100 {
            let backoff = next_backoff(current, multiplier, max);
            assert!(
                backoff >= minimum,
                "Equal jitter backoff {:?} should be >= {:?}",
                backoff,
                minimum
            );
        }
    }

    #[tokio::test]
    async fn reconnecting_sse_stream_classifies_retryable_errors() {
        use tokio::io::AsyncWriteExt;
        use tokio::net::TcpListener;

        // Mock server that accepts connections and sends SSE headers,
        // then goes idle to trigger SseError::IdleTimeout.
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let connection_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let cc = connection_count.clone();
        let server_handle = tokio::spawn(async move {
            loop {
                let (mut socket, _) = match listener.accept().await {
                    Ok(c) => c,
                    Err(_) => break,
                };
                let n = cc.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                // Send SSE headers but no data → idle timeout triggers
                let response = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n";
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.flush().await;
                // Keep connection open — idle timeout fires on the client side
                // Wait a bit then drop
                tokio::time::sleep(Duration::from_millis(200)).await;
                drop(socket);
                if n >= 3 {
                    break;
                }
            }
        });

        // Configure with short idle timeout and limited retries
        let config = ReconnectConfig {
            max_retries: Some(2),
            initial_backoff: Duration::from_millis(1),
            max_backoff: Duration::from_millis(10),
            multiplier: 2.0,
            idle_timeout: Some(Duration::from_millis(50)),
        };

        let base_url = format!("http://{addr}");
        let make_request = move || {
            let url = base_url.clone();
            async move { reqwest::get(&url).await.map_err(|e| anyhow::anyhow!(e)) }
        };

        let stream = reconnecting_sse_stream(make_request, config);
        tokio::pin!(stream);
        let mut got_retryable_error = false;
        while let Some(result) = stream.next().await {
            match result {
                Ok(_) => {} // Ignore events
                Err(e) => {
                    // After retries exhausted, we get the final error.
                    // But SseError::IdleTimeout should have been recognized as retryable
                    // (retries happened before this point).
                    let msg = e.to_string();
                    if msg.contains("reconnection attempts") {
                        got_retryable_error = true;
                    }
                }
            }
        }
        // Stream should have retried (connection_count > 1) and eventually
        // given up with "failed after N reconnection attempts"
        let connections = connection_count.load(std::sync::atomic::Ordering::SeqCst);
        assert!(
            connections > 1,
            "Expected retries (multiple connections), got {connections}"
        );
        assert!(
            got_retryable_error,
            "Expected exhausted-retries error after retrying idle timeouts"
        );

        server_handle.abort();
    }

    #[test]
    fn non_retryable_error_would_propagate() {
        // Verify that errors NOT wrapping SseError are not classified as retryable.
        // This tests the downcast pattern used in the fixed error handling.
        use crate::streaming::SseError;

        // A retryable error: wraps SseError::IdleTimeout
        let retryable: anyhow::Error = anyhow::Error::new(SseError::IdleTimeout { timeout_ms: 5000 });
        assert!(
            retryable.downcast_ref::<SseError>().is_some(),
            "SseError should downcast"
        );
        assert!(
            retryable.downcast_ref::<SseError>().unwrap().is_retryable(),
            "IdleTimeout should be retryable"
        );

        // A non-retryable error: plain anyhow error, no SseError
        let non_retryable: anyhow::Error = anyhow::anyhow!("some other failure");
        assert!(
            non_retryable.downcast_ref::<SseError>().is_none(),
            "Plain error should NOT downcast to SseError"
        );
    }

    #[test]
    #[should_panic(expected = "exceeds u64 nanosecond range")]
    fn next_backoff_panics_on_absurd_duration() {
        // u64::MAX nanoseconds is ~584 years. Anything larger should panic
        // instead of silently truncating.
        let absurd_max = Duration::from_secs(1200 * 365 * 24 * 3600); // 600 years
        let current = absurd_max; // current = max, so capped = max
        let _ = next_backoff(current, 2.0, absurd_max);



    }
}
