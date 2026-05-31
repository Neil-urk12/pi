//! Context-overflow classification helpers.

use crate::error::{PiError, ProviderError};
use crate::types::{AssistantMessage, StopReason};

/// Check if a provider error indicates context-window overflow.
pub fn is_context_overflow_error(error: &PiError) -> bool {
    match error {
        PiError::Provider(ProviderError::ContextOverflow { .. }) => true,
        PiError::Provider(ProviderError::Other { message }) => is_context_overflow_text(message),
        _ => false,
    }
}

/// Check if an assistant message indicates context-window overflow.
///
/// This covers explicit provider error text, silent overflow detected from
/// prompt-token usage, and providers that stop with `length` after filling the
/// context window without producing output.
pub fn is_context_overflow_message(
    message: &AssistantMessage,
    context_window: Option<u32>,
) -> bool {
    if message.stop_reason == StopReason::Error {
        if let Some(error_message) = &message.error_message {
            if is_context_overflow_text(error_message) {
                return true;
            }
        }
    }

    if let Some(context_window) = context_window {
        if message.stop_reason == StopReason::Stop && message.usage.prompt_tokens > context_window {
            return true;
        }

        if message.stop_reason == StopReason::Length && message.usage.completion_tokens == 0 {
            let threshold = (context_window as f64 * 0.99).floor() as u32;
            if message.usage.prompt_tokens >= threshold {
                return true;
            }
        }
    }

    false
}

fn is_context_overflow_text(message: &str) -> bool {
    let lower = message.to_lowercase();
    if lower.trim().is_empty() || is_known_non_overflow(&lower) {
        return false;
    }

    if lower.contains("context") {
        return lower.contains("length")
            || lower.contains("window")
            || lower.contains("exceeded")
            || lower.contains("too long")
            || lower.contains("too many tokens")
            || lower.contains("token limit")
            || lower.contains("too large")
            || lower.contains("input is too long")
            || lower.contains("message too long");
    }

    contains_all(&lower, &["prompt", "too long"])
        || lower.contains("request_too_large")
        || lower.contains("input is too long for requested model")
        || contains_all(&lower, &["maximum", "context length"])
        || contains_all(&lower, &["input token count", "exceeds", "maximum"])
        || lower.contains("exceeds maximum input tokens")
        || lower.contains("maximum prompt length is")
        || lower.contains("reduce the length of the messages")
        || contains_all(&lower, &["maximum allowed input length", "tokens"])
        || contains_all(
            &lower,
            &["input (", "tokens)", "longer than", "context length"],
        )
        || lower.contains("exceeds the limit of")
        || lower.contains("exceeds the available context size")
        || lower.contains("greater than the context length")
        || lower.contains("context window exceeds limit")
        || lower.contains("exceeded model token limit")
        || contains_all(
            &lower,
            &["too large for model with", "maximum context length"],
        )
        || lower.contains("model_context_window_exceeded")
        || contains_all(&lower, &["prompt too long", "exceeded", "context length"])
        || lower.contains("context_length_exceeded")
        || lower.contains("context length exceeded")
        || lower.contains("too many tokens")
        || lower.contains("token limit exceeded")
        || is_empty_body_status_overflow(&lower)
}

fn is_known_non_overflow(lower: &str) -> bool {
    lower.starts_with("throttling error:")
        || lower.starts_with("service unavailable:")
        || lower.contains("rate limit")
        || lower.contains("too many requests")
        || lower.contains("request body")
        || lower.contains("context switch")
        || lower.contains("context deadline")
        || lower.contains("context cancel")
}

fn contains_all(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().all(|needle| haystack.contains(needle))
}
fn is_empty_body_status_overflow(lower: &str) -> bool {
    let trimmed = lower.trim();
    (trimmed.starts_with("400") || trimmed.starts_with("413")) && trimmed.contains("no body")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::PiError;
    use crate::types::{AssistantMessage, ContentBlock, StopReason, Usage};

    fn msg_with_error(error_message: &str) -> AssistantMessage {
        AssistantMessage {
            content: Vec::new(),
            api: "test".to_string(),
            provider: "test".to_string(),
            model: "test".to_string(),
            response_model: None,
            response_id: None,
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
            stop_reason: StopReason::Error,
            error_message: Some(error_message.to_string()),
            timestamp: 0,
        }
    }

    fn msg_with_usage(
        stop_reason: StopReason,
        prompt_tokens: u32,
        completion_tokens: u32,
    ) -> AssistantMessage {
        AssistantMessage {
            content: vec![ContentBlock::Text {
                text: "response".to_string(),
            }],
            api: "test".to_string(),
            provider: "test".to_string(),
            model: "test".to_string(),
            response_model: None,
            response_id: None,
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens.saturating_add(completion_tokens),
            },
            stop_reason,
            error_message: None,
            timestamp: 0,
        }
    }

    // --- is_context_overflow_error ---

    #[test]
    fn error_context_window_exceeded() {
        let err = PiError::provider("context window exceeded");
        assert!(is_context_overflow_error(&err));
    }

    #[test]
    fn error_maximum_context_length() {
        let err = PiError::provider("maximum context length exceeded");
        assert!(is_context_overflow_error(&err));
    }

    #[test]
    fn error_context_length_exceeded() {
        let err = PiError::provider("context_length_exceeded");
        assert!(is_context_overflow_error(&err));
    }

    #[test]
    fn error_rate_limit_not_overflow() {
        let err = PiError::provider("rate limit exceeded");
        assert!(!is_context_overflow_error(&err));
    }

    #[test]
    fn error_request_body_not_overflow() {
        let err =
            PiError::provider("request body too many tokens, please reduce the size");
        assert!(!is_context_overflow_error(&err));
    }

    #[test]
    fn error_http_429_not_overflow() {
        let err = PiError::Provider(ProviderError::RateLimited {
            status: 429,
            body: "rate limit exceeded".to_string(),
        });
        assert!(!is_context_overflow_error(&err));
    }

    #[test]
    fn error_too_many_requests_not_overflow() {
        let err = PiError::provider("too many requests, try again later");
        assert!(!is_context_overflow_error(&err));
    }

    #[test]
    fn error_throttling_not_overflow() {
        let err = PiError::provider("Throttling error: please wait");
        assert!(!is_context_overflow_error(&err));
    }

    #[test]
    fn error_empty_message_not_overflow() {
        let err = PiError::provider("");
        assert!(!is_context_overflow_error(&err));
    }

    #[test]
    fn error_whitespace_only_not_overflow() {
        let err = PiError::provider("   \n\t  ");
        assert!(!is_context_overflow_error(&err));
    }

    #[test]
    fn error_context_overflow_variant() {
        let err = PiError::Provider(ProviderError::ContextOverflow {
            token_count: 100_000,
            context_window: 50_000,
        });
        assert!(is_context_overflow_error(&err));
    }

    #[test]
    fn error_non_provider_not_overflow() {
        let err = PiError::Config("bad config".to_string());
        assert!(!is_context_overflow_error(&err));
    }

    // --- is_context_overflow_message: error text path ---

    #[test]
    fn message_error_context_window_exceeded() {
        let m = msg_with_error("context window exceeded");
        assert!(is_context_overflow_message(&m, Some(100_000)));
    }

    #[test]
    fn message_error_maximum_context_length() {
        let m = msg_with_error("maximum context length exceeded");
        assert!(is_context_overflow_message(&m, Some(100_000)));
    }

    #[test]
    fn message_error_token_limit_exceeded() {
        let m = msg_with_error("token limit exceeded");
        assert!(is_context_overflow_message(&m, Some(100_000)));
    }

    #[test]
    fn message_error_rate_limit_not_overflow() {
        let m = msg_with_error("rate limit exceeded");
        assert!(!is_context_overflow_message(&m, Some(100_000)));
    }

    #[test]
    fn message_error_request_body_not_overflow() {
        let m = msg_with_error("request body too many tokens");
        assert!(!is_context_overflow_message(&m, Some(100_000)));
    }

    #[test]
    fn message_error_empty_not_overflow() {
        let m = msg_with_error("");
        assert!(!is_context_overflow_message(&m, Some(100_000)));
    }

    // --- is_context_overflow_message: usage-based path ---

    #[test]
    fn message_usage_exceeds_context_window() {
        let m = msg_with_usage(StopReason::Stop, 200_001, 10);
        assert!(is_context_overflow_message(&m, Some(200_000)));
    }

    #[test]
    fn message_usage_below_context_window() {
        let m = msg_with_usage(StopReason::Stop, 50_000, 100);
        assert!(!is_context_overflow_message(&m, Some(200_000)));
    }

    #[test]
    fn message_usage_length_stop_zero_completion() {
        let m = msg_with_usage(StopReason::Length, 1_048_512, 0);
        assert!(is_context_overflow_message(&m, Some(1_048_576)));
    }

    #[test]
    fn message_usage_length_stop_with_completion() {
        let m = msg_with_usage(StopReason::Length, 100_000, 4_096);
        assert!(!is_context_overflow_message(&m, Some(200_000)));
    }

    #[test]
    fn message_usage_length_stop_below_threshold() {
        // 1000 << 99% of 200000 (198000) -- not overflow
        let m = msg_with_usage(StopReason::Length, 1_000, 0);
        assert!(!is_context_overflow_message(&m, Some(200_000)));
    }

    #[test]
    fn message_no_context_window_skips_usage_check() {
        let m = msg_with_usage(StopReason::Stop, 999_999, 0);
        assert!(!is_context_overflow_message(&m, None));
    }

    #[test]
    fn message_usage_equal_context_window() {
        // prompt_tokens == context_window (not >) -- not overflow
        let m = msg_with_usage(StopReason::Stop, 200_000, 0);
        assert!(!is_context_overflow_message(&m, Some(200_000)));
    }

    #[test]
    fn message_usage_exactly_one_over_context_window() {
        let m = msg_with_usage(StopReason::Stop, 200_001, 0);
        assert!(is_context_overflow_message(&m, Some(200_000)));
    }

    // --- is_context_overflow_message: stop reason edge cases ---

    #[test]
    fn message_error_stop_reason_with_non_overflow_text() {
        let m = msg_with_error("something went wrong");
        assert!(!is_context_overflow_message(&m, Some(100_000)));
    }

    #[test]
    fn message_stop_reason_without_usage_overflow() {
        let m = msg_with_usage(StopReason::Stop, 100, 50);
        assert!(!is_context_overflow_message(&m, Some(200_000)));
    }

    #[test]
    fn message_length_reason_no_context_window() {
        let m = msg_with_usage(StopReason::Length, 1_048_576, 0);
        assert!(!is_context_overflow_message(&m, None));
    }

    // --- mixed signals: error text says overflow but usage is low ---

    #[test]
    fn message_error_text_takes_precedence_over_low_usage() {
        let mut m = msg_with_error("context window exceeded");
        m.usage = Usage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
        };
        assert!(is_context_overflow_message(&m, Some(200_000)));
    }

    #[test]
    fn message_rate_limit_text_takes_precedence_over_high_usage() {
        // Even with high usage, "rate limit" in error text -> not overflow
        let mut m = msg_with_error("rate limit exceeded, try again later");
        m.usage = Usage {
            prompt_tokens: 200_001,
            completion_tokens: 0,
            total_tokens: 200_001,
        };
        assert!(!is_context_overflow_message(&m, Some(200_000)));
    }

    // --- is_context_overflow_text (private) ---

    #[test]
    fn text_context_window_exceeded() {
        assert!(is_context_overflow_text("context window exceeded"));
    }

    #[test]
    fn text_request_too_large() {
        assert!(is_context_overflow_text("request_too_large"));
    }

    #[test]
    fn text_input_too_long_for_model() {
        assert!(is_context_overflow_text(
            "input is too long for requested model",
        ));
    }

    #[test]
    fn text_context_length_exceeded() {
        assert!(is_context_overflow_text("context_length_exceeded"));
    }

    #[test]
    fn text_model_context_window_exceeded() {
        assert!(is_context_overflow_text("model_context_window_exceeded"));
    }

    #[test]
    fn text_exceeds_limit_of() {
        assert!(is_context_overflow_text("Input exceeds the limit of 128k tokens"));
    }

    #[test]
    fn text_reduce_length_of_messages() {
        assert!(is_context_overflow_text(
            "Please reduce the length of the messages.",
        ));
    }

    #[test]
    fn text_400_no_body() {
        assert!(is_context_overflow_text("400 no body"));
    }

    #[test]
    fn text_413_no_body() {
        assert!(is_context_overflow_text("413 no body"));
    }

    #[test]
    fn text_400_with_body_not_overflow() {
        assert!(!is_context_overflow_text("400 Bad Request: invalid parameter"));
    }

    #[test]
    fn text_context_too_long() {
        assert!(is_context_overflow_text("context is too long"));
    }

    #[test]
    fn text_context_too_many_tokens() {
        assert!(is_context_overflow_text("context: too many tokens"));
    }

    #[test]
    fn text_context_input_is_too_long() {
        assert!(is_context_overflow_text("context: input is too long"));
    }

    #[test]
    fn text_empty_string() {
        assert!(!is_context_overflow_text(""));
    }

    #[test]
    fn text_whitespace_only() {
        assert!(!is_context_overflow_text("   \n"));
    }

    #[test]
    fn text_rate_limit_exceeded() {
        assert!(!is_context_overflow_text("rate limit exceeded"));
    }

    #[test]
    fn text_too_many_requests() {
        assert!(!is_context_overflow_text("too many requests"));
    }

    #[test]
    fn text_service_unavailable() {
        assert!(!is_context_overflow_text(
            "Service Unavailable: please try again",
        ));
    }

    #[test]
    fn text_unrelated_error() {
        assert!(!is_context_overflow_text("connection timeout"));
    }

    // --- contains_all (private) ---

    #[test]
    fn contains_all_all_present() {
        assert!(contains_all("hello world", &["hello", "world"]));
    }

    #[test]
    fn contains_all_one_missing() {
        assert!(!contains_all("hello world", &["hello", "missing"]));
    }

    #[test]
    fn contains_all_empty_needles() {
        assert!(contains_all("anything", &[]));
    }

    // --- is_empty_body_status_overflow (private) ---

    #[test]
    fn empty_body_400() {
        assert!(is_empty_body_status_overflow("400 no body"));
    }

    #[test]
    fn empty_body_413() {
        assert!(is_empty_body_status_overflow("413 no body"));
    }

    #[test]
    fn empty_body_500() {
        assert!(!is_empty_body_status_overflow("500 no body"));
    }

    #[test]
    fn empty_body_400_with_body() {
        assert!(!is_empty_body_status_overflow("400 has body"));
    }
}
