//! Context-overflow classification parity tests.

use rpi_core::{
    AssistantMessage, PiError, StopReason, Usage, is_context_overflow_error,
    is_context_overflow_message,
};

fn assistant_with_error(error_message: &str) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: "openai-completions".to_string(),
        provider: "ollama".to_string(),
        model: "qwen3.5:35b".to_string(),
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

fn assistant_with_usage(
    stop_reason: StopReason,
    prompt_tokens: u32,
    completion_tokens: u32,
) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: "openai-completions".to_string(),
        provider: "xiaomi".to_string(),
        model: "mimo-v2.5-pro".to_string(),
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

#[test]
fn detects_explicit_provider_overflow_messages() {
    let message =
        assistant_with_error("400 `prompt too long; exceeded max context length by 100918 tokens`");

    assert!(is_context_overflow_message(&message, Some(32_768)));
    assert!(is_context_overflow_error(&PiError::provider(
        message.error_message.unwrap()
    )));
}

#[test]
fn excludes_rate_limit_and_service_errors_with_token_words() {
    let throttled =
        assistant_with_error("Throttling error: Too many tokens, please wait before trying again.");
    let rate_limited = assistant_with_error("Rate limit exceeded, please retry after 30 seconds.");
    let too_many_requests = assistant_with_error("Too many requests. Please slow down.");

    assert!(!is_context_overflow_message(&throttled, Some(200_000)));
    assert!(!is_context_overflow_message(&rate_limited, Some(200_000)));
    assert!(!is_context_overflow_message(
        &too_many_requests,
        Some(200_000)
    ));
}

#[test]
fn detects_silent_and_length_stop_overflow_from_usage() {
    let silent = assistant_with_usage(StopReason::Stop, 200_001, 10);
    let filled_context = assistant_with_usage(StopReason::Length, 1_048_512, 0);
    let ordinary_length = assistant_with_usage(StopReason::Length, 1_000, 4_096);

    assert!(is_context_overflow_message(&silent, Some(200_000)));
    assert!(is_context_overflow_message(
        &filled_context,
        Some(1_048_576)
    ));
    assert!(!is_context_overflow_message(
        &ordinary_length,
        Some(200_000)
    ));
}

#[test]
fn request_body_size_not_classified_as_overflow() {
    // "too many tokens" in a request-body-size context should NOT be classified
    // as context window overflow — it's about HTTP request body size.
    let msg = assistant_with_error(
        "400 Bad Request: request body too many tokens, please reduce the size",
    );
    assert!(!is_context_overflow_message(&msg, Some(200_000)));
    assert!(!is_context_overflow_error(&PiError::provider(
        msg.error_message.unwrap()
    )));
}
