//! Structured error tests for Phase 0 error handling.

use rpi_core::{
    CompactionError, ExecutionError, ExecutionErrorCode, ExtensionError, FileError, FileErrorCode,
    PiError, ProviderError, SessionError,
};

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn structured_errors_are_send_and_sync() {
    assert_send_sync::<PiError>();
    assert_send_sync::<FileError>();
    assert_send_sync::<ExecutionError>();
    assert_send_sync::<ProviderError>();
    assert_send_sync::<SessionError>();
    assert_send_sync::<ExtensionError>();
}

#[test]
fn file_error_display_includes_code_path_and_message() {
    let error = FileError::new(
        FileErrorCode::PermissionDenied,
        "/tmp/secret.txt",
        "read denied",
    );

    assert_eq!(
        error.to_string(),
        "file error permission_denied at /tmp/secret.txt: read denied"
    );
}

#[test]
fn execution_error_display_includes_code_and_message() {
    let error = ExecutionError::new(ExecutionErrorCode::Timeout, "process exceeded 30s");

    assert_eq!(
        error.to_string(),
        "execution error timeout: process exceeded 30s"
    );
}

#[test]
fn structured_file_errors_convert_to_pi_error() {
    let error = FileError::new(FileErrorCode::NotFound, "missing.txt", "not found");
    let pi_error = PiError::from(error);

    assert_eq!(
        pi_error.to_string(),
        "file error not_found at missing.txt: not found"
    );
}

#[test]
fn structured_execution_errors_convert_to_pi_error() {
    let error = ExecutionError::new(ExecutionErrorCode::Aborted, "user cancelled");
    let pi_error = PiError::from(error);

    assert_eq!(
        pi_error.to_string(),
        "execution error aborted: user cancelled"
    );
}

#[test]
fn provider_error_http_status_categorizes_rate_limits() {
    let error = ProviderError::http_status(429, "slow down");

    assert_eq!(error.to_string(), "rate limited (429): slow down");
}

#[test]
fn structured_provider_errors_convert_to_pi_error() {
    let error = ProviderError::MalformedResponse {
        message: "missing choices".to_string(),
    };
    let pi_error = PiError::from(error);

    assert_eq!(
        pi_error.to_string(),
        "malformed response: missing choices"
    );
}

#[test]
fn provider_error_converts_to_pi_error_with_structured_info() {
    let error = ProviderError::Http {
        status: 503,
        body: "service unavailable".to_string(),
    };
    let pi_error = PiError::from(error);

    let msg = pi_error.to_string();
    assert!(msg.contains("503"), "should contain status code, got: {msg}");
    assert!(
        msg.contains("http error"),
        "should contain variant name, got: {msg}"
    );
}

#[test]
fn provider_context_overflow_round_trips_correctly() {
    let error = ProviderError::ContextOverflow {
        token_count: 150_000,
        context_window: 128_000,
    };
    let pi_error = PiError::from(error);

    let msg = pi_error.to_string();
    assert!(
        msg.contains("150000"),
        "should contain token count, got: {msg}"
    );
    assert!(
        msg.contains("128000"),
        "should contain context window, got: {msg}"
    );

    // Round-trip: extract structured info back out
    match &pi_error {
        PiError::Provider(ProviderError::ContextOverflow {
            token_count,
            context_window,
        }) => {
            assert_eq!(*token_count, 150_000);
            assert_eq!(*context_window, 128_000);
        }
        other => panic!("expected Provider(ContextOverflow), got: {other:?}"),
    }
}

#[test]
fn string_converts_to_pi_error_via_provider_constructor() {
    let pi_error = PiError::provider("something broke");

    match &pi_error {
        PiError::Provider(ProviderError::Other { message }) => {
            assert_eq!(message, "something broke");
        }
        other => panic!("expected Provider(Other), got: {other:?}"),
    }
}

#[test]
fn compaction_error_converts_to_pi_error() {
    let error = CompactionError::new("accounting bug");
    let pi_error = PiError::from(error);

    assert_eq!(pi_error.to_string(), "compaction error: accounting bug");
}

#[test]
fn pi_error_provider_constructor_creates_structured_provider_error() {
    let error = PiError::provider("something failed");
    assert!(
        matches!(&error, PiError::Provider(ProviderError::Other { message }) if message == "something failed")
    );
    assert!(error.to_string().contains("something failed"));
}

#[test]
fn pi_error_provider_constructor_accepts_format_args() {
    let error = PiError::provider(format!("HTTP {status}", status = 503));
    assert!(
        matches!(&error, PiError::Provider(ProviderError::Other { message }) if message.contains("503"))
    );
}

#[test]
fn provider_error_with_non_ascii_body_does_not_panic_on_display() {
    // Simulate the truncation pattern used in provider error bodies.
    // 511 ASCII chars + 1 two-byte UTF-8 char = 513 bytes, 512 characters.
    let text = "a".repeat(511) + "\u{00e9}"; // \u{00e9} = 'é', 2 UTF-8 bytes
    let safe_end = text.floor_char_boundary(512);
    let preview = format!("{}...(truncated)", &text[..safe_end]);
    assert!(
        preview.len() <= 530,
        "preview too long: {} bytes",
        preview.len()
    );
    assert!(
        std::str::from_utf8(preview.as_bytes()).is_ok(),
        "preview must be valid UTF-8"
    );
}
