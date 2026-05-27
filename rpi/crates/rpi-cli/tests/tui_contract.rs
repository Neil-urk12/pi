use rpi_cli::{safe_tool_summary, validate_tui_mode};
use serde_json::json;

#[test]
fn validate_tui_mode_rejects_single_prompt_mode() {
    let error = validate_tui_mode(false, true).unwrap_err();

    assert!(error.to_string().contains("--tui"));
    assert!(error.to_string().contains("interactive"));
}

#[test]
fn validate_tui_mode_accepts_interactive_tui() {
    validate_tui_mode(true, true).unwrap();
}


#[test]
fn safe_tool_summary_uses_builtin_allowlist() {
    assert_eq!(
        safe_tool_summary("read_file", &json!({"path": "CONTEXT.md", "limit": 10})),
        Some("CONTEXT.md".to_string())
    );
    assert_eq!(
        safe_tool_summary("grep", &json!({"pattern": "secret", "path": "src"})),
        Some("pattern secret in src".to_string())
    );
    assert_eq!(
        safe_tool_summary("unknown", &json!({"token": "do-not-leak"})),
        None
    );
}
