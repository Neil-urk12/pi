//! Streaming JSON repair and partial parse tests.

use rpi_core::{parse_json_with_repair, parse_streaming_json, repair_json};
use serde_json::json;

#[test]
fn repair_json_escapes_control_chars_and_invalid_backslashes() {
    let repaired = repair_json("{\"command\":\"echo hi\\q\n\"}");

    assert_eq!(repaired, "{\"command\":\"echo hi\\\\q\\n\"}");
    assert_eq!(
        parse_json_with_repair(&repaired).unwrap(),
        json!({ "command": "echo hi\\q\n" })
    );
}

#[test]
fn parse_streaming_json_returns_empty_object_for_empty_or_unrecoverable_input() {
    assert_eq!(parse_streaming_json(None), json!({}));
    assert_eq!(parse_streaming_json(Some("")), json!({}));
    assert_eq!(parse_streaming_json(Some("{\"path\":\"REA")), json!({}));
}

#[test]
fn parse_streaming_json_recovers_completed_fields_from_partial_object() {
    let parsed = parse_streaming_json(Some("{\"path\":\"README.md\",\"count\":"));

    assert_eq!(parsed, json!({ "path": "README.md" }));
}
