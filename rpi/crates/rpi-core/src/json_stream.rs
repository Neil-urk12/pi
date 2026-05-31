//! Streaming JSON repair and partial parsing helpers.

use serde_json::Value;

/// Repair malformed JSON string literals by escaping raw control characters
/// and doubling backslashes before invalid escape characters.
pub fn repair_json(json: &str) -> String {
    let chars = json.chars().collect::<Vec<_>>();
    let mut repaired = String::new();
    let mut in_string = false;
    let mut index = 0;

    while index < chars.len() {
        let char = chars[index];

        if !in_string {
            repaired.push(char);
            if char == '"' {
                in_string = true;
            }
            index += 1;
            continue;
        }

        if char == '"' {
            repaired.push(char);
            in_string = false;
            index += 1;
            continue;
        }

        if char == '\\' {
            let Some(next_char) = chars.get(index + 1).copied() else {
                repaired.push_str("\\\\");
                index += 1;
                continue;
            };

            if next_char == 'u' && has_valid_unicode_escape(&chars, index + 2) {
                repaired.push_str("\\u");
                for offset in 2..6 {
                    repaired.push(chars[index + offset]);
                }
                index += 6;
                continue;
            }

            if is_valid_json_escape(next_char) {
                repaired.push('\\');
                repaired.push(next_char);
                index += 2;
                continue;
            }

            repaired.push_str("\\\\");
            index += 1;
            continue;
        }

        if is_control_character(char) {
            repaired.push_str(&escape_control_character(char));
        } else {
            repaired.push(char);
        }
        index += 1;
    }

    repaired
}

/// Parse JSON, retrying once after repairing malformed string literals.
pub fn parse_json_with_repair(json: &str) -> serde_json::Result<Value> {
    serde_json::from_str(json).or_else(|error| {
        let repaired = repair_json(json);
        if repaired == json {
            return Err(error);
        }
        serde_json::from_str(&repaired)
    })
}

/// Parse potentially incomplete JSON from streamed tool arguments.
///
/// Invalid or unrecoverable input returns an empty object.
pub fn parse_streaming_json(partial_json: Option<&str>) -> Value {
    let Some(partial_json) = partial_json else {
        return empty_object();
    };
    if partial_json.trim().is_empty() {
        return empty_object();
    }

    if let Ok(value) = parse_json_with_repair(partial_json) {
        return value;
    }

    parse_partial_object(partial_json).unwrap_or_else(empty_object)
}

fn is_valid_json_escape(char: char) -> bool {
    matches!(char, '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' | 'u')
}

fn has_valid_unicode_escape(chars: &[char], start: usize) -> bool {
    chars
        .get(start..start + 4)
        .is_some_and(|digits| digits.iter().all(|char| char.is_ascii_hexdigit()))
}

fn is_control_character(char: char) -> bool {
    (char as u32) <= 0x1f
}

fn escape_control_character(char: char) -> String {
    match char {
        '\u{08}' => "\\b".to_string(),
        '\u{0c}' => "\\f".to_string(),
        '\n' => "\\n".to_string(),
        '\r' => "\\r".to_string(),
        '\t' => "\\t".to_string(),
        _ => format!("\\u{:04x}", char as u32),
    }
}

fn empty_object() -> Value {
    Value::Object(serde_json::Map::new())
}

fn parse_partial_object(partial_json: &str) -> Option<Value> {
    let repaired = repair_json(partial_json.trim());
    if !repaired.starts_with('{') {
        return None;
    }

    if let Some(value) = parse_with_inferred_closing(&repaired) {
        return Some(value);
    }

    let commas = top_level_commas(&repaired);
    for comma in commas.into_iter().rev() {
        let prefix = &repaired[..comma];
        if let Some(value) = parse_with_inferred_closing(prefix) {
            return Some(value);
        }
    }

    None
}

fn parse_with_inferred_closing(prefix: &str) -> Option<Value> {
    let suffix = inferred_closing_suffix(prefix)?;
    let candidate = format!("{prefix}{suffix}");

    serde_json::from_str(&candidate).ok()
}

fn inferred_closing_suffix(json: &str) -> Option<String> {
    let mut in_string = false;
    let mut escaped = false;
    let mut stack = Vec::new();

    for char in json.chars() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            match char {
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        match char {
            '"' => in_string = true,
            '{' => stack.push('}'),
            '[' => stack.push(']'),
            '}' | ']' => {
                let expected = stack.pop()?;
                if char != expected {
                    return None;
                }
            }
            _ => {}
        }
    }

    if in_string || escaped {
        return None;
    }

    Some(stack.into_iter().rev().collect())
}

fn top_level_commas(json: &str) -> Vec<usize> {
    let mut commas = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    let mut depth = 0_u32;

    for (index, char) in json.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            match char {
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        match char {
            '"' => in_string = true,
            '{' | '[' => depth = depth.saturating_add(1),
            '}' | ']' => depth = depth.saturating_sub(1),
            ',' if depth == 1 => commas.push(index),
            _ => {}
        }
    }

    commas
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ------------------------------------------------------------------
    // repair_json
    // ------------------------------------------------------------------

    #[test]
    fn valid_json_passes_through_unchanged() {
        let input = r#"{"name":"test","age":30}"#;
        assert_eq!(repair_json(input), input);
    }

    #[test]
    fn repair_escapes_raw_newline_in_string() {
        let input = "{\"msg\":\"hello\nworld\"}";
        let repaired = repair_json(input);
        let parsed: serde_json::Value = serde_json::from_str(&repaired).unwrap();
        assert_eq!(parsed["msg"], json!("hello\nworld"));
    }

    #[test]
    fn repair_escapes_tab_in_string() {
        let input = "{\"msg\":\"hello\tworld\"}";
        let repaired = repair_json(input);
        let parsed: serde_json::Value = serde_json::from_str(&repaired).unwrap();
        assert_eq!(parsed["msg"], json!("hello\tworld"));
    }

    #[test]
    fn repair_doubles_bare_backslash_before_invalid_escape_char() {
        // \q is not a valid JSON escape, so the backslash should be doubled
        let input = "{\"path\":\"C:\\temp\\q\"}";
        let repaired = repair_json(input);
        assert!(serde_json::from_str::<serde_json::Value>(&repaired).is_ok());
    }

    #[test]
    fn repair_preserves_valid_unicode_escape() {
        let input = r#"{"emoji":"\u2764"}"#;
        assert_eq!(repair_json(input), input);
    }

    #[test]
    fn repair_escapes_control_character_u0001() {
        // \x01 embedded in a JSON string literal
        let mut input = String::from("{\"msg\":\"");
        input.push('\u{0001}');
        input.push_str("\"}");
        let repaired = repair_json(&input);
        assert!(repaired.contains("\\u0001"));
    }

    #[test]
    fn repair_doubles_trailing_backslash_at_end_of_input() {
        // A backslash at the very end of input with no following character gets doubled
        let input = "{\"msg\":\"hello\\";
        let repaired = repair_json(input);
        // The trailing \ should become \\
        assert!(repaired.ends_with("hello\\\\"));
    }

    // ------------------------------------------------------------------
    // parse_json_with_repair
    // ------------------------------------------------------------------

    #[test]
    fn parse_json_with_repair_succeeds_on_valid_json() {
        let input = r#"{"key":"value"}"#;
        let result = parse_json_with_repair(input).unwrap();
        assert_eq!(result, json!({"key": "value"}));
    }

    #[test]
    fn parse_json_with_repair_repairs_and_parses() {
        let input = "{\"msg\":\"hello\nworld\"}";
        let result = parse_json_with_repair(input).unwrap();
        assert_eq!(result["msg"], json!("hello\nworld"));
    }

    #[test]
    fn parse_json_with_repair_fails_on_unrecoverable() {
        let result = parse_json_with_repair("not json at all");
        assert!(result.is_err());
    }

    #[test]
    fn parse_json_with_repair_returns_original_error_if_repair_unchanged() {
        // Syntax error that repair_json cannot fix (structural, not string-related)
        let input = "{\"a\":1,,}";
        let err = parse_json_with_repair(input).unwrap_err();
        assert!(!err.to_string().is_empty());
    }

    // ------------------------------------------------------------------
    // parse_streaming_json
    // ------------------------------------------------------------------

    #[test]
    fn streaming_none_returns_empty_object() {
        assert_eq!(parse_streaming_json(None), json!({}));
    }

    #[test]
    fn streaming_empty_string_returns_empty_object() {
        assert_eq!(parse_streaming_json(Some("")), json!({}));
    }

    #[test]
    fn streaming_whitespace_only_returns_empty_object() {
        assert_eq!(parse_streaming_json(Some("   \n  ")), json!({}));
    }

    #[test]
    fn streaming_valid_json_passthrough() {
        let input = r#"{"name":"test","count":42}"#;
        let result = parse_streaming_json(Some(input));
        assert_eq!(result, json!({"name": "test", "count": 42}));
    }

    #[test]
    fn streaming_partial_object_recovers_complete_fields() {
        let input = "{\"name\":\"test\",\"age\":30";
        let result = parse_streaming_json(Some(input));
        assert_eq!(result["name"], json!("test"));
    }

    #[test]
    fn streaming_partial_nested_object() {
        let input = "{\"a\":{\"b\":1,\"c\":2}";
        let result = parse_streaming_json(Some(input));
        assert!(result.is_object());
    }

    #[test]
    fn streaming_unrecoverable_returns_empty() {
        let input = "{\"path\":\"REA}";
        let result = parse_streaming_json(Some(input));
        assert_eq!(result, json!({}));
    }

    // ------------------------------------------------------------------
    // inferred_closing_suffix / bracket matching
    // ------------------------------------------------------------------

    #[test]
    fn inferred_closing_handles_nested_brackets() {
        let prefix = "{\"a\":{\"b\":{\"c\":1}";
        let suffix = inferred_closing_suffix(prefix).unwrap();
        assert_eq!(suffix, "}}");
    }

    #[test]
    fn inferred_closing_handles_arrays_inside_objects() {
        let prefix = "{\"items\":[1,2,";
        let suffix = inferred_closing_suffix(prefix).unwrap();
        assert_eq!(suffix, "]}");
    }

    #[test]
    fn inferred_closing_returns_none_for_mismatched_brackets() {
        let prefix = "{\"a\":1}}";
        assert!(inferred_closing_suffix(prefix).is_none());
    }

    #[test]
    fn inferred_closing_returns_none_when_in_open_string() {
        let prefix = "{\"msg\":\"hello";
        assert!(inferred_closing_suffix(prefix).is_none());
    }

    #[test]
    fn inferred_closing_returns_empty_when_already_balanced() {
        let prefix = "{\"a\":1}";
        let suffix = inferred_closing_suffix(prefix).unwrap();
        assert!(suffix.is_empty());
    }

    // ------------------------------------------------------------------
    // top_level_commas
    // ------------------------------------------------------------------

    #[test]
    fn top_level_commas_finds_commas_at_depth_one() {
        let json_str = "{\"a\":1,\"b\":2,\"c\":3}";
        let commas = top_level_commas(json_str);
        assert_eq!(commas.len(), 2);
        assert_eq!(json_str.as_bytes()[commas[0]], b',');
        assert_eq!(json_str.as_bytes()[commas[1]], b',');
    }

    #[test]
    fn top_level_commas_ignores_commas_in_nested_objects() {
        let json_str = "{\"a\":{\"x\":1,\"y\":2},\"b\":3}";
        let commas = top_level_commas(json_str);
        assert_eq!(commas.len(), 1);
    }

    #[test]
    fn top_level_commas_ignores_commas_in_strings() {
        let json_str = "{\"a\":\"hello,world\",\"b\":1}";
        let commas = top_level_commas(json_str);
        assert_eq!(commas.len(), 1);
    }

    #[test]
    fn top_level_commas_ignores_commas_in_arrays() {
        let json_str = "{\"a\":[1,2,3],\"b\":1}";
        let commas = top_level_commas(json_str);
        assert_eq!(commas.len(), 1);
    }

    // ------------------------------------------------------------------
    // Unicode handling
    // ------------------------------------------------------------------

    #[test]
    fn unicode_escape_preserved_through_repair() {
        let input = r#"{"emoji":"\u2764","text":"cafe\u00e9"}"#;
        let repaired = repair_json(input);
        assert_eq!(repaired, input);
        let parsed: serde_json::Value = serde_json::from_str(&repaired).unwrap();
        assert_eq!(parsed["emoji"], json!("\u{2764}"));
    }

    #[test]
    fn multibyte_utf8_preserved_in_repair() {
        // Build a string with a multibyte UTF-8 character inside a JSON string
        let mut input = String::from("{\"msg\":\"");
        input.push('\u{1f600}');
        input.push_str(" hello ");
        input.push('\u{00e9}');
        input.push_str("\"}");
        let repaired = repair_json(&input);
        let parsed: serde_json::Value = serde_json::from_str(&repaired).unwrap();
        let msg = parsed["msg"].as_str().unwrap();
        assert!(msg.contains('\u{1f600}'));
        assert!(msg.contains('\u{00e9}'));
    }

    #[test]

    fn invalid_unicode_escape_passes_through() {
        // \\u is a valid JSON escape char, so repair_json passes it through
        // even when followed by non-hex digits — serde_json still rejects it
        let mut input = String::from("{\"msg\":\"");
        input.push('\\');
        input.push_str("uzzzz\"}");
        let repaired = repair_json(&input);
        // \\u treated as valid escape, passes through unchanged
        assert_eq!(repaired, input);
        // serde_json rejects the malformed \\uXXXX sequence
        assert!(serde_json::from_str::<serde_json::Value>(&repaired).is_err());
    }

    // ------------------------------------------------------------------
    // parse_partial_object internals
    // ------------------------------------------------------------------

    #[test]
    fn partial_object_returns_none_for_non_object_input() {
        assert!(parse_partial_object("[1,2,3]").is_none());
        assert!(parse_partial_object("\"hello\"").is_none());
        assert!(parse_partial_object("123").is_none());
    }

    #[test]
    fn partial_object_returns_none_for_empty() {
        assert!(parse_partial_object("").is_none());
    }

    #[test]
    fn partial_object_recovers_single_complete_field() {
        let input = "{\"name\":\"test\",";
        let result = parse_partial_object(input).unwrap();
        assert_eq!(result["name"], json!("test"));
    }

    #[test]
    fn partial_object_recovers_multiple_complete_fields() {
        let input = "{\"a\":1,\"b\":2,\"c\":3,";
        let result = parse_partial_object(input).unwrap();
        assert_eq!(result["a"], json!(1));
        assert_eq!(result["b"], json!(2));
        assert_eq!(result["c"], json!(3));
    }

    // ------------------------------------------------------------------
    // escape_control_character
    // ------------------------------------------------------------------

    #[test]
    fn escape_known_control_chars() {
        assert_eq!(escape_control_character('\u{08}'), "\\b");
        assert_eq!(escape_control_character('\u{0c}'), "\\f");
        assert_eq!(escape_control_character('\n'), "\\n");
        assert_eq!(escape_control_character('\r'), "\\r");
        assert_eq!(escape_control_character('\t'), "\\t");
    }

    #[test]
    fn escape_unknown_control_char_uses_unicode_format() {
        assert_eq!(escape_control_character('\u{01}'), "\\u0001");
        assert_eq!(escape_control_character('\u{1f}'), "\\u001f");
    }

    // ------------------------------------------------------------------
    // is_valid_json_escape
    // ------------------------------------------------------------------

    #[test]
    fn valid_json_escape_chars() {
        for c in ['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'] {
            assert!(is_valid_json_escape(c), "expected '{c}' to be valid");
        }
    }

    #[test]
    fn invalid_json_escape_chars() {
        for c in ['a', 'x', '0', ' ', 'Z'] {
            assert!(!is_valid_json_escape(c), "expected '{c}' to be invalid");
        }
    }

    // ------------------------------------------------------------------
    // is_control_character
    // ------------------------------------------------------------------

    #[test]
    fn control_character_boundary() {
        assert!(is_control_character('\u{00}'));
        assert!(is_control_character('\u{1f}'));
        assert!(!is_control_character('\u{20}'));
        assert!(!is_control_character('a'));
    }
}
