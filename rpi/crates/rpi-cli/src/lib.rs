use anyhow::{Result, bail};
use serde_json::Value;

pub fn validate_tui_mode(interactive_mode: bool, tui: bool) -> Result<()> {
    if tui && !interactive_mode {
        bail!("--tui is only supported in interactive mode");
    }

    Ok(())
}

pub fn safe_tool_summary(name: &str, arguments: &Value) -> Option<String> {
    match name {
        "read_file" | "write_file" | "edit_file" | "ls" => string_field(arguments, "path"),
        "find" => match (
            string_field(arguments, "pattern"),
            string_field(arguments, "path"),
        ) {
            (Some(pattern), Some(path)) => Some(format!("{pattern} in {path}")),
            (Some(pattern), None) => Some(pattern),
            _ => None,
        },
        "grep" => match (
            string_field(arguments, "pattern"),
            string_field(arguments, "path"),
        ) {
            (Some(pattern), Some(path)) => Some(format!("pattern {pattern} in {path}")),
            (Some(pattern), None) => Some(format!("pattern {pattern}")),
            _ => None,
        },
        "bash" => Some("command".to_string()),
        _ => None,
    }
}

fn string_field(arguments: &Value, field: &str) -> Option<String> {
    arguments
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(String::from)
}
