//! System prompt construction for the agent.

use crate::types::ToolDefinition;

/// Build a system prompt that includes tool descriptions.
pub fn build_system_prompt(
    base_prompt: Option<&str>,
    tools: &[ToolDefinition],
    working_dir: &str,
) -> String {
    let mut prompt = String::new();

    // Base prompt
    prompt.push_str(base_prompt.unwrap_or("You are a helpful coding assistant."));
    prompt.push_str("\n\n");

    // Working directory context
    prompt.push_str(&format!("Working directory: {working_dir}\n\n"));

    // Tool descriptions
    if !tools.is_empty() {
        prompt.push_str("## Available Tools\n\n");
        prompt.push_str("You have access to the following tools. Use them to help the user.\n\n");

        for tool in tools {
            prompt.push_str(&format!("### {}\n", tool.name));
            prompt.push_str(&format!("{}\n\n", tool.description));

            // Add parameter schema if present
            if tool.parameters != serde_json::Value::Null {
                prompt.push_str("Parameters:\n");
                prompt.push_str(&format!(
                    "```json\n{}\n```\n\n",
                    serde_json::to_string_pretty(&tool.parameters).unwrap_or_default()
                ));
            }
        }
    }

    prompt
}

/// Format a tool call for human-readable display.
pub fn format_tool_call_for_display(name: &str, args: &serde_json::Value) -> String {
    let args_str = match serde_json::to_string_pretty(args) {
        Ok(s) => s,
        Err(_) => format!("{args}"),
    };
    format!("Tool: {name}\nArgs: {args_str}")
}

/// Format a tool result for display.
pub fn format_tool_result_for_display(result: &str, is_error: bool) -> String {
    if is_error {
        format!("Error: {result}")
    } else {
        // Truncate long results for display
        let truncated = if result.len() > 500 {
            let end = result.char_indices().nth(500).map(|(i, _)| i).unwrap_or(result.len());
            format!("{}...(truncated)", &result[..end])
        } else {
            result.to_string()
        };
        format!("Result: {truncated}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_build_system_prompt_no_tools() {
        let prompt = build_system_prompt(Some("Be helpful."), &[], "/home/user/project");
        assert!(prompt.contains("Be helpful."));
        assert!(prompt.contains("/home/user/project"));
        assert!(!prompt.contains("Available Tools"));
    }

    #[test]
    fn test_build_system_prompt_with_tools() {
        let tools = vec![ToolDefinition {
            name: "bash".to_string(),
            description: "Execute a bash command.".to_string(),
            parameters: json!({"type": "object", "properties": {"command": {"type": "string"}}}),
        }];
        let prompt = build_system_prompt(None, &tools, "/tmp");
        assert!(prompt.contains("bash"));
        assert!(prompt.contains("Execute a bash command."));
        assert!(prompt.contains("Available Tools"));
    }

    #[test]
    fn test_format_tool_call() {
        let args = json!({"command": "ls -la"});
        let display = format_tool_call_for_display("bash", &args);
        assert!(display.contains("bash"));
        assert!(display.contains("ls -la"));
    }

    #[test]
    fn test_format_tool_result() {
        let display = format_tool_result_for_display("file.txt", false);
        assert!(display.contains("Result:"));

        let error_display = format_tool_result_for_display("not found", true);
        assert!(error_display.contains("Error:"));
    }

    #[test]
    fn test_format_tool_result_truncates_utf8_safely() {
        // Each 'ä' is 2 bytes in UTF-8
        let mut result = String::new();
        for _ in 0..300 {
            result.push_str("ä");
        }
        // 600 bytes total, should truncate at 500 chars (not bytes)
        let formatted = format_tool_result_for_display(&result, false);
        assert!(formatted.contains("...(truncated)"));
    }

    #[test]
    fn test_format_tool_result_short_string() {
        let result = "short string";
        let formatted = format_tool_result_for_display(result, false);
        assert!(!formatted.contains("...(truncated)"));
    }
}
