//! Edit file tool — performs text replacement in a file.

use async_trait::async_trait;
use rpi_core::{PiError, Tool, ToolDefinition};
use serde_json::{json, Value};

/// Performs text replacement in a file.
///
/// Reads the file, replaces all occurrences of `old_text` with `new_text`,
/// and writes the result back. Returns an error if `old_text` is not found.
pub struct EditFileTool;

#[async_trait]
impl Tool for EditFileTool {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "edit_file".to_string(),
            description:
                "Replace all occurrences of `old_text` with `new_text` in a file. \
                 Returns an error if the old text is not found."
                    .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit."
                    },
                    "old_text": {
                        "type": "string",
                        "description": "The text to search for."
                    },
                    "new_text": {
                        "type": "string",
                        "description": "The replacement text."
                    }
                },
                "required": ["path", "old_text", "new_text"]
            }),
        }
    }

    async fn execute(&self, arguments: &Value) -> Result<String, PiError> {
        let path = arguments
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| PiError::Tool {
                tool: "edit_file".to_string(),
                message: "missing required parameter 'path'".to_string(),
            })?;

        let old_text = arguments
            .get("old_text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| PiError::Tool {
                tool: "edit_file".to_string(),
                message: "missing required parameter 'old_text'".to_string(),
            })?;

        let new_text = arguments
            .get("new_text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| PiError::Tool {
                tool: "edit_file".to_string(),
                message: "missing required parameter 'new_text'".to_string(),
            })?;

        if old_text.is_empty() {
            return Err(PiError::Tool {
                tool: "edit_file".to_string(),
                message: "'old_text' must not be empty".to_string(),
            });
        }

        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| PiError::Tool {
                tool: "edit_file".to_string(),
                message: format!("failed to read '{}': {}", path, e),
            })?;

        let count = content.matches(old_text).count();
        if count == 0 {
            return Err(PiError::Tool {
                tool: "edit_file".to_string(),
                message: format!(
                    "'old_text' not found in '{}'. File was not modified.",
                    path
                ),
            });
        }

        let updated = content.replace(old_text, new_text);

        tokio::fs::write(path, &updated)
            .await
            .map_err(|e| PiError::Tool {
                tool: "edit_file".to_string(),
                message: format!("failed to write '{}': {}", path, e),
            })?;

        tracing::debug!(path, replacements = count, "edited file");

        Ok(format!(
            "successfully replaced {} occurrence(s) in '{}'",
            count, path
        ))
    }
}
