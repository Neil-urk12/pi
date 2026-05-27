//! Write file tool — writes content to a file, creating parent directories.

use async_trait::async_trait;
use rpi_core::{PiError, Tool, ToolDefinition};
use serde_json::{json, Value};

/// Writes content to a file. Creates parent directories if they don't exist.
pub struct WriteFileTool;

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "write_file".to_string(),
            description:
                "Write content to a file. Creates parent directories if they don't exist. \
                 Overwrites the file if it already exists."
                    .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write (relative or absolute)."
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file."
                    }
                },
                "required": ["path", "content"]
            }),
        }
    }

    async fn execute(&self, arguments: &Value) -> Result<String, PiError> {
        let path = arguments
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| PiError::Tool {
                tool: "write_file".to_string(),
                message: "missing required parameter 'path'".to_string(),
            })?;

        let content = arguments
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| PiError::Tool {
                tool: "write_file".to_string(),
                message: "missing required parameter 'content'".to_string(),
            })?;

        // Create parent directories.
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                tokio::fs::create_dir_all(parent)
                    .await
                    .map_err(|e| PiError::Tool {
                        tool: "write_file".to_string(),
                        message: format!("failed to create directories for '{}': {}", path, e),
                    })?;
            }
        }

        tokio::fs::write(path, content)
            .await
            .map_err(|e| PiError::Tool {
                tool: "write_file".to_string(),
                message: format!("failed to write '{}': {}", path, e),
            })?;

        let line_count = content.lines().count();
        let byte_count = content.len();
        tracing::debug!(path, line_count, byte_count, "wrote file");

        Ok(format!(
            "successfully wrote {} bytes ({} lines) to '{}'",
            byte_count, line_count, path
        ))
    }
}
