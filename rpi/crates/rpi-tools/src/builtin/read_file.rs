//! Read file tool — reads file contents with line numbers.

use async_trait::async_trait;
use rpi_core::{PiError, Tool, ToolDefinition};
use serde_json::{json, Value};

/// Reads a file and returns its contents with line numbers.
///
/// Supports optional `offset` (1-indexed line to start from) and `limit`
/// (maximum number of lines to return). Binary files are detected and
/// rejected with an error.
pub struct ReadFileTool;

/// Maximum bytes to read (1 MiB).
const MAX_READ_BYTES: usize = 1024 * 1024;

/// Number of bytes to probe for binary detection.
const BINARY_PROBE_LEN: usize = 8192;

impl ReadFileTool {
    /// Detect if a slice of bytes is likely binary (contains null bytes).
    fn is_binary(bytes: &[u8]) -> bool {
        bytes.iter().any(|&b| b == 0)
    }
}

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "read_file".to_string(),
            description: "Read the contents of a file at the given path. Returns the file with \
                line numbers. Use `offset` and `limit` to read a specific range of lines."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read (relative or absolute)."
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from (1-indexed).",
                        "minimum": 1
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read.",
                        "minimum": 1
                    }
                },
                "required": ["path"]
            }),
        }
    }

    async fn execute(&self, arguments: &Value) -> Result<String, PiError> {
        let path = arguments
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| PiError::Tool {
                tool: "read_file".to_string(),
                message: "missing required parameter 'path'".to_string(),
            })?;

        let offset = arguments
            .get("offset")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(1);

        let limit = arguments
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize);

        // Read file with size guard.
        let raw = tokio::fs::read(path).await.map_err(|e| PiError::Tool {
            tool: "read_file".to_string(),
            message: format!("failed to read '{}': {}", path, e),
        })?;

        if raw.len() > MAX_READ_BYTES {
            return Err(PiError::Tool {
                tool: "read_file".to_string(),
                message: format!(
                    "file '{}' is too large ({} bytes, limit is {} bytes)",
                    path,
                    raw.len(),
                    MAX_READ_BYTES
                ),
            });
        }

        // Binary detection.
        let probe_end = raw.len().min(BINARY_PROBE_LEN);
        if Self::is_binary(&raw[..probe_end]) {
            return Err(PiError::Tool {
                tool: "read_file".to_string(),
                message: format!("file '{}' appears to be binary — refusing to read", path),
            });
        }

        let content = String::from_utf8(raw).map_err(|e| PiError::Tool {
            tool: "read_file".to_string(),
            message: format!("file '{}' is not valid UTF-8: {}", path, e),
        })?;

        // Apply offset and limit.
        let lines: Vec<&str> = content.lines().collect();
        let total = lines.len();

        if offset > total {
            return Ok(format!("// file '{}' has {} lines — offset {} is past the end", path, total, offset));
        }

        let start = offset - 1; // convert to 0-indexed
        let end = match limit {
            Some(n) => (start + n).min(total),
            None => total,
        };

        let width = end.to_string().len();
        let mut output = String::new();
        for (i, line) in lines[start..end].iter().enumerate() {
            let line_num = start + i + 1;
            output.push_str(&format!("{:>width$}|{}\n", line_num, line, width = width));
        }

        Ok(output)
    }
}
