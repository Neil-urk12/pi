//! Bash command execution tool.

use async_trait::async_trait;
use rpi_core::{PiError, Tool, ToolDefinition};
use serde_json::{Value, json};

/// Default timeout for bash commands: 120 seconds.
const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// Maximum timeout allowed: 10 minutes.
const MAX_TIMEOUT_SECS: u64 = 600;

/// Executes a shell command via `tokio::process::Command` and returns
/// combined stdout/stderr output. Supports an optional timeout in seconds.
pub struct BashTool;

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "bash".to_string(),
            description: "Execute a bash command and return stdout and stderr. \
                 Use `timeout` to set a time limit in seconds (default 120, max 600)."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute."
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 120, max 600).",
                        "minimum": 1,
                        "maximum": 600
                    }
                },
                "required": ["command"]
            }),
        }
    }

    async fn execute(&self, arguments: &Value) -> Result<String, PiError> {
        let command = arguments
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| PiError::Tool {
                tool: "bash".to_string(),
                message: "missing required parameter 'command'".to_string(),
            })?;

        let timeout = arguments
            .get("timeout")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_TIMEOUT_SECS)
            .min(MAX_TIMEOUT_SECS);

        tracing::debug!(command, timeout, "executing bash command");

        let output = tokio::time::timeout(
            std::time::Duration::from_secs(timeout),
            tokio::process::Command::new("bash")
                .arg("-c")
                .arg(command)
                .output(),
        )
        .await;

        match output {
            Err(_elapsed) => Err(PiError::Tool {
                tool: "bash".to_string(),
                message: format!("command timed out after {} seconds", timeout),
            }),
            Ok(Err(e)) => Err(PiError::Tool {
                tool: "bash".to_string(),
                message: format!("failed to execute command: {}", e),
            }),
            Ok(Ok(result)) => {
                let stdout = String::from_utf8_lossy(&result.stdout);
                let stderr = String::from_utf8_lossy(&result.stderr);
                let code = result.status.code().unwrap_or(-1);

                let mut output = String::new();

                if !stdout.is_empty() {
                    output.push_str("STDOUT:\n");
                    output.push_str(&stdout);
                }

                if !stderr.is_empty() {
                    if !output.is_empty() {
                        output.push('\n');
                    }
                    output.push_str("STDERR:\n");
                    output.push_str(&stderr);
                }

                if !result.status.success() {
                    if !output.is_empty() {
                        output.push('\n');
                    }
                    output.push_str(&format!("EXIT CODE: {}", code));
                }

                if output.is_empty() {
                    output = format!("(exit code: {})", code);
                }

                Ok(output)
            }
        }
    }
}
