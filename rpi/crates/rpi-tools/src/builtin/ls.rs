//! Directory listing tool.

use async_trait::async_trait;
use rpi_core::{PiError, Tool, ToolDefinition};
use serde_json::{Value, json};
use std::path::Path;

/// Lists directory contents with metadata (type, size).
///
/// Directories are shown with a trailing `/`. Hidden files (dotfiles) are
/// included by default.
pub struct LsTool;

#[async_trait]
impl Tool for LsTool {
    fn name(&self) -> &str {
        "ls"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "ls".to_string(),
            description: "List directory contents. Returns entries sorted alphabetically with \
                 type indicators (/ for directories) and file sizes."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to list (default: current directory)."
                    }
                },
                "required": []
            }),
        }
    }

    async fn execute(&self, arguments: &Value) -> Result<String, PiError> {
        let dir = arguments
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or(".");

        let path = Path::new(dir);

        if !path.exists() {
            return Err(PiError::Tool {
                tool: "ls".to_string(),
                message: format!("path '{}' does not exist", dir),
            });
        }

        if path.is_file() {
            let meta = tokio::fs::metadata(path).await.map_err(|e| PiError::Tool {
                tool: "ls".to_string(),
                message: format!("failed to read metadata for '{}': {}", dir, e),
            })?;
            return Ok(format!("{} ({} bytes)", dir, meta.len()));
        }

        let mut entries = tokio::fs::read_dir(path).await.map_err(|e| PiError::Tool {
            tool: "ls".to_string(),
            message: format!("failed to read directory '{}': {}", dir, e),
        })?;

        let mut items: Vec<(String, bool, u64)> = Vec::new();

        loop {
            match entries.next_entry().await {
                Ok(Some(entry)) => {
                    let name = entry.file_name().to_string_lossy().to_string();
                    let meta = entry.metadata().await;
                    let is_dir = meta.as_ref().map(|m| m.is_dir()).unwrap_or(false);
                    let size = meta.as_ref().map(|m| m.len()).unwrap_or(0);
                    items.push((name, is_dir, size));
                }
                Ok(None) => break,
                Err(e) => {
                    return Err(PiError::Tool {
                        tool: "ls".to_string(),
                        message: format!("error reading directory '{}': {}", dir, e),
                    });
                }
            }
        }

        items.sort_by(|a, b| a.0.cmp(&b.0));

        let dirs_count = items.iter().filter(|(_, is_dir, _)| *is_dir).count();
        let files_count = items.len() - dirs_count;

        let mut output = String::new();
        for (name, is_dir, size) in &items {
            if *is_dir {
                output.push_str(&format!("{}/\n", name));
            } else {
                output.push_str(&format!("{} ({} bytes)\n", name, size));
            }
        }

        output.push_str(&format!(
            "\n{} entries ({} directories, {} files)",
            items.len(),
            dirs_count,
            files_count
        ));

        Ok(output)
    }
}
