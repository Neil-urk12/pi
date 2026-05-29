//! Find files by glob or regex pattern.

use async_trait::async_trait;
use rpi_core::{PiError, Tool, ToolDefinition};
use serde_json::{Value, json};
use std::path::PathBuf;

/// Find files matching a pattern.
pub struct FindTool {
    /// Working directory for relative paths.
    cwd: PathBuf,
}

impl FindTool {
    /// Create a new find tool.
    pub fn new(cwd: PathBuf) -> Self {
        Self { cwd }
    }
}

#[async_trait]
impl Tool for FindTool {
    fn name(&self) -> &str {
        "find"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "find".to_string(),
            description: "Find files matching a glob pattern. Returns file paths relative to the search directory.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match (e.g. '*.rs', '**/*.ts', 'src/**/*.json')"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (default: current working directory)"
                    }
                },
                "required": ["pattern"]
            }),
        }
    }

    async fn execute(&self, arguments: &Value) -> Result<String, PiError> {
        let pattern = arguments["pattern"].as_str().ok_or_else(|| PiError::Tool {
            tool: "find".to_string(),
            message: "Missing required parameter: pattern".to_string(),
        })?;

        let search_dir = arguments["path"]
            .as_str()
            .map(|p| self.cwd.join(p))
            .unwrap_or_else(|| self.cwd.clone());

        let glob_pattern = if search_dir.to_string_lossy().contains('*') {
            search_dir.to_string_lossy().to_string()
        } else {
            format!(
                "{}{}{}",
                search_dir.display(),
                std::path::MAIN_SEPARATOR,
                pattern
            )
        };

        let entries: Vec<String> = glob::glob(&glob_pattern)
            .map_err(|e| PiError::Tool {
                tool: "find".to_string(),
                message: format!("Invalid glob pattern: {e}"),
            })?
            .filter_map(|entry| entry.ok())
            .filter(|path| path.is_file())
            .map(|path| {
                path.strip_prefix(&self.cwd)
                    .unwrap_or(&path)
                    .display()
                    .to_string()
            })
            .take(500) // Limit results
            .collect();

        if entries.is_empty() {
            Ok("No files found matching pattern.".to_string())
        } else {
            Ok(entries.join("\n"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_find_tool() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("test.rs"), "fn main() {}").unwrap();
        std::fs::write(dir.path().join("lib.rs"), "pub mod foo;").unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src").join("main.rs"), "fn main() {}").unwrap();

        let tool = FindTool::new(dir.path().to_path_buf());

        // Find all .rs files
        let result = tool.execute(&json!({"pattern": "**/*.rs"})).await.unwrap();
        assert!(result.contains("test.rs"));
        assert!(result.contains("lib.rs"));
        assert!(result.contains("src/main.rs"));
    }
}
