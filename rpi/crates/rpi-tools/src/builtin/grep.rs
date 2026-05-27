//! Grep/search tool — regex search across files.

use async_trait::async_trait;
use regex::Regex;
use rpi_core::{PiError, Tool, ToolDefinition};
use serde_json::{json, Value};
use std::path::Path;

/// Maximum number of matches to return.
const MAX_MATCHES: usize = 200;

/// Default context lines around each match.
const DEFAULT_CONTEXT: usize = 0;

/// Regex search across files in a directory. Returns matching lines with
/// file path, line number, and optional surrounding context lines.
pub struct GrepTool;

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "grep".to_string(),
            description:
                "Search for a regex pattern across files in a directory. \
                 Returns matching lines with file paths and line numbers. \
                 Use `context` to show surrounding lines."
                    .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression pattern to search for."
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search (default: current directory)."
                    },
                    "context": {
                        "type": "integer",
                        "description": "Number of lines of context to show before and after each match (default: 0).",
                        "minimum": 0,
                        "maximum": 10
                    },
                    "glob": {
                        "type": "string",
                        "description": "Glob pattern to filter files (e.g. '*.rs')."
                    },
                    "ignore_case": {
                        "type": "boolean",
                        "description": "Case-insensitive search (default: false)."
                    }
                },
                "required": ["pattern"]
            }),
        }
    }

    async fn execute(&self, arguments: &Value) -> Result<String, PiError> {
        let pattern = arguments
            .get("pattern")
            .and_then(|v| v.as_str())
            .ok_or_else(|| PiError::Tool {
                tool: "grep".to_string(),
                message: "missing required parameter 'pattern'".to_string(),
            })?;

        let search_path = arguments
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or(".");

        let context = arguments
            .get("context")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(DEFAULT_CONTEXT)
            .min(10);

        let glob_pattern = arguments.get("glob").and_then(|v| v.as_str());

        let ignore_case = arguments
            .get("ignore_case")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Build regex — prepend (?i) for case-insensitive.
        let full_pattern = if ignore_case {
            format!("(?i){}", pattern)
        } else {
            pattern.to_string()
        };

        let re = Regex::new(&full_pattern).map_err(|e| PiError::Tool {
            tool: "grep".to_string(),
            message: format!("invalid regex '{}': {}", pattern, e),
        })?;

        let path = Path::new(search_path);
        let mut results: Vec<String> = Vec::new();
        let mut match_count: usize = 0;

        // Collect files to search.
        let files = if path.is_file() {
            vec![path.to_path_buf()]
        } else {
            self.collect_files(path, glob_pattern)?
        };

        for file_path in &files {
            // Skip binary files and non-UTF8.
            let content = match tokio::fs::read_to_string(file_path).await {
                Ok(c) => c,
                Err(_) => continue,
            };

            let lines: Vec<&str> = content.lines().collect();
            let file_display = file_path.display();

            for (idx, line) in lines.iter().enumerate() {
                if !re.is_match(line) {
                    continue;
                }

                match_count += 1;
                if match_count > MAX_MATCHES {
                    results.push(format!(
                        "... (stopped after {} matches, results truncated)",
                        MAX_MATCHES
                    ));
                    return Ok(results.join("\n"));
                }

                // Add context before.
                if context > 0 {
                    let start = idx.saturating_sub(context);
                    for ci in start..idx {
                        results.push(format!("{}:{}: {}", file_display, ci + 1, lines[ci]));
                    }
                }

                // The matching line.
                results.push(format!("{}:{}: {}", file_display, idx + 1, line));

                // Add context after.
                if context > 0 {
                    let end = (idx + context + 1).min(lines.len());
                    for ci in (idx + 1)..end {
                        results.push(format!("{}:{}: {}", file_display, ci + 1, lines[ci]));
                    }
                }

                // Separator between match groups.
                if context > 0 {
                    results.push("--".to_string());
                }
            }
        }

        if results.is_empty() {
            Ok(format!("no matches found for '{}'", pattern))
        } else {
            Ok(results.join("\n"))
        }
    }
}

impl GrepTool {
    /// Collect files to search, optionally filtering by glob pattern.
    fn collect_files(
        &self,
        root: &Path,
        glob_pattern: Option<&str>,
    ) -> Result<Vec<std::path::PathBuf>, PiError> {
        let mut files = Vec::new();

        let matcher = glob_pattern.map(|p| {
            // Build glob relative to root.
            let full = if p.starts_with('/') {
                p.to_string()
            } else {
                format!(
                    "{}{}{}",
                    root.display(),
                    std::path::MAIN_SEPARATOR,
                    p
                )
            };
            glob::Pattern::new(&full).map(|pattern| (pattern, full))
        });

        match matcher {
            Some(Ok((pattern, _full))) => {
                for entry in walkdir::WalkDir::new(root)
                    .follow_links(false)
                    .into_iter()
                    .filter_map(|e| e.ok())
                {
                    let path = entry.path();
                    if path.is_file() && pattern.matches_path(path) {
                        files.push(path.to_path_buf());
                    }
                }
            }
            Some(Err(e)) => {
                return Err(PiError::Tool {
                    tool: "grep".to_string(),
                    message: format!("invalid glob pattern '{}': {}", glob_pattern.unwrap_or(""), e),
                });
            }
            None => {
                for entry in walkdir::WalkDir::new(root)
                    .follow_links(false)
                    .into_iter()
                    .filter_map(|e| e.ok())
                {
                    let path = entry.path();
                    if path.is_file() {
                        files.push(path.to_path_buf());
                    }
                }
            }
        }

        Ok(files)
    }
}
