//! Tool registry — maps tool names to their implementations.
//!
//! The [`ToolRegistry`] is the central lookup table for all built-in tools.
//! It is populated with defaults via [`ToolRegistry::with_defaults()`] and can
//! also be extended with custom tool implementations.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use rpi_core::Tool;

use crate::builtin;

/// A registry of tool name → implementation.
///
/// # Examples
///
/// ```no_run
/// use rpi_tools::ToolRegistry;
///
/// let registry = ToolRegistry::with_defaults();
/// assert!(registry.get("read_file").is_some());
/// assert!(registry.get("nonexistent").is_none());
/// ```
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Create a registry pre-populated with all built-in tools.
    pub fn with_defaults() -> Self {
        Self::with_defaults_in(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
    }

    /// Create a registry pre-populated with all built-in tools, using the given working directory.
    pub fn with_defaults_in(cwd: PathBuf) -> Self {
        let mut reg = Self::new();
        reg.register(Arc::new(builtin::read_file::ReadFileTool));
        reg.register(Arc::new(builtin::write_file::WriteFileTool));
        reg.register(Arc::new(builtin::edit_file::EditFileTool));
        reg.register(Arc::new(builtin::bash::BashTool));
        reg.register(Arc::new(builtin::grep::GrepTool));
        reg.register(Arc::new(builtin::ls::LsTool));
        reg.register(Arc::new(builtin::find::FindTool::new(cwd)));
        reg
    }

    /// Register a tool. Overwrites any existing tool with the same name.
    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    /// Look up a tool by name.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.get(name)
    }

    /// Return all registered tool definitions (for passing to an LLM provider).
    pub fn definitions(&self) -> Vec<rpi_core::ToolDefinition> {
        self.tools.values().map(|t| t.definition()).collect()
    }

    /// Return all registered tools as a Vec of references (for the agent loop).
    pub fn tools(&self) -> Vec<&dyn Tool> {
        self.tools
            .values()
            .map(|t| t.as_ref() as &dyn Tool)
            .collect()
    }

    /// Return the names of all registered tools.
    pub fn names(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }

    /// Return the number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Return whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}
