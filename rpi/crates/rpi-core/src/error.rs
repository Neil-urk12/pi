//! Error types for the pi agent.

use thiserror::Error;

/// The primary error type for all pi agent operations.
#[derive(Error, Debug)]
pub enum PiError {
    /// Errors from an LLM provider (API failures, rate limits, etc.).
    #[error("provider error: {0}")]
    Provider(String),

    /// Errors during tool execution.
    #[error("tool '{tool}' error: {message}")]
    Tool {
        /// Name of the tool that failed.
        tool: String,
        /// Error message.
        message: String,
    },

    /// Configuration errors (missing fields, invalid values).
    #[error("config error: {0}")]
    Config(String),

    /// I/O errors (file system, network).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization/deserialization errors.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Request timed out.
    #[error("timeout after {0}ms")]
    Timeout(u64),

    /// Token limit exceeded.
    #[error("token limit exceeded: {used}/{limit}")]
    TokenLimit {
        /// Tokens used.
        used: u32,
        /// Token limit.
        limit: u32,
    },
}

/// Convenience alias for `Result<T, PiError>`.
pub type Result<T> = std::result::Result<T, PiError>;
