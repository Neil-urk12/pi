//! Error types for the pi agent.

use thiserror::Error;

/// File-system error category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileErrorCode {
    /// File or directory does not exist.
    NotFound,
    /// Permission denied by the operating system or policy.
    PermissionDenied,
    /// Input path was invalid for the requested operation.
    InvalidPath,
    /// File content was malformed or could not be decoded.
    InvalidData,
    /// An unspecified file-system error occurred.
    Other,
}

impl std::fmt::Display for FileErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let code = match self {
            Self::NotFound => "not_found",
            Self::PermissionDenied => "permission_denied",
            Self::InvalidPath => "invalid_path",
            Self::InvalidData => "invalid_data",
            Self::Other => "other",
        };
        f.write_str(code)
    }
}

/// Structured file-system error.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
#[error("file error {code} at {path}: {message}")]
pub struct FileError {
    /// Machine-readable file error code.
    pub code: FileErrorCode,
    /// Path involved in the operation.
    pub path: String,
    /// Human-readable error message.
    pub message: String,
}

impl FileError {
    /// Create a file-system error with a code, path, and message.
    pub fn new(code: FileErrorCode, path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code,
            path: path.into(),
            message: message.into(),
        }
    }
}

/// Process or command execution error category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionErrorCode {
    /// Execution was aborted.
    Aborted,
    /// Execution exceeded its timeout.
    Timeout,
    /// Shell was unavailable.
    ShellUnavailable,
    /// Process exited unsuccessfully.
    ExitStatus,
    /// An unspecified execution error occurred.
    Other,
}

impl std::fmt::Display for ExecutionErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let code = match self {
            Self::Aborted => "aborted",
            Self::Timeout => "timeout",
            Self::ShellUnavailable => "shell_unavailable",
            Self::ExitStatus => "exit_status",
            Self::Other => "other",
        };
        f.write_str(code)
    }
}

/// Structured process or command execution error.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
#[error("execution error {code}: {message}")]
pub struct ExecutionError {
    /// Machine-readable execution error code.
    pub code: ExecutionErrorCode,
    /// Human-readable error message.
    pub message: String,
}

impl ExecutionError {
    /// Create an execution error with a code and message.
    pub fn new(code: ExecutionErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

/// Structured compaction error.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
#[error("compaction error: {message}")]
pub struct CompactionError {
    /// Human-readable error message.
    pub message: String,
}

impl CompactionError {
    /// Create a compaction error.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Structured session persistence error.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
#[error("session error: {message}")]
pub struct SessionError {
    /// Human-readable error message.
    pub message: String,
}

impl SessionError {
    /// Create a session error.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Structured extension runtime error.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
#[error("extension error: {message}")]
pub struct ExtensionError {
    /// Human-readable error message.
    pub message: String,
}

impl ExtensionError {
    /// Create an extension error.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Structured LLM provider error.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum ProviderError {
    /// Provider returned a non-success HTTP status.
    #[error("http error ({status}): {body}")]
    Http {
        /// HTTP status code returned by the provider.
        status: u16,
        /// Provider response body or error text.
        body: String,
    },

    /// Provider rate-limited the request.
    #[error("rate limited ({status}): {body}")]
    RateLimited {
        /// HTTP status code returned by the provider.
        status: u16,
        /// Provider response body or error text.
        body: String,
    },

    /// Provider rejected the request because the context window was exceeded.
    #[error("context window exceeded: {token_count}/{context_window}")]
    ContextOverflow {
        /// Number of tokens in the request.
        token_count: u32,
        /// Model context window size.
        context_window: u32,
    },

    /// Provider returned malformed or unexpected response data.
    #[error("malformed response: {message}")]
    MalformedResponse {
        /// Human-readable error message.
        message: String,
    },

    /// Provider stream failed or ended unexpectedly.
    #[error("stream error: {message}")]
    Stream {
        /// Human-readable error message.
        message: String,
    },

    /// Provider request timed out.
    #[error("timeout after {milliseconds}ms")]
    Timeout {
        /// Timeout duration in milliseconds.
        milliseconds: u64,
    },

    /// Other provider error.
    #[error("{message}")]
    Other {
        /// Human-readable error message.
        message: String,
    },
}

impl ProviderError {
    /// Build a provider error from an HTTP status code and response body.
    pub fn http_status(status: u16, body: impl Into<String>) -> Self {
        let body = body.into();

        if status == 429 {
            Self::RateLimited { status, body }
        } else {
            Self::Http { status, body }
        }
    }
}

/// The primary error type for all pi agent operations.
#[derive(Error, Debug)]
pub enum PiError {
    /// Errors from an LLM provider (API failures, rate limits, etc.).
    #[error(transparent)]
    Provider(#[from] ProviderError),

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

    /// Compaction accounting errors.
    #[error(transparent)]
    Compaction(#[from] CompactionError),

    /// Structured file-system error.
    #[error(transparent)]
    File(#[from] FileError),

    /// Structured process or command execution error.
    #[error(transparent)]
    Execution(#[from] ExecutionError),

    /// Structured session persistence error.
    #[error(transparent)]
    Session(#[from] SessionError),

    /// Structured extension runtime error.
    #[error(transparent)]
    Extension(#[from] ExtensionError),

}

impl PiError {
    /// Create a provider error from a message string.
    ///
    /// Prefer this over `From<String>` for explicit, self-documenting error construction.
    pub fn provider(message: impl Into<String>) -> Self {
        PiError::Provider(ProviderError::Other { message: message.into() })
    }
}


/// Convenience alias for `Result<T, PiError>`.
pub type Result<T> = std::result::Result<T, PiError>;
