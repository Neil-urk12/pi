# Rust Best Practices Reference Guide

## Table of Contents

1. [Error Handling](#1-error-handling)
2. [Async Rust Patterns](#2-async-rust-patterns)
3. [Serde Serialization](#3-serde-serialization)
4. [Rust Security](#4-rust-security)
5. [Testing Patterns](#5-testing-patterns)
6. [API Design](#6-api-design)

---

## 1. Error Handling

### thiserror vs anyhow: When to Use Which

**Use `thiserror` for:**
- Library crates that expose public APIs
- Structured error types with machine-readable variants
- Errors that callers need to match on programmatically
- Domain-specific error hierarchies

**Use `anyhow` for:**
- Application binaries (CLI tools, servers)
- Internal error propagation where callers don't need to match
- Quick prototyping and scripts
- Top-level error handling in `main()`

### Best Practice Pattern (from rpi project)

The rpi project uses **thiserror for structured domain errors** and **anyhow for application-level error handling**:

```rust
// crates/rpi-core/src/error.rs
use thiserror::Error;

/// Structured file-system error with machine-readable code
#[derive(Error, Debug, Clone, PartialEq, Eq)]
#[error("file error {code} at {path}: {message}")]
pub struct FileError {
    pub code: FileErrorCode,
    pub path: String,
    pub message: String,
}

/// Domain-specific error codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileErrorCode {
    NotFound,
    PermissionDenied,
    InvalidPath,
    InvalidData,
    Other,
}

/// Primary error type for the entire ecosystem
#[derive(Error, Debug)]
pub enum PiError {
    /// Errors from LLM providers
    #[error(transparent)]
    Provider(#[from] ProviderError),

    /// Tool execution errors
    #[error("tool '{tool}' error: {message}")]
    Tool {
        tool: String,
        message: String,
    },

    /// I/O errors
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization errors
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Convenience alias
pub type Result<T> = std::result::Result<T, PiError>;
```

### Error Context Chains

Add context to errors using `map_err` or custom methods:

```rust
// Good: Structured error with context
fn read_config(path: &Path) -> Result<Config> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FileError::new(
            FileErrorCode::NotFound,
            path.display().to_string(),
            e.to_string(),
        ))?;

    serde_json::from_str(&content)
        .map_err(|e| PiError::Serialization(e))
}

// Good: Builder pattern for error construction
impl ProviderError {
    pub fn http_status(status: u16, body: impl Into<String>) -> Self {
        let body = body.into();
        if status == 429 {
            Self::RateLimited { status, body }
        } else {
            Self::Http { status, body }
        }
    }
}
```

### Propagation Best Practices

```rust
// Use ? operator for automatic conversion
async fn fetch_data(url: &str) -> Result<Data> {
    let response = client.get(url).send().await?;  // reqwest::Error -> PiError
    let text = response.text().await?;
    let data: Data = serde_json::from_str(&text)?;  // serde_json::Error -> PiError
    Ok(data)
}

// Use .into() for explicit conversion when needed
fn convert_error(err: ProviderError) -> PiError {
    err.into()  // Uses #[from] derive
}
```

### Anti-Patterns to Avoid

```rust
// BAD: Using String for all errors
fn do_something() -> Result<(), String> {
    Err("something failed".to_string())  // Loses structure
}

// BAD: Catching errors you can't handle
fn process() -> Result<()> {
    match risky_operation() {
        Ok(val) => Ok(val),
        Err(e) => {
            log::error!("error: {}", e);
            Err(e)  // Just re-throwing without adding value
        }
    }
}

// GOOD: Let errors propagate with ?
fn process() -> Result<()> {
    let val = risky_operation()?;  // Clean propagation
    Ok(val)
}
```

---

## 2. Async Rust Patterns

### tokio Best Practices

**Runtime Configuration:**
```rust
// In Cargo.toml
[dependencies]
tokio = { version = "1", features = ["full"] }

// In main.rs
#[tokio::main]
async fn main() -> Result<()> {
    // Your async code here
}
```

**Spawning Tasks:**
```rust
use tokio::task;

// Spawn a blocking task for CPU-intensive work
let result = task::spawn_blocking(move || {
    expensive_computation()
}).await?;

// Spawn concurrent async tasks
let (result1, result2) = tokio::join!(
    fetch_data("url1"),
    fetch_data("url2"),
);
```

### async-trait for Async Trait Methods

The rpi project uses `async-trait` for defining async trait methods:

```rust
use async_trait::async_trait;

#[async_trait]
pub trait Provider: Send + Sync {
    /// The provider's identifier
    fn id(&self) -> &str;

    /// Send a chat completion request
    async fn chat(
        &self,
        model: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        config: &AgentConfig,
    ) -> Result<ChatResponse>;

    /// Stream a chat completion response
    async fn chat_stream(
        &self,
        model: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        config: &AgentConfig,
    ) -> Result<ChatStream>;
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn definition(&self) -> ToolDefinition;

    /// Execute the tool with JSON arguments
    async fn execute(&self, arguments: &serde_json::Value) -> Result<String>;
}
```

### Stream Handling

```rust
use futures::StreamExt;
use std::pin::Pin;

/// Type alias for boxed stream
pub type ChatStream = Pin<Box<dyn futures::Stream<Item = Result<StreamChunk>> + Send>>;

/// Process a stream of chunks
async fn process_stream(mut stream: ChatStream) -> Result<String> {
    let mut full_text = String::new();

    while let Some(chunk) = stream.next().await {
        match chunk? {
            StreamChunk { delta: Some(text), .. } => {
                full_text.push_str(&text);
                // Process text delta
            }
            StreamChunk { tool_calls, .. } if !tool_calls.is_empty() => {
                // Handle tool call deltas
                for tc in tool_calls {
                    if let Some(name) = tc.name {
                        // Process tool call start
                    }
                    if let Some(args) = tc.arguments_delta {
                        // Accumulate arguments
                    }
                }
            }
            StreamChunk { finish_reason: Some(reason), .. } => {
                // Stream completed
                break;
            }
            _ => {}
        }
    }

    Ok(full_text)
}
```

### Concurrency Patterns

```rust
use tokio::sync::{mpsc, oneshot};
use std::collections::HashMap;

/// Channel-based message passing
async fn agent_loop() -> Result<()> {
    let (tx, mut rx) = mpsc::channel(100);

    // Spawn worker tasks
    for i in 0..4 {
        let tx = tx.clone();
        tokio::spawn(async move {
            // Process tasks
            tx.send(format!("Worker {} done", i)).await.unwrap();
        });
    }

    // Collect results
    while let Some(msg) = rx.recv().await {
        println!("Received: {}", msg);
    }

    Ok(())
}

/// Select for racing futures
async fn race_operations() -> Result<Data> {
    tokio::select! {
        data = fetch_from_primary() => Ok(data?),
        data = fetch_from_fallback() => Ok(data?),
        _ = tokio::time::sleep(Duration::from_secs(30)) => {
            Err(PiError::Timeout(30_000))
        }
    }
}
```

---

## 3. Serde Serialization

### Derive Macros

```rust
use serde::{Deserialize, Serialize};

/// Basic derive with common traits
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelId {
    pub provider: String,
    pub model: String,
}

/// With default values
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct CacheRetention {
    #[serde(default)]
    pub ephemeral: bool,
}
```

### Enum Tagging Strategies

**Internally tagged (recommended for type-discriminated enums):**
```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    Image {
        media_type: String,
        data: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

// JSON: {"type": "text", "text": "hello"}
```

**Untagged (for flexible parsing):**
```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

// Accepts both: "hello" and [{"type": "text", "text": "hello"}]
```

**Externally tagged (default, for simple enums):**
```rust
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

// JSON: "system", "user", "assistant", "tool"
```

### Custom Deserializers

```rust
use serde::{Deserialize, Deserializer, Serializer};
use serde::de::{self, Visitor};
use std::fmt;

/// Custom string deserializer with validation
fn deserialize_model_id<'de, D>(deserializer: D) -> Result<ModelId, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    ModelId::parse(&s)
        .ok_or_else(|| de::Error::custom(format!("invalid model id: {}", s)))
}

/// Custom serializer for optional fields
fn serialize_optional_string<S>(
    value: &Option<String>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    match value {
        Some(s) => serializer.serialize_str(s),
        None => serializer.serialize_none(),
    }
}
```

### Field Attributes

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Message {
    pub role: Role,

    // Skip serialization if None
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContent>,

    // Default to false if missing
    #[serde(default)]
    pub is_error: bool,

    // Flatten nested struct
    #[serde(flatten)]
    pub metadata: MessageMetadata,

    // Rename field
    #[serde(rename = "tool_call_id")]
    pub call_id: Option<String>,
}
```

---

## 4. Rust Security

### Unsafe Code Guidelines

**Principles:**
1. **Minimize unsafe scope** - Wrap unsafe blocks in safe abstractions
2. **Document invariants** - Explain why the unsafe code is correct
3. **Use safe alternatives** - Prefer `safe` Rust whenever possible
4. **Audit dependencies** - Check for unsafe code in your dependency tree

```rust
// GOOD: Small unsafe block with clear invariant
pub fn get_unchecked(&self, index: usize) -> &T {
    // SAFETY: We ensure index < self.len() before calling
    assert!(index < self.len(), "index out of bounds");
    unsafe { self.data.get_unchecked(index) }
}

// BAD: Large unsafe blocks
pub fn process_data(data: &[u8]) -> &[u8] {
    unsafe {
        // 50 lines of unsafe code...
        // Hard to audit, easy to introduce bugs
    }
}
```

### Dependency Auditing

```bash
# Install cargo-audit
cargo install cargo-audit

# Audit dependencies for known vulnerabilities
cargo audit

# Generate SBOM (Software Bill of Materials)
cargo sbom > sbom.json

# Check for unmaintained dependencies
cargo install cargo-deny
cargo deny check advisories
```

### Secret Handling

```rust
use secrecy::{ExposeSecret, Secret};

/// Wrap sensitive data in Secret type
pub struct Config {
    pub api_key: Secret<String>,
    pub database_url: Secret<String>,
}

impl Config {
    pub fn new(api_key: String, database_url: String) -> Self {
        Self {
            api_key: Secret::new(api_key),
            database_url: Secret::new(database_url),
        }
    }

    pub fn connect(&self) -> Result<Connection> {
        // Use expose_secret() to access the value
        let url = self.database_url.expose_secret();
        Connection::connect(url)
    }
}

// Never log secrets
tracing::info!("Connecting to database");  // GOOD
tracing::info!("Connecting to {}", config.database_url.expose_secret());  // BAD
```

### Input Validation

```rust
use validator::{Validate, ValidationError};

#[derive(Validate)]
pub struct UserInput {
    #[validate(length(min = 1, max = 1000))]
    pub prompt: String,

    #[validate(range(min = 0, max = 100))]
    pub max_tokens: u32,

    #[validate(custom = "validate_model_id")]
    pub model: String,
}

fn validate_model_id(model: &str) -> Result<(), ValidationError> {
    if model.contains('/') && !model.starts_with('/') && !model.ends_with('/') {
        Ok(())
    } else {
        Err(ValidationError::new("invalid_model_id"))
    }
}
```

---

## 5. Testing Patterns

### Test Organization

```
crates/
  rpi-core/
    src/
      lib.rs
      error.rs
      types.rs
    tests/                    # Integration tests
      errors.rs
      types.rs
      event_stream.rs
```

### Unit Tests with #[cfg(test)]

```rust
// In src/types.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_id_parse_valid() {
        let id = ModelId::parse("openai/gpt-4o").unwrap();
        assert_eq!(id.provider, "openai");
        assert_eq!(id.model, "gpt-4o");
    }

    #[test]
    fn model_id_parse_invalid() {
        assert!(ModelId::parse("invalid").is_none());
        assert!(ModelId::parse("/model").is_none());
        assert!(ModelId::parse("provider/").is_none());
    }

    #[test]
    fn role_serialization_roundtrip() {
        let role = Role::Assistant;
        let json = serde_json::to_string(&role).unwrap();
        assert_eq!(json, "\"assistant\"");

        let parsed: Role = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, role);
    }
}
```

### Integration Tests

```rust
// tests/errors.rs
use rpi_core::{
    CompactionError, ExecutionError, ExecutionErrorCode, FileError, FileErrorCode,
    PiError, ProviderError, SessionError,
};

/// Verify all error types are Send + Sync
fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn structured_errors_are_send_and_sync() {
    assert_send_sync::<PiError>();
    assert_send_sync::<FileError>();
    assert_send_sync::<ExecutionError>();
    assert_send_sync::<ProviderError>();
    assert_send_sync::<SessionError>();
}

#[test]
fn file_error_display_includes_code_path_and_message() {
    let error = FileError::new(
        FileErrorCode::PermissionDenied,
        "/tmp/secret.txt",
        "read denied",
    );

    assert_eq!(
        error.to_string(),
        "file error permission_denied at /tmp/secret.txt: read denied"
    );
}

#[test]
fn structured_provider_errors_convert_to_pi_error() {
    let error = ProviderError::MalformedResponse {
        message: "missing choices".to_string(),
    };
    let pi_error = PiError::from(error);

    assert_eq!(
        pi_error.to_string(),
        "malformed response: missing choices"
    );
}

#[test]
fn provider_context_overflow_round_trips_correctly() {
    let error = ProviderError::ContextOverflow {
        token_count: 150_000,
        context_window: 128_000,
    };
    let pi_error = PiError::from(error);

    // Verify display contains structured data
    let msg = pi_error.to_string();
    assert!(msg.contains("150000"));
    assert!(msg.contains("128000"));

    // Round-trip: extract structured info back out
    match &pi_error {
        PiError::Provider(ProviderError::ContextOverflow {
            token_count,
            context_window,
        }) => {
            assert_eq!(*token_count, 150_000);
            assert_eq!(*context_window, 128_000);
        }
        other => panic!("expected Provider(ContextOverflow), got: {other:?}"),
    }
}
```

### Async Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_agent_loop() {
        let provider = MockProvider::new();
        let tools = vec![MockTool::new()];
        let config = AgentConfig::default();

        let result = run_agent_loop(&provider, &tools, &config, vec![]).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_stream_processing() {
        let stream = create_test_stream();
        let result = process_stream(stream).await.unwrap();

        assert_eq!(result, "expected output");
    }
}
```

### Mocking with Mockall

```rust
use mockall::automock;

#[automock]
#[async_trait]
pub trait Provider: Send + Sync {
    fn id(&self) -> &str;
    async fn chat(&self, model: &str, messages: &[Message]) -> Result<ChatResponse>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_with_mock_provider() {
        let mut mock = MockProvider::new();
        mock.expect_id()
            .return_const("mock".to_string());
        mock.expect_chat()
            .returning(|_, _| Ok(ChatResponse::default()));

        // Use mock in tests
        let result = mock.chat("model", &[]).await;
        assert!(result.is_ok());
    }
}
```

### Property-Based Testing

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn model_id_roundtrip(
        provider in "[a-z]{1,10}",
        model in "[a-z0-9-]{1,20}",
    ) {
        let id = ModelId::new(&provider, &model);
        let s = id.to_string();
        let parsed = ModelId::parse(&s).unwrap();
        prop_assert_eq!(id, parsed);
    }
}
```

---

## 6. API Design

### Pub Visibility

```rust
// GOOD: Explicit pub visibility
pub mod error;
pub mod types;

// Re-export only what's needed
pub use error::{PiError, Result};
pub use types::{Message, Role};

// Keep implementation details private
mod internal;
use internal::Helper;
```

### Sealed Traits

```rust
/// Sealed trait pattern - prevent external implementations
mod sealed {
    pub trait Sealed {}
}

pub trait Provider: sealed::Sealed + Send + Sync {
    fn id(&self) -> &str;
    // ...
}

// Only your crate can implement
impl sealed::Sealed for OpenAiProvider {}
impl Provider for OpenAiProvider {
    fn id(&self) -> &str { "openai" }
}
```

### Non-Exhaustive Enums

```rust
/// Future-proof enums that may gain variants
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    // Future variants won't break downstream code
}

impl FinishReason {
    pub fn is_terminal(&self) -> bool {
        match self {
            Self::Stop | Self::Length => true,
            Self::ToolCalls => false,
            // No wildcard needed - #[non_exhaustive] handles this
        }
    }
}
```

### Builder Pattern

```rust
#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub max_tool_rounds: u32,
    pub stream: bool,
    pub thinking_level: Option<ThinkingLevel>,
}

impl AgentConfig {
    pub fn builder() -> AgentConfigBuilder {
        AgentConfigBuilder::default()
    }
}

#[derive(Debug, Default)]
pub struct AgentConfigBuilder {
    max_tool_rounds: Option<u32>,
    stream: Option<bool>,
    thinking_level: Option<ThinkingLevel>,
}

impl AgentConfigBuilder {
    pub fn max_tool_rounds(mut self, rounds: u32) -> Self {
        self.max_tool_rounds = Some(rounds);
        self
    }

    pub fn stream(mut self, stream: bool) -> Self {
        self.stream = Some(stream);
        self
    }

    pub fn thinking_level(mut self, level: ThinkingLevel) -> Self {
        self.thinking_level = Some(level);
        self
    }

    pub fn build(self) -> AgentConfig {
        AgentConfig {
            max_tool_rounds: self.max_tool_rounds.unwrap_or(10),
            stream: self.stream.unwrap_or(false),
            thinking_level: self.thinking_level,
        }
    }
}

// Usage
let config = AgentConfig::builder()
    .max_tool_rounds(5)
    .stream(true)
    .thinking_level(ThinkingLevel::High)
    .build();
```

### Type Aliases for Complex Types

```rust
/// Simplify complex types with aliases
pub type ChatStream = Pin<Box<dyn futures::Stream<Item = Result<StreamChunk>> + Send>>;
pub type Result<T> = std::result::Result<T, PiError>;

/// Generic type aliases
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
pub type AsyncFn<T> = Box<dyn Fn() -> BoxFuture<'static, T> + Send + Sync>;
```

### Documentation

```rust
/// An LLM provider (OpenAI, Anthropic, Ollama, etc.).
///
/// Providers handle API communication, request formatting, and response parsing.
///
/// # Examples
///
/// ```rust
/// use rpi_core::{Provider, OpenAiProvider};
///
/// let provider = OpenAiProvider::new("api-key");
/// let response = provider.chat("gpt-4o", &messages, &tools, &config).await?;
/// ```
///
/// # Errors
///
/// Returns `PiError::Provider` if the API request fails.
#[async_trait]
pub trait Provider: Send + Sync {
    /// The provider's identifier (e.g., "openai", "anthropic").
    fn id(&self) -> &str;

    /// Send a chat completion request.
    ///
    /// # Arguments
    ///
    /// * `model` - The model identifier
    /// * `messages` - Conversation history
    /// * `tools` - Available tools for function calling
    /// * `config` - Agent configuration
    ///
    /// # Returns
    ///
    /// The complete chat response, or an error if the request fails.
    async fn chat(
        &self,
        model: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        config: &AgentConfig,
    ) -> Result<ChatResponse>;
}
```

---

## Quick Reference Checklist

### Error Handling
- [ ] Use `thiserror` for library errors with structured variants
- [ ] Use `anyhow` for application-level error handling
- [ ] Define a crate-level `Result<T>` type alias
- [ ] Implement `From` conversions with `#[from]`
- [ ] Add context to errors with meaningful messages
- [ ] Test error display messages and round-trips

### Async Rust
- [ ] Use `tokio` as the async runtime
- [ ] Use `async-trait` for async trait methods
- [ ] Box complex streams with `Pin<Box<dyn Stream>>`
- [ ] Use `tokio::select!` for racing futures
- [ ] Use `tokio::spawn_blocking` for CPU-intensive work
- [ ] Handle timeouts with `tokio::time::timeout`

### Serde
- [ ] Use `#[serde(tag = "type")]` for type-discriminated enums
- [ ] Use `#[serde(rename_all = "snake_case")]` for JSON conventions
- [ ] Use `#[serde(skip_serializing_if = "Option::is_none")]` for optional fields
- [ ] Use `#[serde(default)]` for fields with defaults
- [ ] Test serialization round-trips

### Security
- [ ] Minimize `unsafe` blocks and document invariants
- [ ] Run `cargo audit` regularly
- [ ] Use `secrecy` crate for sensitive data
- [ ] Validate all external input
- [ ] Never log secrets or credentials

### Testing
- [ ] Unit tests in `#[cfg(test)] mod tests`
- [ ] Integration tests in `tests/` directory
- [ ] Test error paths, not just happy paths
- [ ] Use `mockall` for trait mocking
- [ ] Test async code with `#[tokio::test]`
- [ ] Verify `Send + Sync` bounds on public types

### API Design
- [ ] Use explicit `pub` visibility
- [ ] Consider `#[non_exhaustive]` for enums that may grow
- [ ] Use builder pattern for complex configurations
- [ ] Document all public APIs with examples
- [ ] Re-export only necessary types from `lib.rs`
