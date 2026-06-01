//! Core types and traits for the pi coding agent.
//!
//! This crate provides the foundational types, error definitions, and trait
//! abstractions used across the pi agent ecosystem.

#![warn(missing_docs)]

pub mod agent_loop;
pub mod compaction;
mod error;
pub mod event_stream;
pub mod json_stream;
pub mod message_normalization;
pub mod overflow;
pub mod session;
pub mod system_prompt;
#[cfg(feature = "test-utils")]
pub mod test_utils;
pub mod token_estimation;
pub mod tool_validation;
mod traits;
mod types;

pub use agent_loop::{
    AgentEvent, AgentLoopConfig, apply_compaction, create_compaction_summary_message,
    run_agent_loop,
};
pub use compaction::{
    CompactionPreparation, CompactionResult, CompactionSettings, compact, find_cut_point,
    prepare_compaction, should_compact,
};
pub use error::{
    CompactionError, ExecutionError, ExecutionErrorCode, ExtensionError, FileError, FileErrorCode,
    PiError, ProviderError, Result, SessionError,
};
pub use event_stream::{
    AssistantMessageEventStream, EventStream, EventStreamError,
    create_assistant_message_event_stream,
};
pub use json_stream::{parse_json_with_repair, parse_streaming_json, repair_json};
pub use message_normalization::{
    ToolCallIdNormalizer, normalize_messages, normalize_messages_with_input,
};
pub use overflow::{is_context_overflow_error, is_context_overflow_message};
pub use session::{ContextMessage, Session, SessionEntry, SessionManager, SessionSummary};
pub use system_prompt::{
    build_system_prompt, format_tool_call_for_display, format_tool_result_for_display,
};
pub use token_estimation::{
    estimate_context_tokens, estimate_context_tokens_for_provider_model, estimate_tokens,
    estimate_tokens_for_provider_model,
};
pub use tool_validation::{validate_tool_arguments, validate_tool_call};
pub use traits::{Agent, ChatStream, Provider, StreamChunk, Tool, ToolCallDelta};
pub use types::{
    AgentConfig, AnthropicCompat, AssistantMessage, AssistantMessageEvent, CacheRetention,
    ChatResponse, CompatFlags, ContentBlock, FinishReason, FunctionCall, GoogleCompat, Message,
    MessageContent, Model, ModelCost, ModelId, ModelInputKind, OpenAiCompat, ProviderCapabilities,
    ProviderConfig, ProviderMetadata, ProviderRequestConfig, Role, StopReason, StreamOptions,
    ThinkingFormat, ThinkingLevel, ToolCall, ToolDefinition, Transport, Usage,
};
