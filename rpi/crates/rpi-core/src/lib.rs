//! Core types and traits for the pi coding agent.
//!
//! This crate provides the foundational types, error definitions, and trait
//! abstractions used across the pi agent ecosystem.

#![warn(missing_docs)]

pub mod agent_loop;
pub mod compaction;
mod error;
pub mod session;
pub mod system_prompt;
pub mod token_estimation;
mod traits;
mod types;

pub use agent_loop::{AgentEvent, AgentLoopConfig, apply_compaction, create_compaction_summary_message, run_agent_loop};
pub use compaction::{
    CompactionPreparation, CompactionResult, CompactionSettings, compact, find_cut_point,
    prepare_compaction, should_compact,
};
pub use error::{PiError, Result};
pub use session::{ContextMessage, Session, SessionEntry, SessionManager, SessionSummary};
pub use system_prompt::{
    build_system_prompt, format_tool_call_for_display, format_tool_result_for_display,
};
pub use token_estimation::{estimate_context_tokens, estimate_tokens};
pub use traits::{Agent, ChatStream, Provider, StreamChunk, Tool, ToolCallDelta};
pub use types::{
    AgentConfig, ChatResponse, ContentBlock, FinishReason, FunctionCall, Message, MessageContent,
    ModelId, ProviderConfig, Role, ToolCall, ToolDefinition, Usage,
};
