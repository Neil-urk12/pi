//! Core types and traits for the pi coding agent.
//!
//! This crate provides the foundational types, error definitions, and trait
//! abstractions used across the pi agent ecosystem.

#![warn(missing_docs)]

mod error;
mod traits;
mod types;
pub mod agent_loop;
pub mod session;
pub mod compaction;
pub mod token_estimation;
pub mod system_prompt;

pub use error::{PiError, Result};
pub use traits::{Agent, ChatStream, Provider, StreamChunk, Tool, ToolCallDelta};
pub use types::{
    AgentConfig, ChatResponse, ContentBlock, FinishReason, FunctionCall, Message, MessageContent,
    ModelId, ProviderConfig, Role, ToolCall, ToolDefinition, Usage,
};
pub use agent_loop::{run_agent_loop, AgentEvent, AgentLoopConfig};
pub use session::{Session, SessionEntry, SessionManager, SessionSummary};
pub use system_prompt::{build_system_prompt, format_tool_call_for_display, format_tool_result_for_display};
pub use token_estimation::{estimate_tokens, estimate_context_tokens};
pub use compaction::{compact, find_cut_point, prepare_compaction, should_compact, CompactionPreparation, CompactionResult, CompactionSettings};
