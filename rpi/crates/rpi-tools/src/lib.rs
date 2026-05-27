//! Built-in tool implementations for the pi coding agent.
//!
//! This crate provides the concrete implementations of tools that the agent
//! can invoke during execution — file I/O, shell commands, search, and more.
//!
//! Tools are registered via [`registry::ToolRegistry`] and dispatched by name
//! when the LLM requests a tool call.

#![warn(missing_docs)]

pub mod builtin;
pub mod registry;

pub use registry::ToolRegistry;
