//! AI provider abstraction for the pi coding agent.
//!
//! This crate implements the [`rpi_core::Provider`] trait for multiple LLM
//! backends and exposes a [`provider::create_provider`] factory for runtime
//! provider selection.
//!
//! # Supported providers
//!
//! | Provider | Identifier | Notes |
//! |----------|-----------|-------|
//! | OpenAI / Azure | `openai` | GPT-4o, GPT-4, etc. |
//! | Anthropic | `anthropic` | Claude 3.5 Sonnet, Claude 3 Opus, etc. |
//! | Ollama | `ollama` | Uses the OpenAI-compatible endpoint |
//!
//! # Streaming
//!
//! All providers support SSE (Server-Sent Events) streaming via the
//! [`rpi_core::Provider::chat_stream`] method. The [`streaming`] module
//! provides the low-level SSE parser.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
//! │  OpenAI API  │     │  Anthropic   │     │    Ollama    │
//! └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
//!        │                    │                    │
//!   ┌────▼────┐         ┌────▼────┐         ┌────▼────┐
//!   │ OpenAi  │         │Anthropic│         │ OpenAi  │
//!   │Provider │         │ Provider│         │Provider │
//!   └────┬────┘         └────┬────┘         └────┬────┘
//!        │                    │                    │
//!        └────────────┬───────┘────────────────────┘
//!                     │
//!              ┌──────▼──────┐
//!              │   Provider  │  (rpi_core trait)
//!              │    trait    │
//!              └─────────────┘
//! ```

#![warn(missing_docs)]

pub mod anthropic;
pub mod cloudflare;
pub mod openai;
pub mod provider;
pub mod streaming;

// Re-export the factory for convenience.
pub use provider::create_provider;

use std::time::Duration;

/// Default HTTP client with conservative timeouts.
pub(crate) fn default_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .timeout(Duration::from_secs(300))
        .build()
        // Safe: reqwest::Client::builder().build() only fails if the system TLS
        // backend or DNS resolver is broken — a fatal, unrecoverable condition.
        .expect("HTTP client builder")
}
