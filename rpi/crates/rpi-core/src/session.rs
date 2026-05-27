//! Session management — JSONL-based persistent conversation storage.

use crate::error::Result;
use crate::types::Message;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// A JSONL session entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionEntry {
    /// Session metadata (first line).
    Meta {
        /// Unique session ID.
        session_id: String,
        /// Creation timestamp.
        created_at: DateTime<Utc>,
        /// Model used for this session.
        model: String,
    },
    /// A conversation message.
    Message {
        /// The message.
        message: Message,
    },
}

/// Summary of a session for listing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    /// Session ID.
    pub session_id: String,
    /// When the session was created.
    pub created_at: DateTime<Utc>,
    /// Model used.
    pub model: String,
    /// Number of messages in the session.
    pub message_count: usize,
    /// Preview of the last message.
    pub last_message_preview: Option<String>,
}

/// A loaded session with all its messages.
#[derive(Debug, Clone)]
pub struct Session {
    /// Unique session ID.
    pub session_id: String,
    /// When the session was created.
    pub created_at: DateTime<Utc>,
    /// Model used for this session.
    pub model: String,
    /// All messages in the session.
    pub messages: Vec<Message>,
    /// Additional metadata.
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Session {
    /// Create a new session.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            session_id: Uuid::new_v4().to_string(),
            created_at: Utc::now(),
            model: model.into(),
            messages: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a message to the session.
    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);
    }
}

/// Manages session persistence.
pub struct SessionManager {
    /// Directory where sessions are stored.
    sessions_dir: PathBuf,
}

impl SessionManager {
    /// Create a new session manager with the default directory.
    pub fn new() -> Result<Self> {
        let data_dir = dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("rpi")
            .join("sessions");
        Ok(Self { sessions_dir: data_dir })
    }

    /// Create a new session manager with a custom directory.
    pub fn with_dir(sessions_dir: PathBuf) -> Self {
        Self { sessions_dir }
    }

    /// Get the path for a session file.
    fn session_path(&self, session_id: &str) -> PathBuf {
        self.sessions_dir.join(format!("{session_id}.jsonl"))
    }

    /// Save a session to disk.
    pub fn save(&self, session: &Session) -> Result<()> {
        fs::create_dir_all(&self.sessions_dir)?;

        let path = self.session_path(&session.session_id);
        let mut lines = Vec::new();

        // Write metadata
        let meta = SessionEntry::Meta {
            session_id: session.session_id.clone(),
            created_at: session.created_at,
            model: session.model.clone(),
        };
        lines.push(serde_json::to_string(&meta)?);

        // Write messages
        for message in &session.messages {
            let entry = SessionEntry::Message {
                message: message.clone(),
            };
            lines.push(serde_json::to_string(&entry)?);
        }

        fs::write(&path, lines.join("\n") + "\n")?;
        Ok(())
    }

    /// Load a session from disk.
    pub fn load(&self, session_id: &str) -> Result<Session> {
        let path = self.session_path(session_id);
        let content = fs::read_to_string(&path)?;
        
        let mut session = None;
        let mut messages = Vec::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let entry: SessionEntry = serde_json::from_str(line)?;
            match entry {
                SessionEntry::Meta {
                    session_id,
                    created_at,
                    model,
                } => {
                    session = Some(Session {
                        session_id,
                        created_at,
                        model,
                        messages: Vec::new(),
                        metadata: HashMap::new(),
                    });
                }
                SessionEntry::Message { message } => {
                    messages.push(message);
                }
            }
        }

        let mut session = session.ok_or_else(|| {
            crate::error::PiError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Session file missing metadata",
            ))
        })?;
        session.messages = messages;
        Ok(session)
    }

    /// List all sessions.
    pub fn list(&self) -> Result<Vec<SessionSummary>> {
        if !self.sessions_dir.exists() {
            return Ok(Vec::new());
        }

        let mut summaries = Vec::new();
        for entry in fs::read_dir(&self.sessions_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map_or(true, |e| e != "jsonl") {
                continue;
            }

            let session_id = path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();

            match self.load(&session_id) {
                Ok(session) => {
                    let last_message_preview = session.messages.last().and_then(|m| {
                        let text = match &m.content {
                            Some(crate::types::MessageContent::Text(t)) => t.clone(),
                            Some(crate::types::MessageContent::Blocks(blocks)) => blocks
                                .iter()
                                .filter_map(|b| match b {
                                    crate::types::ContentBlock::Text { text } => Some(text.as_str()),
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join(""),
                            None => String::new(),
                        };
                        if text.len() > 100 {
                            Some(format!("{}...", &text[..100]))
                        } else if text.is_empty() {
                            None
                        } else {
                            Some(text)
                        }
                    });

                    summaries.push(SessionSummary {
                        session_id: session.session_id,
                        created_at: session.created_at,
                        model: session.model,
                        message_count: session.messages.len(),
                        last_message_preview,
                    });
                }
                Err(_) => continue,
            }
        }

        // Sort by creation time, newest first
        summaries.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(summaries)
    }

    /// Delete a session.
    pub fn delete(&self, session_id: &str) -> Result<()> {
        let path = self.session_path(session_id);
        if path.exists() {
            fs::remove_file(&path)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{MessageContent, Role};

    #[test]
    fn test_session_new() {
        let session = Session::new("openai/gpt-4o");
        assert!(!session.session_id.is_empty());
        assert_eq!(session.model, "openai/gpt-4o");
        assert!(session.messages.is_empty());
    }

    #[test]
    fn test_session_save_load() {
        let dir = tempfile::tempdir().unwrap();
        let manager = SessionManager::with_dir(dir.path().to_path_buf());

        let mut session = Session::new("openai/gpt-4o");
        session.add_message(Message {
            role: Role::User,
            content: Some(MessageContent::Text("Hello!".to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        });
        session.add_message(Message {
            role: Role::Assistant,
            content: Some(MessageContent::Text("Hi there!".to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        });

        manager.save(&session).unwrap();

        let loaded = manager.load(&session.session_id).unwrap();
        assert_eq!(loaded.session_id, session.session_id);
        assert_eq!(loaded.messages.len(), 2);
    }

    #[test]
    fn test_session_list() {
        let dir = tempfile::tempdir().unwrap();
        let manager = SessionManager::with_dir(dir.path().to_path_buf());

        // Create and save multiple sessions
        for i in 0..3 {
            let mut session = Session::new("openai/gpt-4o");
            session.add_message(Message {
                role: Role::User,
                content: Some(MessageContent::Text(format!("Message {i}"))),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
            manager.save(&session).unwrap();
        }

        let list = manager.list().unwrap();
        assert_eq!(list.len(), 3);
    }

    #[test]
    fn test_session_delete() {
        let dir = tempfile::tempdir().unwrap();
        let manager = SessionManager::with_dir(dir.path().to_path_buf());

        let session = Session::new("openai/gpt-4o");
        let id = session.session_id.clone();
        manager.save(&session).unwrap();
        assert!(manager.load(&id).is_ok());

        manager.delete(&id).unwrap();
        assert!(manager.load(&id).is_err());
    }
}
