//! Session management — JSONL-based persistent conversation storage.
//!
//! Sessions are tree-structured: each entry has an `id` and optional `parent_id`,
//! forming a linked list. A `Leaf` entry acts as a pointer to the current tip.
//! Compaction entries allow summarising old history while keeping recent messages.

use crate::error::Result;
use crate::types::{escape_xml_tags, Message, MessageContent, Role};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use uuid::Uuid;

/// A JSONL session entry in the tree-structured session model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionEntry {
    /// Session metadata (root entry).
    Meta {
        /// Unique entry ID.
        id: String,
        /// Parent entry ID (always `None` for the root).
        parent_id: Option<String>,
        /// Session identifier (equals `id` for the root meta).
        session_id: String,
        /// Creation timestamp.
        created_at: DateTime<Utc>,
        /// Model used for this session.
        model: String,
    },
    /// A conversation message.
    Message {
        /// Unique entry ID.
        id: String,
        /// Parent entry ID.
        parent_id: Option<String>,
        /// The message.
        message: Message,
    },
    /// A compaction boundary — summarises history before this point.
    Compaction {
        /// Unique entry ID.
        id: String,
        /// Parent entry ID.
        parent_id: Option<String>,
        /// Summary of the compacted history.
        summary: String,
        /// Entry ID where kept (non-compacted) messages resume.
        first_kept_entry_id: String,
        /// Token count before compaction.
        tokens_before: u32,
    },
    /// A leaf pointer — tracks the current tip of the conversation.
    Leaf {
        /// Unique entry ID.
        id: String,
        /// Parent leaf ID (tracks leaf history).
        parent_id: Option<String>,
        /// Entry ID this leaf points to.
        target_id: String,
    },
}

impl SessionEntry {
    /// Return the entry's unique ID.
    pub fn id(&self) -> &str {
        match self {
            Self::Meta { id, .. }
            | Self::Message { id, .. }
            | Self::Compaction { id, .. }
            | Self::Leaf { id, .. } => id,
        }
    }

    /// Return the parent entry ID, if any.
    pub fn parent_id(&self) -> Option<&str> {
        match self {
            Self::Meta { parent_id, .. }
            | Self::Message { parent_id, .. }
            | Self::Compaction { parent_id, .. }
            | Self::Leaf { parent_id, .. } => parent_id.as_deref(),
        }
    }
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

/// A loaded session with all its entries in tree order.
#[derive(Debug, Clone)]
pub struct Session {
    /// Unique session ID (matches the root Meta entry's `id`).
    pub session_id: String,
    /// When the session was created.
    pub created_at: DateTime<Utc>,
    /// Model used for this session.
    pub model: String,
    /// All entries in the session (meta, messages, compactions, leaves).
    pub entries: Vec<SessionEntry>,
    /// ID of the current leaf entry.
    pub leaf_id: String,
    /// Additional metadata.
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Session {
    /// Create a new session with a meta root and an initial leaf.
    pub fn new(model: impl Into<String>) -> Self {
        let meta_id = Uuid::new_v4().to_string();
        let leaf_id = Uuid::new_v4().to_string();
        let model_str = model.into();
        let now = Utc::now();

        let meta = SessionEntry::Meta {
            id: meta_id.clone(),
            parent_id: None,
            session_id: meta_id.clone(),
            created_at: now,
            model: model_str.clone(),
        };
        let leaf = SessionEntry::Leaf {
            id: leaf_id.clone(),
            parent_id: None,
            target_id: meta_id.clone(),
        };

        Self {
            session_id: meta_id,
            created_at: now,
            model: model_str,
            entries: vec![meta, leaf],
            leaf_id,
            metadata: HashMap::new(),
        }
    }

    /// Return the entry ID that the current leaf points to.
    fn current_entry_id(&self) -> String {
        self.entries
            .iter()
            .find_map(|e| match e {
                SessionEntry::Leaf {
                    id,
                    target_id,
                    ..
                } if *id == self.leaf_id => Some(target_id.clone()),
                _ => None,
            })
            .unwrap_or_else(|| self.session_id.clone())
    }

    /// Append a message to the session. Returns the new entry's ID.
    pub fn append_message(&mut self, message: Message) -> String {
        let entry_id = Uuid::new_v4().to_string();
        let old_leaf_target = self.current_entry_id();

        let msg_entry = SessionEntry::Message {
            id: entry_id.clone(),
            parent_id: Some(old_leaf_target),
            message,
        };

        let new_leaf_id = Uuid::new_v4().to_string();
        let new_leaf = SessionEntry::Leaf {
            id: new_leaf_id.clone(),
            parent_id: Some(self.leaf_id.clone()),
            target_id: entry_id.clone(),
        };

        self.entries.push(msg_entry);
        self.entries.push(new_leaf);
        self.leaf_id = new_leaf_id;

        entry_id
    }

    /// Append a compaction entry. Returns the new entry's ID.
    pub fn append_compaction(
        &mut self,
        summary: String,
        first_kept_entry_id: String,
        tokens_before: u32,
    ) -> String {
        let entry_id = Uuid::new_v4().to_string();
        let old_leaf_target = self.current_entry_id();

        let compaction = SessionEntry::Compaction {
            id: entry_id.clone(),
            parent_id: Some(old_leaf_target),
            summary,
            first_kept_entry_id,
            tokens_before,
        };

        let new_leaf_id = Uuid::new_v4().to_string();
        let new_leaf = SessionEntry::Leaf {
            id: new_leaf_id.clone(),
            parent_id: Some(self.leaf_id.clone()),
            target_id: entry_id.clone(),
        };

        self.entries.push(compaction);
        self.entries.push(new_leaf);
        self.leaf_id = new_leaf_id;

        entry_id
    }

    /// Build the conversation context by walking from the leaf to the root.
    ///
    /// When a compaction entry is encountered, a synthetic summary message is
    /// injected and the walk jumps to `first_kept_entry_id`.
    pub fn build_context(&self) -> Vec<Message> {
        let entry_map: HashMap<&str, &SessionEntry> =
            self.entries.iter().map(|e| (e.id(), e)).collect();

        // Collect the ordered tree path from root to leaf.
        let mut path_ids: Vec<String> = Vec::new();
        let mut current_id = self.current_entry_id();
        let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
        while let Some(entry) = entry_map.get(current_id.as_str()) {
            if !visited.insert(entry.id().to_string()) {
                break; // Cycle detected
            }
            path_ids.push(entry.id().to_string());
            match entry.parent_id() {
                Some(pid) => current_id = pid.to_string(),
                None => break,
            }
        }
        path_ids.reverse();

        // --- Pass 1: identify compacted entries and compaction boundaries ---
        let mut compacted: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        let mut summaries_before: HashMap<String, String> = HashMap::new();

        for entry in &self.entries {
            if let SessionEntry::Compaction {
                id: compact_id,
                summary,
                first_kept_entry_id, ..
            } = entry
            {
                if let Some(first_kept) = entry_map.get(first_kept_entry_id.as_str()) {
                    let mut walk_id = match first_kept.parent_id() {
                        Some(pid) => pid.to_string(),
                        None => {
                            summaries_before
                                .entry(first_kept_entry_id.clone())
                                .or_insert_with(|| summary.clone());
                            continue;
                        }
                    };
                    loop {
                        if walk_id == *compact_id
                            || !entry_map.contains_key(walk_id.as_str())
                        {
                            break;
                        }
                        compacted.insert(walk_id.clone());
                        walk_id = match entry_map.get(walk_id.as_str()) {
                            Some(e) => match e.parent_id() {
                                Some(pid) => pid.to_string(),
                                None => break,
                            },
                            None => break,
                        };
                    }
                }
                summaries_before
                    .entry(first_kept_entry_id.clone())
                    .or_insert_with(|| summary.clone());
            }
        }

        // --- Pass 2: build message list from tree path ---
        let mut result: Vec<Message> = Vec::new();

        for id in &path_ids {
            if let Some(summary) = summaries_before.remove(id) {
                result.push(Message {
                    role: Role::User,
                    content: Some(MessageContent::Text(format!(
                        "The conversation history before this point was compacted \
                         into the following summary:\n\n<summary>\n{}\n</summary>",
                        escape_xml_tags(&summary)
                    ))),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                });
            }

            if compacted.contains(id) {
                continue;
            }

            if let Some(entry) = entry_map.get(id.as_str())
                && let SessionEntry::Message { message, .. } = entry {
                    result.push(message.clone());
                }
        }

        result
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
        Ok(Self {
            sessions_dir: data_dir,
        })
    }

    /// Create a new session manager with a custom directory.
    pub fn with_dir(sessions_dir: PathBuf) -> Self {
        Self { sessions_dir }
    }

    /// Get the path for a session file.
    fn session_path(&self, session_id: &str) -> Result<PathBuf> {
        if session_id.is_empty()
            || session_id.starts_with('.')
            || !session_id.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '.')
        {
            return Err(crate::error::PiError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "Invalid session_id '{}': must be non-empty, alphanumeric with hyphens/dots, and cannot start with '.'",
                    session_id
                ),
            )));
        }
        Ok(self.sessions_dir.join(format!("{session_id}.jsonl")))
    }

    /// Save a session to disk.
    pub fn save(&self, session: &Session) -> Result<()> {
        fs::create_dir_all(&self.sessions_dir)?;

        let path = self.session_path(&session.session_id)?;
        let mut lines = Vec::new();

        for entry in &session.entries {
            lines.push(serde_json::to_string(entry)?);
        }

        fs::write(&path, lines.join("\n") + "\n")?;
        Ok(())
    }

    /// Load a session from disk.
    ///
    /// Handles both the new tree-structured format and the legacy flat format.
    pub fn load(&self, session_id: &str) -> Result<Session> {
        let path = self.session_path(session_id)?;
        let content = fs::read_to_string(&path)?;

        let mut entries = Vec::new();
        let mut needs_migration = false;

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // Try parsing as new format first (has "id" field).
            match serde_json::from_str::<SessionEntry>(line) {
                Ok(entry) => entries.push(entry),
                Err(_) => {
                    // Attempt legacy format migration.
                    let old: serde_json::Value = serde_json::from_str(line)?;
                    needs_migration = true;

                    let entry_type = old["type"].as_str().unwrap_or("");
                    match entry_type {
                        "meta" => {
                            entries.push(SessionEntry::Meta {
                                id: Uuid::new_v4().to_string(),
                                parent_id: None,
                                session_id: old["session_id"]
                                    .as_str()
                                    .unwrap_or(session_id)
                                    .to_string(),
                                created_at: serde_json::from_value(
                                    old["created_at"].clone(),
                                )?,
                                model: old["model"]
                                    .as_str()
                                    .unwrap_or("unknown")
                                    .to_string(),
                            });
                        }
                        "message" => {
                            let message: Message =
                                serde_json::from_value(old["message"].clone())?;
                            entries.push(SessionEntry::Message {
                                id: Uuid::new_v4().to_string(),
                                parent_id: None, // filled below
                                message,
                            });
                        }
                        _ => {
                            return Err(crate::error::PiError::Io(std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                format!("Unknown legacy entry type: {}", entry_type),
                            )));
                        }
                    }
                }
            }
        }

        if needs_migration {
            // Chain parent_ids for Message entries in order.
            let mut prev_msg_id: Option<String> = None;
            for entry in entries.iter_mut() {
                if let SessionEntry::Message { id, parent_id, .. } = entry {
                    *parent_id = prev_msg_id.clone();
                    prev_msg_id = Some(id.clone());
                }
            }

            // Find meta to use as the chain root for the first message.
            let meta_id = entries.iter().find_map(|e| match e {
                SessionEntry::Meta { id, .. } => Some(id.clone()),
                _ => None,
            });

            if let Some(first_msg) = entries.iter_mut().find_map(|e| match e {
                SessionEntry::Message { parent_id, .. } if parent_id.is_none() => Some(e),
                _ => None,
            })
                && let SessionEntry::Message { parent_id, .. } = first_msg {
                    *parent_id = meta_id;
                }

            // Add a leaf pointing to the last message (or meta if no messages).
            let last_target = prev_msg_id
                .or_else(|| {
                    entries.iter().find_map(|e| match e {
                        SessionEntry::Meta { id, .. } => Some(id.clone()),
                        _ => None,
                    })
                })
                .unwrap_or_else(|| session_id.to_string());

            let leaf_id = Uuid::new_v4().to_string();
            entries.push(SessionEntry::Leaf {
                id: leaf_id,
                parent_id: None,
                target_id: last_target,
            });
        }

        // Reconstruct Session from entries.
        let (session_id_resolved, created_at, model) = entries
            .iter()
            .find_map(|e| match e {
                SessionEntry::Meta {
                    id: _,
                    session_id,
                    created_at,
                    model,
                    ..
                } => Some((session_id.clone(), *created_at, model.clone())),
                _ => None,
            })
            .ok_or_else(|| {
                crate::error::PiError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Session file missing metadata",
                ))
            })?;

        let leaf_id = entries
            .iter()
            .rev()
            .find_map(|e| match e {
                SessionEntry::Leaf { id, .. } => Some(id.clone()),
                _ => None,
            })
            .unwrap_or_else(|| session_id_resolved.clone());

        Ok(Session {
            session_id: session_id_resolved,
            created_at,
            model,
            entries,
            leaf_id,
            metadata: HashMap::new(),
        })
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
            if path.extension().is_none_or(|e| e != "jsonl") {
                continue;
            }

            let session_id = path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();

            match self.load(&session_id) {
                Ok(session) => {
                    let context = session.build_context();
                    let last_message_preview = context.last().and_then(|m| {
                        let text = match &m.content {
                            Some(MessageContent::Text(t)) => t.clone(),
                            Some(MessageContent::Blocks(blocks)) => blocks
                                .iter()
                                .filter_map(|b| match b {
                                    crate::types::ContentBlock::Text { text } => {
                                        Some(text.as_str())
                                    }
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join(""),
                            None => String::new(),
                        };
                        if text.len() > 100 {
                            let end = text.char_indices().nth(100).map(|(i, _)| i).unwrap_or(text.len());
                            Some(format!("{}...", &text[..end]))
                        } else if text.is_empty() {
                            None
                        } else {
                            Some(text)
                        }
                    });

                    let message_count = session
                        .entries
                        .iter()
                        .filter(|e| matches!(e, SessionEntry::Message { .. }))
                        .count();

                    summaries.push(SessionSummary {
                        session_id: session.session_id,
                        created_at: session.created_at,
                        model: session.model,
                        message_count,
                        last_message_preview,
                    });
                }
                Err(_) => continue,
            }
        }

        // Sort by creation time, newest first.
        summaries.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(summaries)
    }

    /// Delete a session.
    pub fn delete(&self, session_id: &str) -> Result<()> {
        let path = self.session_path(session_id)?;
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

    fn user_msg(text: &str) -> Message {
        Message {
            role: Role::User,
            content: Some(MessageContent::Text(text.to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    fn assistant_msg(text: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: Some(MessageContent::Text(text.to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    #[test]
    fn test_session_new() {
        let session = Session::new("openai/gpt-4o");
        assert!(!session.session_id.is_empty());
        assert_eq!(session.model, "openai/gpt-4o");
        assert_eq!(session.entries.len(), 2); // meta + leaf
        assert!(!session.leaf_id.is_empty());

        // First entry is Meta, second is Leaf.
        assert!(matches!(session.entries[0], SessionEntry::Meta { .. }));
        assert!(matches!(session.entries[1], SessionEntry::Leaf { .. }));
    }

    #[test]
    fn test_session_entry_helpers() {
        let session = Session::new("test-model");
        let meta = &session.entries[0];
        assert_eq!(meta.id(), session.session_id);
        assert!(meta.parent_id().is_none());

        let leaf = &session.entries[1];
        assert_eq!(leaf.id(), session.leaf_id);
        assert!(leaf.parent_id().is_none());
        assert_eq!(leaf.parent_id(), None);
    }

    #[test]
    fn test_append_message() {
        let mut session = Session::new("openai/gpt-4o");
        let initial_len = session.entries.len(); // 2 (meta + leaf)

        session.append_message(user_msg("test"));
        // Should have: meta, old_leaf, message, new_leaf = 4
        assert_eq!(session.entries.len(), initial_len + 2);

        // The new message's parent_id should point to the meta (first real entry).
        let msg_entry = &session.entries[2];
        assert!(matches!(msg_entry, SessionEntry::Message { .. }));
        assert_eq!(
            msg_entry.parent_id().unwrap(),
            session.session_id.as_str()
        );

        // The new leaf's target should be the message.
        let new_leaf = &session.entries[3];
        assert!(matches!(new_leaf, SessionEntry::Leaf { .. }));
        if let SessionEntry::Leaf { target_id, .. } = new_leaf {
            assert_eq!(target_id, msg_entry.id());
        }
    }

    #[test]
    fn test_append_compaction() {
        let mut session = Session::new("openai/gpt-4o");
        session.append_message(user_msg("msg1"));
        let first_kept =
            session.append_message(assistant_msg("msg2"));
        session.append_compaction(
            "Summary of old messages".to_string(),
            first_kept.clone(),
            500,
        );

        let has_compaction = session
            .entries
            .iter()
            .any(|e| matches!(e, SessionEntry::Compaction { .. }));
        assert!(has_compaction);
    }

    #[test]
    fn test_build_context_simple() {
        let mut session = Session::new("openai/gpt-4o");
        session.append_message(user_msg("hello"));
        session.append_message(assistant_msg("hi"));

        let context = session.build_context();
        assert_eq!(context.len(), 2);
        assert_eq!(context[0].role, Role::User);
        assert_eq!(context[1].role, Role::Assistant);
    }

    #[test]
    fn test_build_context_with_compaction() {
        let mut session = Session::new("openai/gpt-4o");
        session.append_message(user_msg("old message 1"));
        session.append_message(assistant_msg("old reply 1"));
        let first_kept =
            session.append_message(user_msg("new message"));
        session.append_compaction(
            "Summary of old conversation".to_string(),
            first_kept,
            200,
        );
        session.append_message(assistant_msg("new reply"));

        let context = session.build_context();
        // Should contain: summary (synthetic) + "new message" + "new reply"
        assert_eq!(context.len(), 3);
        // First message is the synthetic summary from the compaction.
        assert_eq!(context[0].role, Role::User);
        let summary_text = match &context[0].content {
            Some(MessageContent::Text(t)) => t.clone(),
            _ => panic!("Expected text content"),
        };
        assert!(summary_text.contains("<summary>"));
        assert!(summary_text.contains("Summary of old conversation"));
        // Followed by the kept messages.
        assert_eq!(context[1].role, Role::User);
        assert_eq!(context[2].role, Role::Assistant);
    }

    #[test]
    fn test_session_save_load() {
        let dir = tempfile::tempdir().unwrap();
        let manager = SessionManager::with_dir(dir.path().to_path_buf());

        let mut session = Session::new("openai/gpt-4o");
        session.append_message(user_msg("Hello!"));
        session.append_message(assistant_msg("Hi there!"));

        manager.save(&session).unwrap();

        let loaded = manager.load(&session.session_id).unwrap();
        assert_eq!(loaded.session_id, session.session_id);
        assert_eq!(loaded.entries.len(), session.entries.len());
        assert_eq!(loaded.model, session.model);
    }

    #[test]
    fn test_session_list() {
        let dir = tempfile::tempdir().unwrap();
        let manager = SessionManager::with_dir(dir.path().to_path_buf());

        for i in 0..3 {
            let mut session = Session::new("openai/gpt-4o");
            session.append_message(user_msg(&format!("Message {i}")));
            manager.save(&session).unwrap();
        }

        let list = manager.list().unwrap();
        assert_eq!(list.len(),3);
        // Newest first.
        assert!(list[0].created_at >= list[1].created_at);
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

    #[test]
    fn test_session_list_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let manager = SessionManager::with_dir(dir.path().to_path_buf());
        let list = manager.list().unwrap();
        assert!(list.is_empty());
    }

    #[test]
    fn test_session_entry_parent_chain() {
        let mut session = Session::new("openai/gpt-4o");
        let msg1_id = session.append_message(user_msg("first"));
        let msg2_id = session.append_message(assistant_msg("second"));

        // msg2's parent should be msg1.
        let msg2 = session.entries.iter().find(|e| e.id() == msg2_id).unwrap();
        assert_eq!(msg2.parent_id().unwrap(), msg1_id.as_str());

        // msg1's parent should be meta (session_id).
        let msg1 = session.entries.iter().find(|e| e.id() == msg1_id).unwrap();
        assert_eq!(msg1.parent_id().unwrap(), session.session_id.as_str());
    }

    #[test]
    fn test_session_migration_empty_file() {
        // Test migration with empty session file
        let dir = tempfile::tempdir().unwrap();
        let manager = SessionManager::with_dir(dir.path().to_path_buf());
        let session_id = "empty-session";
        let path = dir.path().join(format!("{}.jsonl", session_id));
        std::fs::write(&path, "").unwrap();

        // Loading should fail because empty file has no meta entry
        let result = manager.load(session_id);
        assert!(result.is_err());
    }

    #[test]
    fn test_session_migration_only_meta() {
        // Test migration with only meta entry
        let dir = tempfile::tempdir().unwrap();
        let manager = SessionManager::with_dir(dir.path().to_path_buf());
        let session_id = "meta-only";
        let path = dir.path().join(format!("{}.jsonl", session_id));

        // Write legacy format with only meta
        let meta = serde_json::json!({
            "type": "meta",
            "session_id": session_id,
            "created_at": "2024-01-01T00:00:00Z",
            "model": "openai/gpt-4o"
        });
        std::fs::write(&path, format!("{}\n", meta)).unwrap();

        // Loading should succeed
        let result = manager.load(session_id);
        assert!(result.is_ok());
        let session = result.unwrap();
        assert_eq!(session.entries.len(), 2); // meta + leaf
        assert!(matches!(session.entries[0], SessionEntry::Meta { .. }));
        assert!(matches!(session.entries[1], SessionEntry::Leaf { .. }));
    }

    #[test]
    fn test_session_migration_unknown_entry_type() {
        // Test migration with unknown entry type
        let dir = tempfile::tempdir().unwrap();
        let manager = SessionManager::with_dir(dir.path().to_path_buf());
        let session_id = "unknown-type";
        let path = dir.path().join(format!("{}.jsonl", session_id));

        // Write legacy format with unknown type
        let entry = serde_json::json!({
            "type": "unknown_type",
            "data": "test"
        });
        std::fs::write(&path, format!("{}\n", entry)).unwrap();

        // Loading should fail with error
        let result = manager.load(session_id);
        assert!(result.is_err());
    }

    #[test]
    fn test_session_migration_malformed_json() {
        // Test migration with malformed JSON
        let dir = tempfile::tempdir().unwrap();
        let manager = SessionManager::with_dir(dir.path().to_path_buf());
        let session_id = "malformed";
        let path = dir.path().join(format!("{}.jsonl", session_id));

        // Write malformed JSON
        std::fs::write(&path, "not json").unwrap();

        // Loading should fail with error
        let result = manager.load(session_id);
        assert!(result.is_err());
    }

    #[test]
    fn test_session_migration_missing_fields() {
        // Test migration with missing required fields
        let dir = tempfile::tempdir().unwrap();
        let manager = SessionManager::with_dir(dir.path().to_path_buf());
        let session_id = "missing-fields";
        let path = dir.path().join(format!("{}.jsonl", session_id));

        // Write legacy format with missing fields
        let meta = serde_json::json!({
            "type": "meta"
            // Missing session_id, created_at, model
        });
        std::fs::write(&path, format!("{}\n", meta)).unwrap();

        // Loading should fail with error
        let result = manager.load(session_id);
        assert!(result.is_err());

    }
    // FINDING #8: Compaction summary format consistency
    #[test]
    fn test_compaction_summary_format_consistency() {
        let mut session = Session::new("test-model");
        session.append_message(user_msg("old message"));
        let first_kept = session.append_message(user_msg("new message"));
        session.append_compaction(
            "Test summary content".to_string(),
            first_kept,
            100,
        );

        let context = session.build_context();
        assert_eq!(context.len(), 2);

        let summary_msg = &context[0];
        assert_eq!(summary_msg.role, Role::User);
        let text = match &summary_msg.content {
            Some(MessageContent::Text(t)) => t.clone(),
            _ => panic!("Expected text content"),
        };
        assert!(text.contains("<summary>"), "Should contain <summary> tag");
        assert!(text.contains("</summary>"), "Should contain </summary> tag");
        assert!(text.contains("Test summary content"), "Should contain summary text");
    }

    // FINDING #9: CompactionEntry and LeafEntry serialization
    #[test]
    fn test_compaction_entry_serialization() {
        let entry = SessionEntry::Compaction {
            id: "test-id".to_string(),
            parent_id: Some("parent-id".to_string()),
            summary: "Test summary".to_string(),
            first_kept_entry_id: "kept-id".to_string(),
            tokens_before: 1000,
        };
        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("test-id"));
        let deserialized: SessionEntry = serde_json::from_str(&json).unwrap();
        match &deserialized {
            SessionEntry::Compaction { id, tokens_before, .. } => {
                assert_eq!(id, "test-id");
                assert_eq!(*tokens_before, 1000);
            }
            _ => panic!("Expected Compaction variant"),
        }
    }

    #[test]
    fn test_leaf_entry_serialization() {
        let entry = SessionEntry::Leaf {
            id: "leaf-id".to_string(),
            parent_id: Some("parent-leaf".to_string()),
            target_id: "target-id".to_string(),
        };
        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("leaf-id"));
        let deserialized: SessionEntry = serde_json::from_str(&json).unwrap();
        match &deserialized {
            SessionEntry::Leaf { target_id, .. } => {
                assert_eq!(target_id, "target-id");
            }
            _ => panic!("Expected Leaf variant"),
        }
    }

    #[test]
    fn test_session_path_rejects_dot_dot() {
        let dir = tempfile::tempdir().unwrap();
        let manager = SessionManager::with_dir(dir.path().to_path_buf());
        assert!(manager.load("..").is_err());
    }

    #[test]
    fn test_session_path_rejects_dot() {
        let dir = tempfile::tempdir().unwrap();
        let manager = SessionManager::with_dir(dir.path().to_path_buf());
        assert!(manager.load(".").is_err());
    }

    #[test]
    fn test_session_path_rejects_empty() {
        let dir = tempfile::tempdir().unwrap();
        let manager = SessionManager::with_dir(dir.path().to_path_buf());
        assert!(manager.load("").is_err());
    }

    #[test]
    fn test_session_path_rejects_hidden() {
        let dir = tempfile::tempdir().unwrap();
        let manager = SessionManager::with_dir(dir.path().to_path_buf());
        assert!(manager.load(".hidden").is_err());
    }

    // --- Finding: legacy migration doesn't create Leaf entry ---

    #[test]
    fn test_legacy_migration_creates_leaf_entry() {
        // Bug: loading a legacy (flat) session file doesn't create a Leaf entry.
        let dir = tempfile::tempdir().unwrap();
        let manager = SessionManager::with_dir(dir.path().to_path_buf());

        let legacy_content = r#"{"type":"meta","session_id":"legacy-123","created_at":"2024-01-01T00:00:00Z","model":"gpt-4"}
{"type":"message","message":{"role":"user","content":"hello","tool_calls":null,"tool_call_id":null,"name":null}}
{"type":"message","message":{"role":"assistant","content":"hi","tool_calls":null,"tool_call_id":null,"name":null}}"#;

        let session_path = dir.path().join("legacy-123.jsonl");
        std::fs::write(&session_path, legacy_content).unwrap();

        let session = manager.load("legacy-123").unwrap();

        let has_leaf = session.entries.iter().any(|e| matches!(e, SessionEntry::Leaf { .. }));
        assert!(
            has_leaf,
            "Legacy migration should create a Leaf entry. Found entries: {:?}",
            session.entries.iter().map(|e| e.id()).collect::<Vec<_>>()
        );

        let leaf_valid = session.entries.iter().any(|e| {
            matches!(e, SessionEntry::Leaf { id, .. } if *id == session.leaf_id)
        });
        assert!(leaf_valid, "leaf_id should point to an existing Leaf entry");
    }

    #[test]
    fn test_legacy_migration_preserves_message_content() {
        let dir = tempfile::tempdir().unwrap();
        let manager = SessionManager::with_dir(dir.path().to_path_buf());

        let legacy_content = r#"{"type":"meta","session_id":"content-test","created_at":"2024-01-01T00:00:00Z","model":"gpt-4"}
{"type":"message","message":{"role":"user","content":"first message","tool_calls":null,"tool_call_id":null,"name":null}}
{"type":"message","message":{"role":"assistant","content":"second message","tool_calls":null,"tool_call_id":null,"name":null}}"#;

        let session_path = dir.path().join("content-test.jsonl");
        std::fs::write(&session_path, legacy_content).unwrap();

        let session = manager.load("content-test").unwrap();
        let context = session.build_context();

        assert_eq!(context.len(), 2);
        match &context[0].content {
            Some(MessageContent::Text(t)) => assert_eq!(t, "first message"),
            _ => panic!("Expected text"),
        }
        match &context[1].content {
            Some(MessageContent::Text(t)) => assert_eq!(t, "second message"),
            _ => panic!("Expected text"),
        }
    }

    #[test]
    fn test_legacy_migration_session_id_with_underscores() {
        let dir = tempfile::tempdir().unwrap();
        let manager = SessionManager::with_dir(dir.path().to_path_buf());
        assert!(manager.load("test_session").is_err());
    }

    #[test]
    fn test_legacy_migration_session_id_with_special_chars() {
        let dir = tempfile::tempdir().unwrap();
        let manager = SessionManager::with_dir(dir.path().to_path_buf());
        assert!(manager.load("../etc/passwd").is_err());
        assert!(manager.load("test@session").is_err());
        assert!(manager.load("test session").is_err());
    }
}
