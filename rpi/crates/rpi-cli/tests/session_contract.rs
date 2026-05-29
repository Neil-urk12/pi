use rpi_core::{
    apply_compaction, create_compaction_summary_message, CompactionResult, FunctionCall, Message,
    MessageContent, Role, Session, SessionManager, ToolCall,
};

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

/// Extract text from `Option<MessageContent>` for comparison.
fn content_text(msg: &Message) -> Option<&str> {
    match &msg.content {
        Some(MessageContent::Text(t)) => Some(t.as_str()),
        _ => None,
    }
}

/// Exercises the PRODUCTION `apply_compaction` code path (index arithmetic)
/// and verifies the result can be correctly persisted to and restored from a
/// `Session`. If the index arithmetic in `apply_compaction` diverges from the
/// UUID-based persistence in `Session::build_context`, this test catches it.
#[test]
fn compaction_persistence_matches_apply_compaction() {
    // 1. Build a conversation: 3 turns (user-assistant pairs).
    let messages = vec![
        user_msg("old question 1"),
        assistant_msg("old answer 1"),
        user_msg("old question 2"),
        assistant_msg("old answer 2"),
        user_msg("new question"),
        assistant_msg("new answer"),
    ];

    // 2. Run the PRODUCTION compaction code: keep only the last 2 messages.
    let mut prod_messages = messages.clone();
    let result = CompactionResult {
        summary: "User asked two old questions, got answers.".to_string(),
        first_kept_message_index: 4,
        tokens_before: 100,
        tokens_after: 50,
    };
    apply_compaction(&mut prod_messages, &result);

    // prod_messages should now be: [summary_msg, new_question, new_answer]
    assert_eq!(
        prod_messages.len(),
        3,
        "apply_compaction should keep summary + 2 kept messages"
    );
    assert_eq!(prod_messages[0].role, Role::User); // summary is injected as User
    assert!(
        content_text(&prod_messages[0])
            .is_some_and(|t| t.contains("<summary>")),
        "first message should be the summary"
    );
    assert_eq!(content_text(&prod_messages[1]), Some("new question"));
    assert_eq!(content_text(&prod_messages[2]), Some("new answer"));

    // 3. Create a Session with the same messages and persist the compaction.
    let mut session = Session::new("test-model");
    let entry_ids: Vec<String> = messages
        .iter()
        .map(|m| session.append_message(m.clone()).to_string())
        .collect();

    // Map first_kept_message_index -> first_kept_entry_id.
    // This is the bridge the caller must get right.
    let first_kept_entry_id = entry_ids[result.first_kept_message_index].clone();
    session.append_compaction(
        result.summary.clone(),
        first_kept_entry_id,
        result.tokens_before,
    );

    // 4. Verify Session::build_context produces the same messages as
    //    apply_compaction.
    let session_context = session.build_context();
    assert_eq!(
        session_context.len(),
        prod_messages.len(),
        "Session build_context length should match apply_compaction result"
    );

    for (i, (expected, actual)) in
        prod_messages.iter().zip(session_context.iter()).enumerate()
    {
        assert_eq!(
            expected.role, actual.role,
            "Role mismatch at index {i}: expected {:?}, got {:?}",
            expected.role, actual.role
        );
        assert_eq!(
            content_text(expected),
            content_text(actual),
            "Content mismatch at index {i}"
        );
    }
}

/// Verifies that the summary message format is identical between
/// `apply_compaction` and `Session::build_context` -- they must call
/// the same shared function.
#[test]
fn compaction_summary_format_is_shared() {
    let summary_text = "Summary with <special> & \"chars\"";

    // Production path: create_compaction_summary_message (used by apply_compaction)
    let from_helper = create_compaction_summary_message(summary_text);

    // Session path: build_context with a compaction entry
    let mut session = Session::new("test-model");
    let entry_id = session.append_message(user_msg("msg"));
    session.append_compaction(summary_text.to_string(), entry_id.to_string(), 50);
    let context = session.build_context();

    // first_kept_entry_id = "keep FROM here", so the original message is
    // still present after the summary.
    assert_eq!(context.len(), 2);
    assert_eq!(context[0].role, from_helper.role);
    assert_eq!(content_text(&context[0]), content_text(&from_helper));
    assert_eq!(context[1].role, Role::User);
    assert_eq!(content_text(&context[1]), Some("msg"));
}

/// Verifies that apply_compaction correctly handles a split-turn scenario
/// where the first_kept_message_index lands on a tool result.
#[test]
fn compaction_with_split_turn_preserves_tool_results() {
    let messages = vec![
        user_msg("do something"),
        Message {
            role: Role::Assistant,
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_1".to_string(),
                function: FunctionCall {
                    name: "bash".to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
            tool_call_id: None,
            name: None,
        },
        Message {
            role: Role::Tool,
            content: Some(MessageContent::Text("tool output".to_string())),
            tool_calls: None,
            tool_call_id: Some("call_1".to_string()),
            name: None,
        },
        assistant_msg("final answer"),
    ];

    // Keep from the tool result onwards (split turn)
    let mut prod_messages = messages.clone();
    let result = CompactionResult {
        summary: "User asked to do something, tool ran.".to_string(),
        first_kept_message_index: 2,
        tokens_before: 80,
        tokens_after: 40,
    };
    apply_compaction(&mut prod_messages, &result);

    // Should be: [summary, tool_result, assistant_final]
    assert_eq!(prod_messages.len(), 3);
    assert_eq!(prod_messages[1].role, Role::Tool);
    assert_eq!(prod_messages[2].role, Role::Assistant);

    // Persist to session and verify
    let mut session = Session::new("test-model");
    let entry_ids: Vec<String> = messages
        .iter()
        .map(|m| session.append_message(m.clone()).to_string())
        .collect();

    let first_kept_entry_id = entry_ids[result.first_kept_message_index].clone();
    session.append_compaction(
        result.summary.clone(),
        first_kept_entry_id,
        result.tokens_before,
    );

    let session_context = session.build_context();
    assert_eq!(session_context.len(), prod_messages.len());
    for (i, (expected, actual)) in
        prod_messages.iter().zip(session_context.iter()).enumerate()
    {
        assert_eq!(expected.role, actual.role, "Role mismatch at index {i}");
        assert_eq!(
            content_text(expected),
            content_text(actual),
            "Content mismatch at index {i}"
        );
    }
}
// ── Task 4.3: Contract tests ────────────────────────────────────────────────

#[test]
fn new_session_persists_user_messages() {
    let mut session = Session::new("test-model".to_string());
    session.append_message(user_msg("hello"));

    let context = session.build_context();
    assert_eq!(context.len(), 1);
    assert_eq!(context[0].role, Role::User);
}

#[test]
fn session_persists_multiple_turns() {
    let mut session = Session::new("test-model".to_string());

    session.append_message(user_msg("hello"));
    session.append_message(assistant_msg("hi there"));
    session.append_message(user_msg("how are you"));

    let context = session.build_context();
    assert_eq!(context.len(), 3);
    assert_eq!(context[0].role, Role::User);
    assert_eq!(context[1].role, Role::Assistant);
    assert_eq!(context[2].role, Role::User);
}

#[test]
fn session_save_load_roundtrip() {
    let tmp = tempfile::tempdir().unwrap();
    let manager = SessionManager::with_dir(tmp.path().to_path_buf());

    let mut session = Session::new("test-model".to_string());
    session.append_message(user_msg("test"));
    manager.save(&session).unwrap();

    let loaded = manager.load(&session.session_id).unwrap();
    let context = loaded.build_context();
    assert_eq!(context.len(), 1);
    assert_eq!(context[0].role, Role::User);
}

#[test]
fn context_with_entry_ids_maps_correctly() {
    let mut session = Session::new("test-model".to_string());

    session.append_message(user_msg("hello"));
    session.append_message(assistant_msg("world"));

    let context = session.build_context_with_entry_ids();
    assert_eq!(context.len(), 2);
    // All non-synthetic messages should have entry IDs.
    assert!(context[0].entry_id.is_some());
    assert!(context[1].entry_id.is_some());
    // Entry IDs should be different.
    assert_ne!(context[0].entry_id, context[1].entry_id);
}

// ── Task 5.1: CLI session operation tests ───────────────────────────────────

#[test]
fn session_list_shows_sessions() {
    let tmp = tempfile::tempdir().unwrap();
    let manager = SessionManager::with_dir(tmp.path().to_path_buf());

    let mut session = Session::new("test-model".to_string());
    session.append_message(user_msg("hello"));
    manager.save(&session).unwrap();

    let sessions = manager.list().unwrap();
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].model, "test-model");
    assert_eq!(sessions[0].message_count, 1);
}

#[test]
fn session_delete_removes_session() {
    let tmp = tempfile::tempdir().unwrap();
    let manager = SessionManager::with_dir(tmp.path().to_path_buf());

    let mut session = Session::new("test-model".to_string());
    session.append_message(user_msg("hello"));
    let session_id = session.session_id.clone();
    manager.save(&session).unwrap();

    assert_eq!(manager.list().unwrap().len(), 1);
    manager.delete(&session_id).unwrap();
    assert_eq!(manager.list().unwrap().len(), 0);
}

#[test]
fn load_missing_session_fails() {
    let tmp = tempfile::tempdir().unwrap();
    let manager = SessionManager::with_dir(tmp.path().to_path_buf());

    let result = manager.load("nonexistent-id");
    assert!(result.is_err());
}

// ── Task 5.2: Integration tests for resume context after compaction ────────

#[test]
fn resume_after_compaction_reconstructs_context() {
    let mut session = Session::new("test-model".to_string());

    // Add 5 messages
    for i in 0..5 {
        session.append_message(Message {
            role: if i % 2 == 0 {
                Role::User
            } else {
                Role::Assistant
            },
            content: Some(MessageContent::Text(format!("message {i}"))),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        });
    }

    // Get the entry ID of message 3 (the first kept message after compaction at index 3)
    let context = session.build_context_with_entry_ids();
    let first_kept_entry_id = context[3].entry_id.clone().unwrap();

    // Append compaction entry (summarizes messages 0-2, keeps 3-4)
    session.append_compaction(
        "Summary of messages 0-2".to_string(),
        first_kept_entry_id,
        1000,
    );

    // Add a new message after compaction
    session.append_message(Message {
        role: Role::User,
        content: Some(MessageContent::Text("post-compaction message".to_string())),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    });

    // Rebuild context
    let context = session.build_context();

    // Should have exactly one summary message (User role with "<summary>" wrapper)
    let summary_count = context
        .iter()
        .filter(|m| {
            m.content.as_ref().map_or(
                false,
                |c| matches!(c, MessageContent::Text(t) if t.contains("Summary of messages 0-2")),
            )
        })
        .count();
    assert_eq!(summary_count, 1, "Should have exactly one summary message");

    // Should NOT contain compacted messages 0, 1, 2
    for i in 0..3 {
        let found = context.iter().any(|m| {
            m.content.as_ref().map_or(
                false,
                |c| matches!(c, MessageContent::Text(t) if t == &format!("message {i}")),
            )
        });
        assert!(!found, "Compacted message {i} should not be in context");
    }

    // Should contain kept messages and the new message
    assert!(context.iter().any(|m| {
        m.content.as_ref().map_or(
            false,
            |c| matches!(c, MessageContent::Text(t) if t == "message 3"),
        )
    }));
    assert!(context.iter().any(|m| {
        m.content.as_ref().map_or(
            false,
            |c| matches!(c, MessageContent::Text(t) if t == "message 4"),
        )
    }));
    assert!(context.iter().any(|m| {
        m.content.as_ref().map_or(
            false,
            |c| matches!(c, MessageContent::Text(t) if t == "post-compaction message"),
        )
    }));
}

#[test]
fn save_load_session_with_compaction() {
    let tmp = tempfile::tempdir().unwrap();
    let manager = SessionManager::with_dir(tmp.path().to_path_buf());

    let mut session = Session::new("test-model".to_string());

    // Add messages
    for i in 0..4 {
        session.append_message(Message {
            role: if i % 2 == 0 {
                Role::User
            } else {
                Role::Assistant
            },
            content: Some(MessageContent::Text(format!("msg {i}"))),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        });
    }

    // Compact at index 2
    let context = session.build_context_with_entry_ids();
    let first_kept = context[2].entry_id.clone().unwrap();
    session.append_compaction("Summary".to_string(), first_kept, 500);

    // Save and reload
    manager.save(&session).unwrap();
    let loaded = manager.load(&session.session_id).unwrap();

    // Context should match
    let original_context = session.build_context();
    let loaded_context = loaded.build_context();

    assert_eq!(original_context.len(), loaded_context.len());
    for (orig, load) in original_context.iter().zip(loaded_context.iter()) {
        assert_eq!(orig.role, load.role);
    }
}

#[test]
fn entry_ids_after_compaction_map_correctly() {
    let mut session = Session::new("test-model".to_string());

    for i in 0..4 {
        session.append_message(Message {
            role: if i % 2 == 0 {
                Role::User
            } else {
                Role::Assistant
            },
            content: Some(MessageContent::Text(format!("msg {i}"))),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        });
    }

    let context_pre = session.build_context_with_entry_ids();
    let first_kept = context_pre[2].entry_id.clone().unwrap();
    session.append_compaction("Summary".to_string(), first_kept, 500);

    let context_post = session.build_context_with_entry_ids();

    // First entry is the summary (synthetic) — should be None
    assert!(
        context_post[0].entry_id.is_none(),
        "Summary should map to None"
    );

    // Remaining entries are kept messages — should be Some
    for cm in &context_post[1..] {
        assert!(cm.entry_id.is_some(), "Kept messages should have entry IDs");
    }
}

// ── Finding 4: --resume --model handling ────────────────────────────────────

#[test]
fn resume_preserves_original_model() {
    // Finding 7: `rpi --resume <id> --model other-model` succeeds but session
    // metadata stays at old model. This test documents the current behavior:
    // loading a session for resume preserves the original model field.
    let tmp = tempfile::tempdir().unwrap();
    let manager = SessionManager::with_dir(tmp.path().to_path_buf());

    let mut session = Session::new("original-model".to_string());
    session.append_message(user_msg("hello"));
    manager.save(&session).unwrap();

    let loaded = manager.load(&session.session_id).unwrap();

    assert_eq!(
        loaded.model, "original-model",
        "Session should preserve its original model on resume load"
    );
}

// ── Finding 4: session delete must be idempotent ────────────────────────────

#[test]
fn delete_nonexistent_session_is_idempotent() {
    // Finding 4 fix: SessionManager::delete is idempotent — Ok(()) when file missing.
    let tmp = tempfile::tempdir().unwrap();
    let manager = SessionManager::with_dir(tmp.path().to_path_buf());

    let result = manager.delete("nonexistent-session-id");

    assert!(
        result.is_ok(),
        "Deleting non-existent session should return Ok(()) (idempotent), but got {result:?}"
    );
}

#[test]
fn delete_existing_session_succeeds() {
    let tmp = tempfile::tempdir().unwrap();
    let manager = SessionManager::with_dir(tmp.path().to_path_buf());

    let mut session = Session::new("test-model".to_string());
    session.append_message(user_msg("hello"));
    let session_id = session.session_id.clone();
    manager.save(&session).unwrap();

    assert_eq!(manager.list().unwrap().len(), 1);
    manager.delete(&session_id).unwrap();
    assert_eq!(manager.list().unwrap().len(), 0);
}
