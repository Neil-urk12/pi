use rpi_tui::keybindings::KeybindingsManager;
use rpi_tui::keys::{Key, KeyEvent, Modifiers};
use std::collections::HashMap;

#[test]
fn default_bindings_match_expected_keys() {
    let mgr = KeybindingsManager::new();
    assert!(mgr.matches(b"\x1b[A", "tui.editor.cursorUp"));
    assert!(mgr.matches(b"\x01", "tui.editor.cursorLineStart"));
    assert!(mgr.matches(b"\r", "tui.input.submit"));
}

#[test]
fn user_override_wins_over_default() {
    let mut mgr = KeybindingsManager::new();
    let mut overrides = HashMap::new();
    overrides.insert(
        "tui.input.submit",
        vec![KeyEvent {
            key: Key::Char('s'),
            modifiers: Modifiers::CTRL,
        }],
    );
    mgr.set_user_bindings(overrides);

    assert!(mgr.matches(b"\x13", "tui.input.submit"));
    assert!(!mgr.matches(b"\r", "tui.input.submit"));
}

#[test]
fn conflict_detection() {
    let mut mgr = KeybindingsManager::new();
    let mut overrides = HashMap::new();
    overrides.insert(
        "tui.editor.cursorUp",
        vec![KeyEvent {
            key: Key::Char('k'),
            modifiers: Modifiers::CTRL,
        }],
    );
    overrides.insert(
        "tui.editor.cursorDown",
        vec![KeyEvent {
            key: Key::Char('k'),
            modifiers: Modifiers::CTRL,
        }],
    );
    mgr.set_user_bindings(overrides);

    let conflicts = mgr.get_conflicts();
    assert_eq!(conflicts.len(), 1);
    assert_eq!(conflicts[0].keybindings.len(), 2);
}
