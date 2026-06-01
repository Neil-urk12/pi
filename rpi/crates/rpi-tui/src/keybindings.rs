use crate::keys::{self, KeyEvent, Key, Modifiers, SpecialKey};
use std::collections::HashMap;

/// A keybinding identifier (e.g., "tui.editor.cursorUp").
pub type KeybindingId = &'static str;

/// Default keybinding definitions for the TUI.
pub const TUI_KEYBINDINGS: &[(KeybindingId, &[KeyEvent])] = &[
    ("tui.editor.cursorUp", &[KeyEvent { key: Key::Special(SpecialKey::Up), modifiers: Modifiers::empty() }]),
    ("tui.editor.cursorDown", &[KeyEvent { key: Key::Special(SpecialKey::Down), modifiers: Modifiers::empty() }]),
    ("tui.editor.cursorLeft", &[
        KeyEvent { key: Key::Special(SpecialKey::Left), modifiers: Modifiers::empty() },
        KeyEvent { key: Key::Char('b'), modifiers: Modifiers::CTRL },
    ]),
    ("tui.editor.cursorRight", &[
        KeyEvent { key: Key::Special(SpecialKey::Right), modifiers: Modifiers::empty() },
        KeyEvent { key: Key::Char('f'), modifiers: Modifiers::CTRL },
    ]),
    ("tui.editor.cursorWordLeft", &[
        KeyEvent { key: Key::Special(SpecialKey::Left), modifiers: Modifiers::ALT },
        KeyEvent { key: Key::Special(SpecialKey::Left), modifiers: Modifiers::CTRL },
        KeyEvent { key: Key::Char('b'), modifiers: Modifiers::ALT },
    ]),
    ("tui.editor.cursorWordRight", &[
        KeyEvent { key: Key::Special(SpecialKey::Right), modifiers: Modifiers::ALT },
        KeyEvent { key: Key::Special(SpecialKey::Right), modifiers: Modifiers::CTRL },
        KeyEvent { key: Key::Char('f'), modifiers: Modifiers::ALT },
    ]),
    ("tui.editor.cursorLineStart", &[
        KeyEvent { key: Key::Special(SpecialKey::Home), modifiers: Modifiers::empty() },
        KeyEvent { key: Key::Char('a'), modifiers: Modifiers::CTRL },
    ]),
    ("tui.editor.cursorLineEnd", &[
        KeyEvent { key: Key::Special(SpecialKey::End), modifiers: Modifiers::empty() },
        KeyEvent { key: Key::Char('e'), modifiers: Modifiers::CTRL },
    ]),
    ("tui.editor.jumpForward", &[KeyEvent { key: Key::Char(']'), modifiers: Modifiers::CTRL }]),
    ("tui.editor.jumpBackward", &[KeyEvent { key: Key::Char(']'), modifiers: Modifiers::from_bits_truncate(Modifiers::CTRL.bits() | Modifiers::ALT.bits()) }]),
    ("tui.editor.pageUp", &[KeyEvent { key: Key::Special(SpecialKey::PageUp), modifiers: Modifiers::empty() }]),
    ("tui.editor.pageDown", &[KeyEvent { key: Key::Special(SpecialKey::PageDown), modifiers: Modifiers::empty() }]),
    ("tui.editor.deleteCharBackward", &[KeyEvent { key: Key::Special(SpecialKey::Backspace), modifiers: Modifiers::empty() }]),
    ("tui.editor.deleteCharForward", &[
        KeyEvent { key: Key::Special(SpecialKey::Delete), modifiers: Modifiers::empty() },
        KeyEvent { key: Key::Char('d'), modifiers: Modifiers::CTRL },
    ]),
    ("tui.editor.deleteWordBackward", &[
        KeyEvent { key: Key::Char('w'), modifiers: Modifiers::CTRL },
        KeyEvent { key: Key::Special(SpecialKey::Backspace), modifiers: Modifiers::ALT },
    ]),
    ("tui.editor.deleteWordForward", &[
        KeyEvent { key: Key::Char('d'), modifiers: Modifiers::ALT },
        KeyEvent { key: Key::Special(SpecialKey::Delete), modifiers: Modifiers::ALT },
    ]),
    ("tui.editor.deleteToLineStart", &[KeyEvent { key: Key::Char('u'), modifiers: Modifiers::CTRL }]),
    ("tui.editor.deleteToLineEnd", &[KeyEvent { key: Key::Char('k'), modifiers: Modifiers::CTRL }]),
    ("tui.editor.yank", &[KeyEvent { key: Key::Char('y'), modifiers: Modifiers::CTRL }]),
    ("tui.editor.yankPop", &[KeyEvent { key: Key::Char('y'), modifiers: Modifiers::ALT }]),
    ("tui.editor.undo", &[KeyEvent { key: Key::Char('-'), modifiers: Modifiers::CTRL }]),
    ("tui.input.newLine", &[KeyEvent { key: Key::Special(SpecialKey::Enter), modifiers: Modifiers::SHIFT }]),
    ("tui.input.submit", &[KeyEvent { key: Key::Special(SpecialKey::Enter), modifiers: Modifiers::empty() }]),
    ("tui.input.tab", &[KeyEvent { key: Key::Special(SpecialKey::Tab), modifiers: Modifiers::empty() }]),
    ("tui.input.copy", &[KeyEvent { key: Key::Char('c'), modifiers: Modifiers::CTRL }]),
    ("tui.select.up", &[KeyEvent { key: Key::Special(SpecialKey::Up), modifiers: Modifiers::empty() }]),
    ("tui.select.down", &[KeyEvent { key: Key::Special(SpecialKey::Down), modifiers: Modifiers::empty() }]),
    ("tui.select.pageUp", &[KeyEvent { key: Key::Special(SpecialKey::PageUp), modifiers: Modifiers::empty() }]),
    ("tui.select.pageDown", &[KeyEvent { key: Key::Special(SpecialKey::PageDown), modifiers: Modifiers::empty() }]),
    ("tui.select.confirm", &[KeyEvent { key: Key::Special(SpecialKey::Enter), modifiers: Modifiers::empty() }]),
    ("tui.select.cancel", &[
        KeyEvent { key: Key::Special(SpecialKey::Escape), modifiers: Modifiers::empty() },
        KeyEvent { key: Key::Char('c'), modifiers: Modifiers::CTRL },
    ]),
];

/// A conflict where two keybindings claim the same key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeybindingConflict {
    pub key: KeyEvent,
    pub keybindings: Vec<KeybindingId>,
}

/// Manages keybinding definitions and user overrides.
pub struct KeybindingsManager {
    definitions: Vec<(KeybindingId, Vec<KeyEvent>)>,
    user_overrides: HashMap<KeybindingId, Vec<KeyEvent>>,
    resolved: HashMap<KeybindingId, Vec<KeyEvent>>,
    conflicts: Vec<KeybindingConflict>,
}

impl KeybindingsManager {
    /// Create a new manager with default TUI keybindings.
    pub fn new() -> Self {
        Self::with_definitions(TUI_KEYBINDINGS)
    }

    /// Create a manager with custom definitions.
    pub fn with_definitions(definitions: &[(KeybindingId, &[KeyEvent])]) -> Self {
        let owned: Vec<_> = definitions
            .iter()
            .map(|(id, keys)| (*id, keys.to_vec()))
            .collect();
        let mut mgr = Self {
            definitions: owned,
            user_overrides: HashMap::new(),
            resolved: HashMap::new(),
            conflicts: Vec::new(),
        };
        mgr.rebuild();
        mgr
    }

    /// Set user keybinding overrides.
    pub fn set_user_bindings(&mut self, overrides: HashMap<KeybindingId, Vec<KeyEvent>>) {
        self.user_overrides = overrides;
        self.rebuild();
    }

    /// Check if raw input data matches a named keybinding.
    pub fn matches(&self, data: &[u8], keybinding: KeybindingId) -> bool {
        self.resolved
            .get(keybinding)
            .is_some_and(|k| k.iter().any(|key| keys::matches_key(data, *key)))
    }

    /// Get the resolved keys for a keybinding.
    pub fn get_keys(&self, keybinding: KeybindingId) -> &[KeyEvent] {
        self.resolved
            .get(keybinding)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get any detected conflicts.
    pub fn get_conflicts(&self) -> &[KeybindingConflict] {
        &self.conflicts
    }

    fn rebuild(&mut self) {
        self.resolved.clear();
        self.conflicts.clear();

        // Detect conflicts in user overrides
        let mut claims: HashMap<KeyEvent, Vec<KeybindingId>> = HashMap::new();
        for (id, user_keys) in &self.user_overrides {
            for key in user_keys {
                claims.entry(*key).or_default().push(id);
            }
        }
        for (key, ids) in &claims {
            if ids.len() > 1 {
                self.conflicts.push(KeybindingConflict {
                    key: *key,
                    keybindings: ids.clone(),
                });
            }
        }

        // Resolve: user override wins, else defaults
        for (id, defaults) in &self.definitions {
            let keys = if let Some(user_keys) = self.user_overrides.get(id) {
                user_keys.clone()
            } else {
                defaults.clone()
            };
            self.resolved.insert(*id, keys);
        }
    }
}

impl Default for KeybindingsManager {
    fn default() -> Self {
        Self::new()
    }
}
