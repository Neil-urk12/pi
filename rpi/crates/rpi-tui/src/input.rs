use std::collections::VecDeque;

use crate::keys::{Key, KeyEvent, Modifiers, SpecialKey};
use crate::Frame;

type Callback = Box<dyn FnMut()>;
type SubmitCallback = Box<dyn FnMut(&str)>;

fn is_punctuation(c: char) -> bool {
    !c.is_alphanumeric() && !c.is_whitespace() && c != '_'
}

/// Maximum undo history depth.
const MAX_UNDO: usize = 100;
struct UndoEntry {
    text: String,
    cursor: usize,
}

#[derive(PartialEq)]
enum LastAction {
    TypeWord,
    Kill,
    Yank,
    Other,
}

pub struct Input {
    text: String,
    cursor_pos: usize,
    undo_stack: VecDeque<UndoEntry>,
    kill_ring: Vec<String>,
    last_was_kill: bool,
    last_action: LastAction,
    on_submit: Option<SubmitCallback>,
    on_escape: Option<Callback>,
}

impl Input {
    pub fn new() -> Self {
        Self {
            text: String::new(),
            cursor_pos: 0,
            undo_stack: VecDeque::new(),
            kill_ring: Vec::new(),
            last_was_kill: false,
            last_action: LastAction::Other,
            on_submit: None,
            on_escape: None,
        }
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn cursor(&self) -> usize {
        self.cursor_pos
    }

    pub fn set_value(&mut self, value: &str) {
        self.text = value.to_string();
        self.cursor_pos = self.text.chars().count();
    }

    pub fn handle_key(&mut self, event: KeyEvent) {
        let key = event.key;
        let mods = event.modifiers;

        // Submit
        if key == Key::Special(SpecialKey::Enter) {
            if let Some(ref mut cb) = self.on_submit {
                cb(&self.text);
            }
            return;
        }

        // Escape
        if key == Key::Special(SpecialKey::Escape) {
            if let Some(ref mut cb) = self.on_escape {
                cb();
            }
            return;
        }

        // Track undo for non-movement keys
        let is_movement = matches!(key,
            Key::Special(SpecialKey::Left) | Key::Special(SpecialKey::Right)
            | Key::Special(SpecialKey::Home) | Key::Special(SpecialKey::End)
        ) || (mods.contains(Modifiers::ALT) && matches!(key, Key::Char('b') | Key::Char('f')))
        || (mods.contains(Modifiers::CTRL) && matches!(key, Key::Char('a') | Key::Char('e') | Key::Char('b') | Key::Char('f')));

        let is_kill = (mods.contains(Modifiers::CTRL) && matches!(key, Key::Char('w') | Key::Char('u') | Key::Char('k')))
            || (mods.contains(Modifiers::ALT) && key == Key::Char('d'));

        if !is_movement && !is_kill && key != Key::Special(SpecialKey::Tab) {
            self.last_was_kill = false;
        }

        // Cursor movement
        if key == Key::Special(SpecialKey::Left) || (mods.contains(Modifiers::CTRL) && key == Key::Char('b')) {
            self.cursor_pos = self.cursor_pos.saturating_sub(1);
            self.last_action = LastAction::Other;
            return;
        }
        if key == Key::Special(SpecialKey::Right) || (mods.contains(Modifiers::CTRL) && key == Key::Char('f')) {
            if self.cursor_pos < self.text.chars().count() {
                self.cursor_pos += 1;
            }
            self.last_action = LastAction::Other;
            return;
        }
        if mods.contains(Modifiers::CTRL) && key == Key::Char('a') {
            self.cursor_pos = 0;
            self.last_action = LastAction::Other;
            return;
        }
        if mods.contains(Modifiers::CTRL) && key == Key::Char('e') {
            self.cursor_pos = self.text.chars().count();
            self.last_action = LastAction::Other;
            return;
        }

        // Word movement
        if mods.contains(Modifiers::ALT) && key == Key::Char('b') {
            self.cursor_pos = self.prev_word_boundary();
            self.last_action = LastAction::Other;
            return;
        }
        if mods.contains(Modifiers::ALT) && key == Key::Char('f') {
            self.cursor_pos = self.next_word_boundary();
            self.last_action = LastAction::Other;
            return;
        }

        // Backspace
        if key == Key::Special(SpecialKey::Backspace) || (mods.contains(Modifiers::CTRL) && key == Key::Char('h')) {
            if self.cursor_pos > 0 {
                self.push_undo();
                let byte_pos = self.char_to_byte(self.cursor_pos);
                let prev_byte = self.char_to_byte(self.cursor_pos - 1);
                self.text.drain(prev_byte..byte_pos);
                self.cursor_pos -= 1;
            }
            self.last_action = LastAction::Other;
            return;
        }

        // Delete forward
        if key == Key::Special(SpecialKey::Delete) || (mods.contains(Modifiers::CTRL) && key == Key::Char('d')) {
            if self.cursor_pos < self.text.chars().count() {
                self.push_undo();
                let byte_pos = self.char_to_byte(self.cursor_pos);
                let next_byte = self.char_to_byte(self.cursor_pos + 1);
                self.text.drain(byte_pos..next_byte);
            }
            self.last_action = LastAction::Other;
            return;
        }

        // Kill word backward (Ctrl+W)
        if mods.contains(Modifiers::CTRL) && key == Key::Char('w') {
            if self.cursor_pos > 0 {
                self.push_undo();
                let end = self.cursor_pos;
                let start = self.prev_word_boundary();
                let killed: String = self.text.chars().skip(start).take(end - start).collect();
                if self.last_was_kill && !self.kill_ring.is_empty() {
                    let last = self.kill_ring.last_mut().unwrap();
                    *last = format!("{}{}", killed, last);
                } else {
                    self.kill_ring.push(killed);
                };
                let start_byte = self.char_to_byte(start);
                let end_byte = self.char_to_byte(end);
                self.text.drain(start_byte..end_byte);
                self.cursor_pos = start;
                self.last_was_kill = true;
                self.last_action = LastAction::Kill;
            }
            return;
        }

        // Kill word forward (Alt+D)
        if mods.contains(Modifiers::ALT) && key == Key::Char('d') {
            if self.cursor_pos < self.text.chars().count() {
                self.push_undo();
                let start = self.cursor_pos;
                let end = self.next_word_boundary();
                let killed: String = self.text.chars().skip(start).take(end - start).collect();
                if self.last_was_kill && !self.kill_ring.is_empty() {
                    let last = self.kill_ring.last_mut().unwrap();
                    *last = format!("{}{}", last, killed);
                } else {
                    self.kill_ring.push(killed);
                };
                let start_byte = self.char_to_byte(start);
                let end_byte = self.char_to_byte(end);
                self.text.drain(start_byte..end_byte);
                self.last_was_kill = true;
                self.last_action = LastAction::Kill;
            }
            return;
        }

        // Kill to line start (Ctrl+U)
        if mods.contains(Modifiers::CTRL) && key == Key::Char('u') {
            if self.cursor_pos > 0 {
                self.push_undo();
                let killed: String = self.text.chars().take(self.cursor_pos).collect();
                if self.last_was_kill && !self.kill_ring.is_empty() {
                    let last = self.kill_ring.last_mut().unwrap();
                    *last = format!("{}{}", killed, last);
                } else {
                    self.kill_ring.push(killed);
                };
                let end_byte = self.char_to_byte(self.cursor_pos);
                self.text.drain(0..end_byte);
                self.cursor_pos = 0;
                self.last_was_kill = true;
                self.last_action = LastAction::Kill;
            }
            return;
        }

        // Kill to line end (Ctrl+K)
        if mods.contains(Modifiers::CTRL) && key == Key::Char('k') {
            if self.cursor_pos < self.text.chars().count() {
                self.push_undo();
                let killed: String = self.text.chars().skip(self.cursor_pos).collect();
                if self.last_was_kill && !self.kill_ring.is_empty() {
                    let last = self.kill_ring.last_mut().unwrap();
                    *last = format!("{}{}", last, killed);
                } else {
                    self.kill_ring.push(killed);
                };
                let start_byte = self.char_to_byte(self.cursor_pos);
                self.text.truncate(start_byte);
                self.last_was_kill = true;
                self.last_action = LastAction::Kill;
            }
            return;
        }

        // Yank (Ctrl+Y)
        if mods.contains(Modifiers::CTRL) && key == Key::Char('y') {
            if let Some(entry) = self.kill_ring.last().cloned() {
                self.push_undo();
                let byte_pos = self.char_to_byte(self.cursor_pos);
                self.text.insert_str(byte_pos, &entry);
                self.cursor_pos += entry.chars().count();
            }
            self.last_action = LastAction::Yank;
            return;
        }

        // Yank pop (Alt+Y)
        if mods.contains(Modifiers::ALT) && key == Key::Char('y') {
            if self.last_action != LastAction::Yank || self.kill_ring.len() <= 1 {
                // noop
            } else {
                self.push_undo();
                // Remove previously yanked text
                let prev_text = self.kill_ring.last().unwrap();
                let prev_len = prev_text.chars().count();
                let start = self.cursor_pos - prev_len;
                let text: String = self.text.chars().take(start)
                    .chain(self.text.chars().skip(self.cursor_pos))
                    .collect();
                self.text = text;
                self.cursor_pos = start;
                // Rotate: pop last, insert at front
                let entry = self.kill_ring.pop().unwrap();
                self.kill_ring.insert(0, entry);
                // Yank new last entry
                let new_text = self.kill_ring.last().unwrap().clone();
                let new_len = new_text.chars().count();
                let byte_pos = self.char_to_byte(self.cursor_pos);
                self.text.insert_str(byte_pos, &new_text);
                self.cursor_pos += new_len;
                self.last_action = LastAction::Yank;
            }
            return;
        }

        // Undo (Ctrl+Z or Ctrl+-)
        if (mods.contains(Modifiers::CTRL) && key == Key::Char('z'))
            || (mods.contains(Modifiers::CTRL) && key == Key::Char('-'))
        {
            if let Some(entry) = self.undo_stack.pop_back() {
                self.text = entry.text;
                self.cursor_pos = entry.cursor;
            }
            self.last_action = LastAction::Other;
            return;
        }

        // Insert character with undo coalescing
        if let Key::Char(c) = key {
            if mods.is_empty() || mods == Modifiers::SHIFT {
                // Undo coalescing: whitespace or non-TypeWord breaks coalescing
                if c.is_whitespace() || self.last_action != LastAction::TypeWord {
                    self.push_undo();
                }
                self.last_action = LastAction::TypeWord;
                let byte_pos = self.char_to_byte(self.cursor_pos);
                self.text.insert(byte_pos, c);
                self.cursor_pos += 1;
            }
        }
    }

    pub fn on_submit(&mut self, f: impl FnMut(&str) + 'static) {
        self.on_submit = Some(Box::new(f));
    }

    pub fn on_escape(&mut self, f: impl FnMut() + 'static) {
        self.on_escape = Some(Box::new(f));
    }

    pub fn render(&self, width: usize) -> Frame {
        let prompt = "> ";
        let prompt_width = unicode_width::UnicodeWidthStr::width(prompt);
        let available = width.saturating_sub(prompt_width);

        // Find scroll offset to keep cursor visible
        let mut scroll = 0;
        if self.cursor_pos > 0 {
            // Calculate width up to cursor
            let width_to_cursor: usize = self.text.chars()
                .take(self.cursor_pos)
                .map(|c| unicode_width::UnicodeWidthChar::width(c).unwrap_or(0))
                .sum();
            if width_to_cursor > available {
                // Need to scroll
                let mut acc = 0;
                for (i, c) in self.text.chars().enumerate() {
                    let w = unicode_width::UnicodeWidthChar::width(c).unwrap_or(0);
                    if acc + w > width_to_cursor - available {
                        scroll = i;
                        break;
                    }
                    acc += w;
                }
            }
        }

        // Collect visible characters
        let mut visible = String::new();
        let mut used_width = 0;
        for c in self.text.chars().skip(scroll) {
            let w = unicode_width::UnicodeWidthChar::width(c).unwrap_or(0);
            if used_width + w > available {
                break;
            }
            visible.push(c);
            used_width += w;
        }

        let line = format!("{}{}", prompt, visible);
        Frame::from_plain_lines([&line])
    }

    fn push_undo(&mut self) {
        if self.undo_stack.len() >= MAX_UNDO {
            self.undo_stack.pop_front();
        }
        self.undo_stack.push_back(UndoEntry {
            text: self.text.clone(),
            cursor: self.cursor_pos,
        });
    }

    fn char_to_byte(&self, char_pos: usize) -> usize {
        self.text.char_indices().nth(char_pos).map(|(i, _)| i).unwrap_or(self.text.len())
    }

    fn prev_word_boundary(&self) -> usize {
        let mut pos = self.cursor_pos;

        // Skip trailing whitespace
        while pos > 0 && matches!(self.text.chars().nth(pos - 1), Some(c) if c.is_whitespace()) {
            pos -= 1;
        }
        // Check if last char is punctuation
        if pos > 0 && matches!(self.text.chars().nth(pos - 1), Some(c) if is_punctuation(c)) {
            // Skip punctuation run
            while pos > 0 && matches!(self.text.chars().nth(pos - 1), Some(c) if is_punctuation(c)) {
                pos -= 1;
            }
        } else {
            // Skip word run (stop at whitespace or punctuation)
            while pos > 0 && matches!(self.text.chars().nth(pos - 1), Some(c) if !c.is_whitespace() && !is_punctuation(c)) {
                pos -= 1;
            }
        }
        pos
    }

    fn next_word_boundary(&self) -> usize {
        let len = self.text.chars().count();
        let mut pos = self.cursor_pos;

        // Skip leading whitespace
        while pos < len && matches!(self.text.chars().nth(pos), Some(c) if c.is_whitespace()) {
            pos += 1;
        }
        // Check if first char is punctuation
        if pos < len && matches!(self.text.chars().nth(pos), Some(c) if is_punctuation(c)) {
            // Skip punctuation run
            while pos < len && matches!(self.text.chars().nth(pos), Some(c) if is_punctuation(c)) {
                pos += 1;
            }
        } else {
            // Skip word run (stop at whitespace or punctuation)
            while pos < len && matches!(self.text.chars().nth(pos), Some(c) if !c.is_whitespace() && !is_punctuation(c)) {
                pos += 1;
            }
        }
        pos
    }
}
