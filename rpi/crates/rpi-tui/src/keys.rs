use bitflags::bitflags;
use std::collections::HashMap;
use std::fmt;
use std::sync::LazyLock;

bitflags! {
    /// Keyboard modifier keys. Bitmask values match the Kitty protocol convention.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct Modifiers: u8 {
        const SHIFT = 1;
        const ALT   = 2;
        const CTRL  = 4;
        const SUPER = 8;
    }
}

/// Lock key mask bits (Caps Lock + Num Lock). Stripped before modifier comparison.
pub const LOCK_MASK: u8 = 64 + 128;

/// Special (non-printable) keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpecialKey {
    Escape,
    Enter,
    Tab,
    Space,
    Backspace,
    Delete,
    Insert,
    Clear,
    Home,
    End,
    PageUp,
    PageDown,
    Up,
    Down,
    Left,
    Right,
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
}

/// A base key — either a printable character or a special key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Key {
    /// A single printable character (letter, digit, or symbol).
    Char(char),
    /// A special (non-printable) key.
    Special(SpecialKey),
}

/// A fully qualified key event: base key + modifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KeyEvent {
    pub key: Key,
    pub modifiers: Modifiers,
}

/// Key event type (Kitty protocol).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum KeyEventType {
    Press,
    Repeat,
    Release,
}

// --- Codepoint constants (Kitty protocol) ---

/// Codepoint constants for special keys in the Kitty protocol.
pub mod codepoints {
    pub const ESCAPE: i32 = 27;
    pub const TAB: i32 = 9;
    pub const ENTER: i32 = 13;
    pub const SPACE: i32 = 32;
    pub const BACKSPACE: i32 = 127;
    pub const KP_ENTER: i32 = 57414;

    // Arrow codepoints (negative = functional key encoding)
    pub const UP: i32 = -1;
    pub const DOWN: i32 = -2;
    pub const RIGHT: i32 = -3;
    pub const LEFT: i32 = -4;

    // Functional key codepoints
    pub const DELETE: i32 = -10;
    pub const INSERT: i32 = -11;
    pub const PAGE_UP: i32 = -12;
    pub const PAGE_DOWN: i32 = -13;
    pub const HOME: i32 = -14;
    pub const END: i32 = -15;

    // F-key codepoints (Kitty CSI-u format uses these)
    pub const F1: i32 = -16;
    pub const F2: i32 = -17;
    pub const F3: i32 = -18;
    pub const F4: i32 = -19;
    pub const F5: i32 = -20;
    pub const F6: i32 = -21;
    pub const F7: i32 = -22;
    pub const F8: i32 = -23;
    pub const F9: i32 = -24;
    pub const F10: i32 = -25;
    pub const F11: i32 = -26;
    pub const F12: i32 = -27;
    // KP_BEGIN (numpad 5 without NumLock, aka Clear)
    pub const KP_BEGIN: i32 = 57427;
}

// --- Kitty numpad normalization map ---

/// Maps Kitty numpad codepoints (57399-57427) to their standard equivalents.
pub fn normalize_kitty_functional_codepoint(codepoint: i32) -> i32 {
    match codepoint {
        57399 => 48,                            // KP_0 → '0'
        57400 => 49,                            // KP_1 → '1'
        57401 => 50,                            // KP_2 → '2'
        57402 => 51,                            // KP_3 → '3'
        57403 => 52,                            // KP_4 → '4'
        57404 => 53,                            // KP_5 → '5'
        57405 => 54,                            // KP_6 → '6'
        57406 => 55,                            // KP_7 → '7'
        57407 => 56,                            // KP_8 → '8'
        57408 => 57,                            // KP_9 → '9'
        57409 => 46,                            // KP_DECIMAL → '.'
        57410 => 47,                            // KP_DIVIDE → '/'
        57411 => 42,                            // KP_MULTIPLY → '*'
        57412 => 45,                            // KP_SUBTRACT → '-'
        57413 => 43,                            // KP_ADD → '+'
        57414 => codepoints::ENTER,             // KP_ENTER → Enter
        57415 => 61,                            // KP_EQUAL → '='
        57416 => 44,                            // KP_SEPARATOR → ','
        57417 => codepoints::LEFT,              // KP_LEFT
        57418 => codepoints::RIGHT,             // KP_RIGHT
        57419 => codepoints::UP,                // KP_UP
        57420 => codepoints::DOWN,              // KP_DOWN
        57421 => codepoints::PAGE_UP,           // KP_PAGE_UP
        57422 => codepoints::PAGE_DOWN,         // KP_PAGE_DOWN
        57423 => codepoints::HOME,              // KP_HOME
        57424 => codepoints::END,               // KP_END
        57425 => codepoints::INSERT,            // KP_INSERT
        57426 => codepoints::DELETE,            // KP_DELETE
        57427 => codepoints::KP_BEGIN,          // KP_BEGIN (Clear)
        _ => codepoint,
    }
}

// --- Display impls ---

impl fmt::Display for Modifiers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for (name, flag) in [
            ("ctrl", Modifiers::CTRL),
            ("shift", Modifiers::SHIFT),
            ("alt", Modifiers::ALT),
            ("super", Modifiers::SUPER),
        ] {
            if self.contains(flag) {
                if !first {
                    write!(f, "+")?;
                }
                write!(f, "{name}")?;
                first = false;
            }
        }
        Ok(())
    }
}

impl fmt::Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Key::Char(c) => write!(f, "{c}"),
            Key::Special(s) => write!(f, "{s}"),
        }
    }
}

impl fmt::Display for SpecialKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            SpecialKey::Escape => "escape",
            SpecialKey::Enter => "enter",
            SpecialKey::Tab => "tab",
            SpecialKey::Space => "space",
            SpecialKey::Backspace => "backspace",
            SpecialKey::Delete => "delete",
            SpecialKey::Insert => "insert",
            SpecialKey::Clear => "clear",
            SpecialKey::Home => "home",
            SpecialKey::End => "end",
            SpecialKey::PageUp => "pageUp",
            SpecialKey::PageDown => "pageDown",
            SpecialKey::Up => "up",
            SpecialKey::Down => "down",
            SpecialKey::Left => "left",
            SpecialKey::Right => "right",
            SpecialKey::F1 => "f1",
            SpecialKey::F2 => "f2",
            SpecialKey::F3 => "f3",
            SpecialKey::F4 => "f4",
            SpecialKey::F5 => "f5",
            SpecialKey::F6 => "f6",
            SpecialKey::F7 => "f7",
            SpecialKey::F8 => "f8",
            SpecialKey::F9 => "f9",
            SpecialKey::F10 => "f10",
            SpecialKey::F11 => "f11",
            SpecialKey::F12 => "f12",
        };
        write!(f, "{s}")
    }
}

/// Format a KeyEvent as a KeyId string (e.g. "ctrl+shift+k").
impl fmt::Display for KeyEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.modifiers.is_empty() {
            write!(f, "{}", self.key)
        } else {
            write!(f, "{}+{}", self.modifiers, self.key)
        }
    }
}

// --- KeyId string parsing ---

/// Parse a KeyId string (e.g. "ctrl+shift+k") into a KeyEvent.
///
/// Returns `None` if the string is not a valid KeyId.
pub fn parse_key_id(s: &str) -> Option<KeyEvent> {
    let parts: Vec<&str> = s.split('+').collect();
    if parts.is_empty() {
        return None;
    }

    let mut modifiers = Modifiers::empty();
    let mut key_part = None;

    for (i, part) in parts.iter().enumerate() {
        match *part {
            "ctrl" => modifiers |= Modifiers::CTRL,
            "shift" => modifiers |= Modifiers::SHIFT,
            "alt" => modifiers |= Modifiers::ALT,
            "super" => modifiers |= Modifiers::SUPER,
            _ => {
                if i != parts.len() - 1 || key_part.is_some() {
                    return None;
                }
                key_part = Some(*part);
            }
        }
    }

    let key_str = key_part?;
    let key = parse_base_key(key_str)?;
    Some(KeyEvent { key, modifiers })
}

fn parse_base_key(s: &str) -> Option<Key> {
    match s {
        "escape" | "esc" => Some(Key::Special(SpecialKey::Escape)),
        "enter" | "return" => Some(Key::Special(SpecialKey::Enter)),
        "tab" => Some(Key::Special(SpecialKey::Tab)),
        "space" => Some(Key::Special(SpecialKey::Space)),
        "backspace" => Some(Key::Special(SpecialKey::Backspace)),
        "delete" => Some(Key::Special(SpecialKey::Delete)),
        "insert" => Some(Key::Special(SpecialKey::Insert)),
        "clear" => Some(Key::Special(SpecialKey::Clear)),
        "home" => Some(Key::Special(SpecialKey::Home)),
        "end" => Some(Key::Special(SpecialKey::End)),
        "pageUp" => Some(Key::Special(SpecialKey::PageUp)),
        "pageDown" => Some(Key::Special(SpecialKey::PageDown)),
        "up" => Some(Key::Special(SpecialKey::Up)),
        "down" => Some(Key::Special(SpecialKey::Down)),
        "left" => Some(Key::Special(SpecialKey::Left)),
        "right" => Some(Key::Special(SpecialKey::Right)),
        "f1" => Some(Key::Special(SpecialKey::F1)),
        "f2" => Some(Key::Special(SpecialKey::F2)),
        "f3" => Some(Key::Special(SpecialKey::F3)),
        "f4" => Some(Key::Special(SpecialKey::F4)),
        "f5" => Some(Key::Special(SpecialKey::F5)),
        "f6" => Some(Key::Special(SpecialKey::F6)),
        "f7" => Some(Key::Special(SpecialKey::F7)),
        "f8" => Some(Key::Special(SpecialKey::F8)),
        "f9" => Some(Key::Special(SpecialKey::F9)),
        "f10" => Some(Key::Special(SpecialKey::F10)),
        "f11" => Some(Key::Special(SpecialKey::F11)),
        "f12" => Some(Key::Special(SpecialKey::F12)),
        _ if s.len() == 1 => Some(Key::Char(s.chars().next().unwrap())),
        _ => None,
    }
}

// --- Legacy escape sequence constants ---

/// Maps special key names to their legacy escape sequences.
pub static LEGACY_KEY_SEQUENCES: LazyLock<HashMap<SpecialKey, Vec<Vec<u8>>>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    m.insert(SpecialKey::Up, vec![b"\x1bOA".to_vec(), b"\x1b[A".to_vec()]);
    m.insert(SpecialKey::Down, vec![b"\x1bOB".to_vec(), b"\x1b[B".to_vec()]);
    m.insert(SpecialKey::Right, vec![b"\x1bOC".to_vec(), b"\x1b[C".to_vec()]);
    m.insert(SpecialKey::Left, vec![b"\x1bOD".to_vec(), b"\x1b[D".to_vec()]);
    m.insert(SpecialKey::Home, vec![b"\x1bOH".to_vec(), b"\x1b[H".to_vec()]);
    m.insert(SpecialKey::End, vec![b"\x1bOF".to_vec(), b"\x1b[F".to_vec()]);
    m.insert(SpecialKey::Insert, vec![b"\x1b[2~".to_vec()]);
    m.insert(SpecialKey::Delete, vec![b"\x1b[3~".to_vec()]);
    m.insert(SpecialKey::PageUp, vec![b"\x1b[5~".to_vec()]);
    m.insert(SpecialKey::PageDown, vec![b"\x1b[6~".to_vec()]);
    m.insert(SpecialKey::Clear, vec![b"\x1b[E".to_vec(), b"\x1bOw".to_vec(), b"\x1b[1;2w".to_vec()]);
    m.insert(SpecialKey::F1, vec![b"\x1bOP".to_vec(), b"\x1b[11~".to_vec(), b"\x1b[[A".to_vec()]);
    m.insert(SpecialKey::F2, vec![b"\x1bOQ".to_vec(), b"\x1b[12~".to_vec(), b"\x1b[[B".to_vec()]);
    m.insert(SpecialKey::F3, vec![b"\x1bOR".to_vec(), b"\x1b[13~".to_vec(), b"\x1b[[C".to_vec()]);
    m.insert(SpecialKey::F4, vec![b"\x1bOS".to_vec(), b"\x1b[14~".to_vec(), b"\x1b[[D".to_vec()]);
    m.insert(SpecialKey::F5, vec![b"\x1b[15~".to_vec(), b"\x1b[[E".to_vec()]);
    m.insert(SpecialKey::F6, vec![b"\x1b[17~".to_vec()]);
    m.insert(SpecialKey::F7, vec![b"\x1b[18~".to_vec()]);
    m.insert(SpecialKey::F8, vec![b"\x1b[19~".to_vec()]);
    m.insert(SpecialKey::F9, vec![b"\x1b[20~".to_vec()]);
    m.insert(SpecialKey::F10, vec![b"\x1b[21~".to_vec()]);
    m.insert(SpecialKey::F11, vec![b"\x1b[23~".to_vec()]);
    m.insert(SpecialKey::F12, vec![b"\x1b[24~".to_vec()]);
    m
});

/// Maps legacy escape sequences to their KeyId string representation.
pub static LEGACY_SEQUENCE_KEY_IDS: LazyLock<Vec<(Vec<u8>, &'static str)>> = LazyLock::new(|| {
    vec![
        (b"\x1bOA".to_vec(), "up"),
        (b"\x1b[A".to_vec(), "up"),
        (b"\x1bOB".to_vec(), "down"),
        (b"\x1b[B".to_vec(), "down"),
        (b"\x1bOC".to_vec(), "right"),
        (b"\x1b[C".to_vec(), "right"),
        (b"\x1bOD".to_vec(), "left"),
        (b"\x1b[D".to_vec(), "left"),
        (b"\x1b[1;2A".to_vec(), "shift+up"),
        (b"\x1b[1;2B".to_vec(), "shift+down"),
        (b"\x1b[1;2C".to_vec(), "shift+right"),
        (b"\x1b[1;2D".to_vec(), "shift+left"),
        (b"\x1b[1;5A".to_vec(), "ctrl+up"),
        (b"\x1b[1;5B".to_vec(), "ctrl+down"),
        (b"\x1b[1;5C".to_vec(), "ctrl+right"),
        (b"\x1b[1;5D".to_vec(), "ctrl+left"),
        (b"\x1b[1;3A".to_vec(), "alt+up"),
        (b"\x1b[1;3B".to_vec(), "alt+down"),
        (b"\x1b[1;3C".to_vec(), "alt+right"),
        (b"\x1b[1;3D".to_vec(), "alt+left"),
        (b"\x1b[1;4A".to_vec(), "shift+alt+up"),
        (b"\x1b[1;4B".to_vec(), "shift+alt+down"),
        (b"\x1b[1;4C".to_vec(), "shift+alt+right"),
        (b"\x1b[1;4D".to_vec(), "shift+alt+left"),
        (b"\x1b[1;6A".to_vec(), "ctrl+shift+up"),
        (b"\x1b[1;6B".to_vec(), "ctrl+shift+down"),
        (b"\x1b[1;6C".to_vec(), "ctrl+shift+right"),
        (b"\x1b[1;6D".to_vec(), "ctrl+shift+left"),
        (b"\x1bOH".to_vec(), "home"),
        (b"\x1bOF".to_vec(), "end"),
        (b"\x1b[H".to_vec(), "home"),
        (b"\x1b[F".to_vec(), "end"),
        (b"\x1b[2~".to_vec(), "insert"),
        (b"\x1b[3~".to_vec(), "delete"),
        (b"\x1b[5~".to_vec(), "pageUp"),
        (b"\x1b[6~".to_vec(), "pageDown"),
        (b"\x1b[2;2~".to_vec(), "shift+insert"),
        (b"\x1b[3;2~".to_vec(), "shift+delete"),
        (b"\x1b[5;2~".to_vec(), "shift+pageUp"),
        (b"\x1b[6;2~".to_vec(), "shift+pageDown"),
        (b"\x1b[1;2H".to_vec(), "shift+home"),
        (b"\x1b[1;2F".to_vec(), "shift+end"),
        (b"\x1b[2;5~".to_vec(), "ctrl+insert"),
        (b"\x1b[3;5~".to_vec(), "ctrl+delete"),
        (b"\x1b[5;5~".to_vec(), "ctrl+pageUp"),
        (b"\x1b[6;5~".to_vec(), "ctrl+pageDown"),
        (b"\x1b[1;5H".to_vec(), "ctrl+home"),
        (b"\x1b[1;5F".to_vec(), "ctrl+end"),
        (b"\x1bOP".to_vec(), "f1"),
        (b"\x1bOQ".to_vec(), "f2"),
        (b"\x1bOR".to_vec(), "f3"),
        (b"\x1bOS".to_vec(), "f4"),
        (b"\x1b[15~".to_vec(), "f5"),
        (b"\x1b[17~".to_vec(), "f6"),
        (b"\x1b[18~".to_vec(), "f7"),
        (b"\x1b[19~".to_vec(), "f8"),
        (b"\x1b[20~".to_vec(), "f9"),
        (b"\x1b[21~".to_vec(), "f10"),
        (b"\x1b[23~".to_vec(), "f11"),
        (b"\x1b[24~".to_vec(), "f12"),
        (b"\x1bOM".to_vec(), "enter"),
        (b"\x1b[Z".to_vec(), "shift+tab"),
        (b"\x00".to_vec(), "ctrl+space"),
        (b"\x1b[E".to_vec(), "clear"),
        (b"\x1bOw".to_vec(), "clear"),
        (b"\x1b[1;2w".to_vec(), "clear"),
    ]
});

// --- Kitty/modOK parser ---

/// Parsed Kitty CSI-u sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParsedKittySequence {
    pub codepoint: i32,
    pub shifted_codepoint: Option<i32>,
    pub base_layout_codepoint: Option<i32>,
    pub modifiers: Modifiers,
    pub event_type: KeyEventType,
}

/// Parsed modifyOtherKeys sequence (CSI 27;mod;keycode~).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParsedModifyOtherKeysSequence {
    pub codepoint: i32,
    pub modifiers: Modifiers,
}

/// Parse a Kitty CSI-u escape sequence.
pub(crate) fn parse_kitty_sequence(data: &[u8]) -> Option<ParsedKittySequence> {
    if data.len() < 3 || data[0] != b'\x1b' || data[1] != b'[' {
        return None;
    }

    let inner = &data[2..];

    // Arrow keys: CSI 1;mod A-D
    if inner.len() >= 3
        && inner[0] == b'1'
        && inner[1] == b';'
        && inner.last().is_some_and(|b| matches!(b, b'A'..=b'D'))
    {
        let mod_str = std::str::from_utf8(&inner[2..inner.len() - 1]).ok()?;
        let modifier = if mod_str.is_empty() { 1 } else { mod_str.parse::<u8>().ok()? };
        let codepoint = match inner[inner.len() - 1] {
            b'A' => codepoints::UP,
            b'B' => codepoints::DOWN,
            b'C' => codepoints::RIGHT,
            b'D' => codepoints::LEFT,
            _ => unreachable!(),
        };
        return Some(ParsedKittySequence {
            codepoint,
            shifted_codepoint: None,
            base_layout_codepoint: None,
            modifiers: parse_modifier_byte(modifier),
            event_type: KeyEventType::Press,
        });
    }

    // Home/End: CSI 1;mod H/F
    if inner.len() >= 3
        && inner[0] == b'1'
        && inner[1] == b';'
        && matches!(inner.last(), Some(b'H' | b'F'))
    {
        let mod_str = std::str::from_utf8(&inner[2..inner.len() - 1]).ok()?;
        let modifier = if mod_str.is_empty() { 1 } else { mod_str.parse::<u8>().ok()? };
        let codepoint = if inner[inner.len() - 1] == b'H' {
            codepoints::HOME
        } else {
            codepoints::END
        };
        return Some(ParsedKittySequence {
            codepoint,
            shifted_codepoint: None,
            base_layout_codepoint: None,
            modifiers: parse_modifier_byte(modifier),
            event_type: KeyEventType::Press,
        });
    }

    // Functional keys: CSI num;mod~ or CSI num~
    if inner.len() >= 2 && inner.last() == Some(&b'~') {
        let body = &inner[..inner.len() - 1];
        let (num_str, modifier) = if let Some(semicolon_pos) = body.iter().position(|&b| b == b';') {
            let ns = std::str::from_utf8(&body[..semicolon_pos]).ok()?;
            let ms = std::str::from_utf8(&body[semicolon_pos + 1..]).ok()?.parse::<u8>().unwrap_or(1);
            (ns, ms)
        } else {
            (std::str::from_utf8(body).ok()?, 1u8)
        };
        let num = num_str.parse::<i32>().ok()?;
        let codepoint = match num {
            2 => codepoints::INSERT,
            3 => codepoints::DELETE,
            5 => codepoints::PAGE_UP,
            6 => codepoints::PAGE_DOWN,
            7 => codepoints::HOME,
            8 => codepoints::END,
            11 => codepoints::F1,
            12 => codepoints::F2,
            13 => codepoints::F3,
            14 => codepoints::F4,
            15 => codepoints::F5,
            17 => codepoints::F6,
            18 => codepoints::F7,
            19 => codepoints::F8,
            20 => codepoints::F9,
            21 => codepoints::F10,
            23 => codepoints::F11,
            24 => codepoints::F12,
            57427 => codepoints::KP_BEGIN,
            _ => return None,
        };
        return Some(ParsedKittySequence {
            codepoint,
            shifted_codepoint: None,
            base_layout_codepoint: None,
            modifiers: parse_modifier_byte(modifier),
            event_type: KeyEventType::Press,
        });
    }

    // CSI-u format: CSI codepoint[:shifted[:base]][;modifier[:event]]u
    if inner.last() != Some(&b'u') {
        return None;
    }
    let body = &inner[..inner.len() - 1];

    let (codepoint_part, modifier_part) =
        if let Some(pos) = body.iter().position(|&b| b == b';') {
            (&body[..pos], Some(&body[pos + 1..]))
        } else {
            (body, None)
        };

    let codepoint_parts: Vec<&[u8]> = codepoint_part.split(|&b| b == b':').collect();
    let codepoint = std::str::from_utf8(codepoint_parts[0])
        .ok()?
        .parse::<i32>()
        .ok()?;
    let shifted_codepoint = if codepoint_parts.len() > 1 && !codepoint_parts[1].is_empty() {
        std::str::from_utf8(codepoint_parts[1])
            .ok()?
            .parse::<i32>()
            .ok()
            .filter(|&v| v != 0)
    } else {
        None
    };
    let base_layout_codepoint = if codepoint_parts.len() > 2 && !codepoint_parts[2].is_empty() {
        std::str::from_utf8(codepoint_parts[2])
            .ok()?
            .parse::<i32>()
            .ok()
            .filter(|&v| v != 0)
    } else {
        None
    };

    let (modifier_raw, event_type) = if let Some(mod_bytes) = modifier_part {
        let mod_parts: Vec<&[u8]> = mod_bytes.split(|&b| b == b':').collect();
        let m = if mod_parts[0].is_empty() {
            1
        } else {
            std::str::from_utf8(mod_parts[0])
                .ok()?
                .parse::<u8>()
                .unwrap_or(1)
        };
        let evt = if mod_parts.len() > 1 {
            parse_event_type(mod_parts[1])
        } else {
            KeyEventType::Press
        };
        (m, evt)
    } else {
        (1, KeyEventType::Press)
    };

    Some(ParsedKittySequence {
        codepoint,
        shifted_codepoint,
        base_layout_codepoint,
        modifiers: parse_modifier_byte(modifier_raw),
        event_type,
    })
}

/// Parse a modifyOtherKeys sequence: `\x1b[27;mod;keycode~`
pub(crate) fn parse_modify_other_keys_sequence(data: &[u8]) -> Option<ParsedModifyOtherKeysSequence> {
    if data.len() < 8 || data[0] != b'\x1b' || data[1] != b'[' {
        return None;
    }
    let inner = &data[2..];
    if !inner.starts_with(b"27;") {
        return None;
    }
    let rest = &inner[3..];
    let tilde_pos = rest.iter().position(|&b| b == b'~')?;
    let mid = &rest[..tilde_pos];
    let semi_pos = mid.iter().position(|&b| b == b';')?;
    let mod_str = std::str::from_utf8(&mid[..semi_pos]).ok()?;
    let key_str = std::str::from_utf8(&mid[semi_pos + 1..]).ok()?;
    let modifier = mod_str.parse::<u8>().ok()?;
    let codepoint = key_str.parse::<i32>().ok()?;

    Some(ParsedModifyOtherKeysSequence {
        codepoint,
        modifiers: parse_modifier_byte(modifier),
    })
}

/// Parse a modifier bitmask byte (Kitty convention: shift=1, alt=2, ctrl=4, super=8).
/// Bitmask is 1-indexed (raw value 1 = no modifiers).
fn parse_modifier_byte(raw: u8) -> Modifiers {
    if raw == 0 {
        return Modifiers::empty();
    }
    let bits = raw.saturating_sub(1);
    Modifiers::from_bits_truncate(bits & !LOCK_MASK)
}

fn parse_event_type(raw: &[u8]) -> KeyEventType {
    match raw {
        b"2" => KeyEventType::Repeat,
        b"3" => KeyEventType::Release,
        _ => KeyEventType::Press,
    }
}

// --- Sequence matching ---

/// Check if `data` matches a Kitty sequence for the given codepoint and modifier.
pub fn matches_kitty_sequence(
    data: &[u8],
    expected_codepoint: i32,
    expected_modifiers: Modifiers,
) -> bool {
    let Some(parsed) = parse_kitty_sequence(data) else {
        return false;
    };
    let normalized = normalize_kitty_functional_codepoint(parsed.codepoint);
    // Strip lock bits from modifier for comparison
    let actual_modifiers = parsed.modifiers;

    // Direct match with numpad-normalized codepoint
    if normalized == expected_codepoint && actual_modifiers == expected_modifiers {
        return true;
    }

    // Match uppercase codepoint when expected is lowercase (shift implicit in case)
    // e.g. CSI 65;2u (Shift+A) should match expected 97 ('a') with SHIFT
    if (65..=90).contains(&normalized)
        && normalized + 32 == expected_codepoint
        && actual_modifiers.contains(Modifiers::SHIFT)
        && expected_modifiers.contains(Modifiers::SHIFT)
    {
        return true;
    }

    // Fallback: base layout key for non-Latin keyboards
    if let Some(base) = parsed.base_layout_codepoint
        && !is_standard_printable(expected_codepoint)
        && normalize_kitty_functional_codepoint(base) == expected_codepoint
    {
        return actual_modifiers == expected_modifiers;
    }

    false
}

/// Check if `data` matches a modifyOtherKeys sequence.
pub fn matches_modify_other_keys(
    data: &[u8],
    expected_codepoint: i32,
    expected_modifiers: Modifiers,
) -> bool {
    let Some(parsed) = parse_modify_other_keys_sequence(data) else {
        return false;
    };

    // Direct match
    if parsed.codepoint == expected_codepoint && parsed.modifiers == expected_modifiers {
        return true;
    }

    // Match uppercase codepoint when expected is lowercase (shift implicit in case)
    if (65..=90).contains(&parsed.codepoint)
        && parsed.codepoint + 32 == expected_codepoint
        && parsed.modifiers.contains(Modifiers::SHIFT)
        && expected_modifiers.contains(Modifiers::SHIFT)
    {
        return true;
    }

    false
}

fn strip_shift_from_letter(codepoint: i32, modifiers: Modifiers) -> (i32, Modifiers) {
    if (65..=90).contains(&codepoint) && modifiers.contains(Modifiers::SHIFT) {
        (codepoint + 32, modifiers & !Modifiers::SHIFT)
    } else {
        (codepoint, modifiers)
    }
}

fn is_standard_printable(codepoint: i32) -> bool {
    (32..=126).contains(&codepoint)
}

/// Check if the raw data is a key release event (Kitty protocol).
pub fn is_key_release(data: &[u8]) -> bool {
    parse_kitty_sequence(data).is_some_and(|p| p.event_type == KeyEventType::Release)
}

/// Check if the raw data is a key repeat event (Kitty protocol).
pub fn is_key_repeat(data: &[u8]) -> bool {
    parse_kitty_sequence(data).is_some_and(|p| p.event_type == KeyEventType::Repeat)
}

// --- matches_key / parse_key ---

/// Check if a raw data byte slice matches a `KeyEvent`.
pub fn matches_key(data: &[u8], expected: KeyEvent) -> bool {
    let key = expected.key;
    let modifiers = expected.modifiers;

    match key {
        Key::Special(special) => match_special_key(data, special, modifiers),
        Key::Char(c) => match_char_key(data, c, modifiers),
    }
}

fn match_special_key(data: &[u8], key: SpecialKey, modifiers: Modifiers) -> bool {
    match key {
        SpecialKey::Escape => {
            data == b"\x1b"
                || matches_kitty_sequence(data, codepoints::ESCAPE, modifiers)
                || matches_modify_other_keys(data, codepoints::ESCAPE, modifiers)
        }
        SpecialKey::Tab => {
            data == b"\t"
                || (modifiers.contains(Modifiers::SHIFT) && data == b"\x1b[Z")
                || matches_kitty_sequence(data, codepoints::TAB, modifiers)
                || matches_modify_other_keys(data, codepoints::TAB, modifiers)
        }
        SpecialKey::Enter => {
            data == b"\r"
                || data == b"\n"
                || (modifiers.is_empty() && data == b"\x1bOM")
                || matches_kitty_sequence(data, codepoints::ENTER, modifiers)
                || matches_modify_other_keys(data, codepoints::ENTER, modifiers)
        }
        SpecialKey::Space => {
            (modifiers.is_empty() && data == b" ")
                || (modifiers.contains(Modifiers::CTRL) && data == b"\x00")
                || (modifiers.contains(Modifiers::ALT) && data == b"\x1b ")
                || matches_kitty_sequence(data, codepoints::SPACE, modifiers)
                || matches_modify_other_keys(data, codepoints::SPACE, modifiers)
        }
        SpecialKey::Backspace => {
            matches_raw_backspace(data, modifiers)
                || (modifiers.contains(Modifiers::ALT)
                    && (data == b"\x1b\x7f" || data == b"\x1b\x08"))
                || matches_kitty_sequence(data, codepoints::BACKSPACE, modifiers)
                || matches_modify_other_keys(data, codepoints::BACKSPACE, modifiers)
        }
        SpecialKey::Up => {
            matches_arrow(data, b'A', modifiers)
                || matches_kitty_sequence(data, codepoints::UP, modifiers)
                || matches_modify_other_keys(data, codepoints::UP, modifiers)
        }
        SpecialKey::Down => {
            matches_arrow(data, b'B', modifiers)
                || matches_kitty_sequence(data, codepoints::DOWN, modifiers)
                || matches_modify_other_keys(data, codepoints::DOWN, modifiers)
        }
        SpecialKey::Right => {
            matches_arrow(data, b'C', modifiers)
                || matches_kitty_sequence(data, codepoints::RIGHT, modifiers)
                || matches_modify_other_keys(data, codepoints::RIGHT, modifiers)
        }
        SpecialKey::Left => {
            matches_arrow(data, b'D', modifiers)
                || matches_kitty_sequence(data, codepoints::LEFT, modifiers)
                || matches_modify_other_keys(data, codepoints::LEFT, modifiers)
        }
        SpecialKey::Home => {
            matches_legacy_key(data, SpecialKey::Home)
                || matches_kitty_sequence(data, codepoints::HOME, modifiers)
                || matches_modify_other_keys(data, codepoints::HOME, modifiers)
        }
        SpecialKey::End => {
            matches_legacy_key(data, SpecialKey::End)
                || matches_kitty_sequence(data, codepoints::END, modifiers)
                || matches_modify_other_keys(data, codepoints::END, modifiers)
        }
        SpecialKey::Insert => {
            matches_legacy_key(data, SpecialKey::Insert)
                || matches_kitty_sequence(data, codepoints::INSERT, modifiers)
                || matches_modify_other_keys(data, codepoints::INSERT, modifiers)
        }
        SpecialKey::Delete => {
            matches_legacy_key(data, SpecialKey::Delete)
                || matches_kitty_sequence(data, codepoints::DELETE, modifiers)
                || matches_modify_other_keys(data, codepoints::DELETE, modifiers)
        }
        SpecialKey::PageUp => {
            matches_legacy_key(data, SpecialKey::PageUp)
                || matches_kitty_sequence(data, codepoints::PAGE_UP, modifiers)
                || matches_modify_other_keys(data, codepoints::PAGE_UP, modifiers)
        }
        SpecialKey::PageDown => {
            matches_legacy_key(data, SpecialKey::PageDown)
                || matches_kitty_sequence(data, codepoints::PAGE_DOWN, modifiers)
                || matches_modify_other_keys(data, codepoints::PAGE_DOWN, modifiers)
        }
        SpecialKey::Clear => {
            matches_legacy_key(data, SpecialKey::Clear)
                || matches_kitty_sequence(data, codepoints::KP_BEGIN, modifiers)
                || matches_modify_other_keys(data, codepoints::KP_BEGIN, modifiers)
        }
        SpecialKey::F1 => {
            matches_legacy_key(data, SpecialKey::F1)
                || matches_kitty_sequence(data, codepoints::F1, modifiers)
                || matches_modify_other_keys(data, codepoints::F1, modifiers)
        }
        SpecialKey::F2 => {
            matches_legacy_key(data, SpecialKey::F2)
                || matches_kitty_sequence(data, codepoints::F2, modifiers)
                || matches_modify_other_keys(data, codepoints::F2, modifiers)
        }
        SpecialKey::F3 => {
            matches_legacy_key(data, SpecialKey::F3)
                || matches_kitty_sequence(data, codepoints::F3, modifiers)
                || matches_modify_other_keys(data, codepoints::F3, modifiers)
        }
        SpecialKey::F4 => {
            matches_legacy_key(data, SpecialKey::F4)
                || matches_kitty_sequence(data, codepoints::F4, modifiers)
                || matches_modify_other_keys(data, codepoints::F4, modifiers)
        }
        SpecialKey::F5 => {
            matches_legacy_key(data, SpecialKey::F5)
                || matches_kitty_sequence(data, codepoints::F5, modifiers)
                || matches_modify_other_keys(data, codepoints::F5, modifiers)
        }
        SpecialKey::F6 => {
            matches_legacy_key(data, SpecialKey::F6)
                || matches_kitty_sequence(data, codepoints::F6, modifiers)
                || matches_modify_other_keys(data, codepoints::F6, modifiers)
        }
        SpecialKey::F7 => {
            matches_legacy_key(data, SpecialKey::F7)
                || matches_kitty_sequence(data, codepoints::F7, modifiers)
                || matches_modify_other_keys(data, codepoints::F7, modifiers)
        }
        SpecialKey::F8 => {
            matches_legacy_key(data, SpecialKey::F8)
                || matches_kitty_sequence(data, codepoints::F8, modifiers)
                || matches_modify_other_keys(data, codepoints::F8, modifiers)
        }
        SpecialKey::F9 => {
            matches_legacy_key(data, SpecialKey::F9)
                || matches_kitty_sequence(data, codepoints::F9, modifiers)
                || matches_modify_other_keys(data, codepoints::F9, modifiers)
        }
        SpecialKey::F10 => {
            matches_legacy_key(data, SpecialKey::F10)
                || matches_kitty_sequence(data, codepoints::F10, modifiers)
                || matches_modify_other_keys(data, codepoints::F10, modifiers)
        }
        SpecialKey::F11 => {
            matches_legacy_key(data, SpecialKey::F11)
                || matches_kitty_sequence(data, codepoints::F11, modifiers)
                || matches_modify_other_keys(data, codepoints::F11, modifiers)
        }
        SpecialKey::F12 => {
            matches_legacy_key(data, SpecialKey::F12)
                || matches_kitty_sequence(data, codepoints::F12, modifiers)
                || matches_modify_other_keys(data, codepoints::F12, modifiers)
        }
    }
}

fn match_char_key(data: &[u8], c: char, modifiers: Modifiers) -> bool {
    let codepoint = c as i32;

    // Legacy plain character
    if modifiers.is_empty() && data.len() == 1 && data[0] == c as u8 {
        return true;
    }

    // Ctrl+letter: raw control character (code & 0x1f)
    if modifiers == Modifiers::CTRL
        && let Some(ctrl) = raw_ctrl_char(c)
        && data.len() == 1
        && data[0] == ctrl
    {
        return true;
    }

    // Shift+letter: uppercase letter (legacy terminals)
    if modifiers == Modifiers::SHIFT && c.is_ascii_lowercase() && data.len() == 1 && data[0] == (c as u8 - 32) {
        return true;
    }

    // Alt+character: ESC + character (legacy)
    if modifiers == Modifiers::ALT && data.len() == 2 && data[0] == b'\x1b' && data[1] == c as u8 {
        return true;
    }

    // Kitty/modOtherKeys fallback for all cases
    matches_kitty_sequence(data, codepoint, modifiers)
        || matches_modify_other_keys(data, codepoint, modifiers)
}

fn raw_ctrl_char(c: char) -> Option<u8> {
    match c {
        'a'..='z' => Some((c as u8) & 0x1f),
        '[' => Some(0x1b),
        '\\' => Some(0x1c),
        ']' => Some(0x1d),
        '_' => Some(0x1f),
        '-' => Some(0x1f),
        _ => None,
    }
}

fn matches_arrow(data: &[u8], direction: u8, modifiers: Modifiers) -> bool {
    if modifiers.is_empty() {
        return data == [b'\x1b', b'O', direction] || data == [b'\x1b', b'[', direction];
    }
    if modifiers == Modifiers::SHIFT {
        return data == [b'\x1b', b'[', b'1', b';', b'2', direction];
    }
    if modifiers == Modifiers::CTRL {
        return data == [b'\x1b', b'[', b'1', b';', b'5', direction];
    }
    if modifiers == Modifiers::ALT {
        return data == [b'\x1b', b'[', b'1', b';', b'3', direction];
    }
    if modifiers == Modifiers::SHIFT | Modifiers::ALT {
        return data == [b'\x1b', b'[', b'1', b';', b'4', direction];
    }
    if modifiers == Modifiers::CTRL | Modifiers::SHIFT {
        return data == [b'\x1b', b'[', b'1', b';', b'6', direction];
    }
    false
}

fn matches_legacy_key(data: &[u8], key: SpecialKey) -> bool {
    LEGACY_KEY_SEQUENCES
        .get(&key)
        .is_some_and(|seqs| seqs.iter().any(|seq| data == seq.as_slice()))
}

fn matches_raw_backspace(data: &[u8], modifiers: Modifiers) -> bool {
    if modifiers.is_empty() && data == b"\x7f" {
        return true;
    }
    if modifiers.is_empty() && data == b"\x08" {
        return std::env::var("WT_SESSION").is_err();
    }
    if modifiers == Modifiers::CTRL && data == b"\x08" {
        return std::env::var("WT_SESSION").is_ok();
    }
    false
}

/// Decode a Kitty CSI-u sequence to a printable character.
pub fn decode_kitty_printable(data: &[u8]) -> Option<char> {
    let parsed = parse_kitty_sequence(data)?;
    let modifier_bits = parsed.modifiers.bits();
    if modifier_bits != 0 && (modifier_bits & !(Modifiers::SHIFT.bits() | LOCK_MASK)) != 0 {
        return None;
    }
    let cp = if parsed.modifiers.contains(Modifiers::SHIFT) {
        parsed.shifted_codepoint.unwrap_or(parsed.codepoint)
    } else {
        parsed.codepoint
    };
    if cp >= 32 {
        char::from_u32(cp as u32)
    } else {
        None
    }
}

fn decode_modify_other_keys_printable(data: &[u8]) -> Option<char> {
    let parsed = parse_modify_other_keys_sequence(data)?;
    let modifier_bits = parsed.modifiers.bits();
    if modifier_bits != 0 && (modifier_bits & !(Modifiers::SHIFT.bits() | LOCK_MASK)) != 0 {
        return None;
    }
    if parsed.codepoint >= 32 {
        char::from_u32(parsed.codepoint as u32)
    } else {
        None
    }
}

/// Decode a printable key from raw input.
pub fn decode_printable_key(data: &[u8]) -> Option<char> {
    decode_kitty_printable(data).or_else(|| decode_modify_other_keys_printable(data))
}

/// Parse raw input bytes into a KeyId string.
pub fn parse_key(data: &[u8]) -> Option<String> {
    // 1. Try Kitty CSI-u
    if let Some(parsed) = parse_kitty_sequence(data) {
        let normalized = normalize_kitty_functional_codepoint(parsed.codepoint);
        let (norm_cp, norm_mods) = strip_shift_from_letter(normalized, parsed.modifiers);
        return Some(format_key_with_modifiers(
            norm_cp,
            norm_mods,
            parsed.base_layout_codepoint,
        ));
    }

    // 2. Try modifyOtherKeys
    if let Some(parsed) = parse_modify_other_keys_sequence(data) {
        let (norm_cp, norm_mods) = strip_shift_from_letter(parsed.codepoint, parsed.modifiers);
        return Some(format_key_with_modifiers(norm_cp, norm_mods, None));
    }

    // 3. Legacy lookup
    for (seq, key_id) in LEGACY_SEQUENCE_KEY_IDS.iter() {
        if data == seq.as_slice() {
            return Some(key_id.to_string());
        }
    }

    // 4. Raw bytes
    if data.len() == 1 {
        let b = data[0];
        return match b {
            b'\x1b' => Some("escape".to_string()),
            b'\x1c' => Some("ctrl+\\".to_string()),
            b'\x1d' => Some("ctrl+]".to_string()),
            0x01..=0x1f if b == 0x09 => Some("tab".to_string()),
            0x01..=0x1f if b == 0x0d || b == 0x0a => Some("enter".to_string()),
            0x01..=0x1f => {
                let letter = (b'a' + b - 1) as char;
                Some(format!("ctrl+{letter}"))
            }
            0x20..=0x7e if b == 0x20 => Some("space".to_string()),
            0x20..=0x7e => Some((b as char).to_string()),
            0x7f => Some("backspace".to_string()),
            _ => None,
        };
    }

    // 5. Two-byte ESC sequences (alt+key)
    if data.len() == 2 && data[0] == b'\x1b' {
        let b = data[1];
        match b {
            b'\x1b' => return Some("alt+escape".to_string()),
            b'\r' | b'\n' => return Some("alt+enter".to_string()),
            32..=126 => {
                let c = b as char;
                if c.is_ascii_lowercase() || c.is_ascii_digit() {
                    return Some(format!("alt+{c}"));
                }
            }
            _ => {}
        }
    }

    None
}

fn format_key_with_modifiers(
    codepoint: i32,
    modifiers: Modifiers,
    base_layout: Option<i32>,
) -> String {
    let key_name = codepoint_to_key_name(codepoint, base_layout);
    if modifiers.is_empty() {
        key_name
    } else {
        format!("{modifiers}+{key_name}")
    }
}

fn codepoint_to_key_name(codepoint: i32, base_layout: Option<i32>) -> String {
    let cp = normalize_kitty_functional_codepoint(codepoint);
    match cp {
        codepoints::ESCAPE => "escape".to_string(),
        codepoints::ENTER => "enter".to_string(),
        codepoints::TAB => "tab".to_string(),
        codepoints::BACKSPACE => "backspace".to_string(),
        codepoints::UP => "up".to_string(),
        codepoints::DOWN => "down".to_string(),
        codepoints::LEFT => "left".to_string(),
        codepoints::RIGHT => "right".to_string(),
        codepoints::HOME => "home".to_string(),
        codepoints::END => "end".to_string(),
        codepoints::INSERT => "insert".to_string(),
        codepoints::DELETE => "delete".to_string(),
        codepoints::PAGE_UP => "pageUp".to_string(),
        codepoints::PAGE_DOWN => "pageDown".to_string(),
        codepoints::F1 => "f1".to_string(),
        codepoints::F2 => "f2".to_string(),
        codepoints::F3 => "f3".to_string(),
        codepoints::F4 => "f4".to_string(),
        codepoints::F5 => "f5".to_string(),
        codepoints::F6 => "f6".to_string(),
        codepoints::F7 => "f7".to_string(),
        codepoints::F8 => "f8".to_string(),
        codepoints::F9 => "f9".to_string(),
        codepoints::F10 => "f10".to_string(),
        codepoints::F11 => "f11".to_string(),
        codepoints::F12 => "f12".to_string(),
        codepoints::KP_BEGIN => "clear".to_string(),
        32..=126 if cp == codepoints::SPACE => "space".to_string(),
        32..=126 => (cp as u8 as char).to_string(),
        _ => {
            if let Some(base) = base_layout {
                let normalized_base = normalize_kitty_functional_codepoint(base);
                if is_standard_printable(normalized_base) {
                    return (normalized_base as u8 as char).to_string();
                }
            }
            format!("unknown({cp})")
        }
    }
}
