use rpi_tui::keys::*;

// --- Kitty CSI-u matching ---

#[test]
fn kitty_plain_letter() {
    assert!(matches_key(b"\x1b[97u", KeyEvent { key: Key::Char('a'), modifiers: Modifiers::empty() }));
}

#[test]
fn kitty_ctrl_letter() {
    assert!(matches_key(b"\x1b[97;5u", KeyEvent { key: Key::Char('a'), modifiers: Modifiers::CTRL }));
}

#[test]
fn kitty_shift_letter() {
    assert!(matches_key(b"\x1b[65;2u", KeyEvent { key: Key::Char('a'), modifiers: Modifiers::SHIFT }));
}

#[test]
fn kitty_alt_letter() {
    assert!(matches_key(b"\x1b[107;3u", KeyEvent { key: Key::Char('k'), modifiers: Modifiers::ALT }));
}

#[test]
fn kitty_enter() {
    assert!(matches_key(b"\x1b[13u", KeyEvent { key: Key::Special(SpecialKey::Enter), modifiers: Modifiers::empty() }));
}

#[test]
fn kitty_shift_enter() {
    assert!(matches_key(b"\x1b[13;2u", KeyEvent { key: Key::Special(SpecialKey::Enter), modifiers: Modifiers::SHIFT }));
}

#[test]
fn kitty_tab() {
    assert!(matches_key(b"\x1b[9u", KeyEvent { key: Key::Special(SpecialKey::Tab), modifiers: Modifiers::empty() }));
}

#[test]
fn kitty_escape() {
    assert!(matches_key(b"\x1b[27u", KeyEvent { key: Key::Special(SpecialKey::Escape), modifiers: Modifiers::empty() }));
}

#[test]
fn kitty_backspace() {
    assert!(matches_key(b"\x1b[127u", KeyEvent { key: Key::Special(SpecialKey::Backspace), modifiers: Modifiers::empty() }));
}

#[test]
fn kitty_arrow_up() {
    assert!(matches_key(b"\x1b[1;A", KeyEvent { key: Key::Special(SpecialKey::Up), modifiers: Modifiers::empty() }));
}

#[test]
fn kitty_ctrl_arrow_up() {
    assert!(matches_key(b"\x1b[1;5A", KeyEvent { key: Key::Special(SpecialKey::Up), modifiers: Modifiers::CTRL }));
}

#[test]
fn kitty_shift_arrow_right() {
    assert!(matches_key(b"\x1b[1;2C", KeyEvent { key: Key::Special(SpecialKey::Right), modifiers: Modifiers::SHIFT }));
}

#[test]
fn kitty_home() {
    assert!(matches_key(b"\x1b[1;H", KeyEvent { key: Key::Special(SpecialKey::Home), modifiers: Modifiers::empty() }));
}

#[test]
fn kitty_end() {
    assert!(matches_key(b"\x1b[1;F", KeyEvent { key: Key::Special(SpecialKey::End), modifiers: Modifiers::empty() }));
}

#[test]
fn kitty_delete() {
    assert!(matches_key(b"\x1b[3~", KeyEvent { key: Key::Special(SpecialKey::Delete), modifiers: Modifiers::empty() }));
}

#[test]
fn kitty_insert() {
    assert!(matches_key(b"\x1b[2~", KeyEvent { key: Key::Special(SpecialKey::Insert), modifiers: Modifiers::empty() }));
}

#[test]
fn kitty_page_up() {
    assert!(matches_key(b"\x1b[5~", KeyEvent { key: Key::Special(SpecialKey::PageUp), modifiers: Modifiers::empty() }));
}

#[test]
fn kitty_page_down() {
    assert!(matches_key(b"\x1b[6~", KeyEvent { key: Key::Special(SpecialKey::PageDown), modifiers: Modifiers::empty() }));
}

// --- modifyOtherKeys ---

#[test]
fn modify_other_keys_ctrl_c() {
    assert!(matches_key(b"\x1b[27;5;99~", KeyEvent { key: Key::Char('c'), modifiers: Modifiers::CTRL }));
}

#[test]
fn modify_other_keys_shift_a() {
    assert!(matches_key(b"\x1b[27;2;65~", KeyEvent { key: Key::Char('a'), modifiers: Modifiers::SHIFT }));
}

// --- Legacy matching ---

#[test]
fn legacy_arrow_up() {
    assert!(matches_key(b"\x1bOA", KeyEvent { key: Key::Special(SpecialKey::Up), modifiers: Modifiers::empty() }));
    assert!(matches_key(b"\x1b[A", KeyEvent { key: Key::Special(SpecialKey::Up), modifiers: Modifiers::empty() }));
}

#[test]
fn legacy_shift_arrow_up() {
    assert!(matches_key(b"\x1b[1;2A", KeyEvent { key: Key::Special(SpecialKey::Up), modifiers: Modifiers::SHIFT }));
}

#[test]
fn legacy_ctrl_arrow_left() {
    assert!(matches_key(b"\x1b[1;5D", KeyEvent { key: Key::Special(SpecialKey::Left), modifiers: Modifiers::CTRL }));
}

#[test]
fn legacy_function_keys() {
    assert!(matches_key(b"\x1bOP", KeyEvent { key: Key::Special(SpecialKey::F1), modifiers: Modifiers::empty() }));
    assert!(matches_key(b"\x1bOQ", KeyEvent { key: Key::Special(SpecialKey::F2), modifiers: Modifiers::empty() }));
    assert!(matches_key(b"\x1b[15~", KeyEvent { key: Key::Special(SpecialKey::F5), modifiers: Modifiers::empty() }));
    assert!(matches_key(b"\x1b[24~", KeyEvent { key: Key::Special(SpecialKey::F12), modifiers: Modifiers::empty() }));
}

#[test]
fn legacy_home_end() {
    assert!(matches_key(b"\x1bOH", KeyEvent { key: Key::Special(SpecialKey::Home), modifiers: Modifiers::empty() }));
    assert!(matches_key(b"\x1bOF", KeyEvent { key: Key::Special(SpecialKey::End), modifiers: Modifiers::empty() }));
    assert!(matches_key(b"\x1b[H", KeyEvent { key: Key::Special(SpecialKey::Home), modifiers: Modifiers::empty() }));
    assert!(matches_key(b"\x1b[F", KeyEvent { key: Key::Special(SpecialKey::End), modifiers: Modifiers::empty() }));
}

#[test]
fn legacy_shift_tab() {
    assert!(matches_key(b"\x1b[Z", KeyEvent { key: Key::Special(SpecialKey::Tab), modifiers: Modifiers::SHIFT }));
}

#[test]
fn legacy_numpad_enter() {
    assert!(matches_key(b"\x1bOM", KeyEvent { key: Key::Special(SpecialKey::Enter), modifiers: Modifiers::empty() }));
}

// --- Printable characters ---

#[test]
fn plain_character() {
    assert!(matches_key(b"a", KeyEvent { key: Key::Char('a'), modifiers: Modifiers::empty() }));
    assert!(matches_key(b"5", KeyEvent { key: Key::Char('5'), modifiers: Modifiers::empty() }));
    assert!(matches_key(b"!", KeyEvent { key: Key::Char('!'), modifiers: Modifiers::empty() }));
}

#[test]
fn ctrl_letter() {
    assert!(matches_key(b"\x01", KeyEvent { key: Key::Char('a'), modifiers: Modifiers::CTRL }));
    assert!(matches_key(b"\x03", KeyEvent { key: Key::Char('c'), modifiers: Modifiers::CTRL }));
    assert!(matches_key(b"\x1a", KeyEvent { key: Key::Char('z'), modifiers: Modifiers::CTRL }));
}

#[test]
fn ctrl_bracket() {
    assert!(matches_key(b"\x1b", KeyEvent { key: Key::Char('['), modifiers: Modifiers::CTRL }));
}

#[test]
fn ctrl_backslash() {
    assert!(matches_key(b"\x1c", KeyEvent { key: Key::Char('\\'), modifiers: Modifiers::CTRL }));
}

#[test]
fn ctrl_right_bracket() {
    assert!(matches_key(b"\x1d", KeyEvent { key: Key::Char(']'), modifiers: Modifiers::CTRL }));
}

#[test]
fn alt_character() {
    assert!(matches_key(b"\x1ba", KeyEvent { key: Key::Char('a'), modifiers: Modifiers::ALT }));
    assert!(matches_key(b"\x1bz", KeyEvent { key: Key::Char('z'), modifiers: Modifiers::ALT }));
}

#[test]
fn ctrl_space() {
    assert!(matches_key(b"\x00", KeyEvent { key: Key::Special(SpecialKey::Space), modifiers: Modifiers::CTRL }));
}

#[test]
fn alt_space() {
    assert!(matches_key(b"\x1b ", KeyEvent { key: Key::Special(SpecialKey::Space), modifiers: Modifiers::ALT }));
}

// --- Negative tests ---

#[test]
fn wrong_modifier_no_match() {
    assert!(!matches_key(b"a", KeyEvent { key: Key::Char('a'), modifiers: Modifiers::CTRL }));
    assert!(!matches_key(b"\x01", KeyEvent { key: Key::Char('a'), modifiers: Modifiers::empty() }));
}

#[test]
fn wrong_key_no_match() {
    assert!(!matches_key(b"b", KeyEvent { key: Key::Char('a'), modifiers: Modifiers::empty() }));
    assert!(!matches_key(b"\x1b[A", KeyEvent { key: Key::Special(SpecialKey::Down), modifiers: Modifiers::empty() }));
}

// --- parse_key ---

#[test]
fn parse_key_kitty_letter() {
    assert_eq!(parse_key(b"\x1b[97u").as_deref(), Some("a"));
}

#[test]
fn parse_key_kitty_ctrl_letter() {
    assert_eq!(parse_key(b"\x1b[97;5u").as_deref(), Some("ctrl+a"));
}

#[test]
fn parse_key_kitty_arrow() {
    assert_eq!(parse_key(b"\x1b[1;A").as_deref(), Some("up"));
    assert_eq!(parse_key(b"\x1b[1;5D").as_deref(), Some("ctrl+left"));
}

#[test]
fn parse_key_legacy_arrow() {
    assert_eq!(parse_key(b"\x1bOA").as_deref(), Some("up"));
    assert_eq!(parse_key(b"\x1b[A").as_deref(), Some("up"));
}

#[test]
fn parse_key_legacy_function() {
    assert_eq!(parse_key(b"\x1bOP").as_deref(), Some("f1"));
    assert_eq!(parse_key(b"\x1b[24~").as_deref(), Some("f12"));
}

#[test]
fn parse_key_plain_char() {
    assert_eq!(parse_key(b"a").as_deref(), Some("a"));
    assert_eq!(parse_key(b"5").as_deref(), Some("5"));
}

#[test]
fn parse_key_escape() {
    assert_eq!(parse_key(b"\x1b").as_deref(), Some("escape"));
}

#[test]
fn parse_key_ctrl_letter() {
    assert_eq!(parse_key(b"\x01").as_deref(), Some("ctrl+a"));
    assert_eq!(parse_key(b"\x03").as_deref(), Some("ctrl+c"));
}

#[test]
fn parse_key_alt_letter() {
    assert_eq!(parse_key(b"\x1ba").as_deref(), Some("alt+a"));
}

#[test]
fn parse_key_enter() {
    assert_eq!(parse_key(b"\r").as_deref(), Some("enter"));
    assert_eq!(parse_key(b"\n").as_deref(), Some("enter"));
}

#[test]
fn parse_key_tab() {
    assert_eq!(parse_key(b"\t").as_deref(), Some("tab"));
}

#[test]
fn parse_key_space() {
    assert_eq!(parse_key(b" ").as_deref(), Some("space"));
}

#[test]
fn parse_key_backspace() {
    assert_eq!(parse_key(b"\x7f").as_deref(), Some("backspace"));
}

#[test]
fn parse_key_shift_tab() {
    assert_eq!(parse_key(b"\x1b[Z").as_deref(), Some("shift+tab"));
}

// --- decode_kitty_printable ---

#[test]
fn decode_kitty_printable_plain() {
    assert_eq!(decode_kitty_printable(b"\x1b[97u"), Some('a'));
}

#[test]
fn decode_kitty_printable_shift() {
    assert_eq!(decode_kitty_printable(b"\x1b[97:65;2u"), Some('A'));
}

#[test]
fn decode_kitty_printable_rejects_ctrl() {
    assert_eq!(decode_kitty_printable(b"\x1b[97;5u"), None);
}

#[test]
fn decode_kitty_printable_rejects_alt() {
    assert_eq!(decode_kitty_printable(b"\x1b[97;3u"), None);
}

// --- parse_key_id ---

#[test]
fn parse_key_id_simple() {
    let evt = parse_key_id("a").unwrap();
    assert_eq!(evt.key, Key::Char('a'));
    assert_eq!(evt.modifiers, Modifiers::empty());
}

#[test]
fn parse_key_id_with_modifiers() {
    let evt = parse_key_id("ctrl+shift+k").unwrap();
    assert_eq!(evt.key, Key::Char('k'));
    assert_eq!(evt.modifiers, Modifiers::CTRL | Modifiers::SHIFT);
}

#[test]
fn parse_key_id_special_key() {
    let evt = parse_key_id("ctrl+enter").unwrap();
    assert_eq!(evt.key, Key::Special(SpecialKey::Enter));
    assert_eq!(evt.modifiers, Modifiers::CTRL);
}

#[test]
fn parse_key_id_invalid() {
    assert!(parse_key_id("").is_none());
    assert!(parse_key_id("ctrl+").is_none());
    assert!(parse_key_id("foo").is_none());
}

// --- KeyEventType ---

#[test]
fn kitty_key_release() {
    assert!(is_key_release(b"\x1b[97;1:3u"));
    assert!(!is_key_release(b"\x1b[97u"));
}

#[test]
fn kitty_key_repeat() {
    assert!(is_key_repeat(b"\x1b[97;1:2u"));
    assert!(!is_key_repeat(b"\x1b[97u"));
}

// --- Kitty tilde variants for F-keys and Home/End ---

#[test]
fn kitty_tilde_f1() {
    assert!(matches_key(b"\x1b[11~", KeyEvent { key: Key::Special(SpecialKey::F1), modifiers: Modifiers::empty() }));
}

#[test]
fn kitty_tilde_f2() {
    assert!(matches_key(b"\x1b[12~", KeyEvent { key: Key::Special(SpecialKey::F2), modifiers: Modifiers::empty() }));
}

#[test]
fn kitty_tilde_f3() {
    assert!(matches_key(b"\x1b[13~", KeyEvent { key: Key::Special(SpecialKey::F3), modifiers: Modifiers::empty() }));
}

#[test]
fn kitty_tilde_f4() {
    assert!(matches_key(b"\x1b[14~", KeyEvent { key: Key::Special(SpecialKey::F4), modifiers: Modifiers::empty() }));
}

#[test]
fn kitty_tilde_f5() {
    assert!(matches_key(b"\x1b[15~", KeyEvent { key: Key::Special(SpecialKey::F5), modifiers: Modifiers::empty() }));
}

#[test]
fn kitty_tilde_f6() {
    assert!(matches_key(b"\x1b[17~", KeyEvent { key: Key::Special(SpecialKey::F6), modifiers: Modifiers::empty() }));
}

#[test]
fn kitty_tilde_f7() {
    assert!(matches_key(b"\x1b[18~", KeyEvent { key: Key::Special(SpecialKey::F7), modifiers: Modifiers::empty() }));
}

#[test]
fn kitty_tilde_f8() {
    assert!(matches_key(b"\x1b[19~", KeyEvent { key: Key::Special(SpecialKey::F8), modifiers: Modifiers::empty() }));
}

#[test]
fn kitty_tilde_f9() {
    assert!(matches_key(b"\x1b[20~", KeyEvent { key: Key::Special(SpecialKey::F9), modifiers: Modifiers::empty() }));
}

#[test]
fn kitty_tilde_f10() {
    assert!(matches_key(b"\x1b[21~", KeyEvent { key: Key::Special(SpecialKey::F10), modifiers: Modifiers::empty() }));
}

#[test]
fn kitty_tilde_f11() {
    assert!(matches_key(b"\x1b[23~", KeyEvent { key: Key::Special(SpecialKey::F11), modifiers: Modifiers::empty() }));
}

#[test]
fn kitty_tilde_f12() {
    assert!(matches_key(b"\x1b[24~", KeyEvent { key: Key::Special(SpecialKey::F12), modifiers: Modifiers::empty() }));
}

#[test]
fn kitty_tilde_home() {
    assert!(matches_key(b"\x1b[7~", KeyEvent { key: Key::Special(SpecialKey::Home), modifiers: Modifiers::empty() }));
}

#[test]
fn kitty_tilde_end() {
    assert!(matches_key(b"\x1b[8~", KeyEvent { key: Key::Special(SpecialKey::End), modifiers: Modifiers::empty() }));
}

#[test]
fn kitty_tilde_shift_f1() {
    assert!(matches_key(b"\x1b[11;2~", KeyEvent { key: Key::Special(SpecialKey::F1), modifiers: Modifiers::SHIFT }));
}

#[test]
fn kitty_tilde_ctrl_f5() {
    assert!(matches_key(b"\x1b[15;5~", KeyEvent { key: Key::Special(SpecialKey::F5), modifiers: Modifiers::CTRL }));
}
