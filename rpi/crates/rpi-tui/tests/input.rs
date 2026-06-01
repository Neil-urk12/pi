// Tests for the single-line text Input component.
//
// Key mapping from TS keybindings to Rust KeyEvent:
//   Ctrl+A  -> cursor_line_start
//   Ctrl+E  -> cursor_line_end
//   Ctrl+B  -> cursor_left (alt: Left arrow)
//   Ctrl+F  -> cursor_right (alt: Right arrow)
//   Alt+B   -> cursor_word_left
//   Alt+F   -> cursor_word_right
//   Ctrl+H / Backspace -> delete_char_backward
//   Ctrl+D / Delete    -> delete_char_forward
//   Ctrl+W             -> delete_word_backward
//   Alt+D              -> delete_word_forward
//   Ctrl+U             -> delete_to_line_start
//   Ctrl+K             -> delete_to_line_end
//   Ctrl+Y             -> yank
//   Alt+Y              -> yank_pop
//   Ctrl+-             -> undo
//   Enter              -> submit
//   Escape             -> cancel

use rpi_tui::input::Input;
use rpi_tui::keys::{Key, KeyEvent, Modifiers, SpecialKey};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn char_key(c: char) -> KeyEvent {
    KeyEvent {
        key: Key::Char(c),
        modifiers: Modifiers::empty(),
    }
}

fn ctrl(c: char) -> KeyEvent {
    KeyEvent {
        key: Key::Char(c),
        modifiers: Modifiers::CTRL,
    }
}

fn alt(c: char) -> KeyEvent {
    KeyEvent {
        key: Key::Char(c),
        modifiers: Modifiers::ALT,
    }
}

fn special(key: SpecialKey) -> KeyEvent {
    KeyEvent {
        key: Key::Special(key),
        modifiers: Modifiers::empty(),
    }
}

fn type_str(input: &mut Input, s: &str) {
    for c in s.chars() {
        input.handle_key(char_key(c));
    }
}

// ---------------------------------------------------------------------------
// Construction & basic state
// ---------------------------------------------------------------------------

#[test]
fn new_creates_empty_input() {
    let input = Input::new();
    assert_eq!(input.text(), "");
    assert_eq!(input.cursor(), 0);
}
#[test]
fn set_value() {
    let mut input = Input::new();
    input.set_value("hello world");
    assert_eq!(input.text(), "hello world");
    // Cursor clamps to value length
    assert_eq!(input.cursor(), 11);
}

#[test]
fn set_value_clamps_cursor() {
    let mut input = Input::new();
    input.set_value("abc");
    input.handle_key(ctrl('a')); // cursor to 0
    input.set_value("a"); // shorter value
    assert_eq!(input.cursor(), 1); // clamped to new len
}

#[test]
fn set_value_clamps_cursor_from_high_position() {
    let mut input = Input::new();
    type_str(&mut input, "hello"); // cursor at 5
    input.set_value("hi"); // shorter value
    assert_eq!(input.cursor(), 2); // clamped to end of new value
}

// ---------------------------------------------------------------------------
// Character insertion
// ---------------------------------------------------------------------------

#[test]
fn basic_insertion() {
    let mut input = Input::new();
    type_str(&mut input, "hello");
    assert_eq!(input.text(), "hello");
    assert_eq!(input.cursor(), 5);
}

#[test]
fn insert_at_cursor_middle() {
    let mut input = Input::new();
    type_str(&mut input, "helo");
    input.handle_key(ctrl('a')); // cursor -> 0
    input.handle_key(special(SpecialKey::Right));
    input.handle_key(special(SpecialKey::Right)); // cursor after "he"
    input.handle_key(char_key('l'));
    assert_eq!(input.text(), "hello");
    assert_eq!(input.cursor(), 3);
}

#[test]
fn insert_unicode() {
    let mut input = Input::new();
    type_str(&mut input, "cafe\u{0301}"); // cafe with combining accent
    // Cursor should advance past the combining character
    assert!(input.cursor() > 4);
}

// ---------------------------------------------------------------------------
// Cursor movement
// ---------------------------------------------------------------------------

#[test]
fn cursor_left_from_start_is_noop() {
    let mut input = Input::new();
    type_str(&mut input, "abc");
    input.handle_key(ctrl('a')); // go to start
    assert_eq!(input.cursor(), 0);
    input.handle_key(special(SpecialKey::Left));
    assert_eq!(input.cursor(), 0);
}

#[test]
fn cursor_right_from_end_is_noop() {
    let mut input = Input::new();
    type_str(&mut input, "abc");
    assert_eq!(input.cursor(), 3);
    input.handle_key(special(SpecialKey::Right));
    assert_eq!(input.cursor(), 3);
}

#[test]
fn cursor_left_right() {
    let mut input = Input::new();
    type_str(&mut input, "abc");
    input.handle_key(special(SpecialKey::Left));
    assert_eq!(input.cursor(), 2);
    input.handle_key(special(SpecialKey::Left));
    assert_eq!(input.cursor(), 1);
    input.handle_key(special(SpecialKey::Right));
    assert_eq!(input.cursor(), 2);
}

#[test]
fn cursor_home_ctrl_a() {
    let mut input = Input::new();
    type_str(&mut input, "hello world");
    assert_eq!(input.cursor(), 11);
    input.handle_key(ctrl('a'));
    assert_eq!(input.cursor(), 0);
}

#[test]
fn cursor_end_ctrl_e() {
    let mut input = Input::new();
    type_str(&mut input, "hello");
    input.handle_key(ctrl('a'));
    assert_eq!(input.cursor(), 0);
    input.handle_key(ctrl('e'));
    assert_eq!(input.cursor(), 5);
}

#[test]
fn cursor_word_left() {
    let mut input = Input::new();
    type_str(&mut input, "foo bar baz");
    // cursor at end (11)
    input.handle_key(alt('b'));
    // should jump before "baz"
    assert_eq!(input.cursor(), 8);
    input.handle_key(alt('b'));
    // should jump before "bar"
    assert_eq!(input.cursor(), 4);
    input.handle_key(alt('b'));
    // should jump before "foo"
    assert_eq!(input.cursor(), 0);
}

#[test]
fn cursor_word_right() {
    let mut input = Input::new();
    type_str(&mut input, "foo bar baz");
    input.handle_key(ctrl('a')); // cursor -> 0
    input.handle_key(alt('f'));
    // should jump after "foo"
    assert_eq!(input.cursor(), 3);
    input.handle_key(alt('f'));
    // should jump after "bar"
    assert_eq!(input.cursor(), 7);
    input.handle_key(alt('f'));
    // should jump after "baz"
    assert_eq!(input.cursor(), 11);
}

#[test]
fn cursor_word_left_skips_whitespace() {
    let mut input = Input::new();
    type_str(&mut input, "foo   bar");
    // cursor at end
    input.handle_key(alt('b'));
    // should land before "bar", skipping spaces
    assert_eq!(input.cursor(), 6);
}

#[test]
fn cursor_word_left_stops_at_punctuation() {
    let mut input = Input::new();
    type_str(&mut input, "foo,bar");
    // cursor at end (7)
    input.handle_key(alt('b'));
    // should stop before "bar" (punctuation is boundary)
    assert_eq!(input.cursor(), 4);
}

// ---------------------------------------------------------------------------
// Deletion
// ---------------------------------------------------------------------------

#[test]
fn backspace_deletes_previous_char() {
    let mut input = Input::new();
    type_str(&mut input, "hello");
    input.handle_key(special(SpecialKey::Backspace));
    assert_eq!(input.text(), "hell");
    assert_eq!(input.cursor(), 4);
}

#[test]
fn backspace_at_start_is_noop() {
    let mut input = Input::new();
    type_str(&mut input, "abc");
    input.handle_key(ctrl('a'));
    input.handle_key(special(SpecialKey::Backspace));
    assert_eq!(input.text(), "abc");
    assert_eq!(input.cursor(), 0);
}

#[test]
fn forward_delete() {
    let mut input = Input::new();
    type_str(&mut input, "hello");
    input.handle_key(ctrl('a'));
    input.handle_key(special(SpecialKey::Delete));
    assert_eq!(input.text(), "ello");
    assert_eq!(input.cursor(), 0);
}

#[test]
fn forward_delete_at_end_is_noop() {
    let mut input = Input::new();
    type_str(&mut input, "abc");
    input.handle_key(special(SpecialKey::Delete));
    assert_eq!(input.text(), "abc");
    assert_eq!(input.cursor(), 3);
}

#[test]
fn delete_word_backward() {
    let mut input = Input::new();
    type_str(&mut input, "foo bar baz");
    input.handle_key(ctrl('w'));
    // deletes "baz", trailing whitespace before it
    assert_eq!(input.text(), "foo bar ");
    // do it again
    input.handle_key(ctrl('w'));
    assert_eq!(input.text(), "foo ");
}

#[test]
fn delete_word_backward_at_start_is_noop() {
    let mut input = Input::new();
    type_str(&mut input, "abc");
    input.handle_key(ctrl('a'));
    input.handle_key(ctrl('w'));
    assert_eq!(input.text(), "abc");
}

#[test]
fn delete_word_forward() {
    let mut input = Input::new();
    type_str(&mut input, "foo bar baz");
    input.handle_key(ctrl('a'));
    input.handle_key(alt('d'));
    // deletes "foo" (word forward from start)
    assert_eq!(input.text(), " bar baz");
}

#[test]
fn delete_word_forward_at_end_is_noop() {
    let mut input = Input::new();
    type_str(&mut input, "abc");
    input.handle_key(alt('d'));
    assert_eq!(input.text(), "abc");
}

#[test]
fn delete_to_line_start() {
    let mut input = Input::new();
    type_str(&mut input, "hello world");
    // move cursor to after "hello "
    input.handle_key(ctrl('a'));
    for _ in 0..6 {
        input.handle_key(special(SpecialKey::Right));
    }
    input.handle_key(ctrl('u'));
    assert_eq!(input.text(), "world");
    assert_eq!(input.cursor(), 0);
}

#[test]
fn delete_to_line_start_at_start_is_noop() {
    let mut input = Input::new();
    type_str(&mut input, "abc");
    input.handle_key(ctrl('a'));
    input.handle_key(ctrl('u'));
    assert_eq!(input.text(), "abc");
}

#[test]
fn delete_to_line_end() {
    let mut input = Input::new();
    type_str(&mut input, "hello world");
    input.handle_key(ctrl('a'));
    for _ in 0..6 {
        input.handle_key(special(SpecialKey::Right));
    }
    input.handle_key(ctrl('k'));
    assert_eq!(input.text(), "hello ");
    assert_eq!(input.cursor(), 6);
}

#[test]
fn delete_to_line_end_at_end_is_noop() {
    let mut input = Input::new();
    type_str(&mut input, "abc");
    input.handle_key(ctrl('k'));
    assert_eq!(input.text(), "abc");
}

// ---------------------------------------------------------------------------
// Kill ring & yank
// ---------------------------------------------------------------------------

#[test]
fn yank_from_kill_ring() {
    let mut input = Input::new();
    type_str(&mut input, "foo bar baz");
    input.handle_key(ctrl('e'));
    input.handle_key(ctrl('w')); // kills "baz" (or trailing word)
    let text_after_kill = input.text().to_string();

    input.handle_key(ctrl('a'));
    input.handle_key(ctrl('y'));
    // yanked text inserted at start
    assert!(input.text().len() > text_after_kill.len());
}

#[test]
fn yank_empty_kill_ring_is_noop() {
    let mut input = Input::new();
    type_str(&mut input, "test");
    input.handle_key(ctrl('e'));
    input.handle_key(ctrl('y'));
    assert_eq!(input.text(), "test");
}

#[test]
fn consecutive_kills_accumulate() {
    let mut input = Input::new();
    type_str(&mut input, "one two three");
    input.handle_key(ctrl('e'));
    input.handle_key(ctrl('w')); // kill "three"
    input.handle_key(ctrl('w')); // kill "two " (accumulated)
    input.handle_key(ctrl('w')); // kill "one " (accumulated)
    assert_eq!(input.text(), "");
    // yank should restore all
    input.handle_key(ctrl('y'));
    assert_eq!(input.text(), "one two three");
}

#[test]
fn typing_breaks_kill_accumulation() {
    let mut input = Input::new();
    type_str(&mut input, "foo bar");
    input.handle_key(ctrl('e'));
    input.handle_key(ctrl('w')); // kill "bar"
    input.handle_key(char_key('x')); // breaks kill chain
    input.handle_key(ctrl('w')); // kills "x" separately
    // yank gets most recent kill ("x")
    input.handle_key(ctrl('a'));
    input.handle_key(ctrl('y'));
    assert!(input.text().contains('x'));
}

// ---------------------------------------------------------------------------
// Undo
// ---------------------------------------------------------------------------

#[test]
fn undo_on_empty_stack_is_noop() {
    let mut input = Input::new();
    input.handle_key(ctrl('-'));
    assert_eq!(input.text(), "");
}

#[test]
fn undo_insertion() {
    let mut input = Input::new();
    type_str(&mut input, "hello");
    input.handle_key(ctrl('-'));
    assert_eq!(input.text(), "");
}

#[test]
fn undo_coalesces_word_chars() {
    let mut input = Input::new();
    type_str(&mut input, "hello world");
    // undo removes "world" (the word unit after the space)
    input.handle_key(ctrl('-'));
    assert_eq!(input.text(), "hello");
    // undo removes "hello" (the first word unit)
    input.handle_key(ctrl('-'));
    assert_eq!(input.text(), "");
}

#[test]
fn undo_spaces_one_at_a_time() {
    let mut input = Input::new();
    type_str(&mut input, "ab  ");
    // undo removes second space
    input.handle_key(ctrl('-'));
    assert_eq!(input.text(), "ab ");
    // undo removes first space
    input.handle_key(ctrl('-'));
    assert_eq!(input.text(), "ab");
}

#[test]
fn undo_backspace() {
    let mut input = Input::new();
    type_str(&mut input, "hello");
    input.handle_key(special(SpecialKey::Backspace));
    assert_eq!(input.text(), "hell");
    input.handle_key(ctrl('-'));
    assert_eq!(input.text(), "hello");
}

#[test]
fn undo_forward_delete() {
    let mut input = Input::new();
    type_str(&mut input, "hello");
    input.handle_key(ctrl('a'));
    input.handle_key(special(SpecialKey::Right));
    input.handle_key(special(SpecialKey::Delete));
    assert_eq!(input.text(), "hllo");
    input.handle_key(ctrl('-'));
    assert_eq!(input.text(), "hello");
}

#[test]
fn undo_delete_word_backward() {
    let mut input = Input::new();
    type_str(&mut input, "foo bar baz");
    input.handle_key(ctrl('e'));
    input.handle_key(ctrl('w'));
    let after = input.text().to_string();
    assert_ne!(after, "foo bar baz");
    input.handle_key(ctrl('-'));
    assert_eq!(input.text(), "foo bar baz");
}

#[test]
fn undo_delete_to_line_end() {
    let mut input = Input::new();
    type_str(&mut input, "hello world");
    input.handle_key(ctrl('a'));
    for _ in 0..6 {
        input.handle_key(special(SpecialKey::Right));
    }
    input.handle_key(ctrl('k'));
    assert_eq!(input.text(), "hello ");
    input.handle_key(ctrl('-'));
    assert_eq!(input.text(), "hello world");
}

#[test]
fn undo_delete_to_line_start() {
    let mut input = Input::new();
    type_str(&mut input, "hello world");
    input.handle_key(ctrl('a'));
    for _ in 0..6 {
        input.handle_key(special(SpecialKey::Right));
    }
    input.handle_key(ctrl('u'));
    assert_eq!(input.text(), "world");
    input.handle_key(ctrl('-'));
    assert_eq!(input.text(), "hello world");
}

#[test]
fn undo_yank() {
    let mut input = Input::new();
    type_str(&mut input, "foo");
    input.handle_key(ctrl('e'));
    input.handle_key(ctrl('w')); // kill "foo"
    input.handle_key(ctrl('y')); // yank "foo" back
    assert_eq!(input.text(), "foo");
    input.handle_key(ctrl('-')); // undo yank
    assert_eq!(input.text(), "");
}

#[test]
fn undo_multiple_operations() {
    let mut input = Input::new();
    type_str(&mut input, "aaa");
    input.handle_key(special(SpecialKey::Backspace)); // "aa"
    type_str(&mut input, "bbb"); // "aabbb"

    input.handle_key(ctrl('-')); // undo "bbb"
    assert_eq!(input.text(), "aa");

    input.handle_key(ctrl('-')); // undo backspace -> "aaa"
    assert_eq!(input.text(), "aaa");

    input.handle_key(ctrl('-')); // undo "aaa"
    assert_eq!(input.text(), "");
}

#[test]
fn cursor_movement_starts_new_undo_unit() {
    let mut input = Input::new();
    type_str(&mut input, "abc");
    input.handle_key(ctrl('a')); // move -> new unit boundary
    input.handle_key(ctrl('e')); // move -> another boundary
    type_str(&mut input, "de");

    input.handle_key(ctrl('-')); // undo "de"
    assert_eq!(input.text(), "abc");

    input.handle_key(ctrl('-')); // undo "abc"
    assert_eq!(input.text(), "");
}

// ---------------------------------------------------------------------------
// Submit / Escape callbacks
// ---------------------------------------------------------------------------

#[test]
fn submit_on_enter() {
    let mut input = Input::new();
    type_str(&mut input, "hello");

    use std::cell::RefCell;
    use std::rc::Rc;
    let submitted: Rc<RefCell<Option<String>>> = Rc::new(RefCell::new(None));
    let submitted_clone = submitted.clone();
    input.on_submit(move |value| {
        *submitted_clone.borrow_mut() = Some(value.to_string());
    });

    input.handle_key(special(SpecialKey::Enter));
    assert_eq!(submitted.borrow().as_deref(), Some("hello"));
}

#[test]
fn escape_calls_callback() {
    let mut input = Input::new();
    type_str(&mut input, "hello");

    use std::cell::Cell;
    use std::rc::Rc;
    let escaped = Rc::new(Cell::new(false));
    let escaped_clone = escaped.clone();
    input.on_escape(move || {
        escaped_clone.set(true);
    });

    input.handle_key(special(SpecialKey::Escape));
    assert!(escaped.get());
    // Value unchanged
    assert_eq!(input.text(), "hello");
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
fn empty_input_operations() {
    let mut input = Input::new();
    // All deletions/movements should be noops
    input.handle_key(special(SpecialKey::Backspace));
    input.handle_key(special(SpecialKey::Delete));
    input.handle_key(special(SpecialKey::Left));
    input.handle_key(special(SpecialKey::Right));
    input.handle_key(ctrl('a'));
    input.handle_key(ctrl('e'));
    input.handle_key(ctrl('w'));
    input.handle_key(ctrl('u'));
    input.handle_key(ctrl('k'));
    input.handle_key(alt('b'));
    input.handle_key(alt('f'));
    assert_eq!(input.text(), "");
    assert_eq!(input.cursor(), 0);
}

#[test]
fn single_char_input() {
    let mut input = Input::new();
    input.handle_key(char_key('x'));
    assert_eq!(input.text(), "x");
    assert_eq!(input.cursor(), 1);
    input.handle_key(special(SpecialKey::Backspace));
    assert_eq!(input.text(), "");
    assert_eq!(input.cursor(), 0);
}

#[test]
fn set_value_preserves_cursor_within_bounds() {
    let mut input = Input::new();
    type_str(&mut input, "hello");
    input.handle_key(ctrl('a')); // cursor at 0
    input.set_value("hi");
    // set_value sets cursor to end of new value
    assert_eq!(input.cursor(), 2);
}

#[test]
fn set_value_overwrites_completely() {
    let mut input = Input::new();
    type_str(&mut input, "old text");
    input.set_value("new text");
    assert_eq!(input.text(), "new text");
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

#[test]
fn render_returns_single_line() {
    let mut input = Input::new();
    type_str(&mut input, "hello");
    let frame = input.render(40);
    assert_eq!(frame.lines().len(), 1);
}

#[test]
fn render_empty_input() {
    let input = Input::new();
    let frame = input.render(40);
    assert_eq!(frame.lines().len(), 1);
    // Should contain the prompt "> " at minimum
    let text = frame.lines()[0].to_plain_text();
    assert!(text.starts_with("> "));
}

#[test]
fn render_width_does_not_exceed_given_width() {
    let mut input = Input::new();
    type_str(&mut input, "this is a very long input string that should be scrolled");
    let width = 20;
    let frame = input.render(width);
    assert_eq!(frame.lines().len(), 1);
    let text = frame.lines()[0].to_plain_text();
    // Visible width must not exceed the requested width
    // (unicode_width::UnicodeWidthStr can be used to check)
    assert!(
        unicode_width::UnicodeWidthStr::width(text.as_str()) <= width,
        "rendered line exceeded width {}: {:?}",
        width,
        text
    );
}

#[test]
fn render_wide_chars_do_not_overflow() {
    let mut input = Input::new();
    // CJK characters are double-width
    let text = "\u{ac00}\u{b098}\u{b2e4}\u{b77c}\u{b9c8}\u{bc14}\u{c0ac}\u{c544}\u{c790}\u{cc28}\u{ce74}\u{d0c0}\u{d30c}\u{d558}";
    input.set_value(text);
    input.handle_key(ctrl('a'));
    for _ in 0..5 {
        input.handle_key(special(SpecialKey::Right));
    }
    let width = 20;
    let frame = input.render(width);
    let rendered = frame.lines()[0].to_plain_text();
    assert!(
        unicode_width::UnicodeWidthStr::width(rendered.as_str()) <= width,
        "wide text overflowed: width={}",
        unicode_width::UnicodeWidthStr::width(rendered.as_str())
    );
}

// ---------------------------------------------------------------------------
// Kill-tracking guard alignment
// ---------------------------------------------------------------------------

#[test]
fn empty_kill_does_not_set_kill_state() {
    let mut input = Input::new();

    // Ctrl+W on empty text — should NOT set kill state
    input.handle_key(ctrl('w'));

    // Type new text and kill it
    type_str(&mut input, "fresh");
    input.handle_key(ctrl('e'));
    input.handle_key(ctrl('w')); // kills "fresh"

    // Yank — should get only "fresh", not anything stale
    input.handle_key(ctrl('a'));
    input.handle_key(ctrl('y'));
    assert_eq!(input.text(), "fresh");
}

#[test]
fn empty_alt_d_does_not_set_kill_state() {
    let mut input = Input::new();

    // Alt+D on empty text — should NOT set kill state
    input.handle_key(alt('d'));

    type_str(&mut input, "hello");
    input.handle_key(ctrl('a'));
    input.handle_key(alt('d')); // kills "hello"

    input.handle_key(ctrl('y'));
    assert_eq!(input.text(), "hello");
}

#[test]
fn empty_ctrl_u_does_not_set_kill_state() {
    let mut input = Input::new();

    // Ctrl+U on empty text — should NOT set kill state
    input.handle_key(ctrl('u'));

    type_str(&mut input, "world");
    input.handle_key(ctrl('e'));
    input.handle_key(ctrl('w')); // kills "world"

    input.handle_key(ctrl('a'));
    input.handle_key(ctrl('y'));
    assert_eq!(input.text(), "world");
}

#[test]
fn empty_ctrl_k_does_not_set_kill_state() {
    let mut input = Input::new();

    // Ctrl+K on empty text — should NOT set kill state
    input.handle_key(ctrl('k'));

    type_str(&mut input, "done");
    input.handle_key(ctrl('a'));
    input.handle_key(ctrl('k')); // kills "done"

    input.handle_key(ctrl('y'));
    assert_eq!(input.text(), "done");
}

// ---------------------------------------------------------------------------
// Undo stack cap
// ---------------------------------------------------------------------------

#[test]
fn undo_stack_capped_at_max() {
    let mut input = Input::new();

    // Create more than 100 undo entries by typing words separated by spaces.
    // Each space triggers a new undo entry (whitespace breaks coalescing).
    for i in 0..110 {
        if i > 0 {
            input.handle_key(char_key(' '));
        }
        type_str(&mut input, &format!("w{i}"));
    }

    // Undo 110 times, counting how many actually change the text.
    // With a cap of 100, only 100 undos should be effective.
    let mut effective_undos = 0;
    for _ in 0..115 {
        let before = input.text().to_string();
        input.handle_key(ctrl('-'));
        if input.text() != before {
            effective_undos += 1;
        }
    }

    assert!(
        effective_undos <= 100,
        "undo stack not capped: {} effective undos (expected <= 100)",
        effective_undos,
    );
}

// ---------------------------------------------------------------------------
// Yank pop (Alt+Y)
// ---------------------------------------------------------------------------

#[test]
fn yank_pop_cycles_kill_ring() {
    let mut input = Input::new();
    type_str(&mut input, "aaa bbb ccc");
    // Kill "ccc"
    input.handle_key(ctrl('w'));
    assert_eq!(input.text(), "aaa bbb ");
    // Break kill chain, then kill "ddd"
    type_str(&mut input, "ddd");
    input.handle_key(ctrl('w'));
    assert_eq!(input.text(), "aaa bbb ");
    // Yank at start -- gets most recent kill ("ddd")
    input.handle_key(ctrl('a'));
    input.handle_key(ctrl('y'));
    assert_eq!(input.text(), "dddaaa bbb ");
    assert_eq!(input.cursor(), 3);
    // yank_pop replaces "ddd" with previous entry "ccc"
    input.handle_key(alt('y'));
    assert_eq!(input.text(), "cccaaa bbb ");
    assert_eq!(input.cursor(), 3);
}

#[test]
fn yank_pop_noop_without_prior_yank() {
    let mut input = Input::new();
    type_str(&mut input, "hello");
    input.handle_key(ctrl('e'));
    input.handle_key(ctrl('w')); // kill "hello", text=""
    assert_eq!(input.text(), "");
    // Alt+Y without prior yank -- should be noop
    input.handle_key(alt('y'));
    assert_eq!(input.text(), "");
    assert_eq!(input.cursor(), 0);
}

#[test]
fn yank_pop_noop_single_entry() {
    let mut input = Input::new();
    type_str(&mut input, "hello world");
    input.handle_key(ctrl('e'));
    input.handle_key(ctrl('w')); // kill "world", text="hello "
    assert_eq!(input.text(), "hello ");
    input.handle_key(ctrl('a'));
    input.handle_key(ctrl('y')); // yank "world" at start, text="worldhello "
    assert_eq!(input.text(), "worldhello ");
    // Alt+Y with only one kill ring entry -- should be noop
    input.handle_key(alt('y'));
    assert_eq!(input.text(), "worldhello ");
    assert_eq!(input.cursor(), 5);
}

#[test]
fn yank_pop_repeated() {
    let mut input = Input::new();
    type_str(&mut input, "aa bb cc");
    // Kill "cc"
    input.handle_key(ctrl('w'));
    assert_eq!(input.text(), "aa bb ");
    // Break chain, kill "dd"
    type_str(&mut input, "dd");
    input.handle_key(ctrl('w'));
    assert_eq!(input.text(), "aa bb ");
    // Break chain, kill "ee"
    type_str(&mut input, "ee");
    input.handle_key(ctrl('w'));
    assert_eq!(input.text(), "aa bb ");
    // Yank at start -- gets most recent kill ("ee")
    input.handle_key(ctrl('a'));
    input.handle_key(ctrl('y'));
    assert_eq!(input.text(), "eeaa bb ");
    assert_eq!(input.cursor(), 2);
    // First yank_pop -- replaces "ee" with "dd"
    input.handle_key(alt('y'));
    assert_eq!(input.text(), "ddaa bb ");
    assert_eq!(input.cursor(), 2);
    // Second yank_pop -- replaces "dd" with "cc"
    input.handle_key(alt('y'));
    assert_eq!(input.text(), "ccaa bb ");
    assert_eq!(input.cursor(), 2);
}
