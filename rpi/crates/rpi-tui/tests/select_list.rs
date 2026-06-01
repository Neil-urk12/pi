use rpi_tui::keys::{Key, KeyEvent, Modifiers, SpecialKey};
use rpi_tui::select_list::{SelectItem, SelectList, SelectListResult};

fn item(value: &str, label: &str) -> SelectItem {
    SelectItem {
        value: value.to_string(),
        label: label.to_string(),
        description: None,
    }
}

fn item_with_desc(value: &str, label: &str, description: &str) -> SelectItem {
    SelectItem {
        value: value.to_string(),
        label: label.to_string(),
        description: Some(description.to_string()),
    }
}

fn key_down() -> KeyEvent {
    KeyEvent {
        key: Key::Special(SpecialKey::Down),
        modifiers: Modifiers::empty(),
    }
}

fn key_up() -> KeyEvent {
    KeyEvent {
        key: Key::Special(SpecialKey::Up),
        modifiers: Modifiers::empty(),
    }
}

fn key_enter() -> KeyEvent {
    KeyEvent {
        key: Key::Special(SpecialKey::Enter),
        modifiers: Modifiers::empty(),
    }
}

fn key_escape() -> KeyEvent {
    KeyEvent {
        key: Key::Special(SpecialKey::Escape),
        modifiers: Modifiers::empty(),
    }
}

fn key_ctrl_c() -> KeyEvent {
    KeyEvent {
        key: Key::Char('c'),
        modifiers: Modifiers::CTRL,
    }
}

// --- Construction ---

#[test]
fn new_starts_at_index_zero() {
    let list = SelectList::new(vec![item("a", "A"), item("b", "B")], 5);
    assert_eq!(list.selected_index(), 0);
}

#[test]
fn new_with_empty_items() {
    let list: SelectList = SelectList::new(vec![], 5);
    assert_eq!(list.selected_index(), 0);
    assert_eq!(list.selected_item(), None);
}

#[test]
fn new_with_single_item() {
    let list = SelectList::new(vec![item("only", "Only")], 5);
    assert_eq!(list.selected_index(), 0);
    assert_eq!(list.selected_item().unwrap().value, "only");
}

// --- Down navigation ---

#[test]
fn down_moves_to_next_item() {
    let mut list = SelectList::new(vec![item("a", "A"), item("b", "B"), item("c", "C")], 5);
    list.handle_key(key_down());
    assert_eq!(list.selected_index(), 1);
}

#[test]
fn down_multiple_times() {
    let mut list = SelectList::new(vec![item("a", "A"), item("b", "B"), item("c", "C")], 5);
    list.handle_key(key_down());
    list.handle_key(key_down());
    assert_eq!(list.selected_index(), 2);
}

#[test]
fn down_wraps_to_top_from_bottom() {
    let mut list = SelectList::new(vec![item("a", "A"), item("b", "B"), item("c", "C")], 5);
    list.handle_key(key_down());
    list.handle_key(key_down());
    assert_eq!(list.selected_index(), 2);
    list.handle_key(key_down());
    assert_eq!(list.selected_index(), 0);
}

// --- Up navigation ---

#[test]
fn up_wraps_to_bottom_from_top() {
    let mut list = SelectList::new(vec![item("a", "A"), item("b", "B"), item("c", "C")], 5);
    assert_eq!(list.selected_index(), 0);
    list.handle_key(key_up());
    assert_eq!(list.selected_index(), 2);
}

#[test]
fn up_moves_to_previous_item() {
    let mut list = SelectList::new(vec![item("a", "A"), item("b", "B"), item("c", "C")], 5);
    list.handle_key(key_down());
    list.handle_key(key_down());
    assert_eq!(list.selected_index(), 2);
    list.handle_key(key_up());
    assert_eq!(list.selected_index(), 1);
}

// --- Enter (confirm) ---

#[test]
fn enter_selects_current_item() {
    let mut list = SelectList::new(vec![item("a", "A"), item("b", "B")], 5);
    list.handle_key(key_down());
    let selected = list.select_with_key(key_enter());
    assert!(selected.is_some());
    assert_eq!(selected.unwrap().value, "b");
}

#[test]
fn enter_on_first_item() {
    let mut list = SelectList::new(vec![item("a", "A"), item("b", "B")], 5);
    let selected = list.select_with_key(key_enter());
    assert!(selected.is_some());
    assert_eq!(selected.unwrap().value, "a");
}

// --- Escape / Ctrl+C (cancel) ---

#[test]
fn escape_cancels() {
    let mut list = SelectList::new(vec![item("a", "A"), item("b", "B")], 5);
    let result = list.handle_key(key_escape());
    assert_eq!(result, SelectListResult::Cancelled);
}

#[test]
fn ctrl_c_cancels() {
    let mut list = SelectList::new(vec![item("a", "A"), item("b", "B")], 5);
    let result = list.handle_key(key_ctrl_c());
    assert_eq!(result, SelectListResult::Cancelled);
}

#[test]
fn cancel_does_not_select() {
    let mut list = SelectList::new(vec![item("a", "A")], 5);
    let result = list.handle_key(key_escape());
    assert_eq!(result, SelectListResult::Cancelled);
}

// --- Filtering ---

#[test]
fn filter_reduces_items() {
    let mut list = SelectList::new(
        vec![item("apple", "Apple"), item("banana", "Banana"), item("avocado", "Avocado")],
        5,
    );
    list.set_filter("a");
    assert_eq!(list.filtered_count(), 2);
}

#[test]
fn filter_is_case_insensitive() {
    let mut list = SelectList::new(
        vec![item("Apple", "Apple"), item("Banana", "Banana")],
        5,
    );
    list.set_filter("a");
    assert_eq!(list.filtered_count(), 1);
}

#[test]
fn filter_resets_selection_to_zero() {
    let mut list = SelectList::new(
        vec![item("apple", "Apple"), item("banana", "Banana"), item("avocado", "Avocado")],
        5,
    );
    list.handle_key(key_down());
    list.handle_key(key_down());
    assert_eq!(list.selected_index(), 2);
    list.set_filter("a");
    assert_eq!(list.selected_index(), 0);
}

#[test]
fn filter_matching_nothing_shows_empty() {
    let mut list = SelectList::new(
        vec![item("apple", "Apple"), item("banana", "Banana")],
        5,
    );
    list.set_filter("zzz");
    assert_eq!(list.filtered_count(), 0);
    assert_eq!(list.selected_item(), None);
}

#[test]
fn empty_filter_shows_all_items() {
    let mut list = SelectList::new(
        vec![item("apple", "Apple"), item("banana", "Banana")],
        5,
    );
    list.set_filter("app");
    assert_eq!(list.filtered_count(), 1);
    list.set_filter("");
    assert_eq!(list.filtered_count(), 2);
}

// --- Scrolling / max_visible ---

#[test]
fn render_respects_max_visible() {
    let items: Vec<SelectItem> = (0..20).map(|i| item(&format!("item{i}"), &format!("Item {i}"))).collect();
    let list = SelectList::new(items, 5);
    let frame = list.render(80);
    // Should render at most max_visible + 1 line (for scroll indicator)
    assert!(frame.lines().len() <= 6);
}

#[test]
fn scroll_indicator_shown_when_items_exceed_max_visible() {
    let items: Vec<SelectItem> = (0..20).map(|i| item(&format!("item{i}"), &format!("Item {i}"))).collect();
    let list = SelectList::new(items, 5);
    let frame = list.render(80);
    let plain = frame.to_plain_text();
    // Should contain position indicator like "1/20"
    assert!(plain.contains("/20"));
}

#[test]
fn no_scroll_indicator_when_all_items_visible() {
    let items = vec![item("a", "A"), item("b", "B")];
    let list = SelectList::new(items, 5);
    let frame = list.render(80);
    let plain = frame.to_plain_text();
    assert!(!plain.contains("/2"));
}

#[test]
fn scrolling_follows_selection_down() {
    let items: Vec<SelectItem> = (0..20).map(|i| item(&format!("item{i}"), &format!("Item {i}"))).collect();
    let mut list = SelectList::new(items, 5);
    // Move past the visible window
    for _ in 0..10 {
        list.handle_key(key_down());
    }
    assert_eq!(list.selected_index(), 10);
    let frame = list.render(80);
    let plain = frame.to_plain_text();
    // The selected item should be visible in the rendered output
    assert!(plain.contains("Item 10"));
}

// --- Edge cases ---

#[test]
fn navigation_on_single_item_stays_at_zero() {
    let mut list = SelectList::new(vec![item("only", "Only")], 5);
    list.handle_key(key_down());
    assert_eq!(list.selected_index(), 0);
    list.handle_key(key_up());
    assert_eq!(list.selected_index(), 0);
}

#[test]
fn navigation_on_empty_list_is_noop() {
    let mut list: SelectList = SelectList::new(vec![], 5);
    list.handle_key(key_down());
    assert_eq!(list.selected_index(), 0);
    list.handle_key(key_up());
    assert_eq!(list.selected_index(), 0);
}

#[test]
fn select_on_empty_list_returns_none() {
    let mut list: SelectList = SelectList::new(vec![], 5);
    let result = list.handle_key(key_enter());
    assert!(matches!(result, SelectListResult::Navigating));
}

#[test]
fn set_selected_index_clamps_to_valid_range() {
    let mut list = SelectList::new(vec![item("a", "A"), item("b", "B"), item("c", "C")], 5);
    list.set_selected_index(10);
    assert_eq!(list.selected_index(), 2);
    list.set_selected_index(100);
    assert_eq!(list.selected_index(), 2);
}

#[test]
fn set_selected_index_clamps_below_zero() {
    let mut list = SelectList::new(vec![item("a", "A"), item("b", "B")], 5);
    // After construction at 0, trying to go lower should stay at 0
    list.set_selected_index(0);
    assert_eq!(list.selected_index(), 0);
}

#[test]
fn selected_item_matches_index() {
    let mut list = SelectList::new(
        vec![item("x", "X"), item("y", "Y"), item("z", "Z")],
        5,
    );
    list.handle_key(key_down());
    let sel = list.selected_item().unwrap();
    assert_eq!(sel.value, "y");
    assert_eq!(sel.label, "Y");
}

#[test]
fn items_with_descriptions_are_stored() {
    let list = SelectList::new(
        vec![item_with_desc("cmd", "cmd", "runs the command")],
        5,
    );
    let sel = list.selected_item().unwrap();
    assert_eq!(sel.description.as_deref(), Some("runs the command"));
}

#[test]
fn filtered_navigation_respects_filtered_set() {
    let mut list = SelectList::new(
        vec![
            item("apple", "Apple"),
            item("banana", "Banana"),
            item("avocado", "Avocado"),
        ],
        5,
    );
    list.set_filter("a");
    // Filtered: apple (0), avocado (1)
    list.handle_key(key_down());
    assert_eq!(list.selected_index(), 1);
    let sel = list.selected_item().unwrap();
    assert_eq!(sel.value, "avocado");
}

#[test]
fn down_wraps_around_in_filtered_set() {
    let mut list = SelectList::new(
        vec![
            item("apple", "Apple"),
            item("banana", "Banana"),
            item("avocado", "Avocado"),
        ],
        5,
    );
    list.set_filter("a");
    // Filtered: apple, avocado — 2 items
    list.handle_key(key_down()); // -> avocado (1)
    list.handle_key(key_down()); // wraps -> apple (0)
    assert_eq!(list.selected_index(), 0);
}

#[test]
fn render_empty_filtered_list_shows_no_match() {
    let mut list = SelectList::new(
        vec![item("apple", "Apple"), item("banana", "Banana")],
        5,
    );
    list.set_filter("zzz");
    let frame = list.render(80);
    let plain = frame.to_plain_text();
    assert!(plain.to_lowercase().contains("no match"));
}

#[test]
fn render_with_zero_width_does_not_panic() {
    let list = SelectList::new(vec![item("a", "A")], 5);
    let frame = list.render(0);
    // Should produce some output without panicking
    let _ = frame.to_plain_text();
}

#[test]
fn render_width_limits_output_lines() {
    let items: Vec<SelectItem> = (0..100).map(|i| item(&format!("item{i}"), &format!("Item {i}"))).collect();
    let list = SelectList::new(items, 3);
    let frame = list.render(80);
    // max_visible=3, so at most 3 items + 1 scroll indicator = 4 lines
    assert!(frame.lines().len() <= 4);
}

#[test]
fn filter_then_enter_selects_from_filtered() {
    let mut list = SelectList::new(
        vec![
            item("apple", "Apple"),
            item("banana", "Banana"),
            item("avocado", "Avocado"),
        ],
        5,
    );
    list.set_filter("b");
    let selected = list.select_with_key(key_enter());
    assert!(selected.is_some());
    assert_eq!(selected.unwrap().value, "banana");
}
