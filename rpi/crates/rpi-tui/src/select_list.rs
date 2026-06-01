use crate::keys::{Key, KeyEvent, Modifiers, SpecialKey};
use crate::{Frame, FrameLine};

fn truncate_to_width(s: &str, max_width: usize) -> &str {
    let mut width = 0;
    for (i, c) in s.char_indices() {
        let cw = unicode_width::UnicodeWidthChar::width(c).unwrap_or(0);
        if width + cw > max_width {
            return &s[..i];
        }
        width += cw;
    }
    s
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelectItem {
    pub value: String,
    pub label: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SelectListResult {
    Cancelled,
    Confirmed(Option<SelectItem>),
    Navigating,
}

pub struct SelectList {
    items: Vec<SelectItem>,
    max_visible: usize,
    selected_index: usize,
    filter: String,
    filtered_indices: Vec<usize>,
}

impl SelectList {
    pub fn new(items: Vec<SelectItem>, max_visible: usize) -> Self {
        let filtered_indices = (0..items.len()).collect();
        Self {
            items,
            max_visible,
            selected_index: 0,
            filter: String::new(),
            filtered_indices,
        }
    }

    pub fn selected_index(&self) -> usize {
        self.selected_index
    }

    pub fn selected_item(&self) -> Option<&SelectItem> {
        self.filtered_indices.get(self.selected_index).map(|&i| &self.items[i])
    }

    pub fn handle_key(&mut self, event: KeyEvent) -> SelectListResult {
        // Cancel
        if event.key == Key::Special(SpecialKey::Escape)
            || (event.key == Key::Char('c') && event.modifiers.contains(Modifiers::CTRL))
        {
            return SelectListResult::Cancelled;
        }

        let count = self.filtered_indices.len();

        // Enter: confirm selection
        if event.key == Key::Special(SpecialKey::Enter) {
            if count == 0 {
                return SelectListResult::Navigating;
            }
            let selected = self.filtered_indices.get(self.selected_index).map(|&i| self.items[i].clone());
            return SelectListResult::Confirmed(selected);
        }

        // Navigation (wrap around)
        if count > 0 {
            match event.key {
                Key::Special(SpecialKey::Down) => {
                    self.selected_index = (self.selected_index + 1) % count;
                }
                Key::Special(SpecialKey::Up) => {
                    self.selected_index = if self.selected_index == 0 {
                        count - 1
                    } else {
                        self.selected_index - 1
                    };
                }
                Key::Special(SpecialKey::PageDown) => {
                    self.selected_index = (self.selected_index + self.max_visible).min(count - 1);
                }
                Key::Special(SpecialKey::PageUp) => {
                    self.selected_index = self.selected_index.saturating_sub(self.max_visible);
                }
                Key::Special(SpecialKey::Home) => {
                    self.selected_index = 0;
                }
                Key::Special(SpecialKey::End) => {
                    self.selected_index = count - 1;
                }
                _ => {}
            }
        }

        SelectListResult::Navigating
    }

    pub fn select_with_key(&mut self, event: KeyEvent) -> Option<SelectItem> {
        match self.handle_key(event) {
            SelectListResult::Confirmed(item) => item,
            _ => None,
        }
    }

    pub fn set_filter(&mut self, filter: &str) {
        self.filter = filter.to_string();
        self.recompute_filtered();
        self.selected_index = 0;
    }

    fn recompute_filtered(&mut self) {
        if self.filter.is_empty() {
            self.filtered_indices = (0..self.items.len()).collect();
        } else {
            let lower = self.filter.to_lowercase();
            self.filtered_indices = self.items.iter().enumerate()
                .filter(|(_, item)| item.label.to_lowercase().starts_with(&lower))
                .map(|(i, _)| i)
                .collect();
        }
    }

    pub fn filtered_count(&self) -> usize {
        self.filtered_indices.len()
    }

    pub fn render(&self, width: usize) -> Frame {
        let count = self.filtered_indices.len();

        if count == 0 {
            return Frame::from_plain_lines(["No match"]);
        }

        // Scroll offset: keep selected item visible
        let scroll_offset = if count <= self.max_visible {
            0
        } else if self.selected_index >= self.max_visible {
            self.selected_index - self.max_visible + 1
        } else {
            0
        };

        let visible_count = count.min(self.max_visible);
        let mut lines: Vec<String> = Vec::new();

        for i in 0..visible_count {
            let idx = scroll_offset + i;
            if let Some(&item_idx) = self.filtered_indices.get(idx) {
                let item = &self.items[item_idx];
                let label_width = unicode_width::UnicodeWidthStr::width(item.label.as_str());
                let label = if label_width > width {
                    truncate_to_width(&item.label, width)
                } else {
                    &item.label
                };
                lines.push(label.to_string());
            }
        }

        // Scroll indicator when items overflow visible area
        if count > self.max_visible {
            lines.push(format!(
                "{}/{}",
                self.selected_index + 1,
                count
            ));
        }

        Frame::new(lines.iter().map(|s| FrameLine::plain(s)).collect())
    }

    pub fn set_selected_index(&mut self, index: usize) {
        let count = self.filtered_indices.len();
        if count == 0 {
            self.selected_index = 0;
        } else {
            self.selected_index = index.min(count - 1);
        }
    }

}
