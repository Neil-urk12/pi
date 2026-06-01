use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};
use std::io::IsTerminal;
use std::time::{Duration, Instant};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

pub mod keybindings;
pub mod keys;
pub mod stdin_buffer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    Default,
    Blue,
    Cyan,
    Green,
    Yellow,
    Red,
    Magenta,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Style {
    pub foreground: Color,
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
    pub strikethrough: bool,
}

impl Default for Style {
    fn default() -> Self {
        Self {
            foreground: Color::Default,
            bold: false,
            italic: false,
            underline: false,
            strikethrough: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StyledSpan {
    text: String,
    style: Style,
}

impl StyledSpan {
    pub fn new(text: impl Into<String>, style: Style) -> Self {
        Self {
            text: text.into(),
            style,
        }
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn style(&self) -> Style {
        self.style
    }
}

#[derive(Debug, Clone)]
pub struct DefaultTheme {
    pub text: Style,
    pub heading: Style,
    pub code: Style,
    pub quote: Style,
    pub list_marker: Style,
    pub link: Style,
    pub table_border: Style,
    pub tool: Style,
    pub warning: Style,
}

impl Default for DefaultTheme {
    fn default() -> Self {
        Self {
            text: Style::default(),
            heading: Style {
                foreground: Color::Cyan,
                bold: true,
                underline: true,
                ..Style::default()
            },
            code: Style {
                foreground: Color::Yellow,
                ..Style::default()
            },
            quote: Style {
                foreground: Color::Blue,
                italic: true,
                ..Style::default()
            },
            list_marker: Style {
                foreground: Color::Green,
                ..Style::default()
            },
            link: Style {
                foreground: Color::Blue,
                underline: true,
                ..Style::default()
            },
            table_border: Style {
                foreground: Color::Cyan,
                ..Style::default()
            },
            tool: Style {
                foreground: Color::Magenta,
                ..Style::default()
            },
            warning: Style {
                foreground: Color::Red,
                bold: true,
                ..Style::default()
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FrameLine {
    spans: Vec<StyledSpan>,
}

impl FrameLine {
    pub fn plain(text: impl Into<String>) -> Self {
        Self {
            spans: vec![StyledSpan::new(text, Style::default())],
        }
    }

    pub fn spans(&self) -> &[StyledSpan] {
        &self.spans
    }

    pub fn width(&self) -> usize {
        self.spans
            .iter()
            .map(|span| UnicodeWidthStr::width(span.text.as_str()))
            .sum()
    }

    pub fn to_plain_text(&self) -> String {
        let mut line = String::new();
        for span in &self.spans {
            line.push_str(span.text());
        }
        line
    }

    fn push_span(&mut self, text: String, style: Style) {
        if text.is_empty() {
            return;
        }

        if let Some(last) = self.spans.last_mut()
            && last.style == style
        {
            last.text.push_str(&text);
            return;
        }

        self.spans.push(StyledSpan::new(text, style));
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Frame {
    lines: Vec<FrameLine>,
}

impl Frame {
    pub fn new(lines: Vec<FrameLine>) -> Self {
        Self { lines }
    }

    pub fn from_plain_lines<const N: usize>(lines: [&str; N]) -> Self {
        Self {
            lines: lines.into_iter().map(FrameLine::plain).collect(),
        }
    }

    pub fn lines(&self) -> &[FrameLine] {
        &self.lines
    }

    pub fn to_plain_text(&self) -> String {
        self.lines
            .iter()
            .map(FrameLine::to_plain_text)
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameDiff {
    changed_rows: Vec<usize>,
}

impl FrameDiff {
    pub fn between(previous: &Frame, next: &Frame) -> Self {
        let max_len = previous.lines.len().max(next.lines.len());
        let mut changed_rows = Vec::new();

        for row in 0..max_len {
            if previous.lines.get(row) != next.lines.get(row) {
                changed_rows.push(row);
            }
        }

        Self { changed_rows }
    }

    pub fn changed_rows(&self) -> &[usize] {
        &self.changed_rows
    }
}

#[derive(Debug, Clone)]
pub struct MarkdownRenderer {
    theme: DefaultTheme,
}

impl MarkdownRenderer {
    pub fn new(theme: DefaultTheme) -> Self {
        Self { theme }
    }

    pub fn render(&self, markdown: &str, width: usize) -> Frame {
        let mut options = Options::empty();
        options.insert(Options::ENABLE_TABLES);
        options.insert(Options::ENABLE_TASKLISTS);
        options.insert(Options::ENABLE_STRIKETHROUGH);

        let parser = Parser::new_ext(markdown, options);
        let mut builder = FrameBuilder::new(width.max(1), self.theme.clone());
        builder.render(parser);
        builder.finish()
    }
}

#[derive(Debug)]
struct TableState {
    header_cells: Vec<String>,
    rows: Vec<Vec<String>>,
    current_row: Vec<String>,
    current_cell: String,
    in_header: bool,
}

struct FrameBuilder {
    width: usize,
    theme: DefaultTheme,
    lines: Vec<FrameLine>,
    current: FrameLine,
    in_code_block: bool,
    at_line_start_in_code: bool,
    code_block_has_content: bool,
    style: Style,
    style_stack: Vec<Style>,
    list_stack: Vec<ListState>,
    link_stack: Vec<String>,
    table_state: Option<TableState>,
}

#[derive(Debug, Clone)]
struct ListState {
    next: u64,
    ordered: bool,
}

impl FrameBuilder {
    fn new(width: usize, theme: DefaultTheme) -> Self {
        let style = theme.text;
        Self {
            width,
            theme,
            lines: Vec::new(),
            current: FrameLine::default(),
            style,
            style_stack: Vec::new(),
            list_stack: Vec::new(),
            link_stack: Vec::new(),
            table_state: None,
            in_code_block: false,
            at_line_start_in_code: false,
            code_block_has_content: false,
        }
    }

    fn render<'a>(&mut self, parser: Parser<'a>) {
        for event in parser {
            match event {
                Event::Start(tag) => self.start_tag(tag),
                Event::End(tag) => self.end_tag(tag),
                Event::Text(text) => {
                    if let Some(ts) = &mut self.table_state {
                        ts.current_cell.push_str(text.as_ref());
                    } else {
                        self.append(text.as_ref(), self.style);
                    }
                }
                Event::Code(code) => {
                    if let Some(ts) = &mut self.table_state {
                        ts.current_cell.push('`');
                        ts.current_cell.push_str(code.as_ref());
                        ts.current_cell.push('`');
                    } else {
                        self.append("`", self.theme.code);
                        self.append(code.as_ref(), self.theme.code);
                        self.append("`", self.theme.code);
                    }
                }
                Event::SoftBreak | Event::HardBreak => self.finish_line(),
                Event::Rule => {
                    self.finish_line();
                    self.append(&"-".repeat(self.width.min(40)), self.theme.table_border);
                    self.finish_line();
                }
                Event::TaskListMarker(checked) => {
                    let marker = if checked { "[x] " } else { "[ ] " };
                    self.append(marker, self.theme.list_marker);
                }
                Event::Html(_) | Event::InlineHtml(_) => {}
                Event::FootnoteReference(reference) => self.append(reference.as_ref(), self.style),
                _ => {}
            }
        }
    }

    fn start_tag(&mut self, tag: Tag<'_>) {
        match tag {
            Tag::Paragraph => {}
            Tag::Heading { level, .. } => {
                self.finish_line();
                if level as usize >= 3 {
                    self.append(
                        &format!("{} ", "#".repeat(level as usize)),
                        self.theme.heading,
                    );
                }
                self.push_style(self.theme.heading);
            }
            Tag::BlockQuote(_) => {
                self.finish_line();
                self.append("> ", self.theme.quote);
                self.push_style(self.theme.quote);
            }
            Tag::CodeBlock(_) => {
                self.finish_line();
                self.append("```", self.theme.code);
                self.finish_line();
                self.push_style(self.theme.code);
                self.in_code_block = true;
                self.at_line_start_in_code = true;
                self.code_block_has_content = false;
            }
            Tag::List(start) => {
                self.list_stack.push(ListState {
                    next: start.unwrap_or(1),
                    ordered: start.is_some(),
                });
            }
            Tag::Item => {
                self.finish_line();
                let marker = if let Some(list) = self.list_stack.last_mut() {
                    if list.ordered {
                        let marker = format!("{}. ", list.next);
                        list.next += 1;
                        marker
                    } else {
                        "- ".to_string()
                    }
                } else {
                    "- ".to_string()
                };
                self.append(&marker, self.theme.list_marker);
            }
            Tag::Table(_alignments) => {
                self.table_state = Some(TableState {
                    header_cells: Vec::new(),
                    rows: Vec::new(),
                    current_row: Vec::new(),
                    current_cell: String::new(),
                    in_header: true,
                });
            }
            Tag::TableHead => {}
            Tag::TableRow => {
                if let Some(ts) = &mut self.table_state {
                    ts.current_row = Vec::new();
                    ts.in_header = false;
                }
            }
            Tag::TableCell => {
                if let Some(ts) = &mut self.table_state {
                    ts.current_cell = String::new();
                }
            }
            Tag::Emphasis => self.push_style(Style {
                italic: true,
                ..self.style
            }),
            Tag::Strong => self.push_style(Style {
                bold: true,
                ..self.style
            }),
            Tag::Strikethrough => self.push_style(Style {
                strikethrough: true,
                ..self.style
            }),
            Tag::Link { dest_url, .. } => {
                self.link_stack.push(dest_url.to_string());
                self.push_style(self.theme.link);
            }
            Tag::Image { .. }
            | Tag::HtmlBlock
            | Tag::FootnoteDefinition(_)
            | Tag::DefinitionList
            | Tag::DefinitionListTitle
            | Tag::DefinitionListDefinition
            | Tag::MetadataBlock(_) => {}
        }
    }

    fn end_tag(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Paragraph | TagEnd::Heading(_) | TagEnd::BlockQuote(_) => {
                self.pop_style_for(tag);
                self.blank_line();
            }
            TagEnd::Item => {
                self.pop_style_for(tag);
            }
            TagEnd::CodeBlock => {
                self.in_code_block = false;
                self.at_line_start_in_code = false;
                self.pop_style();
                self.finish_line();
                self.append("```", self.theme.code);
                self.blank_line();
            }
            TagEnd::List(_) => {
                self.list_stack.pop();
                self.blank_line();
            }
            TagEnd::TableCell => {
                if let Some(ts) = &mut self.table_state {
                    let cell = std::mem::take(&mut ts.current_cell);
                    if ts.in_header {
                        ts.header_cells.push(cell);
                    } else {
                        ts.current_row.push(cell);
                    }
                }
            }
            TagEnd::TableHead => {}
            TagEnd::TableRow => {
                if let Some(ts) = &mut self.table_state {
                    let row = std::mem::take(&mut ts.current_row);
                    ts.rows.push(row);
                }
            }
            TagEnd::Table => {
                if let Some(table) = self.table_state.take() {
                    self.finish_line();
                    self.render_table(table);
                }
            }
            TagEnd::Emphasis | TagEnd::Strong | TagEnd::Strikethrough => self.pop_style(),
            TagEnd::Link => {
                self.pop_style();
                if let Some(url) = self.link_stack.pop() {
                    self.append(&format!(" ({url})"), self.theme.link);
                }
            }
            TagEnd::Image
            | TagEnd::HtmlBlock
            | TagEnd::FootnoteDefinition
            | TagEnd::DefinitionList
            | TagEnd::DefinitionListTitle
            | TagEnd::DefinitionListDefinition
            | TagEnd::MetadataBlock(_) => {}
        }
    }

    fn push_style(&mut self, style: Style) {
        self.style_stack.push(self.style);
        self.style = style;
    }

    fn pop_style(&mut self) {
        if let Some(style) = self.style_stack.pop() {
            self.style = style;
        }
    }

    fn pop_style_for(&mut self, tag: TagEnd) {
        if matches!(tag, TagEnd::Heading(_) | TagEnd::BlockQuote(_)) {
            self.pop_style();
        }
    }

    fn append(&mut self, text: &str, style: Style) {
        for ch in text.chars() {
            if ch == '\n' {
                self.finish_line();
                continue;
            }

            // Push "  " prefix before first visible char on each code block line
            if self.at_line_start_in_code {
                self.at_line_start_in_code = false;
                if self.width >= 4 {
                    self.current.push_span("  ".to_string(), self.style);
                }
            }

            let ch_width = ch.width().unwrap_or(0);
            if self.current.width() > 0 && self.current.width() + ch_width > self.width {
                self.finish_line();
            }

            self.current.push_span(ch.to_string(), style);
            if self.in_code_block && !ch.is_whitespace() {
                self.code_block_has_content = true;
            }
        }
    }

    fn finish_line(&mut self) {
        let line = std::mem::take(&mut self.current);
        if !line.spans.is_empty() || (self.in_code_block && self.code_block_has_content) {
            self.lines.push(line);
        }
        if self.in_code_block {
            self.at_line_start_in_code = true;
        }
    }

    /// Render a single table row: border, padded cell content, border for each column.
    fn render_table_row(&mut self, cells: &[String], col_widths: &[usize]) {
        self.current.push_span('\u{2502}'.to_string(), self.theme.table_border);
        for (i, &w) in col_widths.iter().enumerate() {
            let content = cells.get(i).map(|s| s.as_str()).unwrap_or("");
            let mut cell_buf = String::new();
            cell_buf.push(' ');
            let mut used = 0;
            for c in content.chars() {
                let cw = UnicodeWidthChar::width(c).unwrap_or(0);
                if used + cw > w { break; }
                cell_buf.push(c);
                used += cw;
            }
            for _ in 0..w.saturating_sub(used) {
                cell_buf.push(' ');
            }
            cell_buf.push(' ');
            self.current.push_span(cell_buf, self.theme.text);
            self.current.push_span('\u{2502}'.to_string(), self.theme.table_border);
        }
        self.finish_line();
    }

    fn blank_line(&mut self) {
        if self.current.width() > 0 {
            self.finish_line();
        }
        if self.lines.last().is_some_and(|line| line.width() > 0) {
            self.lines.push(FrameLine::default());
        }
    }

    fn finish(mut self) -> Frame {
        if self.current.width() > 0 || self.lines.is_empty() {
            self.finish_line();
        }

        while self.lines.last().is_some_and(|line| line.width() == 0) {
            self.lines.pop();
        }

        Frame::new(self.lines)
    }

    fn render_table(&mut self, table: TableState) {
        let num_cols = table.header_cells.len().max(1);

        // Terminal too narrow for even border structure — skip rendering
        let min_table_width = num_cols * 3 + 1;
        if self.width < min_table_width {
            return;
        }

        // Calculate column widths
        let mut col_widths = vec![0usize; num_cols];
        for (i, cell) in table.header_cells.iter().enumerate() {
            col_widths[i] = col_widths[i].max(UnicodeWidthStr::width(cell.as_str()));
        }
        for row in &table.rows {
            for (i, cell) in row.iter().enumerate() {
                if i < num_cols {
                    col_widths[i] = col_widths[i].max(UnicodeWidthStr::width(cell.as_str()));
                }
            }
        }

        // Ensure minimum width of 1
        for w in &mut col_widths {
            *w = (*w).max(1);
        }

        // Clamp columns to fit terminal width
        // Total: left_border(1) + per_col(width + 2 padding + separator) + right_border(1)
        //      = 1 + num_cols * (w + 2) + (num_cols - 1) + 1 = num_cols * 3 + sum(widths) + 1
        let border_overhead = num_cols.saturating_mul(3).saturating_add(1);
        let total_content: usize = col_widths.iter().sum();
        if border_overhead + total_content > self.width {
            let available = self.width.saturating_sub(border_overhead);
            if available >= num_cols && total_content > 0 {
                // Distribute available width proportionally to content needs
                let mut remaining = available;
                let num_w = col_widths.len();
                for (i, w) in col_widths.iter_mut().enumerate() {
                    if i == num_w - 1 {
                        *w = remaining.max(1);
                    } else {
                        let share = (w.saturating_mul(available) / total_content).max(1).min(remaining);
                        *w = share;
                        remaining = remaining.saturating_sub(*w);
                    }
                }
                // Ensure total width doesn't exceed available (from .max(1) guarantees)
                let total: usize = col_widths.iter().sum();
                if total > available {
                    let excess = total - available;
                    let scale = available as f64 / total as f64;
                    let mut reduced = 0usize;
                    for w in col_widths.iter_mut() {
                        let scaled = ((*w as f64 * scale) as usize).max(1);
                        let cut = (*w).saturating_sub(scaled);
                        *w = scaled;
                        reduced += cut;
                    }
                    // Distribute any remaining excess
                    let mut leftover = excess.saturating_sub(reduced);
                    while leftover > 0 {
                        if let Some(max_w) = col_widths.iter_mut().max() {
                            if *max_w > 1 { *max_w -= 1; leftover -= 1; } else { break; }
                        } else { break; }
                    }
                }
            } else {
                // Not enough space for per-column content — render border-only table
                col_widths.fill(0);
            }
        }

        // Helper to build a border line
        let border_line = |left: char, mid: char, right: char, fill: char, widths: &[usize]| -> String {
            let mut line = String::new();
            line.push(left);
            for (i, &w) in widths.iter().enumerate() {
                for _ in 0..w + 2 {
                    line.push(fill);
                }
                if i < widths.len() - 1 {
                    line.push(mid);
                } else {
                    line.push(right);
                }
            }
            line
        };


        // Top border: ┌─┬─┐
        self.finish_line();
        self.append(
            &border_line('\u{250C}', '\u{252C}', '\u{2510}', '\u{2500}', &col_widths),
            self.theme.table_border,
        );
        self.finish_line();

        // Header row — border chars get table_border, cell content gets text style
        self.render_table_row(&table.header_cells, &col_widths);

        // Header separator: ├─┼─┤ (only if there are data rows)
        if !table.rows.is_empty() {
            self.append(
                &border_line('\u{251C}', '\u{253C}', '\u{2524}', '\u{2500}', &col_widths),
                self.theme.table_border,
            );
            self.finish_line();
        }

        // Data rows
        for row in &table.rows {
            self.render_table_row(row, &col_widths);
        }

        // Bottom border: └─┴─┘
        self.append(
            &border_line('\u{2514}', '\u{2534}', '\u{2518}', '\u{2500}', &col_widths),
            self.theme.table_border,
        );
        self.finish_line();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TuiEvent {
    AssistantDelta(String),
    FinalAssistantMessage(String),
    ToolStarted {
        name: String,
        summary: Option<String>,
    },
    ToolFinished {
        name: String,
        summary: Option<String>,
        is_error: bool,
    },
    RendererWarning(String),
    TurnFinished,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderDecision {
    pub should_render: bool,
}

#[derive(Debug, Clone)]
pub struct TurnView {
    renderer: MarkdownRenderer,
    width: usize,
    assistant_text: String,
    tool_lines: Vec<String>,
    warning_lines: Vec<String>,
    pending_bytes: usize,
    last_render_at: Instant,
    finalized: bool,
}

impl TurnView {
    pub const DEFAULT_BYTE_THRESHOLD: usize = 512;
    pub const DEFAULT_INTERVAL: Duration = Duration::from_millis(33);

    pub fn new(theme: DefaultTheme, width: usize) -> Self {
        Self {
            renderer: MarkdownRenderer::new(theme),
            width: width.max(1),
            assistant_text: String::new(),
            tool_lines: Vec::new(),
            warning_lines: Vec::new(),
            pending_bytes: 0,
            last_render_at: Instant::now(),
            finalized: false,
        }
    }

    pub fn apply_event(&mut self, event: TuiEvent) -> RenderDecision {
        let now = Instant::now();
        let should_render = match event {
            TuiEvent::AssistantDelta(text) => {
                let boundary = text.contains('\n');
                self.pending_bytes += text.len();
                self.assistant_text.push_str(&text);
                boundary
                    || self.pending_bytes >= Self::DEFAULT_BYTE_THRESHOLD
                    || now.duration_since(self.last_render_at) >= Self::DEFAULT_INTERVAL
            }
            TuiEvent::FinalAssistantMessage(text) => {
                self.assistant_text = text;
                self.finalized = true;
                true
            }
            TuiEvent::ToolStarted { name, summary } => {
                self.tool_lines
                    .push(format_tool_line("Running", &name, summary.as_deref()));
                true
            }
            TuiEvent::ToolFinished {
                name,
                summary,
                is_error,
            } => {
                let status = if is_error { "Tool error" } else { "Finished" };
                self.tool_lines
                    .push(format_tool_line(status, &name, summary.as_deref()));
                true
            }
            TuiEvent::RendererWarning(warning) => {
                self.warning_lines
                    .push(format!("Renderer warning: {warning}"));
                true
            }
            TuiEvent::TurnFinished => {
                self.finalized = true;
                true
            }
        };

        if should_render {
            self.pending_bytes = 0;
            self.last_render_at = now;
        }

        RenderDecision { should_render }
    }

    pub fn render_frame(&self) -> Frame {
        let mut lines = Vec::new();
        lines.extend(self.renderer.render(&self.assistant_text, self.width).lines);

        for tool_line in &self.tool_lines {
            lines.push(FrameLine::plain(tool_line));
        }

        for warning_line in &self.warning_lines {
            lines.push(FrameLine::plain(warning_line));
        }

        Frame::new(lines)
    }

    pub fn is_finalized(&self) -> bool {
        self.finalized
    }
}

fn format_tool_line(status: &str, name: &str, summary: Option<&str>) -> String {
    match summary {
        Some(summary) if !summary.is_empty() => {
            format!("{status} {name}: {}", strip_control_chars(summary))
        }
        _ => format!("{status} {name}"),
    }
}

pub struct TerminalBackend;

impl TerminalBackend {
    pub fn terminal_width() -> Option<usize> {
        crossterm::terminal::size()
            .ok()
            .map(|(width, _)| usize::from(width))
    }

    pub fn should_color() -> bool {
        std::env::var_os("NO_COLOR").is_none() && std::io::stdout().is_terminal()
    }

    pub fn encode_frame(frame: &Frame, color: bool) -> String {
        let mut output = String::new();
        for (index, line) in frame.lines().iter().enumerate() {
            if index > 0 {
                output.push('\n');
            }
            if color {
                output.push_str("\x1b[2K");
            }
            for span in line.spans() {
                if color {
                    output.push_str(&ansi_prefix(span.style()));
                    output.push_str(span.text());
                    output.push_str("\x1b[0m");
                } else {
                    output.push_str(span.text());
                }
            }
        }
        output
    }

    pub fn encode_active_region_update(previous: &Frame, next: &Frame, color: bool) -> String {
        let diff = FrameDiff::between(previous, next);
        if diff.changed_rows().is_empty() {
            return String::new();
        }

        let mut output = String::new();
        let previous_line_count = previous.lines().len();
        if previous_line_count > 0 {
            output.push_str(&format!("\x1b[{}F", previous_line_count));
        }

        output.push_str(&Self::encode_frame(next, color));
        output.push('\n');

        if previous_line_count > next.lines().len() {
            for _ in 0..(previous_line_count - next.lines().len()) {
                output.push_str("\x1b[2K\n");
            }
        }

        output
    }
}

fn ansi_prefix(style: Style) -> String {
    let mut codes = Vec::new();

    match style.foreground {
        Color::Default => {}
        Color::Blue => codes.push("34"),
        Color::Cyan => codes.push("36"),
        Color::Green => codes.push("32"),
        Color::Yellow => codes.push("33"),
        Color::Red => codes.push("31"),
        Color::Magenta => codes.push("35"),
    }

    if style.bold {
        codes.push("1");
    }
    if style.italic {
        codes.push("3");
    }
    if style.underline {
        codes.push("4");
    }
    if style.strikethrough {
        codes.push("9");
    }

    if codes.is_empty() {
        String::new()
    } else {
        format!("\x1b[{}m", codes.join(";"))
    }
}

fn strip_control_chars(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            // ESC - start of escape sequence
            '\x1b' => {
                match chars.peek() {
                    Some(&'[') => {
                        // CSI sequence: ESC [ ... final_char (0x40-0x7E)
                        chars.next(); // consume '['
                        while let Some(&next) = chars.peek() {
                            if (0x40..=0x7E).contains(&(next as u32)) {
                                chars.next(); // consume final char
                                break;
                            }
                            chars.next();
                        }
                    }
                    Some(&']') => {
                        // OSC sequence: ESC ] ... BEL or ESC \
                        chars.next(); // consume ']'
                        loop {
                            match chars.next() {
                                Some('\x07') => break, // BEL terminator
                                Some('\x1b') => {
                                    if chars.peek() == Some(&'\\') {
                                        chars.next(); // consume backslash
                                    }
                                    break;
                                }
                                None => break,
                                _ => {}
                            }
                        }
                    }
                    // String-bearing sequences: DCS(P), PM(^), APC(_), SOS(X)
                    // Body terminated by ST (ESC \)
                    Some(&'P') | Some(&'^') | Some(&'_') | Some(&'X') => {
                        chars.next(); // consume Fe byte
                        loop {
                            match chars.next() {
                                Some('\x1b') => {
                                    if chars.peek() == Some(&'\\') {
                                        chars.next();
                                    }
                                    break;
                                }
                                None => break,
                                _ => {}
                            }
                        }
                    }
                    Some(&next) if (0x30..=0x7E).contains(&(next as u32)) => {
                        // Simple ESC + char
                        chars.next();
                    }
                    _ => {}
                }
            }
            // Pass through printable chars and allowed whitespace
            c if !c.is_control() || c == '\t' || c == '\n' || c == '\r' => {
                result.push(c);
            }
            // Skip other control characters
            _ => {}
        }
    }
    result
}
