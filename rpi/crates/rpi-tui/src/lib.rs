use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};
use std::io::IsTerminal;
use std::time::{Duration, Instant};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

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

struct FrameBuilder {
    width: usize,
    theme: DefaultTheme,
    lines: Vec<FrameLine>,
    current: FrameLine,
    style: Style,
    style_stack: Vec<Style>,
    list_stack: Vec<ListState>,
    link_stack: Vec<String>,
    table_cell_started: bool,
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
            table_cell_started: false,
        }
    }

    fn render<'a>(&mut self, parser: Parser<'a>) {
        for event in parser {
            match event {
                Event::Start(tag) => self.start_tag(tag),
                Event::End(tag) => self.end_tag(tag),
                Event::Text(text) => self.append(text.as_ref(), self.style),
                Event::Code(code) => {
                    self.append("`", self.theme.code);
                    self.append(code.as_ref(), self.theme.code);
                    self.append("`", self.theme.code);
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
                self.append(
                    &format!("{} ", "#".repeat(level as usize)),
                    self.theme.heading,
                );
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
            Tag::Table(_) | Tag::TableHead | Tag::TableRow => {
                self.finish_line();
            }
            Tag::TableCell => {
                if self.table_cell_started {
                    self.append(" | ", self.theme.table_border);
                }
                self.table_cell_started = true;
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
            TagEnd::Paragraph | TagEnd::Heading(_) | TagEnd::BlockQuote(_) | TagEnd::Item => {
                self.pop_style_for(tag);
                self.blank_line();
            }
            TagEnd::CodeBlock => {
                self.pop_style();
                self.finish_line();
                self.append("```", self.theme.code);
                self.blank_line();
            }
            TagEnd::List(_) => {
                self.list_stack.pop();
            }
            TagEnd::TableHead | TagEnd::TableRow => {
                self.table_cell_started = false;
                self.finish_line();
            }
            TagEnd::Table => {
                self.table_cell_started = false;
                self.blank_line();
            }
            TagEnd::TableCell => {}
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

            let ch_width = ch.width().unwrap_or(0);
            if self.current.width() > 0 && self.current.width() + ch_width > self.width {
                self.finish_line();
            }

            self.current.push_span(ch.to_string(), style);
        }
    }

    fn finish_line(&mut self) {
        self.lines.push(std::mem::take(&mut self.current));
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
            output.push_str("\x1b[2K");
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
