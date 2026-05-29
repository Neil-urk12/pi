use rpi_tui::{
    DefaultTheme, Frame, FrameDiff, FrameLine, MarkdownRenderer, TerminalBackend, TuiEvent,
    TurnView,
};
use std::thread;
use std::time::Duration;

#[test]
fn markdown_renderer_formats_core_markdown_surface() {
    let renderer = MarkdownRenderer::new(DefaultTheme::default());

    let frame = renderer.render(
        "# Heading\n\nA **bold** and `code` line.\n\n- [x] task\n\n> quote\n\n| A | B |\n| - | - |\n| 1 | 2 |",
        40,
    );

    assert!(frame.to_plain_text().contains("# Heading"));
    assert!(frame.to_plain_text().contains("bold"));
    assert!(frame.to_plain_text().contains("`code`"));
    assert!(frame.to_plain_text().contains("[x] task"));
    assert!(frame.to_plain_text().contains("> quote"));
    assert!(frame.to_plain_text().contains("A | B"));
}

#[test]
fn markdown_renderer_wraps_by_terminal_cell_width() {
    let renderer = MarkdownRenderer::new(DefaultTheme::default());

    let frame = renderer.render("wide 測試 words", 9);

    assert!(frame.lines().len() > 1);
    assert!(frame.lines().iter().all(|line| line.width() <= 9));
}

#[test]
fn frame_diff_reports_changed_lines_only() {
    let previous = Frame::from_plain_lines(["one", "two", "three"]);
    let next = Frame::from_plain_lines(["one", "TWO", "three"]);

    let diff = FrameDiff::between(&previous, &next);

    assert_eq!(diff.changed_rows(), &[1]);
}

#[test]
fn turn_view_coalesces_streaming_events_and_finalizes_transcript() {
    let mut view = TurnView::new(DefaultTheme::default(), 40);

    assert!(
        !view
            .apply_event(TuiEvent::AssistantDelta("hello".into()))
            .should_render
    );
    assert!(
        view.apply_event(TuiEvent::AssistantDelta("\n".into()))
            .should_render
    );
    assert!(
        view.apply_event(TuiEvent::ToolStarted {
            name: "read_file".into(),
            summary: Some("CONTEXT.md".into()),
        })
        .should_render
    );
    assert!(view.apply_event(TuiEvent::TurnFinished).should_render);

    let frame = view.render_frame();
    assert!(frame.to_plain_text().contains("hello"));
    assert!(frame.to_plain_text().contains("read_file"));
    assert!(view.is_finalized());
}

#[test]
fn turn_view_flushes_after_default_time_threshold() {
    let mut view = TurnView::new(DefaultTheme::default(), 40);

    assert!(
        !view
            .apply_event(TuiEvent::AssistantDelta("hello".into()))
            .should_render
    );

    thread::sleep(Duration::from_millis(40));

    assert!(
        view.apply_event(TuiEvent::AssistantDelta(" world".into()))
            .should_render
    );
}

#[test]
fn turn_view_renders_final_message_without_streaming() {
    let mut view = TurnView::new(DefaultTheme::default(), 40);

    let decision = view.apply_event(TuiEvent::FinalAssistantMessage(
        "A **complete** response".into(),
    ));

    assert!(decision.should_render);
    assert!(view.is_finalized());
    assert!(view.render_frame().to_plain_text().contains("complete"));
}

#[test]
fn turn_view_shows_renderer_warning_line() {
    let mut view = TurnView::new(DefaultTheme::default(), 40);

    view.apply_event(TuiEvent::AssistantDelta("plain fallback".into()));
    view.apply_event(TuiEvent::RendererWarning("shown as plain text".into()));

    let frame = view.render_frame();
    assert!(frame.to_plain_text().contains("plain fallback"));
    assert!(
        frame
            .to_plain_text()
            .contains("Renderer warning: shown as plain text")
    );
}

#[test]
fn terminal_backend_encodes_ansi_at_boundary() {
    let frame = MarkdownRenderer::new(DefaultTheme::default()).render("**bold**", 20);

    let encoded = TerminalBackend::encode_frame(&frame, true);

    assert!(encoded.contains("\x1b["));
    assert!(encoded.contains("bold"));
}

#[test]
fn markdown_renderer_drops_raw_html() {
    let renderer = MarkdownRenderer::new(DefaultTheme::default());
    let frame = renderer.render("Hello <script>alert('xss')</script> world", 40);
    let text = frame.to_plain_text();
    assert!(!text.contains("<script>"), "raw HTML should be stripped");
    assert!(text.contains("Hello"));
    assert!(text.contains("world"));
}

#[test]
fn turn_view_strips_control_chars_from_tool_summary() {
    let mut view = TurnView::new(DefaultTheme::default(), 80);
    view.apply_event(TuiEvent::ToolStarted {
        name: "read_file".into(),
        summary: Some("path\x1b[2Jevil".into()),
    });
    let frame = view.render_frame();
    let text = frame.to_plain_text();
    assert!(
        !text.contains("\x1b"),
        "control chars should be stripped from tool output"
    );
    assert!(text.contains("read_file"));
}

#[test]
fn strip_ansi_csi_sequence_no_residue() {
    let mut view = TurnView::new(DefaultTheme::default(), 80);
    view.apply_event(TuiEvent::ToolStarted {
        name: "test".into(),
        summary: Some("\x1b[2Jcleared".into()),
    });
    let text = view.render_frame().to_plain_text();
    assert!(
        !text.contains("[2J"),
        "CSI sequence residue should be stripped, got: {text:?}"
    );
    assert!(text.contains("cleared"));
}

#[test]
fn strip_ansi_color_sequence_full() {
    let mut view = TurnView::new(DefaultTheme::default(), 80);
    view.apply_event(TuiEvent::ToolStarted {
        name: "test".into(),
        summary: Some("\x1b[31mred\x1b[0m".into()),
    });
    let text = view.render_frame().to_plain_text();
    assert!(
        !text.contains("[31m"),
        "color sequence residue should be stripped, got: {text:?}"
    );
    assert!(
        !text.contains("[0m"),
        "reset sequence residue should be stripped, got: {text:?}"
    );
    assert!(text.contains("red"));
}

#[test]
fn strip_ansi_osc_sequence() {
    let mut view = TurnView::new(DefaultTheme::default(), 80);
    view.apply_event(TuiEvent::ToolStarted {
        name: "test".into(),
        summary: Some("\x1b]0;title\x07visible".into()),
    });
    let text = view.render_frame().to_plain_text();
    assert!(
        !text.contains("title"),
        "OSC payload should be stripped, got: {text:?}"
    );
    assert!(text.contains("visible"));
}

#[test]
fn strip_multiple_ansi_sequences() {
    let mut view = TurnView::new(DefaultTheme::default(), 80);
    view.apply_event(TuiEvent::ToolStarted {
        name: "test".into(),
        summary: Some("\x1b[1m\x1b[32mhello\x1b[0m world".into()),
    });
    let text = view.render_frame().to_plain_text();
    assert!(
        !text.contains("\x1b"),
        "all ESC should be stripped, got: {text:?}"
    );
    assert!(
        text.contains("hello world"),
        "should contain stripped text, got: {text:?}"
    );
}

#[test]
fn strip_preserves_whitespace() {
    let mut view = TurnView::new(DefaultTheme::default(), 80);
    view.apply_event(TuiEvent::ToolStarted {
        name: "test".into(),
        summary: Some("hello\tworld\n".into()),
    });
    let text = view.render_frame().to_plain_text();
    assert!(
        text.contains("hello\tworld\n"),
        "tabs and newlines should be preserved, got: {text:?}"
    );
}

#[test]
fn strip_dcs_sequence() {
    let mut view = TurnView::new(DefaultTheme::default(), 80);
    view.apply_event(TuiEvent::ToolStarted {
        name: "test".into(),
        summary: Some("\x1bP$evil_command\x1b\\safe".into()),
    });
    let text = view.render_frame().to_plain_text();
    assert!(
        !text.contains("evil_command"),
        "DCS body should be stripped, got: {text:?}"
    );
    assert!(text.contains("safe"));
}

#[test]
fn strip_pm_apc_sequences() {
    let mut view = TurnView::new(DefaultTheme::default(), 80);
    view.apply_event(TuiEvent::ToolStarted {
        name: "test".into(),
        summary: Some("\x1b^secret\x1b\\ok".into()),
    });
    let text = view.render_frame().to_plain_text();
    assert!(
        !text.contains("secret"),
        "PM body should be stripped, got: {text:?}"
    );
    assert!(text.contains("ok"));
}

// ---------------------------------------------------------------------------
// Layout regression tests for block spacing bugs that were fixed.
//
// These guard against three historical defects reappearing:
//   1. An empty leading line before the first block.
//   2. Double blank lines between blocks (heading→list, item→item, list→code).
//   3. A blank line inside fenced code blocks before the closing fence
//      (caused by `finish_line()` emitting an empty line when the trailing
//      newline in the code content already flushed `current`).
//
// Each test asserts the correct behaviour.  See the
// `full_document_layout_snapshot` test for the canonical expected layout.
// ---------------------------------------------------------------------------

const LAYOUT_MD: &str = "# Title\n\n- item one\n- item two\n\n```rust\nfn main() {}\n```\n\nA paragraph.";

fn render_layout() -> Vec<FrameLine> {
    MarkdownRenderer::new(DefaultTheme::default())
        .render(LAYOUT_MD, 80)
        .lines()
        .to_vec()
}

#[test]
fn heading_spacing_has_no_leading_blank_line() {
    let lines = render_layout();
    assert!(
        !lines.is_empty(),
        "rendered output must not be empty"
    );
    assert_ne!(
        lines[0].to_plain_text(),
        "",
        "first line must not be blank — heading should appear at index 0, \
         but got an empty leading line instead"
    );
    assert_eq!(
        lines[0].to_plain_text(),
        "# Title",
        "first line should be the heading"
    );
}

#[test]
fn heading_to_list_has_single_blank_separator() {
    let lines = render_layout();
    // Find the heading line.
    let heading_idx = lines
        .iter()
        .position(|l: &FrameLine| l.to_plain_text() == "# Title")
        .expect("heading line must exist");

    // The line immediately after the heading should be blank (separator).
    assert_eq!(
        lines[heading_idx + 1].to_plain_text(),
        "",
        "expected blank line after heading"
    );
    // The line after that should be the first list item — NOT another blank.
    assert_eq!(
        lines[heading_idx + 2].to_plain_text(),
        "- item one",
        "expected first list item immediately after the single blank separator; \
         a double-blank indicates a layout bug"
    );
}

#[test]
fn list_items_have_no_double_blank_between() {
    let lines = render_layout();
    let item1_idx = lines
        .iter()
        .position(|l: &FrameLine| l.to_plain_text() == "- item one")
        .expect("first list item must exist");

    // List items should be adjacent — zero blank lines between them.
    assert_eq!(
        lines[item1_idx + 1].to_plain_text(),
        "- item two",
        "list items should be adjacent with no blank line between them; \
         got a blank separator which is a layout bug"
    );
}

#[test]
fn code_block_has_no_inner_blank_before_closing_fence() {
    let lines = render_layout();
    let opening_idx = lines
        .iter()
        .position(|l: &FrameLine| l.to_plain_text() == "```")
        .expect("opening code fence must exist");

    // The line after the opening fence is the code content.
    assert_eq!(
        lines[opening_idx + 1].to_plain_text(),
        "fn main() {}",
        "code content should follow opening fence"
    );
    // The line after code content should be the closing fence — NOT a blank.
    assert_eq!(
        lines[opening_idx + 2].to_plain_text(),
        "```",
        "closing fence should immediately follow code content; \
         a blank line before the closing fence is a layout bug"
    );
}

#[test]
fn code_block_to_paragraph_has_single_blank_separator() {
    let lines = render_layout();
    // Find the closing fence (last occurrence of "```").
    let closing_idx = lines
        .iter()
        .rposition(|l: &FrameLine| l.to_plain_text() == "```")
        .expect("closing code fence must exist");

    // In a correct layout the closing fence sits at index 7 (10 lines total).
    // The buggy renderer pushes it to index 13 due to extra blank lines inside
    // the code block.  Asserting the absolute index catches that shift.
    assert_eq!(
        closing_idx, 7,
        "closing fence should be at index 7 in the correct layout, \
         but was at {closing_idx} — extra blank lines shifted it"
    );
    // Exactly one blank line between closing fence and paragraph.
    assert_eq!(
        lines[closing_idx + 1].to_plain_text(),
        "",
        "expected single blank line after closing fence"
    );
    assert_eq!(
        lines[closing_idx + 2].to_plain_text(),
        "A paragraph.",
        "paragraph should follow the single blank separator"
    );
}

#[test]
fn full_document_layout_snapshot() {
    let frame = MarkdownRenderer::new(DefaultTheme::default())
        .render(LAYOUT_MD, 80);
    let lines = frame.lines();

    // Expected layout (10 lines):
    //   0  "# Title"
    //   1  ""
    //   2  "- item one"
    //   3  "- item two"
    //   4  ""
    //   5  "```"
    //   6  "fn main() {}"
    //   7  "```"
    //   8  ""
    //   9  "A paragraph."
    let expected: &[&str] = &[
        "# Title",
        "",
        "- item one",
        "- item two",
        "",
        "```",
        "fn main() {}",
        "```",
        "",
        "A paragraph.",
    ];

    assert_eq!(
        lines.len(),
        expected.len(),
        "expected {} lines but got {}; actual lines:\n{}",
        expected.len(),
        lines.len(),
        lines
            .iter()
            .enumerate()
            .map(|(i, l)| format!("  {}: {:?}", i, l.to_plain_text()))
            .collect::<Vec<_>>()
            .join("\n"),
    );

    for (i, (line, &expected_text)) in lines.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            line.to_plain_text(),
            expected_text,
            "line {i} mismatch: expected {:?}, got {:?}",
            expected_text,
            line.to_plain_text(),
        );
    }
}

#[test]
fn nested_list_spacing_is_correct() {
    let frame = MarkdownRenderer::new(DefaultTheme::default())
        .render("- outer\n  - inner\n- next outer", 80);
    let lines = frame.lines();
    let texts: Vec<String> = lines.iter().map(|l| l.to_plain_text()).collect();

    // All three items must appear.
    assert!(
        texts.iter().any(|t| t.contains("outer")),
        "first item missing, got: {texts:?}"
    );
    assert!(
        texts.iter().any(|t| t.contains("inner")),
        "nested item missing, got: {texts:?}"
    );
    assert!(
        texts.iter().any(|t| t.contains("next outer")),
        "third item missing, got: {texts:?}"
    );

    // No double-blank between items.
    for window in texts.windows(3) {
        assert!(
            !(window[0].is_empty() && window[1].is_empty()),
            "double blank line found between items: {texts:?}"
        );
    }
}

#[test]
fn single_item_list_followed_by_paragraph() {
    let frame = MarkdownRenderer::new(DefaultTheme::default())
        .render("- only item\n\nParagraph after.", 80);
    let lines = frame.lines();

    assert_eq!(lines.len(), 3, "expected 3 lines");
    assert_eq!(lines[0].to_plain_text(), "- only item");
    assert_eq!(lines[1].to_plain_text(), "");
    assert_eq!(lines[2].to_plain_text(), "Paragraph after.");
}

#[test]
fn adjacent_unordered_then_ordered_list_spacing() {
    let frame = MarkdownRenderer::new(DefaultTheme::default())
        .render("- a\n\n1. b\n\nPara", 80);
    let lines = frame.lines();

    assert_eq!(lines.len(), 5, "expected 5 lines");
    assert_eq!(lines[0].to_plain_text(), "- a");
    assert_eq!(lines[1].to_plain_text(), "");
    assert_eq!(lines[2].to_plain_text(), "1. b");
    assert_eq!(lines[3].to_plain_text(), "");
    assert_eq!(lines[4].to_plain_text(), "Para");
}

#[test]
fn empty_document_produces_empty_frame() {
    let frame = MarkdownRenderer::new(DefaultTheme::default()).render("", 80);
    assert!(
        frame.lines().is_empty(),
        "empty input should produce empty frame, got {} lines",
        frame.lines().len()
    );
}

#[test]
fn empty_code_block_produces_no_inner_blank() {
    let frame = MarkdownRenderer::new(DefaultTheme::default())
        .render("```\n\n```", 80);
    let lines = frame.lines();
    let texts: Vec<String> = lines.iter().map(|l| l.to_plain_text()).collect();

    assert_eq!(
        texts,
        vec!["```", "```"],
        "empty code block should produce just fences, got: {texts:?}"
    );
}