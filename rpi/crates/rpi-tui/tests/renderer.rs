use rpi_tui::{
    Color, DefaultTheme, Frame, FrameDiff, FrameLine, MarkdownRenderer, StyledSpan, TerminalBackend,
    TuiEvent, TurnView,
};
use std::thread;
use std::time::Duration;
use unicode_width::UnicodeWidthStr;

#[test]
fn markdown_renderer_formats_core_markdown_surface() {
    let renderer = MarkdownRenderer::new(DefaultTheme::default());

    let frame = renderer.render(
        "# Heading\n\nA **bold** and `code` line.\n\n- [x] task\n\n> quote\n\n| A | B |\n| - | - |\n| 1 | 2 |",
        40,
    );

    assert!(frame.to_plain_text().contains("Heading"));
    assert!(frame.to_plain_text().contains("bold"));
    assert!(frame.to_plain_text().contains("`code`"));
    assert!(frame.to_plain_text().contains("[x] task"));
    assert!(frame.to_plain_text().contains("> quote"));
    assert!(frame.to_plain_text().contains("│"));
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
        "Title",
        "first line should be the heading"
    );
}

#[test]
fn heading_to_list_has_single_blank_separator() {
    let lines = render_layout();
    // Find the heading line.
    let heading_idx = lines
        .iter()
        .position(|l: &FrameLine| l.to_plain_text() == "Title")
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
        "  fn main() {}",
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
    //   0  "Title"
    //   1  ""
    //   2  "- item one"
    //   3  "- item two"
    //   4  ""
    //   5  "```"
    //   6  "  fn main() {}"
    //   7  "```"
    //   8  ""
    //   9  "A paragraph."
    let expected: &[&str] = &[
        "Title",
        "",
        "- item one",
        "- item two",
        "",
        "```",
        "  fn main() {}",
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

// ---------------------------------------------------------------------------
// Tests for renderer improvements: heading stripping, code indent, table box-drawing.
// ---------------------------------------------------------------------------

#[test]
fn h1_heading_strips_hash_prefix() {
    let renderer = MarkdownRenderer::new(DefaultTheme::default());
    let frame = renderer.render("# Hello", 80);
    let text = frame.to_plain_text();
    assert_eq!(text, "Hello", "h1 should strip '# ' prefix");
}

#[test]
fn h2_heading_strips_hash_prefix() {
    let renderer = MarkdownRenderer::new(DefaultTheme::default());
    let frame = renderer.render("## Python", 80);
    let text = frame.to_plain_text();
    assert_eq!(text, "Python", "h2 should strip '## ' prefix");
}

#[test]
fn h3_heading_keeps_hash_prefix() {
    let renderer = MarkdownRenderer::new(DefaultTheme::default());
    let frame = renderer.render("### Details", 80);
    let text = frame.to_plain_text();
    assert_eq!(text, "### Details", "h3+ should keep '### ' prefix");
}

#[test]
fn code_block_content_is_indented() {
    let renderer = MarkdownRenderer::new(DefaultTheme::default());
    let frame = renderer.render("```\nfn main() {}\n```", 80);
    let lines = frame.lines();
    assert_eq!(lines.len(), 3);
    assert_eq!(lines[0].to_plain_text(), "```");
    assert_eq!(
        lines[1].to_plain_text(),
        "  fn main() {}",
        "code content should be indented with 2 spaces"
    );
    assert_eq!(lines[2].to_plain_text(), "```");
}

#[test]
fn code_block_indent_respects_width() {
    let renderer = MarkdownRenderer::new(DefaultTheme::default());
    let frame = renderer.render("```\nabcdefghijklmnopqrstuvwxyz\n```", 20);
    let lines = frame.lines();
    for line in lines {
        assert!(
            line.width() <= 20,
            "line width {} exceeds 20: {:?}",
            line.width(),
            line.to_plain_text()
        );
    }
    assert!(
        lines[1].to_plain_text().starts_with("  "),
        "code should start with indent"
    );
}

#[test]
fn table_renders_box_drawing() {
    let renderer = MarkdownRenderer::new(DefaultTheme::default());
    let frame = renderer.render("| A | B |\n| - | - |\n| 1 | 2 |", 80);
    let text = frame.to_plain_text();
    assert!(text.contains('\u{250C}'), "should have top-left corner, got: {text}");
    assert!(text.contains('\u{2510}'), "should have top-right corner");
    assert!(text.contains('\u{2514}'), "should have bottom-left corner");
    assert!(text.contains('\u{2518}'), "should have bottom-right corner");
    assert!(text.contains('\u{2502}'), "should have vertical borders");
    assert!(text.contains('\u{2500}'), "should have horizontal borders");
    assert!(text.contains('\u{252C}'), "should have top tee");
    assert!(text.contains('\u{2534}'), "should have bottom tee");
    assert!(text.contains('\u{253C}'), "should have cross");
    assert!(text.contains('\u{251C}'), "should have left tee");
    assert!(text.contains('\u{2524}'), "should have right tee");
}

#[test]
fn table_contains_cell_content() {
    let renderer = MarkdownRenderer::new(DefaultTheme::default());
    let frame = renderer.render("| A | B |\n| - | - |\n| 1 | 2 |", 80);
    let text = frame.to_plain_text();
    assert!(text.contains("A"), "should contain header A");
    assert!(text.contains("B"), "should contain header B");
    assert!(text.contains("1"), "should contain data 1");
    assert!(text.contains("2"), "should contain data 2");
}

#[test]
fn table_lines_fit_within_width() {
    let renderer = MarkdownRenderer::new(DefaultTheme::default());
    let frame =
        renderer.render("| Name | Value |\n| ---- | ----- |\n| hello | world |", 40);
    for line in frame.lines() {
        assert!(
            line.width() <= 40,
            "line width {} exceeds 40: {:?}",
            line.width(),
            line.to_plain_text()
        );
    }
}

#[test]
fn table_cjk_columns_aligned() {
    let renderer = MarkdownRenderer::new(DefaultTheme::default());
    let frame = renderer.render("| 名前 | 値 |\n| -- | -- |\n| 田中太郎 | 123 |", 80);
    let lines: Vec<String> = frame.lines().iter().map(|l| l.to_plain_text()).collect();
    // All table lines should have identical display width
    // CJK chars are 3 bytes but 2 display columns; String::len() will be wrong
    let widths: Vec<usize> = lines
        .iter()
        .map(|l| UnicodeWidthStr::width(l.as_str()))
        .collect();
    let first = widths[0];
    for (i, &w) in widths.iter().enumerate() {
        assert_eq!(
            w, first,
            "CJK table columns misaligned: line {i} display width {w} != {first}"
        );
    }
}

#[test]
fn table_clamps_to_terminal_width() {
    let renderer = MarkdownRenderer::new(DefaultTheme::default());
    // Table with content wider than 30 columns
    let frame = renderer.render(
        "| Name | Value | Description |\n| ---- | ----- | ----------- |\n| alpha | 1 | this is a very long description |",
        30,
    );
    let lines: Vec<String> = frame.lines().iter().map(|l| l.to_plain_text()).collect();
    // All lines must fit within terminal width
    for (i, line) in lines.iter().enumerate() {
        let w = UnicodeWidthStr::width(line.as_str());
        assert!(
            w <= 30,
            "table line {i} width {w} exceeds terminal width 30: {line:?}"
        );
    }
    // Table structure must be preserved: each row has matching │ vertical borders
    // A 3-column table should have exactly 4 │ per data/border row (left + 3 interior + right)
    for (i, line) in lines.iter().enumerate() {
        let pipe_count = line.chars().filter(|&c| c == '\u{2502}').count();
        // Border lines (┌─┬─┐ etc.) use different chars but should have consistent structure
        if pipe_count > 0 {
            assert_eq!(
                pipe_count, 4,
                "table line {i} should have 4 vertical borders, got {pipe_count}: {line:?}"
            );
        }
    }
}

#[test]
fn code_block_indent_multiline_content() {
    let renderer = MarkdownRenderer::new(DefaultTheme::default());
    let frame = renderer.render("```rust\nfn main() {\n    println!(\"hello\");\n}\n```", 80);
    let lines: Vec<String> = frame.lines().iter().map(|l| l.to_plain_text()).collect();
    
    // Expected: opening fence, 3 indented lines, closing fence
    assert_eq!(lines.len(), 5, "expected 5 lines, got {}: {lines:?}", lines.len());
    assert_eq!(lines[0], "```");
    assert_eq!(lines[1], "  fn main() {", "line 1 should be indented");
    assert_eq!(lines[2], "      println!(\"hello\");", "line 2 should be indented (original 4 spaces + 2 prefix)");
    assert_eq!(lines[3], "  }", "line 3 should be indented");
    assert_eq!(lines[4], "```");
}

#[test]
fn full_layout_with_multiline_code_block() {
    let md = "# Title\n\n```rust\nfn main() {\n    println!(\"hi\");\n}\n```\n\nParagraph.";
    let frame = MarkdownRenderer::new(DefaultTheme::default()).render(md, 80);
    let lines: Vec<String> = frame.lines().iter().map(|l| l.to_plain_text()).collect();
    
    // Expected layout (9 lines):
    //   0  "Title"
    //   1  ""
    //   2  "```"
    //   3  "  fn main() {"
    //   4  "      println!(\"hi\");"
    //   5  "  }"
    //   6  "```"
    //   7  ""
    //   8  "Paragraph."
    let expected: &[&str] = &[
        "Title",
        "",
        "```",
        "  fn main() {",
        "      println!(\"hi\");",
        "  }",
        "```",
        "",
        "Paragraph.",
    ];
    
    assert_eq!(lines.len(), expected.len(), "line count mismatch:\nactual: {lines:#?}");
    for (i, (actual, &exp)) in lines.iter().zip(expected.iter()).enumerate() {
        assert_eq!(actual, exp, "line {i} mismatch");
    }
}


#[test]
fn code_block_preserves_blank_line_between_functions() {
    let renderer = MarkdownRenderer::new(DefaultTheme::default());
    let frame = renderer.render("```rust\nfn a() {}\n\nfn b() {}\n```", 80);
    let lines: Vec<String> = frame.lines().iter().map(|l| l.to_plain_text()).collect();
    
    // Expected: ``` / "  fn a() {}" / "" / "  fn b() {}" / ```
    assert_eq!(lines.len(), 5, "expected 5 lines, got {}: {lines:?}", lines.len());
    assert_eq!(lines[0], "```");
    assert_eq!(lines[1], "  fn a() {}");
    assert_eq!(lines[2], "", "blank line in code block should be preserved");
    assert_eq!(lines[3], "  fn b() {}");
    assert_eq!(lines[4], "```");
}

#[test]
fn code_block_in_narrow_terminal_does_not_panic() {
    let renderer = MarkdownRenderer::new(DefaultTheme::default());
    // width=3: "  " prefix (2) + 1 char fits, but barely
    let frame = renderer.render("```\nabc\n```", 3);
    let text = frame.to_plain_text();
    assert!(text.contains("abc"), "code content must survive narrow terminal, got: {text}");
}

#[test]
fn heading_text_has_heading_style() {
    let renderer = MarkdownRenderer::new(DefaultTheme::default());
    let frame = renderer.render("# Hello", 80);

    let lines = frame.lines();
    let hello_span = lines
        .iter()
        .flat_map(|l| l.spans().iter())
        .find(|s| s.text().contains("Hello"))
        .expect("span with 'Hello' must exist");

    let style = hello_span.style();
    assert_eq!(style.foreground, Color::Cyan, "heading foreground should be Cyan");
    assert!(style.bold, "heading should be bold");
    assert!(style.underline, "heading should be underline");
}

#[test]
fn h2_text_has_heading_style() {
    let renderer = MarkdownRenderer::new(DefaultTheme::default());
    let frame = renderer.render("## World", 80);

    let lines = frame.lines();
    let world_span = lines
        .iter()
        .flat_map(|l| l.spans().iter())
        .find(|s| s.text().contains("World"))
        .expect("span with 'World' must exist");

    let style = world_span.style();
    assert_eq!(style.foreground, Color::Cyan, "heading foreground should be Cyan");
    assert!(style.bold, "heading should be bold");
    assert!(style.underline, "heading should be underline");
}

#[test]
fn heading_inside_blockquote_preserves_quote_style_after() {
    let renderer = MarkdownRenderer::new(DefaultTheme::default());
    let frame = renderer.render("> # Title\n> text after heading", 80);

    let lines = frame.lines();
    let after_span = lines
        .iter()
        .flat_map(|l| l.spans().iter())
        .find(|s| s.text().contains("text after heading"))
        .expect("span with 'text after heading' must exist");

    let style = after_span.style();
    assert_eq!(
        style.foreground,
        Color::Blue,
        "text after heading inside blockquote should have quote foreground (Blue)"
    );
    assert!(
        style.italic,
        "text after heading inside blockquote should be italic (quote style)"
    );
    assert!(
        !style.bold,
        "text after heading should not retain heading bold style"
    );
}

#[test]
fn table_width_never_exceeds_terminal_for_many_wide_columns() {
    let renderer = MarkdownRenderer::new(DefaultTheme::default());
    // 6 wide columns on a 25-char terminal — stress test for proportional clamping
    let md = "| Col1 | Col2 | Col3 | Col4 | Col5 | Col6 |
| ---- | ---- | ---- | ---- | ---- | ---- |
| aaa | bbb | ccc | ddd | eee | fff |";
    let frame = renderer.render(md, 25);
    for (i, line) in frame.lines().iter().enumerate() {
        let w = UnicodeWidthStr::width(line.to_plain_text().as_str());
        assert!(
            w <= 25,
            "line {i} display width {w} exceeds 25: {:?}",
            line.to_plain_text()
        );
    }
    // Verify table structure preserved: data rows must have 7 │ chars (6 cols + borders)
    let texts: Vec<String> = frame.lines().iter().map(|l| l.to_plain_text()).collect();
    for (i, text) in texts.iter().enumerate() {
        let pipe_count = text.chars().filter(|&c| c == '\u{2502}').count();
        if pipe_count > 0 {
            assert_eq!(
                pipe_count, 7,
                "line {i} should have 7 vertical borders for 6-column table, got {pipe_count}: {text:?}"
            );
        }
    }
}

#[test]
fn table_no_zero_width_columns() {
    // Regression: when available < num_cols - 1, last column could get width=0,
    // truncating cell content.
    let renderer = MarkdownRenderer::new(DefaultTheme::default());
    // 5 columns, terminal width 40
    // border_overhead = 5*3 + 1 = 16, available = 24
    // First col has wide content forcing proportional clamping
    let md = "| verylongcontentthatexceedswidth | a | b | c | d |\n| --- | --- | --- | --- | --- |\n| z | z | z | z | z |";
    let frame = renderer.render(md, 40);
    let lines: Vec<String> = frame.lines().iter().map(|l| l.to_plain_text()).collect();

    // All lines must fit within terminal width
    for (i, line) in lines.iter().enumerate() {
        let w = unicode_width::UnicodeWidthStr::width(line.as_str());
        assert!(
            w <= 40,
            "line {i} width {w} exceeds 40: {line:?}"
        );
    }

    // Find data row (contains 'z') and verify all 5 cells render their content.
    // With zero-width last column, the 'z' in column 5 would be truncated.
    let data_line = lines
        .iter()
        .find(|l| l.contains('z'))
        .expect("should have a data row");
    let parts: Vec<&str> = data_line.split('\u{2502}').collect();
    // split on │: ["", " content ", " content ", ..., ""] for 5 columns = 7 parts
    assert_eq!(
        parts.len(), 7,
        "data row should split into 7 parts (5 cells + 2 edges), got {}: {data_line:?}",
        parts.len()
    );
    // Every cell should contain its expected content, not just whitespace.
    // A zero-width column truncates content, leaving only padding spaces.
    for (j, part) in parts.iter().enumerate() {
        if j > 0 && j < parts.len() - 1 {
            assert!(
                part.contains('z'),
                "data cell {j} should contain 'z', got {part:?} — column likely has zero width"
            );
        }
    }
}

#[test]
fn table_cell_content_has_text_style_not_border_style() {
    let renderer = MarkdownRenderer::new(DefaultTheme::default());
    let frame = renderer.render("| A | B |\n| - | - |\n| 1 | 2 |", 80);

    // Collect all spans from rendered table
    let all_spans: Vec<&StyledSpan> = frame
        .lines()
        .iter()
        .flat_map(|l| l.spans().iter())
        .collect();

    // Regression: cell content must use text style (Default), not border style (Cyan).
    // Previously, cell_line built the entire row as one string with a single style.

    // Find a span that contains header cell content "A"
    let a_span = all_spans
        .iter()
        .find(|s| s.text().contains('A'))
        .expect("span containing 'A' must exist");
    assert_eq!(
        a_span.style().foreground,
        Color::Default,
        "header cell 'A' should have text style (Default), not border style (Cyan)"
    );

    // Find a span that contains header cell content "B"
    let b_span = all_spans
        .iter()
        .find(|s| s.text().contains('B'))
        .expect("span containing 'B' must exist");
    assert_eq!(
        b_span.style().foreground,
        Color::Default,
        "header cell 'B' should have text style (Default), not border style (Cyan)"
    );

    // Find a span that contains data cell content "1"
    let one_span = all_spans
        .iter()
        .find(|s| s.text().contains('1'))
        .expect("span containing '1' must exist");
    assert_eq!(
        one_span.style().foreground,
        Color::Default,
        "data cell '1' should have text style (Default), not border style (Cyan)"
    );

    // Find a span that contains data cell content "2"
    let two_span = all_spans
        .iter()
        .find(|s| s.text().contains('2'))
        .expect("span containing '2' must exist");
    assert_eq!(
        two_span.style().foreground,
        Color::Default,
        "data cell '2' should have text style (Default), not border style (Cyan)"
    );

    // Border characters should still have Cyan (table_border style)
    let border_span = all_spans
        .iter()
        .find(|s| s.text().contains('\u{250C}') || s.text().contains('\u{2500}'))
        .expect("span with border chars must exist");
    assert_eq!(
        border_span.style().foreground,
        Color::Cyan,
        "border characters should retain table_border style (Cyan)"
    );
}

#[test]
fn table_with_no_data_rows_has_no_header_separator() {
    let renderer = MarkdownRenderer::new(DefaultTheme::default());
    // Header row + separator row in markdown, but NO data rows
    let frame = renderer.render("| A | B |\n| - | - |", 80);
    let text = frame.to_plain_text();

    // Top border must be present
    assert!(
        text.contains('\u{250C}'),
        "should have top-left corner (top border), got: {text}"
    );
    // Header content must be present
    assert!(text.contains("A"), "should contain header A, got: {text}");
    assert!(text.contains("B"), "should contain header B, got: {text}");
    // Bottom border must be present
    assert!(
        text.contains('\u{2514}'),
        "should have bottom-left corner (bottom border), got: {text}"
    );
    // Header separator (left tee) must NOT be present — no data rows to separate from
    assert!(
        !text.contains('\u{251C}'),
        "header separator (left tee) should not appear when there are no data rows, got: {text}"
    );
}