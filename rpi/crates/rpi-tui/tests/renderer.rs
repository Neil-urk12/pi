use rpi_tui::{
    DefaultTheme, Frame, FrameDiff, MarkdownRenderer, TerminalBackend, TuiEvent, TurnView,
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
    assert!(!text.contains("\x1b"), "control chars should be stripped from tool output");
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
    assert!(!text.contains("[2J"), "CSI sequence residue should be stripped, got: {text:?}");
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
    assert!(!text.contains("[31m"), "color sequence residue should be stripped, got: {text:?}");
    assert!(!text.contains("[0m"), "reset sequence residue should be stripped, got: {text:?}");
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
    assert!(!text.contains("title"), "OSC payload should be stripped, got: {text:?}");
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
    assert!(!text.contains("\x1b"), "all ESC should be stripped, got: {text:?}");
    assert!(text.contains("hello world"), "should contain stripped text, got: {text:?}");
}

#[test]
fn strip_preserves_whitespace() {
    let mut view = TurnView::new(DefaultTheme::default(), 80);
    view.apply_event(TuiEvent::ToolStarted {
        name: "test".into(),
        summary: Some("hello\tworld\n".into()),
    });
    let text = view.render_frame().to_plain_text();
    assert!(text.contains("hello\tworld\n"), "tabs and newlines should be preserved, got: {text:?}");
}

#[test]
fn strip_dcs_sequence() {
    let mut view = TurnView::new(DefaultTheme::default(), 80);
    view.apply_event(TuiEvent::ToolStarted {
        name: "test".into(),
        summary: Some("\x1bP$evil_command\x1b\\safe".into()),
    });
    let text = view.render_frame().to_plain_text();
    assert!(!text.contains("evil_command"), "DCS body should be stripped, got: {text:?}");
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
    assert!(!text.contains("secret"), "PM body should be stripped, got: {text:?}");
    assert!(text.contains("ok"));
}
