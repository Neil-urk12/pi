use rpi_tui::stdin_buffer::{StdinBuffer, StdinBufferOptions, StdinEvent};
use std::time::Duration;

fn buf() -> StdinBuffer {
    StdinBuffer::new(StdinBufferOptions::default())
}

fn data_events(buffer: &mut StdinBuffer) -> Vec<Vec<u8>> {
    buffer
        .take_events()
        .into_iter()
        .filter_map(|e| match e {
            StdinEvent::Data(d) => Some(d),
            _ => None,
        })
        .collect()
}

fn paste_events(buffer: &mut StdinBuffer) -> Vec<Vec<u8>> {
    buffer
        .take_events()
        .into_iter()
        .filter_map(|e| match e {
            StdinEvent::Paste(p) => Some(p),
            _ => None,
        })
        .collect()
}

// --- Regular characters ---

#[test]
fn single_printable_char() {
    let mut b = buf();
    b.process(b"a");
    let events = data_events(&mut b);
    assert_eq!(events, vec![b"a".to_vec()]);
}

#[test]
fn multiple_printable_chars_separate_events() {
    let mut b = buf();
    b.process(b"abc");
    let events = data_events(&mut b);
    assert_eq!(events, vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()]);
}

#[test]
fn empty_input_no_events() {
    let mut b = buf();
    b.process(b"");
    assert!(b.take_events().is_empty());
}

// --- Complete escape sequences ---

#[test]
fn csi_arrow_up() {
    let mut b = buf();
    b.process(b"\x1b[A");
    let events = data_events(&mut b);
    assert_eq!(events, vec![b"\x1b[A".to_vec()]);
}

#[test]
fn csi_kitty_letter() {
    let mut b = buf();
    b.process(b"\x1b[97u");
    let events = data_events(&mut b);
    assert_eq!(events, vec![b"\x1b[97u".to_vec()]);
}

#[test]
fn ss3_arrow_up() {
    let mut b = buf();
    b.process(b"\x1bOA");
    let events = data_events(&mut b);
    assert_eq!(events, vec![b"\x1bOA".to_vec()]);
}

#[test]
fn meta_key() {
    let mut b = buf();
    b.process(b"\x1ba");
    let events = data_events(&mut b);
    assert_eq!(events, vec![b"\x1ba".to_vec()]);
}

// --- Incomplete sequences ---

#[test]
fn bare_escape_held_as_pending() {
    let mut b = buf();
    b.process(b"\x1b");
    assert!(b.take_events().is_empty());
    assert!(b.has_pending());
}

#[test]
fn incomplete_csi_held_as_pending() {
    let mut b = buf();
    b.process(b"\x1b[");
    assert!(b.take_events().is_empty());
    assert!(b.has_pending());
}

#[test]
fn incomplete_csi_completed_on_next_chunk() {
    let mut b = buf();
    b.process(b"\x1b[");
    b.process(b"A");
    let events = data_events(&mut b);
    assert_eq!(events, vec![b"\x1b[A".to_vec()]);
}

// --- Mixed content ---

#[test]
fn mixed_printable_and_escape() {
    let mut b = buf();
    b.process(b"a\x1b[Bb");
    let events = data_events(&mut b);
    assert_eq!(events, vec![b"a".to_vec(), b"\x1b[B".to_vec(), b"b".to_vec()]);
}

// --- Bracketed paste ---

#[test]
fn bracketed_paste_basic() {
    let mut b = buf();
    b.process(b"\x1b[200~hello world\x1b[201~");
    let events = paste_events(&mut b);
    assert_eq!(events, vec![b"hello world".to_vec()]);
}

#[test]
fn bracketed_paste_multiline() {
    let mut b = buf();
    b.process(b"\x1b[200~line1\nline2\nline3\x1b[201~");
    let events = paste_events(&mut b);
    assert_eq!(events, vec![b"line1\nline2\nline3".to_vec()]);
}

#[test]
fn bracketed_paste_chunked_delivery() {
    let mut b = buf();
    b.process(b"\x1b[200~hello");
    assert!(b.take_events().is_empty());
    b.process(b" world\x1b[201~");
    let events = paste_events(&mut b);
    assert_eq!(events, vec![b"hello world".to_vec()]);
}

#[test]
fn content_before_paste_emitted_separately() {
    let mut b = buf();
    b.process(b"before\x1b[200~pasted\x1b[201~");
    let all = b.take_events();
    let data: Vec<_> = all
        .iter()
        .filter_map(|e| match e {
            StdinEvent::Data(d) => Some(d.clone()),
            _ => None,
        })
        .collect();
    let pastes: Vec<_> = all
        .iter()
        .filter_map(|e| match e {
            StdinEvent::Paste(p) => Some(p.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(data.len(), 6);
    assert_eq!(pastes, vec![b"pasted".to_vec()]);
}

#[test]
fn content_after_paste_emitted() {
    let mut b = buf();
    b.process(b"\x1b[200~pasted\x1b[201~after");
    let all = b.take_events();
    let pastes: Vec<_> = all
        .iter()
        .filter_map(|e| match e {
            StdinEvent::Paste(p) => Some(p.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(pastes, vec![b"pasted".to_vec()]);
    let has_data = all.iter().any(|e| matches!(e, StdinEvent::Data(_)));
    assert!(has_data);
}

// --- High byte conversion ---

#[test]
fn high_byte_converted_to_meta() {
    let mut b = buf();
    b.process(&[0xE1]);
    let events = data_events(&mut b);
    assert_eq!(events, vec![b"\x1ba".to_vec()]);
}

// --- Clear ---

#[test]
fn clear_resets_state() {
    let mut b = buf();
    b.process(b"\x1b[");
    b.clear();
    assert!(!b.has_pending());
    assert!(b.take_events().is_empty());
}

// --- Flush ---

#[test]
fn flush_returns_buffered_content() {
    let mut b = buf();
    b.process(b"\x1b");
    let flushed = b.flush();
    assert_eq!(flushed.len(), 1);
    match &flushed[0] {
        StdinEvent::Data(d) => assert_eq!(d, b"\x1b"),
        _ => panic!("expected Data event"),
    }
}

// --- OSC sequences ---

#[test]
fn osc_sequence_with_bel_terminator() {
    let mut b = buf();
    b.process(b"\x1b]0;title\x07");
    let events = data_events(&mut b);
    assert_eq!(events, vec![b"\x1b]0;title\x07".to_vec()]);
}

#[test]
fn osc_sequence_with_st_terminator() {
    let mut b = buf();
    b.process(b"\x1b]0;title\x1b\\");
    let events = data_events(&mut b);
    assert_eq!(events, vec![b"\x1b]0;title\x1b\\".to_vec()]);
}

#[test]
fn incomplete_osc_held() {
    let mut b = buf();
    b.process(b"\x1b]0;title");
    assert!(b.take_events().is_empty());
    b.process(b"\x07");
    let events = data_events(&mut b);
    assert_eq!(events, vec![b"\x1b]0;title\x07".to_vec()]);
}

// --- DCS and APC ---

#[test]
fn dcs_sequence() {
    let mut b = buf();
    b.process(b"\x1bPq\x1b\\");
    let events = data_events(&mut b);
    assert_eq!(events.len(), 1);
}

#[test]
fn apc_sequence() {
    let mut b = buf();
    b.process(b"\x1b_some_apc\x1b\\");
    let events = data_events(&mut b);
    assert_eq!(events.len(), 1);
}

// --- Double ESC ---

#[test]
fn double_esc_emits_first_esc_standalone() {
    let mut b = buf();
    b.process(b"\x1b\x1b[A");
    let events = data_events(&mut b);
    assert_eq!(events, vec![b"\x1b".to_vec(), b"\x1b[A".to_vec()]);
}

#[test]
fn buffer_size_cap_prevents_dos() {
    let mut b = StdinBuffer::new(StdinBufferOptions {
        timeout: Duration::from_millis(10),
        max_buffer_size: 64,
    });
    // Feed incomplete escape prefix + many bytes — buffer should flush when cap hit.
    // Use 0x30 (CSI parameter byte, NOT a terminator) so the sequence stays incomplete.
    b.process(&[0x1b, b'[']); // start CSI, stays incomplete
    for _ in 0..100 {
        b.process(b"0"); // 0x30 is a parameter byte, not a final byte
    }
    let events = b.take_events();
    // Verify at least one flush was due to the buffer cap (data > max_buffer_size)
    let total_bytes: usize = events.iter().map(|e| match e {
        StdinEvent::Data(d) => d.len(),
        StdinEvent::Paste(p) => p.len(),
    }).sum();
    assert!(total_bytes >= 64, "total flushed bytes ({total_bytes}) should reach the 64-byte cap");
}

#[test]
fn paste_buffer_size_cap() {
    let mut b = StdinBuffer::new(StdinBufferOptions {
        timeout: Duration::from_millis(10),
        max_buffer_size: 64,
    });
    // Start bracketed paste
    b.process(b"\x1b[200~");
    // Feed many bytes without end marker — paste buffer should flush when cap hit
    for _ in 0..10 {
        b.process(b"abcdefghij"); // 10 bytes per iteration
    }
    let events = b.take_events();
    let paste_bytes: usize = events.iter().map(|e| match e {
        StdinEvent::Paste(p) => p.len(),
        _ => 0,
    }).sum();
    assert!(paste_bytes >= 64, "paste buffer should flush when cap exceeded, got {paste_bytes} bytes");
}
