use std::time::Duration;

const BRACKETED_PASTE_START: &[u8] = b"\x1b[200~";
const BRACKETED_PASTE_END: &[u8] = b"\x1b[201~";

/// Events emitted by the StdinBuffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StdinEvent {
    /// A complete key/escape sequence or printable character.
    Data(Vec<u8>),
    /// Content of a bracketed paste.
    Paste(Vec<u8>),
}

/// Configuration for the stdin buffer.
#[derive(Debug, Clone)]
pub struct StdinBufferOptions {
    /// Timeout before an incomplete escape sequence is flushed as-is. Default: 10ms.
    /// Used to disambiguate bare Escape key presses from the start of escape sequences.
    /// Currently unused — reserved for async stdin event loop integration.
    pub timeout: Duration,
    /// Maximum buffer size in bytes. When exceeded, incomplete sequences are flushed as-is.
    /// Default: 8192. Set to 0 for unlimited.
    pub max_buffer_size: usize,
}

impl Default for StdinBufferOptions {
    fn default() -> Self {
        Self {
            timeout: Duration::from_millis(10),
            max_buffer_size: 8192,
        }
    }
}

/// Buffers raw stdin bytes, splits them into complete escape sequences,
/// and handles bracketed paste mode.
pub struct StdinBuffer {
    buffer: Vec<u8>,
    paste_buffer: Vec<u8>,
    in_paste_mode: bool,
    events: Vec<StdinEvent>,
    /// Timeout for incomplete escape sequence disambiguation (e.g. bare ESC vs CSI prefix).
    /// Not yet wired to an async event loop; reserved for future raw-mode stdin integration.
    timeout: Duration,
    max_buffer_size: usize,
}

impl StdinBuffer {
    pub fn new(options: StdinBufferOptions) -> Self {
        Self {
            buffer: Vec::new(),
            paste_buffer: Vec::new(),
            in_paste_mode: false,
            events: Vec::new(),
            timeout: options.timeout,
            max_buffer_size: options.max_buffer_size,
        }
    }

    /// Feed raw bytes from stdin.
    pub fn process(&mut self, data: &[u8]) {
        if data.is_empty() && self.buffer.is_empty() {
            return;
        }

        let converted = convert_high_bytes(data);
        self.buffer.extend_from_slice(&converted);

        // Flush buffer if it exceeds the size cap (prevents DoS via unbounded growth).
        if self.max_buffer_size > 0 && self.buffer.len() > self.max_buffer_size {
            self.events.push(StdinEvent::Data(std::mem::take(&mut self.buffer)));
            return;
        }

        if self.in_paste_mode {
            self.process_paste();
            return;
        }

        if let Some(start_pos) = find_subsequence(&self.buffer, BRACKETED_PASTE_START) {
            let before = self.buffer[..start_pos].to_vec();
            self.buffer = self.buffer[start_pos + BRACKETED_PASTE_START.len()..].to_vec();
            if !before.is_empty() {
                let (events, remainder) = extract_complete_sequences(&before);
                self.events.extend(events);
                if !remainder.is_empty() {
                    self.events.push(StdinEvent::Data(remainder));
                }
            }
            self.in_paste_mode = true;
            self.process_paste();
            return;
        }

        let (events, remainder) = extract_complete_sequences(&self.buffer);
        self.events.extend(events);
        self.buffer = remainder;
    }

    /// Take all accumulated events.
    pub fn take_events(&mut self) -> Vec<StdinEvent> {
        std::mem::take(&mut self.events)
    }

    /// Force-flush any buffered content as a single data event.
    pub fn flush(&mut self) -> Vec<StdinEvent> {
        let mut events = Vec::new();
        if !self.buffer.is_empty() {
            events.push(StdinEvent::Data(std::mem::take(&mut self.buffer)));
        }
        events
    }

    /// Reset all internal state.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.paste_buffer.clear();
        self.in_paste_mode = false;
        self.events.clear();
    }

    /// Get the configured timeout duration. Currently unused; reserved for async flush integration.
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Check if the buffer has pending content.
    pub fn has_pending(&self) -> bool {
        !self.buffer.is_empty()
    }

    fn process_paste(&mut self) {
        let mut combined = std::mem::take(&mut self.paste_buffer);
        combined.extend_from_slice(&self.buffer);
        self.buffer.clear();

        // Flush paste buffer if it exceeds the size cap (prevents DoS via unbounded growth).
        if self.max_buffer_size > 0 && combined.len() > self.max_buffer_size {
            self.in_paste_mode = false;
            self.events.push(StdinEvent::Paste(combined));
            return;
        }

        if let Some(end_pos) = find_subsequence(&combined, BRACKETED_PASTE_END) {
            let paste_content = combined[..end_pos].to_vec();
            let remainder = combined[end_pos + BRACKETED_PASTE_END.len()..].to_vec();
            self.in_paste_mode = false;
            self.events.push(StdinEvent::Paste(paste_content));
            if !remainder.is_empty() {
                let (events, rem) = extract_complete_sequences(&remainder);
                self.events.extend(events);
                self.buffer = rem;
            }
        } else {
            self.paste_buffer = combined;
        }
    }
}

/// Extract complete escape sequences from `data`.
/// Returns (events, incomplete_remainder).
fn extract_complete_sequences(data: &[u8]) -> (Vec<StdinEvent>, Vec<u8>) {
    let mut events = Vec::new();
    let mut pos = 0;

    while pos < data.len() {
        if data[pos] == b'\x1b' {
            if pos + 1 >= data.len() {
                return (events, data[pos..].to_vec());
            }

            match data[pos + 1] {
                b'[' => {
                    if let Some(end) = find_csi_end(&data[pos..]) {
                        events.push(StdinEvent::Data(data[pos..pos + end].to_vec()));
                        pos += end;
                    } else {
                        return (events, data[pos..].to_vec());
                    }
                }
                b']' => {
                    if let Some(end) = find_osc_end(&data[pos..]) {
                        events.push(StdinEvent::Data(data[pos..pos + end].to_vec()));
                        pos += end;
                    } else {
                        return (events, data[pos..].to_vec());
                    }
                }
                b'P' => {
                    if let Some(end) = find_dcs_end(&data[pos..]) {
                        events.push(StdinEvent::Data(data[pos..pos + end].to_vec()));
                        pos += end;
                    } else {
                        return (events, data[pos..].to_vec());
                    }
                }
                b'_' => {
                    if let Some(end) = find_apc_end(&data[pos..]) {
                        events.push(StdinEvent::Data(data[pos..pos + end].to_vec()));
                        pos += end;
                    } else {
                        return (events, data[pos..].to_vec());
                    }
                }
                b'O' => {
                    // SS3: ESC O + 1 char
                    if pos + 2 < data.len() {
                        events.push(StdinEvent::Data(data[pos..pos + 3].to_vec()));
                        pos += 3;
                    } else {
                        return (events, data[pos..].to_vec());
                    }
                }
                b'\x1b' => {
                    // Double ESC — emit first ESC standalone
                    events.push(StdinEvent::Data(vec![b'\x1b']));
                    pos += 1;
                }
                _ => {
                    // Meta key: ESC + char
                    events.push(StdinEvent::Data(data[pos..pos + 2].to_vec()));
                    pos += 2;
                }
            }
        } else {
            // Non-escape byte
            events.push(StdinEvent::Data(vec![data[pos]]));
            pos += 1;
        }
    }

    (events, Vec::new())
}

/// Find the end of a CSI sequence starting at `data[0]` (which is `\x1b`).
fn find_csi_end(data: &[u8]) -> Option<usize> {
    if data.len() < 3 || data[1] != b'[' {
        return None;
    }
    for (i, &b) in data[2..].iter().enumerate() {
        if (0x40..=0x7E).contains(&b) {
            return Some(i + 3); // +2 for offset, +1 for length
        }
    }
    None
}

/// Find the end of an OSC sequence. Ends with ESC \ or BEL.
fn find_osc_end(data: &[u8]) -> Option<usize> {
    if data.len() < 2 || data[1] != b']' {
        return None;
    }
    for i in 2..data.len() {
        if data[i] == b'\x07' {
            return Some(i + 1);
        }
        if data[i] == b'\x1b' && i + 1 < data.len() && data[i + 1] == b'\\' {
            return Some(i + 2);
        }
    }
    None
}

/// Find the end of a DCS sequence. Ends with ESC \.
fn find_dcs_end(data: &[u8]) -> Option<usize> {
    if data.len() < 2 || data[1] != b'P' {
        return None;
    }
    for i in 2..data.len() {
        if data[i] == b'\x1b' && i + 1 < data.len() && data[i + 1] == b'\\' {
            return Some(i + 2);
        }
    }
    None
}

/// Find the end of an APC sequence. Ends with ESC \.
fn find_apc_end(data: &[u8]) -> Option<usize> {
    if data.len() < 2 || data[1] != b'_' {
        return None;
    }
    for i in 2..data.len() {
        if data[i] == b'\x1b' && i + 1 < data.len() && data[i + 1] == b'\\' {
            return Some(i + 2);
        }
    }
    None
}

/// Convert high-byte values (>127) to ESC + (byte - 128).
///
/// # Precondition
/// Input must be single-byte encoded (e.g. Latin-1 or raw terminal bytes).
/// Multi-byte UTF-8 sequences will be incorrectly split — callers passing
/// decoded text should skip this conversion.
fn convert_high_bytes(data: &[u8]) -> Vec<u8> {
    if data.iter().all(|&b| b <= 127) {
        return data.to_vec();
    }
    let mut result = Vec::with_capacity(data.len());
    for &b in data {
        if b > 127 {
            result.push(b'\x1b');
            result.push(b - 128);
        } else {
            result.push(b);
        }
    }
    result
}

/// Find the first occurrence of `needle` in `haystack`.
fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|w| w == needle)
}
