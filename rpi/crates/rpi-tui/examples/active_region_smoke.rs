use rpi_tui::{Frame, TerminalBackend};
use std::io::{self, Write};
use std::thread;
use std::time::Duration;

fn main() -> io::Result<()> {
    let color = false;
    let first = Frame::from_plain_lines(["streaming **markdown**"]);
    let second =
        Frame::from_plain_lines(["streaming **markdown**", "Finished read_file: CONTEXT.md"]);

    print!(
        "{}",
        TerminalBackend::encode_active_region_update(&Frame::default(), &first, color)
    );
    io::stdout().flush()?;
    thread::sleep(Duration::from_millis(50));
    print!(
        "{}",
        TerminalBackend::encode_active_region_update(&first, &second, color)
    );
    io::stdout().flush()?;

    Ok(())
}
