//! Built-in tool implementations.
//!
//! Each module provides a single [`rpi_core::Tool`] implementation:
//!
//! | Module         | Tool name      | Description                         |
//! |----------------|----------------|-------------------------------------|
//! | [`read_file`]  | `read_file`    | Read file contents with line numbers|
//! | [`write_file`] | `write_file`   | Write content to a file             |
//! | [`edit_file`]  | `edit_file`    | Text replacement in a file          |
//! | [`bash`]       | `bash`         | Execute shell commands              |
//! | [`grep`]       | `grep`         | Regex search across files           |
//! | [`ls`]         | `ls`           | Directory listing                   |
//! | [`find`]       | `find`         | Find files by glob pattern          |

pub mod bash;
pub mod edit_file;
pub mod grep;
pub mod find;
pub mod ls;
pub mod read_file;
pub mod write_file;
