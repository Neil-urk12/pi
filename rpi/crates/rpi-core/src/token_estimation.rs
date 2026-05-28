//! Token estimation for context management.

use crate::types::{ContentBlock, Message, MessageContent, Role, Usage};

/// Content type for token estimation heuristics.
#[derive(Debug, Clone, Copy, PartialEq)]
enum ContentType {
    /// Default content (English text)
    Default,
    /// Programming code
    Code,
    /// Chinese/Japanese/Korean text
    Chinese,
    /// Technical content (JSON, XML, etc.)
    Technical,
}

/// Approximate character count per image for token estimation.
const IMAGE_APPROX_CHARS: u64 = 4800;

/// Detect content type based on text characteristics.
fn detect_content_type(text: &str) -> ContentType {
    // Check for code patterns
    if is_code(text) {
        return ContentType::Code;
    }

    // Check for Chinese characters
    if is_cjk(text) {
        return ContentType::Chinese;
    }

    // Check for technical content
    if is_technical(text) {
        return ContentType::Technical;
    }

    ContentType::Default
}

/// Check for semicolon-newline patterns that look like code (not prose).
/// Matches `42;\n`, `);\n`, `};\n`, but not `done;\n`, `however;\n`.
fn has_code_semicolons(text: &str) -> bool {
    text.lines().any(|line| {
        line.ends_with(';')
            && line.trim_end_matches(';').ends_with(|c: char| !c.is_ascii_lowercase())
    })
}
/// Check if text contains structural patterns that indicate code.
fn has_structural_patterns(text: &str) -> bool {
    // Patterns that strongly indicate code structure beyond just keywords
    text.contains("=>") ||
    text.contains("//") ||
    text.contains("/*") ||
    text.contains("*/") ||
    text.contains("; ") ||
    text.ends_with(';') ||
    has_multi_line_indent(text) ||
    has_code_semicolons(text) ||
    text.contains('\t') // tab indentation
}

/// Check if 3+ lines are indented (code blocks), avoiding false positives on
/// markdown lists which typically have fewer indented lines.
fn has_multi_line_indent(text: &str) -> bool {
    text.lines().filter(|line| line.starts_with("    ")).nth(2).is_some()
}
/// Check if text looks like programming code.
fn is_code(text: &str) -> bool {
    // Common code patterns
    let code_indicators = [
        "fn ", "let ", "const ", "var ", "function ", "class ",
        "def ", "import ", "from ", "return ", "if (", "for (",
        "while (", "match ", "switch ", "case ", "break;", "continue;",
        "//", "/*", "*/", "=>", "println!",
    ];

    let mut code_score = 0;
    for indicator in &code_indicators {
        if text.contains(indicator) {
            code_score += 1;
        }
    }

    // Require both keyword indicators AND structural patterns
    // to avoid false positives on prose that happens to contain code-like words
    code_score >= 2 && has_structural_patterns(text)
}

/// Check if text contains CJK characters (Chinese, Japanese, Korean).
fn is_cjk(text: &str) -> bool {
    text.chars().any(|c| {
        let code = c as u32;
        // CJK Unified Ideographs range
        (0x4E00..=0x9FFF).contains(&code) ||
        // CJK Extension A range
        (0x3400..=0x4DBF).contains(&code) ||
        // CJK Compatibility Ideographs range
        (0xF900..=0xFAFF).contains(&code) ||
        // Hiragana
        (0x3040..=0x309F).contains(&code) ||
        // Katakana
        (0x30A0..=0x30FF).contains(&code)
    })
}

/// Check if text looks like technical content (JSON, XML, etc.).
fn is_technical(text: &str) -> bool {
    // Require structural evidence — not just keyword counts.
    // This avoids false positives on prose that happens to contain
    // words like "true", "null", or isolated braces.
    text.contains("</")
        || text.contains("/>")
        || text.contains("<?")
        || text.contains("<![CDATA[")
        || (text.contains('{') && text.contains(':'))  // JSON-like key-value
        || text.contains("://")  // URLs with protocol
}

/// Estimate token count for a single message using chars/4 heuristic.
pub fn estimate_tokens(message: &Message) -> u32 {
    let mut chars = 0u64;
    let mut content_type = ContentType::Default;

    match &message.content {
        Some(MessageContent::Text(text)) => {
            chars += text.chars().count() as u64;
            content_type = detect_content_type(text);
        }
        Some(MessageContent::Blocks(blocks)) => {
            for block in blocks {
                match block {
                    ContentBlock::Text { text } => {
                        chars += text.chars().count() as u64;
                        // Use the most specific content type detected
                        let block_type = detect_content_type(text);
                        if block_type != ContentType::Default {
                            content_type = block_type;
                        }
                    }
                    ContentBlock::Image { .. } => chars += IMAGE_APPROX_CHARS,
                    ContentBlock::ToolUse { name, input, .. } => {
                        chars += name.chars().count() as u64;
                        chars += input.to_string().chars().count() as u64;
                    }
                    ContentBlock::ToolResult { content, .. } => {
                        chars += content.chars().count() as u64;
                    }
                }
            }
        }
        None => {}
    }
    if let Some(tool_calls) = &message.tool_calls {
        for tc in tool_calls {
            chars += tc.function.name.chars().count() as u64;
            chars += tc.function.arguments.chars().count() as u64;
        }
    }

    // Use different divisors based on content type
    let divisor = match content_type {
        ContentType::Code => 3,
        ContentType::Chinese => 2,
        ContentType::Technical => 3,
        ContentType::Default => 4,
    };

    chars.div_ceil(divisor).min(u32::MAX as u64) as u32
}

/// Estimate total context tokens for a set of messages.
/// Uses usage-based estimation when available, heuristic fallback.
pub fn estimate_context_tokens(messages: &[Message], last_usage: Option<&Usage>) -> u32 {
    match last_usage {
        Some(usage) if usage.total_tokens > 0 => {
            if let Some(last_idx) = messages.iter().rposition(|m| m.role == Role::Assistant) {
                let trailing: u32 = messages.iter().skip(last_idx + 1)
                    .map(estimate_tokens)
                    .sum();
                usage.total_tokens + trailing
            } else {
                messages.iter().map(estimate_tokens).sum()
            }
        }
        _ => messages.iter().map(estimate_tokens).sum(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{FunctionCall, ToolCall};

    fn text_message(text: &str) -> Message {
        Message {
            role: Role::User,
            content: Some(MessageContent::Text(text.to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    #[test]
    fn test_estimate_tokens_text() {
        let msg = text_message("Hello world"); // 11 chars
        assert_eq!(estimate_tokens(&msg), 3); // ceil(11/4) = 3
    }

    #[test]
    fn test_estimate_tokens_tool_call() {
        let msg = Message {
            role: Role::Assistant,
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_1".to_string(),
                function: FunctionCall {
                    name: "read_file".to_string(), // 9 chars
                    arguments: "{\"path\":\"/tmp/test.rs\"}".to_string(), // 22 chars
                },
            }]),
            tool_call_id: None,
            name: None,
        };
        // 9 + 22 = 31 chars, ceil(31/4) = 8
        assert_eq!(estimate_tokens(&msg), 8);
    }

    #[test]
    fn test_estimate_tokens_image() {
        let msg = Message {
            role: Role::User,
            content: Some(MessageContent::Blocks(vec![ContentBlock::Image {
                media_type: "image/png".to_string(),
                data: "base64data".to_string(),
            }])),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        // 4800 chars for image, ceil(4800/4) = 1200
        assert_eq!(estimate_tokens(&msg), 1200);
    }

    #[test]
    fn test_estimate_tokens_empty() {
        let msg = Message {
            role: Role::User,
            content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        assert_eq!(estimate_tokens(&msg), 0);
    }

    #[test]
    fn test_estimate_context_tokens_heuristic() {
        let messages = vec![text_message("hello"), text_message("world")];
        // "hello" = 5 chars → 2 tokens, "world" = 5 chars → 2 tokens
        assert_eq!(estimate_context_tokens(&messages, None), 4);
    }

    #[test]
    fn test_estimate_context_tokens_usage() {
        let messages = vec![
            Message {
                role: Role::Assistant,
                content: Some(MessageContent::Text("response".to_string())),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            text_message("follow up"), // 9 chars → 3 tokens trailing
        ];
        let usage = Usage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
        };
        // 150 (usage) + 3 (trailing "follow up") = 153
        assert_eq!(estimate_context_tokens(&messages, Some(&usage)), 153);
    }

    #[test]
    fn test_estimate_tokens_code() {
        // Rust code has more punctuation and shorter tokens
        let code = "fn main() {\n    println!(\"Hello, world!\");\n}";
        let msg = text_message(code);
        let estimated = estimate_tokens(&msg);
        // Code typically has ~3 chars per token due to punctuation
        // Current heuristic (chars/4) will underestimate
        let chars = code.chars().count() as u32;
        // With improved heuristic, should be closer to chars/3
        let improved_estimate = chars.div_ceil(3);
        // This test FAILS with current implementation (chars/4)
        // but PASSES with improved heuristic (chars/3)
        assert!(
            estimated >= improved_estimate,
            "Code estimation should be at least chars/3 ({}), got {}",
            improved_estimate,
            estimated
        );
    }

    #[test]
    fn test_estimate_tokens_chinese() {
        // Chinese text has ~1.5-2 chars per token
        let chinese = "你好世界，这是一个测试"; // 10 chars
        let msg = text_message(chinese);
        let estimated = estimate_tokens(&msg);
        // Current heuristic (chars/4) will significantly underestimate
        let chars = chinese.chars().count() as u32;
        // With improved heuristic, should be closer to chars/2
        let improved_estimate = chars.div_ceil(2);
        // This test FAILS with current implementation (chars/4)
        // but PASSES with improved heuristic (chars/2)
        assert!(
            estimated >= improved_estimate,
            "Chinese estimation should be at least chars/2 ({}), got {}",
            improved_estimate,
            estimated
        );
    }

    #[test]
    fn test_estimate_tokens_technical() {
        // JSON/technical content has special characters
        let json = r#"{\"key\": \"value\", \"number\": 42, \"array\": [1,2,3]}"#;
        let msg = text_message(json);
        let estimated = estimate_tokens(&msg);
        // Technical content with special characters
        let chars = json.chars().count() as u32;
        // With improved heuristic, should be closer to chars/3
        let improved_estimate = chars.div_ceil(3);
        // This test FAILS with current implementation (chars/4)
        // but PASSES with improved heuristic (chars/3)
        assert!(
            estimated >= improved_estimate,
            "Technical content estimation should be at least chars/3 ({}), got {}",
            improved_estimate,
            estimated
        );
    }

    // FINDING #1: is_code() false positives with prose containing braces
    #[test]
    fn test_is_code_false_positive_prose_with_braces() {
        let prose = "The function (which was defined earlier) returns Ok(result). \
                     Use the {key: value} pattern when needed. See also [1] and [2].";
        // Current is_code() returns true because {, }, (, ), [ ] each add to score
        assert!(!is_code(prose), "Regular prose with braces/parens should not be classified as code");
    }

    // FINDING #1b: is_code() false-positive on prose with multiple code-like keywords
    #[test]
    fn test_is_code_false_positive_prose_with_multiple_keywords() {
        // Each of these prose sentences contains >= 2 code_indicators from is_code(),
        // causing a false-positive classification as code.
        let sentences = [
            "The function returns the result to the caller",         // "function " + "return "
            "Let me import the data from the file",                 // "let " + "import " + "from "
            "We need to return to the class discussion",            // "return " + "class "
        ];
        for prose in &sentences {
            assert!(!is_code(prose),
                "Prose with multiple common English words should not be classified as code: \"{}\"",
                prose
            );
        }
    }

    #[test]
    fn test_is_code_real_code() {
        let code = "fn main() {\n    let x = 42;\n}";
        assert!(is_code(code), "Actual Rust code should be classified as code");
    }

    // FINDING #2: is_cjk() now covers Japanese kana
    #[test]
    fn test_is_cjk_detects_japanese_kana() {
        // Hiragana/Katakana (Japanese) now covered by is_cjk()
        let text = "\u{3053}\u{308c}\u{306f}\u{30c6}\u{30b9}\u{30c8}\u{3067}\u{3059}";
        let detected = is_cjk(text);
        assert!(detected, "Japanese kana should be detected by is_cjk");
    }

    // FINDING #3: is_technical() false positives with prose
    #[test]
    fn test_is_technical_false_positive_prose() {
        let prose = "She said \"hello\" to everyone.\nIt was nice.\nThe API worked.";
        assert!(!is_technical(prose), "Normal prose should not be classified as technical");
    }

    #[test]
    fn test_is_technical_real_json() {
        let json = r#"{\"name\": \"test\", \"items\": [1, 2, 3]}"#;
        assert!(is_technical(json), "JSON should be classified as technical");
    }

    // FINDING #5: Image token estimate reasonableness
    #[test]
    fn test_image_token_estimate_is_reasonable() {
        let msg = Message {
            role: Role::User,
            content: Some(MessageContent::Blocks(vec![ContentBlock::Image {
                media_type: "image/png".to_string(),
                data: "base64data".to_string(),
            }])),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let tokens = estimate_tokens(&msg);
        assert!(tokens > 0, "Image should have non-zero token estimate");
        assert!(tokens < 10000, "Image token estimate should be reasonable, got {}", tokens);
    }

    #[test]
    fn test_estimate_tokens_saturates_at_u32_max() {
        // Verify the u64→u32 cast is safe (uses .min(u32::MAX as u64))
        let msg = Message {
            role: Role::User,
            content: Some(MessageContent::Text("x".repeat(1_000_000))),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let tokens = estimate_tokens(&msg);
        // 1_000_000 / 4 = 250_000 — well within u32 range
        assert_eq!(tokens, 250_000);
        let _: u32 = tokens; // type check
    }

    #[test]
    fn test_has_structural_patterns_markdown_not_code() {
        let markdown = "Here is a list:\n\n    - Item one\n    - Item two\n    - Item three\n\nAnd some more text.";
        assert!(!is_code(markdown), "Markdown with 4-space indented list items should not be classified as code");
    }

    // FINDING: has_structural_patterns `;\n` false positive on prose
    #[test]
    fn test_has_structural_patterns_prose_with_semicolons() {
        // Prose sentences can have semicolons ending a clause before a newline.
        // ";\n" is not exclusive to code — it appears in natural language too.
        let prose = "done;\nhowever, other things happen next.";
        assert!(!has_structural_patterns(prose),
            "Prose with semicolon-before-newline should not trigger structural code detection: {:?}",
            prose
        );
    }

    // FINDING: has_multi_line_indent refactor — positive case preservation
    #[test]
    fn test_has_structural_patterns_multiple_indented_lines() {
        // Text with 3+ lines of 4-space indentation should be detected as structural.
        // This verifies the filter().take(3).count() >= 3 logic (or its nth(2) refactor).
        let indented = "let a = 1;\n    let b = 2;\n    let c = 3;\n    let d = 4;";
        assert!(has_structural_patterns(indented),
            "Text with 3+ indented lines should be detected as having structural patterns"
        );
    }

}
