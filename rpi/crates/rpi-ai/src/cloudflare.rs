//! Cloudflare provider URL helpers.

use rpi_core::{PiError, Result};

/// Workers AI direct OpenAI-compatible endpoint template.
pub const CLOUDFLARE_WORKERS_AI_BASE_URL: &str =
    "https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1";

/// AI Gateway Unified API compatible endpoint template.
pub const CLOUDFLARE_AI_GATEWAY_COMPAT_BASE_URL: &str =
    "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/compat";

/// AI Gateway OpenAI passthrough endpoint template.
pub const CLOUDFLARE_AI_GATEWAY_OPENAI_BASE_URL: &str =
    "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/openai";

/// AI Gateway Anthropic passthrough endpoint template.
pub const CLOUDFLARE_AI_GATEWAY_ANTHROPIC_BASE_URL: &str = "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/anthropic";

/// Return whether a provider id belongs to Cloudflare.
pub fn is_cloudflare_provider(provider: &str) -> bool {
    matches!(provider, "cloudflare-workers-ai" | "cloudflare-ai-gateway")
}

/// Resolve Cloudflare `{ENV_VAR}` placeholders using the process environment.
pub fn resolve_cloudflare_base_url(base_url: &str, provider: &str) -> Result<String> {
    resolve_cloudflare_base_url_with(base_url, provider, |name| std::env::var(name).ok())
}

/// Resolve Cloudflare `{ENV_VAR}` placeholders using a caller-supplied lookup.
pub fn resolve_cloudflare_base_url_with(
    base_url: &str,
    provider: &str,
    lookup: impl Fn(&str) -> Option<String>,
) -> Result<String> {
    let mut resolved = String::with_capacity(base_url.len());
    let mut cursor = 0;

    while let Some(relative_start) = base_url[cursor..].find('{') {
        let start = cursor + relative_start;
        resolved.push_str(&base_url[cursor..start]);

        let Some(relative_end) = base_url[start + 1..].find('}') else {
            resolved.push_str(&base_url[start..]);
            return Ok(resolved);
        };
        let end = start + 1 + relative_end;
        let name = &base_url[start + 1..end];

        if is_env_placeholder(name) {
            let value = lookup(name).ok_or_else(|| {
                PiError::Config(format!(
                    "{name} is required for provider {provider} but is not set."
                ))
            })?;
            resolved.push_str(&value);
        } else {
            resolved.push_str(&base_url[start..=end]);
        }

        cursor = end + 1;
    }

    resolved.push_str(&base_url[cursor..]);
    Ok(resolved)
}

fn is_env_placeholder(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };

    (first.is_ascii_uppercase() || first == '_')
        && chars.all(|character| {
            character.is_ascii_uppercase() || character.is_ascii_digit() || character == '_'
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_cloudflare_provider_ids() {
        assert!(is_cloudflare_provider("cloudflare-workers-ai"));
        assert!(is_cloudflare_provider("cloudflare-ai-gateway"));
        assert!(!is_cloudflare_provider("openai"));
    }

    #[test]
    fn leaves_urls_without_placeholders_unchanged() {
        let url = "https://example.com/v1";

        let resolved = resolve_cloudflare_base_url_with(url, "cloudflare-workers-ai", |_| None)
            .expect("plain URL should resolve");

        assert_eq!(resolved, url);
    }

    #[test]
    fn substitutes_cloudflare_environment_placeholders() {
        let resolved = resolve_cloudflare_base_url_with(
            CLOUDFLARE_AI_GATEWAY_COMPAT_BASE_URL,
            "cloudflare-ai-gateway",
            |name| match name {
                "CLOUDFLARE_ACCOUNT_ID" => Some("account-123".to_string()),
                "CLOUDFLARE_GATEWAY_ID" => Some("gateway-456".to_string()),
                _ => None,
            },
        )
        .expect("gateway URL should resolve");

        assert_eq!(
            resolved,
            "https://gateway.ai.cloudflare.com/v1/account-123/gateway-456/compat"
        );
    }

    #[test]
    fn reports_missing_cloudflare_environment_placeholder() {
        let error = resolve_cloudflare_base_url_with(
            CLOUDFLARE_WORKERS_AI_BASE_URL,
            "cloudflare-workers-ai",
            |_| None,
        )
        .expect_err("missing env var should fail");

        assert_eq!(
            error.to_string(),
            "config error: CLOUDFLARE_ACCOUNT_ID is required for provider cloudflare-workers-ai but is not set."
        );
    }
}
