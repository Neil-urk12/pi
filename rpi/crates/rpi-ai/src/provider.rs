//! Provider factory and registry.
//!
//! Creates the appropriate [`Provider`] implementation from a
//! [`ProviderConfig`], enabling runtime selection of LLM backends.

use rpi_core::ProviderConfig;

use crate::anthropic::AnthropicProvider;
use crate::openai::OpenAiProvider;

/// Create a boxed [`Provider`] from the given configuration.
///
/// # Errors
///
/// Returns `Err` if the configuration is invalid or the provider is not yet
/// supported.
///
/// # Examples
///
/// ```ignore
/// use rpi_ai::provider::create_provider;
/// use rpi_core::types::ProviderConfig;
///
/// let config = ProviderConfig::OpenAi {
///     api_key: "sk-...".into(),
///     base_url: None,
/// };
/// let provider = create_provider(&config)?;
/// ```
pub fn create_provider(
    config: &ProviderConfig,
) -> anyhow::Result<Box<dyn rpi_core::Provider>> {
    match config {
        ProviderConfig::OpenAi { api_key, base_url } => {
            let provider = match base_url {
                Some(url) => OpenAiProvider::with_base_url(url, Some(api_key)),
                None => OpenAiProvider::new(api_key),
            };
            Ok(Box::new(provider))
        }
        ProviderConfig::Anthropic { api_key, base_url } => {
            let provider = match base_url {
                Some(url) => AnthropicProvider::with_base_url(url, api_key),
                None => AnthropicProvider::new(api_key),
            };
            Ok(Box::new(provider))
        }
        ProviderConfig::Ollama { base_url } => {
            // Ollama exposes an OpenAI-compatible endpoint.
            Ok(Box::new(OpenAiProvider::with_base_url(
                &format!("{base_url}/v1"),
                None,
            )))
        }
    }
}

/// Return a list of all supported provider identifiers.
pub fn supported_providers() -> &'static [&'static str] {
    &["openai", "anthropic", "ollama"]
}
