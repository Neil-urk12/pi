use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use rpi_core::compaction::CompactionSettings;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Default model to use (provider/model format)
    pub default_model: Option<String>,

    /// Provider API keys
    #[serde(default)]
    pub api_keys: HashMap<String, String>,

    /// Provider base URLs (for custom endpoints)
    #[serde(default)]
    pub base_urls: HashMap<String, String>,

    /// Default system prompt
    pub system_prompt: Option<String>,

    /// Max tokens for responses
    pub max_tokens: Option<u32>,

    /// Temperature
    pub temperature: Option<f32>,

    /// Compaction settings (optional)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compaction: Option<CompactionConfig>,
}

impl Config {
    pub fn load() -> Result<Self> {
        let config_path = Self::config_path()?;
        if config_path.exists() {
            let content = fs::read_to_string(&config_path)
                .with_context(|| format!("Failed to read config at {}", config_path.display()))?;
            serde_json::from_str(&content)
                .with_context(|| format!("Failed to parse config at {}", config_path.display()))
        } else {
            Ok(Self::default())
        }
    }

    pub fn save(&self) -> Result<()> {
        let config_path = Self::config_path()?;
        if let Some(parent) = config_path.parent() {
            fs::create_dir_all(parent)?;

            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(parent, fs::Permissions::from_mode(0o700))?;
            }
        }
        let content = serde_json::to_string_pretty(self)?;
        fs::write(&config_path, content)?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&config_path, fs::Permissions::from_mode(0o600))?;
        }

        Ok(())
    }

    pub fn set(&self, key: &str, value: &str) -> Result<Config> {
        let mut config = self.clone();
        match key {
            "default_model" => config.default_model = Some(value.to_string()),
            "system_prompt" => config.system_prompt = Some(value.to_string()),
            "max_tokens" => config.max_tokens = Some(value.parse()?),
            "temperature" => config.temperature = Some(value.parse()?),
            k if k.starts_with("api_key.") => {
                let provider = k.strip_prefix("api_key.").unwrap();
                config
                    .api_keys
                    .insert(provider.to_string(), value.to_string());
            }
            k if k.starts_with("base_url.") => {
                let provider = k.strip_prefix("base_url.").unwrap();
                config
                    .base_urls
                    .insert(provider.to_string(), value.to_string());
            }
            k if k.starts_with("compaction.") => {
                let field = k.strip_prefix("compaction.").unwrap();
                if config.compaction.is_none() {
                    config.compaction = Some(CompactionConfig {
                        enabled: false,
                        ..CompactionConfig::default()
                    });
                }
                let compaction = config.compaction.as_mut().unwrap();
                match field {
                    "enabled" => compaction.enabled = value.parse()?,
                    "context_window" => compaction.context_window = Some(value.parse()?),
                    "reserve_tokens" => compaction.reserve_tokens = value.parse()?,
                    "keep_recent_tokens" => compaction.keep_recent_tokens = value.parse()?,
                    _ => anyhow::bail!("Unknown compaction config key: {field}"),
                }
            }
            _ => anyhow::bail!("Unknown config key: {key}"),
        }
        Ok(config)
    }

    pub fn get_api_key(&self, provider: &str) -> Option<&str> {
        self.api_keys.get(provider).map(|s| s.as_str())
    }

    pub fn get_base_url(&self, provider: &str) -> Option<&str> {
        self.base_urls.get(provider).map(|s| s.as_str())
    }

    fn config_path() -> Result<PathBuf> {
        let config_dir = dirs::config_dir().context("Could not determine config directory")?;
        Ok(config_dir.join("rpi").join("config.json"))
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            default_model: Some("openai/gpt-4o".to_string()),
            api_keys: HashMap::new(),
            base_urls: HashMap::new(),
            system_prompt: Some("You are a helpful coding assistant.".to_string()),
            max_tokens: Some(4096),
            temperature: Some(0.7),
            compaction: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionConfig {
    /// Whether compaction is enabled (default: true)
    #[serde(default = "default_compaction_enabled")]
    pub enabled: bool,
    /// Context window size in tokens (optional override)
    pub context_window: Option<u32>,
    /// Token budget reserved for model response (default: 16384)
    #[serde(default = "default_reserve_tokens")]
    pub reserve_tokens: u32,
    /// Tokens to keep from recent messages (default: 20000)
    #[serde(default = "default_keep_recent_tokens")]
    pub keep_recent_tokens: u32,
}

fn default_compaction_enabled() -> bool {
    false
}

fn default_reserve_tokens() -> u32 {
    16_384
}

fn default_keep_recent_tokens() -> u32 {
    20_000
}

impl CompactionConfig {
    pub fn to_core_settings(&self) -> CompactionSettings {
        CompactionSettings {
            reserve_tokens: self.reserve_tokens,
            keep_recent_tokens: self.keep_recent_tokens,
            enabled: self.enabled,
            context_window: self.context_window,
        }
    }
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            context_window: None,
            reserve_tokens: 16_384,
            keep_recent_tokens: 20_000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setting_compaction_field_does_not_implicitly_enable() {
        let config = Config::default();
        assert!(config.compaction.is_none());

        let updated = config.set("compaction.reserve_tokens", "1000").unwrap();

        let compaction = updated
            .compaction
            .as_ref()
            .expect("compaction should be Some after setting a sub-field");
        assert!(
            !compaction.enabled,
            "compaction should not be implicitly enabled"
        );
        assert_eq!(compaction.reserve_tokens, 1000);
    }
}
