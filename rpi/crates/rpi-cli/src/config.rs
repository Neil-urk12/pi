use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
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
        }
        let content = serde_json::to_string_pretty(self)?;
        fs::write(&config_path, content)?;
        Ok(())
    }

    pub fn set(&self, key: &str, value: &str) -> Result<()> {
        let mut config = self.clone();
        match key {
            "default_model" => config.default_model = Some(value.to_string()),
            "system_prompt" => config.system_prompt = Some(value.to_string()),
            "max_tokens" => config.max_tokens = Some(value.parse()?),
            "temperature" => config.temperature = Some(value.parse()?),
            k if k.starts_with("api_key.") => {
                let provider = k.strip_prefix("api_key.").unwrap();
                config.api_keys.insert(provider.to_string(), value.to_string());
            }
            k if k.starts_with("base_url.") => {
                let provider = k.strip_prefix("base_url.").unwrap();
                config.base_urls.insert(provider.to_string(), value.to_string());
            }
            _ => anyhow::bail!("Unknown config key: {key}"),
        }
        config.save()
    }

    pub fn get_api_key(&self, provider: &str) -> Option<&str> {
        self.api_keys.get(provider).map(|s| s.as_str())
    }

    pub fn get_base_url(&self, provider: &str) -> Option<&str> {
        self.base_urls.get(provider).map(|s| s.as_str())
    }

    fn config_path() -> Result<PathBuf> {
        let config_dir = dirs::config_dir()
            .context("Could not determine config directory")?;
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
        }
    }
}
