use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use rpi_core::{Model, compaction::CompactionSettings};
use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Clone, Serialize, Deserialize)]
pub struct Config {
    /// Default model to use (provider/model format)
    pub default_model: Option<String>,

    /// Provider API keys
    #[serde(default)]
    pub api_keys: HashMap<String, String>,

    /// Provider base URLs (for custom endpoints)
    #[serde(default)]
    pub base_urls: HashMap<String, String>,

    /// Provider OAuth tokens
    #[serde(default)]
    pub oauth_tokens: HashMap<String, OAuthToken>,

    /// Custom model definitions keyed by provider/model id
    #[serde(default)]
    pub models: HashMap<String, Model>,

    /// User keybinding overrides
    #[serde(default)]
    pub keybindings: HashMap<String, String>,

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

impl std::fmt::Debug for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Config")
            .field("default_model", &self.default_model)
            .field("api_keys", &"<redacted>")
            .field("base_urls", &self.base_urls)
            .field("oauth_tokens", &"<redacted>")
            .field("models", &self.models)
            .field("keybindings", &self.keybindings)
            .field("system_prompt", &self.system_prompt)
            .field("max_tokens", &self.max_tokens)
            .field("temperature", &self.temperature)
            .field("compaction", &self.compaction)
            .finish()
    }
}

impl Config {
    pub fn load() -> Result<Self> {
        let env = std::env::vars().collect();
        let config_dir = Self::config_dir()?;
        let project_dir = std::env::current_dir().ok().map(|dir| dir.join(".rpi"));
        Self::load_from_dirs_with_env(Some(&config_dir), project_dir.as_deref(), &env, &[])
    }

    #[cfg(test)]
    pub fn load_from_path_with_env(
        config_path: &Path,
        env: &HashMap<String, String>,
    ) -> Result<Self> {
        Self::load_layered_with_env(Some(config_path), None, env, &[])
    }

    #[cfg(test)]
    pub fn load_layered_with_env(
        global_path: Option<&Path>,
        project_path: Option<&Path>,
        env: &HashMap<String, String>,
        cli_overrides: &[(&str, &str)],
    ) -> Result<Self> {
        let mut config = Self::default();
        if let Some(path) = global_path {
            config.merge_patch(ConfigPatch::read(path)?);
        }
        if let Some(path) = project_path {
            config.merge_patch(ConfigPatch::read(path)?);
        }
        config.apply_env(env)?;
        for (key, value) in cli_overrides {
            config = config.set(key, value)?;
        }
        config.interpolate_env(env)?;
        Ok(config)
    }

    pub fn load_from_dirs_with_env(
        global_dir: Option<&Path>,
        project_dir: Option<&Path>,
        env: &HashMap<String, String>,
        cli_overrides: &[(&str, &str)],
    ) -> Result<Self> {
        let mut config = Self::default();
        if let Some(dir) = global_dir {
            config.merge_config_dir(dir)?;
        }
        if let Some(dir) = project_dir {
            config.merge_config_dir(dir)?;
        }
        config.apply_env(env)?;
        for (key, value) in cli_overrides {
            config = config.set(key, value)?;
        }
        config.interpolate_env(env)?;
        Ok(config)
    }

    pub fn save(&self) -> Result<()> {
        let config_path = Self::config_path()?;
        self.save_to_path(&config_path)
    }

    fn save_to_path(&self, config_path: &Path) -> Result<()> {
        if let Some(parent) = config_path.parent() {
            fs::create_dir_all(parent)?;

            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(parent, fs::Permissions::from_mode(0o700))?;
            }
        }

        let _lock = ConfigFileLock::acquire(config_path)?;
        let content = serde_json::to_string_pretty(self)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            let mut file = fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .mode(0o600)
                .open(config_path)?;
            file.write_all(content.as_bytes())?;
        }

        #[cfg(not(unix))]
        {
            fs::write(&config_path, content)?;
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
        Ok(Self::config_dir()?.join("config.json"))
    }

    fn config_dir() -> Result<PathBuf> {
        let config_dir = dirs::config_dir().context("Could not determine config directory")?;
        Ok(config_dir.join("rpi"))
    }

    fn lock_path_for(config_path: &Path) -> PathBuf {
        let file_name = config_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("config.json");
        config_path.with_file_name(format!("{file_name}.lock"))
    }

    fn merge_config_dir(&mut self, dir: &Path) -> Result<()> {
        for file_name in CONFIG_SCHEMA_FILES {
            self.merge_patch(ConfigPatch::read(&dir.join(file_name))?);
        }
        Ok(())
    }

    fn merge_patch(&mut self, patch: ConfigPatch) {
        if let Some(default_model) = patch.default_model {
            self.default_model = Some(default_model);
        }
        self.api_keys.extend(patch.api_keys);
        self.base_urls.extend(patch.base_urls);
        self.oauth_tokens.extend(patch.oauth_tokens);
        self.models.extend(patch.models);
        self.keybindings.extend(patch.keybindings);
        if let Some(system_prompt) = patch.system_prompt {
            self.system_prompt = Some(system_prompt);
        }
        if let Some(max_tokens) = patch.max_tokens {
            self.max_tokens = Some(max_tokens);
        }
        if let Some(temperature) = patch.temperature {
            self.temperature = Some(temperature);
        }
        if let Some(compaction) = patch.compaction {
            let target = self
                .compaction
                .get_or_insert_with(CompactionConfig::default);
            compaction.merge_into(target);
        }
    }

    fn apply_env(&mut self, env: &HashMap<String, String>) -> Result<()> {
        for (key, value) in env {
            match key.as_str() {
                "RPI_DEFAULT_MODEL" => self.default_model = Some(value.clone()),
                "RPI_SYSTEM_PROMPT" => self.system_prompt = Some(value.clone()),
                "RPI_MAX_TOKENS" => {
                    self.max_tokens = Some(
                        value
                            .parse()
                            .with_context(|| format!("Invalid RPI_MAX_TOKENS value: {value}"))?,
                    )
                }
                "RPI_TEMPERATURE" => {
                    self.temperature = Some(
                        value
                            .parse()
                            .with_context(|| format!("Invalid RPI_TEMPERATURE value: {value}"))?,
                    )
                }
                "RPI_COMPACTION_ENABLED" => {
                    let compaction = self
                        .compaction
                        .get_or_insert_with(CompactionConfig::default);
                    compaction.enabled = value.parse().with_context(|| {
                        format!("Invalid RPI_COMPACTION_ENABLED value: {value}")
                    })?;
                }
                "RPI_COMPACTION_CONTEXT_WINDOW" => {
                    let compaction = self
                        .compaction
                        .get_or_insert_with(CompactionConfig::default);
                    compaction.context_window = Some(value.parse().with_context(|| {
                        format!("Invalid RPI_COMPACTION_CONTEXT_WINDOW value: {value}")
                    })?);
                }
                "RPI_COMPACTION_RESERVE_TOKENS" => {
                    let compaction = self
                        .compaction
                        .get_or_insert_with(CompactionConfig::default);
                    compaction.reserve_tokens = value.parse().with_context(|| {
                        format!("Invalid RPI_COMPACTION_RESERVE_TOKENS value: {value}")
                    })?;
                }
                "RPI_COMPACTION_KEEP_RECENT_TOKENS" => {
                    let compaction = self
                        .compaction
                        .get_or_insert_with(CompactionConfig::default);
                    compaction.keep_recent_tokens = value.parse().with_context(|| {
                        format!("Invalid RPI_COMPACTION_KEEP_RECENT_TOKENS value: {value}")
                    })?;
                }
                _ => {
                    if let Some(provider) = key.strip_prefix("RPI_API_KEY_") {
                        self.api_keys
                            .insert(provider.to_ascii_lowercase(), value.clone());
                    } else if let Some(provider) = key.strip_prefix("RPI_BASE_URL_") {
                        self.base_urls
                            .insert(provider.to_ascii_lowercase(), value.clone());
                    }
                }
            }
        }
        Ok(())
    }

    fn interpolate_env(&mut self, env: &HashMap<String, String>) -> Result<()> {
        if let Some(default_model) = &mut self.default_model {
            *default_model = interpolate_env(default_model, env)?;
        }
        for value in self.api_keys.values_mut() {
            *value = interpolate_env(value, env)?;
        }
        for value in self.base_urls.values_mut() {
            *value = interpolate_env(value, env)?;
        }
        for token in self.oauth_tokens.values_mut() {
            token.access_token = interpolate_env(&token.access_token, env)?;
            if let Some(refresh_token) = &mut token.refresh_token {
                *refresh_token = interpolate_env(refresh_token, env)?;
            }
            if let Some(expires_at) = &mut token.expires_at {
                *expires_at = interpolate_env(expires_at, env)?;
            }
        }
        if let Some(system_prompt) = &mut self.system_prompt {
            *system_prompt = interpolate_env(system_prompt, env)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
struct ConfigFileLock {
    path: PathBuf,
}

impl ConfigFileLock {
    fn acquire(config_path: &Path) -> Result<Self> {
        let path = Config::lock_path_for(config_path);
        Self::try_create(&path)
            .or_else(|e| {
                if Self::is_stale(&path) {
                    let _ = fs::remove_file(&path);
                    Self::try_create(&path)
                } else {
                    Err(e)
                }
            })
            .with_context(|| format!("Failed to acquire config lock at {}", path.display()))?;
        Ok(Self { path })
    }

    fn try_create(path: &Path) -> std::io::Result<std::fs::File> {
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
    }

    fn is_stale(path: &Path) -> bool {
        const STALE_THRESHOLD: Duration = Duration::from_secs(30);
        fs::metadata(path)
            .and_then(|m| m.modified())
            .ok()
            .is_some_and(|mtime| mtime.elapsed().unwrap_or(Duration::ZERO) > STALE_THRESHOLD)
}
}

impl Drop for ConfigFileLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

const CONFIG_SCHEMA_FILES: &[&str] = &[
    "config.json",
    "settings.json",
    "auth.json",
    "models.json",
    "keybindings.json",
];

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OAuthToken {
    /// Provider access token.
    pub access_token: String,
    /// Optional refresh token.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    /// Optional token expiry timestamp.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<String>,
}

impl std::fmt::Debug for OAuthToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OAuthToken")
            .field("access_token", &"<redacted>")
            .field("refresh_token", &self.refresh_token.as_ref().map(|_| "<redacted>"))
            .field("expires_at", &self.expires_at)
            .finish()
    }
}

#[derive(Default, Deserialize)]
struct ConfigPatch {
    default_model: Option<String>,
    #[serde(default)]
    api_keys: HashMap<String, String>,
    #[serde(default)]
    base_urls: HashMap<String, String>,
    #[serde(default)]
    oauth_tokens: HashMap<String, OAuthToken>,
    #[serde(default)]
    models: HashMap<String, Model>,
    #[serde(default)]
    keybindings: HashMap<String, String>,
    system_prompt: Option<String>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    compaction: Option<CompactionPatch>,
}

impl std::fmt::Debug for ConfigPatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConfigPatch")
            .field("default_model", &self.default_model)
            .field("api_keys", &"<redacted>")
            .field("base_urls", &self.base_urls)
            .field("oauth_tokens", &"<redacted>")
            .field("models", &self.models)
            .field("keybindings", &self.keybindings)
            .field("system_prompt", &self.system_prompt)
            .field("max_tokens", &self.max_tokens)
            .field("temperature", &self.temperature)
            .field("compaction", &self.compaction)
            .finish()
    }
}

impl ConfigPatch {
    fn read(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read config at {}", path.display()))?;
        serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse config at {}", path.display()))
    }
}

#[derive(Debug, Default, Deserialize)]
struct CompactionPatch {
    enabled: Option<bool>,
    context_window: Option<u32>,
    reserve_tokens: Option<u32>,
    keep_recent_tokens: Option<u32>,
}

impl CompactionPatch {
    fn merge_into(self, config: &mut CompactionConfig) {
        if let Some(enabled) = self.enabled {
            config.enabled = enabled;
        }
        if let Some(context_window) = self.context_window {
            config.context_window = Some(context_window);
        }
        if let Some(reserve_tokens) = self.reserve_tokens {
            config.reserve_tokens = reserve_tokens;
        }
        if let Some(keep_recent_tokens) = self.keep_recent_tokens {
            config.keep_recent_tokens = keep_recent_tokens;
        }
    }
}

fn interpolate_env(value: &str, env: &HashMap<String, String>) -> Result<String> {
    let mut output = String::new();
    let mut rest = value;
    while let Some(start) = rest.find("${") {
        output.push_str(&rest[..start]);
        let after_start = &rest[start + 2..];
        let Some(end) = after_start.find('}') else {
            anyhow::bail!("Unclosed environment interpolation in config value");
        };
        let name = &after_start[..end];
        let replacement = env
            .get(name)
            .with_context(|| format!("Missing environment variable {name}"))?;
        output.push_str(replacement);
        rest = &after_start[end + 1..];
    }
    output.push_str(rest);
    Ok(output)
}

impl Default for Config {
    fn default() -> Self {
        Self {
            default_model: Some("openai/gpt-4o".to_string()),
            api_keys: HashMap::new(),
            base_urls: HashMap::new(),
            oauth_tokens: HashMap::new(),
            models: HashMap::new(),
            keybindings: HashMap::new(),
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
    use std::collections::HashMap;

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

    #[test]
    fn load_from_path_returns_defaults_when_file_is_missing() {
        let temp_dir = tempfile::tempdir().unwrap();
        let missing_path = temp_dir.path().join("missing-config.json");
        let env = HashMap::new();

        let config = Config::load_from_path_with_env(&missing_path, &env).unwrap();

        assert_eq!(config.default_model.as_deref(), Some("openai/gpt-4o"));
    }

    #[test]
    fn load_from_path_interpolates_env_vars_in_string_values() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config_path = temp_dir.path().join("config.json");
        fs::write(
            &config_path,
            r#"{
                "api_keys": {"openai": "${OPENAI_API_KEY}"},
                "base_urls": {"openai": "https://${OPENAI_HOST}/v1"},
                "system_prompt": "Use ${PROJECT_NAME}"
            }"#,
        )
        .unwrap();
        let env = HashMap::from([
            ("OPENAI_API_KEY".to_string(), "test-key".to_string()),
            ("OPENAI_HOST".to_string(), "api.example.test".to_string()),
            ("PROJECT_NAME".to_string(), "rpi".to_string()),
        ]);

        let config = Config::load_from_path_with_env(&config_path, &env).unwrap();

        assert_eq!(config.get_api_key("openai"), Some("test-key"));
        assert_eq!(
            config.get_base_url("openai"),
            Some("https://api.example.test/v1")
        );
        assert_eq!(config.system_prompt.as_deref(), Some("Use rpi"));
    }

    #[test]
    fn load_layered_with_env_applies_defaults_file_env_then_cli_order() {
        let temp_dir = tempfile::tempdir().unwrap();
        let global_path = temp_dir.path().join("global.json");
        let project_path = temp_dir.path().join("project.json");
        fs::write(
            &global_path,
            r#"{
                "default_model": "anthropic/claude",
                "api_keys": {"openai": "global-key"},
                "base_urls": {"openai": "https://global.example/v1"},
                "compaction": {"reserve_tokens": 1000}
            }"#,
        )
        .unwrap();
        fs::write(
            &project_path,
            r#"{
                "api_keys": {"anthropic": "project-key"},
                "base_urls": {"openai": "https://project.example/v1"},
                "compaction": {"keep_recent_tokens": 2000}
            }"#,
        )
        .unwrap();
        let env = HashMap::from([
            ("RPI_DEFAULT_MODEL".to_string(), "google/gemini".to_string()),
            ("RPI_API_KEY_OPENAI".to_string(), "env-key".to_string()),
            (
                "RPI_COMPACTION_RESERVE_TOKENS".to_string(),
                "1500".to_string(),
            ),
        ]);
        let cli_overrides = [("default_model", "openai/gpt-4o")];

        let config = Config::load_layered_with_env(
            Some(&global_path),
            Some(&project_path),
            &env,
            &cli_overrides,
        )
        .unwrap();

        assert_eq!(config.default_model.as_deref(), Some("openai/gpt-4o"));
        assert_eq!(config.get_api_key("openai"), Some("env-key"));
        assert_eq!(config.get_api_key("anthropic"), Some("project-key"));
        assert_eq!(
            config.get_base_url("openai"),
            Some("https://project.example/v1")
        );
        let compaction = config.compaction.as_ref().unwrap();
        assert_eq!(compaction.reserve_tokens, 1500);
        assert_eq!(compaction.keep_recent_tokens, 2000);
    }

    #[test]
    fn load_from_dirs_reads_split_config_schema_files_in_layer_order() {
        let temp_dir = tempfile::tempdir().unwrap();
        let global_dir = temp_dir.path().join("global");
        let project_dir = temp_dir.path().join("project");
        fs::create_dir_all(&global_dir).unwrap();
        fs::create_dir_all(&project_dir).unwrap();

        fs::write(
            global_dir.join("settings.json"),
            r#"{
                "default_model": "anthropic/claude",
                "system_prompt": "Global ${PROJECT_NAME}"
            }"#,
        )
        .unwrap();
        fs::write(
            global_dir.join("auth.json"),
            r#"{
                "api_keys": {"openai": "global-key"},
                "oauth_tokens": {
                    "openai": {
                        "access_token": "${OPENAI_ACCESS_TOKEN}",
                        "refresh_token": "refresh-token",
                        "expires_at": "2026-05-30T00:00:00Z"
                    }
                }
            }"#,
        )
        .unwrap();
        fs::write(
            global_dir.join("models.json"),
            r#"{
                "models": {
                    "openai/custom": {
                        "id": {"provider": "openai", "model": "custom"},
                        "name": "Custom",
                        "api": "openai",
                        "provider": "openai",
                        "base_url": "https://api.example.test",
                        "reasoning": false,
                        "input": ["text"],
                        "cost": {
                            "input": 1.0,
                            "output": 2.0,
                            "cache_read": 0.1,
                            "cache_write": 0.2
                        },
                        "context_window": 128000,
                        "max_tokens": 4096
                    }
                }
            }"#,
        )
        .unwrap();
        fs::write(
            project_dir.join("settings.json"),
            r#"{"default_model": "openai/custom"}"#,
        )
        .unwrap();
        fs::write(
            project_dir.join("auth.json"),
            r#"{"api_keys": {"anthropic": "project-key"}}"#,
        )
        .unwrap();
        fs::write(
            project_dir.join("keybindings.json"),
            r#"{"keybindings": {"submit": "ctrl+j", "cancel": "esc"}}"#,
        )
        .unwrap();

        let env = HashMap::from([
            ("PROJECT_NAME".to_string(), "rpi".to_string()),
            (
                "OPENAI_ACCESS_TOKEN".to_string(),
                "env-access-token".to_string(),
            ),
        ]);

        let config =
            Config::load_from_dirs_with_env(Some(&global_dir), Some(&project_dir), &env, &[])
                .unwrap();

        assert_eq!(config.default_model.as_deref(), Some("openai/custom"));
        assert_eq!(config.system_prompt.as_deref(), Some("Global rpi"));
        assert_eq!(config.get_api_key("openai"), Some("global-key"));
        assert_eq!(config.get_api_key("anthropic"), Some("project-key"));
        assert_eq!(
            config.oauth_tokens["openai"].access_token,
            "env-access-token"
        );
        assert_eq!(config.models["openai/custom"].context_window, 128000);
        assert_eq!(config.keybindings["submit"], "ctrl+j");
    }

    #[test]
    fn save_to_path_refuses_when_lock_file_exists() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let lock_path = Config::lock_path_for(&config_path);
        fs::write(&lock_path, "held").unwrap();

        let err = Config::default().save_to_path(&config_path).unwrap_err();

        assert!(
            err.to_string().contains("config lock"),
            "unexpected error: {err}"
        );
        assert!(
            !config_path.exists(),
            "save should not write config while lock is held"
        );
        assert!(lock_path.exists(), "foreign lock should not be removed");
    }

    #[test]
    fn acquire_removes_stale_lock_older_than_30_seconds() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let lock_path = Config::lock_path_for(&config_path);

        // Create a stale lock file by setting its mtime to 60 seconds ago.
        fs::write(&lock_path, "stale").unwrap();
        std::process::Command::new("touch")
            .args(["-d", "60 seconds ago"])
            .arg(&lock_path)
            .status()
            .unwrap();

        // acquire should succeed by removing the stale lock.
        let lock = ConfigFileLock::acquire(&config_path).unwrap();
        assert_eq!(lock.path, lock_path);
        // Lock file exists while held.
        assert!(lock_path.exists());
        drop(lock);
        // Lock file removed after drop.
        assert!(!lock_path.exists());
    }

    #[test]
    fn acquire_fails_when_lock_is_recent() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let lock_path = Config::lock_path_for(&config_path);

        // Create a fresh lock file (just now).
        fs::write(&lock_path, "held").unwrap();

        // acquire should fail because lock is not stale.
        let err = ConfigFileLock::acquire(&config_path).unwrap_err();
        assert!(
            err.to_string().contains("config lock"),
            "unexpected error: {err}"
        );
        assert!(lock_path.exists(), "recent lock should not be removed");
    }

    #[test]
    fn save_to_path_removes_own_lock_after_success() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let lock_path = Config::lock_path_for(&config_path);

        Config::default().save_to_path(&config_path).unwrap();

        assert!(config_path.exists());
        assert!(!lock_path.exists(), "successful save should release lock");
    }

    #[test]
    #[cfg(unix)]
    fn save_to_path_creates_file_with_0600_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let temp_dir = tempfile::tempdir().unwrap();
        let config_path = temp_dir.path().join("config.json");

        let mut config = Config::default();
        config
            .api_keys
            .insert("openai".to_string(), "sk-secret-key".to_string());
        config.save_to_path(&config_path).unwrap();

        let perms = fs::metadata(&config_path).unwrap().permissions();
        let mode = perms.mode() & 0o777;
        assert_eq!(
            mode, 0o600,
            "config file should be created with 0o600 permissions, got {mode:#o}"
        );
    }

    #[test]
    fn interpolate_errors_on_unclosed_brace() {
        let env = HashMap::new();
        let result = interpolate_env("hello ${UNCLOSED", &env);
        assert!(result.is_err(), "should fail on unclosed ${{");
        assert!(result.unwrap_err().to_string().contains("Unclosed"));
    }

    #[test]
    fn interpolate_errors_on_missing_env_var() {
        let env = HashMap::new();
        let result = interpolate_env("${DEFINITELY_NOT_SET_VAR_XYZ}", &env);
        assert!(result.is_err(), "should fail on missing env var");
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("DEFINITELY_NOT_SET_VAR_XYZ")
        );
    }

    #[test]
    fn interpolate_empty_var_name_errors() {
        let env = HashMap::new();
        let result = interpolate_env("hello ${}", &env);
        // ${} with empty name: env.get("") returns None → error about missing var
        assert!(result.is_err(), "should fail on empty var name");
    }

    #[test]
    fn interpolate_multiple_vars_in_one_string() {
        let env = HashMap::from([
            ("HOST".to_string(), "api.example.com".to_string()),
            ("PORT".to_string(), "8443".to_string()),
        ]);
        let result = interpolate_env("https://${HOST}:${PORT}/v1", &env).unwrap();
        assert_eq!(result, "https://api.example.com:8443/v1");
    }

    #[test]
    fn interpolate_empty_string_passes_through() {
        let env = HashMap::new();
        let result = interpolate_env("", &env).unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn interpolate_no_placeholders_passes_through() {
        let env = HashMap::from([("FOO".to_string(), "bar".to_string())]);
        let result = interpolate_env("just plain text", &env).unwrap();
        assert_eq!(result, "just plain text");
    }

    #[test]
    fn interpolate_empty_env_value_substitutes_empty_string() {
        let env = HashMap::from([("EMPTY".to_string(), String::new())]);
        let result = interpolate_env("prefix-${EMPTY}-suffix", &env).unwrap();
        assert_eq!(result, "prefix--suffix");
    }

    #[test]
    fn interpolate_adjacent_placeholders() {
        let env = HashMap::from([
            ("A".to_string(), "1".to_string()),
            ("B".to_string(), "2".to_string()),
        ]);
        let result = interpolate_env("${A}${B}", &env).unwrap();
        assert_eq!(result, "12");
    }

    #[test]
    fn interpolate_escaped_dollar_sign_not_supported() {
        // Literal $ without { should pass through unchanged
        let env = HashMap::new();
        let result = interpolate_env("price is $5", &env).unwrap();
        assert_eq!(result, "price is $5");
    }
}
