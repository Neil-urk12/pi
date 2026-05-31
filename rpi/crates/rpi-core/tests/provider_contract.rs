//! Provider trait surface contract tests.

use async_trait::async_trait;
use rpi_core::{
    AgentConfig, ChatResponse, ChatStream, Message, MessageContent, ModelId, Provider,
    ProviderCapabilities, ProviderRequestConfig, Role, StreamOptions, ToolDefinition,
};

struct ContractProvider;

#[async_trait]
impl Provider for ContractProvider {
    fn id(&self) -> &str {
        "contract"
    }

    async fn chat(
        &self,
        _model: &str,
        _messages: &[Message],
        _tools: &[ToolDefinition],
        _config: &AgentConfig,
    ) -> rpi_core::Result<ChatResponse> {
        unimplemented!("not needed for provider surface tests")
    }

    async fn chat_stream(
        &self,
        _model: &str,
        _messages: &[Message],
        _tools: &[ToolDefinition],
        _config: &AgentConfig,
    ) -> rpi_core::Result<ChatStream> {
        unimplemented!("not needed for provider surface tests")
    }
}

fn user_message(text: &str) -> Message {
    Message {
        role: Role::User,
        content: Some(MessageContent::Text(text.to_string())),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }
}

#[test]
fn provider_metadata_defaults_to_stable_id_and_empty_capabilities() {
    let provider = ContractProvider;
    let metadata = provider.metadata();

    assert_eq!(metadata.id, "contract");
    assert_eq!(metadata.name, "contract");
    assert_eq!(metadata.supported_apis, Vec::<String>::new());
    assert_eq!(metadata.capabilities, ProviderCapabilities::default());
}

#[test]
fn stream_and_request_options_have_safe_defaults() {
    let stream = StreamOptions::default();
    assert_eq!(stream.max_tokens, None);
    assert_eq!(stream.temperature, None);
    assert_eq!(stream.api_key, None);

    let request = ProviderRequestConfig::default();
    assert_eq!(request.model, None);
    assert_eq!(request.context_window, None);
    assert_eq!(request.stream, stream);
}

#[test]
fn provider_validate_context_rejects_known_over_budget_requests() {
    let provider = ContractProvider;
    let request = ProviderRequestConfig {
        model: Some(ModelId::new("openai", "gpt-4o")),
        context_window: Some(1),
        ..ProviderRequestConfig::default()
    };

    let err = provider
        .validate_context(&[user_message("this exceeds one token")], &request)
        .unwrap_err();

    assert!(
        err.to_string().contains("context window"),
        "unexpected error: {err}"
    );
}
