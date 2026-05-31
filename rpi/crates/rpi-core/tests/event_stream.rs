//! Event stream abstraction contract tests.

use futures::StreamExt;
use rpi_core::{
    AssistantMessage, AssistantMessageEvent, ContentBlock, EventStream, StopReason, Usage,
    create_assistant_message_event_stream,
};

fn assistant_message(text: &str, stop_reason: StopReason) -> AssistantMessage {
    AssistantMessage {
        content: vec![ContentBlock::Text {
            text: text.to_string(),
        }],
        api: "openai".to_string(),
        provider: "openai".to_string(),
        model: "gpt-4o".to_string(),
        response_model: None,
        response_id: None,
        usage: Usage {
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
        },
        stop_reason,
        error_message: None,
        timestamp: 0,
    }
}

#[tokio::test]
async fn event_stream_yields_pushed_events_and_resolves_terminal_result() {
    let mut stream = EventStream::new(|event: &u32| *event == 3, |event: &u32| Ok(*event));

    stream.push(1);
    stream.push(2);
    stream.push(3);
    stream.push(4);

    let events = stream.by_ref().collect::<Vec<_>>().await;
    let result = stream.result().await.unwrap();

    assert_eq!(events, vec![1, 2, 3]);
    assert_eq!(result, 3);
}

#[tokio::test]
async fn assistant_event_stream_resolves_done_and_error_messages() {
    let done_message = assistant_message("ok", StopReason::Stop);
    let mut done_stream = create_assistant_message_event_stream();
    done_stream.push(AssistantMessageEvent::Done {
        reason: StopReason::Stop,
        message: done_message.clone(),
    });

    assert!(matches!(
        done_stream.next().await.unwrap(),
        AssistantMessageEvent::Done { .. }
    ));
    assert_eq!(done_stream.result().await.unwrap(), done_message);

    let error_message = assistant_message("aborted", StopReason::Aborted);
    let mut error_stream = create_assistant_message_event_stream();
    error_stream.push(AssistantMessageEvent::Error {
        reason: StopReason::Aborted,
        error: error_message.clone(),
    });

    assert!(matches!(
        error_stream.next().await.unwrap(),
        AssistantMessageEvent::Error { .. }
    ));
    assert_eq!(error_stream.result().await.unwrap(), error_message);
}

#[tokio::test]
async fn extract_result_on_non_terminal_event_returns_error_not_panic() {
    // Simulate callback desync: is_complete says "done" but extract_result
    // cannot produce a value for non-terminal events.
    use rpi_core::EventStreamError;

    let mut stream: EventStream<u32, u32> = EventStream::new(
        |_: &u32| true, // always complete
        |_: &u32| Err(EventStreamError::NotComplete), // can't extract
    );

    // push must not panic — it should drop the result and leave the stream
    // in a state where result() returns an error.
    stream.push(42);

    let collected = stream.by_ref().collect::<Vec<_>>().await;
    assert_eq!(collected, vec![42]);

    let err = stream.result().await.unwrap_err();
    assert_eq!(err, EventStreamError::ResultClosed);
}
