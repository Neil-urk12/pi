//! Generic async event stream utilities.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures::Stream;
use futures::channel::{mpsc, oneshot};

use crate::types::{AssistantMessage, AssistantMessageEvent};

/// Error returned when an event stream cannot produce its final result.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum EventStreamError {
    /// The final result was already awaited.
    #[error("event stream result was already awaited")]
    ResultAlreadyTaken,
    /// The stream ended without resolving a final result.
    #[error("event stream ended without a final result")]
    ResultClosed,
    /// extract_result was called on a non-terminal event.
    #[error("event stream extract_result called on non-terminal event")]
    NotComplete,
}

/// Generic event stream with explicit push/end/result semantics.
pub struct EventStream<T, R = T> {
    sender: Option<mpsc::UnboundedSender<T>>,
    receiver: mpsc::UnboundedReceiver<T>,
    result_sender: Option<oneshot::Sender<R>>,
    result_receiver: Option<oneshot::Receiver<R>>,
    is_complete: Arc<dyn Fn(&T) -> bool + Send + Sync>,
    extract_result: Arc<dyn Fn(&T) -> Result<R, EventStreamError> + Send + Sync>,
    done: bool,
}

impl<T, R> EventStream<T, R> {
    /// Create a stream from completion/result extraction callbacks.
    pub fn new(
        is_complete: impl Fn(&T) -> bool + Send + Sync + 'static,
        extract_result: impl Fn(&T) -> Result<R, EventStreamError> + Send + Sync + 'static,
    ) -> Self {
        let (sender, receiver) = mpsc::unbounded();
        let (result_sender, result_receiver) = oneshot::channel();
        Self {
            sender: Some(sender),
            receiver,
            result_sender: Some(result_sender),
            result_receiver: Some(result_receiver),
            is_complete: Arc::new(is_complete),
            extract_result: Arc::new(extract_result),
            done: false,
        }
    }

    /// Push an event into the stream.
    pub fn push(&mut self, event: T) {
        if self.done {
            return;
        }

        let is_complete = (self.is_complete)(&event);
        if is_complete {
            self.done = true;
            if let Some(sender) = self.result_sender.take() {
                if let Ok(result) = (self.extract_result)(&event) {
                    let _ = sender.send(result);
                }
            }
        }

        if let Some(sender) = &self.sender {
            let _ = sender.unbounded_send(event);
        }

        if is_complete {
            self.sender.take();
        }
    }

    /// End the stream, optionally resolving the final result.
    pub fn end(&mut self, result: Option<R>) {
        if self.done {
            return;
        }
        self.done = true;

        if let Some(result) = result {
            if let Some(sender) = self.result_sender.take() {
                let _ = sender.send(result);
            }
        }
        self.sender.take();
    }

    /// Await the stream's final result.
    pub async fn result(&mut self) -> Result<R, EventStreamError> {
        let receiver = self
            .result_receiver
            .take()
            .ok_or(EventStreamError::ResultAlreadyTaken)?;
        receiver.await.map_err(|_| EventStreamError::ResultClosed)
    }
}

impl<T, R> Stream for EventStream<T, R> {
    type Item = T;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        Pin::new(&mut this.receiver).poll_next(cx)
    }
}

impl<T, R> Unpin for EventStream<T, R> {}

/// Assistant-message event stream alias.
pub type AssistantMessageEventStream = EventStream<AssistantMessageEvent, AssistantMessage>;

/// Create an assistant-message event stream.
pub fn create_assistant_message_event_stream() -> AssistantMessageEventStream {
    EventStream::new(
        |event: &AssistantMessageEvent| {
            matches!(
                event,
                AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. }
            )
        },
        |event: &AssistantMessageEvent| match event {
            AssistantMessageEvent::Done { message, .. } => Ok(message.clone()),
            AssistantMessageEvent::Error { error, .. } => Ok(error.clone()),
            _ => Err(EventStreamError::NotComplete),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AssistantMessage, ContentBlock, StopReason, Usage};
    use futures::StreamExt;

    // --- push and receive ---

    #[tokio::test]
    async fn push_and_receive_single_event() {
        let mut stream: EventStream<u32, u32> =
            EventStream::new(|_: &u32| false, |_: &u32| Err(EventStreamError::NotComplete));

        stream.push(42);
        stream.end(None);

        let events: Vec<u32> = stream.by_ref().collect().await;
        assert_eq!(events, vec![42]);
    }

    // --- multiple events in order ---

    #[tokio::test]
    async fn push_multiple_events_preserves_order() {
        let mut stream: EventStream<u32, u32> =
            EventStream::new(|_: &u32| false, |_: &u32| Err(EventStreamError::NotComplete));

        stream.push(1);
        stream.push(2);
        stream.push(3);
        stream.end(None);

        let events: Vec<u32> = stream.by_ref().collect().await;
        assert_eq!(events, vec![1, 2, 3]);
    }

    // --- completion via result ---

    #[tokio::test]
    async fn terminal_event_resolves_result_ok() {
        let mut stream: EventStream<u32, u32> =
            EventStream::new(|event: &u32| *event == 3, |event: &u32| Ok(*event));

        stream.push(1);
        stream.push(2);
        stream.push(3);

        let events: Vec<u32> = stream.by_ref().collect().await;
        assert_eq!(events, vec![1, 2, 3]);

        let result = stream.result().await.unwrap();
        assert_eq!(result, 3);
    }

    // --- error via result ---

    #[tokio::test]
    async fn extract_result_error_returns_err() {
        let mut stream: EventStream<u32, u32> = EventStream::new(
            |_: &u32| true,
            |_: &u32| Err(EventStreamError::NotComplete),
        );

        stream.push(42);

        let events: Vec<u32> = stream.by_ref().collect().await;
        assert_eq!(events, vec![42]);

        // result_sender was never sent because extract_result returned Err
        let err = stream.result().await.unwrap_err();
        assert_eq!(err, EventStreamError::ResultClosed);
    }

    // --- stream yields None after completion ---

    #[tokio::test]
    async fn stream_yields_none_after_terminal_event() {
        let mut stream: EventStream<u32, u32> =
            EventStream::new(|event: &u32| *event == 99, |event: &u32| Ok(*event));

        stream.push(99);

        let first = stream.next().await;
        assert_eq!(first, Some(99));

        let second = stream.next().await;
        assert_eq!(second, None);
    }

    // --- push after completion is no-op ---

    #[tokio::test]
    async fn push_after_terminal_is_noop() {
        let mut stream: EventStream<u32, u32> =
            EventStream::new(|event: &u32| *event == 1, |event: &u32| Ok(*event));

        stream.push(1);
        stream.push(2); // should be dropped silently
        stream.push(3); // should be dropped silently

        let events: Vec<u32> = stream.by_ref().collect().await;
        assert_eq!(events, vec![1]);

        let result = stream.result().await.unwrap();
        assert_eq!(result, 1);
    }

    // --- drop before completion does not panic ---

    #[tokio::test]
    async fn drop_before_completion_no_panic() {
        let mut stream: EventStream<u32, u32> =
            EventStream::new(|_: &u32| false, |_: &u32| Err(EventStreamError::NotComplete));

        stream.push(1);
        stream.push(2);

        // Drop without consuming stream or awaiting result -- must not panic.
        drop(stream);
    }

    // --- result() before completion ---

    #[tokio::test]
    async fn result_before_completion_returns_closed() {
        let mut stream: EventStream<u32, u32> =
            EventStream::new(|_: &u32| false, |_: &u32| Err(EventStreamError::NotComplete));

        // Drop the result_sender to close the channel without sending.
        stream.result_sender.take();

        let err = stream.result().await.unwrap_err();
        assert_eq!(err, EventStreamError::ResultClosed);
    }

    // --- result() called twice ---

    #[tokio::test]
    async fn result_called_twice_returns_already_taken() {
        let mut stream: EventStream<u32, u32> =
            EventStream::new(|event: &u32| *event == 1, |event: &u32| Ok(*event));

        stream.push(1);

        let first = stream.result().await;
        assert!(first.is_ok());

        let second = stream.result().await.unwrap_err();
        assert_eq!(second, EventStreamError::ResultAlreadyTaken);
    }

    // --- end() with result ---

    #[tokio::test]
    async fn end_with_result_resolves() {
        let mut stream: EventStream<u32, u32> =
            EventStream::new(|_: &u32| false, |_: &u32| Err(EventStreamError::NotComplete));

        stream.push(1);
        stream.push(2);
        stream.end(Some(999));

        let events: Vec<u32> = stream.by_ref().collect().await;
        assert_eq!(events, vec![1, 2]);

        let result = stream.result().await.unwrap();
        assert_eq!(result, 999);
    }

    // --- end() without result ---

    #[tokio::test]
    async fn end_without_result_closes_stream() {
        let mut stream: EventStream<u32, u32> =
            EventStream::new(|_: &u32| false, |_: &u32| Err(EventStreamError::NotComplete));

        stream.push(1);
        stream.end(None);

        // Stream channel is closed (sender dropped).
        let events: Vec<u32> = stream.by_ref().collect().await;
        assert_eq!(events, vec![1]);

        // end(None) does NOT take result_sender, so result() would hang forever.
        // Drop result_sender manually to close the oneshot channel.
        drop(stream.result_sender.take());
        let err = stream.result().await.unwrap_err();
        assert_eq!(err, EventStreamError::ResultClosed);
    }

    // --- end() after terminal event is no-op ---

    #[tokio::test]
    async fn end_after_terminal_is_noop() {
        let mut stream: EventStream<u32, u32> =
            EventStream::new(|event: &u32| *event == 1, |event: &u32| Ok(*event));

        stream.push(1);
        stream.end(Some(999)); // should be ignored since already done

        let events: Vec<u32> = stream.by_ref().collect().await;
        assert_eq!(events, vec![1]);

        // Result should be 1 from the terminal push, not 999 from end.
        let result = stream.result().await.unwrap();
        assert_eq!(result, 1);
    }

    // --- push after end is no-op ---

    #[tokio::test]
    async fn push_after_end_is_noop() {
        let mut stream: EventStream<u32, u32> =
            EventStream::new(|_: &u32| false, |_: &u32| Err(EventStreamError::NotComplete));

        stream.push(1);
        stream.end(None);
        stream.push(2); // should be silently dropped

        let events: Vec<u32> = stream.by_ref().collect().await;
        assert_eq!(events, vec![1]);
    }

    // --- empty stream with end ---

    #[tokio::test]
    async fn empty_stream_end_yields_no_events() {
        let mut stream: EventStream<u32, u32> =
            EventStream::new(|_: &u32| false, |_: &u32| Err(EventStreamError::NotComplete));

        stream.end(None);

        let events: Vec<u32> = stream.by_ref().collect().await;
        assert!(events.is_empty());
    }

    // --- error event variant with AssistantMessageEvent ---

    #[tokio::test]
    async fn assistant_event_stream_error_terminal() {
        let mut stream = create_assistant_message_event_stream();

        let err_msg = AssistantMessage {
            content: vec![],
            api: "openai".to_string(),
            provider: "openai".to_string(),
            model: "gpt-4o".to_string(),
            response_model: None,
            response_id: None,
            usage: Usage {
                prompt_tokens: 1,
                completion_tokens: 0,
                total_tokens: 1,
            },
            stop_reason: StopReason::Error,
            error_message: Some("something failed".to_string()),
            timestamp: 0,
        };

        stream.push(AssistantMessageEvent::Error {
            reason: StopReason::Error,
            error: err_msg.clone(),
        });

        let event = stream.next().await.unwrap();
        assert!(matches!(event, AssistantMessageEvent::Error { .. }));

        let result = stream.result().await.unwrap();
        assert_eq!(result, err_msg);
    }

    // --- non-terminal events stream without resolving result ---

    #[tokio::test]
    async fn non_terminal_events_stream_then_close() {
        let mut stream = create_assistant_message_event_stream();

        stream.push(AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta: "hello".to_string(),
            partial: AssistantMessage {
                content: vec![ContentBlock::Text {
                    text: "hello".to_string(),
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
                stop_reason: StopReason::Stop,
                error_message: None,
                timestamp: 0,
            },
        });

        // Only non-terminal event should be yielded.
        let event = stream.next().await.unwrap();
        assert!(matches!(event, AssistantMessageEvent::TextDelta { .. }));

        // End without a terminal event.
        stream.end(None);

        drop(stream.result_sender.take());
        let err = stream.result().await.unwrap_err();
        assert_eq!(err, EventStreamError::ResultClosed);
    }

    // --- EventStreamError Display ---

    #[test]
    fn event_stream_error_display() {
        assert_eq!(
            EventStreamError::ResultAlreadyTaken.to_string(),
            "event stream result was already awaited"
        );
        assert_eq!(
            EventStreamError::ResultClosed.to_string(),
            "event stream ended without a final result"
        );
        assert_eq!(
            EventStreamError::NotComplete.to_string(),
            "event stream extract_result called on non-terminal event"
        );
    }

    // --- EventStreamError is PartialEq + Eq ---

    #[test]
    fn event_stream_error_eq() {
        assert_eq!(
            EventStreamError::ResultAlreadyTaken,
            EventStreamError::ResultAlreadyTaken
        );
        assert_ne!(
            EventStreamError::ResultAlreadyTaken,
            EventStreamError::ResultClosed
        );
    }
}
