from collections.abc import Mapping, Sequence
from typing import Protocol

import ollama

from knowledge_platform.rag.generation.base import ChatMessage, ChatModelProvider
from knowledge_platform.rag.ollama_retry import call_with_retries


class ChatResponse(Protocol):
    def __getitem__(self, key: str) -> Mapping[str, str]: ...


class ChatClient(Protocol):
    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, str]],
    ) -> ChatResponse: ...


class OllamaChatModel(ChatModelProvider):
    DEFAULT_MODEL = "qwen2.5:3b"
    DEFAULT_HOST = "http://localhost:11434"
    DEFAULT_TIMEOUT_SECONDS = 60.0
    DEFAULT_RETRIES = 2
    DEFAULT_BACKOFF_SECONDS = 0.5

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        client: ChatClient | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        retries: int = DEFAULT_RETRIES,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    ) -> None:
        self.model = model
        self.retries = max(1, retries)
        self.backoff_seconds = backoff_seconds
        self.client = (
            client
            if client is not None
            else ollama.Client(host=host, timeout=timeout_seconds)
        )

    def chat(self, messages: list[ChatMessage]) -> str:
        response = call_with_retries(
            lambda: self.client.chat(
                model=self.model,
                messages=[
                    {"role": message.role, "content": message.content}
                    for message in messages
                ],
            ),
            attempts=self.retries,
            backoff_seconds=self.backoff_seconds,
        )
        return response["message"]["content"]
