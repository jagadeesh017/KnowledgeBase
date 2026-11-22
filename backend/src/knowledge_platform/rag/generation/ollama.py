from collections.abc import Mapping, Sequence
from typing import Protocol

import ollama

from knowledge_platform.rag.generation.base import ChatMessage, ChatModelProvider


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

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        client: ChatClient | None = None,
    ) -> None:
        self.model = model
        self.client = client if client is not None else ollama.Client(host=host)

    def chat(self, messages: list[ChatMessage]) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": message.role, "content": message.content}
                for message in messages
            ],
        )
        return response["message"]["content"]
