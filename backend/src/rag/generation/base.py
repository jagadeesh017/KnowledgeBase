from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatModelProvider(ABC):
    @abstractmethod
    def chat(self, messages: list[ChatMessage]) -> str:
        """Send a conversation and return the assistant's reply."""
