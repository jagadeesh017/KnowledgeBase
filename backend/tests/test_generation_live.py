import os

import pytest

from knowledge_platform.rag.generation import ChatMessage, OllamaChatModel

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") != "1",
    reason="set RUN_LIVE_TESTS=1 to run against a live Ollama server",
)


def test_live_chat_reply() -> None:
    reply = OllamaChatModel().chat(
        [ChatMessage(role="user", content="Reply with exactly: OK")]
    )

    assert "OK" in reply
