from rag.generation.base import ChatMessage, ChatModelProvider
from rag.generation.prompts import SYSTEM_PROMPT, build_prompt
from rag.retrieval import Retriever, build_context, build_sources
from rag.schemas import RAGAnswer


class RAGAnswerer:
    def __init__(self, retriever: Retriever, chat_model: ChatModelProvider) -> None:
        self.retriever = retriever
        self.chat_model = chat_model

    def answer(
        self,
        question: str,
        top_k: int = 5,
        where: dict[str, object] | None = None,
    ) -> RAGAnswer:
        results = self.retriever.retrieve(question, top_k=top_k, where=where)
        context = build_context(results)
        reply = self.chat_model.chat(
            [
                ChatMessage(role="system", content=SYSTEM_PROMPT),
                ChatMessage(role="user", content=build_prompt(question, context)),
            ]
        )
        return RAGAnswer(answer=reply, sources=build_sources(results))
