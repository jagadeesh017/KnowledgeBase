SYSTEM_PROMPT = (
    "Answer the question using only the numbered context blocks provided by the user. "
    "Cite the block numbers you used, like [1] or [2]. "
    "If the context does not contain the answer, say you do not have enough information. "
    "Do not invent facts or sources."
)


def build_prompt(question: str, context: str) -> str:
    if not context.strip():
        return (
            f"Question: {question}\n\n"
            "No context blocks were retrieved. Reply that you cannot answer."
        )
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
