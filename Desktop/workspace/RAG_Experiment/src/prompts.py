"""Shared prompt templates for all RAG frameworks."""

ANSWER_PROMPT = (
    "Please answer the question based on the following document content.\n\n"
    "Document content:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def format_answer_prompt(question: str, context: str) -> str:
    return ANSWER_PROMPT.format(context=context, question=question)
