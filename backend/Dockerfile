FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

ENV KP_CHROMA_PATH=/data/chroma \
    KP_OLLAMA_HOST=http://ollama:11434

VOLUME ["/data/chroma"]
EXPOSE 8000

CMD ["uv", "run", "--frozen", "--no-dev", "python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
