# Project Status Report

Generated from direct inspection of `/Users/jagadeesh/knowledge-platform` on 2026-11-25 after the production-hardening work was committed as `ee991c5`.

## 1. Current Summary

The project is now a **multi-tenant, production-shaped PDF RAG backend**, not just an early prototype.

It accepts PDF uploads through a durable Postgres-backed job queue, processes them in a background worker, stores versioned chunks in Postgres pgvector with tenant isolation, and answers questions using Ollama embeddings and chat models.

The core production path is:

1. Client sends `POST /documents` with a PDF and API key.
2. API validates file type and size, stores the PDF bytes in Postgres, creates an ingestion job, and returns `202 {job_id}`.
3. `kp-worker` claims jobs with `FOR UPDATE SKIP LOCKED`.
4. Worker parses the PDF, performs semantic chunking, embeds with Ollama, writes new versioned chunks to pgvector, then deletes older versions for the same tenant/document.
5. Client polls `GET /jobs/{job_id}` for status.
6. `POST /ask` retrieves tenant-scoped chunks from pgvector and returns a grounded answer with sources.

## 2. What Is Implemented

### API

- FastAPI app factory and module-level ASGI app.
- Request ID middleware.
- Typed error handlers:
  - domain errors → `422`
  - unavailable dependencies → `503`
  - unexpected errors → `500`
- API-key authentication:
  - `Authorization: Bearer <api_key>`
  - `KP_API_KEYS` maps API key → tenant ID.
  - `/health` and `/ready` are public.
  - `/documents`, `/jobs/{job_id}`, `/ask`, and `DELETE /documents/{id}` are tenant-protected.
- Upload validation:
  - PDF filename check.
  - empty upload rejection.
  - chunked upload reading with `KP_MAX_UPLOAD_BYTES` limit.

### Endpoints

| Endpoint | Method | Behavior |
|---|---|---|
| `/health` | GET | Static service metadata. No dependency checks. |
| `/ready` | GET | Checks RAG dependencies and Postgres job repository. Returns `503` when unavailable. |
| `/documents` | POST | Uploads PDF, creates durable ingestion job, returns `202`. |
| `/jobs/{job_id}` | GET | Returns tenant-scoped job status. |
| `/ask` | POST | Retrieves tenant-scoped chunks and returns grounded answer with sources. |
| `/documents/{document_id}` | DELETE | Deletes tenant-scoped chunks for a document. |

FastAPI docs remain available at `/docs`, `/redoc`, and `/openapi.json`.

### RAG pipeline

- PDF loading from path and bytes using `pypdf`.
- Blank/invalid PDF handling with typed domain errors.
- Semantic chunking:
  - paragraph and sentence units
  - batched Ollama embeddings
  - cosine-distance topic-shift boundaries
  - hard chunk size limits
  - boundary-aware overlap
  - clause/character fallback
- Versioned ingestion:
  - each ingest generates a new `ingest_version`
  - chunk IDs are namespaced by tenant and version
  - new chunks are added first
  - only older versions for the same tenant/document are deleted
  - cleanup failure preserves old data instead of wiping it
- Tenant isolation:
  - tenant ID stored in chunk metadata
  - tenant ID prefixed into chunk IDs
  - tenant filters applied to ingest, query, and delete
- Grounded answering:
  - retriever with `top_k`, `where`, and `max_distance`
  - numbered context blocks
  - system prompt that requires citation and admits insufficient information
  - source metadata returned with answers

### Storage and reliability

- Postgres pgvector is the default vector store.
- SQL migration creates:
  - `document_chunks`
  - `ingestion_jobs`
  - HNSW vector index
  - lookup and metadata indexes
  - job claim and tenant indexes
- Connection pooling with `psycopg_pool`.
- Pgvector filtering supports:
  - equality
  - `$and`
  - `$ne`
  - allowlisted metadata columns only
- ChromaDB remains available as an optional local backend via `KP_VECTOR_STORE_BACKEND=chroma`.
- Ollama calls have:
  - configurable timeout
  - retry/backoff
  - typed `ServiceUnavailableError` mapping to HTTP `503`
- `/ready` verifies Ollama and Postgres paths.

### Durable ingestion jobs

- Jobs are stored in Postgres with PDF bytes, tenant, document ID, filename, status, attempts, max attempts, run time, error, and chunk count.
- Worker claims jobs using `FOR UPDATE SKIP LOCKED`.
- Failed jobs retry with exponential backoff until `KP_JOB_MAX_ATTEMPTS`.
- Stale locked jobs can be reclaimed after 5 minutes.
- Worker entrypoint: `kp-worker` or `python -m knowledge_platform.jobs.worker_loop`.

## 3. Technology Stack

| Area | Technology |
|---|---|
| Language | Python `>=3.12` |
| Package manager | `uv` |
| API | FastAPI, Uvicorn, Pydantic, Pydantic Settings |
| Embeddings/chat | Ollama |
| Primary vector store | Postgres + pgvector |
| Postgres client | `psycopg[binary,pool]`, `pgvector` |
| Optional vector store | ChromaDB |
| PDF parsing | pypdf |
| Tests | pytest, httpx `TestClient`, reportlab |
| Static checks | ruff, mypy |

## 4. Repository Layout

```text
knowledge-platform/
├── PROJECT_STATUS.md
└── backend/
    ├── .env.example
    ├── Dockerfile
    ├── .dockerignore
    ├── pyproject.toml
    ├── uv.lock
    ├── migrations/
    │   └── 001_init.sql
    ├── src/knowledge_platform/
    │   ├── api/                 FastAPI app, schemas, dependencies, error handlers
    │   ├── db/                  Postgres connection provider and migrations
    │   ├── jobs/                Job repository, worker, worker entrypoint
    │   ├── observability.py     Request ID and logging setup
    │   ├── rag/
    │   │   ├── chunking/        Semantic TextChunker
    │   │   ├── embeddings/      Embedding provider + Ollama adapter
    │   │   ├── generation/      Chat provider, prompts, RAGAnswerer
    │   │   ├── loaders/         PDFLoader
    │   │   ├── retrieval/       Retriever and context formatting
    │   │   ├── schemas/         Domain models
    │   │   └── vector_store/    VectorStore, PgVectorStore, ChromaVectorStore
    │   ├── config.py            Environment settings
    │   ├── container.py         Dependency construction
    │   ├── exceptions.py        Typed domain errors
    │   └── __main__.py          API server entrypoint
    └── tests/
```

## 5. Test Status

Last verified after commit `ee991c5`:

```text
87 passed, 2 skipped
```

Also clean:

```bash
.venv/bin/ruff check --no-cache src tests
.venv/bin/mypy --no-incremental --cache-dir=/dev/null
```

Recommended repeat command from `backend/`:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider
```

The two skipped tests are live Ollama tests requiring `RUN_LIVE_TESTS=1` and a running Ollama server with the configured models.

## 6. Environment Variables

Important settings from `backend/.env.example`:

| Variable | Default | Purpose |
|---|---|---|
| `KP_DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/knowledge` | Postgres DSN |
| `KP_VECTOR_STORE_BACKEND` | `pgvector` | `pgvector` or `chroma` |
| `KP_DB_POOL_MIN_SIZE` | `1` | Minimum DB pool size |
| `KP_DB_POOL_MAX_SIZE` | `10` | Maximum DB pool size |
| `KP_JOB_POLL_INTERVAL_SECONDS` | `1` | Worker poll interval |
| `KP_JOB_MAX_ATTEMPTS` | `3` | Max ingestion attempts |
| `KP_JOB_RETRY_BACKOFF_SECONDS` | `5` | Initial retry backoff |
| `KP_OLLAMA_HOST` | `http://localhost:11434` | Ollama server |
| `KP_OLLAMA_TIMEOUT_SECONDS` | `60` | Ollama request timeout |
| `KP_OLLAMA_RETRIES` | `2` | Ollama retry attempts |
| `KP_OLLAMA_RETRY_BACKOFF_SECONDS` | `0.5` | Initial Ollama retry backoff |
| `KP_MAX_UPLOAD_BYTES` | `20971520` | Upload size limit |
| `KP_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `KP_CHAT_MODEL` | `qwen2.5:3b` | Chat model |
| `KP_CHROMA_PATH` | `./chroma` | Local Chroma path when backend is `chroma` |
| `KP_DEFAULT_COLLECTION` | `knowledge` | Logical collection name |
| `KP_CHUNK_SIZE` | `500` | Characters per chunk |
| `KP_CHUNK_OVERLAP` | `50` | Chunk overlap |
| `KP_TOP_K` | `5` | Default retrieval count |
| `KP_SEMANTIC_BREAK_QUANTILE` | `0.75` | Topic-shift threshold |
| `KP_EMBEDDING_BATCH_SIZE` | `64` | Sentences embedded per batch |
| `KP_AUTH_ENABLED` | `true` | Enables API-key auth |
| `KP_API_KEYS` | `{}` | JSON map of API key → tenant ID |
| `KP_LOG_LEVEL` | `INFO` | Root log level |

If `KP_AUTH_ENABLED=true` and `KP_API_KEYS` is empty, protected endpoints fail closed with `401`.

## 7. Setup and Run

From `backend/`:

```bash
uv sync
cp .env.example .env
```

Requirements:

- Postgres reachable at `KP_DATABASE_URL`.
- Postgres user must be able to create the `vector` extension.
- Ollama reachable at `KP_OLLAMA_HOST`.
- Ollama models available: `nomic-embed-text` and `qwen2.5:3b` by default.

Run API:

```bash
uv run python -m knowledge_platform
```

Run worker:

```bash
uv run kp-worker
```

Migrations are applied automatically on API startup and worker startup via `apply_migrations()`.

## 8. Remaining Production Gaps

### Auth hardening

Current auth uses API keys from environment configuration. This is suitable for early multi-tenant isolation but not full identity management.

Missing:

- hashed API keys
- key rotation and expiry
- external identity provider / OAuth / JWT verification
- per-tenant quotas and rate limits

### Observability

Present:

- request IDs
- basic structured log calls
- readiness endpoint

Missing:

- metrics
- tracing
- alerting
- log shipping
- job duration/failure dashboards

### Deployment

Present:

- Dockerfile
- `.dockerignore`

Missing:

- Docker Compose
- Kubernetes manifests
- model preload step
- backup/restore policy
- retention policy
- migration tooling beyond startup SQL

### RAG quality

Missing:

- OCR
- non-PDF formats
- hybrid search
- reranking
- token-aware chunking
- embedding cache
- citation validation

### Product direction

Not implemented:

- frontend
- conversation history
- streaming answers
- agent loop
- tool registry
- MCP integration

## 9. Recommended Next Steps

1. **Auth hardening:** hashed/rotatable API keys or JWT/identity provider integration.
2. **Observability:** metrics, tracing, structured JSON logs, job metrics, and alerts.
3. **Deployment:** Compose/Kubernetes, model preload, backup/restore, and retention.
4. **Retrieval quality:** hybrid search, reranking, embedding cache, and citation validation.
5. **Agent features:** only after the above production foundation is stable.

Do not add agent frameworks or frontend work before auth, observability, and deployment are solid.

## 10. Handoff Summary

The service now has a real production foundation: tenant-scoped access, semantic chunking, Postgres pgvector storage, durable background ingestion, version-safe replacement, worker retries, upload limits, Ollama retries/timeouts, typed dependency failures, and readiness checks.

Current commit: `ee991c5 Add production RAG API with pgvector ingestion queue`.

The most important remaining work is auth hardening and observability. The "agentic" part of the product description is still future work.
