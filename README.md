# RAG Application - VPS Demo Setup

This version is prepared for a lightweight VPS demo:

- FastAPI RAG backend runs in Docker.
- Redis runs in Docker.
- Database operations use Supabase PostgreSQL with pgvector.
- Embeddings, text generation and PDF OCR/vision are served through OpenAI/GPT APIs.

## Architecture

```text
OpenWebUI / API Client
        |
        v
Dockerized RAG API on VPS
        |
        +--> Supabase Postgres + pgvector
        +--> Redis container
        +--> OpenAI embeddings API
        +--> OpenAI chat/completions API
        +--> OpenAI vision model for PDF OCR
```

## Files

Important deployment files:

```text
Dockerfile
docker-compose.yml
.dockerignore
.gitignore
.env.vps.supabase.example
requirements.txt
```

Do not commit real `.env.vps`, `environment.env`, API keys or database passwords.

## Supabase Requirements

Create a Supabase project and enable the `vector` extension.

In Supabase SQL editor:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

The application will create its own tables at startup.

Use the Supabase PostgreSQL connection string in `.env.vps`.

Recommended connection format:

```text
DATABASE_URL=postgresql://postgres.xxxxx:<password>@aws-0-region.pooler.supabase.com:6543/postgres?sslmode=require
```

The app automatically converts `postgresql://` to `postgresql+asyncpg://`.

## Environment Setup

Copy the example file:

```bash
cp .env.vps.supabase.example .env.vps
```

Fill these values:

```text
DATABASE_URL=
OPENAI_API_KEY=
```

Default model settings:

```text
EMBEDDING_PROVIDER=openai
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_EMBEDDING_DIMENSIONS=2048
VLLM_MODEL=gpt-4o-mini
VLM_MODEL=gpt-4o-mini
```

Important: `OPENAI_EMBEDDING_DIMENSIONS` is set to `2048` because the current database vector column is `Vector(2048)`.

## Run on VPS

Build and start:

```bash
docker compose --env-file .env.vps up -d --build
```

View logs:

```bash
docker compose --env-file .env.vps logs -f rag-api
```

Stop:

```bash
docker compose --env-file .env.vps down
```

## Health Check

```bash
curl http://localhost:8005/health
curl http://localhost:8005/api/v2/health
```

Expected result: API, Supabase/Postgres, OpenAI embeddings and GPT model services should be available.

## OpenWebUI Pipe Configuration

In the OpenWebUI pipe:

```text
RAG_API_URL=http://<vps-ip>:8005
UNIT=<unit-name>
COLLECTION=<collection-name>
```

Example:

```text
RAG_API_URL=http://1.2.3.4:8005
UNIT=krediler
COLLECTION=karesi
```

## Ingest Example

```bash
curl -sS -X POST "http://localhost:8005/api/v2/ingest" \
  -F "file=@documents/ornek.pdf" \
  -F "unit=krediler" \
  -F "collection=karesi"
```

## Scoped Query Example

```bash
curl -sS -X POST "http://localhost:8005/api/v2/query/scoped" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Bu dokümandaki ana noktaları özetle.",
    "unit": "krediler",
    "collection": "karesi",
    "top_k": 5,
    "temperature": 0
  }'
```

## Notes for Demo

- Supabase stores all document chunks, metadata and embeddings.
- OpenAI embeddings create 2048-dimensional vectors to match the current schema.
- GPT is used for both answer generation and PDF OCR/vision extraction.
- Large PDF ingestion can be slow and may consume OpenAI API credits.
- For production, add authentication, HTTPS reverse proxy and stricter CORS settings.
