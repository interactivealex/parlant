# PostgreSQL + pgvector Adapter

## Overview

Parlant supports PostgreSQL as an alternative storage backend for both document storage (replacing MongoDB) and vector/embedding storage (replacing ChromaDB). This is achieved through two adapter implementations:

- **`PostgresDocumentDatabase`** — Implements `DocumentDatabase` for structured document CRUD with cursor-based pagination
- **`PostgresVectorDatabase`** — Implements `VectorDatabase` for embedding storage and cosine similarity search using pgvector

Both adapters use `asyncpg` for high-performance async PostgreSQL access.

## Prerequisites

- PostgreSQL 15+ with the `pgvector` extension installed
- Python packages: `asyncpg>=0.29.0`, `pgvector>=0.3.0`

Install the optional dependency group:

```bash
pip install parlant[postgres]
```

### Installing pgvector on PostgreSQL

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

On most managed PostgreSQL services (Supabase, Neon, RDS, Cloud SQL), pgvector is available as a pre-installed extension.

## Configuration

### Connection String

Pass a PostgreSQL connection string (DSN) wherever you currently pass a MongoDB connection string:

```python
from parlant import Server

server = Server(
    session_store="postgresql://user:password@localhost:5432/parlant",
    customer_store="postgresql://user:password@localhost:5432/parlant",
    variable_store="postgresql://user:password@localhost:5432/parlant",
)
```

The SDK auto-detects `postgresql://` or `postgres://` prefixes and routes to the PostgreSQL adapters.

When the session store is a PostgreSQL URL, vector stores (glossary, capabilities, journeys, canned responses) automatically use pgvector for embedding storage and similarity search.

## Schema Design

### Document Database (`PostgresDocumentDatabase`)

Each collection maps to a PostgreSQL table:

```sql
CREATE TABLE "{collection_name}" (
    "id"           TEXT PRIMARY KEY,
    "version"      TEXT,
    "creation_utc" TEXT,
    data           JSONB NOT NULL DEFAULT '{}'::jsonb
);
```

- **`id`, `version`, `creation_utc`** — Indexed columns extracted from the document for fast queries
- **`data`** — JSONB column storing all remaining document fields
- **GIN index** on `data` for efficient JSONB queries
- **B-tree index** on `creation_utc` for cursor-based pagination

### Vector Database (`PostgresVectorDatabase`)

Uses a **dual-table pattern** per collection (matching ChromaDB's approach):

**Unembedded table** (source of truth):
```sql
CREATE TABLE "{name}_unembedded" (
    doc_id    TEXT PRIMARY KEY,
    content   TEXT NOT NULL DEFAULT '',
    checksum  TEXT NOT NULL DEFAULT '',
    metadata  JSONB NOT NULL DEFAULT '{}'::jsonb
);
```

**Embedded table** (with vectors):
```sql
CREATE TABLE "{name}_{EmbedderType}" (
    doc_id    TEXT PRIMARY KEY,
    content   TEXT NOT NULL DEFAULT '',
    checksum  TEXT NOT NULL DEFAULT '',
    metadata  JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(N)
);
```

- **HNSW index** with `vector_cosine_ops` for approximate nearest-neighbor search
- Dimensions up to 2000 use HNSW; larger dimensions (e.g. 3072-dim `text-embedding-3-large`) fall back to sequential scan
- Checksum-based sync ensures embedded table stays consistent with unembedded source of truth

## Query Translation

MongoDB-style `Where` filters are translated to PostgreSQL SQL:

| MongoDB Operator | SQL Translation |
|-----------------|-----------------|
| `{"field": {"$eq": value}}` | `field = $1` |
| `{"field": {"$ne": value}}` | `field != $1` |
| `{"field": {"$gt": value}}` | `field > $1` |
| `{"field": {"$in": [v1, v2]}}` | `field IN ($1, $2)` |
| `{"$and": [...]}` | `(... AND ...)` |
| `{"$or": [...]}` | `(... OR ...)` |

All queries use parameterized values to prevent SQL injection.

## Similarity Search

Vector similarity search uses pgvector's cosine distance operator (`<=>`):

```sql
SELECT doc_id, content, checksum, metadata,
       embedding <=> $1 AS distance
FROM "{embedded_table}"
WHERE {filters}
ORDER BY distance
LIMIT {k}
```

This leverages pgvector's HNSW index for efficient approximate nearest-neighbor search. The distance metric is cosine distance (0 = identical, 2 = opposite).

## Connection Pooling

Both adapters use `asyncpg.create_pool()` with configurable pool sizes (default: min=2, max=10). The pool is created on `__aenter__` and closed on `__aexit__`.

## Migration Support

The `PostgresDocumentDatabase` supports Parlant's document migration infrastructure:

- On `get_collection()`, all existing rows are passed through the `document_loader` callback
- Documents that fail migration are moved to a `{name}_failed_migrations` table
- The `DocumentStoreMigrationHelper` version tracking works identically to MongoDB

## Testing

Set the `TEST_POSTGRES_DSN` environment variable to run integration tests:

```bash
TEST_POSTGRES_DSN="postgresql://user:password@localhost:5432/parlant_test" \
  uv run pytest tests/adapters/db/test_postgres_db.py tests/adapters/vector_db/test_pgvector.py -v
```
