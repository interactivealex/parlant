# Copyright 2026 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Awaitable, Callable, Generic, Mapping, Optional, Sequence, cast

from typing_extensions import override, Self

from parlant.core.async_utils import ReaderWriterLock
from parlant.core.common import JSONSerializable
from parlant.core.loggers import Logger
from parlant.core.tracer import Tracer
from parlant.core.nlp.embedding import (
    Embedder,
    EmbedderFactory,
    EmbeddingCacheProvider,
)
from parlant.core.persistence.common import (
    LiteralValue,
    Where,
    WhereExpression,
    LogicalOperator,
    ensure_is_total,
)
from parlant.core.persistence.vector_database import (
    BaseDocument,
    BaseVectorCollection,
    DeleteResult,
    InsertResult,
    SimilarDocumentResult,
    UpdateResult,
    VectorDatabase,
    TDocument,
)

try:
    import asyncpg  # type: ignore[import-untyped]
except ImportError:
    asyncpg = None  # type: ignore[assignment]

try:
    from pgvector.asyncpg import HalfVector  # type: ignore[import-untyped]
except ImportError:
    HalfVector = None  # type: ignore[assignment, misc]


# Maximum dimension for pgvector HNSW index using halfvec (float16).
# halfvec supports HNSW up to 4000 dimensions (vs 2000 for vector/float32).
# Larger embeddings fall back to sequential scan (still correct results, just slower).
_PGVECTOR_HALFVEC_HNSW_MAX_DIM = 4000


class _VectorWhereTranslator:
    """Translates MongoDB-style Where filters to PostgreSQL SQL for vector tables."""

    # Fields stored as top-level columns (not inside JSONB metadata).
    # "id" maps to the "doc_id" column so lookups use the PRIMARY KEY btree
    # instead of the GIN jsonb_path_ops index.
    INDEXED_FIELDS: dict[str, str] = {"id": "doc_id"}

    def __init__(self) -> None:
        self._params: list[Any] = []
        self._param_idx: int = 0

    def _next_param(self, value: Any) -> str:
        self._param_idx += 1
        self._params.append(value)
        return f"${self._param_idx}"

    def _typed_param(self, value: LiteralValue) -> str:
        placeholder = self._next_param(value)

        if isinstance(value, bool):
            return f"to_jsonb({placeholder}::boolean)"
        if isinstance(value, int):
            return f"to_jsonb({placeholder}::bigint)"
        if isinstance(value, float):
            return f"to_jsonb({placeholder}::double precision)"
        return f"to_jsonb({placeholder}::text)"

    def _containment_param(self, field_name: str, value: LiteralValue) -> str:
        """Create a jsonb containment parameter like '{"field": value}'."""
        self._param_idx += 1
        self._params.append({field_name: value})
        return f"${self._param_idx}::jsonb"

    def translate(self, where: Where) -> tuple[str, list[Any]]:
        self._params = []
        self._param_idx = 0

        if not where:
            return "", []

        clause = self._translate_where(where)
        return clause, self._params

    def _translate_where(self, where: Where) -> str:
        if not where:
            return "TRUE"

        first_key = next(iter(where.keys()))

        if first_key in ("$and", "$or"):
            return self._translate_logical(cast(LogicalOperator, where))
        else:
            return self._translate_expression(cast(WhereExpression, where))

    def _translate_logical(self, op: LogicalOperator) -> str:
        parts: list[str] = []

        if "$and" in op:
            sub_clauses = [self._translate_where(sub) for sub in op["$and"]]
            parts.append("(" + " AND ".join(sub_clauses) + ")")

        if "$or" in op:
            sub_clauses = [self._translate_where(sub) for sub in op["$or"]]
            parts.append("(" + " OR ".join(sub_clauses) + ")")

        return " AND ".join(parts) if parts else "TRUE"

    def _translate_expression(self, expr: WhereExpression) -> str:
        clauses: list[str] = []

        for field_name, field_filter in expr.items():
            column = self.INDEXED_FIELDS.get(field_name)
            is_indexed = column is not None
            ref = f'"{column}"' if is_indexed else f"metadata->'{field_name}'"

            for operator, filter_value in field_filter.items():
                if operator == "$eq":
                    if is_indexed:
                        p = self._next_param(filter_value)
                        clauses.append(f"{ref} = {p}")
                    else:
                        p = self._containment_param(field_name, cast(LiteralValue, filter_value))
                        clauses.append(f"metadata @> {p}")
                elif operator == "$ne":
                    p = (
                        self._next_param(filter_value)
                        if is_indexed
                        else self._typed_param(cast(LiteralValue, filter_value))
                    )
                    clauses.append(f"{ref} != {p}")
                elif operator == "$gt":
                    p = (
                        self._next_param(filter_value)
                        if is_indexed
                        else self._typed_param(cast(LiteralValue, filter_value))
                    )
                    clauses.append(f"{ref} > {p}")
                elif operator == "$gte":
                    p = (
                        self._next_param(filter_value)
                        if is_indexed
                        else self._typed_param(cast(LiteralValue, filter_value))
                    )
                    clauses.append(f"{ref} >= {p}")
                elif operator == "$lt":
                    p = (
                        self._next_param(filter_value)
                        if is_indexed
                        else self._typed_param(cast(LiteralValue, filter_value))
                    )
                    clauses.append(f"{ref} < {p}")
                elif operator == "$lte":
                    p = (
                        self._next_param(filter_value)
                        if is_indexed
                        else self._typed_param(cast(LiteralValue, filter_value))
                    )
                    clauses.append(f"{ref} <= {p}")
                elif operator == "$in":
                    values = cast(list[LiteralValue], filter_value)
                    if not values:
                        clauses.append("FALSE")
                    elif is_indexed:
                        placeholders = [self._next_param(v) for v in values]
                        clauses.append(f"{ref} IN ({', '.join(placeholders)})")
                    else:
                        alternatives = [
                            f"metadata @> {self._containment_param(field_name, v)}" for v in values
                        ]
                        clauses.append(f"({' OR '.join(alternatives)})")
                elif operator == "$nin":
                    values = cast(list[LiteralValue], filter_value)
                    if values:
                        if is_indexed:
                            placeholders = [self._next_param(v) for v in values]
                        else:
                            placeholders = [self._typed_param(v) for v in values]
                        clauses.append(f"{ref} NOT IN ({', '.join(placeholders)})")

        return " AND ".join(clauses) if clauses else "TRUE"


class PostgresVectorDatabase(VectorDatabase):
    def __init__(
        self,
        dsn: str,
        logger: Logger,
        tracer: Tracer,
        embedder_factory: EmbedderFactory,
        embedding_cache_provider: EmbeddingCacheProvider,
        pool: Optional[asyncpg.Pool[asyncpg.Record]] = None,
        statement_timeout_ms: int = 30_000,
        lock_timeout_ms: int = 10_000,
    ) -> None:
        self._dsn = dsn
        self._logger = logger
        self._tracer = tracer
        self._embedder_factory = embedder_factory
        self._embedding_cache_provider = embedding_cache_provider

        self._external_pool = pool
        self._pool: Optional[asyncpg.Pool[asyncpg.Record]] = None
        self._collections: dict[str, PostgresVectorCollection[BaseDocument]] = {}
        self._server_settings = {
            "statement_timeout": str(statement_timeout_ms),
            "lock_timeout": str(lock_timeout_ms),
        }

    async def __aenter__(self) -> Self:
        if self._external_pool is not None:
            self._pool = self._external_pool
            # Ensure pgvector extension exists and register vector types on
            # connections from the shared pool (which only has JSON codec).
            await self._pool.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await self._register_vector_on_pool(self._pool)
        else:
            # Create the pgvector extension BEFORE creating the pool, because
            # _init_connection calls register_vector() which requires the
            # extension's types to already exist in pg_catalog.
            bootstrap_conn = await asyncpg.connect(dsn=self._dsn)
            try:
                await bootstrap_conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            finally:
                await bootstrap_conn.close()

            self._pool = await asyncpg.create_pool(
                dsn=self._dsn,
                min_size=2,
                max_size=10,
                init=self._init_connection,
                server_settings=self._server_settings,
            )
        # Ensure metadata table exists
        await self._pool.execute("""
            CREATE TABLE IF NOT EXISTS _vector_metadata (
                key TEXT PRIMARY KEY,
                value JSONB NOT NULL
            )
        """)
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        if self._pool is not None and self._external_pool is None:
            await self._pool.close()
            self._pool = None

    @staticmethod
    async def _init_connection(conn: asyncpg.Connection[asyncpg.Record]) -> None:
        """Register pgvector type and JSON codec on each connection."""
        from pgvector.asyncpg import register_vector  # type: ignore[import-untyped]

        await register_vector(conn)
        await conn.set_type_codec(
            "jsonb",
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
        )

    @staticmethod
    async def _register_vector_on_pool(pool: asyncpg.Pool[asyncpg.Record]) -> None:
        """Register pgvector types on all existing connections in a shared pool.

        When using a shared pool whose init callback only sets JSON codec,
        this ensures vector types are available on already-open connections.
        """
        from pgvector.asyncpg import register_vector  # type: ignore[import-untyped]

        # Register on the current idle connections by acquiring and releasing them
        async with pool.acquire() as conn:
            await register_vector(conn)

    def _get_pool(self) -> asyncpg.Pool[asyncpg.Record]:
        if self._pool is None:
            raise RuntimeError("Database pool not initialized. Use async with.")
        return self._pool

    @staticmethod
    def _table_name(collection_name: str, suffix: str = "") -> str:
        result = collection_name.replace("-", "_").replace(".", "_")
        if suffix:
            result = f"{result}_{suffix}"

        if not re.match(r"^[a-zA-Z0-9_]+$", result):
            raise ValueError(f"Invalid table name: {result}")

        return result

    def _format_embedded_table(self, name: str, embedder_type: type[Embedder]) -> str:
        return self._table_name(name, embedder_type.__name__)

    async def _table_exists(self, table_name: str) -> bool:
        pool = self._get_pool()
        row = await pool.fetchrow(
            "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_class WHERE relname = $1 AND relkind = 'r')",
            table_name,
        )
        return bool(row and row["exists"])

    async def _create_unembedded_table(self, table_name: str) -> None:
        pool = self._get_pool()
        await pool.execute(f"""
            CREATE TABLE IF NOT EXISTS "{table_name}" (
                doc_id TEXT PRIMARY KEY,
                content TEXT NOT NULL DEFAULT '',
                checksum TEXT NOT NULL DEFAULT '',
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
            )
        """)
        # Drop old default-opclass GIN index if it exists, then create with jsonb_path_ops
        await pool.execute(f'DROP INDEX IF EXISTS "idx_{table_name}_metadata"')
        await pool.execute(
            f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_metadata_pathops" '
            f'ON "{table_name}" USING GIN (metadata jsonb_path_ops)'
        )

    async def _create_embedded_table(self, table_name: str, dimensions: int) -> None:
        pool = self._get_pool()
        await pool.execute(f"""
            CREATE TABLE IF NOT EXISTS "{table_name}" (
                doc_id TEXT PRIMARY KEY,
                content TEXT NOT NULL DEFAULT '',
                checksum TEXT NOT NULL DEFAULT '',
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                embedding halfvec({dimensions})
            )
        """)

        # Drop old default-opclass GIN index if it exists, then create with jsonb_path_ops
        await pool.execute(f'DROP INDEX IF EXISTS "idx_{table_name}_metadata"')
        await pool.execute(
            f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_metadata_pathops" '
            f'ON "{table_name}" USING GIN (metadata jsonb_path_ops)'
        )

        # Create HNSW index for cosine similarity if dimensions are within limit
        if dimensions <= _PGVECTOR_HALFVEC_HNSW_MAX_DIM:
            await pool.execute(f"""
                CREATE INDEX IF NOT EXISTS "idx_{table_name}_hnsw"
                ON "{table_name}"
                USING hnsw (embedding halfvec_cosine_ops)
            """)

    async def _sync_embedded_with_unembedded(
        self,
        unembedded_table: str,
        embedded_table: str,
        embedder: Embedder,
    ) -> None:
        """Ensure embedded table is in sync with unembedded table (source of truth).

        Uses two transaction phases:
        - Phase 1 (REPEATABLE READ): Delete orphans + read diffs for a consistent snapshot.
        - Phase 2: Embed content (external API call, outside any transaction).
        - Phase 3: Write updates/inserts in a single transaction for atomicity.
        """
        pool = self._get_pool()

        # Phase 1: Consistent snapshot — delete orphans and read diffs
        async with pool.acquire() as conn:
            async with conn.transaction(isolation="repeatable_read"):
                # Batch remove orphaned embedded docs (NOT EXISTS enables hash anti-join)
                await conn.execute(
                    f"""
                    DELETE FROM "{embedded_table}" e
                    WHERE NOT EXISTS (
                        SELECT 1 FROM "{unembedded_table}" u WHERE u.doc_id = e.doc_id
                    )
                    """
                )

                # Batch find rows with changed checksums
                changed_rows = await conn.fetch(
                    f"""
                    SELECT u.doc_id, u.content, u.checksum, u.metadata
                    FROM "{unembedded_table}" u
                    JOIN "{embedded_table}" e ON u.doc_id = e.doc_id
                    WHERE u.checksum != e.checksum
                    """
                )

                # Batch find new rows (exist in unembedded but not in embedded)
                new_rows = await conn.fetch(
                    f"""
                    SELECT u.doc_id, u.content, u.checksum, u.metadata
                    FROM "{unembedded_table}" u
                    LEFT JOIN "{embedded_table}" e ON u.doc_id = e.doc_id
                    WHERE e.doc_id IS NULL
                    """
                )

                # Batch find rows where only metadata changed (same checksum, different metadata)
                metadata_changed_rows = await conn.fetch(
                    f"""
                    SELECT u.doc_id, u.content, u.checksum, u.metadata
                    FROM "{unembedded_table}" u
                    JOIN "{embedded_table}" e ON u.doc_id = e.doc_id
                    WHERE u.checksum = e.checksum
                      AND u.metadata IS DISTINCT FROM e.metadata
                    """
                )

        # Phase 2: Embed content (external API call — outside any transaction)
        all_rows_to_embed = list(changed_rows) + list(new_rows)
        all_vectors: Sequence[Sequence[float]] = []
        if all_rows_to_embed:
            contents = [row["content"] for row in all_rows_to_embed]
            all_vectors = (await embedder.embed(contents)).vectors

        # Phase 3: Write all changes in a single transaction for atomicity
        has_writes = metadata_changed_rows or changed_rows or new_rows
        if not has_writes:
            return

        async with pool.acquire() as conn:
            async with conn.transaction():
                if metadata_changed_rows:
                    update_args = [
                        (
                            dict(row["metadata"])
                            if isinstance(row["metadata"], dict)
                            else row["metadata"],
                            row["doc_id"],
                        )
                        for row in metadata_changed_rows
                    ]
                    await conn.executemany(
                        f'UPDATE "{embedded_table}" SET metadata = $1::jsonb WHERE doc_id = $2',
                        update_args,
                    )

                # Batch UPDATE changed rows
                if changed_rows:
                    changed_args = [
                        (
                            row["content"],
                            row["checksum"],
                            dict(row["metadata"])
                            if isinstance(row["metadata"], dict)
                            else row["metadata"],
                            HalfVector(all_vectors[i]),
                            row["doc_id"],
                        )
                        for i, row in enumerate(changed_rows)
                    ]
                    await conn.executemany(
                        f"""
                        UPDATE "{embedded_table}"
                        SET content = $1, checksum = $2, metadata = $3::jsonb, embedding = $4
                        WHERE doc_id = $5
                        """,
                        changed_args,
                    )

                # Batch INSERT new rows
                if new_rows:
                    offset = len(changed_rows)
                    insert_args = [
                        (
                            row["doc_id"],
                            row["content"],
                            row["checksum"],
                            dict(row["metadata"])
                            if isinstance(row["metadata"], dict)
                            else row["metadata"],
                            HalfVector(all_vectors[offset + i]),
                        )
                        for i, row in enumerate(new_rows)
                    ]
                    await conn.executemany(
                        f"""
                        INSERT INTO "{embedded_table}" (doc_id, content, checksum, metadata, embedding)
                        VALUES ($1, $2, $3, $4::jsonb, $5)
                        ON CONFLICT (doc_id) DO UPDATE
                        SET content = EXCLUDED.content, checksum = EXCLUDED.checksum,
                            metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding
                        """,
                        insert_args,
                    )

    @staticmethod
    def _advisory_lock_id(table: str) -> int:
        """Derive a stable advisory-lock ID from a table name."""
        digest = hashlib.md5(table.encode()).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=True)

    async def _load_and_migrate_documents(
        self,
        unembedded_table: str,
        embedded_table: str,
        embedder_type: type[Embedder],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> None:
        """Run document_loader on all documents, migrating as needed.

        Acquires an advisory lock so concurrent workers don't duplicate migration work.
        """
        pool = self._get_pool()
        embedder = self._embedder_factory.create_embedder(embedder_type)

        lock_id = self._advisory_lock_id(unembedded_table)
        async with pool.acquire() as lock_conn:
            await lock_conn.execute("SELECT pg_advisory_lock($1)", lock_id)
            try:
                await self._do_load_and_migrate(
                    pool, unembedded_table, embedded_table, embedder, document_loader
                )
            finally:
                await lock_conn.execute("SELECT pg_advisory_unlock($1)", lock_id)

    _MIGRATION_BATCH_SIZE = 500

    async def _do_load_and_migrate(
        self,
        pool: asyncpg.Pool[asyncpg.Record],
        unembedded_table: str,
        embedded_table: str,
        embedder: Embedder,
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> None:
        """Inner migration logic (must be called under advisory lock).

        Streams rows in batches via a server-side cursor to avoid loading
        the entire collection into memory.
        """
        # Fetch documents in batches to avoid loading entire collection into memory.
        all_docs: list[BaseDocument] = []
        offset = 0
        while True:
            batch = await pool.fetch(
                f'SELECT doc_id, content, checksum, metadata FROM "{unembedded_table}" '
                f"ORDER BY doc_id LIMIT {self._MIGRATION_BATCH_SIZE} OFFSET {offset}",
            )
            if not batch:
                break
            all_docs.extend(self._row_to_document(row) for row in batch)
            offset += len(batch)

        for doc in all_docs:
            try:
                if loaded_doc := await document_loader(doc):
                    if loaded_doc != doc:
                        metadata = {k: v for k, v in loaded_doc.items() if k not in ("content",)}
                        await pool.execute(
                            f"""
                            UPDATE "{unembedded_table}"
                            SET content = $1, checksum = $2, metadata = $3::jsonb
                            WHERE doc_id = $4
                            """,
                            loaded_doc.get("content", ""),
                            loaded_doc.get("checksum", ""),
                            metadata,
                            loaded_doc["id"],
                        )
                else:
                    self._logger.warning(f'Failed to load document "{doc}"')
                    await pool.execute(
                        f'DELETE FROM "{unembedded_table}" WHERE doc_id = $1', doc["id"]
                    )
            except Exception as e:
                self._logger.error(f"Failed to load document '{doc}' with error: {e}.")

        # Now sync embedded table
        await self._sync_embedded_with_unembedded(unembedded_table, embedded_table, embedder)

    @override
    async def create_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
    ) -> PostgresVectorCollection[TDocument]:
        if name in self._collections:
            raise ValueError(f'Collection "{name}" already exists.')

        embedder = self._embedder_factory.create_embedder(embedder_type)

        unembedded_table = self._table_name(name, "unembedded")
        embedded_table = self._format_embedded_table(name, embedder_type)

        await self._create_unembedded_table(unembedded_table)
        await self._create_embedded_table(embedded_table, embedder.dimensions)

        collection = PostgresVectorCollection[TDocument](
            pool=self._get_pool(),
            logger=self._logger,
            tracer=self._tracer,
            unembedded_table=unembedded_table,
            embedded_table=embedded_table,
            name=name,
            schema=schema,
            embedder=embedder,
            embedding_cache_provider=self._embedding_cache_provider,
        )

        self._collections[name] = cast(PostgresVectorCollection[BaseDocument], collection)
        return collection

    @override
    async def get_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> PostgresVectorCollection[TDocument]:
        if collection := self._collections.get(name):
            return cast(PostgresVectorCollection[TDocument], collection)

        embedder = self._embedder_factory.create_embedder(embedder_type)
        unembedded_table = self._table_name(name, "unembedded")
        embedded_table = self._format_embedded_table(name, embedder_type)

        if not await self._table_exists(unembedded_table):
            raise ValueError(f'PostgresVector collection "{name}" not found.')

        # Create embedded table if it doesn't exist yet (e.g. embedder type changed)
        if not await self._table_exists(embedded_table):
            await self._create_embedded_table(embedded_table, embedder.dimensions)

        # Load/migrate documents and sync
        await self._load_and_migrate_documents(
            unembedded_table, embedded_table, embedder_type, document_loader
        )

        collection_obj = PostgresVectorCollection[TDocument](
            pool=self._get_pool(),
            logger=self._logger,
            tracer=self._tracer,
            unembedded_table=unembedded_table,
            embedded_table=embedded_table,
            name=name,
            schema=schema,
            embedder=embedder,
            embedding_cache_provider=self._embedding_cache_provider,
        )

        self._collections[name] = cast(PostgresVectorCollection[BaseDocument], collection_obj)
        return collection_obj

    @override
    async def get_or_create_collection(
        self,
        name: str,
        schema: type[TDocument],
        embedder_type: type[Embedder],
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> PostgresVectorCollection[TDocument]:
        if collection := self._collections.get(name):
            return cast(PostgresVectorCollection[TDocument], collection)

        embedder = self._embedder_factory.create_embedder(embedder_type)
        unembedded_table = self._table_name(name, "unembedded")
        embedded_table = self._format_embedded_table(name, embedder_type)

        await self._create_unembedded_table(unembedded_table)
        await self._create_embedded_table(embedded_table, embedder.dimensions)

        # Load/migrate documents and sync
        await self._load_and_migrate_documents(
            unembedded_table, embedded_table, embedder_type, document_loader
        )

        collection_obj = PostgresVectorCollection[TDocument](
            pool=self._get_pool(),
            logger=self._logger,
            tracer=self._tracer,
            unembedded_table=unembedded_table,
            embedded_table=embedded_table,
            name=name,
            schema=schema,
            embedder=embedder,
            embedding_cache_provider=self._embedding_cache_provider,
        )

        self._collections[name] = cast(PostgresVectorCollection[BaseDocument], collection_obj)
        return collection_obj

    @override
    async def delete_collection(
        self,
        name: str,
    ) -> None:
        if name not in self._collections:
            raise ValueError(f'Collection "{name}" not found.')

        pool = self._get_pool()
        collection = self._collections[name]

        await pool.execute(f'DROP TABLE IF EXISTS "{collection._unembedded_table}"')
        await pool.execute(f'DROP TABLE IF EXISTS "{collection._embedded_table}"')
        del self._collections[name]

    @override
    async def upsert_metadata(
        self,
        key: str,
        value: JSONSerializable,
    ) -> None:
        pool = self._get_pool()
        await pool.execute(
            """
            INSERT INTO _vector_metadata (key, value)
            VALUES ($1, $2::jsonb)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            """,
            key,
            value,
        )

    @override
    async def remove_metadata(
        self,
        key: str,
    ) -> None:
        pool = self._get_pool()
        result = await pool.execute("DELETE FROM _vector_metadata WHERE key = $1", key)
        if result == "DELETE 0":
            raise ValueError(f'Metadata with key "{key}" not found.')

    @override
    async def read_metadata(
        self,
    ) -> Mapping[str, JSONSerializable]:
        pool = self._get_pool()
        rows = await pool.fetch("SELECT key, value FROM _vector_metadata")
        return {row["key"]: cast(JSONSerializable, row["value"]) for row in rows}

    @staticmethod
    def _row_to_document(row: asyncpg.Record) -> BaseDocument:
        raw_meta = row["metadata"]
        if isinstance(raw_meta, str):
            metadata: dict[str, Any] = json.loads(raw_meta) if raw_meta else {}
        elif isinstance(raw_meta, dict):
            metadata = dict(raw_meta)
        else:
            metadata = {}
        # Ensure content and checksum from the row columns are in the doc
        doc: dict[str, Any] = {
            **metadata,
            "content": row["content"],
            "checksum": row["checksum"],
        }
        # doc_id maps to the document's "id" field
        if "id" not in doc:
            doc["id"] = row["doc_id"]
        return cast(BaseDocument, doc)


class PostgresVectorCollection(Generic[TDocument], BaseVectorCollection[TDocument]):
    def __init__(
        self,
        pool: asyncpg.Pool[asyncpg.Record],
        logger: Logger,
        tracer: Tracer,
        unembedded_table: str,
        embedded_table: str,
        name: str,
        schema: type[TDocument],
        embedder: Embedder,
        embedding_cache_provider: EmbeddingCacheProvider,
    ) -> None:
        super().__init__(tracer)

        self._pool = pool
        self._logger = logger
        self._tracer = tracer
        self._unembedded_table = unembedded_table
        self._embedded_table = embedded_table
        self._name = name
        self._schema = schema
        self._embedder = embedder
        self._embedding_cache_provider = embedding_cache_provider

        self._lock = ReaderWriterLock()

    async def _get_embedding(self, content: str) -> Sequence[float]:
        """Get embedding from cache or compute it."""
        if e := await self._embedding_cache_provider().get(
            embedder_type=type(self._embedder),
            texts=[content],
        ):
            return e.vectors[0]

        result = await self._embedder.embed([content])
        await self._embedding_cache_provider().set(
            embedder_type=type(self._embedder),
            texts=[content],
            vectors=list(result.vectors),
        )
        return result.vectors[0]

    @override
    async def find(
        self,
        filters: Where,
    ) -> Sequence[TDocument]:
        async with self._lock.reader_lock:
            translator = _VectorWhereTranslator()
            where_clause, params = translator.translate(filters)

            sql = f'SELECT doc_id, content, checksum, metadata FROM "{self._embedded_table}"'
            if where_clause:
                sql += f" WHERE {where_clause}"

            rows = await self._pool.fetch(sql, *params)
            return [cast(TDocument, PostgresVectorDatabase._row_to_document(r)) for r in rows]

    @override
    async def find_one(
        self,
        filters: Where,
    ) -> Optional[TDocument]:
        async with self._lock.reader_lock:
            translator = _VectorWhereTranslator()
            where_clause, params = translator.translate(filters)

            sql = f'SELECT doc_id, content, checksum, metadata FROM "{self._embedded_table}"'
            if where_clause:
                sql += f" WHERE {where_clause}"
            sql += " LIMIT 1"

            row = await self._pool.fetchrow(sql, *params)
            if row is None:
                return None

            return cast(TDocument, PostgresVectorDatabase._row_to_document(row))

    @override
    async def insert_one(
        self,
        document: TDocument,
    ) -> InsertResult:
        ensure_is_total(document, self._schema)

        content = document["content"]
        embedding = await self._get_embedding(content)

        metadata = {k: v for k, v in document.items() if k not in ("content",)}

        async with self._lock.writer_lock:
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    # Insert into unembedded table (upsert for restart-safety)
                    await conn.execute(
                        f"""
                        INSERT INTO "{self._unembedded_table}" (doc_id, content, checksum, metadata)
                        VALUES ($1, $2, $3, $4::jsonb)
                        ON CONFLICT (doc_id) DO UPDATE
                        SET content = EXCLUDED.content,
                            checksum = EXCLUDED.checksum,
                            metadata = EXCLUDED.metadata
                        """,
                        document["id"],
                        content,
                        document.get("checksum", ""),
                        metadata,
                    )

                    # Insert into embedded table with vector (upsert for restart-safety)
                    await conn.execute(
                        f"""
                        INSERT INTO "{self._embedded_table}" (doc_id, content, checksum, metadata, embedding)
                        VALUES ($1, $2, $3, $4::jsonb, $5)
                        ON CONFLICT (doc_id) DO UPDATE
                        SET content = EXCLUDED.content,
                            checksum = EXCLUDED.checksum,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding
                        """,
                        document["id"],
                        content,
                        document.get("checksum", ""),
                        metadata,
                        HalfVector(embedding),
                    )

        return InsertResult(acknowledged=True)

    @override
    async def update_one(
        self,
        filters: Where,
        params: TDocument,
        upsert: bool = False,
    ) -> UpdateResult[TDocument]:
        async with self._lock.writer_lock:
            translator = _VectorWhereTranslator()
            where_clause, sql_params = translator.translate(filters)

            sql = f'SELECT doc_id, content, checksum, metadata FROM "{self._embedded_table}"'
            if where_clause:
                sql += f" WHERE {where_clause}"
            sql += " LIMIT 1"

            # Phase 1: Read existing document (no transaction held)
            row = await self._pool.fetchrow(sql, *sql_params)

            if row is not None:
                doc = PostgresVectorDatabase._row_to_document(row)
                updated_document = cast(TDocument, {**doc, **params})

                content = str(params.get("content", doc.get("content", "")))
                metadata = {k: v for k, v in updated_document.items() if k not in ("content",)}

                # Phase 2: Compute embedding outside any transaction
                embedding = await self._get_embedding(content)

                # Phase 3: Write in a single transaction (upsert for safety)
                async with self._pool.acquire() as conn:
                    async with conn.transaction():
                        await conn.execute(
                            f"""
                            UPDATE "{self._unembedded_table}"
                            SET content = $1, checksum = $2, metadata = $3::jsonb
                            WHERE doc_id = $4
                            """,
                            content,
                            updated_document.get("checksum", ""),
                            metadata,
                            doc["id"],
                        )

                        await conn.execute(
                            f"""
                            UPDATE "{self._embedded_table}"
                            SET content = $1, checksum = $2, metadata = $3::jsonb, embedding = $4
                            WHERE doc_id = $5
                            """,
                            content,
                            updated_document.get("checksum", ""),
                            metadata,
                            HalfVector(embedding),
                            doc["id"],
                        )

                return UpdateResult(
                    acknowledged=True,
                    matched_count=1,
                    modified_count=1,
                    updated_document=updated_document,
                )

            elif upsert:
                ensure_is_total(params, self._schema)

                content = params["content"]
                metadata = {k: v for k, v in params.items() if k not in ("content",)}

                # Compute embedding outside any transaction
                embedding = await self._get_embedding(content)

                async with self._pool.acquire() as conn:
                    async with conn.transaction():
                        await conn.execute(
                            f"""
                            INSERT INTO "{self._unembedded_table}" (doc_id, content, checksum, metadata)
                            VALUES ($1, $2, $3, $4::jsonb)
                            ON CONFLICT (doc_id) DO UPDATE
                            SET content = EXCLUDED.content,
                                checksum = EXCLUDED.checksum,
                                metadata = EXCLUDED.metadata
                            """,
                            params["id"],
                            content,
                            params.get("checksum", ""),
                            metadata,
                        )

                        await conn.execute(
                            f"""
                            INSERT INTO "{self._embedded_table}" (doc_id, content, checksum, metadata, embedding)
                            VALUES ($1, $2, $3, $4::jsonb, $5)
                            ON CONFLICT (doc_id) DO UPDATE
                            SET content = EXCLUDED.content,
                                checksum = EXCLUDED.checksum,
                                metadata = EXCLUDED.metadata,
                                embedding = EXCLUDED.embedding
                            """,
                            params["id"],
                            content,
                            params.get("checksum", ""),
                            metadata,
                            HalfVector(embedding),
                        )

                return UpdateResult(
                    acknowledged=True,
                    matched_count=0,
                    modified_count=0,
                    updated_document=params,
                )

            return UpdateResult(
                acknowledged=True,
                matched_count=0,
                modified_count=0,
                updated_document=None,
            )

    @override
    async def delete_one(
        self,
        filters: Where,
    ) -> DeleteResult[TDocument]:
        async with self._lock.writer_lock:
            translator = _VectorWhereTranslator()
            where_clause, params = translator.translate(filters)

            sql = f'SELECT doc_id, content, checksum, metadata FROM "{self._embedded_table}"'
            if where_clause:
                sql += f" WHERE {where_clause}"
            sql += " FOR UPDATE LIMIT 1"

            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    row = await conn.fetchrow(sql, *params)

                    if row is None:
                        return DeleteResult(
                            acknowledged=True,
                            deleted_count=0,
                            deleted_document=None,
                        )

                    doc = PostgresVectorDatabase._row_to_document(row)
                    doc_id = row["doc_id"]

                    await conn.execute(
                        f'DELETE FROM "{self._unembedded_table}" WHERE doc_id = $1',
                        doc_id,
                    )
                    await conn.execute(
                        f'DELETE FROM "{self._embedded_table}" WHERE doc_id = $1',
                        doc_id,
                    )

            return DeleteResult(
                acknowledged=True,
                deleted_count=1,
                deleted_document=cast(TDocument, doc),
            )

    @override
    async def do_find_similar_documents(
        self,
        filters: Where,
        query: str,
        k: int,
        hints: Mapping[str, Any] = {},
    ) -> Sequence[SimilarDocumentResult[TDocument]]:
        async with self._lock.reader_lock:
            query_embedding = (await self._embedder.embed([query], hints)).vectors[0]

            translator = _VectorWhereTranslator()
            where_clause, params = translator.translate(filters)

            # Use pgvector cosine distance operator <=>
            # cosine distance = 1 - cosine_similarity, range [0, 2]
            param_idx = len(params) + 1
            params.append(HalfVector(query_embedding))

            sql = f"""
                SELECT doc_id, content, checksum, metadata,
                       embedding <=> ${param_idx} AS distance
                FROM "{self._embedded_table}"
            """

            if where_clause:
                sql += f" WHERE {where_clause}"

            sql += f" ORDER BY distance LIMIT {k}"

            rows = await self._pool.fetch(sql, *params)

            if not rows:
                return []

            results: list[SimilarDocumentResult[TDocument]] = []
            for row in rows:
                doc = PostgresVectorDatabase._row_to_document(row)
                results.append(
                    SimilarDocumentResult(
                        document=cast(TDocument, doc),
                        distance=float(row["distance"]),
                    )
                )

            self._logger.trace(
                f"Similar documents found\n{json.dumps([r.document for r in results[:1]], indent=2, default=str)}"
            )

            return results
