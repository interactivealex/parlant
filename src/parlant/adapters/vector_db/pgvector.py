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

    _VALID_FIELD_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

    def _translate_expression(self, expr: WhereExpression) -> str:
        clauses: list[str] = []

        for field_name, field_filter in expr.items():
            if not self._VALID_FIELD_RE.match(field_name):
                raise ValueError(f"Invalid field name: {field_name!r}")
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
        command_timeout: float = 30.0,
    ) -> None:
        self._dsn = dsn
        self._logger = logger
        self._tracer = tracer
        self._embedder_factory = embedder_factory
        self._embedding_cache_provider = embedding_cache_provider

        self._external_pool = pool
        self._pool: Optional[asyncpg.Pool[asyncpg.Record]] = None
        self._collections: dict[str, PostgresVectorCollection[BaseDocument]] = {}
        self._command_timeout = command_timeout
        self._server_settings = {
            "statement_timeout": str(statement_timeout_ms),
            "lock_timeout": str(lock_timeout_ms),
        }

    async def __aenter__(self) -> Self:
        if self._external_pool is not None:
            self._pool = self._external_pool
            # Ensure pgvector extension exists (idempotent safety net).
            # Vector type registration is handled by the pool's init callback
            # (e.g. _combined_pg_init in sdk.py), which runs on every new connection.
            await self._pool.execute("CREATE EXTENSION IF NOT EXISTS vector")
        else:
            # Create the pgvector extension BEFORE creating the pool, because
            # _init_connection calls register_vector() which requires the
            # extension's types to already exist in pg_catalog.
            bootstrap_conn = await asyncpg.connect(dsn=self._dsn, command_timeout=60.0)
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
                command_timeout=self._command_timeout,
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
        """Register pgvector types on a single connection from the pool.

        Useful for pools created without a vector-aware init callback
        (e.g. in test fixtures). For production use with a shared pool
        whose init callback includes register_vector(), this is not needed.
        """
        from pgvector.asyncpg import register_vector  # type: ignore[import-untyped]

        async with pool.acquire() as conn:
            await register_vector(conn)

    def _get_pool(self) -> asyncpg.Pool[asyncpg.Record]:
        if self._pool is None:
            raise RuntimeError("Database pool not initialized. Use async with.")
        return self._pool

    _PG_MAX_IDENTIFIER = 63

    @staticmethod
    def _table_name(collection_name: str, suffix: str = "") -> str:
        result = collection_name.replace("-", "_").replace(".", "_")
        if suffix:
            result = f"{result}_{suffix}"

        if not re.match(r"^[a-zA-Z0-9_]+$", result):
            raise ValueError(f"Invalid table name: {result}")

        if len(result) > PostgresVectorDatabase._PG_MAX_IDENTIFIER:
            name_hash = hashlib.md5(result.encode()).hexdigest()[:6]
            result = f"{result[: PostgresVectorDatabase._PG_MAX_IDENTIFIER - 7]}_{name_hash}"

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

    _SYNC_BATCH_SIZE = 500

    async def _sync_embedded_with_unembedded(
        self,
        unembedded_table: str,
        embedded_table: str,
        embedder: Embedder,
    ) -> None:
        """Ensure embedded table is in sync with unembedded table (source of truth).

        Processes diffs in batches to keep memory bounded:
        1. Delete orphaned embedded docs in batches.
        2. Read diffs (new, changed, metadata-only) in batches using keyset pagination.
        3. For each batch: embed content outside transaction, write atomically.
        """
        pool = self._get_pool()

        # Phase 1: Delete orphans in batches
        while True:
            result = await pool.execute(
                f"""
                DELETE FROM "{embedded_table}" e
                WHERE e.doc_id IN (
                    SELECT e2.doc_id FROM "{embedded_table}" e2
                    LEFT JOIN "{unembedded_table}" u ON u.doc_id = e2.doc_id
                    WHERE u.doc_id IS NULL
                    LIMIT {self._SYNC_BATCH_SIZE}
                )
                """
            )
            # asyncpg execute() returns the PG command tag, e.g. "DELETE 5"
            deleted_count = int(result.split()[-1])
            if deleted_count < self._SYNC_BATCH_SIZE:
                break

        # Phase 2+3: Read diffs, embed, and write in batches
        last_id = ""
        while True:
            # Read a batch of diffs with consistent classification
            diff_rows = await pool.fetch(
                f"""
                SELECT u.doc_id, u.content, u.checksum, u.metadata,
                       CASE
                           WHEN e.doc_id IS NULL THEN 'new'
                           WHEN u.checksum != e.checksum THEN 'changed'
                           WHEN u.metadata IS DISTINCT FROM e.metadata THEN 'metadata_only'
                       END AS change_type
                FROM "{unembedded_table}" u
                LEFT JOIN "{embedded_table}" e ON u.doc_id = e.doc_id
                WHERE (e.doc_id IS NULL
                   OR u.checksum != e.checksum
                   OR (u.checksum = e.checksum
                       AND u.metadata IS DISTINCT FROM e.metadata))
                  AND u.doc_id > $1
                ORDER BY u.doc_id
                LIMIT {self._SYNC_BATCH_SIZE}
                """,
                last_id,
            )

            if not diff_rows:
                break
            last_id = diff_rows[-1]["doc_id"]

            metadata_only_rows = [r for r in diff_rows if r["change_type"] == "metadata_only"]
            changed_rows = [r for r in diff_rows if r["change_type"] == "changed"]
            new_rows = [r for r in diff_rows if r["change_type"] == "new"]
            rows_to_embed = changed_rows + new_rows

            # Embed outside any transaction to avoid holding connections during API calls
            all_vectors: Sequence[Sequence[float]] = []
            if rows_to_embed:
                contents = [row["content"] for row in rows_to_embed]
                all_vectors = (await embedder.embed(contents)).vectors

            async with pool.acquire() as conn:
                async with conn.transaction():
                    if metadata_only_rows:
                        update_args = [
                            (
                                dict(row["metadata"])
                                if isinstance(row["metadata"], dict)
                                else row["metadata"],
                                row["doc_id"],
                            )
                            for row in metadata_only_rows
                        ]
                        await conn.executemany(
                            f'UPDATE "{embedded_table}" SET metadata = $1::jsonb WHERE doc_id = $2',
                            update_args,
                        )

                    vec_idx = 0
                    if changed_rows:
                        changed_args = [
                            (
                                row["content"],
                                row["checksum"],
                                dict(row["metadata"])
                                if isinstance(row["metadata"], dict)
                                else row["metadata"],
                                HalfVector(all_vectors[vec_idx + i]),
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
                        vec_idx += len(changed_rows)

                    if new_rows:
                        insert_args = [
                            (
                                row["doc_id"],
                                row["content"],
                                row["checksum"],
                                dict(row["metadata"])
                                if isinstance(row["metadata"], dict)
                                else row["metadata"],
                                HalfVector(all_vectors[vec_idx + i]),
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
        collection_name: str,
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
                    pool,
                    collection_name,
                    unembedded_table,
                    embedded_table,
                    embedder,
                    document_loader,
                )
            finally:
                await lock_conn.execute("SELECT pg_advisory_unlock($1)", lock_id)

    _MIGRATION_BATCH_SIZE = 500

    async def _do_load_and_migrate(
        self,
        pool: asyncpg.Pool[asyncpg.Record],
        collection_name: str,
        unembedded_table: str,
        embedded_table: str,
        embedder: Embedder,
        document_loader: Callable[[BaseDocument], Awaitable[Optional[TDocument]]],
    ) -> None:
        """Inner migration logic (must be called under advisory lock).

        Processes rows in batches using keyset pagination to keep memory bounded.
        Failed documents are moved to a dedicated failed_migrations table
        instead of being silently deleted.
        """
        failed_table = self._table_name(collection_name, "failed_migrations")

        # Drop old failed_migrations table from previous runs
        if await self._table_exists(failed_table):
            self._logger.info(f"Deleting old `{failed_table}` table")
            await pool.execute(f'DROP TABLE IF EXISTS "{failed_table}"')

        failed_table_created = False

        last_id = ""
        while True:
            batch = await pool.fetch(
                f'SELECT doc_id, content, checksum, metadata FROM "{unembedded_table}" '
                f"WHERE doc_id > $1 ORDER BY doc_id LIMIT {self._MIGRATION_BATCH_SIZE}",
                last_id,
            )
            if not batch:
                break
            last_id = batch[-1]["doc_id"]

            for row in batch:
                doc = self._row_to_document(row)
                try:
                    if loaded_doc := await document_loader(doc):
                        if loaded_doc != doc:
                            metadata = {
                                k: v for k, v in loaded_doc.items() if k not in ("content",)
                            }
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
                        failed_table_created = await self._handle_failed_migration(
                            pool, failed_table, unembedded_table, doc, failed_table_created
                        )

                except Exception as e:
                    self._logger.error(
                        f"Failed to load document '{doc}' with error: {e}. "
                        f"Added to `{failed_table}` table."
                    )
                    failed_table_created = await self._handle_failed_migration(
                        pool, failed_table, unembedded_table, doc, failed_table_created
                    )

        # Now sync embedded table
        await self._sync_embedded_with_unembedded(unembedded_table, embedded_table, embedder)

    async def _handle_failed_migration(
        self,
        pool: asyncpg.Pool[asyncpg.Record],
        failed_table: str,
        source_table: str,
        doc: BaseDocument,
        failed_table_created: bool,
    ) -> bool:
        """Ensure the failed migrations table exists and move the document into it."""
        if not failed_table_created:
            self._logger.warning(f"Creating `{failed_table}` table to store failed migrations...")
            await self._create_unembedded_table(failed_table)
        await self._move_to_failed_table(pool, failed_table, source_table, doc)
        return True

    async def _move_to_failed_table(
        self,
        pool: asyncpg.Pool[asyncpg.Record],
        failed_table: str,
        source_table: str,
        doc: BaseDocument,
    ) -> None:
        """Move a document from the source table to the failed migrations table."""
        metadata = {k: v for k, v in doc.items() if k not in ("content", "checksum")}
        await pool.execute(
            f"""
            INSERT INTO "{failed_table}" (doc_id, content, checksum, metadata)
            VALUES ($1, $2, $3, $4::jsonb)
            ON CONFLICT (doc_id) DO UPDATE
            SET content = EXCLUDED.content, checksum = EXCLUDED.checksum,
                metadata = EXCLUDED.metadata
            """,
            doc["id"],
            doc.get("content", ""),
            doc.get("checksum", ""),
            metadata,
        )
        await pool.execute(f'DELETE FROM "{source_table}" WHERE doc_id = $1', doc["id"])

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
            name, unembedded_table, embedded_table, embedder_type, document_loader
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
            name, unembedded_table, embedded_table, embedder_type, document_loader
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

    async def _upsert_document(
        self,
        conn: asyncpg.Connection[asyncpg.Record],
        doc_id: str,
        content: str,
        checksum: str,
        metadata: dict[str, Any],
        embedding: Sequence[float],
    ) -> None:
        """Upsert a document into both unembedded and embedded tables."""
        await conn.execute(
            f"""
            INSERT INTO "{self._unembedded_table}" (doc_id, content, checksum, metadata)
            VALUES ($1, $2, $3, $4::jsonb)
            ON CONFLICT (doc_id) DO UPDATE
            SET content = EXCLUDED.content,
                checksum = EXCLUDED.checksum,
                metadata = EXCLUDED.metadata
            """,
            doc_id,
            content,
            checksum,
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
            doc_id,
            content,
            checksum,
            metadata,
            HalfVector(embedding),
        )

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
                    await self._upsert_document(
                        conn,
                        document["id"],
                        content,
                        str(document.get("checksum", "")),
                        metadata,
                        embedding,
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
                content_changed = "content" in params and content != doc.get("content", "")

                # Phase 2: Compute embedding OUTSIDE transaction to avoid
                # holding a connection during external API calls.
                embedding: Optional[Sequence[float]] = None
                if content_changed:
                    embedding = await self._get_embedding(content)

                # Phase 3: Write both tables atomically in a single transaction.
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

                        if content_changed:
                            assert embedding is not None
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
                        else:
                            await conn.execute(
                                f"""
                                UPDATE "{self._embedded_table}"
                                SET content = $1, checksum = $2, metadata = $3::jsonb
                                WHERE doc_id = $4
                                """,
                                content,
                                updated_document.get("checksum", ""),
                                metadata,
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

                embedding = await self._get_embedding(content)

                async with self._pool.acquire() as conn:
                    async with conn.transaction():
                        await self._upsert_document(
                            conn,
                            params["id"],
                            content,
                            str(params.get("checksum", "")),
                            metadata,
                            embedding,
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
