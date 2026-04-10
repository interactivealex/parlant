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

import asyncio
import hashlib
import json
import re
from typing import Any, Awaitable, Callable, Optional, Sequence, cast

from typing_extensions import override, Self

from parlant.core.loggers import Logger
from parlant.core.persistence.common import (
    Cursor,
    LiteralValue,
    LogicalOperator,
    ObjectId,
    SortDirection,
    Where,
    WhereExpression,
)
from parlant.core.persistence.document_database import (
    CollectionIndex,
    CollectionSort,
    BaseDocument,
    DeleteResult,
    DocumentCollection,
    DocumentDatabase,
    FindResult,
    InsertResult,
    TDocument,
    UpdateResult,
)

try:
    import asyncpg  # type: ignore[import-untyped]
except ImportError:
    asyncpg = None  # type: ignore[assignment]


class _WhereTranslator:
    """Translates MongoDB-style Where filters into PostgreSQL SQL with parameters."""

    # Fields stored as top-level columns (not inside JSONB)
    INDEXED_FIELDS = {"id", "version", "creation_utc"}

    def __init__(self) -> None:
        self._params: list[Any] = []
        self._param_idx: int = 0

    def _next_param(self, value: Any) -> str:
        self._param_idx += 1
        self._params.append(value)
        return f"${self._param_idx}"

    _VALID_FIELD_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

    def _field_ref(self, field_name: str) -> str:
        if not self._VALID_FIELD_RE.match(field_name):
            raise ValueError(f"Invalid field name: {field_name!r}")
        if field_name in self.INDEXED_FIELDS:
            return f'"{field_name}"'
        return f"data->'{field_name}'"

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
        """Create a JSONB containment parameter like '{"field": value}'.

        Used with the @> operator to leverage GIN jsonb_path_ops indexes.
        """
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
            ref = self._field_ref(field_name)
            is_indexed_field = field_name in self.INDEXED_FIELDS

            for operator, filter_value in field_filter.items():
                if operator == "$eq":
                    if is_indexed_field:
                        p = self._next_param(filter_value)
                        clauses.append(f"{ref} = {p}")
                    else:
                        # Use containment (@>) to leverage GIN jsonb_path_ops index
                        p = self._containment_param(field_name, cast(LiteralValue, filter_value))
                        clauses.append(f"data @> {p}")
                elif operator == "$ne":
                    p = (
                        self._next_param(filter_value)
                        if is_indexed_field
                        else self._typed_param(cast(LiteralValue, filter_value))
                    )
                    clauses.append(f"{ref} != {p}")
                elif operator == "$gt":
                    p = (
                        self._next_param(filter_value)
                        if is_indexed_field
                        else self._typed_param(cast(LiteralValue, filter_value))
                    )
                    clauses.append(f"{ref} > {p}")
                elif operator == "$gte":
                    p = (
                        self._next_param(filter_value)
                        if is_indexed_field
                        else self._typed_param(cast(LiteralValue, filter_value))
                    )
                    clauses.append(f"{ref} >= {p}")
                elif operator == "$lt":
                    p = (
                        self._next_param(filter_value)
                        if is_indexed_field
                        else self._typed_param(cast(LiteralValue, filter_value))
                    )
                    clauses.append(f"{ref} < {p}")
                elif operator == "$lte":
                    p = (
                        self._next_param(filter_value)
                        if is_indexed_field
                        else self._typed_param(cast(LiteralValue, filter_value))
                    )
                    clauses.append(f"{ref} <= {p}")
                elif operator == "$in":
                    values = cast(list[LiteralValue], filter_value)
                    if not values:
                        clauses.append("FALSE")
                    elif is_indexed_field:
                        placeholders = [self._next_param(v) for v in values]
                        clauses.append(f"{ref} IN ({', '.join(placeholders)})")
                    else:
                        # Use containment (@>) for each value to leverage GIN index
                        alternatives = [
                            f"data @> {self._containment_param(field_name, v)}" for v in values
                        ]
                        clauses.append(f"({' OR '.join(alternatives)})")
                elif operator == "$nin":
                    values = cast(list[LiteralValue], filter_value)
                    if not values:
                        pass  # $nin with empty list matches everything
                    else:
                        placeholders = [
                            self._next_param(v) if is_indexed_field else self._typed_param(v)
                            for v in values
                        ]
                        clauses.append(f"{ref} NOT IN ({', '.join(placeholders)})")

        return " AND ".join(clauses) if clauses else "TRUE"


class PostgresDocumentDatabase(DocumentDatabase):
    def __init__(
        self,
        dsn: str,
        logger: Logger,
        table_prefix: str = "",
        pool: Optional[asyncpg.Pool[asyncpg.Record]] = None,
        statement_timeout_ms: int = 30_000,
        lock_timeout_ms: int = 10_000,
        command_timeout: float = 30.0,
    ) -> None:
        self._dsn = dsn
        self._logger = logger
        self._table_prefix = table_prefix
        self._external_pool = pool
        self._pool: Optional[asyncpg.Pool[asyncpg.Record]] = None
        self._collections: dict[str, PostgresDocumentCollection[Any]] = {}
        self._command_timeout = command_timeout
        self._server_settings = {
            "statement_timeout": str(statement_timeout_ms),
            "lock_timeout": str(lock_timeout_ms),
        }

    @staticmethod
    async def _init_connection(conn: asyncpg.Connection[asyncpg.Record]) -> None:
        """Register JSON codec so JSONB columns are returned as Python dicts."""
        await conn.set_type_codec(
            "jsonb",
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
        )

    async def __aenter__(self) -> Self:
        if self._external_pool is not None:
            self._pool = self._external_pool
        else:
            self._pool = await asyncpg.create_pool(
                dsn=self._dsn,
                min_size=2,
                max_size=10,
                init=self._init_connection,
                server_settings=self._server_settings,
                command_timeout=self._command_timeout,
            )
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> bool:
        if self._pool is not None and self._external_pool is None:
            await self._pool.close()
            self._pool = None
        return False

    def _get_pool(self) -> asyncpg.Pool[asyncpg.Record]:
        if self._pool is None:
            raise RuntimeError("Database pool not initialized. Use async with.")
        return self._pool

    _PG_MAX_IDENTIFIER = 63

    def _table_name(self, collection_name: str) -> str:
        """Sanitize collection name for use as a table name, with optional prefix.

        PostgreSQL identifiers are limited to 63 characters.
        If the result exceeds that, we shorten the prefix and/or the name,
        appending a short hash for uniqueness.
        """
        name = collection_name.replace("-", "_").replace(".", "_")
        if self._table_prefix:
            full = f"{self._table_prefix}_{name}"
            if len(full) <= self._PG_MAX_IDENTIFIER:
                result = full
            else:
                # Both prefix and name may be long.  Truncate prefix first,
                # then name if still too long, always keeping a 6-char hash.
                prefix_hash = hashlib.md5(self._table_prefix.encode()).hexdigest()[:6]
                name_hash = hashlib.md5(name.encode()).hexdigest()[:6]

                # Budget: prefix_part + "_" + prefix_hash + "_" + name_part + "_" + name_hash
                # = prefix_part + name_part + 16 fixed chars
                available = self._PG_MAX_IDENTIFIER - 16  # 47 chars for prefix+name
                half = available // 2
                prefix_part = self._table_prefix[:half]
                name_part = name[: available - len(prefix_part)]

                result = f"{prefix_part}_{prefix_hash}_{name_part}_{name_hash}"
        else:
            if len(name) > self._PG_MAX_IDENTIFIER:
                name_hash = hashlib.md5(name.encode()).hexdigest()[:6]
                result = f"{name[: self._PG_MAX_IDENTIFIER - 7]}_{name_hash}"
            else:
                result = name

        if not re.match(r"^[a-zA-Z0-9_]+$", result):
            raise ValueError(f"Invalid table name: {result}")

        if len(result) > self._PG_MAX_IDENTIFIER:
            raise ValueError(
                f"Table name too long ({len(result)} > {self._PG_MAX_IDENTIFIER}): {result}"
            )

        return result

    async def _table_exists(self, table_name: str) -> bool:
        pool = self._get_pool()
        row = await pool.fetchrow(
            "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_class WHERE relname = $1 AND relkind = 'r')",
            table_name,
        )
        return bool(row and row["exists"])

    @override
    async def create_collection(
        self,
        name: str,
        schema: type[TDocument],
    ) -> DocumentCollection[TDocument]:
        pool = self._get_pool()
        table = self._table_name(name)

        await pool.execute(f"""
            CREATE TABLE IF NOT EXISTS "{table}" (
                "id" TEXT PRIMARY KEY,
                "version" TEXT,
                "creation_utc" TEXT,
                data JSONB NOT NULL DEFAULT '{{}}'::jsonb
            )
        """)

        # Index on creation_utc for pagination
        await pool.execute(
            f'CREATE INDEX IF NOT EXISTS "idx_{table}_creation_utc" ON "{table}" ("creation_utc")'
        )

        # GIN index on JSONB data for efficient $eq containment queries
        await pool.execute(
            f'CREATE INDEX IF NOT EXISTS "idx_{table}_data_pathops" '
            f'ON "{table}" USING GIN (data jsonb_path_ops)'
        )

        collection: PostgresDocumentCollection[TDocument] = PostgresDocumentCollection(
            pool=pool,
            table_name=table,
            logger=self._logger,
        )
        self._collections[name] = collection
        return collection

    @staticmethod
    def _advisory_lock_id(table: str) -> int:
        """Derive a stable advisory-lock ID from a table name.

        PostgreSQL advisory locks use a bigint key.  We hash the table name so
        that each collection gets its own lock without requiring a shared
        registry.
        """
        # Use first 8 bytes of the hash → signed 64-bit int
        digest = hashlib.md5(table.encode()).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=True)

    @override
    async def get_collection(
        self,
        name: str,
        schema: type[TDocument],
        document_loader: Callable[[BaseDocument], Awaitable[TDocument | None]],
    ) -> DocumentCollection[TDocument]:
        pool = self._get_pool()
        table = self._table_name(name)

        if not await self._table_exists(table):
            raise ValueError(f'Collection "{name}" does not exist.')

        # Acquire advisory lock so concurrent workers don't duplicate migration work
        lock_id = self._advisory_lock_id(table)
        async with pool.acquire() as lock_conn:
            await lock_conn.execute("SELECT pg_advisory_lock($1)", lock_id)
            try:
                await self._run_migration(pool, table, name, document_loader)
            finally:
                await lock_conn.execute("SELECT pg_advisory_unlock($1)", lock_id)

        collection: PostgresDocumentCollection[TDocument] = PostgresDocumentCollection(
            pool=pool,
            table_name=table,
            logger=self._logger,
        )
        self._collections[name] = collection
        return collection

    _MIGRATION_BATCH_SIZE = 500

    async def _run_migration(
        self,
        pool: asyncpg.Pool[asyncpg.Record],
        table: str,
        name: str,
        document_loader: Callable[[BaseDocument], Awaitable[TDocument | None]],
    ) -> None:
        """Run document migration/loading (must be called under advisory lock).

        Streams rows in batches via a server-side cursor to avoid loading
        the entire collection into memory.
        """
        failed_table = self._table_name(f"{name}_failed_migrations")

        # Check if failed migrations table exists from a previous run and drop it
        if await self._table_exists(failed_table):
            self._logger.info(f"Deleting old `{failed_table}` table")
            await pool.execute(f'DROP TABLE IF EXISTS "{failed_table}"')

        failed_collection: Optional[DocumentCollection[TDocument]] = None

        # Process documents in batches using keyset pagination to keep memory bounded.
        last_id = ""
        while True:
            batch = await pool.fetch(
                f'SELECT "id", "version", "creation_utc", data FROM "{table}" '
                f'WHERE "id" > $1 ORDER BY "id" LIMIT {self._MIGRATION_BATCH_SIZE}',
                last_id,
            )
            if not batch:
                break
            last_id = batch[-1]["id"]

            for row in batch:
                doc = self._row_to_document(row)
                try:
                    if loaded_doc := await document_loader(doc):
                        await self._replace_document(pool, table, doc["id"], loaded_doc)
                        continue

                    if failed_collection is None:
                        self._logger.warning(
                            f"Creating `{failed_table}` table to store failed migrations..."
                        )
                        failed_collection = await self._ensure_failed_migrations_table(
                            pool, failed_table
                        )

                    self._logger.warning(f'Failed to load document "{doc}"')
                    await failed_collection.insert_one(cast(TDocument, doc))
                    await pool.execute(f'DELETE FROM "{table}" WHERE "id" = $1', doc["id"])

                except Exception as e:
                    if failed_collection is None:
                        self._logger.warning(
                            f"Creating `{failed_table}` table to store failed migrations..."
                        )
                        failed_collection = await self._ensure_failed_migrations_table(
                            pool, failed_table
                        )

                    self._logger.error(
                        f"Failed to load document '{doc}' with error: {e}. "
                        f"Added to `{failed_table}` table."
                    )
                    await failed_collection.insert_one(cast(TDocument, doc))
                    await pool.execute(f'DELETE FROM "{table}" WHERE "id" = $1', doc["id"])

    @override
    async def get_or_create_collection(
        self,
        name: str,
        schema: type[TDocument],
        document_loader: Callable[[BaseDocument], Awaitable[TDocument | None]],
    ) -> DocumentCollection[TDocument]:
        table = self._table_name(name)

        if not await self._table_exists(table):
            return await self.create_collection(name, schema)

        return await self.get_collection(name, schema, document_loader)

    @override
    async def delete_collection(self, name: str) -> None:
        pool = self._get_pool()
        table = self._table_name(name)

        if not await self._table_exists(table):
            raise ValueError(f'Collection "{name}" does not exist.')

        await pool.execute(f'DROP TABLE IF EXISTS "{table}"')
        self._collections.pop(name, None)

    async def _ensure_failed_migrations_table(
        self,
        pool: asyncpg.Pool[asyncpg.Record],
        table_name: str,
    ) -> PostgresDocumentCollection[Any]:
        await pool.execute(f"""
            CREATE TABLE IF NOT EXISTS "{table_name}" (
                "id" TEXT PRIMARY KEY,
                "version" TEXT,
                "creation_utc" TEXT,
                data JSONB NOT NULL DEFAULT '{{}}'::jsonb
            )
        """)
        return PostgresDocumentCollection(pool=pool, table_name=table_name, logger=self._logger)

    async def _replace_document(
        self,
        pool: asyncpg.Pool[asyncpg.Record],
        table: str,
        doc_id: str,
        new_doc: BaseDocument,
    ) -> None:
        data = {k: v for k, v in new_doc.items() if k not in ("id", "version", "creation_utc")}
        await pool.execute(
            f"""
            UPDATE "{table}"
            SET "version" = $1, "creation_utc" = $2, data = $3::jsonb
            WHERE "id" = $4
            """,
            new_doc.get("version", ""),
            new_doc.get("creation_utc", ""),
            data,
            doc_id,
        )

    @staticmethod
    def _row_to_document(row: asyncpg.Record) -> BaseDocument:
        """Reconstruct a BaseDocument from a PostgreSQL row."""
        raw_data = row["data"]
        if isinstance(raw_data, str):
            data: dict[str, Any] = json.loads(raw_data) if raw_data else {}
        elif isinstance(raw_data, dict):
            data = dict(raw_data)
        else:
            data = {}
        doc: dict[str, Any] = {
            "id": row["id"],
            **data,
        }
        if row["version"] is not None:
            doc["version"] = row["version"]
        if row["creation_utc"] is not None:
            doc["creation_utc"] = row["creation_utc"]
        return cast(BaseDocument, doc)


class PostgresDocumentCollection(DocumentCollection[TDocument]):
    def __init__(
        self,
        pool: asyncpg.Pool[asyncpg.Record],
        table_name: str,
        logger: Logger,
    ) -> None:
        self._pool = pool
        self._table = table_name
        self._logger = logger

    @override
    async def find(
        self,
        filters: Where,
        limit: Optional[int] = None,
        cursor: Optional[Cursor] = None,
        sort_direction: Optional[SortDirection] = None,
    ) -> FindResult[TDocument]:
        sort_direction = sort_direction or SortDirection.ASC

        translator = _WhereTranslator()
        where_clause, params = translator.translate(filters)

        if cursor is not None:
            # Continue parameter numbering from the WHERE translation
            cursor_translator = _WhereTranslator()
            cursor_translator._param_idx = translator._param_idx
            cursor_translator._params = list(params)

            if sort_direction == SortDirection.DESC:
                cursor_cond = (
                    f'("creation_utc" < {cursor_translator._next_param(cursor.creation_utc)}'
                    f" OR ("
                    f'"creation_utc" = {cursor_translator._next_param(cursor.creation_utc)}'
                    f' AND "id" < {cursor_translator._next_param(cursor.id)}'
                    f"))"
                )
            else:
                cursor_cond = (
                    f'("creation_utc" > {cursor_translator._next_param(cursor.creation_utc)}'
                    f" OR ("
                    f'"creation_utc" = {cursor_translator._next_param(cursor.creation_utc)}'
                    f' AND "id" > {cursor_translator._next_param(cursor.id)}'
                    f"))"
                )

            params = cursor_translator._params
            if where_clause:
                where_clause = f"({where_clause}) AND {cursor_cond}"
            else:
                where_clause = cursor_cond

        sort_order = "DESC" if sort_direction == SortDirection.DESC else "ASC"
        order_by = f'"creation_utc" {sort_order}, "id" {sort_order}'

        # Query one extra to detect has_more
        query_limit = (limit + 1) if limit is not None else None

        # Fetch matching rows
        sql = f'SELECT "id", "version", "creation_utc", data FROM "{self._table}"'
        if where_clause:
            sql += f" WHERE {where_clause}"
        sql += f" ORDER BY {order_by}"
        if query_limit is not None:
            sql += f" LIMIT {query_limit}"

        # Separate COUNT query for correct total (window functions see only LIMITed rows)
        count_sql = f'SELECT COUNT(*) AS cnt FROM "{self._table}"'
        if where_clause:
            count_sql += f" WHERE {where_clause}"

        rows, count_row = await asyncio.gather(
            self._pool.fetch(sql, *params),
            self._pool.fetchrow(count_sql, *params),
        )

        total_count = int(count_row["cnt"]) if count_row else 0
        items = [PostgresDocumentDatabase._row_to_document(r) for r in rows]

        has_more = False
        next_cursor = None

        if limit is not None and len(items) > limit:
            has_more = True
            items = items[:limit]

            if items:
                last_item = items[-1]
                next_cursor = Cursor(
                    creation_utc=str(last_item.get("creation_utc", "")),
                    id=ObjectId(str(last_item.get("id", ""))),
                )

        return FindResult(
            items=cast(Sequence[TDocument], items),
            total_count=total_count,
            has_more=has_more,
            next_cursor=next_cursor,
        )

    @override
    async def find_one(
        self,
        filters: Where,
        sort: Optional[CollectionSort] = None,
    ) -> Optional[TDocument]:
        translator = _WhereTranslator()
        where_clause, params = translator.translate(filters)

        sql = f'SELECT "id", "version", "creation_utc", data FROM "{self._table}"'
        if where_clause:
            sql += f" WHERE {where_clause}"

        if sort:
            order_parts = []
            for field_name, direction in sort:
                if not _WhereTranslator._VALID_FIELD_RE.match(field_name):
                    raise ValueError(f"Invalid field name: {field_name!r}")
                order = "DESC" if direction == SortDirection.DESC else "ASC"
                if field_name in _WhereTranslator.INDEXED_FIELDS:
                    order_parts.append(f'"{field_name}" {order}')
                else:
                    order_parts.append(f"(data->'{field_name}') {order}")
            sql += f" ORDER BY {', '.join(order_parts)}"

        sql += " LIMIT 1"

        row = await self._pool.fetchrow(sql, *params)
        if row is None:
            return None

        return cast(TDocument, PostgresDocumentDatabase._row_to_document(row))

    @override
    async def ensure_indexes(
        self,
        indexes: Sequence[CollectionIndex],
    ) -> None:
        for idx, index in enumerate(indexes):
            cols = []
            for field_name, direction in index.fields:
                if not _WhereTranslator._VALID_FIELD_RE.match(field_name):
                    raise ValueError(f"Invalid field name: {field_name!r}")
                order = "DESC" if direction == SortDirection.DESC else "ASC"
                if field_name in _WhereTranslator.INDEXED_FIELDS:
                    cols.append(f'"{field_name}" {order}')
                else:
                    cols.append(f"(data->'{field_name}') {order}")

            unique = "UNIQUE" if index.unique else ""
            col_str = ", ".join(cols)
            index_signature = hashlib.sha1(f"{index.unique}:{col_str}".encode("utf-8")).hexdigest()[
                :12
            ]
            idx_name = f"idx_{self._table[:32]}_{idx}_{index_signature}"
            await self._pool.execute(
                f'CREATE {unique} INDEX IF NOT EXISTS "{idx_name}" ON "{self._table}" ({col_str})'
            )

    async def _insert_one_with_conn(
        self,
        conn: asyncpg.Connection[asyncpg.Record],
        document: TDocument,
    ) -> None:
        doc = dict(document)
        doc_id = doc.pop("id", "")
        version = doc.pop("version", None)
        creation_utc = doc.pop("creation_utc", None)

        await conn.execute(
            f"""
            INSERT INTO "{self._table}" ("id", "version", "creation_utc", data)
            VALUES ($1, $2, $3, $4::jsonb)
            ON CONFLICT ("id") DO UPDATE
            SET "version" = EXCLUDED."version",
                "creation_utc" = EXCLUDED."creation_utc",
                data = EXCLUDED.data
            """,
            doc_id,
            version,
            creation_utc,
            doc,
        )

    @override
    async def insert_one(self, document: TDocument) -> InsertResult:
        async with self._pool.acquire() as conn:
            await self._insert_one_with_conn(conn, document)

        return InsertResult(acknowledged=True)

    @override
    async def update_one(
        self,
        filters: Where,
        params: TDocument,
        upsert: bool = False,
    ) -> UpdateResult[TDocument]:
        translator = _WhereTranslator()
        where_clause, sql_params = translator.translate(filters)

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Lock the matching row
                sql = f'SELECT "id", "version", "creation_utc", data FROM "{self._table}"'
                if where_clause:
                    sql += f" WHERE {where_clause}"
                sql += " FOR UPDATE LIMIT 1"

                row = await conn.fetchrow(sql, *sql_params)

                if row is not None:
                    existing = PostgresDocumentDatabase._row_to_document(row)
                    merged = {**existing, **params}
                    doc_id = merged["id"]
                    data = {
                        k: v
                        for k, v in merged.items()
                        if k not in ("id", "version", "creation_utc")
                    }

                    updated_row = await conn.fetchrow(
                        f"""
                        UPDATE "{self._table}"
                        SET "version" = $1, "creation_utc" = $2, data = $3::jsonb
                        WHERE "id" = $4
                        RETURNING "id", "version", "creation_utc", data
                        """,
                        merged.get("version"),
                        merged.get("creation_utc"),
                        data,
                        doc_id,
                    )

                    assert updated_row is not None
                    updated_doc = PostgresDocumentDatabase._row_to_document(updated_row)
                    return UpdateResult(
                        acknowledged=True,
                        matched_count=1,
                        modified_count=1,
                        updated_document=cast(TDocument, updated_doc),
                    )

                if upsert:
                    await self._insert_one_with_conn(conn, params)
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
    async def delete_one(self, filters: Where) -> DeleteResult[TDocument]:
        translator = _WhereTranslator()
        where_clause, sql_params = translator.translate(filters)

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Lock the matching row
                sql = f'SELECT "id" FROM "{self._table}"'
                if where_clause:
                    sql += f" WHERE {where_clause}"
                sql += " FOR UPDATE LIMIT 1"

                row = await conn.fetchrow(sql, *sql_params)

                if row is None:
                    return DeleteResult(acknowledged=True, deleted_count=0, deleted_document=None)

                doc_id = row["id"]
                deleted_row = await conn.fetchrow(
                    f"""
                    DELETE FROM "{self._table}" WHERE "id" = $1
                    RETURNING "id", "version", "creation_utc", data
                    """,
                    doc_id,
                )

                assert deleted_row is not None
                deleted_doc = PostgresDocumentDatabase._row_to_document(deleted_row)

        return DeleteResult(
            acknowledged=True,
            deleted_count=1,
            deleted_document=cast(TDocument, deleted_doc),
        )
