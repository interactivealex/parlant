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

import os
from contextlib import AsyncExitStack
from collections.abc import Mapping
from typing import Any, AsyncIterator, Optional, TypedDict, cast

import pytest
from lagom import Container
from pytest import fixture
from typing_extensions import Required, override

from parlant.adapters.db.postgres_db import PostgresDocumentDatabase
from parlant.adapters.vector_db.pgvector import PostgresVectorDatabase
from parlant.core.common import IdGenerator, Version, md5_checksum
from parlant.core.capabilities import CapabilityVectorStore
from parlant.core.nlp.embedding import (
    Embedder,
    EmbedderFactory,
    EmbeddingResult,
    NullEmbedder,
    NullEmbeddingCache,
)
from parlant.core.nlp.tokenization import EstimatingTokenizer, ZeroEstimatingTokenizer
from parlant.core.loggers import Logger
from parlant.core.persistence.common import ObjectId
from parlant.core.persistence.document_database import BaseDocument as DocBaseDocument
from parlant.core.persistence.vector_database import BaseDocument
from parlant.core.tracer import Tracer

try:
    import asyncpg  # type: ignore[import-untyped]
except ImportError:
    asyncpg = None  # type: ignore[assignment]


async def _null_embedder_type_provider() -> type[Embedder]:
    return NullEmbedder


class _TestDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    content: str
    checksum: Required[str]
    name: str
    rank: int
    active: bool


@fixture
def postgres_dsn() -> str:
    dsn = os.environ.get("TEST_POSTGRES_DSN")
    if dsn:
        return dsn
    print("Could not find `TEST_POSTGRES_DSN` in environment, skipping pgvector tests...")
    raise pytest.skip()


@fixture
async def pgvector_db(
    container: Container,
    postgres_dsn: str,
) -> AsyncIterator[object]:
    from parlant.adapters.vector_db.pgvector import PostgresVectorDatabase

    db = PostgresVectorDatabase(
        dsn=postgres_dsn,
        logger=container[Logger],
        tracer=container[Tracer],
        embedder_factory=EmbedderFactory(container),
        embedding_cache_provider=NullEmbeddingCache,
    )

    async with db as opened_db:
        # Clean up test tables
        pool = opened_db._get_pool()
        tables = await pool.fetch("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        for table in tables:
            await pool.execute(f'DROP TABLE IF EXISTS "{table["tablename"]}" CASCADE')
        # Re-create metadata table and pgvector extension
        await pool.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await pool.execute("""
            CREATE TABLE IF NOT EXISTS _vector_metadata (
                key TEXT PRIMARY KEY,
                value JSONB NOT NULL
            )
        """)
        yield opened_db


def _make_doc(
    doc_id: str,
    content: str,
    name: str = "test",
    rank: int = 0,
    active: bool = False,
) -> _TestDocument:
    doc = _TestDocument(
        id=ObjectId(doc_id),
        version=Version.String("0.1.0"),
        content=content,
        checksum=md5_checksum(content),
        name=name,
        rank=rank,
        active=active,
    )
    return doc


async def test_that_collection_can_be_created(
    container: Container,
    pgvector_db: object,
) -> None:
    from parlant.adapters.vector_db.pgvector import PostgresVectorDatabase

    db = pgvector_db
    assert isinstance(db, PostgresVectorDatabase)

    collection = await db.create_collection(
        name="test_col",
        schema=_TestDocument,
        embedder_type=NullEmbedder,
    )
    assert collection is not None


async def test_that_document_can_be_inserted_and_found(
    container: Container,
    pgvector_db: object,
) -> None:
    from parlant.adapters.vector_db.pgvector import PostgresVectorDatabase

    db = pgvector_db
    assert isinstance(db, PostgresVectorDatabase)

    collection = await db.create_collection(
        name="insert_test",
        schema=_TestDocument,
        embedder_type=NullEmbedder,
    )

    doc = _make_doc("doc1", "Hello world", "greeting")
    await collection.insert_one(doc)

    results = await collection.find({})
    assert len(results) == 1
    assert results[0]["id"] == "doc1"
    assert results[0]["content"] == "Hello world"
    assert results[0]["name"] == "greeting"


async def test_that_find_one_works(
    container: Container,
    pgvector_db: object,
) -> None:
    from parlant.adapters.vector_db.pgvector import PostgresVectorDatabase

    db = pgvector_db
    assert isinstance(db, PostgresVectorDatabase)

    collection = await db.create_collection(
        name="findone_test",
        schema=_TestDocument,
        embedder_type=NullEmbedder,
    )

    doc = _make_doc("doc1", "Hello world", "greeting")
    await collection.insert_one(doc)

    result = await collection.find_one({"id": {"$eq": "doc1"}})
    assert result is not None
    assert result["content"] == "Hello world"

    no_result = await collection.find_one({"id": {"$eq": "nonexistent"}})
    assert no_result is None


async def test_that_document_can_be_updated(
    container: Container,
    pgvector_db: object,
) -> None:
    from parlant.adapters.vector_db.pgvector import PostgresVectorDatabase

    db = pgvector_db
    assert isinstance(db, PostgresVectorDatabase)

    collection = await db.create_collection(
        name="update_test",
        schema=_TestDocument,
        embedder_type=NullEmbedder,
    )

    doc = _make_doc("doc1", "original content", "original")
    await collection.insert_one(doc)

    updated = _TestDocument(
        id=ObjectId("doc1"),
        version=Version.String("0.1.0"),
        content="updated content",
        checksum=md5_checksum("updated content"),
        name="updated",
    )

    result = await collection.update_one({"id": {"$eq": "doc1"}}, updated)
    assert result.matched_count == 1
    assert result.updated_document is not None
    assert result.updated_document["name"] == "updated"

    found = await collection.find_one({"id": {"$eq": "doc1"}})
    assert found is not None
    assert found["content"] == "updated content"


async def test_that_document_can_be_deleted(
    container: Container,
    pgvector_db: object,
) -> None:
    from parlant.adapters.vector_db.pgvector import PostgresVectorDatabase

    db = pgvector_db
    assert isinstance(db, PostgresVectorDatabase)

    collection = await db.create_collection(
        name="delete_test",
        schema=_TestDocument,
        embedder_type=NullEmbedder,
    )

    doc = _make_doc("doc1", "to be deleted")
    await collection.insert_one(doc)

    result = await collection.delete_one({"id": {"$eq": "doc1"}})
    assert result.deleted_count == 1
    assert result.deleted_document is not None

    found = await collection.find_one({"id": {"$eq": "doc1"}})
    assert found is None


async def test_that_delete_nonexistent_returns_zero(
    container: Container,
    pgvector_db: object,
) -> None:
    from parlant.adapters.vector_db.pgvector import PostgresVectorDatabase

    db = pgvector_db
    assert isinstance(db, PostgresVectorDatabase)

    collection = await db.create_collection(
        name="delete_none_test",
        schema=_TestDocument,
        embedder_type=NullEmbedder,
    )

    result = await collection.delete_one({"id": {"$eq": "nonexistent"}})
    assert result.deleted_count == 0
    assert result.deleted_document is None


async def test_that_upsert_inserts_when_not_found(
    container: Container,
    pgvector_db: object,
) -> None:
    from parlant.adapters.vector_db.pgvector import PostgresVectorDatabase

    db = pgvector_db
    assert isinstance(db, PostgresVectorDatabase)

    collection = await db.create_collection(
        name="upsert_test",
        schema=_TestDocument,
        embedder_type=NullEmbedder,
    )

    doc = _make_doc("upsert_doc", "upserted content", "upserted")
    result = await collection.update_one({"id": {"$eq": "upsert_doc"}}, doc, upsert=True)

    assert result.updated_document is not None
    assert result.matched_count == 0

    found = await collection.find_one({"id": {"$eq": "upsert_doc"}})
    assert found is not None
    assert found["content"] == "upserted content"


async def test_that_find_with_filters_works(
    container: Container,
    pgvector_db: object,
) -> None:
    db = pgvector_db
    assert isinstance(db, PostgresVectorDatabase)

    collection = await db.create_collection(
        name="filter_test",
        schema=_TestDocument,
        embedder_type=NullEmbedder,
    )

    await collection.insert_one(_make_doc("doc1", "alpha content", "alpha"))
    await collection.insert_one(_make_doc("doc2", "beta content", "beta"))

    results = await collection.find({"name": {"$eq": "alpha"}})
    assert len(results) == 1
    assert results[0]["name"] == "alpha"


async def test_that_numeric_metadata_filters_use_jsonb_numeric_semantics(
    container: Container,
    pgvector_db: object,
) -> None:
    db = pgvector_db
    assert isinstance(db, PostgresVectorDatabase)

    collection = await db.create_collection(
        name="numeric_filter_test",
        schema=_TestDocument,
        embedder_type=NullEmbedder,
    )

    await collection.insert_one(_make_doc("doc2", "two", rank=2))
    await collection.insert_one(_make_doc("doc10", "ten", rank=10))
    await collection.insert_one(_make_doc("doc11", "eleven", rank=11))

    gt_results = await collection.find({"rank": {"$gt": 9}})
    in_results = await collection.find({"rank": {"$in": [2, 11]}})
    nin_results = await collection.find({"rank": {"$nin": [10]}})

    assert {doc["rank"] for doc in gt_results} == {10, 11}
    assert {doc["rank"] for doc in in_results} == {2, 11}
    assert {doc["rank"] for doc in nin_results} == {2, 11}


async def test_that_boolean_metadata_filters_are_not_coerced_to_text(
    container: Container,
    pgvector_db: object,
) -> None:
    db = pgvector_db
    assert isinstance(db, PostgresVectorDatabase)

    collection = await db.create_collection(
        name="boolean_filter_test",
        schema=_TestDocument,
        embedder_type=NullEmbedder,
    )

    await collection.insert_one(_make_doc("doc_true", "true", active=True))
    await collection.insert_one(_make_doc("doc_false", "false", active=False))

    results = await collection.find({"active": {"$eq": True}})

    assert len(results) == 1
    assert results[0]["id"] == "doc_true"


async def test_that_metadata_can_be_stored_and_retrieved(
    container: Container,
    pgvector_db: object,
) -> None:
    from parlant.adapters.vector_db.pgvector import PostgresVectorDatabase

    db = pgvector_db
    assert isinstance(db, PostgresVectorDatabase)

    await db.upsert_metadata("test_key", "test_value")
    metadata = await db.read_metadata()
    assert metadata["test_key"] == "test_value"

    await db.upsert_metadata("test_key", "updated_value")
    metadata = await db.read_metadata()
    assert metadata["test_key"] == "updated_value"


async def test_that_metadata_can_be_removed(
    container: Container,
    pgvector_db: object,
) -> None:
    from parlant.adapters.vector_db.pgvector import PostgresVectorDatabase

    db = pgvector_db
    assert isinstance(db, PostgresVectorDatabase)

    await db.upsert_metadata("removable", "value")
    await db.remove_metadata("removable")

    metadata = await db.read_metadata()
    assert "removable" not in metadata


async def test_that_collection_can_be_deleted(
    container: Container,
    pgvector_db: object,
) -> None:
    from parlant.adapters.vector_db.pgvector import PostgresVectorDatabase

    db = pgvector_db
    assert isinstance(db, PostgresVectorDatabase)

    await db.create_collection(
        name="deletable",
        schema=_TestDocument,
        embedder_type=NullEmbedder,
    )

    await db.delete_collection("deletable")

    with pytest.raises(ValueError, match="not found"):
        await db.delete_collection("deletable")


async def test_that_get_or_create_collection_is_idempotent(
    container: Container,
    pgvector_db: object,
) -> None:
    from parlant.adapters.vector_db.pgvector import PostgresVectorDatabase

    db = pgvector_db
    assert isinstance(db, PostgresVectorDatabase)

    async def identity(doc: BaseDocument) -> Optional[_TestDocument]:
        return cast(_TestDocument, doc)

    col1 = await db.get_or_create_collection(
        name="idempotent_test",
        schema=_TestDocument,
        embedder_type=NullEmbedder,
        document_loader=identity,
    )

    await col1.insert_one(_make_doc("doc1", "test content"))

    col2 = await db.get_or_create_collection(
        name="idempotent_test",
        schema=_TestDocument,
        embedder_type=NullEmbedder,
        document_loader=identity,
    )

    results = await col2.find({})
    assert len(results) == 1


async def test_that_similarity_search_returns_empty_for_zero_vectors(
    container: Container,
    pgvector_db: object,
) -> None:
    """NullEmbedder produces zero vectors. pgvector's cosine distance
    of zero vectors is NaN, so no results are returned. This is expected."""
    from parlant.adapters.vector_db.pgvector import PostgresVectorDatabase

    db = pgvector_db
    assert isinstance(db, PostgresVectorDatabase)

    collection = await db.create_collection(
        name="similarity_test",
        schema=_TestDocument,
        embedder_type=NullEmbedder,
    )

    await collection.insert_one(_make_doc("doc1", "hello world"))
    await collection.insert_one(_make_doc("doc2", "goodbye world"))

    # Zero vectors produce NaN cosine distance, pgvector returns empty results
    results = await collection.find_similar_documents(
        filters={},
        query="hello",
        k=2,
    )

    assert len(results) == 0


async def test_that_capability_documents_persist_across_postgres_restarts(
    container: Container,
    pgvector_db: object,
    postgres_dsn: str,
) -> None:
    del pgvector_db

    created_capability_id = None

    async with AsyncExitStack() as stack:
        vector_db = await stack.enter_async_context(
            PostgresVectorDatabase(
                dsn=postgres_dsn,
                logger=container[Logger],
                tracer=container[Tracer],
                embedder_factory=EmbedderFactory(container),
                embedding_cache_provider=NullEmbeddingCache,
            )
        )
        document_db = await stack.enter_async_context(
            PostgresDocumentDatabase(
                dsn=postgres_dsn,
                logger=container[Logger],
                table_prefix="capabilities",
            )
        )
        store = await stack.enter_async_context(
            CapabilityVectorStore(
                container[IdGenerator],
                vector_db=vector_db,
                document_db=document_db,
                embedder_factory=EmbedderFactory(container),
                embedder_type_provider=_null_embedder_type_provider,
            )
        )

        capability = await store.create_capability(
            title="Persisted capability",
            description="Survives restart",
            signals=["restart"],
        )
        created_capability_id = capability.id

        assert len(await store.list_capabilities()) == 1

    async with AsyncExitStack() as stack:
        vector_db = await stack.enter_async_context(
            PostgresVectorDatabase(
                dsn=postgres_dsn,
                logger=container[Logger],
                tracer=container[Tracer],
                embedder_factory=EmbedderFactory(container),
                embedding_cache_provider=NullEmbeddingCache,
            )
        )
        document_db = await stack.enter_async_context(
            PostgresDocumentDatabase(
                dsn=postgres_dsn,
                logger=container[Logger],
                table_prefix="capabilities",
            )
        )
        store = await stack.enter_async_context(
            CapabilityVectorStore(
                container[IdGenerator],
                vector_db=vector_db,
                document_db=document_db,
                embedder_factory=EmbedderFactory(container),
                embedder_type_provider=_null_embedder_type_provider,
            )
        )

        capabilities = await store.list_capabilities()

    assert len(capabilities) == 1
    assert capabilities[0].id == created_capability_id


async def test_that_pgvector_invalid_collection_names_raise_value_error(
    container: Container,
    pgvector_db: object,
) -> None:
    db = pgvector_db
    assert isinstance(db, PostgresVectorDatabase)

    with pytest.raises(ValueError, match="Invalid table name"):
        await db.create_collection(
            name="has spaces",
            schema=_TestDocument,
            embedder_type=NullEmbedder,
        )

    with pytest.raises(ValueError, match="Invalid table name"):
        await db.create_collection(
            name="semi;colon",
            schema=_TestDocument,
            embedder_type=NullEmbedder,
        )


async def test_that_insert_rolls_back_unembedded_row_when_embedded_insert_fails(
    container: Container,
    pgvector_db: object,
) -> None:
    db = pgvector_db
    assert isinstance(db, PostgresVectorDatabase)

    collection = await db.create_collection(
        name="rollback_insert_test",
        schema=_TestDocument,
        embedder_type=NullEmbedder,
    )

    doc = _make_doc("rollback_doc", "test content")

    # Corrupt the embedded table name to force a failure on the second INSERT
    original_embedded = collection._embedded_table
    collection._embedded_table = "nonexistent_table"

    with pytest.raises(Exception):
        await collection.insert_one(doc)

    collection._embedded_table = original_embedded

    # The unembedded row should have been rolled back
    pool = db._get_pool()
    row = await pool.fetchrow(
        f'SELECT doc_id FROM "{collection._unembedded_table}" WHERE doc_id = $1',
        "rollback_doc",
    )
    assert row is None


async def test_that_delete_rolls_back_unembedded_delete_when_embedded_delete_fails(
    container: Container,
    pgvector_db: object,
) -> None:
    db = pgvector_db
    assert isinstance(db, PostgresVectorDatabase)

    collection = await db.create_collection(
        name="rollback_delete_test",
        schema=_TestDocument,
        embedder_type=NullEmbedder,
    )

    doc = _make_doc("rb_del_doc", "delete test content")
    await collection.insert_one(doc)

    # Corrupt the embedded table name to force a failure on the second DELETE
    original_embedded = collection._embedded_table
    collection._embedded_table = "nonexistent_table"

    with pytest.raises(Exception):
        await collection.delete_one({"id": {"$eq": "rb_del_doc"}})

    collection._embedded_table = original_embedded

    # The unembedded row should still exist (rolled back)
    pool = db._get_pool()
    row = await pool.fetchrow(
        f'SELECT doc_id FROM "{collection._unembedded_table}" WHERE doc_id = $1',
        "rb_del_doc",
    )
    assert row is not None


@fixture
async def shared_pool(
    postgres_dsn: str,
) -> AsyncIterator[Any]:
    pool = await asyncpg.create_pool(
        dsn=postgres_dsn,
        min_size=2,
        max_size=10,
        init=PostgresVectorDatabase._init_connection,
    )
    try:
        yield pool
    finally:
        await pool.close()


async def test_that_vector_db_and_document_db_can_share_a_pool(
    container: Container,
    shared_pool: Any,
) -> None:
    async with (
        PostgresVectorDatabase(
            dsn="unused",
            logger=container[Logger],
            tracer=container[Tracer],
            embedder_factory=EmbedderFactory(container),
            embedding_cache_provider=NullEmbeddingCache,
            pool=shared_pool,
        ) as vector_db,
        PostgresDocumentDatabase(
            dsn="unused",
            logger=container[Logger],
            table_prefix="shared_test",
            pool=shared_pool,
        ) as doc_db,
    ):
        # Both should be able to use the pool
        vcol = await vector_db.create_collection(
            name="shared_vec",
            schema=_TestDocument,
            embedder_type=NullEmbedder,
        )
        dcol = await doc_db.create_collection(
            name="shared_doc",
            schema=DocBaseDocument,
        )

        await vcol.insert_one(_make_doc("vdoc1", "vector content"))
        await dcol.insert_one(
            DocBaseDocument(
                id=ObjectId("ddoc1"),
                version=Version.String("0.1.0"),
                creation_utc="2024-01-01T00:00:00Z",
            )
        )

        vresults = await vcol.find({})
        dresults = await dcol.find({})
        assert len(vresults) == 1
        assert dresults.total_count == 1

    # Pool should still be usable
    row = await shared_pool.fetchval("SELECT 1")
    assert row == 1


async def test_that_sync_handles_new_documents_from_unembedded_table(
    container: Container,
    pgvector_db: object,
    postgres_dsn: str,
) -> None:
    """Inserting a doc into unembedded only, then syncing, should populate embedded."""
    db = pgvector_db
    assert isinstance(db, PostgresVectorDatabase)

    collection = await db.create_collection(
        name="sync_new_test",
        schema=_TestDocument,
        embedder_type=NullEmbedder,
    )

    pool = db._get_pool()

    # Directly insert into unembedded table (simulating out-of-band insert)
    await pool.execute(
        f"""
        INSERT INTO "{collection._unembedded_table}" (doc_id, content, checksum, metadata)
        VALUES ($1, $2, $3, $4::jsonb)
        """,
        "sync_doc",
        "synced content",
        md5_checksum("synced content"),
        {"id": "sync_doc", "name": "synced"},
    )

    embedder = NullEmbedder()
    await db._sync_embedded_with_unembedded(
        collection._unembedded_table,
        collection._embedded_table,
        embedder,
    )

    # Should now be findable via the collection
    found = await collection.find_one({"id": {"$eq": "sync_doc"}})
    assert found is not None
    assert found["content"] == "synced content"


async def test_that_sync_removes_embedded_documents_not_in_unembedded(
    container: Container,
    pgvector_db: object,
) -> None:
    """If an embedded doc has no matching unembedded doc, sync should remove it."""
    db = pgvector_db
    assert isinstance(db, PostgresVectorDatabase)

    collection = await db.create_collection(
        name="sync_orphan_test",
        schema=_TestDocument,
        embedder_type=NullEmbedder,
    )

    doc = _make_doc("orphan_doc", "orphan content")
    await collection.insert_one(doc)

    pool = db._get_pool()

    # Remove from unembedded only, leaving an orphan in embedded
    await pool.execute(
        f'DELETE FROM "{collection._unembedded_table}" WHERE doc_id = $1',
        "orphan_doc",
    )

    embedder = NullEmbedder()
    await db._sync_embedded_with_unembedded(
        collection._unembedded_table,
        collection._embedded_table,
        embedder,
    )

    # Orphan should be cleaned up from embedded
    found = await collection.find_one({"id": {"$eq": "orphan_doc"}})
    assert found is None


async def test_that_sync_handles_mixed_new_updated_and_deleted_documents(
    container: Container,
    pgvector_db: object,
) -> None:
    """Sync should handle a mix of new, updated (checksum changed), and deleted docs."""
    db = pgvector_db
    assert isinstance(db, PostgresVectorDatabase)

    collection = await db.create_collection(
        name="sync_mixed_test",
        schema=_TestDocument,
        embedder_type=NullEmbedder,
    )

    # Insert initial docs
    await collection.insert_one(_make_doc("keep_doc", "keep content", "keeper"))
    await collection.insert_one(_make_doc("update_doc", "old content", "updater"))
    await collection.insert_one(_make_doc("delete_doc", "delete content", "deleter"))

    pool = db._get_pool()

    # Simulate out-of-band changes to unembedded table:
    # 1. Add a new doc
    await pool.execute(
        f"""
        INSERT INTO "{collection._unembedded_table}" (doc_id, content, checksum, metadata)
        VALUES ($1, $2, $3, $4::jsonb)
        """,
        "new_doc",
        "new content",
        md5_checksum("new content"),
        {"id": "new_doc", "name": "newbie"},
    )

    # 2. Update checksum for update_doc
    new_checksum = md5_checksum("updated content")
    await pool.execute(
        f"""
        UPDATE "{collection._unembedded_table}"
        SET content = $1, checksum = $2
        WHERE doc_id = $3
        """,
        "updated content",
        new_checksum,
        "update_doc",
    )

    # 3. Remove delete_doc from unembedded
    await pool.execute(
        f'DELETE FROM "{collection._unembedded_table}" WHERE doc_id = $1',
        "delete_doc",
    )

    embedder = NullEmbedder()
    await db._sync_embedded_with_unembedded(
        collection._unembedded_table,
        collection._embedded_table,
        embedder,
    )

    # Verify results
    keep = await collection.find_one({"id": {"$eq": "keep_doc"}})
    assert keep is not None
    assert keep["content"] == "keep content"

    updated = await collection.find_one({"id": {"$eq": "update_doc"}})
    assert updated is not None
    assert updated["content"] == "updated content"

    deleted = await collection.find_one({"id": {"$eq": "delete_doc"}})
    assert deleted is None

    new = await collection.find_one({"id": {"$eq": "new_doc"}})
    assert new is not None
    assert new["content"] == "new content"


async def test_that_sync_detects_metadata_only_changes(
    container: Container,
    pgvector_db: object,
) -> None:
    """If metadata changes but content/checksum stays the same, sync should update embedded."""
    db = pgvector_db
    assert isinstance(db, PostgresVectorDatabase)

    collection = await db.create_collection(
        name="sync_meta_test",
        schema=_TestDocument,
        embedder_type=NullEmbedder,
    )

    await collection.insert_one(_make_doc("meta_doc", "same content", "original_name"))

    pool = db._get_pool()

    # Directly update metadata in unembedded table (same content/checksum, different name)
    await pool.execute(
        f"""
        UPDATE "{collection._unembedded_table}"
        SET metadata = metadata || '{{"name": "updated_name"}}'::jsonb
        WHERE doc_id = $1
        """,
        "meta_doc",
    )

    embedder = NullEmbedder()
    await db._sync_embedded_with_unembedded(
        collection._unembedded_table,
        collection._embedded_table,
        embedder,
    )

    found = await collection.find_one({"id": {"$eq": "meta_doc"}})
    assert found is not None
    assert found["name"] == "updated_name"
    assert found["content"] == "same content"


async def test_that_update_one_uses_transactional_read_modify_write(
    container: Container,
    pgvector_db: object,
) -> None:
    """Update a document and verify both tables are consistent."""
    db = pgvector_db
    assert isinstance(db, PostgresVectorDatabase)

    collection = await db.create_collection(
        name="txn_update_test",
        schema=_TestDocument,
        embedder_type=NullEmbedder,
    )

    doc = _make_doc("txn_doc", "original content", "original")
    await collection.insert_one(doc)

    updated = _TestDocument(
        id=ObjectId("txn_doc"),
        version=Version.String("0.1.0"),
        content="updated content",
        checksum=md5_checksum("updated content"),
        name="updated",
    )

    result = await collection.update_one({"id": {"$eq": "txn_doc"}}, updated)
    assert result.matched_count == 1
    assert result.updated_document is not None
    assert result.updated_document["name"] == "updated"

    # Verify the embedded table reflects the update
    found = await collection.find_one({"id": {"$eq": "txn_doc"}})
    assert found is not None
    assert found["content"] == "updated content"
    assert found["name"] == "updated"

    # Verify the unembedded table also reflects the update
    pool = db._get_pool()
    row = await pool.fetchrow(
        f'SELECT doc_id, content, metadata FROM "{collection._unembedded_table}" WHERE doc_id = $1',
        "txn_doc",
    )
    assert row is not None
    assert row["content"] == "updated content"
    assert row["metadata"]["name"] == "updated"


class _HighDimEmbedder(Embedder):
    """Embedder with 3072 dimensions to test HNSW index creation beyond 2000 dims."""

    def __init__(self) -> None:
        self._tokenizer = ZeroEstimatingTokenizer()

    async def embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        return EmbeddingResult(vectors=[[0.0] * self.dimensions for _ in texts])

    @property
    @override
    def id(self) -> str:
        return "high_dim_test"

    @property
    @override
    def max_tokens(self) -> int:
        return 8192

    @property
    @override
    def tokenizer(self) -> EstimatingTokenizer:
        return self._tokenizer

    @property
    @override
    def dimensions(self) -> int:
        return 3072


async def test_that_hnsw_index_is_created_for_high_dimension_embedder(
    container: Container,
    pgvector_db: object,
) -> None:
    db = pgvector_db
    assert isinstance(db, PostgresVectorDatabase)

    container[_HighDimEmbedder] = _HighDimEmbedder()

    collection = await db.create_collection(
        name="high_dim_test",
        schema=_TestDocument,
        embedder_type=_HighDimEmbedder,
    )

    pool = db._get_pool()
    row = await pool.fetchrow(
        "SELECT indexname FROM pg_indexes WHERE tablename = $1 AND indexname LIKE '%hnsw%'",
        collection._embedded_table,
    )
    assert row is not None, "HNSW index should be created for 3072 dimensions with halfvec"


async def test_that_pgvector_ne_filter_works(
    pgvector_db: object,
) -> None:
    db = pgvector_db
    assert isinstance(db, PostgresVectorDatabase)

    collection = await db.create_collection(
        name="ne_filter_test",
        schema=_TestDocument,
        embedder_type=NullEmbedder,
    )
    await collection.insert_one(_make_doc("doc1", "alpha content", "alpha"))
    await collection.insert_one(_make_doc("doc2", "beta content", "beta"))

    results = await collection.find({"name": {"$ne": "alpha"}})
    assert len(results) == 1
    assert results[0]["name"] == "beta"


async def test_that_pgvector_or_logical_filter_works(
    pgvector_db: object,
) -> None:
    db = pgvector_db
    assert isinstance(db, PostgresVectorDatabase)

    collection = await db.create_collection(
        name="or_filter_test",
        schema=_TestDocument,
        embedder_type=NullEmbedder,
    )
    await collection.insert_one(_make_doc("doc1", "alpha content", "alpha"))
    await collection.insert_one(_make_doc("doc2", "beta content", "beta"))
    await collection.insert_one(_make_doc("doc3", "gamma content", "gamma"))

    results = await collection.find(
        {
            "$or": [
                {"name": {"$eq": "alpha"}},
                {"name": {"$eq": "gamma"}},
            ]
        }
    )
    assert len(results) == 2
    names = {r["name"] for r in results}
    assert names == {"alpha", "gamma"}


async def test_that_similarity_search_with_filters_works(
    pgvector_db: object,
) -> None:
    db = pgvector_db
    assert isinstance(db, PostgresVectorDatabase)

    collection = await db.create_collection(
        name="sim_filter_test",
        schema=_TestDocument,
        embedder_type=NullEmbedder,
    )
    await collection.insert_one(_make_doc("doc1", "hello world", "alpha", rank=1))
    await collection.insert_one(_make_doc("doc2", "goodbye world", "beta", rank=2))
    await collection.insert_one(_make_doc("doc3", "hello again", "alpha", rank=3))

    # Search with a filter that excludes doc2
    results = await collection.find_similar_documents(
        filters={"name": {"$eq": "alpha"}},
        query="hello",
        k=5,
    )

    # Only alpha docs should be returned
    assert all(r.document["name"] == "alpha" for r in results)
    assert len(results) == 2


async def test_that_pgvector_get_collection_raises_for_nonexistent(
    pgvector_db: object,
) -> None:
    db = pgvector_db
    assert isinstance(db, PostgresVectorDatabase)

    async def loader(doc: BaseDocument) -> Optional[_TestDocument]:
        return cast(_TestDocument, doc)

    with pytest.raises(ValueError, match="not found"):
        await db.get_collection(
            "nonexistent_collection",
            _TestDocument,
            NullEmbedder,
            loader,
        )


async def test_that_pgvector_failed_migrations_are_stored_in_separate_table(
    pgvector_db: object,
) -> None:
    db = pgvector_db
    assert isinstance(db, PostgresVectorDatabase)

    # Create a collection and insert documents
    collection = await db.create_collection(
        name="fail_migrate_test",
        schema=_TestDocument,
        embedder_type=NullEmbedder,
    )
    await collection.insert_one(_make_doc("good_doc", "good content", "good"))
    await collection.insert_one(_make_doc("bad_doc", "bad content", "bad"))

    pool = db._get_pool()

    # Tamper with the unembedded table to set an unrecognized version
    await pool.execute(
        f"""UPDATE "{collection._unembedded_table}"
            SET metadata = metadata || '{{"version": "99.0.0"}}'::jsonb
            WHERE doc_id = $1""",
        "bad_doc",
    )

    # Clear the in-memory cache so get_collection re-runs migration
    db._collections.pop("fail_migrate_test", None)

    async def rejecting_loader(doc: BaseDocument) -> Optional[_TestDocument]:
        raw_meta = doc.get("version", "")
        if raw_meta == "99.0.0":
            return None  # Signal failure
        return cast(_TestDocument, doc)

    await db.get_collection(
        "fail_migrate_test",
        _TestDocument,
        NullEmbedder,
        rejecting_loader,
    )

    # Verify failed migrations table exists and contains the bad doc
    failed_table = db._table_name("fail_migrate_test", "failed_migrations")
    assert await db._table_exists(failed_table)

    row = await pool.fetchrow(f'SELECT doc_id FROM "{failed_table}" WHERE doc_id = $1', "bad_doc")
    assert row is not None, "bad_doc should be in the failed migrations table"


async def test_that_get_or_create_collection_skips_per_row_migration_when_migration_required_is_false(
    pgvector_db: object,
) -> None:
    from unittest.mock import AsyncMock, patch

    db = pgvector_db
    assert isinstance(db, PostgresVectorDatabase)

    async def identity(doc: BaseDocument) -> Optional[_TestDocument]:
        return cast(_TestDocument, doc)

    collection = await db.get_or_create_collection(
        name="skip_migration_test",
        schema=_TestDocument,
        embedder_type=NullEmbedder,
        document_loader=identity,
    )
    await collection.insert_one(_make_doc("doc1", "existing content", "existing"))
    db._collections.pop("skip_migration_test", None)

    with patch.object(db, "_do_load_and_migrate", new_callable=AsyncMock) as spy:
        reopened = await db.get_or_create_collection(
            name="skip_migration_test",
            schema=_TestDocument,
            embedder_type=NullEmbedder,
            document_loader=identity,
            migration_required=False,
        )

    spy.assert_not_awaited()

    found = await reopened.find_one({"id": {"$eq": "doc1"}})
    assert found is not None
    assert found["content"] == "existing content"


async def test_that_get_or_create_collection_runs_per_row_migration_when_migration_required_is_true_or_omitted(
    pgvector_db: object,
) -> None:
    from unittest.mock import AsyncMock, patch

    db = pgvector_db
    assert isinstance(db, PostgresVectorDatabase)

    async def identity(doc: BaseDocument) -> Optional[_TestDocument]:
        return cast(_TestDocument, doc)

    collection = await db.get_or_create_collection(
        name="run_migration_test",
        schema=_TestDocument,
        embedder_type=NullEmbedder,
        document_loader=identity,
    )
    await collection.insert_one(_make_doc("doc1", "content", "name"))

    db._collections.pop("run_migration_test", None)
    with patch.object(
        db, "_do_load_and_migrate", new_callable=AsyncMock, return_value=(0, 0)
    ) as spy_default:
        await db.get_or_create_collection(
            name="run_migration_test",
            schema=_TestDocument,
            embedder_type=NullEmbedder,
            document_loader=identity,
        )
    spy_default.assert_awaited_once()

    db._collections.pop("run_migration_test", None)
    with patch.object(
        db, "_do_load_and_migrate", new_callable=AsyncMock, return_value=(0, 0)
    ) as spy_explicit:
        await db.get_or_create_collection(
            name="run_migration_test",
            schema=_TestDocument,
            embedder_type=NullEmbedder,
            document_loader=identity,
            migration_required=True,
        )
    spy_explicit.assert_awaited_once()


async def test_that_get_or_create_collection_still_syncs_embedded_table_when_migration_skipped(
    pgvector_db: object,
) -> None:
    from unittest.mock import AsyncMock, patch

    db = pgvector_db
    assert isinstance(db, PostgresVectorDatabase)

    async def identity(doc: BaseDocument) -> Optional[_TestDocument]:
        return cast(_TestDocument, doc)

    collection = await db.get_or_create_collection(
        name="skip_still_sync_test",
        schema=_TestDocument,
        embedder_type=NullEmbedder,
        document_loader=identity,
    )

    pool = db._get_pool()
    await pool.execute(
        f'INSERT INTO "{collection._unembedded_table}" (doc_id, content, checksum, metadata) '
        f"VALUES ($1, $2, $3, $4::jsonb)",
        "oob_doc",
        "out-of-band content",
        md5_checksum("out-of-band content"),
        {"id": "oob_doc", "name": "oob"},
    )

    db._collections.pop("skip_still_sync_test", None)
    with patch.object(
        db, "_do_load_and_migrate", new_callable=AsyncMock, return_value=(0, 0)
    ) as migrate_spy:
        reopened = await db.get_or_create_collection(
            name="skip_still_sync_test",
            schema=_TestDocument,
            embedder_type=NullEmbedder,
            document_loader=identity,
            migration_required=False,
        )

    # Sync ran (oob doc made it into the embedded table) without migration running.
    migrate_spy.assert_not_awaited()
    found = await reopened.find_one({"id": {"$eq": "oob_doc"}})
    assert found is not None
    assert found["content"] == "out-of-band content"


async def test_that_get_collection_skips_per_row_migration_when_migration_required_is_false(
    pgvector_db: object,
) -> None:
    from unittest.mock import AsyncMock, patch

    db = pgvector_db
    assert isinstance(db, PostgresVectorDatabase)

    async def identity(doc: BaseDocument) -> Optional[_TestDocument]:
        return cast(_TestDocument, doc)

    collection = await db.get_or_create_collection(
        name="get_skip_test",
        schema=_TestDocument,
        embedder_type=NullEmbedder,
        document_loader=identity,
    )
    await collection.insert_one(_make_doc("doc1", "content", "name"))
    db._collections.pop("get_skip_test", None)

    with patch.object(db, "_do_load_and_migrate", new_callable=AsyncMock) as spy:
        await db.get_collection(
            "get_skip_test",
            _TestDocument,
            NullEmbedder,
            identity,
            migration_required=False,
        )
    spy.assert_not_awaited()


async def test_that_vector_document_store_migration_helper_exposes_migration_required_after_aenter(
    pgvector_db: object,
) -> None:
    from parlant.core.persistence.vector_database_helper import (
        VectorDocumentStoreMigrationHelper,
    )

    db = pgvector_db
    assert isinstance(db, PostgresVectorDatabase)

    class _StubVersionedStore:
        VERSION = Version.from_string("1.0.0")

    original_version = _StubVersionedStore.VERSION
    try:
        async with VectorDocumentStoreMigrationHelper(
            store=cast(Any, _StubVersionedStore()),
            database=db,
            allow_migration=True,
        ) as helper:
            assert helper.migration_required is False

        async with VectorDocumentStoreMigrationHelper(
            store=cast(Any, _StubVersionedStore()),
            database=db,
            allow_migration=True,
        ) as helper:
            assert helper.migration_required is False

        _StubVersionedStore.VERSION = Version.from_string("2.0.0")
        async with VectorDocumentStoreMigrationHelper(
            store=cast(Any, _StubVersionedStore()),
            database=db,
            allow_migration=True,
        ) as helper:
            assert helper.migration_required is True

        # After the bump-and-migrate cycle finishes, __aexit__ must have
        # stamped the new version on the per-store key. The next boot at the
        # same version must therefore see migration_required=False — otherwise
        # the per-row walk repeats every warm start after any version bump.
        async with VectorDocumentStoreMigrationHelper(
            store=cast(Any, _StubVersionedStore()),
            database=db,
            allow_migration=True,
        ) as helper:
            assert helper.migration_required is False
    finally:
        _StubVersionedStore.VERSION = original_version
