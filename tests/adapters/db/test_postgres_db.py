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
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Optional, TypedDict, cast

import pytest
from lagom import Container
from pytest import fixture
from typing_extensions import Self

from parlant.adapters.db.postgres_db import PostgresDocumentDatabase
from parlant.core.agents import AgentId
from parlant.core.async_utils import Timeout
from parlant.core.common import Version
from parlant.core.customers import CustomerId
from parlant.core.persistence.common import Cursor, ObjectId, SortDirection
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentCollection,
    FindResult,
)
from parlant.core.persistence.document_database_helper import DocumentStoreMigrationHelper
from parlant.core.loggers import Logger
from parlant.core.sessions import (
    EventKind,
    EventSource,
    PollingSessionListener,
    SessionDocumentStore,
)

try:
    import asyncpg  # type: ignore[import-untyped]
except ImportError:
    asyncpg = None  # type: ignore[assignment]


class PostgresTestDocument(TypedDict, total=False):
    id: ObjectId
    creation_utc: str
    version: Version.String
    name: str
    count: int
    active: bool


class DummyStore:
    VERSION = Version.from_string("2.0.0")

    class DummyDocumentV1(TypedDict, total=False):
        id: ObjectId
        creation_utc: str
        version: Version.String
        name: str

    class DummyDocumentV2(TypedDict, total=False):
        id: ObjectId
        creation_utc: str
        version: Version.String
        name: str
        additional_field: str

    def __init__(
        self,
        database: Any,
        allow_migration: bool = True,
    ) -> None:
        self._database = database
        self._collection: DocumentCollection[DummyStore.DummyDocumentV2]
        self.allow_migration = allow_migration

    async def _document_loader(self, doc: BaseDocument) -> Optional[DummyDocumentV2]:
        if doc["version"] == "1.0.0":
            doc = cast(DummyStore.DummyDocumentV1, doc)
            return self.DummyDocumentV2(
                id=doc["id"],
                version=Version.String("2.0.0"),
                name=doc["name"],
                additional_field="default_value",
                creation_utc=str(doc.get("creation_utc", "2023-01-01T00:00:00Z")),
            )
        elif doc["version"] == "2.0.0":
            doc_with_creation = dict(doc)
            if "creation_utc" not in doc_with_creation:
                doc_with_creation["creation_utc"] = "2023-01-01T00:00:00Z"
            return cast(DummyStore.DummyDocumentV2, doc_with_creation)
        return None

    async def __aenter__(self) -> Self:
        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._database,
            allow_migration=self.allow_migration,
        ):
            self._collection = await self._database.get_or_create_collection(
                name="dummy_collection",
                schema=DummyStore.DummyDocumentV2,
                document_loader=self._document_loader,
            )
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        pass

    async def list_dummy(
        self,
        limit: Optional[int] = None,
        cursor: Optional[Cursor] = None,
        sort_direction: Optional[SortDirection] = None,
    ) -> FindResult[DummyDocumentV2]:
        if sort_direction is not None:
            return await self._collection.find(
                {}, limit=limit, cursor=cursor, sort_direction=sort_direction
            )
        return await self._collection.find({}, limit=limit, cursor=cursor)

    async def create_dummy(self, name: str, additional_field: str = "default") -> DummyDocumentV2:
        doc = self.DummyDocumentV2(
            id=ObjectId(f"dummy_{name}"),
            version=Version.String("2.0.0"),
            name=name,
            additional_field=additional_field,
            creation_utc=datetime.now(timezone.utc).isoformat(),
        )
        await self._collection.insert_one(doc)
        return doc

    async def read_dummy(self, doc_id: str) -> Optional[DummyDocumentV2]:
        return await self._collection.find_one({"id": {"$eq": doc_id}})

    async def update_dummy(self, doc_id: str, name: str) -> Optional[DummyDocumentV2]:
        existing = await self._collection.find_one({"id": {"$eq": doc_id}})
        if existing is None:
            return None

        updated_doc = self.DummyDocumentV2(
            id=existing["id"],
            version=existing["version"],
            name=name,
            additional_field=existing["additional_field"],
            creation_utc=existing["creation_utc"],
        )

        result = await self._collection.update_one({"id": {"$eq": doc_id}}, updated_doc)
        return result.updated_document

    async def delete_dummy(self, doc_id: str) -> bool:
        result = await self._collection.delete_one({"id": {"$eq": doc_id}})
        return result.acknowledged and result.deleted_count > 0


@fixture
def postgres_dsn() -> str:
    dsn = os.environ.get("TEST_POSTGRES_DSN")
    if dsn:
        return dsn
    print("Could not find `TEST_POSTGRES_DSN` in environment, skipping postgres tests...")
    raise pytest.skip()


@fixture
async def postgres_db(
    container: Container,
    postgres_dsn: str,
) -> AsyncIterator[Any]:
    async with PostgresDocumentDatabase(
        dsn=postgres_dsn,
        logger=container[Logger],
    ) as db:
        # Clean up test tables before each test
        pool = db._get_pool()
        tables = await pool.fetch("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        for table in tables:
            await pool.execute(f'DROP TABLE IF EXISTS "{table["tablename"]}" CASCADE')

        yield db


async def test_that_documents_can_be_created_and_found(
    container: Container,
    postgres_db: Any,
) -> None:
    async with DummyStore(postgres_db) as store:
        created = await store.create_dummy(name="test-dummy")
        dummies = await store.list_dummy()

        assert dummies.total_count == 1
        assert dummies.items[0]["name"] == "test-dummy"
        assert dummies.items[0]["id"] == created["id"]


async def test_that_documents_can_be_retrieved_by_id(
    container: Container,
    postgres_db: Any,
) -> None:
    async with DummyStore(postgres_db) as store:
        created = await store.create_dummy(
            name="retrievable_dummy", additional_field="custom_value"
        )
        retrieved = await store.read_dummy(created["id"])

        assert retrieved is not None
        assert retrieved["name"] == "retrievable_dummy"
        assert retrieved["additional_field"] == "custom_value"


async def test_that_multiple_documents_can_be_created_and_retrieved(
    container: Container,
    postgres_db: Any,
) -> None:
    async with DummyStore(postgres_db) as store:
        first = await store.create_dummy(name="first", additional_field="first_val")
        second = await store.create_dummy(name="second", additional_field="second_val")

        dummies = await store.list_dummy()
        assert dummies.total_count == 2

        ids = [d["id"] for d in dummies.items]
        assert first["id"] in ids
        assert second["id"] in ids


async def test_that_documents_can_be_updated(
    container: Container,
    postgres_db: Any,
) -> None:
    async with DummyStore(postgres_db) as store:
        created = await store.create_dummy(name="original")
        updated = await store.update_dummy(created["id"], "updated_name")

        assert updated is not None
        assert updated["name"] == "updated_name"

        retrieved = await store.read_dummy(created["id"])
        assert retrieved is not None
        assert retrieved["name"] == "updated_name"


async def test_that_documents_can_be_deleted(
    container: Container,
    postgres_db: Any,
) -> None:
    async with DummyStore(postgres_db) as store:
        created = await store.create_dummy(name="deletable")

        result = await store.delete_dummy(created["id"])
        assert result is True

        retrieved = await store.read_dummy(created["id"])
        assert retrieved is None


async def test_that_delete_returns_false_for_nonexistent_document(
    container: Container,
    postgres_db: Any,
) -> None:
    async with DummyStore(postgres_db) as store:
        result = await store.delete_dummy("nonexistent_id")
        assert result is False


async def test_that_find_one_returns_none_when_no_match(
    container: Container,
    postgres_db: Any,
) -> None:
    async with DummyStore(postgres_db) as store:
        result = await store.read_dummy("nonexistent_id")
        assert result is None


async def test_that_find_with_limit_works(
    container: Container,
    postgres_db: Any,
) -> None:
    async with DummyStore(postgres_db) as store:
        for i in range(5):
            await store.create_dummy(name=f"item_{i}")

        result = await store.list_dummy(limit=3)
        assert len(result.items) == 3
        assert result.total_count == 5
        assert result.has_more is True
        assert result.next_cursor is not None


async def test_that_cursor_pagination_works(
    container: Container,
    postgres_db: Any,
) -> None:
    async with DummyStore(postgres_db) as store:
        for i in range(5):
            await store.create_dummy(name=f"item_{i}")

        # Get first page
        page1 = await store.list_dummy(limit=2)
        assert len(page1.items) == 2
        assert page1.total_count == 5
        assert page1.has_more is True
        assert page1.next_cursor is not None

        # Get second page
        page2 = await store.list_dummy(limit=2, cursor=page1.next_cursor)
        assert len(page2.items) == 2
        assert page2.total_count == 3
        assert page2.has_more is True

        # Get third page
        page3 = await store.list_dummy(limit=2, cursor=page2.next_cursor)
        assert len(page3.items) == 1
        assert page3.total_count == 1
        assert page3.has_more is False

        # Ensure no duplicate IDs across pages
        all_ids = [d["id"] for d in list(page1.items) + list(page2.items) + list(page3.items)]
        assert len(all_ids) == len(set(all_ids))


async def test_that_descending_sort_works(
    container: Container,
    postgres_db: Any,
) -> None:
    async with DummyStore(postgres_db) as store:
        import time

        for i in range(3):
            await store.create_dummy(name=f"item_{i}")
            time.sleep(0.01)  # Ensure different timestamps

        result = await store.list_dummy(sort_direction=SortDirection.DESC)
        names = [d["name"] for d in result.items]
        assert names == ["item_2", "item_1", "item_0"]


async def test_that_find_with_filters_works(
    container: Container,
    postgres_db: Any,
) -> None:
    collection = await postgres_db.create_collection(
        name="filter_test", schema=PostgresTestDocument
    )

    doc1 = PostgresTestDocument(
        id=ObjectId("doc1"),
        creation_utc=datetime.now(timezone.utc).isoformat(),
        version=Version.String("1.0.0"),
        name="alpha",
    )
    doc2 = PostgresTestDocument(
        id=ObjectId("doc2"),
        creation_utc=datetime.now(timezone.utc).isoformat(),
        version=Version.String("1.0.0"),
        name="beta",
    )

    await collection.insert_one(doc1)
    await collection.insert_one(doc2)

    # Filter by name
    result = await collection.find({"name": {"$eq": "alpha"}})
    assert result.total_count == 1
    assert result.items[0]["name"] == "alpha"


async def test_that_jsonb_numeric_filters_use_numeric_semantics(
    container: Container,
    postgres_db: Any,
) -> None:
    collection = await postgres_db.create_collection(
        name="numeric_filter_test", schema=PostgresTestDocument
    )

    for doc_id, count in [("doc2", 2), ("doc10", 10), ("doc11", 11)]:
        await collection.insert_one(
            PostgresTestDocument(
                id=ObjectId(doc_id),
                creation_utc=datetime.now(timezone.utc).isoformat(),
                version=Version.String("1.0.0"),
                name=doc_id,
                count=count,
            )
        )

    gt_result = await collection.find({"count": {"$gt": 9}})
    gte_result = await collection.find({"count": {"$gte": 10}})
    lt_result = await collection.find({"count": {"$lt": 10}})
    lte_result = await collection.find({"count": {"$lte": 10}})
    in_result = await collection.find({"count": {"$in": [2, 11]}})
    nin_result = await collection.find({"count": {"$nin": [10]}})

    assert {doc["count"] for doc in gt_result.items} == {10, 11}
    assert {doc["count"] for doc in gte_result.items} == {10, 11}
    assert {doc["count"] for doc in lt_result.items} == {2}
    assert {doc["count"] for doc in lte_result.items} == {2, 10}
    assert {doc["count"] for doc in in_result.items} == {2, 11}
    assert {doc["count"] for doc in nin_result.items} == {2, 11}


async def test_that_jsonb_boolean_filters_are_not_coerced_to_text(
    container: Container,
    postgres_db: Any,
) -> None:
    collection = await postgres_db.create_collection(
        name="boolean_filter_test", schema=PostgresTestDocument
    )

    await collection.insert_one(
        PostgresTestDocument(
            id=ObjectId("active_doc"),
            creation_utc=datetime.now(timezone.utc).isoformat(),
            version=Version.String("1.0.0"),
            name="active",
            active=True,
        )
    )
    await collection.insert_one(
        PostgresTestDocument(
            id=ObjectId("inactive_doc"),
            creation_utc=datetime.now(timezone.utc).isoformat(),
            version=Version.String("1.0.0"),
            name="inactive",
            active=False,
        )
    )

    result = await collection.find({"active": {"$eq": True}})

    assert result.total_count == 1
    assert result.items[0]["id"] == "active_doc"


async def test_that_find_one_sorts_jsonb_numbers_numerically(
    container: Container,
    postgres_db: Any,
) -> None:
    collection = await postgres_db.create_collection(
        name="numeric_sort_test", schema=PostgresTestDocument
    )

    for doc_id, count in [("doc2", 2), ("doc10", 10), ("doc11", 11)]:
        await collection.insert_one(
            PostgresTestDocument(
                id=ObjectId(doc_id),
                creation_utc=datetime.now(timezone.utc).isoformat(),
                version=Version.String("1.0.0"),
                name=doc_id,
                count=count,
            )
        )

    highest = await collection.find_one({}, sort=(("count", SortDirection.DESC),))
    lowest = await collection.find_one({}, sort=(("count", SortDirection.ASC),))

    assert highest is not None
    assert lowest is not None
    assert highest["count"] == 11
    assert lowest["count"] == 2


async def test_that_in_filter_works(
    container: Container,
    postgres_db: Any,
) -> None:
    collection = await postgres_db.create_collection(
        name="in_filter_test", schema=PostgresTestDocument
    )

    for name in ["alpha", "beta", "gamma"]:
        await collection.insert_one(
            PostgresTestDocument(
                id=ObjectId(f"doc_{name}"),
                creation_utc=datetime.now(timezone.utc).isoformat(),
                version=Version.String("1.0.0"),
                name=name,
            )
        )

    result = await collection.find({"name": {"$in": ["alpha", "gamma"]}})
    assert result.total_count == 2
    names = {d["name"] for d in result.items}
    assert names == {"alpha", "gamma"}


async def test_that_logical_and_filter_works(
    container: Container,
    postgres_db: Any,
) -> None:
    collection = await postgres_db.create_collection(
        name="and_filter_test", schema=PostgresTestDocument
    )

    await collection.insert_one(
        PostgresTestDocument(
            id=ObjectId("doc1"),
            creation_utc=datetime.now(timezone.utc).isoformat(),
            version=Version.String("1.0.0"),
            name="alpha",
        )
    )
    await collection.insert_one(
        PostgresTestDocument(
            id=ObjectId("doc2"),
            creation_utc=datetime.now(timezone.utc).isoformat(),
            version=Version.String("2.0.0"),
            name="alpha",
        )
    )

    result = await collection.find(
        {
            "$and": [
                {"name": {"$eq": "alpha"}},
                {"version": {"$eq": "1.0.0"}},
            ]
        }
    )
    assert result.total_count == 1
    assert result.items[0]["id"] == "doc1"


async def test_that_upsert_inserts_when_no_match(
    container: Container,
    postgres_db: Any,
) -> None:
    collection = await postgres_db.create_collection(
        name="upsert_test", schema=PostgresTestDocument
    )

    doc = PostgresTestDocument(
        id=ObjectId("upsert_doc"),
        creation_utc=datetime.now(timezone.utc).isoformat(),
        version=Version.String("1.0.0"),
        name="upserted",
    )

    result = await collection.update_one({"id": {"$eq": "upsert_doc"}}, doc, upsert=True)
    assert result.updated_document is not None
    assert result.matched_count == 0

    retrieved = await collection.find_one({"id": {"$eq": "upsert_doc"}})
    assert retrieved is not None
    assert retrieved["name"] == "upserted"


async def test_that_delete_collection_works(
    container: Container,
    postgres_db: Any,
) -> None:
    collection = await postgres_db.create_collection(
        name="deletable_col", schema=PostgresTestDocument
    )
    await collection.insert_one(
        PostgresTestDocument(
            id=ObjectId("doc1"),
            creation_utc=datetime.now(timezone.utc).isoformat(),
            version=Version.String("1.0.0"),
            name="test",
        )
    )

    await postgres_db.delete_collection("deletable_col")

    # Table should be gone
    pool = postgres_db._get_pool()
    exists = await pool.fetchrow(
        "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_class WHERE relname = 'deletable_col' AND relkind = 'r')"
    )
    assert not exists["exists"]


async def test_that_session_events_keep_numeric_offsets_and_polling_semantics(
    container: Container,
    postgres_db: Any,
) -> None:
    async with SessionDocumentStore(postgres_db) as session_store:
        session = await session_store.create_session(
            customer_id=CustomerId("customer"),
            agent_id=AgentId("agent"),
        )

        created_events = [
            await session_store.create_event(
                session_id=session.id,
                source=EventSource.CUSTOMER,
                kind=EventKind.CUSTOM,
                trace_id=f"trace-{index}",
                data={"index": index},
            )
            for index in range(12)
        ]

        assert [event.offset for event in created_events] == list(range(12))

        offset_filtered_events = await session_store.list_events(session.id, min_offset=11)
        assert [event.offset for event in offset_filtered_events] == [11]

        listener = PollingSessionListener(session_store)
        assert (
            await listener.wait_for_more_events(
                session.id,
                min_offset=12,
                timeout=Timeout.none(),
            )
            is False
        )

        next_event = await session_store.create_event(
            session_id=session.id,
            source=EventSource.CUSTOMER,
            kind=EventKind.CUSTOM,
            trace_id="trace-12",
            data={"index": 12},
        )
        assert next_event.offset == 12
        assert (
            await listener.wait_for_more_events(
                session.id,
                min_offset=12,
                timeout=Timeout.none(),
            )
            is True
        )


async def test_that_valid_collection_names_are_accepted(
    container: Container,
    postgres_db: Any,
) -> None:
    for name in ["simple", "with_underscores", "CamelCase", "mix123"]:
        collection = await postgres_db.create_collection(name=name, schema=PostgresTestDocument)
        assert collection is not None


async def test_that_invalid_collection_names_raise_value_error(
    container: Container,
    postgres_dsn: str,
) -> None:
    async with PostgresDocumentDatabase(
        dsn=postgres_dsn,
        logger=container[Logger],
    ) as db:
        with pytest.raises(ValueError, match="Invalid table name"):
            await db.create_collection(
                name="has spaces",
                schema=PostgresTestDocument,
            )

        with pytest.raises(ValueError, match="Invalid table name"):
            await db.create_collection(
                name="semi;colon",
                schema=PostgresTestDocument,
            )

        with pytest.raises(ValueError, match="Invalid table name"):
            await db.create_collection(
                name="quote'mark",
                schema=PostgresTestDocument,
            )


async def test_that_update_one_returns_updated_document_atomically(
    container: Container,
    postgres_db: Any,
) -> None:
    collection = await postgres_db.create_collection(
        name="atomic_update_test", schema=PostgresTestDocument
    )

    doc = PostgresTestDocument(
        id=ObjectId("atomic_doc"),
        creation_utc=datetime.now(timezone.utc).isoformat(),
        version=Version.String("1.0.0"),
        name="original",
    )
    await collection.insert_one(doc)

    updated_params = PostgresTestDocument(
        id=ObjectId("atomic_doc"),
        creation_utc=doc["creation_utc"],
        version=Version.String("1.0.0"),
        name="updated",
    )

    result = await collection.update_one({"id": {"$eq": "atomic_doc"}}, updated_params)
    assert result.matched_count == 1
    assert result.modified_count == 1
    assert result.updated_document is not None
    assert result.updated_document["name"] == "updated"


async def test_that_delete_one_returns_deleted_document_via_returning(
    container: Container,
    postgres_db: Any,
) -> None:
    collection = await postgres_db.create_collection(
        name="atomic_delete_test", schema=PostgresTestDocument
    )

    doc = PostgresTestDocument(
        id=ObjectId("del_doc"),
        creation_utc=datetime.now(timezone.utc).isoformat(),
        version=Version.String("1.0.0"),
        name="to_delete",
    )
    await collection.insert_one(doc)

    result = await collection.delete_one({"id": {"$eq": "del_doc"}})
    assert result.deleted_count == 1
    assert result.deleted_document is not None
    assert result.deleted_document["name"] == "to_delete"

    # Confirm it's gone
    found = await collection.find_one({"id": {"$eq": "del_doc"}})
    assert found is None


async def test_that_update_one_with_no_match_returns_zero_matched_count(
    container: Container,
    postgres_db: Any,
) -> None:
    collection = await postgres_db.create_collection(
        name="no_match_update_test", schema=PostgresTestDocument
    )

    result = await collection.update_one(
        {"id": {"$eq": "nonexistent"}},
        PostgresTestDocument(
            id=ObjectId("nonexistent"),
            creation_utc=datetime.now(timezone.utc).isoformat(),
            version=Version.String("1.0.0"),
            name="ghost",
        ),
    )
    assert result.matched_count == 0
    assert result.updated_document is None


@fixture
async def shared_pool(
    postgres_dsn: str,
) -> AsyncIterator[Any]:
    pool = await asyncpg.create_pool(
        dsn=postgres_dsn,
        min_size=2,
        max_size=10,
        init=PostgresDocumentDatabase._init_connection,
    )
    try:
        yield pool
    finally:
        await pool.close()


async def test_that_two_document_databases_can_share_a_single_pool(
    container: Container,
    shared_pool: Any,
) -> None:
    async with (
        PostgresDocumentDatabase(
            dsn="unused",
            logger=container[Logger],
            table_prefix="db1",
            pool=shared_pool,
        ) as db1,
        PostgresDocumentDatabase(
            dsn="unused",
            logger=container[Logger],
            table_prefix="db2",
            pool=shared_pool,
        ) as db2,
    ):
        col1 = await db1.create_collection(name="test", schema=PostgresTestDocument)
        col2 = await db2.create_collection(name="test", schema=PostgresTestDocument)

        await col1.insert_one(
            PostgresTestDocument(
                id=ObjectId("doc1"),
                creation_utc=datetime.now(timezone.utc).isoformat(),
                version=Version.String("1.0.0"),
                name="from_db1",
            )
        )
        await col2.insert_one(
            PostgresTestDocument(
                id=ObjectId("doc2"),
                creation_utc=datetime.now(timezone.utc).isoformat(),
                version=Version.String("1.0.0"),
                name="from_db2",
            )
        )

        result1 = await col1.find({})
        result2 = await col2.find({})

        assert result1.total_count == 1
        assert result1.items[0]["name"] == "from_db1"
        assert result2.total_count == 1
        assert result2.items[0]["name"] == "from_db2"


async def test_that_shared_pool_is_not_closed_when_individual_database_exits(
    container: Container,
    shared_pool: Any,
) -> None:
    async with PostgresDocumentDatabase(
        dsn="unused",
        logger=container[Logger],
        table_prefix="ephemeral",
        pool=shared_pool,
    ) as db:
        await db.create_collection(name="test", schema=PostgresTestDocument)

    # Pool should still be usable after database exits
    row = await shared_pool.fetchval("SELECT 1")
    assert row == 1
