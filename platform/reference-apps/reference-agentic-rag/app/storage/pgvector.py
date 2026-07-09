"""pgvector storage adapter — implements VectorStorePort for PostgreSQL + pgvector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..search.ports import VectorStorePort


@dataclass(frozen=True)
class PgVectorConfig:
    connection_uri: str = "postgresql://rag:rag_dev_only@localhost:5432/rag_factory"
    table_name: str = "documents"
    embedding_column: str = "embedding"
    content_column: str = "content"
    dimensions: int = 1536
    index_type: str = "hnsw"  # hnsw or ivfflat


class PgVectorStore:
    """PostgreSQL + pgvector vector store implementation.

    Implements VectorStorePort protocol for dependency injection.
    """

    def __init__(self, config: PgVectorConfig | None = None) -> None:
        self._config = config or PgVectorConfig()
        self._pool = None

    async def initialize(self) -> None:
        """Create the table and index if they don't exist."""
        try:
            import asyncpg
        except ImportError as e:
            raise ImportError("Install asyncpg: pip install asyncpg") from e

        self._pool = await asyncpg.create_pool(self._config.connection_uri)
        async with self._pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._config.table_name} (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata JSONB DEFAULT '{{}}',
                    source TEXT DEFAULT '',
                    namespace TEXT DEFAULT '',
                    {self._config.embedding_column} vector({self._config.dimensions})
                )
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._config.table_name}_embedding
                ON {self._config.table_name}
                USING {self._config.index_type} ({self._config.embedding_column} vector_cosine_ops)
            """)

    async def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        namespace: str = "",
    ) -> list[dict[str, Any]]:
        if not self._pool:
            await self.initialize()

        embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
        where_clauses = ["namespace = $2"] if namespace else []

        if filters:
            for key, value in filters.items():
                where_clauses.append(f"metadata->>'{key}' = '{value}'")

        where = " AND ".join(where_clauses) if where_clauses else "TRUE"

        query = f"""
            SELECT id, content, metadata, source,
                   1 - ({self._config.embedding_column} <=> $1::vector) AS score
            FROM {self._config.table_name}
            WHERE {where}
            ORDER BY {self._config.embedding_column} <=> $1::vector
            LIMIT $3
        """

        async with self._pool.acquire() as conn:
            params = [embedding_str, namespace, top_k] if namespace else [embedding_str, top_k]
            if not namespace:
                query = query.replace("WHERE namespace = $2", "WHERE TRUE").replace("$3", "$2")

            rows = await conn.fetch(query, *params)

        return [
            {
                "id": row["id"],
                "content": row["content"],
                "score": float(row["score"]),
                "metadata": dict(row["metadata"]) if row["metadata"] else {},
                "source": row["source"],
            }
            for row in rows
        ]

    async def upsert(
        self,
        documents: list[dict[str, Any]],
        *,
        namespace: str = "",
    ) -> int:
        if not self._pool:
            await self.initialize()

        count = 0
        async with self._pool.acquire() as conn:
            for doc in documents:
                embedding_str = "[" + ",".join(str(v) for v in doc["embedding"]) + "]"
                await conn.execute(
                    f"""
                    INSERT INTO {self._config.table_name}
                        (id, content, metadata, source, namespace, {self._config.embedding_column})
                    VALUES ($1, $2, $3::jsonb, $4, $5, $6::vector)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        source = EXCLUDED.source,
                        {self._config.embedding_column} = EXCLUDED.{self._config.embedding_column}
                    """,
                    doc.get("id", ""),
                    doc.get("content", ""),
                    str(doc.get("metadata", {})),
                    doc.get("source", ""),
                    namespace,
                    embedding_str,
                )
                count += 1
        return count

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
