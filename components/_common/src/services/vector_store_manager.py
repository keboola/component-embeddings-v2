"""Vector store manager for handling different vector databases."""
import asyncio
import logging
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeAlias

from keboola.component.exceptions import UserException
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_milvus import Milvus
from langchain_pinecone import PineconeVectorStore
from langchain_postgres import PGVector
from langchain_qdrant import QdrantVectorStore
from langchain_redis import RedisVectorStore
from opensearchpy import AsyncOpenSearch
from pinecone import Pinecone
from pymilvus import connections
from qdrant_client import QdrantClient
from redis.asyncio import Redis as AsyncRedis
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type

sys.path.append(str(Path(__file__).parent.parent))

from configuration import ComponentConfig  # noqa

# Type aliases
VectorData: TypeAlias = dict[str, str | list[float]]
VectorBatch: TypeAlias = list[VectorData]

# Constants
MAX_RETRIES = 3
MIN_BACKOFF = 4
MAX_BACKOFF = 10


class VectorStoreManager:
    """Manager for vector store operations."""

    def __init__(self, config: ComponentConfig, embedding_model: Embeddings) -> None:
        """Initialize vector store manager."""
        self.config = config
        self.embedding_model = embedding_model
        self.vector_store = None
        self.semaphore = asyncio.Semaphore(5)  # Limit concurrent connections
        self.stored_ids = []

        self.vector_store = self._initialize_vector_store()

    def _initialize_vector_store(self):
        """Initialize the vector store based on configuration."""
        db_config = self.config.vector_db
        try:
            match db_config.db_type:
                case "pgvector":
                    settings = db_config.pgvector_settings
                    connection_string = (
                        f"postgresql+psycopg://{settings.username}:{settings.password}"
                        f"@{settings.host}:{settings.port}/{settings.database}"
                    )
                    return PGVector(
                        connection=connection_string,
                        embeddings=self.embedding_model,
                        collection_name=settings.collection_name,
                        async_mode=True
                    )

                case "pinecone":
                    settings = db_config.pinecone_settings

                    pc = Pinecone(api_key=settings.api_key)
                    index_name = settings.index_name

                    if index_name not in [index_info["name"] for index_info in pc.list_indexes()]:
                        raise UserException(f"Index '{index_name}' not found in Pinecone.")

                    index = pc.Index(index_name)
                    return PineconeVectorStore(index=index, embedding=self.embedding_model)

                case "qdrant":
                    settings = db_config.qdrant_settings
                    client = QdrantClient(
                        url=settings.url,
                        api_key=settings.api_key,
                        prefer_grpc=True,
                        timeout=30  # Increase timeout for batch operations
                    )
                    return QdrantVectorStore(
                        client=client,
                        collection_name=settings.collection_name,
                        embedding=self.embedding_model
                    )

                case "milvus":
                    settings = db_config.milvus_settings

                    connections.connect(
                        alias="default",
                        host=settings.host,
                        port=settings.port,
                        user=settings.username,
                        password=settings.password
                    )
                    return Milvus(
                        collection_name="embeddings",
                        embedding_function=self.embedding_model,
                        connection_args={
                            "host": settings.host,
                            "port": settings.port,
                            "user": settings.username,
                            "password": settings.password,
                            "secure": True
                        }
                    )

                case "redis":
                    settings = db_config.redis_settings

                    client = AsyncRedis(
                        host=settings.host,
                        port=settings.port,
                        password=settings.password,
                        ssl=True
                    )
                    return RedisVectorStore(
                        redis_client=client,
                        index_name="embeddings",
                        embeddings=self.embedding_model
                    )

                case "opensearch":
                    settings = db_config.opensearch_settings

                    client = AsyncOpenSearch(
                        hosts=[{"host": settings.host, "port": settings.port}],
                        http_auth=(settings.username, settings.password),
                        use_ssl=True,
                        timeout=30,
                        max_retries=MAX_RETRIES,
                        retry_on_timeout=True,
                        verify_certs=True
                    )
                    return OpenSearchVectorSearch(
                        client=client,
                        index_name=settings.index_name,
                        embedding_function=self.embedding_model,
                        engine="nmslib",
                        space_type="cosinesimil",
                        ef_construction=512,
                        m=16,
                        opensearch_url="TODO"
                    )
                case _:
                    raise UserException(f"Unsupported vector store type: {db_config.db_type}")
        except Exception as e:
            if "details =" in str(e):
                detail = str(e).split("details =")[1].strip().strip('"')
                raise UserException(detail)
            raise UserException(f"Failed to initialize vector client: {str(e)}")

    @staticmethod
    def _create_documents(
            texts: Sequence[str],
            embeddings: Sequence[Sequence[float]],
            metadata: Sequence[dict]
    ) -> list[Document]:
        """Create LangChain documents with embeddings and metadata.
        Args:
            texts: Sequence of text content
            embeddings: Sequence of embedding vectors
            metadata: Sequence of metadata dictionaries
        Returns:
            List of Document objects with properly formatted metadata
        """
        current_time = datetime.now(timezone.utc).isoformat()

        return [
            Document(
                page_content=text,
                metadata={
                    "source": "keboola",
                    "created_at": current_time,
                    **meta  # Unpack user metadata directly into root
                }
            )
            for text, embedding, meta in zip(texts, embeddings, metadata)
        ]

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=MIN_BACKOFF, max=MAX_BACKOFF),
        retry=lambda e: not isinstance(e, UserException),
        reraise=True
    )
    async def store_vectors(
            self,
            texts: Sequence[str],
            embeddings: Sequence[Sequence[float]],
            metadata: Sequence[dict]
    ) -> None:
        """Store vectors in the vector store."""
        if not self.vector_store:
            return

        # Create LangChain documents with embeddings
        documents = self._create_documents(texts, embeddings, metadata)

        # Process in batches using configured batch size
        batch_size = self.config.advanced_options.batch_size
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            match type(self.vector_store).__name__:
                case "PGVector" | "RedisVectorStore" | "OpenSearchVectorSearch":
                    # These support and benefit from async operations
                    async with self.semaphore:
                        ids = await self.vector_store.aadd_documents(batch)
                        logging.debug(f"Async Stored {len(ids)} vectors")
                        self.stored_ids.extend(ids)
                case _:
                    # Others are better with synchronous batch operations
                    ids = await asyncio.to_thread(
                        self.vector_store.add_documents,
                        batch
                    )
                    logging.debug(f"Stored {len(ids)} vectors")
                    self.stored_ids.extend(ids)

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=MIN_BACKOFF, max=MAX_BACKOFF),
        retry=retry_if_not_exception_type(UserException),
        reraise=True
    )
    async def upsert_vectors(
            self,
            texts: Sequence[str],
            embeddings: Sequence[Sequence[float]],
            metadata: Sequence[dict],
            ids: Sequence[str]
    ) -> None:
        """Upsert vectors in the vector store (update if exists, insert if not)."""
        try:
            if not self.vector_store:
                return

            documents = self._create_documents(texts, embeddings, metadata)
            batch_size = self.config.advanced_options.batch_size

            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]

                async with self.semaphore:
                    # All vector stores support upsert via add_documents with ids
                    if isinstance(self.vector_store, OpenSearchVectorSearch):
                        # OpenSearch has special handling via native client
                        await self.vector_store.aadd_documents(batch)
                    else:
                        # For other vector stores use standard add_documents with ids
                        await self.vector_store.aadd_documents(batch, ids=batch_ids)
                        logging.debug(f"Upserted {len(batch)} vectors with IDs: {batch_ids}")
                        self.stored_ids.extend(batch_ids)
        except Exception as e:
            if "details =" in str(e):
                detail = str(e).split("details =")[1].strip().strip('"')
                raise UserException(detail)
            raise UserException(f"Failed to upsert vectors: {str(e)}")
