"""Vector store manager for handling different vector databases."""
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias

from keboola.component.exceptions import UserException
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector
from langchain_pinecone import Pinecone
import pinecone
from langchain_qdrant import QdrantVectorStore
from langchain_redis import RedisVectorStore
from langchain_milvus import Milvus
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import AsyncOpenSearch
from pymilvus import connections
from qdrant_client import QdrantClient
from redis.asyncio import Redis as AsyncRedis
from tenacity import retry, stop_after_attempt, wait_exponential

sys.path.append(str(Path(__file__).parent.parent))

from configuration import ComponentConfig  # noqa

# Type aliases
VectorData: TypeAlias = dict[str, str | list[float]]
VectorBatch: TypeAlias = list[VectorData]

# Constants
BATCH_SIZE = 100
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

        if config.vector_db:
            self.vector_store = self._initialize_vector_store()

    def _initialize_vector_store(self):
        """Initialize the vector store based on configuration."""
        db_config = self.config.vector_db

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
                    collection_name=settings.table_name,
                    async_mode=True
                )

            case "pinecone":
                settings = db_config.pinecone_settings

                pinecone.init(
                    api_key=settings.api_key,
                    environment=settings.environment
                )

                return Pinecone.from_existing_index(
                    index_name=settings.index_name,
                    embedding=self.embedding_model,
                    namespace="keboola"
                )

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
                    collection_name="embeddings",
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

    @staticmethod
    def _create_documents(texts: Sequence[str], embeddings: Sequence[Sequence[float]]) -> list[Document]:
        """Create LangChain documents with embeddings."""
        return [
            Document(
                page_content=text,
                metadata={
                    "embedding": list(embedding),
                    "source": "keboola",
                    "created_at": "now()"
                }
            )
            for text, embedding in zip(texts, embeddings)
        ]

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=MIN_BACKOFF, max=MAX_BACKOFF)
    )
    async def store_vectors(
            self,
            texts: Sequence[str],
            embeddings: Sequence[Sequence[float]]
    ) -> None:
        """Store vectors in the vector store."""
        if not self.vector_store:
            return

        try:
            # Create LangChain documents with embeddings
            documents = self._create_documents(texts, embeddings)

            # Process in batches
            for i in range(0, len(documents), BATCH_SIZE):
                batch = documents[i:i + BATCH_SIZE]

                match type(self.vector_store).__name__:
                    case "PGVector" | "RedisVectorStore" | "OpenSearchVectorSearch":
                        # These support and benefit from async operations
                        async with self.semaphore:
                            ids = await self.vector_store.aadd_documents(batch)
                            self.stored_ids.extend(ids)
                    case _:
                        # Others are better with synchronous batch operations
                        ids = await asyncio.to_thread(
                            self.vector_store.add_documents,
                            batch
                        )
                        self.stored_ids.extend(ids)

        except Exception as e:
            raise UserException(f"Failed to store vectors: {str(e)}")
