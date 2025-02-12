"""Configuration models for the embedding component."""
from typing import Literal, Optional
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, model_validator


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface_hub"
    GOOGLE_VERTEX = "google_vertex"
    BEDROCK = "bedrock"


class OpenAIModel(str, Enum):
    """Supported OpenAI embedding models."""
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"


class CohereModel(str, Enum):
    """Supported Cohere embedding models."""
    EMBED_ENGLISH_V3 = "embed-english-v3.0"
    EMBED_ENGLISH_LIGHT_V3 = "embed-english-light-v3.0"
    EMBED_MULTILINGUAL_V3 = "embed-multilingual-v3.0"
    EMBED_MULTILINGUAL_LIGHT_V3 = "embed-multilingual-light-v3.0"


class VectorStoreType(str, Enum):
    """Supported vector store types."""
    PGVECTOR = "pgvector"
    PINECONE = "pinecone"
    QDRANT = "qdrant"
    MILVUS = "milvus"
    REDIS = "redis"
    OPENSEARCH = "opensearch"


class PGVectorSettings(BaseModel):
    """PostgreSQL vector database settings."""
    model_config = ConfigDict(populate_by_name=True)

    host: str
    port: int = 5432
    database: str
    username: str
    password: str = Field(validation_alias="#password")
    collection_name: str = "keboola_embeddings"


class PineconeSettings(BaseModel):
    """Pinecone vector database settings."""
    model_config = ConfigDict(populate_by_name=True)

    api_key: str = Field(validation_alias="#api_key")
    environment: str
    index_name: str


class QdrantSettings(BaseModel):
    """Qdrant vector database settings."""
    model_config = ConfigDict(populate_by_name=True)

    url: str
    api_key: str = Field(validation_alias="#api_key")


class MilvusSettings(BaseModel):
    """Milvus vector database settings."""
    model_config = ConfigDict(populate_by_name=True)

    host: str
    port: int = 19530
    username: str
    password: str = Field(validation_alias="#password")


class RedisSettings(BaseModel):
    """Redis vector database settings."""
    model_config = ConfigDict(populate_by_name=True)

    host: str
    port: int = 6379
    password: str = Field(validation_alias="#password")


class OpenSearchSettings(BaseModel):
    """OpenSearch vector database settings."""
    model_config = ConfigDict(populate_by_name=True)

    host: str
    port: int = 9200
    username: str
    password: str = Field(validation_alias="#password")
    index_name: str = "embeddings"


class VectorDBConfig(BaseModel):
    """Vector database configuration."""
    model_config = ConfigDict(populate_by_name=True)

    db_type: VectorStoreType
    pgvector_settings: Optional[PGVectorSettings] = None
    pinecone_settings: Optional[PineconeSettings] = None
    qdrant_settings: Optional[QdrantSettings] = None
    milvus_settings: Optional[MilvusSettings] = None
    redis_settings: Optional[RedisSettings] = None
    opensearch_settings: Optional[OpenSearchSettings] = None

    @model_validator(mode='after')
    def validate_settings(self) -> 'VectorDBConfig':
        """Validate that appropriate settings are provided for the selected db_type."""
        settings_map = {
            VectorStoreType.PGVECTOR: self.pgvector_settings,
            VectorStoreType.PINECONE: self.pinecone_settings,
            VectorStoreType.QDRANT: self.qdrant_settings,
            VectorStoreType.MILVUS: self.milvus_settings,
            VectorStoreType.REDIS: self.redis_settings,
            VectorStoreType.OPENSEARCH: self.opensearch_settings
        }

        if settings_map[self.db_type] is None:
            raise ValueError(f"{self.db_type.value}_settings must be provided when db_type is {self.db_type.value}")

        for db_type, settings in settings_map.items():
            if db_type != self.db_type and settings is not None:
                raise ValueError(
                    f"{db_type.value}_settings should not be provided when db_type is {self.db_type.value}")

        return self


class ChunkingSettings(BaseModel):
    """Settings for text chunking."""
    model_config = ConfigDict(populate_by_name=True)

    chunk_size: int = Field(default=1000, ge=100, le=8000)
    chunk_overlap: int = Field(default=100, ge=0, le=1000)
    chunk_strategy: Literal["character", "sentence", "word", "paragraph"] = "paragraph"

    @model_validator(mode='after')
    def validate_overlap(self) -> 'ChunkingSettings':
        """Validate that overlap is smaller than chunk size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class AdvancedOptions(BaseModel):
    """Advanced processing options."""
    model_config = ConfigDict(populate_by_name=True)

    batch_size: int = Field(default=100, ge=1, le=1000)
    enable_chunking: bool = False
    chunking_settings: Optional[ChunkingSettings] = None

    @model_validator(mode='after')
    def validate_chunking_settings(self) -> 'AdvancedOptions':
        """Validate that chunking settings are present when chunking is enabled."""
        if self.enable_chunking and self.chunking_settings is None:
            raise ValueError("chunking_settings must be provided when enable_chunking is True")
        return self


class OutputConfig(BaseModel):
    """Configuration for output handling."""
    model_config = ConfigDict(populate_by_name=True)

    save_to_storage: bool = True
    save_to_vectordb: bool = False


class OpenAISettings(BaseModel):
    """OpenAI settings."""
    model_config = ConfigDict(populate_by_name=True)

    model: OpenAIModel
    api_key: str = Field(validation_alias="#api_key")


class AzureOpenAISettings(BaseModel):
    """Azure OpenAI settings."""
    model_config = ConfigDict(populate_by_name=True)

    deployment_name: str
    api_key: str = Field(validation_alias="#api_key")
    azure_endpoint: str
    api_version: str = "2024-02-01"


class CohereSettings(BaseModel):
    """Cohere settings."""
    model_config = ConfigDict(populate_by_name=True)

    model: CohereModel
    api_key: str = Field(validation_alias="#api_key")


class HuggingFaceSettings(BaseModel):
    """HuggingFace Hub settings."""
    model_config = ConfigDict(populate_by_name=True)

    model: str
    api_key: str = Field(validation_alias="#api_key")
    normalize_embeddings: bool = True
    show_progress: bool = False


class GoogleVertexSettings(BaseModel):
    """Google Vertex AI settings."""
    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    project: str
    credentials: str = Field(validation_alias="#credentials")
    location: str = "us-central1"
    model_name: str = "textembedding-gecko@latest"


class BedrockSettings(BaseModel):
    """AWS Bedrock settings."""
    model_config = ConfigDict(populate_by_name=True)

    aws_access_key: str = Field(validation_alias="#aws_access_key")
    aws_secret_key: str = Field(validation_alias="#aws_secret_key")
    region: str
    model_id: str = model_config.update(protected_namespaces=())


class EmbeddingSettings(BaseModel):
    """Embedding provider settings."""
    model_config = ConfigDict(populate_by_name=True)

    provider_type: EmbeddingProvider
    openai_settings: Optional[OpenAISettings] = None
    azure_settings: Optional[AzureOpenAISettings] = None
    cohere_settings: Optional[CohereSettings] = None
    huggingface_settings: Optional[HuggingFaceSettings] = None
    google_vertex_settings: Optional[GoogleVertexSettings] = None
    bedrock_settings: Optional[BedrockSettings] = None

    @model_validator(mode='after')
    def validate_settings(self) -> 'EmbeddingSettings':
        """Validate that appropriate settings are provided for the selected provider."""
        settings_map = {
            EmbeddingProvider.OPENAI: self.openai_settings,
            EmbeddingProvider.AZURE_OPENAI: self.azure_settings,
            EmbeddingProvider.COHERE: self.cohere_settings,
            EmbeddingProvider.HUGGINGFACE: self.huggingface_settings,
            EmbeddingProvider.GOOGLE_VERTEX: self.google_vertex_settings,
            EmbeddingProvider.BEDROCK: self.bedrock_settings
        }

        if settings_map[self.provider_type] is None:
            raise ValueError(f"{self.provider_type.value}_settings must be provided")

        for provider, settings in settings_map.items():
            if provider != self.provider_type and settings is not None:
                raise ValueError(f"{provider.value}_settings should not be provided")

        return self


class Destination(BaseModel):
    incremental_load: bool = Field(default=False)
    output_table_name: str
    primary_keys_array: list[str] = Field(default_factory=list)


class ComponentConfig(BaseModel):
    """Main configuration for the embedding component."""
    model_config = ConfigDict(populate_by_name=True)

    text_column: str
    metadata_columns: list[str]
    embedding_settings: EmbeddingSettings
    output_config: OutputConfig
    destination: Destination
    vector_db: Optional[VectorDBConfig] = None
    advanced_options: AdvancedOptions = Field(default_factory=AdvancedOptions)

    @model_validator(mode='after')
    def validate_vector_db(self) -> 'ComponentConfig':
        """Validate that vector_db is present when save_to_vectordb is True."""
        if self.output_config.save_to_vectordb and self.vector_db is None:
            raise ValueError("vector_db must be provided when save_to_vectordb is True")
        return self


def get_component_configuration() -> ComponentConfig:
    """Parse and validate the component configuration."""
    return ComponentConfig.model_validate({})  # This will be populated from component input
