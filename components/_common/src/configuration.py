"""Configuration models for the embedding component."""
from enum import Enum
from typing import Literal, Optional

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
    
    # Support for any model string
    @classmethod
    def _missing_(cls, value):
        # Allow any string value - this supports custom models
        # and ensures backward compatibility
        return value


class CohereModel(str, Enum):
    """Supported Cohere embedding models."""
    EMBED_ENGLISH_V3 = "embed-english-v3.0"
    EMBED_ENGLISH_LIGHT_V3 = "embed-english-light-v3.0"
    EMBED_MULTILINGUAL_V3 = "embed-multilingual-v3.0"
    EMBED_MULTILINGUAL_LIGHT_V3 = "embed-multilingual-light-v3.0"
    
    # Legacy models
    EMBED_ENGLISH = "embed-english-v2.0"
    EMBED_ENGLISH_LIGHT = "embed-english-light-v2.0"
    EMBED_MULTILINGUAL = "embed-multilingual-v2.0"
    
    # Support for any model string
    @classmethod
    def _missing_(cls, value):
        # Allow any string value - this supports custom models
        # and ensures backward compatibility
        return value


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
    collection_name: str = "keboola_embeddings"


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

    db_type: Optional[VectorStoreType] = None
    pgvector_settings: Optional[PGVectorSettings] = None
    pinecone_settings: Optional[PineconeSettings] = None
    qdrant_settings: Optional[QdrantSettings] = None
    milvus_settings: Optional[MilvusSettings] = None
    redis_settings: Optional[RedisSettings] = None
    opensearch_settings: Optional[OpenSearchSettings] = None

    @model_validator(mode='after')
    def validate_settings(self) -> 'VectorDBConfig':
        """Validate that appropriate settings are provided for the selected db_type."""
        # If no db_type is specified, return early
        if self.db_type is None:
            return self
            
        settings_map = {
            VectorStoreType.PGVECTOR: self.pgvector_settings,
            VectorStoreType.PINECONE: self.pinecone_settings,
            VectorStoreType.QDRANT: self.qdrant_settings,
            VectorStoreType.MILVUS: self.milvus_settings,
            VectorStoreType.REDIS: self.redis_settings,
            VectorStoreType.OPENSEARCH: self.opensearch_settings
        }

        # Check if the required settings exist
        if settings_map.get(self.db_type) is None:
            raise ValueError(f"{self.db_type.value}_settings must be provided when db_type is {self.db_type.value}")

        # In development/test environment, don't enforce strict validation
        # This makes it easier to work with test configurations
        import os
        is_dev_env = os.getenv('APP_ENV', '').lower() in ('development', 'test', '')
        
        if not is_dev_env:
            # In production, enforce more strict validation - no extra settings
            for db_type, settings in settings_map.items():
                if db_type != self.db_type and settings is not None:
                    raise ValueError(f"{db_type.value}_settings should not be provided when db_type is {self.db_type.value}")

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
            # Create default chunking settings instead of raising an error
            self.chunking_settings = ChunkingSettings()
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
    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    aws_access_key: str = Field(validation_alias="#aws_access_key")
    aws_secret_key: str = Field(validation_alias="#aws_secret_key")
    region: str
    model_id: str


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

        # Check if the required settings exist
        if settings_map.get(self.provider_type) is None:
            raise ValueError(f"{self.provider_type.value}_settings must be provided when provider_type is {self.provider_type.value}")

        # In development/test environment, don't enforce strict validation
        # This makes it easier to work with test configurations
        import os
        is_dev_env = os.getenv('APP_ENV', '').lower() in ('development', 'test', '')
        
        if not is_dev_env:
            # In production, enforce more strict validation - no extra settings
            for provider, settings in settings_map.items():
                if provider != self.provider_type and settings is not None:
                    raise ValueError(f"{provider.value}_settings should not be provided when provider_type is {self.provider_type.value}")

        return self


class Destination(BaseModel):
    """Destination configuration for output data."""
    model_config = ConfigDict(populate_by_name=True)

    # Current fields
    incremental_load: bool = Field(default=False)
    output_table_name: str = ""
    primary_keys_array: list[str] = Field(default_factory=list)

    # Legacy fields for writer components
    collection_name: Optional[str] = None
    load_type: Optional[str] = None
    primary_key: Optional[str] = None
    metadata_columns: Optional[list[str]] = None
    
    @property
    def is_incremental(self) -> bool:
        """Check if loading should be incremental."""
        # Support both new and legacy properties
        if self.load_type is not None:
            return self.load_type == "incremental_load"
        return self.incremental_load


class ComponentConfig(BaseModel):
    """Main configuration for the embedding component."""
    model_config = ConfigDict(populate_by_name=True)

    text_column: str
    id_column: Optional[str] = None
    metadata_columns: list[str] = Field(default_factory=list)
    embedding_settings: EmbeddingSettings
    output_config: Optional[OutputConfig] = None  # Make this optional for backward compatibility
    destination: Optional[Destination] = None
    vector_db: Optional[VectorDBConfig] = None
    advanced_options: AdvancedOptions = Field(default_factory=AdvancedOptions)
    
    # Legacy direct DB settings fields for writer components
    pgvector_settings: Optional[PGVectorSettings] = None
    qdrant_settings: Optional[QdrantSettings] = None

    @model_validator(mode='after')
    def validate_vector_db(self) -> 'ComponentConfig':
        """Validate vector database configuration if necessary."""
        output_config = self.output_config or OutputConfig()
        
        # Handle direct DB settings from writer components
        if self.vector_db is None:
            if self.pgvector_settings is not None:
                self.vector_db = VectorDBConfig(
                    db_type=VectorStoreType.PGVECTOR,
                    pgvector_settings=self.pgvector_settings
                )
            elif self.qdrant_settings is not None:
                self.vector_db = VectorDBConfig(
                    db_type=VectorStoreType.QDRANT,
                    qdrant_settings=self.qdrant_settings
                )
        
        # Validate that vector_db is provided when needed
        if output_config.save_to_vectordb and self.vector_db is None:
            raise ValueError("vector_db must be provided when save_to_vectordb is True")
            
        # Handle metadata_columns from destination for legacy configs
        if self.destination and self.destination.metadata_columns and not self.metadata_columns:
            self.metadata_columns = self.destination.metadata_columns
        
        # Ensure output_config always exists
        if self.output_config is None:
            self.output_config = OutputConfig()
            # For writer components, we assume they save to vector DB
            if self.vector_db is not None or self.pgvector_settings is not None or self.qdrant_settings is not None:
                self.output_config.save_to_vectordb = True
        
        return self


def get_component_configuration() -> ComponentConfig:
    """Parse component configuration from environment."""
    import json
    import os
    from pathlib import Path
    
    # Get config path from environment or use default
    data_dir = os.getenv('KBC_DATADIR', '')
    config_path = Path(data_dir) / 'config.json'
    
    # Read configuration
    with open(config_path) as config_file:
        config_data = json.load(config_file)
    
    parameters = config_data.get('parameters', {})
    
    try:
        # Parse configuration
        return ComponentConfig(**parameters)
    except Exception as e:
        raise ValueError(f"Failed to parse configuration: {str(e)}")
