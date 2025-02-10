"""Service for handling embeddings through Langchain."""
import asyncio
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias

from tenacity import retry, stop_after_attempt, wait_exponential
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_cohere import CohereEmbeddings
from cohere import ClientV2 as CohereClientV2, AsyncClientV2 as AsyncCohereClientV2
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_aws import BedrockEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings
from boto3 import client as boto3_client
from keboola.component.exceptions import UserException

sys.path.append(str(Path(__file__).parent.parent))

from configuration import ComponentConfig  # noqa

# Type aliases
TextChunk: TypeAlias = str
Embedding: TypeAlias = list[float]
ChunkList: TypeAlias = list[TextChunk]
EmbeddingList: TypeAlias = list[Embedding]


class EmbeddingManager:
    """Manager for handling different embedding providers through Langchain."""

    def __init__(self, config: ComponentConfig) -> None:
        self.config = config
        self.embedding_model = self._initialize_embeddings()
        self.text_splitter = self._initialize_text_splitter() if getattr(config.advanced_options, "enable_chunking",
                                                                         False) else None
        self.semaphore = asyncio.Semaphore(5)  # Limit concurrent connections

    def _initialize_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Initialize text splitter based on configuration."""
        chunk_settings = self.config.advanced_options.chunking_settings
        strategy = chunk_settings.chunk_strategy

        separators = []
        match strategy:
            case "paragraph":
                separators = ["\n\n", "\n", ". ", ", ", " ", ""]
            case "sentence":
                separators = [". ", "! ", "? ", "\n", ", ", " ", ""]
            case "word":
                separators = [" ", "\n", "", "-"]
            case _:  # character
                separators = [""]

        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_settings.chunk_size,
            chunk_overlap=chunk_settings.chunk_overlap,
            length_function=len,
            separators=separators,
            keep_separator=True,
            add_start_index=True,
            strip_whitespace=True
        )

    def _initialize_embeddings(self):
        """Initialize the embedding model based on configuration."""
        settings = self.config.embedding_settings

        match settings.provider_type:
            case "openai":
                openai_settings = settings.openai_settings
                return OpenAIEmbeddings(
                    model=openai_settings.model,
                    api_key=openai_settings.api_key,
                    show_progress_bar=False,
                    retry_min_seconds=4,
                    retry_max_seconds=10,
                    max_retries=3
                )
            case "azure_openai":
                azure_settings = settings.azure_settings
                return AzureOpenAIEmbeddings(
                    model=azure_settings.deployment_name,
                    openai_api_type="azure",
                    api_key=azure_settings.api_key,
                    azure_endpoint=azure_settings.azure_endpoint,
                    api_version=azure_settings.api_version
                )
            case "cohere":
                cohere_settings = settings.cohere_settings
                cohere_client = CohereClientV2(
                    api_key=cohere_settings.api_key
                )
                cohere_async_client = AsyncCohereClientV2(
                    api_key=cohere_settings.api_key
                )
                return CohereEmbeddings(
                    cohere_api_key=cohere_settings.api_key,
                    model=cohere_settings.model,
                    max_retries=3,
                    client=cohere_client,
                    async_client=cohere_async_client
                )
            case "huggingface_hub":
                hf_settings = settings.huggingface_settings
                return HuggingFaceEmbeddings(
                    model_name=hf_settings.model,
                    cache_folder=None,  # Use default cache
                    encode_kwargs={
                        "normalize_embeddings": hf_settings.normalize_embeddings,
                        "show_progress_bar": hf_settings.show_progress
                    }
                )
            case "google_vertex":
                vertex_settings = settings.google_vertex_settings
                return VertexAIEmbeddings(
                    project_id=vertex_settings.project,
                    location=vertex_settings.location,
                    model_name=vertex_settings.model_name,
                    credentials=vertex_settings.credentials
                )
            case "bedrock":
                bedrock_settings = settings.bedrock_settings
                bedrock_client = boto3_client(
                    service_name="bedrock",
                    aws_access_key_id=bedrock_settings.aws_access_key,
                    aws_secret_access_key=bedrock_settings.aws_secret_key
                )
                return BedrockEmbeddings(
                    model_id=bedrock_settings.model_id,
                    region_name=bedrock_settings.region,
                    client=bedrock_client
                )
            case _:
                raise UserException(f"Unsupported embedding provider: {settings.provider_type}")

    def _split_text(self, text: str) -> ChunkList:
        """Split text into chunks."""
        if not self.text_splitter:
            return [text]
        try:
            return self.text_splitter.split_text(text)
        except Exception as e:
            raise UserException(f"Failed to split text: {str(e)}")

    def test_connection(self):
        """Test connection to the embedding service."""
        resp = None
        try:
            resp = self.embedding_model.embed_query("")
            logging.info("Connection to embedding service successful")
        except Exception as e:
            raise UserException(f"Failed to connect to embedding service: {str(e)} response: {resp}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _process_batch_async(self, batch: list[str]) -> EmbeddingList:
        """Process a batch of texts to embeddings asynchronously."""
        try:
            async with self.semaphore:
                if hasattr(self.embedding_model, "aembed_documents"):
                    return await self.embedding_model.aembed_documents(batch)
                else:
                    return await asyncio.to_thread(
                        self.embedding_model.embed_documents,
                        batch
                    )
        except Exception as e:
            raise UserException(f"Failed to create embeddings: {str(e)}")

    async def process_texts(self, texts: Sequence[str]) -> tuple[ChunkList, EmbeddingList]:
        """Process texts to create embeddings asynchronously."""
        try:
            # Split texts into chunks
            all_chunks = []
            for text in texts:
                chunks = self._split_text(text)
                all_chunks.extend(chunks)

            # Process chunks
            embeddings = await self._process_batch_async(all_chunks)
            return all_chunks, embeddings
            
        except Exception as e:
            raise UserException(f"Failed to process texts: {str(e)}")
