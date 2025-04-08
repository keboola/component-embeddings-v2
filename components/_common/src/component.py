"""Main component file for the embedding service."""
import warnings

warnings.filterwarnings("ignore",
                        message="Field \"model_arn\" in BedrockRerank has conflict with protected namespace \"model_\"")
warnings.filterwarnings("ignore",
                        message="As the c extension couldn't be imported, `google-crc32c` is using a pure python implementation that is significantly slower")

import asyncio
import csv
import logging
import sys

from collections import OrderedDict

from keboola.component.base import ComponentBase, sync_action
from keboola.component.dao import ColumnDefinition, DataType
from keboola.component.exceptions import UserException
from keboola.component.sync_actions import ValidationResult, MessageType

from configuration import ComponentConfig
from services.csv_manager import CSVManager
from services.embedding_manager import EmbeddingManager
from services.vector_store_manager import VectorStoreManager

csv.field_size_limit(sys.maxsize)


class Component(ComponentBase):
    """Main component class."""

    def __init__(self):
        super().__init__()
        self.config = None
        self.embedding_manager = None
        self.vector_store_manager = None
        self.csv_manager = None

    def _initialize_services(self):
        """Initialize component services."""
        self.config = ComponentConfig.model_validate(self.configuration.parameters)
        self.embedding_manager = EmbeddingManager(self.config)
        self._test_embedding_service_connection(self.embedding_manager)

        if self.config.vector_db:
            self.vector_store_manager = VectorStoreManager(
                self.config,
                self.embedding_manager.embedding_model
            )
            self._test_vector_store_connection(self.vector_store_manager)
            logging.info("Vector store connection successful")

        logging.info("Services initialized")

    def _prepare_tables(self):
        """Prepare input and output tables."""
        tables = self.get_input_tables_definitions()
        if not tables:
            raise UserException("No input table specified!")
        if len(tables) > 1:
            raise UserException("Only one input table is supported")

        self.input_table_definition = tables[0]
        self.csv_manager = CSVManager()

        if self.config.output_config.save_to_storage:
            self._build_output_table()
            self.csv_manager.output_table_definition = self.output_table_definition

    def _build_output_table(self) -> None:
        """Build output table definition with proper schema."""
        dest_config = self.config.destination
        out_table_name = dest_config.output_table_name or f"embeddings-{self.environment_variables.config_row_id}"
        out_table_name = f"{out_table_name}.csv"

        # Create schema for output table
        output_schema = OrderedDict()

        # Add ID column if configured
        if self.config.id_column:
            if self.config.id_column in self.input_table_definition.schema:
                output_schema["id"] = self.input_table_definition.schema[self.config.id_column]

        # Add text column
        if self.config.text_column in self.input_table_definition.schema:
            output_schema["text"] = self.input_table_definition.schema[self.config.text_column]

        # Add metadata columns (sorted to match CSV order)
        for col_name in sorted(self.config.metadata_columns):
            if col_name in self.input_table_definition.schema:
                output_schema[col_name] = self.input_table_definition.schema[col_name]

        # Add embedding column with appropriate backend-specific type
        output_schema["embedding"] = ColumnDefinition(
            data_types={
                "base": DataType(dtype="STRING"),
                "snowflake": DataType(dtype="VECTOR", length="FLOAT,256"),
                "bigquery": DataType(dtype="STRING")
            },
            nullable=True,
            primary_key=False,
            description="Embedding vector",
            metadata=None
        )

        self.output_table_definition = self.create_out_table_definition(
            name=out_table_name,
            primary_key=dest_config.primary_keys_array,
            incremental=dest_config.incremental_load,
            schema=output_schema
        )

        if self.output_table_definition:
            self.write_manifest(self.output_table_definition)

    async def _process_batch(self, texts: list[str], metadata: list[dict]) \
            -> tuple[list[str], list[dict], list[list[float]]]:
        """Process a batch of texts.
        Args:
            texts: List of text content to embed
            metadata: List of metadata dictionaries (passed through unchanged)
        Returns:
            Tuple of (processed_texts, metadata, embeddings)
        """
        # Get embeddings
        embeddings = await self.embedding_manager.process_texts(texts)

        # If chunking is enabled, we need to duplicate metadata for each chunk
        if self.config.advanced_options.enable_chunking:
            # Create corresponding list of chunked texts and expanded metadata
            chunked_texts = []
            expanded_metadata = []

            # Process each text and its metadata
            for text, meta in zip(texts, metadata):
                chunks = self.embedding_manager.split_text(text)
                chunked_texts.extend(chunks)
                # Duplicate metadata for each chunk of this text
                expanded_metadata.extend([meta] * len(chunks))

            texts = chunked_texts
            metadata = expanded_metadata

        return texts, metadata, embeddings

    async def _save_results(
            self,
            texts: list[str],
            metadata: list[dict],
            embeddings: list[list[float]],
            ids: list[str] = None
    ) -> None:
        """Save results to file and/or vector database.
        Args:
            texts: List of text content
            metadata: List of metadata dictionaries
            embeddings: List of embedding vectors
            ids: Optional list of record IDs for upsert
        """
        if self.config.output_config.save_to_storage:
            logging.info(f"Saving {len(texts)} embeddings to CSV")
            self.csv_manager.save_embeddings_to_csv(texts, metadata, embeddings, ids)

        if self.config.output_config.save_to_vectordb and self.config.vector_db:
            logging.info(f"Storing {len(texts)} embeddings in vector database")
            if ids:
                logging.info("Using upsert operation with provided IDs")
                await self.vector_store_manager.upsert_vectors(texts, embeddings, metadata, ids)
            else:
                await self.vector_store_manager.store_vectors(texts, embeddings, metadata)
            logging.info(f"Total embeddings in vector database: {len(self.vector_store_manager.stored_ids)}")

    async def _process_all_data(self):
        """Process all input data in batches."""
        batch_size = self.config.advanced_options.batch_size
        logging.info(f"Processing data in batches of size {batch_size}")

        current_batch_texts = []
        current_batch_metadata = []
        current_batch_ids = []
        processed_count = 0

        text_generator = self.csv_manager.read_input_table(
            self.input_table_definition,
            self.config.text_column,
            self.config.metadata_columns,
            self.config.id_column
        )

        for text, metadata, record_id in text_generator:
            current_batch_texts.append(text)
            current_batch_metadata.append(metadata)
            if record_id:
                current_batch_ids.append(record_id)

            if len(current_batch_texts) >= batch_size:
                texts, metadata, embeddings = await self._process_batch(current_batch_texts, current_batch_metadata)
                batch_ids = current_batch_ids if current_batch_ids else None
                await self._save_results(texts, metadata, embeddings, batch_ids)
                processed_count += len(texts)
                logging.info(f"Processed {processed_count} texts")
                current_batch_texts, current_batch_metadata, current_batch_ids = [], [], []

        if current_batch_texts:
            texts, metadata, embeddings = await self._process_batch(current_batch_texts, current_batch_metadata)
            batch_ids = current_batch_ids if current_batch_ids else None
            await self._save_results(texts, metadata, embeddings, batch_ids)
            processed_count += len(texts)
            logging.info(f"Processed total of {processed_count} texts")

    @staticmethod
    def _test_vector_store_connection(vector_store_manager: VectorStoreManager) -> None:
        """Test connection to vector database."""
        try:
            vector_store_manager.vector_store.search("test", "similarity")
        except Exception as e:
            raise UserException(f"Failed to connect to vector database: {str(e)}")

    @sync_action("testVectorStoreConnection")
    def test_vector_store_connection(self) -> ValidationResult:
        """Sync action to test connection to vector database."""
        try:
            # Load config
            self.config = ComponentConfig.model_validate(self.configuration.parameters)

            # Check if vector db is configured
            if not self.config.vector_db:
                raise UserException("Vector database configuration is missing")

            # Try to initialize vector store
            vector_store = VectorStoreManager(self.config, None)

            vector_store._initialize_vector_store()

            return ValidationResult("Connection to vector database successful.", MessageType.SUCCESS)

        except Exception as e:
            raise UserException(f"Failed to connect to vector database: {str(e)}")

    @staticmethod
    def _test_embedding_service_connection(embedding_manager: EmbeddingManager) -> None:
        return embedding_manager.test_connection()

    @sync_action("testEmbeddingServiceConnection")
    def test_embedding_service_connection(self) -> ValidationResult:
        """Test connection to embedding service."""
        try:
            # Load config
            self.config = ComponentConfig.model_validate(self.configuration.parameters)

            # Check if vector db is configured
            if not self.config.embedding_settings:
                raise UserException("Embedding service configuration is missing")

            # Try to initialize vector store
            embedding_manager = EmbeddingManager(self.config)
            embedding_manager._initialize_embeddings()
            self._test_embedding_service_connection(embedding_manager)

            return ValidationResult("Embedding service connection successful.", MessageType.SUCCESS)

        except Exception as e:
            raise UserException(f"Failed to connect to embedding service: {str(e)}")

    async def _run_async(self):
        self._prepare_tables()
        await self._process_all_data()

    def run(self):
        """Main execution code."""
        self._initialize_services()
        asyncio.run(self._run_async())


if __name__ == "__main__":
    try:
        comp = Component()
        comp.execute_action()
    except UserException as exc:
        logging.exception(exc)
        exit(1)
    except Exception as exc:
        logging.exception(exc)
        exit(2)
