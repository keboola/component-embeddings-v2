"""Main component file for the embedding service."""
import logging
import sys
import csv
import asyncio

from keboola.component.base import ComponentBase, sync_action
from keboola.component.sync_actions import ValidationResult, MessageType
from keboola.component.exceptions import UserException

from configuration import ComponentConfig
from services import EmbeddingManager, VectorStoreManager
from services.csv_manager import CSVManager

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
            self._build_out_csv_table()
            self.csv_manager.output_table_definition = self.output_table_definition

    def _build_out_csv_table(self):
        """Configure output table."""
        dest_config = self.config.destination
        out_table_name = dest_config.output_table_name or f"embeddings-{self.environment_variables.config_row_id}"
        out_table_name = f"{out_table_name}.csv"

        self.output_table_definition = self.create_out_table_definition(
            out_table_name,
            primary_key=dest_config.primary_keys_array,
            incremental=dest_config.incremental_load
        )

        if self.output_table_definition:
            self.write_manifest(self.output_table_definition)

    async def _process_batch(self, texts: list[str], metadata: list[str]) \
            -> tuple[list[str], list[dict], list[list[float]]]:
        """Process a batch of texts."""
        return await self.embedding_manager.process_texts(texts, metadata)

    async def _save_results(self, texts: list[str], metadata: list[dict], embeddings: list[list[float]]) -> None:
        """Save results to file and/or vector database."""
        if self.config.output_config.save_to_storage:
            logging.info(f"Saving {len(texts)} embeddings to CSV")
            self.csv_manager.save_embeddings_to_csv(texts, metadata, embeddings)

        if self.config.output_config.save_to_vectordb and self.config.vector_db:
            logging.info(f"Storing {len(texts)} embeddings in vector database")
            await self.vector_store_manager.store_vectors(texts, embeddings, metadata)
            logging.info(f"Total embeddings in vector database: {len(self.vector_store_manager.stored_ids)}")

    async def _process_all_data(self):
        """Process all input data in batches."""
        batch_size = self.config.advanced_options.batch_size
        logging.info(f"Processing data in batches of size {batch_size}")

        current_batch_texts = []
        current_batch_metadata = []
        processed_count = 0

        text_generator = self.csv_manager.read_input_table(
            self.input_table_definition,
            self.config.text_column,
            self.config.metadata_columns
        )

        for text, metadata in text_generator:
            current_batch_texts.append(text)
            current_batch_metadata.append(metadata)

            if len(current_batch_texts) >= batch_size:
                texts, metadata, embeddings = await self._process_batch(current_batch_texts, current_batch_metadata)
                await self._save_results(texts, metadata, embeddings)
                processed_count += len(texts)
                logging.info(f"Processed {processed_count} texts")
                current_batch_texts, current_batch_metadata = [], []

        if current_batch_texts:
            texts, metadata, embeddings = await self._process_batch(current_batch_texts, current_batch_metadata)
            await self._save_results(texts, metadata, embeddings)
            processed_count += len(texts)
            logging.info(f"Processed total of {processed_count} texts")

    @staticmethod
    def _test_vector_store_connection(vector_store_manager: VectorStoreManager) -> None:
        """Test connection to vector database."""
        try:
            asyncio.run(vector_store_manager.vector_store.aget_by_ids([]))
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
