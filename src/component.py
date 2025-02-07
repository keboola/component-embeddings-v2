"""Main component file for the embedding service."""
import logging
import os
import asyncio
from itertools import islice

from keboola.component.base import ComponentBase, sync_action
from keboola.component.sync_actions import ValidationResult, MessageType
from keboola.component.exceptions import UserException

from configuration import ComponentConfig
from services import EmbeddingManager, VectorStoreManager
from utils import save_embeddings_to_csv, read_input_table


class Component(ComponentBase):
    """Main component class."""

    def __init__(self):
        super().__init__()
        self.config: ComponentConfig = None
        self.embedding_manager = None
        self.vector_store_manager = None

    def _initialize_services(self) -> None:
        """Initialize configuration and service managers."""
        self.config = ComponentConfig.model_validate(self.configuration.parameters)
        self.embedding_manager = EmbeddingManager(self.config)
        self._test_embedding_service_connection(self.embedding_manager)
        if self.config.output_config.save_to_vectordb:
            self.vector_store_manager = VectorStoreManager(
                self.config,
                self.embedding_manager.embedding_model
            )
        logging.info("Services initialized")

    @sync_action("testVectorStoreConnection")
    def test_vector_store_connection(self) -> ValidationResult:
        """Test connection to vector database."""
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

            return ValidationResult("Connection to embedding service successful.", MessageType.SUCCESS)

        except Exception as e:
            raise UserException(f"Failed to connect to embedding service: {str(e)}")

    def _read_input_data(self):
        """Read and validate input data."""
        tables = self._get_input_tables()

        input_table_path = tables.full_path
        if not os.path.exists(input_table_path):
            raise UserException(f"Input table not found: {input_table_path}")

        return read_input_table(
            input_table_path,
            self.config.text_column
        )

    async def _process_batch(
            self,
            texts: list[str]
    ) -> tuple[list[str], list[list[float]]]:
        """Process a batch of texts and return chunks with their embeddings."""
        return await self.embedding_manager.process_texts(texts)

    async def _process_all_data(
            self,
            text_generator
    ) -> tuple[list[str], list[list[float]]]:
        """Process all input data in batches."""
        all_texts = []
        all_embeddings = []
        batch_size = self.config.advanced_options.batch_size
        logging.info(f"Processing data in batches of size {batch_size}")
        while True:
            # Get next batch of texts
            batch_texts = list(islice(text_generator, batch_size))
            if not batch_texts:
                break

            # Process batch
            processed_texts, embeddings = await self._process_batch(batch_texts)

            all_texts.extend(processed_texts)
            all_embeddings.extend(embeddings)

            # Save intermediate results if needed
            if len(all_texts) >= 10000:  # Save every 10k records
                await self._save_results(all_texts, all_embeddings)
                all_texts = []
                all_embeddings = []

        return all_texts, all_embeddings

    async def _save_results(
            self,
            texts: list[str],
            embeddings: list[list[float]]
    ) -> None:
        """Save results to file and/or vector database."""
        # Save to CSV if configured
        # TODO use keboola create_out_table_definition or create_out_table_definition_from_schema
        # Add option to save in raw structure to zip
        if self.config.output_config.output_type == "csv":
            logging.info(f"Saving {len(texts)} embeddings to CSV")
            output_path = os.path.join(self.tables_out_path, "embeddings.csv")
            save_embeddings_to_csv(
                output_path,
                texts,
                embeddings,
                False  # zip_output is now controlled by output_type
            )

        # Store in vector database if configured
        if self.config.output_config.save_to_vectordb and self.config.vector_db:
            logging.info(f"Storing {len(texts)} embeddings in vector database")
            await self.vector_store_manager.store_vectors(
                texts,
                embeddings
            )
            logging.info(
                "Embeddings stored in vector database total: " + str(len(self.vector_store_manager.stored_ids)))

    def run(self):
        """Main execution code."""
        # Initialize services
        self._initialize_services()

        asyncio.run(self._run_async())

    async def _run_async(self):
        # Read input data
        text_generator = self._read_input_data()

        processed_texts, embeddings = await self._process_all_data(text_generator)

        if processed_texts:
            logging.info(f"Saving {len(processed_texts)} results")
            await self._save_results(processed_texts, embeddings)

    def _get_input_tables(self):
        if not self.get_input_tables_definitions():
            raise UserException("No input table specified. Please provide one input table in the input mapping!")

        if len(self.get_input_tables_definitions()) > 1:
            raise UserException("Only one input table is supported")

        return self.get_input_tables_definitions()[0]


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
