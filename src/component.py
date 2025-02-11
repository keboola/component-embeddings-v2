"""Main component file for the embedding service."""
import csv
import json
import logging
import sys
from typing import Generator, Any
import asyncio
import os

from keboola.component.base import ComponentBase, sync_action
from keboola.component.sync_actions import ValidationResult, MessageType
from keboola.component.exceptions import UserException

from configuration import ComponentConfig
from services import EmbeddingManager, VectorStoreManager

EMBEDDING_RESULT_COLUMN_NAME = "embedding"
CSV_TABLES_FIELDS = ["text", "metadata", EMBEDDING_RESULT_COLUMN_NAME]

csv.field_size_limit(sys.maxsize)


class Component(ComponentBase):
    """Main component class."""

    def __init__(self):
        super().__init__()
        self.config = None
        self.embedding_manager = None
        self.vector_store_manager = None

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
            # TODO PineconeVectorStore does not yet support get_by_ids.
            # self._test_vector_store_connection(self.vector_store_manager)
            logging.info("Vector store connection successful")
        logging.info("Services initialized")

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

    def _prepare_tables(self):
        if not self.get_input_tables_definitions():
            raise UserException("No input table specified. Please provide one input table in the input mapping!")

        if len(self.get_input_tables_definitions()) > 1:
            raise UserException("Only one input table is supported")

        self.input_table_definition = self.get_input_tables_definitions()[0]

        if self.config.output_config.save_to_storage:
            self._build_out_csv_table()

    def _build_out_csv_table(self):
        destination_config = self.config.destination

        if not (out_table_name := destination_config.output_table_name):
            out_table_name = f"app-embeddings-{self.environment_variables.config_row_id}.csv"
        else:
            out_table_name = f"{out_table_name}.csv"

        primary_key = destination_config.primary_keys_array

        incremental_load = destination_config.incremental_load
        self.output_table_definition = self.create_out_table_definition(out_table_name,
                                                                        primary_key=primary_key,
                                                                        incremental=incremental_load)

    def _save_embeddings_to_csv(
            self,
            texts: list[str],
            metadata: list[str],
            embeddings: list[list[float]]
    ) -> None:
        """Save embeddings to CSV file."""
        if not (len(texts) == len(embeddings) == len(metadata)):
            raise ValueError("Length mismatch between texts, metadata and embeddings")

        # Write to CSV - append mode if file exists, write mode with header if not
        file_exists = os.path.exists(self.output_table_definition.full_path)
        mode = "a" if file_exists else "w"

        with open(self.output_table_definition.full_path, mode, encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_TABLES_FIELDS)

            # Write header only for new file
            if not file_exists:
                writer.writeheader()
                self.write_manifest(self.output_table_definition)

            # Write rows
            for text, meta, embedding in zip(texts, metadata, embeddings):
                row = {
                    "text": text,
                    "metadata": meta,
                    "embedding": json.dumps(embedding)
                }
                writer.writerow(row)

    async def _process_batch(
            self,
            texts: list[str],
            metadata: list[str]
    ) -> tuple[list[str], list[str], list[list[float]]]:
        """Process a batch of texts and return chunks with their embeddings."""
        return await self.embedding_manager.process_texts(texts, metadata)

    async def _process_all_data(
            self,
            text_generator
    ) -> None:
        """Process all input data in batches."""
        batch_size = self.config.advanced_options.batch_size
        logging.info(f"Processing data in batches of size {batch_size}")

        current_batch_texts = []
        current_batch_metadata = []
        processed_count = 0

        for text, metadata in text_generator:
            current_batch_texts.append(text)
            current_batch_metadata.append(metadata)

            if len(current_batch_texts) >= batch_size:
                # Process current batch
                texts, metadata, embeddings = await self._process_batch(current_batch_texts, current_batch_metadata)
                await self._save_results(texts, metadata, embeddings)

                processed_count += len(texts)
                logging.info(f"Processed {processed_count} texts")

                current_batch_texts = []
                current_batch_metadata = []

        # Process remaining texts
        if current_batch_texts:
            texts, metadata, embeddings = await self._process_batch(current_batch_texts, current_batch_metadata)
            await self._save_results(texts, metadata, embeddings)
            processed_count += len(texts)
            logging.info(f"Processed total of {processed_count} texts")

    async def _save_results(
            self,
            texts: list[str],
            metadata: list[str],
            embeddings: list[list[float]]
    ) -> None:
        """Save results to file and/or vector database."""
        # Add option to save in raw structure to zip
        if self.config.output_config.save_to_storage:
            logging.info(f"Saving {len(texts)} embeddings to CSV")
            self._save_embeddings_to_csv(
                texts,
                metadata,
                embeddings
            )

        if self.config.output_config.save_to_vectordb and self.config.vector_db:
            logging.info(f"Storing {len(texts)} embeddings in vector database")
            await self.vector_store_manager.store_vectors(
                texts,
                embeddings,
                metadata
            )
            logging.info(
                "Embeddings stored in vector database total: " + str(len(self.vector_store_manager.stored_ids)))

    def _read_input_table(self) -> Generator[tuple[str | Any, str | Any], None, None]:
        """Read input table and yield texts one by one."""

        text_column = self.config.text_column
        metadata_column = self.config.metadata_column
        with open(self.input_table_definition.full_path, "r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            if text_column not in reader.fieldnames:
                raise ValueError(f"Text column '{text_column}' not found in input table")
            if metadata_column not in reader.fieldnames:
                raise ValueError(f"Metadata column '{metadata_column}' not found in input table")

            for row in reader:
                yield row[text_column], row[metadata_column]

    async def _run_async(self):
        self._prepare_tables()

        text_generator = self._read_input_table()
        # TODO we need more metadata columns, dynamic number of columns
        await self._process_all_data(text_generator)

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
