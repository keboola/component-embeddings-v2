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
        self.config = None
        self.embedding_manager = None
        self.vector_store_manager = None

    def _initialize_services(self):
        """Initialize component services."""
        self.config = ComponentConfig.model_validate(self.configuration.parameters)
        self.embedding_manager = EmbeddingManager(self.config)
        if self.config.vector_db:
            self.vector_store_manager = VectorStoreManager(
                self.config,
                self.embedding_manager.embedding_model
            )

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
    ) -> None:
        """Process all input data in batches."""
        batch_size = self.config.advanced_options.batch_size
        logging.info(f"Processing data in batches of size {batch_size}")

        current_batch = []
        processed_count = 0

        for text in text_generator:
            current_batch.append(text)

            if len(current_batch) >= batch_size:
                # Process current batch
                texts, embeddings = await self._process_batch(current_batch)
                await self._save_results(texts, embeddings)

                processed_count += len(texts)
                logging.info(f"Processed {processed_count} texts")

                current_batch = []

        # Process remaining texts
        if current_batch:
            texts, embeddings = await self._process_batch(current_batch)
            await self._save_results(texts, embeddings)
            processed_count += len(texts)
            logging.info(f"Processed total of {processed_count} texts")

    async def _save_results(
            self,
            texts: list[str],
            embeddings: list[list[float]]
    ) -> None:
        """Save results to file and/or vector database."""
        if self.config.output_config.output_type == "csv":
            logging.info(f"Saving {len(texts)} embeddings to CSV")
            output_path = os.path.join(self.tables_out_path, "embeddings.csv")
            save_embeddings_to_csv(
                output_path,
                texts,
                embeddings,
                False
            )

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
        self._initialize_services()
        asyncio.run(self._run_async())

    async def _run_async(self):
        text_generator = self._read_input_data()
        await self._process_all_data(text_generator)

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
