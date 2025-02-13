"""CSV handling service for the embedding component."""
import csv
import json
import os
from collections import OrderedDict
from typing import Generator

from keboola.component.dao import ColumnDefinition, DataType


class CSVManager:
    def __init__(self, output_table_definition=None):
        self.output_table_definition = output_table_definition
        self.EMBEDDING_RESULT_COLUMN_NAME = "embedding"

    @property
    def is_output_configured(self) -> bool:
        """Check if output is properly configured."""
        return self.output_table_definition is not None

    def build_output_table(self,
                           input_table_definition,
                           config,
                           environment_variables) -> None:
        """Build output table definition with proper schema.
        Args:
            input_table_definition: Input table definition with schema
            config: Component configuration
            environment_variables: Environment variables for table name generation
        """
        dest_config = config.destination
        out_table_name = dest_config.output_table_name or f"embeddings-{environment_variables.config_row_id}"
        out_table_name = f"{out_table_name}.csv"

        # Create schema for output table
        output_schema = OrderedDict()

        # Add ID column if configured
        if config.id_column:
            if config.id_column in input_table_definition.schema:
                output_schema["id"] = input_table_definition.schema[config.id_column]

        # Add text column
        if config.text_column in input_table_definition.schema:
            output_schema["text"] = input_table_definition.schema[config.text_column]

        # Add metadata columns (sorted to match CSV order)
        for col_name in sorted(config.metadata_columns):
            if col_name in input_table_definition.schema:
                output_schema[col_name] = input_table_definition.schema[col_name]

        # Add embedding column with appropriate backend-specific type
        output_schema["embedding"] = ColumnDefinition(
            data_types={
                "base": DataType(dtype="STRING"),
                "snowflake": DataType(dtype="VECTOR(FLOAT, 256)"),
                "bigquery": DataType(dtype="ARRAY<FLOAT64>")
            },
            nullable=True,
            primary_key=False,
            description="Embedding vector",
            metadata=None
        )

        self.output_table_definition = input_table_definition.create_out_table_definition(
            out_table_name,
            primary_key=dest_config.primary_keys_array,
            incremental=dest_config.incremental_load,
            schema=output_schema
        )

        if self.output_table_definition:
            input_table_definition.write_manifest(self.output_table_definition)

    def save_embeddings_to_csv(
            self,
            texts: list[str],
            metadata: list[dict],
            embeddings: list[list[float]],
            ids: list[str] = None
    ) -> None:
        """Save embeddings to CSV file.
        Args:
            texts: List of text content
            metadata: List of metadata dictionaries
            embeddings: List of embedding vectors
            ids: Optional list of record IDs
        """
        if not self.is_output_configured:
            raise ValueError("Output table definition not set")

        if not (len(texts) == len(embeddings) == len(metadata)):
            raise ValueError("Length mismatch between texts, metadata and embeddings")

        if ids and len(ids) != len(texts):
            raise ValueError("Length mismatch between texts and IDs")

        # Get all unique metadata keys
        metadata_keys = set()
        for meta in metadata:
            metadata_keys.update(meta.keys())

        # Define CSV fields - add id as first column if present
        fields = (["id"] if ids else []) + ["text"] + sorted(list(metadata_keys)) + [self.EMBEDDING_RESULT_COLUMN_NAME]

        # Write to CSV
        file_exists = os.path.exists(self.output_table_definition.full_path)
        mode = "a" if file_exists else "w"

        with open(self.output_table_definition.full_path, mode, encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields)

            if not file_exists:
                writer.writeheader()

            for idx, (text, meta, embedding) in enumerate(zip(texts, metadata, embeddings)):
                row = {"text": text, self.EMBEDDING_RESULT_COLUMN_NAME: json.dumps(embedding)}
                if ids:
                    row["id"] = ids[idx]
                row.update(meta)
                writer.writerow(row)

    @staticmethod
    def read_input_table(
            input_table_definition,
            text_column: str,
            metadata_columns: list[str],
            id_column: str = None
    ) -> Generator[tuple[str, dict, str | None], None, None]:
        """Read input table and yield texts with their metadata dictionary and optional ID.
        Args:
            input_table_definition: Input table definition
            text_column: Name of the column containing text to embed
            metadata_columns: List of column names to include as metadata
            id_column: Optional name of the column containing unique identifiers
        Returns:
            Generator yielding tuples of (text, metadata_dict, id)
        """
        with open(input_table_definition.full_path, "r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            if text_column not in reader.fieldnames:
                raise ValueError(f"Text column '{text_column}' not found in input table")

            for col in metadata_columns:
                if col not in reader.fieldnames:
                    raise ValueError(f"Metadata column '{col}' not found in input table")

            if id_column and id_column not in reader.fieldnames:
                raise ValueError(f"ID column '{id_column}' not found in input table")

            for row in reader:
                metadata = {col: row[col] for col in metadata_columns} if metadata_columns else {}
                record_id = row[id_column] if id_column else None
                yield row[text_column], metadata, record_id
