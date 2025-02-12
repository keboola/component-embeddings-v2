"""CSV handling service for the embedding component."""
import csv
import json
import os
from typing import Generator


class CSVManager:
    def __init__(self, output_table_definition=None):
        self.output_table_definition = output_table_definition
        self.EMBEDDING_RESULT_COLUMN_NAME = "embedding"

    @property
    def is_output_configured(self) -> bool:
        """Check if output is properly configured."""
        return self.output_table_definition is not None

    def save_embeddings_to_csv(
            self,
            texts: list[str],
            metadata: list[dict],
            embeddings: list[list[float]]
    ) -> None:
        """Save embeddings to CSV file."""
        if not self.is_output_configured:
            raise ValueError("Output table definition not set")

        if not (len(texts) == len(embeddings) == len(metadata)):
            raise ValueError("Length mismatch between texts, metadata and embeddings")

        # Get all unique metadata keys
        metadata_keys = set()
        for meta in metadata:
            metadata_keys.update(meta.keys())

        # Define CSV fields
        fields = ["text"] + sorted(list(metadata_keys)) + [self.EMBEDDING_RESULT_COLUMN_NAME]

        # Write to CSV
        file_exists = os.path.exists(self.output_table_definition.full_path)
        mode = "a" if file_exists else "w"

        with open(self.output_table_definition.full_path, mode, encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields)

            if not file_exists:
                writer.writeheader()

            for text, meta, embedding in zip(texts, metadata, embeddings):
                row = {"text": text, self.EMBEDDING_RESULT_COLUMN_NAME: json.dumps(embedding)}
                row.update(meta)
                writer.writerow(row)

    @staticmethod
    def read_input_table(input_table_definition, text_column: str, metadata_columns: list[str]) \
            -> Generator[tuple[str, dict], None, None]:
        """Read input table and yield texts with their metadata dictionary."""
        with open(input_table_definition.full_path, "r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            if text_column not in reader.fieldnames:
                raise ValueError(f"Text column '{text_column}' not found in input table")

            for col in metadata_columns:
                if col not in reader.fieldnames:
                    raise ValueError(f"Metadata column '{col}' not found in input table")

            for row in reader:
                metadata = {col: row[col] for col in metadata_columns} if metadata_columns else {}
                yield row[text_column], metadata
