"""Utility functions for the embedding service component."""
import csv
import json
import os
import zipfile
from typing import Generator


def read_input_table(
        input_table_path: str,
        text_column: str
) -> Generator[str, None, None]:
    """Read input table and yield texts one by one."""
    with open(input_table_path, "r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if text_column not in reader.fieldnames:
            raise ValueError(f"Text column '{text_column}' not found in input table")

        for row in reader:
            yield row[text_column]


def save_embeddings_to_csv(
        output_path: str,
        texts: list[str],
        embeddings: list[list[float]],
        zip_output: bool = False
) -> None:
    """Save embeddings to CSV file."""
    if len(texts) != len(embeddings):
        raise ValueError("Length mismatch between texts and embeddings")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write to CSV
    with open(output_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["text", "embedding"])
        writer.writeheader()

        # Write rows
        for text, embedding in zip(texts, embeddings):
            row = {
                "text": text,
                "embedding": json.dumps(embedding)
            }
            writer.writerow(row)

    # Zip the output if requested
    if zip_output:
        zip_path = output_path + ".zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.write(output_path, os.path.basename(output_path))
        os.remove(output_path)  # Remove the original CSV file
