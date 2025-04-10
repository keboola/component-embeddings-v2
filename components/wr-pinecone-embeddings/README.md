# Keboola Writer - Pinecone Embeddings

This component is responsible for writing embeddings to Pinecone vector database. It enables efficient storage and similarity search of vector embeddings for AI applications.

## Features

- Direct integration with Pinecone vector database
- Efficient batch writing of embeddings
- Support for metadata storage
- Namespace organization
- Support for various embedding providers
- Configurable upsert behavior

## Prerequisites

- Pinecone account and API key
- Existing Pinecone index with appropriate dimension size
- Python 3.12+
- Docker for local development

## Configuration

### Pinecone Connection
```json
{
    "vector_db": {
        "db_type": "pinecone",
        "pinecone_settings": {
            "#api_key": "your-pinecone-api-key",
            "environment": "your-environment",
            "index_name": "your-index-name"
        }
    }
}
```

### Embedding Provider
```json
{
    "embedding_settings": {
        "provider_type": "openai",
        "openai_settings": {
            "model": "text-embedding-3-small",
            "#api_key": "your-openai-api-key"
        }
    }
}
```

### Input Configuration
```json
{
    "text_column": "text",
    "id_column": "id",
    "metadata_columns": ["category", "source", "date"]
}
```

## Development

1. Navigate to the component directory:
   ```bash
   cd components/wr-pinecone-embeddings
   ```

2. Build and run development environment:
   ```bash
   docker-compose build
   docker-compose up dev
   ```

3. Run tests:
   ```bash
   docker-compose run --rm test
   ```

## Input

The component expects input tables with:
- Text data to embed
- Optional unique identifiers
- Optional metadata columns

## Output

- Creates or updates vectors in Pinecone index
- Optionally saves embeddings back to Keboola Storage
- Returns operation statistics

## Advanced Features

### Namespaces
Use the `pinecone_namespace` parameter to organize vectors within logical collections in your index.

### Upsert Modes
- **Replace**: Completely replace existing vectors with the same ID
- **Update**: Only update metadata fields that are new or changed

### Text Chunking
Enables splitting long texts into smaller chunks for more effective embedding:
- Configure chunk size, overlap, and strategy
- Choose from character, word, sentence, or paragraph-based splitting

## Error Handling

- Connection error management
- Data validation
- Batch operation recovery
- Automatic retries with exponential backoff

## License

MIT Licensed. See LICENSE file for details.
