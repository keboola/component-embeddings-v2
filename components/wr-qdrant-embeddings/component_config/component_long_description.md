# Keboola Writer - Qdrant Embeddings

This component enables storage and retrieval of vector embeddings in Qdrant vector database. Qdrant is optimized for high-performance vector similarity search and filtering.

## Key Features

- Writes embeddings to Qdrant collections with configurable settings
- Supports both cloud-hosted and self-hosted Qdrant instances
- Configurable batch operations for efficient data loading
- Automatic collection creation with customizable schema
- Support for metadata storage alongside vector embeddings
- Document upsert capabilities when ID column is provided
- Compatible with all embedding providers supported by the App Embeddings V2

The writer integrates with the Keboola App Embeddings V2 UI component and is ideal for applications requiring fast vector search with filtering, including recommendation systems, semantic search, and clustering operations.