# Keboola Writer - PGVector Embeddings

This component enables efficient storage and similarity search of vector embeddings in PostgreSQL databases using the pgvector extension. It handles the integration between Keboola data sources and PostgreSQL vector databases.

## Key Features

- Writes embeddings directly to PostgreSQL databases with pgvector extension
- Configurable batch operations for efficient data loading
- Automatic table creation with customizable schema
- Support for metadata storage alongside vector embeddings
- Index creation and optimization (ivfflat, hnsw) for fast similarity search
- Compatible with all embedding providers supported by the App Embeddings V2

The writer works seamlessly with the Keboola App Embeddings V2 UI component and can be used in data workflows requiring semantic search, clustering, or other vector-based operations in PostgreSQL environments.