# Keboola Embeddings Component V2

A powerful component for generating and managing text embeddings in the Keboola Connection platform. This component
supports multiple embedding providers and vector databases, making it a versatile solution for various AI and machine
learning applications.

## Features

### Embedding Providers

- **OpenAI**
    - Latest models including text-embedding-3-small/large
    - Legacy ada-002 support
- **Azure OpenAI**
    - Full Azure OpenAI API support
    - Custom deployment configurations
- **Cohere**
    - English and multilingual models
    - Light and standard versions
- **HuggingFace Hub**
    - Support for custom models
    - Optimized for sentence-transformers
- **Google Vertex AI**
    - Native integration with Google Cloud
    - Support for latest Vertex AI models
- **AWS Bedrock**
    - Amazon Titan models
    - Cohere models via AWS

### Vector Database Support

- **PostgreSQL (pgvector)**
    - Native vector similarity search
    - Efficient indexing and querying
- **Pinecone**
    - Managed vector database service
    - High-performance similarity search
- **Qdrant**
    - Self-hosted or cloud options
    - Advanced filtering capabilities
- **Milvus**
    - Scalable vector database
    - Hybrid search support
- **Redis**
    - Vector similarity with Redis
    - High-performance in-memory operations
- **OpenSearch**
    - Full-text and vector search
    - AWS OpenSearch compatible

### Advanced Features

- **Batch Processing**
    - Configurable batch sizes (1-1000)
    - Optimized for performance
- **Text Chunking**
    - Multiple chunking strategies
    - Configurable overlap
    - Support for long documents
- **Metadata Handling**
    - Custom metadata storage
    - Rich query capabilities
- **Storage Options**
    - Keboola Storage integration
    - Vector database storage
    - Dual storage support

## Usage

### Basic Configuration

1. **Select Embedding Provider**
   ```json
   {
     "embedding_settings": {
       "provider_type": "openai",
       "openai_settings": {
         "model": "text-embedding-3-small",
         "#api_key": "your-api-key"
       }
     }
   }
   ```

2. **Configure Input**
   ```json
   {
     "text_column": "description",
     "metadata_column": "id"
   }
   ```

3. **Set Output Options**
   ```json
   {
     "output_config": {
       "save_to_storage": true,
       "save_to_vectordb": true
     }
   }
   ```

### Vector Database Setup

Example for PostgreSQL (pgvector):

```json
{
    "vector_db": {
        "db_type": "pgvector",
        "pgvector_settings": {
            "host": "your-host",
            "port": 5432,
            "database": "your-db",
            "#username": "user",
            "#password": "pass",
            "table_name": "embeddings"
        }
    }
}
```

### Advanced Configuration

Enable text chunking:

```json
{
    "advanced_options": {
        "batch_size": 100,
        "enable_chunking": true,
        "chunking_settings": {
            "chunk_size": 1000,
            "chunk_overlap": 100,
            "chunk_strategy": "paragraph"
        }
    }
}
```

## Development

### Prerequisites

- Python 3.13+
- Docker
- Access to embedding service APIs
- Vector database instance (if using)

### Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/keboola/component-embeddings-v2
   cd keboola.app_embeddings_v2
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run tests:
   ```bash
   python -m pytest
   ```

### Docker Development

Build and run the component:

```bash
docker-compose build
docker-compose run --rm dev
```

Run tests in Docker:

```bash
docker-compose run --rm test
```

## Output

The component produces two types of outputs:

1. **Storage Tables** (if enabled)
    - Text content
    - Metadata
    - Embedding vectors

2. **Vector Database Records**
    - Embeddings with metadata
    - Queryable via vector similarity

## Error Handling

The component includes robust error handling for:

- API rate limits
- Connection issues
- Invalid configurations
- Data format problems

## Performance Considerations

- Use appropriate batch sizes for your use case
- Enable chunking for long texts
- Consider vector database performance characteristics
- Monitor API usage and costs

## License

MIT Licensed. See LICENSE file for details.

**Table of Contents:**

[TOC]

Functionality Notes
===================

Prerequisites
=============

Ensure you have the necessary API token, register the application, etc.

Features
========

| **Feature**             | **Description**                               |
|-------------------------|-----------------------------------------------|
| Generic UI Form         | Dynamic UI form for easy configuration.       |
| Row-Based Configuration | Allows structuring the configuration in rows. |
| OAuth                   | OAuth authentication enabled.                 |
| Incremental Loading     | Fetch data in new increments.                 |
| Backfill Mode           | Supports seamless backfill setup.             |
| Date Range Filter       | Specify the date range for data retrieval.    |

Supported Endpoints
===================

If you need additional endpoints, please submit your request to
[ideas.keboola.com](https://ideas.keboola.com/).

Configuration
=============

Param 1
-------
Details about parameter 1.

Param 2
-------
Details about parameter 2.

Output
======

Provides a list of tables, foreign keys, and schema.

Development
-----------

To customize the local data folder path, replace the `CUSTOM_FOLDER` placeholder with your desired path in the
`docker-compose.yml` file:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    volumes:
      - ./:/code
      - ./CUSTOM_FOLDER:/data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clone this repository, initialize the workspace, and run the component using the following
commands:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
git clone https://github.com/keboola/component-embeddings-v2 keboola.app_embeddings_v2
cd keboola.app_embeddings_v2
docker-compose build
docker-compose run --rm dev
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the test suite and perform lint checks using this command:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
docker-compose run --rm test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Integration
===========

For details about deployment and integration with Keboola, refer to the
[deployment section of the developer
documentation](https://developers.keboola.com/extend/component/deployment/).
