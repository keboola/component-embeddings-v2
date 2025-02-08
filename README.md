# Keboola Embeddings Component V2

This component generates embeddings from text data using various embedding providers and can store them in vector databases.

## Component Initialization

The component was initialized using the Keboola Python Component template:

```bash
cookiecutter https://github.com/keboola/cookiecutter-python-component
```

With the following configuration:
- template_variant: GitHub
- repository_url: https://github.com/keboola/component-embeddings-v2
- component_name: keboola.app-embeddings-v2

## Features

- Multiple embedding providers support:
  - OpenAI
  - Cohere
  - HuggingFace
  - Azure OpenAI

- Vector database integrations:
  - Chroma
  - PostgreSQL (pgvector)
  - FAISS
  - Pinecone

- Advanced text processing:
  - Configurable batch processing
  - Text chunking with overlap
  - Metadata handling

## Configuration

### Embedding Provider Configuration

```json
{
  "embedding_provider": "openai",
  "model_name": "text-embedding-3-small",
  "provider_params": {
    "#api_key": "your-api-key"
  }
}
```

### Input Configuration

```json
{
  "input_table": {
    "input_table_id": "your-table-id",
    "text_column": "text",
    "id_column": "id"
  }
}
```

### Vector Database Configuration (Optional)

```json
{
  "vector_db": {
    "db_type": "pgvector",
    "connection_params": {
      "connection_string": "postgresql://user:pass@host:5432/db",
      "index_name": "embeddings"
    }
  }
}
```

## Development

1. Clone the repository:
```bash
git clone https://github.com/keboola/component-embeddings-v2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run tests:
```bash
python -m pytest
```

## License

MIT

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

To customize the local data folder path, replace the `CUSTOM_FOLDER` placeholder with your desired path in the `docker-compose.yml` file:

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
