# Keboola PGVector Embeddings Writer

This document describes how to configure the Keboola PGVector Embeddings Writer component.

## Main Configuration

The component configuration consists of two main parts:

1. Component configuration (`configSchema.json`)
2. Row configuration (`configRowSchema.json`)

## Embedding Settings

### Provider Selection

Choose from the following embedding providers:

- OpenAI
- Azure OpenAI
- Cohere
- HuggingFace Hub
- Google Vertex AI
- AWS Bedrock

### Provider-Specific Settings

#### OpenAI

- **Model**: Choose from:
    - text-embedding-3-small (recommended)
    - text-embedding-3-large
    - text-embedding-ada-002
- **API Key**: Your OpenAI API key

#### Azure OpenAI

- **Deployment Name**: Your Azure OpenAI deployment name
- **API Key**: Your Azure OpenAI API key
- **Azure Endpoint**: Your Azure OpenAI endpoint URL
- **API Version**: API version (default: 2024-02-01)

#### Cohere

- **Model**: Choose from:
    - embed-english-v3.0
    - embed-english-light-v3.0
    - embed-multilingual-v3.0
    - embed-multilingual-light-v3.0
- **API Key**: Your Cohere API key

#### HuggingFace Hub

- **Model Name**: Model name (default: sentence-transformers/all-mpnet-base-v2)
- **API Key**: Your HuggingFace API key
- **Normalize Embeddings**: Whether to normalize embeddings (default: true)
- **Show Progress**: Show progress during embedding generation

#### Google Vertex AI

- **Service Account JSON**: Google Cloud service account credentials
- **Project ID**: Google Cloud project ID
- **Location**: Google Cloud region (default: us-central1)
- **Model Name**: Vertex AI model name (default: textembedding-gecko@latest)

#### AWS Bedrock

- **AWS Access Key**: Your AWS access key
- **AWS Secret Key**: Your AWS secret key
- **Region**: AWS region where Bedrock is available
- **Model ID**: Choose from:
    - amazon.titan-embed-text-v1
    - amazon.titan-embed-g1-text-02
    - cohere.embed-english-v3
    - cohere.embed-multilingual-v3

## Row Configuration

### Input Configuration

- **Text Column**: Column containing text to embed
- **Metadata Columns**: Optional columns to store as metadata

### Output Configuration

- **Save to Storage**: Save embeddings to Keboola Storage (as CSV) also

### Storage Destination (if saving to storage)

- **Output Table Name**: Name of the output table
- **Incremental Load**: Update existing table instead of rewriting
- **Primary Keys**: Define primary keys for the table

### Vector Database Configuration

**PostgreSQL (pgvector)**

- Host, Port, Database
- Username, Password
- Table Name

### Advanced Options

- **Batch Size**: Number of texts to process in one batch (1-1000)
- **Enable Text Chunking**: Split long texts into smaller chunks
    - Chunk Size: Maximum characters per chunk
    - Chunk Overlap: Characters to overlap between chunks
    - Chunking Strategy: character/sentence/word/paragraph