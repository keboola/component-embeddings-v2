# Keboola PGVector Embeddings Writer

This document describes how to configure the Keboola PGVector Embeddings Writer component.

## Main Configuration

The component configuration consists of two main parts:

## Embedding Settings

### Provider Selection

Choose from the following embedding providers:

- OpenAI
- Azure OpenAI
- Cohere
- HuggingFace Hub
- Google Vertex AI
- AWS Bedrock

## Row Configuration

### Input Configuration

- **Text Column**: Column containing text to embed
- **Metadata Columns**: Optional columns to store as metadata
- **Id Column**: Optional columns to store as ID - **if it set the index will be updated**
- 
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