# Pinecone Embeddings Writer Configuration

## Embedding Provider
Choose the service that will generate vector embeddings from your text data. The component supports:
- OpenAI
- Azure OpenAI
- Cohere
- HuggingFace Hub
- Google Vertex AI
- AWS Bedrock

## Vector Database Configuration
Configure your Pinecone connection:

```json
{
  "vector_db": {
    "pinecone_settings": {
      "#api_key": "your-pinecone-api-key",
      "environment": "your-environment",
      "index_name": "your-index-name"
    }
  }
}
```

## Input Processing
- **Text Column**: Column containing the text to embed
- **ID Column**: Optional column with unique identifiers
- **Metadata Columns**: Additional columns to store as metadata in Pinecone

## Advanced Options
- **Batch Size**: Number of records to process at once
- **Text Chunking**: Split long texts before embedding
- **Pinecone Namespace**: Logical namespace within the index
- **Upsert Mode**: How to handle existing records with the same ID

## Output Configuration
- **Save to Storage**: Whether to save embeddings to Keboola Storage
- **Save to Vector Database**: Whether to save to Pinecone (default: true)