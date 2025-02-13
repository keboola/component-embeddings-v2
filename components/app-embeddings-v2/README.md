# Keboola App Embeddings V2 - UI Component

This is the UI component of the Keboola Embeddings V2 project that provides a user interface for managing and configuring embedding operations.

## Features

- Configuration of embedding providers and their settings
- Management of vector database connections
- Text processing and chunking configuration
- Batch processing settings
- Visual feedback and validation of settings

## Supported Embedding Providers

- OpenAI (text-embedding-3-small/large, ada-002)
- Azure OpenAI
- Cohere
- HuggingFace Hub
- Google Vertex AI
- AWS Bedrock

## Development

### Prerequisites

- Python 3.12+
- Docker
- Docker Compose

### Local Setup

1. Clone the repository and navigate to the component directory:
   ```bash
   cd components/app-embeddings-v2
   ```

2. Build and run the development environment:
   ```bash
   docker-compose build
   docker-compose up dev
   ```

3. Run tests:
   ```bash
   docker-compose run --rm test
   ```

## Configuration Options

The UI provides configuration for:

### Embedding Provider Settings
- Provider selection
- API credentials
- Model selection
- Custom parameters

### Vector Database Configuration
- Database type selection
- Connection settings
- Table/collection configuration

### Processing Options
- Batch size
- Chunking settings
- Metadata handling
- Output configuration

## License

MIT Licensed. See LICENSE file for details.
