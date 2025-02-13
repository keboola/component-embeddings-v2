# Keboola Writer - PGVector Embeddings

This component is responsible for writing embeddings to PostgreSQL databases using the pgvector extension. It enables efficient storage and similarity search of vector embeddings in PostgreSQL.

## Features

- Direct integration with PostgreSQL + pgvector
- Efficient batch writing of embeddings
- Support for metadata storage
- Automatic index creation and optimization
- Configurable table schema

## Prerequisites

- PostgreSQL database with pgvector extension installed
- Database credentials with appropriate permissions
- Python 3.12+
- Docker for local development

## Configuration

### Database Connection
```json
{
    "db_connection": {
        "host": "your-host",
        "port": 5432,
        "database": "your-db",
        "#username": "user",
        "#password": "pass"
    }
}
```

### Table Configuration
```json
{
    "table_settings": {
        "table_name": "embeddings",
        "create_if_not_exists": true,
        "vector_dimension": 1536,
        "metadata_columns": ["id", "text", "metadata"]
    }
}
```

### Index Settings
```json
{
    "index_settings": {
        "create_index": true,
        "index_type": "ivfflat",
        "lists": 100,
        "probes": 10
    }
}
```

## Development

1. Navigate to the component directory:
   ```bash
   cd components/wr-pgvector-embeddings
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
- Vector embeddings in a specified format
- Optional metadata columns
- Proper dimension size matching the configuration

## Output

- Creates or updates tables in PostgreSQL
- Optionally creates optimized indexes
- Returns operation statistics

## Error Handling

- Connection error management
- Data validation
- Batch operation recovery
- Index creation monitoring

## License

MIT Licensed. See LICENSE file for details.
