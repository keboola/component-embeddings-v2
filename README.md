# Keboola App Embeddings V2

This monorepo contains multiple components that share common code base for embeddings functionality.

## Repository Structure

```
keboola.app_embeddings_v2/
├── components/              # All components
│   ├── _common/            # Shared code and configuration
│   │   ├── src/           # Common source code
│   │   ├── tests/        # Common tests
│   │   └── ...
│   ├── app-embeddings-v2/  # UI Application component
│   └── wr-pgvector-embeddings/  # Writer component
├── scripts/                # Shared scripts
└── .github/                # GitHub workflows
```

## Local Development

### Prerequisites

- Docker
- Docker Compose
- Python 3.12+
- Make (optional)

### Setting Up Development Environment

1. Clone the repository:

```bash
git clone git@github.com:keboola/app-embeddings-v2.git
cd keboola.app_embeddings_v2
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

### Building and Testing Components

Each component can be built and tested independently. Navigate to the component directory and use docker-compose
commands.

#### Writer Component (wr-pgvector-embeddings)

```bash
cd components/wr-pgvector-embeddings

# Build the component
docker-compose build

# Run development environment
docker-compose up dev

# Run tests
docker-compose run --rm test
```

#### UI Application (app-embeddings-v2)

```bash
cd components/app-embeddings-v2

# Build the component
docker-compose build

# Run development environment
docker-compose up dev

# Run tests
docker-compose run --rm test
```

### Development Notes

- Common code is located in `components/_common/` and is shared between all components
- Each component has its own:
    - `component_config/` - Component specific configuration
    - `Dockerfile` - Component build definition
    - `docker-compose.yml` - Local development setup
    - `Version` - Component version for CI/CD

### Testing

Each component can be tested independently:

1. Unit tests:
```bash
cd components/<component-name>
docker-compose run --rm test
```

2. Local development:

```bash
cd components/<component-name>
docker-compose up dev
```

The test environment mounts:

- Component specific configuration from `./component_config`
- Common source code from `../_common/src`
- Common tests from `../_common/tests` (for test service)

### Adding a New Component

1. Create a new directory in `components/`
2. Copy the basic structure:
    - `component_config/` - Component specific configuration
    - `Dockerfile` - Based on common template
    - `docker-compose.yml` - Based on common template
    - `Version` - Start with version 1.0.0

3. Update paths in Dockerfile and docker-compose.yml to use common code from `../_common` 