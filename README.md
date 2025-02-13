# Keboola App Embeddings V2

This monorepo contains multiple components that share common code base for embeddings functionality.

## Repository Structure

```
keboola.app_embeddings_v2/
├── .github/                    # GitHub configuration
│   └── workflows/             # CI/CD workflow definitions
│       ├── common-component-workflow.yml  # Shared workflow steps
│       ├── app-embeddings-v2.yml         # UI app specific workflow
│       ├── wr-pgvector-embeddings.yml    # PGVector writer workflow
│       └── wr-qdrant-embeddings.yml      # Qdrant writer workflow
├── components/                # All components
│   ├── _common/              # Shared code and configuration
│   │   ├── src/             # Common source code
│   │   │   ├── db/         # Database interfaces
│   │   │   ├── embeddings/ # Embedding providers
│   │   │   └── utils/     # Shared utilities
│   │   └── tests/          # Common test utilities
│   ├── app-embeddings-v2/   # UI Application component
│   ├── wr-pgvector-embeddings/ # PostgreSQL writer
│   └── wr-qdrant-embeddings/   # Qdrant writer
├── .gitignore                # Git ignore rules
└── LICENSE.md                # Project license
```

## Local Development

### Prerequisites

- Docker
- Docker Compose
- Python 3.12+
- Make (optional)


### Development Notes

- Common code is located in `components/_common/` and is shared between all components
- Each component has its own:
    - `component_config/` - Component specific configuration
    - `Dockerfile` - Component build definition
    - `docker-compose.yml` - Local development setup
    - `Version` - Component version for CI/CD


### Adding a New Component

The easiest way is to copy an existing similar component (e.g., `wr-pgvector-embeddings`) and modify it:

1. **Copy the Component**
   ```bash
   cd components
   cp -r wr-pgvector-embeddings your-component-name
   cd your-component-name
   ```

2. **Modify Files**
   
   Required changes in files:

   - `VERSION` - change to `0.0.1`
   
   - `docker-compose.yml` - update service names if using specific ones (e.g., if you have `pgvector-dev`, change to `your-component-dev`)
   
   - `component_config/` - update configuration files:
     - `configSchema.json` - main component configuration schema
     - `configRowSchema.json` - row configuration schema (if component supports row configuration)
     - `configuration_description.md` - configuration documentation
     - `component_short_description.md` - brief component description
     - `component_long_description.md` - detailed component description

3. **Add GitHub Workflow**
   
   Add new workflow file `.github/workflows/your-component-name.yml` - copy from another component and update the component name.

## CI/CD Pipeline Structure

The CI/CD pipeline is organized as follows:

1. **Common Workflow (`common-component-workflow.yml`)**
   - Shared steps for all components
   - Handles building, testing, and deployment
   - Manages version bumping and releases

2. **Component-Specific Workflows**
   - Triggered on changes to component files
   - Use the common workflow with component-specific parameters
   - Handle component-specific deployment needs

### CI/CD Process

1. **On Pull Request:**
   - Builds component
   - Runs tests
   - Checks code style
   - Validates configuration

2. **On Merge to Main:**
   - Builds component
   - Runs tests
   - Creates new version
   - Deploys to staging
   - Runs integration tests
   - Deploys to production

### Environment Variables

Required environment variables for CI/CD:
```
KBC_DEVELOPERPORTAL_USERNAME=your-username
KBC_DEVELOPERPORTAL_PASSWORD=your-password
KBC_DEVELOPERPORTAL_VENDOR=ypur-vendor
```

## License

MIT Licensed. See LICENSE file for details. 