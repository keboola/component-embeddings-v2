# Build stage
FROM python:3.12-slim-bullseye AS builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Install build tools
RUN pip install --no-cache-dir uv tree

# Copy only requirements first to leverage Docker cache
COPY components/_common/pyproject.toml components/_common/uv.lock /code/

# Install dependencies
WORKDIR /code
RUN uv pip install -r pyproject.toml --system --no-cache

# Runtime stage
FROM python:3.12-slim-bullseye AS runtime

# Install test tools
RUN pip install --no-cache-dir flake8 tree

# Copy only necessary files
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY components/_common/src /code/src/
COPY components/_common/tests /code/tests/
COPY scripts /code/scripts/
COPY components/_common/flake8.cfg /code/flake8.cfg
COPY components/_common/deploy.sh /code/deploy.sh
COPY components/_common/pyproject.toml /code/pyproject.toml
COPY components/_common/uv.lock /code/uv.lock

# Copy component specific configuration
COPY components/app-embeddings-v2/component_config /code/component_config/

WORKDIR /code/

CMD ["python", "-u", "/code/src/component.py"]