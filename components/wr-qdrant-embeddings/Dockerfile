FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install build tools
RUN pip install --no-cache-dir flake8

# Copy common files first
COPY components/_common/pyproject.toml components/_common/uv.lock /code/
COPY components/_common/src /code/src/
COPY components/_common/flake8.cfg /code/flake8.cfg
COPY components/_common/deploy.sh /code/deploy.sh

# Copy component specific files
COPY components/wr-qdrant-embeddings/tests /code/tests/
COPY components/wr-qdrant-embeddings/scripts /code/scripts/

WORKDIR /code/

# Install dependencies using uv into .venv
RUN uv sync --locked

# Set the command to run using python from .venv
CMD ["/code/.venv/bin/python", "-u", "/code/src/component.py"]