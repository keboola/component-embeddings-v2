FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install build tools (if needed, e.g., flake8 for linting outside uv sync)
RUN pip install --no-cache-dir flake8

# Copy common files first
COPY components/_common/pyproject.toml components/_common/uv.lock /code/
COPY components/_common/src /code/src/
COPY components/_common/flake8.cfg /code/flake8.cfg
COPY components/_common/deploy.sh /code/deploy.sh

# Copy component specific files
COPY components/app-embeddings-v2/tests /code/tests/
COPY components/app-embeddings-v2/scripts /code/scripts/

WORKDIR /code/

# Install dependencies using uv into .venv
# uv automatically creates and uses .venv when pyproject.toml is present
RUN uv sync --locked

# Set the command to run using python from .venv
CMD ["/code/.venv/bin/python", "-u", "/code/src/component.py"]