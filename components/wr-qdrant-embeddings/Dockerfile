FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install build tools
RUN pip install --no-cache-dir flake8

# Copy only requirements first to leverage Docker cache
COPY components/_common/pyproject.toml components/_common/uv.lock /code/

WORKDIR /code/
RUN uv sync

COPY components/_common/src /code/src/
COPY components/_common/flake8.cfg /code/flake8.cfg
COPY components/_common/deploy.sh /code/deploy.sh
COPY components/wr-qdrant-embeddings/tests /code/tests/
COPY components/wr-qdrant-embeddings/scripts /code/scripts/

WORKDIR /code/

CMD ["python", "-u", "/code/src/component.py"]