FROM python:3.13-slim-bullseye

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Install build tools
RUN pip install --no-cache-dir uv flake8

# Copy only requirements first to leverage Docker cache
COPY components/_common/pyproject.toml components/_common/uv.lock /code/

COPY components/_common/src /code/src/
COPY components/_common/flake8.cfg /code/flake8.cfg
COPY components/_common/deploy.sh /code/deploy.sh
COPY components/wr-qdrant-embeddings/tests /code/tests/
COPY components/wr-qdrant-embeddings/scripts /code/scripts/

WORKDIR /code/
RUN uv sync

CMD ["python", "-u", "/code/src/component.py"]