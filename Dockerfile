# Build stage
FROM python:3.12-slim-bullseye AS builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Install build tools
RUN pip install --no-cache-dir uv 

# Copy only requirements first to leverage Docker cache
COPY pyproject.toml uv.lock /code/

# Install dependencies
WORKDIR /code
RUN uv pip install -r pyproject.toml --system --no-cache

# Runtime stage
FROM python:3.12-slim-bullseye AS runtime

# Install test tools
RUN pip install --no-cache-dir flake8

# Copy only necessary files
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY /src /code/src/
COPY /tests /code/tests/
COPY /scripts /code/scripts/
COPY flake8.cfg /code/flake8.cfg
COPY deploy.sh /code/deploy.sh
COPY pyproject.toml /code/pyproject.toml
COPY uv.lock /code/uv.lock

WORKDIR /code/

CMD ["python", "-u", "/code/src/component.py"]