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
RUN uv pip install -r pyproject.toml --system --no-cache --no-deps

# Runtime stage
FROM python:3.12-slim-bullseye

# Copy only necessary files
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY /src /code/src/

WORKDIR /code/

CMD ["python", "-u", "/code/src/component.py"]