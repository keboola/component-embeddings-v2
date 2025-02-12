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

# Test stage
FROM python:3.12-slim-bullseye AS tester

# Install test tools
RUN pip install --no-cache-dir flake8 pytest

# Copy dependencies and test files
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY /src /code/src/
COPY /tests /code/tests/
COPY flake8.cfg /code/flake8.cfg

WORKDIR /code/

# Runtime stage (minimal)
FROM python:3.12-slim-bullseye AS runtime

# Create non-root user
RUN useradd -m -U app && \
    mkdir -p /code && \
    chown -R app:app /code

# Remove unnecessary files and clean up
RUN apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    find /usr/local/lib/python3.12 -name '__pycache__' -type d -exec rm -r {} +

# Copy only runtime dependencies and source code
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY /src /code/src/

WORKDIR /code/
USER app

CMD ["python", "-u", "/code/src/component.py"]