FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install uv
RUN uv pip install --no-cache-dir -r requirements.txt

# Copy mock server code
COPY server.py .

# Expose port
EXPOSE 8000

# Simple healthcheck that just checks if the container is running
HEALTHCHECK --interval=5s --timeout=5s --retries=5 \
    CMD true

# Run the mock server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"] 