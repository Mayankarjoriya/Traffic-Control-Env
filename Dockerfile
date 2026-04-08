# Multi-stage Dockerfile for Hugging Face Spaces
# Build context is the repo root (Traffic-Control-Env/).
# All environment code lives under smart_traffic_env/.

FROM python:3.13-slim-bookworm AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

# Copy only the environment package (not the whole repo root)
COPY smart_traffic_env/ /app/env/

WORKDIR /app/env

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi


# ── Final runtime stage ──────────────────────────────────────────────────────
FROM python:3.13-slim-bookworm

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy venv and source code from builder
COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env

# Use the venv
ENV PATH="/app/.venv/bin:$PATH"

# Make the package importable
ENV PYTHONPATH="/app/env:$PYTHONPATH"

ENV ENABLE_WEB_INTERFACE=true

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the FastAPI server
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 7860"]
