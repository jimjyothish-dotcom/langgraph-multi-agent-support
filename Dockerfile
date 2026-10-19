# =============================================================================
# LangGraph Multi-Agent Support System — Production Dockerfile
# Galaxy Telecom CRM AI Support — multi-agent support
#
# Multi-stage build:
#   Stage 1 (builder) — installs dependencies into an isolated layer
#   Stage 2 (runtime) — copies only the installed packages, no build tools
#
# This keeps the final image lean: no pip, no compilers, no build cache.
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder — install Python dependencies
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build-time system dependencies (needed to compile some packages)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libffi-dev \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies into a target directory so we can
# copy just the installed packages into the final image
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --target=/build/packages -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: Runtime — lean production image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Metadata labels — useful for Docker Hub and production tooling
LABEL maintainer="Galaxy Telecom Support"
LABEL org.opencontainers.image.title="LangGraph Multi-Agent Support System"
LABEL org.opencontainers.image.description="Multi-agent AI support system — Galaxy Telecom CRM scenario"
LABEL org.opencontainers.image.source="https://github.com/<your-org>/langgraph-multi-agent-support"
LABEL org.opencontainers.image.version="1.0.0"

# Security: run as a non-root user
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Runtime system dependencies only (no compilers)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /build/packages /usr/local/lib/python3.11/site-packages

# Copy application source
COPY --chown=appuser:appgroup . .

# Create the data directory for SQLite volume mount
# The volume will be mounted here; this ensures it exists if not mounted
RUN mkdir -p /app/data && chown appuser:appgroup /app/data

# Switch to non-root user
USER appuser

# Environment defaults — these are safe non-secret defaults only.
# All secrets MUST be supplied via .env file or docker run --env-file.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    SQLITE_DB_PATH=/app/data/audit.db

# The SQLite audit log is written here — mount a named volume at /app/data
# to persist it across container restarts (see docker-compose.yml)
VOLUME ["/app/data"]

# Entrypoint: run the CLI support agent
CMD ["python", "main.py"]
