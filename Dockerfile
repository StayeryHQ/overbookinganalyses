# Slim-based build (manylinux wheels for reliability)
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies (most packages ship pre-built wheels on glibc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv via pip
RUN pip install --no-cache-dir uv

# Copy dependency manifests first (for layer caching)
COPY pyproject.toml uv.lock ./

ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN uv sync --no-dev --frozen --no-cache

# Stage 2: Runtime
FROM python:3.12-slim

WORKDIR /app

# Install only curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY dash_app ./dash_app
COPY src ./src
COPY main.py ./main.py
COPY configs ./configs
COPY pyproject.toml ./
COPY Data/ /app/.builtin-models/

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8050

HEALTHCHECK CMD curl --fail http://localhost:8050/_carogician/

ENTRYPOINT ["gunicorn", "dash_app.app:server", "--bind", "0.0.0.0:8050", "--workers", "4", "--timeout", "120", "--access-logfile", "-"]