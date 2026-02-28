# =============================================================================
# WINDMAR API - Production Dockerfile
# =============================================================================
# Multi-stage build optimized for security and performance
#
# Build: docker build -t windmar-api:latest .
# Run:   docker run -p 8000:8000 windmar-api:latest
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Build dependencies
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Set build-time environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gfortran \
    libeccodes-dev \
    libgeos-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --target=/build/deps -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Production runtime
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ARG BUILD_COMMIT=unknown

# Labels for container metadata
LABEL org.opencontainers.image.title="WINDMAR API" \
      org.opencontainers.image.description="Maritime Route Optimization API" \
      org.opencontainers.image.vendor="SL Mar" \
      org.opencontainers.image.version="2.1.0" \
      org.opencontainers.image.revision="${BUILD_COMMIT}" \
      org.opencontainers.image.licenses="Apache-2.0"

# Security: Run as non-root user
RUN groupadd --gid 1000 windmar \
    && useradd --uid 1000 --gid windmar --shell /bin/bash --create-home windmar

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/app/deps \
    PATH="/app/deps/bin:$PATH" \
    BUILD_COMMIT=${BUILD_COMMIT} \
    # Application defaults (override via environment)
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    LOG_LEVEL=info \
    ENVIRONMENT=production

WORKDIR /app

# Install runtime system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libeccodes0 \
    libgeos-c1v5 \
    libproj25 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python dependencies from builder
COPY --from=builder /build/deps /app/deps

# Copy application code
COPY --chown=windmar:windmar src/ ./src/
COPY --chown=windmar:windmar api/ ./api/
COPY --chown=windmar:windmar docker/migrations/ ./docker/migrations/
COPY --chown=windmar:windmar LICENSE ./

# Create necessary directories with correct permissions
RUN mkdir -p data/grib data/gfs_cache data/vessel_database data/calibration data/weather_cache data/copernicus_cache data/climatology_cache logs \
    /tmp/windmar_cache/wind /tmp/windmar_cache/wave /tmp/windmar_cache/current /tmp/windmar_cache/ice /tmp/windmar_cache/sst /tmp/windmar_cache/vis \
    /tmp/windmar_tiles \
    && chown -R windmar:windmar /app /tmp/windmar_cache /tmp/windmar_tiles

# Switch to non-root user
USER windmar

# Pre-download GSHHS shapefiles so first request doesn't incur 70MB download.
# Must run AFTER USER windmar so data lands in /home/windmar/.local/share/cartopy/
# (same path the runtime process will look in).
RUN PYTHONPATH=/app/deps python -c "import cartopy.io.shapereader as s; s.gshhs('i', 1)" || true

# Expose API port
EXPOSE 8000

# Health check with curl (more reliable than Python in minimal image)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run the API server with gunicorn + uvicorn workers
# - gunicorn manages worker lifecycle with configurable timeout
# - --timeout 600 allows long-running CMEMS downloads (3-5 min each)
#   and VISIR optimizations (~15-30s). Blocking I/O in sync ingestion
#   functions stalls the event loop, preventing heartbeats.
# - uvicorn workers handle async endpoints (FastAPI)
CMD ["gunicorn", "api.main:app", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--timeout", "600", \
     "--graceful-timeout", "60", \
     "--forwarded-allow-ips", "172.16.0.0/12,10.0.0.0/8,127.0.0.1", \
     "--access-logfile", "-", \
     "--log-level", "info"]
