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

# Labels for container metadata
LABEL org.opencontainers.image.title="WINDMAR API" \
      org.opencontainers.image.description="Maritime Route Optimization API" \
      org.opencontainers.image.vendor="SL Mar" \
      org.opencontainers.image.version="2.1.0" \
      org.opencontainers.image.licenses="Commercial"

# Security: Run as non-root user
RUN groupadd --gid 1000 windmar \
    && useradd --uid 1000 --gid windmar --shell /bin/bash --create-home windmar

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH="/app/deps/bin:$PATH" \
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
COPY --chown=windmar:windmar LICENSE ./

# Create necessary directories with correct permissions
RUN mkdir -p data/grib data/vessel_database data/calibration data/weather_cache logs \
    && chown -R windmar:windmar /app

# Switch to non-root user
USER windmar

# Expose API port
EXPOSE 8000

# Health check with curl (more reliable than Python in minimal image)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run the API server with production settings
# - Workers based on CPU cores (2 * cores + 1 is recommended)
# - Access log disabled for performance (structured logging handles this)
# - Proxy headers enabled for load balancer compatibility
CMD ["python", "-m", "uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*", \
     "--access-log", \
     "--log-level", "info"]
