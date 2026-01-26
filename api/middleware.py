"""
Production-grade middleware for WINDMAR API.

Provides:
- Security headers (CSP, XSS protection, etc.)
- Request ID tracking for distributed tracing
- Structured logging with correlation IDs
- Request/response timing metrics
- Error handling and sanitization
"""
import time
import uuid
import logging
import json
from typing import Callable, Optional
from contextvars import ContextVar
from datetime import datetime

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from fastapi import FastAPI

# Context variable for request ID (thread-safe)
request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def get_request_id() -> Optional[str]:
    """Get the current request ID from context."""
    return request_id_ctx.get()


class StructuredLogger:
    """
    Structured JSON logger for production environments.

    Outputs logs in JSON format with consistent fields for log aggregation
    systems like ELK, Datadog, or CloudWatch.
    """

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def _log(self, level: str, message: str, **kwargs):
        """Internal log method with structured output."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "message": message,
            "service": "windmar-api",
            "request_id": get_request_id(),
            **kwargs
        }

        # Remove None values
        log_entry = {k: v for k, v in log_entry.items() if v is not None}

        getattr(self.logger, level.lower())(json.dumps(log_entry))

    def info(self, message: str, **kwargs):
        self._log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs):
        self._log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs):
        self._log("ERROR", message, **kwargs)

    def debug(self, message: str, **kwargs):
        self._log("DEBUG", message, **kwargs)


# Global structured logger instance
structured_logger = StructuredLogger("windmar")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Adds security headers to all responses.

    Headers added:
    - X-Content-Type-Options: Prevents MIME type sniffing
    - X-Frame-Options: Prevents clickjacking
    - X-XSS-Protection: Enables XSS filtering
    - Referrer-Policy: Controls referrer information
    - Permissions-Policy: Restricts browser features
    - Content-Security-Policy: Controls resource loading
    - Strict-Transport-Security: Enforces HTTPS (when enabled)
    """

    def __init__(self, app, enable_hsts: bool = False):
        super().__init__(app)
        self.enable_hsts = enable_hsts

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"

        # Enable XSS filtering
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Control referrer information
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Restrict browser features
        response.headers["Permissions-Policy"] = (
            "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
            "magnetometer=(), microphone=(), payment=(), usb=()"
        )

        # Content Security Policy - Strict mode for production security
        # Note: If you need inline scripts/styles, use nonces or hashes instead
        # See: https://developer.mozilla.org/en-US/docs/Web/HTTP/CSP
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self' https:; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'; "
            "upgrade-insecure-requests;"
        )

        # HSTS - only enable in production with HTTPS
        if self.enable_hsts:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )

        return response


class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    Adds unique request ID to each request for distributed tracing.

    The request ID is:
    - Generated as a UUID4 if not provided
    - Accepted from X-Request-ID header if provided (for tracing across services)
    - Added to response headers for client correlation
    - Available via get_request_id() for logging
    """

    HEADER_NAME = "X-Request-ID"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get or generate request ID
        request_id = request.headers.get(self.HEADER_NAME) or str(uuid.uuid4())

        # Set in context for access throughout request lifecycle
        token = request_id_ctx.set(request_id)

        try:
            response = await call_next(request)
            response.headers[self.HEADER_NAME] = request_id
            return response
        finally:
            request_id_ctx.reset(token)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs all requests with timing and metadata.

    Logs include:
    - Method, path, query parameters
    - Response status code
    - Request duration in milliseconds
    - Client IP address
    - User agent
    """

    # Paths to exclude from logging (health checks, metrics)
    EXCLUDED_PATHS = {"/api/health", "/api/metrics", "/health", "/metrics"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip logging for health checks
        if request.url.path in self.EXCLUDED_PATHS:
            return await call_next(request)

        start_time = time.perf_counter()

        # Get client info
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")

        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start_time) * 1000

            structured_logger.info(
                "Request completed",
                method=request.method,
                path=request.url.path,
                query=str(request.query_params) if request.query_params else None,
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
                client_ip=client_ip,
                user_agent=user_agent[:100] if user_agent else None,
            )

            return response

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            structured_logger.error(
                "Request failed",
                method=request.method,
                path=request.url.path,
                error=str(e),
                error_type=type(e).__name__,
                duration_ms=round(duration_ms, 2),
                client_ip=client_ip,
            )
            raise


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Global error handling with sanitized responses.

    In production:
    - Internal errors return generic messages (no stack traces)
    - All errors are logged with full details
    - Error responses include request ID for support

    In development:
    - Full error details are returned
    """

    def __init__(self, app, debug: bool = False):
        super().__init__(app)
        self.debug = debug

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as e:
            request_id = get_request_id()

            # Log the full error
            structured_logger.error(
                "Unhandled exception",
                error=str(e),
                error_type=type(e).__name__,
                path=request.url.path,
                method=request.method,
            )

            # Return sanitized error response
            if self.debug:
                detail = str(e)
            else:
                detail = "An internal error occurred. Please contact support with the request ID."

            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "detail": detail,
                    "request_id": request_id,
                },
                headers={"X-Request-ID": request_id} if request_id else {},
            )


# Metrics storage for Prometheus-style metrics
class MetricsCollector:
    """
    Simple in-memory metrics collector for request statistics.

    Collects:
    - Request counts by method, path, status
    - Request duration histograms
    - Error counts
    """

    def __init__(self):
        self.request_count: dict = {}
        self.request_duration_sum: dict = {}
        self.request_duration_count: dict = {}
        self.error_count: dict = {}
        self.start_time = datetime.utcnow()

    def record_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_seconds: float
    ):
        """Record a completed request."""
        # Normalize path (remove IDs for aggregation)
        normalized_path = self._normalize_path(path)
        key = f"{method}:{normalized_path}:{status_code}"

        self.request_count[key] = self.request_count.get(key, 0) + 1
        self.request_duration_sum[key] = (
            self.request_duration_sum.get(key, 0) + duration_seconds
        )
        self.request_duration_count[key] = (
            self.request_duration_count.get(key, 0) + 1
        )

        if status_code >= 500:
            error_key = f"{method}:{normalized_path}"
            self.error_count[error_key] = self.error_count.get(error_key, 0) + 1

    def _normalize_path(self, path: str) -> str:
        """Normalize path by replacing UUIDs and IDs with placeholders."""
        import re
        # Replace UUIDs
        path = re.sub(
            r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '{id}',
            path,
            flags=re.IGNORECASE
        )
        # Replace numeric IDs
        path = re.sub(r'/\d+(?=/|$)', '/{id}', path)
        return path

    def get_metrics(self) -> dict:
        """Get all metrics as a dictionary."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()

        return {
            "uptime_seconds": uptime,
            "requests": {
                "total": sum(self.request_count.values()),
                "by_endpoint": self.request_count,
            },
            "latency": {
                "sum_seconds": self.request_duration_sum,
                "count": self.request_duration_count,
            },
            "errors": {
                "total": sum(self.error_count.values()),
                "by_endpoint": self.error_count,
            },
        }

    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus exposition format."""
        lines = []

        # Uptime
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        lines.append(f"# HELP windmar_uptime_seconds Time since service start")
        lines.append(f"# TYPE windmar_uptime_seconds gauge")
        lines.append(f"windmar_uptime_seconds {uptime}")

        # Request count
        lines.append(f"# HELP windmar_requests_total Total request count")
        lines.append(f"# TYPE windmar_requests_total counter")
        for key, count in self.request_count.items():
            method, path, status = key.split(":")
            lines.append(
                f'windmar_requests_total{{method="{method}",path="{path}",status="{status}"}} {count}'
            )

        # Request duration
        lines.append(f"# HELP windmar_request_duration_seconds Request duration")
        lines.append(f"# TYPE windmar_request_duration_seconds summary")
        for key, total in self.request_duration_sum.items():
            method, path, status = key.split(":")
            count = self.request_duration_count.get(key, 1)
            avg = total / count if count > 0 else 0
            lines.append(
                f'windmar_request_duration_seconds_sum{{method="{method}",path="{path}",status="{status}"}} {total}'
            )
            lines.append(
                f'windmar_request_duration_seconds_count{{method="{method}",path="{path}",status="{status}"}} {count}'
            )

        # Errors
        lines.append(f"# HELP windmar_errors_total Total error count")
        lines.append(f"# TYPE windmar_errors_total counter")
        for key, count in self.error_count.items():
            method, path = key.split(":")
            lines.append(
                f'windmar_errors_total{{method="{method}",path="{path}"}} {count}'
            )

        return "\n".join(lines)


# Global metrics collector
metrics_collector = MetricsCollector()


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Collects request metrics for monitoring.
    """

    EXCLUDED_PATHS = {"/api/health", "/api/metrics", "/health", "/metrics"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self.EXCLUDED_PATHS:
            return await call_next(request)

        start_time = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start_time

        metrics_collector.record_request(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_seconds=duration,
        )

        return response


def setup_middleware(app: FastAPI, debug: bool = False, enable_hsts: bool = False):
    """
    Configure all production middleware for the application.

    Args:
        app: FastAPI application instance
        debug: Enable debug mode (detailed error messages)
        enable_hsts: Enable HSTS header (requires HTTPS)

    Order matters! Middleware is executed in reverse order of addition.
    """
    # These are added last but execute first
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(SecurityHeadersMiddleware, enable_hsts=enable_hsts)
    app.add_middleware(ErrorHandlingMiddleware, debug=debug)
