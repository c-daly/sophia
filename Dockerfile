FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==2.2.1

# Copy poetry files
COPY pyproject.toml poetry.lock ./

# Configure poetry to not create a virtual environment
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --only main --no-interaction --no-ansi

# Copy application code
COPY src/ ./src/

# Create non-root user
RUN useradd -m -u 1000 sophia && \
    chown -R sophia:sophia /app

USER sophia

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "sophia.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
