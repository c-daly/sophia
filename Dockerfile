FROM ghcr.io/c-daly/logos-foundry:0.5.0

# Set working directory
WORKDIR /app/sophia

# Clear PYTHONPATH from base image — the base image sets PYTHONPATH=/app which
# causes Python to import logos_hcg from the base image's source tree (/app/logos_hcg/)
# instead of the git-pinned version installed by poetry to site-packages.
ENV PYTHONPATH=

# Copy application code and configuration
COPY src/ ./src/
COPY pyproject.toml poetry.lock README.md ./

# Install ML dependencies only when requested to keep standard images small
ARG SOPHIA_INSTALL_ML=0

RUN if [ "$SOPHIA_INSTALL_ML" = "1" ]; then \
      poetry install --only main --with ml --no-interaction --no-ansi; \
    else \
      poetry install --only main --no-interaction --no-ansi; \
    fi

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "sophia.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
