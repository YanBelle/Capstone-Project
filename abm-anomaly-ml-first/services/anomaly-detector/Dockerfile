FROM python:3.10-slim

# Build argument to optionally skip model downloads
ARG SKIP_MODEL_DOWNLOAD=false

WORKDIR /app

# Install system dependencies for ML libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Increase pip timeout for large models
ENV PIP_DEFAULT_TIMEOUT=200
ENV PYTHONPATH=/app

# Copy requirements first for better caching
COPY requirements.txt .

# Install requirements with specific version handling for compatibility
RUN pip install --no-cache-dir -r requirements.txt

# Fix huggingface_hub compatibility issues
RUN pip install --upgrade "huggingface_hub==0.15.1" "transformers==4.31.0"

# Copy model download script
COPY download_models.py /tmp/download_models.py

# Download ML models during build (optional)
RUN if [ "$SKIP_MODEL_DOWNLOAD" != "true" ]; then \
        echo "Downloading ML models during build..."; \
        python /tmp/download_models.py; \
    else \
        echo "Skipping model download - will download at runtime"; \
    fi && rm /tmp/download_models.py

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/models /app/logs /app/input/processed /app/output /app/data/sessions /app/cache

CMD ["python", "main.py"]
