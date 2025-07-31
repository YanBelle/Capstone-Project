FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create directories for models and data
RUN mkdir -p /app/models /app/data

EXPOSE 8000

# Use debug version temporarily to isolate issues
CMD ["uvicorn", "app.main_debug:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
