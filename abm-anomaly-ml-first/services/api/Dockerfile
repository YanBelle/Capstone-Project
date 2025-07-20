FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed for psutil and other packages
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create necessary directories for models and data
RUN mkdir -p /app/models /app/logs /app/input/processed /app/output /app/cache

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
