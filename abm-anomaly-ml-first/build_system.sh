#!/bin/bash

echo "🚀 Building ML-First ABM Anomaly Detection System..."

# Set environment variables for better builds
export DOCKER_BUILDKIT=0
export COMPOSE_DOCKER_CLI_BUILD=0

# Pre-pull base images
echo "📥 Pre-pulling base images..."
docker pull postgres:15-alpine &
docker pull redis:7-alpine &
docker pull python:3.10-slim &
docker pull grafana/grafana:latest &
docker pull prom/prometheus:latest &
wait

echo "✅ Base images ready!"

# Build services in optimal order (lightweight services first)
echo "🔨 Building services..."

echo "Building PostgreSQL (pre-built)..."
docker compose build postgres

echo "Building Redis (pre-built)..."
docker compose build redis

echo "Building API service..."
docker compose build api

echo "Building Dashboard..."
docker compose build dashboard

echo "Building Jupyter service..."
docker compose build jupyter

echo "Building Anomaly Detector (this will take longer due to ML libraries)..."
docker compose build anomaly-detector

echo "✅ All services built successfully!"

# Start the system
echo "🚀 Starting ML-First ABM System..."
make up

echo "🎉 System is starting up!"
echo ""
echo "📊 Access URLs:"
echo "  Dashboard: http://localhost:3000"
echo "  API Docs:  http://localhost:8000/docs"
echo "  Jupyter:   http://localhost:8888 (token: ml_jupyter_token_123)"
echo "  Grafana:   http://localhost:3001 (admin/ml_admin)"
