# BERT Debugging Dashboard

## Overview
This is a Dockerized web application for debugging and optimizing BERT models trained on Electronic Journal (EJ) logs.

## Features
- Token Attention Visualization
- Prediction Analysis
- Saliency Maps
- Embedding Visualization (t-SNE/UMAP)
- Performance Metrics
- Misclassification Explorer
- Token Bias Analysis

## Setup

### Prerequisites
- Docker and Docker Compose installed
- BERT model files (place in `models/bert_ej_model/`)

### Quick Start
1. Run the setup script:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

2. Place your BERT model files in `models/bert_ej_model/`

3. Start the application:
   ```bash
   docker-compose up --build
   ```

4. Access the dashboard at http://localhost:3333

### Model Requirements
Your BERT model directory should contain:
- `config.json`
- `pytorch_model.bin` or `model.safetensors`
- `tokenizer_config.json`
- `vocab.txt`

### Sample Data Format
CSV files should have the following columns:
- `text`: The EJ log text
- `label` (optional): True label for metrics calculation

## Port Configuration
- Frontend: 3333
- Backend API: 8000

## Troubleshooting
- If the model fails to load, the application will use a default BERT model
- Check Docker logs: `docker-compose logs -f`
- Ensure all ports are free before starting
