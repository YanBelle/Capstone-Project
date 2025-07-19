#!/bin/bash
# setup_ml_first_abm_system.sh - Complete ML-First ABM Anomaly Detection with Expert Labeling

echo "================================================================"
echo "ML-First ABM EJ Log Anomaly Detection System"
echo "With Expert Labeling and Supervised Learning"
echo "================================================================"

# Create main project directory
PROJECT_NAME="abm-anomaly-ml-first"
echo "Creating project directory: $PROJECT_NAME"
mkdir -p $PROJECT_NAME
cd $PROJECT_NAME

# Create directory structure
echo "Creating directory structure..."
mkdir -p services/{anomaly-detector,api,dashboard/{src,public},jupyter}
mkdir -p data/{models,logs,input/{processed},output,sessions}
mkdir -p notebooks
mkdir -p init-db
mkdir -p grafana/{provisioning/{dashboards,datasources},dashboards}
mkdir -p prometheus
mkdir -p docs

# Create .env file
echo "Creating environment configuration..."
cat > .env << 'EOF'
# Database Configuration
POSTGRES_DB=abm_ml_db
POSTGRES_USER=abm_user
POSTGRES_PASSWORD=secure_ml_password123
POSTGRES_HOST=postgres

# Redis Configuration
REDIS_HOST=redis
REDIS_PASSWORD=redis_ml_password123

# Security
JWT_SECRET=your_jwt_secret_key_ml_first

# ML Configuration
ANOMALY_THRESHOLD=0.7
MODEL_UPDATE_INTERVAL=3600
LOG_LEVEL=INFO
BERT_MODEL=bert-base-uncased

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Dashboard Configuration
REACT_APP_API_URL=http://localhost:8000
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
*.egg-info/

# Data
data/logs/*
data/input/*
!data/input/processed/
data/output/*
data/models/*.pkl
data/models/*.h5
data/sessions/*

# Jupyter
.ipynb_checkpoints/
notebooks/.ipynb_checkpoints/

# Docker
*.log

# Environment
.env.local
.env.*.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Node
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# React
services/dashboard/build/
services/dashboard/.env.local

# Models
*.pkl
*.h5
*.pth
*.bin
EOF

# Create docker-compose.yml
echo "Creating Docker Compose configuration..."
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  # ML-First anomaly detection service
  anomaly-detector:
    build: ./services/anomaly-detector
    container_name: abm-ml-anomaly-detector
    environment:
      - PYTHONUNBUFFERED=1
      - LOG_LEVEL=${LOG_LEVEL}
      - MODEL_UPDATE_INTERVAL=${MODEL_UPDATE_INTERVAL}
      - ANOMALY_THRESHOLD=${ANOMALY_THRESHOLD}
      - REDIS_HOST=${REDIS_HOST}
      - REDIS_PASSWORD=${REDIS_PASSWORD}
      - POSTGRES_HOST=${POSTGRES_HOST}
      - POSTGRES_DB=${POSTGRES_DB}
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - BERT_MODEL=${BERT_MODEL}
      - TRANSFORMERS_CACHE=/app/cache
    volumes:
      - ./data/models:/app/models
      - ./data/logs:/app/logs
      - ./data/input:/app/input
      - ./data/output:/app/output
      - ./data/sessions:/app/data/sessions
      - transformer-cache:/app/cache
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    networks:
      - abm-network

  # API service with expert labeling endpoints
  api:
    build: ./services/api
    container_name: abm-ml-api
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
      - REDIS_HOST=${REDIS_HOST}
      - REDIS_PASSWORD=${REDIS_PASSWORD}
      - POSTGRES_HOST=${POSTGRES_HOST}
      - POSTGRES_DB=${POSTGRES_DB}
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - JWT_SECRET=${JWT_SECRET}
    volumes:
      - ./data/models:/app/models
      - ./data/input:/app/input
      - ./data/sessions:/app/data/sessions
    depends_on:
      - postgres
      - redis
      - anomaly-detector
    restart: unless-stopped
    networks:
      - abm-network

  # Enhanced dashboard with expert labeling interface
  dashboard:
    build: ./services/dashboard
    container_name: abm-ml-dashboard
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=${REACT_APP_API_URL}
    depends_on:
      - api
    restart: unless-stopped
    networks:
      - abm-network

  # PostgreSQL with ML schema
  postgres:
    image: postgres:15-alpine
    container_name: abm-ml-postgres
    environment:
      - POSTGRES_DB=${POSTGRES_DB}
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-db:/docker-entrypoint-initdb.d
    ports:
      - "5432:5432"
    restart: unless-stopped
    networks:
      - abm-network

  # Redis for caching and real-time
  redis:
    image: redis:7-alpine
    container_name: abm-ml-redis
    command: redis-server --requirepass ${REDIS_PASSWORD}
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    networks:
      - abm-network

  # Jupyter for ML analysis
  jupyter:
    build: ./services/jupyter
    container_name: abm-ml-jupyter
    ports:
      - "8888:8888"
    environment:
      - JUPYTER_ENABLE_LAB=yes
      - JUPYTER_TOKEN=ml_jupyter_token_123
    volumes:
      - ./notebooks:/home/jovyan/work
      - ./data:/home/jovyan/data
    networks:
      - abm-network

  # Grafana for monitoring
  grafana:
    image: grafana/grafana:latest
    container_name: abm-ml-grafana
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=ml_admin
      - GF_SECURITY_ADMIN_USER=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
      - ./grafana/dashboards:/var/lib/grafana/dashboards
    depends_on:
      - prometheus
    restart: unless-stopped
    networks:
      - abm-network

  # Prometheus for metrics
  prometheus:
    image: prom/prometheus:latest
    container_name: abm-ml-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    restart: unless-stopped
    networks:
      - abm-network

volumes:
  postgres_data:
  redis_data:
  grafana_data:
  prometheus_data:
  transformer-cache:

networks:
  abm-network:
    driver: bridge
EOF

# Create Anomaly Detector Dockerfile
echo "Setting up ML-First Anomaly Detector service..."
cat > services/anomaly-detector/Dockerfile << 'EOF'
FROM python:3.10-slim

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
RUN pip install --no-cache-dir -r requirements.txt

# Download BERT model during build
RUN python -c "from transformers import BertTokenizer, BertModel; \
    BertTokenizer.from_pretrained('bert-base-uncased'); \
    BertModel.from_pretrained('bert-base-uncased')"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/models /app/logs /app/input/processed /app/output /app/data/sessions /app/cache

CMD ["python", "main.py"]
EOF

# Create requirements.txt for ML-first detector
cat > services/anomaly-detector/requirements.txt << 'EOF'
# Core Data Science
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0

# Deep Learning
tensorflow==2.13.0
torch==2.0.1
transformers==4.31.0

# NLP and Embeddings
sentence-transformers==2.2.2

# Time Series and Statistics
statsmodels==0.14.0
scipy==1.11.1

# Visualization
matplotlib==3.7.2
seaborn==0.12.2

# Model Persistence
joblib==1.3.1
h5py==3.9.0

# Database and Caching
psycopg2-binary==2.9.7
sqlalchemy==2.0.19
redis==4.6.0

# Environment and Configuration
python-dotenv==1.0.0

# Logging and Monitoring
loguru==0.7.0

# Scheduling
schedule==1.2.0

# Additional ML Tools
imbalanced-learn==0.11.0
xgboost==1.7.6
lightgbm==4.0.0

# Clustering
hdbscan==0.8.33
umap-learn==0.5.3
EOF

# Create main.py for ML-first anomaly detector
cat > services/anomaly-detector/main.py << 'EOF'
# ML-First ABM Anomaly Detection Service
import os
import sys
import time
import schedule
from loguru import logger
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, text
import redis
import json
import numpy as np

# Import the ML-first anomaly detector
from ml_analyzer import MLFirstAnomalyDetector

load_dotenv()

logger.add("/app/logs/anomaly_detector_{time}.log", rotation="100 MB")


class MLFirstEJProcessor:
    """Main processor for ML-first anomaly detection"""
    
    def __init__(self):
        # Database connection
        self.db_engine = create_engine(
            f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
            f"@{os.getenv('POSTGRES_HOST', 'postgres')}:5432/{os.getenv('POSTGRES_DB')}"
        )
        
        # Redis connection
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'redis'),
            port=6379,
            password=os.getenv('REDIS_PASSWORD'),
            decode_responses=True
        )
        
        # Initialize the ML-first detector
        self.detector = MLFirstAnomalyDetector()
        
        # Load existing models if available
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models if they exist"""
        model_dir = "/app/models"
        if os.path.exists(os.path.join(model_dir, "isolation_forest.pkl")):
            logger.info("Loading existing ML models...")
            try:
                import joblib
                self.detector.isolation_forest = joblib.load(
                    os.path.join(model_dir, "isolation_forest.pkl")
                )
                self.detector.one_class_svm = joblib.load(
                    os.path.join(model_dir, "one_class_svm.pkl")
                )
                self.detector.scaler = joblib.load(
                    os.path.join(model_dir, "scaler.pkl")
                )
                if os.path.exists(os.path.join(model_dir, "pca.pkl")):
                    self.detector.pca = joblib.load(
                        os.path.join(model_dir, "pca.pkl")
                    )
                logger.info("Models loaded successfully")
            except Exception as e:
                logger.error(f"Error loading models: {str(e)}")
        else:
            logger.info("No existing models found. Will train on first batch.")
    
    def process_ej_file(self, file_path: str):
        """Process an EJ log file using ML-first approach"""
        logger.info(f"Processing EJ file: {file_path}")
        
        try:
            # Run ML-first detection pipeline
            results_df = self.detector.process_ej_logs(file_path)
            
            # Store sessions in database
            self.store_sessions(results_df)
            
            # Store anomalies
            anomalies_df = results_df[results_df['is_anomaly']]
            if len(anomalies_df) > 0:
                self.store_anomalies(anomalies_df)
                self.generate_alerts(anomalies_df)
            
            # Publish real-time updates
            self.publish_updates(results_df)
            
            # Save updated models
            self.detector.save_models("/app/models")
            
            logger.info(f"Processing complete. Found {len(anomalies_df)} anomalies.")
            
            # Generate report
            self.generate_anomaly_report(anomalies_df)
            
        except Exception as e:
            logger.error(f"Error processing EJ file: {str(e)}")
            raise
    
    def store_sessions(self, results_df: pd.DataFrame):
        """Store all sessions in database with embeddings"""
        sessions_data = []
        
        for i, (_, row) in enumerate(results_df.iterrows()):
            # Get the embedding for this session
            embedding = self.detector.sessions[i].embedding
            
            # Store raw text
            session_id = row['session_id']
            raw_text = self.detector.sessions[i].raw_text
            self.store_session_raw_text(session_id, raw_text)
            
            session_data = {
                'session_id': session_id,
                'timestamp': row['start_time'] if pd.notna(row['start_time']) else datetime.now(),
                'session_length': row['session_length'],
                'is_anomaly': row['is_anomaly'],
                'anomaly_score': row['anomaly_score'],
                'anomaly_type': row['anomaly_type'] if row['anomaly_type'] else None,
                'detected_patterns': json.dumps(row['detected_patterns']),
                'critical_events': json.dumps(row['critical_events']),
                'embedding_vector': embedding.tobytes() if embedding is not None else None,
                'created_at': datetime.now()
            }
            sessions_data.append(session_data)
        
        # Store in database
        pd.DataFrame(sessions_data).to_sql(
            'ml_sessions', 
            self.db_engine, 
            if_exists='append', 
            index=False
        )
    
    def store_session_raw_text(self, session_id: str, raw_text: str):
        """Store raw text for a session"""
        # Store in file system
        output_dir = f"/app/data/sessions/{session_id[:2]}"
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f"{output_dir}/{session_id}.txt", 'w') as f:
            f.write(raw_text)
    
    def store_anomalies(self, anomalies_df: pd.DataFrame):
        """Store detected anomalies with ML-based details"""
        for _, anomaly in anomalies_df.iterrows():
            anomaly_data = {
                'session_id': anomaly['session_id'],
                'anomaly_type': anomaly['anomaly_type'] if anomaly['anomaly_type'] else 'unknown',
                'anomaly_score': float(anomaly['anomaly_score']),
                'detected_patterns': json.dumps(anomaly['detected_patterns']),
                'critical_events': json.dumps(anomaly['critical_events']),
                'model_name': 'ml_ensemble',
                'detected_at': datetime.now()
            }
            
            pd.DataFrame([anomaly_data]).to_sql(
                'ml_anomalies', 
                self.db_engine, 
                if_exists='append', 
                index=False
            )
    
    def generate_alerts(self, anomalies_df: pd.DataFrame):
        """Generate alerts for detected anomalies"""
        for _, anomaly in anomalies_df.iterrows():
            # Determine alert level
            alert_level = 'LOW'
            if anomaly['anomaly_score'] > 0.8:
                alert_level = 'HIGH'
            elif anomaly['anomaly_score'] > 0.6:
                alert_level = 'MEDIUM'
            
            # Check for critical patterns
            critical_patterns = [
                'unable_to_dispense', 
                'device_error', 
                'power_reset',
                'cash_retract',
                'recovery_failed'
            ]
            
            if any(pattern in anomaly['detected_patterns'] for pattern in critical_patterns):
                alert_level = 'HIGH'
            
            alert_data = {
                'alert_level': alert_level,
                'message': json.dumps({
                    'session_id': anomaly['session_id'],
                    'anomaly_type': anomaly['anomaly_type'],
                    'anomaly_score': float(anomaly['anomaly_score']),
                    'patterns': anomaly['detected_patterns'],
                    'critical_events': anomaly['critical_events'],
                    'description': self.generate_alert_description(anomaly)
                }),
                'is_resolved': False,
                'created_at': datetime.now()
            }
            
            pd.DataFrame([alert_data]).to_sql(
                'alerts', 
                self.db_engine, 
                if_exists='append', 
                index=False
            )
            
            # Publish real-time alert
            self.redis_client.publish(
                'anomaly_alerts',
                json.dumps({
                    'session_id': anomaly['session_id'],
                    'alert_level': alert_level,
                    'anomaly_score': float(anomaly['anomaly_score']),
                    'patterns': anomaly['detected_patterns'],
                    'critical_events': anomaly['critical_events'],
                    'timestamp': datetime.now().isoformat()
                })
            )
    
    def generate_alert_description(self, anomaly):
        """Generate human-readable description of the anomaly"""
        descriptions = []
        
        # Map patterns to descriptions
        pattern_descriptions = {
            'supervisor_mode': 'Supervisor mode activity detected',
            'unable_to_dispense': 'ATM unable to dispense cash',
            'device_error': 'Hardware device error occurred',
            'power_reset': 'Power reset or restart detected',
            'cash_retract': 'Cash retraction initiated',
            'no_dispense': 'Cash dispensing failed',
            'notes_issue': 'Issue with note handling',
            'note_error': 'Note processing error',
            'recovery_failed': 'Recovery operation failed'
        }
        
        for pattern in anomaly['detected_patterns']:
            if pattern in pattern_descriptions:
                descriptions.append(pattern_descriptions[pattern])
        
        # Add critical events
        for event in anomaly['critical_events']:
            descriptions.append(event)
        
        return '; '.join(descriptions) if descriptions else 'Anomalous pattern detected'
    
    def publish_updates(self, results_df: pd.DataFrame):
        """Publish dashboard updates via Redis"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_sessions': len(results_df),
            'total_anomalies': int(results_df['is_anomaly'].sum()),
            'anomaly_rate': float(results_df['is_anomaly'].mean()),
            'anomaly_types': {},
            'pattern_summary': {},
            'processing_mode': 'ml_first'
        }
        
        # Count anomaly types
        anomaly_types = results_df[results_df['is_anomaly']]['anomaly_type'].value_counts()
        summary['anomaly_types'] = anomaly_types.to_dict() if len(anomaly_types) > 0 else {}
        
        # Pattern frequency
        all_patterns = []
        for patterns in results_df[results_df['is_anomaly']]['detected_patterns']:
            all_patterns.extend(patterns)
        
        if all_patterns:
            pattern_counts = pd.Series(all_patterns).value_counts().head(5)
            summary['pattern_summary'] = pattern_counts.to_dict()
        
        # Publish to Redis
        self.redis_client.publish('dashboard_updates', json.dumps(summary))
        self.redis_client.setex('latest_ml_summary', 3600, json.dumps(summary))
    
    def generate_anomaly_report(self, anomalies_df: pd.DataFrame):
        """Generate detailed anomaly report"""
        if len(anomalies_df) == 0:
            return
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'total_anomalies': len(anomalies_df),
            'anomaly_breakdown': {},
            'critical_findings': [],
            'pattern_analysis': {},
            'recommendations': []
        }
        
        # Anomaly type breakdown
        type_counts = anomalies_df['anomaly_type'].value_counts()
        report['anomaly_breakdown'] = type_counts.to_dict()
        
        # Critical findings
        for _, anomaly in anomalies_df.iterrows():
            if anomaly['anomaly_score'] > 0.8:
                finding = {
                    'session_id': anomaly['session_id'],
                    'score': float(anomaly['anomaly_score']),
                    'events': anomaly['critical_events']
                }
                report['critical_findings'].append(finding)
        
        # Pattern analysis
        all_patterns = []
        for patterns in anomalies_df['detected_patterns']:
            all_patterns.extend(patterns)
        
        pattern_counts = pd.Series(all_patterns).value_counts()
        report['pattern_analysis'] = pattern_counts.to_dict()
        
        # Generate recommendations
        if 'device_error' in pattern_counts:
            report['recommendations'].append(
                f"Hardware maintenance recommended - {pattern_counts['device_error']} device errors detected"
            )
        
        if 'power_reset' in pattern_counts:
            report['recommendations'].append(
                f"Power stability check needed - {pattern_counts['power_reset']} unexpected resets"
            )
        
        if 'unable_to_dispense' in pattern_counts:
            report['recommendations'].append(
                f"Cash handling mechanism inspection required - {pattern_counts['unable_to_dispense']} dispense failures"
            )
        
        # Save report
        report_path = f"/app/output/anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Anomaly report generated: {report_path}")
    
    def scan_input_directory(self):
        """Scan for new EJ log files"""
        input_dir = "/app/input"
        processed_dir = "/app/input/processed"
        
        os.makedirs(processed_dir, exist_ok=True)
        
        for filename in os.listdir(input_dir):
            if filename.endswith('.txt') or filename.endswith('.log'):
                file_path = os.path.join(input_dir, filename)
                
                try:
                    # Process the file
                    self.process_ej_file(file_path)
                    
                    # Move to processed directory
                    os.rename(
                        file_path,
                        os.path.join(processed_dir, filename)
                    )
                    
                    logger.info(f"Successfully processed {filename}")
                    
                except Exception as e:
                    logger.error(f"Failed to process {filename}: {str(e)}")
    
    def process_realtime_session(self, session_text: str) -> dict:
        """Process a single session in real-time"""
        try:
            # Create a temporary session
            from ml_analyzer import TransactionSession
            
            session = TransactionSession(
                session_id=f"realtime_{datetime.now().timestamp()}",
                raw_text=session_text,
                start_time=datetime.now(),
                end_time=None
            )
            
            # Get embedding
            embeddings = self.detector.convert_to_embeddings([session])
            
            # Check if anomaly using existing models
            if hasattr(self.detector, 'scaler') and self.detector.scaler is not None:
                embeddings_scaled = self.detector.scaler.transform(embeddings)
                
                # Get predictions
                if_score = self.detector.isolation_forest.score_samples(embeddings_scaled)[0]
                if_pred = self.detector.isolation_forest.predict(embeddings_scaled)[0]
                
                # Normalize score
                anomaly_score = (if_score - self.detector.isolation_forest.offset_) / -self.detector.isolation_forest.offset_
                anomaly_score = max(0, min(1, anomaly_score))
                
                is_anomaly = if_pred == -1
                
                result = {
                    'session_id': session.session_id,
                    'is_anomaly': bool(is_anomaly),
                    'anomaly_score': float(anomaly_score),
                    'timestamp': datetime.now().isoformat()
                }
                
                # If anomaly, extract reasons
                if is_anomaly:
                    session.is_anomaly = True
                    session.anomaly_score = anomaly_score
                    extracted = self.detector.extract_anomaly_reasons(session)
                    result['patterns'] = extracted['detected_patterns']
                    result['critical_events'] = extracted['critical_events']
                
                return result
            else:
                # Models not trained yet
                return {
                    'session_id': session.session_id,
                    'is_anomaly': False,
                    'anomaly_score': 0.0,
                    'message': 'ML models not yet trained'
                }
                
        except Exception as e:
            logger.error(f"Error processing realtime session: {str(e)}")
            raise


def run_ml_anomaly_detection():
    """Run the ML-first anomaly detection process"""
    processor = MLFirstEJProcessor()
    processor.scan_input_directory()


def main():
    logger.info("ML-First ABM Anomaly Detector Service Started")
    
    # Schedule periodic runs
    interval = int(os.getenv('MODEL_UPDATE_INTERVAL', 3600))
    schedule.every(interval).seconds.do(run_ml_anomaly_detection)
    
    # Run once on startup
    run_ml_anomaly_detection()
    
    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()
EOF

# Create placeholder for ml_analyzer.py
cat > services/anomaly-detector/ml_analyzer.py << 'EOF'
# IMPORTANT: Replace this file with the complete ML analyzer code
# from the "ML Analyzer with Supervised Learning Integration" artifact

# Placeholder to prevent import errors
class MLFirstAnomalyDetector:
    def __init__(self):
        pass
    
    def process_ej_logs(self, file_path):
        import pandas as pd
        return pd.DataFrame({'is_anomaly': [], 'anomaly_score': []})

class TransactionSession:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

# NOTE: Copy the full implementation from the ML artifact
EOF

# Create API Dockerfile
echo "Setting up API service with expert labeling..."
cat > services/api/Dockerfile << 'EOF'
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
EOF

# Create API requirements.txt
cat > services/api/requirements.txt << 'EOF'
fastapi==0.101.0
uvicorn==0.23.2
pydantic==2.1.1
redis==4.6.0
psycopg2-binary==2.9.7
sqlalchemy==2.0.19
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
joblib==1.3.1
python-multipart==0.0.6
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
prometheus-client==0.17.1
httpx==0.24.1
python-dotenv==1.0.0
loguru==0.7.0
EOF

# Create the API main.py with expert labeling endpoints
cat > services/api/main.py << 'EOF'
# Placeholder for API - Copy the complete API code with expert labeling endpoints
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ABM ML Anomaly Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "ABM ML Anomaly Detection API", "status": "operational"}

# NOTE: Add all the expert labeling endpoints from the artifacts
EOF

# Create Dashboard Dockerfile
echo "Setting up Dashboard with expert labeling interface..."
cat > services/dashboard/Dockerfile << 'EOF'
FROM node:18-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

FROM node:18-alpine

WORKDIR /app

COPY --from=builder /app/build ./build
COPY --from=builder /app/package*.json ./

RUN npm install --production

EXPOSE 3000

CMD ["npx", "serve", "-s", "build", "-l", "3000"]
EOF

# Create Dashboard package.json
cat > services/dashboard/package.json << 'EOF'
{
  "name": "abm-ml-anomaly-dashboard",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "axios": "^1.4.0",
    "recharts": "^2.7.2",
    "lucide-react": "^0.263.1",
    "serve": "^14.2.0"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "eslintConfig": {
    "extends": [
      "react-app"
    ]
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  }
}
EOF

# Create Dashboard files
mkdir -p services/dashboard/public
cat > services/dashboard/public/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta name="description" content="ML-First ABM Anomaly Detection Dashboard" />
    <title>ABM ML Anomaly Detection</title>
    <script src="https://cdn.tailwindcss.com"></script>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>
EOF

# Create placeholder Dashboard files
cat > services/dashboard/src/App.js << 'EOF'
import React from 'react';
import Dashboard from './Dashboard';

function App() {
  return <Dashboard />;
}

export default App;
EOF

cat > services/dashboard/src/index.js << 'EOF'
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
EOF

cat > services/dashboard/src/Dashboard.js << 'EOF'
// Placeholder - Copy the complete dashboard code with expert labeling
import React from 'react';

const Dashboard = () => {
  return (
    <div className="min-h-screen bg-gray-100">
      <div className="max-w-7xl mx-auto p-6">
        <h1 className="text-3xl font-bold">ML-First ABM Anomaly Detection</h1>
        <p className="mt-4">Dashboard placeholder - Replace with full implementation</p>
      </div>
    </div>
  );
};

export default Dashboard;
EOF

cat > services/dashboard/src/ExpertLabelingInterface.js << 'EOF'
// Placeholder - Copy the complete expert labeling interface
import React from 'react';

const ExpertLabelingInterface = () => {
  return (
    <div>
      <h2>Expert Labeling Interface</h2>
      <p>Replace with full implementation</p>
    </div>
  );
};

export default ExpertLabelingInterface;
EOF

# Create Jupyter Dockerfile
echo "Setting up Jupyter service..."
cat > services/jupyter/Dockerfile << 'EOF'
FROM jupyter/scipy-notebook:latest

USER root

RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

USER $NB_UID

RUN pip install --no-cache-dir \
    torch==2.0.1 \
    transformers==4.31.0 \
    psycopg2-binary==2.9.7 \
    sqlalchemy==2.0.19 \
    redis==4.6.0 \
    plotly==5.15.0 \
    joblib==1.3.1 \
    scikit-learn==1.3.0

COPY --chown=${NB_UID}:${NB_GID} notebooks /home/jovyan/work
EOF

# Create database initialization scripts
echo "Creating database schema..."
cat > init-db/01-base-schema.sql << 'EOF'
-- Base schema for ABM anomaly detection
CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    card_number VARCHAR(20),
    transaction_type VARCHAR(50),
    amount DECIMAL(10, 2),
    status VARCHAR(20),
    error_type VARCHAR(50),
    response_time INTEGER,
    terminal_id VARCHAR(20),
    session_id VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_transactions_timestamp ON transactions(timestamp);
CREATE INDEX idx_transactions_session ON transactions(session_id);

-- Alerts table
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    anomaly_id INTEGER,
    alert_level VARCHAR(20),
    message TEXT,
    is_resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_alerts_level ON alerts(alert_level);
CREATE INDEX idx_alerts_resolved ON alerts(is_resolved);
EOF

cat > init-db/02-ml-schema.sql << 'EOF'
-- ML-specific schema
CREATE TABLE IF NOT EXISTS ml_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(50) UNIQUE NOT NULL,
    timestamp TIMESTAMP,
    session_length INTEGER,
    is_anomaly BOOLEAN DEFAULT FALSE,
    anomaly_score DECIMAL(5, 4),
    anomaly_type VARCHAR(50),
    detected_patterns JSONB,
    critical_events JSONB,
    embedding_vector BYTEA,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_ml_sessions_anomaly ON ml_sessions(is_anomaly);
CREATE INDEX idx_ml_sessions_score ON ml_sessions(anomaly_score);
CREATE INDEX idx_ml_sessions_type ON ml_sessions(anomaly_type);

-- ML anomalies table
CREATE TABLE IF NOT EXISTS ml_anomalies (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(50) REFERENCES ml_sessions(session_id),
    anomaly_type VARCHAR(50),
    anomaly_score DECIMAL(5, 4),
    cluster_id INTEGER,
    detected_patterns JSONB,
    critical_events JSONB,
    error_codes JSONB,
    model_name VARCHAR(50),
    model_version VARCHAR(20),
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Anomaly clusters
CREATE TABLE IF NOT EXISTS anomaly_clusters (
    id SERIAL PRIMARY KEY,
    cluster_id INTEGER UNIQUE NOT NULL,
    cluster_name VARCHAR(100),
    cluster_description TEXT,
    typical_patterns JSONB,
    member_count INTEGER DEFAULT 0,
    centroid_vector BYTEA,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Labeled training data
CREATE TABLE IF NOT EXISTS labeled_anomalies (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(50) REFERENCES ml_sessions(session_id),
    anomaly_label VARCHAR(100) NOT NULL,
    label_confidence DECIMAL(3, 2),
    labeled_by VARCHAR(100),
    label_reason TEXT,
    is_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_labeled_anomalies_label ON labeled_anomalies(anomaly_label);

-- ML model metadata
CREATE TABLE IF NOT EXISTS ml_models (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50),
    model_version VARCHAR(20),
    training_date TIMESTAMP,
    training_samples INTEGER,
    anomaly_threshold DECIMAL(5, 4),
    performance_metrics JSONB,
    model_parameters JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Pattern definitions
CREATE TABLE IF NOT EXISTS anomaly_patterns (
    id SERIAL PRIMARY KEY,
    pattern_name VARCHAR(100) UNIQUE NOT NULL,
    pattern_regex TEXT,
    pattern_description TEXT,
    severity_level VARCHAR(20),
    recommended_action TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert predefined patterns
INSERT INTO anomaly_patterns (pattern_name, pattern_regex, pattern_description, severity_level, recommended_action) VALUES
('supervisor_mode', 'SUPERVISOR\s+MODE\s+(ENTRY|EXIT)', 'Supervisor mode activity detected', 'MEDIUM', 'Review supervisor access logs'),
('unable_to_dispense', 'UNABLE\s+TO\s+DISPENSE', 'ATM unable to dispense cash', 'HIGH', 'Check cash cassettes and dispensing mechanism'),
('device_error', 'DEVICE\s+ERROR', 'Hardware device error', 'HIGH', 'Schedule immediate maintenance'),
('power_reset', 'POWER-UP/RESET', 'Power reset or restart', 'HIGH', 'Check power supply and UPS status'),
('cash_retract', 'CASHIN\s+RETRACT\s+STARTED', 'Cash retraction initiated', 'HIGH', 'Review deposit module functionality'),
('no_dispense', 'NO\s+DISPENSE\s+SUCCESS', 'Cash dispensing failed', 'HIGH', 'Inspect cash handling mechanism'),
('note_error', 'NOTE\s+ERROR\s+OCCURRED', 'Note processing error', 'MEDIUM', 'Check note reader and validator'),
('recovery_failed', 'RECOVERY\s+FAILED', 'Recovery operation failed', 'CRITICAL', 'Immediate technical intervention required')
ON CONFLICT (pattern_name) DO NOTHING;

-- Views for analysis
CREATE OR REPLACE VIEW ml_anomaly_summary AS
SELECT 
    DATE(s.timestamp) as date,
    COUNT(DISTINCT s.id) as total_sessions,
    COUNT(DISTINCT CASE WHEN s.is_anomaly THEN s.id END) as anomaly_count,
    AVG(CASE WHEN s.is_anomaly THEN s.anomaly_score END) as avg_anomaly_score,
    COUNT(DISTINCT a.cluster_id) as unique_clusters,
    ARRAY_AGG(DISTINCT a.anomaly_type) as anomaly_types
FROM ml_sessions s
LEFT JOIN ml_anomalies a ON s.session_id = a.session_id
GROUP BY DATE(s.timestamp);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO abm_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO abm_user;
EOF

# Create Prometheus configuration
echo "Creating monitoring configuration..."
cat > prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/api/v1/metrics'
    
  - job_name: 'anomaly-detector'
    static_configs:
      - targets: ['anomaly-detector:9091']
    metrics_path: '/metrics'
EOF

# Create Grafana configuration
cat > grafana/provisioning/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
    
  - name: PostgreSQL
    type: postgres
    url: postgres:5432
    database: abm_ml_db
    user: abm_user
    secureJsonData:
      password: secure_ml_password123
    jsonData:
      sslmode: 'disable'
EOF

# Create README.md
echo "Creating documentation..."
cat > README.md << 'EOF'
# ML-First ABM EJ Log Anomaly Detection System

A pure ML-first approach to detecting anomalies in ABM Electronic Journal logs using BERT embeddings, unsupervised learning, and expert-guided supervised learning.

## 🚀 Features

- **ML-First Detection**: Uses BERT embeddings before any regex parsing
- **Unsupervised Learning**: Isolation Forest, One-Class SVM, Autoencoders
- **Expert Labeling Interface**: Web UI for domain experts to label anomalies
- **Supervised Learning**: Trains on expert-labeled data
- **Real-time Processing**: Stream processing for live anomaly detection
- **Pattern Discovery**: Automatically finds new anomaly types
- **Explainable AI**: Shows why anomalies were detected

## 📋 Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- 16GB RAM recommended
- 30GB free disk space (for ML models)

## 🛠️ Quick Start

### 1. Run Setup Script

```bash
chmod +x setup_ml_first_abm_system.sh
./setup_ml_first_abm_system.sh
```

### 2. Add ML Components

**IMPORTANT**: You must add the actual ML code:

1. Copy the ML analyzer code to `services/anomaly-detector/ml_analyzer.py`
2. Copy the API endpoints to `services/api/main.py`
3. Copy the dashboard components to `services/dashboard/src/`

### 3. Build and Deploy

```bash
# Build all services
docker-compose build

# Start services
docker-compose up -d

# Verify status
docker-compose ps
```

### 4. Access the System

- **Dashboard**: http://localhost:3000
- **API**: http://localhost:8000/docs
- **Jupyter**: http://localhost:8888 (token: ml_jupyter_token_123)
- **Grafana**: http://localhost:3001 (admin/ml_admin)

## 📊 How It Works

### ML-First Pipeline

```
Raw Logs → Session Splitting → BERT Embeddings → ML Detection → Clustering → Expert Review → Supervised Learning
```

### No Initial Regex Parsing

- Works directly on unstructured text
- BERT understands context and semantics
- Discovers unknown patterns automatically

## 🎯 Usage

### Process EJ Logs

Drop files in `data/input/` or upload via dashboard.

### Expert Labeling

1. Navigate to "Expert Review" tab
2. Review detected anomalies
3. Label or exclude false positives
4. Train supervised model

### API Examples

```bash
# Upload EJ log file
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@abm_logs.txt"

# Get anomalies for labeling
curl http://localhost:8000/api/v1/expert/anomalies

# Train supervised model
curl -X POST http://localhost:8000/api/v1/expert/train-supervised
```

## 🔧 Configuration

Edit `.env` file for customization:
- `ANOMALY_THRESHOLD`: Detection sensitivity
- `BERT_MODEL`: Pre-trained model to use
- Database credentials

## 📈 Monitoring

- Grafana dashboards at http://localhost:3001
- Real-time metrics and alerts
- Model performance tracking

## 🐛 Troubleshooting

Check logs:
```bash
docker-compose logs -f anomaly-detector
docker-compose logs -f api
```

## 📄 License

MIT License
EOF

# Create Makefile
echo "Creating Makefile..."
cat > Makefile << 'EOF'
.PHONY: help build up down logs clean test components

help:
	@echo "ML-First ABM Anomaly Detection System"
	@echo ""
	@echo "Available commands:"
	@echo "  make build      - Build all Docker images"
	@echo "  make up         - Start all services"
	@echo "  make down       - Stop all services"
	@echo "  make logs       - View logs"
	@echo "  make clean      - Clean up volumes and images"
	@echo "  make test       - Run tests"
	@echo "  make components - Show status of required components"

build:
	docker-compose build

up:
	docker-compose up -d
	@echo ""
	@echo "Services starting..."
	@echo "Dashboard: http://localhost:3000"
	@echo "API: http://localhost:8000/docs"
	@echo "Jupyter: http://localhost:8888 (token: ml_jupyter_token_123)"
	@echo "Grafana: http://localhost:3001 (admin/ml_admin)"

down:
	docker-compose down

logs:
	docker-compose logs -f

clean:
	docker-compose down -v
	docker system prune -f

test:
	@echo "Checking service health..."
	@curl -s http://localhost:8000/api/v1/health || echo "API not ready"
	@docker exec abm-ml-postgres pg_isready || echo "Database not ready"

components:
	@echo "Checking required ML components..."
	@echo -n "ml_analyzer.py: "
	@if [ -f "services/anomaly-detector/ml_analyzer.py" ] && [ -s "services/anomaly-detector/ml_analyzer.py" ]; then echo "✓ Present"; else echo "✗ Missing - Add ML analyzer code"; fi
	@echo -n "API endpoints: "
	@if grep -q "expert/anomalies" services/api/main.py 2>/dev/null; then echo "✓ Present"; else echo "✗ Missing - Add expert labeling endpoints"; fi
	@echo -n "Dashboard components: "
	@if grep -q "ExpertLabelingInterface" services/dashboard/src/Dashboard.js 2>/dev/null; then echo "✓ Present"; else echo "✗ Missing - Add dashboard components"; fi
EOF

# Create sample test data generator
echo "Creating test data generator..."
cat > generate_test_abm_logs.py << 'EOF'
#!/usr/bin/env python3
"""Generate test ABM EJ logs with various anomaly patterns"""

import random
from datetime import datetime, timedelta

def generate_test_logs(filename="test_abm_logs.txt", num_sessions=100):
    """Generate test ABM logs with anomalies from the requirements"""
    
    with open(filename, 'w') as f:
        base_time = datetime.now() - timedelta(days=7)
        
        for i in range(num_sessions):
            session_time = base_time + timedelta(hours=i)
            
            # Normal transaction template
            transaction = f"""*{i:03d}*{session_time.strftime('%m/%d/%Y')}*{session_time.strftime('%H:%M')}*
     *TRANSACTION START*
CARD INSERTED
{session_time.strftime('%H:%M:%S')} ATR RECEIVED T=0
{(session_time + timedelta(seconds=5)).strftime('%H:%M:%S')} PIN ENTERED
{(session_time + timedelta(seconds=10)).strftime('%H:%M:%S')} OPCODE = BBD
GENAC 1 : ARQC
GENAC 2 : TC

  PAN 0004263********{random.randint(1000, 9999)}
  ---START OF TRANSACTION---
  
       N.C.B. MIDAS
   NCB RUBIS BORDER AVE
     DATE        TIME
   {session_time.strftime('%Y/%m/%d')}   {session_time.strftime('%H:%M:%S')}
   MACHINE       0163
   TRAN NO       {100000 + i}
"""
            
            # Add anomalies based on examples
            if i % 20 == 0:  # Unable to dispense
                transaction += "   UNABLE TO DISPENSE\n"
            elif i % 25 == 1:  # Supervisor mode after transaction
                transaction += f"{(session_time + timedelta(seconds=20)).strftime('%H:%M:%S')} TRANSACTION END\n"
                transaction += "SUPERVISOR MODE ENTRY\nSUPERVISOR MODE EXIT\n"
                continue
            elif i % 30 == 2:  # Power reset after transaction
                transaction += f"{(session_time + timedelta(seconds=20)).strftime('%H:%M:%S')} TRANSACTION END\n"
                transaction += "[05pPOWER-UP/RESET\nAPTRA ADVANCE NDC 05.01.00[00p\n"
                continue
            elif i % 35 == 3:  # Cash retract error
                transaction += """A/C
DEVICE ERROR
ESC: 000
VAL: 000
REF: 000
REJECTS:000
CASHIN RETRACT STARTED - RETRACT BIN
"""
            elif i % 40 == 4:  # Long delay in note taking
                transaction += f"{(session_time + timedelta(seconds=20)).strftime('%H:%M:%S')} NOTES PRESENTED 0,0,0,6\n"
                transaction += f"{(session_time + timedelta(seconds=35)).strftime('%H:%M:%S')} NOTES TAKEN\n"
            else:  # Normal completion
                transaction += f"   WITHDRAWAL    {random.choice([50000, 100000, 200000])}.00\n"
                transaction += "   FROM CHEQUING\n"
            
            transaction += "         THANK YOU\n"
            transaction += f"{(session_time + timedelta(seconds=25)).strftime('%H:%M:%S')} CARD TAKEN\n"
            transaction += f"{(session_time + timedelta(seconds=30)).strftime('%H:%M:%S')} TRANSACTION END\n"
            
            f.write(transaction)
            f.write("\n")
    
    print(f"Generated {num_sessions} test sessions in {filename}")
    print("Anomaly types included:")
    print("- Unable to dispense")
    print("- Supervisor mode after transaction")
    print("- Power reset after transaction")
    print("- Cash retract with device error")
    print("- Long delay between notes presented and taken")

if __name__ == "__main__":
    generate_test_logs()
EOF

chmod +x generate_test_abm_logs.py

# Create component integration script
cat > integrate_components.sh << 'EOF'
#!/bin/bash
# Script to help integrate the ML components

echo "ML-First ABM Component Integration Helper"
echo "========================================"
echo ""
echo "To complete the setup, you need to add the following components:"
echo ""
echo "1. ML Analyzer (services/anomaly-detector/ml_analyzer.py):"
echo "   - Copy the complete code from 'ML Analyzer with Supervised Learning Integration'"
echo ""
echo "2. API Endpoints (services/api/main.py):"
echo "   - Copy all the expert labeling endpoints"
echo "   - Include the complete API implementation"
echo ""
echo "3. Dashboard Components:"
echo "   - Copy Dashboard.js with all tabs including expert-labeling"
echo "   - Copy ExpertLabelingInterface.js component"
echo ""
echo "After adding these components, run:"
echo "  make build"
echo "  make up"
echo ""
echo "To verify components are present:"
echo "  make components"
EOF

chmod +x integrate_components.sh

# Final message
echo ""
echo "================================================================"
echo "✅ ML-First ABM Anomaly Detection System Setup Complete!"
echo "================================================================"
echo ""
echo "📁 Project created in: $PROJECT_NAME"
echo ""
echo "🔧 IMPORTANT NEXT STEPS:"
echo ""
echo "1. ADD THE ML COMPONENTS:"
echo "   ./integrate_components.sh"
echo ""
echo "2. BUILD THE SYSTEM:"
echo "   cd $PROJECT_NAME"
echo "   make build"
echo ""
echo "3. START SERVICES:"
echo "   make up"
echo ""
echo "4. GENERATE TEST DATA:"
echo "   python3 generate_test_abm_logs.py"
echo "   cp test_abm_logs.txt data/input/"
echo ""
echo "5. ACCESS THE SYSTEM:"
echo "   - Dashboard: http://localhost:3000"
echo "   - API Docs: http://localhost:8000/docs"
echo "   - Expert Labeling: http://localhost:3000 (Expert Review tab)"
echo ""
echo "📚 For help: make help"
echo "================================================================"