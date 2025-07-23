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
docker compose logs -f anomaly-detector
docker compose logs -f api
```

## 📄 License

MIT License


docker compose logs -f anomaly-detector 
docker compose up -d anomaly-detector 
docker compose down anomaly-detector  

docker compose restart api dashboard
docker compose restart anomaly-detector
./clear_ml_sessions.sh

git push -u origin feature/cp
git push -u origin feature/cp


docker compose -f docker-compose-flyway.yml up

docker compose stop grafana jupyter prometheus
docker compose rm grafana jupyter prometheus

sudo chown -R deploy:deploy /home/deploy/Capstone-Project/abm-anomaly-ml-first



docker exec -it abm-ml-dashboard sh
ls /usr/share/nginx/html
ls /usr/share/nginx/html/static/js
cat /usr/share/nginx/html/index.html | grep main

docker exec -it abm-ml-nginx

docker exec -it abm-ml-postgres psql -U abm_user -d abm_ml_db
docker exec -i abm-ml-postgres psql -U abm_user -d abm_ml_db < ./data/db/schema/updated_schema.sql

docker exec -i abm-ml-postgres psql -U abm_user -d abm_ml_db < ./database/migrations/002_multi_anomaly_support.sql
docker exec -i abm-ml-postgres psql -U abm_user -d abm_ml_db < ./database/migrations/003_fix_missing_schema.sql

the below should have been an anomaly, the customer basically attempted a transaction and it appears to show nothing happening.
why were they not flagged as anomalies?
<txn1>
[020t15706/18/202513:39
TRANSACTION START
[020t CARD INSERTED
[020t 13:39:56 CARD TAKEN
[000p[040q(I 75561D(10,M-090B0210B9,R-4S
[000p[040q(I 75561D(10,M-00,R-4S
[020t 13:39:56 TRANSACTION END
[020t15806/18/202513:39
PRIMARY CARD READER ACTIVATED
</txn1>

 <txn2>
 [020t*209*06/18/2025*14:23*
      *TRANSACTION START*
 [020t CARD INSERTED
  14:23:03 ATR RECEIVED T=0
 [020t 14:23:06 OPCODE = FI      
 
   PAN 0004263********6687
   ---START OF TRANSACTION---
  
 [020t 14:23:22 PIN ENTERED
 [020t 14:23:36 OPCODE = BC      
 
   PAN 0004263********6687
   ---START OF TRANSACTION---
  
 [020t 14:24:28 CARD TAKEN
 [020t 14:24:29 TRANSACTION END
 [020t*210*06/18/2025*14:24*
      *PRIMARY CARD READER ACTIVATED*
 </txn2>