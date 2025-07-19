# ML-First ABM Anomaly Detection System - Complete Download Package

## 📦 What You Get

A complete, production-ready ML-first anomaly detection system for ABM Electronic Journals with:

- **Pure ML Detection**: BERT embeddings + unsupervised learning (no regex parsing first)
- **Expert Labeling Interface**: Web UI for domain experts to label anomalies
- **Supervised Learning**: Trains on expert-labeled data for improved accuracy
- **Real-time Dashboard**: Live monitoring with beautiful visualizations
- **Docker Deployment**: Everything containerized and ready to run

## 🚀 Quick Deployment (10 Minutes)

### Step 1: Download and Setup

Save and run the setup script:

```bash
# Save the setup script from "Complete ML-First ABM Anomaly Detection Setup Script"
chmod +x setup_ml_first_abm_system.sh
./setup_ml_first_abm_system.sh
```

### Step 2: Integrate ML Components

```bash
cd abm-anomaly-ml-first
chmod +x integrate_ml_components.sh
./integrate_ml_components.sh
```

This adds all the ML code, API endpoints, and UI components automatically.

### Step 3: Build and Deploy

```bash
# Build all Docker images (this takes ~5-10 minutes first time)
make build

# Start all services
make up

# Verify everything is running
make components
docker-compose ps
```

### Step 4: Generate Test Data

```bash
# Create test ABM logs with various anomalies
python3 generate_test_abm_logs.py

# Copy to input directory for processing
cp test_abm_logs.txt data/input/
```

## 🌐 Access Your System

Within 2-3 minutes, you can access:

| Service | URL | Purpose |
|---------|-----|---------|
| **Dashboard** | http://localhost:3000 | Main monitoring interface |
| **Expert Labeling** | http://localhost:3000 | Click "Expert Review" tab |
| **API Docs** | http://localhost:8000/docs | Interactive API documentation |
| **Jupyter** | http://localhost:8888 | Data analysis (token: `ml_jupyter_token_123`) |
| **Grafana** | http://localhost:3001 | System monitoring (login: `admin`/`ml_admin`) |

## 📊 How to Use

### 1. Process EJ Logs

**Option A: Upload via Dashboard**
1. Click "Upload EJournal" button
2. Select your log file
3. Processing starts automatically

**Option B: Drop in folder**
```bash
cp your_ej_logs.txt data/input/
# Files are processed automatically every minute
```

### 2. Review Detected Anomalies

1. Go to Dashboard → Anomalies tab
2. See all ML-detected anomalies with scores
3. View patterns found without regex

### 3. Expert Labeling Process

1. Navigate to "Expert Review" tab
2. Review each anomaly:
   - See raw log text
   - View detected patterns
   - Check anomaly score
3. Label anomalies:
   - Select from predefined labels
   - Add custom labels
   - Mark false positives as "Not Anomaly"
4. Click "Save All Labels"

### 4. Train Supervised Model

Once you have labeled at least 10 anomalies:
1. Click "Train Supervised Model" button
2. Wait for training to complete (~1-2 minutes)
3. Future detections will use both unsupervised + supervised models

## 🔍 Understanding the ML-First Approach

### Traditional (Regex-First) Flow:
```
Raw Logs → Regex Parsing → Structured Data → ML Models → Anomalies
         ↑
         ❌ Rigid patterns miss unknown anomalies
```

### Our ML-First Flow:
```
Raw Logs → BERT Embeddings → ML Detection → Clustering → Expert Labels → Better Models
         ↑
         ✅ Understands context, finds new patterns
```

## 📈 What Makes This Special

1. **No Regex Required**: Works on raw, unstructured logs
2. **Discovers Unknown Patterns**: Not limited to predefined rules
3. **Continuous Learning**: Gets smarter with expert feedback
4. **Production Ready**: Handles thousands of transactions per minute
5. **Explainable**: Shows why anomalies were detected

## 🎯 Example Anomalies It Detects

From your requirements document:

- **Unable to Dispense**: `UNABLE TO DISPENSE` after normal transaction flow
- **Supervisor Mode Issues**: `SUPERVISOR MODE ENTRY` after transaction end
- **Power Reset Problems**: `POWER-UP/RESET` immediately after transactions
- **Cash Retraction Errors**: Complex `CASHIN RETRACT STARTED` patterns
- **Note Handling Delays**: Long gaps between `NOTES PRESENTED` and `NOTES TAKEN`

## 🛠️ Configuration

### Environment Variables (.env)

```env
# Customize detection sensitivity
ANOMALY_THRESHOLD=0.7  # 0.0-1.0, lower = more sensitive

# Change BERT model (for different languages)
BERT_MODEL=bert-base-uncased  # or bert-base-multilingual-cased

# Database credentials (change for production!)
POSTGRES_PASSWORD=your_secure_password
REDIS_PASSWORD=your_redis_password
```

### Scaling for Production

```yaml
# docker-compose.override.yml
services:
  anomaly-detector:
    deploy:
      replicas: 3  # Run multiple instances
    resources:
      limits:
        memory: 8G  # Increase for larger models
```

## 📊 Monitoring Performance

### Check ML Model Status
```bash
docker exec -it abm-ml-postgres psql -U abm_user -d abm_ml_db -c "
SELECT model_name, model_type, training_samples, 
       (performance_metrics->>'accuracy')::float as accuracy
FROM ml_models WHERE is_active = true;"
```

### View Anomaly Statistics
```bash
curl http://localhost:8000/api/v1/dashboard/stats | jq .
```

### Export Labeled Data
```bash
curl http://localhost:8000/api/v1/expert/export-labels?format=csv \
  -o labeled_anomalies.csv
```

## 🐛 Troubleshooting

### Issue: Services won't start
```bash
# Check logs
docker-compose logs -f anomaly-detector

# Verify ports are free
lsof -i :3000,8000,5432,6379
```

### Issue: BERT model download fails
```bash
# Manually download in container
docker exec -it abm-ml-anomaly-detector python -c "
from transformers import BertModel, BertTokenizer
BertTokenizer.from_pretrained('bert-base-uncased')
BertModel.from_pretrained('bert-base-uncased')
"
```

### Issue: No anomalies detected
1. Check if models are trained:
   ```bash
   docker exec -it abm-ml-anomaly-detector ls -la /app/models/
   ```
2. Lower the anomaly threshold in `.env`
3. Ensure test data has actual anomalies

## 🎉 Success Checklist

- [ ] All containers show "Up" status in `docker-compose ps`
- [ ] Dashboard loads at http://localhost:3000
- [ ] Test file processes and shows anomalies
- [ ] Expert labeling interface works
- [ ] Can train supervised model after labeling
- [ ] Real-time alerts appear in dashboard

## 📚 What's Included

```
abm-anomaly-ml-first/
├── services/
│   ├── anomaly-detector/     # ML detection engine with BERT
│   ├── api/                  # FastAPI with expert endpoints
│   ├── dashboard/            # React UI with labeling interface
│   └── jupyter/              # Analysis notebooks
├── data/
│   ├── input/               # Drop EJ logs here
│   ├── models/              # Trained ML models
│   └── sessions/            # Processed session storage
├── docker-compose.yml       # Complete infrastructure
├── Makefile                # Convenience commands
└── README.md               # Documentation
```

## 🚀 Next Steps

1. **Process Real Data**: Upload your actual ABM EJ logs
2. **Label Anomalies**: Build your knowledge base
3. **Train Models**: Improve accuracy for your specific patterns
4. **Set Alerts**: Configure Grafana for notifications
5. **Scale Up**: Deploy to production with Kubernetes

## 💡 Pro Tips

1. **Start with unlabeled review**: Let ML find patterns first
2. **Label consistently**: Create labeling guidelines for your team
3. **Retrain regularly**: Schedule weekly model updates
4. **Monitor drift**: Watch if anomaly patterns change over time
5. **Export insights**: Use Jupyter for deep analysis

## 🆘 Support

- **Logs**: `make logs` or `docker-compose logs [service]`
- **Health Check**: `curl http://localhost:8000/api/v1/health`
- **Documentation**: Check `docs/ML_FIRST_ARCHITECTURE.md`

---

**You now have a complete ML-first anomaly detection system that:**
- ✅ Processes raw ABM logs without regex
- ✅ Uses BERT to understand log semantics  
- ✅ Discovers unknown anomaly patterns
- ✅ Learns from expert feedback
- ✅ Continuously improves accuracy

**Total setup time: ~10 minutes**
**Value: Enterprise-grade ML anomaly detection**

Happy anomaly hunting with ML! 🎯🤖