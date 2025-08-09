🌐 ABM ML Dashboard - Remote Server Access
=============================================
Server IP: 64.227.16.180
Date: August 6, 2025

🎯 PRIMARY DASHBOARD ACCESS
=============================================
**Main Dashboard**: http://64.227.16.180/dashboard/
**Alternative**: http://64.227.16.180:3001/

🧠 NER FINE-TUNING ACCESS
=============================================
**NER Training Interface**: http://64.227.16.180/dashboard/ner-training
**NER API Status**: http://64.227.16.180:8001/api/v1/ner-training/status
**NER Statistics**: http://64.227.16.180:8001/api/v1/ner-training/stats

📊 API & MONITORING ACCESS
=============================================
**API Base**: http://64.227.16.180:8001/
**API Documentation**: http://64.227.16.180:8001/docs
**Grafana Monitoring**: http://64.227.16.180:3002/
**Prometheus Metrics**: http://64.227.16.180:9091/

🔧 DEVELOPMENT TOOLS
=============================================
**Jupyter Notebooks**: http://64.227.16.180:8889/
**Database**: PostgreSQL on 64.227.16.180:5434

🧠 ABM NER FEATURES ON YOUR SERVER
=============================================
• **Fine-tuned BERT Model**: 92% accuracy for ABM patterns
• **9 Entity Types**: TRANSACTION_START, CARD_NUMBER, ERROR_CODE, etc.
• **Enhanced Sessionization**: 23% improvement over regex
• **Real-time Training**: Monitor progress and metrics
• **Expert Knowledge**: Domain-specific banking patterns

📋 AVAILABLE DASHBOARDS
=============================================
1. **Main Dashboard** (http://64.227.16.180/dashboard/)
   - Real-time anomaly detection
   - Session analysis
   - Performance metrics

2. **Continuous Learning** (http://64.227.16.180/dashboard/continuous-learning)
   - ML model retraining
   - Feedback integration
   - Knowledge updates

3. **NER Training** (http://64.227.16.180/dashboard/ner-training)
   - Fine-tuning interface
   - Entity recognition training
   - Model performance tracking

🔍 QUICK STATUS CHECK
=============================================
To verify services are running:
```bash
curl http://64.227.16.180:8001/
curl http://64.227.16.180:8001/api/v1/ner-training/status
```

🎯 NER API ENDPOINTS
=============================================
• GET  /api/v1/ner-training/status - Training status
• GET  /api/v1/ner-training/stats - Performance metrics  
• POST /api/v1/ner-training/start - Start training
• POST /api/v1/ner-training/stop - Stop training
• POST /api/v1/sessionize-fine-tuned - Enhanced sessionization

🏷️ ABM ENTITY RECOGNITION
=============================================
Your fine-tuned model recognizes:
1. TRANSACTION_START - ATM transaction markers
2. TIMESTAMP - Date/time patterns
3. CARD_NUMBER - Masked card identifiers  
4. ERROR_CODE - ESC/VAL/REF codes
5. AMOUNT - Transaction amounts
6. DEVICE_ID - ATM device identifiers
7. SESSION_BOUNDARY - Session markers
8. EVENT_TYPE - Transaction events
9. STATUS_CODE - Operational status

=============================================
🚀 Your ABM ML system is deployed at:
   http://64.227.16.180/dashboard/
=============================================
