# ABM Anomaly Detection System - Main Implementation Guide

## 📋 **Overview**

This document provides a comprehensive technical analysis of the anomaly detection implementation in `main.py` from the anomaly-detector service. The system implements an ML-first approach for detecting anomalies in ABM (Automated Banking Machine) transaction logs using ensemble machine learning techniques.

---

## 🏗️ **System Architecture**

### **Core Components**

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   MLFirstEJProcessor │───▶│  MLFirstAnomalyDet  │───▶│   Database Storage  │
│   (main.py)         │    │  (ml_analyzer.py)   │    │   & Redis Cache     │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
         │                          │                          │
         ▼                          ▼                          ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   File Processing   │    │   ML Model Pipeline │    │   Alert Generation  │
│   & Monitoring      │    │   & Predictions     │    │   & Reporting       │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

### **MLFirstEJProcessor Class**

The main orchestrator that coordinates all anomaly detection activities:

```python
class MLFirstEJProcessor:
    """Main processor for ML-first anomaly detection"""
    
    def __init__(self):
        # Database connection (PostgreSQL)
        self.db_engine = create_engine(...)
        
        # Redis connection (for real-time updates)
        self.redis_client = redis.Redis(...)
        
        # Initialize ML detector with database connection
        self.detector = MLFirstAnomalyDetector(db_engine=self.db_engine)
        
        # Load existing models if available
        self.load_models()
```

---

## 🔧 **Anomaly Detection Pipeline**

### **1. File Processing Workflow**

```python
def process_ej_file(self, file_path: str):
    """Process an EJ log file using ML-first approach"""
    
    # 1. Check for duplicate processing
    if self.should_skip_file(file_path):
        return
    
    # 2. Run ML-first detection pipeline
    results_df = self.detector.process_ej_logs(file_path)
    
    # 3. Store results in database
    self.store_sessions(results_df)
    
    # 4. Process anomalies specifically
    anomalies_df = results_df[results_df['is_anomaly']]
    if len(anomalies_df) > 0:
        self.store_anomalies(anomalies_df)
        self.generate_alerts(anomalies_df)
    
    # 5. Publish real-time updates
    self.publish_updates(results_df)
    
    # 6. Save updated models
    self.detector.save_models("/app/models")
    
    # 7. Generate comprehensive report
    self.generate_anomaly_report(anomalies_df)
```

### **2. ML Model Management**

#### **A. Model Loading Strategy**

```python
def load_models(self):
    """Load pre-trained models if they exist"""
    model_dir = "/app/models"
    
    if os.path.exists(os.path.join(model_dir, "isolation_forest.pkl")):
        # Load ensemble ML models
        self.detector.isolation_forest = joblib.load("isolation_forest.pkl")
        self.detector.one_class_svm = joblib.load("one_class_svm.pkl")
        self.detector.scaler = joblib.load("scaler.pkl")
        self.detector.pca = joblib.load("pca.pkl")  # Optional
    else:
        # Will train new models on first batch
        logger.info("No existing models found. Will train on first batch.")
```

**Models Used:**
- **Isolation Forest**: For multivariate outlier detection
- **One-Class SVM**: For boundary-based anomaly detection  
- **StandardScaler**: For feature normalization
- **PCA**: For dimensionality reduction (optional)

#### **B. Real-time Processing Capability**

```python
def process_realtime_session(self, session_text: str) -> dict:
    """Process a single session in real-time"""
    
    # Create temporary session object
    session = TransactionSession(
        session_id=f"realtime_{datetime.now().timestamp()}",
        raw_text=session_text,
        start_time=datetime.now()
    )
    
    # Convert to embeddings
    embeddings = self.detector.convert_to_embeddings([session])
    
    # Scale features
    embeddings_scaled = self.detector.scaler.transform(embeddings)
    
    # Get anomaly predictions
    if_score = self.detector.isolation_forest.score_samples(embeddings_scaled)[0]
    if_pred = self.detector.isolation_forest.predict(embeddings_scaled)[0]
    
    # Normalize anomaly score
    anomaly_score = (if_score - self.detector.isolation_forest.offset_) / -self.detector.isolation_forest.offset_
    anomaly_score = max(0, min(1, anomaly_score))
    
    is_anomaly = if_pred == -1
    
    return {
        'session_id': session.session_id,
        'is_anomaly': bool(is_anomaly),
        'anomaly_score': float(anomaly_score),
        'timestamp': datetime.now().isoformat()
    }
```

---

## 💾 **Data Storage & Management**

### **1. Session Storage with Multi-Anomaly Support**

```python
def store_sessions(self, results_df: pd.DataFrame):
    """Store all sessions in database with embeddings and multi-anomaly support"""
    
    for i, (_, row) in enumerate(results_df.iterrows()):
        # Extract session data
        embedding = self.detector.sessions[i].embedding
        session_id = row['session_id']
        raw_text = self.detector.sessions[i].raw_text
        
        session_data = {
            'session_id': session_id,
            'timestamp': row['start_time'],
            'session_length': row['session_length'],
            'is_anomaly': row['is_anomaly'],
            'anomaly_score': row['anomaly_score'],
            'anomaly_type': row['anomaly_type'],
            'detected_patterns': json.dumps(row['detected_patterns']),
            'critical_events': json.dumps(row['critical_events']),
            'embedding_vector': embedding.tobytes(),
            'raw_text': raw_text,
            
            # Multi-anomaly support fields
            'anomaly_count': row.get('anomaly_count', 0),
            'anomaly_types': json.dumps(row.get('anomaly_types', [])),
            'max_severity': row.get('max_severity', 'normal'),
            'overall_anomaly_score': row.get('overall_anomaly_score', 0.0),
            'critical_anomalies_count': row.get('critical_anomalies_count', 0),
            'high_severity_anomalies_count': row.get('high_severity_anomalies_count', 0),
            'detection_methods': json.dumps(row.get('detection_methods', [])),
            'anomalies_detail': json.dumps(row.get('anomalies_detail', [])),
            
            'created_at': datetime.now()
        }
```

### **2. Conflict Resolution Strategy**

```python
def store_sessions_with_conflict_resolution(self, sessions_data: List[Dict]):
    """Store sessions individually with conflict resolution"""
    
    for session_data in sessions_data:
        # Check if session already exists
        check_query = text("SELECT COUNT(*) FROM ml_sessions WHERE session_id = :session_id")
        
        if exists:
            # Update existing session with new data
            update_query = text("""UPDATE ml_sessions SET ...""")
        else:
            # Insert new session
            insert_query = text("""INSERT INTO ml_sessions (...) VALUES (...)""")
```

---

## 🚨 **Alert Generation System**

### **1. Dynamic Alert Level Determination**

```python
def generate_alerts(self, anomalies_df: pd.DataFrame):
    """Generate alerts for detected anomalies"""
    
    for _, anomaly in anomalies_df.iterrows():
        # Base alert level from anomaly score
        alert_level = 'LOW'
        if anomaly['anomaly_score'] > 0.8:
            alert_level = 'HIGH'
        elif anomaly['anomaly_score'] > 0.6:
            alert_level = 'MEDIUM'
        
        # Critical pattern escalation
        critical_patterns = [
            'unable_to_dispense', 
            'device_error', 
            'power_reset',
            'cash_retract',
            'recovery_failed'
        ]
        
        if any(pattern in anomaly['detected_patterns'] for pattern in critical_patterns):
            alert_level = 'HIGH'
```

### **2. Human-Readable Alert Descriptions**

```python
def generate_alert_description(self, anomaly):
    """Generate human-readable description of the anomaly"""
    
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
    
    descriptions = []
    for pattern in anomaly['detected_patterns']:
        if pattern in pattern_descriptions:
            descriptions.append(pattern_descriptions[pattern])
    
    return '; '.join(descriptions) if descriptions else 'Anomalous pattern detected'
```

### **3. Real-time Alert Publishing**

```python
# Publish real-time alert via Redis
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
```

---

## 📊 **Real-time Dashboard Updates**

### **1. Summary Statistics Publishing**

```python
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
    summary['anomaly_types'] = anomaly_types.to_dict()
    
    # Pattern frequency analysis
    all_patterns = []
    for patterns in results_df[results_df['is_anomaly']]['detected_patterns']:
        all_patterns.extend(patterns)
    
    if all_patterns:
        pattern_counts = pd.Series(all_patterns).value_counts().head(5)
        summary['pattern_summary'] = pattern_counts.to_dict()
    
    # Publish to Redis channels
    self.redis_client.publish('dashboard_updates', json.dumps(summary))
    self.redis_client.setex('latest_ml_summary', 3600, json.dumps(summary))
```

---

## 📈 **Comprehensive Reporting System**

### **1. Anomaly Analysis Report**

```python
def generate_anomaly_report(self, anomalies_df: pd.DataFrame):
    """Generate detailed anomaly report"""
    
    report = {
        'report_timestamp': datetime.now().isoformat(),
        'total_anomalies': len(anomalies_df),
        'anomaly_breakdown': {},
        'critical_findings': [],
        'pattern_analysis': {},
        'recommendations': []
    }
    
    # Critical findings extraction
    for _, anomaly in anomalies_df.iterrows():
        if anomaly['anomaly_score'] > 0.8:
            finding = {
                'session_id': anomaly['session_id'],
                'score': float(anomaly['anomaly_score']),
                'events': anomaly['critical_events']
            }
            report['critical_findings'].append(finding)
```

### **2. Actionable Recommendations**

```python
# Generate maintenance recommendations
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
```

### **3. Enhanced Analysis Integration**

```python
# Add comprehensive anomaly summary if available
try:
    if hasattr(self, 'detector') and self.detector:
        anomaly_summary = self.detector.generate_anomaly_summary_report()
        if anomaly_summary:
            report['comprehensive_analysis'] = anomaly_summary
            logger.info("Added comprehensive anomaly analysis to report")
except Exception as e:
    logger.warning(f"Could not generate comprehensive anomaly summary: {e}")
```

---

## ⚡ **Automated Processing Pipeline**

### **1. Directory Monitoring**

```python
def scan_input_directory(self):
    """Scan for new EJ log files"""
    input_dir = "/app/input"
    processed_dir = "/app/input/processed"
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt') or filename.endswith('.log'):
            file_path = os.path.join(input_dir, filename)
            
            try:
                # Check if file should be skipped
                if self.should_skip_file(file_path):
                    continue
                
                # Process the file
                self.process_ej_file(file_path)
                
                # Move to processed directory
                os.rename(file_path, os.path.join(processed_dir, filename))
                
                logger.info(f"Successfully processed {filename}")
                
            except Exception as e:
                logger.error(f"Failed to process {filename}: {str(e)}")
```

### **2. Duplicate Prevention**

```python
def should_skip_file(self, file_path: str) -> bool:
    """Check if file has already been processed recently"""
    
    # Extract file identifier for pattern matching
    file_match = re.search(r'ABM(\d+)EJ_(\d{8})_(\d{8})', file_name)
    if file_match:
        abm_num = file_match.group(1)
        start_date = file_match.group(2)
        file_pattern = f"ABM{abm_num}_{start_date}%"
        
        # Check if processed in last 24 hours
        check_query = text("""
            SELECT COUNT(*) FROM ml_sessions 
            WHERE session_id LIKE :file_pattern 
            AND created_at > NOW() - INTERVAL '24 hours'
        """)
        
        if count > 0:
            logger.info(f"Skipping {file_name} - already processed {count} sessions")
            return True
    
    return False
```

### **3. Scheduled Execution**

```python
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
```

---

## 🔧 **Key Features & Capabilities**

### **1. ML-First Architecture**
- **Primary Detection**: Uses ensemble ML models (Isolation Forest + One-Class SVM)
- **Feature Engineering**: Automatic embedding generation from transaction text
- **Model Persistence**: Saves and loads trained models between runs

### **2. Multi-Anomaly Session Support**
- **Complex Transactions**: Can detect multiple anomaly types within a single session
- **Severity Hierarchies**: Automatic severity escalation based on anomaly types
- **Detailed Metadata**: Rich anomaly descriptions with contextual information

### **3. Real-time Processing**
- **Live Analysis**: Can process individual sessions immediately
- **Low Latency**: Optimized for immediate anomaly detection
- **Streaming Updates**: Real-time dashboard updates via Redis

### **4. Comprehensive Storage**
- **Full Session Data**: Stores raw text, embeddings, and analysis results
- **Conflict Resolution**: Handles duplicate processing gracefully
- **Multi-dimensional Analysis**: Supports complex querying and reporting

### **5. Intelligent Alerting**
- **Dynamic Prioritization**: Alert levels based on anomaly scores and patterns
- **Business Context**: Human-readable explanations with domain knowledge
- **Actionable Recommendations**: Specific maintenance and investigation suggestions

---

## 📋 **Data Flow Summary**

```
1. 📁 File Detection
   └── Scan /app/input directory for new EJ log files
   
2. 🔍 Duplicate Check
   └── Verify file hasn't been processed in last 24 hours
   
3. 🤖 ML Processing
   └── MLFirstAnomalyDetector.process_ej_logs()
   
4. 💾 Data Storage
   ├── Store sessions with embeddings in PostgreSQL
   ├── Store individual anomalies with metadata
   └── Save session raw text to filesystem
   
5. 🚨 Alert Generation
   ├── Determine alert levels based on scores/patterns
   ├── Generate human-readable descriptions
   └── Publish real-time alerts via Redis
   
6. 📊 Dashboard Updates
   ├── Calculate summary statistics
   ├── Analyze pattern frequencies
   └── Publish to Redis for real-time dashboard
   
7. 📈 Report Generation
   ├── Create detailed anomaly analysis
   ├── Generate actionable recommendations
   └── Save comprehensive JSON reports
   
8. 🔄 Model Persistence
   └── Save updated ML models for future use
```

---

## 🎯 **Key Innovations**

### **1. ML-First Approach**
- **Reduced Rule Dependency**: Minimizes hard-coded detection rules
- **Adaptive Learning**: Models improve with new data automatically
- **Contextual Understanding**: Uses embeddings to capture semantic meaning

### **2. Multi-Modal Detection**
- **Ensemble Voting**: Combines multiple ML algorithms for robust detection
- **Multi-Anomaly Support**: Single session can have multiple anomaly types
- **Severity-Aware Processing**: Automatic escalation based on business impact

### **3. Real-time Capabilities**
- **Streaming Processing**: Can analyze sessions as they arrive
- **Live Dashboard**: Real-time updates for operational monitoring
- **Immediate Alerting**: Critical issues flagged instantly

### **4. Production-Ready Design**
- **Scalable Architecture**: Handles high-volume transaction streams
- **Fault Tolerance**: Graceful error handling and recovery
- **Operational Intelligence**: Comprehensive logging and monitoring

---

## 🔮 **Future Enhancement Opportunities**

### **1. Advanced ML Integration**
- **Transformer Models**: Integration of BERT/GPT for better text understanding
- **Time Series Analysis**: Temporal pattern detection across sessions
- **Federated Learning**: Multi-institution model training without data sharing

### **2. Enhanced Analytics**
- **Predictive Modeling**: Forecasting potential anomalies before they occur
- **Root Cause Analysis**: Automated investigation of anomaly patterns
- **Business Impact Assessment**: Financial impact quantification

### **3. Operational Intelligence**
- **Automated Response**: Triggering maintenance workflows based on anomaly types
- **Integration APIs**: Connect with existing banking systems and workflows
- **Regulatory Compliance**: Automated reporting for banking regulatory requirements

---

**Document Version**: 1.0  
**Last Updated**: August 10, 2025  
**Source File**: `services/anomaly-detector/main.py`  
**System**: ABM ML-First Anomaly Detection

---

*This document provides a comprehensive technical overview of the anomaly detection implementation in main.py, highlighting the sophisticated ML-first architecture and real-time processing capabilities of the ABM transaction log analyzer.*
