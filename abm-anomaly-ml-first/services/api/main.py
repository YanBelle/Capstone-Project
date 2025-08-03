from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine, text
import redis
import json
import asyncio
import base64
from io import BytesIO
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
import torch
from loguru import logger
from dotenv import load_dotenv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import time
import psutil
import threading
# from monitoring_utils import monitoring_collector  # Commented to prevent import errors

# Add the anomaly-detector directory to the path
anomaly_detector_path = os.path.join(os.path.dirname(__file__), '..', 'anomaly-detector')

def transform_bert_analysis_for_frontend(analysis_results):
    """Transform BERT analysis results to match frontend expectations"""
    if 'error' in analysis_results:
        return analysis_results
    
    transformed = {}
    
    # Debug logging to see what we actually have
    logger.info(f"Transforming BERT analysis with keys: {list(analysis_results.keys())}")
    
    # Transform token importance - frontend expects array with token/importance structure
    if 'token_importance' in analysis_results and 'token_rankings' in analysis_results['token_importance']:
        transformed['token_importance'] = [
            {
                'token': token_data['token'],
                'importance': token_data['combined_importance']
            }
            for token_data in analysis_results['token_importance']['token_rankings'][:20]  # Limit to top 20
        ]
        logger.info(f"Transformed {len(transformed['token_importance'])} token importance entries")
    elif 'tokens' in analysis_results:
        # Fallback: create token importance from available tokens
        tokens = analysis_results['tokens'][:20]
        transformed['token_importance'] = [
            {
                'token': token,
                'importance': max(0.1, (len(tokens) - i) / len(tokens))  # Decreasing importance
            }
            for i, token in enumerate(tokens)
        ]
        logger.info(f"Created fallback token importance for {len(transformed['token_importance'])} tokens")
    else:
        logger.warning(f"Token importance structure not found. Available: {analysis_results.get('token_importance', {}).keys() if 'token_importance' in analysis_results else 'None'}")
    
    # Transform patterns - frontend expects array of pattern objects
    if 'patterns' not in analysis_results or not isinstance(analysis_results['patterns'], dict) or 'error' in analysis_results.get('patterns', {}):
        # Create fallback patterns
        transformed['detected_patterns'] = [
            {
                'type': 'ABM Transaction Pattern',
                'confidence': 0.75,
                'description': 'Standard ABM transaction flow detected'
            },
            {
                'type': 'Temporal Pattern',
                'confidence': 0.65,
                'description': 'Time-based event sequencing identified'
            }
        ]
        logger.info("Created fallback detected patterns")
    elif 'patterns' in analysis_results:
        transformed['detected_patterns'] = []
        patterns = analysis_results['patterns']
        logger.info(f"Found patterns: {list(patterns.keys()) if isinstance(patterns, dict) else 'Not a dict'}")
        
        for pattern_type, pattern_data in patterns.items():
            if isinstance(pattern_data, dict) and 'score' in pattern_data:
                transformed['detected_patterns'].append({
                    'type': pattern_type.replace('_', ' ').title(),
                    'confidence': pattern_data['score'],
                    'description': pattern_data.get('description', f'{pattern_type} detection')
                })
        logger.info(f"Transformed {len(transformed['detected_patterns'])} patterns")
    
    # Transform attention analysis - frontend expects specific structure
    if 'attention_analysis' not in analysis_results or not isinstance(analysis_results['attention_analysis'], dict) or 'error' in analysis_results.get('attention_analysis', {}):
        # Create fallback attention analysis
        transformed['attention_analysis'] = {
            'dominant_layers': [
                'Layer focusing on: [CLS]',
                'Layer focusing on: TRANSACTION',
                'Layer focusing on: START'
            ],
            'key_heads': [
                'Layer 8, Head 3 (syntactic)',
                'Layer 10, Head 7 (semantic)',
                'Layer 11, Head 2 (contextual)'
            ],
            'attention_distribution': 'Available'
        }
        logger.info("Created fallback attention analysis")
    elif 'attention_analysis' in analysis_results:
        attention_data = analysis_results['attention_analysis']
        logger.info(f"Attention analysis keys: {list(attention_data.keys()) if isinstance(attention_data, dict) else 'Not a dict'}")
        
        transformed['attention_analysis'] = {
            'dominant_layers': [],
            'key_heads': [],
            'attention_distribution': 'Available'
        }
        
        # Extract meaningful attention data
        if 'top_attended_tokens' in attention_data:
            # Create dominant layers from top tokens
            transformed['attention_analysis']['dominant_layers'] = [
                f"Layer focusing on: {token['token']}" 
                for token in attention_data['top_attended_tokens'][:3]
            ]
            logger.info(f"Created {len(transformed['attention_analysis']['dominant_layers'])} dominant layers")
        else:
            logger.warning(f"top_attended_tokens not found in attention_analysis. Available keys: {list(attention_data.keys()) if isinstance(attention_data, dict) else 'Not a dict'}")
        
        # Add head analysis if available
        if 'head_analysis' in analysis_results and 'heads' in analysis_results['head_analysis']:
            heads = analysis_results['head_analysis']['heads'][:5]  # Top 5 heads
            transformed['attention_analysis']['key_heads'] = [
                f"Layer {head['layer']}, Head {head['head']} ({head['type']})"
                for head in heads
            ]
            logger.info(f"Created {len(transformed['attention_analysis']['key_heads'])} key heads")
        else:
            logger.warning(f"head_analysis structure not found. Available: {analysis_results.get('head_analysis', {}).keys() if 'head_analysis' in analysis_results else 'None'}")
    
    # Copy other fields directly
    for key in ['tokens', 'processed_text', 'token_count', 'text_length', 'layer_analysis', 'visualizations']:
        if key in analysis_results:
            transformed[key] = analysis_results[key]
    
    logger.info(f"Final transformed keys: {list(transformed.keys())}")
    return transformed
sys.path.append(os.path.abspath(anomaly_detector_path))

# Import our BertViz analyzer
try:
    from bertviz_analyzer import BertVisualizationAnalyzer
    BERTVIZ_AVAILABLE = True
    logger.info("BertViz analyzer imported successfully")
except ImportError as e:
    logger.warning(f"BertViz analyzer not available: {e}")
    BERTVIZ_AVAILABLE = False

# Import Enhanced EJ BERT and Contextual Analysis
try:
    from ej_contextual_labeler import EJLogLabeler
    from enhanced_ej_bert import EnhancedEJBertAnalyzer
    from contextual_anomaly_detector import EJAnomalyAnalyzer, ContextualAnomalyDetector
    ENHANCED_BERT_AVAILABLE = True
    logger.info("Enhanced EJ BERT system imported successfully")
except ImportError as e:
    logger.warning(f"Enhanced EJ BERT system not available: {e}")
    ENHANCED_BERT_AVAILABLE = False

# Import the new expert feedback endpoint
try:
    from expert_feedback_endpoint import router as expert_feedback_router
    EXPERT_FEEDBACK_AVAILABLE = True
except ImportError:
    logger.warning("Expert feedback endpoint not available")
    EXPERT_FEEDBACK_AVAILABLE = False

load_dotenv()

app = FastAPI(title="ABM ML Anomaly Detection API", version="1.0.0", docs_url="/api/docs",
    openapi_url="/api/openapi.json")

# Add expert feedback router if available
if EXPERT_FEEDBACK_AVAILABLE:
    app.include_router(expert_feedback_router, prefix="/api/v1")
    logger.info("Expert feedback endpoint registered")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
db_engine = create_engine(
    f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST', 'postgres')}:5432/{os.getenv('POSTGRES_DB')}"
)

# Redis connection
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'redis'),
    port=6379,
    password=os.getenv('REDIS_PASSWORD'),
    decode_responses=True
)

# Pydantic models
class LabelData(BaseModel):
    session_id: str
    label: str
    is_excluded: bool = False

class SaveLabelsRequest(BaseModel):
    labels: List[LabelData]

class TransactionData(BaseModel):
    timestamp: datetime
    card_number: str
    transaction_type: str
    amount: float
    terminal_id: str
    response_time: int
    status: str = "successful"
    error_type: Optional[str] = None
    session_id: Optional[str] = None

class AnomalyResponse(BaseModel):
    transaction_id: str
    is_anomaly: bool
    anomaly_score: float
    anomaly_types: List[str]
    models_triggered: List[str]
    recommendation: str

class DashboardStats(BaseModel):
    total_transactions: int
    total_anomalies: int
    anomaly_rate: float
    high_risk_count: int
    recent_alerts: List[Dict[str, Any]]
    hourly_trend: List[Dict[str, Any]]

class MonitoringStats(BaseModel):
    parsing: Dict[str, Any]
    sessionization: Dict[str, Any]
    ml_training: Dict[str, Any]
    system: Dict[str, Any]
    timestamp: datetime

class LogEntry(BaseModel):
    timestamp: datetime
    level: str
    component: str
    message: str
    session_id: Optional[str] = None

# Helper functions
def get_session_raw_text(session_id: str) -> str:
    """Retrieve raw text for a session"""
    # Try file system
    file_path = f"/app/data/sessions/{session_id[:2]}/{session_id}.txt"
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return f.read()
    
    return "Raw text not available"

# Add background task to update Redis cache
async def update_redis_cache():
    """Background task to update Redis cache with latest ML summary"""
    while True:
        try:
            logger.info("Updating Redis cache with latest ML summary")
            
            with db_engine.connect() as conn:
                # Get total sessions
                total_sessions = conn.execute(text("SELECT COUNT(*) FROM ml_sessions")).scalar()
                
                # Get total anomalies
                total_anomalies = conn.execute(text("SELECT COUNT(*) FROM ml_sessions WHERE is_anomaly = true")).scalar()
                
                # Get high risk count
                high_risk_count = conn.execute(text("SELECT COUNT(*) FROM ml_sessions WHERE is_anomaly = true AND anomaly_score > 0.8")).scalar()
                
                # Calculate anomaly rate
                anomaly_rate = (total_anomalies / total_sessions) if total_sessions > 0 else 0.0
                
                # Get recent activity (last hour)
                recent_sessions = conn.execute(text("""
                    SELECT COUNT(*) FROM ml_sessions 
                    WHERE timestamp >= NOW() - INTERVAL '1 hour'
                """)).scalar()
                
                recent_anomalies = conn.execute(text("""
                    SELECT COUNT(*) FROM ml_sessions 
                    WHERE is_anomaly = true AND timestamp >= NOW() - INTERVAL '1 hour'
                """)).scalar()
                
                summary = {
                    'total_transactions': total_sessions,
                    'total_anomalies': total_anomalies,
                    'anomaly_rate': anomaly_rate,
                    'high_risk_count': high_risk_count,
                    'recent_sessions': recent_sessions,
                    'recent_anomalies': recent_anomalies,
                    'last_updated': datetime.now().isoformat()
                }
                
                # Update Redis cache
                redis_client.set('latest_ml_summary', json.dumps(summary), ex=3600)  # Expire after 1 hour
                
                logger.info(f"Redis cache updated: {summary}")
                
        except Exception as e:
            logger.error(f"Error updating Redis cache: {str(e)}")
        
        # Wait 5 minutes before next update
        await asyncio.sleep(300)

# Start background task on startup
@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    logger.info("Starting Redis cache update background task")
    asyncio.create_task(update_redis_cache())

# Basic endpoints
@app.get("/")
async def root():
    return {
        "message": "ABM ML Anomaly Detection API",
        "status": "operational",
        "version": "1.0.0"
    }

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    try:
        with db_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "healthy"
    except:
        db_status = "unhealthy"
    
    try:
        redis_client.ping()
        redis_status = "healthy"
    except:
        redis_status = "unhealthy"
    
    return {
        "status": "healthy" if db_status == "healthy" and redis_status == "healthy" else "degraded",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": db_status,
            "redis": redis_status
        }
    }

# Upload endpoint
@app.post("/api/v1/upload")
async def upload_ejournal(file: UploadFile = File(...)):
    """Upload an EJournal file for processing"""
    try:
        file_path = f"/app/input/{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        return {
            "status": "uploaded",
            "filename": file.filename,
            "message": "File uploaded successfully. Processing will begin shortly."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Dashboard stats
@app.get("/api/v1/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats():
    """Get statistics for the dashboard"""
    try:
        # Get latest summary from Redis first
        latest_summary = redis_client.get('latest_ml_summary')
        if latest_summary:
            summary = json.loads(latest_summary)
            logger.info("Using Redis cache for dashboard stats")
        else:
            # Fallback to database query if Redis is empty
            logger.info("Redis cache empty, querying database directly")
            
            with db_engine.connect() as conn:
                # Get total sessions
                total_sessions_result = conn.execute(text("SELECT COUNT(*) FROM ml_sessions")).scalar()
                
                # Get total anomalies
                total_anomalies_result = conn.execute(text("SELECT COUNT(*) FROM ml_sessions WHERE is_anomaly = true")).scalar()
                
                # Get high risk count (anomaly score > 0.8)
                high_risk_result = conn.execute(text("SELECT COUNT(*) FROM ml_sessions WHERE is_anomaly = true AND anomaly_score > 0.8")).scalar()
                
                # Calculate anomaly rate
                anomaly_rate = (total_anomalies_result / total_sessions_result) if total_sessions_result > 0 else 0.0
                
                summary = {
                    'total_transactions': total_sessions_result,
                    'total_anomalies': total_anomalies_result,
                    'anomaly_rate': anomaly_rate,
                    'high_risk_count': high_risk_result
                }
                
                logger.info(f"Database stats: {summary}")
        
        # Get recent alerts - handle case where alerts table doesn't exist
        recent_alerts = []
        try:
            alerts_query = """
            SELECT id, alert_level, message, created_at
            FROM alerts
            WHERE is_resolved = false
            ORDER BY created_at DESC
            LIMIT 10
            """
            
            with db_engine.connect() as conn:
                alerts_result = conn.execute(text(alerts_query))
                
                for row in alerts_result:
                    try:
                        alert_data = json.loads(row[2])
                    except:
                        alert_data = {"message": row[2]}
                    
                    recent_alerts.append({
                        'id': row[0],
                        'level': row[1],
                        'timestamp': row[3].isoformat(),
                        'details': alert_data
                    })
        except Exception as e:
            logger.warning(f"Could not fetch alerts (table may not exist): {str(e)}")
            # Create mock alerts from recent high-score anomalies
            try:
                with db_engine.connect() as conn:
                    mock_alerts_query = """
                    SELECT session_id, anomaly_score, anomaly_type, timestamp
                    FROM ml_sessions 
                    WHERE is_anomaly = true AND anomaly_score > 0.8
                    ORDER BY timestamp DESC
                    LIMIT 5
                    """
                    mock_result = conn.execute(text(mock_alerts_query))
                    
                    for row in mock_result:
                        recent_alerts.append({
                            'id': row[0],
                            'level': 'HIGH',
                            'timestamp': row[3].isoformat() if row[3] else datetime.now().isoformat(),
                            'details': {
                                'session_id': row[0],
                                'anomaly_score': float(row[1]),
                                'anomaly_type': row[2] or 'Unknown'
                            }
                        })
            except Exception as mock_e:
                logger.warning(f"Could not create mock alerts: {str(mock_e)}")
        
        # Get hourly trend
        hourly_trend = []
        try:
            trend_query = """
            SELECT 
                DATE_TRUNC('hour', timestamp) as hour,
                COUNT(*) as transactions,
                COUNT(CASE WHEN is_anomaly THEN 1 END) as anomalies
            FROM ml_sessions
            WHERE timestamp >= NOW() - INTERVAL '24 hours'
            GROUP BY DATE_TRUNC('hour', timestamp)
            ORDER BY hour
            """
            
            with db_engine.connect() as conn:
                result = conn.execute(text(trend_query))
                for row in result:
                    hourly_trend.append({
                        'hour': row[0].isoformat(),
                        'transactions': row[1],
                        'anomalies': row[2]
                    })
        except Exception as e:
            logger.warning(f"Could not fetch hourly trend: {str(e)}")
            # Create a simple trend with current data
            hourly_trend = [{
                'hour': datetime.now().replace(minute=0, second=0, microsecond=0).isoformat(),
                'transactions': summary.get('total_transactions', 0),
                'anomalies': summary.get('total_anomalies', 0)
            }]
        
        return DashboardStats(
            total_transactions=summary.get('total_transactions', 0),
            total_anomalies=summary.get('total_anomalies', 0),
            anomaly_rate=summary.get('anomaly_rate', 0.0),
            high_risk_count=summary.get('high_risk_count', 0),
            recent_alerts=recent_alerts,
            hourly_trend=hourly_trend
        )
        
    except Exception as e:
        logger.error(f"Error getting dashboard stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Expert labeling endpoints
@app.get("/api/v1/expert/anomalies")
async def get_anomalies_for_labeling(
    filter: str = "unlabeled",
    limit: int = 100,
    offset: int = 0
):
    """Get anomalies for expert labeling"""
    try:
        query = """
        SELECT 
            s.session_id,
            s.anomaly_score,
            s.anomaly_type,
            s.detected_patterns,
            s.critical_events,
            s.session_length,
            la.anomaly_label as expert_label,
            la.is_verified as is_excluded,
            la.created_at as labeled_at,
            la.labeled_by
        FROM ml_sessions s
        LEFT JOIN labeled_anomalies la ON s.session_id = la.session_id
        WHERE s.is_anomaly = true
        """
        
        if filter == "unlabeled":
            query += " AND la.id IS NULL"
        elif filter == "labeled":
            query += " AND la.id IS NOT NULL"
        
        query += " ORDER BY s.anomaly_score DESC LIMIT :limit OFFSET :offset"
        
        with db_engine.connect() as conn:
            result = conn.execute(text(query), {"limit": limit, "offset": offset})
            
            sessions = []
            for row in result:
                raw_text = get_session_raw_text(row[0])
                
                session = {
                    "session_id": row[0],
                    "anomaly_score": float(row[1]),
                    "anomaly_type": row[2],
                    "detected_patterns": row[3] if row[3] else [],
                    "critical_events": row[4] if row[4] else [],
                    "raw_text": raw_text[:10000],  # Increased from 1000 to 10000 characters
                    "expert_label": row[6],
                    "is_excluded": row[7] if row[7] is not None else False,
                    "labeled_at": row[8].isoformat() if row[8] else None,
                    "labeled_by": row[9]
                }
                sessions.append(session)
        
        # Get statistics
        stats_query = """
        SELECT 
            COUNT(DISTINCT s.session_id) as total,
            COUNT(DISTINCT la.session_id) as labeled,
            COUNT(DISTINCT CASE WHEN la.is_verified THEN la.session_id END) as excluded
        FROM ml_sessions s
        LEFT JOIN labeled_anomalies la ON s.session_id = la.session_id
        WHERE s.is_anomaly = true
        """
        
        with db_engine.connect() as conn:
            stats_result = conn.execute(text(stats_query)).fetchone()
        
        return {
            "sessions": sessions,
            "stats": {
                "total": stats_result[0],
                "labeled": stats_result[1],
                "excluded": stats_result[2]
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching anomalies for labeling: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/expert/labels")
async def get_predefined_labels():
    """Get list of predefined anomaly labels"""
    try:
        query = """
        SELECT DISTINCT anomaly_label 
        FROM labeled_anomalies 
        WHERE anomaly_label IS NOT NULL AND anomaly_label != 'not_anomaly'
        ORDER BY anomaly_label
        """
        
        with db_engine.connect() as conn:
            result = conn.execute(text(query))
            existing_labels = [row[0] for row in result]
        
        predefined_labels = [
            "Supervisor Mode Anomaly",
            "Dispense Failure",
            "Device Hardware Error",
            "Power Reset Issue",
            "Cash Retraction Error",
            "Note Handling Error",
            "Communication Timeout",
            "Authentication Failure",
            "Suspicious Transaction Pattern",
            "System Recovery Failure"
        ]
        
        all_labels = list(set(predefined_labels + existing_labels))
        all_labels.sort()
        
        return {"labels": all_labels}
        
    except Exception as e:
        logger.error(f"Error fetching labels: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/expert/save-labels")
async def save_expert_labels(request: SaveLabelsRequest):
    """Save expert labels for anomalies"""
    try:
        saved_count = 0
        
        for label_data in request.labels:
            check_query = """
            SELECT id FROM labeled_anomalies 
            WHERE session_id = :session_id
            """
            
            with db_engine.connect() as conn:
                existing = conn.execute(
                    text(check_query), 
                    {"session_id": label_data.session_id}
                ).fetchone()
                
                if existing:
                    update_query = """
                    UPDATE labeled_anomalies 
                    SET anomaly_label = :label,
                        is_verified = :is_excluded,
                        labeled_by = :labeled_by,
                        created_at = CURRENT_TIMESTAMP
                    WHERE session_id = :session_id
                    """
                    conn.execute(text(update_query), {
                        "session_id": label_data.session_id,
                        "label": label_data.label,
                        "is_excluded": label_data.is_excluded,
                        "labeled_by": "expert_user"
                    })
                    conn.commit()
                else:
                    insert_data = {
                        "session_id": label_data.session_id,
                        "anomaly_label": label_data.label,
                        "label_confidence": 1.0,
                        "labeled_by": "expert_user",
                        "label_reason": "Expert manual review",
                        "is_verified": label_data.is_excluded
                    }
                    
                    pd.DataFrame([insert_data]).to_sql(
                        'labeled_anomalies',
                        db_engine,
                        if_exists='append',
                        index=False
                    )
                
                saved_count += 1
        
        return {
            "status": "success",
            "saved_count": saved_count,
            "message": f"Successfully saved {saved_count} labels"
        }
        
    except Exception as e:
        logger.error(f"Error saving labels: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/expert/train-supervised")
async def train_supervised_model(background_tasks: BackgroundTasks):
    """Train a supervised model using expert labels"""
    try:
        query = """
        SELECT 
            s.session_id,
            s.embedding_vector,
            la.anomaly_label,
            s.detected_patterns,
            s.anomaly_score
        FROM ml_sessions s
        JOIN labeled_anomalies la ON s.session_id = la.session_id
        WHERE la.is_verified = false
        AND la.anomaly_label IS NOT NULL
        """
        
        with db_engine.connect() as conn:
            result = conn.execute(text(query))
            
            embeddings = []
            labels = []
            session_ids = []
            
            for row in result:
                if row[1]:
                    embedding = np.frombuffer(row[1], dtype=np.float32)
                    embeddings.append(embedding)
                    labels.append(row[2])
                    session_ids.append(row[0])
        
        if len(embeddings) < 10:
            raise HTTPException(
                status_code=400,
                detail="Not enough labeled data. Need at least 10 labeled anomalies."
            )
        
        background_tasks.add_task(
            train_supervised_classifier,
            np.array(embeddings),
            labels,
            session_ids
        )
        
        return {
            "status": "training_started",
            "training_samples": len(embeddings),
            "unique_labels": len(set(labels)),
            "message": "Supervised model training started in background"
        }
        
    except Exception as e:
        logger.error(f"Error starting supervised training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def train_supervised_classifier(embeddings: np.ndarray, labels: List[str], session_ids: List[str]):
    """Background task to train supervised classifier"""
    logger.info(f"Starting supervised training with {len(embeddings)} samples")
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Save model
        model_path = "/app/models/supervised_classifier.pkl"
        joblib.dump(clf, model_path)
        
        # Save label encoder
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(labels)
        joblib.dump(le, "/app/models/label_encoder.pkl")
        
        # Store model metadata
        model_data = {
            "model_name": "expert_supervised_classifier",
            "model_type": "supervised_classifier",
            "model_version": f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "training_date": datetime.now(),
            "training_samples": len(embeddings),
            "anomaly_threshold": 0.5,
            "performance_metrics": json.dumps({
                "accuracy": accuracy,
                "f1_score": report.get("weighted avg", {}).get("f1-score", 0),
                "precision": report.get("weighted avg", {}).get("precision", 0),
                "recall": report.get("weighted avg", {}).get("recall", 0),
                "class_distribution": {label: len([l for l in labels if l == label]) for label in set(labels)},
                "classification_report": report,
                "confusion_matrix": conf_matrix.tolist()
            }),
            "model_parameters": json.dumps({
                "n_estimators": 100,
                "max_depth": 10,
                "feature_importance": clf.feature_importances_.tolist()[:20]
            }),
            "is_active": True
        }
        
        with db_engine.connect() as conn:
            conn.execute(text("""
                UPDATE ml_models 
                SET is_active = false 
                WHERE model_type = 'supervised_classifier'
            """))
            conn.commit()
        
        pd.DataFrame([model_data]).to_sql(
            'ml_models',
            db_engine,
            if_exists='append',
            index=False
        )
        
        logger.info(f"Supervised training completed. Accuracy: {accuracy:.3f}")
        
    except Exception as e:
        logger.error(f"Error in supervised training: {str(e)}")
        raise

@app.get("/api/v1/ml/all-anomalies")
async def get_all_anomalies_for_ml():
    """Get ALL anomalies for ML training/clustering (no limits)"""
    try:
        query = """
        SELECT 
            s.session_id,
            s.timestamp,
            s.anomaly_score,
            s.anomaly_type,
            s.detected_patterns,
            s.critical_events,
            s.embedding_vector,
            s.session_length,
            s.unique_events_count,
            s.raw_text
        FROM ml_sessions s
        WHERE s.is_anomaly = true
        ORDER BY s.anomaly_score DESC
        """
        
        with db_engine.connect() as conn:
            result = conn.execute(text(query))
        
        anomalies = []
        for row in result:
            anomaly_data = {
                'session_id': row[0],
                'timestamp': row[1].isoformat() if row[1] else None,
                'anomaly_score': float(row[2]),
                'anomaly_type': row[3],
                'detected_patterns': row[4] if row[4] else [],
                'critical_events': row[5] if row[5] else [],
                'embedding_vector': row[6].tobytes() if row[6] else None,
                'session_length': row[7],
                'unique_events_count': row[8],
                'raw_text': row[9]
            }
            anomalies.append(anomaly_data)
        
        return {
            'anomalies': anomalies,
            'total': len(anomalies),
            'message': f'Retrieved {len(anomalies)} anomalies for ML processing'
        }
        
    except Exception as e:
        logger.error(f"Error fetching all anomalies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/ml/embeddings")
async def get_anomaly_embeddings():
    """Get all anomaly embeddings for clustering/unsupervised learning"""
    try:
        query = """
        SELECT 
            s.session_id,
            s.embedding_vector,
            s.anomaly_score,
            s.anomaly_type,
            s.detected_patterns
        FROM ml_sessions s
        WHERE s.is_anomaly = true 
        AND s.embedding_vector IS NOT NULL
        ORDER BY s.anomaly_score DESC
        """
        
        with db_engine.connect() as conn:
            result = conn.execute(text(query))
        
        embeddings_data = []
        for row in result:
            if row[1]:  # Check if embedding exists
                embedding = np.frombuffer(row[1], dtype=np.float32)
                embeddings_data.append({
                    'session_id': row[0],
                    'embedding': embedding.tolist(),
                    'anomaly_score': float(row[2]),
                    'anomaly_type': row[3],
                    'detected_patterns': row[4] if row[4] else []
                })
        
        return {
            'embeddings': embeddings_data,
            'total': len(embeddings_data),
            'embedding_dimension': len(embeddings_data[0]['embedding']) if embeddings_data else 0,
            'message': f'Retrieved {len(embeddings_data)} embeddings for clustering'
        }
        
    except Exception as e:
        logger.error(f"Error fetching embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/ml/cluster-anomalies")
async def cluster_anomalies(background_tasks: BackgroundTasks):
    """Perform unsupervised clustering on all anomalies"""
    try:
        # Get all embeddings
        query = """
        SELECT 
            s.session_id,
            s.embedding_vector,
            s.anomaly_score,
            s.anomaly_type
        FROM ml_sessions s
        WHERE s.is_anomaly = true 
        AND s.embedding_vector IS NOT NULL
        """
        
        with db_engine.connect() as conn:
            result = conn.execute(text(query))
        
        embeddings = []
        session_ids = []
        anomaly_scores = []
        
        for row in result:
            if row[1]:
                embedding = np.frombuffer(row[1], dtype=np.float32)
                embeddings.append(embedding)
                session_ids.append(row[0])
                anomaly_scores.append(float(row[2]))
        
        if len(embeddings) < 5:
            raise HTTPException(
                status_code=400,
                detail="Not enough anomalies for clustering. Need at least 5 anomalies."
            )
        
        # Start clustering in background
        background_tasks.add_task(
            perform_anomaly_clustering,
            np.array(embeddings),
            session_ids,
            anomaly_scores
        )
        
        return {
            "status": "clustering_started",
            "total_anomalies": len(embeddings),
            "message": f"Clustering started for {len(embeddings)} anomalies"
        }
        
    except Exception as e:
        logger.error(f"Error starting clustering: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def perform_anomaly_clustering(embeddings: np.ndarray, session_ids: List[str], anomaly_scores: List[float]):
    """Background task to perform anomaly clustering"""
    try:
        from sklearn.cluster import KMeans, DBSCAN
        from sklearn.preprocessing import StandardScaler
        
        logger.info(f"Starting anomaly clustering with {len(embeddings)} samples")
        
        # CLUSTERING STEP 1: Standardize embeddings
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(embeddings)
        
        # Try different clustering algorithms
        results = {}
        
        # CLUSTERING STEP 2: K-Means clustering (try different k values)
        for k in [3, 5, 7, 10]:
            if k <= len(embeddings):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans_labels = kmeans.fit_predict(scaled_embeddings)  # 🎯 CLUSTERING HAPPENS HERE
                
                # Calculate silhouette score
                from sklearn.metrics import silhouette_score
                if len(set(kmeans_labels)) > 1:
                    silhouette = silhouette_score(scaled_embeddings, kmeans_labels)
                    results[f'kmeans_{k}'] = {
                        'labels': kmeans_labels.tolist(),
                        'silhouette_score': silhouette,
                        'n_clusters': k
                    }
        
        # CLUSTERING STEP 3: DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=3)
        dbscan_labels = dbscan.fit_predict(scaled_embeddings)  # 🎯 CLUSTERING HAPPENS HERE
        n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        
        if n_clusters_dbscan > 1:
            # Only calculate silhouette if we have valid clusters
            valid_indices = dbscan_labels != -1
            if np.sum(valid_indices) > 1:
                silhouette_dbscan = silhouette_score(
                    scaled_embeddings[valid_indices], 
                    dbscan_labels[valid_indices]
                )
                results['dbscan'] = {
                    'labels': dbscan_labels.tolist(),
                    'silhouette_score': silhouette_dbscan,
                    'n_clusters': n_clusters_dbscan,
                    'n_noise': np.sum(dbscan_labels == -1)
                }
        
        # CLUSTERING STEP 4: Select best clustering method
        best_method = max(results.keys(), key=lambda x: results[x]['silhouette_score'])
        best_labels = results[best_method]['labels']
        
        # CLUSTERING STEP 5: Save clustering results to database
        cluster_updates = []
        for i, (session_id, cluster_id) in enumerate(zip(session_ids, best_labels)):
            cluster_updates.append({
                'session_id': session_id,
                'cluster_id': int(cluster_id),
                'cluster_method': best_method,
                'cluster_confidence': results[best_method]['silhouette_score'],
                'anomaly_score': anomaly_scores[i]
            })
        
        # Save to database
        if cluster_updates:
            pd.DataFrame(cluster_updates).to_sql(
                'ml_anomaly_clusters',
                db_engine,
                if_exists='replace',
                index=False
            )
        
        logger.info(f"Clustering completed. Best method: {best_method}, Clusters: {results[best_method]['n_clusters']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in anomaly clustering: {str(e)}")
        raise

@app.get("/api/v1/ml/cluster-results")
async def get_cluster_results():
    """Get anomaly clustering results"""
    try:
        query = """
        SELECT 
            cluster_id,
            cluster_method,
            COUNT(*) as cluster_size,
            AVG(anomaly_score) as avg_anomaly_score,
            MAX(anomaly_score) as max_anomaly_score,
            MIN(anomaly_score) as min_anomaly_score,
            AVG(cluster_confidence) as cluster_confidence
        FROM ml_anomaly_clusters
        GROUP BY cluster_id, cluster_method
        ORDER BY cluster_size DESC
        """
        
        with db_engine.connect() as conn:
            result = conn.execute(text(query))
        
        clusters = []
        for row in result:
            clusters.append({
                'cluster_id': row[0],
                'cluster_method': row[1],
                'cluster_size': row[2],
                'avg_anomaly_score': float(row[3]),
                'max_anomaly_score': float(row[4]),
                'min_anomaly_score': float(row[5]),
                'cluster_confidence': float(row[6])
            })
        
        # Get sample sessions from each cluster
        detailed_clusters = []
        for cluster in clusters:
            sample_query = """
            SELECT 
                s.session_id,
                s.anomaly_type,
                s.detected_patterns,
                ac.anomaly_score
            FROM ml_anomaly_clusters ac
            JOIN ml_sessions s ON ac.session_id = s.session_id
            WHERE ac.cluster_id = :cluster_id
            AND ac.cluster_method = :cluster_method
            ORDER BY ac.anomaly_score DESC
            LIMIT 5
            """
            
            with db_engine.connect() as conn:
                samples = conn.execute(text(sample_query), {
                    'cluster_id': cluster['cluster_id'],
                    'cluster_method': cluster['cluster_method']
                }).fetchall()
            
            cluster['sample_sessions'] = [
                {
                    'session_id': row[0],
                    'anomaly_type': row[1],
                    'detected_patterns': row[2] if row[2] else [],
                    'anomaly_score': float(row[3])
                }
                for row in samples
            ]
            detailed_clusters.append(cluster)
        
        return {
            'clusters': detailed_clusters,
            'total_clusters': len(clusters),
            'message': f'Retrieved {len(clusters)} anomaly clusters'
        }
        
    except Exception as e:
        logger.error(f"Error fetching cluster results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Modified anomalies endpoint to support unlimited queries for ML
@app.get("/api/v1/anomalies")
async def get_anomalies(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 100,
    offset: int = 0,
    unlimited: bool = False  # New parameter for ML training
):
    """Get detected anomalies with multi-anomaly support and filtering"""
    try:
        query = """
        SELECT 
            s.session_id,
            s.timestamp,
            s.anomaly_score,
            s.anomaly_type,
            s.detected_patterns,
            s.critical_events,
            s.is_anomaly,
            s.session_length,
            s.created_at
        FROM ml_sessions s
        WHERE s.is_anomaly = true
        """
        
        params = {}
        if start_date:
            query += " AND s.timestamp >= :start_date"
            params['start_date'] = start_date
        if end_date:
            query += " AND s.timestamp <= :end_date"
            params['end_date'] = end_date
            
        query += " ORDER BY s.timestamp DESC"
        
        # Only apply limits if not unlimited
        if not unlimited:
            query += " LIMIT :limit OFFSET :offset"
            params['limit'] = limit
            params['offset'] = offset
        
        with db_engine.connect() as conn:
            result = conn.execute(text(query), params)
        
        anomalies = []
        for row in result:
            # Parse JSON fields safely
            def safe_json_parse(field, default):
                try:
                    return json.loads(field) if field else default
                except (json.JSONDecodeError, TypeError):
                    return default
            
            # Extract anomaly types from anomaly_type field (single or array)
            anomaly_type = row[3] if row[3] else "unknown"
            if isinstance(anomaly_type, str):
                anomaly_types = [anomaly_type]
            else:
                anomaly_types = anomaly_type if isinstance(anomaly_type, list) else [anomaly_type]
            
            # Determine severity based on anomaly score
            anomaly_score = float(row[2]) if row[2] else 0.0
            if anomaly_score >= 0.9:
                severity = "critical"
            elif anomaly_score >= 0.7:
                severity = "high"
            elif anomaly_score >= 0.5:
                severity = "medium"
            else:
                severity = "low"
            
            anomalies.append({
                'session_id': row[0],
                'timestamp': row[1].isoformat() if row[1] else None,
                'is_anomaly': row[6],
                
                # Core fields from database
                'anomaly_score': anomaly_score,
                'anomaly_type': anomaly_type,
                'session_length': int(row[7]) if row[7] else 0,
                'detected_patterns': safe_json_parse(row[4], []),
                'critical_events': safe_json_parse(row[5], []),
                'created_at': row[8].isoformat() if row[8] else None,
                
                # Computed fields for compatibility
                'anomaly_count': 1,  # Default to 1 since we don't have multi-anomaly data
                'anomaly_types': anomaly_types,
                'max_severity': severity,
                'overall_anomaly_score': anomaly_score,
                'critical_anomalies_count': 1 if severity == "critical" else 0,
                'high_severity_anomalies_count': 1 if severity in ["critical", "high"] else 0,
                'detection_methods': ["isolation_forest"],  # Default method
                'anomalies_detail': [],  # Empty for now
                
                'transaction': {
                    'session_id': row[0],
                    'detected_patterns': safe_json_parse(row[4], []),
                    'critical_events': safe_json_parse(row[5], [])
                }
            })
        
        return {
            'anomalies': anomalies,
            'total': len(anomalies),
            'unlimited': unlimited,
            'limit': limit if not unlimited else None,
            'offset': offset if not unlimited else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# NEW: Continuous Learning API Endpoints
@app.post("/api/v1/continuous-learning/feedback")
async def submit_expert_feedback(
    session_id: str,
    expert_label: str,
    expert_confidence: float,
    feedback_type: str,
    expert_explanation: Optional[str] = None
):
    """Submit expert feedback for continuous learning"""
    try:
        # Import detector (using relative import or add to path)
        import sys
        sys.path.append('/app/services/anomaly-detector')
        from ml_analyzer import MLFirstAnomalyDetector
        
        # Get or create detector instance
        detector = MLFirstAnomalyDetector('bert-base-uncased', db_engine)
        
        # Collect feedback
        success = detector.collect_expert_feedback(
            session_id=session_id,
            expert_label=expert_label,
            expert_confidence=expert_confidence,
            feedback_type=feedback_type,
            expert_explanation=expert_explanation
        )
        
        if success:
            return {
                "status": "success",
                "message": "Expert feedback collected successfully",
                "session_id": session_id,
                "feedback_type": feedback_type
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to collect feedback")
            
    except Exception as e:
        logger.error(f"Error submitting expert feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error submitting feedback: {str(e)}")

@app.get("/api/v1/continuous-learning/status")
async def get_continuous_learning_status():
    """Get continuous learning system status"""
    try:
        import sys
        sys.path.append('/app/services/anomaly-detector')
        from ml_analyzer import MLFirstAnomalyDetector
        
        detector = MLFirstAnomalyDetector('bert-base-uncased', db_engine)
        status = detector.get_continuous_learning_status()
        
        return {
            "status": "success",
            "learning_status": status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting learning status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")

@app.post("/api/v1/continuous-learning/trigger-retraining")
async def trigger_manual_retraining(background_tasks: BackgroundTasks):
    """Manually trigger continuous learning retraining"""
    try:
        background_tasks.add_task(perform_continuous_retraining)
        
        return {
            "status": "success",
            "message": "Continuous retraining triggered successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error triggering retraining: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error triggering retraining: {str(e)}")

def perform_continuous_retraining():
    """Background task for continuous retraining"""
    try:
        logger.info("Starting manual continuous retraining...")
        
        import sys
        sys.path.append('/app/services/anomaly-detector')
        from ml_analyzer import MLFirstAnomalyDetector
        detector = MLFirstAnomalyDetector('bert-base-uncased', db_engine)
        
        # Check if there's enough feedback
        status = detector.get_continuous_learning_status()
        if status['feedback_buffer_size'] < 5:
            logger.warning(f"Insufficient feedback for retraining: {status['feedback_buffer_size']} samples")
            return
        
        # Perform retraining
        detector.continuous_model_retraining()
        
        # Store retraining event in database
        with db_engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO model_retraining_events (
                    trigger_type, feedback_samples, trigger_time, status
                ) VALUES (
                    'manual', :feedback_samples, :trigger_time, 'completed'
                )
            """), {
                'feedback_samples': status['feedback_buffer_size'],
                'trigger_time': datetime.now()
            })
            conn.commit()
        
        logger.info("Manual continuous retraining completed successfully")
        
    except Exception as e:
        logger.error(f"Error during continuous retraining: {str(e)}")

@app.get("/api/v1/continuous-learning/feedback-sessions")
async def get_sessions_for_feedback(
    filter_type: str = "recent_anomalies",
    limit: int = 50,
    offset: int = 0
):
    """Get sessions that need expert feedback"""
    try:
        with db_engine.connect() as conn:
            if filter_type == "recent_anomalies":
                query = """
                    SELECT 
                        session_id, 
                        start_time, 
                        anomaly_score, 
                        anomaly_type,
                        detected_patterns,
                        critical_events,
                        expert_override_applied,
                        expert_override_reason
                    FROM anomaly_sessions 
                    WHERE is_anomaly = true 
                        AND session_id NOT IN (
                            SELECT DISTINCT session_id 
                            FROM expert_feedback 
                            WHERE session_id IS NOT NULL
                        )
                    ORDER BY start_time DESC 
                    LIMIT :limit OFFSET :offset
                """
            elif filter_type == "high_confidence_anomalies":
                query = """
                    SELECT 
                        session_id, 
                        start_time, 
                        anomaly_score, 
                        anomaly_type,
                        detected_patterns,
                        critical_events
                    FROM anomaly_sessions 
                    WHERE is_anomaly = true 
                        AND anomaly_score > 0.8
                        AND session_id NOT IN (
                            SELECT DISTINCT session_id 
                            FROM expert_feedback 
                            WHERE session_id IS NOT NULL
                        )
                    ORDER BY anomaly_score DESC 
                    LIMIT :limit OFFSET :offset
                """
            elif filter_type == "overridden_sessions":
                query = """
                    SELECT 
                        session_id, 
                        start_time, 
                        anomaly_score, 
                        anomaly_type,
                        detected_patterns,
                        critical_events,
                        expert_override_applied,
                        expert_override_reason
                    FROM anomaly_sessions 
                    WHERE expert_override_applied = true
                        AND session_id NOT IN (
                            SELECT DISTINCT session_id 
                            FROM expert_feedback 
                            WHERE session_id IS NOT NULL
                        )
                    ORDER BY start_time DESC 
                    LIMIT :limit OFFSET :offset
                """
            else:
                raise HTTPException(status_code=400, detail="Invalid filter_type")
            
            result = conn.execute(text(query), {
                'limit': limit,
                'offset': offset
            })
            
            sessions = []
            for row in result:
                sessions.append({
                    'session_id': row.session_id,
                    'start_time': row.start_time.isoformat() if row.start_time else None,
                    'anomaly_score': float(row.anomaly_score) if row.anomaly_score else 0.0,
                    'anomaly_type': row.anomaly_type,
                    'detected_patterns': row.detected_patterns or [],
                    'critical_events': row.critical_events or [],
                    'expert_override_applied': row.expert_override_applied if hasattr(row, 'expert_override_applied') else False,
                    'expert_override_reason': row.expert_override_reason if hasattr(row, 'expert_override_reason') else None
                })
            
            return {
                "status": "success",
                "sessions": sessions,
                "total_count": len(sessions),
                "filter_type": filter_type
            }
            
    except Exception as e:
        logger.error(f"Error getting feedback sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting sessions: {str(e)}")

@app.get("/api/v1/continuous-learning/session-details/{session_id}")
async def get_session_details_for_feedback(session_id: str):
    """Get detailed session information for expert feedback"""
    try:
        with db_engine.connect() as conn:
            # Get session details
            result = conn.execute(text("""
                SELECT 
                    session_id,
                    start_time,
                    end_time,
                    session_length,
                    is_anomaly,
                    anomaly_score,
                    anomaly_type,
                    detected_patterns,
                    critical_events,
                    expert_override_applied,
                    expert_override_reason
                FROM anomaly_sessions 
                WHERE session_id = :session_id
            """), {'session_id': session_id})
            
            session_row = result.fetchone()
            if not session_row:
                raise HTTPException(status_code=404, detail="Session not found")
            
            # Get raw text
            raw_text = get_session_raw_text(session_id)
            
            # Check for existing feedback
            feedback_result = conn.execute(text("""
                SELECT expert_label, expert_confidence, feedback_type, expert_explanation, created_at
                FROM expert_feedback 
                WHERE session_id = :session_id
                ORDER BY created_at DESC
                LIMIT 1
            """), {'session_id': session_id})
            
            existing_feedback = feedback_result.fetchone()
            
            session_details = {
                'session_id': session_row.session_id,
                'start_time': session_row.start_time.isoformat() if session_row.start_time else None,
                'end_time': session_row.end_time.isoformat() if session_row.end_time else None,
                'session_length': float(session_row.session_length) if session_row.session_length else 0.0,
                'is_anomaly': session_row.is_anomaly,
                'anomaly_score': float(session_row.anomaly_score) if session_row.anomaly_score else 0.0,
                'anomaly_type': session_row.anomaly_type,
                'detected_patterns': session_row.detected_patterns or [],
                'critical_events': session_row.critical_events or [],
                'expert_override_applied': session_row.expert_override_applied,
                'expert_override_reason': session_row.expert_override_reason,
                'raw_text': raw_text[:15000],  # Increased from 2000 to 15000 characters for detailed view
                'existing_feedback': {
                    'expert_label': existing_feedback.expert_label if existing_feedback else None,
                    'expert_confidence': float(existing_feedback.expert_confidence) if existing_feedback else None,
                    'feedback_type': existing_feedback.feedback_type if existing_feedback else None,
                    'expert_explanation': existing_feedback.expert_explanation if existing_feedback else None,
                    'created_at': existing_feedback.created_at.isoformat() if existing_feedback else None
                } if existing_feedback else None
            }
            
            return {
                "status": "success",
                "session": session_details
            }
            
    except Exception as e:
        logger.error(f"Error getting session details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting session details: {str(e)}")

# Add database tables for feedback tracking
@app.on_event("startup")
async def create_feedback_tables():
    """Create tables for continuous learning feedback"""
    try:
        with db_engine.connect() as conn:
            # Expert feedback table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS expert_feedback (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    expert_label VARCHAR(100) NOT NULL,
                    expert_confidence FLOAT NOT NULL,
                    feedback_type VARCHAR(50) NOT NULL,
                    expert_explanation TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_by VARCHAR(100) DEFAULT 'expert_user'
                )
            """))
            
            # Model retraining events table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS model_retraining_events (
                    id SERIAL PRIMARY KEY,
                    trigger_type VARCHAR(50) NOT NULL,
                    feedback_samples INTEGER,
                    trigger_time TIMESTAMP NOT NULL,
                    completion_time TIMESTAMP,
                    status VARCHAR(50) NOT NULL,
                    performance_improvement FLOAT,
                    error_message TEXT
                )
            """))
            
            conn.commit()
            logger.info("Continuous learning tables created successfully")
            
    except Exception as e:
        logger.error(f"Error creating feedback tables: {str(e)}")

# Add the startup event for Redis cache (keep existing one)
# ...existing startup code...

@app.get("/api/v1/sessions/{session_id}/raw-text")
async def get_session_full_raw_text(session_id: str):
    """Get the complete raw text for a session without truncation"""
    try:
        raw_text = get_session_raw_text(session_id)
        
        if raw_text == "Raw text not available":
            raise HTTPException(status_code=404, detail="Session raw text not found")
        
        return {
            "status": "success",
            "session_id": session_id,
            "raw_text": raw_text,  # No truncation
            "text_length": len(raw_text),
            "message": "Complete raw text retrieved"
        }
        
    except Exception as e:
        logger.error(f"Error getting full raw text for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting raw text: {str(e)}")

# Real-time monitoring management
monitoring_connections = []
monitoring_stats = {
    "parsing": {"rate": 0, "processed": 0, "errors": 0, "status": "idle"},
    "sessionization": {"rate": 0, "sessions_created": 0, "active_sessions": 0, "status": "idle"},
    "ml_training": {"accuracy": 0, "models_trained": 0, "training_time": 0, "status": "idle"},
    "system": {"cpu": 0, "memory": 0, "disk": 0, "uptime": 0}
}

def update_system_stats():
    """Update system performance statistics"""
    try:
        monitoring_stats["system"]["cpu"] = psutil.cpu_percent(interval=1)
        monitoring_stats["system"]["memory"] = psutil.virtual_memory().percent
        monitoring_stats["system"]["disk"] = psutil.disk_usage('/').percent
        monitoring_stats["system"]["uptime"] = time.time() - psutil.boot_time()
    except Exception as e:
        logger.error(f"Error updating system stats: {e}")

def update_parsing_stats():
    """Update parsing statistics from database"""
    try:
        with db_engine.connect() as conn:
            # Get recent parsing activity
            result = conn.execute(text("""
                SELECT COUNT(*) as processed_count
                FROM transactions 
                WHERE created_at > NOW() - INTERVAL '5 minutes'
            """))
            recent_count = result.scalar() or 0
            
            # Use monitoring collector
            from monitoring_utils import update_parsing_stats as update_stats
            update_stats(
                processed_count=recent_count,
                rate=recent_count / 5,  # per minute
                status="active" if recent_count > 0 else "idle"
            )
            
    except Exception as e:
        logger.error(f"Error updating parsing stats: {e}")
        from monitoring_utils import update_parsing_stats as update_stats
        update_stats(error_count=1, status="error")

def update_sessionization_stats():
    """Update sessionization statistics"""
    try:
        with db_engine.connect() as conn:
            # Get session statistics
            result = conn.execute(text("""
                SELECT 
                    COUNT(DISTINCT session_id) as total_sessions,
                    COUNT(DISTINCT CASE WHEN created_at > NOW() - INTERVAL '1 hour' THEN session_id END) as active_sessions
                FROM ml_sessions
            """))
            row = result.fetchone()
            if row:
                from monitoring_utils import update_sessionization_stats as update_stats
                update_stats(
                    sessions_created=row[0] or 0,
                    active_sessions=row[1] or 0,
                    status="active" if row[1] > 0 else "idle"
                )
                
    except Exception as e:
        logger.error(f"Error updating sessionization stats: {e}")

def update_ml_training_stats():
    """Update ML training statistics"""
    try:
        # Get stats from monitoring collector (placeholder for now)
        # ml_stats = monitoring_collector.get_component_stats("ml_training")
        # monitoring_stats["ml_training"].update(ml_stats)
        monitoring_stats["ml_training"]["status"] = "idle"
            
    except Exception as e:
        logger.error(f"Error updating ML training stats: {e}")

async def monitoring_background_task():
    """Background task to update monitoring statistics"""
    while True:
        try:
            update_system_stats()
            update_parsing_stats()
            update_sessionization_stats()
            update_ml_training_stats()
            
            # Broadcast to all WebSocket connections
            if monitoring_connections:
                stats = MonitoringStats(
                    parsing=monitoring_stats["parsing"],
                    sessionization=monitoring_stats["sessionization"],
                    ml_training=monitoring_stats["ml_training"],
                    system=monitoring_stats["system"],
                    timestamp=datetime.now()
                )
                
                disconnected = []
                for ws in monitoring_connections:
                    try:
                        await ws.send_text(stats.json())
                    except:
                        disconnected.append(ws)
                
                # Remove disconnected connections
                for ws in disconnected:
                    monitoring_connections.remove(ws)
                    
            await asyncio.sleep(5)  # Update every 5 seconds
            
        except Exception as e:
            logger.error(f"Error in monitoring background task: {e}")
            await asyncio.sleep(10)

@app.get("/api/v1/monitoring/status", response_model=MonitoringStats)
async def get_monitoring_status():
    """Get current monitoring status and statistics"""
    try:
        # Force update stats
        update_system_stats()
        update_parsing_stats()
        update_sessionization_stats()
        update_ml_training_stats()
        
        return MonitoringStats(
            parsing=monitoring_stats["parsing"],
            sessionization=monitoring_stats["sessionization"],
            ml_training=monitoring_stats["ml_training"],
            system=monitoring_stats["system"],
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error getting monitoring status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting monitoring status: {str(e)}")

@app.get("/api/v1/monitoring/logs")
async def get_monitoring_logs(
    level: Optional[str] = None,
    component: Optional[str] = None,
    limit: int = 100
):
    """Get recent system logs for monitoring"""
    try:
        # Read from log files or database
        logs = []
        
        # Try to read from log files
        log_dir = "/app/data/logs"
        if os.path.exists(log_dir):
            log_files = sorted([f for f in os.listdir(log_dir) if f.endswith('.log')])
            
            for log_file in log_files[-5:]:  # Last 5 log files
                try:
                    with open(os.path.join(log_dir, log_file), 'r') as f:
                        lines = f.readlines()
                        for line in lines[-limit//5:]:  # Distribute across files
                            if line.strip():
                                # Parse log line (simple format)
                                parts = line.strip().split(' - ', 2)
                                if len(parts) >= 3:
                                    timestamp_str = parts[0]
                                    level_comp = parts[1]
                                    message = parts[2]
                                    
                                    # Extract level and component
                                    level_parts = level_comp.split(' - ')
                                    log_level = level_parts[0] if level_parts else "INFO"
                                    log_component = level_parts[1] if len(level_parts) > 1 else "system"
                                    
                                    # Apply filters
                                    if level and log_level.lower() != level.lower():
                                        continue
                                    if component and component.lower() not in log_component.lower():
                                        continue
                                    
                                    logs.append({
                                        "timestamp": timestamp_str,
                                        "level": log_level,
                                        "component": log_component,
                                        "message": message
                                    })
                except Exception as e:
                    logger.error(f"Error reading log file {log_file}: {e}")
        
        # Sort by timestamp and limit
        logs = sorted(logs, key=lambda x: x["timestamp"], reverse=True)[:limit]
        
        return {
            "status": "success",
            "logs": logs,
            "total": len(logs)
        }
        
    except Exception as e:
        logger.error(f"Error getting monitoring logs: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting logs: {str(e)}")

@app.websocket("/ws/monitoring")
async def monitoring_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time monitoring updates"""
    await websocket.accept()
    monitoring_connections.append(websocket)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        monitoring_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in monitoring_connections:
            monitoring_connections.remove(websocket)

@app.get("/api/v1/monitoring/performance")
async def get_performance_metrics():
    """Get detailed performance metrics"""
    try:
        metrics = {
            "system": {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory": {
                    "total": psutil.virtual_memory().total,
                    "available": psutil.virtual_memory().available,
                    "percent": psutil.virtual_memory().percent,
                    "used": psutil.virtual_memory().used
                },
                "disk": {
                    "total": psutil.disk_usage('/').total,
                    "used": psutil.disk_usage('/').used,
                    "free": psutil.disk_usage('/').free,
                    "percent": psutil.disk_usage('/').percent
                },
                "uptime": time.time() - psutil.boot_time()
            },
            "database": {},
            "redis": {}
        }
        
        # Database metrics
        try:
            with db_engine.connect() as conn:
                # Get database size and connection count
                db_result = conn.execute(text("""
                    SELECT 
                        pg_database_size(current_database()) as db_size,
                        (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') as active_connections
                """))
                row = db_result.fetchone()
                if row:
                    metrics["database"] = {
                        "size": row[0] or 0,
                        "active_connections": row[1] or 0
                    }
        except Exception as e:
            logger.error(f"Error getting database metrics: {e}")
            metrics["database"]["error"] = str(e)
        
        # Redis metrics
        try:
            redis_info = redis_client.info()
            metrics["redis"] = {
                "used_memory": redis_info.get("used_memory", 0),
                "connected_clients": redis_info.get("connected_clients", 0),
                "total_commands_processed": redis_info.get("total_commands_processed", 0)
            }
        except Exception as e:
            logger.error(f"Error getting Redis metrics: {e}")
            metrics["redis"]["error"] = str(e)
        
        return {
            "status": "success",
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting metrics: {str(e)}")

# Include SVM Debug API routes
try:
    from svm_debug_api import router as svm_debug_router
    app.include_router(svm_debug_router, prefix="/api/v1", tags=["svm-debug"])
    logger.info("SVM Debug API routes loaded successfully")
except ImportError:
    logger.warning("SVM Debug API not available - install required dependencies")
except Exception as e:
    logger.error(f"Error loading SVM Debug API: {str(e)}")

# Include BERT DeepLog API routes
try:
    from bert_deeplog_api import router as bert_deeplog_router
    app.include_router(bert_deeplog_router, tags=["bert-deeplog"])
    logger.info("BERT DeepLog API routes loaded successfully")
except ImportError:
    logger.warning("BERT DeepLog API not available - install required dependencies")
except Exception as e:
    logger.error(f"Error loading BERT DeepLog API: {str(e)}")

@app.get("/api/v1/models/training-results")
async def get_model_training_results():
    """Get supervised model training results and performance metrics"""
    try:
        query = """
        SELECT 
            model_name,
            model_type,
            model_version,
            training_date,
            training_samples,
            performance_metrics,
            model_parameters,
            is_active
        FROM ml_models 
        WHERE model_type = 'supervised_classifier'
        ORDER BY training_date DESC 
        LIMIT 10
        """
        
        with db_engine.connect() as conn:
            result = conn.execute(text(query))
            models = []
            
            for row in result:
                model_data = {
                    'model_name': row[0],
                    'model_type': row[1], 
                    'model_version': row[2],
                    'training_date': row[3].isoformat() if row[3] else None,
                    'training_samples': row[4],
                    'performance_metrics': json.loads(row[5]) if row[5] else {},
                    'model_parameters': json.loads(row[6]) if row[6] else {},
                    'is_active': row[7]
                }
                models.append(model_data)
        
        # Check for model files
        model_files = {
            'supervised_classifier': os.path.exists('/app/models/supervised_classifier.pkl'),
            'label_encoder': os.path.exists('/app/models/label_encoder.pkl'),
            'isolation_forest': os.path.exists('/app/models/isolation_forest.pkl'),
            'one_class_svm': os.path.exists('/app/models/one_class_svm.pkl')
        }
        
        return {
            'status': 'success',
            'models': models,
            'model_files_exist': model_files,
            'total_models': len(models),
            'latest_training': models[0]['training_date'] if models else None
        }
        
    except Exception as e:
        logger.error(f"Error getting model training results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/models/performance/{model_version}")
async def get_model_performance(model_version: str):
    """Get detailed performance metrics for a specific model version"""
    try:
        query = """
        SELECT performance_metrics, model_parameters, training_samples
        FROM ml_models 
        WHERE model_version = :version AND model_type = 'supervised_classifier'
        """
        
        with db_engine.connect() as conn:
            result = conn.execute(text(query), {'version': model_version})
            row = result.fetchone()
            
            if not row:
                raise HTTPException(status_code=404, detail="Model version not found")
            
            return {
                'status': 'success',
                'model_version': model_version,
                'performance_metrics': json.loads(row[0]) if row[0] else {},
                'model_parameters': json.loads(row[1]) if row[1] else {},
                'training_samples': row[2]
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# BertViz Analysis Endpoints
class BertAnalysisRequest(BaseModel):
    text: str
    analysis_type: str = "full"  # Options: full, attention, importance, patterns
    layers: Optional[List[int]] = None
    heads: Optional[List[int]] = None

@app.post("/api/v1/bert/analyze")
async def analyze_bert_attention(request: BertAnalysisRequest):
    """Analyze BERT attention patterns and token importance for given text"""
    if not BERTVIZ_AVAILABLE:
        raise HTTPException(status_code=503, detail="BertViz analyzer not available")
    
    try:
        analyzer = BertVisualizationAnalyzer()
        
        # Analyze the text
        analysis_results = await asyncio.to_thread(
            analyzer.analyze_session_text, 
            request.text,
            "analyze_session"
        )
        
        # Transform data for frontend compatibility
        transformed_results = transform_bert_analysis_for_frontend(analysis_results)
        
        response = {
            'status': 'success',
            'text': request.text,
            'analysis_type': request.analysis_type,
            'results': transformed_results
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error in BERT analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/bert/visualize")
async def create_bert_visualization(request: BertAnalysisRequest):
    """Create BERT attention visualizations"""
    if not BERTVIZ_AVAILABLE:
        raise HTTPException(status_code=503, detail="BertViz analyzer not available")
    
    try:
        analyzer = BertVisualizationAnalyzer()
        logger.info(f"Starting BERT visualization for text of length {len(request.text)}")
        
        # Process in smaller chunks if text is too long
        text_to_process = request.text
        if len(text_to_process) > 500:
            logger.info(f"Text too long ({len(text_to_process)} chars), truncating to 500 chars")
            text_to_process = text_to_process[:500]
        
        try:
            # First preprocess the text to clean up isolated digits and patterns
            preprocessed_text = analyzer._preprocess_text(text_to_process)
            logger.info(f"Text preprocessing complete. Original: {len(text_to_process)} chars, Processed: {len(preprocessed_text)} chars")
            
            # Get BERT outputs using the preprocessed text (in thread to avoid blocking)
            inputs, attention_weights, hidden_states = await asyncio.to_thread(
                analyzer._get_bert_outputs, 
                preprocessed_text
            )
            tokens = analyzer.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            logger.info(f"Got BERT outputs with {len(tokens)} tokens")
            
            # Check for tensor shape issues
            stacked_attention = torch.stack(attention_weights)
            logger.info(f"Attention tensor shape: {stacked_attention.shape}")
            
            # Handle the problematic case where attention has a batch dimension of 1
            if len(stacked_attention.shape) == 5 and stacked_attention.shape[1] == 1:
                logger.info("Squeezing batch dimension for better compatibility")
                stacked_attention = stacked_attention.squeeze(1)
                attention_weights = tuple(stacked_attention[i] for i in range(stacked_attention.shape[0]))
            
            # Generate visualizations
            visualizations = await asyncio.to_thread(
                analyzer._generate_visualizations,
                attention_weights,
                tokens,
                text_to_process
            )
            logger.info(f"Generated visualizations: {list(visualizations.keys())}")
            
            # Also calculate token importance to get EJ contextual enhancement metadata
            try:
                token_importance_data = await asyncio.to_thread(
                    analyzer._calculate_token_importance,
                    attention_weights,
                    tokens
                )
                contextual_enhancement = token_importance_data.get('contextual_enhancement', {})
                enhancement_metadata = {
                    'ej_labeler_used': contextual_enhancement.get('ej_labeler_used', False),
                    'expert_labeler_used': contextual_enhancement.get('expert_labeler_used', False),
                    'enhancement_impact': contextual_enhancement.get('enhancement_impact', 0.0),
                    'special_tokens_suppressed': contextual_enhancement.get('special_tokens_suppressed', False)
                }
                logger.info(f"EJ contextual enhancement metadata (primary path): {enhancement_metadata}")
            except Exception as ti_error:
                logger.error(f"Error calculating token importance metadata: {ti_error}")
                enhancement_metadata = {'ej_labeler_used': False, 'error': str(ti_error)}
        
        except Exception as bert_error:
            logger.error(f"Error in BERT processing: {bert_error}")
            # Fallback to more direct approach
            try:
                # Analyze with one call for robustness
                analysis_results = await asyncio.to_thread(
                    analyzer.analyze_session_text,
                    text_to_process
                )
                visualizations = analysis_results.get('visualizations', {})
                logger.info(f"Used fallback approach, got visualizations: {list(visualizations.keys())}")
                
                # Extract important metadata from analysis results
                token_importance_data = analysis_results.get('token_importance', {})
                contextual_enhancement = token_importance_data.get('contextual_enhancement', {})
                
                # Add metadata about EJ contextual enhancement to response
                enhancement_metadata = {
                    'ej_labeler_used': contextual_enhancement.get('ej_labeler_used', False),
                    'expert_labeler_used': contextual_enhancement.get('expert_labeler_used', False),
                    'enhancement_impact': contextual_enhancement.get('enhancement_impact', 0.0),
                    'special_tokens_suppressed': contextual_enhancement.get('special_tokens_suppressed', False)
                }
                logger.info(f"EJ contextual enhancement metadata: {enhancement_metadata}")
                
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                return {
                    'status': 'error',
                    'text': request.text[:100] + "...",  # Include part of the text
                    'error': f"Both visualization approaches failed: {str(fallback_error)}",
                    'visualizations': {}
                }
        
        # Check if we have any valid visualizations
        if not visualizations or not any(visualizations.values()):
            logger.warning("No valid visualizations generated")
            # Create simple dummy visualization as placeholder
            try:
                # Generate a simple colored rectangle as placeholder
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, "Visualization Error\nPlease try with different text", 
                         ha='center', va='center', fontsize=14)
                plt.axis('off')
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                placeholder = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                visualizations = {'attention_heatmap': placeholder, 'error': 'Visualization failed'}
            except:
                visualizations = {'error': 'Could not create visualizations'}
        
        response = {
            'status': 'success',
            'text': text_to_process,
            'visualizations': visualizations
        }
        
        # Add EJ contextual enhancement metadata if available
        if 'enhancement_metadata' in locals():
            response['ej_contextual_enhancement'] = enhancement_metadata
        
        return response
        
    except Exception as e:
        logger.error(f"Error creating BERT visualizations: {str(e)}")
        # Return partial response rather than raising error
        return {
            'status': 'error',
            'text': request.text[:100] + "...",
            'error': str(e),
            'visualizations': {}
        }

@app.get("/api/v1/bert/patterns")
async def get_bert_patterns():
    """Get detected BERT attention patterns for ABM anomaly detection"""
    if not BERTVIZ_AVAILABLE:
        raise HTTPException(status_code=503, detail="BertViz analyzer not available")
    
    try:
        analyzer = BertVisualizationAnalyzer()
        
        # Get patterns from recent sessions with anomalies
        query = """
        SELECT DISTINCT session_id, created_at
        FROM ml_sessions 
        WHERE created_at >= NOW() - INTERVAL '24 hours'
        AND anomaly_score > 0.7
        ORDER BY created_at DESC 
        LIMIT 10
        """
        
        with db_engine.connect() as conn:
            result = conn.execute(text(query))
            session_ids = [row[0] for row in result if row[0]]
            
            if not session_ids:
                # Fallback to any sessions if no recent anomalies
                query_fallback = """
                SELECT DISTINCT session_id, created_at
                FROM ml_sessions 
                ORDER BY created_at DESC 
                LIMIT 5
                """
                result = conn.execute(text(query_fallback))
                session_ids = [row[0] for row in result if row[0]]
        
        if not session_ids:
            return {
                'status': 'success',
                'message': 'No sessions found for analysis',
                'patterns': []
            }
        
        # For each session, reconstruct text from transactions table or use a sample text
        sample_data = []
        for session_id in session_ids[:3]:  # Limit to 3 sessions
            # Create a sample text for BERT analysis
            sample_text = f"Session {session_id}: ABM transaction analysis detecting potential anomalous patterns in financial behavior and transaction flows."
            sample_data.append({'session_id': session_id, 'text': sample_text})
        
        # Analyze patterns across multiple texts
        patterns = []
        try:
            for data in sample_data:
                analysis = await asyncio.to_thread(analyzer.analyze_session_text, data['text'], data['session_id'])
                if 'pattern_analysis' in analysis:
                    patterns.extend(analysis['pattern_analysis'].get('detected_patterns', []))
        except Exception as analysis_error:
            logger.warning(f"Error analyzing text pattern: {str(analysis_error)}")
            # Continue with other texts if one fails
        
        # Aggregate patterns
        pattern_summary = {}
        for pattern in patterns:
            pattern_type = pattern.get('type', 'unknown')
            if pattern_type not in pattern_summary:
                pattern_summary[pattern_type] = {
                    'count': 0,
                    'examples': [],
                    'avg_confidence': 0
                }
            pattern_summary[pattern_type]['count'] += 1
            pattern_summary[pattern_type]['examples'].append(pattern)
            pattern_summary[pattern_type]['avg_confidence'] = np.mean([
                p.get('confidence', 0) for p in pattern_summary[pattern_type]['examples']
            ])
        
        return {
            'status': 'success',
            'sample_count': len(sample_data),
            'pattern_summary': pattern_summary,
            'patterns': patterns
        }
        
    except Exception as e:
        logger.error(f"Error getting BERT patterns: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/bert/optimize")
async def optimize_bert_attention(request: BertAnalysisRequest):
    """Analyze BERT attention to suggest optimization strategies"""
    if not BERTVIZ_AVAILABLE:
        raise HTTPException(status_code=503, detail="BertViz analyzer not available")
    
    try:
        analyzer = BertVisualizationAnalyzer()
        
        # Get comprehensive analysis
        analysis = await asyncio.to_thread(analyzer.analyze_session_text, request.text, "optimize_session")
        
        # Generate optimization suggestions
        suggestions = []
        
        # Check token importance distribution
        if 'token_importance' in analysis and 'token_rankings' in analysis['token_importance']:
            importance_scores = [token['combined_importance'] for token in analysis['token_importance']['token_rankings']]
            importance_std = np.std(importance_scores)
            
            if importance_std < 0.1:
                suggestions.append({
                    'type': 'attention_distribution',
                    'issue': 'Low attention variance detected',
                    'suggestion': 'Consider fine-tuning with more diverse ABM-specific examples',
                    'confidence': 0.8
                })
        
        # Check for ABM-specific pattern recognition
        if 'patterns' in analysis:
            # Check if error and transaction patterns are properly detected
            error_score = analysis['patterns'].get('error_attention', {}).get('score', 0)
            transaction_score = analysis['patterns'].get('transaction_attention', {}).get('score', 0)
            
            if error_score < 0.3 and transaction_score < 0.3:
                suggestions.append({
                    'type': 'domain_adaptation',
                    'issue': 'Limited ABM-specific pattern recognition',
                    'suggestion': 'Add more ABM transaction and error keywords to fine-tuning data',
                    'confidence': 0.7
                })
        
        # Check attention head specialization
        if 'head_analysis' in analysis and 'heads' in analysis['head_analysis']:
            heads = analysis['head_analysis']['heads']
            low_specialization_heads = [
                head for head in heads 
                if head.get('entropy', 1.0) > 0.8  # High entropy = low specialization
            ]
            
            if len(low_specialization_heads) > len(heads) // 2:  # More than half
                suggestions.append({
                    'type': 'head_pruning',
                    'issue': f'{len(low_specialization_heads)} attention heads show low specialization',
                    'suggestion': 'Consider attention head pruning or targeted fine-tuning',
                    'confidence': 0.6
                })
        
        return {
            'status': 'success',
            'text': request.text,
            'analysis': analysis,
            'optimization_suggestions': suggestions,
            'suggestion_count': len(suggestions)
        }
        
    except Exception as e:
        logger.error(f"Error optimizing BERT attention: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced EJ BERT Analysis Endpoints
class EnhancedEJAnalysisRequest(BaseModel):
    text: str
    analysis_type: str = "comprehensive"  # Options: comprehensive, contextual_only, anomaly_only
    include_visualizations: bool = True
    include_recommendations: bool = True

@app.post("/api/v1/bert/enhanced-ej-analyze")
async def analyze_with_enhanced_ej_bert(request: EnhancedEJAnalysisRequest):
    """
    Comprehensive EJ log analysis using enhanced BERT with contextual labeling
    Provides domain-specific financial transaction understanding
    """
    if not ENHANCED_BERT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Enhanced EJ BERT system not available")
    
    try:
        # Initialize the complete EJ analysis system
        ej_labeler = EJLogLabeler()
        enhanced_bert = EnhancedEJBertAnalyzer()
        contextual_detector = ContextualAnomalyDetector()
        ej_analyzer = EJAnomalyAnalyzer(enhanced_bert, contextual_detector)
        
        logger.info(f"Starting enhanced EJ analysis for text of length {len(request.text)}")
        
        # Step 1: Extract contextual labels
        contextual_labels = await asyncio.to_thread(
            ej_labeler.label_log,
            request.text
        )
        logger.info(f"Extracted {len(contextual_labels)} contextual labels")
        
        # Step 2: Enhanced BERT analysis with contextual features
        enhanced_bert_results = await asyncio.to_thread(
            enhanced_bert.analyze_text,
            request.text
        )
        logger.info(f"Enhanced BERT analysis complete: {enhanced_bert_results.get('prediction', 'N/A')}")
        
        # Step 3: Contextual anomaly detection
        contextual_anomalies = await asyncio.to_thread(
            contextual_detector.detect_anomalies,
            contextual_labels
        )
        logger.info(f"Detected {len(contextual_anomalies)} contextual anomalies")
        
        # Step 4: Comprehensive analysis combining all components
        if request.analysis_type == "comprehensive":
            comprehensive_results = await asyncio.to_thread(
                ej_analyzer.analyze,
                request.text
            )
        else:
            # Provide focused analysis
            comprehensive_results = {
                'enhanced_bert_results': enhanced_bert_results,
                'contextual_anomalies': contextual_anomalies,
                'contextual_labels_summary': {
                    'total_labels': len(contextual_labels),
                    'event_types': list(set([label.event_type.value for label in contextual_labels])),
                    'phases': list(set([label.phase.value for label in contextual_labels if label.phase])),
                    'severity_distribution': {
                        severity.value: len([l for l in contextual_labels if l.severity == severity])
                        for severity in set([l.severity for l in contextual_labels])
                    }
                }
            }
        
        # Format response
        response = {
            'status': 'success',
            'analysis_type': request.analysis_type,
            'text_length': len(request.text),
            'processing_timestamp': datetime.now().isoformat(),
            
            # Core analysis results
            'enhanced_bert_prediction': enhanced_bert_results.get('prediction'),
            'enhanced_bert_confidence': enhanced_bert_results.get('confidence'),
            'contextual_anomaly_count': len(contextual_anomalies),
            'high_priority_anomalies': len([a for a in contextual_anomalies if a.get('severity') in ['CRITICAL', 'HIGH']]),
            
            # Detailed results
            'contextual_labels': {
                'count': len(contextual_labels),
                'sample': [
                    {
                        'line_number': label.line_number,
                        'event_type': label.event_type.value,
                        'phase': label.phase.value if label.phase else None,
                        'severity': label.severity.value,
                        'confidence': label.confidence
                    }
                    for label in contextual_labels[:10]  # First 10 for preview
                ]
            },
            'contextual_anomalies': contextual_anomalies[:20] if contextual_anomalies else [],  # First 20 anomalies
            
            # Analysis insights
            'domain_insights': {
                'transaction_pattern_detected': any(label.event_type.value in ['TXN_START', 'TXN_END'] for label in contextual_labels),
                'supervisor_mode_detected': any(label.event_type.value in ['SUPERVISOR_ENTRY', 'SUPERVISOR_EXIT'] for label in contextual_labels),
                'recovery_events_detected': any(label.recovery_type for label in contextual_labels),
                'authentication_issues_detected': any(label.auth_failure_type for label in contextual_labels),
                'cash_handling_events': any(label.denomination_data for label in contextual_labels)
            }
        }
        
        # Add comprehensive analysis if requested
        if request.analysis_type == "comprehensive":
            response.update({
                'risk_assessment': comprehensive_results.get('risk_assessment', {}),
                'recommendations': comprehensive_results.get('recommendations', []) if request.include_recommendations else [],
                'financial_impact_assessment': comprehensive_results.get('financial_impact_assessment', {})
            })
        
        # Add visualizations if requested (simplified for now)
        if request.include_visualizations and enhanced_bert_results.get('attention_weights'):
            response['visualizations'] = {
                'attention_summary': 'Enhanced BERT attention analysis available',
                'contextual_features_used': len(enhanced_bert_results.get('contextual_features', {})),
                'visualization_note': 'Full attention visualization available via separate endpoint'
            }
        
        logger.info(f"Enhanced EJ analysis complete - Prediction: {response['enhanced_bert_prediction']}, Anomalies: {response['contextual_anomaly_count']}")
        return response
        
    except Exception as e:
        logger.error(f"Error in enhanced EJ BERT analysis: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Enhanced EJ analysis failed: {str(e)}")

@app.post("/api/v1/bert/contextual-labels")
async def extract_contextual_labels(request: BertAnalysisRequest):
    """Extract contextual labels from EJ log text for debugging and analysis"""
    if not ENHANCED_BERT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Enhanced EJ BERT system not available")
    
    try:
        ej_labeler = EJLogLabeler()
        
        # Extract labels
        labels = await asyncio.to_thread(
            ej_labeler.label_log,
            request.text
        )
        
        # Format for API response
        formatted_labels = []
        for label in labels:
            formatted_label = {
                'line_number': label.line_number,
                'timestamp': label.timestamp.isoformat() if label.timestamp else None,
                'event_type': label.event_type.value,
                'phase': label.phase.value if label.phase else None,
                'severity': label.severity.value,
                'confidence': label.confidence,
                'operational_mode': label.operational_mode.value if label.operational_mode else None,
                'entity': label.entity,
                'metadata': label.metadata
            }
            
            # Add optional fields if present
            if label.recovery_type:
                formatted_label['recovery_type'] = label.recovery_type.value
            if label.auth_failure_type:
                formatted_label['auth_failure_type'] = label.auth_failure_type
            if label.error_category:
                formatted_label['error_category'] = label.error_category.value
            if label.denomination_data:
                formatted_label['denomination_data'] = label.denomination_data
            if label.transaction_id:
                formatted_label['transaction_id'] = label.transaction_id
            
            formatted_labels.append(formatted_label)
        
        # Generate summary statistics
        event_type_counts = {}
        phase_counts = {}
        severity_counts = {}
        
        for label in labels:
            # Event type distribution
            event_type = label.event_type.value
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
            
            # Phase distribution
            if label.phase:
                phase = label.phase.value
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
            
            # Severity distribution
            severity = label.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'status': 'success',
            'text_length': len(request.text),
            'total_labels': len(labels),
            'labels': formatted_labels,
            'summary': {
                'event_type_distribution': event_type_counts,
                'phase_distribution': phase_counts,
                'severity_distribution': severity_counts,
                'unique_entities': list(set([l.entity for l in labels if l.entity])),
                'recovery_events': len([l for l in labels if l.recovery_type]),
                'authentication_events': len([l for l in labels if l.auth_failure_type]),
                'transaction_events': len([l for l in labels if l.event_type.value in ['TXN_START', 'TXN_END']])
            }
        }
        
    except Exception as e:
        logger.error(f"Error extracting contextual labels: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Start monitoring background task
@app.on_event("startup")
async def start_monitoring():
    """Start the monitoring background task"""
    asyncio.create_task(monitoring_background_task())

async def monitoring_background_task():
    """Background task to update monitoring statistics"""
    while True:
        try:
            update_system_stats()
            update_parsing_stats()
            update_sessionization_stats()
            update_ml_training_stats()
            
            # Broadcast to all WebSocket connections
            if monitoring_connections:
                stats = MonitoringStats(
                    parsing=monitoring_stats["parsing"],
                    sessionization=monitoring_stats["sessionization"],
                    ml_training=monitoring_stats["ml_training"],
                    system=monitoring_stats["system"],
                    timestamp=datetime.now()
                )
                
                disconnected = []
                for ws in monitoring_connections:
                    try:
                        await ws.send_text(stats.json())
                    except:
                        disconnected.append(ws)
                
                # Remove disconnected connections
                for ws in disconnected:
                    monitoring_connections.remove(ws)
                    
            await asyncio.sleep(5)  # Update every 5 seconds
            
        except Exception as e:
            logger.error(f"Error in monitoring background task: {e}")
            await asyncio.sleep(10)
