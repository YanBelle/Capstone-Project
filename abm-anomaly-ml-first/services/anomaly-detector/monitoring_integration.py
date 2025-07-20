"""
Monitoring integration for ML Analyzer
Provides functions to update monitoring statistics during ML operations
"""
import redis
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
import os

try:
    # Redis connection for monitoring data
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'redis'),
        port=6379,
        password=os.getenv('REDIS_PASSWORD', ''),
        decode_responses=True
    )
except Exception as e:
    print(f"Warning: Could not connect to Redis for monitoring: {e}")
    redis_client = None

def update_ml_training_stats(accuracy: float = 0, models_trained: int = 0,
                           training_time: float = 0, status: str = "idle",
                           model_type: Optional[str] = None):
    """Update ML training statistics"""
    if not redis_client:
        return
        
    try:
        ml_stats = {
            "accuracy": accuracy,
            "models_trained": models_trained,
            "training_time": training_time,
            "status": status,
            "last_model_update": datetime.now().isoformat() if models_trained > 0 else None,
            "last_updated": time.time()
        }
        
        if model_type:
            ml_stats["last_model_type"] = model_type
        
        # Store in Redis
        redis_client.setex(
            "monitoring:ml_training",
            300,  # 5 minute expiry
            json.dumps(ml_stats)
        )
        
        print(f"Updated ML training stats: accuracy={accuracy}, models={models_trained}, time={training_time}s")
        
    except Exception as e:
        print(f"Error updating ML training stats: {e}")

def log_ml_activity(activity: str, session_id: Optional[str] = None, 
                   details: Optional[Dict[str, Any]] = None):
    """Log ML component activity for monitoring"""
    if not redis_client:
        return
        
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "component": "ml_training",
            "activity": activity,
            "session_id": session_id,
            "details": details or {}
        }
        
        # Store in Redis list (keep last 1000 entries)
        redis_client.lpush("monitoring:activity_log", json.dumps(log_entry))
        redis_client.ltrim("monitoring:activity_log", 0, 999)
        
        print(f"Logged ML activity: {activity}")
        
    except Exception as e:
        print(f"Error logging ML activity: {e}")

def mark_ml_training_start(model_type: str = "unknown"):
    """Mark the start of ML training"""
    update_ml_training_stats(status="training", model_type=model_type)
    log_ml_activity(f"Started training {model_type} model")

def mark_ml_training_complete(accuracy: float, training_time: float, 
                            model_type: str = "unknown"):
    """Mark the completion of ML training"""
    update_ml_training_stats(
        accuracy=accuracy,
        models_trained=1,
        training_time=training_time,
        status="idle",
        model_type=model_type
    )
    log_ml_activity(f"Completed training {model_type} model", 
                   details={"accuracy": accuracy, "training_time": training_time})

def mark_ml_detection_run(session_count: int, anomaly_count: int):
    """Mark an anomaly detection run"""
    log_ml_activity("Anomaly detection run", 
                   details={"sessions_processed": session_count, "anomalies_detected": anomaly_count})

def mark_ml_error(error_message: str, context: Optional[str] = None):
    """Mark an ML error"""
    update_ml_training_stats(status="error")
    log_ml_activity(f"ML Error: {error_message}", details={"context": context})
