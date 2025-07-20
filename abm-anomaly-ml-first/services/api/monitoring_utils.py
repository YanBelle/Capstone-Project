"""
Monitoring Utility Module
Provides functions to update monitoring statistics that can be called from various services
"""
import redis
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
from loguru import logger
import os

# Redis connection for monitoring data
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'redis'),
    port=6379,
    password=os.getenv('REDIS_PASSWORD'),
    decode_responses=True
)

class MonitoringCollector:
    """Collects and stores monitoring data across different components"""
    
    def __init__(self):
        self.component_stats = {
            "parsing": {
                "rate": 0,
                "processed": 0,
                "errors": 0,
                "status": "idle",
                "last_updated": time.time()
            },
            "sessionization": {
                "rate": 0,
                "sessions_created": 0,
                "active_sessions": 0,
                "status": "idle",
                "last_updated": time.time()
            },
            "ml_training": {
                "accuracy": 0,
                "models_trained": 0,
                "training_time": 0,
                "status": "idle",
                "last_model_update": None,
                "last_updated": time.time()
            }
        }
    
    def update_parsing_stats(self, processed_count: int = 0, error_count: int = 0, 
                           status: str = "active", rate: float = 0):
        """Update parsing component statistics"""
        try:
            self.component_stats["parsing"].update({
                "processed": self.component_stats["parsing"]["processed"] + processed_count,
                "errors": self.component_stats["parsing"]["errors"] + error_count,
                "status": status,
                "rate": rate,
                "last_updated": time.time()
            })
            
            # Store in Redis
            redis_client.setex(
                "monitoring:parsing", 
                300,  # 5 minute expiry
                json.dumps(self.component_stats["parsing"])
            )
            
            logger.info(f"Updated parsing stats: processed={processed_count}, errors={error_count}, rate={rate}")
            
        except Exception as e:
            logger.error(f"Error updating parsing stats: {e}")
    
    def update_sessionization_stats(self, sessions_created: int = 0, 
                                  active_sessions: int = 0, status: str = "active"):
        """Update sessionization component statistics"""
        try:
            self.component_stats["sessionization"].update({
                "sessions_created": sessions_created,
                "active_sessions": active_sessions,
                "status": status,
                "last_updated": time.time()
            })
            
            # Calculate rate based on recent activity
            current_time = time.time()
            time_diff = current_time - self.component_stats["sessionization"]["last_updated"]
            if time_diff > 0:
                self.component_stats["sessionization"]["rate"] = sessions_created / (time_diff / 60)  # per minute
            
            # Store in Redis
            redis_client.setex(
                "monitoring:sessionization",
                300,  # 5 minute expiry
                json.dumps(self.component_stats["sessionization"])
            )
            
            logger.info(f"Updated sessionization stats: sessions={sessions_created}, active={active_sessions}")
            
        except Exception as e:
            logger.error(f"Error updating sessionization stats: {e}")
    
    def update_ml_training_stats(self, accuracy: float = 0, models_trained: int = 0,
                               training_time: float = 0, status: str = "idle",
                               model_type: Optional[str] = None):
        """Update ML training component statistics"""
        try:
            self.component_stats["ml_training"].update({
                "accuracy": accuracy,
                "models_trained": self.component_stats["ml_training"]["models_trained"] + models_trained,
                "training_time": training_time,
                "status": status,
                "last_model_update": datetime.now().isoformat() if models_trained > 0 else self.component_stats["ml_training"]["last_model_update"],
                "last_updated": time.time()
            })
            
            if model_type:
                self.component_stats["ml_training"]["last_model_type"] = model_type
            
            # Store in Redis
            redis_client.setex(
                "monitoring:ml_training",
                300,  # 5 minute expiry
                json.dumps(self.component_stats["ml_training"])
            )
            
            logger.info(f"Updated ML training stats: accuracy={accuracy}, models={models_trained}, time={training_time}s")
            
        except Exception as e:
            logger.error(f"Error updating ML training stats: {e}")
    
    def log_component_activity(self, component: str, activity: str, 
                             session_id: Optional[str] = None, 
                             details: Optional[Dict[str, Any]] = None):
        """Log component activity for monitoring"""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "component": component,
                "activity": activity,
                "session_id": session_id,
                "details": details or {}
            }
            
            # Store in Redis list (keep last 1000 entries)
            redis_client.lpush("monitoring:activity_log", json.dumps(log_entry))
            redis_client.ltrim("monitoring:activity_log", 0, 999)
            
            logger.info(f"Logged activity: {component} - {activity}")
            
        except Exception as e:
            logger.error(f"Error logging component activity: {e}")
    
    def get_component_stats(self, component: str) -> Dict[str, Any]:
        """Get statistics for a specific component"""
        try:
            redis_key = f"monitoring:{component}"
            cached_stats = redis_client.get(redis_key)
            
            if cached_stats:
                return json.loads(cached_stats)
            else:
                return self.component_stats.get(component, {})
                
        except Exception as e:
            logger.error(f"Error getting component stats: {e}")
            return {}
    
    def get_activity_log(self, limit: int = 100, component: Optional[str] = None) -> list:
        """Get recent activity log entries"""
        try:
            entries = redis_client.lrange("monitoring:activity_log", 0, limit - 1)
            activities = [json.loads(entry) for entry in entries]
            
            if component:
                activities = [a for a in activities if a.get("component") == component]
            
            return activities
            
        except Exception as e:
            logger.error(f"Error getting activity log: {e}")
            return []

# Global monitoring collector instance
monitoring_collector = MonitoringCollector()

# Convenience functions for easy use in other modules
def update_parsing_stats(processed_count: int = 0, error_count: int = 0, 
                        status: str = "active", rate: float = 0):
    """Update parsing statistics"""
    monitoring_collector.update_parsing_stats(processed_count, error_count, status, rate)

def update_sessionization_stats(sessions_created: int = 0, active_sessions: int = 0, 
                               status: str = "active"):
    """Update sessionization statistics"""
    monitoring_collector.update_sessionization_stats(sessions_created, active_sessions, status)

def update_ml_training_stats(accuracy: float = 0, models_trained: int = 0,
                           training_time: float = 0, status: str = "idle",
                           model_type: Optional[str] = None):
    """Update ML training statistics"""
    monitoring_collector.update_ml_training_stats(accuracy, models_trained, training_time, status, model_type)

def log_component_activity(component: str, activity: str, 
                         session_id: Optional[str] = None, 
                         details: Optional[Dict[str, Any]] = None):
    """Log component activity"""
    monitoring_collector.log_component_activity(component, activity, session_id, details)

def mark_component_idle(component: str):
    """Mark a component as idle"""
    if component == "parsing":
        update_parsing_stats(status="idle")
    elif component == "sessionization":
        update_sessionization_stats(status="idle")
    elif component == "ml_training":
        update_ml_training_stats(status="idle")

def mark_component_active(component: str):
    """Mark a component as active"""
    if component == "parsing":
        update_parsing_stats(status="active")
    elif component == "sessionization":
        update_sessionization_stats(status="active")
    elif component == "ml_training":
        update_ml_training_stats(status="training")
