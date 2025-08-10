"""
Enhanced Monitoring Utility Module
Provides functions to update monitoring statistics that can be called from various services
Includes progress tracking for EJ loading and model training
"""
import redis
import json
import time
import threading
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

class ProgressTracker:
    """Thread-safe progress tracking for various operations"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._operations = {}
        
    def start_operation(self, operation_id: str, total_items: int, operation_type: str):
        """Start tracking a new operation"""
        with self._lock:
            self._operations[operation_id] = {
                'type': operation_type,
                'total_items': total_items,
                'completed_items': 0,
                'current_item': None,
                'start_time': time.time(),
                'last_update': time.time(),
                'status': 'running',
                'error_count': 0,
                'errors': [],
                'rate': 0.0,
                'eta': None,
                'progress_percent': 0.0
            }
            logger.info(f"Started operation {operation_id} ({operation_type}) with {total_items} items")
    
    def update_progress(self, operation_id: str, completed_items: int, current_item: str = None, error: str = None):
        """Update progress for an operation"""
        with self._lock:
            if operation_id not in self._operations:
                return
                
            op = self._operations[operation_id]
            op['completed_items'] = completed_items
            if current_item:
                op['current_item'] = current_item
            
            current_time = time.time()
            time_elapsed = current_time - op['start_time']
            
            # Calculate progress percentage
            if op['total_items'] > 0:
                op['progress_percent'] = (completed_items / op['total_items']) * 100
            
            # Calculate rate and ETA
            if time_elapsed > 0:
                op['rate'] = completed_items / time_elapsed
                if op['rate'] > 0 and op['total_items'] > completed_items:
                    remaining = op['total_items'] - completed_items
                    op['eta'] = remaining / op['rate']
            
            op['last_update'] = current_time
            
            if error:
                op['error_count'] += 1
                op['errors'].append({
                    'timestamp': current_time,
                    'message': error
                })
                # Keep only last 10 errors
                op['errors'] = op['errors'][-10:]
                
            # Store in Redis for persistence
            try:
                redis_client.setex(
                    f"progress:{operation_id}",
                    3600,  # 1 hour expiry
                    json.dumps(op, default=str)
                )
            except Exception as e:
                logger.error(f"Error storing progress in Redis: {e}")
    
    def complete_operation(self, operation_id: str, success: bool = True):
        """Mark an operation as completed"""
        with self._lock:
            if operation_id not in self._operations:
                return
                
            op = self._operations[operation_id]
            op['status'] = 'completed' if success else 'failed'
            op['end_time'] = time.time()
            op['total_time'] = op['end_time'] - op['start_time']
            op['progress_percent'] = 100.0 if success else op.get('progress_percent', 0)
            
            logger.info(f"Completed operation {operation_id} in {op['total_time']:.2f}s (success: {success})")
    
    def get_operation_status(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of an operation"""
        with self._lock:
            return self._operations.get(operation_id, {}).copy()
    
    def get_all_operations(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all operations"""
        with self._lock:
            return {k: v.copy() for k, v in self._operations.items()}

# Global progress tracker instance
progress_tracker = ProgressTracker()

class MonitoringCollector:
    """Collects and stores monitoring data across different components"""
    
    def __init__(self):
        self.component_stats = {
            "parsing": {
                "rate": 0,
                "processed": 0,
                "total_files": 0,
                "current_file": None,
                "errors": 0,
                "status": "idle",
                "progress_percent": 0.0,
                "eta_seconds": None,
                "last_updated": time.time()
            },
            "sessionization": {
                "rate": 0,
                "sessions_created": 0,
                "current_session": None,
                "active_sessions": 0,
                "status": "idle",
                "last_updated": time.time()
            },
            "ml_training": {
                "accuracy": 0,
                "models_trained": 0,
                "training_time": 0,
                "status": "idle",
                "model_type": None,
                "training_progress": 0.0,
                "current_epoch": 0,
                "total_epochs": 0,
                "current_loss": 0.0,
                "best_accuracy": 0.0,
                "training_samples": 0,
                "eta_seconds": None,
                "last_model_update": None,
                "last_updated": time.time()
            }
        }
    
    def update_parsing_stats(self, processed_count: int = 0, total_files: int = 0, 
                           current_file: str = None, error_count: int = 0, 
                           status: str = "active", rate: float = 0, progress_percent: float = 0,
                           eta_seconds: float = None):
        """Update parsing component statistics"""
        try:
            self.component_stats["parsing"].update({
                "processed": self.component_stats["parsing"]["processed"] + processed_count,
                "total_files": total_files if total_files > 0 else self.component_stats["parsing"]["total_files"],
                "current_file": current_file,
                "errors": self.component_stats["parsing"]["errors"] + error_count,
                "status": status,
                "rate": rate,
                "progress_percent": progress_percent,
                "eta_seconds": eta_seconds,
                "last_updated": time.time()
            })
            
            # Store in Redis
            redis_client.setex(
                "monitoring:parsing", 
                300,  # 5 minute expiry
                json.dumps(self.component_stats["parsing"], default=str)
            )
            
            logger.info(f"Updated parsing stats: {processed_count}/{total_files} files, {progress_percent:.1f}% complete")
            
        except Exception as e:
            logger.error(f"Error updating parsing stats: {e}")
    
    def update_sessionization_stats(self, sessions_created: int = 0, 
                                  current_session: str = None,
                                  active_sessions: int = 0, status: str = "active"):
        """Update sessionization component statistics"""
        try:
            self.component_stats["sessionization"].update({
                "sessions_created": sessions_created,
                "current_session": current_session,
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
                json.dumps(self.component_stats["sessionization"], default=str)
            )
            
            logger.info(f"Updated sessionization stats: sessions={sessions_created}, active={active_sessions}")
            
        except Exception as e:
            logger.error(f"Error updating sessionization stats: {e}")
    
    def update_ml_training_stats(self, accuracy: float = 0, models_trained: int = 0,
                               training_time: float = 0, status: str = "idle",
                               model_type: Optional[str] = None, training_progress: float = 0,
                               current_epoch: int = 0, total_epochs: int = 0,
                               current_loss: float = 0, best_accuracy: float = 0,
                               training_samples: int = 0, eta_seconds: float = None):
        """Update ML training component statistics"""
        try:
            self.component_stats["ml_training"].update({
                "accuracy": accuracy,
                "models_trained": self.component_stats["ml_training"]["models_trained"] + models_trained,
                "training_time": training_time,
                "status": status,
                "model_type": model_type,
                "training_progress": training_progress,
                "current_epoch": current_epoch,
                "total_epochs": total_epochs,
                "current_loss": current_loss,
                "best_accuracy": best_accuracy,
                "training_samples": training_samples,
                "eta_seconds": eta_seconds,
                "last_model_update": datetime.now().isoformat() if models_trained > 0 else self.component_stats["ml_training"]["last_model_update"],
                "last_updated": time.time()
            })
            
            # Store in Redis
            redis_client.setex(
                "monitoring:ml_training",
                300,  # 5 minute expiry
                json.dumps(self.component_stats["ml_training"], default=str)
            )
            
            logger.info(f"Updated ML training stats: {training_progress:.1f}% complete, accuracy={accuracy:.3f}")
            
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
def update_parsing_stats(processed_count: int = 0, total_files: int = 0, 
                        current_file: str = None, error_count: int = 0, 
                        status: str = "active", rate: float = 0, progress_percent: float = 0,
                        eta_seconds: float = None):
    """Update parsing statistics"""
    monitoring_collector.update_parsing_stats(
        processed_count, total_files, current_file, error_count, 
        status, rate, progress_percent, eta_seconds
    )

def update_sessionization_stats(sessions_created: int = 0, current_session: str = None,
                               active_sessions: int = 0, status: str = "active"):
    """Update sessionization statistics"""
    monitoring_collector.update_sessionization_stats(sessions_created, current_session, active_sessions, status)

def update_ml_training_stats(accuracy: float = 0, models_trained: int = 0,
                           training_time: float = 0, status: str = "idle",
                           model_type: Optional[str] = None, training_progress: float = 0,
                           current_epoch: int = 0, total_epochs: int = 0,
                           current_loss: float = 0, best_accuracy: float = 0,
                           training_samples: int = 0, eta_seconds: float = None):
    """Update ML training statistics"""
    monitoring_collector.update_ml_training_stats(
        accuracy, models_trained, training_time, status, model_type,
        training_progress, current_epoch, total_epochs, current_loss,
        best_accuracy, training_samples, eta_seconds
    )

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

# Progress tracking convenience functions
def start_ej_processing(total_files: int) -> str:
    """Start tracking EJ file processing"""
    operation_id = f"ej_processing_{int(time.time())}"
    progress_tracker.start_operation(operation_id, total_files, "ej_processing")
    update_parsing_stats(total_files=total_files, status="active", progress_percent=0.0)
    return operation_id

def update_ej_processing_progress(operation_id: str, completed_files: int, current_file: str = None, error: str = None):
    """Update EJ processing progress"""
    progress_tracker.update_progress(operation_id, completed_files, current_file, error)
    
    # Get operation details for stats update
    op_status = progress_tracker.get_operation_status(operation_id)
    if op_status:
        update_parsing_stats(
            processed_count=0,  # Don't increment, use absolute
            current_file=current_file,
            error_count=1 if error else 0,
            status="active",
            rate=op_status.get('rate', 0),
            progress_percent=op_status.get('progress_percent', 0),
            eta_seconds=op_status.get('eta')
        )

def complete_ej_processing(operation_id: str, success: bool = True):
    """Complete EJ processing operation"""
    progress_tracker.complete_operation(operation_id, success)
    update_parsing_stats(status="completed" if success else "failed", progress_percent=100.0 if success else None)

def start_model_training(model_type: str, total_epochs: int = 100, training_samples: int = 0) -> str:
    """Start tracking model training"""
    operation_id = f"training_{model_type}_{int(time.time())}"
    progress_tracker.start_operation(operation_id, total_epochs, "model_training")
    update_ml_training_stats(
        status="training",
        model_type=model_type,
        total_epochs=total_epochs,
        training_samples=training_samples,
        training_progress=0.0
    )
    return operation_id

def update_model_training_progress(operation_id: str, current_epoch: int, accuracy: float = 0, 
                                 loss: float = 0, error: str = None):
    """Update model training progress"""
    progress_tracker.update_progress(operation_id, current_epoch, f"Epoch {current_epoch}", error)
    
    # Get operation details for stats update
    op_status = progress_tracker.get_operation_status(operation_id)
    if op_status:
        update_ml_training_stats(
            accuracy=accuracy,
            current_epoch=current_epoch,
            current_loss=loss,
            best_accuracy=max(accuracy, op_status.get('best_accuracy', 0)),
            training_progress=op_status.get('progress_percent', 0),
            eta_seconds=op_status.get('eta'),
            status="training"
        )

def complete_model_training(operation_id: str, final_accuracy: float = 0, success: bool = True):
    """Complete model training operation"""
    progress_tracker.complete_operation(operation_id, success)
    update_ml_training_stats(
        accuracy=final_accuracy,
        models_trained=1 if success else 0,
        status="completed" if success else "failed",
        training_progress=100.0 if success else None
    )

def get_monitoring_summary() -> Dict[str, Any]:
    """Get comprehensive monitoring summary including progress"""
    all_stats = monitoring_collector.component_stats.copy()
    
    # Add active operations from progress tracker
    active_operations = {}
    for op_id, op_data in progress_tracker.get_all_operations().items():
        if op_data.get('status') == 'running':
            active_operations[op_id] = {
                'type': op_data.get('type'),
                'progress': op_data.get('progress_percent', 0),
                'current_item': op_data.get('current_item'),
                'rate': op_data.get('rate', 0),
                'eta': op_data.get('eta'),
                'error_count': op_data.get('error_count', 0)
            }
    
    return {
        'components': all_stats,
        'active_operations': active_operations,
        'timestamp': datetime.now().isoformat()
    }
