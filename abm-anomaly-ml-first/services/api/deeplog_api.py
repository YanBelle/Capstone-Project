"""
DeepLog API Integration Module
API endpoints for DeepLog training and prediction functionality
"""

from fastapi import HTTPException
from typing import Dict, Any, List
import asyncio
import glob
import os
from loguru import logger

# DeepLog Training Integration
try:
    from deeplog_bert_trainer import get_deeplog_trainer
    deeplog_available = True
except ImportError:
    logger.warning("DeepLog BERT trainer not available - PyTorch/transformers not installed")
    deeplog_available = False

async def retrain_deeplog_model(get_db_connection):
    """Retrain DeepLog model using BERT embeddings on available session data"""
    try:
        if not deeplog_available:
            return {
                "status": "error",
                "message": "DeepLog training not available - PyTorch/transformers dependencies missing"
            }
        
        # Get training data from database
        async with get_db_connection() as conn:
            sessions = await conn.fetch("""
                SELECT session_id, raw_text, anomaly_score, 
                       (anomaly_score > 0.5) as is_anomaly,
                       CASE 
                           WHEN anomaly_score > 0.8 THEN 'high_anomaly'
                           WHEN anomaly_score > 0.5 THEN 'medium_anomaly'
                           ELSE 'normal'
                       END as anomaly_type
                FROM ml_sessions 
                WHERE raw_text IS NOT NULL 
                AND raw_text != 'Raw text not available'
                ORDER BY created_at DESC
                LIMIT 1000
            """)
        
        if len(sessions) < 10:
            # If not enough data in database, try to collect from file system
            file_data = []
            processed_files = glob.glob("/app/input/processed/*.txt")
            
            for file_path in processed_files[:100]:  # Limit to 100 files
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            # Simple heuristic for anomaly detection during collection
                            anomaly_score = 0.1  # Default normal
                            if any(keyword in content.lower() for keyword in ['error', 'exception', 'fail', 'timeout']):
                                anomaly_score = 0.7
                            
                            file_data.append({
                                'session_id': os.path.basename(file_path).replace('.txt', ''),
                                'raw_text': content,
                                'anomaly_score': anomaly_score,
                                'is_anomaly': anomaly_score > 0.5,
                                'anomaly_type': 'error_pattern' if anomaly_score > 0.5 else 'normal'
                            })
                except Exception as e:
                    logger.warning(f"Error reading file {file_path}: {e}")
                    continue
            
            sessions.extend(file_data)
        
        if len(sessions) < 5:
            return {
                "status": "error",
                "message": f"Not enough training data available. Found {len(sessions)} sessions, need at least 5."
            }
        
        # Convert to list of dicts for training
        training_data = [dict(session) for session in sessions]
        
        # Initialize trainer and train model
        trainer = get_deeplog_trainer()
        training_results = trainer.train_model(
            sessions_data=training_data,
            epochs=10,
            batch_size=4,  # Small batch size for limited data
            learning_rate=1e-4
        )
        
        return {
            "status": "success",
            "message": f"DeepLog model retrained successfully on {len(training_data)} sessions",
            "training_results": training_results
        }
        
    except Exception as e:
        logger.error(f"Error retraining DeepLog model: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to retrain DeepLog model: {str(e)}"
        }

async def predict_with_deeplog(request: Dict[str, Any]):
    """Use DeepLog model to predict if a session is anomalous"""
    try:
        if not deeplog_available:
            return {
                "status": "error",
                "message": "DeepLog prediction not available - PyTorch/transformers dependencies missing"
            }
        
        raw_text = request.get('raw_text', '').strip()
        if not raw_text:
            return {
                "status": "error",
                "message": "No raw_text provided for prediction"
            }
        
        # Get trainer and make prediction
        trainer = get_deeplog_trainer()
        
        if trainer.deeplog_model is None:
            return {
                "status": "error",
                "message": "DeepLog model not trained yet. Please run /api/v1/deeplog/retrain first."
            }
        
        prediction = trainer.predict_anomaly(raw_text)
        
        return {
            "status": "success",
            "prediction": prediction
        }
        
    except Exception as e:
        logger.error(f"Error making DeepLog prediction: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to make prediction: {str(e)}"
        }

async def deeplog_model_status():
    """Get status of DeepLog model including training history"""
    try:
        if not deeplog_available:
            return {
                "status": "unavailable",
                "message": "DeepLog not available - PyTorch/transformers dependencies missing"
            }
        
        trainer = get_deeplog_trainer()
        
        if trainer.deeplog_model is None:
            return {
                "status": "not_trained",
                "message": "DeepLog model not trained yet",
                "available": True
            }
        
        return {
            "status": "ready",
            "message": "DeepLog model trained and ready for predictions",
            "available": True,
            "num_classes": len(trainer.label_encoder),
            "labels": list(trainer.label_encoder.keys()),
            "training_history": trainer.training_history[-5:] if trainer.training_history else []  # Last 5 epochs
        }
        
    except Exception as e:
        logger.error(f"Error getting DeepLog status: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to get status: {str(e)}"
        }
