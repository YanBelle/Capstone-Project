"""
DeepLog + BERT Integration for existing ABM Anomaly Detection System
Integrates the DeepLog + BERT model with the main anomaly detection service
"""

import os
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class DeepLogBertIntegration:
    """
    Integration layer for DeepLog + BERT with existing anomaly detection system
    """
    
    def __init__(self, model_path: str = "/app/models/deeplog_bert_model.pt"):
        self.model_path = model_path
        self.trainer = None
        self.is_model_loaded = False
        self.training_in_progress = False
        
        # Integration settings
        self.confidence_threshold = 0.7
        self.enable_realtime_detection = True
        self.batch_training_enabled = True
        
        logger.info("DeepLogBertIntegration initialized")
    
    async def initialize_model(self) -> bool:
        """
        Initialize the DeepLog + BERT model
        """
        try:
            # Import here to handle potential dependency issues gracefully
            from deeplog_bert_training_clean import create_deeplog_bert_trainer
            
            self.trainer = create_deeplog_bert_trainer(self.model_path)
            
            # Try to load existing model
            if os.path.exists(self.model_path):
                self.is_model_loaded = self.trainer.load_model()
                if self.is_model_loaded:
                    logger.info(f"DeepLog + BERT model loaded from {self.model_path}")
                else:
                    logger.warning("Failed to load existing model, will need training")
            else:
                logger.info("No existing model found, will need training")
            
            return True
            
        except ImportError as e:
            logger.error(f"DeepLog + BERT dependencies not available: {e}")
            logger.info("Install dependencies with: pip install -r deeplog_bert_requirements.txt")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize DeepLog + BERT model: {e}")
            return False
    
    async def train_model_async(self, session_texts: List[str], labels: List[int] = None) -> Dict[str, Any]:
        """
        Train the DeepLog + BERT model asynchronously
        """
        if self.training_in_progress:
            return {
                "status": "error",
                "message": "Training already in progress"
            }
        
        if not self.trainer:
            await self.initialize_model()
            if not self.trainer:
                return {
                    "status": "error",
                    "message": "Model initialization failed"
                }
        
        try:
            self.training_in_progress = True
            logger.info(f"Starting DeepLog + BERT training with {len(session_texts)} sessions")
            
            # Run training in a separate thread to avoid blocking
            def train_in_thread():
                return self.trainer.train_model(session_texts, labels)
            
            # Use asyncio to run in thread pool
            loop = asyncio.get_event_loop()
            training_results = await loop.run_in_executor(None, train_in_thread)
            
            if training_results.get("status") == "success":
                self.is_model_loaded = True
                logger.info("DeepLog + BERT training completed successfully")
            
            return training_results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
        finally:
            self.training_in_progress = False
    
    async def detect_anomaly_async(self, session_text: str) -> Dict[str, Any]:
        """
        Detect anomalies in a session using DeepLog + BERT
        """
        if not self.is_model_loaded:
            # Try to initialize if not already done
            if not await self.initialize_model():
                return {
                    "status": "error",
                    "message": "DeepLog + BERT model not available"
                }
        
        if not self.trainer:
            return {
                "status": "error",
                "message": "Model trainer not initialized"
            }
        
        try:
            # Run prediction in thread pool to avoid blocking
            def predict_in_thread():
                return self.trainer.predict_anomaly(session_text)
            
            loop = asyncio.get_event_loop()
            prediction = await loop.run_in_executor(None, predict_in_thread)
            
            # Add integration metadata
            prediction["model_type"] = "deeplog_bert"
            prediction["timestamp"] = datetime.now().isoformat()
            prediction["confidence_threshold"] = self.confidence_threshold
            
            # Adjust confidence based on threshold
            if prediction.get("confidence", 0) >= self.confidence_threshold:
                prediction["high_confidence"] = True
            else:
                prediction["high_confidence"] = False
            
            return prediction
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "model_type": "deeplog_bert"
            }
    
    def get_model_status(self) -> Dict[str, Any]:
        """
        Get current status of the DeepLog + BERT model
        """
        return {
            "model_loaded": self.is_model_loaded,
            "training_in_progress": self.training_in_progress,
            "model_path": self.model_path,
            "model_exists": os.path.exists(self.model_path),
            "confidence_threshold": self.confidence_threshold,
            "realtime_detection_enabled": self.enable_realtime_detection,
            "batch_training_enabled": self.batch_training_enabled
        }
    
    async def batch_train_from_database(self, database_connection) -> Dict[str, Any]:
        """
        Train model using sessions from the database
        """
        try:
            # Query recent sessions for training
            query = """
            SELECT session_text, is_anomaly 
            FROM ml_sessions 
            WHERE session_text IS NOT NULL 
            AND created_at > NOW() - INTERVAL '30 days'
            ORDER BY created_at DESC
            LIMIT 1000
            """
            
            cursor = database_connection.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            
            if not results:
                return {
                    "status": "error",
                    "message": "No training data found in database"
                }
            
            # Extract texts and labels
            session_texts = [row[0] for row in results]
            labels = [row[1] for row in results if row[1] is not None]
            
            logger.info(f"Retrieved {len(session_texts)} sessions from database for training")
            
            # Train model
            training_result = await self.train_model_async(session_texts, labels if labels else None)
            
            return training_result
            
        except Exception as e:
            logger.error(f"Batch training from database failed: {e}")
            return {
                "status": "error",
                "message": str(e)
            }


# Integration functions for existing anomaly detector service
deeplog_bert_integration = DeepLogBertIntegration()

async def initialize_deeplog_bert() -> bool:
    """
    Initialize DeepLog + BERT integration
    """
    return await deeplog_bert_integration.initialize_model()

async def predict_with_deeplog_bert(session_text: str) -> Dict[str, Any]:
    """
    Predict anomaly using DeepLog + BERT
    """
    return await deeplog_bert_integration.detect_anomaly_async(session_text)

async def train_deeplog_bert_model(session_texts: List[str], labels: List[int] = None) -> Dict[str, Any]:
    """
    Train DeepLog + BERT model
    """
    return await deeplog_bert_integration.train_model_async(session_texts, labels)

def get_deeplog_bert_status() -> Dict[str, Any]:
    """
    Get DeepLog + BERT model status
    """
    return deeplog_bert_integration.get_model_status()


# Enhanced anomaly detection function that combines multiple models
async def enhanced_anomaly_detection(session_text: str, session_id: str) -> Dict[str, Any]:
    """
    Enhanced anomaly detection combining multiple approaches
    """
    results = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "detections": {},
        "final_decision": {
            "is_anomaly": False,
            "confidence": 0.0,
            "reasoning": []
        }
    }
    
    # DeepLog + BERT Detection
    try:
        deeplog_result = await predict_with_deeplog_bert(session_text)
        results["detections"]["deeplog_bert"] = deeplog_result
        
        if deeplog_result.get("is_anomaly"):
            results["final_decision"]["reasoning"].append("DeepLog + BERT detected sequential anomaly")
            results["final_decision"]["confidence"] = max(
                results["final_decision"]["confidence"],
                deeplog_result.get("confidence", 0)
            )
    except Exception as e:
        logger.error(f"DeepLog + BERT detection failed: {e}")
        results["detections"]["deeplog_bert"] = {"status": "error", "message": str(e)}
    
    # Rule-based detection (placeholder for existing logic)
    try:
        rule_based_result = await rule_based_detection(session_text)
        results["detections"]["rule_based"] = rule_based_result
        
        if rule_based_result.get("is_anomaly"):
            results["final_decision"]["reasoning"].append("Rule-based detection triggered")
            results["final_decision"]["confidence"] = max(
                results["final_decision"]["confidence"],
                rule_based_result.get("confidence", 0)
            )
    except Exception as e:
        logger.error(f"Rule-based detection failed: {e}")
        results["detections"]["rule_based"] = {"status": "error", "message": str(e)}
    
    # Statistical detection (placeholder for existing logic)
    try:
        statistical_result = await statistical_detection(session_text)
        results["detections"]["statistical"] = statistical_result
        
        if statistical_result.get("is_anomaly"):
            results["final_decision"]["reasoning"].append("Statistical anomaly detected")
            results["final_decision"]["confidence"] = max(
                results["final_decision"]["confidence"],
                statistical_result.get("confidence", 0)
            )
    except Exception as e:
        logger.error(f"Statistical detection failed: {e}")
        results["detections"]["statistical"] = {"status": "error", "message": str(e)}
    
    # Final decision logic
    detection_count = sum(1 for det in results["detections"].values() 
                         if det.get("is_anomaly") == True)
    
    if detection_count >= 2:  # Consensus of at least 2 methods
        results["final_decision"]["is_anomaly"] = True
        results["final_decision"]["reasoning"].append("Multiple detection methods agree")
    elif detection_count == 1 and results["final_decision"]["confidence"] > 0.8:
        results["final_decision"]["is_anomaly"] = True
        results["final_decision"]["reasoning"].append("Single high-confidence detection")
    
    return results


# Placeholder functions for existing detection methods
async def rule_based_detection(session_text: str) -> Dict[str, Any]:
    """Placeholder for existing rule-based detection"""
    return {"is_anomaly": False, "confidence": 0.3, "method": "rule_based"}

async def statistical_detection(session_text: str) -> Dict[str, Any]:
    """Placeholder for existing statistical detection"""
    return {"is_anomaly": False, "confidence": 0.2, "method": "statistical"}


# Export main functions
__all__ = [
    'initialize_deeplog_bert',
    'predict_with_deeplog_bert',
    'train_deeplog_bert_model',
    'get_deeplog_bert_status',
    'enhanced_anomaly_detection',
    'DeepLogBertIntegration'
]
