"""
API endpoints for BERT-Enhanced DeepLog Model
Provides training, prediction, and monitoring capabilities
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import logging
import asyncio
from datetime import datetime
import os
import sys

# Add the anomaly detector path
sys.path.append('/app/services/anomaly-detector')

try:
    from bert_deeplog_model import BertDeepLogAnalyzer
    DEEPLOG_AVAILABLE = True
except ImportError as e:
    logging.error(f"Could not import BERT DeepLog components: {e}")
    DEEPLOG_AVAILABLE = False
    BertDeepLogAnalyzer = None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/bert-deeplog", tags=["BERT DeepLog"])

# Global analyzer instance
deeplog_analyzer = None

# Pydantic models
class TrainingRequest(BaseModel):
    sessions: List[Dict[str, Any]]
    validation_split: float = 0.2
    normal_sessions_only: bool = True

class PredictionRequest(BaseModel):
    session_text: str
    session_id: Optional[str] = None

class BatchPredictionRequest(BaseModel):
    sessions: List[Dict[str, str]]  # List of {session_id, session_text}

class ModelConfigRequest(BaseModel):
    window_size: Optional[int] = None
    anomaly_threshold: Optional[float] = None
    learning_rate: Optional[float] = None
    num_epochs: Optional[int] = None

class TrainingResponse(BaseModel):
    success: bool
    message: str
    training_stats: Optional[Dict[str, Any]] = None
    training_history: Optional[List[Dict[str, Any]]] = None

class PredictionResponse(BaseModel):
    session_id: Optional[str]
    is_anomaly: bool
    anomaly_probability: float
    confidence: float
    important_events: List[Dict[str, Any]]
    explanation: Optional[Dict[str, Any]] = None
    processing_time_ms: float

@router.on_event("startup")
async def initialize_deeplog():
    """Initialize the DeepLog analyzer on startup"""
    global deeplog_analyzer
    
    if not DEEPLOG_AVAILABLE:
        logger.error("BERT DeepLog not available - endpoints will return errors")
        return
    
    try:
        deeplog_analyzer = BertDeepLogAnalyzer()
        logger.info("BERT DeepLog analyzer initialized successfully")
        
        # Try to load existing model
        try:
            deeplog_analyzer.load_model()
            logger.info("Loaded existing BERT DeepLog model")
        except:
            logger.info("No existing model found - will need training")
            
    except Exception as e:
        logger.error(f"Failed to initialize BERT DeepLog analyzer: {e}")

@router.post("/train", response_model=TrainingResponse)
async def train_deeplog_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train the BERT-DeepLog model on provided sessions"""
    
    if not DEEPLOG_AVAILABLE or deeplog_analyzer is None:
        raise HTTPException(status_code=500, detail="BERT DeepLog not available")
    
    if len(request.sessions) < 10:
        raise HTTPException(status_code=400, detail="Need at least 10 sessions for training")
    
    try:
        start_time = datetime.now()
        
        # Prepare training data
        num_sequences = deeplog_analyzer.prepare_training_data(
            request.sessions, 
            normal_sessions_only=request.normal_sessions_only
        )
        
        if num_sequences < 5:
            raise HTTPException(
                status_code=400, 
                detail=f"Insufficient training sequences ({num_sequences}). Need at least 5."
            )
        
        # Train model (run in background for long training)
        def train_model():
            try:
                training_history = deeplog_analyzer.train_model(
                    validation_split=request.validation_split
                )
                logger.info(f"BERT DeepLog training completed with {len(training_history)} epochs")
            except Exception as e:
                logger.error(f"Training failed: {e}")
        
        if len(request.sessions) > 100:
            # Run training in background for large datasets
            background_tasks.add_task(train_model)
            
            training_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return TrainingResponse(
                success=True,
                message=f"Training started in background with {num_sequences} sequences",
                training_stats={
                    'num_input_sessions': len(request.sessions),
                    'num_training_sequences': num_sequences,
                    'training_time_ms': training_time,
                    'validation_split': request.validation_split,
                    'training_status': 'background'
                }
            )
        else:
            # Train synchronously for smaller datasets
            training_history = deeplog_analyzer.train_model(
                validation_split=request.validation_split
            )
            
            training_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return TrainingResponse(
                success=True,
                message=f"Training completed successfully with {num_sequences} sequences",
                training_stats={
                    'num_input_sessions': len(request.sessions),
                    'num_training_sequences': num_sequences,
                    'training_time_ms': training_time,
                    'validation_split': request.validation_split,
                    'final_train_loss': training_history[-1]['train_loss'] if training_history else None,
                    'final_val_loss': training_history[-1]['val_loss'] if training_history else None,
                    'training_status': 'completed'
                },
                training_history=training_history
            )
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@router.post("/predict", response_model=PredictionResponse)
async def predict_anomaly(request: PredictionRequest):
    """Predict if a session is anomalous using BERT-DeepLog"""
    
    if not DEEPLOG_AVAILABLE or deeplog_analyzer is None:
        raise HTTPException(status_code=500, detail="BERT DeepLog not available")
    
    if not deeplog_analyzer.model_trained:
        raise HTTPException(status_code=400, detail="Model not trained. Train the model first.")
    
    start_time = datetime.now()
    
    try:
        # Make prediction
        prediction_result = deeplog_analyzer.predict_anomaly(
            request.session_text,
            request.session_id
        )
        
        if 'error' in prediction_result:
            raise HTTPException(status_code=500, detail=prediction_result['error'])
        
        # Get explanation
        explanation = None
        if request.session_id:
            try:
                explanation = deeplog_analyzer.explain_prediction(request.session_id)
            except Exception as e:
                logger.warning(f"Could not generate explanation: {e}")
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return PredictionResponse(
            session_id=prediction_result.get('session_id'),
            is_anomaly=prediction_result['is_anomaly'],
            anomaly_probability=prediction_result['anomaly_probability'],
            confidence=prediction_result['confidence'],
            important_events=prediction_result['important_events'],
            explanation=explanation,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/predict-batch")
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction for multiple sessions"""
    
    if not DEEPLOG_AVAILABLE or deeplog_analyzer is None:
        raise HTTPException(status_code=500, detail="BERT DeepLog not available")
    
    if not deeplog_analyzer.model_trained:
        raise HTTPException(status_code=400, detail="Model not trained. Train the model first.")
    
    start_time = datetime.now()
    results = []
    
    for session in request.sessions:
        try:
            prediction_result = deeplog_analyzer.predict_anomaly(
                session['session_text'],
                session.get('session_id')
            )
            results.append(prediction_result)
        except Exception as e:
            results.append({
                'session_id': session.get('session_id'),
                'error': str(e),
                'is_anomaly': False,
                'anomaly_probability': 0.0
            })
    
    processing_time = (datetime.now() - start_time).total_seconds() * 1000
    
    # Calculate batch statistics
    successful_predictions = [r for r in results if 'error' not in r]
    anomaly_count = sum(1 for r in successful_predictions if r['is_anomaly'])
    
    return {
        'total_sessions': len(request.sessions),
        'successful_predictions': len(successful_predictions),
        'failed_predictions': len(results) - len(successful_predictions),
        'anomalies_detected': anomaly_count,
        'anomaly_rate': anomaly_count / max(len(successful_predictions), 1),
        'processing_time_ms': processing_time,
        'results': results
    }

@router.get("/model-info")
async def get_model_info():
    """Get information about the current model"""
    
    if not DEEPLOG_AVAILABLE or deeplog_analyzer is None:
        raise HTTPException(status_code=500, detail="BERT DeepLog not available")
    
    model_stats = deeplog_analyzer.get_model_stats()
    
    return {
        'model_available': True,
        'model_stats': model_stats,
        'training_history_length': len(deeplog_analyzer.training_history),
        'cached_predictions': len(deeplog_analyzer.prediction_cache),
        'last_training': deeplog_analyzer.training_history[-1]['timestamp'] if deeplog_analyzer.training_history else None
    }

@router.get("/training-history")
async def get_training_history():
    """Get model training history"""
    
    if not DEEPLOG_AVAILABLE or deeplog_analyzer is None:
        raise HTTPException(status_code=500, detail="BERT DeepLog not available")
    
    return {
        'training_history': deeplog_analyzer.training_history,
        'total_epochs': len(deeplog_analyzer.training_history),
        'model_trained': deeplog_analyzer.model_trained
    }

@router.get("/prediction-cache")
async def get_prediction_cache():
    """Get cached predictions for analysis"""
    
    if not DEEPLOG_AVAILABLE or deeplog_analyzer is None:
        raise HTTPException(status_code=500, detail="BERT DeepLog not available")
    
    cache_summary = {}
    anomaly_count = 0
    
    for session_id, prediction in deeplog_analyzer.prediction_cache.items():
        cache_summary[session_id] = {
            'is_anomaly': prediction['is_anomaly'],
            'anomaly_probability': prediction['anomaly_probability'],
            'confidence': prediction['confidence'],
            'num_important_events': len(prediction['important_events']),
            'prediction_timestamp': prediction['prediction_timestamp']
        }
        
        if prediction['is_anomaly']:
            anomaly_count += 1
    
    return {
        'total_cached_predictions': len(deeplog_analyzer.prediction_cache),
        'anomalies_in_cache': anomaly_count,
        'cache_summary': cache_summary
    }

@router.post("/configure")
async def configure_model(request: ModelConfigRequest):
    """Configure model parameters"""
    
    if not DEEPLOG_AVAILABLE or deeplog_analyzer is None:
        raise HTTPException(status_code=500, detail="BERT DeepLog not available")
    
    updated_params = {}
    
    if request.window_size is not None:
        deeplog_analyzer.window_size = request.window_size
        updated_params['window_size'] = request.window_size
    
    if request.anomaly_threshold is not None:
        deeplog_analyzer.anomaly_threshold = request.anomaly_threshold
        updated_params['anomaly_threshold'] = request.anomaly_threshold
    
    if request.learning_rate is not None:
        deeplog_analyzer.learning_rate = request.learning_rate
        updated_params['learning_rate'] = request.learning_rate
    
    if request.num_epochs is not None:
        deeplog_analyzer.num_epochs = request.num_epochs
        updated_params['num_epochs'] = request.num_epochs
    
    return {
        'success': True,
        'message': 'Model configuration updated',
        'updated_parameters': updated_params,
        'current_config': {
            'window_size': deeplog_analyzer.window_size,
            'anomaly_threshold': deeplog_analyzer.anomaly_threshold,
            'learning_rate': deeplog_analyzer.learning_rate,
            'num_epochs': deeplog_analyzer.num_epochs
        }
    }

@router.get("/explanation/{session_id}")
async def get_prediction_explanation(session_id: str):
    """Get detailed explanation for a prediction"""
    
    if not DEEPLOG_AVAILABLE or deeplog_analyzer is None:
        raise HTTPException(status_code=500, detail="BERT DeepLog not available")
    
    try:
        explanation = deeplog_analyzer.explain_prediction(session_id)
        
        if 'error' in explanation:
            raise HTTPException(status_code=404, detail=explanation['error'])
        
        return explanation
        
    except Exception as e:
        logger.error(f"Error getting explanation for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get explanation: {str(e)}")

@router.delete("/clear-cache")
async def clear_prediction_cache():
    """Clear the prediction cache"""
    
    if not DEEPLOG_AVAILABLE or deeplog_analyzer is None:
        raise HTTPException(status_code=500, detail="BERT DeepLog not available")
    
    cache_size = len(deeplog_analyzer.prediction_cache)
    deeplog_analyzer.prediction_cache.clear()
    
    return {
        'success': True,
        'message': f'Cleared {cache_size} cached predictions',
        'cache_size_before': cache_size,
        'cache_size_after': 0
    }

@router.post("/save-model")
async def save_model():
    """Save the current model to disk"""
    
    if not DEEPLOG_AVAILABLE or deeplog_analyzer is None:
        raise HTTPException(status_code=500, detail="BERT DeepLog not available")
    
    if not deeplog_analyzer.model_trained:
        raise HTTPException(status_code=400, detail="No trained model to save")
    
    try:
        deeplog_analyzer.save_model()
        
        return {
            'success': True,
            'message': 'Model saved successfully',
            'model_path': os.path.join(deeplog_analyzer.model_dir, 'bert_deeplog_model.pth')
        }
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save model: {str(e)}")

@router.post("/load-model")
async def load_model():
    """Load a saved model from disk"""
    
    if not DEEPLOG_AVAILABLE or deeplog_analyzer is None:
        raise HTTPException(status_code=500, detail="BERT DeepLog not available")
    
    try:
        deeplog_analyzer.load_model()
        
        return {
            'success': True,
            'message': 'Model loaded successfully',
            'model_trained': deeplog_analyzer.model_trained,
            'training_history_length': len(deeplog_analyzer.training_history)
        }
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No saved model found")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@router.get("/load-ej-sessions")
async def load_ej_sessions(include_errors: bool = False, limit: Optional[int] = None):
    """Load processed EJ sessions from the EJ Rule-Based Processor output"""
    
    try:
        import glob
        import base64
        
        # Find the latest processed session files
        data_dir = "/app/data/processed"
        
        # Look for normal sessions
        normal_pattern = os.path.join(data_dir, "normal_sessions_full_*.json")
        normal_files = glob.glob(normal_pattern)
        
        if not normal_files:
            raise HTTPException(status_code=404, detail="No processed EJ sessions found")
        
        # Get the latest file
        latest_normal_file = max(normal_files, key=os.path.getctime)
        
        # Load normal sessions
        with open(latest_normal_file, 'r', encoding='utf-8') as f:
            normal_sessions = json.load(f)
        
        # Convert to the format expected by BERT-DeepLog
        training_sessions = []
        
        for session in normal_sessions:
            # Use BERT preprocessed text if available, otherwise decode raw text
            if session.get('bert_preprocessed_text'):
                session_text = session['bert_preprocessed_text']
            else:
                # Decode base64 raw text
                raw_text_b64 = session.get('raw_text_base64', '')
                if raw_text_b64:
                    session_text = base64.b64decode(raw_text_b64).decode('utf-8')
                else:
                    continue  # Skip sessions without text
            
            training_sessions.append({
                'session_id': session['session_id'],
                'raw_text': session_text,  # Use preprocessed or decoded text
                'text': session_text,  # Alias for compatibility
                'is_anomaly': session.get('has_errors', False),
                'source': 'ej_rule_processor',
                'file_source': session.get('file_source', 'unknown'),
                'transaction_type': session.get('transaction_type'),
                'preprocessing_info': session.get('preprocessing_info', {}),
                'error_types': session.get('error_types', [])
            })
        
        # Optionally include error sessions
        if include_errors:
            error_pattern = os.path.join(data_dir, "error_sessions_full_*.json")
            error_files = glob.glob(error_pattern)
            
            if error_files:
                latest_error_file = max(error_files, key=os.path.getctime)
                
                with open(latest_error_file, 'r', encoding='utf-8') as f:
                    error_sessions = json.load(f)
                
                for session in error_sessions:
                    if session.get('bert_preprocessed_text'):
                        session_text = session['bert_preprocessed_text']
                    else:
                        raw_text_b64 = session.get('raw_text_base64', '')
                        if raw_text_b64:
                            session_text = base64.b64decode(raw_text_b64).decode('utf-8')
                        else:
                            continue
                    
                    training_sessions.append({
                        'session_id': session['session_id'],
                        'raw_text': session_text,
                        'text': session_text,
                        'is_anomaly': True,  # Error sessions are anomalies
                        'source': 'ej_rule_processor',
                        'file_source': session.get('file_source', 'unknown'),
                        'transaction_type': session.get('transaction_type'),
                        'preprocessing_info': session.get('preprocessing_info', {}),
                        'error_types': session.get('error_types', [])
                    })
        
        # Apply limit if specified
        if limit and limit > 0:
            training_sessions = training_sessions[:limit]
        
        return {
            'success': True,
            'message': f'Loaded {len(training_sessions)} EJ sessions from processor output',
            'sessions': training_sessions,
            'data_sources': {
                'normal_file': os.path.basename(latest_normal_file),
                'error_file': os.path.basename(max(error_files, key=os.path.getctime)) if include_errors and error_files else None,
                'total_normal': len(normal_sessions),
                'total_errors': len(error_sessions) if include_errors and 'error_sessions' in locals() else 0,
                'returned_sessions': len(training_sessions)
            },
            'preprocessing_stats': {
                'sessions_with_bert_preprocessing': sum(1 for s in training_sessions if s['preprocessing_info']),
                'average_compression_ratio': sum(s['preprocessing_info'].get('compression_ratio', 0) for s in training_sessions if s['preprocessing_info']) / len(training_sessions) if training_sessions else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error loading EJ sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load EJ sessions: {str(e)}")
