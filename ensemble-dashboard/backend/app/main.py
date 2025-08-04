from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import pandas as pd
import numpy as np
import io
import os
import sys
import glob
import base64
from datetime import datetime

# Add the backend directory to the path to import ensemble_detector
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from enhanced_ensemble_detector import EnhancedEnsembleAnomalyDetector as EnsembleAnomalyDetector
    print("Using Enhanced Ensemble Detector with DBSCAN")
except ImportError:
    from ensemble_detector import EnsembleAnomalyDetector

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    import numpy as np
    
    # Handle None
    if obj is None:
        return None
    
    # Handle numpy scalars and arrays first
    if hasattr(obj, 'dtype'):  # All numpy types have dtype
        if hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        else:
            return str(obj)
    
    # Handle basic numpy types explicitly
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.str_):
        return str(obj)
    
    # Handle collections
    elif isinstance(obj, dict):
        return {str(key): convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, set):
        return [convert_numpy_types(item) for item in obj]
    
    # Handle objects with __dict__ (but avoid complex objects that might cause issues)
    elif hasattr(obj, '__dict__') and not hasattr(obj, '__call__'):
        try:
            return {key: convert_numpy_types(value) for key, value in obj.__dict__.items()}
        except (AttributeError, TypeError):
            return str(obj)
    
    # Handle basic Python types
    elif isinstance(obj, (int, float, str, bool)):
        return obj
    
    # Default fallback
    else:
        try:
            return str(obj)
        except:
            return f"<unserializable: {type(obj).__name__}>"
    print("Using Standard Ensemble Detector")

app = FastAPI(title="Ensemble Anomaly Detection Dashboard API")

# Configure CORS - More permissive for debugging
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for debugging
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global ensemble model
ensemble_model = EnsembleAnomalyDetector()

class TrainingRequest(BaseModel):
    sessions: List[str]
    text_weight: Optional[float] = 0.6
    statistical_weight: Optional[float] = 0.4
    threshold: Optional[float] = 0.5

class PredictionRequest(BaseModel):
    text: str

@app.on_event("startup")
async def startup_event():
    """Initialize the ensemble model on startup"""
    print("Ensemble Dashboard API starting up...")
    
    # Try to load existing model
    model_path = "./models/ensemble_model.pkl"
    if os.path.exists(model_path):
        try:
            ensemble_model.load_model(model_path)
            print("Existing ensemble model loaded successfully")
        except Exception as e:
            print(f"Failed to load existing model: {e}")
    else:
        print("No existing model found - ready for training")

@app.get("/")
async def root():
    return {"message": "Ensemble Anomaly Detection Dashboard API", "status": "running"}

@app.get("/api/health")
async def health_check():
    """Enhanced health check endpoint with detailed model status"""
    try:
        model_status = {
            "model_exists": ensemble_model is not None,
            "is_trained": getattr(ensemble_model, 'is_trained', False),
            "has_training_stats": hasattr(ensemble_model, 'training_stats'),
            "training_stats_type": type(getattr(ensemble_model, 'training_stats', None)).__name__,
            "has_cluster_profiles": hasattr(ensemble_model, 'cluster_profiles'),
            "has_get_cluster_insights": hasattr(ensemble_model, 'get_cluster_insights'),
            "model_type": type(ensemble_model).__name__
        }
        
        return {
            "status": "healthy",
            "model_loaded": ensemble_model.is_trained if ensemble_model else False,
            "timestamp": datetime.now().isoformat(),
            "detailed_status": model_status
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "model_loaded": False,
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/model_info")
async def get_model_info():
    """Get ensemble model information"""
    try:
        model_info = ensemble_model.get_model_info()
        return convert_numpy_types({
            "success": True,
            "model_info": model_info
        })
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "model_info": {"is_trained": False}
        }

@app.post("/api/load_ej_sessions")
async def load_ej_sessions(
    file: UploadFile = File(None),
    text: str = Form(None),
    include_errors: bool = Form(False),
    limit: Optional[int] = Form(None)
):
    """
    Load EJ sessions from multiple sources:
    1. Processed data (same location as DeepLog) - primary source
    2. Uploaded file (CSV or raw text)
    3. Text input
    """
    try:
        sessions = []
        data_source = "uploaded"
        
        # First try to load from processed data (same as DeepLog)
        if not file and not text:
            try:
                import glob
                import base64
                
                # Use the same data directory as DeepLog
                # Try multiple possible locations for the data
                possible_data_dirs = [
                    "/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/data/processed",
                    "/data/processed",  # Docker volume mount
                    "../../abm-anomaly-ml-first/data/processed",  # Relative from backend/app
                    "../../../abm-anomaly-ml-first/data/processed",  # Relative from ensemble-dashboard 
                    "./data/processed"  # Local data directory
                ]
                
                data_dir = None
                for possible_dir in possible_data_dirs:
                    print(f"Checking directory: {possible_dir}")
                    if os.path.exists(possible_dir):
                        print(f"✓ Found directory: {possible_dir}")
                        data_dir = possible_dir
                        break
                    else:
                        print(f"✗ Directory not found: {possible_dir}")
                
                if not data_dir:
                    print(f"❌ No data directory found in: {possible_data_dirs}")
                    raise Exception(f"No data directory found in: {possible_data_dirs}")
                
                print(f"Using data directory: {data_dir}")
                
                # Look for normal sessions (same pattern as DeepLog)
                normal_pattern = os.path.join(data_dir, "normal_sessions_full_*.json")
                normal_files = glob.glob(normal_pattern)
                
                print(f"Looking for pattern: {normal_pattern}")
                print(f"Found {len(normal_files)} normal session files: {normal_files}")
                
                if normal_files:
                    # Get the latest file (same logic as DeepLog)
                    latest_normal_file = max(normal_files, key=os.path.getctime)
                    
                    # Load normal sessions
                    with open(latest_normal_file, 'r', encoding='utf-8') as f:
                        normal_sessions = json.load(f)
                    
                    # Convert to format expected by ensemble detector
                    for session in normal_sessions:
                        # Decode base64 raw text
                        raw_text_b64 = session.get('raw_text_base64', '')
                        if raw_text_b64:
                            session_text = base64.b64decode(raw_text_b64).decode('utf-8')
                            sessions.append(session_text)
                    
                    # Optionally include error sessions
                    if include_errors:
                        error_pattern = os.path.join(data_dir, "error_sessions_full_*.json")
                        error_files = glob.glob(error_pattern)
                        
                        if error_files:
                            latest_error_file = max(error_files, key=os.path.getctime)
                            
                            with open(latest_error_file, 'r', encoding='utf-8') as f:
                                error_sessions = json.load(f)
                            
                            for session in error_sessions:
                                raw_text_b64 = session.get('raw_text_base64', '')
                                if raw_text_b64:
                                    session_text = base64.b64decode(raw_text_b64).decode('utf-8')
                                    sessions.append(session_text)
                    
                    data_source = f"processed_data (normal: {len(normal_sessions)}" + (f", errors: {len(error_sessions)}" if include_errors and 'error_sessions' in locals() else "") + ")"
                    
                    # Apply limit if specified
                    if limit and limit > 0:
                        sessions = sessions[:limit]
                        
            except Exception as e:
                print(f"Could not load from processed data: {e}")
                # Fall back to requiring file/text upload
                
        # Handle file upload
        if file and not sessions:
            content = await file.read()
            
            # Try to parse as CSV
            try:
                df = pd.read_csv(io.BytesIO(content))
                if 'text' in df.columns:
                    sessions = df['text'].tolist()
                elif 'session_text' in df.columns:
                    sessions = df['session_text'].tolist()
                else:
                    raise HTTPException(status_code=400, detail="CSV must contain 'text' or 'session_text' column")
            except:
                # If CSV parsing fails, try as plain text
                content_str = content.decode('utf-8')
                # Split into sessions using ensemble model's sessionizer
                sessions = ensemble_model.sessionize_ej_log(content_str)
                
            data_source = f"uploaded_file ({file.filename})"
        
        # Handle text input
        elif text and not sessions:
            # Sessionize provided text
            sessions = ensemble_model.sessionize_ej_log(text)
            data_source = "text_input"
        
        # If no sessions loaded, provide helpful message
        if not sessions:
            return {
                "success": False,
                "message": f"No EJ sessions found. Data directories checked: {', '.join(possible_data_dirs) if 'possible_data_dirs' in locals() else 'None'}. Either upload a file/text or ensure processed data exists with pattern: normal_sessions_full_*.json",
                "data_source": "none",
                "count": 0,
                "suggestions": [
                    "1. Upload CSV file with 'text' column containing EJ sessions",
                    "2. Paste raw EJ log text directly", 
                    "3. Ensure processed data exists in abm-anomaly-ml-first/data/processed/",
                    "4. Run EJ processor first to generate session data"
                ]
            }
        
        # Filter out empty sessions
        valid_sessions = [s for s in sessions if s and str(s).strip()]
        
        return {
            "success": True,
            "message": f"Loaded {len(valid_sessions)} sessions from {data_source}",
            "sessions": valid_sessions,
            "data_source": data_source,
            "count": len(valid_sessions),
            "includes_errors": include_errors if data_source.startswith("processed_data") else False
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load EJ sessions: {str(e)}")

@app.post("/api/train")
async def train_ensemble(request: TrainingRequest):
    """
    Train the ensemble model on normal sessions
    """
    try:
        if not request.sessions:
            raise HTTPException(status_code=400, detail="No sessions provided for training")
        
        # Update ensemble weights if provided
        ensemble_model.text_weight = request.text_weight
        ensemble_model.statistical_weight = request.statistical_weight
        ensemble_model.threshold = request.threshold
        
        # Filter normal sessions (assume all provided sessions are normal for unsupervised training)
        normal_sessions = [s for s in request.sessions if s and str(s).strip()]
        
        if len(normal_sessions) < 3:
            raise HTTPException(status_code=400, detail="Need at least 3 sessions for training")
        
        # Train the model
        training_stats = ensemble_model.train(normal_sessions)
        
        # Save the trained model
        ensemble_model.save_model()
        
        return {
            "success": True,
            "message": "Ensemble model trained successfully",
            "training_stats": convert_numpy_types(training_stats)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/api/predict")
async def predict_session(text: str = Form(...)):
    """
    Predict anomaly for a single EJ session
    """
    try:
        if not ensemble_model.is_trained:
            raise HTTPException(status_code=400, detail="Model not trained. Please train the model first.")
        
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Text input is required")
        
        # Make prediction
        result = ensemble_model.predict(text.strip())
        
        return {
            "success": True,
            "prediction": convert_numpy_types(result)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/batch_predict")
async def batch_predict(file: UploadFile = File(...)):
    """
    Predict anomalies for batch of EJ sessions
    """
    try:
        if not ensemble_model.is_trained:
            raise HTTPException(status_code=400, detail="Model not trained. Please train the model first.")
        
        content = await file.read()
        
        # Parse sessions
        try:
            df = pd.read_csv(io.BytesIO(content))
            if 'text' not in df.columns:
                raise HTTPException(status_code=400, detail="CSV must contain 'text' column")
            sessions = df['text'].tolist()
            true_labels = df['label'].tolist() if 'label' in df.columns else None
        except:
            # Try as raw text
            content_str = content.decode('utf-8')
            sessions = ensemble_model.sessionize_ej_log(content_str)
            true_labels = None
        
        # Filter valid sessions
        valid_sessions = [s for s in sessions if s and str(s).strip()]
        
        if not valid_sessions:
            raise HTTPException(status_code=400, detail="No valid sessions found")
        
        # Make batch predictions
        predictions = ensemble_model.batch_predict(valid_sessions)
        
        # Calculate metrics if true labels provided
        metrics = None
        if true_labels:
            # Simple accuracy calculation
            correct = 0
            total = min(len(predictions), len(true_labels))
            for i in range(total):
                if (predictions[i]['is_anomaly'] and true_labels[i] == 1) or \
                   (not predictions[i]['is_anomaly'] and true_labels[i] == 0):
                    correct += 1
            
            metrics = {
                "accuracy": correct / total if total > 0 else 0,
                "total_samples": total,
                "correct_predictions": correct,
                "anomalies_detected": sum(1 for p in predictions if p['is_anomaly']),
                "normal_detected": sum(1 for p in predictions if not p['is_anomaly'])
            }
        
        return {
            "success": True,
            "predictions": convert_numpy_types(predictions),
            "metrics": convert_numpy_types(metrics) if metrics else None,
            "total_sessions": len(predictions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/api/sessionize")
async def sessionize_ej_log(text: str = Form(...)):
    """
    Sessionize raw EJ log text into individual sessions
    """
    try:
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Text input is required")
        
        sessions = ensemble_model.sessionize_ej_log(text.strip())
        
        return {
            "success": True,
            "sessions": sessions,
            "count": len(sessions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sessionization failed: {str(e)}")

@app.get("/api/training_stats")
async def get_training_stats():
    """Get training statistics and model performance"""
    try:
        print("Training stats endpoint called")
        
        if not ensemble_model.is_trained:
            print("Model not trained")
            return {
                "success": False,
                "message": "Model not trained"
            }
        
        print("Getting training stats from model...")
        raw_stats = ensemble_model.training_stats
        print(f"Raw stats type: {type(raw_stats)}")
        
        if raw_stats is None:
            print("Training stats is None")
            return {
                "success": False,
                "message": "Training stats not available"
            }
        
        print("Converting numpy types...")
        converted_stats = convert_numpy_types(raw_stats)
        print("Conversion successful")
        
        return {
            "success": True,
            "stats": converted_stats
        }
        
    except Exception as e:
        print(f"Error in training_stats endpoint: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to get training stats: {str(e)}")

@app.post("/api/update_config")
async def update_config(
    text_weight: float = Form(0.4),
    statistical_weight: float = Form(0.3),
    density_weight: float = Form(0.3),
    threshold: float = Form(0.5)
):
    """Update ensemble configuration with DBSCAN density weight"""
    try:
        # Validate weights sum to 1.0
        total_weight = text_weight + statistical_weight + density_weight
        if abs(total_weight - 1.0) > 0.01:
            raise HTTPException(status_code=400, detail=f"Weights must sum to 1.0, got {total_weight}")
        
        ensemble_model.text_weight = text_weight
        ensemble_model.statistical_weight = statistical_weight
        if hasattr(ensemble_model, 'density_weight'):
            ensemble_model.density_weight = density_weight
        ensemble_model.threshold = threshold
        
        return {
            "success": True,
            "config": {
                "text_weight": text_weight,
                "statistical_weight": statistical_weight,
                "density_weight": density_weight if hasattr(ensemble_model, 'density_weight') else 0.0,
                "threshold": threshold
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Configuration update failed: {str(e)}")

@app.get("/api/cluster_insights")
async def get_cluster_insights():
    """Get DBSCAN cluster insights and analysis"""
    try:
        print("cluster_insights endpoint called")
        
        if not ensemble_model.is_trained:
            print("Model not trained")
            raise HTTPException(status_code=400, detail="Model not trained")
        
        if hasattr(ensemble_model, 'get_cluster_insights'):
            print("Getting cluster insights from model...")
            insights = ensemble_model.get_cluster_insights()
            print(f"Insights type: {type(insights)}")
            
            converted_insights = convert_numpy_types(insights)
            print("Insights converted successfully")
            
            return {
                "success": True,
                "insights": converted_insights
            }
        else:
            print("Enhanced DBSCAN features not available")
            return {
                "success": False,
                "message": "Enhanced DBSCAN features not available in current model"
            }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in cluster_insights: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to get cluster insights: {str(e)}")

@app.post("/api/cluster_visualization_data")
async def get_cluster_visualization_data():
    """Get data for DBSCAN cluster visualization"""
    try:
        print("cluster_visualization_data endpoint called")
        
        if not ensemble_model.is_trained:
            print("Model not trained")
            raise HTTPException(status_code=400, detail="Model not trained")
        
        if not hasattr(ensemble_model, 'cluster_profiles'):
            print("No cluster_profiles attribute")
            return {
                "success": False,
                "message": "DBSCAN clustering not available in current model"
            }
        
        print("Generating visualization data...")
        # Generate visualization data from cluster profiles
        viz_data = {
            "text_clusters": [],
            "numerical_clusters": [],
            "combined_clusters": [],
            "cluster_statistics": {}
        }
        
        for cluster_type in ['text', 'numerical', 'combined']:
            cluster_key = f'{cluster_type}_clusters'
            if cluster_key in ensemble_model.cluster_profiles:
                print(f"Processing {cluster_type} clusters...")
                analysis = ensemble_model.cluster_profiles[cluster_key]
                
                # Extract cluster centers and sizes for visualization
                clusters = []
                for cluster_name, profile in analysis.get('cluster_profiles', {}).items():
                    if cluster_name == 'noise':
                        continue
                    
                    # For visualization, we'll use first 2 dimensions of cluster center
                    center = profile.get('center', [])
                    if len(center) >= 2:
                        clusters.append({
                            'name': cluster_name,
                            'x': center[0],
                            'y': center[1],
                            'size': profile.get('size', 0),
                            'std_x': profile.get('std', [0, 0])[0] if len(profile.get('std', [])) > 0 else 0,
                            'std_y': profile.get('std', [0, 0])[1] if len(profile.get('std', [])) > 1 else 0
                        })
                
                viz_data[f'{cluster_type}_clusters'] = clusters
                
                # Add statistics
                viz_data['cluster_statistics'][cluster_type] = {
                    'n_clusters': analysis.get('n_clusters', 0),
                    'noise_ratio': analysis.get('noise_ratio', 0),
                    'total_points': sum(analysis.get('cluster_sizes', {}).values())
                }
        
        print("Converting visualization data...")
        converted_data = convert_numpy_types(viz_data)
        print("Conversion successful")
        
        return {
            "success": True,
            "visualization_data": converted_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in cluster_visualization_data: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to get visualization data: {str(e)}")

@app.post("/api/performance_comparison")
async def get_performance_comparison():
    """Get performance comparison between traditional and DBSCAN-enhanced ensemble"""
    try:
        print("performance_comparison endpoint called")
        
        if not ensemble_model.is_trained:
            print("Model not trained")
            raise HTTPException(status_code=400, detail="Model not trained")
        
        print("Getting training statistics...")
        # Get training statistics
        stats = ensemble_model.training_stats
        print(f"Stats type: {type(stats)}")
        
        if stats is None:
            print("Training stats is None")
            raise HTTPException(status_code=500, detail="Training statistics not available")
        
        print("Preparing comparison data...")
        # Prepare comparison data
        comparison_data = {
            "model_type": "Enhanced with DBSCAN" if hasattr(ensemble_model, 'density_weight') else "Traditional",
            "training_sessions": stats.get('num_training_sessions', 0),
            "feature_dimensions": {
                "text": stats.get('text_feature_dims', 0),
                "numerical": stats.get('numerical_feature_dims', 0)
            },
            "average_scores": {
                "text": stats.get('avg_svm_score', 0),
                "statistical": stats.get('avg_isolation_score', 0),
                "density": stats.get('avg_density_score', 0) if hasattr(ensemble_model, 'density_weight') else None,
                "ensemble": stats.get('avg_ensemble_score', 0)
            },
            "weights": stats.get('weights', {
                "text_weight": ensemble_model.text_weight,
                "statistical_weight": ensemble_model.statistical_weight,
                "density_weight": getattr(ensemble_model, 'density_weight', 0)
            }),
            "dbscan_parameters": stats.get('dbscan_params', {}) if hasattr(ensemble_model, 'density_weight') else None
        }
        
        print("Converting numpy types...")
        converted_data = convert_numpy_types(comparison_data)
        print("Conversion successful")
        
        return {
            "success": True,
            "comparison_data": converted_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in performance_comparison: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to get performance comparison: {str(e)}")

# Cluster interaction endpoints for expert labeling
class ClusterSessionsRequest(BaseModel):
    cluster_id: int
    feature_type: Optional[str] = 'combined'

class LabelClusterRequest(BaseModel):
    cluster_id: int
    label: str
    feature_type: Optional[str] = 'combined'
    label_name: Optional[str] = None
    label_description: Optional[str] = None
    confidence: Optional[float] = 0.8

class SupervisedTrainingRequest(BaseModel):
    force_retrain: Optional[bool] = False

class SupervisedPredictionRequest(BaseModel):
    session_text: str

@app.post("/api/cluster_sessions")
async def get_cluster_sessions(request: ClusterSessionsRequest):
    """Get EJ sessions belonging to a specific cluster"""
    try:
        print(f"cluster_sessions called with cluster_id={request.cluster_id}, feature_type={request.feature_type}")
        
        if not ensemble_model.is_trained:
            raise HTTPException(status_code=400, detail="Model not trained")
        
        if not hasattr(ensemble_model, 'get_cluster_sessions'):
            raise HTTPException(status_code=400, detail="Enhanced DBSCAN features not available")
        
        # Get sessions from the model
        try:
            print("Calling ensemble_model.get_cluster_sessions...")
            raw_sessions = ensemble_model.get_cluster_sessions(request.cluster_id, request.feature_type)
            print(f"Raw sessions received: {type(raw_sessions)}, length: {len(raw_sessions) if raw_sessions else 0}")
        except Exception as e:
            print(f"Error from get_cluster_sessions: {type(e).__name__}: {str(e)}")
            # Handle cluster not found errors as 400 status
            error_message = str(e)
            if "not found" in error_message.lower() and "cluster" in error_message.lower():
                raise HTTPException(status_code=400, detail=error_message)
            else:
                raise
        
        # Convert to properly structured JSON objects
        print("Converting sessions to structured format...")
        sessions = []
        if raw_sessions:
            for i, session in enumerate(raw_sessions):
                try:
                    # Check if session is a dictionary-like object
                    if hasattr(session, 'keys') or isinstance(session, dict):
                        # Convert to a proper dictionary
                        session_dict = {}
                        if hasattr(session, 'items'):
                            for key, value in session.items():
                                session_dict[str(key)] = convert_numpy_types(value)
                        else:
                            # If it's not iterable, convert to string
                            session_dict = {
                                "session_text": str(session),
                                "session_id": f"session_{request.cluster_id}_{i}",
                                "cluster_id": request.cluster_id,
                                "feature_type": request.feature_type
                            }
                        sessions.append(session_dict)
                    else:
                        # For simple string sessions
                        sessions.append({
                            "session_text": str(session),
                            "session_id": f"session_{request.cluster_id}_{i}",
                            "cluster_id": request.cluster_id,
                            "feature_type": request.feature_type
                        })
                    if i == 0:  # Debug first session
                        print(f"First session converted: {type(session)} -> structured dict")
                except Exception as e:
                    print(f"Warning: Could not convert session {i}: {e}")
                    sessions.append({
                        "session_text": f"<session conversion error: {str(e)}>",
                        "session_id": f"session_{request.cluster_id}_{i}",
                        "cluster_id": request.cluster_id,
                        "feature_type": request.feature_type,
                        "error": True
                    })
        
        print(f"Converted {len(sessions)} sessions to structured objects")
        
        # Build response with only basic Python types
        response_data = {
            "success": True,
            "cluster_id": int(request.cluster_id),
            "feature_type": str(request.feature_type),
            "sessions": sessions,
            "count": len(sessions)
        }
        
        print("Response data built, attempting JSON round-trip...")
        
        # Use JSON round-trip to ensure everything is serializable
        import json
        try:
            json_str = json.dumps(response_data)
            final_response = json.loads(json_str)
            print("JSON round-trip successful!")
            return final_response
        except Exception as json_error:
            print(f"JSON serialization error: {json_error}")
            # Return a safe fallback response
            fallback_response = {
                "success": True,
                "cluster_id": int(request.cluster_id),
                "feature_type": str(request.feature_type),
                "sessions": [f"<{len(sessions)} sessions - serialization error>"],
                "count": len(sessions),
                "error": f"Serialization issue: {str(json_error)}"
            }
            print("Returning fallback response")
            return fallback_response
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        print("Re-raising HTTPException")
        raise
    except Exception as e:
        # Log the full error for debugging
        print(f"Exception in cluster_sessions: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to get cluster sessions: {str(e)}")

@app.post("/api/label_cluster")
async def label_cluster(request: LabelClusterRequest):
    """Label a cluster with expert knowledge"""
    try:
        if not ensemble_model.is_trained:
            raise HTTPException(status_code=400, detail="Model not trained")
        
        if not hasattr(ensemble_model, 'label_cluster'):
            raise HTTPException(status_code=400, detail="Enhanced DBSCAN features not available")
        
        # Use label_name if provided, otherwise fall back to label
        label_to_use = request.label_name if request.label_name else request.label
        feature_type_to_use = request.feature_type if request.feature_type else 'combined'
        
        result = ensemble_model.label_cluster(request.cluster_id, label_to_use, feature_type_to_use)
        
        return {
            "success": True,
            "message": f"Cluster {request.cluster_id} labeled as '{label_to_use}'",
            "cluster_id": request.cluster_id,
            "label": label_to_use,
            "feature_type": feature_type_to_use
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to label cluster: {str(e)}")

@app.get("/api/cluster_labels")
async def get_cluster_labels():
    """Get all cluster labels"""
    try:
        if not ensemble_model.is_trained:
            raise HTTPException(status_code=400, detail="Model not trained")
        
        if not hasattr(ensemble_model, 'cluster_labels'):
            return {"success": True, "labels": {}}
        
        labels = getattr(ensemble_model, 'cluster_labels', {})
        
        return convert_numpy_types({
            "success": True,
            "labels": labels,
            "count": len(labels)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cluster labels: {str(e)}")

@app.post("/api/train_supervised_classifier")
async def train_supervised_classifier(request: SupervisedTrainingRequest):
    """Train supervised classifier using labeled clusters"""
    try:
        if not ensemble_model.is_trained:
            raise HTTPException(status_code=400, detail="Model not trained")
        
        if not hasattr(ensemble_model, 'train_supervised_classifier'):
            raise HTTPException(status_code=400, detail="Enhanced DBSCAN features not available")
        
        result = ensemble_model.train_supervised_classifier(force_retrain=request.force_retrain)
        
        return {
            "success": True,
            "message": "Supervised classifier trained successfully",
            "training_stats": convert_numpy_types(result)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to train supervised classifier: {str(e)}")

@app.post("/api/predict_with_supervised")
async def predict_with_supervised(request: SupervisedPredictionRequest):
    """Predict cluster label for new session using supervised classifier"""
    try:
        if not ensemble_model.is_trained:
            raise HTTPException(status_code=400, detail="Model not trained")
        
        if not hasattr(ensemble_model, 'predict_supervised'):
            raise HTTPException(status_code=400, detail="Enhanced DBSCAN features not available")
        
        prediction = ensemble_model.predict_supervised(request.session_text)
        
        return {
            "success": True,
            "session_text": request.session_text,
            "prediction": convert_numpy_types(prediction)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to predict with supervised classifier: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
