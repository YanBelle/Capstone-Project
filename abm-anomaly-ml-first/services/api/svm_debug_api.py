"""
SVM Debug API Endpoints for ABM Anomaly Detection System
Provides REST API endpoints for SVM visualization and debugging
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import logging
from datetime import datetime
import os
import sys

# Add the anomaly detector path
sys.path.append('/app/services/anomaly-detector')

try:
    from svm_visualizer import OneClassSVMVisualizer
    from ml_analyzer import MLFirstAnomalyDetector, TransactionSession
except ImportError as e:
    logging.warning(f"Could not import SVM visualizer components: {e}")
    OneClassSVMVisualizer = None
    MLFirstAnomalyDetector = None
    TransactionSession = None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/svm-debug", tags=["SVM Debug"])

class SVMDebugRequest(BaseModel):
    session_id: str
    raw_text: str
    include_visualization: bool = True

class SVMDebugResponse(BaseModel):
    session_id: str
    decision_score: float
    prediction: str
    confidence: float
    support_vectors_used: int
    feature_contributions: Dict[str, float]
    visualization_url: Optional[str] = None
    processing_time_ms: float

class SVMModelInfo(BaseModel):
    parameters: Dict[str, Any]
    support_vectors_count: int
    is_fitted: bool
    feature_dimensions: Optional[int]
    last_training_time: Optional[str]

class BatchAnalysisRequest(BaseModel):
    session_ids: List[str]
    include_visualizations: bool = False

class BatchAnalysisResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_processed: int
    processing_time_ms: float
    summary_stats: Dict[str, Any]

@router.post("/analyze-session", response_model=SVMDebugResponse)
async def debug_svm_session(request: SVMDebugRequest):
    """Debug SVM decision for a specific session"""
    
    if not OneClassSVMVisualizer or not MLFirstAnomalyDetector:
        raise HTTPException(status_code=500, detail="SVM visualizer not available")
    
    start_time = datetime.now()
    
    try:
        # Initialize ML analyzer
        analyzer = MLFirstAnomalyDetector()
        
        # Process the session
        session = TransactionSession(
            session_id=request.session_id,
            raw_text=request.raw_text
        )
        
        analyzer.sessions = [session]
        
        # Extract embeddings
        analyzer.extract_embeddings()
        
        if not hasattr(session, 'embedding_vector') or session.embedding_vector is None:
            raise HTTPException(status_code=400, detail="Could not extract embeddings from session")
        
        # Get SVM prediction
        import numpy as np
        embedding_scaled = analyzer.scaler.transform(session.embedding_vector.reshape(1, -1))
        decision_score = analyzer.one_class_svm.decision_function(embedding_scaled)[0]
        prediction = analyzer.one_class_svm.predict(embedding_scaled)[0]
        
        # Calculate feature contributions (approximate)
        feature_contributions = {}
        if hasattr(analyzer, 'feature_names') and analyzer.feature_names:
            for i, name in enumerate(analyzer.feature_names[:10]):  # Top 10 features
                if i < len(embedding_scaled[0]):
                    feature_contributions[name] = float(embedding_scaled[0][i])
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Determine prediction label and confidence
        is_anomaly = prediction == -1
        confidence = abs(decision_score)
        
        # Generate visualization if requested
        visualization_url = None
        if request.include_visualization and OneClassSVMVisualizer:
            try:
                visualizer = OneClassSVMVisualizer()
                visualization_url = visualizer.create_session_visualization(
                    session.embedding_vector, decision_score, request.session_id
                )
            except Exception as e:
                logger.warning(f"Could not generate visualization: {e}")
        
        return SVMDebugResponse(
            session_id=request.session_id,
            decision_score=decision_score,
            prediction="anomaly" if is_anomaly else "normal",
            confidence=confidence,
            support_vectors_used=analyzer.one_class_svm.n_support_[0] if hasattr(analyzer.one_class_svm, 'n_support_') else 0,
            feature_contributions=feature_contributions,
            visualization_url=visualization_url,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error analyzing session {request.session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/tfidf-analysis")
async def get_tfidf_analysis(request: SVMDebugRequest):
    """Get detailed TF-IDF feature analysis for outlier sessions"""
    
    try:
        # Import One-Class SVM detector
        sys.path.append('/app/services/anomaly-detector')
        from oneclass_svm_detector import OneClassSVMAnomalyDetector
        
        # Initialize detector
        detector = OneClassSVMAnomalyDetector()
        
        # Check if model exists and load it
        model_path = os.path.join(detector.model_dir, 'oneclass_svm_model.pkl')
        if not os.path.exists(model_path):
            # Try to train with sample data or return error
            raise HTTPException(status_code=400, detail="SVM model not trained. Please train the model first.")
        
        detector.load_model()
        
        # Get complete outlier analysis with TF-IDF
        analysis_result = detector.get_outlier_analysis(
            session_text=request.raw_text,
            session_id=request.session_id
        )
        
        return {
            'session_id': request.session_id,
            'is_anomaly': analysis_result.get('is_anomaly', False),
            'decision_score': analysis_result.get('decision_score', 0.0),
            'tfidf_analysis': analysis_result.get('tfidf_analysis', []),
            'word_categories': analysis_result.get('word_categories', {}),
            'feature_analysis': analysis_result.get('feature_analysis', {}),
            'processing_timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in TF-IDF analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TF-IDF analysis failed: {str(e)}")

@router.get("/model-tfidf-vocab")
async def get_model_vocabulary():
    """Get the TF-IDF vocabulary from the trained model"""
    
    try:
        sys.path.append('/app/services/anomaly-detector')
        from oneclass_svm_detector import OneClassSVMAnomalyDetector
        
        detector = OneClassSVMAnomalyDetector()
        
        # Load model
        model_path = os.path.join(detector.model_dir, 'oneclass_svm_model.pkl')
        if not os.path.exists(model_path):
            raise HTTPException(status_code=400, detail="SVM model not trained")
        
        detector.load_model()
        
        # Get vocabulary
        if hasattr(detector, 'vectorizer') and detector.vectorizer:
            vocab = detector.vectorizer.get_feature_names_out()
            vocab_size = len(vocab)
            
            return {
                'vocabulary_size': vocab_size,
                'top_100_words': vocab[:100].tolist(),  # First 100 words
                'feature_extraction_config': {
                    'max_features': detector.vectorizer.max_features,
                    'ngram_range': detector.vectorizer.ngram_range,
                    'stop_words': detector.vectorizer.stop_words
                }
            }
        else:
            raise HTTPException(status_code=500, detail="TF-IDF vectorizer not available")
            
    except Exception as e:
        logger.error(f"Error getting vocabulary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Vocabulary retrieval failed: {str(e)}")
                    contribution = float(embedding_scaled[0][i] * decision_score)
                    feature_contributions[name] = contribution
        else:
            # Use generic feature names
            for i in range(min(10, len(embedding_scaled[0]))):
                contribution = float(embedding_scaled[0][i] * decision_score)
                feature_contributions[f'Feature_{i}'] = contribution
        
        visualization_url = None
        if request.include_visualization:
            try:
                # Generate visualization
                visualizer = OneClassSVMVisualizer(analyzer)
                sessions_data = [{
                    'session_id': session.session_id,
                    'embedding': session.embedding_vector,
                    'is_anomaly': prediction == -1,
                    'raw_text': session.raw_text
                }]
                
                # Create debug output directory
                debug_dir = "/app/static/debug"
                os.makedirs(debug_dir, exist_ok=True)
                
                vis_path = os.path.join(debug_dir, f"svm_debug_{session.session_id}.html")
                visualizer.generate_svm_debug_report(sessions_data, vis_path)
                visualization_url = f"/static/debug/svm_debug_{session.session_id}.html"
            except Exception as e:
                logger.warning(f"Could not generate visualization: {e}")
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return SVMDebugResponse(
            session_id=request.session_id,
            decision_score=float(decision_score),
            prediction="Anomaly" if prediction == -1 else "Normal",
            confidence=float(abs(decision_score)),
            support_vectors_used=len(analyzer.one_class_svm.support_vectors_),
            feature_contributions=feature_contributions,
            visualization_url=visualization_url,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"SVM debug error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"SVM debug error: {str(e)}")

@router.get("/model-info", response_model=SVMModelInfo)
async def get_svm_model_info():
    """Get current SVM model information"""
    
    if not MLFirstAnomalyDetector:
        raise HTTPException(status_code=500, detail="ML analyzer not available")
    
    try:
        analyzer = MLFirstAnomalyDetector()
        
        # Check if model is fitted
        is_fitted = hasattr(analyzer.one_class_svm, 'support_vectors_')
        
        model_info = SVMModelInfo(
            parameters={
                'nu': analyzer.one_class_svm.nu,
                'gamma': analyzer.one_class_svm.gamma,
                'kernel': analyzer.one_class_svm.kernel
            },
            support_vectors_count=len(analyzer.one_class_svm.support_vectors_) if is_fitted else 0,
            is_fitted=is_fitted,
            feature_dimensions=analyzer.one_class_svm.support_vectors_.shape[1] if is_fitted else None,
            last_training_time=None  # Could be tracked in future
        )
        
        return model_info
        
    except Exception as e:
        logger.error(f"Error getting SVM model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@router.post("/batch-analyze", response_model=BatchAnalysisResponse)
async def batch_analyze_sessions(request: BatchAnalysisRequest, background_tasks: BackgroundTasks):
    """Analyze multiple sessions and return SVM decision summary"""
    
    if not MLFirstAnomalyDetector:
        raise HTTPException(status_code=500, detail="ML analyzer not available")
    
    start_time = datetime.now()
    
    try:
        analyzer = MLFirstAnomalyDetector()
        results = []
        decision_scores = []
        
        for session_id in request.session_ids:
            try:
                # In a real implementation, you would fetch session data from database
                # For now, we'll create a placeholder
                session_result = {
                    'session_id': session_id,
                    'decision_score': 0.0,  # Would be actual SVM score
                    'prediction': 'Normal',  # Would be actual prediction
                    'confidence': 0.0,       # Would be actual confidence
                    'processing_time_ms': 0,  # Would be actual processing time
                    'error': None
                }
                
                # TODO: Implement actual session processing
                # This would involve:
                # 1. Fetching session from database by session_id
                # 2. Processing with ML analyzer
                # 3. Getting SVM decision
                
                results.append(session_result)
                decision_scores.append(session_result['decision_score'])
                
            except Exception as e:
                logger.error(f"Error processing session {session_id}: {e}")
                results.append({
                    'session_id': session_id,
                    'error': str(e),
                    'decision_score': 0.0,
                    'prediction': 'Error',
                    'confidence': 0.0,
                    'processing_time_ms': 0
                })
        
        # Calculate summary statistics
        import numpy as np
        valid_scores = [r['decision_score'] for r in results if r.get('error') is None]
        
        summary_stats = {
            'total_sessions': len(request.session_ids),
            'successful_analyses': len(valid_scores),
            'error_count': len(request.session_ids) - len(valid_scores),
            'mean_decision_score': float(np.mean(valid_scores)) if valid_scores else 0.0,
            'std_decision_score': float(np.std(valid_scores)) if valid_scores else 0.0,
            'anomaly_count': sum(1 for r in results if r.get('prediction') == 'Anomaly'),
            'anomaly_rate': sum(1 for r in results if r.get('prediction') == 'Anomaly') / len(results) if results else 0.0
        }
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # If visualizations requested, generate them in background
        if request.include_visualizations:
            background_tasks.add_task(generate_batch_visualizations, request.session_ids)
        
        return BatchAnalysisResponse(
            results=results,
            total_processed=len(request.session_ids),
            processing_time_ms=processing_time,
            summary_stats=summary_stats
        )
        
    except Exception as e:
        logger.error(f"Batch analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch analysis error: {str(e)}")

@router.get("/performance-metrics")
async def get_svm_performance_metrics():
    """Get SVM performance metrics over time"""
    
    if not MLFirstAnomalyDetector:
        raise HTTPException(status_code=500, detail="ML analyzer not available")
    
    try:
        analyzer = MLFirstAnomalyDetector()
        
        # Get performance metrics
        performance_metrics = {
            'timestamp': datetime.now().isoformat(),
            'model_status': 'active' if hasattr(analyzer.one_class_svm, 'support_vectors_') else 'not_trained',
            'support_vector_count': len(analyzer.one_class_svm.support_vectors_) if hasattr(analyzer.one_class_svm, 'support_vectors_') else 0,
            'model_parameters': {
                'nu': analyzer.one_class_svm.nu,
                'gamma': analyzer.one_class_svm.gamma,
                'kernel': analyzer.one_class_svm.kernel
            }
        }
        
        # Add session statistics if available
        if analyzer.sessions:
            embeddings = []
            for session in analyzer.sessions:
                if hasattr(session, 'embedding_vector') and session.embedding_vector is not None:
                    embeddings.append(session.embedding_vector)
            
            if embeddings:
                import numpy as np
                embeddings_scaled = analyzer.scaler.transform(np.array(embeddings))
                decision_scores = analyzer.one_class_svm.decision_function(embeddings_scaled)
                predictions = analyzer.one_class_svm.predict(embeddings_scaled)
                
                performance_metrics.update({
                    'total_sessions_analyzed': len(embeddings),
                    'anomalies_detected': int(np.sum(predictions == -1)),
                    'anomaly_rate': float(np.sum(predictions == -1) / len(predictions)),
                    'decision_score_stats': {
                        'mean': float(np.mean(decision_scores)),
                        'std': float(np.std(decision_scores)),
                        'min': float(np.min(decision_scores)),
                        'max': float(np.max(decision_scores))
                    }
                })
        
        return performance_metrics
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting performance metrics: {str(e)}")

@router.post("/tune-parameters")
async def tune_svm_parameters(nu_values: List[float] = None, gamma_values: List[str] = None):
    """Automatically tune SVM parameters for better performance"""
    
    if not MLFirstAnomalyDetector or not OneClassSVMVisualizer:
        raise HTTPException(status_code=500, detail="SVM components not available")
    
    try:
        analyzer = MLFirstAnomalyDetector()
        
        if not analyzer.sessions:
            raise HTTPException(status_code=400, detail="No sessions available for parameter tuning")
        
        # Default parameter ranges if not provided
        if nu_values is None:
            nu_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
        if gamma_values is None:
            gamma_values = ['scale', 'auto', 0.001, 0.01, 0.1, 1.0]
        
        visualizer = OneClassSVMVisualizer(analyzer)
        
        # Prepare session data
        sessions_data = []
        for session in analyzer.sessions:
            if hasattr(session, 'embedding_vector') and session.embedding_vector is not None:
                sessions_data.append({
                    'session_id': session.session_id,
                    'embedding': session.embedding_vector,
                    'is_anomaly': getattr(session, 'is_anomaly', False),
                    'raw_text': getattr(session, 'raw_text', '')
                })
        
        if not sessions_data:
            raise HTTPException(status_code=400, detail="No valid sessions with embeddings found")
        
        # Run parameter debugging
        param_fig, param_df = visualizer.debug_svm_parameters(sessions_data)
        
        if param_df is not None:
            # Find optimal parameters (example: minimize false positive rate while maintaining detection)
            optimal_params = param_df.loc[param_df['anomaly_rate'].idxmin()]
            
            return {
                'current_parameters': {
                    'nu': analyzer.one_class_svm.nu,
                    'gamma': analyzer.one_class_svm.gamma
                },
                'recommended_parameters': {
                    'nu': optimal_params['nu'],
                    'gamma': optimal_params['gamma']
                },
                'parameter_analysis': param_df.to_dict('records'),
                'tuning_timestamp': datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Parameter tuning failed")
        
    except Exception as e:
        logger.error(f"Parameter tuning error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Parameter tuning error: {str(e)}")

async def generate_batch_visualizations(session_ids: List[str]):
    """Background task to generate visualizations for batch analysis"""
    try:
        logger.info(f"Generating batch visualizations for {len(session_ids)} sessions")
        # Implementation would create comprehensive visualizations
        # This is a placeholder for the background task
    except Exception as e:
        logger.error(f"Error generating batch visualizations: {e}")

# Health check endpoint for SVM debug service
@router.get("/health")
async def svm_debug_health():
    """Health check for SVM debug service"""
    
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'visualizer': OneClassSVMVisualizer is not None,
            'ml_analyzer': MLFirstAnomalyDetector is not None,
            'transaction_session': TransactionSession is not None
        }
    }
    
    if not all(health_status['components'].values()):
        health_status['status'] = 'degraded'
        health_status['warnings'] = [
            comp for comp, available in health_status['components'].items() 
            if not available
        ]
    
    return health_status
