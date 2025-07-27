"""
Expert Feedback API Endpoint
Handles expert input for continuous learning and model improvement
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
from .ml_analyzer import MLFirstAnomalyDetector
from .database import get_db_connection

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/expert-feedback", tags=["Expert Feedback"])

# Global ML analyzer instance (shared across requests)
ml_analyzer = None

class ExpertFeedbackRequest(BaseModel):
    session_id: str
    expert_label: str  # 'normal', 'anomaly', or specific anomaly type
    expert_confidence: float  # 0.0 to 1.0
    feedback_type: str  # 'confirmation', 'correction', 'new_discovery'
    expert_explanation: Optional[str] = None
    expert_name: Optional[str] = None

class ExpertFeedbackResponse(BaseModel):
    success: bool
    message: str
    feedback_id: Optional[str] = None
    training_triggered: bool = False
    model_performance: Optional[Dict[str, Any]] = None

class FeedbackStatsResponse(BaseModel):
    total_feedback_count: int
    feedback_by_type: Dict[str, int]
    model_accuracy_by_method: Dict[str, Dict[str, float]]
    recent_training_sessions: List[Dict[str, Any]]
    pending_feedback_count: int

def get_ml_analyzer():
    """Get or create the ML analyzer instance"""
    global ml_analyzer
    if ml_analyzer is None:
        ml_analyzer = MLFirstAnomalyDetector()
        logger.info("ML Analyzer initialized for expert feedback")
    return ml_analyzer

@router.post("/submit", response_model=ExpertFeedbackResponse)
async def submit_expert_feedback(
    feedback: ExpertFeedbackRequest,
    db=Depends(get_db_connection)
):
    """
    Submit expert feedback for a specific session
    This triggers the continuous learning pipeline
    """
    try:
        analyzer = get_ml_analyzer()
        
        # Validate the feedback
        if feedback.expert_confidence < 0.0 or feedback.expert_confidence > 1.0:
            raise HTTPException(status_code=400, detail="Expert confidence must be between 0.0 and 1.0")
        
        if feedback.feedback_type not in ['confirmation', 'correction', 'new_discovery']:
            raise HTTPException(status_code=400, detail="Invalid feedback type")
        
        # Collect the feedback using the ML analyzer
        success = analyzer.collect_expert_feedback(
            session_id=feedback.session_id,
            expert_label=feedback.expert_label,
            expert_confidence=feedback.expert_confidence,
            feedback_type=feedback.feedback_type,
            expert_explanation=feedback.expert_explanation
        )
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Session {feedback.session_id} not found")
        
        # Store feedback in database for audit trail
        feedback_id = await store_feedback_in_db(db, feedback)
        
        # Check if training was triggered
        training_triggered = len(analyzer.feedback_buffer) == 0  # Buffer is cleared after training
        
        # Get current model performance statistics
        model_performance = get_model_performance_stats(analyzer)
        
        logger.info(f"Expert feedback submitted for session {feedback.session_id} by {feedback.expert_name}")
        
        return ExpertFeedbackResponse(
            success=True,
            message="Expert feedback submitted successfully",
            feedback_id=feedback_id,
            training_triggered=training_triggered,
            model_performance=model_performance
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit expert feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/stats", response_model=FeedbackStatsResponse)
async def get_feedback_stats(db=Depends(get_db_connection)):
    """
    Get statistics about expert feedback and model performance
    """
    try:
        analyzer = get_ml_analyzer()
        
        # Get feedback statistics from database
        feedback_stats = await get_feedback_stats_from_db(db)
        
        # Calculate model accuracy by detection method
        accuracy_by_method = {}
        for method, stats in analyzer.detection_method_feedback.items():
            total = stats['tp'] + stats['fp'] + stats['tn'] + stats['fn']
            if total > 0:
                accuracy = (stats['tp'] + stats['tn']) / total
                precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0.0
                recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0.0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                accuracy_by_method[method] = {
                    'accuracy': round(accuracy, 3),
                    'precision': round(precision, 3),
                    'recall': round(recall, 3),
                    'f1_score': round(f1_score, 3),
                    'total_samples': total
                }
        
        # Get recent training sessions
        recent_training = analyzer.model_performance_history[-5:] if analyzer.model_performance_history else []
        
        return FeedbackStatsResponse(
            total_feedback_count=feedback_stats['total_count'],
            feedback_by_type=feedback_stats['by_type'],
            model_accuracy_by_method=accuracy_by_method,
            recent_training_sessions=recent_training,
            pending_feedback_count=len(analyzer.feedback_buffer)
        )
        
    except Exception as e:
        logger.error(f"Failed to get feedback stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/trigger-training")
async def trigger_manual_training():
    """
    Manually trigger model retraining with current feedback
    """
    try:
        analyzer = get_ml_analyzer()
        
        if len(analyzer.feedback_buffer) == 0:
            return {"message": "No feedback available for training", "triggered": False}
        
        # Trigger retraining
        analyzer.continuous_model_retraining()
        
        return {
            "message": f"Manual training triggered with {len(analyzer.feedback_buffer)} feedback samples",
            "triggered": True
        }
        
    except Exception as e:
        logger.error(f"Failed to trigger manual training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/model-performance")
async def get_model_performance():
    """
    Get detailed model performance metrics
    """
    try:
        analyzer = get_ml_analyzer()
        
        performance = get_model_performance_stats(analyzer)
        
        # Add additional metrics
        performance["feedback_buffer_size"] = len(analyzer.feedback_buffer)
        performance["learning_threshold"] = analyzer.learning_threshold
        performance["ensemble_weights"] = getattr(analyzer, 'ensemble_weights', {})
        performance["dynamic_thresholds"] = {
            "semantic_threshold": getattr(analyzer, 'semantic_threshold', 0.75),
            "sequence_threshold": getattr(analyzer, 'sequence_threshold', 0.7),
            "ensemble_threshold": getattr(analyzer, 'ensemble_threshold', 0.6)
        }
        
        return performance
        
    except Exception as e:
        logger.error(f"Failed to get model performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Helper functions

async def store_feedback_in_db(db, feedback: ExpertFeedbackRequest) -> str:
    """Store expert feedback in database for audit trail"""
    try:
        query = """
        INSERT INTO expert_feedback (
            session_id, expert_label, expert_confidence, feedback_type,
            expert_explanation, expert_name, created_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        RETURNING id
        """
        
        result = await db.fetchrow(
            query,
            feedback.session_id,
            feedback.expert_label,
            feedback.expert_confidence,
            feedback.feedback_type,
            feedback.expert_explanation,
            feedback.expert_name,
            datetime.now()
        )
        
        return str(result['id'])
        
    except Exception as e:
        logger.warning(f"Failed to store feedback in database: {str(e)}")
        return "db_error"

async def get_feedback_stats_from_db(db) -> Dict[str, Any]:
    """Get feedback statistics from database"""
    try:
        # Total count
        total_result = await db.fetchrow("SELECT COUNT(*) as count FROM expert_feedback")
        total_count = total_result['count'] if total_result else 0
        
        # By type
        type_results = await db.fetch("""
            SELECT feedback_type, COUNT(*) as count 
            FROM expert_feedback 
            GROUP BY feedback_type
        """)
        
        by_type = {row['feedback_type']: row['count'] for row in type_results}
        
        return {
            'total_count': total_count,
            'by_type': by_type
        }
        
    except Exception as e:
        logger.warning(f"Failed to get feedback stats from database: {str(e)}")
        return {'total_count': 0, 'by_type': {}}

def get_model_performance_stats(analyzer) -> Dict[str, Any]:
    """Calculate current model performance statistics"""
    try:
        # Calculate overall statistics
        all_stats = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
        method_count = 0
        
        for method, stats in analyzer.detection_method_feedback.items():
            if stats['tp'] + stats['fp'] + stats['tn'] + stats['fn'] > 0:
                all_stats['tp'] += stats['tp']
                all_stats['fp'] += stats['fp']
                all_stats['tn'] += stats['tn']
                all_stats['fn'] += stats['fn']
                method_count += 1
        
        total_samples = sum(all_stats.values())
        
        if total_samples > 0:
            overall_accuracy = (all_stats['tp'] + all_stats['tn']) / total_samples
            overall_precision = all_stats['tp'] / (all_stats['tp'] + all_stats['fp']) if (all_stats['tp'] + all_stats['fp']) > 0 else 0.0
            overall_recall = all_stats['tp'] / (all_stats['tp'] + all_stats['fn']) if (all_stats['tp'] + all_stats['fn']) > 0 else 0.0
        else:
            overall_accuracy = overall_precision = overall_recall = 0.0
        
        return {
            'overall_accuracy': round(overall_accuracy, 3),
            'overall_precision': round(overall_precision, 3),
            'overall_recall': round(overall_recall, 3),
            'total_feedback_samples': total_samples,
            'active_detection_methods': method_count,
            'last_updated': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.warning(f"Failed to calculate model performance stats: {str(e)}")
        return {'error': str(e)}
