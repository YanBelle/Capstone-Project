# ML-First ABM Anomaly Detection with Supervised Learning
import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import Dict, List, Tuple, Optional, Any
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
import joblib
import os
import hashlib
import time

# Monitoring integration
try:
    from monitoring_integration import (
        mark_ml_training_start, mark_ml_training_complete, 
        mark_ml_detection_run, mark_ml_error, log_ml_activity
    )
except ImportError:
    # Fallback functions if monitoring not available
    def mark_ml_training_start(model_type="unknown"): pass
    def mark_ml_training_complete(accuracy, training_time, model_type="unknown"): pass
    def mark_ml_detection_run(session_count, anomaly_count): pass
    def mark_ml_error(error_message, context=None): pass
    def log_ml_activity(activity, **kwargs): pass

# NLP and ML imports
from transformers import BertTokenizer, BertModel
import torch
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Optional TensorFlow imports for API service
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available in API service. Some features may be limited.")

# Import simple embeddings fallback
from simple_embeddings import SimpleEmbeddingGenerator

# Additional ML imports for sentiment and negative text detection
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from scipy import stats
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AnomalyDetection:
    """Represents a single anomaly detection within a session"""
    anomaly_type: str
    confidence: float
    detection_method: str  # 'isolation_forest', 'one_class_svm', 'expert_rule', 'supervised'
    description: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    timestamp: Optional[datetime] = None
    details: Optional[Dict[str, Any]] = None

@dataclass
class TransactionSession:
    """Represents a single transaction session from EJ logs with support for multiple anomalies"""
    session_id: str
    raw_text: str
    clean_text: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    terminal_id: Optional[str] = None  # ABM Terminal ID extracted from filename
    embedding: Optional[np.ndarray] = None
    
    # Multi-anomaly support
    anomalies: List[AnomalyDetection] = field(default_factory=list)
    overall_anomaly_score: float = 0.0
    max_severity: str = "normal"  # highest severity among all anomalies
    
    # Legacy fields for backwards compatibility
    is_anomaly: bool = False
    anomaly_score: float = 0.0
    anomaly_type: Optional[str] = None
    supervised_label: Optional[str] = None
    supervised_confidence: float = 0.0
    extracted_details: Optional[Dict[str, Any]] = None
    
    def add_anomaly(self, anomaly_type: str, confidence: float = 0.5, detection_method: str = "unknown", 
                   description: str = "Anomaly detected", severity: str = "medium", details: Dict[str, Any] = None):
        """Add a new anomaly detection to this session"""
        anomaly = AnomalyDetection(
            anomaly_type=anomaly_type,
            confidence=confidence,
            detection_method=detection_method,
            description=description,
            severity=severity,
            timestamp=datetime.now(),
            details=details or {}
        )
        self.anomalies.append(anomaly)
        
        # Update session-level flags
        self.is_anomaly = True
        self.overall_anomaly_score = max(self.overall_anomaly_score, confidence)
        self._update_max_severity()
        
        # Update legacy fields for backwards compatibility
        if not self.anomaly_type or confidence > self.anomaly_score:
            self.anomaly_type = anomaly_type
            self.anomaly_score = confidence
    
    def _update_max_severity(self):
        """Update the maximum severity level across all anomalies"""
        severity_levels = {"normal": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        max_level = 0
        for anomaly in self.anomalies:
            level = severity_levels.get(anomaly.severity, 0)
            max_level = max(max_level, level)
        
        severity_names = {0: "normal", 1: "low", 2: "medium", 3: "high", 4: "critical"}
        self.max_severity = severity_names[max_level]
    
    def get_anomaly_types(self) -> List[str]:
        """Get list of all anomaly types detected in this session"""
        return [anomaly.anomaly_type for anomaly in self.anomalies]
    
    def get_anomalies_by_severity(self, min_severity: str = "low") -> List[AnomalyDetection]:
        """Get anomalies filtered by minimum severity level"""
        severity_levels = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        min_level = severity_levels.get(min_severity, 1)
        
        return [anomaly for anomaly in self.anomalies 
                if severity_levels.get(anomaly.severity, 0) >= min_level]
    
    def has_anomaly_type(self, anomaly_type: str) -> bool:
        """Check if session has a specific anomaly type"""
        return any(anomaly.anomaly_type == anomaly_type for anomaly in self.anomalies)
    
    def get_anomaly_count(self) -> int:
        """Get the total number of anomalies detected in this session"""
        return len(self.anomalies)
    
    def get_max_severity(self) -> str:
        """Get the maximum severity level across all anomalies"""
        return self.max_severity
    
    def has_critical_anomalies(self) -> bool:
        """Check if session has any critical anomalies"""
        return any(anomaly.severity == "critical" for anomaly in self.anomalies)
    
    def get_critical_anomalies_count(self) -> int:
        """Get count of critical anomalies"""
        return len([a for a in self.anomalies if a.severity == "critical"])
    
    def get_high_severity_anomalies_count(self) -> int:
        """Get count of high severity anomalies"""
        return len([a for a in self.anomalies if a.severity == "high"])
    
    def get_detection_methods(self) -> List[str]:
        """Get list of unique detection methods used"""
        return list(set(anomaly.detection_method for anomaly in self.anomalies))
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API response"""
        return {
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'is_anomaly': self.is_anomaly,
            'anomaly_count': self.get_anomaly_count(),
            'anomaly_types': [a.anomaly_type for a in self.anomalies],
            'max_severity': self.max_severity,
            'overall_anomaly_score': self.overall_anomaly_score,
            'critical_anomalies_count': self.get_critical_anomalies_count(),
            'high_severity_anomalies_count': self.get_high_severity_anomalies_count(),
            'detection_methods': self.get_detection_methods(),
            'anomalies_detail': [
                {
                    'type': a.anomaly_type,
                    'confidence': a.confidence,
                    'method': a.detection_method,
                    'severity': a.severity,
                    'description': a.description,
                    'timestamp': a.timestamp.isoformat() if a.timestamp else None,
                    'details': a.details
                }
                for a in self.anomalies
            ]
        }
    


class MLFirstAnomalyDetector:
    """ML-First approach with supervised learning integration and expert knowledge"""
    
    def __init__(self, model_name: str = 'bert-base-uncased'):
        # Initialize ML models
        self.model_name = model_name
        self.tokenizer = None
        self.bert_model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=50)
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.one_class_svm = OneClassSVM(gamma='auto')
        
        # Initialize ML-first components
        self.normal_embeddings_cluster = None
        self.anomaly_embeddings_cluster = None
        self.cluster_centers = None
        self.cluster_threshold = 2.0
        self.learned_normal_sequences = []
        self.learned_anomaly_sequences = []
        self.expert_normal_patterns = {}
        
        # Dynamic thresholds (adjusted by expert feedback)
        self.semantic_threshold = 0.75
        self.sequence_threshold = 0.7
        self.ensemble_threshold = 0.6
        
        # Ensemble weights (adjusted by expert feedback)
        self.ensemble_weights = {
            'bert_semantic': 0.3,
            'lstm_sequence': 0.25,
            'ml_ensemble': 0.25,
            'clustering': 0.2
        }
        
        # Load expert rules (kept minimal for critical safety only)
        self.expert_rules = self.load_expert_rules()
        
        # Initialize supervised model components
        self.supervised_classifier = None
        self.label_encoder = None
        
        # ML-first continuous learning system
        self.initialize_feedback_system()
        
        # Load pre-trained models if available
        self.load_pretrained_ml_models()
        
        # Initialize explanation patterns (reduced to minimal critical set)
        self.explanation_patterns = self._get_minimal_explanation_patterns()
        
        # Enhanced session tracking
        self.sessions = []
        self.embeddings_matrix = None
        self.latest_anomaly_summary = None
    
    def _get_minimal_explanation_patterns(self) -> Dict[str, re.Pattern]:
        """Get minimal set of critical explanation patterns (safety-focused only)"""
        return {
            # Only critical safety patterns that require immediate action
            'critical_hardware_fault': re.compile(r'HARDWARE\s+FAULT|SYSTEM\s+FAILURE|CRITICAL\s+ERROR', re.IGNORECASE),
            'security_violation': re.compile(r'UNAUTHORIZED|SECURITY\s+VIOLATION|TAMPER\s+DETECTED', re.IGNORECASE),
            'power_failure': re.compile(r'POWER\s+FAILURE|UPS\s+FAILURE|EMERGENCY\s+SHUTDOWN', re.IGNORECASE)
        }
    
    def initialize_feedback_system(self):
        """Enhanced feedback system initialization for ML-first approach"""
        self.feedback_buffer = []
        self.learning_threshold = 5  # Reduced threshold for more frequent learning
        self.feedback_weights = {
            'expert_confirmed_anomaly': 3.0,
            'expert_confirmed_normal': 3.0,
            'expert_new_anomaly_type': 4.0,
            'false_positive_correction': 2.5,
            'false_negative_correction': 3.5
        }
        self.model_performance_history = []
        
        # ML-first specific feedback tracking
        self.detection_method_feedback = {
            'bert_semantic': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
            'lstm_sequence': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
            'ml_ensemble': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
            'clustering': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
        }

    def load_pretrained_ml_models(self):
        """Load pre-trained ML models if available"""
        try:
            # Try to load BERT model and tokenizer
            if self.model_name:
                try:
                    from transformers import BertTokenizer, BertModel
                    self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
                    self.bert_model = BertModel.from_pretrained(self.model_name)
                    logger.info(f"Successfully loaded BERT model: {self.model_name}")
                except Exception as e:
                    logger.warning(f"Could not load BERT model {self.model_name}: {e}")
                    self.tokenizer = None
                    self.bert_model = None
            
            # Try to load any saved ML models
            models_dir = '/app/models'
            if os.path.exists(models_dir):
                try:
                    isolation_forest_path = os.path.join(models_dir, 'isolation_forest.pkl')
                    if os.path.exists(isolation_forest_path):
                        self.isolation_forest = joblib.load(isolation_forest_path)
                        logger.info("Loaded pre-trained Isolation Forest model")
                except Exception as e:
                    logger.warning(f"Could not load Isolation Forest model: {e}")
                
                try:
                    svm_path = os.path.join(models_dir, 'one_class_svm.pkl')
                    if os.path.exists(svm_path):
                        self.one_class_svm = joblib.load(svm_path)
                        logger.info("Loaded pre-trained One-Class SVM model")
                except Exception as e:
                    logger.warning(f"Could not load One-Class SVM model: {e}")
                    
        except Exception as e:
            logger.warning(f"Error loading pre-trained models: {e}")

    def collect_expert_feedback(self, session_id: str, expert_label: str, 
                               expert_confidence: float, feedback_type: str, 
                               expert_explanation: str = None) -> bool:
        """
        Enhanced expert feedback collection for ML-first continuous learning
        
        Args:
            session_id: ID of the session being corrected
            expert_label: Expert's classification ('normal', 'anomaly', or specific type)
            expert_confidence: Expert's confidence (0.0 to 1.0)
            feedback_type: Type of feedback ('confirmation', 'correction', 'new_discovery')
            expert_explanation: Optional explanation from expert
        """
        try:
            # Find the session
            session = next((s for s in self.sessions if s.session_id == session_id), None)
            if not session:
                logger.warning(f"Session {session_id} not found for feedback collection")
                return False
            
            # Get current ML predictions for this session
            ml_prediction = session.is_anomaly
            ml_confidence = getattr(session, 'overall_anomaly_score', 0.0)
            detection_methods = [anomaly.detection_method for anomaly in session.anomalies]
            
            # Create enhanced feedback record
            feedback_record = {
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'session_text': session.raw_text,
                'expert_label': expert_label,
                'expert_confidence': expert_confidence,
                'expert_explanation': expert_explanation,
                'feedback_type': feedback_type,
                'ml_prediction': ml_prediction,
                'ml_confidence': ml_confidence,
                'detection_methods': detection_methods,
                'anomaly_types': session.get_anomaly_types() if hasattr(session, 'get_anomaly_types') else [],
                'feedback_weight': self._calculate_feedback_weight(ml_prediction, expert_label, expert_confidence, feedback_type)
            }
            
            # Add to feedback buffer
            self.feedback_buffer.append(feedback_record)
            
            # Update method-specific feedback statistics
            self._update_method_feedback_stats(feedback_record)
            
            # Log the feedback
            logger.info(f"Expert feedback collected for session {session_id}: "
                       f"Expert={expert_label} (conf: {expert_confidence}), "
                       f"ML={ml_prediction} (conf: {ml_confidence}), "
                       f"Type={feedback_type}")
            
            # Trigger retraining if threshold reached
            if len(self.feedback_buffer) >= self.learning_threshold:
                logger.info(f"Feedback threshold reached ({len(self.feedback_buffer)} samples), triggering retraining")
                self.continuous_model_retraining()
            
            return True
            
        except Exception as e:
            logger.error(f"Expert feedback collection failed for session {session_id}: {str(e)}")
            return False
    
    def _calculate_feedback_weight(self, ml_prediction: bool, expert_label: str, 
                                  expert_confidence: float, feedback_type: str) -> float:
        """Calculate the weight of feedback based on agreement and confidence"""
        base_weight = self.feedback_weights.get(f"expert_{feedback_type}", 1.0)
        
        # Increase weight for high-confidence expert corrections
        confidence_multiplier = 1.0 + (expert_confidence - 0.5)  # 0.5 to 1.5 range
        
        # Increase weight for corrections vs confirmations
        expert_is_anomaly = expert_label != 'normal' and expert_label != ''
        if ml_prediction != expert_is_anomaly:
            # This is a correction
            base_weight *= 1.5
        
        return base_weight * confidence_multiplier
    
    def _update_method_feedback_stats(self, feedback_record: Dict):
        """Update statistics for each detection method based on expert feedback"""
        expert_label = feedback_record['expert_label']
        ml_prediction = feedback_record['ml_prediction']
        detection_methods = feedback_record['detection_methods']
        
        expert_is_anomaly = expert_label != 'normal' and expert_label != ''
        
        # Update stats for each method that participated in detection
        for method in detection_methods:
            if method in self.detection_method_feedback:
                stats = self.detection_method_feedback[method]
                
                if expert_is_anomaly and ml_prediction:
                    stats['tp'] += 1  # True Positive
                elif expert_is_anomaly and not ml_prediction:
                    stats['fn'] += 1  # False Negative
                elif not expert_is_anomaly and ml_prediction:
                    stats['fp'] += 1  # False Positive
                elif not expert_is_anomaly and not ml_prediction:
                    stats['tn'] += 1  # True Negative
        # Initialize BERT for embeddings
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert_model = BertModel.from_pretrained(model_name)
        self.bert_model.eval()
        
        # Initialize unsupervised models
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        
        self.one_class_svm = OneClassSVM(
            kernel='rbf',
            gamma='auto',
            nu=0.05
        )
        
        self.autoencoder = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=50)
        
        # Supervised model (loaded if available)
        self.supervised_classifier = None
        self.label_encoder = None
        self.load_supervised_model()
        
        # Storage
        self.sessions: List[TransactionSession] = []
        self.embeddings_matrix = None
        
        # Expert knowledge system to avoid false positives
        self.expert_rules = self.load_expert_rules()
        
        # Regex patterns for explanation
        self.explanation_patterns = {
            'supervisor_mode': re.compile(r'SUPERVISOR\s+MODE\s+(ENTRY|EXIT)', re.IGNORECASE),
            'unable_to_dispense': re.compile(r'UNABLE\s+TO\s+DISPENSE', re.IGNORECASE),
            'device_error': re.compile(r'DEVICE\s+ERROR', re.IGNORECASE),
            'power_reset': re.compile(r'POWER-UP/RESET', re.IGNORECASE),
            'cash_retract': re.compile(r'CASHIN\s+RETRACT\s+STARTED', re.IGNORECASE),
            'no_dispense': re.compile(r'NO\s+DISPENSE\s+SUCCESS', re.IGNORECASE),
            'notes_issue': re.compile(r'NOTES\s+(TAKEN|PRESENTED)', re.IGNORECASE),
            'error_codes': re.compile(r'(ESC|VAL|REF|REJECTS):\s*(\d+)', re.IGNORECASE),
            'note_error': re.compile(r'NOTE\s+ERROR\s+OCCURRED', re.IGNORECASE),
            'recovery_failed': re.compile(r'RECOVERY\s+FAILED', re.IGNORECASE)
        }
        
        # Import DeepLog analyzer
        try:
            from deeplog_analyzer import DeepLogAnalyzer, create_normal_training_patterns
        except ImportError:
            logger.warning("DeepLog analyzer not available")
            DeepLogAnalyzer = None
            create_normal_training_patterns = None
        
        # Initialize DeepLog analyzer
        self.deeplog_analyzer = None
        self.deeplog_trained = False
        if DeepLogAnalyzer:
            try:
                self.deeplog_analyzer = DeepLogAnalyzer(window_size=8, top_k=7)
                # Try to load existing model
                if self.deeplog_analyzer.load_model():
                    self.deeplog_trained = True
                    logger.info("DeepLog model loaded successfully")
                else:
                    logger.info("DeepLog model not found - will need training")
            except Exception as e:
                logger.error(f"Error initializing DeepLog: {e}")
                self.deeplog_analyzer = None
        
        # Initialize sentiment analysis and negative text detection models
        self.initialize_sentiment_models()
        
        # Initialize advanced ML ensemble detector
        # self.ensemble_detector = EnsembleAnomalyDetector()  # TODO: Implement if needed
        
        # Initialize continuous learning feedback system
        self.initialize_feedback_system()
    
    def load_supervised_model(self):
        """Load supervised model if available"""
        model_path = "/app/models/supervised_classifier.pkl"
        encoder_path = "/app/models/label_encoder.pkl"
        
        if os.path.exists(model_path):
            try:
                self.supervised_classifier = joblib.load(model_path)
                if os.path.exists(encoder_path):
                    self.label_encoder = joblib.load(encoder_path)
                logger.info("Supervised model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading supervised model: {str(e)}")
                self.supervised_classifier = None
                self.label_encoder = None
        else:
            logger.info("No supervised model found. Using unsupervised detection only.")
    
    def load_expert_rules(self) -> Dict:
        """Load expert-defined rules for transaction classification to avoid false positives"""
        return {
            "normal_transaction_indicators": [
                # Successful withdrawal patterns - THESE ARE NORMAL, NOT ANOMALIES
                ("NOTES PRESENTED", "NOTES TAKEN"),
                ("CARD INSERTED", "PIN ENTERED", "NOTES PRESENTED", "NOTES TAKEN", "CARD TAKEN"),
                ("NOTES STACKED", "NOTES PRESENTED", "NOTES TAKEN"),
                
                # Successful inquiry patterns  
                ("CARD INSERTED", "PIN ENTERED", "CARD TAKEN"),
                ("BALANCE INQUIRY", "RECEIPT PRINTED", "CARD TAKEN"),
            ],
            
            "genuine_anomaly_indicators": [
                # Actual problems requiring attention
                ("UNABLE TO DISPENSE",),
                ("DEVICE ERROR",),
                ("NOTES PRESENTED", "TIMEOUT"),
                ("NOTES PRESENTED", "NOTES RETRACTED"),
                ("HARDWARE FAULT",),
                ("COMMUNICATION ERROR",),
                ("SUPERVISOR MODE ENTRY",),
            ],
            
            "maintenance_indicators": [
                # Normal maintenance activities
                ("POWER-UP/RESET",),
                ("SUPERVISOR MODE EXIT",),
                ("CASSETTE REPLENISHED",),
                ("SYSTEM STARTUP",),
            ]
        }
    
    def process_ej_logs(self, file_path: str) -> pd.DataFrame:
        """Main entry point for processing EJ logs"""
        logger.info(f"Processing EJ logs from {file_path}")
        log_ml_activity("Started processing EJ logs", details={"file_path": file_path})
        
        # Step 1: Read raw logs
        raw_logs = self.read_raw_logs(file_path)
        
        # Step 2: Split into sessions
        self.sessions = self.split_into_sessions(raw_logs, file_path)
        logger.info(f"Found {len(self.sessions)} transaction sessions")
        
        if len(self.sessions) == 0:
            logger.warning("No transaction sessions found in the log file")
            log_ml_activity("No sessions found in log file", details={"file_path": file_path})
            return pd.DataFrame()
        
        # TEMPORARY: Process only first 1000 sessions for testing
        if len(self.sessions) > 4000:
            logger.warning(f"Processing only first 4000 sessions out of {len(self.sessions)} for faster results")
            self.sessions = self.sessions[:4000]

        # Step 3: Generate embeddings
        log_ml_activity("Generating embeddings", details={"session_count": len(self.sessions)})
        self.embeddings_matrix = self.generate_embeddingsUsingSentence(self.sessions)
        
        # Step 4: Unsupervised anomaly detection
        log_ml_activity("Running unsupervised anomaly detection")
        anomaly_results = self.detect_anomalies_unsupervised()
        
        # Step 5: Supervised classification (if model available)
        if self.supervised_classifier is not None:
            log_ml_activity("Applying supervised classification")
            self.apply_supervised_classification()
        
        # Step 6: Extract explanations
        self.extract_anomaly_explanations()
        
        # Step 6.5: Train DeepLog model if not already trained
        if self.deeplog_analyzer and not self.deeplog_trained:
            log_ml_activity("Training DeepLog model on current sessions")
            self.train_deeplog_model()
        
        # Step 7: Final expert validation and reporting
        self.perform_final_expert_validation()
        
        # Step 8: Create results dataframe
        results_df = self.create_results_dataframe()
        
        # Step 9: Generate comprehensive anomaly summary report
        anomaly_summary = self.generate_anomaly_summary_report()
        
        # Log summary with false positive prevention details
        total_anomalies = results_df['is_anomaly'].sum()
        total_overrides = sum(1 for session in self.sessions 
                            if session.extracted_details and 
                            session.extracted_details.get('expert_override', False))
        
        logger.info(f"Processing complete. Found {total_anomalies} genuine anomalies")
        logger.info(f"Expert system prevented {total_overrides} false positives")
        
        # Log anomaly breakdown
        logger.info(f"Anomaly Type Breakdown: {anomaly_summary['anomaly_type_summary']['counts']}")
        if 'host_decline' in anomaly_summary['anomaly_type_summary']['counts']:
            decline_count = anomaly_summary['anomaly_type_summary']['counts']['host_decline']
            logger.info(f"Host Declines (UNABLE TO PROCESS): {decline_count}")
        
        # Mark detection run for monitoring
        mark_ml_detection_run(session_count=len(self.sessions), anomaly_count=total_anomalies)
        log_ml_activity("Completed EJ log processing", 
                       details={
                           "sessions_processed": len(self.sessions),
                           "anomalies_detected": total_anomalies,
                           "false_positives_prevented": total_overrides,
                           "anomaly_breakdown": anomaly_summary['anomaly_type_summary']['counts']
                       })
        
        # Store the summary for external access
        self.latest_anomaly_summary = anomaly_summary
        
        return results_df
    
    def read_raw_logs(self, file_path: str) -> str:
        """Step 1: Read raw EJ logs as-is"""
        logger.info(f"Reading raw EJ logs from {file_path}")
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            raw_content = file.read()
        return raw_content
    
    def _apply_bertviz_cleaning(self, raw_text: str) -> str:
        """
        Apply BertViz _preprocess_text method to clean EJ text
        Falls back to original text if BertViz is not available
        """
        try:
            # Import BertViz analyzer
            from bertviz_analyzer import BertVisualizationAnalyzer
            bert_analyzer = BertVisualizationAnalyzer()
            cleaned_text = bert_analyzer._preprocess_text(raw_text)
            logger.info("Applied BertViz preprocessing to session text")
            return cleaned_text
        except ImportError:
            logger.warning("BertViz analyzer not available, using original text")
            return raw_text
        except Exception as e:
            logger.error(f"Error applying BertViz cleaning: {str(e)}, using original text")
            return raw_text

    def split_into_sessions(self, raw_logs: str, file_path: str = None) -> List[TransactionSession]:
        """Step 2: Split logs into transaction sessions with unique IDs
        
        Sessionization logic:
        - Start a session when encountering "TRANSACTION START" or "CARDLESS TRANSACTION START"
        - Take the session start time from the line immediately above the session start marker
        - Continue capturing all lines until the next "TRANSACTION START" or "CARDLESS TRANSACTION START" is found
        - This ensures we capture everything including post-transaction errors
        - Extract terminal ID from filename in format: ABM{terminal_id}EJ_YYYYMMDD_YYYYMMDD.txt
        """
        logger.info("Splitting logs into transaction sessions")
        
        sessions = []
        
        # Extract file identifier and terminal ID for unique session IDs
        file_identifier = "unknown"
        terminal_id = None
        
        if file_path:
            file_name = os.path.basename(file_path)
            # Extract ABM number and date from filename like ABM416EJ_20250101_20250630.txt
            file_match = re.search(r'ABM(\d+)EJ_(\d{8})_(\d{8})', file_name)
            if file_match:
                terminal_id = file_match.group(1)  # Extract terminal ID (e.g., "416")
                abm_num = file_match.group(1)
                start_date = file_match.group(2)
                file_identifier = f"ABM{abm_num}_{start_date}"
                logger.info(f"Extracted terminal ID: {terminal_id} from filename: {file_name}")
            else:
                logger.warning(f"Could not extract terminal ID from filename: {file_name}. Expected format: ABM{{terminal_id}}EJ_YYYYMMDD_YYYYMMDD.txt")
                file_identifier = file_name.replace('.txt', '').replace('.', '_')
        
        # Add timestamp to ensure uniqueness across runs
        timestamp_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Split log into lines for line-by-line processing
        log_lines = raw_logs.split('\n')
        
        # Find all transaction start markers with their line numbers
        transaction_start_pattern = re.compile(
            r'(\*(?:TRANSACTION|CARDLESS TRANSACTION)\s+START\*)',
            re.IGNORECASE
        )
        
        # Find all start line numbers
        start_line_numbers = []
        for line_num, line in enumerate(log_lines):
            if transaction_start_pattern.search(line):
                start_line_numbers.append(line_num)
        
        if not start_line_numbers:
            # Try alternative patterns for your specific log format
            # Based on your sample, transactions seem to be bounded by timestamps and transaction numbers
            alternative_pattern = re.compile(
                r'\*(\d+)\*(\d{2}/\d{2}/\d{4})\*(\d{2}:\d{2})\*.*?(?=\*\d+\*\d{2}/\d{2}/\d{4}\*\d{2}:\d{2}\*|\Z)',
                re.DOTALL
            )
            matches = list(alternative_pattern.finditer(raw_logs))
            
            for i, match in enumerate(matches):
                trans_num = match.group(1)
                date_str = match.group(2)
                time_str = match.group(3)
                
                session_text = match.group(0)
                
                # Clean the raw text using BertViz _preprocess_text method
                cleaned_session_text = self._apply_bertviz_cleaning(session_text)
                
                # Generate unique session ID with file info and timestamp
                session_id = f"{file_identifier}_TXN_{trans_num}_{date_str.replace('/', '')}_{time_str.replace(':', '')}_{timestamp_suffix}_{i}"
                
                # Parse timestamps
                try:
                    start_time = datetime.strptime(f"{date_str} {time_str}", "%m/%d/%Y %H:%M")
                except:
                    start_time = None
                
                # Look for TRANSACTION END in the text
                end_time = None
                end_match = re.search(r'(\d{2}:\d{2}:\d{2})\s+TRANSACTION\s+END', session_text)
                if end_match and start_time:
                    try:
                        end_time_str = end_match.group(1)
                        end_time = datetime.strptime(f"{date_str} {end_time_str}", "%m/%d/%Y %H:%M:%S")
                    except:
                        pass
                
                session = TransactionSession(
                    session_id=session_id,
                    raw_text=session_text,
                    clean_text = cleaned_session_text,  # Store cleaned text as raw_text
                    start_time=start_time,
                    end_time=end_time,
                    terminal_id=terminal_id  # Include terminal ID from filename
                )
                sessions.append(session)
        else:
            # Process sessions based on transaction start markers
            for i, start_line_num in enumerate(start_line_numbers):
                # Find the end line number (start of next transaction or end of file)
                if i + 1 < len(start_line_numbers):
                    # End should be the line before the timestamp line that precedes the next TRANSACTION START
                    next_transaction_line = start_line_numbers[i + 1]
                    # Look for the timestamp line before the next transaction start
                    if next_transaction_line > 0:
                        # Find the timestamp line before the next transaction start
                        # We need to include all lines up to (but not including) the timestamp line
                        # that precedes the next transaction start
                        end_line_num = next_transaction_line - 1  # This is the timestamp line before next transaction
                    else:
                        end_line_num = next_transaction_line
                else:
                    end_line_num = len(log_lines)
                
                # Include the timestamp line that comes before this TRANSACTION START
                # We want to start from the timestamp line, not the TRANSACTION START line itself
                session_start_line = start_line_num
                if start_line_num > 0:
                    # Include the timestamp line before TRANSACTION START
                    session_start_line = start_line_num - 1
                
                # Extract session text - include everything from timestamp line to just before next timestamp line
                session_lines = log_lines[session_start_line:end_line_num]
                session_text = '\n'.join(session_lines)
                
                # Generate unique session ID with file info, content hash, timestamp, and index
                content_hash = hashlib.md5(session_text.encode()).hexdigest()[:8]
                session_id = f"{file_identifier}_SESSION_{i+1}_{content_hash}_{timestamp_suffix}"
                
                # Extract start time from the line immediately ABOVE the "TRANSACTION START" marker
                start_time = None
                if start_line_num > 0:
                    # Look at the line above the TRANSACTION START marker
                    previous_line = log_lines[start_line_num - 1]
                    start_time = self.extract_timestamp_from_line(previous_line)
                
                # Extract end time from the session content
                end_time = self.extract_timestamp(session_text, "end")
                
                # Clean the raw text using BertViz _preprocess_text method before creating the session
                cleaned_session_text = self._apply_bertviz_cleaning(session_text)
                
                session = TransactionSession(
                    session_id=session_id,
                    raw_text=cleaned_session_text,  # Store cleaned text as raw_text
                    start_time=start_time,
                    end_time=end_time,
                    terminal_id=terminal_id  # Include terminal ID from filename
                )
                sessions.append(session)
        
        logger.info(f"Created {len(sessions)} transaction sessions")
        return sessions
    
    def extract_timestamp(self, text: str, position: str) -> Optional[datetime]:
        """Extract timestamp from session text"""
        timestamp_patterns = [
            r'(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2}:\d{2})',
            r'(\d{4}/\d{2}/\d{2})\s+(\d{2}:\d{2}:\d{2})',
            r'(\d{2}:\d{2}:\d{2})'
        ]
        
        for pattern in timestamp_patterns:
            matches = re.findall(pattern, text)
            if matches:
                if position == "start" and matches:
                    return self.parse_timestamp(matches[0])
                elif position == "end" and matches:
                    return self.parse_timestamp(matches[-1])
        
        return None
    
    def parse_timestamp(self, timestamp_match) -> Optional[datetime]:
        """Parse timestamp from regex match"""
        try:
            if isinstance(timestamp_match, tuple):
                if len(timestamp_match) == 2:
                    return datetime.strptime(f"{timestamp_match[0]} {timestamp_match[1]}", "%m/%d/%Y %H:%M:%S")
            else:
                return datetime.strptime(timestamp_match, "%H:%M:%S")
        except:
            return None
    
    #using BERT for embeddings
    def generate_embeddingsUsingBERT(self, sessions: List[TransactionSession]) -> np.ndarray:
        """Step 3: Generate BERT embeddings for each session"""
        logger.info("Generating BERT embeddings for sessions")
        
        embeddings = []
        
        with torch.no_grad():
            for session in sessions:
                # For longer sessions, we need to be smarter about text processing
                # Instead of truncating, let's extract key patterns and summarize
                text = self.prepare_text_for_embedding(session.raw_text)
                
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                
                # Get BERT embeddings
                outputs = self.bert_model(**inputs)
                
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[0, 0, :].numpy()
                
                session.embedding = embedding
                embeddings.append(embedding)
        
        return np.array(embeddings)
    
    #using sentence-transformers for faster embeddings
    def generate_embeddingsUsingSentence(self, sessions: List[TransactionSession]) -> np.ndarray:
        """Step 3: Generate BERT embeddings for each session - OPTIMIZED"""
        logger.info("Generating BERT embeddings for sessions")
        
        embeddings = []
        batch_size = 32  # Process in batches
        
        # Use sentence-transformers for faster processing with error handling
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Using SentenceTransformer for embeddings")
            model = SentenceTransformer('all-MiniLM-L6-v2')  # Much faster than BERT
            
            # Process in batches with progress tracking
            for i in range(0, len(sessions), batch_size):
                batch_sessions = sessions[i:i+batch_size]
                batch_texts = [self.prepare_text_for_embedding(session.raw_text) for session in batch_sessions]
                
                # Generate embeddings for batch
                batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
                
                # Store embeddings
                for j, session in enumerate(batch_sessions):
                    session.embedding = batch_embeddings[j]
                    embeddings.append(batch_embeddings[j])
                
                # Log progress every 1000 sessions
                if (i + batch_size) % 5000 == 0:
                    logger.info(f"Processed {i + batch_size}/{len(sessions)} sessions")
                    
        except ImportError as e:
            logger.error(f"SentenceTransformer import failed: {e}")
            logger.info("Falling back to BERT embeddings")
            try:
                return self.generate_embeddingsUsingBERT(sessions)
            except Exception as bert_error:
                logger.error(f"BERT fallback also failed: {bert_error}")
                logger.info("Using simple TF-IDF embeddings as final fallback")
                return self.generate_simple_embeddings(sessions)
        except Exception as e:
            logger.error(f"Error with SentenceTransformer: {e}")
            logger.info("Falling back to BERT embeddings")
            try:
                return self.generate_embeddingsUsingBERT(sessions)
            except Exception as bert_error:
                logger.error(f"BERT fallback also failed: {bert_error}")
                logger.info("Using simple TF-IDF embeddings as final fallback")
                return self.generate_simple_embeddings(sessions)
        
        logger.info("Embedding generation complete")
        return np.array(embeddings)

    def detect_anomalies_unsupervised(self) -> Dict[str, np.ndarray]:
        """Step 4: Unsupervised anomaly detection with multi-anomaly support"""
        logger.info("Running unsupervised anomaly detection with multi-anomaly support")
        
        # Scale embeddings
        embeddings_scaled = self.scaler.fit_transform(self.embeddings_matrix)
        
        # Apply PCA if we have enough samples
        if len(self.sessions) > 50:
            embeddings_scaled = self.pca.fit_transform(embeddings_scaled)
        
        # Isolation Forest
        if_predictions = self.isolation_forest.fit_predict(embeddings_scaled)
        if_scores = self.isolation_forest.score_samples(embeddings_scaled)
        
        # One-Class SVM
        svm_predictions = self.one_class_svm.fit_predict(embeddings_scaled)
        svm_scores = self.one_class_svm.decision_function(embeddings_scaled)
        
        # Update sessions with results and apply expert knowledge for multi-anomaly detection
        for i, session in enumerate(self.sessions):
            # Normalize scores to 0-1 range
            if_score_norm = (if_scores[i] - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-8)
            svm_score_norm = (svm_scores[i] - svm_scores.min()) / (svm_scores.max() - svm_scores.min() + 1e-8)
            
            # Check for multiple types of anomalies
            self._detect_multiple_anomalies(session, if_predictions[i], svm_predictions[i], 
                                           if_score_norm, svm_score_norm)
            
            # Update legacy fields for backwards compatibility
            session.overall_anomaly_score = max(if_score_norm, svm_score_norm)
            session.is_anomaly = len(session.anomalies) > 0
            
            if session.anomalies:
                # Set primary anomaly type (highest confidence)
                primary_anomaly = max(session.anomalies, key=lambda x: x.confidence)
                session.anomaly_type = primary_anomaly.anomaly_type
                session.anomaly_score = primary_anomaly.confidence
        
        return {
            'if_predictions': if_predictions,
            'if_scores': if_scores,
            'svm_predictions': svm_predictions,
            'svm_scores': svm_scores
        }
    
    def _detect_multiple_anomalies(self, session: TransactionSession, if_pred: int, svm_pred: int, 
                                  if_score: float, svm_score: float):
        """Detect multiple types of anomalies in a single session"""
        events = self.extract_key_events(session.raw_text)
        
        # First check for normal patterns that should override anomaly detection
        if self._check_and_apply_normal_overrides(session, events):
            return  # Session is normal, no anomalies to add
        
        # Check for ML-detected anomalies
        if if_pred == -1:
            session.add_anomaly(
                anomaly_type="statistical_outlier_isolation",
                confidence=1.0 - if_score,
                detection_method="isolation_forest",
                description="Session identified as statistical outlier by Isolation Forest",
                severity=self._determine_severity(1.0 - if_score)
            )
        
        if svm_pred == -1:
            session.add_anomaly(
                anomaly_type="statistical_outlier_svm",
                confidence=1.0 - svm_score,
                detection_method="one_class_svm",
                description="Session identified as statistical outlier by One-Class SVM",
                severity=self._determine_severity(1.0 - svm_score)
            )
        
        # Check for specific anomaly patterns
        self._detect_specific_anomalies(session, events)
        
        # DeepLog sequential anomaly detection
        self._detect_deeplog_anomalies(session, events)
        
        # NEW: DeepLog-enhanced sentiment anomaly detection
        self._detect_deeplog_sentiment_anomalies(session, events)
        
        # Incomplete/Failed Transactions
        self._detect_incomplete_transactions(session, events, session.raw_text)
        
        # Machine Status anomalies detection
        self._detect_machine_status_anomalies(session, session.raw_text)
        
        # Advanced ML-based anomaly detection
        # TODO: Implement ensemble detector if needed
        # try:
        #     ensemble_anomalies = self.ensemble_detector.detect_ensemble_anomalies(session)
        #     for anomaly in ensemble_anomalies:
        #         if anomaly['type'] != 'ensemble_summary':  # Skip summary for individual anomalies
        #             session.add_anomaly(
        #                 anomaly_type=f"ml_{anomaly['type']}",
        #                 confidence=anomaly['confidence'],
        #                 detection_method=f"ml_{anomaly['detector']}",
        #                 description=anomaly['description'],
        #                 severity=self._determine_ml_severity(anomaly['confidence']),
        #                 details=anomaly
        #             )
        # except Exception as e:
        #     logger.warning(f"Advanced ML detection failed for session {session.session_id}: {str(e)}")
    
    def _determine_ml_severity(self, confidence: float) -> str:
        """Determine severity for ML-detected anomalies"""
        if confidence >= 0.9:
            return "critical"
        elif confidence >= 0.75:
            return "high"
        elif confidence >= 0.6:
            return "medium"
        else:
            return "low"
    
    def _check_and_apply_normal_overrides(self, session: TransactionSession, events: List[Dict]) -> bool:
        """Check for normal patterns and apply overrides"""
        if self.is_successful_withdrawal(session.raw_text, events):
            logger.info(f"Expert override applied for {session.session_id}: Successful withdrawal detected")
            session.extracted_details = {
                'expert_override': True,
                'override_reason': 'NOTES PRESENTED followed by NOTES TAKEN indicates successful cash collection',
                'detected_patterns': ['successful_withdrawal'],
                'critical_events': ['notes_issue: PRESENTED', 'notes_issue: TAKEN']
            }
            return True
        
        if self.is_successful_inquiry(session.raw_text, events):
            logger.info(f"Expert override applied for {session.session_id}: Successful inquiry detected")
            session.extracted_details = {
                'expert_override': True,
                'override_reason': 'Card inserted, transaction completed, card returned',
                'detected_patterns': ['successful_inquiry'],
                'critical_events': ['card_flow: INSERTED → TAKEN']
            }
            return True
        
        return False
    
    def _detect_specific_anomalies(self, session: TransactionSession, events: List[Dict]):
        """Detect specific types of anomalies using ML-first approach with minimal rule-based fallback"""
        
        # ML-First Anomaly Detection using Advanced Models
        ml_anomalies = self._detect_ml_anomalies(session)
        for anomaly in ml_anomalies:
            session.add_anomaly(**anomaly)
        
        # Only use rule-based detection for critical safety patterns (minimal set)
        self._detect_critical_safety_patterns(session)
    
    def _detect_ml_anomalies(self, session: TransactionSession) -> List[Dict]:
        """Advanced ML-based anomaly detection using multiple models"""
        anomalies = []
        
        try:
            # 1. BERT-based semantic anomaly detection
            semantic_anomalies = self._detect_semantic_anomalies(session)
            anomalies.extend(semantic_anomalies)
            
            # 2. Sequence-based anomaly detection using LSTM
            sequence_anomalies = self._detect_sequence_anomalies(session)
            anomalies.extend(sequence_anomalies)
            
            # 3. Statistical ensemble detection
            ensemble_anomalies = self._detect_ensemble_anomalies(session)
            anomalies.extend(ensemble_anomalies)
            
            # 4. Pattern clustering anomaly detection
            cluster_anomalies = self._detect_cluster_anomalies(session)
            anomalies.extend(cluster_anomalies)
            
        except Exception as e:
            logger.warning(f"ML anomaly detection failed for session {session.session_id}: {str(e)}")
            # Fallback to minimal rule-based detection only for critical cases
        
        return anomalies
    
    def _detect_critical_safety_patterns(self, session: TransactionSession):
        """Minimal rule-based detection only for critical safety patterns"""
        text = session.raw_text.upper()
        
        # Only detect truly critical patterns that require immediate attention
        critical_patterns = {
            "hardware_failure": {
                "patterns": ['HARDWARE FAULT', 'SYSTEM FAILURE', 'CRITICAL ERROR'],
                "confidence": 0.98,
                "severity": "critical"
            },
            "security_breach": {
                "patterns": ['UNAUTHORIZED ACCESS', 'SECURITY VIOLATION', 'TAMPER DETECTED'],
                "confidence": 0.99,
                "severity": "critical"
            }
        }
        
        for anomaly_type, config in critical_patterns.items():
            if any(pattern in text for pattern in config["patterns"]):
                session.add_anomaly(
                    anomaly_type=anomaly_type,
                    confidence=config["confidence"],
                    detection_method="critical_safety_rule",
                    description=f"Critical safety pattern detected: {anomaly_type}",
                    severity=config["severity"],
                    details={"matched_patterns": [p for p in config["patterns"] if p in text]}
                )
    
    def _detect_semantic_anomalies(self, session: TransactionSession) -> List[Dict]:
        """BERT-based semantic anomaly detection"""
        anomalies = []
        
        try:
            # Generate embedding for this session
            session_embedding = self._generate_single_embedding(session.raw_text)
            
            # Compare against learned normal patterns
            if hasattr(self, 'normal_embeddings_cluster'):
                semantic_score = self._calculate_semantic_distance(session_embedding, self.normal_embeddings_cluster)
                
                if semantic_score > 0.75:  # Threshold for semantic anomaly
                    anomalies.append({
                        "anomaly_type": "semantic_anomaly",
                        "confidence": semantic_score,
                        "detection_method": "bert_semantic",
                        "description": "Transaction semantically differs from normal patterns",
                        "severity": self._determine_severity(semantic_score),
                        "details": {"semantic_distance": semantic_score}
                    })
        except Exception as e:
            logger.warning(f"Semantic anomaly detection failed: {str(e)}")
        
        return anomalies
    
    def _detect_sequence_anomalies(self, session: TransactionSession) -> List[Dict]:
        """LSTM-based sequence anomaly detection"""
        anomalies = []
        
        try:
            # Extract event sequence from session
            events_sequence = self._extract_event_sequence(session.raw_text)
            
            if hasattr(self, 'sequence_model') and len(events_sequence) > 3:
                # Predict next events and calculate anomaly score
                sequence_score = self._calculate_sequence_anomaly_score(events_sequence)
                
                if sequence_score > 0.7:  # Threshold for sequence anomaly
                    anomalies.append({
                        "anomaly_type": "sequence_anomaly", 
                        "confidence": sequence_score,
                        "detection_method": "lstm_sequence",
                        "description": "Transaction event sequence is unusual",
                        "severity": self._determine_severity(sequence_score),
                        "details": {"sequence_score": sequence_score, "events": events_sequence}
                    })
        except Exception as e:
            logger.warning(f"Sequence anomaly detection failed: {str(e)}")
        
        return anomalies
    
    def _detect_ensemble_anomalies(self, session: TransactionSession) -> List[Dict]:
        """Ensemble-based anomaly detection using multiple ML models"""
        anomalies = []
        
        try:
            # Get session features
            features = self._extract_ml_features(session)
            
            # Apply multiple models and combine predictions
            model_scores = {}
            
            if hasattr(self, 'autoencoder_model'):
                model_scores['autoencoder'] = self._autoencoder_anomaly_score(features)
            
            if hasattr(self, 'dbscan_model'):
                model_scores['clustering'] = self._clustering_anomaly_score(features)
            
            if hasattr(self, 'local_outlier_factor'):
                model_scores['lof'] = self._lof_anomaly_score(features)
            
            # Ensemble voting
            if model_scores:
                ensemble_score = np.mean(list(model_scores.values()))
                voting_threshold = 0.6
                
                if ensemble_score > voting_threshold:
                    anomalies.append({
                        "anomaly_type": "ensemble_anomaly",
                        "confidence": ensemble_score,
                        "detection_method": "ml_ensemble",
                        "description": f"Multiple ML models indicate anomaly (consensus: {len(model_scores)} models)",
                        "severity": self._determine_severity(ensemble_score),
                        "details": {"model_scores": model_scores, "ensemble_score": ensemble_score}
                    })
        except Exception as e:
            logger.warning(f"Ensemble anomaly detection failed: {str(e)}")
        
        return anomalies
    
    def _detect_cluster_anomalies(self, session: TransactionSession) -> List[Dict]:
        """Clustering-based anomaly detection"""
        anomalies = []
        
        try:
            # Generate embedding for this session
            session_embedding = self._generate_single_embedding(session.raw_text)
            
            # Check distance to nearest cluster centers
            if hasattr(self, 'cluster_centers'):
                min_distance = float('inf')
                nearest_cluster = -1
                
                for i, center in enumerate(self.cluster_centers):
                    distance = np.linalg.norm(session_embedding - center)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_cluster = i
                
                # If distance is too large, it's an anomaly
                cluster_threshold = getattr(self, 'cluster_threshold', 2.0)
                if min_distance > cluster_threshold:
                    anomalies.append({
                        "anomaly_type": "cluster_outlier",
                        "confidence": min(0.95, min_distance / cluster_threshold * 0.5),
                        "detection_method": "clustering",
                        "description": f"Transaction doesn't fit any known cluster (distance: {min_distance:.2f})",
                        "severity": self._determine_severity(min_distance / cluster_threshold * 0.5),
                        "details": {"cluster_distance": min_distance, "nearest_cluster": nearest_cluster}
                    })
        except Exception as e:
            logger.warning(f"Cluster anomaly detection failed: {str(e)}")
        
        return anomalies
    
    # Helper methods for ML-based anomaly detection
    def _generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text session"""
        try:
            from sentence_transformers import SentenceTransformer
            if not hasattr(self, '_embedding_model'):
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Clean and prepare text
            cleaned_text = re.sub(r'\s+', ' ', text).strip()
            if len(cleaned_text) > 512:
                cleaned_text = cleaned_text[:512]  # Truncate for performance
            
            embedding = self._embedding_model.encode([cleaned_text])[0]
            return embedding
        except Exception as e:
            logger.warning(f"Embedding generation failed: {str(e)}")
            # Fallback to simple feature vector
            return self._create_simple_feature_vector(text)
    
    def _create_simple_feature_vector(self, text: str) -> np.ndarray:
        """Create a simple feature vector as fallback"""
        features = [
            len(text),
            text.count('CARD'),
            text.count('PIN'),
            text.count('NOTES'),
            text.count('ERROR'),
            text.count('TIMEOUT'),
            text.count('SUPERVISOR'),
            len(re.findall(r'\d+', text)),
            text.count('\n')
        ]
        return np.array(features, dtype=np.float32)
    
    def _calculate_semantic_distance(self, embedding: np.ndarray, cluster_center: np.ndarray) -> float:
        """Calculate semantic distance between embedding and cluster center"""
        try:
            distance = np.linalg.norm(embedding - cluster_center)
            # Normalize to 0-1 range
            return min(1.0, distance / 2.0)
        except:
            return 0.0
    
    def _extract_event_sequence(self, text: str) -> List[str]:
        """Extract sequence of events from session text"""
        events = []
        event_patterns = {
            'CARD_INSERT': r'CARD INSERTED',
            'PIN_ENTRY': r'PIN ENTERED',
            'TRANSACTION_START': r'TRANSACTION START',
            'NOTES_PRESENT': r'NOTES PRESENTED',
            'NOTES_TAKEN': r'NOTES TAKEN',
            'CARD_TAKEN': r'CARD TAKEN',
            'ERROR': r'ERROR|FAULT|FAIL',
            'TIMEOUT': r'TIMEOUT',
            'TRANSACTION_END': r'TRANSACTION END'
        }
        
        for event_name, pattern in event_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                events.append(event_name)
        
        return events
    
    def _calculate_sequence_anomaly_score(self, events_sequence: List[str]) -> float:
        """Calculate anomaly score based on event sequence"""
        # Simple sequence analysis - can be enhanced with LSTM
        normal_sequences = [
            ['CARD_INSERT', 'PIN_ENTRY', 'NOTES_PRESENT', 'NOTES_TAKEN', 'CARD_TAKEN'],
            ['CARD_INSERT', 'PIN_ENTRY', 'CARD_TAKEN'],
            ['TRANSACTION_START', 'CARD_INSERT', 'PIN_ENTRY', 'TRANSACTION_END']
        ]
        
        # Check similarity to normal sequences
        max_similarity = 0.0
        for normal_seq in normal_sequences:
            similarity = self._sequence_similarity(events_sequence, normal_seq)
            max_similarity = max(max_similarity, similarity)
        
        # Return anomaly score (1 - similarity)
        return 1.0 - max_similarity
    
    def _sequence_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """Calculate similarity between two sequences"""
        if not seq1 or not seq2:
            return 0.0
        
        # Simple Jaccard similarity
        set1, set2 = set(seq1), set(seq2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_ml_features(self, session: TransactionSession) -> np.ndarray:
        """Extract numerical features for ML models"""
        text = session.raw_text
        
        features = [
            # Basic text statistics
            len(text),
            len(text.split('\n')),
            len(text.split()),
            
            # Event counts
            text.count('CARD'),
            text.count('PIN'),
            text.count('NOTES'),
            text.count('ERROR'),
            text.count('TIMEOUT'),
            text.count('SUPERVISOR'),
            text.count('FAULT'),
            text.count('FAIL'),
            
            # Timing indicators
            len(re.findall(r'\d{2}:\d{2}:\d{2}', text)),
            
            # Transaction indicators
            len(re.findall(r'TRANSACTION', text)),
            len(re.findall(r'OPCODE', text)),
            
            # Error indicators
            len(re.findall(r'ESC:\s*\d+', text)),
            len(re.findall(r'VAL:\s*\d+', text)),
            
            # Session characteristics
            session.session_length if hasattr(session, 'session_length') else len(text),
            session.overall_anomaly_score if hasattr(session, 'overall_anomaly_score') else 0.0
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _autoencoder_anomaly_score(self, features: np.ndarray) -> float:
        """Calculate anomaly score using autoencoder (placeholder)"""
        # This would use a trained autoencoder model
        # For now, return a simple statistical measure
        mean_val = np.mean(features)
        std_val = np.std(features)
        z_score = abs(mean_val - 50) / (std_val + 1e-8)  # Arbitrary baseline
        return min(1.0, z_score / 3.0)
    
    def _clustering_anomaly_score(self, features: np.ndarray) -> float:
        """Calculate anomaly score using clustering"""
        # Simple distance-based scoring
        # In practice, this would use DBSCAN or similar
        feature_norm = np.linalg.norm(features)
        baseline_norm = 100.0  # Arbitrary baseline
        return min(1.0, abs(feature_norm - baseline_norm) / baseline_norm)
    
    def _lof_anomaly_score(self, features: np.ndarray) -> float:
        """Calculate Local Outlier Factor score"""
        # Simplified LOF calculation
        # In practice, use sklearn's LocalOutlierFactor
        feature_sum = np.sum(features)
        baseline_sum = 500.0  # Arbitrary baseline
        return min(1.0, abs(feature_sum - baseline_sum) / baseline_sum)
        
        # Security Issues
        if 'SUPERVISOR MODE' in text and 'UNAUTHORIZED' in text:
            session.add_anomaly(
                anomaly_type="unauthorized_access",
                confidence=0.98,
                detection_method="expert_rule",
                description="Unauthorized supervisor mode access attempt",
                severity="critical",
                details={"security_event": "unauthorized_supervisor"}
            )
        
        # Excessive Supervisor Mode Entries (NEW DETECTION)
        supervisor_entries = len(re.findall(r'SUPERVISOR MODE ENTRY', text, re.IGNORECASE))
        if supervisor_entries > 5:
            session.add_anomaly(
                anomaly_type="excessive_supervisor_mode",
                confidence=min(0.95, 0.5 + (supervisor_entries / 20.0)),
                detection_method="expert_rule",
                description=f"Excessive supervisor mode entries: {supervisor_entries} times",
                severity="high" if supervisor_entries > 10 else "medium",
                details={"supervisor_entries": supervisor_entries, "threshold": 5}
            )
        
        # Excessive Diagnostic Messages (NEW DETECTION)
        diagnostic_patterns = len(re.findall(r'\[000p', text, re.IGNORECASE))
        if diagnostic_patterns > 50:
            session.add_anomaly(
                anomaly_type="excessive_diagnostics",
                severity="high" if diagnostic_patterns > 100 else "medium",
                details={"diagnostic_patterns": diagnostic_patterns, "threshold": 50}
            )
        
        # Repetitive Pattern Loops (NEW DETECTION)
        lines = session.raw_text.split('\n')
        if len(lines) > 500:  # Large sessions that might have repetitive loops
            # Count unique vs total lines (excluding timestamps)
            non_timestamp_lines = [line for line in lines if not re.match(r'^\*\d+\*\d{2}/\d{2}/\d{4}\*', line)]
            if non_timestamp_lines:
                unique_lines = len(set(non_timestamp_lines))
                total_lines = len(non_timestamp_lines)
                repetition_ratio = (total_lines - unique_lines) / total_lines
                
                if repetition_ratio > 0.8:  # More than 80% repetitive content
                    session.add_anomaly(
                        anomaly_type="repetitive_pattern_loop",
                        confidence=min(0.95, repetition_ratio),
                        detection_method="expert_rule",
                        description=f"High repetition ratio: {repetition_ratio:.2f} ({total_lines} lines, {unique_lines} unique)",
                        severity="high",
                        details={
                            "repetition_ratio": repetition_ratio,
                            "total_lines": total_lines,
                            "unique_lines": unique_lines
                        }
                    )
        
        # Card Issues
        if 'CARD RETAINED' in text or 'CARD CAPTURED' in text:
            session.add_anomaly(
                anomaly_type="card_retained",
                confidence=0.85,
                detection_method="expert_rule",
                description="Card was retained by the ATM",
                severity="medium",
                details={"retention_reason": "security_or_malfunction"}
            )
        
        # Cash Handling Issues
        if 'CASH CASSETTE' in text and any(issue in text for issue in ['EMPTY', 'FAULT', 'ERROR']):
            session.add_anomaly(
                anomaly_type="cash_handling_issue",
                confidence=0.88,
                detection_method="expert_rule",
                description="Cash cassette related problem",
                severity="high",
                details={"cassette_issues": [issue for issue in ['EMPTY', 'FAULT', 'ERROR'] if issue in text]}
            )
        
        # Transaction Timeout Issues
        if 'TIMEOUT' in text or 'NO RESPONSE' in text:
            session.add_anomaly(
                anomaly_type="timeout_error",
                confidence=0.75,
                detection_method="expert_rule",
                description="Transaction timeout or communication error",
                severity="medium",
                details={"timeout_indicators": ["TIMEOUT", "NO RESPONSE"]}
            )
        
        # System Reset/Recovery
        if 'SYSTEM RESET' in text or 'POWER CYCLE' in text:
            session.add_anomaly(
                anomaly_type="system_reset",
                confidence=0.70,
                detection_method="expert_rule",
                description="System reset or power cycle occurred",
                severity="medium",
                details={"reset_type": "system_recovery"}
            )
        
        # Error Codes (ESC, VAL, REF, REJECTS)
        error_code_pattern = re.compile(r'(ESC|VAL|REF|REJECTS):\s*(\d+)', re.IGNORECASE)
        error_matches = error_code_pattern.findall(text)
        if error_matches:
            session.add_anomaly(
                anomaly_type="error_codes",
                confidence=0.85,
                detection_method="expert_rule",
                description="Device error codes detected",
                severity="high",
                details={"error_codes": [f"{code}: {value}" for code, value in error_matches]}
            )
        
        # Host Transaction Declines - "UNABLE TO PROCESS"
        if 'UNABLE TO PROCESS' in text:
            # Determine decline context for better categorization
            decline_context = self._analyze_unable_to_process_context(text, events)
            
            session.add_anomaly(
                anomaly_type="host_decline",
                confidence=0.85,  # High confidence as this is a definitive host response
                detection_method="expert_rule", 
                description=f"Host declined transaction: {decline_context['reason']}",
                severity="medium",  # Host declines indicate potential issues that need monitoring
                details={
                    "decline_type": "unable_to_process",
                    "context": decline_context,
                    "customer_initiated": True,
                    "system_fault": False
                }
            )
    
    def _detect_incomplete_transactions(self, session: TransactionSession, events: List[str], text: str):
        """Enhanced detection for incomplete or failed transactions that should be flagged as anomalies"""
        
        # Pattern 1: Card inserted and immediately taken without PIN (suspicious - like txn1)
        if ("CARD_INSERTED" in events and "CARD_TAKEN" in events and 
            "PIN_ENTERED" not in events and
            not re.search(r'AUTHORIZATION', text, re.IGNORECASE) and
            not re.search(r'BALANCE.*\d+', text, re.IGNORECASE)):
            
            session.add_anomaly(
                anomaly_type="incomplete_transaction",
                confidence=0.95,  # Increased confidence
                detection_method="expert_rule",
                description="Card inserted and immediately removed without PIN entry or transaction completion",
                severity="high",
                details={
                    "pattern": "card_inserted_no_pin",
                    "indicators": ["CARD_INSERTED", "CARD_TAKEN", "NO_PIN_ENTERED"],
                    "user_example": "txn1_pattern"
                }
            )
        
        # Pattern 2: PIN entered but transaction incomplete (no authorization, account info, or completion - like txn2)
        elif ("CARD_INSERTED" in events and "PIN_ENTERED" in events and "CARD_TAKEN" in events and
              not any(indicator in text.upper() for indicator in [
                  'AUTHORIZATION', 'ACCOUNT', 'BALANCE', 'WITHDRAWAL', 'DEPOSIT', 
                  'NOTES STACKED', 'NOTES PRESENTED', 'RECEIPT PRINTED', 'CASH DISPENSED'
              ])):
            
            session.add_anomaly(
                anomaly_type="incomplete_transaction", 
                confidence=0.90,  # Increased confidence
                detection_method="expert_rule",
                description="PIN entered but transaction failed to complete normally",
                severity="high",
                details={
                    "pattern": "pin_entered_incomplete", 
                    "indicators": ["CARD_INSERTED", "PIN_ENTERED", "CARD_TAKEN", "NO_COMPLETION"],
                    "user_example": "txn2_pattern"
                }
            )
        
        # Pattern 3: OPCODE operations started but not completed (enhanced for txn2 case)
        elif (re.search(r'OPCODE\s*=\s*(FI|BC|WD|IN)', text, re.IGNORECASE) and
              "PIN_ENTERED" in events and "CARD_TAKEN" in events and
              not re.search(r'(NOTES|CASH|WITHDRAWAL.*COMPLETE|BALANCE.*\d+|DISPENSE)', text, re.IGNORECASE)):
            
            session.add_anomaly(
                anomaly_type="incomplete_transaction",
                confidence=0.88,
                detection_method="expert_rule",
                description="OPCODE operations initiated but transaction not completed",
                severity="high",
                details={
                    "pattern": "opcode_incomplete",
                    "indicators": ["OPCODE_OPERATIONS", "PIN_ENTERED", "CARD_TAKEN", "NO_COMPLETION"],
                    "user_example": "txn2_opcode_pattern"
                }
            )
        
        # Pattern 4: Very short sessions with transaction boundaries but no meaningful activity
        elif ("TRANSACTION START" in text and "TRANSACTION END" in text and
              len(text.strip()) < 300 and  # Very short session
              "CARD_TAKEN" in events and
              not any(activity in text.upper() for activity in [
                  'NOTES', 'BALANCE', 'WITHDRAWAL', 'DEPOSIT', 'RECEIPT', 'AUTHORIZATION'
              ])):
            
            session.add_anomaly(
                anomaly_type="incomplete_transaction",
                confidence=0.80,
                detection_method="expert_rule", 
                description="Very short transaction session with no meaningful activity",
                severity="medium",
                details={
                    "pattern": "short_session_no_activity",
                    "session_length": len(text.strip()),
                    "indicators": ["TRANSACTION_START", "TRANSACTION_END", "CARD_TAKEN", "NO_ACTIVITY"]
                }
            )
        
        # Pattern 5: Enhanced detection for specific user examples
        # Direct text pattern matching for cases like the provided examples
        text_upper = text.upper()
        
        # Check for txn1-like pattern: CARD INSERTED followed by CARD TAKEN quickly without PIN
        if (re.search(r'CARD\s+INSERTED', text_upper) and 
            re.search(r'CARD\s+TAKEN', text_upper) and
            not re.search(r'PIN\s+ENTERED', text_upper) and
            not re.search(r'OPCODE', text_upper)):
            
            session.add_anomaly(
                anomaly_type="incomplete_transaction",
                confidence=0.95,
                detection_method="expert_rule",
                description="Card inserted and taken without PIN entry - possible card skimming or customer abandonment",
                severity="high",
                details={
                    "pattern": "card_inserted_taken_no_pin_direct",
                    "indicators": ["CARD_INSERTED", "CARD_TAKEN", "NO_PIN", "NO_OPCODE"],
                    "detection_method": "direct_text_pattern"
                }
            )
        
        # Check for txn2-like pattern: PIN + OPCODE but no transaction completion
        elif (re.search(r'PIN\s+ENTERED', text_upper) and 
              re.search(r'OPCODE\s*=', text_upper) and
              re.search(r'CARD\s+TAKEN', text_upper) and
              not any(re.search(pattern, text_upper) for pattern in [
                  r'NOTES\s+PRESENTED', r'RECEIPT\s+PRINTED', r'CASH\s+DISPENSED',
                  r'BALANCE\s+\d+', r'WITHDRAWAL\s+COMPLETE', r'TRANSACTION\s+COMPLETE'
              ])):
            
            session.add_anomaly(
                anomaly_type="incomplete_transaction",
                confidence=0.92,
                detection_method="expert_rule",
                description="PIN entered and OPCODE operations initiated but transaction not completed",
                severity="high",
                details={
                    "pattern": "pin_opcode_incomplete_direct",
                    "indicators": ["PIN_ENTERED", "OPCODE_OPERATIONS", "CARD_TAKEN", "NO_COMPLETION"],
                    "detection_method": "direct_text_pattern"
                }
            )
            
            session.add_anomaly(
                anomaly_type="incomplete_transaction",
                confidence=0.88,
                detection_method="expert_rule",
                description="Transaction operation initiated but not completed",
                severity="high",
                details={
                    "pattern": "opcode_initiated_incomplete",
                    "opcode_found": re.search(r'OPCODE\s*=\s*(\w+)', text, re.IGNORECASE).group(1) if re.search(r'OPCODE\s*=\s*(\w+)', text, re.IGNORECASE) else "unknown",
                    "indicators": ["OPCODE_OPERATION", "PIN_ENTERED", "CARD_TAKEN", "NO_COMPLETION"]
                }
            )
            
            session.add_anomaly(
                anomaly_type="abnormal_termination",
                confidence=0.80, 
                detection_method="expert_rule",
                description="Transaction started but terminated abnormally without proper completion",
                severity="medium",
                details={
                    "pattern": "abnormal_termination",
                    "indicators": ["TRANSACTION_START", "CARD_TAKEN", "NO_PROPER_END"]
                }
            )
    
    def _detect_machine_status_anomalies(self, session: TransactionSession, text: str):
        """Detect anomalies based on Machine Status codes in transaction logs
        
        Pattern: *<TransactionNo>*<DeviceID>*<StatusType>*<ErrorCode>,M-<ModuleCode>,R-<RetryCount>
        Example: *7252*1*D*3,M-02,R-10011 (Module Code 02 indicates error)
        Example: *7258*1*D*9,M-81,R-0 (Module Code 81 is ignorable - chip read failure)
        """
        
        # Regex pattern to extract machine status codes
        # Pattern: *TransactionNo*DeviceID*StatusType*ErrorCode*SubCode,M-ModuleCode,R-RetryCount
        machine_status_pattern = re.compile(
            r'\*(\d+)\*(\d+)\*([A-Z]*)\(?(\d*)\*([^,]*),M-([^,]+),R-(\d+)',
            re.IGNORECASE
        )
        
        machine_status_matches = machine_status_pattern.findall(text)
        
        if not machine_status_matches:
            return  # No machine status codes found
        
        # Analyze each machine status code
        error_modules = []
        warning_modules = []
        ignored_modules = []
        high_retry_counts = []
        
        # Define module code classifications
        error_module_codes = {
            '02': 'Communication Error',
            '03': 'Hardware Fault', 
            '04': 'Cash Dispenser Error',
            '05': 'Card Reader Error',
            '06': 'Receipt Printer Error',
            '07': 'Cash Cassette Error',
            '08': 'Security Module Error',
            '09': 'Pin Pad Error',
            '10': 'Display Error',
            '11': 'Network Communication Error',
            '12': 'Transaction Processing Error'
        }
        
        warning_module_codes = {
            '01': 'Minor Warning',
            '20': 'Maintenance Required',
            '21': 'Low Cash Warning',
            '22': 'Paper Low Warning'
        }
        
        # Module codes to ignore (known non-critical issues)
        ignored_module_codes = {
            '81': 'Chip Read Failure (Normal)',
            '00': 'Status OK',
            '090B0210B9': 'Diagnostic Status'
        }
        
        for match in machine_status_matches:
            # Handle the 7-element tuple from our regex
            if len(match) >= 7:
                trans_no, device_id, status_type, error_code, sub_code, module_code, retry_count = match
            else:
                continue  # Skip malformed matches
                
            retry_count_int = int(retry_count) if retry_count.isdigit() else 0
            
            # Check for high retry counts (indicates persistent issues)
            if retry_count_int > 5:
                high_retry_counts.append({
                    'transaction': trans_no,
                    'module_code': module_code,
                    'retry_count': retry_count_int,
                    'error_code': error_code
                })
            
            # Classify module codes
            if module_code in error_module_codes:
                error_modules.append({
                    'transaction': trans_no,
                    'module_code': module_code,
                    'description': error_module_codes[module_code],
                    'error_code': error_code,
                    'retry_count': retry_count_int
                })
            elif module_code in warning_module_codes:
                warning_modules.append({
                    'transaction': trans_no,
                    'module_code': module_code,
                    'description': warning_module_codes[module_code],
                    'error_code': error_code,
                    'retry_count': retry_count_int
                })
            elif module_code in ignored_module_codes:
                ignored_modules.append({
                    'transaction': trans_no,
                    'module_code': module_code,
                    'description': ignored_module_codes[module_code]
                })
            else:
                # Unknown module code - treat as potential error
                error_modules.append({
                    'transaction': trans_no,
                    'module_code': module_code,
                    'description': f'Unknown Module Code: {module_code}',
                    'error_code': error_code,
                    'retry_count': retry_count_int
                })
        
        # Generate anomalies based on findings
        if error_modules:
            # Group errors by module type for better reporting
            module_counts = {}
            for error in error_modules:
                module_type = error['description']
                module_counts[module_type] = module_counts.get(module_type, 0) + 1
            
            session.add_anomaly(
                anomaly_type="machine_status_error",
                confidence=min(0.95, 0.7 + (len(error_modules) * 0.05)),
                detection_method="machine_status_analysis",
                description=f"Machine status errors detected: {', '.join([f'{desc}({count})' for desc, count in module_counts.items()])}",
                severity="high" if len(error_modules) > 3 else "medium",
                details={
                    "error_modules": error_modules,
                    "error_count": len(error_modules),
                    "module_breakdown": module_counts,
                    "total_status_codes": len(machine_status_matches)
                }
            )
        
        if high_retry_counts:
            session.add_anomaly(
                anomaly_type="high_retry_count",
                confidence=min(0.90, 0.6 + (max(r['retry_count'] for r in high_retry_counts) * 0.02)),
                detection_method="machine_status_analysis", 
                description=f"High retry counts detected: max {max(r['retry_count'] for r in high_retry_counts)} retries",
                severity="medium",
                details={
                    "high_retry_operations": high_retry_counts,
                    "max_retry_count": max(r['retry_count'] for r in high_retry_counts),
                    "operations_with_high_retries": len(high_retry_counts)
                }
            )
        
        if warning_modules:
            session.add_anomaly(
                anomaly_type="machine_status_warning",
                confidence=0.60,
                detection_method="machine_status_analysis",
                description=f"Machine status warnings: {len(warning_modules)} warning codes detected",
                severity="low",
                details={
                    "warning_modules": warning_modules,
                    "warning_count": len(warning_modules)
                }
            )
        
        # Log ignored modules for debugging (but don't create anomalies)
        if ignored_modules:
            logger.debug(f"Session {session.session_id}: Ignored {len(ignored_modules)} non-critical status codes")

    def _determine_severity(self, confidence: float) -> str:
        """Determine severity based on confidence score"""
        if confidence >= 0.9:
            return "critical"
        elif confidence >= 0.75:
            return "high"
        elif confidence >= 0.5:
            return "medium"
        else:
            return "low"
    
    def _analyze_unable_to_process_context(self, text: str, events: List[str]) -> Dict[str, Any]:
        """Analyze the context of 'UNABLE TO PROCESS' to categorize the host decline reason"""
        
        context = {
            "reason": "host_decline",
            "transaction_stage": "unknown",
            "likely_cause": "host_system_decline",
            "had_authentication": False,
            "had_account_access": False,
            "system_errors_present": False,
            "decline_category": "unknown"
        }
        
        # Check transaction stage when decline occurred
        if re.search(r'(OPCODE\s*=\s*(FI|BC|WD|IN))', text, re.IGNORECASE):
            context["transaction_stage"] = "after_transaction_type_selection"
            
        if re.search(r'(PIN ENTERED|GENAC|EXTERNAL AUTHENTICATE)', text, re.IGNORECASE):
            context["had_authentication"] = True
            context["transaction_stage"] = "after_authentication"
            
        if re.search(r'(PAN \d{4}\*+\d{4}|ACCOUNT|BALANCE)', text, re.IGNORECASE):
            context["had_account_access"] = True 
            context["transaction_stage"] = "after_account_access"
            
        # Check for system errors that might have caused the decline
        if any(error in text.upper() for error in ['HARDWARE ERROR', 'DEVICE ERROR', 'COMMUNICATION ERROR']):
            context["system_errors_present"] = True
            context["likely_cause"] = "system_error_decline"
            context["decline_category"] = "technical_failure"
            
        # Check for specific decline reasons in the text
        if re.search(r'(INSUFFICIENT|BALANCE|FUNDS)', text, re.IGNORECASE):
            context["decline_category"] = "insufficient_funds"
            context["likely_cause"] = "account_balance_insufficient"
        elif re.search(r'(INVALID|EXPIRED|BLOCKED)', text, re.IGNORECASE):
            context["decline_category"] = "card_issue"
            context["likely_cause"] = "card_validation_failed"
        elif re.search(r'(TIMEOUT|TIME.*OUT)', text, re.IGNORECASE):
            context["decline_category"] = "timeout"
            context["likely_cause"] = "host_timeout"
        elif re.search(r'(LIMIT|EXCEED)', text, re.IGNORECASE):
            context["decline_category"] = "limit_exceeded"
            context["likely_cause"] = "transaction_limit_exceeded"
            
        # Determine likely reason based on context
        if context["had_account_access"] and not context["system_errors_present"]:
            if context["decline_category"] == "unknown":
                context["likely_cause"] = "host_business_rule_decline"
                context["decline_category"] = "business_rule"
        elif context["had_authentication"] and not context["system_errors_present"]:
            context["likely_cause"] = "host_authorization_decline"
            context["decline_category"] = "authorization_failure"
        elif not context["had_authentication"]:
            context["likely_cause"] = "host_early_decline"
            context["decline_category"] = "pre_authorization"
            
        return context
    
    def apply_expert_override(self, session: TransactionSession) -> bool:
        """
        Apply expert knowledge to override ML predictions and prevent false positives
        Returns True if the session should be considered normal (override anomaly detection)
        """
        events = self.extract_key_events(session.raw_text)
        
        # Check for definitive normal patterns first
        if self.is_successful_withdrawal(session.raw_text, events):
            logger.info(f"Expert override applied for {session.session_id}: Successful withdrawal detected")
            session.anomaly_type = "normal_withdrawal"
            session.extracted_details = {
                'expert_override': True,
                'override_reason': 'NOTES PRESENTED followed by NOTES TAKEN indicates successful cash collection',
                'detected_patterns': ['successful_withdrawal'],
                'critical_events': ['notes_issue: PRESENTED', 'notes_issue: TAKEN']
            }
            return True
        
        if self.is_successful_inquiry(session.raw_text, events):
            logger.info(f"Expert override applied for {session.session_id}: Successful inquiry detected")
            session.anomaly_type = "normal_inquiry"
            session.extracted_details = {
                'expert_override': True,
                'override_reason': 'Card inserted, transaction completed, card returned',
                'detected_patterns': ['successful_inquiry'],
                'critical_events': ['card_flow: INSERTED → TAKEN']
            }
            return True
        
        # Check for genuine anomalies that should NOT be overridden
        if self.has_genuine_anomaly(session.raw_text, events):
            logger.info(f"Genuine anomaly confirmed for {session.session_id}")
            return False
        
        # For unclear cases, apply conservative override if ML confidence is low
        if session.anomaly_score < 0.7:
            logger.info(f"Conservative expert override applied for {session.session_id}: Low ML confidence")
            session.extracted_details = {
                'expert_override': True,
                'override_reason': 'Low ML confidence and no clear anomaly indicators',
                'detected_patterns': ['unclear_pattern'],
                'critical_events': []
            }
            return True
        
        return False
    
    def extract_key_events(self, session_text: str) -> List[str]:
        """Extract key events from session text for expert analysis"""
        events = []
        
        event_patterns = {
            'CARD_INSERTED': r'CARD INSERTED',
            'PIN_ENTERED': r'PIN ENTERED', 
            'NOTES_PRESENTED': r'NOTES PRESENTED',
            'NOTES_TAKEN': r'NOTES TAKEN',
            'NOTES_STACKED': r'NOTES STACKED',
            'CARD_TAKEN': r'CARD TAKEN',
            'UNABLE_TO_DISPENSE': r'UNABLE TO DISPENSE',
            'DEVICE_ERROR': r'DEVICE ERROR',
            'TIMEOUT': r'TIMEOUT',
            'NOTES_RETRACTED': r'NOTES RETRACTED',
            'RECEIPT_PRINTED': r'RECEIPT PRINTED',
            'BALANCE_INQUIRY': r'BALANCE INQUIRY',
            'SUPERVISOR_MODE': r'SUPERVISOR MODE',
            'POWER_RESET': r'POWER-UP/RESET'
        }
        
        for event_name, pattern in event_patterns.items():
            if re.search(pattern, session_text, re.IGNORECASE):
                events.append(event_name)
        
        return events
    
    def is_successful_withdrawal(self, session_text: str, events: List[str]) -> bool:
        """Check if this is a successful withdrawal (NOTES PRESENTED + NOTES TAKEN)"""
        return ("NOTES_PRESENTED" in events and 
                "NOTES_TAKEN" in events and
                "UNABLE_TO_DISPENSE" not in events and
                "DEVICE_ERROR" not in events and
                "TIMEOUT" not in events)
    
    def is_successful_inquiry(self, session_text: str, events: List[str]) -> bool:
        """Check if this is a successful inquiry transaction"""
        # A successful inquiry should have:
        # 1. Card inserted and taken
        # 2. Some form of authentication or transaction activity
        # 3. No errors
        
        basic_card_flow = ("CARD_INSERTED" in events and "CARD_TAKEN" in events)
        no_errors = ("UNABLE_TO_DISPENSE" not in events and "DEVICE_ERROR" not in events)
        
        # Must have some indication of actual transaction processing
        has_transaction_activity = (
            "PIN_ENTERED" in events or
            "BALANCE_INQUIRY" in events or 
            "RECEIPT_PRINTED" in events or
            re.search(r'AUTHORIZATION', session_text, re.IGNORECASE) or
            re.search(r'ACCOUNT', session_text, re.IGNORECASE) or
            re.search(r'BALANCE.*\d+', session_text, re.IGNORECASE)
        )
        
        return basic_card_flow and no_errors and has_transaction_activity
    
    def has_genuine_anomaly(self, session_text: str, events: List[str]) -> bool:
        """Check for genuine anomaly indicators"""
        return ("UNABLE_TO_DISPENSE" in events or
                "DEVICE_ERROR" in events or
                ("NOTES_PRESENTED" in events and "TIMEOUT" in events) or
                ("NOTES_PRESENTED" in events and "NOTES_RETRACTED" in events) or
                re.search(r"HARDWARE\s+FAULT", session_text, re.IGNORECASE) or
                re.search(r"COMMUNICATION\s+ERROR", session_text, re.IGNORECASE))
    
    def apply_supervised_classification(self):
        """Step 5: Apply supervised classification if available"""
        if self.supervised_classifier is None:
            return
        
        logger.info("Applying supervised classification")
        
        # Use scaled embeddings
        embeddings_scaled = self.scaler.transform(self.embeddings_matrix)
        
        if hasattr(self.pca, 'components_'):
            embeddings_scaled = self.pca.transform(embeddings_scaled)
        
        # Get predictions
        predictions = self.supervised_classifier.predict(embeddings_scaled)
        probabilities = self.supervised_classifier.predict_proba(embeddings_scaled)
        
        # Update sessions
        for i, session in enumerate(self.sessions):
            if self.label_encoder:
                session.supervised_label = self.label_encoder.inverse_transform([predictions[i]])[0]
            else:
                session.supervised_label = str(predictions[i])
            
            session.supervised_confidence = probabilities[i].max()
            
            # Override unsupervised if supervised is confident
            if session.supervised_confidence > 0.8 and session.supervised_label != "normal":
                session.is_anomaly = True
                session.anomaly_type = session.supervised_label
    
    def extract_anomaly_explanations(self):
        """Step 6: Extract explanations for detected anomalies"""
        logger.info("Extracting anomaly explanations")
        
        for session in self.sessions:
            if session.is_anomaly:
                patterns_found = []
                critical_events = []
                
                # Check each pattern
                for pattern_name, pattern_regex in self.explanation_patterns.items():
                    matches = pattern_regex.findall(session.raw_text)
                    if matches:
                        patterns_found.append(pattern_name)
                        # Extract context around match
                        for match in matches[:3]:  # Limit to first 3 matches
                            match_str = str(match) if isinstance(match, tuple) else match
                            critical_events.append(f"{pattern_name}: {match_str}")
                
                # Additional analysis
                # Check for long delays
                if session.start_time and session.end_time:
                    duration = (session.end_time - session.start_time).total_seconds()
                    if duration > 300:  # 5 minutes
                        patterns_found.append('long_duration')
                        critical_events.append(f"Session duration: {duration:.0f} seconds")
                
                # Store extracted details
                session.extracted_details = {
                    'detected_patterns': patterns_found,
                    'critical_events': critical_events[:5]  # Limit to 5 events
                }
                
                # Set anomaly type if not already set
                if not session.anomaly_type:
                    if 'unable_to_dispense' in patterns_found:
                        session.anomaly_type = 'dispense_failure'
                    elif 'device_error' in patterns_found:
                        session.anomaly_type = 'hardware_error'
                    elif 'power_reset' in patterns_found:
                        session.anomaly_type = 'system_reset'
                    elif 'supervisor_mode' in patterns_found:
                        session.anomaly_type = 'supervisor_activity'
                    elif 'cash_retract' in patterns_found:
                        session.anomaly_type = 'cash_handling_issue'
                    else:
                        session.anomaly_type = 'unknown_anomaly'
    
    def perform_final_expert_validation(self):
        """Perform final expert validation to ensure no false positives"""
        logger.info("Performing final expert validation")
        
        normal_reclassified = 0
        
        for session in self.sessions:
            if session.is_anomaly:
                events = self.extract_key_events(session.raw_text)
                
                # Double-check for successful withdrawal pattern that might have been missed
                if self.is_successful_withdrawal(session.raw_text, events):
                    session.is_anomaly = False
                    session.anomaly_score = 0.0
                    session.anomaly_type = "normal_withdrawal"
                    
                    # Update or create extracted details
                    if not session.extracted_details:
                        session.extracted_details = {}
                    
                    session.extracted_details.update({
                        'final_expert_override': True,
                        'final_override_reason': 'NOTES PRESENTED + NOTES TAKEN pattern detected in final validation',
                        'validation_stage': 'final_expert_check'
                    })
                    
                    normal_reclassified += 1
                    logger.info(f"Final validation: Reclassified {session.session_id} as normal withdrawal")
        
        if normal_reclassified > 0:
            logger.info(f"Final expert validation prevented {normal_reclassified} additional false positives")
        else:
            logger.info("Final expert validation: No additional false positives detected")
    
    def create_results_dataframe(self) -> pd.DataFrame:
        """Step 8: Create results dataframe with multi-anomaly support and expert override information"""
        results = []
        
        for session in self.sessions:
            # Check if expert override was applied
            expert_override = False
            override_reason = None
            if session.extracted_details:
                expert_override = (session.extracted_details.get('expert_override', False) or 
                                 session.extracted_details.get('final_expert_override', False))
                override_reason = (session.extracted_details.get('override_reason') or 
                                 session.extracted_details.get('final_override_reason'))
            
            # Multi-anomaly information
            anomaly_types = session.get_anomaly_types()
            anomaly_count = len(session.anomalies)
            critical_anomalies = len(session.get_anomalies_by_severity("critical"))
            high_severity_anomalies = len(session.get_anomalies_by_severity("high"))
            
            # Detection methods summary
            detection_methods = list(set(anomaly.detection_method for anomaly in session.anomalies))
            
            # Calculate session length with validation
            session_length = 0
            if session.start_time and session.end_time:
                time_diff = (session.end_time - session.start_time).total_seconds()
                # Validate that session length is reasonable (between 0 and 24 hours)
                if 0 <= time_diff <= 86400:  # 24 hours in seconds
                    session_length = int(time_diff)
                else:
                    # If time difference is unreasonable, use character count as proxy
                    session_length = min(len(session.raw_text), 86400)  # Cap at 24 hours worth
            else:
                # If no timestamps available, use character count as proxy
                session_length = min(len(session.raw_text), 86400)  # Cap at 24 hours worth
            
            result = {
                'session_id': session.session_id,
                'start_time': session.start_time,
                'end_time': session.end_time,
                'session_length': session_length,
                
                # Legacy fields for backwards compatibility
                'is_anomaly': session.is_anomaly,
                'anomaly_score': session.anomaly_score,
                'anomaly_type': session.anomaly_type,
                
                # Multi-anomaly fields
                'anomaly_count': anomaly_count,
                'anomaly_types': anomaly_types,
                'max_severity': session.max_severity,
                'overall_anomaly_score': session.overall_anomaly_score,
                'critical_anomalies_count': critical_anomalies,
                'high_severity_anomalies_count': high_severity_anomalies,
                'detection_methods': detection_methods,
                'anomalies_detail': [
                    {
                        'type': anomaly.anomaly_type,
                        'confidence': anomaly.confidence,
                        'method': anomaly.detection_method,
                        'severity': anomaly.severity,
                        'description': anomaly.description
                    } for anomaly in session.anomalies
                ],
                
                # Supervised learning fields
                'supervised_label': session.supervised_label,
                'supervised_confidence': session.supervised_confidence,
                
                # Expert override fields
                'expert_override_applied': expert_override,
                'expert_override_reason': override_reason,
                'detected_patterns': session.extracted_details.get('detected_patterns', []) if session.extracted_details else [],
                'critical_events': session.extracted_details.get('critical_events', []) if session.extracted_details else []
            }
            results.append(result)
        
        return pd.DataFrame(results)
    
    # NEW: Continuous Learning and Feedback Loop System
    def initialize_feedback_system(self):
        """Initialize the continuous learning feedback system"""
        self.feedback_buffer = []  # Store expert corrections
        self.learning_threshold = 50  # Retrain after 50 corrections
        self.feedback_weights = {
            'expert_confirmed_anomaly': 2.0,
            'expert_confirmed_normal': 2.0,
            'expert_new_anomaly_type': 3.0,
            'false_positive_correction': 1.5,
            'false_negative_correction': 2.5
        }
        self.model_performance_history = []
        
    def collect_expert_feedback(self, session_id: str, expert_label: str, 
                               expert_confidence: float, feedback_type: str, 
                               expert_explanation: str = None) -> bool:
        """
        Collect expert feedback on predictions for continuous learning
        
        Args:
            session_id: ID of the session being corrected
            expert_label: Expert's classification ('anomaly', 'normal', or specific type)
            expert_confidence: Expert's confidence (0.0 to 1.0)
            feedback_type: Type of feedback ('confirmation', 'correction', 'new_discovery')
            expert_explanation: Optional explanation from expert
        """
        try:
            # Find the session
            session = next((s for s in self.sessions if s.session_id == session_id), None)
            if not session:
                logger.error(f"Session {session_id} not found for feedback")
                return False
            
            # Create feedback record
            feedback_record = {
                'timestamp': datetime.now(),
                'session_id': session_id,
                'original_ml_prediction': session.is_anomaly,
                'original_ml_score': session.anomaly_score,
                'original_ml_type': session.anomaly_type,
                'expert_label': expert_label,
                'expert_confidence': expert_confidence,
                'feedback_type': feedback_type,
                'expert_explanation': expert_explanation,
                'session_embedding': session.embedding.copy() if session.embedding is not None else None,
                'session_text_hash': hashlib.md5(session.raw_text.encode()).hexdigest()
            }
            
            # Calculate feedback weight
            weight_key = self._determine_feedback_weight_key(
                session.is_anomaly, expert_label, feedback_type
            )
            feedback_record['learning_weight'] = self.feedback_weights.get(weight_key, 1.0)
            
            # Add to feedback buffer
            self.feedback_buffer.append(feedback_record)
            
            # Update session with expert feedback
            session.expert_feedback = {
                'expert_label': expert_label,
                'expert_confidence': expert_confidence,
                'feedback_type': feedback_type,
                'correction_applied': True
            }
            
            logger.info(f"Expert feedback collected for session {session_id}: {expert_label} ({feedback_type})")
            
            # Check if we should trigger retraining
            if len(self.feedback_buffer) >= self.learning_threshold:
                logger.info("Feedback threshold reached. Triggering continuous retraining...")
                self.continuous_model_retraining()
            
            return True
            
        except Exception as e:
            logger.error(f"Error collecting expert feedback: {str(e)}")
            return False
    
    def _determine_feedback_weight(self, ml_prediction: bool, expert_label: str, feedback_type: str) -> float:
        """Determine the weight of the feedback based on its type and correctness"""
        base_weight = 1.0
        if feedback_type == 'confirmation':
            base_weight = 1.0
        elif feedback_type == 'correction':
            base_weight = 2.0
        elif feedback_type == 'new_discovery':
            base_weight = 3.0
        
        # Increase weight if expert confidence is high
        if expert_label != 'normal' and expert_label is not None:
            base_weight += 1.0
        
        return base_weight
    
    def _determine_feedback_weight_key(self, ml_prediction: bool, expert_label: str, feedback_type: str) -> str:
        """Determine the appropriate weight key for the feedback"""
        if feedback_type == 'confirmation':
            return 'expert_confirmed_anomaly' if expert_label != 'normal' else 'expert_confirmed_normal'
        elif feedback_type == 'correction':
            if ml_prediction and expert_label == 'normal':
                return 'false_positive_correction'
            elif not ml_prediction and expert_label != 'normal':
                return 'false_negative_correction'
        elif feedback_type == 'new_discovery':
            return 'expert_new_anomaly_type'
        
        return 'expert_confirmed_anomaly'
    
    def continuous_model_retraining(self):
        """
        Enhanced continuous retraining with ML-first approach
        This is the core feedback loop that makes the system learn from expert input
        """
        if len(self.feedback_buffer) < 5:  # Reduced threshold for more frequent learning
            logger.info(f"Insufficient feedback for retraining: {len(self.feedback_buffer)} samples (need 5+)")
            return
        
        logger.info(f"Starting ML-first continuous retraining with {len(self.feedback_buffer)} feedback samples")
        
        # Mark training start for monitoring
        mark_ml_training_start("continuous_learning_ml_first")
        training_start_time = time.time()
        
        try:
            # 1. Update embeddings model with expert feedback
            self._update_embeddings_model_with_feedback()
            
            # 2. Retrain clustering models with new data
            self._retrain_clustering_models()
            
            # 3. Update anomaly thresholds based on expert corrections
            self._update_anomaly_thresholds()
            
            # 4. Train sequence model with expert-labeled sequences
            self._retrain_sequence_model()
            
            # 5. Update ensemble weights based on expert feedback accuracy
            self._update_ensemble_weights()
            
            # 6. Create expert-informed normal patterns
            self._build_expert_normal_patterns()
            
            # Clear processed feedback
            processed_feedback = len(self.feedback_buffer)
            self.feedback_buffer.clear()
            
            training_duration = time.time() - training_start_time
            logger.info(f"ML-first continuous retraining completed in {training_duration:.2f}s")
            logger.info(f"Processed {processed_feedback} expert feedback samples")
            
            # Save updated models
            self._save_updated_models()
            
            # Log training success
            mark_ml_training_complete("continuous_learning_ml_first", 
                                    feedback_samples=processed_feedback,
                                    training_duration=training_duration)
            
        except Exception as e:
            logger.error(f"Continuous retraining failed: {str(e)}")
            mark_ml_training_error("continuous_learning_ml_first", str(e))
            
            # 3. Retrain supervised classifier if enough labeled data
            self._retrain_supervised_with_feedback()
            
            # 4. Update expert rules based on new patterns
            self._update_expert_rules_from_feedback()
            
            # 5. Evaluate performance improvement
            performance_improvement = self._evaluate_feedback_performance()
            
            # 6. Save updated models
            self.save_models("/app/models/continuous_learning")
            
            # 7. Archive feedback buffer and track performance
            self._archive_feedback_buffer(performance_improvement)
            
            training_time = time.time() - training_start_time
            logger.info(f"Continuous retraining completed. Performance change: {performance_improvement:.3f}")
            
            # Mark training completion for monitoring
            mark_ml_training_complete(
                accuracy=performance_improvement,
                training_time=training_time,
                model_type="continuous_learning"
            )
            
        except Exception as e:
            logger.error(f"Error during continuous retraining: {str(e)}")
            mark_ml_error(f"Continuous retraining failed: {str(e)}", "continuous_model_retraining")
    
    def _retrain_isolation_forest_with_feedback(self):
        """Retrain Isolation Forest incorporating expert feedback"""
        # Get embeddings and weights from feedback
        feedback_embeddings = []
        feedback_weights = []
        
        for feedback in self.feedback_buffer:
            if feedback['session_embedding'] is not None:
                feedback_embeddings.append(feedback['session_embedding'])
                # Weight based on expert confidence and feedback type
                weight = feedback['learning_weight'] * feedback['expert_confidence']
                feedback_weights.append(weight)
        
        if len(feedback_embeddings) < 5:
            return
        
        feedback_embeddings = np.array(feedback_embeddings)
        feedback_weights = np.array(feedback_weights)
        
        # Scale embeddings
        feedback_scaled = self.scaler.transform(feedback_embeddings)
        if hasattr(self.pca, 'components_'):
            feedback_scaled = self.pca.transform(feedback_scaled)
        
        # Create a new Isolation Forest with adjusted parameters based on feedback
        expert_normal_ratio = sum(1 for f in self.feedback_buffer if f['expert_label'] == 'normal') / len(self.feedback_buffer)
        adjusted_contamination = max(0.01, min(0.3, 1.0 - expert_normal_ratio))
        
        # Retrain with original data + weighted feedback data
        original_data = self.scaler.transform(self.embeddings_matrix)
        if hasattr(self.pca, 'components_'):
            original_data = self.pca.transform(original_data)
        
        # Combine original and feedback data with weights
        combined_data = np.vstack([original_data, feedback_scaled])
        
        # Create new model with adjusted contamination
        self.isolation_forest = IsolationForest(
            contamination=adjusted_contamination,
            random_state=42,
            n_estimators=150  # Increase trees for better performance
        )
        
        self.isolation_forest.fit(combined_data)
        logger.info(f"Isolation Forest retrained with contamination={adjusted_contamination:.3f}")
    
    def _retrain_svm_with_feedback(self):
        """Retrain One-Class SVM with feedback-informed parameters"""
        # Adjust nu parameter based on expert feedback
        expert_anomaly_ratio = sum(1 for f in self.feedback_buffer if f['expert_label'] != 'normal') / len(self.feedback_buffer)
        adjusted_nu = max(0.01, min(0.2, expert_anomaly_ratio))
        
        # Get feedback data
        feedback_embeddings = [f['session_embedding'] for f in self.feedback_buffer if f['session_embedding'] is not None]
        
        if len(feedback_embeddings) < 5:
            return
        
        feedback_embeddings = np.array(feedback_embeddings)
        feedback_scaled = self.scaler.transform(feedback_embeddings)
        if hasattr(self.pca, 'components_'):
            feedback_scaled = self.pca.transform(feedback_scaled)
        
        # Combine with original data
        original_data = self.scaler.transform(self.embeddings_matrix)
        if hasattr(self.pca, 'components_'):
            original_data = self.pca.transform(original_data)
        
        combined_data = np.vstack([original_data, feedback_scaled])
        
        # Retrain SVM
        self.one_class_svm = OneClassSVM(
            kernel='rbf',
            gamma='auto',
            nu=adjusted_nu
        )
        
        self.one_class_svm.fit(combined_data)
        logger.info(f"One-Class SVM retrained with nu={adjusted_nu:.3f}")
    
    def _retrain_supervised_with_feedback(self):
        """Retrain supervised classifier with accumulated feedback"""
        # Prepare labeled data from feedback
        X_feedback = []
        y_feedback = []
        
        for feedback in self.feedback_buffer:
            if feedback['session_embedding'] is not None:
                X_feedback.append(feedback['session_embedding'])
                y_feedback.append(feedback['expert_label'])
        
        if len(X_feedback) < 10:
            logger.info("Insufficient labeled feedback for supervised retraining")
            return
        
        X_feedback = np.array(X_feedback)
        y_feedback = np.array(y_feedback)
        
        # Scale and transform
        X_scaled = self.scaler.transform(X_feedback)
        if hasattr(self.pca, 'components_'):
            X_scaled = self.pca.transform(X_scaled)
        
        # Initialize or update label encoder
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        
        # Handle new labels that weren't in original training
        unique_labels = np.unique(y_feedback)
        if hasattr(self.label_encoder, 'classes_'):
            all_labels = np.unique(np.concatenate([self.label_encoder.classes_, unique_labels]))
            self.label_encoder.classes_ = all_labels
        
        y_encoded = self.label_encoder.fit_transform(y_feedback)
        
        # Train or retrain classifier
        if self.supervised_classifier is None:
            self.supervised_classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
        
        self.supervised_classifier.fit(X_scaled, y_encoded)
        logger.info(f"Supervised classifier retrained with {len(X_feedback)} feedback samples")
    
    def _update_expert_rules_from_feedback(self):
        """Update expert rules based on feedback patterns"""
        # Analyze feedback to discover new patterns
        false_positive_patterns = []
        false_negative_patterns = []
        
        for feedback in self.feedback_buffer:
            if feedback['feedback_type'] == 'correction':
                if feedback['original_ml_prediction'] and feedback['expert_label'] == 'normal':
                    # False positive - should update normal indicators
                    false_positive_patterns.append(feedback['expert_explanation'])
                elif not feedback['original_ml_prediction'] and feedback['expert_label'] != 'normal':
                    # False negative - should update anomaly indicators
                    false_negative_patterns.append(feedback['expert_explanation'])
        
        # Log pattern insights for manual rule updates
        if false_positive_patterns:
            logger.info(f"Patterns causing false positives: {false_positive_patterns[:5]}")
        if false_negative_patterns:
            logger.info(f"Patterns causing false negatives: {false_negative_patterns[:5]}")
    
    def _evaluate_feedback_performance(self) -> float:
        """Evaluate how well the updated models perform on feedback data"""
        if len(self.feedback_buffer) < 5:
            return 0.0
        
        correct_predictions = 0
        total_predictions = 0
        
        for feedback in self.feedback_buffer:
            if feedback['session_embedding'] is not None:
                # Test updated models on this feedback sample
                embedding_scaled = self.scaler.transform([feedback['session_embedding']])
                if hasattr(self.pca, 'components_'):
                    embedding_scaled = self.pca.transform(embedding_scaled)
                

                
                # Get new predictions
                if_pred = self.isolation_forest.predict(embedding_scaled)[0]
                svm_pred = self.one_class_svm.predict(embedding_scaled)[0]
                
                ml_predicts_anomaly = (if_pred == -1) or (svm_pred == -1)
                expert_says_anomaly = feedback['expert_label'] != 'normal'
                
                if ml_predicts_anomaly == expert_says_anomaly:
                    correct_predictions += 1
                total_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        return accuracy
    
    def _archive_feedback_buffer(self, performance_improvement: float):
        """Archive processed feedback and clear buffer"""
        # Save feedback history
        feedback_archive = {
            'timestamp': datetime.now().isoformat(),
            'feedback_count': len(self.feedback_buffer),
            'performance_improvement': performance_improvement,
            'feedback_summary': {
                'confirmations': sum(1 for f in self.feedback_buffer if f['feedback_type'] == 'confirmation'),
                'corrections': sum(1 for f in self.feedback_buffer if f['feedback_type'] == 'correction'),
                'new_discoveries': sum(1 for f in self.feedback_buffer if f['feedback_type'] == 'new_discovery')
            }
        }
        
        self.model_performance_history.append(feedback_archive)
        
        # Clear buffer for next cycle
        self.feedback_buffer = []
        
        logger.info(f"Feedback buffer archived. Performance improvement: {performance_improvement:.3f}")
    
    def get_continuous_learning_status(self) -> Dict:
        """Get status of the continuous learning system"""
        return {
            'feedback_buffer_size': len(self.feedback_buffer),
            'learning_threshold': self.learning_threshold,
            'retraining_cycles': len(self.model_performance_history),
            'last_performance_improvement': self.model_performance_history[-1]['performance_improvement'] if self.model_performance_history else 0.0,
            'total_feedback_processed': sum(h['feedback_count'] for h in self.model_performance_history),
            'feedback_types_summary': {
                'confirmations': sum(1 for f in self.feedback_buffer if f['feedback_type'] == 'confirmation'),
                'corrections': sum(1 for f in self.feedback_buffer if f['feedback_type'] == 'correction'),
                'new_discoveries': sum(1 for f in self.feedback_buffer if f['feedback_type'] == 'new_discovery')
            }
        }

    # Convenience methods for expert feedback
    def expert_confirm_anomaly(self, session_id: str, anomaly_type: str = None, confidence: float = 1.0):
        """Expert confirms ML detected an anomaly correctly"""
        return self.collect_expert_feedback(
            session_id=session_id,
            expert_label=anomaly_type or 'anomaly',
            expert_confidence=confidence,
            feedback_type='confirmation',
            expert_explanation=f"Expert confirmed anomaly detection"
        )
    
    def expert_confirm_normal(self, session_id: str, confidence: float = 1.0):
        """Expert confirms ML correctly identified normal transaction"""
        return self.collect_expert_feedback(
            session_id=session_id,
            expert_label='normal',
            expert_confidence=confidence,
            feedback_type='confirmation',
            expert_explanation="Expert confirmed normal transaction"
        )
    
    def expert_correct_false_positive(self, session_id: str, explanation: str, confidence: float = 1.0):
        """Expert corrects a false positive (ML said anomaly, but it's normal)"""
        return self.collect_expert_feedback(
            session_id=session_id,
            expert_label='normal',
            expert_confidence=confidence,
            feedback_type='correction',
            expert_explanation=f"False positive correction: {explanation}"
        )
    
    def expert_correct_false_negative(self, session_id: str, true_anomaly_type: str, explanation: str, confidence: float = 1.0):
        """Expert corrects a false negative (ML said normal, but it's an anomaly)"""
        return self.collect_expert_feedback(
            session_id=session_id,
            expert_label=true_anomaly_type,
            expert_confidence=confidence,
            feedback_type='correction',
            expert_explanation=f"False negative correction: {explanation}"
        )
    
    def expert_discover_new_anomaly_type(self, session_id: str, new_anomaly_type: str, explanation: str, confidence: float = 1.0):
        """Expert identifies a new type of anomaly not previously known"""
        return self.collect_expert_feedback(
            session_id=session_id,
            expert_label=new_anomaly_type,
            expert_confidence=confidence,
            feedback_type='new_discovery',
            expert_explanation=f"New anomaly type discovered: {explanation}"
        )

    def generate_simple_embeddings(self, sessions: List[TransactionSession]) -> np.ndarray:
        """Fallback method using TF-IDF embeddings when transformers fail"""
        logger.info("Generating simple TF-IDF embeddings as fallback")
        
        # Extract text from sessions
        texts = [session.raw_text[:1000] for session in sessions]  # Limit text length
        
        try:
            # Use simple embedding generator
            generator = SimpleEmbeddingGenerator(n_components=384)
            embeddings = generator.fit_transform(texts)
            
            # Store embeddings back to sessions
            for i, session in enumerate(sessions):
                session.embedding = embeddings[i]
            
            logger.info(f"Generated {len(embeddings)} simple embeddings successfully")
            return embeddings
            
        except Exception as e:
            logger.error(f"Simple embeddings also failed: {e}")
            # Final fallback - create random embeddings for basic functionality
            logger.warning("Creating random embeddings for basic functionality")
            embeddings = np.random.randn(len(sessions), 384)
            
            for i, session in enumerate(sessions):
                session.embedding = embeddings[i]
            
            return embeddings
    
    def save_models(self, model_dir: str):
        """Save trained models to disk"""
        import os
        import joblib
        
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            # Save anomaly detection models
            if hasattr(self, 'isolation_forest') and self.isolation_forest is not None:
                joblib.dump(self.isolation_forest, os.path.join(model_dir, 'isolation_forest.pkl'))
                logger.info("Saved Isolation Forest model")
            
            if hasattr(self, 'one_class_svm') and self.one_class_svm is not None:
                joblib.dump(self.one_class_svm, os.path.join(model_dir, 'one_class_svm.pkl'))
                logger.info("Saved One-Class SVM model")
                
            # Save supervised classifier if it exists
            if hasattr(self, 'supervised_classifier') and self.supervised_classifier is not None:
                joblib.dump(self.supervised_classifier, os.path.join(model_dir, 'supervised_classifier.pkl'))
                logger.info("Saved supervised classifier model")
                
            # Save expert rules
            if hasattr(self, 'expert_rules') and self.expert_rules:
                import json
                with open(os.path.join(model_dir, 'expert_rules.json'), 'w') as f:
                    json.dump(self.expert_rules, f, indent=2)
                logger.info("Saved expert rules")
                
            # Save feedback history for continuous learning
            if hasattr(self, 'feedback_history') and self.feedback_history:
                import json
                with open(os.path.join(model_dir, 'feedback_history.json'), 'w') as f:
                    json.dump(self.feedback_history, f, indent=2)
                logger.info("Saved feedback history")
                
            logger.info(f"Models saved successfully to {model_dir}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")

    def load_models(self, model_dir: str):
        """Load trained models from disk"""
        import os
        import joblib
        
        try:
            # Load anomaly detection models
            isolation_forest_path = os.path.join(model_dir, 'isolation_forest.pkl')
            if os.path.exists(isolation_forest_path):
                self.isolation_forest = joblib.load(isolation_forest_path)
                logger.info("Loaded Isolation Forest model")
            
            one_class_svm_path = os.path.join(model_dir, 'one_class_svm.pkl')
            if os.path.exists(one_class_svm_path):
                self.one_class_svm = joblib.load(one_class_svm_path)
                logger.info("Loaded One-Class SVM model")
                
            # Load supervised classifier if it exists
            supervised_path = os.path.join(model_dir, 'supervised_classifier.pkl')
            if os.path.exists(supervised_path):
                self.supervised_classifier = joblib.load(supervised_path)
                logger.info("Loaded supervised classifier model")
                
            # Load expert rules
            expert_rules_path = os.path.join(model_dir, 'expert_rules.json')
            if os.path.exists(expert_rules_path):
                import json
                with open(expert_rules_path, 'r') as f:
                    self.expert_rules = json.load(f)
                logger.info("Loaded expert rules")
                
            # Load feedback history
            feedback_path = os.path.join(model_dir, 'feedback_history.json')
            if os.path.exists(feedback_path):
                import json
                with open(feedback_path, 'r') as f:
                    self.feedback_history = json.load(f)
                logger.info("Loaded feedback history")
                
            logger.info(f"Models loaded successfully from {model_dir}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def initialize_sentiment_models(self):
        """Initialize sentiment analysis and negative text detection models with DeepLog integration"""
        logger.info("Initializing advanced sentiment analysis and negative text detection models")
        
        try:
            # 1. VADER Sentiment Analyzer (Rule-based, good for technical text)
            self.vader_analyzer = SentimentIntensityAnalyzer()
            
            # 2. TextBlob (Statistical sentiment analysis)
            self.textblob_enabled = True
            
            # 3. Domain-specific negative phrase classifier
            self.initialize_negative_phrase_classifier()
            
            # 4. Technical failure sentiment model (transformer-based)
            self.initialize_technical_failure_model()
            
            # 5. Error severity classifier
            self.initialize_error_severity_classifier()
            
            # 6. NEW: DeepLog-enhanced sentiment context analyzer
            self.initialize_deeplog_sentiment_analyzer()
            
            # 7. NEW: Contextual emotion detection for ATM transactions
            self.initialize_contextual_emotion_detector()
            
            # 8. NEW: Adaptive negative pattern learner
            self.initialize_adaptive_pattern_learner()
            
            logger.info("Advanced sentiment analysis models with DeepLog integration initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing sentiment models: {str(e)}")
            # Fallback to basic text analysis
            self.vader_analyzer = None
            self.textblob_enabled = False
    
    def initialize_negative_phrase_classifier(self):
        """Initialize domain-specific negative phrase classifier for ATM logs"""
        
        # ATM-specific negative phrases and their severity weights
        self.atm_negative_phrases = {
            # Critical failures
            'DISPENSE FAIL': 0.95,
            'HARDWARE ERROR': 0.90,
            'SENSOR ERROR': 0.85,
            'MOTOR ERROR': 0.88,
            'UNABLE TO DISPENSE': 0.95,
            'DEVICE ERROR': 0.80,
            'RECOVERY FAILED': 0.92,
            'NOTES JAMMED': 0.85,
            'CASH CASSETTE ERROR': 0.90,
            
            # Security issues
            'UNAUTHORIZED ACCESS': 0.98,
            'CARD RETAINED': 0.75,
            'CARD CAPTURED': 0.80,
            'SUPERVISOR MODE UNAUTHORIZED': 0.95,
            
            # Communication/Network issues
            'TIMEOUT': 0.65,
            'NO RESPONSE': 0.70,
            'CONNECTION FAILED': 0.75,
            'NETWORK ERROR': 0.72,
            
            # Cash handling issues
            'CASH RETRACT': 0.70,
            'NOTES NOT TAKEN': 0.60,
            'CASH EMPTY': 0.85,
            'CASSETTE FAULT': 0.80,
            
            # System issues
            'POWER RESET': 0.65,
            'SYSTEM FAULT': 0.78,
            'REBOOT REQUIRED': 0.70,
            'SERVICE REQUIRED': 0.75
        }
        
        # Initialize TF-IDF vectorizer for negative text classification
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        
        # Initialize Naive Bayes classifier for failure detection
        self.failure_classifier = MultinomialNB(alpha=0.1)
        
        # Initialize logistic regression for severity classification
        self.severity_classifier = LogisticRegression(random_state=42)
        
        logger.info("Negative phrase classifier initialized with ATM-specific vocabulary")
    
    def initialize_technical_failure_model(self):
        """Initialize transformer-based technical failure detection model"""
        try:
            # Use a pre-trained model fine-tuned for technical/error text
            self.technical_failure_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            # Alternative: Use a general-purpose model
            # self.technical_failure_pipeline = pipeline(
            #     "text-classification",
            #     model="distilbert-base-uncased-finetuned-sst-2-english"
            # )
            
            logger.info("Technical failure detection model initialized")
            
        except Exception as e:
            logger.warning(f"Could not initialize transformer model: {str(e)}")
            self.technical_failure_pipeline = None
    
    def initialize_error_severity_classifier(self):
        """Initialize error severity classification model"""
        
        # Error severity mapping based on ATM operations
        self.error_severity_mapping = {
            'CRITICAL': ['DISPENSE FAIL', 'HARDWARE ERROR', 'UNAUTHORIZED ACCESS', 'CASH CASSETTE ERROR'],
            'HIGH': ['SENSOR ERROR', 'MOTOR ERROR', 'RECOVERY FAILED', 'DEVICE ERROR', 'NOTES JAMMED'],
            'MEDIUM': ['TIMEOUT', 'CONNECTION FAILED', 'CARD RETAINED', 'CASH RETRACT', 'SYSTEM FAULT'],
            'LOW': ['POWER RESET', 'NOTES NOT TAKEN', 'SERVICE REQUIRED', 'REBOOT REQUIRED']
        }
        
        # Create training data for severity classification
        severity_texts = []
        severity_labels = []
        
        for severity, phrases in self.error_severity_mapping.items():
            for phrase in phrases:
                severity_texts.append(phrase)
                severity_labels.append(severity)
        
        # Train a simple severity classifier
        if len(severity_texts) > 0:
            try:
                severity_tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
                severity_features = severity_tfidf.fit_transform(severity_texts)
                
                self.severity_classifier = LogisticRegression(random_state=42)
                self.severity_classifier.fit(severity_features, severity_labels)
                self.severity_tfidf = severity_tfidf
                
                logger.info("Error severity classifier trained successfully")
                
            except Exception as e:
                logger.error(f"Error training severity classifier: {str(e)}")
                self.severity_classifier = None
        else:
            self.severity_classifier = None

    def analyze_negative_sentiment(self, session: TransactionSession) -> Dict[str, Any]:
        """
        Analyze text for negative sentiment and failure indicators
        Returns sentiment scores and detected negative patterns
        """
        text = session.raw_text
        sentiment_results = {
            'vader_score': 0.0,
            'textblob_score': 0.0,
            'technical_failure_score': 0.0,
            'negative_phrases': [],
            'severity_level': 'LOW',
            'confidence': 0.0,
            'detected_patterns': []
        }
        
        try:
            # 1. VADER Sentiment Analysis (good for technical text)
            if self.vader_analyzer:
                vader_scores = self.vader_analyzer.polarity_scores(text)
                sentiment_results['vader_score'] = vader_scores['compound']
                sentiment_results['vader_negative'] = vader_scores['neg']
                sentiment_results['vader_details'] = vader_scores

            # 2. TextBlob Sentiment Analysis
            if self.textblob_enabled:
                try:
                    blob = TextBlob(text)
                    sentiment_results['textblob_score'] = blob.sentiment.polarity
                    sentiment_results['textblob_subjectivity'] = blob.sentiment.subjectivity
                except:
                    logger.warning("TextBlob analysis failed, skipping")

            # 3. Domain-specific negative phrase detection
            negative_phrases, phrase_score = self.detect_negative_phrases(text)
            sentiment_results['negative_phrases'] = negative_phrases
            sentiment_results['phrase_score'] = phrase_score

            # 4. Technical failure detection using transformer model
            if self.technical_failure_pipeline:
                try:
                    # Split text into chunks for processing
                    chunks = self.split_text_for_analysis(text)
                    technical_scores = []

                    for chunk in chunks:
                        if len(chunk.strip()) > 10:  # Skip very short chunks
                            result = self.technical_failure_pipeline(chunk)
                            if isinstance(result, list) and len(result) > 0:
                                # Extract negative sentiment score
                                for item in result[0]:
                                    if item['label'] in ['NEGATIVE', 'LABEL_0']:
                                        technical_scores.append(item['score'])

                    if technical_scores:
                        sentiment_results['technical_failure_score'] = max(technical_scores)
                        sentiment_results['avg_technical_score'] = sum(technical_scores) / len(technical_scores)

                except Exception as e:
                    logger.warning(f"Technical failure detection failed: {str(e)}")

            # 5. Error severity classification
            if self.severity_classifier and hasattr(self, 'severity_tfidf'):
                try:
                    severity_features = self.severity_tfidf.transform([text])
                    severity_pred = self.severity_classifier.predict(severity_features)[0]
                    severity_proba = self.severity_classifier.predict_proba(severity_features)[0]

                    sentiment_results['severity_level'] = severity_pred
                    sentiment_results['severity_confidence'] = max(severity_proba)

                except Exception as e:
                    logger.warning(f"Severity classification failed: {str(e)}")

            # 6. Calculate overall confidence score
            sentiment_results['confidence'] = self.calculate_sentiment_confidence(sentiment_results)

            # 7. Detect specific failure patterns
            sentiment_results['detected_patterns'] = self.detect_failure_patterns(text)

            return sentiment_results

        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return sentiment_results
    
    def detect_negative_phrases(self, text: str) -> Tuple[List[str], float]:
        """Detect ATM-specific negative phrases and calculate severity score"""
        detected_phrases = []
        severity_scores = []
        
        text_upper = text.upper()
        
        for phrase, severity in self.atm_negative_phrases.items():
            if phrase in text_upper:
                detected_phrases.append(phrase)
                severity_scores.append(severity)
        
        # Calculate weighted average severity
        if severity_scores:
            avg_severity = sum(severity_scores) / len(severity_scores)
            max_severity = max(severity_scores)
            # Use weighted combination of average and max
            phrase_score = (avg_severity * 0.6) + (max_severity * 0.4)
        else:
            phrase_score = 0.0
        
        return detected_phrases, phrase_score
    
    def split_text_for_analysis(self, text: str, max_length: int = 512) -> List[str]:
        """Split text into chunks for transformer model processing"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= max_length:
                current_chunk.append(word)
                current_length += len(word) + 1
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def calculate_sentiment_confidence(self, sentiment_results: Dict[str, Any]) -> float:
        """Calculate overall confidence score for sentiment analysis"""
        confidence_factors = []
        
        # VADER score contribution
        if 'vader_score' in sentiment_results:
            confidence_factors.append(sentiment_results['vader_score'])
        
        # TextBlob score contribution
        if 'textblob_score' in sentiment_results:
            confidence_factors.append(sentiment_results['textblob_score'])
        
        # Technical failure score contribution
        if 'technical_failure_score' in sentiment_results:
            confidence_factors.append(sentiment_results['technical_failure_score'])
        
        # Severity classification confidence
        if 'severity_confidence' in sentiment_results:
            confidence_factors.append(sentiment_results['severity_confidence'])
        
        # Overall confidence is the product of individual confidences
        overall_confidence = 1.0
        for factor in confidence_factors:
            overall_confidence *= factor + 0.01  # Avoid multiplication by zero
        
        return min(overall_confidence, 1.0)  # Cap at 1.0
    
    def initialize_deeplog_sentiment_analyzer(self):
        """Initialize DeepLog-enhanced sentiment context analyzer"""
        logger.info("Initializing DeepLog-enhanced sentiment context analyzer")
        
        try:
            # DeepLog-sentiment integration parameters
            self.deeplog_sentiment_config = {
                'sequence_window': 5,  # Analyze sentiment in context of 5 events
                'sentiment_threshold': -0.3,  # Negative sentiment threshold
                'context_weight': 0.7,  # Weight for contextual sentiment vs isolated sentiment
                'emotion_escalation_threshold': 2  # Number of consecutive negative events
            }
            
            # Context-aware sentiment patterns for ATM transactions
            self.contextual_sentiment_patterns = {
                'escalating_frustration': {
                    'pattern': ['TIMEOUT', 'ERROR', 'RETRY', 'FAIL'],
                    'sentiment_weight': 0.8,
                    'description': 'Progressive user frustration pattern'
                },
                'critical_failure_cascade': {
                    'pattern': ['ERROR', 'FAULT', 'UNABLE', 'FAIL'],
                    'sentiment_weight': 0.9,
                    'description': 'Multiple system failures in sequence'
                },
                'security_concern_pattern': {
                    'pattern': ['UNAUTHORIZED', 'RETAINED', 'CAPTURED', 'SUPERVISOR'],
                    'sentiment_weight': 0.95,
                    'description': 'Security-related negative events'
                },
                'incomplete_transaction_frustration': {
                    'pattern': ['START', 'INVALID', 'UNABLE', 'END'],
                    'sentiment_weight': 0.75,
                    'description': 'Transaction starts but fails to complete properly'
                }
            }
            
            # Initialize sentiment-sequence correlation model
            self.sentiment_sequence_model = {
                'normal_sentiment_sequences': [],
                'anomaly_sentiment_sequences': [],
                'learned_emotional_escalations': []
            }
            
            logger.info("DeepLog-sentiment analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing DeepLog-sentiment analyzer: {str(e)}")
            self.deeplog_sentiment_config = None
    
    def initialize_contextual_emotion_detector(self):
        """Initialize contextual emotion detection for ATM transactions"""
        logger.info("Initializing contextual emotion detector")
        
        try:
            # ATM-specific emotional indicators and their contexts
            self.atm_emotional_indicators = {
                # Frustration indicators
                'frustration': {
                    'keywords': ['TIMEOUT', 'RETRY', 'AGAIN', 'REPEAT', 'MULTIPLE'],
                    'context_multipliers': {
                        'sequential': 1.5,  # Multiple frustration events in sequence
                        'timeout_related': 1.3,  # Timeout-related frustration
                        'repeated_attempts': 1.4  # Multiple retry attempts
                    },
                    'base_weight': 0.6
                },
                
                # Anxiety/concern indicators
                'anxiety': {
                    'keywords': ['CARD RETAINED', 'CARD CAPTURED', 'UNAUTHORIZED', 'SECURITY'],
                    'context_multipliers': {
                        'security_related': 1.8,
                        'card_capture': 1.6,
                        'unauthorized_access': 1.9
                    },
                    'base_weight': 0.8
                },
                
                # Confusion indicators
                'confusion': {
                    'keywords': ['INVALID', 'UNKNOWN', 'UNEXPECTED', 'UNRECOGNIZED'],
                    'context_multipliers': {
                        'invalid_operations': 1.2,
                        'unknown_errors': 1.3,
                        'unexpected_behavior': 1.4
                    },
                    'base_weight': 0.5
                },
                
                # Urgency/critical indicators
                'urgency': {
                    'keywords': ['CRITICAL', 'EMERGENCY', 'IMMEDIATE', 'URGENT', 'FAULT'],
                    'context_multipliers': {
                        'hardware_fault': 1.7,
                        'critical_error': 1.8,
                        'emergency_situation': 1.9
                    },
                    'base_weight': 0.9
                }
            }
            
            # Initialize emotional pattern learning
            self.emotional_pattern_learner = {
                'learned_patterns': {},
                'pattern_frequencies': {},
                'expert_validated_emotions': {}
            }
            
            logger.info("Contextual emotion detector initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing contextual emotion detector: {str(e)}")
            self.atm_emotional_indicators = {}
    
    def initialize_adaptive_pattern_learner(self):
        """Initialize adaptive negative pattern learner that evolves with data"""
        logger.info("Initializing adaptive negative pattern learner")
        
        try:
            # Adaptive learning configuration
            self.adaptive_learner_config = {
                'min_pattern_frequency': 3,  # Minimum occurrences to consider a pattern
                'learning_rate': 0.1,  # How quickly to adapt to new patterns
                'expert_feedback_weight': 2.0,  # Weight for expert-validated patterns
                'auto_discovery_threshold': 0.7  # Threshold for auto-discovering new patterns
            }
            
            # Dynamic pattern storage
            self.discovered_negative_patterns = {
                'auto_discovered': {},  # Patterns discovered automatically
                'expert_validated': {},  # Patterns validated by experts
                'false_positive_patterns': {},  # Patterns marked as false positives
                'evolving_patterns': {}  # Patterns that are still being learned
            }
            
            # Pattern evolution tracking
            self.pattern_evolution_tracker = {
                'pattern_performance': {},  # Track how well patterns perform
                'pattern_confidence': {},  # Confidence scores for each pattern
                'pattern_context': {}  # Contextual information for patterns
            }
            
            logger.info("Adaptive pattern learner initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing adaptive pattern learner: {str(e)}")
            self.adaptive_learner_config = None
    
    # Override the default method to extract timestamp from individual lines
    def extract_timestamp_from_line(self, line: str) -> Optional[datetime]:
        """Extract timestamp from a single line, specifically for the line above TRANSACTION START"""
        # Pattern for lines like: [020t*632*06/18/2025*04:48*
        timestamp_pattern = re.compile(r'\*(\d{2}/\d{2}/\d{4})\*(\d{2}:\d{2})\*')
        match = timestamp_pattern.search(line)
        
        if match:
            date_str = match.group(1)
            time_str = match.group(2)
            try:
                # Parse the date and time
                return datetime.strptime(f"{date_str} {time_str}", "%m/%d/%Y %H:%M")
            except ValueError:
                logger.warning(f"Could not parse timestamp from line: {line}")
                return None
        
        # Fallback to the original timestamp extraction patterns
        timestamp_patterns = [
            r'(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2}:\d{2})',
            r'(\d{4}/\d{2}/\d{2})\s+(\d{2}:\d{2}:\d{2})',
            r'(\d{2}:\d{2}:\d{2})'
        ]
        
        for pattern in timestamp_patterns:
            match = re.search(pattern, line)
            if match:
                return self.parse_timestamp(match.groups())
        
        return None

    def prepare_text_for_embedding(self, raw_text: str, max_length: int = 2048) -> str:
        """
        Prepare text for embedding generation with intelligent handling of long sessions.
        Instead of simple truncation, extract key patterns and summarize important information.
        """
        if len(raw_text) <= max_length:
            return raw_text
        
        # For longer sessions, extract key patterns and create a summary
        logger.info(f"Processing long session ({len(raw_text)} chars) for embedding")
        
        # Extract important patterns from the entire text
        key_patterns = []
        
        # 1. Extract unique error patterns
        error_patterns = set()
        error_matches = re.finditer(r'(ERROR|FAULT|FAILED|TIMEOUT|EXCEPTION|REJECT)', raw_text, re.IGNORECASE)
        for match in error_matches:
            # Get context around the error
            start = max(0, match.start() - 50)
            end = min(len(raw_text), match.end() + 50)
            error_patterns.add(raw_text[start:end].strip())
        
        # 2. Extract supervisor mode entries (these could indicate issues)
        supervisor_patterns = set()
        supervisor_matches = re.finditer(r'SUPERVISOR MODE (ENTRY|EXIT)', raw_text, re.IGNORECASE)
        for match in supervisor_matches:
            start = max(0, match.start() - 30)
            end = min(len(raw_text), match.end() + 30)
            supervisor_patterns.add(raw_text[start:end].strip())
        
        # 3. Count repetitive patterns to detect anomalies
        repetitive_patterns = {}
        diagnostic_matches = re.finditer(r'(\*.*?\*[0-9D]*\*.*?R-[0-9]+)', raw_text)
        for match in diagnostic_matches:
            pattern = match.group(1)
            repetitive_patterns[pattern] = repetitive_patterns.get(pattern, 0) + 1
        
        # 4. Extract transaction boundaries
        transaction_boundaries = []
        boundary_matches = re.finditer(r'(TRANSACTION START|TRANSACTION END|CARDLESS TRANSACTION)', raw_text, re.IGNORECASE)
        for match in boundary_matches:
            start = max(0, match.start() - 20)
            end = min(len(raw_text), match.end() + 20)
            transaction_boundaries.append(raw_text[start:end].strip())
        
        # Build summarized text
        summary_parts = []
        
        # Always include the beginning of the session
        summary_parts.append("SESSION_START: " + raw_text[:200])
        
        # Add error patterns
        if error_patterns:
            summary_parts.append("ERRORS: " + " | ".join(list(error_patterns)[:5]))
        
        # Add supervisor mode patterns
        if supervisor_patterns:
            summary_parts.append("SUPERVISOR: " + " | ".join(list(supervisor_patterns)[:3]))
        
        # Add information about repetitive patterns
        if repetitive_patterns:
            most_common = sorted(repetitive_patterns.items(), key=lambda x: x[1], reverse=True)[:3]
            rep_info = []
            for pattern, count in most_common:
                if count > 5:  # Only include patterns that repeat significantly
                    rep_info.append(f"{pattern}(x{count})")
            if rep_info:
                summary_parts.append("REPETITIVE: " + " | ".join(rep_info))
        
        # Add transaction boundaries
        if transaction_boundaries:
            summary_parts.append("BOUNDARIES: " + " | ".join(transaction_boundaries[:3]))
        
        # Always include the end of the session
        summary_parts.append("SESSION_END: " + raw_text[-200:])
        
        # Join all parts and ensure we don't exceed max_length
        summarized_text = " || ".join(summary_parts)
        
        if len(summarized_text) > max_length:
            # If still too long, truncate but try to keep the most important parts
            return summarized_text[:max_length]
        
        return summarized_text
    def _detect_deeplog_anomalies(self, session: TransactionSession, events: List[str]):
        """Detect sequential pattern anomalies using DeepLog LSTM model"""
        if not self.deeplog_analyzer or not self.deeplog_trained:
            return
        
        try:
            # Extract event sequence for DeepLog analysis
            event_sequence = self.deeplog_analyzer.extract_event_sequence(session.raw_text)
            
            if len(event_sequence) < 2:  # Need at least 2 events for sequence analysis
                return
            
            # Check for anomalous patterns
            is_anomalous, confidence, anomaly_details = self.deeplog_analyzer.detect_anomaly(event_sequence)
            
            if is_anomalous:
                session.add_anomaly(
                    anomaly_type="sequential_pattern_anomaly",
                    confidence=confidence,
                    detection_method="deeplog_lstm",
                    description=f"Anomalous transaction sequence detected: {anomaly_details.get('description', 'Unexpected event sequence')}",
                    severity=self._determine_severity(confidence),
                    details={
                        "event_sequence": event_sequence,
                        "anomaly_type": anomaly_details.get('anomaly_type', 'unknown'),
                        "expected_next_events": anomaly_details.get('expected_events', []),
                        "actual_events": anomaly_details.get('actual_events', []),
                        "sequence_analysis": anomaly_details
                    }
                )
                
                logger.info(f"DeepLog detected sequential anomaly in {session.session_id}: {anomaly_details.get('description', 'Unknown pattern')}")
            
            # Check for incomplete transaction patterns
            self._check_transaction_completeness_deeplog(session, event_sequence)
            
        except Exception as e:
            logger.error(f"Error in DeepLog anomaly detection for session {session.session_id}: {e}")
    
    def _check_transaction_completeness_deeplog(self, session: TransactionSession, event_sequence: List[str]):
        """Use DeepLog to detect incomplete transaction patterns"""
        if not self.deeplog_analyzer:
            return
        
        try:
            # Check if transaction appears complete based on learned patterns
            is_complete, completeness_score, missing_events = self.deeplog_analyzer.check_transaction_completeness(event_sequence)
            
            if not is_complete and completeness_score < 0.5:
                session.add_anomaly(
                    anomaly_type="incomplete_transaction_deeplog",
                    confidence=1.0 - completeness_score,
                    detection_method="deeplog_completeness",
                    description=f"Transaction appears incomplete based on learned patterns. Missing expected events: {', '.join(missing_events)}",
                    severity=self._determine_severity(1.0 - completeness_score),
                    details={
                        "event_sequence": event_sequence,
                        "completeness_score": completeness_score,
                        "missing_events": missing_events,
                        "pattern_type": "incomplete_transaction"
                    }
                )
                
                logger.info(f"DeepLog detected incomplete transaction in {session.session_id}: completeness score {completeness_score:.3f}")
                
        except Exception as e:
            logger.error(f"Error in DeepLog completeness check for session {session.session_id}: {e}")

    def train_deeplog_model(self, training_sessions: List[TransactionSession] = None):
        """Train the DeepLog model on normal transaction patterns"""
        if not self.deeplog_analyzer:
            logger.warning("DeepLog analyzer not available for training")
            return False
        
        try:
            # Use provided sessions or current sessions for training
            sessions_to_use = training_sessions or self.sessions
            
            if not sessions_to_use:
                logger.warning("No sessions available for DeepLog training")
                return False
            
            # Extract normal transaction sequences for training
            normal_sequences = []
            for session in sessions_to_use:
                # Only use sessions that are not anomalies or are confirmed normal transactions
                if (not session.is_anomaly or 
                    session.anomaly_type in ['normal_withdrawal', 'normal_inquiry'] or
                    (session.extracted_details and session.extracted_details.get('expert_override', False))):
                    
                    event_sequence = self.deeplog_analyzer.extract_event_sequence(session.raw_text)
                    if len(event_sequence) >= 3:  # Need minimum sequence length
                        normal_sequences.append(event_sequence)
            
            if len(normal_sequences) < 10:
                logger.warning(f"Only {len(normal_sequences)} normal sequences available - need at least 10 for training")
                return False
            
            # Train the model
            logger.info(f"Training DeepLog model on {len(normal_sequences)} normal transaction sequences")
            success = self.deeplog_analyzer.train(normal_sequences)
            
            if success:
                self.deeplog_trained = True
                # Save the trained model
                if self.deeplog_analyzer.save_model():
                    logger.info("DeepLog model trained and saved successfully")
                else:
                    logger.warning("DeepLog model trained but failed to save")
                return True
            else:
                logger.error("Failed to train DeepLog model")
                return False
                
        except Exception as e:
            logger.error(f"Error training DeepLog model: {e}")
            return False
    
    def _detect_deeplog_sentiment_anomalies(self, session: TransactionSession, events: List[str]):
        """
        Advanced DeepLog-enhanced sentiment anomaly detection
        Combines sequential pattern analysis with contextual sentiment analysis
        """
        if not hasattr(self, 'deeplog_sentiment_config') or not self.deeplog_sentiment_config:
            return
        
        try:
            # 1. Analyze overall sentiment of the session
            sentiment_analysis = self.analyze_negative_sentiment(session)
            
            # 2. Extract event sequence for contextual analysis
            event_sequence = []
            if self.deeplog_analyzer:
                event_sequence = self.deeplog_analyzer.extract_event_sequence(session.raw_text)
            
            # 3. Detect sentiment-sequence correlation anomalies
            sentiment_anomalies = self._analyze_sentiment_sequence_correlation(
                session, event_sequence, sentiment_analysis
            )
            
            # 4. Detect contextual emotional escalation
            emotion_anomalies = self._detect_emotional_escalation_patterns(
                session, event_sequence, sentiment_analysis
            )
            
            # 5. Apply adaptive pattern learning
            adaptive_anomalies = self._apply_adaptive_negative_pattern_detection(
                session, sentiment_analysis
            )
            
            # Add detected anomalies to session
            all_detected_anomalies = sentiment_anomalies + emotion_anomalies + adaptive_anomalies
            for anomaly in all_detected_anomalies:
                session.add_anomaly(**anomaly)
            
            # 6. Learn from this session for future improvement
            self._update_sentiment_learning_models(session, sentiment_analysis, event_sequence)
            
        except Exception as e:
            logger.error(f"Error in DeepLog-sentiment anomaly detection: {str(e)}")
    
    def _analyze_sentiment_sequence_correlation(self, session: TransactionSession, 
                                              event_sequence: List[str], 
                                              sentiment_analysis: Dict) -> List[Dict]:
        """Analyze correlation between event sequences and sentiment patterns"""
        anomalies = []
        
        try:
            # Check if negative sentiment correlates with specific event patterns
            negative_sentiment_score = min(
                sentiment_analysis.get('vader_score', 0),
                sentiment_analysis.get('textblob_score', 0)
            )
            
            # High negative sentiment threshold
            if negative_sentiment_score < -0.5:
                
                # Pattern 1: Negative sentiment with incomplete sequences
                if len(event_sequence) < 4 and any(neg_phrase in session.raw_text.upper() 
                                                 for neg_phrase in ['INVALID', 'UNABLE', 'ERROR', 'FAIL']):
                    anomalies.append({
                        'anomaly_type': 'sentiment_sequence_mismatch',
                        'confidence': abs(negative_sentiment_score) * 0.9,
                        'detection_method': 'deeplog_sentiment_correlation',
                        'description': f'High negative sentiment ({negative_sentiment_score:.3f}) with incomplete transaction sequence',
                        'severity': 'high' if abs(negative_sentiment_score) > 0.7 else 'medium',
                        'details': {
                            'sentiment_score': negative_sentiment_score,
                            'event_sequence_length': len(event_sequence),
                            'detected_negative_phrases': sentiment_analysis.get('negative_phrases', []),
                            'correlation_type': 'negative_sentiment_incomplete_sequence'
                        }
                    })
                
                # Pattern 2: Progressive sentiment degradation
                if self._detect_progressive_sentiment_degradation(session, event_sequence):
                    anomalies.append({
                        'anomaly_type': 'progressive_sentiment_degradation',
                        'confidence': 0.8,
                        'detection_method': 'deeplog_sentiment_progression',
                        'description': 'Sentiment progressively worsens throughout transaction sequence',
                        'severity': 'high',
                        'details': {
                            'sentiment_progression': 'degrading',
                            'final_sentiment': negative_sentiment_score,
                            'event_sequence': event_sequence
                        }
                    })
            
            # Pattern 3: Sentiment-sequence mismatch (positive sequence, negative sentiment)
            if self._is_positive_sequence(event_sequence) and negative_sentiment_score < -0.3:
                anomalies.append({
                    'anomaly_type': 'sentiment_sequence_contradiction',
                    'confidence': 0.75,
                    'detection_method': 'deeplog_sentiment_contradiction',
                    'description': 'Positive transaction sequence with unexpected negative sentiment indicators',
                    'severity': 'medium',
                    'details': {
                        'sequence_type': 'positive',
                        'sentiment_score': negative_sentiment_score,
                        'contradiction_indicator': True
                    }
                })
            
        except Exception as e:
            logger.warning(f"Error in sentiment-sequence correlation analysis: {str(e)}")
        
        return anomalies
    
    def _detect_emotional_escalation_patterns(self, session: TransactionSession,
                                            event_sequence: List[str],
                                            sentiment_analysis: Dict) -> List[Dict]:
        """Detect emotional escalation patterns using contextual analysis"""
        anomalies = []
        
        try:
            if not hasattr(self, 'atm_emotional_indicators'):
                return anomalies
            
            session_text = session.raw_text.upper()
            detected_emotions = {}
            
            # Analyze each emotional category
            for emotion_type, config in self.atm_emotional_indicators.items():
                emotion_score = 0
                detected_keywords = []
                
                # Check for emotion keywords
                for keyword in config['keywords']:
                    if keyword in session_text:
                        detected_keywords.append(keyword)
                        base_score = config['base_weight']
                        
                        # Apply context multipliers
                        context_score = base_score
                        for context, multiplier in config['context_multipliers'].items():
                            if self._check_emotional_context(session_text, keyword, context):
                                context_score *= multiplier
                        
                        emotion_score = max(emotion_score, context_score)
                
                if emotion_score > 0.6:  # Significant emotional indicator
                    detected_emotions[emotion_type] = {
                        'score': emotion_score,
                        'keywords': detected_keywords
                    }
            
            # Check for escalating emotional patterns
            if len(detected_emotions) >= 2:  # Multiple emotions detected
                high_emotion_types = [k for k, v in detected_emotions.items() if v['score'] > 0.7]
                
                if len(high_emotion_types) >= 2:
                    anomalies.append({
                        'anomaly_type': 'multi_emotional_escalation',
                        'confidence': min(0.95, max(e['score'] for e in detected_emotions.values())),
                        'detection_method': 'contextual_emotion_detection',
                        'description': f'Multiple high-intensity emotions detected: {", ".join(high_emotion_types)}',
                        'severity': 'high',
                        'details': {
                            'detected_emotions': detected_emotions,
                            'escalation_type': 'multi_emotional',
                            'dominant_emotions': high_emotion_types
                        }
                    })
            
            # Check for critical emotional states
            if 'urgency' in detected_emotions and detected_emotions['urgency']['score'] > 0.8:
                anomalies.append({
                    'anomaly_type': 'critical_emotional_state',
                    'confidence': detected_emotions['urgency']['score'],
                    'detection_method': 'urgency_emotion_detection',
                    'description': 'Critical urgency indicators detected in transaction',
                    'severity': 'critical',
                    'details': {
                        'emotion_type': 'urgency',
                        'emotion_score': detected_emotions['urgency']['score'],
                        'triggering_keywords': detected_emotions['urgency']['keywords']
                    }
                })
            
        except Exception as e:
            logger.warning(f"Error in emotional escalation detection: {str(e)}")
        
        return anomalies
    
    def _apply_adaptive_negative_pattern_detection(self, session: TransactionSession,
                                                 sentiment_analysis: Dict) -> List[Dict]:
        """Apply adaptive learning to detect evolving negative patterns"""
        anomalies = []
        
        try:
            if not hasattr(self, 'discovered_negative_patterns'):
                return anomalies
            
            session_text = session.raw_text.upper()
            
            # Check against auto-discovered patterns
            for pattern, pattern_info in self.discovered_negative_patterns.get('auto_discovered', {}).items():
                if pattern in session_text and pattern_info.get('confidence', 0) > 0.6:
                    anomalies.append({
                        'anomaly_type': 'adaptive_negative_pattern',
                        'confidence': pattern_info['confidence'],
                        'detection_method': 'adaptive_pattern_learning',
                        'description': f'Auto-discovered negative pattern detected: {pattern}',
                        'severity': self._determine_adaptive_severity(pattern_info),
                        'details': {
                            'pattern': pattern,
                            'discovery_method': 'automatic',
                            'pattern_frequency': pattern_info.get('frequency', 0),
                            'pattern_contexts': pattern_info.get('contexts', [])
                        }
                    })
            
            # Check against expert-validated patterns
            for pattern, pattern_info in self.discovered_negative_patterns.get('expert_validated', {}).items():
                if pattern in session_text:
                    anomalies.append({
                        'anomaly_type': 'expert_validated_negative_pattern',
                        'confidence': pattern_info.get('confidence', 0.9),
                        'detection_method': 'expert_validated_pattern',
                        'description': f'Expert-validated negative pattern detected: {pattern}',
                        'severity': pattern_info.get('severity', 'medium'),
                        'details': {
                            'pattern': pattern,
                            'validation_method': 'expert',
                            'expert_notes': pattern_info.get('expert_notes', ''),
                            'validation_date': pattern_info.get('validation_date', '')
                        }
                    })
            
        except Exception as e:
            logger.warning(f"Error in adaptive negative pattern detection: {str(e)}")
        
        return anomalies
    
    def _detect_progressive_sentiment_degradation(self, session: TransactionSession, 
                                                event_sequence: List[str]) -> bool:
        """Detect if sentiment progressively worsens throughout the session"""
        try:
            # Split session text into chunks and analyze sentiment progression
            text_chunks = self._split_session_into_temporal_chunks(session.raw_text)
            
            if len(text_chunks) < 3:  # Need at least 3 chunks for progression analysis
                return False
            
            sentiment_scores = []
            for chunk in text_chunks:
                if self.vader_analyzer:
                    chunk_sentiment = self.vader_analyzer.polarity_scores(chunk)['compound']
                    sentiment_scores.append(chunk_sentiment)
            
            # Check for progressive degradation (each chunk more negative than previous)
            degradation_count = 0
            for i in range(1, len(sentiment_scores)):
                if sentiment_scores[i] < sentiment_scores[i-1] - 0.1:  # Significant degradation
                    degradation_count += 1
            
            # Consider it progressive degradation if more than half the transitions are negative
            return degradation_count >= len(sentiment_scores) // 2
            
        except:
            return False
    
    def _is_positive_sequence(self, event_sequence: List[str]) -> bool:
        """Determine if an event sequence represents a positive/successful transaction"""
        positive_indicators = ['NOTES_TAKEN', 'CARD_TAKEN', 'RECEIPT_PRINTED', 'BALANCE_INQUIRY', 'SUCCESSFUL']
        negative_indicators = ['ERROR', 'FAIL', 'TIMEOUT', 'UNABLE', 'INVALID']
        
        positive_count = sum(1 for event in event_sequence if any(pos in event.upper() for pos in positive_indicators))
        negative_count = sum(1 for event in event_sequence if any(neg in event.upper() for neg in negative_indicators))
        
        return positive_count > negative_count
    
    def _check_emotional_context(self, text: str, keyword: str, context: str) -> bool:
        """Check if a keyword appears in a specific emotional context"""
        context_patterns = {
            'sequential': lambda t, k: text.count(k) > 1,
            'timeout_related': lambda t, k: any(timeout in t for timeout in ['TIMEOUT', 'NO RESPONSE']),
            'repeated_attempts': lambda t, k: any(repeat in t for repeat in ['RETRY', 'AGAIN', 'REPEAT']),
            'security_related': lambda t, k: any(sec in t for sec in ['UNAUTHORIZED', 'SECURITY', 'VIOLATION']),
            'card_capture': lambda t, k: any(card in t for card in ['CARD RETAINED', 'CARD CAPTURED']),
            'unauthorized_access': lambda t, k: 'UNAUTHORIZED' in t,
            'invalid_operations': lambda t, k: 'INVALID' in t,
            'unknown_errors': lambda t, k: any(unknown in t for unknown in ['UNKNOWN', 'UNRECOGNIZED']),
            'unexpected_behavior': lambda t, k: 'UNEXPECTED' in t,
            'hardware_fault': lambda t, k: any(hw in t for hw in ['HARDWARE', 'DEVICE ERROR', 'SENSOR']),
            'critical_error': lambda t, k: 'CRITICAL' in t,
            'emergency_situation': lambda t, k: 'EMERGENCY' in t
        }
        
        if context in context_patterns:
            return context_patterns[context](text, keyword)
        return False
    
    def _determine_adaptive_severity(self, pattern_info: Dict) -> str:
        """Determine severity level for adaptively discovered patterns"""
        confidence = pattern_info.get('confidence', 0)
        frequency = pattern_info.get('frequency', 0)
        
        if confidence > 0.9 or frequency > 10:
            return 'high'
        elif confidence > 0.7 or frequency > 5:
            return 'medium'
        else:
            return 'low'
    
    def _split_session_into_temporal_chunks(self, text: str, num_chunks: int = 4) -> List[str]:
        """Split session text into temporal chunks for progression analysis"""
        lines = text.split('\n')
        chunk_size = max(1, len(lines) // num_chunks)
        
        chunks = []
        for i in range(0, len(lines), chunk_size):
            chunk = '\n'.join(lines[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def _update_sentiment_learning_models(self, session: TransactionSession,
                                        sentiment_analysis: Dict,
                                        event_sequence: List[str]):
        """Update learning models based on current session analysis"""
        try:
            # Update discovered patterns based on this session
            session_text = session.raw_text.upper()
            negative_phrases = sentiment_analysis.get('negative_phrases', [])
            
            # Learn new negative patterns automatically
            for phrase in negative_phrases:
                if phrase not in self.discovered_negative_patterns.get('auto_discovered', {}):
                    self.discovered_negative_patterns.setdefault('auto_discovered', {})[phrase] = {
                        'confidence': 0.5,  # Start with moderate confidence
                        'frequency': 1,
                        'contexts': [session.session_id],
                        'discovery_date': datetime.now().isoformat()
                    }
                else:
                    # Increase frequency and confidence
                    pattern_info = self.discovered_negative_patterns['auto_discovered'][phrase]
                    pattern_info['frequency'] += 1
                    pattern_info['confidence'] = min(0.95, pattern_info['confidence'] + 0.05)
                    pattern_info['contexts'].append(session.session_id)
            
        except Exception as e:
            logger.warning(f"Error updating sentiment learning models: {str(e)}")
    
    def generate_anomaly_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive anomaly grouping and tallying report"""
        
        # Collect all anomalies across all sessions
        all_anomalies = []
        for session in self.sessions:
            for anomaly in session.anomalies:
                anomaly_data = {
                    'session_id': session.session_id,
                    'anomaly_type': anomaly.anomaly_type,
                    'severity': anomaly.severity,
                    'confidence': anomaly.confidence,
                    'detection_method': anomaly.detection_method,
                    'description': anomaly.description,
                    'timestamp': session.start_time,
                    'details': anomaly.details or {}
                }
                all_anomalies.append(anomaly_data)
        
        # Group anomalies by type
        anomaly_type_counts = {}
        anomaly_type_details = {}
        
        for anomaly in all_anomalies:
            anom_type = anomaly['anomaly_type']
            if anom_type not in anomaly_type_counts:
                anomaly_type_counts[anom_type] = 0
                anomaly_type_details[anom_type] = {
                    'count': 0,
                    'severity_breakdown': {'low': 0, 'medium': 0, 'high': 0, 'critical': 0},
                    'detection_methods': set(),
                    'avg_confidence': 0.0,
                    'sessions_affected': set(),
                    'descriptions': set(),
                    'context_details': []
                }
            
            anomaly_type_counts[anom_type] += 1
            anomaly_type_details[anom_type]['count'] += 1
            anomaly_type_details[anom_type]['severity_breakdown'][anomaly['severity']] += 1
            anomaly_type_details[anom_type]['detection_methods'].add(anomaly['detection_method'])
            anomaly_type_details[anom_type]['sessions_affected'].add(anomaly['session_id'])
            anomaly_type_details[anom_type]['descriptions'].add(anomaly['description'])
            
            # Special handling for host declines
            if anom_type == 'host_decline':
                context = anomaly['details'].get('context', {})
                anomaly_type_details[anom_type]['context_details'].append(context)
        
        # Calculate averages and convert sets to lists
        for anom_type, details in anomaly_type_details.items():
            # Calculate average confidence
            confidences = [a['confidence'] for a in all_anomalies if a['anomaly_type'] == anom_type]
            details['avg_confidence'] = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Convert sets to lists for JSON serialization
            details['detection_methods'] = list(details['detection_methods'])
            details['sessions_affected'] = list(details['sessions_affected'])
            details['descriptions'] = list(details['descriptions'])
            details['sessions_affected_count'] = len(details['sessions_affected'])
        
        # Group by severity
        severity_breakdown = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        for anomaly in all_anomalies:
            severity_breakdown[anomaly['severity']] += 1
        
        # Group by detection method
        detection_method_breakdown = {}
        for anomaly in all_anomalies:
            method = anomaly['detection_method']
            if method not in detection_method_breakdown:
                detection_method_breakdown[method] = 0
            detection_method_breakdown[method] += 1
        
        # Special analysis for host declines
        host_decline_analysis = self._analyze_host_declines(all_anomalies)
        
        # Time-based analysis (if timestamps available)
        time_analysis = self._analyze_anomalies_by_time(all_anomalies)
        
        # Generate overall statistics
        total_sessions = len(self.sessions)
        sessions_with_anomalies = len([s for s in self.sessions if len(s.anomalies) > 0])
        anomaly_rate = sessions_with_anomalies / total_sessions if total_sessions > 0 else 0
        
        summary_report = {
            'report_timestamp': datetime.now().isoformat(),
            'total_sessions_analyzed': total_sessions,
            'sessions_with_anomalies': sessions_with_anomalies,
            'total_anomalies_detected': len(all_anomalies),
            'overall_anomaly_rate': round(anomaly_rate * 100, 2),
            
            # Anomaly type breakdown
            'anomaly_type_summary': {
                'counts': anomaly_type_counts,
                'detailed_breakdown': anomaly_type_details
            },
            
            # Severity breakdown
            'severity_summary': severity_breakdown,
            
            # Detection method breakdown
            'detection_method_summary': detection_method_breakdown,
            
            # Special analysis
            'host_decline_analysis': host_decline_analysis,
            'time_based_analysis': time_analysis,
            
            # Top anomaly types (for quick reference)
            'top_anomaly_types': sorted(anomaly_type_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        }
        
        return summary_report
    
    def _analyze_host_declines(self, all_anomalies: List[Dict]) -> Dict[str, Any]:
        """Detailed analysis of host decline patterns"""
        declines = [a for a in all_anomalies if a['anomaly_type'] == 'host_decline']
        
        if not declines:
            return {'total_declines': 0, 'analysis': 'No host declines detected'}
        
        # Analyze decline reasons
        decline_reasons = {}
        decline_categories = {}
        transaction_stages = {}
        
        for decline in declines:
            context = decline['details'].get('context', {})
            reason = context.get('likely_cause', 'unknown')
            category = context.get('decline_category', 'unknown')
            stage = context.get('transaction_stage', 'unknown')
            
            decline_reasons[reason] = decline_reasons.get(reason, 0) + 1
            decline_categories[category] = decline_categories.get(category, 0) + 1
            transaction_stages[stage] = transaction_stages.get(stage, 0) + 1
        
        return {
            'total_declines': len(declines),
            'decline_reasons': decline_reasons,
            'decline_categories': decline_categories,
            'transaction_stages_when_declined': transaction_stages,
            'percentage_of_total_anomalies': round(len(declines) / len(all_anomalies) * 100, 2) if all_anomalies else 0,
            'recommendations': [
                'Monitor high-frequency decline reasons for host system issues',
                'Investigate patterns in decline categories for business impact',
                'Review transaction flow at stages with high decline rates',
                'Coordinate with host systems team on recurring decline patterns'
            ]
        }
    
    def _analyze_anomalies_by_time(self, all_anomalies: List[Dict]) -> Dict[str, Any]:
        """Analyze anomalies by time patterns"""
        anomalies_with_time = [a for a in all_anomalies if a['timestamp'] is not None]
        
        if not anomalies_with_time:
            return {'analysis': 'No timestamp data available for time-based analysis'}
        
        # Group by hour of day
        hourly_breakdown = {}
        for anomaly in anomalies_with_time:
            hour = anomaly['timestamp'].hour
            hourly_breakdown[hour] = hourly_breakdown.get(hour, 0) + 1
        
        # Find peak anomaly hours
        peak_hours = sorted(hourly_breakdown.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'anomalies_with_timestamps': len(anomalies_with_time),
            'hourly_breakdown': hourly_breakdown,
            'peak_anomaly_hours': peak_hours
        }