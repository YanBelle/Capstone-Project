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
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector
from tensorflow.keras.optimizers import Adam

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
    start_time: Optional[datetime]
    end_time: Optional[datetime]
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
    
    def __init__(self, model_name: str = 'bert-base-uncased', db_engine=None):
        # Database connection for loading labeled anomalies
        self.db_engine = db_engine
        
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
    
    def split_into_sessions(self, raw_logs: str, file_path: str = None) -> List[TransactionSession]:
        """Step 2: Split logs into transaction sessions with unique IDs
        
        Sessionization logic:
        - Start a session when encountering "TRANSACTION START" or "CARDLESS TRANSACTION START"
        - Take the session start time from the line immediately above the session start marker
        - Continue capturing all lines until the next "TRANSACTION START" or "CARDLESS TRANSACTION START" is found
        - This ensures we capture everything including post-transaction errors
        """
        logger.info("Splitting logs into transaction sessions")
        
        sessions = []
        
        # Extract file identifier for unique session IDs
        file_identifier = "unknown"
        if file_path:
            file_name = os.path.basename(file_path)
            # Extract ABM number and date from filename like ABM175EJ_20250624_20250624.txt
            file_match = re.search(r'ABM(\d+)EJ_(\d{8})_(\d{8})', file_name)
            if file_match:
                abm_num = file_match.group(1)
                start_date = file_match.group(2)
                file_identifier = f"ABM{abm_num}_{start_date}"
            else:
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
                    start_time=start_time,
                    end_time=end_time
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
                
                session = TransactionSession(
                    session_id=session_id,
                    raw_text=session_text,
                    start_time=start_time,
                    end_time=end_time
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
        """Detect specific types of anomalies based on event patterns"""
        text = session.raw_text.upper()
        
        # Hardware/Mechanical Issues
        if 'DISPENSE FAIL' in text or 'DISPENSER FAULT' in text:
            session.add_anomaly(
                anomaly_type="dispense_failure",
                confidence=0.95,
                detection_method="expert_rule",
                description="Cash dispenser failed to dispense notes",
                severity="high",
                details={"keywords": ["DISPENSE FAIL", "DISPENSER FAULT"]}
            )
        
        if any(error in text for error in ['HARDWARE ERROR', 'SENSOR ERROR', 'MOTOR ERROR', 'DEVICE ERROR']):
            session.add_anomaly(
                anomaly_type="hardware_error",
                confidence=0.90,
                detection_method="expert_rule",
                description="Hardware component malfunction detected",
                severity="high",
                details={"detected_errors": [error for error in ['HARDWARE ERROR', 'SENSOR ERROR', 'MOTOR ERROR', 'DEVICE ERROR'] if error in text]}
            )
        
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
    
    def load_labeled_anomalies_from_database(self):
        """Load labeled anomalies from database into feedback buffer for retraining"""
        if not self.db_engine:
            logger.warning("No database connection available to load labeled anomalies")
            return
        
        try:
            from sqlalchemy import text
            with self.db_engine.connect() as conn:
                # Get all labeled anomalies from database
                result = conn.execute(text("""
                    SELECT session_id, anomaly_type, label, confidence, explanation
                    FROM labeled_anomalies
                    ORDER BY created_at DESC
                """))
                
                labeled_anomalies = result.fetchall()
                
                # Convert to feedback buffer format
                for anomaly in labeled_anomalies:
                    session_id, anomaly_type, label, confidence, explanation = anomaly
                    
                    # Skip if already in feedback buffer
                    if any(f['session_id'] == session_id for f in self.feedback_buffer):
                        continue
                    
                    # Determine feedback type based on label
                    if label == 'anomaly':
                        feedback_type = 'confirmation'
                        expert_label = anomaly_type or 'unknown_anomaly'
                    else:  # label == 'normal'
                        feedback_type = 'correction'  # ML thought anomaly, expert says normal
                        expert_label = 'normal'
                    
                    # Add to feedback buffer
                    feedback_entry = {
                        'session_id': session_id,
                        'expert_label': expert_label,
                        'expert_confidence': confidence or 0.8,
                        'feedback_type': feedback_type,
                        'expert_explanation': explanation or 'Database labeled anomaly',
                        'original_ml_prediction': anomaly_type,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    self.feedback_buffer.append(feedback_entry)
                
                logger.info(f"Loaded {len(labeled_anomalies)} labeled anomalies from database into feedback buffer")
                
        except Exception as e:
            logger.error(f"Error loading labeled anomalies from database: {e}")
    
    def continuous_model_retraining(self):
        """
        Continuously retrain models based on accumulated expert feedback
        This implements the true feedback loop for unsupervised learning improvement
        """
        # First, load labeled anomalies from database
        self.load_labeled_anomalies_from_database()
        
        if len(self.feedback_buffer) < 5:  # Reduced threshold since we're loading from database
            logger.info(f"Insufficient feedback for retraining: {len(self.feedback_buffer)} samples")
            return
        
        logger.info(f"Starting continuous retraining with {len(self.feedback_buffer)} feedback samples")
        
        # Mark training start for monitoring
        mark_ml_training_start("continuous_learning")
        training_start_time = time.time()
        
        try:
            # 1. Update Isolation Forest with weighted feedback
            self._retrain_isolation_forest_with_feedback()
            
            # 2. Update One-Class SVM decision boundary
            self._retrain_svm_with_feedback()
            
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
        feedback_buffer_size = len(self.feedback_buffer)
        
        # Also check database for labeled anomalies
        db_labeled_count = 0
        if self.db_engine:
            try:
                from sqlalchemy import text
                with self.db_engine.connect() as conn:
                    result = conn.execute(text("SELECT COUNT(*) FROM labeled_anomalies"))
                    db_labeled_count = result.scalar()
            except Exception as e:
                logger.warning(f"Could not check database for labeled anomalies: {e}")
        
        total_feedback_size = feedback_buffer_size + db_labeled_count
        
        return {
            'feedback_buffer_size': total_feedback_size,  # Include both buffer and database
            'feedback_buffer_memory': feedback_buffer_size,
            'feedback_database_count': db_labeled_count,
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
        """Initialize sentiment analysis and negative text detection models"""
        logger.info("Initializing sentiment analysis and negative text detection models")
        
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
            
            logger.info("Sentiment analysis models initialized successfully")
            
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