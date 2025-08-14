"""
Unified ML-First ABM Anomaly Detection System
Consolidates both API and Anomaly Detector service implementations
Enhanced with DeepLog + BERT sequence anomaly detection
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import re
import json
import time
import pickle
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging

# ML and NLP imports
from transformers import BertTokenizer, BertModel
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Database imports
from sqlalchemy import create_engine, text

# Setup logging first
logger = logging.getLogger(__name__)

# DeepLog integration
try:
    from deeplog_service_integration import DeepLogServiceIntegration
    DEEPLOG_AVAILABLE = True
except ImportError:
    DEEPLOG_AVAILABLE = False
    logger.warning("DeepLog integration not available")

@dataclass
class TransactionSession:
    """Unified transaction session representation"""
    session_id: str
    raw_text: str
    cleaned_text: str = ""
    content: List[str] = field(default_factory=list)  # For compatibility with original sessionization
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    terminal_id: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    
    # Multi-anomaly support
    anomalies: List[Dict] = field(default_factory=list)
    overall_anomaly_score: float = 0.0
    max_severity: str = "normal"
    
    # Legacy compatibility
    is_anomaly: bool = False
    anomaly_score: float = 0.0
    anomaly_type: Optional[str] = None
    
    # Additional metadata
    processed_events: str = "[]"
    cassette_counters: Dict[str, Any] = field(default_factory=dict)
    critical_events: List[str] = field(default_factory=list)
    detected_patterns: List[str] = field(default_factory=list)
    
    def add_anomaly(self, anomaly_type: str, confidence: float = 0.5, 
                   detection_method: str = "unknown", description: str = "Anomaly detected", 
                   severity: str = "medium", details: Dict[str, Any] = None):
        """Add anomaly with unified format"""
        anomaly = {
            'anomaly_type': anomaly_type,
            'confidence': confidence,
            'detection_method': detection_method,
            'description': description,
            'severity': severity,
            'timestamp': datetime.now(),
            'details': details or {}
        }
        self.anomalies.append(anomaly)
        
        # Update session-level flags
        self.is_anomaly = True
        self.overall_anomaly_score = max(self.overall_anomaly_score, confidence)
        self._update_max_severity()
    
    def _update_max_severity(self):
        """Update max severity based on all anomalies"""
        severity_levels = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        max_level = 0
        max_severity = "normal"
        
        for anomaly in self.anomalies:
            level = severity_levels.get(anomaly.get('severity', 'medium'), 2)
            if level > max_level:
                max_level = level
                max_severity = anomaly.get('severity', 'medium')
        
        self.max_severity = max_severity


class UnifiedMLAnomalyDetector:
    """
    Unified ML-First Anomaly Detector
    Consolidates API and Anomaly Detector service implementations
    """
    
    def __init__(self, model_name: str = 'bert-base-uncased', db_engine=None, 
                 service_mode: str = "api"):
        """
        Initialize unified ML analyzer
        
        Args:
            model_name: BERT model name
            db_engine: Database connection (for anomaly-detector service)
            service_mode: "api" or "anomaly-detector" for service-specific features
        """
        self.model_name = model_name
        self.db_engine = db_engine
        self.service_mode = service_mode
        
        # Initialize BERT components
        self._initialize_bert()
        
        # Initialize ML models
        self._initialize_ml_models()
        
        # Initialize service-specific components
        if service_mode == "anomaly-detector":
            self._initialize_anomaly_detector_features()
        elif service_mode == "api":
            self._initialize_api_features()
        
        # Shared components
        self.sessions: List[TransactionSession] = []
        self.embeddings_matrix = None
        self.expert_rules = self._load_expert_rules()
        
        # Initialize cassette counter patterns
        self._initialize_cassette_patterns()
        
    def _initialize_bert(self):
        """Initialize BERT components with error handling"""
        try:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.bert_model = BertModel.from_pretrained(self.model_name)
            self.bert_model.eval()
            logger.info(f"BERT model {self.model_name} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize BERT: {e}")
            self.tokenizer = None
            self.bert_model = None
    
    def _initialize_ml_models(self):
        """Initialize ML models with unified parameters"""
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=50)
        
        # Isolation Forest with optimized parameters
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        
        # One-Class SVM with optimized parameters
        self.one_class_svm = OneClassSVM(
            kernel='rbf',
            gamma='auto',
            nu=0.05
        )
        
        # DBSCAN for density-based detection
        self.dbscan = DBSCAN(
            eps=0.5,
            min_samples=3,
            metric='cosine'
        )
        
        # Supervised model (loaded if available)
        self.supervised_classifier = None
        self.label_encoder = None
        self._load_supervised_model()
    
    def _initialize_anomaly_detector_features(self):
        """Initialize features specific to anomaly-detector service"""
        # Enhanced unsupervised analyzer
        try:
            from enhanced_unsupervised_analyzer import EnhancedUnsupervisedEJAnalyzer
            self.unsupervised_analyzer = EnhancedUnsupervisedEJAnalyzer()
        except ImportError:
            self.unsupervised_analyzer = None
        
        # Continuous learning system
        self.feedback_buffer = []
        self.learning_threshold = 50
    
    def _initialize_api_features(self):
        """Initialize features specific to API service"""
        # API-specific features like caching, simplified processing
        self.cache_embeddings = True
        self.max_sessions_per_batch = 1000
    
    def _initialize_cassette_patterns(self):
        """Initialize cassette counter parsing patterns"""
        self.cassette_patterns = {
            'cassette_1': re.compile(r'(?:CASSETTE\s*1|CAS\s*1).*?(?:COUNT|CNT).*?(\d+)', re.IGNORECASE),
            'cassette_2': re.compile(r'(?:CASSETTE\s*2|CAS\s*2).*?(?:COUNT|CNT).*?(\d+)', re.IGNORECASE),
            'cassette_3': re.compile(r'(?:CASSETTE\s*3|CAS\s*3).*?(?:COUNT|CNT).*?(\d+)', re.IGNORECASE),
            'cassette_4': re.compile(r'(?:CASSETTE\s*4|CAS\s*4).*?(?:COUNT|CNT).*?(\d+)', re.IGNORECASE),
            'total_notes': re.compile(r'TOTAL.*?NOTES.*?(\d+)', re.IGNORECASE),
            'dispensed': re.compile(r'DISPENSED.*?(\d+)', re.IGNORECASE),
            'rejected': re.compile(r'REJECTED.*?(\d+)', re.IGNORECASE)
        }
    
    def _load_expert_rules(self):
        """Load expert rules for anomaly detection"""
        return {
            'supervisor_mode': [
                r'SUPERVISOR\s+MODE',
                r'ADMIN\s+ACCESS',
                r'MAINTENANCE\s+MODE'
            ],
            'cash_errors': [
                r'UNABLE\s+TO\s+DISPENSE',
                r'CASH\s+RETRACT',
                r'NOTES?\s+JAM',
                r'DISPENSE\s+ERROR'
            ],
            'device_errors': [
                r'DEVICE\s+ERROR',
                r'HARDWARE\s+FAULT',
                r'SENSOR\s+ERROR'
            ],
            'power_issues': [
                r'POWER\s+RESET',
                r'UPS\s+FAILURE',
                r'BATTERY\s+LOW'
            ]
        }
    
    def _load_supervised_model(self):
        """Load supervised model if available"""
        try:
            model_path = "/app/models/supervised_classifier.pkl"
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.supervised_classifier = model_data.get('classifier')
                    self.label_encoder = model_data.get('label_encoder')
                logger.info("Supervised model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load supervised model: {e}")
    
    def process_ej_logs(self, file_path: str) -> pd.DataFrame:
        """
        UNIFIED entry point for processing EJ logs
        Consolidates both service implementations
        """
        logger.info(f"Processing EJ logs from {file_path} in {self.service_mode} mode")
        
        # Step 1: Read and split into sessions (UNIFIED)
        raw_logs = self._read_raw_logs(file_path)
        self.sessions = self.split_into_sessions(raw_logs, file_path)
        
        if len(self.sessions) == 0:
            logger.warning("No sessions found")
            return pd.DataFrame()
        
        # Limit sessions for performance if in API mode
        if self.service_mode == "api" and len(self.sessions) > self.max_sessions_per_batch:
            logger.info(f"API mode: limiting to {self.max_sessions_per_batch} sessions")
            self.sessions = self.sessions[:self.max_sessions_per_batch]
        
        # Step 2: Generate embeddings (UNIFIED)
        self.embeddings_matrix = self._generate_embeddings_unified(self.sessions)
        
        # Step 3: Anomaly detection (UNIFIED)
        self._detect_anomalies_unified()
        
        # Step 4: Service-specific processing
        if self.service_mode == "anomaly-detector":
            self._anomaly_detector_post_processing()
        elif self.service_mode == "api":
            self._api_post_processing()
        
        # Step 5: Create results dataframe (UNIFIED)
        results_df = self._create_results_dataframe()
        
        logger.info(f"Processing complete: {len(self.sessions)} sessions, "
                   f"{results_df['is_anomaly'].sum()} anomalies")
        
        return results_df
    
    def split_into_sessions(self, raw_logs: str, file_path: str = None) -> List[TransactionSession]:
        """
        UNIFIED sessionization logic
        Consolidates both implementations with best practices from each
        """
        logger.info("Splitting logs into transaction sessions (UNIFIED)")
        
        sessions = []
        
        # Extract terminal ID from filename (unified approach)
        terminal_id, file_identifier = self._extract_file_metadata(file_path)
        
        # Split by transaction boundaries (consolidated logic)
        log_lines = raw_logs.split('\n')
        
        # Find transaction start patterns
        transaction_pattern = re.compile(
            r'(\*(?:TRANSACTION|CARDLESS TRANSACTION)\s+START\*)',
            re.IGNORECASE
        )
        
        start_positions = []
        for line_num, line in enumerate(log_lines):
            if transaction_pattern.search(line):
                start_positions.append(line_num)
        
        if not start_positions:
            logger.warning("No transaction start markers found")
            return sessions
        
        # Create sessions from boundaries
        timestamp_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, start_pos in enumerate(start_positions):
            end_pos = start_positions[i + 1] if i + 1 < len(start_positions) else len(log_lines)
            
            # Extract session content
            session_lines = log_lines[start_pos:end_pos]
            session_content = '\n'.join(session_lines)
            
            if len(session_content.strip()) < 10:
                continue
            
            # Create session with unified format
            session_id = f"{file_identifier}_txn_{i+1:04d}_{timestamp_suffix}"
            
            # Extract timestamps
            start_time = self._extract_timestamp_from_lines(session_lines[:5])
            end_time = self._extract_timestamp_from_lines(session_lines[-5:])
            
            # Apply text cleaning (unified approach)
            cleaned_content = self._apply_unified_cleaning(session_content)
            
            # Parse cassette counters from this session
            cassette_counters = self.parse_cassette_counters(session_content)
            
            # Extract critical events and patterns
            critical_events = self._extract_critical_events(session_content)
            detected_patterns = self._extract_patterns(session_content)
            
            session = TransactionSession(
                session_id=session_id,
                raw_text=session_content,
                cleaned_text=cleaned_content,
                content=session_lines,  # For compatibility
                start_time=start_time,
                end_time=end_time,
                terminal_id=terminal_id,
                cassette_counters=cassette_counters,
                critical_events=critical_events,
                detected_patterns=detected_patterns
            )
            
            sessions.append(session)
        
        logger.info(f"Created {len(sessions)} sessions using unified sessionization")
        return sessions
    
    def parse_cassette_counters(self, session_text: str) -> Dict[str, Any]:
        """
        Parse cassette counter information from session text
        Maintains compatibility with existing cassette counter functionality
        """
        counters = {}
        
        try:
            # Extract cassette counts using patterns
            for counter_name, pattern in self.cassette_patterns.items():
                matches = pattern.findall(session_text)
                if matches:
                    # Take the last match as it's likely the most recent count
                    counters[counter_name] = int(matches[-1])
            
            # Calculate derived metrics
            if 'cassette_1' in counters and 'cassette_2' in counters:
                counters['total_cassettes'] = counters['cassette_1'] + counters['cassette_2']
                if 'cassette_3' in counters:
                    counters['total_cassettes'] += counters['cassette_3']
                if 'cassette_4' in counters:
                    counters['total_cassettes'] += counters['cassette_4']
            
            # Extract dispense information
            dispense_pattern = re.compile(r'DISPENSE.*?(\d+).*?NOTES?', re.IGNORECASE)
            dispense_matches = dispense_pattern.findall(session_text)
            if dispense_matches:
                counters['dispensed_this_transaction'] = int(dispense_matches[-1])
            
            # Extract denomination information
            denom_pattern = re.compile(r'(\d+)\s*(?:DOLLAR|USD|PHP)', re.IGNORECASE)
            denom_matches = denom_pattern.findall(session_text)
            if denom_matches:
                counters['denominations'] = [int(d) for d in denom_matches]
            
            logger.debug(f"Parsed cassette counters: {counters}")
            
        except Exception as e:
            logger.error(f"Error parsing cassette counters: {e}")
            counters = {}
        
        return counters
    
    def _extract_file_metadata(self, file_path: str) -> Tuple[Optional[str], str]:
        """Extract terminal ID and file identifier from filename"""
        terminal_id = None
        file_identifier = "unknown"
        
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
                logger.warning(f"Could not extract terminal ID from filename: {file_name}")
                file_identifier = file_name.replace('.txt', '').replace('.', '_')
        
        return terminal_id, file_identifier
    
    def _read_raw_logs(self, file_path: str) -> str:
        """Read raw logs from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return ""
    
    def _extract_timestamp_from_lines(self, lines: List[str]) -> Optional[datetime]:
        """Extract timestamp from log lines"""
        timestamp_pattern = re.compile(r'(\d{4}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2}:\d{2})')
        
        for line in lines:
            match = timestamp_pattern.search(line)
            if match:
                try:
                    timestamp_str = match.group(1)
                    # Try different timestamp formats
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S']:
                        try:
                            return datetime.strptime(timestamp_str, fmt)
                        except ValueError:
                            continue
                except Exception as e:
                    logger.debug(f"Could not parse timestamp {match.group(1)}: {e}")
        
        return None
    
    def _apply_unified_cleaning(self, text: str) -> str:
        """Apply unified text cleaning from both implementations"""
        try:
            # Try BertViz cleaning if available
            from bertviz_analyzer import BertVisualizationAnalyzer
            bert_analyzer = BertVisualizationAnalyzer()
            return bert_analyzer._preprocess_text(text)
        except ImportError:
            # Fallback to basic cleaning
            return self._apply_basic_cleaning(text)
        except Exception as e:
            logger.warning(f"BertViz cleaning failed: {e}, using basic cleaning")
            return self._apply_basic_cleaning(text)
    
    def _apply_basic_cleaning(self, text: str) -> str:
        """Apply basic text cleaning"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\-\.\,\:\;\(\)\*\[\]]', ' ', text)
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        return text.strip()
    
    def _extract_critical_events(self, session_text: str) -> List[str]:
        """Extract critical events from session text"""
        critical_events = []
        
        # Define critical event patterns
        critical_patterns = [
            r'ERROR',
            r'FAULT',
            r'FAILURE',
            r'TIMEOUT',
            r'UNABLE\s+TO',
            r'CASH\s+RETRACT',
            r'DISPENSE\s+FAIL',
            r'SUPERVISOR\s+MODE'
        ]
        
        for pattern in critical_patterns:
            matches = re.findall(pattern, session_text, re.IGNORECASE)
            critical_events.extend(matches)
        
        return list(set(critical_events))  # Remove duplicates
    
    def _extract_patterns(self, session_text: str) -> List[str]:
        """Extract detected patterns from session text"""
        patterns = []
        
        for pattern_name, pattern_list in self.expert_rules.items():
            for pattern in pattern_list:
                if re.search(pattern, session_text, re.IGNORECASE):
                    patterns.append(pattern_name)
                    break  # Only add pattern name once
        
        return patterns
    
    def _generate_embeddings_unified(self, sessions: List[TransactionSession]) -> np.ndarray:
        """
        UNIFIED embedding generation
        Primary: BERT, Fallback: Sentence Transformers, Final fallback: TF-IDF
        """
        logger.info(f"Generating embeddings for {len(sessions)} sessions (UNIFIED)")
        
        # Try BERT first (highest quality)
        if self.bert_model is not None:
            try:
                return self._generate_bert_embeddings(sessions)
            except Exception as e:
                logger.warning(f"BERT failed: {e}, trying Sentence Transformers")
        
        # Fallback to Sentence Transformers
        try:
            return self._generate_sentence_embeddings(sessions)
        except Exception as e:
            logger.warning(f"Sentence Transformers failed: {e}, using TF-IDF")
        
        # Final fallback to TF-IDF
        return self._generate_tfidf_embeddings(sessions)
    
    def _generate_bert_embeddings(self, sessions: List[TransactionSession]) -> np.ndarray:
        """Generate BERT embeddings for sessions"""
        embeddings = []
        
        for session in sessions:
            try:
                # Use cleaned text for embedding
                text = session.cleaned_text[:512]  # BERT max length
                
                # Tokenize and encode
                inputs = self.tokenizer(text, return_tensors='pt', 
                                      truncation=True, padding=True, max_length=512)
                
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    # Use [CLS] token embedding
                    embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
                
                embeddings.append(embedding)
                
            except Exception as e:
                logger.error(f"Error generating BERT embedding for session {session.session_id}: {e}")
                # Fallback to zero embedding
                embeddings.append(np.zeros(768))  # BERT base hidden size
        
        return np.array(embeddings)
    
    def _generate_sentence_embeddings(self, sessions: List[TransactionSession]) -> np.ndarray:
        """Generate Sentence Transformer embeddings"""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            texts = [session.cleaned_text for session in sessions]
            embeddings = model.encode(texts)
            return embeddings
            
        except ImportError:
            logger.warning("Sentence Transformers not available")
            raise
    
    def _generate_tfidf_embeddings(self, sessions: List[TransactionSession]) -> np.ndarray:
        """Generate TF-IDF embeddings as final fallback"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        texts = [session.cleaned_text for session in sessions]
        vectorizer = TfidfVectorizer(max_features=512, stop_words='english')
        embeddings = vectorizer.fit_transform(texts).toarray()
        
        return embeddings
    
    def _detect_anomalies_unified(self):
        """
        UNIFIED anomaly detection
        Uses ensemble of all three methods: Isolation Forest, One-Class SVM, DBSCAN
        """
        logger.info("Running unified anomaly detection")
        
        if self.embeddings_matrix is None or len(self.embeddings_matrix) == 0:
            logger.warning("No embeddings available for anomaly detection")
            return
        
        # Scale embeddings
        embeddings_scaled = self.scaler.fit_transform(self.embeddings_matrix)
        
        # Apply PCA if enough samples
        if len(self.sessions) > 50:
            embeddings_scaled = self.pca.fit_transform(embeddings_scaled)
        
        # Run all three algorithms
        if_predictions = self.isolation_forest.fit_predict(embeddings_scaled)
        if_scores = self.isolation_forest.score_samples(embeddings_scaled)
        
        svm_predictions = self.one_class_svm.fit_predict(embeddings_scaled)
        svm_scores = self.one_class_svm.decision_function(embeddings_scaled)
        
        # Optimize DBSCAN parameters if enough data
        if len(self.sessions) >= 20:
            self._optimize_dbscan_parameters(embeddings_scaled)
        
        dbscan_labels = self.dbscan.fit_predict(embeddings_scaled)
        dbscan_predictions = np.where(dbscan_labels == -1, -1, 1)
        
        # Apply results to sessions
        for i, session in enumerate(self.sessions):
            self._apply_anomaly_results_to_session(
                session, if_predictions[i], svm_predictions[i], dbscan_predictions[i],
                if_scores[i], svm_scores[i], dbscan_labels[i]
            )
        
        # Apply expert rules and pattern detection
        self._apply_expert_rules_and_patterns()
    
    def _optimize_dbscan_parameters(self, embeddings_scaled: np.ndarray):
        """Optimize DBSCAN parameters based on data"""
        try:
            # Try different eps values and choose the one with best silhouette score
            best_eps = 0.5
            best_score = -1
            
            for eps in [0.3, 0.4, 0.5, 0.6, 0.7]:
                dbscan_test = DBSCAN(eps=eps, min_samples=3, metric='cosine')
                labels = dbscan_test.fit_predict(embeddings_scaled)
                
                # Check if we have valid clusters
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters > 1:
                    try:
                        score = silhouette_score(embeddings_scaled, labels)
                        if score > best_score:
                            best_score = score
                            best_eps = eps
                    except ValueError:
                        continue
            
            # Update DBSCAN with optimized parameters
            self.dbscan = DBSCAN(eps=best_eps, min_samples=3, metric='cosine')
            logger.info(f"Optimized DBSCAN eps to {best_eps} (silhouette score: {best_score:.3f})")
            
        except Exception as e:
            logger.warning(f"DBSCAN optimization failed: {e}")
    
    def _apply_anomaly_results_to_session(self, session: TransactionSession, 
                                        if_pred: int, svm_pred: int, dbscan_pred: int,
                                        if_score: float, svm_score: float, dbscan_label: int):
        """Apply anomaly detection results to session"""
        # Count how many algorithms flagged as anomaly
        anomaly_votes = sum([if_pred == -1, svm_pred == -1, dbscan_pred == -1])
        
        # Session is anomaly if majority vote or any critical pattern
        session.is_anomaly = (anomaly_votes >= 2) or len(session.detected_patterns) > 0
        
        # Calculate overall anomaly score
        scores = []
        if if_pred == -1:
            scores.append(abs(if_score))
        if svm_pred == -1:
            scores.append(abs(svm_score))
        if dbscan_pred == -1:
            scores.append(0.7)  # Fixed score for density-based anomaly
        
        session.anomaly_score = np.mean(scores) if scores else 0.0
        session.overall_anomaly_score = session.anomaly_score
        
        # Determine anomaly type based on detection method
        if anomaly_votes >= 2:
            session.anomaly_type = "statistical"
            session.add_anomaly(
                "statistical", 
                confidence=session.anomaly_score,
                detection_method="ensemble",
                description=f"Detected by {anomaly_votes}/3 algorithms"
            )
        
        # Add pattern-based anomalies
        for pattern in session.detected_patterns:
            session.add_anomaly(
                pattern,
                confidence=0.8,
                detection_method="expert_rules",
                description=f"Pattern: {pattern}",
                severity="high" if pattern in ['supervisor_mode', 'power_issues'] else "medium"
            )
    
    def _apply_expert_rules_and_patterns(self):
        """Apply expert rules and pattern-based detection"""
        for session in self.sessions:
            # Apply rule-based detection
            for pattern_name, patterns in self.expert_rules.items():
                for pattern in patterns:
                    if re.search(pattern, session.raw_text, re.IGNORECASE):
                        if pattern_name not in session.detected_patterns:
                            session.detected_patterns.append(pattern_name)
                        break
    
    def _anomaly_detector_post_processing(self):
        """Post-processing specific to anomaly-detector service"""
        # Enhanced analysis with unsupervised analyzer if available
        if self.unsupervised_analyzer:
            try:
                for session in self.sessions:
                    enhanced_analysis = self.unsupervised_analyzer.analyze_session(session.raw_text)
                    if enhanced_analysis.get('is_anomaly', False):
                        session.add_anomaly(
                            "enhanced_unsupervised",
                            confidence=enhanced_analysis.get('confidence', 0.5),
                            detection_method="enhanced_unsupervised",
                            description="Enhanced unsupervised analysis"
                        )
            except Exception as e:
                logger.warning(f"Enhanced unsupervised analysis failed: {e}")
    
    def _api_post_processing(self):
        """Post-processing specific to API service"""
        # API-specific optimizations
        # Limit detailed analysis for performance
        pass
    
    def _create_results_dataframe(self) -> pd.DataFrame:
        """Create results dataframe from processed sessions"""
        results = []
        
        for session in self.sessions:
            # Calculate session length (number of characters in raw text)
            session_length = len(session.raw_text) if session.raw_text else 0
            
            result = {
                'session_id': session.session_id,
                'terminal_id': session.terminal_id,
                'start_time': session.start_time,
                'end_time': session.end_time,
                'session_length': session_length,  # Add session_length field
                'raw_text': session.raw_text,
                'cleaned_text': session.cleaned_text,
                'processed_events': session.processed_events,
                'is_anomaly': session.is_anomaly,
                'anomaly_score': session.anomaly_score,
                'anomaly_type': session.anomaly_type,
                'overall_anomaly_score': session.overall_anomaly_score,
                'max_severity': session.max_severity,
                'detected_patterns': session.detected_patterns,
                'critical_events': session.critical_events,
                'cassette_counters': session.cassette_counters,
                'anomalies': session.anomalies,
                'embedding_vector': session.embedding.tolist() if session.embedding is not None else None
            }
            results.append(result)
        
        return pd.DataFrame(results)
    
    def store_unified_sessions(self, results_df: pd.DataFrame, source_file: str = None) -> Dict[str, int]:
        """
        Store sessions using the preferred API approach with cleaned_text and raw_text
        Plus terminal_id detection from anomaly-detector approach
        """
        if self.db_engine is None:
            logger.warning("No database engine available for storing sessions")
            return {'success_count': 0, 'duplicate_count': 0, 'error_count': len(results_df)}
        
        success_count = 0
        duplicate_count = 0
        error_count = 0
        
        for _, row in results_df.iterrows():
            try:
                with self.db_engine.connect() as conn:
                    # Check if session already exists
                    existing = conn.execute(
                        text("SELECT session_id FROM ml_sessions WHERE session_id = :session_id"),
                        {"session_id": row['session_id']}
                    ).fetchone()
                    
                    if existing:
                        # Update existing session
                        conn.execute(text("""
                            UPDATE ml_sessions 
                            SET raw_text = :raw_text, 
                                cleaned_text = :cleaned_text, 
                                processed_events = :processed_events,
                                terminal_id = :terminal_id,
                                is_anomaly = :is_anomaly,
                                anomaly_score = :anomaly_score,
                                anomaly_type = :anomaly_type,
                                detected_patterns = :detected_patterns,
                                critical_events = :critical_events,
                                cassette_counters = :cassette_counters,
                                embedding_vector = :embedding_vector,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE session_id = :session_id
                        """), {
                            "session_id": row['session_id'],
                            "raw_text": row['raw_text'],
                            "cleaned_text": row['cleaned_text'],
                            "processed_events": row['processed_events'],
                            "terminal_id": row['terminal_id'],
                            "is_anomaly": row['is_anomaly'],
                            "anomaly_score": float(row['anomaly_score']),
                            "anomaly_type": row['anomaly_type'],
                            "detected_patterns": json.dumps(row['detected_patterns']),
                            "critical_events": json.dumps(row['critical_events']),
                            "cassette_counters": json.dumps(row['cassette_counters']),
                            "embedding_vector": json.dumps(row['embedding_vector']) if row['embedding_vector'] else None
                        })
                        conn.commit()
                        duplicate_count += 1
                        logger.debug(f"Updated existing session {row['session_id']}")
                    else:
                        # Create new session
                        conn.execute(text("""
                            INSERT INTO ml_sessions 
                            (session_id, terminal_id, raw_text, cleaned_text, processed_events,
                             is_anomaly, anomaly_score, anomaly_type, detected_patterns, 
                             critical_events, cassette_counters, embedding_vector, 
                             source_file, created_at)
                            VALUES 
                            (:session_id, :terminal_id, :raw_text, :cleaned_text, :processed_events,
                             :is_anomaly, :anomaly_score, :anomaly_type, :detected_patterns,
                             :critical_events, :cassette_counters, :embedding_vector,
                             :source_file, CURRENT_TIMESTAMP)
                        """), {
                            "session_id": row['session_id'],
                            "terminal_id": row['terminal_id'],
                            "raw_text": row['raw_text'],
                            "cleaned_text": row['cleaned_text'],
                            "processed_events": row['processed_events'],
                            "is_anomaly": row['is_anomaly'],
                            "anomaly_score": float(row['anomaly_score']),
                            "anomaly_type": row['anomaly_type'],
                            "detected_patterns": json.dumps(row['detected_patterns']),
                            "critical_events": json.dumps(row['critical_events']),
                            "cassette_counters": json.dumps(row['cassette_counters']),
                            "embedding_vector": json.dumps(row['embedding_vector']) if row['embedding_vector'] else None,
                            "source_file": source_file
                        })
                        conn.commit()
                        success_count += 1
                        logger.debug(f"Created new session {row['session_id']}")
                        
            except Exception as e:
                logger.error(f"Error storing session {row['session_id']}: {e}")
                error_count += 1
        
        logger.info(f"Session storage complete - New: {success_count}, Updated: {duplicate_count}, Errors: {error_count}")
        return {
            'success_count': success_count,
            'duplicate_count': duplicate_count,
            'error_count': error_count
        }


# Factory function for easy instantiation
def create_unified_analyzer(service_mode: str = "api", db_engine=None, **kwargs) -> UnifiedMLAnomalyDetector:
    """
    Factory function to create unified ML analyzer
    
    Args:
        service_mode: "api" or "anomaly-detector"
        db_engine: Database engine (required for anomaly-detector mode)
        **kwargs: Additional arguments passed to analyzer
    
    Returns:
        Configured UnifiedMLAnomalyDetector instance
    """
    return UnifiedMLAnomalyDetector(
        service_mode=service_mode,
        db_engine=db_engine,
        **kwargs
    )
