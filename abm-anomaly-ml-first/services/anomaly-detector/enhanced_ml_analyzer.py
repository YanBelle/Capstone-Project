"""
Enhanced ML-First ABM Anomaly Detection with Knowledge-Guided Learning
Replaces rigid expert overrides with collaborative knowledge-ML system
"""

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

# NLP and ML imports
from transformers import BertTokenizer, BertModel
import torch
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector
from collections import Counter
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class KnowledgeSignal:
    """Represents domain knowledge signals that influence ML confidence"""
    signal_type: str  # 'normal_pattern', 'error_pattern', 'variation', 'novel'
    confidence: float  # How confident we are in this signal
    description: str
    pattern_elements: List[str]
    adjustment_factor: float  # How much this should adjust ML confidence


@dataclass
class EnhancedAnomalyDetection:
    """Enhanced anomaly detection with knowledge integration"""
    anomaly_type: str
    ml_confidence: float
    knowledge_adjusted_confidence: float
    detection_method: str
    description: str
    severity: str
    knowledge_signals: List[KnowledgeSignal] = field(default_factory=list)
    supporting_evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None


@dataclass
class EnhancedTransactionSession:
    """Enhanced transaction session with knowledge-guided analysis"""
    session_id: str
    raw_text: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    embedding: Optional[np.ndarray] = None
    
    # Enhanced anomaly detection
    anomaly_detections: List[EnhancedAnomalyDetection] = field(default_factory=list)
    overall_ml_score: float = 0.0
    overall_knowledge_adjusted_score: float = 0.0
    confidence_in_assessment: float = 0.0
    
    # Pattern analysis
    detected_base_patterns: List[str] = field(default_factory=list)
    pattern_variations: List[str] = field(default_factory=list)
    novel_elements: List[str] = field(default_factory=list)
    
    # Legacy compatibility
    is_anomaly: bool = False
    anomaly_score: float = 0.0
    anomaly_type: Optional[str] = None


class KnowledgeGuidedMLAnalyzer:
    """Enhanced ML-First Anomaly Detector with Knowledge Guidance"""
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        logger.info("Initializing Knowledge-Guided ML Analyzer")
        
        # Initialize BERT models
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert_model = BertModel.from_pretrained(model_name)
        self.bert_model.eval()
        
        # Initialize unsupervised ML models
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
        self.one_class_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=50)
        
        # Knowledge base for pattern recognition
        self.knowledge_base = self.initialize_knowledge_base()
        
        # Pattern similarity analyzer
        self.pattern_analyzer = PatternSimilarityAnalyzer()
        
        # Temporal analysis models
        self.temporal_analyzer = TemporalAnomalyAnalyzer()
        
        # Sequence analysis models  
        self.sequence_analyzer = SequenceAnomalyAnalyzer()
        
        # Storage
        self.sessions: List[EnhancedTransactionSession] = []
        self.embeddings_matrix = None
        
        logger.info("Knowledge-Guided ML Analyzer initialized successfully")
    
    def initialize_knowledge_base(self) -> Dict:
        """Initialize knowledge base with base patterns and their characteristics"""
        return {
            'base_patterns': {
                'successful_withdrawal': {
                    'core_sequence': ['CARD_INSERTED', 'PIN_ENTERED', 'NOTES_PRESENTED', 'NOTES_TAKEN', 'CARD_TAKEN'],
                    'required_elements': ['NOTES_PRESENTED', 'NOTES_TAKEN'],
                    'confidence_threshold': 0.85,
                    'is_normal': True,
                    'confidence_adjustment': 0.2  # Reduce anomaly confidence by 80%
                },
                'successful_inquiry': {
                    'core_sequence': ['CARD_INSERTED', 'PIN_ENTERED', 'BALANCE_INQUIRY', 'CARD_TAKEN'],
                    'required_elements': ['CARD_INSERTED', 'CARD_TAKEN'],
                    'confidence_threshold': 0.80,
                    'is_normal': True,
                    'confidence_adjustment': 0.3
                },
                'dispense_failure': {
                    'core_sequence': ['DISPENSE_COMMAND', 'UNABLE_TO_DISPENSE'],
                    'required_elements': ['UNABLE_TO_DISPENSE'],
                    'confidence_threshold': 0.90,
                    'is_normal': False,
                    'confidence_adjustment': 1.5  # Increase anomaly confidence by 50%
                },
                'hardware_error': {
                    'core_sequence': ['DEVICE_ERROR', 'HARDWARE_FAULT'],
                    'required_elements': ['DEVICE_ERROR', 'HARDWARE_FAULT', 'SENSOR_ERROR'],
                    'confidence_threshold': 0.85,
                    'is_normal': False,
                    'confidence_adjustment': 1.4
                },
                'supervisor_intervention': {
                    'core_sequence': ['SUPERVISOR_MODE_ENTRY'],
                    'required_elements': ['SUPERVISOR_MODE'],
                    'confidence_threshold': 0.75,
                    'is_normal': None,  # Context dependent
                    'confidence_adjustment': 1.0  # No adjustment, let ML decide
                }
            },
            
            'pattern_variations': {
                'timing_variations': ['delayed_response', 'quick_sequence', 'timeout_occurred'],
                'retry_patterns': ['retry_attempt', 'multiple_attempts', 'recovery_action'],
                'error_variations': ['temporary_error', 'persistent_error', 'cascading_errors'],
                'user_variations': ['user_canceled', 'user_timeout', 'multiple_pin_attempts']
            },
            
            'contextual_indicators': {
                'maintenance_context': ['POWER_UP_RESET', 'SYSTEM_STARTUP', 'CASSETTE_REPLENISHED'],
                'operational_context': ['BUSY_PERIOD', 'LOW_CASH', 'END_OF_DAY'],
                'security_context': ['MULTIPLE_FAILURES', 'SUSPICIOUS_ACTIVITY', 'UNAUTHORIZED_ACCESS']
            }
        }
    
    def process_ej_logs_enhanced(self, file_path: str) -> pd.DataFrame:
        """Enhanced processing with knowledge-guided ML analysis"""
        logger.info(f"Processing EJ logs with knowledge-guided analysis: {file_path}")
        
        # Step 1: Read and sessionize
        raw_logs = self.read_raw_logs(file_path)
        self.sessions = self.split_into_enhanced_sessions(raw_logs, file_path)
        logger.info(f"Found {len(self.sessions)} transaction sessions")
        
        if len(self.sessions) == 0:
            logger.warning("No transaction sessions found")
            return pd.DataFrame()
        
        # Limit for performance during development
        if len(self.sessions) > 2000:
            logger.info(f"Processing first 2000 sessions for performance")
            self.sessions = self.sessions[:2000]
        
        # Step 2: Generate embeddings
        self.embeddings_matrix = self.generate_embeddings(self.sessions)
        
        # Step 3: Knowledge-guided pattern analysis
        self.analyze_patterns_with_knowledge()
        
        # Step 4: ML anomaly detection with knowledge adjustment
        self.detect_anomalies_with_knowledge_guidance()
        
        # Step 5: Temporal and sequence analysis
        self.perform_advanced_analysis()
        
        # Step 6: Final assessment integration
        self.integrate_all_assessments()
        
        # Step 7: Create enhanced results
        results_df = self.create_enhanced_results_dataframe()
        
        # Log comprehensive summary
        self.log_analysis_summary(results_df)
        
        return results_df
    
    def analyze_patterns_with_knowledge(self):
        """Analyze each session for known patterns and variations"""
        logger.info("Analyzing patterns with knowledge base")
        
        for session in self.sessions:
            pattern_analysis = self.pattern_analyzer.analyze_session_patterns(
                session, self.knowledge_base
            )
            
            session.detected_base_patterns = pattern_analysis['base_patterns']
            session.pattern_variations = pattern_analysis['variations']
            session.novel_elements = pattern_analysis['novel_elements']
    
    def detect_anomalies_with_knowledge_guidance(self):
        """ML anomaly detection with knowledge-based confidence adjustment"""
        logger.info("Running ML anomaly detection with knowledge guidance")
        
        if self.embeddings_matrix is None or len(self.embeddings_matrix) == 0:
            logger.warning("No embeddings available for anomaly detection")
            return
        
        # Scale embeddings
        embeddings_scaled = self.scaler.fit_transform(self.embeddings_matrix)
        
        # Apply PCA if sufficient samples
        if len(self.sessions) > 50:
            embeddings_scaled = self.pca.fit_transform(embeddings_scaled)
        
        # Run ML models
        if_predictions = self.isolation_forest.fit_predict(embeddings_scaled)
        if_scores = self.isolation_forest.score_samples(embeddings_scaled)
        
        svm_predictions = self.one_class_svm.fit_predict(embeddings_scaled)
        svm_scores = self.one_class_svm.decision_function(embeddings_scaled)
        
        # Process each session with knowledge guidance
        for i, session in enumerate(self.sessions):
            # Normalize ML scores
            if_score_norm = self.normalize_score(if_scores[i], if_scores)
            svm_score_norm = self.normalize_score(svm_scores[i], svm_scores)
            
            # Combine ML scores
            combined_ml_score = max(1.0 - if_score_norm, 1.0 - svm_score_norm)
            session.overall_ml_score = combined_ml_score
            
            # Apply knowledge-guided adjustment
            knowledge_adjustment = self.calculate_knowledge_adjustment(session)
            
            session.overall_knowledge_adjusted_score = min(1.0, 
                combined_ml_score * knowledge_adjustment['adjustment_factor']
            )
            
            session.confidence_in_assessment = knowledge_adjustment['confidence']
            
            # Create enhanced anomaly detection if score is significant
            if session.overall_knowledge_adjusted_score > 0.3:
                detection = EnhancedAnomalyDetection(
                    anomaly_type=self.determine_anomaly_type(session),
                    ml_confidence=combined_ml_score,
                    knowledge_adjusted_confidence=session.overall_knowledge_adjusted_score,
                    detection_method="knowledge_guided_ml",
                    description=self.generate_anomaly_description(session, knowledge_adjustment),
                    severity=self.determine_severity(session.overall_knowledge_adjusted_score),
                    knowledge_signals=knowledge_adjustment['signals'],
                    supporting_evidence=knowledge_adjustment['evidence'],
                    timestamp=datetime.now()
                )
                session.anomaly_detections.append(detection)
            
            # Update legacy fields
            session.is_anomaly = session.overall_knowledge_adjusted_score > 0.5
            session.anomaly_score = session.overall_knowledge_adjusted_score
            session.anomaly_type = self.determine_anomaly_type(session) if session.is_anomaly else None
    
    def calculate_knowledge_adjustment(self, session: EnhancedTransactionSession) -> Dict:
        """Calculate how domain knowledge should adjust ML confidence"""
        
        adjustment_factor = 1.0
        confidence = 0.5  # Base confidence in our assessment
        signals = []
        evidence = {}
        
        # Check against base patterns
        base_patterns = self.knowledge_base['base_patterns']
        
        for pattern_name, pattern_def in base_patterns.items():
            if pattern_name in session.detected_base_patterns:
                pattern_adjustment = pattern_def.get('confidence_adjustment', 1.0)
                
                if pattern_def.get('is_normal', False):
                    # Known normal pattern - reduce anomaly confidence
                    adjustment_factor *= pattern_adjustment
                    confidence = max(confidence, 0.8)  # High confidence in normal assessment
                    
                    signal = KnowledgeSignal(
                        signal_type='normal_pattern',
                        confidence=0.9,
                        description=f"Matches known normal pattern: {pattern_name}",
                        pattern_elements=pattern_def['core_sequence'],
                        adjustment_factor=pattern_adjustment
                    )
                    signals.append(signal)
                    
                elif pattern_def.get('is_normal', True) == False:  # Explicitly marked as abnormal
                    # Known anomaly pattern - increase anomaly confidence  
                    adjustment_factor *= pattern_adjustment
                    confidence = max(confidence, 0.85)
                    
                    signal = KnowledgeSignal(
                        signal_type='error_pattern',
                        confidence=0.9,
                        description=f"Matches known error pattern: {pattern_name}",
                        pattern_elements=pattern_def['core_sequence'],
                        adjustment_factor=pattern_adjustment
                    )
                    signals.append(signal)
        
        # Check for pattern variations
        if session.pattern_variations:
            # Pattern variations get moderate confidence reduction
            variation_factor = 0.7
            adjustment_factor *= variation_factor
            confidence = max(confidence, 0.6)
            
            signal = KnowledgeSignal(
                signal_type='variation',
                confidence=0.7,
                description=f"Pattern variations detected: {', '.join(session.pattern_variations)}",
                pattern_elements=session.pattern_variations,
                adjustment_factor=variation_factor
            )
            signals.append(signal)
        
        # Check for novel elements
        if session.novel_elements:
            # Novel elements slightly increase confidence (might be new anomaly type)
            novel_factor = 1.2
            adjustment_factor *= novel_factor
            confidence = max(confidence, 0.4)  # Lower confidence for novel patterns
            
            signal = KnowledgeSignal(
                signal_type='novel',
                confidence=0.4,
                description=f"Novel elements detected: {', '.join(session.novel_elements)}",
                pattern_elements=session.novel_elements,
                adjustment_factor=novel_factor
            )
            signals.append(signal)
        
        # Contextual adjustments
        context_adjustment = self.apply_contextual_knowledge(session)
        adjustment_factor *= context_adjustment['factor']
        confidence = max(confidence, context_adjustment['confidence'])
        signals.extend(context_adjustment['signals'])
        
        evidence = {
            'base_patterns_found': session.detected_base_patterns,
            'variations_found': session.pattern_variations,
            'novel_elements_found': session.novel_elements,
            'context_factors': context_adjustment['factors']
        }
        
        return {
            'adjustment_factor': adjustment_factor,
            'confidence': confidence,
            'signals': signals,
            'evidence': evidence
        }
    
    def apply_contextual_knowledge(self, session: EnhancedTransactionSession) -> Dict:
        """Apply contextual domain knowledge"""
        
        adjustment_factor = 1.0
        confidence = 0.5
        signals = []
        factors = []
        
        text_upper = session.raw_text.upper()
        
        # Maintenance context
        maintenance_indicators = self.knowledge_base['contextual_indicators']['maintenance_context']
        maintenance_count = sum(1 for indicator in maintenance_indicators if indicator in text_upper)
        
        if maintenance_count > 0:
            # During maintenance, reduce anomaly confidence slightly
            adjustment_factor *= 0.8
            confidence = max(confidence, 0.7)
            factors.append(f"maintenance_context:{maintenance_count}")
            
            signal = KnowledgeSignal(
                signal_type='maintenance_context',
                confidence=0.7,
                description=f"Maintenance context detected ({maintenance_count} indicators)",
                pattern_elements=maintenance_indicators,
                adjustment_factor=0.8
            )
            signals.append(signal)
        
        # Security context
        security_indicators = self.knowledge_base['contextual_indicators']['security_context']
        security_count = sum(1 for indicator in security_indicators if indicator in text_upper)
        
        if security_count > 0:
            # Security issues increase anomaly confidence
            adjustment_factor *= 1.3
            confidence = max(confidence, 0.8)
            factors.append(f"security_context:{security_count}")
            
            signal = KnowledgeSignal(
                signal_type='security_context',
                confidence=0.8,
                description=f"Security context detected ({security_count} indicators)",
                pattern_elements=security_indicators,
                adjustment_factor=1.3
            )
            signals.append(signal)
        
        return {
            'factor': adjustment_factor,
            'confidence': confidence,
            'signals': signals,
            'factors': factors
        }
    
    def normalize_score(self, score: float, all_scores: np.ndarray) -> float:
        """Normalize score to 0-1 range"""
        min_score = all_scores.min()
        max_score = all_scores.max()
        if max_score == min_score:
            return 0.5
        return (score - min_score) / (max_score - min_score)
    
    def determine_anomaly_type(self, session: EnhancedTransactionSession) -> str:
        """Determine the type of anomaly based on analysis"""
        
        # Check for specific known patterns first
        if 'dispense_failure' in session.detected_base_patterns:
            return 'dispense_failure'
        elif 'hardware_error' in session.detected_base_patterns:
            return 'hardware_error'
        elif 'supervisor_intervention' in session.detected_base_patterns:
            return 'supervisor_intervention'
        
        # Check for pattern variations
        if session.pattern_variations:
            return 'pattern_variation'
        
        # Check for novel elements
        if session.novel_elements:
            return 'novel_pattern'
        
        # Default to statistical anomaly
        return 'statistical_anomaly'
    
    def generate_anomaly_description(self, session: EnhancedTransactionSession, knowledge_adjustment: Dict) -> str:
        """Generate human-readable description of the anomaly"""
        
        base_desc = f"Session {session.session_id} shows anomalous behavior"
        
        # Add knowledge-based insights
        if knowledge_adjustment['signals']:
            signal_types = [signal.signal_type for signal in knowledge_adjustment['signals']]
            
            if 'error_pattern' in signal_types:
                base_desc += " with known error patterns"
            elif 'normal_pattern' in signal_types:
                base_desc += " but matches known normal patterns"
            elif 'variation' in signal_types:
                base_desc += " with pattern variations from known normal behavior"
            elif 'novel' in signal_types:
                base_desc += " with novel elements not seen in training data"
        
        # Add confidence information
        confidence_level = knowledge_adjustment['confidence']
        if confidence_level > 0.8:
            base_desc += " (high confidence)"
        elif confidence_level > 0.6:
            base_desc += " (medium confidence)"
        else:
            base_desc += " (low confidence - requires expert review)"
        
        return base_desc
    
    def determine_severity(self, confidence: float) -> str:
        """Determine severity based on knowledge-adjusted confidence"""
        if confidence >= 0.9:
            return "critical"
        elif confidence >= 0.75:
            return "high"
        elif confidence >= 0.5:
            return "medium"
        else:
            return "low"
    
    def perform_advanced_analysis(self):
        """Perform temporal and sequence analysis"""
        logger.info("Performing advanced temporal and sequence analysis")
        
        for session in self.sessions:
            # Temporal analysis
            temporal_results = self.temporal_analyzer.analyze_session(session)
            
            # Sequence analysis
            sequence_results = self.sequence_analyzer.analyze_session(session)
            
            # Add results as additional evidence
            if hasattr(session, 'anomaly_detections') and session.anomaly_detections:
                for detection in session.anomaly_detections:
                    detection.supporting_evidence.update({
                        'temporal_analysis': temporal_results,
                        'sequence_analysis': sequence_results
                    })
    
    def integrate_all_assessments(self):
        """Integrate all analysis results into final assessment"""
        logger.info("Integrating all assessment results")
        
        for session in self.sessions:
            # Combine all confidence scores with weighted average
            ml_weight = 0.4
            knowledge_weight = 0.4
            temporal_weight = 0.1
            sequence_weight = 0.1
            
            final_score = (
                session.overall_ml_score * ml_weight +
                session.overall_knowledge_adjusted_score * knowledge_weight
                # Add temporal and sequence weights when implemented
            )
            
            session.overall_knowledge_adjusted_score = min(1.0, final_score)
            session.is_anomaly = final_score > 0.5
            session.anomaly_score = final_score
    
    def create_enhanced_results_dataframe(self) -> pd.DataFrame:
        """Create enhanced results dataframe with knowledge insights"""
        
        results = []
        
        for session in self.sessions:
            result = {
                'session_id': session.session_id,
                'start_time': session.start_time,
                'end_time': session.end_time,
                
                # ML Analysis
                'ml_anomaly_score': session.overall_ml_score,
                'knowledge_adjusted_score': session.overall_knowledge_adjusted_score,
                'confidence_in_assessment': session.confidence_in_assessment,
                
                # Pattern Analysis
                'detected_base_patterns': ', '.join(session.detected_base_patterns),
                'pattern_variations': ', '.join(session.pattern_variations),
                'novel_elements': ', '.join(session.novel_elements),
                
                # Final Assessment
                'is_anomaly': session.is_anomaly,
                'anomaly_score': session.anomaly_score,
                'anomaly_type': session.anomaly_type,
                
                # Knowledge Signals
                'knowledge_signals_count': len(session.anomaly_detections[0].knowledge_signals) if session.anomaly_detections else 0,
                'knowledge_signal_types': ', '.join([signal.signal_type for detection in session.anomaly_detections for signal in detection.knowledge_signals]),
                
                # Raw data
                'raw_text': session.raw_text[:500] + "..." if len(session.raw_text) > 500 else session.raw_text
            }
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def log_analysis_summary(self, results_df: pd.DataFrame):
        """Log comprehensive analysis summary"""
        
        total_sessions = len(results_df)
        anomalies_found = results_df['is_anomaly'].sum()
        
        # Knowledge impact analysis
        ml_only_anomalies = (results_df['ml_anomaly_score'] > 0.5).sum()
        knowledge_adjusted_anomalies = anomalies_found
        
        logger.info("="*60)
        logger.info("KNOWLEDGE-GUIDED ML ANALYSIS SUMMARY")
        logger.info("="*60)
        logger.info(f"Total sessions analyzed: {total_sessions}")
        logger.info(f"ML-only anomalies: {ml_only_anomalies}")
        logger.info(f"Knowledge-adjusted anomalies: {knowledge_adjusted_anomalies}")
        logger.info(f"Knowledge impact: {knowledge_adjusted_anomalies - ml_only_anomalies:+d} anomalies")
        
        # Pattern analysis summary
        base_patterns_found = results_df['detected_base_patterns'].str.len() > 0
        variations_found = results_df['pattern_variations'].str.len() > 0
        novel_elements_found = results_df['novel_elements'].str.len() > 0
        
        logger.info(f"Sessions with base patterns: {base_patterns_found.sum()}")
        logger.info(f"Sessions with pattern variations: {variations_found.sum()}")
        logger.info(f"Sessions with novel elements: {novel_elements_found.sum()}")
        
        # Confidence analysis
        high_confidence = (results_df['confidence_in_assessment'] > 0.8).sum()
        medium_confidence = ((results_df['confidence_in_assessment'] > 0.6) & (results_df['confidence_in_assessment'] <= 0.8)).sum()
        low_confidence = (results_df['confidence_in_assessment'] <= 0.6).sum()
        
        logger.info(f"High confidence assessments: {high_confidence}")
        logger.info(f"Medium confidence assessments: {medium_confidence}")
        logger.info(f"Low confidence assessments: {low_confidence} (require expert review)")
        logger.info("="*60)
    
    # Additional helper methods would be implemented here
    def read_raw_logs(self, file_path: str) -> str:
        """Read raw EJ logs"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read()
    
    def split_into_enhanced_sessions(self, raw_logs: str, file_path: str) -> List[EnhancedTransactionSession]:
        """Split logs into enhanced transaction sessions"""
        # Implementation would be similar to original but create EnhancedTransactionSession objects
        # This is a simplified version - full implementation would follow original logic
        sessions = []
        # ... sessionization logic ...
        return sessions
    
    def generate_embeddings(self, sessions: List[EnhancedTransactionSession]) -> np.ndarray:
        """Generate BERT embeddings for sessions"""
        # Implementation similar to original
        embeddings = []
        # ... embedding generation logic ...
        return np.array(embeddings) if embeddings else np.array([])


class PatternSimilarityAnalyzer:
    """Analyzes pattern similarity and variations"""
    
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    
    def analyze_session_patterns(self, session: EnhancedTransactionSession, knowledge_base: Dict) -> Dict:
        """Analyze session against known patterns"""
        
        results = {
            'base_patterns': [],
            'variations': [],
            'novel_elements': []
        }
        
        text_upper = session.raw_text.upper()
        
        # Check against base patterns
        for pattern_name, pattern_def in knowledge_base['base_patterns'].items():
            similarity = self.calculate_pattern_similarity(text_upper, pattern_def)
            
            if similarity > pattern_def['confidence_threshold']:
                results['base_patterns'].append(pattern_name)
        
        # Check for variations
        for variation_type, variation_list in knowledge_base['pattern_variations'].items():
            for variation in variation_list:
                if variation.upper() in text_upper:
                    results['variations'].append(variation)
        
        # Identify novel elements (simplified)
        if not results['base_patterns'] and not results['variations']:
            results['novel_elements'].append('unknown_pattern')
        
        return results
    
    def calculate_pattern_similarity(self, text: str, pattern_def: Dict) -> float:
        """Calculate similarity between text and pattern definition"""
        
        # Count required elements present
        required_elements = pattern_def.get('required_elements', [])
        elements_found = sum(1 for element in required_elements if element in text)
        
        if not required_elements:
            return 0.0
        
        return elements_found / len(required_elements)


class TemporalAnomalyAnalyzer:
    """Analyzes temporal anomalies in session timing"""
    
    def analyze_session(self, session: EnhancedTransactionSession) -> Dict:
        """Analyze temporal patterns in session"""
        
        results = {
            'duration_anomaly': False,
            'timing_score': 0.0,
            'temporal_patterns': []
        }
        
        # Calculate session duration
        if session.start_time and session.end_time:
            duration = (session.end_time - session.start_time).total_seconds()
            
            # Simple duration-based anomaly detection
            if duration > 300:  # More than 5 minutes
                results['duration_anomaly'] = True
                results['timing_score'] = min(1.0, duration / 600)  # Normalize to max 10 minutes
                results['temporal_patterns'].append('long_duration')
            elif duration < 10:  # Less than 10 seconds
                results['duration_anomaly'] = True
                results['timing_score'] = 0.7
                results['temporal_patterns'].append('short_duration')
        
        return results


class SequenceAnomalyAnalyzer:
    """Analyzes sequence anomalies in transaction flow"""
    
    def analyze_session(self, session: EnhancedTransactionSession) -> Dict:
        """Analyze sequence patterns in session"""
        
        results = {
            'sequence_anomaly': False,
            'sequence_score': 0.0,
            'sequence_patterns': []
        }
        
        # Extract event sequence
        events = self.extract_event_sequence(session.raw_text)
        
        # Simple sequence analysis
        if len(events) > 100:  # Too many events
            results['sequence_anomaly'] = True
            results['sequence_score'] = min(1.0, len(events) / 200)
            results['sequence_patterns'].append('excessive_events')
        elif len(events) < 3:  # Too few events
            results['sequence_anomaly'] = True
            results['sequence_score'] = 0.6
            results['sequence_patterns'].append('insufficient_events')
        
        return results
    
    def extract_event_sequence(self, text: str) -> List[str]:
        """Extract sequence of events from session text"""
        
        # Simple event extraction based on common patterns
        events = []
        
        event_patterns = [
            'CARD INSERTED', 'PIN ENTERED', 'NOTES PRESENTED', 'NOTES TAKEN',
            'CARD TAKEN', 'DISPENSE', 'ERROR', 'SUPERVISOR', 'TIMEOUT'
        ]
        
        for pattern in event_patterns:
            if pattern in text.upper():
                events.append(pattern)
        
        return events
