"""
Ensemble Anomaly Detection Model
"""

import numpy as np
import re
import json
import pickle
import os
from typing import Dict, List, Tuple, Any
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

class EnsembleAnomalyDetector:
    """
    Complete ensemble anomaly detection system combining text and statistical analysis
    """
    
    def __init__(self, model_dir: str = "./models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Model components
        self.text_vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2), lowercase=True)
        self.svm_model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
        self.isolation_model = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        
        # Training state
        self.is_trained = False
        self.training_stats = {}
        self.feature_names = []
        
        # Weights for ensemble
        self.text_weight = 0.6
        self.statistical_weight = 0.4
        self.threshold = 0.5
        
    def extract_text_features(self, session_text: str) -> Dict[str, float]:
        """Extract text-based features with enhanced critical anomaly detection"""
        text_lower = session_text.lower()
        text_upper = session_text.upper()
        words = text_lower.split()
        
        # Enhanced term categories based on contextual labeler knowledge
        normal_terms = ['card', 'pin', 'verified', 'completed', 'successful', 'dispensed', 'printed', 'ejected', 'taken', 'approved']
        
        # Critical error terms that should trigger immediate anomaly detection
        critical_error_terms = ['device error', 'device offline', 'critical', 'fatal', 'malfunction', 'communication failure']
        hardware_error_terms = ['hardware error', 'power-up/reset', 'cim-reset', 'recovery failed', 'capture failed', 'jam']
        general_error_terms = ['error', 'fail', 'timeout', 'reset', 'offline', 'fault', 'unable', 'declined']
        
        # Machine status codes that indicate problems (M-XX patterns)
        machine_status_patterns = re.findall(r'M-\d+', text_upper)
        critical_machine_codes = ['M-65', 'M-01', 'M-15', 'M-23', 'M-45', 'M-67']  # Known problematic codes
        
        # Error codes and patterns from EJ contextual knowledge
        error_code_patterns = re.findall(r'[ME]-\d+', text_upper)
        device_error_count = len(re.findall(r'device\s+error', text_lower))
        aac_errors = len(re.findall(r'aac|no arpc', text_lower))
        communication_errors = len(re.findall(r'communication\s+failure|comm\s+error|timeout', text_lower))
        
        # Supervisor mode indicators (high anomaly weight)
        supervisor_patterns = len(re.findall(r'supervisor\s+mode|supervisor\s+entry|supervisor\s+exit', text_lower))
        
        # Recovery operation indicators
        recovery_patterns = len(re.findall(r'recovery|cim-reset|init\s+bna|device\s+init|retract\s+bin', text_lower))
        
        # Cash handling anomalies
        cash_anomalies = len(re.findall(r'cash\s+error|dispenser\s+error|notes\s+jam|cash\s+jam', text_lower))
        retract_operations = len(re.findall(r'retract|capture\s+failed', text_lower))
        
        # Authentication failures
        auth_failures = len(re.findall(r'external\s+authenticate.*fail|pin.*fail|auth.*fail', text_lower))
        
        # Calculate base features
        features = {
            'total_words': len(words),
            'unique_words': len(set(words)),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            
            # Term counting with enhanced weights
            'normal_term_count': sum(1 for word in words if any(term in word for term in normal_terms)),
            'critical_error_count': sum(1 for phrase in critical_error_terms if phrase in text_lower),
            'hardware_error_count': sum(1 for phrase in hardware_error_terms if phrase in text_lower),
            'general_error_count': sum(1 for phrase in general_error_terms if phrase in text_lower),
            
            # Critical anomaly indicators - these should have high weight
            'device_error_explicit': device_error_count,
            'machine_status_codes': len(machine_status_patterns),
            'critical_machine_codes': sum(1 for code in machine_status_patterns if code in critical_machine_codes),
            'error_codes_total': len(error_code_patterns),
            'aac_authentication_errors': aac_errors,
            'communication_failures': communication_errors,
            
            # Operational anomaly indicators
            'supervisor_mode_indicators': supervisor_patterns,
            'recovery_operations': recovery_patterns,
            'cash_handling_anomalies': cash_anomalies,
            'retract_operations': retract_operations,
            'authentication_failures': auth_failures,
            
            # Pattern-based anomaly detection
            'error_pattern_density': 0.0,  # Will calculate below
            'critical_anomaly_score': 0.0,  # Will calculate below
        }
        
        # Calculate ratios and derived features
        if features['total_words'] > 0:
            features['error_ratio'] = (features['critical_error_count'] + features['hardware_error_count'] + features['general_error_count']) / features['total_words']
            features['normal_ratio'] = features['normal_term_count'] / features['total_words']
            features['anomaly_term_density'] = (features['device_error_explicit'] + features['critical_machine_codes'] + features['supervisor_mode_indicators']) / features['total_words']
        else:
            features['error_ratio'] = features['normal_ratio'] = features['anomaly_term_density'] = 0
            
        # Calculate error pattern density (concentration of error indicators)
        total_error_indicators = (features['critical_error_count'] + features['hardware_error_count'] + 
                                features['device_error_explicit'] + features['critical_machine_codes'] + 
                                features['communication_failures'] + features['recovery_operations'])
        
        if features['total_words'] > 0:
            features['error_pattern_density'] = total_error_indicators / features['total_words']
        
        # Calculate critical anomaly score (0-1, higher = more anomalous)
        # This heavily weights the specific patterns you mentioned
        critical_score = 0.0
        
        # "DEVICE ERROR" pattern gets maximum weight
        if features['device_error_explicit'] > 0:
            critical_score += 0.8  # Very high weight for explicit device errors
            
        # Machine status codes like "M-65" get high weight  
        if features['critical_machine_codes'] > 0:
            critical_score += 0.7  # High weight for known bad machine codes
            
        # Multiple error codes indicate serious problems
        if features['error_codes_total'] > 2:
            critical_score += 0.6
        elif features['error_codes_total'] > 0:
            critical_score += 0.3
            
        # Communication and authentication failures
        if features['communication_failures'] > 0:
            critical_score += 0.5
        if features['aac_authentication_errors'] > 0:
            critical_score += 0.4
            
        # Recovery operations suggest previous failures
        if features['recovery_operations'] > 1:
            critical_score += 0.4
        elif features['recovery_operations'] > 0:
            critical_score += 0.2
            
        # Supervisor mode during transactions is highly suspicious
        if features['supervisor_mode_indicators'] > 0:
            critical_score += 0.6
            
        # High error density
        if features['error_pattern_density'] > 0.1:  # More than 10% error terms
            critical_score += 0.5
        elif features['error_pattern_density'] > 0.05:  # More than 5% error terms
            critical_score += 0.3
            
        # Cap the score at 1.0 but allow accumulation for severe cases
        features['critical_anomaly_score'] = min(1.0, critical_score)
        
        # Add specific pattern flags for interpretability
        features['has_device_error'] = 1.0 if features['device_error_explicit'] > 0 else 0.0
        features['has_critical_machine_code'] = 1.0 if features['critical_machine_codes'] > 0 else 0.0
        features['has_supervisor_anomaly'] = 1.0 if features['supervisor_mode_indicators'] > 0 else 0.0
        features['has_recovery_operations'] = 1.0 if features['recovery_operations'] > 0 else 0.0
        features['multiple_error_codes'] = 1.0 if features['error_codes_total'] > 1 else 0.0
        
        return features
    
    def extract_numerical_features(self, session_text: str) -> Dict[str, float]:
        """Extract numerical/statistical features with enhanced anomaly detection"""
        lines = session_text.strip().split('\n')
        text_lower = session_text.lower()
        text_upper = session_text.upper()
        
        # Basic session structure
        features = {
            # Session structure
            'line_count': len(lines),
            'total_chars': len(session_text),
            'avg_line_length': np.mean([len(line) for line in lines]) if lines else 0,
            'empty_lines': sum(1 for line in lines if not line.strip()),
            
            # Enhanced error pattern detection based on contextual labeler
            'error_count': len(re.findall(r'error', text_lower)),
            'fail_count': len(re.findall(r'fail', text_lower)),
            'malfunction_count': len(re.findall(r'malfunction', text_lower)),
            'timeout_count': len(re.findall(r'timeout', text_lower)),
            'device_error_count': len(re.findall(r'device\s+error', text_lower)),  # Specific "DEVICE ERROR" pattern
            
            # Machine status codes (critical for ATM anomaly detection)
            'machine_status_codes': len(re.findall(r'M-\d+', text_upper)),
            'critical_m_codes': len(re.findall(r'M-(?:01|15|23|38|45|65|67)', text_upper)),  # Known critical codes
            'error_codes_total': len(re.findall(r'[ME]-\d+', text_upper)),
            
            # Hardware-specific patterns from contextual labeler
            'hardware_mentions': len(re.findall(r'hardware', text_lower)),
            'power_reset_count': len(re.findall(r'power.*reset|reset.*power|power-up/reset', text_lower)),
            'cim_mentions': len(re.findall(r'cim', text_lower)),
            'recovery_failures': len(re.findall(r'recovery.*fail', text_lower)),
            'capture_failures': len(re.findall(r'capture.*fail', text_lower)),
            
            # Critical hardware patterns (high anomaly weight)
            'critical_hardware_patterns': len(re.findall(
                r'power-up/reset|hardware.*error|cim-reset|recovery.*failed|capture.*failed|device\s+error',
                text_lower
            )),
            
            # Enhanced communication and authentication patterns
            'communication_errors': len(re.findall(r'communication\s+failure|comm\s+error|no arpc|aac', text_lower)),
            'authentication_failures': len(re.findall(r'external\s+authenticate.*fail|pin.*fail|genac.*aac', text_lower)),
            'network_errors': len(re.findall(r'network.*error|connection.*lost|timeout', text_lower)),
            
            # Cash dispenser and handling patterns
            'cash_errors': len(re.findall(r'cash.*error|dispenser.*error|jam', text_lower)),
            'retract_operations': len(re.findall(r'retract|capture\s+failed', text_lower)),
            'dispensing_issues': len(re.findall(r'notes.*jam|cash.*jam|dispenser.*malfunction', text_lower)),
            
            # Supervisor mode patterns (highly suspicious during transactions)
            'supervisor_patterns': len(re.findall(r'supervisor\s+mode|supervisor\s+entry|supervisor\s+exit', text_lower)),
            
            # Recovery operation indicators (suggest previous failures)
            'recovery_operations': len(re.findall(r'init\s+bna|cim-reset|device\s+init|recovery', text_lower)),
            'reset_operations': len(re.findall(r'reset|init.*started|recovery.*ok', text_lower)),
            
            # Transaction integrity patterns
            'transaction_start_count': len(re.findall(r'transaction\s+start', text_lower)),
            'transaction_end_count': len(re.findall(r'transaction\s+end', text_lower)),
            'incomplete_transaction_ratio': 0.0,  # Will calculate below
            
            # Success indicators (should be lower in anomalous sessions)
            'success_indicators': len(re.findall(
                r'completed|successful|verified|dispensed|printed|taken|approved',
                text_lower
            )),
            
            # Specific anomaly pattern scores
            'anomaly_density_score': 0.0,  # Will calculate below
            'critical_error_density': 0.0,  # Will calculate below
            'hardware_failure_score': 0.0,  # Will calculate below
        }
        
        # Calculate derived features and anomaly scores
        
        # Transaction completeness check
        if features['transaction_start_count'] > 0:
            features['incomplete_transaction_ratio'] = abs(features['transaction_start_count'] - features['transaction_end_count']) / features['transaction_start_count']
        
        # Calculate anomaly density (proportion of anomalous patterns)
        total_anomaly_indicators = (features['device_error_count'] + features['critical_m_codes'] + 
                                  features['critical_hardware_patterns'] + features['communication_errors'] + 
                                  features['supervisor_patterns'] + features['recovery_operations'])
        
        if features['line_count'] > 0:
            features['anomaly_density_score'] = total_anomaly_indicators / features['line_count']
            features['critical_error_density'] = (features['device_error_count'] + features['critical_m_codes']) / features['line_count']
        
        # Hardware failure composite score
        hardware_failures = (features['critical_hardware_patterns'] + features['power_reset_count'] + 
                           features['recovery_failures'] + features['capture_failures'])
        features['hardware_failure_score'] = min(1.0, hardware_failures / 5.0)  # Normalize to 0-1
        
        # Calculate error-to-success ratio
        total_errors = (features['error_count'] + features['fail_count'] + features['device_error_count'] + 
                       features['critical_m_codes'])
        if features['success_indicators'] > 0:
            features['error_to_success_ratio'] = total_errors / features['success_indicators']
        else:
            features['error_to_success_ratio'] = total_errors  # No success indicators = very bad
        
        # Line-based error concentration
        if features['line_count'] > 0:
            features['error_to_line_ratio'] = total_errors / features['line_count']
        else:
            features['error_to_line_ratio'] = 0
        
        # Enhanced session health scoring
        features['session_health_score'] = self._calculate_session_health_score(features, text_lower)
        
        return features
    
    def _calculate_session_health_score(self, features: Dict[str, float], text_lower: str) -> float:
        """Calculate a comprehensive session health score (0=very unhealthy, 1=healthy)"""
        health_score = 1.0  # Start with perfect health
        
        # Critical deductions for specific patterns you mentioned
        
        # "DEVICE ERROR" pattern - immediate severe deduction
        if features['device_error_count'] > 0:
            health_score -= 0.8  # Severe deduction for device errors
        
        # Critical machine codes like "M-65" - severe deduction
        if features['critical_m_codes'] > 0:
            health_score -= 0.7  # High deduction for known bad machine codes
        
        # Multiple error codes indicate cascading failures
        if features['error_codes_total'] > 2:
            health_score -= 0.6
        elif features['error_codes_total'] > 0:
            health_score -= 0.3
        
        # Communication failures
        if features['communication_errors'] > 0:
            health_score -= 0.5
        
        # Hardware failures and recovery operations
        if features['critical_hardware_patterns'] > 1:
            health_score -= 0.6
        elif features['critical_hardware_patterns'] > 0:
            health_score -= 0.4
        
        # Supervisor mode during transaction (highly suspicious)
        if features['supervisor_patterns'] > 0:
            health_score -= 0.7
        
        # Recovery operations suggest previous failures
        if features['recovery_operations'] > 1:
            health_score -= 0.4
        elif features['recovery_operations'] > 0:
            health_score -= 0.2
        
        # High error density
        if features['anomaly_density_score'] > 0.2:  # >20% of lines contain anomalies
            health_score -= 0.5
        elif features['anomaly_density_score'] > 0.1:  # >10% of lines contain anomalies
            health_score -= 0.3
        
        # Authentication failures
        if features['authentication_failures'] > 0:
            health_score -= 0.4
        
        # Incomplete transactions
        if features['incomplete_transaction_ratio'] > 0:
            health_score -= 0.3
        
        # Very high error to success ratio
        if features['error_to_success_ratio'] > 2.0:
            health_score -= 0.4
        elif features['error_to_success_ratio'] > 1.0:
            health_score -= 0.2
        
        # Ensure score stays in valid range
        return max(0.0, min(1.0, health_score))
    
    def sessionize_ej_log(self, ej_log_text: str) -> List[str]:
        """
        Split EJ log into individual sessions based on transaction boundaries
        """
        sessions = []
        
        # Split by common session delimiters
        session_delimiters = [
            r'TRANSACTION START|SESSION START',
            r'TRANSACTION END|SESSION END',
            r'\[020t\*\d+\*',  # Transaction ID pattern
            r'---START OF TRANSACTION---',
            r'---END OF TRANSACTION---'
        ]
        
        # First try to split by explicit session markers
        lines = ej_log_text.split('\n')
        current_session = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for session start markers
            if any(re.search(delimiter, line, re.IGNORECASE) for delimiter in session_delimiters[:3]):
                if current_session:
                    sessions.append('\n'.join(current_session))
                    current_session = []
                current_session.append(line)
            elif current_session:
                current_session.append(line)
            else:
                # Start a new session if we don't have one
                current_session.append(line)
        
        # Add the last session
        if current_session:
            sessions.append('\n'.join(current_session))
        
        # If no clear sessions found, treat as single session
        if not sessions:
            sessions = [ej_log_text]
        
        # Filter out very short sessions (likely incomplete)
        sessions = [s for s in sessions if len(s.split()) > 5]
        
        return sessions
    
    def train(self, normal_sessions: List[str]) -> Dict[str, Any]:
        """
        Train the ensemble model on normal sessions only (unsupervised)
        """
        print(f"Training ensemble on {len(normal_sessions)} normal sessions...")
        
        # Extract features from normal sessions
        texts = []
        numerical_features_list = []
        
        for session in normal_sessions:
            texts.append(session)
            num_features = self.extract_numerical_features(session)
            numerical_features_list.append(list(num_features.values()))
        
        # Store feature names
        sample_num_features = self.extract_numerical_features(normal_sessions[0])
        self.feature_names = list(sample_num_features.keys())
        
        # Train text vectorizer and SVM
        text_features = self.text_vectorizer.fit_transform(texts).toarray()
        self.svm_model.fit(text_features)
        
        # Train scaler and isolation forest
        numerical_features = np.array(numerical_features_list)
        numerical_features = self.scaler.fit_transform(numerical_features)
        self.isolation_model.fit(numerical_features)
        
        # Calculate training statistics
        svm_scores = self.svm_model.decision_function(text_features)
        iso_scores = self.isolation_model.decision_function(numerical_features)
        
        # Convert to probabilities
        svm_probabilities = 1 / (1 + np.exp(svm_scores))
        iso_probabilities = 1 / (1 + np.exp(iso_scores))
        ensemble_scores = self.text_weight * svm_probabilities + self.statistical_weight * iso_probabilities
        
        self.training_stats = {
            'num_training_sessions': len(normal_sessions),
            'text_feature_dims': text_features.shape[1],
            'numerical_feature_dims': len(self.feature_names),
            'avg_svm_score': float(np.mean(svm_probabilities)),
            'avg_isolation_score': float(np.mean(iso_probabilities)),
            'avg_ensemble_score': float(np.mean(ensemble_scores)),
            'feature_names': self.feature_names,
            'text_weight': self.text_weight,
            'statistical_weight': self.statistical_weight,
            'threshold': self.threshold
        }
        
        self.is_trained = True
        print("Training complete!")
        
        return self.training_stats
    
    def predict(self, session_text: str) -> Dict[str, Any]:
        """
        Predict anomaly for a single session with enhanced critical pattern detection
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features
        text_features = self.text_vectorizer.transform([session_text]).toarray()
        num_features_dict = self.extract_numerical_features(session_text)
        text_features_dict = self.extract_text_features(session_text)
        
        # Get base model predictions
        num_features = np.array([list(num_features_dict.values())])
        num_features = self.scaler.transform(num_features)
        
        svm_score = self.svm_model.decision_function(text_features)[0]
        iso_score = self.isolation_model.decision_function(num_features)[0]
        
        # Convert to probabilities (higher = more anomalous)
        svm_probability = 1 / (1 + np.exp(svm_score))
        iso_probability = 1 / (1 + np.exp(iso_score))
        
        # Apply critical anomaly amplification based on domain knowledge
        amplified_scores = self._apply_critical_anomaly_amplification(
            svm_probability, iso_probability, text_features_dict, num_features_dict, session_text
        )
        
        amplified_svm = amplified_scores['amplified_text_score']
        amplified_iso = amplified_scores['amplified_statistical_score']
        critical_boost = amplified_scores['critical_boost']
        anomaly_reasons = amplified_scores['anomaly_reasons']
        
        # Calculate final ensemble score with amplification
        base_ensemble_score = self.text_weight * amplified_svm + self.statistical_weight * amplified_iso
        
        # Apply additional boost for critical patterns
        final_ensemble_score = min(1.0, base_ensemble_score + critical_boost)
        
        # Determine if anomaly (adjusted threshold for critical patterns)
        effective_threshold = self.threshold
        if critical_boost > 0.3:  # Significant critical patterns detected
            effective_threshold = max(0.3, self.threshold - 0.2)  # Lower threshold for critical cases
        
        is_anomaly = final_ensemble_score > effective_threshold
        
        # Enhanced confidence calculation
        confidence = self._calculate_enhanced_confidence(
            final_ensemble_score, effective_threshold, critical_boost, anomaly_reasons
        )
        
        return {
            'session_text': session_text,
            'text_anomaly_score': float(amplified_svm),
            'statistical_anomaly_score': float(amplified_iso),
            'ensemble_score': float(final_ensemble_score),
            'base_ensemble_score': float(base_ensemble_score),
            'critical_boost': float(critical_boost),
            'is_anomaly': bool(is_anomaly),
            'confidence': confidence,
            'threshold': effective_threshold,
            'original_threshold': self.threshold,
            'text_features': text_features_dict,
            'numerical_features': num_features_dict,
            'anomaly_reasons': anomaly_reasons,
            'prediction_breakdown': {
                'text_component': {
                    'original_score': float(svm_probability),
                    'amplified_score': float(amplified_svm),
                    'weight': self.text_weight,
                    'contribution': float(self.text_weight * amplified_svm)
                },
                'statistical_component': {
                    'original_score': float(iso_probability),
                    'amplified_score': float(amplified_iso),
                    'weight': self.statistical_weight,
                    'contribution': float(self.statistical_weight * amplified_iso)
                },
                'critical_amplification': {
                    'boost_applied': float(critical_boost),
                    'reasons': anomaly_reasons,
                    'threshold_adjustment': float(self.threshold - effective_threshold)
                }
            }
        }
    
    def _apply_critical_anomaly_amplification(self, svm_prob: float, iso_prob: float, 
                                            text_features: Dict[str, float], 
                                            num_features: Dict[str, float], 
                                            session_text: str) -> Dict[str, Any]:
        """
        Apply amplification for critical anomaly patterns based on contextual labeler knowledge
        """
        anomaly_reasons = []
        critical_boost = 0.0
        
        # Critical Pattern 1: "DEVICE ERROR" - This should be heavily weighted
        if text_features.get('device_error_explicit', 0) > 0:
            boost = 0.6  # Major boost for explicit device errors
            critical_boost += boost
            anomaly_reasons.append(f"DEVICE ERROR detected - critical hardware failure indicator (+{boost:.1f})")
        
        # Critical Pattern 2: Machine status codes like "M-65"
        if num_features.get('critical_m_codes', 0) > 0:
            boost = 0.5  # High boost for known critical machine codes
            critical_boost += boost
            # Extract specific codes for detailed reporting
            m_codes = re.findall(r'M-\d+', session_text.upper())
            anomaly_reasons.append(f"Critical machine status codes detected: {', '.join(m_codes)} (+{boost:.1f})")
        
        # Critical Pattern 3: Multiple error codes indicate cascading failures
        error_code_count = num_features.get('error_codes_total', 0)
        if error_code_count > 2:
            boost = 0.4
            critical_boost += boost
            anomaly_reasons.append(f"Multiple error codes ({int(error_code_count)}) - cascading system failures (+{boost:.1f})")
        elif error_code_count > 0:
            boost = 0.2
            critical_boost += boost
            anomaly_reasons.append(f"Error codes detected ({int(error_code_count)}) - system issues (+{boost:.1f})")
        
        # Critical Pattern 4: Communication failures
        if num_features.get('communication_errors', 0) > 0:
            boost = 0.4
            critical_boost += boost
            anomaly_reasons.append(f"Communication failures detected - network/host connectivity issues (+{boost:.1f})")
        
        # Critical Pattern 5: Supervisor mode patterns (highly suspicious during transactions)
        if num_features.get('supervisor_patterns', 0) > 0:
            boost = 0.5
            critical_boost += boost
            anomaly_reasons.append(f"Supervisor mode activity detected - unusual operational pattern (+{boost:.1f})")
        
        # Critical Pattern 6: Critical hardware patterns
        if num_features.get('critical_hardware_patterns', 0) > 1:
            boost = 0.4
            critical_boost += boost
            anomaly_reasons.append(f"Multiple critical hardware patterns detected - device reliability issues (+{boost:.1f})")
        elif num_features.get('critical_hardware_patterns', 0) > 0:
            boost = 0.2
            critical_boost += boost
            anomaly_reasons.append(f"Critical hardware pattern detected - potential device issue (+{boost:.1f})")
        
        # Critical Pattern 7: High anomaly density (concentrated error patterns)
        anomaly_density = num_features.get('anomaly_density_score', 0)
        if anomaly_density > 0.2:  # More than 20% of lines contain anomalies
            boost = 0.3
            critical_boost += boost
            anomaly_reasons.append(f"High anomaly density ({anomaly_density:.1%}) - concentrated error patterns (+{boost:.1f})")
        elif anomaly_density > 0.1:  # More than 10% of lines contain anomalies
            boost = 0.15
            critical_boost += boost
            anomaly_reasons.append(f"Elevated anomaly density ({anomaly_density:.1%}) - multiple error indicators (+{boost:.1f})")
        
        # Critical Pattern 8: Poor session health
        session_health = num_features.get('session_health_score', 1.0)
        if session_health < 0.3:
            boost = 0.4
            critical_boost += boost
            anomaly_reasons.append(f"Very poor session health (score: {session_health:.2f}) - multiple critical issues (+{boost:.1f})")
        elif session_health < 0.5:
            boost = 0.2
            critical_boost += boost
            anomaly_reasons.append(f"Poor session health (score: {session_health:.2f}) - concerning patterns (+{boost:.1f})")
        
        # Critical Pattern 9: Authentication failures
        if num_features.get('authentication_failures', 0) > 0:
            boost = 0.3
            critical_boost += boost
            anomaly_reasons.append(f"Authentication failures detected - security concerns (+{boost:.1f})")
        
        # Critical Pattern 10: Recovery operations (suggest previous failures)
        recovery_ops = num_features.get('recovery_operations', 0)
        if recovery_ops > 1:
            boost = 0.3
            critical_boost += boost
            anomaly_reasons.append(f"Multiple recovery operations ({int(recovery_ops)}) - device instability (+{boost:.1f})")
        elif recovery_ops > 0:
            boost = 0.15
            critical_boost += boost
            anomaly_reasons.append(f"Recovery operation detected - previous device issue (+{boost:.1f})")
        
        # Apply amplification to scores
        # For text score: amplify based on text-based critical indicators
        text_amplification = (text_features.get('critical_anomaly_score', 0) * 0.5 + 
                            min(0.3, critical_boost * 0.6))
        amplified_svm = min(1.0, svm_prob + text_amplification)
        
        # For statistical score: amplify based on numerical critical indicators  
        statistical_amplification = min(0.4, critical_boost * 0.7)
        amplified_iso = min(1.0, iso_prob + statistical_amplification)
        
        # Cap total critical boost
        critical_boost = min(0.4, critical_boost)
        
        return {
            'amplified_text_score': amplified_svm,
            'amplified_statistical_score': amplified_iso,
            'critical_boost': critical_boost,
            'anomaly_reasons': anomaly_reasons,
            'text_amplification': text_amplification,
            'statistical_amplification': statistical_amplification
        }
    
    def _calculate_enhanced_confidence(self, ensemble_score: float, threshold: float, 
                                     critical_boost: float, anomaly_reasons: List[str]) -> str:
        """Calculate enhanced confidence based on score, threshold, and critical patterns"""
        
        # Distance from threshold
        threshold_distance = abs(ensemble_score - threshold)
        
        # Base confidence from threshold distance
        if threshold_distance > 0.4:
            base_confidence = "HIGH"
        elif threshold_distance > 0.2:
            base_confidence = "MEDIUM"
        else:
            base_confidence = "LOW"
        
        # Boost confidence for critical patterns
        if critical_boost > 0.3 and len(anomaly_reasons) >= 3:
            # Multiple critical patterns detected - very high confidence
            return "VERY_HIGH"
        elif critical_boost > 0.2 and len(anomaly_reasons) >= 2:
            # Significant critical patterns - upgrade confidence
            if base_confidence == "LOW":
                return "MEDIUM"
            elif base_confidence == "MEDIUM":
                return "HIGH"
            else:
                return "VERY_HIGH"
        elif critical_boost > 0.1:
            # Some critical patterns - slight confidence boost
            if base_confidence == "LOW" and threshold_distance > 0.1:
                return "MEDIUM"
        
        return base_confidence
    
    def batch_predict(self, sessions: List[str]) -> List[Dict[str, Any]]:
        """
        Predict anomalies for multiple sessions
        """
        return [self.predict(session) for session in sessions]
    
    def save_model(self, filepath: str = None):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        if filepath is None:
            filepath = os.path.join(self.model_dir, "ensemble_model.pkl")
        
        model_data = {
            'text_vectorizer': self.text_vectorizer,
            'svm_model': self.svm_model,
            'isolation_model': self.isolation_model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'training_stats': self.training_stats,
            'feature_names': self.feature_names,
            'text_weight': self.text_weight,
            'statistical_weight': self.statistical_weight,
            'threshold': self.threshold
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = None):
        """Load a trained model"""
        if filepath is None:
            filepath = os.path.join(self.model_dir, "ensemble_model.pkl")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.text_vectorizer = model_data['text_vectorizer']
        self.svm_model = model_data['svm_model']
        self.isolation_model = model_data['isolation_model']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
        self.training_stats = model_data['training_stats']
        self.feature_names = model_data['feature_names']
        self.text_weight = model_data['text_weight']
        self.statistical_weight = model_data['statistical_weight']
        self.threshold = model_data['threshold']
        
        print(f"Model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'is_trained': self.is_trained,
            'training_stats': self.training_stats,
            'ensemble_config': {
                'text_weight': self.text_weight,
                'statistical_weight': self.statistical_weight,
                'threshold': self.threshold
            },
            'feature_names': self.feature_names
        }
