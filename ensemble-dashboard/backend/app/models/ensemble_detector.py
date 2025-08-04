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
        """Extract text-based features"""
        text_lower = session_text.lower()
        words = text_lower.split()
        
        # Define term categories
        normal_terms = ['card', 'pin', 'verified', 'completed', 'successful', 'dispensed', 'printed', 'ejected']
        error_terms = ['error', 'fail', 'malfunction', 'timeout', 'reset', 'offline', 'jam', 'lost']
        hardware_terms = ['hardware', 'power-up/reset', 'cim-reset', 'recovery', 'capture', 'device']
        
        features = {
            'total_words': len(words),
            'normal_term_count': sum(1 for word in words if any(term in word for term in normal_terms)),
            'error_term_count': sum(1 for word in words if any(term in word for term in error_terms)),
            'hardware_term_count': sum(1 for word in words if any(term in word for term in hardware_terms)),
            'unique_words': len(set(words)),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
        }
        
        # Calculate ratios
        if features['total_words'] > 0:
            features['error_ratio'] = features['error_term_count'] / features['total_words']
            features['hardware_ratio'] = features['hardware_term_count'] / features['total_words']
            features['normal_ratio'] = features['normal_term_count'] / features['total_words']
        else:
            features['error_ratio'] = features['hardware_ratio'] = features['normal_ratio'] = 0
            
        return features
    
    def extract_numerical_features(self, session_text: str) -> Dict[str, float]:
        """Extract numerical/statistical features"""
        lines = session_text.strip().split('\n')
        text_lower = session_text.lower()
        
        features = {
            # Session structure
            'line_count': len(lines),
            'total_chars': len(session_text),
            'avg_line_length': np.mean([len(line) for line in lines]) if lines else 0,
            'empty_lines': sum(1 for line in lines if not line.strip()),
            
            # Error patterns
            'error_count': len(re.findall(r'error', text_lower)),
            'fail_count': len(re.findall(r'fail', text_lower)),
            'malfunction_count': len(re.findall(r'malfunction', text_lower)),
            'timeout_count': len(re.findall(r'timeout', text_lower)),
            
            # Hardware-specific
            'hardware_mentions': len(re.findall(r'hardware', text_lower)),
            'power_reset_count': len(re.findall(r'power.*reset|reset.*power|power-up/reset', text_lower)),
            'cim_mentions': len(re.findall(r'cim', text_lower)),
            'recovery_failures': len(re.findall(r'recovery.*fail', text_lower)),
            'capture_failures': len(re.findall(r'capture.*fail', text_lower)),
            
            # Critical patterns
            'critical_hardware_patterns': len(re.findall(
                r'power-up/reset|hardware.*error|cim-reset|recovery.*failed|capture.*failed',
                text_lower
            )),
            
            # Success indicators
            'success_indicators': len(re.findall(
                r'completed|successful|verified|dispensed|printed',
                text_lower
            )),
            
            # Network patterns
            'network_errors': len(re.findall(r'network.*error|connection.*lost|timeout', text_lower)),
            
            # Cash dispenser patterns
            'cash_errors': len(re.findall(r'cash.*error|dispenser.*error|jam', text_lower)),
        }
        
        # Calculate ratios
        if features['line_count'] > 0:
            features['error_to_line_ratio'] = (features['error_count'] + features['fail_count']) / features['line_count']
        else:
            features['error_to_line_ratio'] = 0
            
        return features
    
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
        Predict anomaly for a single session
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract text features
        text_features = self.text_vectorizer.transform([session_text]).toarray()
        svm_score = self.svm_model.decision_function(text_features)[0]
        svm_probability = 1 / (1 + np.exp(svm_score))
        
        # Extract numerical features
        num_features_dict = self.extract_numerical_features(session_text)
        num_features = np.array([list(num_features_dict.values())])
        num_features = self.scaler.transform(num_features)
        iso_score = self.isolation_model.decision_function(num_features)[0]
        iso_probability = 1 / (1 + np.exp(iso_score))
        
        # Ensemble prediction
        ensemble_score = self.text_weight * svm_probability + self.statistical_weight * iso_probability
        is_anomaly = ensemble_score > self.threshold
        
        # Calculate confidence
        confidence = "HIGH" if abs(ensemble_score - 0.5) > 0.3 else "MEDIUM" if abs(ensemble_score - 0.5) > 0.15 else "LOW"
        
        return {
            'session_text': session_text,
            'text_anomaly_score': float(svm_probability),
            'statistical_anomaly_score': float(iso_probability),
            'ensemble_score': float(ensemble_score),
            'is_anomaly': bool(is_anomaly),
            'confidence': confidence,
            'threshold': self.threshold,
            'text_features': num_features_dict,
            'prediction_breakdown': {
                'text_component': {
                    'score': float(svm_probability),
                    'weight': self.text_weight,
                    'contribution': float(self.text_weight * svm_probability)
                },
                'statistical_component': {
                    'score': float(iso_probability),
                    'weight': self.statistical_weight,
                    'contribution': float(self.statistical_weight * iso_probability)
                }
            }
        }
    
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
