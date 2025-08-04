"""
Ensemble Anomaly Detector combining multiple models for robust EJ session analysis
Combines One-Class SVM, Isolation Forest, and optionally LSTM Autoencoder
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
import json
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import re

logger = logging.getLogger(__name__)

class IsolationForestDetector:
    """
    Isolation Forest detector for multivariate anomaly detection
    Complements One-Class SVM by focusing on feature-based outliers
    """
    
    def __init__(self, contamination=0.1):
        self.contamination = contamination
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.feature_names = []
        self.model_trained = False
    
    def extract_features(self, session_text: str) -> np.ndarray:
        """Extract numerical features for isolation forest"""
        lines = session_text.strip().split('\n')
        text_lower = session_text.lower()
        
        features = {
            # Session structure features
            'line_count': len(lines),
            'total_chars': len(session_text),
            'avg_line_length': np.mean([len(line) for line in lines]) if lines else 0,
            'empty_lines': sum(1 for line in lines if not line.strip()),
            
            # Error frequency features
            'error_count': len(re.findall(r'error', text_lower)),
            'fail_count': len(re.findall(r'fail', text_lower)),
            'malfunction_count': len(re.findall(r'malfunction', text_lower)),
            'timeout_count': len(re.findall(r'timeout', text_lower)),
            
            # Hardware-specific features
            'hardware_mentions': len(re.findall(r'hardware', text_lower)),
            'power_reset_count': len(re.findall(r'power.*reset|reset.*power|power-up/reset', text_lower)),
            'cim_mentions': len(re.findall(r'cim', text_lower)),
            'recovery_failures': len(re.findall(r'recovery.*fail', text_lower)),
            'capture_failures': len(re.findall(r'capture.*fail', text_lower)),
            
            # Transaction features
            'card_mentions': len(re.findall(r'card', text_lower)),
            'pin_mentions': len(re.findall(r'pin', text_lower)),
            'cash_mentions': len(re.findall(r'cash', text_lower)),
            'transaction_mentions': len(re.findall(r'transaction', text_lower)),
            'customer_mentions': len(re.findall(r'customer', text_lower)),
            
            # Critical pattern combinations
            'critical_hardware_patterns': len(re.findall(
                r'power-up/reset|hardware.*error|cim-reset|recovery.*failed|capture.*failed',
                text_lower
            )),
            
            # Ratio features
            'error_to_line_ratio': 0,
            'hardware_to_transaction_ratio': 0,
        }
        
        # Calculate ratios (avoid division by zero)
        if features['line_count'] > 0:
            features['error_to_line_ratio'] = (
                features['error_count'] + features['fail_count']
            ) / features['line_count']
        
        if features['transaction_mentions'] > 0:
            features['hardware_to_transaction_ratio'] = (
                features['hardware_mentions']
            ) / features['transaction_mentions']
        
        self.feature_names = list(features.keys())
        return np.array(list(features.values()))
    
    def train_model(self, ej_sessions: List[Dict]):
        """Train isolation forest on normal sessions"""
        logger.info("Training Isolation Forest detector")
        
        # Filter normal sessions
        normal_sessions = [
            session for session in ej_sessions 
            if not session.get('is_anomaly', False)
        ]
        
        # Extract features
        features_list = []
        for session in normal_sessions:
            session_text = session.get('raw_text', session.get('text', ''))
            if session_text.strip():
                features = self.extract_features(session_text)
                features_list.append(features)
        
        if not features_list:
            raise ValueError("No valid training data")
        
        X = np.array(features_list)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train isolation forest
        self.isolation_forest.fit(X_scaled)
        self.model_trained = True
        
        logger.info(f"Isolation Forest trained on {len(X)} sessions with {len(self.feature_names)} features")
        
        return {
            'training_samples': len(X),
            'feature_count': len(self.feature_names),
            'contamination': self.contamination
        }
    
    def predict_anomaly(self, session_text: str, session_id: str = None) -> Dict[str, Any]:
        """Predict anomaly using isolation forest"""
        if not self.model_trained:
            raise ValueError("Model not trained")
        
        try:
            # Extract features
            features = self.extract_features(session_text)
            features_scaled = self.scaler.transform([features])
            
            # Predict
            prediction = self.isolation_forest.predict(features_scaled)[0]
            anomaly_score = self.isolation_forest.decision_function(features_scaled)[0]
            
            # Convert to probability (isolation forest returns negative scores for anomalies)
            anomaly_probability = 1 / (1 + np.exp(anomaly_score))
            is_anomaly = prediction == -1
            
            return {
                'session_id': session_id,
                'is_anomaly': bool(is_anomaly),
                'anomaly_probability': float(anomaly_probability),
                'anomaly_score': float(anomaly_score),
                'detection_method': 'isolation_forest',
                'prediction_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in isolation forest prediction: {e}")
            return {
                'error': str(e),
                'session_id': session_id,
                'is_anomaly': False,
                'anomaly_probability': 0.0
            }

class EnsembleAnomalyDetector:
    """
    Ensemble detector combining multiple anomaly detection models
    """
    
    def __init__(self, model_dir="/app/data/models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize component models
        self.svm_detector = None  # Will be imported when needed
        self.isolation_detector = IsolationForestDetector()
        self.lstm_detector = None  # Optional third model
        
        # Ensemble configuration
        self.weights = {
            'svm': 0.6,           # Higher weight for text-based detection
            'isolation': 0.4,     # Lower weight for feature-based detection
            'lstm': 0.0           # Optional, adjust when using 3-model ensemble
        }
        
        self.model_trained = False
        self.ensemble_stats = {}
    
    def set_svm_detector(self, svm_detector):
        """Set the SVM detector (dependency injection)"""
        self.svm_detector = svm_detector
    
    def set_lstm_detector(self, lstm_detector):
        """Set optional LSTM detector for 3-model ensemble"""
        self.lstm_detector = lstm_detector
        # Adjust weights for 3-model ensemble
        self.weights = {'svm': 0.4, 'isolation': 0.3, 'lstm': 0.3}
    
    def train_models(self, ej_sessions: List[Dict]):
        """Train all component models"""
        logger.info("Training ensemble models")
        
        training_results = {}
        
        # Train SVM detector
        if self.svm_detector:
            try:
                svm_result = self.svm_detector.train_model(ej_sessions)
                training_results['svm'] = svm_result
                logger.info("✅ SVM detector trained successfully")
            except Exception as e:
                logger.error(f"❌ SVM training failed: {e}")
                training_results['svm'] = {'error': str(e)}
        
        # Train Isolation Forest
        try:
            isolation_result = self.isolation_detector.train_model(ej_sessions)
            training_results['isolation'] = isolation_result
            logger.info("✅ Isolation Forest trained successfully")
        except Exception as e:
            logger.error(f"❌ Isolation Forest training failed: {e}")
            training_results['isolation'] = {'error': str(e)}
        
        # Train LSTM detector (if available)
        if self.lstm_detector:
            try:
                lstm_result = self.lstm_detector.train_model(ej_sessions)
                training_results['lstm'] = lstm_result
                logger.info("✅ LSTM Autoencoder trained successfully")
            except Exception as e:
                logger.error(f"❌ LSTM training failed: {e}")
                training_results['lstm'] = {'error': str(e)}
        
        self.model_trained = True
        self.ensemble_stats = training_results
        
        logger.info("🎯 Ensemble training completed")
        return training_results
    
    def predict_anomaly(self, session_text: str, session_id: str = None) -> Dict[str, Any]:
        """
        Ensemble prediction combining multiple models
        """
        if not self.model_trained:
            raise ValueError("Ensemble not trained. Call train_models() first.")
        
        individual_results = {}
        probabilities = []
        votes = []
        
        # Get SVM prediction
        if self.svm_detector:
            try:
                svm_result = self.svm_detector.predict_anomaly(session_text, session_id)
                individual_results['svm'] = svm_result
                probabilities.append(('svm', svm_result.get('anomaly_probability', 0.0)))
                votes.append(svm_result.get('is_anomaly', False))
            except Exception as e:
                logger.error(f"SVM prediction error: {e}")
                individual_results['svm'] = {'error': str(e)}
                probabilities.append(('svm', 0.0))
                votes.append(False)
        
        # Get Isolation Forest prediction
        try:
            iso_result = self.isolation_detector.predict_anomaly(session_text, session_id)
            individual_results['isolation'] = iso_result
            probabilities.append(('isolation', iso_result.get('anomaly_probability', 0.0)))
            votes.append(iso_result.get('is_anomaly', False))
        except Exception as e:
            logger.error(f"Isolation Forest prediction error: {e}")
            individual_results['isolation'] = {'error': str(e)}
            probabilities.append(('isolation', 0.0))
            votes.append(False)
        
        # Get LSTM prediction (if available)
        if self.lstm_detector:
            try:
                lstm_result = self.lstm_detector.predict_anomaly(session_text, session_id)
                individual_results['lstm'] = lstm_result
                probabilities.append(('lstm', lstm_result.get('anomaly_probability', 0.0)))
                votes.append(lstm_result.get('is_anomaly', False))
            except Exception as e:
                logger.error(f"LSTM prediction error: {e}")
                individual_results['lstm'] = {'error': str(e)}
                probabilities.append(('lstm', 0.0))
                votes.append(False)
        
        # Calculate ensemble score
        ensemble_probability = 0.0
        for model_name, prob in probabilities:
            weight = self.weights.get(model_name, 0.0)
            ensemble_probability += weight * prob
        
        # Consensus analysis
        consensus_votes = sum(votes)
        total_models = len(votes)
        agreement_score = consensus_votes / total_models if total_models > 0 else 0.0
        
        # Final decision
        is_anomaly = ensemble_probability > 0.5 or consensus_votes >= (total_models // 2 + 1)
        
        # Confidence based on agreement
        if consensus_votes == 0 or consensus_votes == total_models:
            confidence = 'HIGH'  # All models agree
        elif consensus_votes == total_models - 1 or consensus_votes == 1:
            confidence = 'MEDIUM'  # Most models agree
        else:
            confidence = 'LOW'  # Models disagree
        
        # Create detailed result
        ensemble_result = {
            'session_id': session_id,
            'is_anomaly': bool(is_anomaly),
            'ensemble_probability': float(ensemble_probability),
            'consensus_votes': f"{consensus_votes}/{total_models}",
            'agreement_score': float(agreement_score),
            'confidence_level': confidence,
            'detection_method': 'ensemble',
            'prediction_timestamp': datetime.now().isoformat(),
            
            # Individual model results
            'individual_results': individual_results,
            
            # Detailed analysis
            'ensemble_analysis': {
                'weights_used': self.weights,
                'model_probabilities': dict(probabilities),
                'voting_result': votes,
                'high_confidence': confidence == 'HIGH'
            }
        }
        
        return ensemble_result
    
    def explain_prediction(self, session_id: str = None, result: Dict = None) -> Dict[str, Any]:
        """Provide detailed explanation of ensemble prediction"""
        if not result:
            return {'error': 'No result provided for explanation'}
        
        explanation = {
            'ensemble_summary': {
                'final_decision': 'ANOMALY' if result['is_anomaly'] else 'NORMAL',
                'confidence': result['confidence_level'],
                'ensemble_score': result['ensemble_probability'],
                'consensus': result['consensus_votes']
            },
            'model_breakdown': [],
            'key_factors': []
        }
        
        # Analyze each model's contribution
        for model_name, model_result in result['individual_results'].items():
            if 'error' not in model_result:
                explanation['model_breakdown'].append({
                    'model': model_name,
                    'decision': 'ANOMALY' if model_result['is_anomaly'] else 'NORMAL',
                    'probability': model_result['anomaly_probability'],
                    'weight': self.weights.get(model_name, 0.0),
                    'contribution': self.weights.get(model_name, 0.0) * model_result['anomaly_probability']
                })
        
        # Key factors for decision
        if result['is_anomaly']:
            explanation['key_factors'].extend([
                f"Ensemble probability ({result['ensemble_probability']:.3f}) exceeds threshold (0.5)",
                f"Model consensus: {result['consensus_votes']} models detected anomaly",
                f"Confidence level: {result['confidence_level']}"
            ])
        else:
            explanation['key_factors'].extend([
                f"Ensemble probability ({result['ensemble_probability']:.3f}) below threshold (0.5)",
                f"Model consensus: {result['consensus_votes']} indicates normal session",
                f"Confidence level: {result['confidence_level']}"
            ])
        
        return explanation
    
    def save_ensemble(self, model_path: str = None):
        """Save the ensemble configuration and models"""
        if model_path is None:
            model_path = os.path.join(self.model_dir, 'ensemble_detector.pkl')
        
        ensemble_data = {
            'weights': self.weights,
            'model_trained': self.model_trained,
            'ensemble_stats': self.ensemble_stats,
            'isolation_detector': self.isolation_detector
        }
        
        joblib.dump(ensemble_data, model_path)
        
        # Save individual models
        if self.svm_detector and hasattr(self.svm_detector, 'save_model'):
            self.svm_detector.save_model()
        
        if self.lstm_detector and hasattr(self.lstm_detector, 'save_model'):
            self.lstm_detector.save_model()
        
        logger.info(f"Ensemble saved to {model_path}")
    
    def load_ensemble(self, model_path: str = None):
        """Load the ensemble configuration and models"""
        if model_path is None:
            model_path = os.path.join(self.model_dir, 'ensemble_detector.pkl')
        
        ensemble_data = joblib.load(model_path)
        
        self.weights = ensemble_data['weights']
        self.model_trained = ensemble_data['model_trained']
        self.ensemble_stats = ensemble_data['ensemble_stats']
        self.isolation_detector = ensemble_data['isolation_detector']
        
        # Load individual models (would need to be implemented separately)
        logger.info(f"Ensemble loaded from {model_path}")

# Alias for easy integration
BERTDeepLogAnomalyDetector = EnsembleAnomalyDetector
