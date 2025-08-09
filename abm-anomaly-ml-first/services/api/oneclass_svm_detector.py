"""
One-Class SVM Anomaly Detector for EJ Sessions
Specialized for detecting hardware errors and unusual patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.svm import OneClassSVM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import logging
from datetime import datetime
import re
import os

logger = logging.getLogger(__name__)

class OneClassSVMAnomalyDetector:
    """
    One-Class SVM model specifically designed for EJ session anomaly detection.
    Trains only on normal data and detects anomalies as outliers.
    """
    
    def __init__(self, model_dir="/app/data/models", contamination=0.1):
        """
        Initialize One-Class SVM Anomaly Detector
        
        Args:
            model_dir: Directory to save trained models
            contamination: Expected proportion of anomalies in training data
        """
        self.model_dir = model_dir
        self.contamination = contamination
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize components
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),  # Capture phrases like "POWER-UP/RESET"
            stop_words=None,  # Keep all words for technical logs
            lowercase=True,
            token_pattern=r'\b\w+(?:[-/]\w+)*\b'  # Handle hyphenated terms
        )
        
        self.scaler = StandardScaler()
        
        self.svm_model = OneClassSVM(
            kernel='rbf',
            gamma='scale',
            nu=contamination,  # Expected fraction of anomalies
            cache_size=1000
        )
        
        # Feature extractors
        self.feature_extractors = [
            self._extract_text_features,
            self._extract_error_features,
            self._extract_session_features,
            self._extract_hardware_features
        ]
        
        self.model_trained = False
        self.feature_names = []
        
    def _extract_text_features(self, session_text: str) -> Dict[str, float]:
        """Extract TF-IDF text features"""
        return {'raw_text': session_text}  # Will be processed by vectorizer
    
    def _extract_error_features(self, session_text: str) -> Dict[str, float]:
        """Extract error-related features"""
        text_lower = session_text.lower()
        
        # Error indicators
        error_patterns = [
            r'error', r'fail', r'malfunction', r'fault', r'exception',
            r'timeout', r'abort', r'reject', r'denied', r'invalid'
        ]
        
        features = {}
        for pattern in error_patterns:
            count = len(re.findall(pattern, text_lower))
            features[f'error_{pattern}_count'] = count
        
        # Total error count
        features['total_error_count'] = sum(features.values())
        
        return features
    
    def _extract_session_features(self, session_text: str) -> Dict[str, float]:
        """Extract session-level features"""
        lines = session_text.strip().split('\n')
        
        features = {
            'session_length_lines': len(lines),
            'session_length_chars': len(session_text),
            'avg_line_length': np.mean([len(line) for line in lines]) if lines else 0,
            'empty_lines_count': sum(1 for line in lines if not line.strip()),
        }
        
        # Transaction indicators
        transaction_patterns = [
            r'transaction', r'withdraw', r'deposit', r'balance',
            r'pin', r'card', r'cash', r'receipt'
        ]
        
        for pattern in transaction_patterns:
            count = len(re.findall(pattern, session_text.lower()))
            features[f'transaction_{pattern}_count'] = count
        
        return features
    
    def _extract_hardware_features(self, session_text: str) -> Dict[str, float]:
        """Extract hardware-specific features"""
        text_lower = session_text.lower()
        
        # Hardware error patterns (the key ones you want to detect)
        hardware_patterns = {
            'power_reset': [r'power-up/reset', r'power.*reset', r'reset'],
            'hardware_error': [r'hardware.*error', r'hardwareerror'],
            'component_failure': [r'cim-reset', r'capture.*failed', r'recovery.*failed'],
            'device_issues': [r'malfunction', r'device.*error', r'component.*error']
        }
        
        features = {}
        for category, patterns in hardware_patterns.items():
            total_count = 0
            for pattern in patterns:
                count = len(re.findall(pattern, text_lower))
                features[f'hw_{category}_{pattern.replace(".*", "_").replace("[", "").replace("]", "")}'] = count
                total_count += count
            features[f'hw_{category}_total'] = total_count
        
        # Critical hardware indicators
        critical_terms = ['power-up/reset', 'hardware error', 'cim-reset', 'recovery failed']
        features['critical_hardware_score'] = sum(
            len(re.findall(term.replace(' ', r'\s*'), text_lower))
            for term in critical_terms
        )
        
        return features
    
    def extract_features(self, session_text: str) -> np.ndarray:
        """Extract all features from a session"""
        all_features = {}
        
        # Extract from all feature extractors
        for extractor in self.feature_extractors:
            features = extractor(session_text)
            all_features.update(features)
        
        # Handle text separately for TF-IDF
        text_content = all_features.pop('raw_text', '')
        
        if self.model_trained:
            # Transform text with fitted vectorizer
            text_features = self.vectorizer.transform([text_content]).toarray()[0]
        else:
            # During training, we'll fit the vectorizer
            text_features = None
        
        # Convert other features to array
        feature_values = list(all_features.values())
        
        if text_features is not None:
            # Combine text and other features
            combined_features = np.concatenate([text_features, feature_values])
        else:
            # During training preparation
            combined_features = np.array(feature_values)
        
        return combined_features, all_features, text_content
    
    def prepare_training_data(self, ej_sessions: List[Dict]) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare training data from normal EJ sessions only
        
        Args:
            ej_sessions: List of EJ session dictionaries
            
        Returns:
            Feature matrix and session texts
        """
        logger.info(f"Preparing training data from {len(ej_sessions)} sessions")
        
        # Filter to normal sessions only (One-Class SVM trains on normal data)
        normal_sessions = [
            session for session in ej_sessions 
            if not session.get('is_anomaly', False)
        ]
        
        logger.info(f"Using {len(normal_sessions)} normal sessions for training")
        
        session_texts = []
        feature_lists = []
        
        for session in normal_sessions:
            session_text = session.get('raw_text', session.get('text', ''))
            if not session_text.strip():
                continue
                
            session_texts.append(session_text)
            
            # Extract non-text features
            _, other_features, _ = self.extract_features(session_text)
            feature_lists.append(list(other_features.values()))
        
        # Fit TF-IDF vectorizer on all session texts
        text_features = self.vectorizer.fit_transform(session_texts).toarray()
        
        # Combine text and other features
        other_features_array = np.array(feature_lists)
        combined_features = np.concatenate([text_features, other_features_array], axis=1)
        
        # Store feature names for interpretation
        self.feature_names = (
            list(self.vectorizer.get_feature_names_out()) + 
            list(feature_lists[0]) if feature_lists else []
        )
        
        logger.info(f"Extracted {combined_features.shape[1]} features from {combined_features.shape[0]} sessions")
        
        return combined_features, session_texts
    
    def train_model(self, ej_sessions: List[Dict]):
        """
        Train the One-Class SVM model
        
        Args:
            ej_sessions: List of EJ session dictionaries
        """
        logger.info("Starting One-Class SVM training")
        
        # Prepare training data
        X_train, session_texts = self.prepare_training_data(ej_sessions)
        
        if len(X_train) == 0:
            raise ValueError("No training data available")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train One-Class SVM
        self.svm_model.fit(X_train_scaled)
        
        self.model_trained = True
        logger.info("One-Class SVM training completed")
        
        # Evaluate on training data to get baseline
        train_predictions = self.svm_model.predict(X_train_scaled)
        n_outliers = np.sum(train_predictions == -1)
        
        logger.info(f"Training evaluation: {n_outliers}/{len(train_predictions)} sessions flagged as outliers")
        
        return {
            'training_samples': len(X_train),
            'features_count': X_train.shape[1],
            'outliers_in_training': n_outliers,
            'outlier_rate': n_outliers / len(train_predictions)
        }
    
    def predict_anomaly(self, session_text: str, session_id: str = None) -> Dict[str, Any]:
        """
        Predict if a session is anomalous
        
        Args:
            session_text: Raw EJ session text
            session_id: Optional session identifier
            
        Returns:
            Dictionary containing prediction results
        """
        if not self.model_trained:
            raise ValueError("Model not trained. Call train_model() first.")
        
        try:
            # Extract features
            combined_features, other_features, text_content = self.extract_features(session_text)
            
            # Scale features
            features_scaled = self.scaler.transform([combined_features])
            
            # Predict
            prediction = self.svm_model.predict(features_scaled)[0]
            decision_score = self.svm_model.decision_function(features_scaled)[0]
            
            # Convert to probability-like score
            # SVM decision function: positive = normal, negative = anomaly
            anomaly_probability = 1 / (1 + np.exp(decision_score))  # Sigmoid transformation
            is_anomaly = prediction == -1
            
            # Calculate confidence based on distance from decision boundary
            confidence = abs(decision_score)
            
            # Identify key contributing features
            feature_importance = self._analyze_feature_importance(
                combined_features, other_features, text_content
            )
            
            result = {
                'session_id': session_id,
                'is_anomaly': bool(is_anomaly),
                'anomaly_probability': float(anomaly_probability),
                'confidence': float(confidence),
                'decision_score': float(decision_score),
                'prediction_timestamp': datetime.now().isoformat(),
                'detection_method': 'one_class_svm',
                'feature_analysis': feature_importance,
                'model_prediction': int(prediction)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting anomaly for session {session_id}: {e}")
            return {
                'error': str(e),
                'session_id': session_id,
                'is_anomaly': False,
                'anomaly_probability': 0.0
            }
    
    def _analyze_feature_importance(self, features: np.ndarray, other_features: Dict, text_content: str) -> Dict:
        """Analyze which features contributed to the decision"""
        # Get feature values that are non-zero
        significant_features = {}
        
        # Analyze text features (TF-IDF)
        text_features = features[:len(self.vectorizer.get_feature_names_out())]
        vocab = self.vectorizer.get_feature_names_out()
        
        # Get top text features
        top_text_indices = np.argsort(text_features)[-10:]  # Top 10
        for idx in top_text_indices:
            if text_features[idx] > 0:
                significant_features[f'text_{vocab[idx]}'] = float(text_features[idx])
        
        # Analyze other features
        for feature_name, value in other_features.items():
            if value > 0:
                significant_features[feature_name] = value
        
        return significant_features
    
    def get_tfidf_feature_importance(self, session_text: str, top_n: int = 15) -> List[Dict[str, Any]]:
        """
        Get TF-IDF feature importance for a session (especially for outliers)
        
        Args:
            session_text: Raw session text
            top_n: Number of top features to return
            
        Returns:
            List of dictionaries with word and importance scores
        """
        if not self.model_trained:
            raise ValueError("Model not trained")
        
        # Transform text to TF-IDF features
        tfidf_features = self.vectorizer.transform([session_text]).toarray()[0]
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get top features by TF-IDF score
        top_indices = np.argsort(tfidf_features)[-top_n:][::-1]  # Descending order
        
        top_features = []
        for idx in top_indices:
            if tfidf_features[idx] > 0:  # Only include non-zero features
                top_features.append({
                    'word': feature_names[idx],
                    'tfidf_score': float(tfidf_features[idx]),
                    'importance': float(tfidf_features[idx] / np.max(tfidf_features))  # Normalized
                })
        
        return top_features
    
    def get_outlier_analysis(self, session_text: str, session_id: str = None) -> Dict[str, Any]:
        """
        Complete outlier analysis including TF-IDF contributions
        
        Args:
            session_text: Raw session text
            session_id: Optional session ID
            
        Returns:
            Dictionary with prediction and detailed TF-IDF analysis
        """
        # Get basic prediction
        prediction_result = self.predict_anomaly(session_text, session_id)
        
        # Add detailed TF-IDF analysis
        if prediction_result.get('is_anomaly', False):
            tfidf_analysis = self.get_tfidf_feature_importance(session_text, top_n=20)
            prediction_result['tfidf_analysis'] = tfidf_analysis
            
            # Group words by categories for better visualization
            prediction_result['word_categories'] = self._categorize_important_words(tfidf_analysis)
        
        return prediction_result
    
    def _categorize_important_words(self, tfidf_features: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize TF-IDF words into logical groups"""
        categories = {
            'error_terms': [],
            'hardware_terms': [],
            'transaction_terms': [],
            'status_terms': [],
            'other_terms': []
        }
        
        # Define category patterns
        category_patterns = {
            'error_terms': ['error', 'fail', 'timeout', 'abort', 'reject', 'invalid', 'malfunction'],
            'hardware_terms': ['power', 'reset', 'hardware', 'device', 'component', 'cim', 'recovery'],
            'transaction_terms': ['transaction', 'withdraw', 'deposit', 'balance', 'pin', 'card', 'cash'],
            'status_terms': ['start', 'end', 'success', 'complete', 'activated', 'taken', 'inserted']
        }
        
        for feature in tfidf_features:
            word = feature['word'].lower()
            categorized = False
            
            for category, patterns in category_patterns.items():
                if any(pattern in word for pattern in patterns):
                    categories[category].append(feature)
                    categorized = True
                    break
            
            if not categorized:
                categories['other_terms'].append(feature)
        
        return categories
    
    def save_model(self, model_path: str = None):
        """Save the trained model"""
        if model_path is None:
            model_path = os.path.join(self.model_dir, 'oneclass_svm_model.pkl')
        
        model_data = {
            'svm_model': self.svm_model,
            'vectorizer': self.vectorizer,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_trained': self.model_trained,
            'contamination': self.contamination
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str = None):
        """Load a saved model"""
        if model_path is None:
            model_path = os.path.join(self.model_dir, 'oneclass_svm_model.pkl')
        
        model_data = joblib.load(model_path)
        
        self.svm_model = model_data['svm_model']
        self.vectorizer = model_data['vectorizer']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_trained = model_data['model_trained']
        self.contamination = model_data['contamination']
        
        logger.info(f"Model loaded from {model_path}")

# Alias for compatibility
BERTDeepLogAnomalyDetector = OneClassSVMAnomalyDetector
