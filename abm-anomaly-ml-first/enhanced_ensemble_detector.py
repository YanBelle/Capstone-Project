"""
Enhanced Ensemble Anomaly Detector with DBSCAN Integration
Combines traditional anomaly detection with density-based clustering for improved accuracy
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from typing import Dict, List, Any, Tuple, Optional
import json
import joblib
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedEnsembleDetector:
    """
    Enhanced ensemble anomaly detector with DBSCAN clustering capabilities
    
    Features:
    - Multi-modal clustering (text, numerical, combined features)
    - Automatic parameter optimization
    - Comprehensive model information
    - Density-based anomaly scoring
    """
    
    def __init__(self, 
                 contamination: float = 0.1,
                 random_state: int = 42,
                 models_dir: str = "./models"):
        """
        Initialize the enhanced ensemble detector
        
        Args:
            contamination: Expected proportion of outliers
            random_state: Random seed for reproducibility
            models_dir: Directory to save/load models
        """
        self.contamination = contamination
        self.random_state = random_state
        self.models_dir = models_dir
        
        # Core models
        self.tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.one_class_svm = OneClassSVM(nu=contamination, kernel='rbf', gamma='scale')
        self.isolation_forest = IsolationForest(contamination=contamination, random_state=random_state)
        
        # DBSCAN models for different feature spaces
        self.dbscan_text = None
        self.dbscan_numerical = None
        self.dbscan_combined = None
        
        # Scalers and transformers
        self.text_scaler = StandardScaler()
        self.numerical_scaler = StandardScaler()
        self.combined_scaler = StandardScaler()
        self.pca_reducer = PCA(n_components=0.95)  # Keep 95% of variance
        
        # Training data and results
        self.text_features = None
        self.numerical_features = None
        self.combined_features = None
        self.cluster_results = {}
        self.is_trained = False
        self.training_timestamp = None
        
        # Model metadata
        self.feature_names = []
        self.training_stats = {}
        
        # Ensure models directory exists
        os.makedirs(models_dir, exist_ok=True)
    
    def extract_features(self, sessions: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract text and numerical features from session data
        
        Args:
            sessions: List of session dictionaries
            
        Returns:
            Tuple of (text_features, numerical_features)
        """
        logger.info(f"Extracting features from {len(sessions)} sessions")
        
        # Extract text data (EJ logs)
        text_data = []
        numerical_data = []
        
        for session in sessions:
            # Combine EJ logs into text
            ej_text = ""
            if 'ej_logs' in session and session['ej_logs']:
                if isinstance(session['ej_logs'], list):
                    ej_text = " ".join([str(log) for log in session['ej_logs']])
                else:
                    ej_text = str(session['ej_logs'])
            
            text_data.append(ej_text)
            
            # Extract numerical features
            num_features = []
            
            # Session-level features
            num_features.append(len(session.get('ej_logs', [])))  # Log count
            num_features.append(session.get('duration', 0))  # Session duration
            num_features.append(len(str(ej_text)))  # Text length
            
            # Transaction features
            transactions = session.get('transactions', [])
            num_features.append(len(transactions))  # Transaction count
            
            # Amount statistics
            amounts = []
            for trans in transactions:
                if 'amount' in trans and trans['amount'] is not None:
                    try:
                        amounts.append(float(trans['amount']))
                    except (ValueError, TypeError):
                        pass
            
            num_features.extend([
                len(amounts),  # Valid amount count
                np.mean(amounts) if amounts else 0,  # Mean amount
                np.std(amounts) if len(amounts) > 1 else 0,  # Amount std
                np.max(amounts) if amounts else 0,  # Max amount
                np.min(amounts) if amounts else 0,  # Min amount
            ])
            
            # Operation type counts
            operation_types = [trans.get('operation_type', '') for trans in transactions]
            unique_ops = list(set(operation_types))
            num_features.append(len(unique_ops))  # Unique operation types
            
            # Common operation counts
            for op_type in ['CASH_DISPENSING', 'BALANCE_INQUIRY', 'DEPOSIT', 'TRANSFER']:
                num_features.append(operation_types.count(op_type))
            
            # Error and status features
            error_count = sum(1 for trans in transactions if trans.get('status') == 'ERROR')
            success_count = sum(1 for trans in transactions if trans.get('status') == 'SUCCESS')
            num_features.extend([error_count, success_count])
            
            # Time-based features
            timestamps = [trans.get('timestamp') for trans in transactions if trans.get('timestamp')]
            if len(timestamps) > 1:
                time_diffs = []
                for i in range(1, len(timestamps)):
                    try:
                        diff = (pd.to_datetime(timestamps[i]) - pd.to_datetime(timestamps[i-1])).total_seconds()
                        time_diffs.append(diff)
                    except:
                        pass
                
                num_features.extend([
                    np.mean(time_diffs) if time_diffs else 0,  # Mean time between transactions
                    np.std(time_diffs) if len(time_diffs) > 1 else 0,  # Time variance
                    np.max(time_diffs) if time_diffs else 0,  # Max gap
                ])
            else:
                num_features.extend([0, 0, 0])
            
            # Card and account features
            unique_cards = len(set(trans.get('card_number', '') for trans in transactions if trans.get('card_number')))
            unique_accounts = len(set(trans.get('account_number', '') for trans in transactions if trans.get('account_number')))
            num_features.extend([unique_cards, unique_accounts])
            
            # Location features
            unique_locations = len(set(trans.get('location', '') for trans in transactions if trans.get('location')))
            num_features.append(unique_locations)
            
            # ATM-specific features
            atm_ids = [trans.get('atm_id', '') for trans in transactions if trans.get('atm_id')]
            unique_atms = len(set(atm_ids))
            num_features.append(unique_atms)
            
            # Pad to fixed length (37 features total)
            while len(num_features) < 37:
                num_features.append(0)
            
            numerical_data.append(num_features[:37])  # Ensure exactly 37 features
        
        # Convert text to TF-IDF features
        text_features = self.tfidf_vectorizer.fit_transform(text_data).toarray()
        numerical_features = np.array(numerical_data)
        
        # Store feature names for interpretability
        self.feature_names = {
            'text_features': self.tfidf_vectorizer.get_feature_names_out().tolist(),
            'numerical_features': [
                'log_count', 'duration', 'text_length', 'transaction_count',
                'valid_amount_count', 'mean_amount', 'amount_std', 'max_amount', 'min_amount',
                'unique_operation_types', 'cash_dispensing_count', 'balance_inquiry_count',
                'deposit_count', 'transfer_count', 'error_count', 'success_count',
                'mean_time_between_trans', 'time_variance', 'max_time_gap',
                'unique_cards', 'unique_accounts', 'unique_locations', 'unique_atms'
            ] + [f'feature_{i}' for i in range(23, 37)]  # Padding features
        }
        
        logger.info(f"Extracted text features: {text_features.shape}")
        logger.info(f"Extracted numerical features: {numerical_features.shape}")
        
        return text_features, numerical_features
    
    def optimize_dbscan_parameters(self, features: np.ndarray, 
                                  feature_type: str = "unknown") -> Tuple[float, int]:
        """
        Optimize DBSCAN parameters using silhouette analysis
        
        Args:
            features: Feature matrix
            feature_type: Type of features for logging
            
        Returns:
            Tuple of (optimal_eps, optimal_min_samples)
        """
        logger.info(f"Optimizing DBSCAN parameters for {feature_type} features")
        
        best_silhouette = -1
        best_eps = 0.5
        best_min_samples = 5
        
        # Test different parameter combinations
        eps_values = np.arange(0.1, 2.0, 0.2)
        min_samples_values = [3, 5, 10, 15]
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = dbscan.fit_predict(features)
                    
                    # Check if we have valid clusters
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    if n_clusters > 1:
                        # Calculate silhouette score for non-noise points
                        valid_indices = labels != -1
                        if np.sum(valid_indices) > min_samples:
                            silhouette = silhouette_score(features[valid_indices], labels[valid_indices])
                            
                            if silhouette > best_silhouette:
                                best_silhouette = silhouette
                                best_eps = eps
                                best_min_samples = min_samples
                
                except Exception as e:
                    logger.debug(f"DBSCAN failed with eps={eps}, min_samples={min_samples}: {e}")
                    continue
        
        logger.info(f"Best parameters for {feature_type}: eps={best_eps}, min_samples={best_min_samples}, silhouette={best_silhouette:.3f}")
        return best_eps, best_min_samples
    
    def perform_clustering_analysis(self, features: np.ndarray, 
                                  feature_type: str) -> Dict[str, Any]:
        """
        Perform comprehensive clustering analysis
        
        Args:
            features: Feature matrix
            feature_type: Type of features
            
        Returns:
            Dictionary with clustering results
        """
        logger.info(f"Performing clustering analysis for {feature_type} features")
        
        # Optimize parameters
        eps, min_samples = self.optimize_dbscan_parameters(features, feature_type)
        
        # Perform clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(features)
        
        # Calculate metrics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        noise_ratio = n_noise / len(labels) if len(labels) > 0 else 0
        
        # Calculate silhouette score for valid clusters
        silhouette = -1
        if n_clusters > 1:
            valid_indices = labels != -1
            if np.sum(valid_indices) > 1:
                try:
                    silhouette = silhouette_score(features[valid_indices], labels[valid_indices])
                except:
                    silhouette = -1
        
        # Cluster analysis
        cluster_info = {}
        for cluster_id in set(labels):
            if cluster_id != -1:  # Skip noise
                cluster_mask = labels == cluster_id
                cluster_features = features[cluster_mask]
                
                cluster_info[int(cluster_id)] = {
                    'size': int(np.sum(cluster_mask)),
                    'percentage': float(np.sum(cluster_mask) / len(labels) * 100),
                    'center': cluster_features.mean(axis=0).tolist(),
                    'std': cluster_features.std(axis=0).tolist()
                }
        
        results = {
            'dbscan_model': dbscan,
            'labels': labels.tolist(),
            'n_clusters': int(n_clusters),
            'n_noise': int(n_noise),
            'noise_ratio': float(noise_ratio),
            'silhouette_score': float(silhouette),
            'eps': float(eps),
            'min_samples': int(min_samples),
            'cluster_info': cluster_info,
            'feature_type': feature_type
        }
        
        logger.info(f"{feature_type} clustering: {n_clusters} clusters, {noise_ratio:.1%} noise, silhouette={silhouette:.3f}")
        
        return results
    
    def train(self, sessions: List[str]) -> Dict[str, Any]:
        """
        Train the enhanced ensemble detector
        
        Args:
            sessions: List of training session strings
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Training enhanced ensemble detector with {len(sessions)} sessions")
        
        # Store training sessions for cluster analysis
        self.training_sessions = sessions
        
        # Extract features
        text_features, numerical_features = self.extract_features(sessions)
        
        # Scale features
        text_features_scaled = self.text_scaler.fit_transform(text_features)
        numerical_features_scaled = self.numerical_scaler.fit_transform(numerical_features)
        
        # Create combined features
        combined_features = np.hstack([text_features_scaled, numerical_features_scaled])
        
        # Apply PCA to combined features for dimensionality reduction
        combined_features_pca = self.pca_reducer.fit_transform(combined_features)
        combined_features_scaled = self.combined_scaler.fit_transform(combined_features_pca)
        
        # Store processed features
        self.text_features = text_features_scaled
        self.numerical_features = numerical_features_scaled
        self.combined_features = combined_features_scaled
        
        # Train traditional models
        logger.info("Training One-Class SVM...")
        self.one_class_svm.fit(text_features_scaled)
        
        logger.info("Training Isolation Forest...")
        self.isolation_forest.fit(combined_features_scaled)
        
        # Perform clustering analysis for each feature space
        logger.info("Performing clustering analysis...")
        
        # Text features clustering
        text_results = self.perform_clustering_analysis(text_features_scaled, "text")
        self.dbscan_text = text_results['dbscan_model']
        
        # Numerical features clustering
        numerical_results = self.perform_clustering_analysis(numerical_features_scaled, "numerical")
        self.dbscan_numerical = numerical_results['dbscan_model']
        
        # Combined features clustering
        combined_results = self.perform_clustering_analysis(combined_features_scaled, "combined")
        self.dbscan_combined = combined_results['dbscan_model']
        
        # Store clustering results
        self.cluster_results = {
            'text': text_results,
            'numerical': numerical_results,
            'combined': combined_results
        }
        
        # Calculate training statistics
        self.training_stats = {
            'n_sessions': len(sessions),
            'text_features_shape': text_features.shape,
            'numerical_features_shape': numerical_features.shape,
            'combined_features_shape': combined_features_scaled.shape,
            'pca_explained_variance': float(self.pca_reducer.explained_variance_ratio_.sum()),
            'training_timestamp': datetime.now().isoformat()
        }
        
        # Mark as trained
        self.is_trained = True
        self.training_timestamp = datetime.now().isoformat()
        
        logger.info("Training completed successfully")
        
        # Save models
        self.save_models()
        
        return {
            'status': 'success',
            'training_stats': self.training_stats,
            'cluster_results': self.cluster_results
        }
    
    def predict(self, sessions: List[Dict]) -> Dict[str, Any]:
        """
        Predict anomalies using the ensemble approach
        
        Args:
            sessions: List of sessions to predict
            
        Returns:
            Prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        logger.info(f"Making predictions for {len(sessions)} sessions")
        
        # Extract and scale features
        text_features, numerical_features = self.extract_features(sessions)
        text_features_scaled = self.text_scaler.transform(text_features)
        numerical_features_scaled = self.numerical_scaler.transform(numerical_features)
        
        # Create combined features
        combined_features = np.hstack([text_features_scaled, numerical_features_scaled])
        combined_features_pca = self.pca_reducer.transform(combined_features)
        combined_features_scaled = self.combined_scaler.transform(combined_features_pca)
        
        # Get predictions from traditional models
        svm_scores = self.one_class_svm.decision_function(text_features_scaled)
        isolation_scores = self.isolation_forest.decision_function(combined_features_scaled)
        
        # Get clustering-based anomaly scores
        text_clusters = self.dbscan_text.fit_predict(text_features_scaled)
        numerical_clusters = self.dbscan_numerical.fit_predict(numerical_features_scaled)
        combined_clusters = self.dbscan_combined.fit_predict(combined_features_scaled)
        
        # Store cluster labels for expert labeling
        self.text_cluster_labels = text_clusters
        self.numerical_cluster_labels = numerical_clusters
        self.combined_cluster_labels = combined_clusters
        
        # Calculate density-based anomaly scores
        density_scores = []
        for i in range(len(sessions)):
            # Sessions in noise clusters get higher anomaly scores
            text_anomaly = 1.0 if text_clusters[i] == -1 else 0.0
            numerical_anomaly = 1.0 if numerical_clusters[i] == -1 else 0.0
            combined_anomaly = 1.0 if combined_clusters[i] == -1 else 0.0
            
            # Weighted combination
            density_score = 0.3 * text_anomaly + 0.3 * numerical_anomaly + 0.4 * combined_anomaly
            density_scores.append(density_score)
        
        density_scores = np.array(density_scores)
        
        # Normalize scores to [0, 1] range
        svm_scores_norm = (svm_scores - svm_scores.min()) / (svm_scores.max() - svm_scores.min() + 1e-8)
        isolation_scores_norm = (isolation_scores - isolation_scores.min()) / (isolation_scores.max() - isolation_scores.min() + 1e-8)
        
        # Ensemble scoring (higher means more anomalous)
        ensemble_scores = (
            0.3 * (1 - svm_scores_norm) +  # Invert SVM scores (lower = more anomalous)
            0.3 * (1 - isolation_scores_norm) +  # Invert isolation scores
            0.4 * density_scores  # Density scores already proper direction
        )
        
        # Determine anomalies based on threshold
        threshold = np.percentile(ensemble_scores, (1 - self.contamination) * 100)
        predictions = (ensemble_scores > threshold).astype(int)
        
        return {
            'predictions': predictions.tolist(),
            'anomaly_scores': ensemble_scores.tolist(),
            'svm_scores': svm_scores.tolist(),
            'isolation_scores': isolation_scores.tolist(),
            'density_scores': density_scores.tolist(),
            'text_clusters': text_clusters.tolist(),
            'numerical_clusters': numerical_clusters.tolist(),
            'combined_clusters': combined_clusters.tolist(),
            'threshold': float(threshold)
        }
    
    def get_tfidf_analysis_for_session(self, session_text: str, session_id: str = None) -> Dict[str, Any]:
        """
        Get detailed TF-IDF analysis for a specific session
        
        Args:
            session_text: Raw session text
            session_id: Optional session ID
            
        Returns:
            Dictionary with TF-IDF analysis results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before analysis")
        
        # Transform text to TF-IDF features
        tfidf_features = self.tfidf_vectorizer.transform([session_text]).toarray()[0]
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # Get top features by TF-IDF score
        top_indices = np.argsort(tfidf_features)[-20:][::-1]  # Top 20, descending
        
        top_features = []
        for idx in top_indices:
            if tfidf_features[idx] > 0:  # Only include non-zero features
                top_features.append({
                    'word': feature_names[idx],
                    'tfidf_score': float(tfidf_features[idx]),
                    'importance': float(tfidf_features[idx] / np.max(tfidf_features)) if np.max(tfidf_features) > 0 else 0.0
                })
        
        # Categorize words
        word_categories = self._categorize_tfidf_words(top_features)
        
        # Get prediction for this session
        prediction_result = self.predict_single_session(session_text, session_id)
        
        return {
            'session_id': session_id,
            'tfidf_analysis': top_features,
            'word_categories': word_categories,
            'prediction_result': prediction_result,
            'vocabulary_size': len(feature_names),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def predict_single_session(self, session_text: str, session_id: str = None) -> Dict[str, Any]:
        """
        Predict anomaly for a single session
        
        Args:
            session_text: Raw session text
            session_id: Optional session ID
            
        Returns:
            Prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Create session dict
        session = {
            'raw_text': session_text,
            'session_id': session_id or f'session_{datetime.now().timestamp()}',
            'transactions': []  # Will be populated by feature extraction
        }
        
        # Extract features
        text_features, numerical_features = self.extract_features([session])
        
        # Scale features
        text_features_scaled = self.text_scaler.transform(text_features)
        numerical_features_scaled = self.numerical_scaler.transform(numerical_features)
        
        # Create combined features
        combined_features = np.hstack([text_features_scaled, numerical_features_scaled])
        combined_features_pca = self.pca_reducer.transform(combined_features)
        combined_features_scaled = self.combined_scaler.transform(combined_features_pca)
        
        # Get predictions from models
        svm_score = self.one_class_svm.decision_function(text_features_scaled)[0]
        isolation_score = self.isolation_forest.decision_function(combined_features_scaled)[0]
        
        # Get clustering predictions
        text_cluster = self.dbscan_text.fit_predict(text_features_scaled)[0]
        numerical_cluster = self.dbscan_numerical.fit_predict(numerical_features_scaled)[0]
        combined_cluster = self.dbscan_combined.fit_predict(combined_features_scaled)[0]
        
        # Calculate ensemble score
        svm_score_norm = 1 / (1 + np.exp(svm_score))  # Sigmoid transform
        isolation_score_norm = 1 / (1 + np.exp(isolation_score))
        
        # Density-based scoring
        density_score = 0.3 * (1.0 if text_cluster == -1 else 0.0) + \
                       0.3 * (1.0 if numerical_cluster == -1 else 0.0) + \
                       0.4 * (1.0 if combined_cluster == -1 else 0.0)
        
        # Final ensemble score
        ensemble_score = 0.3 * svm_score_norm + 0.3 * isolation_score_norm + 0.4 * density_score
        
        # Determine if anomaly
        is_anomaly = ensemble_score > 0.5
        
        return {
            'session_id': session_id,
            'is_anomaly': bool(is_anomaly),
            'ensemble_score': float(ensemble_score),
            'svm_score': float(svm_score),
            'isolation_score': float(isolation_score),
            'density_score': float(density_score),
            'text_cluster': int(text_cluster),
            'numerical_cluster': int(numerical_cluster),
            'combined_cluster': int(combined_cluster),
            'prediction_timestamp': datetime.now().isoformat()
        }
    
    def _categorize_tfidf_words(self, tfidf_features: List[Dict]) -> Dict[str, List[Dict]]:
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

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information
        
        Returns:
            Dictionary with model status and details
        """
        if not self.is_trained:
            return {
                'is_trained': False,
                'message': 'Model has not been trained yet'
            }
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            """Recursively convert numpy types to Python native types"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                return {key: convert_numpy_types(value) for key, value in obj.__dict__.items() if not key.startswith('_')}
            else:
                return obj
        
        # Get cluster insights
        cluster_insights = {}
        for feature_type, results in self.cluster_results.items():
            cluster_insights[feature_type] = {
                'n_clusters': results['n_clusters'],
                'noise_ratio': results['noise_ratio'],
                'silhouette_score': results['silhouette_score'],
                'parameters': {
                    'eps': results['eps'],
                    'min_samples': results['min_samples']
                },
                'cluster_distribution': {
                    str(cluster_id): info['size'] 
                    for cluster_id, info in results['cluster_info'].items()
                } if results['cluster_info'] else {}
            }
        
        model_info = {
            'is_trained': True,
            'training_timestamp': self.training_timestamp,
            'training_stats': self.training_stats,
            'model_parameters': {
                'contamination': self.contamination,
                'random_state': self.random_state
            },
            'feature_info': {
                'text_features_count': len(self.feature_names.get('text_features', [])),
                'numerical_features_count': len(self.feature_names.get('numerical_features', [])),
                'feature_names': self.feature_names
            },
            'cluster_insights': cluster_insights,
            'ensemble_components': {
                'one_class_svm': {
                    'kernel': self.one_class_svm.kernel,
                    'nu': self.one_class_svm.nu,
                    'gamma': self.one_class_svm.gamma
                },
                'isolation_forest': {
                    'contamination': self.isolation_forest.contamination,
                    'n_estimators': self.isolation_forest.n_estimators,
                    'random_state': self.isolation_forest.random_state
                },
                'dbscan_text': {
                    'eps': self.cluster_results['text']['eps'],
                    'min_samples': self.cluster_results['text']['min_samples']
                },
                'dbscan_numerical': {
                    'eps': self.cluster_results['numerical']['eps'],
                    'min_samples': self.cluster_results['numerical']['min_samples']
                },
                'dbscan_combined': {
                    'eps': self.cluster_results['combined']['eps'],
                    'min_samples': self.cluster_results['combined']['min_samples']
                }
            }
        }
        
        # Apply numpy type conversion
        return convert_numpy_types(model_info)
    
    def get_dbscan_analysis(self) -> Dict[str, Any]:
        """
        Get detailed DBSCAN analysis results
        
        Returns:
            Dictionary with DBSCAN analysis for all feature spaces
        """
        if not self.is_trained:
            return {
                'error': 'Model must be trained before getting DBSCAN analysis'
            }
        
        # Convert numpy types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Get detailed analysis for visualization
        analysis = {}
        
        for feature_type, results in self.cluster_results.items():
            # Prepare data for scatter plot
            features = None
            if feature_type == 'text':
                features = self.text_features
            elif feature_type == 'numerical':
                features = self.numerical_features
            elif feature_type == 'combined':
                features = self.combined_features
            
            if features is not None:
                # Use PCA for 2D visualization if features have more than 2 dimensions
                if features.shape[1] > 2:
                    pca_viz = PCA(n_components=2, random_state=self.random_state)
                    features_2d = pca_viz.fit_transform(features)
                    explained_variance = pca_viz.explained_variance_ratio_.sum()
                else:
                    features_2d = features
                    explained_variance = 1.0
                
                analysis[feature_type] = {
                    'clustering_results': convert_numpy_types(results),
                    'visualization_data': {
                        'features_2d': features_2d.tolist(),
                        'labels': results['labels'],
                        'explained_variance': float(explained_variance)
                    },
                    'cluster_statistics': {
                        'total_points': len(results['labels']),
                        'n_clusters': results['n_clusters'],
                        'n_noise': results['n_noise'],
                        'noise_percentage': results['noise_ratio'] * 100,
                        'silhouette_score': results['silhouette_score']
                    }
                }
        
        return convert_numpy_types(analysis)
    
    def save_models(self) -> bool:
        """
        Save all trained models to disk
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save traditional models
            joblib.dump(self.tfidf_vectorizer, os.path.join(self.models_dir, 'tfidf_vectorizer.pkl'))
            joblib.dump(self.one_class_svm, os.path.join(self.models_dir, 'one_class_svm.pkl'))
            joblib.dump(self.isolation_forest, os.path.join(self.models_dir, 'isolation_forest.pkl'))
            
            # Save DBSCAN models
            joblib.dump(self.dbscan_text, os.path.join(self.models_dir, 'dbscan_text.pkl'))
            joblib.dump(self.dbscan_numerical, os.path.join(self.models_dir, 'dbscan_numerical.pkl'))
            joblib.dump(self.dbscan_combined, os.path.join(self.models_dir, 'dbscan_combined.pkl'))
            
            # Save scalers and transformers
            joblib.dump(self.text_scaler, os.path.join(self.models_dir, 'text_scaler.pkl'))
            joblib.dump(self.numerical_scaler, os.path.join(self.models_dir, 'numerical_scaler.pkl'))
            joblib.dump(self.combined_scaler, os.path.join(self.models_dir, 'combined_scaler.pkl'))
            joblib.dump(self.pca_reducer, os.path.join(self.models_dir, 'pca_reducer.pkl'))
            
            # Save metadata
            metadata = {
                'is_trained': self.is_trained,
                'training_timestamp': self.training_timestamp,
                'contamination': self.contamination,
                'random_state': self.random_state,
                'feature_names': self.feature_names,
                'training_stats': self.training_stats,
                'cluster_results': self.cluster_results
            }
            
            with open(os.path.join(self.models_dir, 'model_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info("Models saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
    
    def load_models(self) -> bool:
        """
        Load trained models from disk
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if metadata file exists
            metadata_path = os.path.join(self.models_dir, 'model_metadata.json')
            if not os.path.exists(metadata_path):
                logger.warning("No saved models found")
                return False
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Load traditional models
            self.tfidf_vectorizer = joblib.load(os.path.join(self.models_dir, 'tfidf_vectorizer.pkl'))
            self.one_class_svm = joblib.load(os.path.join(self.models_dir, 'one_class_svm.pkl'))
            self.isolation_forest = joblib.load(os.path.join(self.models_dir, 'isolation_forest.pkl'))
            
            # Load DBSCAN models
            self.dbscan_text = joblib.load(os.path.join(self.models_dir, 'dbscan_text.pkl'))
            self.dbscan_numerical = joblib.load(os.path.join(self.models_dir, 'dbscan_numerical.pkl'))
            self.dbscan_combined = joblib.load(os.path.join(self.models_dir, 'dbscan_combined.pkl'))
            
            # Load scalers and transformers
            self.text_scaler = joblib.load(os.path.join(self.models_dir, 'text_scaler.pkl'))
            self.numerical_scaler = joblib.load(os.path.join(self.models_dir, 'numerical_scaler.pkl'))
            self.combined_scaler = joblib.load(os.path.join(self.models_dir, 'combined_scaler.pkl'))
            self.pca_reducer = joblib.load(os.path.join(self.models_dir, 'pca_reducer.pkl'))
            
            # Restore metadata
            self.is_trained = metadata.get('is_trained', False)
            self.training_timestamp = metadata.get('training_timestamp')
            self.feature_names = metadata.get('feature_names', {})
            self.training_stats = metadata.get('training_stats', {})
            self.cluster_results = metadata.get('cluster_results', {})
            
            logger.info("Models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def get_cluster_sessions(self, cluster_id: int, feature_type: str = 'combined') -> List[Dict[str, Any]]:
        """
        Get all sessions belonging to a specific cluster
        
        Args:
            cluster_id: The cluster ID to get sessions for
            feature_type: Type of features used for clustering ('text', 'numerical', 'combined')
        
        Returns:
            List of session dictionaries with text and metadata
        """
        try:
            if not self.is_trained or not hasattr(self, 'training_sessions'):
                raise ValueError("Model not trained or training data not available")
            
            # Get the appropriate cluster labels
            if feature_type == 'text' and hasattr(self, 'text_cluster_labels'):
                cluster_labels = self.text_cluster_labels
            elif feature_type == 'numerical' and hasattr(self, 'numerical_cluster_labels'):
                cluster_labels = self.numerical_cluster_labels
            elif feature_type == 'combined' and hasattr(self, 'combined_cluster_labels'):
                cluster_labels = self.combined_cluster_labels
            else:
                raise ValueError(f"No cluster labels found for feature_type: {feature_type}")
            
            # Find sessions in the specified cluster
            cluster_sessions = []
            for i, label in enumerate(cluster_labels):
                if label == cluster_id and i < len(self.training_sessions):
                    session_text = self.training_sessions[i]
                    
                    # Create session metadata
                    session_info = {
                        'index': i,
                        'text': session_text,
                        'cluster_id': cluster_id,
                        'feature_type': feature_type,
                        'length': len(session_text),
                        'word_count': len(session_text.split()) if isinstance(session_text, str) else 0
                    }
                    
                    cluster_sessions.append(session_info)
            
            return cluster_sessions
            
        except Exception as e:
            logger.error(f"Error getting cluster sessions: {e}")
            raise
    
    def label_cluster(self, cluster_id: int, label: str, feature_type: str = 'combined') -> bool:
        """
        Assign a human-readable label to a cluster
        
        Args:
            cluster_id: The cluster ID to label
            label: Human-readable label for the cluster
            feature_type: Type of features used for clustering
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Initialize cluster labels storage if it doesn't exist
            if not hasattr(self, 'cluster_labels'):
                self.cluster_labels = {}
            
            # Store the label with feature type context
            label_key = f"{feature_type}_{cluster_id}"
            self.cluster_labels[label_key] = {
                'cluster_id': cluster_id,
                'label': label,
                'feature_type': feature_type,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Labeled cluster {cluster_id} ({feature_type}) as '{label}'")
            return True
            
        except Exception as e:
            logger.error(f"Error labeling cluster: {e}")
            return False
    
    def train_supervised_classifier(self, force_retrain: bool = False) -> Dict[str, Any]:
        """
        Train a supervised classifier using labeled clusters
        
        Args:
            force_retrain: Whether to retrain even if classifier exists
        
        Returns:
            Training statistics and performance metrics
        """
        try:
            if not hasattr(self, 'cluster_labels') or not self.cluster_labels:
                raise ValueError("No labeled clusters available for supervised training")
            
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
            from sklearn.metrics import classification_report
            
            # Prepare training data
            X_train = []
            y_train = []
            
            for label_key, label_info in self.cluster_labels.items():
                cluster_id = label_info['cluster_id']
                feature_type = label_info['feature_type']
                label = label_info['label']
                
                # Get sessions for this cluster
                try:
                    sessions = self.get_cluster_sessions(cluster_id, feature_type)
                    
                    # Extract features for each session
                    for session in sessions:
                        session_text = session['text']
                        
                        # Create feature vector (use combined features for consistency)
                        if hasattr(self, 'tfidf_vectorizer'):
                            text_features = self.tfidf_vectorizer.transform([session_text])
                            numerical_features = self._extract_numerical_features([session_text])
                            
                            # Combine features
                            combined_features = np.hstack([
                                text_features.toarray(),
                                numerical_features
                            ])
                            
                            X_train.append(combined_features[0])
                            y_train.append(label)
                            
                except Exception as e:
                    logger.warning(f"Error processing cluster {cluster_id}: {e}")
                    continue
            
            if len(X_train) == 0:
                raise ValueError("No training data could be extracted from labeled clusters")
            
            # Train supervised classifier
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            self.supervised_classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced'
            )
            
            self.supervised_classifier.fit(X_train, y_train)
            
            # Evaluate classifier
            cv_scores = cross_val_score(self.supervised_classifier, X_train, y_train, cv=5)
            
            # Store training statistics
            training_stats = {
                'num_samples': len(X_train),
                'num_classes': len(np.unique(y_train)),
                'classes': list(np.unique(y_train)),
                'cv_accuracy_mean': cv_scores.mean(),
                'cv_accuracy_std': cv_scores.std(),
                'feature_dimensions': X_train.shape[1],
                'training_timestamp': datetime.now().isoformat()
            }
            
            # Store classifier state
            self.supervised_training_stats = training_stats
            
            logger.info(f"Supervised classifier trained successfully: {training_stats}")
            return training_stats
            
        except Exception as e:
            logger.error(f"Error training supervised classifier: {e}")
            raise
    
    def predict_supervised(self, session_text: str) -> Dict[str, Any]:
        """
        Predict cluster label for a session using supervised classifier
        
        Args:
            session_text: The session text to classify
        
        Returns:
            Prediction results with confidence scores
        """
        try:
            if not hasattr(self, 'supervised_classifier'):
                raise ValueError("Supervised classifier not trained. Call train_supervised_classifier() first.")
            
            # Extract features
            text_features = self.tfidf_vectorizer.transform([session_text])
            numerical_features = self._extract_numerical_features([session_text])
            
            # Combine features
            combined_features = np.hstack([
                text_features.toarray(),
                numerical_features
            ])
            
            # Make prediction
            prediction = self.supervised_classifier.predict(combined_features)[0]
            probabilities = self.supervised_classifier.predict_proba(combined_features)[0]
            
            # Get class labels
            classes = self.supervised_classifier.classes_
            
            # Create prediction result
            prediction_result = {
                'predicted_label': prediction,
                'confidence': max(probabilities),
                'all_probabilities': {
                    classes[i]: probabilities[i] for i in range(len(classes))
                },
                'session_text': session_text,
                'prediction_timestamp': datetime.now().isoformat()
            }
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error making supervised prediction: {e}")
            raise
