"""
Enhanced Ensemble Anomaly Detection Model with DBSCAN Integration
"""

import numpy as np
import re
import json
import pickle
import os
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class EnhancedEnsembleAnomalyDetector:
    """
    Enhanced ensemble anomaly detection system with DBSCAN integration
    Combines text analysis, statistical analysis, and density-based clustering
    """
    
    def __init__(self, model_dir: str = "./models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Original ensemble components
        self.text_vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2), lowercase=True)
        self.svm_model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
        self.isolation_model = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        
        # New DBSCAN components
        self.text_dbscan = DBSCAN(eps=0.5, min_samples=3, metric='cosine')
        self.numerical_dbscan = DBSCAN(eps=0.5, min_samples=3)
        self.combined_dbscan = DBSCAN(eps=0.5, min_samples=3)
        
        # Feature reduction for high-dimensional clustering
        self.text_pca = PCA(n_components=50, random_state=42)
        self.numerical_pca = PCA(n_components=20, random_state=42)
        
        # Training state
        self.is_trained = False
        self.training_stats = {}
        self.feature_names = []
        self.cluster_profiles = {}
        
        # Weights for enhanced ensemble
        self.text_weight = 0.4
        self.statistical_weight = 0.3
        self.density_weight = 0.3
        self.threshold = 0.5
        
        # DBSCAN parameters
        self.dbscan_params = {
            'text_eps': 0.5,
            'text_min_samples': 3,
            'numerical_eps': 0.5,
            'numerical_min_samples': 3,
            'combined_eps': 0.5,
            'combined_min_samples': 3
        }
    
    def extract_text_features(self, session_text: str) -> Dict[str, float]:
        """Extract text-based features (same as original but with clustering preparation)"""
        text_lower = session_text.lower()
        text_upper = session_text.upper()
        words = text_lower.split()
        
        # Enhanced term categories
        normal_terms = ['card', 'pin', 'verified', 'completed', 'successful', 'dispensed', 'printed', 'ejected', 'taken', 'approved']
        critical_error_terms = ['device error', 'device offline', 'critical', 'fatal', 'malfunction', 'communication failure']
        hardware_error_terms = ['hardware error', 'power-up/reset', 'cim-reset', 'recovery failed', 'capture failed', 'jam']
        general_error_terms = ['error', 'fail', 'timeout', 'reset', 'offline', 'fault', 'unable', 'declined']
        
        # Machine status patterns
        machine_status_patterns = re.findall(r'M-\d+', text_upper)
        critical_machine_codes = ['M-65', 'M-01', 'M-15', 'M-23', 'M-45', 'M-67']
        
        # Error patterns
        error_code_patterns = re.findall(r'[ME]-\d+', text_upper)
        device_error_count = len(re.findall(r'device\s+error', text_lower))
        aac_errors = len(re.findall(r'aac|no arpc', text_lower))
        communication_errors = len(re.findall(r'communication\s+failure|comm\s+error|timeout', text_lower))
        
        # Operational patterns
        supervisor_patterns = len(re.findall(r'supervisor\s+mode|supervisor\s+entry|supervisor\s+exit', text_lower))
        recovery_patterns = len(re.findall(r'recovery|cim-reset|init\s+bna|device\s+init|retract\s+bin', text_lower))
        cash_anomalies = len(re.findall(r'cash\s+error|dispenser\s+error|notes\s+jam|cash\s+jam', text_lower))
        retract_operations = len(re.findall(r'retract|capture\s+failed', text_lower))
        auth_failures = len(re.findall(r'external\s+authenticate.*fail|pin.*fail|auth.*fail', text_lower))
        
        # Base features
        features = {
            'total_words': len(words),
            'unique_words': len(set(words)),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            
            # Term counting
            'normal_term_count': sum(1 for word in words if any(term in word for term in normal_terms)),
            'critical_error_count': sum(1 for phrase in critical_error_terms if phrase in text_lower),
            'hardware_error_count': sum(1 for phrase in hardware_error_terms if phrase in text_lower),
            'general_error_count': sum(1 for phrase in general_error_terms if phrase in text_lower),
            
            # Critical indicators
            'device_error_explicit': device_error_count,
            'machine_status_codes': len(machine_status_patterns),
            'critical_machine_codes': sum(1 for code in machine_status_patterns if code in critical_machine_codes),
            'error_codes_total': len(error_code_patterns),
            'aac_authentication_errors': aac_errors,
            'communication_failures': communication_errors,
            
            # Operational indicators
            'supervisor_mode_indicators': supervisor_patterns,
            'recovery_operations': recovery_patterns,
            'cash_handling_anomalies': cash_anomalies,
            'retract_operations': retract_operations,
            'authentication_failures': auth_failures,
            
            # Pattern-based features
            'error_pattern_density': 0.0,
            'critical_anomaly_score': 0.0,
        }
        
        # Calculate derived features
        if features['total_words'] > 0:
            features['error_ratio'] = (features['critical_error_count'] + features['hardware_error_count'] + features['general_error_count']) / features['total_words']
            features['normal_ratio'] = features['normal_term_count'] / features['total_words']
            features['anomaly_term_density'] = (features['device_error_explicit'] + features['critical_machine_codes'] + features['supervisor_mode_indicators']) / features['total_words']
        else:
            features['error_ratio'] = features['normal_ratio'] = features['anomaly_term_density'] = 0
            
        # Error pattern density
        total_error_indicators = (features['critical_error_count'] + features['hardware_error_count'] + 
                                features['device_error_explicit'] + features['critical_machine_codes'] + 
                                features['communication_failures'] + features['recovery_operations'])
        
        if features['total_words'] > 0:
            features['error_pattern_density'] = total_error_indicators / features['total_words']
        
        # Critical anomaly score
        critical_score = 0.0
        if features['device_error_explicit'] > 0:
            critical_score += 0.8
        if features['critical_machine_codes'] > 0:
            critical_score += 0.7
        if features['error_codes_total'] > 2:
            critical_score += 0.6
        elif features['error_codes_total'] > 0:
            critical_score += 0.3
        if features['communication_failures'] > 0:
            critical_score += 0.5
        if features['aac_authentication_errors'] > 0:
            critical_score += 0.4
        if features['recovery_operations'] > 1:
            critical_score += 0.4
        elif features['recovery_operations'] > 0:
            critical_score += 0.2
        if features['supervisor_mode_indicators'] > 0:
            critical_score += 0.6
        if features['error_pattern_density'] > 0.1:
            critical_score += 0.5
        elif features['error_pattern_density'] > 0.05:
            critical_score += 0.3
            
        features['critical_anomaly_score'] = min(1.0, critical_score)
        
        # Add binary flags for clustering
        features['has_device_error'] = 1.0 if features['device_error_explicit'] > 0 else 0.0
        features['has_critical_machine_code'] = 1.0 if features['critical_machine_codes'] > 0 else 0.0
        features['has_supervisor_anomaly'] = 1.0 if features['supervisor_mode_indicators'] > 0 else 0.0
        features['has_recovery_operations'] = 1.0 if features['recovery_operations'] > 0 else 0.0
        features['multiple_error_codes'] = 1.0 if features['error_codes_total'] > 1 else 0.0
        
        return features
    
    def extract_numerical_features(self, session_text: str) -> Dict[str, float]:
        """Extract numerical/statistical features (enhanced for clustering)"""
        lines = session_text.strip().split('\n')
        text_lower = session_text.lower()
        text_upper = session_text.upper()
        
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
            'device_error_count': len(re.findall(r'device\s+error', text_lower)),
            
            # Machine status
            'machine_status_codes': len(re.findall(r'M-\d+', text_upper)),
            'critical_m_codes': len(re.findall(r'M-(?:01|15|23|38|45|65|67)', text_upper)),
            'error_codes_total': len(re.findall(r'[ME]-\d+', text_upper)),
            
            # Hardware patterns
            'hardware_mentions': len(re.findall(r'hardware', text_lower)),
            'power_reset_count': len(re.findall(r'power.*reset|reset.*power|power-up/reset', text_lower)),
            'cim_mentions': len(re.findall(r'cim', text_lower)),
            'recovery_failures': len(re.findall(r'recovery.*fail', text_lower)),
            'capture_failures': len(re.findall(r'capture.*fail', text_lower)),
            
            # Critical patterns
            'critical_hardware_patterns': len(re.findall(
                r'power-up/reset|hardware.*error|cim-reset|recovery.*failed|capture.*failed|device\s+error',
                text_lower
            )),
            
            # Communication and authentication
            'communication_errors': len(re.findall(r'communication\s+failure|comm\s+error|no arpc|aac', text_lower)),
            'authentication_failures': len(re.findall(r'external\s+authenticate.*fail|pin.*fail|genac.*aac', text_lower)),
            'network_errors': len(re.findall(r'network.*error|connection.*lost|timeout', text_lower)),
            
            # Cash handling
            'cash_errors': len(re.findall(r'cash.*error|dispenser.*error|jam', text_lower)),
            'retract_operations': len(re.findall(r'retract|capture\s+failed', text_lower)),
            'dispensing_issues': len(re.findall(r'notes.*jam|cash.*jam|dispenser.*malfunction', text_lower)),
            
            # Operational patterns
            'supervisor_patterns': len(re.findall(r'supervisor\s+mode|supervisor\s+entry|supervisor\s+exit', text_lower)),
            'recovery_operations': len(re.findall(r'init\s+bna|cim-reset|device\s+init|recovery', text_lower)),
            'reset_operations': len(re.findall(r'reset|init.*started|recovery.*ok', text_lower)),
            
            # Transaction integrity
            'transaction_start_count': len(re.findall(r'transaction\s+start', text_lower)),
            'transaction_end_count': len(re.findall(r'transaction\s+end', text_lower)),
            'incomplete_transaction_ratio': 0.0,
            
            # Success indicators
            'success_indicators': len(re.findall(
                r'completed|successful|verified|dispensed|printed|taken|approved',
                text_lower
            )),
            
            # Derived scores
            'anomaly_density_score': 0.0,
            'critical_error_density': 0.0,
            'hardware_failure_score': 0.0,
        }
        
        # Calculate derived features
        if features['transaction_start_count'] > 0:
            features['incomplete_transaction_ratio'] = abs(features['transaction_start_count'] - features['transaction_end_count']) / features['transaction_start_count']
        
        total_anomaly_indicators = (features['device_error_count'] + features['critical_m_codes'] + 
                                  features['critical_hardware_patterns'] + features['communication_errors'] + 
                                  features['supervisor_patterns'] + features['recovery_operations'])
        
        if features['line_count'] > 0:
            features['anomaly_density_score'] = total_anomaly_indicators / features['line_count']
            features['critical_error_density'] = (features['device_error_count'] + features['critical_m_codes']) / features['line_count']
        
        hardware_failures = (features['critical_hardware_patterns'] + features['power_reset_count'] + 
                           features['recovery_failures'] + features['capture_failures'])
        features['hardware_failure_score'] = min(1.0, hardware_failures / 5.0)
        
        total_errors = (features['error_count'] + features['fail_count'] + features['device_error_count'] + 
                       features['critical_m_codes'])
        if features['success_indicators'] > 0:
            features['error_to_success_ratio'] = total_errors / features['success_indicators']
        else:
            features['error_to_success_ratio'] = total_errors
        
        if features['line_count'] > 0:
            features['error_to_line_ratio'] = total_errors / features['line_count']
        else:
            features['error_to_line_ratio'] = 0
        
        features['session_health_score'] = self._calculate_session_health_score(features, text_lower)
        
        return features
    
    def _calculate_session_health_score(self, features: Dict[str, float], text_lower: str) -> float:
        """Calculate session health score"""
        health_score = 1.0
        
        if features['device_error_count'] > 0:
            health_score -= 0.8
        if features['critical_m_codes'] > 0:
            health_score -= 0.7
        if features['error_codes_total'] > 2:
            health_score -= 0.6
        elif features['error_codes_total'] > 0:
            health_score -= 0.3
        if features['communication_errors'] > 0:
            health_score -= 0.5
        if features['critical_hardware_patterns'] > 1:
            health_score -= 0.6
        elif features['critical_hardware_patterns'] > 0:
            health_score -= 0.4
        if features['supervisor_patterns'] > 0:
            health_score -= 0.7
        if features['recovery_operations'] > 1:
            health_score -= 0.4
        elif features['recovery_operations'] > 0:
            health_score -= 0.2
        if features['anomaly_density_score'] > 0.2:
            health_score -= 0.5
        elif features['anomaly_density_score'] > 0.1:
            health_score -= 0.3
        if features['authentication_failures'] > 0:
            health_score -= 0.4
        if features['incomplete_transaction_ratio'] > 0:
            health_score -= 0.3
        if features['error_to_success_ratio'] > 2.0:
            health_score -= 0.4
        elif features['error_to_success_ratio'] > 1.0:
            health_score -= 0.2
        
        return max(0.0, min(1.0, health_score))
    
    def _optimize_dbscan_parameters(self, X: np.ndarray, parameter_name: str) -> Dict[str, float]:
        """Optimize DBSCAN parameters using silhouette analysis"""
        best_eps = 0.5
        best_min_samples = 3
        best_score = -1
        
        eps_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        min_samples_values = [2, 3, 4, 5]
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                try:
                    if parameter_name == 'text':
                        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
                    else:
                        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    
                    labels = dbscan.fit_predict(X)
                    
                    # Check if we have at least 2 clusters and not all noise
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    if n_clusters >= 2 and np.sum(labels == -1) < len(labels) * 0.8:  # Less than 80% noise
                        score = silhouette_score(X, labels)
                        if score > best_score:
                            best_score = score
                            best_eps = eps
                            best_min_samples = min_samples
                except:
                    continue
        
        return {
            f'{parameter_name}_eps': best_eps,
            f'{parameter_name}_min_samples': best_min_samples,
            f'{parameter_name}_silhouette_score': best_score
        }
    
    def _analyze_clusters(self, X: np.ndarray, labels: np.ndarray, sessions: List[str], 
                         feature_names: List[str] = None) -> Dict[str, Any]:
        """Analyze cluster characteristics"""
        cluster_analysis = {
            'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
            'n_noise': np.sum(labels == -1),
            'noise_ratio': np.sum(labels == -1) / len(labels),
            'cluster_sizes': {},
            'cluster_profiles': {}
        }
        
        # Analyze each cluster
        for cluster_id in set(labels):
            if cluster_id == -1:  # Noise points
                cluster_analysis['cluster_sizes']['noise'] = np.sum(labels == -1)
                continue
            
            cluster_mask = labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            cluster_analysis['cluster_sizes'][f'cluster_{cluster_id}'] = cluster_size
            
            # Calculate cluster statistics
            cluster_data = X[cluster_mask]
            cluster_center = np.mean(cluster_data, axis=0)
            cluster_std = np.std(cluster_data, axis=0)
            
            profile = {
                'size': cluster_size,
                'center': cluster_center.tolist(),
                'std': cluster_std.tolist(),
                'sessions_sample': sessions[:3] if len(sessions) > 0 else []  # Sample sessions
            }
            
            # Add feature-specific analysis if feature names provided
            if feature_names:
                top_features_idx = np.argsort(cluster_center)[-5:]  # Top 5 features
                profile['top_features'] = [(feature_names[i], cluster_center[i]) for i in top_features_idx]
            
            cluster_analysis['cluster_profiles'][f'cluster_{cluster_id}'] = profile
        
        return cluster_analysis
    
    def train(self, normal_sessions: List[str]) -> Dict[str, Any]:
        """Enhanced training with DBSCAN integration"""
        print(f"Training enhanced ensemble with DBSCAN on {len(normal_sessions)} sessions...")
        
        # Extract features
        texts = []
        numerical_features_list = []
        text_features_list = []
        
        for session in normal_sessions:
            texts.append(session)
            num_features = self.extract_numerical_features(session)
            text_features_dict = self.extract_text_features(session)
            
            numerical_features_list.append(list(num_features.values()))
            text_features_list.append(list(text_features_dict.values()))
        
        # Store feature names
        sample_num_features = self.extract_numerical_features(normal_sessions[0])
        sample_text_features = self.extract_text_features(normal_sessions[0])
        self.feature_names = list(sample_num_features.keys())
        self.text_feature_names = list(sample_text_features.keys())
        
        # Train original ensemble components
        text_vectors = self.text_vectorizer.fit_transform(texts).toarray()
        self.svm_model.fit(text_vectors)
        
        numerical_features = np.array(numerical_features_list)
        numerical_features = self.scaler.fit_transform(numerical_features)
        self.isolation_model.fit(numerical_features)
        
        # Train DBSCAN components
        
        # 1. Text-based clustering (on TF-IDF vectors)
        print("Training text-based DBSCAN...")
        if text_vectors.shape[1] > 50:
            text_features_reduced = self.text_pca.fit_transform(text_vectors)
        else:
            text_features_reduced = text_vectors
            
        # Optimize text DBSCAN parameters
        text_params = self._optimize_dbscan_parameters(text_features_reduced, 'text')
        self.dbscan_params.update(text_params)
        
        self.text_dbscan = DBSCAN(
            eps=text_params['text_eps'], 
            min_samples=text_params['text_min_samples'], 
            metric='cosine'
        )
        text_clusters = self.text_dbscan.fit_predict(text_features_reduced)
        
        # 2. Numerical features clustering
        print("Training numerical DBSCAN...")
        if numerical_features.shape[1] > 20:
            numerical_features_reduced = self.numerical_pca.fit_transform(numerical_features)
        else:
            numerical_features_reduced = numerical_features
            
        numerical_params = self._optimize_dbscan_parameters(numerical_features_reduced, 'numerical')
        self.dbscan_params.update(numerical_params)
        
        self.numerical_dbscan = DBSCAN(
            eps=numerical_params['numerical_eps'],
            min_samples=numerical_params['numerical_min_samples']
        )
        numerical_clusters = self.numerical_dbscan.fit_predict(numerical_features_reduced)
        
        # 3. Combined features clustering
        print("Training combined features DBSCAN...")
        text_features_array = np.array(text_features_list)
        text_features_scaled = StandardScaler().fit_transform(text_features_array)
        
        combined_features = np.hstack([text_features_scaled, numerical_features])
        combined_params = self._optimize_dbscan_parameters(combined_features, 'combined')
        self.dbscan_params.update(combined_params)
        
        self.combined_dbscan = DBSCAN(
            eps=combined_params['combined_eps'],
            min_samples=combined_params['combined_min_samples']
        )
        combined_clusters = self.combined_dbscan.fit_predict(combined_features)
        
        # Analyze clusters
        text_cluster_analysis = self._analyze_clusters(
            text_features_reduced, text_clusters, normal_sessions
        )
        numerical_cluster_analysis = self._analyze_clusters(
            numerical_features_reduced, numerical_clusters, normal_sessions, self.feature_names
        )
        combined_cluster_analysis = self._analyze_clusters(
            combined_features, combined_clusters, normal_sessions
        )
        
        # Store cluster profiles for anomaly detection
        self.cluster_profiles = {
            'text_clusters': text_cluster_analysis,
            'numerical_clusters': numerical_cluster_analysis,
            'combined_clusters': combined_cluster_analysis
        }
        
        # Calculate original ensemble statistics
        svm_scores = self.svm_model.decision_function(text_vectors)
        iso_scores = self.isolation_model.decision_function(numerical_features)
        
        svm_probabilities = 1 / (1 + np.exp(svm_scores))
        iso_probabilities = 1 / (1 + np.exp(iso_scores))
        
        # Calculate DBSCAN-based anomaly scores
        text_density_scores = self._calculate_density_scores(text_clusters)
        numerical_density_scores = self._calculate_density_scores(numerical_clusters)
        combined_density_scores = self._calculate_density_scores(combined_clusters)
        
        # Ensemble density score
        density_ensemble_scores = (text_density_scores + numerical_density_scores + combined_density_scores) / 3
        
        # Final ensemble with density component
        ensemble_scores = (self.text_weight * svm_probabilities + 
                         self.statistical_weight * iso_probabilities + 
                         self.density_weight * density_ensemble_scores)
        
        self.training_stats = {
            'num_training_sessions': len(normal_sessions),
            'text_feature_dims': text_vectors.shape[1],
            'numerical_feature_dims': len(self.feature_names),
            'avg_svm_score': float(np.mean(svm_probabilities)),
            'avg_isolation_score': float(np.mean(iso_probabilities)),
            'avg_density_score': float(np.mean(density_ensemble_scores)),
            'avg_ensemble_score': float(np.mean(ensemble_scores)),
            'feature_names': self.feature_names,
            'text_feature_names': self.text_feature_names,
            'weights': {
                'text_weight': self.text_weight,
                'statistical_weight': self.statistical_weight,
                'density_weight': self.density_weight
            },
            'threshold': self.threshold,
            'dbscan_params': self.dbscan_params,
            'cluster_analysis': {
                'text_clusters': text_cluster_analysis,
                'numerical_clusters': numerical_cluster_analysis,
                'combined_clusters': combined_cluster_analysis
            }
        }
        
        self.is_trained = True
        print("Enhanced training with DBSCAN complete!")
        
        return self.training_stats
    
    def _calculate_density_scores(self, cluster_labels: np.ndarray) -> np.ndarray:
        """Calculate anomaly scores based on cluster density"""
        scores = np.zeros(len(cluster_labels))
        
        # Noise points (label = -1) get high anomaly scores
        noise_mask = cluster_labels == -1
        scores[noise_mask] = 0.9  # High anomaly score for noise points
        
        # For clustered points, score based on cluster size (smaller clusters = more anomalous)
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Skip noise
                continue
            
            cluster_mask = cluster_labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            # Smaller clusters get higher anomaly scores
            # Large clusters (>20% of data) get low scores
            # Small clusters (<5% of data) get high scores
            total_points = len(cluster_labels)
            cluster_ratio = cluster_size / total_points
            
            if cluster_ratio > 0.2:  # Large cluster
                cluster_score = 0.1
            elif cluster_ratio > 0.1:  # Medium cluster  
                cluster_score = 0.3
            elif cluster_ratio > 0.05:  # Small cluster
                cluster_score = 0.6
            else:  # Very small cluster
                cluster_score = 0.8
            
            scores[cluster_mask] = cluster_score
        
        return scores
    
    def predict(self, session_text: str) -> Dict[str, Any]:
        """Enhanced prediction with DBSCAN-based density analysis"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features
        text_vectors = self.text_vectorizer.transform([session_text]).toarray()
        num_features_dict = self.extract_numerical_features(session_text)
        text_features_dict = self.extract_text_features(session_text)
        
        # Get original ensemble predictions
        num_features = np.array([list(num_features_dict.values())])
        num_features = self.scaler.transform(num_features)
        
        svm_score = self.svm_model.decision_function(text_vectors)[0]
        iso_score = self.isolation_model.decision_function(num_features)[0]
        
        svm_probability = 1 / (1 + np.exp(svm_score))
        iso_probability = 1 / (1 + np.exp(iso_score))
        
        # Get DBSCAN-based density predictions
        density_scores = self._predict_density_anomaly(text_vectors[0], num_features[0], 
                                                     list(text_features_dict.values()))
        
        # Apply critical pattern amplification (same as original)
        amplified_scores = self._apply_critical_anomaly_amplification(
            svm_probability, iso_probability, text_features_dict, num_features_dict, session_text
        )
        
        amplified_svm = amplified_scores['amplified_text_score']
        amplified_iso = amplified_scores['amplified_statistical_score']
        critical_boost = amplified_scores['critical_boost']
        anomaly_reasons = amplified_scores['anomaly_reasons']
        
        # Enhanced ensemble with density component
        base_ensemble_score = (self.text_weight * amplified_svm + 
                             self.statistical_weight * amplified_iso + 
                             self.density_weight * density_scores['ensemble_density_score'])
        
        final_ensemble_score = min(1.0, base_ensemble_score + critical_boost)
        
        # Dynamic threshold adjustment
        effective_threshold = self.threshold
        if critical_boost > 0.3:
            effective_threshold = max(0.3, self.threshold - 0.2)
        
        # Additional threshold adjustment for density-based anomalies
        if density_scores['is_density_anomaly']:
            effective_threshold = max(0.2, effective_threshold - 0.1)
            anomaly_reasons.append(f"Density-based anomaly detected - unusual pattern clustering")
        
        is_anomaly = final_ensemble_score > effective_threshold
        
        confidence = self._calculate_enhanced_confidence(
            final_ensemble_score, effective_threshold, critical_boost, anomaly_reasons
        )
        
        return {
            'session_text': session_text,
            'text_anomaly_score': float(amplified_svm),
            'statistical_anomaly_score': float(amplified_iso),
            'density_anomaly_score': float(density_scores['ensemble_density_score']),
            'ensemble_score': float(final_ensemble_score),
            'base_ensemble_score': float(base_ensemble_score),
            'critical_boost': float(critical_boost),
            'is_anomaly': bool(is_anomaly),
            'confidence': confidence,
            'threshold': effective_threshold,
            'original_threshold': self.threshold,
            'text_features': text_features_dict,
            'numerical_features': num_features_dict,
            'density_analysis': density_scores,
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
                'density_component': {
                    'density_score': float(density_scores['ensemble_density_score']),
                    'weight': self.density_weight,
                    'contribution': float(self.density_weight * density_scores['ensemble_density_score']),
                    'is_density_anomaly': density_scores['is_density_anomaly'],
                    'cluster_distances': density_scores['cluster_distances']
                },
                'critical_amplification': {
                    'boost_applied': float(critical_boost),
                    'reasons': anomaly_reasons,
                    'threshold_adjustment': float(self.threshold - effective_threshold)
                }
            }
        }
    
    def _predict_density_anomaly(self, text_vector: np.ndarray, num_features: np.ndarray, 
                               text_features_list: List[float]) -> Dict[str, Any]:
        """Predict anomaly based on density clustering"""
        
        # Prepare features for clustering
        if hasattr(self, 'text_pca') and text_vector.shape[0] > 50:
            text_features_reduced = self.text_pca.transform([text_vector])
        else:
            text_features_reduced = [text_vector]
            
        if hasattr(self, 'numerical_pca') and num_features.shape[0] > 20:
            numerical_features_reduced = self.numerical_pca.transform([num_features])
        else:
            numerical_features_reduced = [num_features]
        
        text_features_scaled = StandardScaler().fit_transform([text_features_list])
        combined_features = np.hstack([text_features_scaled, [num_features]])
        
        # Predict cluster membership (distance to existing clusters)
        text_distances = self._calculate_cluster_distances(text_features_reduced[0], 'text')
        numerical_distances = self._calculate_cluster_distances(numerical_features_reduced[0], 'numerical')
        combined_distances = self._calculate_cluster_distances(combined_features[0], 'combined')
        
        # Calculate density-based anomaly scores
        text_density_score = min(text_distances.values()) if text_distances else 0.9
        numerical_density_score = min(numerical_distances.values()) if numerical_distances else 0.9
        combined_density_score = min(combined_distances.values()) if combined_distances else 0.9
        
        # Ensemble density score
        ensemble_density_score = (text_density_score + numerical_density_score + combined_density_score) / 3
        
        # Determine if this is a density-based anomaly
        is_density_anomaly = ensemble_density_score > 0.7  # High distance from all clusters
        
        return {
            'text_density_score': text_density_score,
            'numerical_density_score': numerical_density_score,
            'combined_density_score': combined_density_score,
            'ensemble_density_score': ensemble_density_score,
            'is_density_anomaly': is_density_anomaly,
            'cluster_distances': {
                'text': text_distances,
                'numerical': numerical_distances,
                'combined': combined_distances
            }
        }
    
    def _calculate_cluster_distances(self, point: np.ndarray, cluster_type: str) -> Dict[str, float]:
        """Calculate distances to existing cluster centers"""
        distances = {}
        
        if cluster_type not in self.cluster_profiles:
            return distances
        
        cluster_analysis = self.cluster_profiles[f'{cluster_type}_clusters']
        
        for cluster_name, profile in cluster_analysis['cluster_profiles'].items():
            if cluster_name == 'noise':
                continue
                
            cluster_center = np.array(profile['center'])
            distance = np.linalg.norm(point - cluster_center)
            
            # Normalize by cluster standard deviation
            cluster_std = np.array(profile['std'])
            avg_std = np.mean(cluster_std)
            if avg_std > 0:
                normalized_distance = distance / avg_std
            else:
                normalized_distance = distance
            
            distances[cluster_name] = normalized_distance
        
        return distances
    
    def _apply_critical_anomaly_amplification(self, svm_prob: float, iso_prob: float, 
                                            text_features: Dict[str, float], 
                                            num_features: Dict[str, float], 
                                            session_text: str) -> Dict[str, Any]:
        """Apply critical pattern amplification (same as original implementation)"""
        anomaly_reasons = []
        critical_boost = 0.0
        
        # Critical patterns with same logic as original
        if text_features.get('device_error_explicit', 0) > 0:
            boost = 0.6
            critical_boost += boost
            anomaly_reasons.append(f"DEVICE ERROR detected - critical hardware failure indicator (+{boost:.1f})")
        
        if num_features.get('critical_m_codes', 0) > 0:
            boost = 0.5
            critical_boost += boost
            m_codes = re.findall(r'M-\d+', session_text.upper())
            anomaly_reasons.append(f"Critical machine status codes detected: {', '.join(m_codes)} (+{boost:.1f})")
        
        error_code_count = num_features.get('error_codes_total', 0)
        if error_code_count > 2:
            boost = 0.4
            critical_boost += boost
            anomaly_reasons.append(f"Multiple error codes ({int(error_code_count)}) - cascading system failures (+{boost:.1f})")
        elif error_code_count > 0:
            boost = 0.2
            critical_boost += boost
            anomaly_reasons.append(f"Error codes detected ({int(error_code_count)}) - system issues (+{boost:.1f})")
        
        if num_features.get('communication_errors', 0) > 0:
            boost = 0.4
            critical_boost += boost
            anomaly_reasons.append(f"Communication failures detected - network/host connectivity issues (+{boost:.1f})")
        
        if num_features.get('supervisor_patterns', 0) > 0:
            boost = 0.5
            critical_boost += boost
            anomaly_reasons.append(f"Supervisor mode activity detected - unusual operational pattern (+{boost:.1f})")
        
        if num_features.get('critical_hardware_patterns', 0) > 1:
            boost = 0.4
            critical_boost += boost
            anomaly_reasons.append(f"Multiple critical hardware patterns detected - device reliability issues (+{boost:.1f})")
        elif num_features.get('critical_hardware_patterns', 0) > 0:
            boost = 0.2
            critical_boost += boost
            anomaly_reasons.append(f"Critical hardware pattern detected - potential device issue (+{boost:.1f})")
        
        anomaly_density = num_features.get('anomaly_density_score', 0)
        if anomaly_density > 0.2:
            boost = 0.3
            critical_boost += boost
            anomaly_reasons.append(f"High anomaly density ({anomaly_density:.1%}) - concentrated error patterns (+{boost:.1f})")
        elif anomaly_density > 0.1:
            boost = 0.15
            critical_boost += boost
            anomaly_reasons.append(f"Elevated anomaly density ({anomaly_density:.1%}) - multiple error indicators (+{boost:.1f})")
        
        session_health = num_features.get('session_health_score', 1.0)
        if session_health < 0.3:
            boost = 0.4
            critical_boost += boost
            anomaly_reasons.append(f"Very poor session health (score: {session_health:.2f}) - multiple critical issues (+{boost:.1f})")
        elif session_health < 0.5:
            boost = 0.2
            critical_boost += boost
            anomaly_reasons.append(f"Poor session health (score: {session_health:.2f}) - concerning patterns (+{boost:.1f})")
        
        if num_features.get('authentication_failures', 0) > 0:
            boost = 0.3
            critical_boost += boost
            anomaly_reasons.append(f"Authentication failures detected - security concerns (+{boost:.1f})")
        
        recovery_ops = num_features.get('recovery_operations', 0)
        if recovery_ops > 1:
            boost = 0.3
            critical_boost += boost
            anomaly_reasons.append(f"Multiple recovery operations ({int(recovery_ops)}) - device instability (+{boost:.1f})")
        elif recovery_ops > 0:
            boost = 0.15
            critical_boost += boost
            anomaly_reasons.append(f"Recovery operation detected - previous device issue (+{boost:.1f})")
        
        # Apply amplification
        text_amplification = (text_features.get('critical_anomaly_score', 0) * 0.5 + 
                            min(0.3, critical_boost * 0.6))
        amplified_svm = min(1.0, svm_prob + text_amplification)
        
        statistical_amplification = min(0.4, critical_boost * 0.7)
        amplified_iso = min(1.0, iso_prob + statistical_amplification)
        
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
        """Calculate enhanced confidence"""
        threshold_distance = abs(ensemble_score - threshold)
        
        if threshold_distance > 0.4:
            base_confidence = "HIGH"
        elif threshold_distance > 0.2:
            base_confidence = "MEDIUM"
        else:
            base_confidence = "LOW"
        
        if critical_boost > 0.3 and len(anomaly_reasons) >= 3:
            return "VERY_HIGH"
        elif critical_boost > 0.2 and len(anomaly_reasons) >= 2:
            if base_confidence == "LOW":
                return "MEDIUM"
            elif base_confidence == "MEDIUM":
                return "HIGH"
            else:
                return "VERY_HIGH"
        elif critical_boost > 0.1:
            if base_confidence == "LOW" and threshold_distance > 0.1:
                return "MEDIUM"
        
        return base_confidence
    
    def get_cluster_insights(self) -> Dict[str, Any]:
        """Get insights about discovered clusters"""
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        insights = {
            'cluster_summary': {},
            'anomaly_patterns': {},
            'recommendations': []
        }
        
        for cluster_type in ['text', 'numerical', 'combined']:
            cluster_key = f'{cluster_type}_clusters'
            if cluster_key in self.cluster_profiles:
                analysis = self.cluster_profiles[cluster_key]
                
                insights['cluster_summary'][cluster_type] = {
                    'n_clusters': analysis['n_clusters'],
                    'noise_ratio': analysis['noise_ratio'],
                    'largest_cluster_size': max(analysis['cluster_sizes'].values()) if analysis['cluster_sizes'] else 0
                }
                
                # Identify potential anomaly patterns
                high_noise_ratio = analysis['noise_ratio'] > 0.1
                many_small_clusters = analysis['n_clusters'] > 5
                
                insights['anomaly_patterns'][cluster_type] = {
                    'high_noise_ratio': high_noise_ratio,
                    'many_small_clusters': many_small_clusters,
                    'fragmented_patterns': high_noise_ratio and many_small_clusters
                }
        
        # Generate recommendations
        if insights['anomaly_patterns']['combined']['high_noise_ratio']:
            insights['recommendations'].append("High noise ratio detected - consider reviewing DBSCAN parameters")
        
        if insights['anomaly_patterns']['text']['many_small_clusters']:
            insights['recommendations'].append("Many small text clusters found - potential for fine-grained anomaly detection")
        
        return insights
    
    # Include all other methods from original class (save_model, load_model, etc.)
    def save_model(self, filepath: str = None):
        """Save the enhanced model with DBSCAN components"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        if filepath is None:
            filepath = os.path.join(self.model_dir, "enhanced_ensemble_model.pkl")
        
        model_data = {
            'text_vectorizer': self.text_vectorizer,
            'svm_model': self.svm_model,
            'isolation_model': self.isolation_model,
            'scaler': self.scaler,
            'text_dbscan': self.text_dbscan,
            'numerical_dbscan': self.numerical_dbscan,
            'combined_dbscan': self.combined_dbscan,
            'text_pca': self.text_pca,
            'numerical_pca': self.numerical_pca,
            'is_trained': self.is_trained,
            'training_stats': self.training_stats,
            'feature_names': self.feature_names,
            'text_feature_names': getattr(self, 'text_feature_names', []),
            'cluster_profiles': self.cluster_profiles,
            'text_weight': self.text_weight,
            'statistical_weight': self.statistical_weight,
            'density_weight': self.density_weight,
            'threshold': self.threshold,
            'dbscan_params': self.dbscan_params
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Enhanced model saved to {filepath}")
    
    def load_model(self, filepath: str = None):
        """Load the enhanced model with DBSCAN components"""
        if filepath is None:
            filepath = os.path.join(self.model_dir, "enhanced_ensemble_model.pkl")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Load all components
        self.text_vectorizer = model_data['text_vectorizer']
        self.svm_model = model_data['svm_model']
        self.isolation_model = model_data['isolation_model']
        self.scaler = model_data['scaler']
        self.text_dbscan = model_data['text_dbscan']
        self.numerical_dbscan = model_data['numerical_dbscan']
        self.combined_dbscan = model_data['combined_dbscan']
        self.text_pca = model_data['text_pca']
        self.numerical_pca = model_data['numerical_pca']
        self.is_trained = model_data['is_trained']
        self.training_stats = model_data['training_stats']
        self.feature_names = model_data['feature_names']
        self.text_feature_names = model_data.get('text_feature_names', [])
        self.cluster_profiles = model_data['cluster_profiles']
        self.text_weight = model_data['text_weight']
        self.statistical_weight = model_data['statistical_weight']
        self.density_weight = model_data['density_weight']
        self.threshold = model_data['threshold']
        self.dbscan_params = model_data['dbscan_params']
        
        print(f"Enhanced model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get enhanced model information"""
        base_info = {
            'is_trained': self.is_trained,
            'training_stats': self.training_stats,
            'ensemble_config': {
                'text_weight': self.text_weight,
                'statistical_weight': self.statistical_weight,
                'density_weight': self.density_weight,
                'threshold': self.threshold
            },
            'feature_names': self.feature_names,
            'dbscan_params': self.dbscan_params
        }
        
        if self.is_trained:
            base_info['cluster_insights'] = self.get_cluster_insights()
        
        return base_info
    
    def sessionize_ej_log(self, ej_log_text):
        """
        Sessionize raw EJ log text into individual sessions
        Compatible with existing API
        """
        import re
        
        # Split on transaction markers
        sessions = []
        
        # Look for transaction start/end patterns
        transaction_pattern = r'\[020t.*?\*.*?\*.*?\*.*?\*.*?\*TRANSACTION START\*.*?(?=\[020t.*?\*TRANSACTION START\*|\[020t.*?\*PRIMARY CARD READER ACTIVATED\*|$)'
        
        matches = re.findall(transaction_pattern, ej_log_text, re.DOTALL)
        
        for match in matches:
            # Clean up the session text
            cleaned_session = match.strip()
            if cleaned_session and len(cleaned_session) > 50:  # Filter out very short sessions
                sessions.append(cleaned_session)
        
        # If no matches found, try simpler pattern
        if not sessions:
            # Split on timestamp patterns
            lines = ej_log_text.split('\n')
            current_session = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if this is a new session start
                if '[020t*' in line and '*TRANSACTION START*' in line:
                    if current_session:
                        session_text = '\n'.join(current_session)
                        if len(session_text) > 50:
                            sessions.append(session_text)
                    current_session = [line]
                elif '[020t*' in line and '*PRIMARY CARD READER ACTIVATED*' in line:
                    if current_session:
                        session_text = '\n'.join(current_session)
                        if len(session_text) > 50:
                            sessions.append(session_text)
                    current_session = []
                else:
                    current_session.append(line)
            
            # Add the last session if it exists
            if current_session:
                session_text = '\n'.join(current_session)
                if len(session_text) > 50:
                    sessions.append(session_text)
        
        # If still no sessions, split by major sections
        if not sessions:
            sections = re.split(r'\[020t.*?\*\d+\*.*?\*.*?\*', ej_log_text)
            for section in sections:
                section = section.strip()
                if section and len(section) > 50:
                    sessions.append(section)
        
        return sessions
