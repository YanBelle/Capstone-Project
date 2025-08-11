"""
Session-Specific Model Evaluation Module
Provides individual session analysis for each model in the ensemble
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import base64
from io import BytesIO

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

from loguru import logger

class SessionModelEvaluator:
    """Evaluate individual EJ sessions against specific models"""
    
    def __init__(self, ml_analyzer=None):
        self.ml_analyzer = ml_analyzer
        
    def evaluate_session_isolation_forest(self, session_id: str, cleaned_text: str) -> Dict[str, Any]:
        """Evaluate a single session using Isolation Forest"""
        try:
            if not self.ml_analyzer or not hasattr(self.ml_analyzer, 'isolation_forest'):
                return {
                    'error': 'Isolation Forest not available', 
                    'status': 'ERROR',
                    'message': 'The Isolation Forest model is not loaded. Please ensure models are trained.'
                }
            
            # Check if the model is fitted
            try:
                # Try to access model attributes that only exist after fitting
                if not hasattr(self.ml_analyzer.isolation_forest, 'offset_'):
                    return {
                        'error': 'Isolation Forest not fitted', 
                        'status': 'ERROR',
                        'message': 'The Isolation Forest model is not fitted yet. Call fit() with appropriate arguments before using this estimator.'
                    }
            except Exception as fit_check_error:
                return {
                    'error': f'Isolation Forest not properly initialized: {str(fit_check_error)}',
                    'status': 'ERROR',
                    'message': 'The Isolation Forest model is not properly initialized. Please retrain the models.'
                }
            
            # Generate embedding for this session
            embedding = self._generate_session_embedding(cleaned_text)
            if embedding is None:
                return {'error': 'Could not generate embedding', 'status': 'ERROR'}
            
            # Check if scaler is fitted
            if not hasattr(self.ml_analyzer.scaler, 'scale_'):
                return {
                    'error': 'Scaler not fitted',
                    'status': 'ERROR', 
                    'message': 'The feature scaler is not fitted. Please train the models first.'
                }
            
            # Scale the embedding
            embedding_scaled = self.ml_analyzer.scaler.transform([embedding])
            
            # Apply PCA if trained
            if hasattr(self.ml_analyzer.pca, 'components_'):
                embedding_scaled = self.ml_analyzer.pca.transform(embedding_scaled)
            
            # Get Isolation Forest prediction and score
            prediction = self.ml_analyzer.isolation_forest.predict(embedding_scaled)[0]
            score = self.ml_analyzer.isolation_forest.score_samples(embedding_scaled)[0]
            
            # Calculate decision details
            decision_path = self.ml_analyzer.isolation_forest.decision_function(embedding_scaled)[0]
            
            result = {
                'session_id': session_id,
                'model': 'isolation_forest',
                'prediction': 'anomaly' if prediction == -1 else 'normal',
                'anomaly_score': float(score),
                'decision_function': float(decision_path),
                'confidence': float(1.0 - score) if prediction == -1 else float(score),
                'explanation': self._explain_isolation_forest_decision(score, prediction),
                'raw_text_length': len(cleaned_text),
                'embedding_dimension': len(embedding),
                'processed_features': embedding_scaled.shape[1],
                'evaluation_timestamp': datetime.now().isoformat()
            }
            
            # Add visualization if available
            if VISUALIZATION_AVAILABLE:
                viz_data = self._create_session_if_visualization(embedding_scaled, score, prediction)
                result['visualization'] = viz_data
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating session {session_id} with Isolation Forest: {e}")
            return {'error': str(e), 'session_id': session_id, 'model': 'isolation_forest'}
    
    def evaluate_session_svm(self, session_id: str, cleaned_text: str) -> Dict[str, Any]:
        """Evaluate a single session using One-Class SVM"""
        try:
            if not self.ml_analyzer or not hasattr(self.ml_analyzer, 'one_class_svm'):
                return {
                    'error': 'One-Class SVM not available',
                    'status': 'ERROR',
                    'message': 'The One-Class SVM model is not loaded. Please ensure models are trained.'
                }
            
            # Check if the model is fitted
            try:
                if not hasattr(self.ml_analyzer.one_class_svm, 'support_vectors_'):
                    return {
                        'error': 'One-Class SVM not fitted',
                        'status': 'ERROR',
                        'message': 'The One-Class SVM model is not fitted yet. Call fit() with appropriate arguments before using this estimator.'
                    }
            except Exception as fit_check_error:
                return {
                    'error': f'One-Class SVM not properly initialized: {str(fit_check_error)}',
                    'status': 'ERROR',
                    'message': 'The One-Class SVM model is not properly initialized. Please retrain the models.'
                }
            
            # Generate embedding for this session
            embedding = self._generate_session_embedding(cleaned_text)
            if embedding is None:
                return {'error': 'Could not generate embedding', 'status': 'ERROR'}
            
            # Check if scaler is fitted
            if not hasattr(self.ml_analyzer.scaler, 'scale_'):
                return {
                    'error': 'Scaler not fitted',
                    'status': 'ERROR',
                    'message': 'The feature scaler is not fitted. Please train the models first.'
                }
            
            # Scale the embedding
            embedding_scaled = self.ml_analyzer.scaler.transform([embedding])
            
            # Apply PCA if trained
            if hasattr(self.ml_analyzer.pca, 'components_'):
                embedding_scaled = self.ml_analyzer.pca.transform(embedding_scaled)
            
            # Get SVM prediction and decision function
            prediction = self.ml_analyzer.one_class_svm.predict(embedding_scaled)[0]
            decision_score = self.ml_analyzer.one_class_svm.decision_function(embedding_scaled)[0]
            
            # Calculate distance to decision boundary
            distance_to_boundary = abs(decision_score)
            
            result = {
                'session_id': session_id,
                'model': 'one_class_svm',
                'prediction': 'anomaly' if prediction == -1 else 'normal',
                'decision_score': float(decision_score),
                'distance_to_boundary': float(distance_to_boundary),
                'confidence': float(distance_to_boundary),
                'explanation': self._explain_svm_decision(decision_score, prediction),
                'raw_text_length': len(cleaned_text),
                'embedding_dimension': len(embedding),
                'processed_features': embedding_scaled.shape[1],
                'evaluation_timestamp': datetime.now().isoformat()
            }
            
            # Add support vector information if available
            if hasattr(self.ml_analyzer.one_class_svm, 'support_vectors_'):
                n_support_vectors = len(self.ml_analyzer.one_class_svm.support_vectors_)
                result['support_vectors_count'] = n_support_vectors
                
                # Calculate distance to nearest support vector
                support_vectors = self.ml_analyzer.one_class_svm.support_vectors_
                distances = [np.linalg.norm(embedding_scaled[0] - sv) for sv in support_vectors]
                result['distance_to_nearest_sv'] = float(min(distances)) if distances else None
            
            # Add visualization if available
            if VISUALIZATION_AVAILABLE:
                viz_data = self._create_session_svm_visualization(embedding_scaled, decision_score, prediction)
                result['visualization'] = viz_data
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating session {session_id} with One-Class SVM: {e}")
            return {'error': str(e), 'session_id': session_id, 'model': 'one_class_svm'}
    
    def evaluate_session_dbscan(self, session_id: str, cleaned_text: str) -> Dict[str, Any]:
        """Evaluate a single session using DBSCAN clustering"""
        try:
            if not self.ml_analyzer or not hasattr(self.ml_analyzer, 'dbscan'):
                return {
                    'error': 'DBSCAN not available',
                    'status': 'ERROR', 
                    'message': 'DBSCAN clustering model is not loaded. Please ensure models are trained.'
                }
            
            # Check if the model is fitted
            try:
                if not hasattr(self.ml_analyzer.dbscan, 'labels_'):
                    return {
                        'error': 'DBSCAN not fitted',
                        'status': 'ERROR',
                        'message': 'DBSCAN clustering model is not fitted yet. Call fit() with appropriate arguments before using this estimator.'
                    }
            except Exception as fit_check_error:
                return {
                    'error': f'DBSCAN not properly initialized: {str(fit_check_error)}',
                    'status': 'ERROR',
                    'message': 'DBSCAN clustering model is not properly initialized. Please retrain the models.'
                }
            
            # Generate embedding for this session
            embedding = self._generate_session_embedding(cleaned_text)
            if embedding is None:
                return {'error': 'Could not generate embedding', 'status': 'ERROR'}
            
            # Check if scaler is fitted
            if not hasattr(self.ml_analyzer.scaler, 'scale_'):
                return {
                    'error': 'Scaler not fitted',
                    'status': 'ERROR',
                    'message': 'The feature scaler is not fitted. Please train the models first.'
                }
            
            # Scale the embedding
            embedding_scaled = self.ml_analyzer.scaler.transform([embedding])
            
            # Apply PCA if trained
            if hasattr(self.ml_analyzer.pca, 'components_'):
                embedding_scaled = self.ml_analyzer.pca.transform(embedding_scaled)
            
            # For DBSCAN, we need to use the trained model to find the closest cluster
            # Since DBSCAN doesn't have a predict method, we'll use distance-based approach
            cluster_label = self._predict_dbscan_cluster(embedding_scaled[0])
            
            # Calculate distance to cluster centers
            cluster_distances = self._calculate_cluster_distances(embedding_scaled[0])
            
            result = {
                'session_id': session_id,
                'model': 'dbscan',
                'predicted_cluster': int(cluster_label) if cluster_label != -1 else 'outlier',
                'is_outlier': cluster_label == -1,
                'cluster_distances': cluster_distances,
                'explanation': self._explain_dbscan_decision(cluster_label, cluster_distances),
                'raw_text_length': len(cleaned_text),
                'embedding_dimension': len(embedding),
                'processed_features': embedding_scaled.shape[1],
                'evaluation_timestamp': datetime.now().isoformat()
            }
            
            # Add DBSCAN parameters
            result['dbscan_parameters'] = {
                'eps': getattr(self.ml_analyzer.dbscan, 'eps', 'unknown'),
                'min_samples': getattr(self.ml_analyzer.dbscan, 'min_samples', 'unknown'),
                'metric': getattr(self.ml_analyzer.dbscan, 'metric', 'unknown')
            }
            
            # Add visualization if available
            if VISUALIZATION_AVAILABLE:
                viz_data = self._create_session_dbscan_visualization(embedding_scaled, cluster_label, cluster_distances)
                result['visualization'] = viz_data
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating session {session_id} with DBSCAN: {e}")
            return {'error': str(e), 'session_id': session_id, 'model': 'dbscan'}
    
    def evaluate_session_deeplog(self, session_id: str, cleaned_text: str) -> Dict[str, Any]:
        """Evaluate a single session using DeepLog LSTM"""
        try:
            if not self.ml_analyzer or not hasattr(self.ml_analyzer, 'deeplog_analyzer') or not self.ml_analyzer.deeplog_analyzer:
                return {
                    'error': 'DeepLog analyzer not available',
                    'status': 'ERROR',
                    'message': 'DeepLog LSTM analyzer is not loaded. Please ensure the DeepLog module is properly initialized.'
                }
            
            if not hasattr(self.ml_analyzer, 'deeplog_trained') or not self.ml_analyzer.deeplog_trained:
                return {
                    'error': 'DeepLog model not trained',
                    'status': 'ERROR',
                    'message': 'DeepLog LSTM model is not trained yet. Please train the model first.'
                }
            
            # Extract event sequence from the session
            event_sequence = self.ml_analyzer.deeplog_analyzer.extract_event_sequence(cleaned_text)
            
            if len(event_sequence) < 2:
                return {
                    'session_id': session_id,
                    'model': 'deeplog_lstm',
                    'prediction': 'insufficient_data',
                    'event_sequence': event_sequence,
                    'explanation': 'Insufficient events for sequence analysis (need at least 2 events)',
                    'raw_text_length': len(cleaned_text),
                    'evaluation_timestamp': datetime.now().isoformat()
                }
            
            # Detect anomalies using DeepLog
            is_anomalous, confidence, anomaly_details = self.ml_analyzer.deeplog_analyzer.detect_anomaly(event_sequence)
            
            # Check transaction completeness
            is_complete, completeness_score, missing_events = self.ml_analyzer.deeplog_analyzer.check_transaction_completeness(event_sequence)
            
            result = {
                'session_id': session_id,
                'model': 'deeplog_lstm',
                'prediction': 'anomaly' if is_anomalous else 'normal',
                'confidence': float(confidence),
                'is_complete_transaction': is_complete,
                'completeness_score': float(completeness_score),
                'missing_events': missing_events,
                'event_sequence': event_sequence,
                'sequence_length': len(event_sequence),
                'anomaly_details': anomaly_details,
                'explanation': self._explain_deeplog_decision(is_anomalous, confidence, anomaly_details, is_complete),
                'raw_text_length': len(cleaned_text),
                'evaluation_timestamp': datetime.now().isoformat()
            }
            
            # Add sequence analysis
            result['sequence_analysis'] = {
                'unique_events': len(set(event_sequence)),
                'event_repetition_ratio': 1.0 - (len(set(event_sequence)) / len(event_sequence)),
                'common_patterns': self._identify_common_patterns(event_sequence),
                'transition_analysis': self._analyze_event_transitions(event_sequence)
            }
            
            # Add visualization if available
            if VISUALIZATION_AVAILABLE:
                viz_data = self._create_session_deeplog_visualization(event_sequence, is_anomalous, confidence)
                result['visualization'] = viz_data
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating session {session_id} with DeepLog: {e}")
            return {'error': str(e), 'session_id': session_id, 'model': 'deeplog_lstm'}
    
    def evaluate_session_sentiment(self, session_id: str, cleaned_text: str) -> Dict[str, Any]:
        """Evaluate a single session using sentiment analysis (VADER + TextBlob)"""
        try:
            if not self.ml_analyzer:
                return {'error': 'ML analyzer not available'}
            
            # Create a dummy session object for sentiment analysis
            class DummySession:
                def __init__(self, text):
                    self.raw_text = text
                    self.session_id = session_id
            
            dummy_session = DummySession(cleaned_text)
            
            # Run sentiment analysis
            sentiment_result = self.ml_analyzer.analyze_negative_sentiment(dummy_session)
            
            # Extract key components
            vader_score = sentiment_result.get('vader_score', 0.0)
            textblob_score = sentiment_result.get('textblob_score', 0.0)
            confidence = sentiment_result.get('confidence', 0.0)
            severity_level = sentiment_result.get('severity_level', 'LOW')
            negative_phrases = sentiment_result.get('negative_phrases', [])
            
            # Determine overall sentiment classification
            combined_score = min(vader_score, textblob_score)
            is_negative = combined_score < -0.3
            
            result = {
                'session_id': session_id,
                'model': 'sentiment_analysis',
                'prediction': 'negative_sentiment' if is_negative else 'neutral_positive',
                'vader_score': float(vader_score),
                'textblob_score': float(textblob_score),
                'combined_score': float(combined_score),
                'confidence': float(confidence),
                'severity_level': severity_level,
                'negative_phrases': negative_phrases,
                'explanation': self._explain_sentiment_decision(vader_score, textblob_score, negative_phrases),
                'raw_text_length': len(cleaned_text),
                'evaluation_timestamp': datetime.now().isoformat()
            }
            
            # Add detailed sentiment breakdown
            result['sentiment_breakdown'] = {
                'vader_details': sentiment_result.get('vader_details', {}),
                'textblob_subjectivity': sentiment_result.get('textblob_subjectivity', 0.0),
                'technical_failure_score': sentiment_result.get('technical_failure_score', 0.0),
                'detected_patterns': sentiment_result.get('detected_patterns', [])
            }
            
            # Add phrase analysis
            if negative_phrases:
                result['phrase_analysis'] = self._analyze_negative_phrases(cleaned_text, negative_phrases)
            
            # Add visualization if available
            if VISUALIZATION_AVAILABLE:
                viz_data = self._create_session_sentiment_visualization(sentiment_result)
                result['visualization'] = viz_data
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating session {session_id} with sentiment analysis: {e}")
            return {'error': str(e), 'session_id': session_id, 'model': 'sentiment_analysis'}
    
    def evaluate_session_preprocessing(self, session_id: str, cleaned_text: str) -> Dict[str, Any]:
        """Evaluate preprocessing impact on a single session"""
        try:
            if not self.ml_analyzer:
                return {'error': 'ML analyzer not available'}
            
            # Generate original embedding
            original_embedding = self._generate_session_embedding(cleaned_text)
            if original_embedding is None:
                return {'error': 'Could not generate embedding'}
            
            # Apply StandardScaler
            scaled_embedding = self.ml_analyzer.scaler.transform([original_embedding])[0]
            
            # Apply PCA if available
            pca_embedding = None
            if hasattr(self.ml_analyzer.pca, 'components_'):
                pca_embedding = self.ml_analyzer.pca.transform([scaled_embedding])[0]
            
            result = {
                'session_id': session_id,
                'model': 'preprocessing',
                'original_embedding': {
                    'dimension': len(original_embedding),
                    'mean': float(np.mean(original_embedding)),
                    'std': float(np.std(original_embedding)),
                    'min': float(np.min(original_embedding)),
                    'max': float(np.max(original_embedding))
                },
                'scaled_embedding': {
                    'dimension': len(scaled_embedding),
                    'mean': float(np.mean(scaled_embedding)),
                    'std': float(np.std(scaled_embedding)),
                    'min': float(np.min(scaled_embedding)),
                    'max': float(np.max(scaled_embedding))
                },
                'explanation': self._explain_preprocessing_impact(original_embedding, scaled_embedding, pca_embedding),
                'raw_text_length': len(cleaned_text),
                'evaluation_timestamp': datetime.now().isoformat()
            }
            
            # Add PCA information if available
            if pca_embedding is not None:
                result['pca_embedding'] = {
                    'dimension': len(pca_embedding),
                    'mean': float(np.mean(pca_embedding)),
                    'std': float(np.std(pca_embedding)),
                    'variance_explained': float(np.sum(self.ml_analyzer.pca.explained_variance_ratio_)),
                    'components_used': len(pca_embedding)
                }
            
            # Calculate preprocessing impact metrics
            result['impact_metrics'] = {
                'scale_factor': float(np.std(original_embedding) / np.std(scaled_embedding)) if np.std(scaled_embedding) > 0 else 0,
                'dimension_reduction': (len(original_embedding) - len(pca_embedding)) / len(original_embedding) if pca_embedding is not None else 0,
                'information_retention': float(np.sum(self.ml_analyzer.pca.explained_variance_ratio_)) if pca_embedding is not None else 1.0
            }
            
            # Add visualization if available
            if VISUALIZATION_AVAILABLE:
                viz_data = self._create_session_preprocessing_visualization(original_embedding, scaled_embedding, pca_embedding)
                result['visualization'] = viz_data
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating session {session_id} preprocessing: {e}")
            return {'error': str(e), 'session_id': session_id, 'model': 'preprocessing'}
    
    def _generate_session_embedding(self, cleaned_text: str):
        """Generate embedding for a single session"""
        try:
            if hasattr(self.ml_analyzer, 'generate_embeddingsUsingBERT'):
                # Create dummy session object
                class DummySession:
                    def __init__(self, text):
                        self.raw_text = text
                
                dummy_session = DummySession(cleaned_text)
                embeddings = self.ml_analyzer.generate_embeddingsUsingBERT([dummy_session])
                return embeddings[0] if len(embeddings) > 0 else None
            else:
                return None
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def _predict_dbscan_cluster(self, embedding):
        """Predict DBSCAN cluster for a new point"""
        try:
            # This is a simplified approach - in practice, you'd need to retrain or use a more sophisticated method
            if hasattr(self.ml_analyzer, 'sessions') and self.ml_analyzer.sessions:
                # Find the trained cluster centers (stored during training)
                if hasattr(self.ml_analyzer, 'cluster_centers_'):
                    distances = []
                    for center in self.ml_analyzer.cluster_centers_.values():
                        dist = np.linalg.norm(embedding - center)
                        distances.append(dist)
                    
                    min_distance = min(distances)
                    min_cluster = list(self.ml_analyzer.cluster_centers_.keys())[distances.index(min_distance)]
                    
                    # Check if point is within eps distance
                    eps = getattr(self.ml_analyzer.dbscan, 'eps', 0.5)
                    if min_distance <= eps:
                        return min_cluster
                    else:
                        return -1  # Outlier
                else:
                    return -1  # No cluster centers available
            else:
                return -1
        except Exception:
            return -1
    
    def _calculate_cluster_distances(self, embedding):
        """Calculate distances to all cluster centers"""
        try:
            if hasattr(self.ml_analyzer, 'cluster_centers_'):
                distances = {}
                for cluster_id, center in self.ml_analyzer.cluster_centers_.items():
                    dist = float(np.linalg.norm(embedding - center))
                    distances[f'cluster_{cluster_id}'] = dist
                return distances
            else:
                return {}
        except Exception:
            return {}
    
    def _identify_common_patterns(self, event_sequence):
        """Identify common patterns in event sequence"""
        patterns = []
        
        # Look for repeated subsequences
        for length in range(2, min(5, len(event_sequence))):
            for i in range(len(event_sequence) - length + 1):
                pattern = tuple(event_sequence[i:i+length])
                count = 0
                for j in range(len(event_sequence) - length + 1):
                    if tuple(event_sequence[j:j+length]) == pattern:
                        count += 1
                if count > 1:
                    patterns.append({'pattern': list(pattern), 'count': count, 'length': length})
        
        return patterns[:5]  # Return top 5 patterns
    
    def _analyze_event_transitions(self, event_sequence):
        """Analyze transitions between events"""
        transitions = {}
        for i in range(len(event_sequence) - 1):
            transition = f"{event_sequence[i]} -> {event_sequence[i+1]}"
            transitions[transition] = transitions.get(transition, 0) + 1
        
        return dict(sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:10])
    
    def _analyze_negative_phrases(self, text, negative_phrases):
        """Analyze context around negative phrases"""
        analysis = []
        text_upper = text.upper()
        
        for phrase in negative_phrases:
            matches = []
            start = 0
            while True:
                pos = text_upper.find(phrase, start)
                if pos == -1:
                    break
                
                # Get context around the phrase
                context_start = max(0, pos - 50)
                context_end = min(len(text), pos + len(phrase) + 50)
                context = text[context_start:context_end]
                
                matches.append({
                    'position': pos,
                    'context': context.strip()
                })
                start = pos + 1
            
            if matches:
                analysis.append({
                    'phrase': phrase,
                    'occurrences': len(matches),
                    'contexts': matches[:3]  # First 3 contexts
                })
        
        return analysis
    
    # Explanation methods
    def _explain_isolation_forest_decision(self, score, prediction):
        if prediction == -1:
            return f"Isolation Forest classified this session as an anomaly with score {score:.4f}. Lower scores indicate higher anomaly likelihood. This session required fewer splits than normal sessions to isolate, suggesting unusual patterns."
        else:
            return f"Isolation Forest classified this session as normal with score {score:.4f}. The session follows typical patterns and required many splits to isolate, indicating normal behavior."
    
    def _explain_svm_decision(self, decision_score, prediction):
        if prediction == -1:
            return f"One-Class SVM classified this session as an anomaly with decision score {decision_score:.4f}. Negative scores indicate the session falls outside the learned normal region in the feature space."
        else:
            return f"One-Class SVM classified this session as normal with decision score {decision_score:.4f}. Positive scores indicate the session falls within the learned normal region."
    
    def _explain_dbscan_decision(self, cluster_label, cluster_distances):
        if cluster_label == -1:
            min_distance = min(cluster_distances.values()) if cluster_distances else float('inf')
            return f"DBSCAN classified this session as an outlier. It doesn't belong to any cluster and has minimum distance {min_distance:.4f} to the nearest cluster center."
        else:
            return f"DBSCAN assigned this session to cluster {cluster_label}. It shares similar characteristics with other sessions in this cluster."
    
    def _explain_deeplog_decision(self, is_anomalous, confidence, anomaly_details, is_complete):
        explanation = f"DeepLog LSTM analyzed the event sequence with {confidence:.2%} confidence. "
        
        if is_anomalous:
            explanation += f"The sequence contains anomalous patterns: {anomaly_details.get('description', 'Unexpected event sequence')}. "
        else:
            explanation += "The event sequence follows learned normal patterns. "
        
        if not is_complete:
            explanation += "The transaction appears incomplete based on learned completion patterns."
        else:
            explanation += "The transaction appears complete."
        
        return explanation
    
    def _explain_sentiment_decision(self, vader_score, textblob_score, negative_phrases):
        explanation = f"Sentiment analysis using VADER (score: {vader_score:.3f}) and TextBlob (score: {textblob_score:.3f}). "
        
        if vader_score < -0.3 or textblob_score < -0.3:
            explanation += "Negative sentiment detected, indicating potential issues or failures. "
        else:
            explanation += "Neutral or positive sentiment detected. "
        
        if negative_phrases:
            explanation += f"Found {len(negative_phrases)} negative phrases indicating problems."
        
        return explanation
    
    def _explain_preprocessing_impact(self, original, scaled, pca):
        explanation = f"Preprocessing transformed {len(original)}-dimensional embedding. "
        explanation += f"StandardScaler normalized features (std: {np.std(original):.4f} -> {np.std(scaled):.4f}). "
        
        if pca is not None:
            explanation += f"PCA reduced dimensions to {len(pca)} while retaining most information."
        
        return explanation
    
    # Visualization methods (simplified versions)
    def _create_session_if_visualization(self, embedding_scaled, score, prediction):
        """Create Isolation Forest visualization for single session"""
        if not VISUALIZATION_AVAILABLE:
            return None
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Score visualization
            ax1.bar(['Session Score'], [score], color='red' if prediction == -1 else 'blue', alpha=0.7)
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax1.set_ylabel('Anomaly Score')
            ax1.set_title('Isolation Forest Score')
            ax1.grid(True, alpha=0.3)
            
            # Feature distribution
            if len(embedding_scaled[0]) <= 20:  # Only for small feature sets
                ax2.bar(range(len(embedding_scaled[0])), embedding_scaled[0], alpha=0.7)
                ax2.set_xlabel('Feature Index')
                ax2.set_ylabel('Feature Value')
                ax2.set_title('Processed Feature Values')
            else:
                ax2.hist(embedding_scaled[0], bins=20, alpha=0.7)
                ax2.set_xlabel('Feature Value')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Feature Distribution')
            
            plt.tight_layout()
            
            # Convert to base64
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return img_base64
            
        except Exception as e:
            logger.error(f"Error creating IF visualization: {e}")
            return None
    
    def _create_session_svm_visualization(self, embedding_scaled, decision_score, prediction):
        """Create SVM visualization for single session"""
        if not VISUALIZATION_AVAILABLE:
            return None
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Decision score visualization
            ax1.bar(['Session'], [decision_score], color='red' if prediction == -1 else 'blue', alpha=0.7)
            ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Decision Boundary')
            ax1.set_ylabel('Decision Score')
            ax1.set_title('SVM Decision Score')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Feature radar chart (if small number of features)
            if len(embedding_scaled[0]) <= 10:
                angles = np.linspace(0, 2*np.pi, len(embedding_scaled[0]), endpoint=False)
                values = embedding_scaled[0]
                angles = np.concatenate((angles, [angles[0]]))
                values = np.concatenate((values, [values[0]]))
                
                ax2 = plt.subplot(122, projection='polar')
                ax2.plot(angles, values, 'o-', linewidth=2, alpha=0.7)
                ax2.fill(angles, values, alpha=0.25)
                ax2.set_title('Feature Profile')
            else:
                ax2.hist(embedding_scaled[0], bins=15, alpha=0.7, color='lightcoral')
                ax2.set_xlabel('Feature Value')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Feature Distribution')
            
            plt.tight_layout()
            
            # Convert to base64
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return img_base64
            
        except Exception as e:
            logger.error(f"Error creating SVM visualization: {e}")
            return None
    
    def _create_session_dbscan_visualization(self, embedding_scaled, cluster_label, cluster_distances):
        """Create DBSCAN visualization for single session"""
        if not VISUALIZATION_AVAILABLE:
            return None
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Cluster assignment
            color = 'red' if cluster_label == -1 else 'blue'
            label = 'Outlier' if cluster_label == -1 else f'Cluster {cluster_label}'
            
            ax1.bar(['Session'], [1], color=color, alpha=0.7, label=label)
            ax1.set_ylabel('Assignment')
            ax1.set_title('DBSCAN Cluster Assignment')
            ax1.legend()
            ax1.set_ylim(0, 1.2)
            
            # Distance to clusters
            if cluster_distances:
                clusters = list(cluster_distances.keys())
                distances = list(cluster_distances.values())
                
                bars = ax2.bar(clusters, distances, alpha=0.7)
                # Highlight closest cluster
                min_idx = distances.index(min(distances))
                bars[min_idx].set_color('orange')
                
                ax2.set_xlabel('Cluster')
                ax2.set_ylabel('Distance')
                ax2.set_title('Distance to Cluster Centers')
                plt.setp(ax2.get_xticklabels(), rotation=45)
            else:
                ax2.text(0.5, 0.5, 'No cluster distance\ndata available', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Cluster Distances')
            
            plt.tight_layout()
            
            # Convert to base64
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return img_base64
            
        except Exception as e:
            logger.error(f"Error creating DBSCAN visualization: {e}")
            return None
    
    def _create_session_deeplog_visualization(self, event_sequence, is_anomalous, confidence):
        """Create DeepLog visualization for single session"""
        if not VISUALIZATION_AVAILABLE:
            return None
        
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Event sequence timeline
            y_pos = range(len(event_sequence))
            colors = ['red' if is_anomalous else 'blue'] * len(event_sequence)
            
            ax1.barh(y_pos, [1] * len(event_sequence), color=colors, alpha=0.7)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(event_sequence)
            ax1.set_xlabel('Event Occurrence')
            ax1.set_title(f'Event Sequence ({"Anomalous" if is_anomalous else "Normal"}, Confidence: {confidence:.2%})')
            ax1.grid(True, alpha=0.3)
            
            # Event frequency
            event_counts = pd.Series(event_sequence).value_counts()
            ax2.bar(range(len(event_counts)), event_counts.values, alpha=0.7)
            ax2.set_xticks(range(len(event_counts)))
            ax2.set_xticklabels(event_counts.index, rotation=45, ha='right')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Event Frequency Distribution')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convert to base64
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return img_base64
            
        except Exception as e:
            logger.error(f"Error creating DeepLog visualization: {e}")
            return None
    
    def _create_session_sentiment_visualization(self, sentiment_result):
        """Create sentiment analysis visualization for single session"""
        if not VISUALIZATION_AVAILABLE:
            return None
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # Sentiment scores comparison
            scores = [
                sentiment_result.get('vader_score', 0),
                sentiment_result.get('textblob_score', 0)
            ]
            models = ['VADER', 'TextBlob']
            colors = ['red' if score < -0.3 else 'green' for score in scores]
            
            ax1.bar(models, scores, color=colors, alpha=0.7)
            ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax1.axhline(y=-0.3, color='red', linestyle='--', alpha=0.5, label='Negative Threshold')
            ax1.set_ylabel('Sentiment Score')
            ax1.set_title('Sentiment Scores')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # VADER breakdown (if available)
            vader_details = sentiment_result.get('vader_details', {})
            if vader_details:
                components = ['neg', 'neu', 'pos', 'compound']
                values = [vader_details.get(comp, 0) for comp in components]
                
                ax2.bar(components, values, alpha=0.7, color=['red', 'gray', 'green', 'blue'])
                ax2.set_ylabel('Score')
                ax2.set_title('VADER Components')
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'VADER details\nnot available', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('VADER Components')
            
            # Negative phrases
            negative_phrases = sentiment_result.get('negative_phrases', [])
            if negative_phrases:
                phrase_counts = pd.Series(negative_phrases).value_counts()[:10]
                
                ax3.barh(range(len(phrase_counts)), phrase_counts.values, alpha=0.7, color='red')
                ax3.set_yticks(range(len(phrase_counts)))
                ax3.set_yticklabels(phrase_counts.index)
                ax3.set_xlabel('Frequency')
                ax3.set_title('Top Negative Phrases')
            else:
                ax3.text(0.5, 0.5, 'No negative phrases\ndetected', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Negative Phrases')
            
            # Confidence and severity
            confidence = sentiment_result.get('confidence', 0)
            severity = sentiment_result.get('severity_level', 'LOW')
            
            severity_colors = {'LOW': 'green', 'MEDIUM': 'yellow', 'HIGH': 'orange', 'CRITICAL': 'red'}
            severity_color = severity_colors.get(severity, 'gray')
            
            ax4.bar(['Confidence'], [confidence], alpha=0.7, color='blue')
            ax4.bar(['Severity'], [{'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}.get(severity, 1)], 
                   alpha=0.7, color=severity_color)
            ax4.set_ylabel('Level')
            ax4.set_title(f'Confidence & Severity ({severity})')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convert to base64
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return img_base64
            
        except Exception as e:
            logger.error(f"Error creating sentiment visualization: {e}")
            return None
    
    def _create_session_preprocessing_visualization(self, original, scaled, pca):
        """Create preprocessing visualization for single session"""
        if not VISUALIZATION_AVAILABLE:
            return None
        
        try:
            fig_width = 15 if pca is not None else 10
            fig, axes = plt.subplots(1, 3 if pca is not None else 2, figsize=(fig_width, 5))
            if not isinstance(axes, np.ndarray):
                axes = [axes]
            
            # Original embedding distribution
            axes[0].hist(original, bins=20, alpha=0.7, color='blue', label='Original')
            axes[0].set_xlabel('Feature Value')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title(f'Original Embedding (dim={len(original)})')
            axes[0].grid(True, alpha=0.3)
            
            # Scaled embedding distribution
            axes[1].hist(scaled, bins=20, alpha=0.7, color='green', label='Scaled')
            axes[1].set_xlabel('Feature Value')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title(f'Scaled Embedding (dim={len(scaled)})')
            axes[1].grid(True, alpha=0.3)
            
            # PCA embedding (if available)
            if pca is not None and len(axes) > 2:
                axes[2].hist(pca, bins=15, alpha=0.7, color='red', label='PCA')
                axes[2].set_xlabel('Feature Value')
                axes[2].set_ylabel('Frequency')
                axes[2].set_title(f'PCA Embedding (dim={len(pca)})')
                axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convert to base64
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return img_base64
            
        except Exception as e:
            logger.error(f"Error creating preprocessing visualization: {e}")
            return None
