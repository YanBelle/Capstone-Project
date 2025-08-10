"""
Unsupervised EJ Log Analyzer
Completely unsupervised anomaly detection system for ATM transaction logs
Requires NO labeled data and automatically discovers patterns and anomalies
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import hdbscan
import umap
from typing import List, Dict, Tuple, Optional
import warnings
from loguru import logger
from collections import Counter
import asyncio

warnings.filterwarnings('ignore')

class UnsupervisedEJAnalyzer:
    """
    Completely unsupervised EJ log analyzer - requires NO labeled data
    """
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the analyzer with pre-trained sentence transformer
        
        Args:
            embedding_model: Name of the sentence transformer model to use
        """
        logger.info(f"Initializing UnsupervisedEJAnalyzer with model: {embedding_model}")
        
        # Initialize embedder
        try:
            self.embedder = SentenceTransformer(embedding_model)
            logger.info("✅ SentenceTransformer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer: {e}")
            raise
        
        # Initialize unsupervised models
        self.isolation_forest = IsolationForest(
            contamination='auto',
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        
        self.hdbscan = hdbscan.HDBSCAN(
            min_cluster_size=5,
            min_samples=3,
            cluster_selection_method='eom'
        )
        
        self.lof = LocalOutlierFactor(
            n_neighbors=20,
            novelty=False,
            contamination='auto'
        )
        
        # Storage for analysis results
        self.embeddings = None
        self.embeddings_normalized = None
        self.sequences = None
        self.results = {}
        
        logger.info("UnsupervisedEJAnalyzer initialized successfully")
        
    def analyze_sequences(self, sequences: List[str], 
                         perform_dim_reduction: bool = True) -> Dict:
        """
        Perform complete unsupervised analysis on EJ sequences
        
        Args:
            sequences: List of preprocessed EJ log sequences
            perform_dim_reduction: Whether to compute UMAP/PCA projections
            
        Returns:
            Dictionary containing all analysis results
        """
        if not sequences:
            raise ValueError("Empty sequences list provided")
            
        self.sequences = sequences
        logger.info(f"Starting analysis of {len(sequences)} sequences")
        
        # Step 1: Create embeddings
        logger.info("Step 1: Creating sequence embeddings...")
        self._create_embeddings()
        
        # Step 2: Anomaly detection with multiple methods
        logger.info("Step 2: Running anomaly detection algorithms...")
        self._detect_anomalies()
        
        # Step 3: Clustering analysis
        logger.info("Step 3: Performing clustering analysis...")
        self._perform_clustering()
        
        # Step 4: Dimensionality reduction for visualization
        if perform_dim_reduction:
            logger.info("Step 4: Computing dimensionality reductions...")
            self._compute_projections()
        
        # Step 5: Pattern analysis
        logger.info("Step 5: Analyzing discovered patterns...")
        self._analyze_patterns()
        
        # Step 6: Calculate performance metrics
        logger.info("Step 6: Calculating performance metrics...")
        self._calculate_metrics()
        
        logger.info("✅ Unsupervised analysis completed successfully")
        return self.results
    
    def _create_embeddings(self):
        """Create and normalize sequence embeddings"""
        try:
            # Create embeddings in batches for memory efficiency
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(self.sequences), batch_size):
                batch = self.sequences[i:i+batch_size]
                batch_embeddings = self.embedder.encode(
                    batch,
                    batch_size=len(batch),
                    show_progress_bar=False
                )
                all_embeddings.extend(batch_embeddings)
            
            self.embeddings = np.array(all_embeddings)
            
            # Normalize embeddings
            scaler = StandardScaler()
            self.embeddings_normalized = scaler.fit_transform(self.embeddings)
            
            logger.info(f"Created embeddings with shape: {self.embeddings.shape}")
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
        
    def _detect_anomalies(self):
        """Run multiple anomaly detection algorithms"""
        try:
            # Isolation Forest
            iso_predictions = self.isolation_forest.fit_predict(self.embeddings_normalized)
            iso_scores = self.isolation_forest.score_samples(self.embeddings_normalized)
            iso_anomalies = iso_predictions == -1
            
            # Local Outlier Factor
            lof_predictions = self.lof.fit_predict(self.embeddings_normalized)
            lof_scores = self.lof.negative_outlier_factor_
            lof_anomalies = lof_predictions == -1
            
            # Statistical outliers (based on embedding distances)
            distances = np.linalg.norm(
                self.embeddings_normalized - np.mean(self.embeddings_normalized, axis=0),
                axis=1
            )
            statistical_threshold = np.percentile(distances, 95)
            statistical_anomalies = distances > statistical_threshold
            
            # Consensus anomalies (detected by multiple methods)
            consensus_anomalies = iso_anomalies & lof_anomalies
            
            self.results['anomaly_detection'] = {
                'isolation_forest': {
                    'predictions': iso_predictions,
                    'scores': iso_scores,
                    'anomalies': iso_anomalies,
                    'n_anomalies': np.sum(iso_anomalies),
                    'anomaly_rate': np.mean(iso_anomalies)
                },
                'lof': {
                    'predictions': lof_predictions,
                    'scores': lof_scores,
                    'anomalies': lof_anomalies,
                    'n_anomalies': np.sum(lof_anomalies),
                    'anomaly_rate': np.mean(lof_anomalies)
                },
                'statistical': {
                    'distances': distances,
                    'threshold': statistical_threshold,
                    'anomalies': statistical_anomalies,
                    'n_anomalies': np.sum(statistical_anomalies),
                    'anomaly_rate': np.mean(statistical_anomalies)
                },
                'consensus': {
                    'anomalies': consensus_anomalies,
                    'n_anomalies': np.sum(consensus_anomalies),
                    'anomaly_rate': np.mean(consensus_anomalies)
                }
            }
            
            logger.info(f"Anomaly detection completed:")
            logger.info(f"  - Isolation Forest: {np.sum(iso_anomalies)} anomalies")
            logger.info(f"  - LOF: {np.sum(lof_anomalies)} anomalies")
            logger.info(f"  - Statistical: {np.sum(statistical_anomalies)} anomalies")
            logger.info(f"  - Consensus: {np.sum(consensus_anomalies)} anomalies")
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            raise
    
    def _perform_clustering(self):
        """Perform clustering analysis"""
        try:
            # HDBSCAN clustering
            cluster_labels = self.hdbscan.fit_predict(self.embeddings_normalized)
            
            # Calculate cluster probabilities
            probabilities = self.hdbscan.probabilities_
            
            # Find exemplars (most representative points) for each cluster
            exemplars = {}
            unique_clusters = [c for c in set(cluster_labels) if c != -1]
            
            for cluster_id in unique_clusters:
                cluster_mask = cluster_labels == cluster_id
                cluster_probs = probabilities[cluster_mask]
                cluster_indices = np.where(cluster_mask)[0]
                
                # Find the point with highest probability in this cluster
                exemplar_idx = cluster_indices[np.argmax(cluster_probs)]
                exemplars[cluster_id] = {
                    'index': exemplar_idx,
                    'sequence': self.sequences[exemplar_idx],
                    'probability': np.max(cluster_probs)
                }
            
            # Count noise points
            n_noise = np.sum(cluster_labels == -1)
            n_clusters = len(unique_clusters)
            
            self.results['clustering'] = {
                'labels': cluster_labels,
                'probabilities': probabilities,
                'exemplars': exemplars,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'cluster_sizes': pd.Series(cluster_labels).value_counts().to_dict(),
                'noise_ratio': n_noise / len(cluster_labels)
            }
            
            logger.info(f"Clustering completed: {n_clusters} clusters, {n_noise} noise points")
            
        except Exception as e:
            logger.error(f"Error in clustering: {e}")
            raise
    
    def _compute_projections(self):
        """Compute 2D projections for visualization"""
        try:
            # PCA projection
            pca = PCA(n_components=2, random_state=42)
            pca_projection = pca.fit_transform(self.embeddings_normalized)
            
            # UMAP projection
            umap_model = umap.UMAP(
                n_components=2,
                random_state=42,
                n_neighbors=15,
                min_dist=0.1
            )
            umap_projection = umap_model.fit_transform(self.embeddings_normalized)
            
            self.results['projections'] = {
                'pca': {
                    'coordinates': pca_projection,
                    'explained_variance_ratio': pca.explained_variance_ratio_,
                    'total_variance_explained': np.sum(pca.explained_variance_ratio_)
                },
                'umap': {
                    'coordinates': umap_projection
                }
            }
            
            logger.info(f"Projections computed - PCA variance explained: {np.sum(pca.explained_variance_ratio_):.3f}")
            
        except Exception as e:
            logger.error(f"Error computing projections: {e}")
            raise
    
    def _analyze_patterns(self):
        """Analyze patterns in clusters and anomalies"""
        try:
            patterns = []
            
            # Analyze each cluster
            cluster_labels = self.results['clustering']['labels']
            unique_clusters = [c for c in set(cluster_labels) if c != -1]
            
            for cluster_id in unique_clusters:
                cluster_mask = cluster_labels == cluster_id
                cluster_sequences = [self.sequences[i] for i, mask in enumerate(cluster_mask) if mask]
                cluster_size = len(cluster_sequences)
                
                # Calculate anomaly ratio in this cluster
                iso_anomalies = self.results['anomaly_detection']['isolation_forest']['anomalies']
                cluster_anomaly_ratio = np.mean(iso_anomalies[cluster_mask])
                
                # Get common tokens and pattern signature
                common_tokens = self._get_common_tokens(cluster_sequences)
                pattern_signature = self._get_pattern_signature(cluster_sequences)
                
                # Get exemplar
                exemplar = self.results['clustering']['exemplars'][cluster_id]
                
                patterns.append({
                    'cluster_id': cluster_id,
                    'size': cluster_size,
                    'percentage': cluster_size / len(self.sequences) * 100,
                    'anomaly_ratio': cluster_anomaly_ratio,
                    'pattern_signature': pattern_signature,
                    'common_tokens': common_tokens,
                    'exemplar_sequence': exemplar['sequence'][:100] + '...' if len(exemplar['sequence']) > 100 else exemplar['sequence'],
                    'exemplar_probability': exemplar['probability']
                })
            
            # Analyze noise points
            noise_mask = cluster_labels == -1
            if np.any(noise_mask):
                noise_sequences = [self.sequences[i] for i, mask in enumerate(noise_mask) if mask]
                noise_size = len(noise_sequences)
                
                # Noise points are often anomalies
                noise_anomaly_ratio = np.mean(self.results['anomaly_detection']['isolation_forest']['anomalies'][noise_mask])
                
                patterns.append({
                    'cluster_id': -1,
                    'size': noise_size,
                    'percentage': noise_size / len(self.sequences) * 100,
                    'anomaly_ratio': noise_anomaly_ratio,
                    'pattern_signature': 'UNCLUSTERED_NOISE',
                    'common_tokens': self._get_common_tokens(noise_sequences),
                    'exemplar_sequence': noise_sequences[0][:100] + '...' if len(noise_sequences[0]) > 100 else noise_sequences[0],
                    'exemplar_probability': 0.0
                })
            
            self.results['patterns'] = pd.DataFrame(patterns).sort_values('size', ascending=False)
            
            logger.info(f"Pattern analysis completed - found {len(patterns)} patterns")
            
        except Exception as e:
            logger.error(f"Error in pattern analysis: {e}")
            raise
    
    def _get_common_tokens(self, sequences: List[str], top_n: int = 5) -> List[Tuple[str, float]]:
        """Extract most common tokens from sequences"""
        try:
            all_tokens = []
            sample_size = min(50, len(sequences))  # Sample for efficiency
            
            for seq in sequences[:sample_size]:
                all_tokens.extend(seq.split())
            
            if not all_tokens:
                return []
            
            token_counts = Counter(all_tokens)
            total_tokens = len(all_tokens)
            
            common = [(token, count/total_tokens) for token, count in token_counts.most_common(top_n)]
            return common
            
        except Exception as e:
            logger.warning(f"Error extracting common tokens: {e}")
            return []
    
    def _get_pattern_signature(self, sequences: List[str]) -> str:
        """Infer pattern type from sequences"""
        try:
            if not sequences:
                return 'EMPTY_PATTERN'
            
            sample_text = ' '.join(sequences[:10]).upper()
            
            # Check for known patterns
            if 'GENAC_2_TC' in sample_text and 'RECEIPT_PRINTED' in sample_text:
                return 'SUCCESSFUL_TRANSACTION'
            elif 'M_' in sample_text or 'ERROR' in sample_text:
                return 'DEVICE_ERROR'
            elif 'GENAC_2_AAC' in sample_text:
                return 'AUTH_FAILURE'
            elif sample_text.count('TRANSACTION_END') < sample_text.count('TRANSACTION_START'):
                return 'INCOMPLETE_TRANSACTION'
            elif 'TIMEOUT' in sample_text:
                return 'TIMEOUT_ERROR'
            elif 'CARD_TAKEN' in sample_text and 'NOTES_TAKEN' in sample_text:
                return 'COMPLETED_TRANSACTION'
            elif 'PIN_ENTERED' in sample_text:
                return 'PIN_TRANSACTION'
            elif 'CONTACTLESS' in sample_text:
                return 'CONTACTLESS_TRANSACTION'
            else:
                return 'UNKNOWN_PATTERN'
                
        except Exception as e:
            logger.warning(f"Error determining pattern signature: {e}")
            return 'ERROR_PATTERN'
    
    def _calculate_metrics(self):
        """Calculate clustering and anomaly detection performance metrics"""
        try:
            cluster_labels = self.results['clustering']['labels']
            
            # Only calculate metrics if we have valid clusters
            valid_labels = cluster_labels[cluster_labels != -1]
            valid_embeddings = self.embeddings_normalized[cluster_labels != -1]
            
            metrics = {}
            
            if len(valid_labels) > 1 and len(set(valid_labels)) > 1:
                # Clustering metrics
                try:
                    metrics['silhouette_score'] = silhouette_score(valid_embeddings, valid_labels)
                except:
                    metrics['silhouette_score'] = 0.0
                    
                try:
                    metrics['calinski_harabasz_score'] = calinski_harabasz_score(valid_embeddings, valid_labels)
                except:
                    metrics['calinski_harabasz_score'] = 0.0
                    
                try:
                    metrics['davies_bouldin_score'] = davies_bouldin_score(valid_embeddings, valid_labels)
                except:
                    metrics['davies_bouldin_score'] = float('inf')
            
            # Anomaly detection agreement
            iso_anomalies = self.results['anomaly_detection']['isolation_forest']['anomalies']
            lof_anomalies = self.results['anomaly_detection']['lof']['anomalies']
            
            metrics['anomaly_agreement'] = np.mean(iso_anomalies == lof_anomalies)
            metrics['total_sequences'] = len(self.sequences)
            metrics['embedding_dimension'] = self.embeddings.shape[1]
            
            # Clustering quality indicators
            metrics['n_clusters'] = self.results['clustering']['n_clusters']
            metrics['noise_ratio'] = self.results['clustering']['noise_ratio']
            
            self.results['metrics'] = metrics
            
            logger.info(f"Metrics calculated - Silhouette: {metrics.get('silhouette_score', 'N/A')}")
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            self.results['metrics'] = {'error': str(e)}
    
    def get_anomalous_sequences(self, method: str = 'consensus') -> List[Tuple[int, str]]:
        """Get sequences identified as anomalous"""
        if method == 'consensus':
            anomaly_mask = self.results['anomaly_detection']['consensus']['anomalies']
        elif method == 'isolation_forest':
            anomaly_mask = self.results['anomaly_detection']['isolation_forest']['anomalies']
        elif method == 'lof':
            anomaly_mask = self.results['anomaly_detection']['lof']['anomalies']
        elif method == 'statistical':
            anomaly_mask = self.results['anomaly_detection']['statistical']['anomalies']
        else:
            raise ValueError(f"Unknown method: {method}")
        
        anomalous = [(i, self.sequences[i]) for i in np.where(anomaly_mask)[0]]
        return anomalous
    
    def get_cluster_sequences(self, cluster_id: int) -> List[Tuple[int, str]]:
        """Get sequences belonging to a specific cluster"""
        cluster_mask = self.results['clustering']['labels'] == cluster_id
        sequences = [(i, self.sequences[i]) for i in np.where(cluster_mask)[0]]
        return sequences
    
    def get_analysis_summary(self) -> Dict:
        """Get a comprehensive summary of the analysis"""
        if not self.results:
            return {'error': 'No analysis results available'}
        
        summary = {
            'total_sequences': len(self.sequences),
            'clusters_found': self.results['clustering']['n_clusters'],
            'noise_points': self.results['clustering']['n_noise'],
            'anomaly_rates': {
                'isolation_forest': self.results['anomaly_detection']['isolation_forest']['anomaly_rate'],
                'lof': self.results['anomaly_detection']['lof']['anomaly_rate'],
                'statistical': self.results['anomaly_detection']['statistical']['anomaly_rate'],
                'consensus': self.results['anomaly_detection']['consensus']['anomaly_rate']
            },
            'top_patterns': []
        }
        
        # Add top patterns
        if 'patterns' in self.results and not self.results['patterns'].empty:
            for _, pattern in self.results['patterns'].head(5).iterrows():
                summary['top_patterns'].append({
                    'pattern_type': pattern['pattern_signature'],
                    'size': pattern['size'],
                    'percentage': pattern['percentage'],
                    'anomaly_ratio': pattern['anomaly_ratio']
                })
        
        return summary
