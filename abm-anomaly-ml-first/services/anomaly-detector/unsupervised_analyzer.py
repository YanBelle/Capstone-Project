"""
Enhanced Unsupervised EJ Log Analyzer
Integrates with existing ML-first anomaly detection system to provide more robust,
less rule-based anomaly detection using multiple unsupervised techniques.
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
from typing import List, Dict, Tuple, Optional, Any
import warnings
import logging
from collections import Counter

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class EnhancedUnsupervisedEJAnalyzer:
    """
    Enhanced unsupervised EJ log analyzer that integrates with existing system
    Provides more robust anomaly detection with less reliance on rules
    """
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the enhanced analyzer
        
        Args:
            embedding_model: Name of the sentence transformer model to use
        """
        logger.info(f"Initializing Enhanced Unsupervised Analyzer with model: {embedding_model}")
        
        # Initialize embedder
        try:
            self.embedder = SentenceTransformer(embedding_model)
        except Exception as e:
            logger.warning(f"Failed to load {embedding_model}, falling back to simple embeddings")
            self.embedder = None
        
        # Initialize unsupervised models with optimized parameters for EJ logs
        self.isolation_forest = IsolationForest(
            contamination='auto',
            random_state=42,
            n_jobs=-1,
            max_samples=0.8,  # Use 80% of samples for training each tree
            n_estimators=200   # More trees for better performance
        )
        
        self.hdbscan = hdbscan.HDBSCAN(
            min_cluster_size=3,     # Smaller clusters for transaction patterns
            min_samples=2,          # More sensitive to local structure
            metric='euclidean',
            cluster_selection_method='eom',
            alpha=1.0
        )
        
        self.lof = LocalOutlierFactor(
            n_neighbors=10,         # Reduced for transaction data
            contamination='auto',
            novelty=False,
            algorithm='auto'
        )
        
        # Additional models for ensemble approach
        self.ensemble_models = {
            'isolation_forest_conservative': IsolationForest(
                contamination=0.05,  # More conservative
                random_state=42,
                n_jobs=-1
            ),
            'isolation_forest_sensitive': IsolationForest(
                contamination=0.15,  # More sensitive
                random_state=42,
                n_jobs=-1
            )
        }
        
        # Storage for analysis results
        self.embeddings = None
        self.embeddings_normalized = None
        self.sequences = None
        self.results = {}
        self.scaler = StandardScaler()
        
    def analyze_sessions(self, sessions: List[Any], 
                        perform_dim_reduction: bool = True,
                        ensemble_voting: bool = True) -> Dict:
        """
        Perform enhanced unsupervised analysis on EJ sessions
        
        Args:
            sessions: List of TransactionSession objects or preprocessed sequences
            perform_dim_reduction: Whether to compute UMAP/PCA projections
            ensemble_voting: Whether to use ensemble voting for final decisions
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        logger.info(f"Starting enhanced unsupervised analysis on {len(sessions)} sessions")
        
        # Convert sessions to text sequences if needed
        if hasattr(sessions[0], 'raw_text'):
            self.sequences = [session.raw_text for session in sessions]
            self.session_objects = sessions
        else:
            self.sequences = sessions
            self.session_objects = None
        
        # Step 1: Create embeddings
        logger.info("Step 1: Creating advanced sequence embeddings...")
        self.embeddings = self._create_embeddings(self.sequences)
        
        # Normalize embeddings
        self.embeddings_normalized = self.scaler.fit_transform(self.embeddings)
        
        # Step 2: Multi-method anomaly detection
        logger.info("Step 2: Running ensemble anomaly detection...")
        self._detect_anomalies_ensemble(ensemble_voting)
        
        # Step 3: Advanced clustering analysis
        logger.info("Step 3: Performing hierarchical clustering analysis...")
        self._perform_advanced_clustering()
        
        # Step 4: Pattern discovery and characterization
        logger.info("Step 4: Discovering and characterizing patterns...")
        self._discover_patterns()
        
        # Step 5: Dimensionality reduction for insights
        if perform_dim_reduction:
            logger.info("Step 5: Computing dimensionality reductions...")
            self._compute_projections()
        
        # Step 6: Advanced metrics and validation
        logger.info("Step 6: Calculating advanced performance metrics...")
        self._calculate_advanced_metrics()
        
        # Step 7: Generate actionable insights
        logger.info("Step 7: Generating actionable insights...")
        self._generate_insights()
        
        logger.info("Enhanced unsupervised analysis complete")
        return self.results
    
    def _create_embeddings(self, sequences: List[str]) -> np.ndarray:
        """Create embeddings using sentence transformers or fallback methods"""
        if self.embedder:
            try:
                embeddings = self.embedder.encode(
                    sequences,
                    show_progress_bar=True,
                    batch_size=32,
                    normalize_embeddings=True
                )
                logger.info(f"Created sentence transformer embeddings: {embeddings.shape}")
                return embeddings
            except Exception as e:
                logger.warning(f"Sentence transformer failed: {e}, using TF-IDF fallback")
        
        # Fallback to TF-IDF
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8
        )
        embeddings = vectorizer.fit_transform(sequences).toarray()
        logger.info(f"Created TF-IDF embeddings: {embeddings.shape}")
        return embeddings
    
    def _detect_anomalies_ensemble(self, use_voting: bool = True):
        """Run ensemble anomaly detection with multiple methods"""
        
        # Primary anomaly detection methods
        iso_predictions = self.isolation_forest.fit_predict(self.embeddings_normalized)
        iso_scores = self.isolation_forest.score_samples(self.embeddings_normalized)
        
        lof_predictions = self.lof.fit_predict(self.embeddings_normalized)
        lof_scores = self.lof.negative_outlier_factor_
        
        # Additional ensemble models
        ensemble_predictions = {}
        ensemble_scores = {}
        
        for name, model in self.ensemble_models.items():
            try:
                predictions = model.fit_predict(self.embeddings_normalized)
                scores = model.score_samples(self.embeddings_normalized)
                ensemble_predictions[name] = predictions
                ensemble_scores[name] = scores
            except Exception as e:
                logger.warning(f"Failed to run {name}: {e}")
        
        # Statistical outliers based on embedding distances
        centroid = np.mean(self.embeddings_normalized, axis=0)
        distances = np.linalg.norm(self.embeddings_normalized - centroid, axis=1)
        
        # Dynamic threshold based on data distribution
        q75, q25 = np.percentile(distances, [75, 25])
        iqr = q75 - q25
        statistical_threshold = q75 + 1.5 * iqr  # IQR-based outlier detection
        statistical_anomalies = distances > statistical_threshold
        
        # Density-based outliers
        density_scores = self._calculate_density_scores()
        density_threshold = np.percentile(density_scores, 5)
        density_anomalies = density_scores < density_threshold
        
        # Ensemble voting if enabled
        if use_voting and ensemble_predictions:
            voting_results = self._ensemble_voting([
                iso_predictions,
                lof_predictions,
                *ensemble_predictions.values()
            ])
        else:
            voting_results = iso_predictions
        
        # Consensus anomalies (detected by multiple methods)
        consensus_anomalies = (
            (iso_predictions == -1) & 
            (lof_predictions == -1) &
            statistical_anomalies
        )
        
        # Store comprehensive results
        self.results['anomaly_detection'] = {
            'isolation_forest': {
                'predictions': iso_predictions,
                'scores': iso_scores,
                'n_anomalies': np.sum(iso_predictions == -1),
                'anomaly_rate': np.mean(iso_predictions == -1)
            },
            'lof': {
                'predictions': lof_predictions,
                'scores': lof_scores,
                'n_anomalies': np.sum(lof_predictions == -1),
                'anomaly_rate': np.mean(lof_predictions == -1)
            },
            'statistical': {
                'anomalies': statistical_anomalies,
                'distances': distances,
                'threshold': statistical_threshold,
                'n_anomalies': np.sum(statistical_anomalies),
                'anomaly_rate': np.mean(statistical_anomalies)
            },
            'density': {
                'anomalies': density_anomalies,
                'scores': density_scores,
                'threshold': density_threshold,
                'n_anomalies': np.sum(density_anomalies),
                'anomaly_rate': np.mean(density_anomalies)
            },
            'consensus': {
                'anomalies': consensus_anomalies,
                'n_anomalies': np.sum(consensus_anomalies),
                'anomaly_rate': np.mean(consensus_anomalies)
            },
            'ensemble': {
                'predictions': ensemble_predictions,
                'scores': ensemble_scores,
                'voting_results': voting_results if use_voting else None
            }
        }
    
    def _calculate_density_scores(self) -> np.ndarray:
        """Calculate local density scores for each point"""
        from sklearn.neighbors import NearestNeighbors
        
        nbrs = NearestNeighbors(n_neighbors=10).fit(self.embeddings_normalized)
        distances, indices = nbrs.kneighbors(self.embeddings_normalized)
        
        # Calculate local density as inverse of mean distance to k-nearest neighbors
        density_scores = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-8)
        return density_scores
    
    def _ensemble_voting(self, predictions_list: List[np.ndarray]) -> np.ndarray:
        """Perform ensemble voting across multiple anomaly detection methods"""
        # Convert predictions to binary (normal=0, anomaly=1)
        binary_predictions = []
        for pred in predictions_list:
            binary_pred = (pred == -1).astype(int)
            binary_predictions.append(binary_pred)
        
        # Stack predictions and use majority voting
        stacked = np.stack(binary_predictions, axis=1)
        majority_vote = np.mean(stacked, axis=1)
        
        # Use threshold for final decision (can be tuned)
        threshold = 0.5  # Majority rule
        final_predictions = np.where(majority_vote >= threshold, -1, 1)
        
        return final_predictions
    
    def _perform_advanced_clustering(self):
        """Perform hierarchical clustering with multiple resolution levels"""
        
        # Primary HDBSCAN clustering
        cluster_labels = self.hdbscan.fit_predict(self.embeddings_normalized)
        probabilities = self.hdbscan.probabilities_
        
        # Multi-resolution clustering (different parameters)
        cluster_variants = {}
        
        # Fine-grained clustering
        hdbscan_fine = hdbscan.HDBSCAN(
            min_cluster_size=2,
            min_samples=1,
            cluster_selection_method='eom'
        )
        cluster_variants['fine'] = hdbscan_fine.fit_predict(self.embeddings_normalized)
        
        # Coarse clustering
        hdbscan_coarse = hdbscan.HDBSCAN(
            min_cluster_size=10,
            min_samples=5,
            cluster_selection_method='eom'
        )
        cluster_variants['coarse'] = hdbscan_coarse.fit_predict(self.embeddings_normalized)
        
        # Find cluster exemplars
        exemplars = self._find_cluster_exemplars(cluster_labels, probabilities)
        
        # Calculate cluster stability and quality metrics
        cluster_stability = self._calculate_cluster_stability(cluster_labels)
        
        self.results['clustering'] = {
            'labels': cluster_labels,
            'probabilities': probabilities,
            'variants': cluster_variants,
            'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
            'n_noise': np.sum(cluster_labels == -1),
            'noise_ratio': np.mean(cluster_labels == -1),
            'exemplars': exemplars,
            'stability': cluster_stability,
            'cluster_sizes': pd.Series(cluster_labels).value_counts().to_dict()
        }
    
    def _find_cluster_exemplars(self, cluster_labels: np.ndarray, probabilities: np.ndarray) -> Dict[int, int]:
        """Find most representative points (exemplars) for each cluster"""
        exemplars = {}
        
        for cluster_id in set(cluster_labels):
            if cluster_id != -1:  # Skip noise points
                cluster_mask = cluster_labels == cluster_id
                cluster_probs = probabilities[cluster_mask]
                
                if len(cluster_probs) > 0:
                    # Find point with highest membership probability
                    exemplar_idx = np.where(cluster_mask)[0][np.argmax(cluster_probs)]
                    exemplars[cluster_id] = exemplar_idx
        
        return exemplars
    
    def _calculate_cluster_stability(self, cluster_labels: np.ndarray) -> Dict[str, float]:
        """Calculate various cluster stability metrics"""
        stability_metrics = {}
        
        # Cluster persistence across different subsamples
        n_samples = len(cluster_labels)
        n_trials = 10
        stability_scores = []
        
        for _ in range(n_trials):
            # Subsample data
            subsample_idx = np.random.choice(n_samples, size=int(0.8 * n_samples), replace=False)
            subsample_embeddings = self.embeddings_normalized[subsample_idx]
            
            # Re-cluster subsample
            hdbscan_temp = hdbscan.HDBSCAN(
                min_cluster_size=self.hdbscan.min_cluster_size,
                min_samples=self.hdbscan.min_samples
            )
            subsample_labels = hdbscan_temp.fit_predict(subsample_embeddings)
            
            # Calculate adjusted rand index with original clustering
            from sklearn.metrics import adjusted_rand_score
            original_subsample = cluster_labels[subsample_idx]
            ari_score = adjusted_rand_score(original_subsample, subsample_labels)
            stability_scores.append(ari_score)
        
        stability_metrics['mean_ari'] = np.mean(stability_scores)
        stability_metrics['std_ari'] = np.std(stability_scores)
        
        return stability_metrics
    
    def _discover_patterns(self):
        """Advanced pattern discovery and characterization"""
        patterns = []
        cluster_labels = self.results['clustering']['labels']
        unique_clusters = [c for c in set(cluster_labels) if c != -1]
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_sequences = [self.sequences[i] for i, mask in enumerate(cluster_mask) if mask]
            
            # Advanced pattern analysis
            pattern_info = self._analyze_cluster_pattern(cluster_id, cluster_sequences, cluster_mask)
            patterns.append(pattern_info)
        
        # Analyze noise/outlier patterns
        noise_mask = cluster_labels == -1
        if np.any(noise_mask):
            noise_sequences = [self.sequences[i] for i, mask in enumerate(noise_mask) if mask]
            noise_info = self._analyze_noise_patterns(noise_sequences, noise_mask)
            patterns.append(noise_info)
        
        # Sort patterns by significance
        patterns_df = pd.DataFrame(patterns)
        if not patterns_df.empty:
            patterns_df = patterns_df.sort_values(['anomaly_severity', 'size'], ascending=[False, False])
        
        self.results['patterns'] = patterns_df
    
    def _analyze_cluster_pattern(self, cluster_id: int, sequences: List[str], mask: np.ndarray) -> Dict:
        """Analyze patterns within a specific cluster"""
        
        # Calculate anomaly scores for this cluster
        iso_scores = self.results['anomaly_detection']['isolation_forest']['scores'][mask]
        lof_scores = self.results['anomaly_detection']['lof']['scores'][mask]
        
        # Pattern characterization
        common_tokens = self._extract_common_tokens(sequences)
        pattern_signature = self._infer_pattern_type(sequences)
        sequence_similarity = self._calculate_sequence_similarity(sequences)
        
        # Anomaly assessment
        anomaly_indicators = self._assess_anomaly_indicators(sequences)
        
        return {
            'cluster_id': cluster_id,
            'type': 'cluster',
            'size': len(sequences),
            'avg_isolation_score': np.mean(iso_scores),
            'avg_lof_score': np.mean(lof_scores),
            'anomaly_severity': self._calculate_anomaly_severity(iso_scores, lof_scores),
            'common_tokens': common_tokens,
            'pattern_signature': pattern_signature,
            'sequence_similarity': sequence_similarity,
            'anomaly_indicators': anomaly_indicators,
            'sample_sequences': sequences[:3] if sequences else []
        }
    
    def _analyze_noise_patterns(self, sequences: List[str], mask: np.ndarray) -> Dict:
        """Analyze patterns in noise/outlier sequences"""
        
        iso_scores = self.results['anomaly_detection']['isolation_forest']['scores'][mask]
        lof_scores = self.results['anomaly_detection']['lof']['scores'][mask]
        
        return {
            'cluster_id': -1,
            'type': 'noise/outliers',
            'size': len(sequences),
            'avg_isolation_score': np.mean(iso_scores),
            'avg_lof_score': np.mean(lof_scores),
            'anomaly_severity': self._calculate_anomaly_severity(iso_scores, lof_scores),
            'common_tokens': self._extract_common_tokens(sequences[:20]),  # Sample
            'pattern_signature': 'OUTLIERS/ANOMALIES',
            'sequence_similarity': 'LOW',
            'anomaly_indicators': self._assess_anomaly_indicators(sequences[:20]),
            'sample_sequences': sequences[:5] if sequences else []
        }
    
    def _extract_common_tokens(self, sequences: List[str], top_n: int = 10) -> List[Tuple[str, float]]:
        """Extract most common tokens with better preprocessing"""
        if not sequences:
            return []
        
        # Tokenize and clean
        all_tokens = []
        for seq in sequences[:50]:  # Sample for efficiency
            # Simple tokenization and cleaning
            tokens = seq.replace('\n', ' ').replace('\r', ' ').split()
            # Filter out very short tokens and common noise
            clean_tokens = [t for t in tokens if len(t) > 2 and not t.isdigit()]
            all_tokens.extend(clean_tokens)
        
        if not all_tokens:
            return []
        
        token_counts = Counter(all_tokens)
        total_tokens = len(all_tokens)
        
        return [(token, count/total_tokens) for token, count in token_counts.most_common(top_n)]
    
    def _infer_pattern_type(self, sequences: List[str]) -> str:
        """Infer transaction pattern type from sequences"""
        if not sequences:
            return 'UNKNOWN'
        
        sample_text = ' '.join(sequences[:10]).upper()
        
        # Enhanced pattern detection
        patterns = {
            'SUCCESSFUL_WITHDRAWAL': ['NOTES PRESENTED', 'NOTES TAKEN', 'CASH DISPENSED'],
            'SUCCESSFUL_INQUIRY': ['BALANCE INQUIRY', 'RECEIPT PRINTED'],
            'CARD_ERROR': ['CARD ERROR', 'INVALID CARD', 'CARD CAPTURE'],
            'DEVICE_ERROR': ['DEVICE ERROR', 'HARDWARE ERROR', 'UNABLE TO DISPENSE'],
            'TIMEOUT_ERROR': ['TIMEOUT', 'CUSTOMER TIMEOUT'],
            'AUTHENTICATION_FAILURE': ['PIN INVALID', 'AUTH FAILURE', 'DECLINED'],
            'INCOMPLETE_TRANSACTION': ['TRANSACTION START', 'CARD TAKEN'],
            'NETWORK_ERROR': ['HOST UNREACHABLE', 'CONNECTION ERROR'],
            'CASH_HANDLING_ERROR': ['CASH RETRACT', 'DISPENSER ERROR']
        }
        
        for pattern_name, keywords in patterns.items():
            if any(keyword in sample_text for keyword in keywords):
                return pattern_name
        
        return 'UNKNOWN_PATTERN'
    
    def _calculate_sequence_similarity(self, sequences: List[str]) -> str:
        """Calculate similarity within sequences"""
        if len(sequences) < 2:
            return 'N/A'
        
        # Simple similarity based on token overlap
        similarities = []
        sample_sequences = sequences[:10]  # Sample for efficiency
        
        for i in range(len(sample_sequences)):
            for j in range(i+1, len(sample_sequences)):
                seq1_tokens = set(sample_sequences[i].split())
                seq2_tokens = set(sample_sequences[j].split())
                
                if len(seq1_tokens.union(seq2_tokens)) == 0:
                    similarity = 0
                else:
                    similarity = len(seq1_tokens.intersection(seq2_tokens)) / len(seq1_tokens.union(seq2_tokens))
                similarities.append(similarity)
        
        if not similarities:
            return 'N/A'
        
        avg_similarity = np.mean(similarities)
        
        if avg_similarity > 0.7:
            return 'HIGH'
        elif avg_similarity > 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _assess_anomaly_indicators(self, sequences: List[str]) -> List[str]:
        """Assess various anomaly indicators in sequences"""
        indicators = []
        sample_text = ' '.join(sequences[:10]).upper()
        
        # Check for various anomaly indicators
        anomaly_checks = {
            'ERROR_MESSAGES': ['ERROR', 'FAILED', 'UNABLE'],
            'INCOMPLETE_FLOW': ['CARD INSERTED', 'CARD TAKEN'],
            'TIMEOUT_ISSUES': ['TIMEOUT', 'NO RESPONSE'],
            'HARDWARE_PROBLEMS': ['DEVICE', 'HARDWARE', 'MALFUNCTION'],
            'SECURITY_ISSUES': ['INVALID', 'DECLINED', 'CAPTURE'],
            'CASH_PROBLEMS': ['DISPENSER', 'RETRACT', 'JAM']
        }
        
        for indicator_type, keywords in anomaly_checks.items():
            if any(keyword in sample_text for keyword in keywords):
                indicators.append(indicator_type)
        
        return indicators
    
    def _calculate_anomaly_severity(self, iso_scores: np.ndarray, lof_scores: np.ndarray) -> str:
        """Calculate overall anomaly severity for a group"""
        # Normalize scores
        iso_norm = np.mean(iso_scores)
        lof_norm = np.mean(lof_scores)
        
        # Combine scores (lower isolation scores = more anomalous)
        combined_score = -iso_norm + abs(lof_norm)  # Higher = more anomalous
        
        if combined_score > 2.0:
            return 'CRITICAL'
        elif combined_score > 1.0:
            return 'HIGH'
        elif combined_score > 0.5:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _compute_projections(self):
        """Compute dimensionality reductions for visualization"""
        # PCA projection
        pca = PCA(n_components=2, random_state=42)
        pca_projection = pca.fit_transform(self.embeddings_normalized)
        
        # UMAP projection with optimized parameters
        umap_model = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, len(self.sequences)//4),
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        umap_projection = umap_model.fit_transform(self.embeddings_normalized)
        
        # t-SNE projection for comparison
        try:
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.sequences)//4))
            tsne_projection = tsne.fit_transform(self.embeddings_normalized)
        except Exception as e:
            logger.warning(f"t-SNE failed: {e}")
            tsne_projection = None
        
        self.results['projections'] = {
            'pca': {
                'coordinates': pca_projection,
                'explained_variance': pca.explained_variance_ratio_,
                'total_variance_explained': np.sum(pca.explained_variance_ratio_)
            },
            'umap': {
                'coordinates': umap_projection
            },
            'tsne': {
                'coordinates': tsne_projection
            } if tsne_projection is not None else None
        }
    
    def _calculate_advanced_metrics(self):
        """Calculate comprehensive performance metrics"""
        cluster_labels = self.results['clustering']['labels']
        metrics = {}
        
        # Clustering quality metrics
        valid_mask = cluster_labels != -1
        if np.sum(valid_mask) > 1:
            valid_labels = cluster_labels[valid_mask]
            valid_embeddings = self.embeddings_normalized[valid_mask]
            
            if len(set(valid_labels)) > 1:
                metrics['silhouette_score'] = silhouette_score(valid_embeddings, valid_labels)
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(valid_embeddings, valid_labels)
                metrics['davies_bouldin_score'] = davies_bouldin_score(valid_embeddings, valid_labels)
        
        # Anomaly detection consistency
        anomaly_methods = ['isolation_forest', 'lof', 'statistical', 'density']
        method_agreements = {}
        
        for i, method1 in enumerate(anomaly_methods):
            for method2 in anomaly_methods[i+1:]:
                pred1 = self.results['anomaly_detection'][method1]['predictions'] == -1 if 'predictions' in self.results['anomaly_detection'][method1] else self.results['anomaly_detection'][method1]['anomalies']
                pred2 = self.results['anomaly_detection'][method2]['predictions'] == -1 if 'predictions' in self.results['anomaly_detection'][method2] else self.results['anomaly_detection'][method2]['anomalies']
                
                agreement = np.mean(pred1 == pred2)
                method_agreements[f'{method1}_vs_{method2}'] = agreement
        
        metrics['method_agreements'] = method_agreements
        metrics['total_sequences'] = len(self.sequences)
        metrics['clustering_efficiency'] = 1.0 - (self.results['clustering']['n_noise'] / len(self.sequences))
        
        self.results['metrics'] = metrics
    
    def _generate_insights(self):
        """Generate actionable insights from the analysis"""
        insights = {
            'summary': {},
            'alerts': [],
            'recommendations': [],
            'pattern_insights': []
        }
        
        # Summary statistics
        total_sequences = len(self.sequences)
        consensus_anomalies = self.results['anomaly_detection']['consensus']['n_anomalies']
        anomaly_rate = consensus_anomalies / total_sequences
        
        insights['summary'] = {
            'total_transactions': total_sequences,
            'consensus_anomalies': consensus_anomalies,
            'anomaly_rate': f"{anomaly_rate:.2%}",
            'clusters_found': self.results['clustering']['n_clusters'],
            'outlier_transactions': self.results['clustering']['n_noise']
        }
        
        # Generate alerts based on thresholds
        if anomaly_rate > 0.15:  # More than 15% anomalies
            insights['alerts'].append({
                'level': 'HIGH',
                'type': 'ANOMALY_RATE',
                'message': f"High anomaly rate detected: {anomaly_rate:.1%} of transactions flagged",
                'recommendation': 'Investigate system stability and transaction processing'
            })
        
        if self.results['clustering']['noise_ratio'] > 0.25:  # More than 25% noise
            insights['alerts'].append({
                'level': 'MEDIUM',
                'type': 'HIGH_NOISE',
                'message': f"High proportion of unclustered transactions: {self.results['clustering']['noise_ratio']:.1%}",
                'recommendation': 'Review transaction patterns and clustering parameters'
            })
        
        # Pattern-based insights
        if not self.results['patterns'].empty:
            patterns_df = self.results['patterns']
            
            # Find high-severity patterns
            critical_patterns = patterns_df[patterns_df['anomaly_severity'] == 'CRITICAL']
            if not critical_patterns.empty:
                for _, pattern in critical_patterns.iterrows():
                    insights['alerts'].append({
                        'level': 'CRITICAL',
                        'type': 'PATTERN_ANOMALY',
                        'message': f"Critical anomaly pattern detected: {pattern['pattern_signature']} ({pattern['size']} transactions)",
                        'recommendation': f"Immediate investigation required for {pattern['pattern_signature']} pattern"
                    })
            
            # Pattern insights
            for _, pattern in patterns_df.head(5).iterrows():
                if pattern['size'] > 1:  # Only meaningful patterns
                    insights['pattern_insights'].append({
                        'pattern': pattern['pattern_signature'],
                        'size': pattern['size'],
                        'severity': pattern['anomaly_severity'],
                        'description': f"{pattern['size']} transactions showing {pattern['pattern_signature']} pattern with {pattern['anomaly_severity']} anomaly severity"
                    })
        
        # Method agreement insights
        if 'metrics' in self.results and 'method_agreements' in self.results['metrics']:
            agreements = self.results['metrics']['method_agreements']
            avg_agreement = np.mean(list(agreements.values()))
            
            if avg_agreement < 0.7:
                insights['alerts'].append({
                    'level': 'MEDIUM',
                    'type': 'METHOD_DISAGREEMENT',
                    'message': f"Low agreement between anomaly detection methods: {avg_agreement:.1%}",
                    'recommendation': 'Consider tuning detection parameters or investigating data quality'
                })
        
        # Recommendations
        insights['recommendations'].extend([
            'Focus investigation on consensus anomalies detected by multiple methods',
            'Review patterns with CRITICAL or HIGH severity ratings',
            'Monitor trends in anomaly rates over time',
            'Validate detected patterns with domain experts'
        ])
        
        self.results['insights'] = insights
    
    def get_anomalous_sessions_enhanced(self, method: str = 'consensus', 
                                      include_scores: bool = True) -> List[Dict]:
        """Get enhanced anomaly information for sessions"""
        
        if method == 'consensus':
            anomaly_mask = self.results['anomaly_detection']['consensus']['anomalies']
        elif method in self.results['anomaly_detection']:
            if 'predictions' in self.results['anomaly_detection'][method]:
                anomaly_mask = self.results['anomaly_detection'][method]['predictions'] == -1
            else:
                anomaly_mask = self.results['anomaly_detection'][method]['anomalies']
        else:
            raise ValueError(f"Unknown method: {method}")
        
        anomalous_sessions = []
        anomaly_indices = np.where(anomaly_mask)[0]
        
        for idx in anomaly_indices:
            session_info = {
                'index': idx,
                'sequence': self.sequences[idx],
                'cluster_id': self.results['clustering']['labels'][idx],
                'cluster_probability': self.results['clustering']['probabilities'][idx]
            }
            
            if include_scores:
                session_info.update({
                    'isolation_score': self.results['anomaly_detection']['isolation_forest']['scores'][idx],
                    'lof_score': self.results['anomaly_detection']['lof']['scores'][idx],
                    'distance_score': self.results['anomaly_detection']['statistical']['distances'][idx]
                })
            
            # Add pattern information if available
            cluster_id = session_info['cluster_id']
            if not self.results['patterns'].empty:
                pattern_info = self.results['patterns'][self.results['patterns']['cluster_id'] == cluster_id]
                if not pattern_info.empty:
                    session_info['pattern_signature'] = pattern_info.iloc[0]['pattern_signature']
                    session_info['anomaly_severity'] = pattern_info.iloc[0]['anomaly_severity']
            
            anomalous_sessions.append(session_info)
        
        return anomalous_sessions
    
    def integrate_with_existing_system(self, existing_sessions: List[Any]) -> Dict[str, Any]:
        """
        Integration method to enhance existing anomaly detection results
        
        Args:
            existing_sessions: List of TransactionSession objects from existing system
            
        Returns:
            Enhanced anomaly information for integration
        """
        logger.info("Integrating unsupervised analysis with existing system")
        
        # Run analysis on existing sessions
        analysis_results = self.analyze_sessions(existing_sessions)
        
        # Create integration mapping
        integration_data = []
        
        for i, session in enumerate(existing_sessions):
            # Get unsupervised anomaly assessment
            unsupervised_anomaly = self.results['anomaly_detection']['consensus']['anomalies'][i]
            cluster_id = self.results['clustering']['labels'][i]
            
            # Get pattern information
            pattern_info = None
            if not self.results['patterns'].empty:
                pattern_matches = self.results['patterns'][self.results['patterns']['cluster_id'] == cluster_id]
                if not pattern_matches.empty:
                    pattern_info = pattern_matches.iloc[0].to_dict()
            
            integration_entry = {
                'session_id': getattr(session, 'session_id', f'session_{i}'),
                'unsupervised_anomaly': unsupervised_anomaly,
                'unsupervised_cluster': cluster_id,
                'unsupervised_pattern': pattern_info['pattern_signature'] if pattern_info else 'UNKNOWN',
                'anomaly_severity': pattern_info['anomaly_severity'] if pattern_info else 'LOW',
                'isolation_score': self.results['anomaly_detection']['isolation_forest']['scores'][i],
                'lof_score': self.results['anomaly_detection']['lof']['scores'][i],
                'cluster_probability': self.results['clustering']['probabilities'][i]
            }
            
            integration_data.append(integration_entry)
        
        return {
            'integration_data': integration_data,
            'analysis_results': analysis_results,
            'enhancement_summary': {
                'total_sessions_analyzed': len(existing_sessions),
                'unsupervised_anomalies_found': np.sum(self.results['anomaly_detection']['consensus']['anomalies']),
                'patterns_discovered': len(self.results['patterns']) if not self.results['patterns'].empty else 0,
                'clusters_identified': self.results['clustering']['n_clusters']
            }
        }
