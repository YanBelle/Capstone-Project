# EJ Log Unsupervised Analysis Implementation Guide

## Overview
This guide instructs how to integrate a completely unsupervised anomaly detection system into the existing EJ log analysis codebase. The solution requires NO labeled data and automatically discovers patterns and anomalies in ATM transaction logs.

## Implementation Instructions

### Step 1: Update Dependencies
Add these packages to your `requirements.txt`:
```txt
sentence-transformers==2.2.2
hdbscan==0.8.33
scikit-learn==1.3.2
umap-learn==0.5.4
plotly==5.18.0
seaborn==0.13.0
pandas==2.1.3
numpy==1.24.3
matplotlib==3.8.0
```

### Step 2: Create the Core Unsupervised Analyzer
Replace or add this module to your existing codebase as `unsupervised_analyzer.py`:

```python
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
        # Initialize embedder
        self.embedder = SentenceTransformer(embedding_model)
        
        # Initialize unsupervised models
        self.isolation_forest = IsolationForest(
            contamination='auto',
            random_state=42,
            n_jobs=-1
        )
        
        self.hdbscan = hdbscan.HDBSCAN(
            min_cluster_size=5,
            min_samples=3,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        
        self.lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination='auto',
            novelty=False
        )
        
        # Storage for analysis results
        self.embeddings = None
        self.sequences = None
        self.results = {}
        
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
        self.sequences = sequences
        
        # Step 1: Create embeddings
        print("Step 1: Creating sequence embeddings...")
        self.embeddings = self.embedder.encode(
            sequences,
            show_progress_bar=True,
            batch_size=32
        )
        
        # Normalize embeddings
        scaler = StandardScaler()
        self.embeddings_normalized = scaler.fit_transform(self.embeddings)
        
        # Step 2: Anomaly detection with multiple methods
        print("\nStep 2: Running anomaly detection algorithms...")
        self._detect_anomalies()
        
        # Step 3: Clustering analysis
        print("\nStep 3: Performing clustering analysis...")
        self._perform_clustering()
        
        # Step 4: Dimensionality reduction for visualization
        if perform_dim_reduction:
            print("\nStep 4: Computing dimensionality reductions...")
            self._compute_projections()
        
        # Step 5: Pattern analysis
        print("\nStep 5: Analyzing discovered patterns...")
        self._analyze_patterns()
        
        # Step 6: Calculate performance metrics
        print("\nStep 6: Calculating performance metrics...")
        self._calculate_metrics()
        
        return self.results
    
    def _detect_anomalies(self):
        """Run multiple anomaly detection algorithms"""
        # Isolation Forest
        iso_predictions = self.isolation_forest.fit_predict(self.embeddings_normalized)
        iso_scores = self.isolation_forest.score_samples(self.embeddings_normalized)
        
        # Local Outlier Factor
        lof_predictions = self.lof.fit_predict(self.embeddings_normalized)
        lof_scores = self.lof.negative_outlier_factor_
        
        # Consensus anomalies (detected by multiple methods)
        consensus_anomalies = (iso_predictions == -1) & (lof_predictions == -1)
        
        # Statistical outliers (based on embedding distances)
        distances = np.linalg.norm(
            self.embeddings_normalized - np.mean(self.embeddings_normalized, axis=0),
            axis=1
        )
        statistical_threshold = np.percentile(distances, 95)
        statistical_anomalies = distances > statistical_threshold
        
        self.results['anomaly_detection'] = {
            'isolation_forest': {
                'predictions': iso_predictions,
                'scores': iso_scores,
                'n_anomalies': np.sum(iso_predictions == -1)
            },
            'lof': {
                'predictions': lof_predictions,
                'scores': lof_scores,
                'n_anomalies': np.sum(lof_predictions == -1)
            },
            'consensus': {
                'anomalies': consensus_anomalies,
                'n_anomalies': np.sum(consensus_anomalies)
            },
            'statistical': {
                'anomalies': statistical_anomalies,
                'distances': distances,
                'threshold': statistical_threshold,
                'n_anomalies': np.sum(statistical_anomalies)
            }
        }
    
    def _perform_clustering(self):
        """Perform clustering analysis"""
        # HDBSCAN clustering
        cluster_labels = self.hdbscan.fit_predict(self.embeddings_normalized)
        
        # Calculate cluster probabilities
        probabilities = self.hdbscan.probabilities_
        
        # Find exemplars (most representative points) for each cluster
        exemplars = {}
        for cluster_id in set(cluster_labels):
            if cluster_id != -1:  # Skip noise points
                cluster_mask = cluster_labels == cluster_id
                cluster_probs = probabilities[cluster_mask]
                if len(cluster_probs) > 0:
                    exemplar_idx = np.where(cluster_mask)[0][np.argmax(cluster_probs)]
                    exemplars[cluster_id] = exemplar_idx
        
        self.results['clustering'] = {
            'labels': cluster_labels,
            'probabilities': probabilities,
            'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
            'n_noise': np.sum(cluster_labels == -1),
            'exemplars': exemplars,
            'cluster_sizes': pd.Series(cluster_labels).value_counts().to_dict()
        }
    
    def _compute_projections(self):
        """Compute 2D projections for visualization"""
        # PCA projection
        pca = PCA(n_components=2, random_state=42)
        pca_projection = pca.fit_transform(self.embeddings_normalized)
        
        # UMAP projection
        umap_model = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        umap_projection = umap_model.fit_transform(self.embeddings_normalized)
        
        self.results['projections'] = {
            'pca': {
                'coordinates': pca_projection,
                'explained_variance': pca.explained_variance_ratio_
            },
            'umap': {
                'coordinates': umap_projection
            }
        }
    
    def _analyze_patterns(self):
        """Analyze patterns in clusters and anomalies"""
        patterns = []
        
        # Analyze each cluster
        cluster_labels = self.results['clustering']['labels']
        unique_clusters = [c for c in set(cluster_labels) if c != -1]
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_sequences = [self.sequences[i] for i, mask in enumerate(cluster_mask) if mask]
            
            pattern_info = {
                'cluster_id': cluster_id,
                'size': len(cluster_sequences),
                'avg_anomaly_score': np.mean(
                    self.results['anomaly_detection']['isolation_forest']['scores'][cluster_mask]
                ),
                'common_tokens': self._get_common_tokens(cluster_sequences),
                'pattern_signature': self._get_pattern_signature(cluster_sequences),
                'anomaly_ratio': np.mean(
                    self.results['anomaly_detection']['consensus']['anomalies'][cluster_mask]
                )
            }
            patterns.append(pattern_info)
        
        # Analyze noise points
        noise_mask = cluster_labels == -1
        if np.any(noise_mask):
            noise_sequences = [self.sequences[i] for i, mask in enumerate(noise_mask) if mask]
            noise_info = {
                'cluster_id': -1,
                'size': len(noise_sequences),
                'avg_anomaly_score': np.mean(
                    self.results['anomaly_detection']['isolation_forest']['scores'][noise_mask]
                ),
                'common_tokens': self._get_common_tokens(noise_sequences[:10]),  # Sample
                'pattern_signature': 'NOISE/OUTLIERS',
                'anomaly_ratio': np.mean(
                    self.results['anomaly_detection']['consensus']['anomalies'][noise_mask]
                )
            }
            patterns.append(noise_info)
        
        self.results['patterns'] = pd.DataFrame(patterns).sort_values('size', ascending=False)
    
    def _get_common_tokens(self, sequences: List[str], top_n: int = 5) -> List[Tuple[str, float]]:
        """Extract most common tokens from sequences"""
        from collections import Counter
        
        all_tokens = []
        for seq in sequences[:50]:  # Sample for efficiency
            all_tokens.extend(seq.split())
        
        token_counts = Counter(all_tokens)
        total_tokens = len(all_tokens)
        
        common = [(token, count/total_tokens) for token, count in token_counts.most_common(top_n)]
        return common
    
    def _get_pattern_signature(self, sequences: List[str]) -> str:
        """Infer pattern type from sequences"""
        sample_text = ' '.join(sequences[:10])
        
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
        else:
            return 'UNKNOWN_PATTERN'
    
    def _calculate_metrics(self):
        """Calculate clustering and anomaly detection performance metrics"""
        cluster_labels = self.results['clustering']['labels']
        
        # Only calculate metrics if we have valid clusters
        valid_labels = cluster_labels[cluster_labels != -1]
        valid_embeddings = self.embeddings_normalized[cluster_labels != -1]
        
        metrics = {}
        
        if len(valid_labels) > 1 and len(set(valid_labels)) > 1:
            # Clustering metrics
            metrics['silhouette_score'] = silhouette_score(valid_embeddings, valid_labels)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(valid_embeddings, valid_labels)
            metrics['davies_bouldin_score'] = davies_bouldin_score(valid_embeddings, valid_labels)
        
        # Anomaly detection agreement
        iso_anomalies = self.results['anomaly_detection']['isolation_forest']['predictions'] == -1
        lof_anomalies = self.results['anomaly_detection']['lof']['predictions'] == -1
        
        metrics['anomaly_agreement'] = np.mean(iso_anomalies == lof_anomalies)
        metrics['total_sequences'] = len(self.sequences)
        
        self.results['metrics'] = metrics
    
    def get_anomalous_sequences(self, method: str = 'consensus') -> List[Tuple[int, str]]:
        """Get sequences identified as anomalous"""
        if method == 'consensus':
            anomaly_mask = self.results['anomaly_detection']['consensus']['anomalies']
        elif method == 'isolation_forest':
            anomaly_mask = self.results['anomaly_detection']['isolation_forest']['predictions'] == -1
        elif method == 'lof':
            anomaly_mask = self.results['anomaly_detection']['lof']['predictions'] == -1
        else:
            raise ValueError(f"Unknown method: {method}")
        
        anomalous = [(i, self.sequences[i]) for i in np.where(anomaly_mask)[0]]
        return anomalous
    
    def get_cluster_sequences(self, cluster_id: int) -> List[Tuple[int, str]]:
        """Get sequences belonging to a specific cluster"""
        cluster_mask = self.results['clustering']['labels'] == cluster_id
        sequences = [(i, self.sequences[i]) for i in np.where(cluster_mask)[0]]
        return sequences

### Step 3: Create the Visualization Module
Add this as `unsupervised_visualizer.py`:

```python
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class UnsupervisedEJVisualizer:
    """
    Visualization module for unsupervised EJ analysis results
    """
    
    def __init__(self, analyzer: UnsupervisedEJAnalyzer):
        """
        Initialize visualizer with analyzer instance
        
        Args:
            analyzer: Trained UnsupervisedEJAnalyzer instance
        """
        self.analyzer = analyzer
        self.results = analyzer.results
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def create_comprehensive_dashboard(self, save_path: Optional[str] = None, 
                                     interactive: bool = True) -> None:
        """
        Create comprehensive visualization dashboard
        
        Args:
            save_path: Path to save static plot (for matplotlib)
            interactive: Whether to create interactive plotly dashboard
        """
        if interactive:
            self._create_interactive_dashboard()
        else:
            self._create_static_dashboard(save_path)
    
    def _create_interactive_dashboard(self):
        """Create interactive Plotly dashboard"""
        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'UMAP Projection with Clusters',
                'Anomaly Scores Distribution',
                'Cluster Sizes',
                'PCA Projection with Anomalies',
                'Anomaly Detection Comparison',
                'Pattern Analysis',
                'Cluster Quality Metrics',
                'Consensus Anomalies',
                'Distance Distribution'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'histogram'}, {'type': 'bar'}],
                [{'type': 'scatter'}, {'type': 'bar'}, {'type': 'table'}],
                [{'type': 'bar'}, {'type': 'scatter'}, {'type': 'histogram'}]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # 1. UMAP with clusters
        if 'projections' in self.results:
            umap_coords = self.results['projections']['umap']['coordinates']
            cluster_labels = self.results['clustering']['labels']
            
            # Create color map for clusters
            unique_labels = sorted(set(cluster_labels))
            colors = px.colors.qualitative.Plotly
            color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
            
            for label in unique_labels:
                mask = cluster_labels == label
                name = f'Cluster {label}' if label != -1 else 'Noise'
                
                fig.add_trace(
                    go.Scatter(
                        x=umap_coords[mask, 0],
                        y=umap_coords[mask, 1],
                        mode='markers',
                        name=name,
                        marker=dict(
                            size=6,
                            color=color_map[label],
                            opacity=0.7
                        ),
                        text=[f'Seq {i}' for i in np.where(mask)[0]],
                        hovertemplate='%{text}<br>UMAP1: %{x:.2f}<br>UMAP2: %{y:.2f}'
                    ),
                    row=1, col=1
                )
        
        # 2. Anomaly scores distribution
        iso_scores = self.results['anomaly_detection']['isolation_forest']['scores']
        fig.add_trace(
            go.Histogram(
                x=iso_scores,
                nbinsx=50,
                name='Anomaly Scores',
                marker_color='lightblue'
            ),
            row=1, col=2
        )
        
        # Add threshold line
        threshold = np.percentile(iso_scores, 5)
        fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                     annotation_text="Anomaly Threshold", row=1, col=2)
        
        # 3. Cluster sizes
        cluster_sizes = pd.Series(self.results['clustering']['labels']).value_counts().sort_index()
        fig.add_trace(
            go.Bar(
                x=[f'Cluster {i}' if i != -1 else 'Noise' for i in cluster_sizes.index],
                y=cluster_sizes.values,
                name='Cluster Sizes',
                marker_color='lightgreen'
            ),
            row=1, col=3
        )
        
        # 4. PCA with anomalies
        if 'projections' in self.results:
            pca_coords = self.results['projections']['pca']['coordinates']
            anomaly_mask = self.results['anomaly_detection']['consensus']['anomalies']
            
            # Normal points
            fig.add_trace(
                go.Scatter(
                    x=pca_coords[~anomaly_mask, 0],
                    y=pca_coords[~anomaly_mask, 1],
                    mode='markers',
                    name='Normal',
                    marker=dict(size=5, color='blue', opacity=0.5)
                ),
                row=2, col=1
            )
            
            # Anomalies
            fig.add_trace(
                go.Scatter(
                    x=pca_coords[anomaly_mask, 0],
                    y=pca_coords[anomaly_mask, 1],
                    mode='markers',
                    name='Anomalies',
                    marker=dict(size=8, color='red', symbol='x')
                ),
                row=2, col=1
            )
        
        # 5. Anomaly detection comparison
        methods = ['Isolation Forest', 'LOF', 'Statistical', 'Consensus']
        anomaly_counts = [
            self.results['anomaly_detection']['isolation_forest']['n_anomalies'],
            self.results['anomaly_detection']['lof']['n_anomalies'],
            self.results['anomaly_detection']['statistical']['n_anomalies'],
            self.results['anomaly_detection']['consensus']['n_anomalies']
        ]
        
        fig.add_trace(
            go.Bar(
                x=methods,
                y=anomaly_counts,
                name='Anomaly Counts',
                marker_color=['blue', 'green', 'orange', 'red']
            ),
            row=2, col=2
        )
        
        # 6. Pattern analysis table
        patterns_df = self.results['patterns']
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Cluster', 'Size', 'Anomaly Ratio', 'Pattern'],
                    fill_color='lightgray',
                    align='left'
                ),
                cells=dict(
                    values=[
                        patterns_df['cluster_id'],
                        patterns_df['size'],
                        patterns_df['anomaly_ratio'].round(3),
                        patterns_df['pattern_signature']
                    ],
                    fill_color='white',
                    align='left'
                )
            ),
            row=2, col=3
        )
        
        # 7. Cluster quality metrics
        if 'metrics' in self.results and 'silhouette_score' in self.results['metrics']:
            metrics = self.results['metrics']
            metric_names = ['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin']
            metric_values = [
                metrics.get('silhouette_score', 0),
                metrics.get('calinski_harabasz_score', 0) / 1000,  # Scale down
                1 / (metrics.get('davies_bouldin_score', 1) + 1)  # Invert (lower is better)
            ]
            
            fig.add_trace(
                go.Bar(
                    x=metric_names,
                    y=metric_values,
                    name='Quality Metrics',
                    marker_color='purple'
                ),
                row=3, col=1
            )
        
        # 8. Consensus anomalies scatter
        if 'projections' in self.results:
            umap_coords = self.results['projections']['umap']['coordinates']
            iso_anomalies = self.results['anomaly_detection']['isolation_forest']['predictions'] == -1
            lof_anomalies = self.results['anomaly_detection']['lof']['predictions'] == -1
            
            # Create categories
            categories = np.zeros(len(iso_anomalies))
            categories[iso_anomalies & lof_anomalies] = 3  # Both
            categories[iso_anomalies & ~lof_anomalies] = 1  # ISO only
            categories[~iso_anomalies & lof_anomalies] = 2  # LOF only
            
            category_names = ['Normal', 'ISO Only', 'LOF Only', 'Both']
            colors = ['blue', 'orange', 'green', 'red']
            
            for cat_id, (cat_name, color) in enumerate(zip(category_names, colors)):
                mask = categories == cat_id
                if np.any(mask):
                    fig.add_trace(
                        go.Scatter(
                            x=umap_coords[mask, 0],
                            y=umap_coords[mask, 1],
                            mode='markers',
                            name=cat_name,
                            marker=dict(size=6, color=color, opacity=0.7)
                        ),
                        row=3, col=2
                    )
        
        # 9. Distance distribution
        distances = self.results['anomaly_detection']['statistical']['distances']
        threshold = self.results['anomaly_detection']['statistical']['threshold']
        
        fig.add_trace(
            go.Histogram(
                x=distances,
                nbinsx=50,
                name='Distance Distribution',
                marker_color='lightcoral'
            ),
            row=3, col=3
        )
        
        fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                     annotation_text="95th Percentile", row=3, col=3)
        
        # Update layout
        fig.update_layout(
            title_text="Unsupervised EJ Log Analysis Dashboard",
            height=1200,
            showlegend=True,
            template='plotly_white'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="UMAP 1", row=1, col=1)
        fig.update_yaxes(title_text="UMAP 2", row=1, col=1)
        fig.update_xaxes(title_text="Anomaly Score", row=1, col=2)
        fig.update_xaxes(title_text="PCA 1", row=2, col=1)
        fig.update_yaxes(title_text="PCA 2", row=2, col=1)
        fig.update_xaxes(title_text="UMAP 1", row=3, col=2)
        fig.update_yaxes(title_text="UMAP 2", row=3, col=2)
        fig.update_xaxes(title_text="Distance from Center", row=3, col=3)
        
        fig.show()
    
    def _create_static_dashboard(self, save_path: Optional[str]):
        """Create static matplotlib dashboard"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle('Unsupervised EJ Log Analysis Dashboard', fontsize=20)
        
        # Similar plots as interactive but using matplotlib
        # ... (implementation details for matplotlib version)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_anomaly_analysis(self, sequence_idx: int):
        """
        Detailed analysis plot for a specific sequence
        
        Args:
            sequence_idx: Index of sequence to analyze
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Get sequence data
        sequence = self.analyzer.sequences[sequence_idx]
        embedding = self.analyzer.embeddings_normalized[sequence_idx]
        
        # 1. Sequence tokens
        ax = axes[0, 0]
        tokens = sequence.split()[:20]  # First 20 tokens
        ax.text(0.5, 0.5, f"Sequence {sequence_idx}:\n" + "\n".join(tokens),
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, wrap=True)
        ax.set_title('Sequence Tokens (First 20)')
        ax.axis('off')
        
        # 2. Anomaly scores
        ax = axes[0, 1]
        methods = ['Isolation Forest', 'LOF', 'Statistical']
        scores = [
            self.results['anomaly_detection']['isolation_forest']['scores'][sequence_idx],
            -self.results['anomaly_detection']['lof']['scores'][sequence_idx],  # Negative for LOF
            self.results['anomaly_detection']['statistical']['distances'][sequence_idx]
        ]
        
        bars = ax.bar(methods, scores)
        ax.set_title(f'Anomaly Scores for Sequence {sequence_idx}')
        ax.set_ylabel('Score')
        
        # Color bars based on anomaly detection
        colors = ['red' if s > np.percentile(
            self.results['anomaly_detection']['isolation_forest']['scores'], 5
        ) else 'green' for s in scores]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # 3. Nearest neighbors
        ax = axes[1, 0]
        distances = np.linalg.norm(
            self.analyzer.embeddings_normalized - embedding, axis=1
        )
        nearest_indices = np.argsort(distances)[1:6]  # Skip self
        
        neighbor_data = []
        for idx in nearest_indices:
            neighbor_data.append({
                'Index': idx,
                'Distance': distances[idx],
                'Cluster': self.results['clustering']['labels'][idx],
                'Preview': self.analyzer.sequences[idx][:30] + '...'
            })
        
        neighbor_df = pd.DataFrame(neighbor_data)
        ax.axis('off')
        table = ax.table(cellText=neighbor_df.values,
                        colLabels=neighbor_df.columns,
                        cellLoc='left',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        ax.set_title('5 Nearest Neighbors')
        
        # 4. Position in embedding space
        ax = axes[1, 1]
        if 'projections' in self.results:
            umap_coords = self.results['projections']['umap']['coordinates']
            
            # Plot all points
            ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                      c='lightgray', alpha=0.3, s=10)
            
            # Highlight this sequence
            ax.scatter(umap_coords[sequence_idx, 0], umap_coords[sequence_idx, 1],
                      c='red', s=200, marker='*', edgecolor='black', linewidth=2)
            
            # Highlight neighbors
            ax.scatter(umap_coords[nearest_indices, 0], umap_coords[nearest_indices, 1],
                      c='blue', s=100, marker='o', edgecolor='black', linewidth=1)
            
            ax.set_title('Position in UMAP Space')
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
        
        plt.tight_layout()
        plt.show()
    
    def generate_analysis_report(self) -> pd.DataFrame:
        """Generate comprehensive analysis report as DataFrame"""
        report_data = []
        
        # Overall statistics
        report_data.append({
            'Metric': 'Total Sequences',
            'Value': len(self.analyzer.sequences),
            'Category': 'Overview'
        })
        
        report_data.append({
            'Metric': 'Clusters Found',
            'Value': self.results['clustering']['n_clusters'],
            'Category': 'Clustering'
        })
        
        report_data.append({
            'Metric': 'Noise Points',
            'Value': self.results['clustering']['n_noise'],
            'Category': 'Clustering'
        })
        
        # Anomaly statistics
        for method, data in self.results['anomaly_detection'].items():
            if 'n_anomalies' in data:
                report_data.append({
                    'Metric': f'{method} Anomalies',
                    'Value': data['n_anomalies'],
                    'Category': 'Anomaly Detection'
                })
        
        # Cluster quality metrics
        if 'metrics' in self.results:
            for metric_name, value in self.results['metrics'].items():
                if isinstance(value, (int, float)):
                    report_data.append({
                        'Metric': metric_name.replace('_', ' ').title(),
                        'Value': f'{value:.3f}' if isinstance(value, float) else value,
                        'Category': 'Performance'
                    })
        
        # Pattern distribution
        patterns = self.results['patterns']
        for _, pattern in patterns.iterrows():
            report_data.append({
                'Metric': f'Cluster {pattern["cluster_id"]} Pattern',
                'Value': pattern['pattern_signature'],
                'Category': 'Patterns'
            })
        
        return pd.DataFrame(report_data)

### Step 4: Integration Module
Add this as `integrate_unsupervised.py` to connect with your existing code:

```python
from typing import List, Dict, Optional
import pandas as pd
from unsupervised_analyzer import UnsupervisedEJAnalyzer
from unsupervised_visualizer import UnsupervisedEJVisualizer

class EJUnsupervisedIntegration:
    """
    Integration module to connect unsupervised analysis with existing EJ log pipeline
    """
    
    def __init__(self, existing_pipeline=None):
        """
        Initialize integration module
        
        Args:
            existing_pipeline: Your existing EJ log processing pipeline (optional)
        """
        self.existing_pipeline = existing_pipeline
        self.analyzer = UnsupervisedEJAnalyzer()
        self.visualizer = None
        
    def process_ej_logs(self, ej_logs: List[str], 
                       preprocess: bool = True,
                       visualize: bool = True) -> Dict:
        """
        Process EJ logs through unsupervised analysis
        
        Args:
            ej_logs: Raw or preprocessed EJ log sequences
            preprocess: Whether to apply preprocessing
            visualize: Whether to generate visualizations
            
        Returns:
            Complete analysis results
        """
        # Step 1: Preprocess if needed (integrate with your existing preprocessing)
        if preprocess and self.existing_pipeline:
            sequences = [self.existing_pipeline.preprocess(log) for log in ej_logs]
        else:
            sequences = ej_logs
        
        # Step 2: Run unsupervised analysis
        print(f"Analyzing {len(sequences)} EJ log sequences...")
        results = self.analyzer.analyze_sequences(sequences)
        
        # Step 3: Create visualizations
        if visualize:
            self.visualizer = UnsupervisedEJVisualizer(self.analyzer)
            print("\nGenerating visualizations...")
            self.visualizer.create_comprehensive_dashboard(interactive=True)
        
        # Step 4: Generate insights
        insights = self._generate_insights(results)
        
        return {
            'results': results,
            'insights': insights,
            'analyzer': self.analyzer,
            'visualizer': self.visualizer
        }
    
    def _generate_insights(self, results: Dict) -> Dict:
        """Generate actionable insights from results"""
        insights = {
            'summary': {},
            'alerts': [],
            'recommendations': []
        }
        
        # Summary statistics
        total_sequences = len(self.analyzer.sequences)
        anomaly_rate = results['anomaly_detection']['consensus']['n_anomalies'] / total_sequences
        
        insights['summary'] = {
            'total_transactions': total_sequences,
            'anomaly_rate': f'{anomaly_rate:.2%}',
            'distinct_patterns': results['clustering']['n_clusters'],
            'outliers': results['clustering']['n_noise']
        }
        
        # Generate alerts
        if anomaly_rate > 0.1:  # More than 10% anomalies
            insights['alerts'].append({
                'level': 'HIGH',
                'message': f'High anomaly rate detected: {anomaly_rate:.2%}',
                'action': 'Investigate common patterns in anomalous transactions'
            })
        
        # Pattern-based alerts
        patterns_df = results['patterns']
        for _, pattern in patterns_df.iterrows():
            if pattern['anomaly_ratio'] > 0.5:
                insights['alerts'].append({
                    'level': 'MEDIUM',
                    'message': f'Cluster {pattern["cluster_id"]} has high anomaly ratio: {pattern["anomaly_ratio"]:.2%}',
                    'pattern': pattern['pattern_signature'],
                    'size': pattern['size']
                })
        
        # Recommendations
        if results['clustering']['n_noise'] > total_sequences * 0.2:
            insights['recommendations'].append(
                'High number of outliers detected. Consider investigating these unique transaction patterns.'
            )
        
        if 'DEVICE_ERROR' in patterns_df['pattern_signature'].values:
            insights['recommendations'].append(
                'Device error patterns detected. Schedule maintenance for affected ATMs.'
            )
        
        return insights
    
    def export_results(self, output_dir: str = './ej_analysis_output'):
        """Export analysis results and visualizations"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Export patterns
        self.analyzer.results['patterns'].to_csv(
            os.path.join(output_dir, 'discovered_patterns.csv'), 
            index=False
        )
        
        # Export anomalies
        anomalies = self.analyzer.get_anomalous_sequences()
        anomaly_df = pd.DataFrame(anomalies, columns=['index', 'sequence'])
        anomaly_df.to_csv(
            os.path.join(output_dir, 'anomalous_sequences.csv'), 
            index=False
        )
        
        # Export metrics
        if self.visualizer:
            report = self.visualizer.generate_analysis_report()
            report.to_csv(
                os.path.join(output_dir, 'analysis_report.csv'), 
                index=False
            )
        
        print(f"\nResults exported to {output_dir}")
    
    def compare_with_labeled_data(self, labels: Optional[List[int]] = None) -> Dict:
        """
        Optional: Compare unsupervised results with known labels if available
        
        Args:
            labels: Known labels for sequences (if available)
            
        Returns:
            Comparison metrics
        """
        if labels is None:
            return {}
        
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        
        cluster_labels = self.analyzer.results['clustering']['labels']
        
        # Calculate agreement metrics
        comparison = {
            'adjusted_rand_score': adjusted_rand_score(labels, cluster_labels),
            'normalized_mutual_info': normalized_mutual_info_score(labels, cluster_labels)
        }
        
        # Check if anomalies align with known failures
        if 'failure' in set(labels):
            failure_mask = np.array(labels) == 'failure'
            detected_anomalies = self.analyzer.results['anomaly_detection']['consensus']['anomalies']
            
            comparison['anomaly_precision'] = np.sum(
                detected_anomalies & failure_mask
            ) / np.sum(detected_anomalies)
            
            comparison['anomaly_recall'] = np.sum(
                detected_anomalies & failure_mask
            ) / np.sum(failure_mask)
        
        return comparison

### Step 5: Main Execution Script
Create `run_unsupervised_analysis.py`:

```python
import sys
from integrate_unsupervised import EJUnsupervisedIntegration

def main():
    """
    Main execution function for unsupervised EJ log analysis
    """
    # Example EJ sequences (replace with your data loading logic)
    ej_sequences = [
        "TRANSACTION_START CARD_INSERTED ATR_RECEIVED_T_0 OPCODE_FI CardNumber PIN_ENTERED OPCODE_BBC GENAC_1_ARQC GENAC_2_TC NOTES_STACKED CARD_TAKEN CardNumber NOTES_PRESENTED NOTES_TAKEN CASH_DISPENSED_SUMMARY RECEIPT_PRINTED TRANSACTION_END PRIMARY_CARD_READER_ACTIVATED",
        "TRANSACTION_START CARD_INSERTED ATR_RECEIVED_T_0 OPCODE_FI CardNumber PIN_ENTERED OPCODE_IB CardNumber CARD_TAKEN TRANSACTION_END PRIMARY_CARD_READER_ACTIVATED",
        "TRANSACTION_START CARD_INSERTED D M_81 R- CARD INITIALISE ATTEMPT ATR_RECEIVED_T_0 OPCODE_FI CardNumber PIN_ENTERED OPCODE_DAAC GENAC ARQC GENAC TC CardNumber RECEIPT_PRINTED CARD_TAKEN TRANSACTION_END PRIMARY_CARD_READER_ACTIVATED",
        # Add more sequences...
    ]
    
    # Initialize integration module
    integration = EJUnsupervisedIntegration()
    
    # Process logs
    analysis_results = integration.process_ej_logs(
        ej_sequences,
        preprocess=False,  # Already preprocessed
        visualize=True
    )
    
    # Print insights
    print("\n" + "="*50)
    print("ANALYSIS INSIGHTS")
    print("="*50)
    
    insights = analysis_results['insights']
    
    print("\nSUMMARY:")
    for key, value in insights['summary'].items():
        print(f"  {key}: {value}")
    
    print("\nALERTS:")
    for alert in insights['alerts']:
        print(f"  [{alert['level']}] {alert['message']}")
    
    print("\nRECOMMENDATIONS:")
    for rec in insights['recommendations']:
        print(f"  - {rec}")
    
    # Export results
    integration.export_results()
    
    # Optional: Analyze specific sequences
    if insights['alerts']:
        print("\n" + "="*50)
        print("DETAILED ANALYSIS OF ANOMALOUS SEQUENCE")
        print("="*50)
        
        # Get first anomalous sequence
        anomalies = integration.analyzer.get_anomalous_sequences()
        if anomalies:
            idx, seq = anomalies[0]
            print(f"\nAnalyzing sequence {idx}:")
            print(f"Preview: {seq[:100]}...")
            
            # Detailed plot
            integration.visualizer.plot_anomaly_analysis(idx)

if __name__ == "__main__":
    main()
```

## Integration Instructions for Claude/Copilot

To integrate this unsupervised analysis into your existing codebase:

1. **Install Dependencies**: Update your requirements.txt with the packages listed in Step 1

2. **Add Modules**: Create the three Python files (unsupervised_analyzer.py, unsupervised_visualizer.py, integrate_unsupervised.py) in your project

3. **Connect to Existing Pipeline**:
   ```python
   # In your existing code, replace or supplement your current analysis with:
   from integrate_unsupervised import EJUnsupervisedIntegration
   
   # Initialize with your existing pipeline
   unsupervised = EJUnsupervisedIntegration(existing_pipeline=your_pipeline)
   
   # Process logs
   results = unsupervised.process_ej_logs(your_preprocessed_sequences)
   ```

4. **Access Results**:
   - Clusters: `results['analyzer'].results['clustering']['labels']`
   - Anomalies: `results['analyzer'].get_anomalous_sequences()`
   - Patterns: `results['analyzer'].results['patterns']`
   - Visualizations: Auto-generated interactive dashboard

5. **Performance Metrics**: The system automatically calculates:
   - Silhouette Score (clustering quality)
   - Anomaly detection agreement between methods
   - Pattern discovery statistics

## Key Advantages

1. **No Labels Required**: Works with completely unlabeled data
2. **Multiple Methods**: Combines 4 different anomaly detection approaches
3. **Automatic Pattern Discovery**: Identifies and names patterns without manual rules
4. **Performance Visualization**: Interactive dashboards show model performance
5. **Easy Integration**: Designed to work alongside existing preprocessing

## Customization Options

- Change embedding model: `UnsupervisedEJAnalyzer(embedding_model='your-model')`
- Adjust clustering sensitivity: Modify `min_cluster_size` in HDBSCAN
- Change anomaly threshold: Adjust `contamination` parameter
- Add custom visualizations: Extend `UnsupervisedEJVisualizer` class

This solution provides true unsupervised learning value - discovering patterns and anomalies without any manual labeling or rule definition.