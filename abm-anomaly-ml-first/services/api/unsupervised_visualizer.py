"""
Unsupervised EJ Visualizer
Visualization module for unsupervised EJ analysis results
Creates interactive dashboards and detailed analysis plots
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, Optional, TYPE_CHECKING
import warnings
from loguru import logger

if TYPE_CHECKING:
    from unsupervised_analyzer import UnsupervisedEJAnalyzer

warnings.filterwarnings('ignore')

class UnsupervisedEJVisualizer:
    """
    Visualization module for unsupervised EJ analysis results
    """
    
    def __init__(self, analyzer: 'UnsupervisedEJAnalyzer'):
        """
        Initialize visualizer with analyzer instance
        
        Args:
            analyzer: Trained UnsupervisedEJAnalyzer instance
        """
        self.analyzer = analyzer
        self.results = analyzer.results
        
        # Set style
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            plt.style.use('default')
        sns.set_palette("husl")
        
        logger.info("UnsupervisedEJVisualizer initialized")
    
    def create_comprehensive_dashboard(self, save_path: Optional[str] = None, 
                                     interactive: bool = True) -> None:
        """
        Create comprehensive visualization dashboard
        
        Args:
            save_path: Path to save static plot (for matplotlib)
            interactive: Whether to create interactive plotly dashboard
        """
        try:
            if interactive:
                self._create_interactive_dashboard()
            else:
                self._create_static_dashboard(save_path)
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            raise
    
    def _create_interactive_dashboard(self):
        """Create interactive Plotly dashboard"""
        try:
            # Create subplots
            fig = make_subplots(
                rows=3, cols=3,
                subplot_titles=[
                    'UMAP Clustering View', 'Anomaly Scores Distribution', 'Cluster Sizes',
                    'PCA with Anomalies', 'Anomaly Detection Comparison', 'Pattern Analysis',
                    'Cluster Quality Metrics', 'Consensus Anomalies', 'Distance Distribution'
                ],
                specs=[[{'type': 'scatter'}, {'type': 'histogram'}, {'type': 'bar'}],
                       [{'type': 'scatter'}, {'type': 'bar'}, {'type': 'table'}],
                       [{'type': 'bar'}, {'type': 'scatter'}, {'type': 'histogram'}]],
                horizontal_spacing=0.1,
                vertical_spacing=0.1
            )
            
            # 1. UMAP with clusters
            if 'projections' in self.results:
                umap_coords = self.results['projections']['umap']['coordinates']
                cluster_labels = self.results['clustering']['labels']
                
                # Create color map for clusters
                unique_clusters = sorted(list(set(cluster_labels)))
                colors = px.colors.qualitative.Set3[:len(unique_clusters)]
                
                for i, cluster_id in enumerate(unique_clusters):
                    cluster_mask = cluster_labels == cluster_id
                    cluster_name = f'Cluster {cluster_id}' if cluster_id != -1 else 'Noise'
                    
                    fig.add_trace(
                        go.Scatter(
                            x=umap_coords[cluster_mask, 0],
                            y=umap_coords[cluster_mask, 1],
                            mode='markers',
                            name=cluster_name,
                            marker=dict(
                                color=colors[i] if i < len(colors) else 'black',
                                size=6,
                                opacity=0.7
                            ),
                            showlegend=True
                        ),
                        row=1, col=1
                    )
            
            # 2. Anomaly scores distribution
            iso_scores = self.results['anomaly_detection']['isolation_forest']['scores']
            fig.add_trace(
                go.Histogram(
                    x=iso_scores,
                    nbinsx=30,
                    name='Isolation Forest Scores',
                    marker_color='lightblue',
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # Add threshold line
            threshold = np.percentile(iso_scores, 5)
            fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                         annotation_text="Anomaly Threshold", row=1, col=2)
            
            # 3. Cluster sizes
            cluster_sizes = pd.Series(self.results['clustering']['labels']).value_counts().sort_index()
            cluster_names = [f'Cluster {i}' if i != -1 else 'Noise' for i in cluster_sizes.index]
            
            fig.add_trace(
                go.Bar(
                    x=cluster_names,
                    y=cluster_sizes.values,
                    name='Cluster Sizes',
                    marker_color='lightgreen',
                    showlegend=False
                ),
                row=1, col=3
            )
            
            # 4. PCA with anomalies
            if 'projections' in self.results:
                pca_coords = self.results['projections']['pca']['coordinates']
                iso_anomalies = self.results['anomaly_detection']['isolation_forest']['anomalies']
                
                # Normal points
                fig.add_trace(
                    go.Scatter(
                        x=pca_coords[~iso_anomalies, 0],
                        y=pca_coords[~iso_anomalies, 1],
                        mode='markers',
                        name='Normal',
                        marker=dict(color='blue', size=4, opacity=0.6),
                        showlegend=True
                    ),
                    row=2, col=1
                )
                
                # Anomalous points
                fig.add_trace(
                    go.Scatter(
                        x=pca_coords[iso_anomalies, 0],
                        y=pca_coords[iso_anomalies, 1],
                        mode='markers',
                        name='Anomalies',
                        marker=dict(color='red', size=6, opacity=0.8),
                        showlegend=True
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
                    marker_color='orange',
                    showlegend=False
                ),
                row=2, col=2
            )
            
            # 6. Pattern analysis table
            if 'patterns' in self.results and not self.results['patterns'].empty:
                patterns_df = self.results['patterns'].head(10)
                
                fig.add_trace(
                    go.Table(
                        header=dict(
                            values=['Pattern', 'Size', '%', 'Anomaly %'],
                            fill_color='lightgray',
                            align='left'
                        ),
                        cells=dict(
                            values=[
                                patterns_df['pattern_signature'],
                                patterns_df['size'],
                                [f"{x:.1f}%" for x in patterns_df['percentage']],
                                [f"{x:.1f}%" for x in patterns_df['anomaly_ratio'] * 100]
                            ],
                            fill_color='white',
                            align='left'
                        ),
                        showlegend=False
                    ),
                    row=2, col=3
                )
            
            # 7. Cluster quality metrics
            if 'metrics' in self.results:
                metrics = self.results['metrics']
                metric_names = []
                metric_values = []
                
                if 'silhouette_score' in metrics:
                    metric_names.append('Silhouette Score')
                    metric_values.append(metrics['silhouette_score'])
                
                if 'calinski_harabasz_score' in metrics:
                    metric_names.append('Calinski-Harabasz')
                    metric_values.append(metrics['calinski_harabasz_score'] / 1000)  # Scale down
                
                if 'anomaly_agreement' in metrics:
                    metric_names.append('Anomaly Agreement')
                    metric_values.append(metrics['anomaly_agreement'])
                
                if metric_names:
                    fig.add_trace(
                        go.Bar(
                            x=metric_names,
                            y=metric_values,
                            name='Quality Metrics',
                            marker_color='purple',
                            showlegend=False
                        ),
                        row=3, col=1
                    )
            
            # 8. Consensus anomalies scatter
            if 'projections' in self.results:
                umap_coords = self.results['projections']['umap']['coordinates']
                consensus_anomalies = self.results['anomaly_detection']['consensus']['anomalies']
                
                # Normal points
                fig.add_trace(
                    go.Scatter(
                        x=umap_coords[~consensus_anomalies, 0],
                        y=umap_coords[~consensus_anomalies, 1],
                        mode='markers',
                        name='Normal (Consensus)',
                        marker=dict(color='lightblue', size=4, opacity=0.5),
                        showlegend=False
                    ),
                    row=3, col=2
                )
                
                # Consensus anomalies
                if np.any(consensus_anomalies):
                    fig.add_trace(
                        go.Scatter(
                            x=umap_coords[consensus_anomalies, 0],
                            y=umap_coords[consensus_anomalies, 1],
                            mode='markers',
                            name='Consensus Anomalies',
                            marker=dict(color='darkred', size=8, opacity=0.9),
                            showlegend=False
                        ),
                        row=3, col=2
                    )
            
            # 9. Distance distribution
            distances = self.results['anomaly_detection']['statistical']['distances']
            threshold = self.results['anomaly_detection']['statistical']['threshold']
            
            fig.add_trace(
                go.Histogram(
                    x=distances,
                    nbinsx=30,
                    name='Distance Distribution',
                    marker_color='lightcoral',
                    showlegend=False
                ),
                row=3, col=3
            )
            
            fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                         annotation_text="95th Percentile", row=3, col=3)
            
            # Update layout
            fig.update_layout(
                title_text="Unsupervised EJ Log Analysis Dashboard",
                height=1200,
                width=1800,
                template='plotly_white',
                showlegend=True
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="UMAP 1", row=1, col=1)
            fig.update_yaxes(title_text="UMAP 2", row=1, col=1)
            fig.update_xaxes(title_text="Anomaly Score", row=1, col=2)
            fig.update_yaxes(title_text="Count", row=1, col=2)
            fig.update_xaxes(title_text="Cluster", row=1, col=3)
            fig.update_yaxes(title_text="Size", row=1, col=3)
            fig.update_xaxes(title_text="PCA 1", row=2, col=1)
            fig.update_yaxes(title_text="PCA 2", row=2, col=1)
            fig.update_xaxes(title_text="Method", row=2, col=2)
            fig.update_yaxes(title_text="Anomaly Count", row=2, col=2)
            fig.update_xaxes(title_text="Metric", row=3, col=1)
            fig.update_yaxes(title_text="Score", row=3, col=1)
            fig.update_xaxes(title_text="UMAP 1", row=3, col=2)
            fig.update_yaxes(title_text="UMAP 2", row=3, col=2)
            fig.update_xaxes(title_text="Distance from Center", row=3, col=3)
            fig.update_yaxes(title_text="Count", row=3, col=3)
            
            fig.show()
            logger.info("✅ Interactive dashboard created successfully")
            
        except Exception as e:
            logger.error(f"Error creating interactive dashboard: {e}")
            raise
    
    def _create_static_dashboard(self, save_path: Optional[str]):
        """Create static matplotlib dashboard"""
        try:
            fig, axes = plt.subplots(3, 3, figsize=(20, 16))
            fig.suptitle('Unsupervised EJ Log Analysis Dashboard', fontsize=20)
            
            # 1. UMAP Clustering
            if 'projections' in self.results:
                ax = axes[0, 0]
                umap_coords = self.results['projections']['umap']['coordinates']
                cluster_labels = self.results['clustering']['labels']
                
                scatter = ax.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                                   c=cluster_labels, cmap='tab10', alpha=0.7, s=30)
                ax.set_title('UMAP Clustering View')
                ax.set_xlabel('UMAP 1')
                ax.set_ylabel('UMAP 2')
                plt.colorbar(scatter, ax=ax)
            
            # 2. Anomaly Scores
            ax = axes[0, 1]
            iso_scores = self.results['anomaly_detection']['isolation_forest']['scores']
            ax.hist(iso_scores, bins=30, alpha=0.7, color='lightblue')
            threshold = np.percentile(iso_scores, 5)
            ax.axvline(threshold, color='red', linestyle='--', label='Threshold')
            ax.set_title('Anomaly Scores Distribution')
            ax.set_xlabel('Isolation Forest Score')
            ax.set_ylabel('Count')
            ax.legend()
            
            # 3. Cluster Sizes
            ax = axes[0, 2]
            cluster_sizes = pd.Series(self.results['clustering']['labels']).value_counts().sort_index()
            cluster_names = [f'C{i}' if i != -1 else 'Noise' for i in cluster_sizes.index]
            ax.bar(cluster_names, cluster_sizes.values, color='lightgreen')
            ax.set_title('Cluster Sizes')
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Size')
            plt.setp(ax.get_xticklabels(), rotation=45)
            
            # 4. PCA with Anomalies
            if 'projections' in self.results:
                ax = axes[1, 0]
                pca_coords = self.results['projections']['pca']['coordinates']
                iso_anomalies = self.results['anomaly_detection']['isolation_forest']['anomalies']
                
                ax.scatter(pca_coords[~iso_anomalies, 0], pca_coords[~iso_anomalies, 1], 
                          c='blue', alpha=0.6, s=20, label='Normal')
                ax.scatter(pca_coords[iso_anomalies, 0], pca_coords[iso_anomalies, 1], 
                          c='red', alpha=0.8, s=30, label='Anomalies')
                ax.set_title('PCA with Anomalies')
                ax.set_xlabel('PCA 1')
                ax.set_ylabel('PCA 2')
                ax.legend()
            
            # 5. Anomaly Detection Comparison
            ax = axes[1, 1]
            methods = ['Iso Forest', 'LOF', 'Statistical', 'Consensus']
            anomaly_counts = [
                self.results['anomaly_detection']['isolation_forest']['n_anomalies'],
                self.results['anomaly_detection']['lof']['n_anomalies'],
                self.results['anomaly_detection']['statistical']['n_anomalies'],
                self.results['anomaly_detection']['consensus']['n_anomalies']
            ]
            ax.bar(methods, anomaly_counts, color='orange')
            ax.set_title('Anomaly Detection Comparison')
            ax.set_xlabel('Method')
            ax.set_ylabel('Anomaly Count')
            plt.setp(ax.get_xticklabels(), rotation=45)
            
            # 6. Pattern Analysis
            ax = axes[1, 2]
            if 'patterns' in self.results and not self.results['patterns'].empty:
                patterns_df = self.results['patterns'].head(8)
                y_pos = np.arange(len(patterns_df))
                ax.barh(y_pos, patterns_df['size'], color='lightcoral')
                ax.set_yticks(y_pos)
                ax.set_yticklabels([p[:15] + '...' if len(p) > 15 else p 
                                   for p in patterns_df['pattern_signature']])
                ax.set_title('Top Patterns by Size')
                ax.set_xlabel('Size')
            
            # 7. Quality Metrics
            ax = axes[2, 0]
            if 'metrics' in self.results:
                metrics = self.results['metrics']
                metric_names = []
                metric_values = []
                
                if 'silhouette_score' in metrics:
                    metric_names.append('Silhouette')
                    metric_values.append(metrics['silhouette_score'])
                
                if 'anomaly_agreement' in metrics:
                    metric_names.append('Agreement')
                    metric_values.append(metrics['anomaly_agreement'])
                
                if metric_names:
                    ax.bar(metric_names, metric_values, color='purple')
                    ax.set_title('Quality Metrics')
                    ax.set_ylabel('Score')
            
            # 8. Consensus Anomalies
            if 'projections' in self.results:
                ax = axes[2, 1]
                umap_coords = self.results['projections']['umap']['coordinates']
                consensus_anomalies = self.results['anomaly_detection']['consensus']['anomalies']
                
                ax.scatter(umap_coords[~consensus_anomalies, 0], umap_coords[~consensus_anomalies, 1], 
                          c='lightblue', alpha=0.5, s=20, label='Normal')
                if np.any(consensus_anomalies):
                    ax.scatter(umap_coords[consensus_anomalies, 0], umap_coords[consensus_anomalies, 1], 
                              c='darkred', alpha=0.9, s=40, label='Consensus Anomalies')
                ax.set_title('Consensus Anomalies')
                ax.set_xlabel('UMAP 1')
                ax.set_ylabel('UMAP 2')
                ax.legend()
            
            # 9. Distance Distribution
            ax = axes[2, 2]
            distances = self.results['anomaly_detection']['statistical']['distances']
            threshold = self.results['anomaly_detection']['statistical']['threshold']
            ax.hist(distances, bins=30, alpha=0.7, color='lightcoral')
            ax.axvline(threshold, color='red', linestyle='--', label='95th Percentile')
            ax.set_title('Distance Distribution')
            ax.set_xlabel('Distance from Center')
            ax.set_ylabel('Count')
            ax.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Static dashboard saved to {save_path}")
            
            plt.show()
            logger.info("✅ Static dashboard created successfully")
            
        except Exception as e:
            logger.error(f"Error creating static dashboard: {e}")
            raise
    
    def plot_anomaly_analysis(self, sequence_idx: int):
        """
        Detailed analysis plot for a specific sequence
        
        Args:
            sequence_idx: Index of sequence to analyze
        """
        try:
            if sequence_idx >= len(self.analyzer.sequences):
                raise ValueError(f"Sequence index {sequence_idx} out of range")
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Detailed Analysis for Sequence {sequence_idx}', fontsize=16)
            
            # Get sequence data
            sequence = self.analyzer.sequences[sequence_idx]
            embedding = self.analyzer.embeddings_normalized[sequence_idx]
            
            # 1. Sequence tokens
            ax = axes[0, 0]
            tokens = sequence.split()[:20]  # First 20 tokens
            ax.text(0.05, 0.95, f"Sequence {sequence_idx}:\n" + "\n".join(tokens),
                    ha='left', va='top', transform=ax.transAxes,
                    fontsize=10, wrap=True)
            ax.set_title('Sequence Tokens (First 20)')
            ax.axis('off')
            
            # 2. Anomaly scores
            ax = axes[0, 1]
            methods = ['Isolation Forest', 'Statistical']
            scores = [
                self.results['anomaly_detection']['isolation_forest']['scores'][sequence_idx],
                self.results['anomaly_detection']['statistical']['distances'][sequence_idx]
            ]
            
            bars = ax.bar(methods, scores)
            ax.set_title(f'Anomaly Scores for Sequence {sequence_idx}')
            ax.set_ylabel('Score')
            
            # Color bars based on anomaly detection
            iso_threshold = np.percentile(
                self.results['anomaly_detection']['isolation_forest']['scores'], 5
            )
            stat_threshold = self.results['anomaly_detection']['statistical']['threshold']
            
            colors = []
            colors.append('red' if scores[0] < iso_threshold else 'green')
            colors.append('red' if scores[1] > stat_threshold else 'green')
            
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            # 3. Nearest neighbors
            ax = axes[1, 0]
            distances = np.linalg.norm(
                self.analyzer.embeddings_normalized - embedding, axis=1
            )
            nearest_indices = np.argsort(distances)[1:6]  # Skip self
            
            neighbor_data = []
            for i, idx in enumerate(nearest_indices):
                neighbor_data.append({
                    'Rank': i + 1,
                    'Index': idx,
                    'Distance': f"{distances[idx]:.3f}",
                    'Cluster': self.results['clustering']['labels'][idx],
                    'Pattern': self.results['patterns'].iloc[
                        self.results['clustering']['labels'][idx]
                    ]['pattern_signature'] if self.results['clustering']['labels'][idx] != -1 else 'Noise'
                })
            
            neighbor_df = pd.DataFrame(neighbor_data)
            ax.axis('off')
            table = ax.table(cellText=neighbor_df.values,
                            colLabels=neighbor_df.columns,
                            loc='center',
                            cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            ax.set_title('5 Nearest Neighbors')
            
            # 4. Position in embedding space
            ax = axes[1, 1]
            if 'projections' in self.results:
                umap_coords = self.results['projections']['umap']['coordinates']
                cluster_labels = self.results['clustering']['labels']
                
                # Plot all points
                scatter = ax.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                                   c=cluster_labels, cmap='tab10', alpha=0.5, s=20)
                
                # Highlight the specific sequence
                ax.scatter(umap_coords[sequence_idx, 0], umap_coords[sequence_idx, 1], 
                          c='red', s=100, marker='x', linewidths=3, 
                          label=f'Sequence {sequence_idx}')
                
                ax.set_title('Position in UMAP Space')
                ax.set_xlabel('UMAP 1')
                ax.set_ylabel('UMAP 2')
                ax.legend()
            
            plt.tight_layout()
            plt.show()
            logger.info(f"✅ Anomaly analysis plot created for sequence {sequence_idx}")
            
        except Exception as e:
            logger.error(f"Error creating anomaly analysis plot: {e}")
            raise
    
    def generate_analysis_report(self) -> pd.DataFrame:
        """Generate comprehensive analysis report as DataFrame"""
        try:
            report_data = []
            
            # Overall statistics
            report_data.append({
                'Metric': 'Total Sequences',
                'Value': len(self.analyzer.sequences),
                'Category': 'Overview',
                'Description': 'Total number of sequences analyzed'
            })
            
            report_data.append({
                'Metric': 'Embedding Dimension',
                'Value': self.analyzer.embeddings.shape[1] if self.analyzer.embeddings is not None else 0,
                'Category': 'Overview',
                'Description': 'Dimensionality of sentence embeddings'
            })
            
            # Clustering statistics
            report_data.append({
                'Metric': 'Clusters Found',
                'Value': self.results['clustering']['n_clusters'],
                'Category': 'Clustering',
                'Description': 'Number of distinct clusters discovered'
            })
            
            report_data.append({
                'Metric': 'Noise Points',
                'Value': self.results['clustering']['n_noise'],
                'Category': 'Clustering',
                'Description': 'Number of sequences that could not be clustered'
            })
            
            report_data.append({
                'Metric': 'Noise Ratio',
                'Value': f"{self.results['clustering']['noise_ratio']:.1%}",
                'Category': 'Clustering',
                'Description': 'Percentage of sequences classified as noise'
            })
            
            # Anomaly statistics
            for method, data in self.results['anomaly_detection'].items():
                if 'n_anomalies' in data:
                    report_data.append({
                        'Metric': f'{method.title()} Anomalies',
                        'Value': data['n_anomalies'],
                        'Category': 'Anomaly Detection',
                        'Description': f'Anomalies detected by {method.replace("_", " ").title()}'
                    })
                    
                    report_data.append({
                        'Metric': f'{method.title()} Rate',
                        'Value': f"{data['anomaly_rate']:.1%}",
                        'Category': 'Anomaly Detection',
                        'Description': f'Anomaly rate for {method.replace("_", " ").title()}'
                    })
            
            # Cluster quality metrics
            if 'metrics' in self.results:
                for metric_name, value in self.results['metrics'].items():
                    if metric_name in ['silhouette_score', 'calinski_harabasz_score', 
                                     'davies_bouldin_score', 'anomaly_agreement']:
                        report_data.append({
                            'Metric': metric_name.replace('_', ' ').title(),
                            'Value': f"{value:.3f}" if isinstance(value, float) else value,
                            'Category': 'Quality Metrics',
                            'Description': f'Quality metric: {metric_name.replace("_", " ")}'
                        })
            
            # Pattern distribution
            if 'patterns' in self.results and not self.results['patterns'].empty:
                patterns = self.results['patterns']
                for _, pattern in patterns.head(10).iterrows():
                    report_data.append({
                        'Metric': f'Pattern: {pattern["pattern_signature"][:20]}...',
                        'Value': f'{pattern["size"]} ({pattern["percentage"]:.1f}%)',
                        'Category': 'Patterns',
                        'Description': f'Size and percentage of pattern cluster'
                    })
            
            report_df = pd.DataFrame(report_data)
            logger.info("✅ Analysis report generated successfully")
            return report_df
            
        except Exception as e:
            logger.error(f"Error generating analysis report: {e}")
            return pd.DataFrame([{
                'Metric': 'Error',
                'Value': str(e),
                'Category': 'Error',
                'Description': 'Error occurred during report generation'
            }])
    
    def export_visualizations(self, output_dir: str = './ej_visualizations'):
        """Export all visualizations to files"""
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            # Create static dashboard
            static_path = os.path.join(output_dir, 'dashboard_static.png')
            self._create_static_dashboard(static_path)
            
            # Export individual plots
            self._export_individual_plots(output_dir)
            
            logger.info(f"✅ Visualizations exported to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error exporting visualizations: {e}")
            raise
    
    def _export_individual_plots(self, output_dir: str):
        """Export individual analysis plots"""
        try:
            # UMAP clustering plot
            if 'projections' in self.results:
                fig, ax = plt.subplots(figsize=(10, 8))
                umap_coords = self.results['projections']['umap']['coordinates']
                cluster_labels = self.results['clustering']['labels']
                
                scatter = ax.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                                   c=cluster_labels, cmap='tab10', alpha=0.7, s=30)
                ax.set_title('UMAP Clustering View')
                ax.set_xlabel('UMAP 1')
                ax.set_ylabel('UMAP 2')
                plt.colorbar(scatter, ax=ax)
                
                plt.savefig(os.path.join(output_dir, 'umap_clustering.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
            
            # Pattern distribution pie chart
            if 'patterns' in self.results and not self.results['patterns'].empty:
                fig, ax = plt.subplots(figsize=(10, 8))
                patterns = self.results['patterns'].head(8)
                
                ax.pie(patterns['size'], labels=patterns['pattern_signature'], 
                      autopct='%1.1f%%', startangle=90)
                ax.set_title('Pattern Distribution')
                
                plt.savefig(os.path.join(output_dir, 'pattern_distribution.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
            
            logger.info("Individual plots exported successfully")
            
        except Exception as e:
            logger.error(f"Error exporting individual plots: {e}")
