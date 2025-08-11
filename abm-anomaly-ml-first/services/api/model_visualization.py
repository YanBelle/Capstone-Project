"""
Model Performance Visualization Module
Provides comprehensive visualization capabilities for ensemble models
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import silhouette_score, silhouette_samples, roc_curve, auc, f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from io import BytesIO
import base64
import json
from loguru import logger
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from datetime import datetime

class EnsembleVisualizationEngine:
    """Main visualization engine for ensemble model performance"""
    
    def __init__(self, ml_analyzer=None):
        self.ml_analyzer = ml_analyzer
        self.plt_style = 'seaborn-v0_8'
        plt.style.use('default')  # Use default if seaborn not available
        
    def create_isolation_forest_visualization(self, if_predictions, if_scores, embeddings_scaled, 
                                            session_ids=None, ground_truth=None) -> Dict[str, Any]:
        """Create comprehensive Isolation Forest visualization"""
        
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Anomaly score distribution
        plt.subplot(2, 3, 1)
        plt.hist(if_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=np.mean(if_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(if_scores):.3f}')
        plt.axvline(x=np.percentile(if_scores, 10), color='orange', linestyle='--', 
                   label=f'10th percentile: {np.percentile(if_scores, 10):.3f}')
        plt.title('Isolation Forest Anomaly Scores Distribution')
        plt.xlabel('Anomaly Score (lower = more anomalous)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Decision boundary visualization (2D projection)
        plt.subplot(2, 3, 2)
        pca_2d = PCA(n_components=2).fit_transform(embeddings_scaled)
        colors = ['red' if pred == -1 else 'blue' for pred in if_predictions]
        scatter = plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c=colors, alpha=0.6, s=30)
        plt.title('Isolation Forest Decisions (PCA Projection)')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        
        # Add legend
        red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                              markersize=8, label='Anomaly')
        blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                               markersize=8, label='Normal')
        plt.legend(handles=[red_patch, blue_patch])
        plt.grid(True, alpha=0.3)
        
        # 3. ROC curve (if ground truth available)
        plt.subplot(2, 3, 3)
        if ground_truth is not None:
            fpr, tpr, _ = roc_curve(ground_truth, -if_scores)  # Negative because lower scores = more anomalous
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve - Isolation Forest')
            plt.legend(loc="lower right")
        else:
            plt.text(0.5, 0.5, 'ROC Curve\n(Ground truth not available)', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
            plt.title('ROC Curve - Isolation Forest')
        plt.grid(True, alpha=0.3)
        
        # 4. Score vs prediction scatter
        plt.subplot(2, 3, 4)
        anomaly_mask = if_predictions == -1
        normal_scores = if_scores[~anomaly_mask]
        anomaly_scores = if_scores[anomaly_mask]
        
        plt.scatter(range(len(normal_scores)), normal_scores, c='blue', alpha=0.6, 
                   label=f'Normal ({len(normal_scores)})', s=20)
        plt.scatter(range(len(normal_scores), len(if_scores)), anomaly_scores, c='red', alpha=0.6, 
                   label=f'Anomaly ({len(anomaly_scores)})', s=20)
        plt.xlabel('Sample Index')
        plt.ylabel('Anomaly Score')
        plt.title('Anomaly Scores by Prediction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Statistics summary
        plt.subplot(2, 3, 5)
        stats_text = f"""
        Isolation Forest Statistics:
        
        Total Samples: {len(if_scores)}
        Anomalies Detected: {np.sum(if_predictions == -1)}
        Anomaly Rate: {(np.sum(if_predictions == -1) / len(if_predictions)):.1%}
        
        Score Statistics:
        Mean Score: {np.mean(if_scores):.4f}
        Std Score: {np.std(if_scores):.4f}
        Min Score: {np.min(if_scores):.4f}
        Max Score: {np.max(if_scores):.4f}
        
        Anomaly Threshold: ~{np.percentile(if_scores, 10):.4f}
        """
        plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        plt.axis('off')
        
        # 6. Feature importance (if available)
        plt.subplot(2, 3, 6)
        if hasattr(self.ml_analyzer, 'isolation_forest') and hasattr(self.ml_analyzer.isolation_forest, 'estimators_'):
            # Calculate feature importance from trees
            try:
                importances = []
                for estimator in self.ml_analyzer.isolation_forest.estimators_:
                    if hasattr(estimator, 'feature_importances_'):
                        importances.append(estimator.feature_importances_)
                
                if importances:
                    mean_importance = np.mean(importances, axis=0)
                    top_features = np.argsort(mean_importance)[-20:]  # Top 20 features
                    
                    plt.barh(range(len(top_features)), mean_importance[top_features])
                    plt.xlabel('Feature Importance')
                    plt.ylabel('Feature Index')
                    plt.title('Top 20 Feature Importances')
                    plt.yticks(range(len(top_features)), top_features)
                else:
                    plt.text(0.5, 0.5, 'Feature Importance\n(Not available)', 
                            ha='center', va='center', transform=plt.gca().transAxes)
            except Exception as e:
                plt.text(0.5, 0.5, f'Feature Importance\n(Error: {str(e)})', 
                        ha='center', va='center', transform=plt.gca().transAxes)
        else:
            plt.text(0.5, 0.5, 'Feature Importance\n(Not available)', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Feature Importance Analysis')
        
        plt.tight_layout()
        
        # Convert to base64 for API response
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        # Calculate additional metrics
        anomaly_rate = np.sum(if_predictions == -1) / len(if_predictions)
        score_stats = {
            'mean': float(np.mean(if_scores)),
            'std': float(np.std(if_scores)),
            'min': float(np.min(if_scores)),
            'max': float(np.max(if_scores)),
            'anomaly_threshold': float(np.percentile(if_scores, 10))
        }
        
        return {
            'visualization': img_base64,
            'metrics': {
                'total_samples': len(if_scores),
                'anomalies_detected': int(np.sum(if_predictions == -1)),
                'anomaly_rate': float(anomaly_rate),
                'score_statistics': score_stats
            },
            'model_type': 'isolation_forest'
        }
    
    def create_svm_visualization(self, svm_predictions, svm_scores, embeddings_scaled, 
                               session_ids=None, ground_truth=None) -> Dict[str, Any]:
        """Create comprehensive One-Class SVM visualization"""
        
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Decision function distribution
        plt.subplot(2, 3, 1)
        plt.hist(svm_scores, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Decision Boundary')
        plt.axvline(x=np.mean(svm_scores), color='blue', linestyle='--', 
                   label=f'Mean: {np.mean(svm_scores):.3f}')
        plt.title('One-Class SVM Decision Scores Distribution')
        plt.xlabel('Decision Score (negative = anomaly)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Support vectors highlight
        plt.subplot(2, 3, 2)
        pca_2d = PCA(n_components=2).fit_transform(embeddings_scaled)
        colors = ['red' if pred == -1 else 'blue' for pred in svm_predictions]
        plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c=colors, alpha=0.6, s=30)
        
        # Highlight support vectors if available
        if hasattr(self.ml_analyzer, 'one_class_svm') and hasattr(self.ml_analyzer.one_class_svm, 'support_'):
            support_indices = self.ml_analyzer.one_class_svm.support_
            if len(support_indices) > 0 and len(support_indices) < len(pca_2d):
                plt.scatter(pca_2d[support_indices, 0], pca_2d[support_indices, 1], 
                           s=100, facecolors='none', edgecolors='black', linewidth=2,
                           label=f'Support Vectors ({len(support_indices)})')
        
        plt.title('SVM Decision Space (PCA Projection)')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Decision boundary analysis
        plt.subplot(2, 3, 3)
        # Create decision boundary visualization in 2D space
        if len(pca_2d) > 0:
            h = 0.1  # Step size
            x_min, x_max = pca_2d[:, 0].min() - 1, pca_2d[:, 0].max() + 1
            y_min, y_max = pca_2d[:, 1].min() - 1, pca_2d[:, 1].max() + 1
            
            # Create a coarse grid for visualization
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h*5),
                               np.arange(y_min, y_max, h*5))
            
            # Create dummy data for the mesh
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            
            # Since we can't directly apply SVM to 2D PCA space, we'll show the actual data points
            plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c=colors, alpha=0.6)
            plt.title('SVM Decision Boundary Approximation')
            plt.xlabel('First Principal Component')
            plt.ylabel('Second Principal Component')
        else:
            plt.text(0.5, 0.5, 'Decision Boundary\n(Insufficient data)', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        # 4. Score distribution by prediction
        plt.subplot(2, 3, 4)
        anomaly_mask = svm_predictions == -1
        normal_scores = svm_scores[~anomaly_mask]
        anomaly_scores = svm_scores[anomaly_mask]
        
        plt.boxplot([normal_scores, anomaly_scores], labels=['Normal', 'Anomaly'])
        plt.ylabel('Decision Score')
        plt.title('Score Distribution by Prediction')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        if len(normal_scores) > 0:
            plt.text(1, np.median(normal_scores), f'n={len(normal_scores)}', 
                    ha='center', va='bottom')
        if len(anomaly_scores) > 0:
            plt.text(2, np.median(anomaly_scores), f'n={len(anomaly_scores)}', 
                    ha='center', va='bottom')
        
        # 5. SVM Statistics
        plt.subplot(2, 3, 5)
        n_support = 0
        if hasattr(self.ml_analyzer, 'one_class_svm') and hasattr(self.ml_analyzer.one_class_svm, 'support_'):
            n_support = len(self.ml_analyzer.one_class_svm.support_)
        
        stats_text = f"""
        One-Class SVM Statistics:
        
        Total Samples: {len(svm_scores)}
        Anomalies Detected: {np.sum(svm_predictions == -1)}
        Anomaly Rate: {(np.sum(svm_predictions == -1) / len(svm_predictions)):.1%}
        Support Vectors: {n_support}
        
        Score Statistics:
        Mean Score: {np.mean(svm_scores):.4f}
        Std Score: {np.std(svm_scores):.4f}
        Min Score: {np.min(svm_scores):.4f}
        Max Score: {np.max(svm_scores):.4f}
        
        Decision Boundary: 0.0
        """
        plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        plt.axis('off')
        
        # 6. ROC curve or performance metrics
        plt.subplot(2, 3, 6)
        if ground_truth is not None:
            fpr, tpr, _ = roc_curve(ground_truth, svm_scores)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve - One-Class SVM')
            plt.legend(loc="lower right")
        else:
            # Show kernel information instead
            kernel_info = "Unknown"
            if hasattr(self.ml_analyzer, 'one_class_svm'):
                kernel_info = getattr(self.ml_analyzer.one_class_svm, 'kernel', 'Unknown')
            
            plt.text(0.5, 0.5, f'One-Class SVM\nKernel: {kernel_info}\n\n(ROC requires ground truth)', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
            plt.title('SVM Configuration')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return {
            'visualization': img_base64,
            'metrics': {
                'total_samples': len(svm_scores),
                'anomalies_detected': int(np.sum(svm_predictions == -1)),
                'anomaly_rate': float(np.sum(svm_predictions == -1) / len(svm_predictions)),
                'support_vectors': n_support,
                'score_statistics': {
                    'mean': float(np.mean(svm_scores)),
                    'std': float(np.std(svm_scores)),
                    'min': float(np.min(svm_scores)),
                    'max': float(np.max(svm_scores))
                }
            },
            'model_type': 'one_class_svm'
        }
    
    def create_dbscan_visualization(self, dbscan_labels, dbscan_scores, embeddings_scaled, 
                                  session_ids=None) -> Dict[str, Any]:
        """Create comprehensive DBSCAN clustering visualization"""
        
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Cluster visualization
        plt.subplot(2, 3, 1)
        pca_2d = PCA(n_components=2).fit_transform(embeddings_scaled)
        unique_labels = set(dbscan_labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black for noise points (outliers)
                col = [0, 0, 0, 1]
                label = f'Outliers ({np.sum(dbscan_labels == k)})'
            else:
                label = f'Cluster {k} ({np.sum(dbscan_labels == k)})'
            
            class_member_mask = (dbscan_labels == k)
            xy = pca_2d[class_member_mask]
            plt.scatter(xy[:, 0], xy[:, 1], c=[col], alpha=0.6, s=30, label=label)
        
        plt.title('DBSCAN Clustering Results')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 2. Cluster size distribution
        plt.subplot(2, 3, 2)
        cluster_sizes = pd.Series(dbscan_labels).value_counts().sort_index()
        
        # Separate outliers from clusters
        outliers = cluster_sizes.get(-1, 0)
        clusters = cluster_sizes.drop(-1, errors='ignore')
        
        if len(clusters) > 0:
            clusters.plot(kind='bar', color='skyblue', alpha=0.7)
            plt.xlabel('Cluster ID')
            plt.ylabel('Number of Points')
            plt.title('Cluster Size Distribution')
            plt.xticks(rotation=45)
            
            # Add outliers as a separate bar
            if outliers > 0:
                plt.bar(len(clusters), outliers, color='red', alpha=0.7, label=f'Outliers: {outliers}')
                plt.legend()
        else:
            plt.text(0.5, 0.5, 'No clusters found\n(All points are outliers)', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Cluster Size Distribution')
        
        plt.grid(True, alpha=0.3)
        
        # 3. Silhouette analysis
        plt.subplot(2, 3, 3)
        if len(set(dbscan_labels)) > 1 and -1 not in dbscan_labels:
            try:
                silhouette_avg = silhouette_score(embeddings_scaled, dbscan_labels)
                sample_silhouette_values = silhouette_samples(embeddings_scaled, dbscan_labels)
                
                y_lower = 10
                for i in sorted(set(dbscan_labels)):
                    if i != -1:  # Skip outliers
                        cluster_silhouette_values = sample_silhouette_values[dbscan_labels == i]
                        cluster_silhouette_values.sort()
                        
                        size_cluster_i = cluster_silhouette_values.shape[0]
                        y_upper = y_lower + size_cluster_i
                        
                        color = plt.cm.nipy_spectral(float(i) / len(set(dbscan_labels)))
                        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_values,
                                        facecolor=color, edgecolor=color, alpha=0.7)
                        
                        # Label the silhouette plots with their cluster numbers at the middle
                        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                        y_lower = y_upper + 10
                
                plt.axvline(x=silhouette_avg, color="red", linestyle="--", 
                           label=f'Average Score: {silhouette_avg:.3f}')
                plt.xlabel('Silhouette Coefficient Values')
                plt.ylabel('Cluster Label')
                plt.title(f'Silhouette Analysis (avg={silhouette_avg:.3f})')
                plt.legend()
            except Exception as e:
                plt.text(0.5, 0.5, f'Silhouette Analysis\n(Error: {str(e)})', 
                        ha='center', va='center', transform=plt.gca().transAxes)
        else:
            plt.text(0.5, 0.5, 'Silhouette Analysis\n(Requires multiple clusters)', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Silhouette Analysis')
        
        # 4. Distance distribution
        plt.subplot(2, 3, 4)
        plt.hist(dbscan_scores, bins=30, alpha=0.7, color='green', edgecolor='black')
        plt.xlabel('Distance to Cluster Center')
        plt.ylabel('Frequency')
        plt.title('Anomaly Score Distribution (DBSCAN)')
        plt.axvline(x=np.mean(dbscan_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(dbscan_scores):.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. DBSCAN Statistics
        plt.subplot(2, 3, 5)
        n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_outliers = np.sum(dbscan_labels == -1)
        
        # Get DBSCAN parameters
        eps = getattr(self.ml_analyzer.dbscan, 'eps', 'Unknown') if hasattr(self.ml_analyzer, 'dbscan') else 'Unknown'
        min_samples = getattr(self.ml_analyzer.dbscan, 'min_samples', 'Unknown') if hasattr(self.ml_analyzer, 'dbscan') else 'Unknown'
        
        stats_text = f"""
        DBSCAN Statistics:
        
        Parameters:
        - eps: {eps}
        - min_samples: {min_samples}
        
        Results:
        Total Samples: {len(dbscan_labels)}
        Clusters Found: {n_clusters}
        Outliers: {n_outliers}
        Outlier Rate: {(n_outliers / len(dbscan_labels)):.1%}
        
        Score Statistics:
        Mean Distance: {np.mean(dbscan_scores):.4f}
        Std Distance: {np.std(dbscan_scores):.4f}
        Max Distance: {np.max(dbscan_scores):.4f}
        """
        plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        plt.axis('off')
        
        # 6. Cluster characteristics
        plt.subplot(2, 3, 6)
        if n_clusters > 0:
            cluster_stats = []
            for cluster_id in sorted(set(dbscan_labels)):
                if cluster_id != -1:
                    cluster_mask = dbscan_labels == cluster_id
                    cluster_scores = dbscan_scores[cluster_mask]
                    cluster_stats.append({
                        'cluster': cluster_id,
                        'size': np.sum(cluster_mask),
                        'avg_distance': np.mean(cluster_scores),
                        'max_distance': np.max(cluster_scores)
                    })
            
            if cluster_stats:
                df_stats = pd.DataFrame(cluster_stats)
                x_pos = np.arange(len(df_stats))
                
                plt.bar(x_pos, df_stats['avg_distance'], alpha=0.7, 
                       color='lightblue', label='Avg Distance')
                plt.xlabel('Cluster ID')
                plt.ylabel('Average Distance')
                plt.title('Cluster Cohesion (Lower = More Cohesive)')
                plt.xticks(x_pos, df_stats['cluster'])
                
                # Add size annotations
                for i, (idx, row) in enumerate(df_stats.iterrows()):
                    plt.text(i, row['avg_distance'] + 0.01, f"n={row['size']}", 
                            ha='center', va='bottom', fontsize=8)
            else:
                plt.text(0.5, 0.5, 'No clusters to analyze', 
                        ha='center', va='center', transform=plt.gca().transAxes)
        else:
            plt.text(0.5, 0.5, 'No clusters found', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return {
            'visualization': img_base64,
            'metrics': {
                'total_samples': len(dbscan_labels),
                'clusters_found': n_clusters,
                'outliers_detected': n_outliers,
                'outlier_rate': float(n_outliers / len(dbscan_labels)),
                'parameters': {
                    'eps': eps,
                    'min_samples': min_samples
                },
                'score_statistics': {
                    'mean': float(np.mean(dbscan_scores)),
                    'std': float(np.std(dbscan_scores)),
                    'max': float(np.max(dbscan_scores))
                }
            },
            'model_type': 'dbscan'
        }
    
    def create_ensemble_dashboard(self, anomaly_results: Dict, embeddings_scaled: np.ndarray, 
                                session_data: List, ground_truth=None) -> Dict[str, Any]:
        """Create comprehensive ensemble performance dashboard"""
        
        fig = plt.figure(figsize=(20, 15))
        
        # Extract results
        if_predictions = anomaly_results.get('if_predictions', [])
        svm_predictions = anomaly_results.get('svm_predictions', [])
        dbscan_predictions = anomaly_results.get('dbscan_predictions', [])
        
        # 1. Model agreement matrix
        plt.subplot(3, 4, 1)
        model_predictions = pd.DataFrame({
            'Isolation_Forest': if_predictions,
            'One_Class_SVM': svm_predictions,
            'DBSCAN': dbscan_predictions
        })
        agreement_matrix = model_predictions.corr()
        sns.heatmap(agreement_matrix, annot=True, cmap='RdYlBu', center=0, 
                   square=True, cbar_kws={'label': 'Agreement'})
        plt.title('Model Agreement Matrix')
        
        # 2. Ensemble voting distribution
        plt.subplot(3, 4, 2)
        ensemble_votes = (model_predictions == -1).sum(axis=1)
        vote_counts = ensemble_votes.value_counts().sort_index()
        
        plt.bar(vote_counts.index, vote_counts.values, alpha=0.7, color=['green', 'yellow', 'orange', 'red'][:len(vote_counts)])
        plt.xlabel('Number of Models Voting Anomaly')
        plt.ylabel('Number of Sessions')
        plt.title('Ensemble Voting Distribution')
        plt.xticks(range(4))
        
        # Add percentages
        total = len(ensemble_votes)
        for i, count in enumerate(vote_counts.values):
            plt.text(vote_counts.index[i], count + total*0.01, f'{count/total:.1%}', 
                    ha='center', va='bottom')
        
        # 3. Anomaly type distribution
        plt.subplot(3, 4, 3)
        anomaly_types = {}
        for session in session_data:
            if hasattr(session, 'anomalies'):
                for anomaly in session.anomalies:
                    anomaly_types[anomaly.anomaly_type] = anomaly_types.get(anomaly.anomaly_type, 0) + 1
        
        if anomaly_types:
            # Show top 10 anomaly types
            sorted_types = sorted(anomaly_types.items(), key=lambda x: x[1], reverse=True)[:10]
            types, counts = zip(*sorted_types)
            
            plt.pie(counts, labels=types, autopct='%1.1f%%', startangle=90)
            plt.title('Top 10 Anomaly Types')
        else:
            plt.text(0.5, 0.5, 'No anomaly types\navailable', ha='center', va='center')
            plt.title('Anomaly Type Distribution')
        
        # 4. Confidence score distributions by model
        plt.subplot(3, 4, 4)
        confidence_data = {'IF': [], 'SVM': [], 'DBSCAN': [], 'DeepLog': []}
        
        for session in session_data:
            if hasattr(session, 'anomalies'):
                for anomaly in session.anomalies:
                    if 'isolation' in anomaly.detection_method.lower():
                        confidence_data['IF'].append(anomaly.confidence)
                    elif 'svm' in anomaly.detection_method.lower():
                        confidence_data['SVM'].append(anomaly.confidence)
                    elif 'dbscan' in anomaly.detection_method.lower():
                        confidence_data['DBSCAN'].append(anomaly.confidence)
                    elif 'deeplog' in anomaly.detection_method.lower():
                        confidence_data['DeepLog'].append(anomaly.confidence)
        
        box_data = [data for data in confidence_data.values() if data]
        box_labels = [name for name, data in confidence_data.items() if data]
        
        if box_data:
            plt.boxplot(box_data, labels=box_labels)
            plt.ylabel('Confidence Score')
            plt.title('Confidence by Detection Method')
            plt.xticks(rotation=45)
        else:
            plt.text(0.5, 0.5, 'No confidence data\navailable', ha='center', va='center')
            plt.title('Confidence Distributions')
        
        # 5. Model performance comparison (if ground truth available)
        plt.subplot(3, 4, 5)
        if ground_truth is not None:
            models = ['Isolation Forest', 'One-Class SVM', 'DBSCAN']
            predictions = [if_predictions, svm_predictions, dbscan_predictions]
            
            f1_scores = []
            for pred in predictions:
                try:
                    f1 = f1_score(ground_truth, [1 if p == -1 else 0 for p in pred])
                    f1_scores.append(f1)
                except:
                    f1_scores.append(0)
            
            plt.bar(models, f1_scores, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen'])
            plt.ylabel('F1 Score')
            plt.title('Model Performance Comparison')
            plt.xticks(rotation=45)
            
            # Add values on bars
            for i, score in enumerate(f1_scores):
                plt.text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom')
        else:
            plt.text(0.5, 0.5, 'Performance Comparison\n(Requires ground truth)', 
                    ha='center', va='center')
            plt.title('Model Performance')
        
        # 6. Anomaly severity distribution
        plt.subplot(3, 4, 6)
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for session in session_data:
            if hasattr(session, 'anomalies'):
                for anomaly in session.anomalies:
                    severity = getattr(anomaly, 'severity', 'medium').lower()
                    if severity in severity_counts:
                        severity_counts[severity] += 1
        
        if sum(severity_counts.values()) > 0:
            severities = list(severity_counts.keys())
            counts = list(severity_counts.values())
            colors = ['green', 'yellow', 'orange', 'red']
            
            plt.bar(severities, counts, color=colors, alpha=0.7)
            plt.ylabel('Number of Anomalies')
            plt.title('Anomaly Severity Distribution')
            
            # Add percentages
            total = sum(counts)
            for i, count in enumerate(counts):
                if count > 0:
                    plt.text(i, count + total*0.01, f'{count/total:.1%}', 
                            ha='center', va='bottom')
        else:
            plt.text(0.5, 0.5, 'No severity data\navailable', ha='center', va='center')
            plt.title('Severity Distribution')
        
        # 7. Detection method effectiveness
        plt.subplot(3, 4, 7)
        method_counts = {}
        for session in session_data:
            if hasattr(session, 'anomalies'):
                for anomaly in session.anomalies:
                    method = anomaly.detection_method
                    method_counts[method] = method_counts.get(method, 0) + 1
        
        if method_counts:
            methods = list(method_counts.keys())
            counts = list(method_counts.values())
            
            plt.barh(methods, counts, alpha=0.7)
            plt.xlabel('Number of Detections')
            plt.title('Detection Method Effectiveness')
            
            # Add counts on bars
            for i, count in enumerate(counts):
                plt.text(count + max(counts)*0.01, i, str(count), va='center')
        else:
            plt.text(0.5, 0.5, 'No detection method\ndata available', ha='center', va='center')
            plt.title('Detection Methods')
        
        # 8. Temporal anomaly distribution (if timestamps available)
        plt.subplot(3, 4, 8)
        try:
            anomaly_times = []
            for session in session_data:
                if hasattr(session, 'anomalies') and session.anomalies and hasattr(session, 'start_time'):
                    if session.start_time:
                        anomaly_times.append(session.start_time)
            
            if anomaly_times:
                # Group by hour
                hours = [t.hour for t in anomaly_times if t]
                if hours:
                    hour_counts = pd.Series(hours).value_counts().sort_index()
                    plt.bar(hour_counts.index, hour_counts.values, alpha=0.7, color='purple')
                    plt.xlabel('Hour of Day')
                    plt.ylabel('Number of Anomalies')
                    plt.title('Anomaly Distribution by Hour')
                    plt.xticks(range(0, 24, 2))
                else:
                    plt.text(0.5, 0.5, 'No valid timestamps\navailable', ha='center', va='center')
            else:
                plt.text(0.5, 0.5, 'No temporal data\navailable', ha='center', va='center')
        except Exception as e:
            plt.text(0.5, 0.5, f'Temporal Analysis\nError: {str(e)}', ha='center', va='center')
        plt.title('Temporal Distribution')
        
        # 9. Model processing time comparison (if available)
        plt.subplot(3, 4, 9)
        processing_times = {
            'Isolation Forest': getattr(self.ml_analyzer, 'if_processing_time', 0),
            'One-Class SVM': getattr(self.ml_analyzer, 'svm_processing_time', 0),
            'DBSCAN': getattr(self.ml_analyzer, 'dbscan_processing_time', 0),
            'DeepLog': getattr(self.ml_analyzer, 'deeplog_processing_time', 0)
        }
        
        if any(processing_times.values()):
            models = list(processing_times.keys())
            times = list(processing_times.values())
            
            plt.bar(models, times, alpha=0.7, color='lightblue')
            plt.ylabel('Processing Time (seconds)')
            plt.title('Model Processing Time')
            plt.xticks(rotation=45)
            
            # Add values on bars
            for i, time in enumerate(times):
                if time > 0:
                    plt.text(i, time + max(times)*0.01, f'{time:.2f}s', 
                            ha='center', va='bottom')
        else:
            plt.text(0.5, 0.5, 'Processing time\ndata not available', 
                    ha='center', va='center')
            plt.title('Processing Time')
        
        # 10. Summary statistics
        plt.subplot(3, 4, 10)
        stats_text = f"""
        Ensemble Summary Statistics:
        
        Total Sessions: {len(session_data)}
        Total Anomalies: {sum(len(getattr(s, 'anomalies', [])) for s in session_data)}
        
        Model Results:
        - IF Anomalies: {np.sum(if_predictions == -1) if len(if_predictions) > 0 else 0}
        - SVM Anomalies: {np.sum(svm_predictions == -1) if len(svm_predictions) > 0 else 0}
        - DBSCAN Outliers: {np.sum(dbscan_predictions == -1) if len(dbscan_predictions) > 0 else 0}
        
        Agreement:
        - All Models Agree: {np.sum(ensemble_votes == 3) if len(ensemble_votes) > 0 else 0}
        - Majority Vote: {np.sum(ensemble_votes >= 2) if len(ensemble_votes) > 0 else 0}
        - No Agreement: {np.sum(ensemble_votes == 0) if len(ensemble_votes) > 0 else 0}
        """
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        plt.axis('off')
        plt.title('Summary Statistics')
        
        # 11. Feature space visualization (3D projection)
        plt.subplot(3, 4, 11)
        if embeddings_scaled.shape[1] >= 3:
            pca_3d = PCA(n_components=3).fit_transform(embeddings_scaled)
            
            # Color by ensemble vote
            colors = plt.cm.RdYlGn_r(ensemble_votes / 3.0) if len(ensemble_votes) > 0 else 'blue'
            
            # Create 3D-like visualization in 2D
            plt.scatter(pca_3d[:, 0], pca_3d[:, 1], c=colors, alpha=0.6, s=20)
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title('Feature Space (Colored by Ensemble Votes)')
            
            # Add colorbar
            if len(ensemble_votes) > 0:
                cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='RdYlGn_r'), ax=plt.gca())
                cbar.set_label('Ensemble Votes (0=Normal, 3=All Anomaly)')
        else:
            plt.text(0.5, 0.5, 'Feature space\nvisualization requires\n≥3 dimensions', 
                    ha='center', va='center')
            plt.title('Feature Space')
        
        # 12. Model calibration (if ground truth available)
        plt.subplot(3, 4, 12)
        if ground_truth is not None:
            # Plot precision-recall for each model
            from sklearn.metrics import precision_recall_curve, average_precision_score
            
            scores_data = [
                (anomaly_results.get('if_scores', []), 'Isolation Forest'),
                (anomaly_results.get('svm_scores', []), 'One-Class SVM'),
                (anomaly_results.get('dbscan_scores', []), 'DBSCAN')
            ]
            
            for scores, name in scores_data:
                if len(scores) > 0:
                    try:
                        # Convert scores to probability-like values
                        y_score = -np.array(scores) if name != 'DBSCAN' else np.array(scores)
                        precision, recall, _ = precision_recall_curve(ground_truth, y_score)
                        ap = average_precision_score(ground_truth, y_score)
                        plt.plot(recall, precision, label=f'{name} (AP={ap:.3f})')
                    except:
                        pass
            
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curves')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Model Calibration\n(Requires ground truth)', 
                    ha='center', va='center')
            plt.title('Model Calibration')
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return {
            'visualization': img_base64,
            'metrics': {
                'total_sessions': len(session_data),
                'total_anomalies': sum(len(getattr(s, 'anomalies', [])) for s in session_data),
                'model_agreement': {
                    'all_agree_anomaly': int(np.sum(ensemble_votes == 3)) if len(ensemble_votes) > 0 else 0,
                    'majority_vote_anomaly': int(np.sum(ensemble_votes >= 2)) if len(ensemble_votes) > 0 else 0,
                    'no_agreement': int(np.sum(ensemble_votes == 0)) if len(ensemble_votes) > 0 else 0
                },
                'model_results': {
                    'isolation_forest_anomalies': int(np.sum(if_predictions == -1)) if len(if_predictions) > 0 else 0,
                    'svm_anomalies': int(np.sum(svm_predictions == -1)) if len(svm_predictions) > 0 else 0,
                    'dbscan_outliers': int(np.sum(dbscan_predictions == -1)) if len(dbscan_predictions) > 0 else 0
                }
            },
            'model_type': 'ensemble'
        }
