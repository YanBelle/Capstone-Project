#!/usr/bin/env python3
"""
SVM Visualization and Debugging System for ABM Anomaly Detection
Provides comprehensive visualization and debugging tools for One-Class SVM
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import json
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class OneClassSVMVisualizer:
    """Comprehensive visualization and debugging for One-Class SVM anomaly detection"""
    
    def __init__(self, ml_analyzer):
        self.ml_analyzer = ml_analyzer
        self.svm_model = ml_analyzer.one_class_svm
        self.scaler = ml_analyzer.scaler
        self.pca = ml_analyzer.pca if hasattr(ml_analyzer, 'pca') else None
        
    def visualize_decision_boundary_2d(self, sessions_data: List[Dict], save_path: str = None):
        """Visualize SVM decision boundary in 2D using PCA"""
        
        try:
            # Extract embeddings and labels
            embeddings = np.array([session['embedding'] for session in sessions_data])
            labels = [session.get('is_anomaly', False) for session in sessions_data]
            session_ids = [session.get('session_id', f'Session_{i}') for i, session in enumerate(sessions_data)]
            
            if len(embeddings) == 0:
                logger.warning("No embeddings found for visualization")
                return None
            
            # Scale embeddings
            embeddings_scaled = self.scaler.transform(embeddings)
            
            # Reduce to 2D using PCA
            pca_2d = PCA(n_components=2)
            embeddings_2d = pca_2d.fit_transform(embeddings_scaled)
            
            # Get SVM decision function scores
            decision_scores = self.svm_model.decision_function(embeddings_scaled)
            
            # Create mesh grid for decision boundary
            h = 0.02  # step size in mesh
            x_min, x_max = embeddings_2d[:, 0].min() - 1, embeddings_2d[:, 0].max() + 1
            y_min, y_max = embeddings_2d[:, 1].min() - 1, embeddings_2d[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            
            # Transform mesh grid back to original space for SVM prediction
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            mesh_scaled = pca_2d.inverse_transform(mesh_points)
            mesh_scores = self.svm_model.decision_function(mesh_scaled)
            mesh_scores = mesh_scores.reshape(xx.shape)
            
            # Create interactive plot
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'SVM Decision Boundary & Data Points',
                    'Decision Function Heatmap', 
                    'Anomaly Score Distribution',
                    'Session Details'
                ),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"type": "table"}]]
            )
            
            # Plot 1: Decision boundary with data points
            fig.add_trace(
                go.Contour(
                    x=xx[0], y=yy[:, 0], z=mesh_scores,
                    colorscale='RdBu', showscale=False,
                    contours=dict(start=-2, end=2, size=0.2),
                    opacity=0.3, name='Decision Boundary'
                ), row=1, col=1
            )
            
            # Add data points colored by anomaly status
            colors = ['red' if label else 'blue' for label in labels]
            fig.add_trace(
                go.Scatter(
                    x=embeddings_2d[:, 0], y=embeddings_2d[:, 1],
                    mode='markers',
                    marker=dict(
                        color=colors,
                        size=8,
                        line=dict(width=1, color='black')
                    ),
                    text=[f"ID: {sid}<br>Score: {score:.3f}" for sid, score in zip(session_ids, decision_scores)],
                    hovertemplate='%{text}<extra></extra>',
                    name='Sessions'
                ), row=1, col=1
            )
            
            # Plot 2: Decision function heatmap
            fig.add_trace(
                go.Heatmap(
                    z=mesh_scores, x=xx[0], y=yy[:, 0],
                    colorscale='RdBu', showscale=True,
                    name='Decision Function'
                ), row=1, col=2
            )
            
            # Plot 3: Score distribution
            fig.add_trace(
                go.Histogram(
                    x=decision_scores,
                    nbinsx=20,
                    name='Score Distribution',
                    marker_color='lightblue'
                ), row=2, col=1
            )
            
            # Plot 4: Session details table
            table_data = pd.DataFrame({
                'Session ID': session_ids,
                'Decision Score': [f"{score:.3f}" for score in decision_scores],
                'Prediction': ['Anomaly' if score < 0 else 'Normal' for score in decision_scores],
                'Actual': ['Anomaly' if label else 'Normal' for label in labels]
            })
            
            fig.add_trace(
                go.Table(
                    header=dict(values=list(table_data.columns)),
                    cells=dict(values=[table_data[col] for col in table_data.columns])
                ), row=2, col=2
            )
            
            fig.update_layout(
                title="One-Class SVM Anomaly Detection Analysis",
                height=800,
                showlegend=True
            )
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"SVM visualization saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating SVM visualization: {str(e)}")
            return None
    
    def debug_svm_parameters(self, sessions_data: List[Dict]):
        """Debug SVM model parameters and their effect on detection"""
        
        try:
            embeddings = np.array([session['embedding'] for session in sessions_data])
            embeddings_scaled = self.scaler.transform(embeddings)
            
            # Test different nu values
            nu_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
            gamma_values = ['scale', 'auto', 0.001, 0.01, 0.1, 1.0]
            
            results = []
            
            for nu in nu_values:
                for gamma in gamma_values:
                    try:
                        from sklearn.svm import OneClassSVM
                        test_svm = OneClassSVM(nu=nu, gamma=gamma)
                        predictions = test_svm.fit_predict(embeddings_scaled)
                        decision_scores = test_svm.decision_function(embeddings_scaled)
                        
                        anomaly_count = np.sum(predictions == -1)
                        anomaly_rate = anomaly_count / len(predictions)
                        
                        results.append({
                            'nu': nu,
                            'gamma': gamma,
                            'anomaly_count': anomaly_count,
                            'anomaly_rate': anomaly_rate,
                            'mean_decision_score': np.mean(decision_scores),
                            'std_decision_score': np.std(decision_scores),
                            'min_score': np.min(decision_scores),
                            'max_score': np.max(decision_scores)
                        })
                    except Exception as e:
                        logger.warning(f"Error with nu={nu}, gamma={gamma}: {e}")
            
            # Create visualization
            df = pd.DataFrame(results)
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Anomaly Rate by Parameters',
                    'Score Range by Parameters',
                    'Parameter Sensitivity',
                    'Optimal Parameter Region'
                )
            )
            
            # Heatmap of anomaly rates
            pivot_anomaly = df.pivot(index='nu', columns='gamma', values='anomaly_rate')
            fig.add_trace(
                go.Heatmap(
                    z=pivot_anomaly.values,
                    x=pivot_anomaly.columns,
                    y=pivot_anomaly.index,
                    colorscale='Viridis',
                    name='Anomaly Rate'
                ), row=1, col=1
            )
            
            # Score ranges
            fig.add_trace(
                go.Scatter(
                    x=df['nu'],
                    y=df['std_decision_score'],
                    mode='markers',
                    marker=dict(
                        size=df['anomaly_rate'] * 100,
                        color=df['anomaly_rate'],
                        colorscale='Viridis'
                    ),
                    text=[f"gamma: {g}" for g in df['gamma']],
                    name='Score Std Dev'
                ), row=1, col=2
            )
            
            fig.update_layout(
                title="SVM Parameter Sensitivity Analysis",
                height=600
            )
            
            return fig, df
            
        except Exception as e:
            logger.error(f"Error in parameter debugging: {str(e)}")
            return None, None
    
    def analyze_feature_importance(self, sessions_data: List[Dict], feature_names: List[str] = None):
        """Analyze which features contribute most to SVM decisions"""
        
        try:
            embeddings = np.array([session['embedding'] for session in sessions_data])
            embeddings_scaled = self.scaler.transform(embeddings)
            
            decision_scores = self.svm_model.decision_function(embeddings_scaled)
            
            # Calculate feature correlations with decision scores
            correlations = []
            for i in range(embeddings_scaled.shape[1]):
                corr = np.corrcoef(embeddings_scaled[:, i], decision_scores)[0, 1]
                correlations.append(abs(corr) if not np.isnan(corr) else 0.0)
            
            if feature_names is None:
                feature_names = [f'Feature_{i}' for i in range(len(correlations))]
            
            # Create feature importance plot
            fig = go.Figure()
            
            # Sort by importance
            sorted_indices = np.argsort(correlations)[::-1][:20]  # Top 20 features
            sorted_names = [feature_names[i] for i in sorted_indices]
            sorted_correlations = [correlations[i] for i in sorted_indices]
            
            fig.add_trace(
                go.Bar(
                    x=sorted_names,
                    y=sorted_correlations,
                    marker_color='lightblue',
                    name='Feature Importance'
                )
            )
            
            fig.update_layout(
                title="Feature Importance for SVM Decision Function",
                xaxis_title="Features",
                yaxis_title="Absolute Correlation with Decision Score",
                xaxis_tickangle=-45,
                height=500
            )
            
            return fig, correlations
            
        except Exception as e:
            logger.error(f"Error in feature importance analysis: {str(e)}")
            return None, []
    
    def real_time_svm_monitor(self, new_session_data: Dict):
        """Real-time monitoring of SVM decisions for new sessions"""
        
        try:
            embedding = np.array(new_session_data['embedding']).reshape(1, -1)
            embedding_scaled = self.scaler.transform(embedding)
            
            # Get SVM prediction and score
            prediction = self.svm_model.predict(embedding_scaled)[0]
            decision_score = self.svm_model.decision_function(embedding_scaled)[0]
            
            # Calculate distance to decision boundary
            distance_to_boundary = abs(decision_score)
            
            # Get support vectors info
            n_support_vectors = len(self.svm_model.support_vectors_)
            
            # Create real-time monitoring dashboard
            monitoring_data = {
                'session_id': new_session_data.get('session_id', 'Unknown'),
                'prediction': 'Anomaly' if prediction == -1 else 'Normal',
                'decision_score': float(decision_score),
                'confidence': float(distance_to_boundary),
                'anomaly_probability': float(1 / (1 + np.exp(decision_score))),  # Sigmoid transformation
                'support_vectors_count': n_support_vectors,
                'timestamp': datetime.now().isoformat(),
                'raw_text_length': len(new_session_data.get('raw_text', '')),
                'processing_status': 'Processed'
            }
            
            return monitoring_data
            
        except Exception as e:
            logger.error(f"Error in real-time SVM monitoring: {str(e)}")
            return None
    
    def generate_svm_debug_report(self, sessions_data: List[Dict], output_path: str = "svm_debug_report.html"):
        """Generate comprehensive SVM debugging report"""
        
        try:
            # Collect all visualizations
            boundary_fig = self.visualize_decision_boundary_2d(sessions_data)
            param_fig, param_df = self.debug_svm_parameters(sessions_data)
            
            # Extract feature names from first session if available
            feature_names = None
            if sessions_data and 'feature_names' in sessions_data[0]:
                feature_names = sessions_data[0]['feature_names']
            
            importance_fig, correlations = self.analyze_feature_importance(sessions_data, feature_names)
            
            # Model statistics
            embeddings = np.array([session['embedding'] for session in sessions_data])
            embeddings_scaled = self.scaler.transform(embeddings)
            decision_scores = self.svm_model.decision_function(embeddings_scaled)
            predictions = self.svm_model.predict(embeddings_scaled)
            
            # Create comprehensive HTML report
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>One-Class SVM Debug Report</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                    .header {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                    .section {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                    .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #f8f9fa; border-radius: 5px; border-left: 4px solid #007bff; }}
                    .plot-container {{ width: 100%; height: 600px; }}
                    h1 {{ color: #2c3e50; }}
                    h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>One-Class SVM Anomaly Detection Debug Report</h1>
                    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>Total Sessions Analyzed:</strong> {len(sessions_data)}</p>
                </div>
                
                <div class="section">
                    <h2>Model Configuration</h2>
                    <div class="stats-grid">
                        <div class="metric">
                            <strong>Nu Parameter:</strong><br>
                            {self.svm_model.nu}
                        </div>
                        <div class="metric">
                            <strong>Gamma:</strong><br>
                            {self.svm_model.gamma}
                        </div>
                        <div class="metric">
                            <strong>Kernel:</strong><br>
                            {self.svm_model.kernel}
                        </div>
                        <div class="metric">
                            <strong>Support Vectors:</strong><br>
                            {len(self.svm_model.support_vectors_)}
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Decision Statistics</h2>
                    <div class="stats-grid">
                        <div class="metric">
                            <strong>Mean Score:</strong><br>
                            {np.mean(decision_scores):.3f}
                        </div>
                        <div class="metric">
                            <strong>Std Score:</strong><br>
                            {np.std(decision_scores):.3f}
                        </div>
                        <div class="metric">
                            <strong>Anomalies Detected:</strong><br>
                            {np.sum(predictions == -1)}/{len(predictions)}
                        </div>
                        <div class="metric">
                            <strong>Anomaly Rate:</strong><br>
                            {np.sum(predictions == -1)/len(predictions)*100:.1f}%
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Decision Boundary Analysis</h2>
                    <div id="boundary_plot" class="plot-container"></div>
                </div>
                
                <div class="section">
                    <h2>Parameter Sensitivity Analysis</h2>
                    <div id="param_plot" class="plot-container"></div>
                </div>
                
                <div class="section">
                    <h2>Feature Importance Analysis</h2>
                    <div id="importance_plot" class="plot-container"></div>
                </div>
                
                <script>
                    {f"var boundary_data = {boundary_fig.to_json()};" if boundary_fig else "var boundary_data = null;"}
                    {f"var param_data = {param_fig.to_json()};" if param_fig else "var param_data = null;"}
                    {f"var importance_data = {importance_fig.to_json()};" if importance_fig else "var importance_data = null;"}
                    
                    if (boundary_data) {{
                        Plotly.newPlot('boundary_plot', boundary_data.data, boundary_data.layout);
                    }}
                    if (param_data) {{
                        Plotly.newPlot('param_plot', param_data.data, param_data.layout);
                    }}
                    if (importance_data) {{
                        Plotly.newPlot('importance_plot', importance_data.data, importance_data.layout);
                    }}
                </script>
            </body>
            </html>
            """
            
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"SVM debug report generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating SVM debug report: {str(e)}")
            return None
