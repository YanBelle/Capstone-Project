"""
EJ Unsupervised Integration Module
Integration module to connect unsupervised analysis with existing EJ log pipeline
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import os
import asyncio
from loguru import logger

try:
    from unsupervised_analyzer import UnsupervisedEJAnalyzer
    from unsupervised_visualizer import UnsupervisedEJVisualizer
    unsupervised_available = True
except ImportError as e:
    logger.warning(f"Unsupervised analysis not available: {e}")
    unsupervised_available = False

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
        self.analyzer = None
        self.visualizer = None
        
        if unsupervised_available:
            try:
                self.analyzer = UnsupervisedEJAnalyzer()
                logger.info("✅ EJUnsupervisedIntegration initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize UnsupervisedEJAnalyzer: {e}")
                raise
        else:
            raise ImportError("Unsupervised analysis dependencies not available")
        
    def process_ej_logs(self, ej_logs: List[str], 
                       preprocess: bool = True,
                       visualize: bool = True) -> Dict:
        """
        Process EJ logs through unsupervised analysis
        
        Args:
            ej_logs: Raw or preprocessed EJ log sequences
            preprocess: Whether to preprocess the logs
            visualize: Whether to generate visualizations
            
        Returns:
            Complete analysis results with insights
        """
        try:
            # Step 1: Preprocess if needed (integrate with your existing preprocessing)
            if preprocess and self.existing_pipeline:
                logger.info("Preprocessing logs with existing pipeline...")
                sequences = [self.existing_pipeline.preprocess(log) for log in ej_logs]
            else:
                sequences = ej_logs
            
            # Filter out empty sequences
            sequences = [seq for seq in sequences if seq and seq.strip()]
            
            if not sequences:
                raise ValueError("No valid sequences after preprocessing")
            
            # Step 2: Run unsupervised analysis
            logger.info(f"Analyzing {len(sequences)} EJ log sequences...")
            results = self.analyzer.analyze_sequences(sequences)
            
            # Step 3: Create visualizations
            if visualize:
                logger.info("Creating visualizations...")
                self.visualizer = UnsupervisedEJVisualizer(self.analyzer)
                try:
                    self.visualizer.create_comprehensive_dashboard(interactive=True)
                except Exception as viz_error:
                    logger.warning(f"Interactive visualization failed: {viz_error}")
                    logger.info("Falling back to static visualization...")
                    self.visualizer.create_comprehensive_dashboard(interactive=False)
            
            # Step 4: Generate insights
            logger.info("Generating insights...")
            insights = self._generate_insights(results)
            
            return {
                'results': results,
                'insights': insights,
                'analyzer': self.analyzer,
                'visualizer': self.visualizer,
                'summary': self.analyzer.get_analysis_summary()
            }
            
        except Exception as e:
            logger.error(f"Error processing EJ logs: {e}")
            raise
    
    def _generate_insights(self, results: Dict) -> Dict:
        """Generate actionable insights from results"""
        try:
            insights = {
                'summary': {},
                'alerts': [],
                'recommendations': [],
                'pattern_insights': []
            }
            
            # Summary statistics
            total_sequences = len(self.analyzer.sequences)
            anomaly_rate = results['anomaly_detection']['consensus']['anomaly_rate']
            
            insights['summary'] = {
                'total_transactions': total_sequences,
                'clusters_discovered': results['clustering']['n_clusters'],
                'outliers': results['clustering']['n_noise'],
                'overall_anomaly_rate': f"{anomaly_rate:.1%}",
                'consensus_anomalies': results['anomaly_detection']['consensus']['n_anomalies']
            }
            
            # Generate alerts based on thresholds
            if anomaly_rate > 0.1:  # More than 10% anomalies
                insights['alerts'].append({
                    'level': 'HIGH',
                    'message': f'High anomaly rate detected: {anomaly_rate:.1%} of transactions',
                    'recommendation': 'Investigate recent system changes or data quality issues'
                })
            
            if results['clustering']['noise_ratio'] > 0.3:  # More than 30% noise
                insights['alerts'].append({
                    'level': 'MEDIUM',
                    'message': f'High noise ratio: {results["clustering"]["noise_ratio"]:.1%} of transactions could not be clustered',
                    'recommendation': 'Consider adjusting clustering parameters or data preprocessing'
                })
            
            # Pattern-based alerts
            if 'patterns' in results and not results['patterns'].empty:
                patterns_df = results['patterns']
                
                for _, pattern in patterns_df.iterrows():
                    if pattern['anomaly_ratio'] > 0.5 and pattern['size'] > total_sequences * 0.05:
                        insights['alerts'].append({
                            'level': 'MEDIUM',
                            'message': f'Pattern "{pattern["pattern_signature"]}" has high anomaly rate: {pattern["anomaly_ratio"]:.1%}',
                            'recommendation': f'Investigate {pattern["pattern_signature"]} pattern for potential issues'
                        })
                
                # Pattern insights
                for _, pattern in patterns_df.head(5).iterrows():
                    insights['pattern_insights'].append({
                        'pattern': pattern['pattern_signature'],
                        'size': pattern['size'],
                        'percentage': f"{pattern['percentage']:.1f}%",
                        'anomaly_rate': f"{pattern['anomaly_ratio']:.1%}",
                        'description': self._describe_pattern(pattern)
                    })
            
            # Recommendations based on analysis
            if results['clustering']['n_noise'] > total_sequences * 0.2:
                insights['recommendations'].append(
                    "Consider data preprocessing improvements - high number of unclustered sequences suggests data quality issues"
                )
            
            # Check for specific error patterns
            if 'patterns' in results and not results['patterns'].empty:
                patterns_df = results['patterns']
                
                if 'DEVICE_ERROR' in patterns_df['pattern_signature'].values:
                    error_pattern = patterns_df[patterns_df['pattern_signature'] == 'DEVICE_ERROR'].iloc[0]
                    insights['recommendations'].append(
                        f"Device errors found in {error_pattern['percentage']:.1f}% of transactions - schedule maintenance"
                    )
                
                if 'AUTH_FAILURE' in patterns_df['pattern_signature'].values:
                    auth_pattern = patterns_df[patterns_df['pattern_signature'] == 'AUTH_FAILURE'].iloc[0]
                    insights['recommendations'].append(
                        f"Authentication failures in {auth_pattern['percentage']:.1f}% of transactions - review security protocols"
                    )
                
                if 'TIMEOUT_ERROR' in patterns_df['pattern_signature'].values:
                    timeout_pattern = patterns_df[patterns_df['pattern_signature'] == 'TIMEOUT_ERROR'].iloc[0]
                    insights['recommendations'].append(
                        f"Timeout errors in {timeout_pattern['percentage']:.1f}% of transactions - optimize network/processing speed"
                    )
            
            # Quality recommendations
            if 'metrics' in results:
                metrics = results['metrics']
                
                if 'silhouette_score' in metrics and metrics['silhouette_score'] < 0.3:
                    insights['recommendations'].append(
                        "Low clustering quality detected - consider adjusting clustering parameters or data preprocessing"
                    )
                
                if 'anomaly_agreement' in metrics and metrics['anomaly_agreement'] < 0.7:
                    insights['recommendations'].append(
                        "Low agreement between anomaly detection methods - results may be less reliable"
                    )
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {
                'summary': {'error': str(e)},
                'alerts': [{'level': 'ERROR', 'message': f'Error generating insights: {e}'}],
                'recommendations': [],
                'pattern_insights': []
            }
    
    def _describe_pattern(self, pattern: pd.Series) -> str:
        """Generate a description for a pattern"""
        try:
            pattern_type = pattern['pattern_signature']
            size = pattern['size']
            percentage = pattern['percentage']
            anomaly_rate = pattern['anomaly_ratio']
            
            if pattern_type == 'SUCCESSFUL_TRANSACTION':
                return f"Normal successful transactions ({size} sequences, {percentage:.1f}% of total)"
            elif pattern_type == 'DEVICE_ERROR':
                return f"Device malfunction pattern with {anomaly_rate:.1%} anomaly rate"
            elif pattern_type == 'AUTH_FAILURE':
                return f"Authentication failure pattern - potential security concern"
            elif pattern_type == 'INCOMPLETE_TRANSACTION':
                return f"Incomplete transactions - may indicate user abandonment or technical issues"
            elif pattern_type == 'TIMEOUT_ERROR':
                return f"Timeout-related errors - network or processing delays"
            elif pattern_type == 'UNCLUSTERED_NOISE':
                return f"Diverse/outlier transactions that don't fit standard patterns"
            else:
                return f"Pattern with {anomaly_rate:.1%} anomaly rate - requires investigation"
                
        except Exception as e:
            return f"Pattern description unavailable: {e}"
    
    def export_results(self, output_dir: str = './ej_analysis_output'):
        """Export analysis results and visualizations"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            if not self.analyzer or not self.analyzer.results:
                raise ValueError("No analysis results available to export")
            
            # Export patterns
            if 'patterns' in self.analyzer.results and not self.analyzer.results['patterns'].empty:
                patterns_path = os.path.join(output_dir, 'discovered_patterns.csv')
                self.analyzer.results['patterns'].to_csv(patterns_path, index=False)
                logger.info(f"Patterns exported to {patterns_path}")
            
            # Export anomalies
            anomalies = self.analyzer.get_anomalous_sequences()
            if anomalies:
                anomaly_df = pd.DataFrame(anomalies, columns=['index', 'sequence'])
                anomaly_path = os.path.join(output_dir, 'anomalous_sequences.csv')
                anomaly_df.to_csv(anomaly_path, index=False)
                logger.info(f"Anomalies exported to {anomaly_path}")
            
            # Export cluster assignments
            cluster_data = []
            for i, (seq, label) in enumerate(zip(self.analyzer.sequences, self.analyzer.results['clustering']['labels'])):
                cluster_data.append({
                    'sequence_index': i,
                    'cluster_id': label,
                    'cluster_name': f'Cluster_{label}' if label != -1 else 'Noise',
                    'sequence_preview': seq[:100] + '...' if len(seq) > 100 else seq
                })
            
            cluster_df = pd.DataFrame(cluster_data)
            cluster_path = os.path.join(output_dir, 'cluster_assignments.csv')
            cluster_df.to_csv(cluster_path, index=False)
            logger.info(f"Cluster assignments exported to {cluster_path}")
            
            # Export metrics and summary
            if self.visualizer:
                report = self.visualizer.generate_analysis_report()
                report_path = os.path.join(output_dir, 'analysis_report.csv')
                report.to_csv(report_path, index=False)
                logger.info(f"Analysis report exported to {report_path}")
            
            # Export visualizations
            if self.visualizer:
                viz_dir = os.path.join(output_dir, 'visualizations')
                self.visualizer.export_visualizations(viz_dir)
            
            # Export raw results as JSON
            import json
            
            # Convert numpy arrays to lists for JSON serialization
            export_results = {}
            for key, value in self.analyzer.results.items():
                if key == 'patterns' and isinstance(value, pd.DataFrame):
                    export_results[key] = value.to_dict('records')
                elif isinstance(value, dict):
                    export_results[key] = {}
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, np.ndarray):
                            export_results[key][subkey] = subvalue.tolist()
                        else:
                            export_results[key][subkey] = subvalue
                else:
                    export_results[key] = value
            
            json_path = os.path.join(output_dir, 'analysis_results.json')
            with open(json_path, 'w') as f:
                json.dump(export_results, f, indent=2, default=str)
            logger.info(f"Raw results exported to {json_path}")
            
            logger.info(f"✅ All results exported to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            raise
    
    def compare_with_labeled_data(self, labels: Optional[List[str]] = None) -> Dict:
        """
        Optional: Compare unsupervised results with known labels if available
        
        Args:
            labels: Known labels for sequences (if available)
            
        Returns:
            Comparison metrics
        """
        if labels is None or not self.analyzer or not self.analyzer.results:
            return {'message': 'No labeled data or analysis results available for comparison'}
        
        try:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
            
            cluster_labels = self.analyzer.results['clustering']['labels']
            
            if len(labels) != len(cluster_labels):
                return {'error': 'Label length mismatch with analysis results'}
            
            # Calculate agreement metrics
            comparison = {
                'adjusted_rand_score': adjusted_rand_score(labels, cluster_labels),
                'normalized_mutual_info': normalized_mutual_info_score(labels, cluster_labels)
            }
            
            # Check if anomalies align with known failures
            unique_labels = set(labels)
            if any('fail' in str(label).lower() or 'error' in str(label).lower() for label in unique_labels):
                failure_indices = [i for i, label in enumerate(labels) 
                                 if 'fail' in str(label).lower() or 'error' in str(label).lower()]
                
                if failure_indices:
                    iso_anomalies = self.analyzer.results['anomaly_detection']['isolation_forest']['anomalies']
                    detected_failures = sum(iso_anomalies[i] for i in failure_indices)
                    failure_detection_rate = detected_failures / len(failure_indices)
                    
                    comparison['failure_detection_rate'] = failure_detection_rate
            
            logger.info("✅ Comparison with labeled data completed")
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing with labeled data: {e}")
            return {'error': str(e)}
    
    def get_sequence_details(self, sequence_idx: int) -> Dict:
        """Get detailed information about a specific sequence"""
        try:
            if not self.analyzer or sequence_idx >= len(self.analyzer.sequences):
                return {'error': 'Invalid sequence index or no analysis available'}
            
            sequence = self.analyzer.sequences[sequence_idx]
            cluster_id = self.analyzer.results['clustering']['labels'][sequence_idx]
            
            # Get anomaly scores
            iso_score = self.analyzer.results['anomaly_detection']['isolation_forest']['scores'][sequence_idx]
            lof_score = self.analyzer.results['anomaly_detection']['lof']['scores'][sequence_idx]
            stat_distance = self.analyzer.results['anomaly_detection']['statistical']['distances'][sequence_idx]
            
            # Check if it's an anomaly by different methods
            is_iso_anomaly = self.analyzer.results['anomaly_detection']['isolation_forest']['anomalies'][sequence_idx]
            is_lof_anomaly = self.analyzer.results['anomaly_detection']['lof']['anomalies'][sequence_idx]
            is_stat_anomaly = self.analyzer.results['anomaly_detection']['statistical']['anomalies'][sequence_idx]
            is_consensus_anomaly = self.analyzer.results['anomaly_detection']['consensus']['anomalies'][sequence_idx]
            
            # Get pattern information
            pattern_info = None
            if cluster_id != -1 and 'patterns' in self.analyzer.results:
                patterns_df = self.analyzer.results['patterns']
                cluster_patterns = patterns_df[patterns_df['cluster_id'] == cluster_id]
                if not cluster_patterns.empty:
                    pattern_info = cluster_patterns.iloc[0].to_dict()
            
            details = {
                'sequence_index': sequence_idx,
                'sequence_text': sequence,
                'sequence_length': len(sequence.split()),
                'cluster_id': cluster_id,
                'cluster_name': f'Cluster_{cluster_id}' if cluster_id != -1 else 'Noise',
                'anomaly_scores': {
                    'isolation_forest': float(iso_score),
                    'lof': float(lof_score),
                    'statistical_distance': float(stat_distance)
                },
                'anomaly_flags': {
                    'isolation_forest': bool(is_iso_anomaly),
                    'lof': bool(is_lof_anomaly),
                    'statistical': bool(is_stat_anomaly),
                    'consensus': bool(is_consensus_anomaly)
                },
                'pattern_info': pattern_info
            }
            
            return details
            
        except Exception as e:
            logger.error(f"Error getting sequence details: {e}")
            return {'error': str(e)}
    
    async def analyze_session_from_database(self, session_id: str, get_db_connection) -> Dict:
        """Analyze a specific session from the database"""
        try:
            # Get session data from database
            async with get_db_connection() as conn:
                session_data = await conn.fetchrow(
                    "SELECT session_id, raw_text FROM ml_sessions WHERE session_id = $1",
                    session_id
                )
            
            if not session_data or not session_data['raw_text']:
                return {'error': f'No raw text found for session {session_id}'}
            
            raw_text = session_data['raw_text']
            
            # Analyze single sequence
            analyzer = UnsupervisedEJAnalyzer()
            results = analyzer.analyze_sequences([raw_text], perform_dim_reduction=False)
            
            # Get insights for this single sequence
            insights = {
                'session_id': session_id,
                'is_anomaly': {
                    'isolation_forest': bool(results['anomaly_detection']['isolation_forest']['anomalies'][0]),
                    'lof': bool(results['anomaly_detection']['lof']['anomalies'][0]),
                    'statistical': bool(results['anomaly_detection']['statistical']['anomalies'][0]),
                    'consensus': bool(results['anomaly_detection']['consensus']['anomalies'][0])
                },
                'anomaly_scores': {
                    'isolation_forest': float(results['anomaly_detection']['isolation_forest']['scores'][0]),
                    'lof': float(results['anomaly_detection']['lof']['scores'][0]),
                    'statistical_distance': float(results['anomaly_detection']['statistical']['distances'][0])
                },
                'sequence_length': len(raw_text.split()),
                'raw_text_preview': raw_text[:200] + '...' if len(raw_text) > 200 else raw_text
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error analyzing session from database: {e}")
            return {'error': str(e)}
