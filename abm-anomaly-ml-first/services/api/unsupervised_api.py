"""
Unsupervised EJ Analysis API Module
API endpoints for unsupervised EJ log analysis functionality
"""

from fastapi import HTTPException
from typing import Dict, Any, List, Optional
import asyncio
import glob
import os
import json
from loguru import logger

# Unsupervised Analysis Integration
try:
    from unsupervised_integration import EJUnsupervisedIntegration
    unsupervised_available = True
except ImportError as e:
    logger.warning(f"Unsupervised analysis not available: {e}")
    unsupervised_available = False

# Global integration instance
unsupervised_integration = None

def get_unsupervised_integration():
    """Get or create the global unsupervised integration instance"""
    global unsupervised_integration
    if unsupervised_integration is None and unsupervised_available:
        try:
            unsupervised_integration = EJUnsupervisedIntegration()
            logger.info("✅ Unsupervised integration initialized")
        except Exception as e:
            logger.error(f"Failed to initialize unsupervised integration: {e}")
            return None
    return unsupervised_integration

async def run_unsupervised_analysis(get_db_connection, limit: int = 1000, include_files: bool = True) -> Dict:
    """
    Run unsupervised analysis on available EJ log data
    
    Args:
        get_db_connection: Database connection function
        limit: Maximum number of sessions to analyze
        include_files: Whether to include file system data if database is insufficient
        
    Returns:
        Complete analysis results with insights
    """
    try:
        if not unsupervised_available:
            return {
                "status": "error",
                "message": "Unsupervised analysis not available - dependencies missing"
            }
        
        integration = get_unsupervised_integration()
        if not integration:
            return {
                "status": "error",
                "message": "Failed to initialize unsupervised analysis"
            }
        
        # Collect EJ log sequences
        sequences = []
        
        # Get data from database first
        try:
            async with get_db_connection() as conn:
                db_sessions = await conn.fetch("""
                    SELECT session_id, raw_text 
                    FROM ml_sessions 
                    WHERE raw_text IS NOT NULL 
                    AND raw_text != 'Raw text not available'
                    AND length(raw_text) > 10
                    ORDER BY created_at DESC
                    LIMIT $1
                """, limit)
            
            for session in db_sessions:
                if session['raw_text'] and len(session['raw_text'].strip()) > 10:
                    sequences.append(session['raw_text'].strip())
            
            logger.info(f"Collected {len(sequences)} sequences from database")
            
        except Exception as db_error:
            logger.warning(f"Database collection failed: {db_error}")
        
        # If insufficient data from database, supplement with files
        if len(sequences) < 20 and include_files:
            logger.info("Supplementing with file system data...")
            
            file_patterns = [
                "/app/input/processed/*.txt",
                "/app/input/*.txt",
                "/app/data/sessions/*.txt"
            ]
            
            for pattern in file_patterns:
                files = glob.glob(pattern)
                files_added = 0
                
                for file_path in files[:100]:  # Limit file reads
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            if content and len(content) > 10 and content not in sequences:
                                sequences.append(content)
                                files_added += 1
                                
                                if len(sequences) >= limit:
                                    break
                                    
                    except Exception as file_error:
                        logger.warning(f"Error reading file {file_path}: {file_error}")
                        continue
                
                logger.info(f"Added {files_added} sequences from {pattern}")
                
                if len(sequences) >= limit:
                    break
        
        if len(sequences) < 5:
            return {
                "status": "error",
                "message": f"Insufficient data for analysis. Found {len(sequences)} sequences, need at least 5."
            }
        
        # Limit sequences to avoid memory issues
        if len(sequences) > limit:
            sequences = sequences[:limit]
        
        logger.info(f"Starting unsupervised analysis on {len(sequences)} sequences")
        
        # Run unsupervised analysis
        analysis_results = integration.process_ej_logs(
            ej_logs=sequences,
            preprocess=False,  # Assume already preprocessed
            visualize=False    # Don't auto-visualize in API
        )
        
        # Extract key metrics for API response
        summary = analysis_results['summary']
        insights = analysis_results['insights']
        
        # Get top patterns
        top_patterns = []
        if 'patterns' in analysis_results['results'] and not analysis_results['results']['patterns'].empty:
            patterns_df = analysis_results['results']['patterns'].head(10)
            for _, pattern in patterns_df.iterrows():
                top_patterns.append({
                    'pattern_type': pattern['pattern_signature'],
                    'size': int(pattern['size']),
                    'percentage': round(pattern['percentage'], 1),
                    'anomaly_rate': round(pattern['anomaly_ratio'] * 100, 1),
                    'exemplar': pattern['exemplar_sequence'][:100] + '...' if len(pattern['exemplar_sequence']) > 100 else pattern['exemplar_sequence']
                })
        
        # Get anomalies summary
        anomalies_summary = {}
        for method in ['isolation_forest', 'lof', 'statistical', 'consensus']:
            if method in analysis_results['results']['anomaly_detection']:
                anomalies_summary[method] = {
                    'count': int(analysis_results['results']['anomaly_detection'][method]['n_anomalies']),
                    'rate': round(analysis_results['results']['anomaly_detection'][method]['anomaly_rate'] * 100, 1)
                }
        
        return {
            "status": "success",
            "message": f"Unsupervised analysis completed on {len(sequences)} sequences",
            "summary": summary,
            "insights": insights,
            "top_patterns": top_patterns,
            "anomalies_summary": anomalies_summary,
            "analysis_id": id(integration),  # Simple ID for tracking
            "sequences_analyzed": len(sequences)
        }
        
    except Exception as e:
        logger.error(f"Error in unsupervised analysis: {str(e)}")
        return {
            "status": "error",
            "message": f"Analysis failed: {str(e)}"
        }

async def get_unsupervised_anomalies(method: str = 'consensus', limit: int = 50) -> Dict:
    """
    Get anomalous sequences detected by unsupervised analysis
    
    Args:
        method: Anomaly detection method ('consensus', 'isolation_forest', 'lof', 'statistical')
        limit: Maximum number of anomalies to return
        
    Returns:
        List of anomalous sequences with details
    """
    try:
        if not unsupervised_available:
            return {
                "status": "error",
                "message": "Unsupervised analysis not available"
            }
        
        integration = get_unsupervised_integration()
        if not integration or not integration.analyzer or not integration.analyzer.results:
            return {
                "status": "error", 
                "message": "No analysis results available. Run analysis first."
            }
        
        # Get anomalous sequences
        anomalies = integration.analyzer.get_anomalous_sequences(method=method)
        
        if not anomalies:
            return {
                "status": "success",
                "message": f"No anomalies found using {method} method",
                "anomalies": []
            }
        
        # Limit results
        if len(anomalies) > limit:
            anomalies = anomalies[:limit]
        
        # Format anomalies with additional details
        formatted_anomalies = []
        for idx, sequence in anomalies:
            anomaly_details = integration.get_sequence_details(idx)
            formatted_anomalies.append({
                'sequence_index': idx,
                'sequence_preview': sequence[:200] + '...' if len(sequence) > 200 else sequence,
                'full_sequence': sequence,
                'anomaly_scores': anomaly_details.get('anomaly_scores', {}),
                'anomaly_flags': anomaly_details.get('anomaly_flags', {}),
                'cluster_info': {
                    'cluster_id': anomaly_details.get('cluster_id'),
                    'cluster_name': anomaly_details.get('cluster_name')
                }
            })
        
        return {
            "status": "success",
            "method": method,
            "total_anomalies": len(formatted_anomalies),
            "anomalies": formatted_anomalies
        }
        
    except Exception as e:
        logger.error(f"Error getting anomalies: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to get anomalies: {str(e)}"
        }

async def get_unsupervised_patterns() -> Dict:
    """Get discovered patterns from unsupervised analysis"""
    try:
        if not unsupervised_available:
            return {
                "status": "error",
                "message": "Unsupervised analysis not available"
            }
        
        integration = get_unsupervised_integration()
        if not integration or not integration.analyzer or not integration.analyzer.results:
            return {
                "status": "error",
                "message": "No analysis results available. Run analysis first."
            }
        
        if 'patterns' not in integration.analyzer.results:
            return {
                "status": "success",
                "message": "No patterns discovered",
                "patterns": []
            }
        
        patterns_df = integration.analyzer.results['patterns']
        
        if patterns_df.empty:
            return {
                "status": "success", 
                "message": "No patterns discovered",
                "patterns": []
            }
        
        # Format patterns for API response
        patterns = []
        for _, pattern in patterns_df.iterrows():
            # Get sample sequences from this pattern
            cluster_sequences = integration.analyzer.get_cluster_sequences(pattern['cluster_id'])
            sample_sequences = [seq for _, seq in cluster_sequences[:3]]  # First 3 sequences
            
            patterns.append({
                'cluster_id': int(pattern['cluster_id']) if pattern['cluster_id'] != -1 else -1,
                'pattern_type': pattern['pattern_signature'],
                'size': int(pattern['size']),
                'percentage': round(pattern['percentage'], 1),
                'anomaly_rate': round(pattern['anomaly_ratio'] * 100, 1),
                'common_tokens': pattern['common_tokens'][:5] if pattern['common_tokens'] else [],
                'exemplar_sequence': pattern['exemplar_sequence'],
                'sample_sequences': [seq[:100] + '...' if len(seq) > 100 else seq for seq in sample_sequences],
                'description': integration._describe_pattern(pattern)
            })
        
        return {
            "status": "success",
            "total_patterns": len(patterns),
            "patterns": patterns
        }
        
    except Exception as e:
        logger.error(f"Error getting patterns: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to get patterns: {str(e)}"
        }

async def analyze_single_session_unsupervised(session_id: str, get_db_connection) -> Dict:
    """
    Analyze a single session using unsupervised methods
    
    Args:
        session_id: Session ID to analyze
        get_db_connection: Database connection function
        
    Returns:
        Analysis results for the single session
    """
    try:
        if not unsupervised_available:
            return {
                "status": "error",
                "message": "Unsupervised analysis not available"
            }
        
        integration = get_unsupervised_integration()
        if not integration:
            return {
                "status": "error",
                "message": "Failed to initialize unsupervised analysis"
            }
        
        # Analyze session from database
        result = await integration.analyze_session_from_database(session_id, get_db_connection)
        
        if 'error' in result:
            return {
                "status": "error",
                "message": result['error']
            }
        
        return {
            "status": "success",
            "session_analysis": result
        }
        
    except Exception as e:
        logger.error(f"Error analyzing session {session_id}: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to analyze session: {str(e)}"
        }

async def get_unsupervised_status() -> Dict:
    """Get status of unsupervised analysis system"""
    try:
        status_info = {
            "unsupervised_available": unsupervised_available,
            "integration_initialized": unsupervised_integration is not None
        }
        
        if unsupervised_available and unsupervised_integration:
            # Check if analysis has been run
            if (unsupervised_integration.analyzer and 
                unsupervised_integration.analyzer.results):
                
                results = unsupervised_integration.analyzer.results
                status_info.update({
                    "analysis_completed": True,
                    "sequences_analyzed": len(unsupervised_integration.analyzer.sequences),
                    "clusters_found": results['clustering']['n_clusters'],
                    "anomalies_found": results['anomaly_detection']['consensus']['n_anomalies'],
                    "patterns_discovered": len(results['patterns']) if 'patterns' in results else 0
                })
            else:
                status_info.update({
                    "analysis_completed": False,
                    "message": "No analysis has been run yet"
                })
        else:
            status_info.update({
                "message": "Unsupervised analysis dependencies not available"
            })
        
        return {
            "status": "success",
            "system_status": status_info
        }
        
    except Exception as e:
        logger.error(f"Error getting unsupervised status: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to get status: {str(e)}"
        }

async def export_unsupervised_results(output_format: str = 'json') -> Dict:
    """Export unsupervised analysis results"""
    try:
        if not unsupervised_available:
            return {
                "status": "error",
                "message": "Unsupervised analysis not available"
            }
        
        integration = get_unsupervised_integration()
        if not integration or not integration.analyzer or not integration.analyzer.results:
            return {
                "status": "error",
                "message": "No analysis results available to export"
            }
        
        # Export to temporary directory
        output_dir = f"/tmp/ej_unsupervised_export_{id(integration)}"
        integration.export_results(output_dir)
        
        # List exported files
        exported_files = []
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, output_dir)
                exported_files.append(relative_path)
        
        return {
            "status": "success",
            "message": f"Results exported to {output_dir}",
            "export_directory": output_dir,
            "exported_files": exported_files
        }
        
    except Exception as e:
        logger.error(f"Error exporting results: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to export results: {str(e)}"
        }

async def create_unsupervised_dashboard() -> Dict:
    """Create interactive dashboard for unsupervised analysis results"""
    try:
        if not unsupervised_available:
            return {
                "status": "error",
                "message": "Unsupervised analysis not available"
            }
        
        integration = get_unsupervised_integration()
        if not integration or not integration.analyzer or not integration.analyzer.results:
            return {
                "status": "error",
                "message": "No analysis results available for visualization"
            }
        
        # Create visualizer if not exists
        if not integration.visualizer:
            from unsupervised_visualizer import UnsupervisedEJVisualizer
            integration.visualizer = UnsupervisedEJVisualizer(integration.analyzer)
        
        # Create dashboard
        try:
            integration.visualizer.create_comprehensive_dashboard(interactive=True)
            dashboard_status = "Interactive dashboard created successfully"
        except Exception as viz_error:
            logger.warning(f"Interactive dashboard failed: {viz_error}")
            integration.visualizer.create_comprehensive_dashboard(interactive=False)
            dashboard_status = "Static dashboard created (interactive mode failed)"
        
        return {
            "status": "success",
            "message": dashboard_status
        }
        
    except Exception as e:
        logger.error(f"Error creating dashboard: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to create dashboard: {str(e)}"
        }
