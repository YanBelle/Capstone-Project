#!/usr/bin/env python3
"""
Debug script to test cluster_sessions functionality directly
"""

import sys
import os
import logging
import json
import traceback

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.api.enhanced_ensemble_detector import EnhancedEnsembleDetector

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_cluster_sessions():
    """Test the get_cluster_sessions method directly"""
    try:
        # Initialize the detector
        detector = EnhancedEnsembleDetector()
        
        # Try to load the model
        model_path = "./models/enhanced_ensemble_model.pkl"
        if not os.path.exists(model_path):
            logger.error("Model file not found: %s", model_path)
            return
        
        success = detector.load_model(model_path)
        if not success:
            logger.error("Failed to load model")
            return
        
        logger.info("Model loaded successfully. Is trained: %s", detector.is_trained)
        
        # Check available attributes
        logger.info("Available cluster attributes:")
        for attr in ['text_cluster_labels', 'numerical_cluster_labels', 'combined_cluster_labels']:
            if hasattr(detector, attr):
                labels = getattr(detector, attr)
                unique_labels = set(labels) if labels is not None else set()
                logger.info(f"  {attr}: {len(labels) if labels else 0} labels, unique: {unique_labels}")
            else:
                logger.info(f"  {attr}: NOT FOUND")
        
        # Check training sessions
        if hasattr(detector, 'training_sessions'):
            logger.info(f"Training sessions: {len(detector.training_sessions)} sessions")
            if detector.training_sessions:
                logger.info(f"First session keys: {list(detector.training_sessions[0].keys())}")
        else:
            logger.info("No training_sessions attribute found")
        
        # Try to get cluster sessions for different clusters and feature types
        for feature_type in ['text', 'numerical', 'combined']:
            try:
                # Find available cluster IDs for this feature type
                if feature_type == 'text' and hasattr(detector, 'text_cluster_labels'):
                    labels = detector.text_cluster_labels
                elif feature_type == 'numerical' and hasattr(detector, 'numerical_cluster_labels'):
                    labels = detector.numerical_cluster_labels
                elif feature_type == 'combined' and hasattr(detector, 'combined_cluster_labels'):
                    labels = detector.combined_cluster_labels
                else:
                    logger.info(f"No labels found for {feature_type}")
                    continue
                
                if labels:
                    unique_clusters = sorted(set(labels))
                    logger.info(f"Testing {feature_type} clusters: {unique_clusters}")
                    
                    # Test first non-noise cluster (not -1)
                    test_clusters = [c for c in unique_clusters if c != -1][:2]  # Test first 2 non-noise clusters
                    
                    for cluster_id in test_clusters:
                        logger.info(f"\nTesting cluster {cluster_id} for {feature_type} features:")
                        try:
                            sessions = detector.get_cluster_sessions(cluster_id, feature_type)
                            logger.info(f"  Found {len(sessions)} sessions")
                            if sessions:
                                logger.info(f"  First session keys: {list(sessions[0].keys())}")
                                logger.info(f"  Sample session: {json.dumps(sessions[0], indent=2, default=str)}")
                        except Exception as e:
                            logger.error(f"  Error getting sessions for cluster {cluster_id}: {e}")
                            logger.error(f"  Traceback: {traceback.format_exc()}")
                
            except Exception as e:
                logger.error(f"Error testing {feature_type}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
        
    except Exception as e:
        logger.error(f"Error in test: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    test_cluster_sessions()
