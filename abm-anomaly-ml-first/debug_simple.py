#!/usr/bin/env python3
"""Debug script to test cluster_sessions functionality"""

import sys
import os
import logging

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enhanced_ensemble_detector import EnhancedEnsembleDetector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test():
    try:
        detector = EnhancedEnsembleDetector()
        
        model_path = "./models/enhanced_ensemble_model.pkl"
        if not os.path.exists(model_path):
            logger.error("Model file not found")
            return
        
        success = detector.load_model(model_path)
        if not success:
            logger.error("Failed to load model")
            return
        
        logger.info("Model loaded. Is trained: %s", detector.is_trained)
        
        # Check combined cluster labels
        if hasattr(detector, 'combined_cluster_labels'):
            labels = detector.combined_cluster_labels
            if labels:
                unique_clusters = sorted(set(labels))
                logger.info("Clusters: %s", unique_clusters)
                
                # Test first non-noise cluster
                test_clusters = [c for c in unique_clusters if c != -1]
                if test_clusters:
                    cluster_id = test_clusters[0]
                    logger.info("Testing cluster %s", cluster_id)
                    
                    sessions = detector.get_cluster_sessions(cluster_id, 'combined')
                    logger.info("Found %d sessions", len(sessions))
                else:
                    logger.error("No non-noise clusters")
            else:
                logger.error("No labels")
        else:
            logger.error("No combined_cluster_labels attribute")
        
    except Exception as e:
        logger.error("Error: %s", str(e))
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    test()
