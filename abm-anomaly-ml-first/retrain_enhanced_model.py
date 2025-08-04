#!/usr/bin/env python3
"""
Retrain Enhanced Ensemble Model Script
Retrains the enhanced ensemble detector with sample training data
"""

import json
import requests
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_training_data():
    """Load training data from sample_training_data.json"""
    try:
        with open('sample_training_data.json', 'r') as f:
            data = json.load(f)
        
        sessions = data.get('sessions', [])
        logger.info(f"Loaded {len(sessions)} training sessions")
        return sessions
    
    except FileNotFoundError:
        logger.error("sample_training_data.json not found")
        return []
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        return []

def retrain_model(sessions):
    """Send training request to API"""
    try:
        url = "http://localhost:8000/api/train_enhanced_ensemble"
        
        payload = {
            "sessions": sessions
        }
        
        logger.info(f"Sending training request with {len(sessions)} sessions...")
        response = requests.post(url, json=payload, timeout=300)  # 5 minute timeout
        
        if response.status_code == 200:
            result = response.json()
            logger.info("Training completed successfully!")
            
            # Print training statistics
            if 'training_stats' in result:
                stats = result['training_stats']
                logger.info(f"Training Statistics:")
                logger.info(f"  - Sessions processed: {stats.get('n_sessions', 'Unknown')}")
                logger.info(f"  - Text features shape: {stats.get('text_features_shape', 'Unknown')}")
                logger.info(f"  - Numerical features shape: {stats.get('numerical_features_shape', 'Unknown')}")
                logger.info(f"  - Combined features shape: {stats.get('combined_features_shape', 'Unknown')}")
                logger.info(f"  - PCA explained variance: {stats.get('pca_explained_variance', 'Unknown'):.3f}")
            
            # Print clustering results
            if 'cluster_results' in result:
                cluster_results = result['cluster_results']
                logger.info(f"Clustering Results:")
                
                for feature_type, results in cluster_results.items():
                    logger.info(f"  {feature_type.upper()} Features:")
                    logger.info(f"    - Clusters: {results.get('n_clusters', 0)}")
                    logger.info(f"    - Noise points: {results.get('n_noise', 0)}")
                    logger.info(f"    - Noise ratio: {results.get('noise_ratio', 0):.1%}")
                    logger.info(f"    - Silhouette score: {results.get('silhouette_score', -1):.3f}")
                    logger.info(f"    - Parameters: eps={results.get('eps', 0):.2f}, min_samples={results.get('min_samples', 0)}")
                    
                    # Show cluster distribution
                    if 'cluster_info' in results and results['cluster_info']:
                        logger.info(f"    - Cluster sizes:")
                        for cluster_id, info in results['cluster_info'].items():
                            logger.info(f"      Cluster {cluster_id}: {info['size']} points ({info['percentage']:.1f}%)")
            
            return True
        else:
            logger.error(f"Training failed with status {response.status_code}: {response.text}")
            return False
    
    except requests.exceptions.Timeout:
        logger.error("Training request timed out")
        return False
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return False

def check_model_status():
    """Check if model is trained"""
    try:
        url = "http://localhost:8000/api/model_info"
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            info = response.json()
            is_trained = info.get('is_trained', False)
            logger.info(f"Model trained status: {is_trained}")
            
            if is_trained:
                logger.info(f"Training timestamp: {info.get('training_timestamp', 'Unknown')}")
                training_stats = info.get('training_stats', {})
                if training_stats:
                    logger.info(f"Number of training sessions: {training_stats.get('n_sessions', 'Unknown')}")
            
            return is_trained
        else:
            logger.error(f"Failed to get model status: {response.status_code}")
            return False
    
    except Exception as e:
        logger.error(f"Error checking model status: {e}")
        return False

def main():
    """Main execution function"""
    logger.info("Starting enhanced ensemble model retraining...")
    
    # Check current model status
    logger.info("Checking current model status...")
    is_trained = check_model_status()
    
    if is_trained:
        logger.info("Model is already trained. Proceeding with retraining...")
    else:
        logger.info("Model is not trained. Starting initial training...")
    
    # Load training data
    sessions = load_training_data()
    if not sessions:
        logger.error("No training data available. Exiting.")
        sys.exit(1)
    
    # Retrain the model
    success = retrain_model(sessions)
    
    if success:
        logger.info("Model retraining completed successfully!")
        
        # Verify the model is now trained
        logger.info("Verifying model status...")
        if check_model_status():
            logger.info("✅ Model is now trained and ready for use!")
        else:
            logger.warning("⚠️  Model training may not have persisted correctly")
    else:
        logger.error("❌ Model retraining failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
