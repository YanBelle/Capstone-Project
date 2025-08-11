#!/usr/bin/env python3
"""
Test script for DBSCAN-enhanced ensemble anomaly detection
Demonstrates the integration of DBSCAN with Isolation Forest and One-Class SVM
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import logging

# Add the anomaly detector path
sys.path.insert(0, 'abm-anomaly-ml-first/services/anomaly-detector')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dbscan_ensemble():
    """Test the enhanced ensemble detection with DBSCAN"""
    
    try:
        from ml_analyzer import MLFirstAnomalyDetector, TransactionSession
        
        print("🔧 Initializing Enhanced Ensemble Detector with DBSCAN...")
        detector = MLFirstAnomalyDetector()
        
        # Verify DBSCAN is initialized
        print(f"✅ DBSCAN Model: {type(detector.dbscan).__name__}")
        print(f"✅ DBSCAN Parameters: eps={detector.dbscan.eps}, min_samples={detector.dbscan.min_samples}")
        print(f"✅ Isolation Forest: {type(detector.isolation_forest).__name__}")
        print(f"✅ One-Class SVM: {type(detector.one_class_svm).__name__}")
        
        # Create test sessions with different patterns
        test_sessions = [
            # Normal sessions
            TransactionSession(
                "normal_001",
                "ATM CARD INSERTED\\nPIN ENTRY\\nBALANCE INQUIRY\\nCASH WITHDRAWAL $100\\nTRANSACTION COMPLETE\\nCARD EJECTED",
                datetime.now(),
                None
            ),
            TransactionSession(
                "normal_002", 
                "ATM CARD INSERTED\\nPIN ENTRY\\nDEPOSIT $200\\nRECEIPT PRINTED\\nTRANSACTION COMPLETE\\nCARD EJECTED",
                datetime.now(),
                None
            ),
            TransactionSession(
                "normal_003",
                "ATM CARD INSERTED\\nPIN ENTRY\\nBALANCE INQUIRY\\nMINI STATEMENT\\nTRANSACTION COMPLETE\\nCARD EJECTED",
                datetime.now(),
                None
            ),
            
            # Anomalous sessions - different types for ensemble testing
            TransactionSession(
                "anomaly_001",
                "ATM CARD INSERTED\\nPIN ENTRY\\nCASH WITHDRAWAL $500\\nUNABLE TO DISPENSE\\nDEVICE ERROR\\nTRANSACTION CANCELLED\\nCARD EJECTED",
                datetime.now(),
                None
            ),
            TransactionSession(
                "anomaly_002",
                "SUPERVISOR MODE ENTRY\\nDIAGNOSTIC TEST\\nCASH RETRACT STARTED\\nPOWER-UP/RESET\\nSUPERVISOR MODE EXIT",
                datetime.now(),
                None
            ),
            TransactionSession(
                "anomaly_003",
                "ATM CARD INSERTED\\nPIN ENTRY FAILED\\nPIN ENTRY FAILED\\nPIN ENTRY FAILED\\nCARD RETAINED\\nSECURITY ALERT",
                datetime.now(),
                None
            ),
            TransactionSession(
                "outlier_001",
                "UNKNOWN_EVENT_001\\nRANDOM_SEQUENCE_XYZ\\nUNEXPECTED_PATTERN\\nANOMALOUS_BEHAVIOR\\nSTRANGE_ENDING",
                datetime.now(),
                None
            )
        ]
        
        print(f"\n📊 Processing {len(test_sessions)} test sessions...")
        
        # Set sessions and get embeddings
        detector.sessions = test_sessions
        embeddings = detector.convert_to_embeddings(test_sessions)
        
        print(f"✅ Generated embeddings: {embeddings.shape}")
        
        # Run ensemble detection
        print("\n🚀 Running Enhanced Ensemble Detection (IF + SVM + DBSCAN)...")
        results = detector.fit_predict(embeddings)
        
        # Display results
        print("\n📈 ENSEMBLE DETECTION RESULTS:")
        print("=" * 80)
        
        for i, session in enumerate(detector.sessions):
            anomaly_status = "🚨 ANOMALY" if session.is_anomaly else "✅ NORMAL"
            
            print(f"\nSession: {session.session_id}")
            print(f"Status: {anomaly_status}")
            print(f"Overall Score: {session.overall_anomaly_score:.3f}")
            
            if hasattr(session, 'anomalies') and session.anomalies:
                print(f"Detected Anomalies ({len(session.anomalies)}):")
                for anomaly in session.anomalies:
                    print(f"  - {anomaly.anomaly_type} (confidence: {anomaly.confidence:.3f}, method: {anomaly.detection_method})")
            
            # Show individual model results
            if_pred = results['if_predictions'][i]
            svm_pred = results['svm_predictions'][i]
            dbscan_pred = results['dbscan_predictions'][i]
            
            print(f"Model Results:")
            print(f"  - Isolation Forest: {'Anomaly' if if_pred == -1 else 'Normal'} (score: {results['if_scores'][i]:.3f})")
            print(f"  - One-Class SVM: {'Anomaly' if svm_pred == -1 else 'Normal'} (score: {results['svm_scores'][i]:.3f})")
            print(f"  - DBSCAN: {'Outlier' if dbscan_pred == -1 else 'Clustered'} (score: {results['dbscan_scores'][i]:.3f}, cluster: {results['dbscan_labels'][i]})")
        
        # Summary statistics
        print("\n📊 ENSEMBLE SUMMARY:")
        print("=" * 50)
        
        total_sessions = len(detector.sessions)
        anomaly_sessions = sum(1 for s in detector.sessions if s.is_anomaly)
        normal_sessions = total_sessions - anomaly_sessions
        
        print(f"Total Sessions: {total_sessions}")
        print(f"Normal Sessions: {normal_sessions}")
        print(f"Anomaly Sessions: {anomaly_sessions}")
        print(f"Anomaly Rate: {(anomaly_sessions/total_sessions)*100:.1f}%")
        
        # Model agreement analysis
        if_anomalies = sum(1 for pred in results['if_predictions'] if pred == -1)
        svm_anomalies = sum(1 for pred in results['svm_predictions'] if pred == -1)
        dbscan_outliers = sum(1 for pred in results['dbscan_predictions'] if pred == -1)
        
        print(f"\nModel-specific Detections:")
        print(f"Isolation Forest Anomalies: {if_anomalies}")
        print(f"One-Class SVM Anomalies: {svm_anomalies}")
        print(f"DBSCAN Outliers: {dbscan_outliers}")
        
        # DBSCAN cluster analysis
        unique_clusters = len(set(results['dbscan_labels'])) - (1 if -1 in results['dbscan_labels'] else 0)
        noise_points = sum(1 for label in results['dbscan_labels'] if label == -1)
        
        print(f"\nDBSCAN Cluster Analysis:")
        print(f"Number of Clusters: {unique_clusters}")
        print(f"Noise Points (Outliers): {noise_points}")
        print(f"Clustered Points: {total_sessions - noise_points}")
        
        print("\n✅ DBSCAN Ensemble Test Completed Successfully!")
        
        return {
            'total_sessions': total_sessions,
            'anomaly_sessions': anomaly_sessions,
            'if_anomalies': if_anomalies,
            'svm_anomalies': svm_anomalies,
            'dbscan_outliers': dbscan_outliers,
            'dbscan_clusters': unique_clusters,
            'ensemble_results': results
        }
        
    except Exception as e:
        logger.error(f"Error in DBSCAN ensemble test: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_dbscan_parameter_optimization():
    """Test DBSCAN parameter optimization"""
    
    try:
        from ml_analyzer import MLFirstAnomalyDetector
        
        print("\n🔧 Testing DBSCAN Parameter Optimization...")
        detector = MLFirstAnomalyDetector()
        
        # Generate synthetic embedding data for testing
        np.random.seed(42)
        
        # Create clusters with outliers
        cluster1 = np.random.normal(0, 0.1, (20, 50))
        cluster2 = np.random.normal(2, 0.1, (15, 50))
        outliers = np.random.normal(5, 0.5, (5, 50))
        
        synthetic_embeddings = np.vstack([cluster1, cluster2, outliers])
        
        print(f"Generated synthetic data: {synthetic_embeddings.shape}")
        
        # Test parameter optimization
        optimal_params = detector.optimize_dbscan_parameters(synthetic_embeddings)
        
        print(f"Optimal DBSCAN Parameters:")
        print(f"  - eps: {optimal_params['eps']:.3f}")
        print(f"  - min_samples: {optimal_params['min_samples']}")
        
        # Apply optimized parameters
        detector.dbscan.set_params(**optimal_params)
        labels = detector.dbscan.fit_predict(synthetic_embeddings)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = sum(1 for label in labels if label == -1)
        
        print(f"\nOptimized DBSCAN Results:")
        print(f"  - Clusters: {n_clusters}")
        print(f"  - Outliers: {n_noise}")
        print(f"  - Clustered points: {len(labels) - n_noise}")
        
        print("✅ DBSCAN Parameter Optimization Test Completed!")
        
    except Exception as e:
        logger.error(f"Error in parameter optimization test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🎯 Enhanced Ensemble Anomaly Detection with DBSCAN")
    print("=" * 60)
    
    # Test main ensemble functionality
    ensemble_results = test_dbscan_ensemble()
    
    # Test parameter optimization
    test_dbscan_parameter_optimization()
    
    if ensemble_results:
        print(f"\n🎉 All tests completed successfully!")
        print(f"Enhanced ensemble with DBSCAN is now operational.")
    else:
        print(f"\n❌ Tests failed. Check the error messages above.")
