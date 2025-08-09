#!/usr/bin/env python3
"""
Simple test of anomaly detection for the DEVICE ERROR case
"""

import sys
import os
import re

# Add the backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_feature_extraction():
    """Test just the feature extraction on the anomalous session"""
    
    anomalous_session = """[020t*455*06/13/2025*06:10*
     *TRANSACTION START*
[020t CARD INSERTED
 06:10:47 ATR RECEIVED T=0
[020t 06:10:50 OPCODE = FI      

  PAN 0004263********2423
  ---START OF TRANSACTION---
 
[020t 06:11:03 PIN ENTERED
[020t 06:11:10 OPCODE = IB      

  PAN 0004263********2423
  ---START OF TRANSACTION---
 
*456*06/13/2025*06:11*
*8409*1*(Iw(1*3, M-65, R-100110021
A/C 
DEVICE ERROR
ESC: 000
VAL: 000
REF: 000
REJECTS:000*(1
S
[020t*457*06/13/2025*06:12*
000025_8409_061325_061208.610_2423_002
[020t 06:12:08 CARD TAKEN
[020t 06:12:09 TRANSACTION END
[020t*458*06/13/2025*06:12*
     *PRIMARY CARD READER ACTIVATED*"""
    
    print("🔍 Testing Critical Pattern Detection")
    print("=" * 50)
    
    # Test regex patterns directly
    text_lower = anomalous_session.lower()
    text_upper = anomalous_session.upper()
    
    print("📝 Direct Pattern Detection:")
    
    # Test DEVICE ERROR pattern
    device_errors = len(re.findall(r'device\s+error', text_lower))
    print(f"   DEVICE ERROR occurrences: {device_errors}")
    
    # Test M-XX machine codes
    machine_codes = re.findall(r'M-\d+', text_upper)
    print(f"   Machine status codes found: {machine_codes}")
    
    # Test critical machine codes
    critical_codes = re.findall(r'M-(?:01|15|23|38|45|65|67)', text_upper)
    print(f"   Critical machine codes: {critical_codes}")
    
    # Test error codes
    error_codes = re.findall(r'[ME]-\d+', text_upper)
    print(f"   All error codes: {error_codes}")
    
    # Test general error patterns
    general_errors = len(re.findall(r'error', text_lower))
    print(f"   General 'error' mentions: {general_errors}")
    
    print()
    print("🎯 Expected Detection Results:")
    print("   ✅ Should find 'DEVICE ERROR' pattern")
    print("   ✅ Should find 'M-65' machine status code")
    print("   ✅ Should classify as critical anomaly")
    
    print()
    
    # Test with actual detector if available
    try:
        from ensemble_detector import EnsembleAnomalyDetector
        
        print("🔧 Testing with Enhanced Ensemble Detector:")
        print("-" * 40)
        
        detector = EnsembleAnomalyDetector()
        
        # Extract features
        text_features = detector.extract_text_features(anomalous_session)
        num_features = detector.extract_numerical_features(anomalous_session)
        
        print(f"📊 Text Features:")
        print(f"   device_error_explicit: {text_features.get('device_error_explicit', 0)}")
        print(f"   critical_anomaly_score: {text_features.get('critical_anomaly_score', 0):.3f}")
        print(f"   has_device_error: {text_features.get('has_device_error', 0)}")
        print(f"   has_critical_machine_code: {text_features.get('has_critical_machine_code', 0)}")
        
        print(f"📈 Numerical Features:")
        print(f"   device_error_count: {num_features.get('device_error_count', 0)}")
        print(f"   critical_m_codes: {num_features.get('critical_m_codes', 0)}")
        print(f"   machine_status_codes: {num_features.get('machine_status_codes', 0)}")
        print(f"   session_health_score: {num_features.get('session_health_score', 1.0):.3f}")
        print(f"   anomaly_density_score: {num_features.get('anomaly_density_score', 0):.3f}")
        
        # Test training and prediction
        print()
        print("🚀 Testing Full Prediction Pipeline:")
        print("-" * 40)
        
        # Simple normal sessions for training
        normal_sessions = [
            "TRANSACTION START\nCARD INSERTED\nPIN VERIFIED\nCASH DISPENSED\nRECEIPT PRINTED\nCARD EJECTED\nTRANSACTION END",
            "TRANSACTION START\nCARD INSERTED\nPIN VERIFIED\nBALANCE INQUIRY\nRECEIPT PRINTED\nCARD EJECTED\nTRANSACTION END",
            "TRANSACTION START\nCARD INSERTED\nPIN VERIFIED\nDEPOSIT COMPLETED\nRECEIPT PRINTED\nCARD EJECTED\nTRANSACTION END"
        ]
        
        # Train the model
        print("📚 Training model...")
        training_stats = detector.train(normal_sessions)
        print(f"   ✅ Trained on {training_stats['num_training_sessions']} sessions")
        
        # Test prediction
        print("🎯 Predicting anomaly...")
        result = detector.predict(anomalous_session)
        
        print(f"   Is Anomaly: {'🚨 YES' if result['is_anomaly'] else '❌ NO'}")
        print(f"   Ensemble Score: {result['ensemble_score']:.3f}")
        print(f"   Threshold: {result['threshold']:.3f}")
        print(f"   Confidence: {result['confidence']}")
        
        if 'critical_boost' in result and result['critical_boost'] > 0:
            print(f"   Critical Boost: +{result['critical_boost']:.3f}")
            
        if 'anomaly_reasons' in result and result['anomaly_reasons']:
            print("   🔍 Anomaly Reasons:")
            for reason in result['anomaly_reasons']:
                print(f"      • {reason}")
                
        return result
        
    except ImportError as e:
        print(f"❌ Could not import enhanced detector: {e}")
        return None
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return None

if __name__ == "__main__":
    test_feature_extraction()
