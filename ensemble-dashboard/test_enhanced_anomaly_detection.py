#!/usr/bin/env python3
"""
Test Enhanced Anomaly Detection for Critical EJ Patterns
Test the specific case where "DEVICE ERROR" and "M-65" should be detected as anomalies
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from ensemble_detector import EnsembleAnomalyDetector

def test_critical_anomaly_patterns():
    """Test the enhanced anomaly detection on the specific case mentioned"""
    
    # Initialize detector
    detector = EnsembleAnomalyDetector()
    
    # Sample normal sessions for training
    normal_sessions = [
        """[020t*455*06/13/2025*06:10*
     *TRANSACTION START*
[020t CARD INSERTED
 06:10:47 ATR RECEIVED T=0
[020t 06:10:50 OPCODE = FI      
  PAN 0004263********2423
  ---START OF TRANSACTION---
[020t 06:11:03 PIN ENTERED
[020t 06:11:10 OPCODE = IB      
[020t 06:11:15 NOTES DISPENSED
[020t 06:11:18 RECEIPT PRINTED
[020t 06:11:20 CARD EJECTED
[020t 06:11:22 TRANSACTION END""",
        
        """[020t*456*06/13/2025*08:15*
     *TRANSACTION START*
[020t CARD INSERTED
 08:15:30 ATR RECEIVED T=0
[020t 08:15:35 PIN ENTERED
[020t 08:15:40 PIN VERIFIED
[020t 08:15:45 BALANCE INQUIRY
[020t 08:15:50 RECEIPT PRINTED
[020t 08:15:55 CARD EJECTED
[020t 08:15:58 TRANSACTION END""",
        
        """[020t*457*06/13/2025*10:20*
     *TRANSACTION START*
[020t CARD INSERTED
 10:20:15 ATR RECEIVED T=0
[020t 10:20:20 PIN ENTERED
[020t 10:20:25 PIN VERIFIED
[020t 10:20:30 CASH WITHDRAW SELECTED
[020t 10:20:35 AMOUNT ENTERED: $100
[020t 10:20:40 NOTES DISPENSED
[020t 10:20:45 RECEIPT PRINTED
[020t 10:20:50 CARD EJECTED
[020t 10:20:55 TRANSACTION END"""
    ]
    
    # The anomalous session you mentioned
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

    print("🔧 Testing Enhanced Ensemble Anomaly Detection")
    print("=" * 60)
    
    # Train the model
    print("📚 Training ensemble model on normal sessions...")
    training_stats = detector.train(normal_sessions)
    print(f"✅ Training complete: {training_stats['num_training_sessions']} sessions")
    print(f"   Text features: {training_stats['text_feature_dims']}")
    print(f"   Statistical features: {training_stats['numerical_feature_dims']}")
    print()
    
    # Test the specific anomalous case
    print("🚨 Testing the reported anomalous session:")
    print("Session contains: 'DEVICE ERROR' and 'M-65'")
    print("-" * 40)
    
    result = detector.predict(anomalous_session)
    
    # Display results
    print(f"🎯 PREDICTION RESULTS:")
    print(f"   Is Anomaly: {'🚨 YES' if result['is_anomaly'] else '✅ NO'}")
    print(f"   Ensemble Score: {result['ensemble_score']:.3f}")
    print(f"   Threshold: {result['threshold']:.3f}")
    print(f"   Confidence: {result['confidence']}")
    print()
    
    if 'critical_boost' in result:
        print(f"🔥 CRITICAL PATTERN ANALYSIS:")
        print(f"   Base Score: {result['base_ensemble_score']:.3f}")
        print(f"   Critical Boost: +{result['critical_boost']:.3f}")
        print(f"   Final Score: {result['ensemble_score']:.3f}")
        print()
        
        if result['anomaly_reasons']:
            print(f"🔍 DETECTED ANOMALY PATTERNS:")
            for reason in result['anomaly_reasons']:
                print(f"   • {reason}")
            print()
    
    # Show component breakdown
    print(f"📊 COMPONENT BREAKDOWN:")
    breakdown = result['prediction_breakdown']
    
    if 'text_component' in breakdown:
        text_comp = breakdown['text_component']
        print(f"   📝 Text Analysis:")
        if 'original_score' in text_comp:
            print(f"      Original: {text_comp['original_score']:.3f}")
            print(f"      Amplified: {text_comp['amplified_score']:.3f}")
        else:
            print(f"      Score: {text_comp['score']:.3f}")
        print(f"      Weight: {text_comp['weight']:.1f}")
        print(f"      Contribution: {text_comp['contribution']:.3f}")
    
    if 'statistical_component' in breakdown:
        stat_comp = breakdown['statistical_component']
        print(f"   📊 Statistical Analysis:")
        if 'original_score' in stat_comp:
            print(f"      Original: {stat_comp['original_score']:.3f}")
            print(f"      Amplified: {stat_comp['amplified_score']:.3f}")
        else:
            print(f"      Score: {stat_comp['score']:.3f}")
        print(f"      Weight: {stat_comp['weight']:.1f}")
        print(f"      Contribution: {stat_comp['contribution']:.3f}")
    
    print()
    
    # Show key features detected
    print(f"🔎 KEY FEATURES DETECTED:")
    text_features = result.get('text_features', {})
    numerical_features = result.get('numerical_features', {})
    
    critical_features = []
    if text_features.get('has_device_error', 0) > 0:
        critical_features.append("✅ DEVICE ERROR pattern detected")
    if text_features.get('has_critical_machine_code', 0) > 0:
        critical_features.append("✅ Critical machine code (M-65) detected")
    if numerical_features.get('critical_m_codes', 0) > 0:
        critical_features.append(f"✅ {int(numerical_features['critical_m_codes'])} critical machine status codes")
    if numerical_features.get('device_error_count', 0) > 0:
        critical_features.append(f"✅ {int(numerical_features['device_error_count'])} explicit DEVICE ERROR mentions")
    if numerical_features.get('error_codes_total', 0) > 0:
        critical_features.append(f"✅ {int(numerical_features['error_codes_total'])} total error codes")
    
    for feature in critical_features:
        print(f"   {feature}")
    
    if not critical_features:
        print("   ❌ No critical features detected (this indicates a problem with feature extraction)")
    
    print()
    
    # Test a few normal sessions for comparison
    print("🔍 Testing normal sessions for comparison:")
    print("-" * 40)
    
    for i, normal_session in enumerate(normal_sessions[:2]):
        print(f"Normal Session {i+1}:")
        normal_result = detector.predict(normal_session)
        print(f"   Is Anomaly: {'🚨 YES' if normal_result['is_anomaly'] else '✅ NO'}")
        print(f"   Score: {normal_result['ensemble_score']:.3f}")
        print(f"   Confidence: {normal_result['confidence']}")
        print()
    
    # Final assessment
    print("🎯 FINAL ASSESSMENT:")
    print("=" * 60)
    
    if result['is_anomaly']:
        print("✅ SUCCESS: The enhanced detector correctly identified the session as anomalous!")
        print(f"   The combination of 'DEVICE ERROR' and 'M-65' was properly detected")
        print(f"   Final anomaly score: {result['ensemble_score']:.3f} > threshold {result['threshold']:.3f}")
        if result.get('critical_boost', 0) > 0:
            print(f"   Critical pattern boost: +{result['critical_boost']:.3f}")
    else:
        print("❌ ISSUE: The detector did not classify this as anomalous")
        print("   This suggests the feature extraction or weighting needs adjustment")
        print(f"   Score: {result['ensemble_score']:.3f} <= threshold {result['threshold']:.3f}")
    
    return result

if __name__ == "__main__":
    test_critical_anomaly_patterns()
