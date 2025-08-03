#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test script for the Enhanced EJ BERT system
Validates basic functionality without complex formatting
"""

import sys
import os
import traceback

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """Test basic imports"""
    print("Testing basic imports...")
    
    try:
        from ej_contextual_labeler import EJLogLabeler, EventType
        print("+ EJ Contextual Labeler imported")
        
        from enhanced_ej_bert import EnhancedEJBertAnalyzer
        print("+ Enhanced EJ BERT imported")
        
        from contextual_anomaly_detector import ContextualAnomalyDetector
        print("+ Contextual Anomaly Detector imported")
        
        return True
    except Exception as e:
        print("X Import failed: " + str(e))
        traceback.print_exc()
        return False

def test_contextual_labeling_basic():
    """Test basic contextual labeling"""
    print("\nTesting contextual labeling...")
    
    try:
        from ej_contextual_labeler import EJLogLabeler
        
        labeler = EJLogLabeler()
        
        # Simple test log
        test_log = "2024-01-15 10:30:15,123 [INFO] Transaction started - Card inserted"
        
        labels = labeler.label_ej_log(test_log)
        print("+ Labeling completed. Labels: " + str(len(labels)))
        
        if len(labels) > 0:
            label = labels[0]
            print("  First label event type: " + label.event_type.value)
            print("  First label severity: " + label.severity.value)
        
        return True
        
    except Exception as e:
        print("X Labeling test failed: " + str(e))
        traceback.print_exc()
        return False

def test_anomaly_detection_basic():
    """Test basic anomaly detection"""
    print("\nTesting anomaly detection...")
    
    try:
        from ej_contextual_labeler import EJLogLabeler
        from contextual_anomaly_detector import ContextualAnomalyDetector
        
        labeler = EJLogLabeler()
        detector = ContextualAnomalyDetector()
        
        # Test with multiple error logs
        error_log = """
2024-01-15 10:30:15,123 [ERROR] Card read failed
2024-01-15 10:30:16,456 [ERROR] Card read failed  
2024-01-15 10:30:17,789 [ERROR] Card read failed
2024-01-15 10:30:18,012 [CRITICAL] Card reader malfunction
        """.strip()
        
        labels = labeler.label_ej_log(error_log)
        anomalies = detector.detect_anomalies(labels)
        
        print("+ Anomaly detection completed")
        print("  Labels found: " + str(len(labels)))
        print("  Anomalies detected: " + str(len(anomalies)))
        
        if len(anomalies) > 0:
            print("  First anomaly type: " + anomalies[0].get('type', 'Unknown'))
        
        return True
        
    except Exception as e:
        print("X Anomaly detection failed: " + str(e))
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Enhanced EJ BERT System - Basic Test")
    print("=" * 40)
    
    tests = [
        ("Imports", test_basic_imports),
        ("Contextual Labeling", test_contextual_labeling_basic),
        ("Anomaly Detection", test_anomaly_detection_basic)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print("\n" + "=" * 20 + " " + test_name + " " + "=" * 20)
        try:
            result = test_func()
            if result:
                passed += 1
                print("RESULT: PASS")
            else:
                print("RESULT: FAIL")
        except Exception as e:
            print("RESULT: CRASH - " + str(e))
    
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("Passed: " + str(passed) + "/" + str(total))
    
    if passed == total:
        print("SUCCESS: All tests passed!")
        return 0
    else:
        print("WARNING: Some tests failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
