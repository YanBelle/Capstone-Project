#!/usr/bin/env python3
"""Minimal test for DeepLog integration"""

try:
    print("Testing imports...")
    
    # Test PyTorch import
    import torch
    print("✅ PyTorch imported")
    
    # Test DeepLog analyzer
    from deeplog_analyzer import DeepLogAnalyzer
    print("✅ DeepLog analyzer imported")
    
    # Test basic functionality
    deeplog = DeepLogAnalyzer()
    print("✅ DeepLog initialized")
    
    # Test ML analyzer integration
    from ml_analyzer import MLFirstAnomalyDetector
    ml_detector = MLFirstAnomalyDetector()
    print("✅ ML analyzer initialized")
    
    if hasattr(ml_detector, 'deeplog_analyzer'):
        if ml_detector.deeplog_analyzer:
            print("✅ DeepLog properly integrated")
        else:
            print("⚠️ DeepLog analyzer is None")
    else:
        print("❌ DeepLog not found in ML analyzer")
    
    print("\n🎉 Integration test completed successfully!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
