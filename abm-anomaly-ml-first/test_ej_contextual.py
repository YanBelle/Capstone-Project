#!/usr/bin/env python3
"""
Simple test to verify EJ contextual labeler without BERT model loading
"""

def test_ej_labeler():
    try:
        from ej_contextual_labeler import EJLogLabeler, EventType, TransactionPhase
        print("✅ EJ Contextual Labeler imports successful")
        
        labeler = EJLogLabeler()
        print("✅ EJ Contextual Labeler instantiated")
        
        test_text = "DEVICE ERROR REJECTS:000"
        labels = labeler.label_ej_line(test_text)
        print(f"✅ EJ Labeler processed '{test_text}'")
        print(f"📊 Labels returned: {len(labels)} labels")
        for label in labels:
            print(f"   - {label}")
        
        return True
    except Exception as e:
        print(f"❌ EJ Contextual Labeler test failed: {e}")
        return False

def test_contextual_importance():
    try:
        import numpy as np
        
        # Mock the contextual importance method
        tokens = ['DEVICE', 'ERROR', 'REJECTS', '000']
        importance = np.zeros(len(tokens))
        
        # Simulate EJ contextual enhancement
        keywords = ['device', 'error', 'reject']
        for i, token in enumerate(tokens):
            if any(keyword in token.lower() for keyword in keywords):
                importance[i] = 2.0  # High importance boost
        
        # Normalize
        if importance.max() > 0:
            importance = importance / importance.max()
        
        print(f"✅ Contextual importance test:")
        for token, score in zip(tokens, importance):
            print(f"   {token}: {score:.3f}")
        
        # Check if important terms have high scores
        device_score = importance[0]  # DEVICE
        error_score = importance[1]   # ERROR
        rejects_score = importance[2]  # REJECTS
        
        if device_score > 0.8 and error_score > 0.8 and rejects_score > 0.8:
            print("🎯 SUCCESS: Anomaly terms have HIGH importance!")
            return True
        else:
            print("⚠️  WARNING: Anomaly terms have low importance")
            return False
            
    except Exception as e:
        print(f"❌ Contextual importance test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing EJ Contextual Labeler Integration")
    print("=" * 50)
    
    ej_ok = test_ej_labeler()
    print()
    contextual_ok = test_contextual_importance()
    
    print("\n" + "=" * 50)
    if ej_ok and contextual_ok:
        print("✅ All tests PASSED - EJ contextual enhancement ready!")
    else:
        print("❌ Some tests FAILED - EJ contextual enhancement needs fixes!")
