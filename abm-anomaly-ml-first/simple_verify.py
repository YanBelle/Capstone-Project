#!/usr/bin/env python3
import requests
import json

# Test our API fix
payload = {
    "text": "DEVICE ERROR: ATM MALFUNCTION DETECTED REJECTS:000 TRANSACTION FAILED",
    "session_id": "TEST_VERIFICATION",
    "return_vectors": True,
    "debug": True
}

print("🔍 Testing EJ Contextual Enhancement Fix...")
try:
    response = requests.post("http://localhost:8000/api/v1/bert/visualize", json=payload, timeout=30)
    
    if response.status_code == 200:
        data = response.json()
        
        print("✅ API call successful!")
        
        # Check for our critical fix
        if 'ej_contextual_enhancement' in data:
            ej_metadata = data['ej_contextual_enhancement']
            print("🎉 EJ CONTEXTUAL ENHANCEMENT METADATA FOUND!")
            print(f"   📈 EJ Labeler Used: {ej_metadata.get('ej_labeler_used')}")
            print(f"   🧠 Expert Labeler Used: {ej_metadata.get('expert_labeler_used')}")
            print(f"   💪 Enhancement Impact: {ej_metadata.get('enhancement_impact')}")
            print(f"   🚫 Special Tokens Suppressed: {ej_metadata.get('special_tokens_suppressed')}")
            
            # Success criteria
            if (ej_metadata.get('ej_labeler_used') == True and 
                ej_metadata.get('enhancement_impact', 0) > 0):
                print("\n🎯 SUCCESS: EJ Contextual Labeler is WORKING!")
                print("   The method name fix (label_ej_line → label_log) has resolved the issue!")
                print("   BERT should now properly prioritize 'DEVICE ERROR' and 'REJECTS:000' terms.")
            else:
                print("\n❌ Issue: EJ Labeler metadata indicates problems")
        else:
            print("❌ No EJ contextual enhancement metadata found")
    else:
        print(f"❌ API error: {response.status_code}")
        
except Exception as e:
    print(f"❌ Request failed: {e}")
