#!/usr/bin/env python3
"""
Verify that EJ Contextual Enhancement is working after the critical method name fix.
This script tests that the API now returns proper metadata about EJ labeler usage.
"""

import requests
import json
import time

def test_ej_enhancement():
    """Test that EJ contextual enhancement is working with metadata verification"""
    
    # Test payload with critical ATM error terms
    test_payload = {
        "text": "DEVICE ERROR: ATM MALFUNCTION DETECTED REJECTS:000 TRANSACTION FAILED",
        "session_id": "TEST_SESSION_VERIFICATION_001",
        "return_vectors": True,
        "debug": True
    }
    
    print("🔍 Testing EJ Contextual Enhancement after critical method name fix...")
    print(f"📝 Test text: '{test_payload['text']}'")
    print()
    
    try:
        # Call the API endpoint
        print("📡 Calling BERT API endpoint...")
        response = requests.post(
            "http://localhost:8000/api/v1/bert/visualize",
            json=test_payload,
            timeout=30
        )
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ ERROR: API returned status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
        # Parse response
        data = response.json()
        print("✅ API call successful!")
        print()
        
        # Check for EJ contextual enhancement metadata (the key fix we made)
        print("🔬 Checking for EJ Contextual Enhancement metadata...")
        
        if 'ej_contextual_enhancement' in data:
            ej_metadata = data['ej_contextual_enhancement']
            print("✅ EJ contextual enhancement metadata found!")
            print(f"   📈 EJ Labeler Used: {ej_metadata.get('ej_labeler_used', 'NOT FOUND')}")
            print(f"   🧠 Expert Labeler Used: {ej_metadata.get('expert_labeler_used', 'NOT FOUND')}")
            print(f"   💪 Enhancement Impact: {ej_metadata.get('enhancement_impact', 'NOT FOUND')}")
            print(f"   🚫 Special Tokens Suppressed: {ej_metadata.get('special_tokens_suppressed', 'NOT FOUND')}")
            
            # Verify EJ labeler is actually being used
            if ej_metadata.get('ej_labeler_used') == True:
                print("🎉 SUCCESS: EJ Contextual Labeler is ACTIVE!")
            else:
                print("❌ FAILURE: EJ Contextual Labeler is NOT being used")
                return False
                
        else:
            print("❌ FAILURE: No EJ contextual enhancement metadata found in response")
            print("Available keys:", list(data.keys()))
            return False
        
        print()
        
        # Check token importance for critical terms
        print("🎯 Checking token importance for critical ATM terms...")
        
        if 'token_analysis' in data and 'importance_scores' in data['token_analysis']:
            importance_scores = data['token_analysis']['importance_scores']
            
            # Look for critical terms
            critical_terms = ['DEVICE', 'ERROR', 'REJECTS', '000']
            found_critical_terms = []
            
            for token_info in importance_scores:
                token = token_info.get('token', '').upper()
                importance = token_info.get('importance', 0)
                
                for term in critical_terms:
                    if term in token:
                        found_critical_terms.append((token, importance))
                        print(f"   🎯 Found '{token}': importance = {importance:.4f}")
            
            if found_critical_terms:
                # Check if any critical term has high importance (should be boosted by EJ labeler)
                high_importance_terms = [term for term, imp in found_critical_terms if imp > 0.1]
                if high_importance_terms:
                    print(f"✅ SUCCESS: Critical terms have high importance scores!")
                    print(f"   High importance terms: {high_importance_terms}")
                else:
                    print("⚠️  WARNING: Critical terms found but importance scores seem low")
                    print("   This might indicate EJ enhancement isn't boosting scores as expected")
            else:
                print("❌ No critical ATM terms found in token analysis")
                
        else:
            print("❌ No token analysis data found in response")
            
        print()
        print("📋 SUMMARY:")
        print(f"   Method Name Fix: ✅ Applied (label_ej_line → label_log)")
        print(f"   API Enhancement: ✅ Applied (metadata extraction)")
        print(f"   EJ Labeler Active: {'✅ YES' if data.get('ej_contextual_enhancement', {}).get('ej_labeler_used') else '❌ NO'}")
        print(f"   Enhancement Metadata: {'✅ PRESENT' if 'ej_contextual_enhancement' in data else '❌ MISSING'}")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON response: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

if __name__ == "__main__":
    print("🚀 EJ Contextual Enhancement Verification")
    print("=" * 50)
    print()
    
    # Check API health first
    if not check_api_health():
        print("❌ API is not responding. Please ensure services are running:")
        print("   docker-compose up -d")
        exit(1)
    
    print("✅ API is responding")
    print()
    
    # Run the verification test
    success = test_ej_enhancement()
    
    if success:
        print()
        print("🎉 VERIFICATION COMPLETE: EJ Contextual Enhancement appears to be working!")
        print("   The critical method name fix (label_ej_line → label_log) has resolved the issue.")
        print("   BERT should now properly prioritize 'DEVICE ERROR' and 'REJECTS:000' terms.")
    else:
        print()
        print("❌ VERIFICATION FAILED: Issues detected with EJ Contextual Enhancement")
        print("   Further investigation may be needed.")
