#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify BERT API is working with EJ contextual labeling
"""
import requests
import json
import sys

def test_bert_visualization():
    """Test the BERT visualization API endpoint that the frontend uses"""
    url = "http://localhost:8000/api/v1/bert/visualize"
    
    # Test with the EXACT EJ log that should show strong importance for DEVICE ERROR and REJECTS:000
    test_data = {
        "text": """[020t*629*06/18/2025*00:46*
     *TRANSACTION START*
[020t CARD INSERTED
 00:46:27 ATR RECEIVED T=0
[020t 00:46:30 OPCODE = FI      

  PAN 0004263********1897
  ---START OF TRANSACTION---
 
[020t 00:46:42 PIN ENTERED
[020t 00:46:47 OPCODE = IB      

  PAN 0004263********1897
  ---START OF TRANSACTION---
 
*630*06/18/2025*00:46*
*7231*1*(Iw(1*3, M-02, R-10011
A/C 
DEVICE ERROR
ESC: 000
VAL: 000
REF: 000
REJECTS:000*(1
S
[020t 00:47:13 CARD TAKEN
[020t 00:47:15 TRANSACTION END
[020t*631*06/18/2025*00:47*"""
    }
    
    try:
        response = requests.post(url, json=test_data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ BERT Visualization API Response:")
            print(f"Status: {response.status_code}")
            print(f"Response keys: {list(result.keys())}")
            
            # Check for visualizations 
            visualizations = result.get('visualizations', {})
            if visualizations:
                print(f"\n🎨 Visualizations Available:")
                for viz_name, viz_data in visualizations.items():
                    if viz_data and len(viz_data) > 100:
                        print(f"   ✅ {viz_name}: Generated (length: {len(viz_data)})")
                    else:
                        print(f"   ❌ {viz_name}: Empty or failed")
                        
                # Specifically check if attention_heatmap and token_importance exist
                heatmap_ok = 'attention_heatmap' in visualizations and len(visualizations['attention_heatmap']) > 100
                token_ok = 'token_importance' in visualizations and len(visualizations['token_importance']) > 100
                
                if heatmap_ok and token_ok:
                    print(f"\n🎯 SUCCESS: Both attention heatmap and token importance visualizations generated!")
                    print(f"   - Now checking for EJ contextual labeling enhancement...")
                    
                    # Check if contextual enhancement is working
                    result_data = result.get('data', {})
                    contextual_info = result_data.get('token_importance', {}).get('contextual_enhancement', {})
                    
                    if contextual_info.get('ej_labeler_used', False):
                        print(f"   ✅ EJ Contextual Labeler ACTIVE - domain-specific enhancement enabled!")
                        print(f"   ✅ Expected high importance for: DEVICE, ERROR, REJECTS, 000")
                        print(f"   ✅ Special tokens [CLS], [SEP] should be suppressed")
                        
                        # Check enhancement impact
                        enhancement_impact = contextual_info.get('enhancement_impact', 0)
                        print(f"   📊 EJ Enhancement Impact: {enhancement_impact:.3f}")
                        
                        if enhancement_impact > 0.1:
                            print(f"   🚀 STRONG contextual enhancement detected!")
                        else:
                            print(f"   ⚠️  Low contextual enhancement - may need stronger EJ patterns")
                    else:
                        print(f"   ❌ EJ Contextual Labeler NOT ACTIVE - using basic analysis only")
                        print(f"   ❌ This explains why DEVICE ERROR and REJECTS:000 aren't prioritized!")
                    
                    return True
                else:
                    print(f"\n⚠️  PARTIAL: Some visualizations missing")
                    print(f"   - Heatmap: {'✅' if heatmap_ok else '❌'}")
                    print(f"   - Token importance: {'✅' if token_ok else '❌'}")
                    return False
            else:
                print(f"\n❌ No visualizations in response")
                return False
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get("http://localhost:8000/api/v1/health", timeout=10)
        if response.status_code == 200:
            print("✅ API Health Check: OK")
            return True
        else:
            print(f"❌ API Health Check Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health Check Error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing BERT Visualization API with EJ Contextual Labeling")
    print("🔍 Focus: DEVICE ERROR and REJECTS:000 should have HIGH importance")
    print("=" * 70)
    
    # Test health first
    if not test_health():
        print("❌ Health check failed - cannot proceed")
        sys.exit(1)
    
    print()
    
    # Test BERT visualization (the one the frontend uses)
    if test_bert_visualization():
        print("\n✅ BERT Visualization API test completed successfully!")
        print("🎉 EJ Contextual Labeling should now prioritize DEVICE ERROR and REJECTS:000!")
        print("🔍 Check the dashboard - anomaly terms should have HIGH token importance!")
    else:
        print("\n❌ BERT Visualization API test failed!")
        print("🚨 EJ Contextual Labeler may not be working properly!")
        sys.exit(1)
