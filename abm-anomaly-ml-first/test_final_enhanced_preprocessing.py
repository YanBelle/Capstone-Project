#!/usr/bin/env python3
"""
Final validation test for enhanced BERT preprocessing
Tests the exact EJ sample provided by the user to ensure all tokenization issues are resolved
"""

import requests
import json
from datetime import datetime

def test_enhanced_preprocessing():
    """Test the enhanced preprocessing with the actual EJ sample"""
    
    # The exact EJ sample that had the tokenization issues
    ej_sample = """[020t*629*06/18/2025*00:46*
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

    print("🔬 Testing Enhanced BERT Preprocessing")
    print("=" * 60)
    print(f"Testing at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Test the enhanced EJ BERT analysis endpoint
        payload = {
            "text": ej_sample,
            "session_id": "TEST_ENHANCED_PREPROCESSING_001", 
            "include_visualization": True,
            "include_preprocessing_details": True
        }
        
        print("📤 Sending request to BERT analysis API...")
        response = requests.post(
            "http://localhost:8000/api/v1/bert/analyze",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ API Response received successfully!")
            print()
            
            # Check if BERT analysis is present
            if 'analysis' in result:
                bert_analysis = result['analysis']
                
                print("🧠 BERT Analysis Results:")
                print("-" * 40)
                
                # Check processed tokens
                if 'processed_text' in bert_analysis:
                    processed_text = bert_analysis['processed_text']
                    print(f"Processed Text: {processed_text}")
                    print()
                    
                    # Validate specific improvements
                    validation_checks = [
                        ("ESC_000 token present", "ESC_000" in processed_text),
                        ("VAL_000 token present", "VAL_000" in processed_text),
                        ("REF_000 token present", "REF_000" in processed_text),
                        ("REJECTS_000 token present", "REJECTS_000" in processed_text),
                        ("ATR_RECEIVED_T_0 present", "ATR_RECEIVED_T_0" in processed_text),
                        ("No isolated '1' tokens", " 1 " not in processed_text and not processed_text.endswith(" 1")),
                        ("No isolated 'S' tokens", " S " not in processed_text and not processed_text.endswith(" S")),
                        ("TRANSACTION_START present", "TRANSACTION_START" in processed_text),
                        ("DEVICE_ERROR present", "DEVICE_ERROR" in processed_text),
                        ("CARD_INSERTED present", "CARD_INSERTED" in processed_text),
                        ("CARD_TAKEN present", "CARD_TAKEN" in processed_text),
                        ("PIN_ENTERED present", "PIN_ENTERED" in processed_text),
                        ("TRANSACTION_END present", "TRANSACTION_END" in processed_text),
                        ("M_02 present", "M_02" in processed_text),
                        ("R_10011 present", "R_10011" in processed_text),
                        ("OPCODE_FI present", "OPCODE_FI" in processed_text),
                        ("OPCODE_IB present", "OPCODE_IB" in processed_text),
                        ("CardNumber present", "CardNumber" in processed_text),
                        ("No noise patterns", "*7231*1*(Iw(1*3," not in processed_text),
                        ("No timestamps", "00:46:27" not in processed_text),
                    ]
                    
                    print("🔍 Validation Results:")
                    print("-" * 40)
                    passed = 0
                    total = len(validation_checks)
                    
                    for check_name, check_result in validation_checks:
                        status = "✅ PASS" if check_result else "❌ FAIL"
                        print(f"  {check_name}: {status}")
                        if check_result:
                            passed += 1
                    
                    print()
                    print("📊 Summary:")
                    print(f"  Validation Score: {passed}/{total} checks passed")
                    
                    if passed == total:
                        print("🎉 Perfect! All enhanced preprocessing patterns working correctly!")
                        print("   - Isolated noise tokens eliminated")
                        print("   - ESC/VAL/REF properly combined")
                        print("   - ATR patterns correctly formatted")
                        print("   - Transaction codes cleaned up")
                        print("   - Compound tokens preserved")
                    else:
                        print(f"⚠️  {total - passed} validation checks failed")
                        print("   Some preprocessing patterns need further refinement")
                    
                    print()
                
                # Check if attention weights look better
                if 'token_importance' in bert_analysis:
                    token_importance = bert_analysis['token_importance']
                    print("💡 Top Important Tokens (BERT Attention):")
                    print("-" * 40)
                    
                    # Sort tokens by importance
                    sorted_tokens = sorted(token_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    for token, importance in sorted_tokens:
                        print(f"  {token:20} {importance:.4f}")
                    
                    print()
                    
                    # Check that important tokens are meaningful (not noise)
                    top_tokens = [token for token, _ in sorted_tokens[:5]]
                    meaningful_tokens = [
                        'DEVICE_ERROR', 'CARD_INSERTED', 'CARD_TAKEN', 'PIN_ENTERED',
                        'TRANSACTION_START', 'TRANSACTION_END', 'ESC_000', 'VAL_000', 
                        'REF_000', 'REJECTS_000', 'ATR_RECEIVED_T_0', 'OPCODE_FI', 
                        'OPCODE_IB', 'CardNumber', 'M_02', 'R_10011'
                    ]
                    
                    meaningful_count = sum(1 for token in top_tokens if token in meaningful_tokens)
                    print(f"🎯 Attention Quality: {meaningful_count}/5 top tokens are meaningful")
                    
                    if meaningful_count >= 3:
                        print("✅ BERT attention is focusing on important ATM events!")
                    else:
                        print("⚠️  BERT attention may still be distracted by noise tokens")
                
                print()
                
            else:
                print("❌ No BERT analysis found in response")
                
            # Check overall anomaly score
            if 'anomaly_score' in result:
                anomaly_score = result['anomaly_score']
                print(f"🔍 Anomaly Score: {anomaly_score:.4f}")
                print(f"🏷️  Classification: {'ANOMALY' if anomaly_score > 0.5 else 'NORMAL'}")
                
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection Error: {e}")
        print("Make sure the services are running with: docker compose up -d")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

if __name__ == "__main__":
    test_enhanced_preprocessing()
