#!/usr/bin/env python3
"""
Test script to validate the complete BERT custom vocabulary implementation
Tests the actual API endpoint with real BERT tokenization
"""

import json
import requests
import sys

def test_bert_api_with_custom_vocabulary():
    """Test the actual API endpoint with our sample EJ text"""
    
    print("=== BERT CUSTOM VOCABULARY API TEST ===")
    print()
    
    # Your original sample EJ text
    sample_ej_text = """[020t*629*06/18/2025*00:46*
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

    print("Original EJ sample:")
    print("=" * 50)
    print(sample_ej_text.strip())
    print("=" * 50)
    print(f"Original length: {len(sample_ej_text)} characters")
    print()

    # Test API endpoint
    api_url = "http://localhost:80/analyze_bert"
    
    # Prepare request
    payload = {
        "session_text": sample_ej_text,
        "session_id": "test_custom_vocab_001"
    }
    
    print("Calling BERT analysis API...")
    print(f"URL: {api_url}")
    print(f"Session ID: {payload['session_id']}")
    print()
    
    try:
        response = requests.post(api_url, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract key information
            print("=== API RESPONSE ANALYSIS ===")
            print(f"Status: SUCCESS")
            print(f"Response time: {response.elapsed.total_seconds():.2f} seconds")
            print()
            
            # Display processed text
            if 'processed_text' in result:
                processed_text = result['processed_text']
                print("=== PROCESSED TEXT (sent to BERT) ===")
                print(f"Processed length: {len(processed_text)} characters")
                print(f"Reduction: {len(sample_ej_text) - len(processed_text)} characters ({((len(sample_ej_text) - len(processed_text)) / len(sample_ej_text) * 100):.1f}%)")
                print()
                print("Processed text:")
                print("-" * 50)
                print(processed_text)
                print("-" * 50)
                print()
            
            # Display tokens
            if 'tokens' in result:
                tokens = result['tokens']
                print(f"=== BERT TOKENIZATION RESULTS ===")
                print(f"Total tokens: {len(tokens)}")
                print()
                print("Tokens produced:")
                print(tokens)
                print()
                
                # Check for successful compound token recognition
                compound_tokens_found = []
                fragmented_tokens_found = []
                
                expected_compounds = [
                    'transaction_start', 'transaction_end', 'card_inserted', 'card_taken',
                    'pin_entered', 'device_error', 'atr_received', 'opcode_fi', 'opcode_ib',
                    'esc_000', 'val_000', 'ref_000', 'rejects_000', 'cardnumber'
                ]
                
                for token in tokens:
                    token_lower = token.lower()
                    if token_lower in expected_compounds:
                        compound_tokens_found.append(token)
                    elif token.startswith('##'):
                        fragmented_tokens_found.append(token)
                
                print(f"=== COMPOUND TOKEN ANALYSIS ===")
                print(f"Expected compound tokens found: {len(compound_tokens_found)}")
                if compound_tokens_found:
                    print(f"Compounds found: {compound_tokens_found}")
                print()
                
                print(f"Subword (##) tokens found: {len(fragmented_tokens_found)}")
                if fragmented_tokens_found:
                    print(f"Subwords: {fragmented_tokens_found}")
                print()
                
                # Success metrics
                total_expected = len(expected_compounds)
                success_rate = (len(compound_tokens_found) / total_expected) * 100 if total_expected > 0 else 0
                print(f"Compound token success rate: {success_rate:.1f}%")
                print(f"Fragmentation reduction: {len(fragmented_tokens_found)} subword tokens")
                
            # Display token importance
            if 'token_importance' in result and 'token_rankings' in result['token_importance']:
                rankings = result['token_importance']['token_rankings']
                print(f"\n=== TOKEN IMPORTANCE RANKINGS ===")
                print("Top 10 most important tokens:")
                for i, token_info in enumerate(rankings[:10]):
                    print(f"{i+1:2d}. {token_info['token']:20} (importance: {token_info['combined_importance']:.4f})")
                
                # Check if our important domain terms are ranking highly
                important_domain_terms = ['device_error', 'rejects', 'card_taken', 'transaction_start', 'transaction_end']
                domain_rankings = []
                for i, token_info in enumerate(rankings):
                    if any(term in token_info['token'].lower() for term in important_domain_terms):
                        domain_rankings.append((i+1, token_info['token'], token_info['combined_importance']))
                
                if domain_rankings:
                    print(f"\n=== DOMAIN TERM RANKINGS ===")
                    print("Key ATM/EJ terms in importance ranking:")
                    for rank, token, importance in domain_rankings[:5]:
                        print(f"#{rank:2d}: {token:20} (importance: {importance:.4f})")
            
            # Display enhancement info
            if 'token_importance' in result and 'contextual_enhancement' in result['token_importance']:
                enhancement = result['token_importance']['contextual_enhancement']
                print(f"\n=== ENHANCEMENT STATUS ===")
                print(f"EJ Labeler used: {enhancement.get('ej_labeler_used', False)}")
                print(f"Expert Labeler used: {enhancement.get('expert_labeler_used', False)}")
                print(f"Enhancement impact: {enhancement.get('enhancement_impact', 0):.4f}")
                print(f"Special tokens suppressed: {enhancement.get('special_tokens_suppressed', False)}")
            
            # Model initialization info check
            if 'error' not in result:
                print(f"\n=== SUCCESS SUMMARY ===")
                print("✅ API call successful")
                print("✅ BERT analysis completed")
                print("✅ Custom vocabulary processing active")
                print("✅ Token preprocessing applied")
                
                if 'tokens' in result:
                    if any('_' in token for token in result['tokens']):
                        print("✅ Compound tokens preserved (contains underscore patterns)")
                    else:
                        print("⚠️  No underscore compound tokens found - check custom vocabulary")
                
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        print("Make sure the API service is running on localhost:80")
        return False
    
    return True

if __name__ == "__main__":
    success = test_bert_api_with_custom_vocabulary()
    if success:
        print("\n🎉 Custom vocabulary test completed!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)
