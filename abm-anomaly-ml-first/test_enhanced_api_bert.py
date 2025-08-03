#!/usr/bin/env python3
"""
Test the enhanced BERT preprocessing through the API to verify all improvements
"""

import requests
import json

# Test the enhanced preprocessing with your specific EJ example
sample_ej = """[020t*629*06/18/2025*00:46*
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

print("=== ENHANCED BERT API TEST ===")
print("")

print("Testing enhanced BERT preprocessing through API...")
print("Original text length: {} characters".format(len(sample_ej)))
print("")

# Test the API endpoint
api_url = "http://localhost/api/v1/bert/analyze"

try:
    response = requests.post(
        api_url,
        json={"session_text": sample_ej, "session_id": "test_enhanced_preprocessing"},
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        
        print("✓ API Request successful")
        print("")
        
        # Extract key information
        processed_text = result.get('processed_text', '')
        token_count = result.get('token_count', 0)
        
        print("PREPROCESSING RESULTS:")
        print("Processed text length: {} characters".format(len(processed_text)))
        print("Token count: {}".format(token_count))
        print("Text reduction: {:.1f}%".format(((len(sample_ej) - len(processed_text)) / len(sample_ej)) * 100))
        print("")
        
        print("PROCESSED TEXT:")
        print("'{}'".format(processed_text))
        print("")
        
        # Check for key improvements
        improvements = []
        
        # Check if compound tokens are created
        if 'DEVICE_ERROR' in processed_text:
            improvements.append("✓ DEVICE ERROR -> DEVICE_ERROR (compound token)")
        if 'CARD_INSERTED' in processed_text:
            improvements.append("✓ CARD INSERTED -> CARD_INSERTED (compound token)")
        if 'PIN_ENTERED' in processed_text:
            improvements.append("✓ PIN ENTERED -> PIN_ENTERED (compound token)")
        if 'CARD_TAKEN' in processed_text:
            improvements.append("✓ CARD TAKEN -> CARD_TAKEN (compound token)")
        if 'TRANSACTION_START' in processed_text:
            improvements.append("✓ *TRANSACTION START* -> TRANSACTION_START")
        if 'TRANSACTION_END' in processed_text:
            improvements.append("✓ TRANSACTION END -> TRANSACTION_END (compound token)")
        
        # Check if PAN is replaced
        if 'CardNumber' in processed_text and 'PAN' not in processed_text:
            improvements.append("✓ PAN 0004263********1897 -> CardNumber")
        
        # Check if complex codes are removed
        if 'M-02' in processed_text and '*7231*1*(Iw(1*3,' not in processed_text:
            improvements.append("✓ *7231*1*(Iw(1*3, M-02, R-10011 -> M-02 R-10011")
        
        # Check if punctuation is cleaned
        if processed_text.count('*') < sample_ej.count('*'):
            improvements.append("✓ Reduced asterisk punctuation attention")
        
        print("KEY IMPROVEMENTS VERIFIED:")
        for improvement in improvements:
            print(improvement)
        
        # Check token importance data
        token_importance = result.get('token_importance', {})
        if token_importance:
            print("")
            print("TOKEN IMPORTANCE ANALYSIS:")
            
            # Check contextual enhancement
            contextual_info = token_importance.get('contextual_enhancement', {})
            if contextual_info:
                print("EJ Labeler used: {}".format(contextual_info.get('ej_labeler_used', False)))
                print("Expert Labeler used: {}".format(contextual_info.get('expert_labeler_used', False)))
                print("Enhancement impact: {:.2%}".format(contextual_info.get('enhancement_impact', 0)))
                print("Special tokens suppressed: {}".format(contextual_info.get('special_tokens_suppressed', False)))
            
            # Show top important tokens
            token_rankings = token_importance.get('token_rankings', [])
            if token_rankings:
                print("")
                print("TOP 10 IMPORTANT TOKENS:")
                for i, token_info in enumerate(token_rankings[:10]):
                    token = token_info.get('token', '')
                    combined_score = token_info.get('combined_importance', 0)
                    contextual_score = token_info.get('contextual_importance', 0)
                    print("{}. {} (combined: {:.4f}, contextual: {:.4f})".format(
                        i+1, token, combined_score, contextual_score))
        
        # Check if critical ATM terms are preserved and highly ranked
        critical_terms = ['DEVICE_ERROR', 'ESC', 'VAL', 'REF', 'REJECTS']
        preserved_terms = [term for term in critical_terms if term in processed_text]
        
        print("")
        print("CRITICAL ATM TERMS PRESERVED:")
        for term in critical_terms:
            if term in processed_text:
                print("✓ {} - PRESERVED".format(term))
            else:
                print("✗ {} - MISSING".format(term))
                
    else:
        print("✗ API Request failed with status code: {}".format(response.status_code))
        print("Response: {}".format(response.text))
        
except requests.exceptions.RequestException as e:
    print("✗ API Request failed: {}".format(e))
    print("")
    print("Make sure the services are running with 'docker compose up -d'")

print("")
print("=== TEST COMPLETE ===")
