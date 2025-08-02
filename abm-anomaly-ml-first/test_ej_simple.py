#!/usr/bin/env python3
"""
Simple test for EJ text cleaning implementation.
Tests both manual regex patterns and API integration to verify timestamp removal.
"""

import re
import requests

# Test data with [020t timestamp patterns
test_cases = [
    # Original test data with timestamp patterns
    "[020t 00:47:13 some text DEVICE ERROR more text",
    "[020t 15:30:22 REJECTS:000 normal operation", 
    "Some normal text [020t 09:12:45 CASH DISPENSER",
    "[020t timestamp only",
    "Normal text with DEVICE ERROR but no timestamp",
    "REJECTS:000 without timestamp",
    "[020t multiple [020t 10:20:30 timestamps [020t 14:15:16"
]

def test_regex_patterns():
    """Test the regex patterns manually"""
    print("=" * 60)
    print("🧪 TESTING REGEX PATTERNS MANUALLY")
    print("=" * 60)
    
    # The patterns we implemented
    pattern1 = r'\[020t\s+\d{2}:\d{2}:\d{2}'  # [020t + time format
    pattern2 = r'\[020t\s+'  # [020t + any whitespace 
    
    for i, test_text in enumerate(test_cases, 1):
        print("\nTest case {}: {}".format(i, test_text))
        
        # Apply both patterns like in our implementation
        cleaned = re.sub(pattern1, '', test_text)
        cleaned = re.sub(pattern2, '', cleaned)
        
        print("Cleaned:      {}".format(cleaned))
        
        # Check if cleaning worked
        if '[020t' in cleaned:
            print("❌ FAILED - Still contains [020t")
        else:
            print("✅ SUCCESS - [020t patterns removed")
            
        # Check if important content is preserved
        important_terms = ['DEVICE ERROR', 'REJECTS:000', 'CASH DISPENSER']
        preserved = [term for term in important_terms if term in test_text and term in cleaned]
        if preserved:
            print("✅ Preserved: {}".format(', '.join(preserved)))

def test_api_integration():
    """Test the API integration with EJ cleaning"""
    print("\n" + "=" * 60)
    print("🌐 TESTING API INTEGRATION")
    print("=" * 60)
    
    # Use a test case with both timestamp and important content
    test_text = "[020t 00:47:13 Transaction error DEVICE ERROR in cash dispenser [020t 15:30:22 REJECTS:000"
    
    print("Original text: {}".format(test_text))
    
    try:
        # Make API request
        print("\n📡 Making API request to BERT visualizer...")
        response = requests.post(
            'http://localhost:8000/api/v1/bert/visualize',
            json={'text': test_text},
            timeout=30
        )
        
        print("Status: {}".format(response.status_code))
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ BERT API Response received")
            
            # Check if we can see the processed text
            processed_text = result.get('data', {}).get('processed_text', '')
            tokens = result.get('data', {}).get('tokens', [])
            
            if processed_text:
                print("\nProcessed text length: {} characters".format(len(processed_text)))
                print("Token count: {}".format(len(tokens)))
                print("Processed text preview: {}".format(processed_text[:200] + "..." if len(processed_text) > 200 else processed_text))
                
                # Check for timestamp pattern removal
                ej_patterns = ['[020t 00:47:13', '[020t 15:30:22', '[020t']
                ej_patterns_found = [p for p in ej_patterns if p in processed_text]
                
                if ej_patterns_found:
                    print("❌ EJ cleaning FAILED - still found: {}".format(', '.join(ej_patterns_found)))
                else:
                    print("✅ EJ cleaning SUCCESS - no [020t patterns found")
                
                # Check if important content is preserved
                important_terms = ['DEVICE ERROR', 'REJECTS:000', 'Transaction error', 'cash dispenser']
                important_content = [term for term in important_terms if term.lower() in processed_text.lower()]
                
                if important_content:
                    print("✅ Important content preserved: {}".format(', '.join(important_content)))
                else:
                    print("⚠️  Warning: Important content may have been removed")
                
                # Check EJ contextual enhancement status
                ej_metadata = result.get('metadata', {}).get('ej_labeler', {})
                if ej_metadata.get('used'):
                    print("✅ EJ contextual enhancement still working after text cleaning")
                    print("   Enhancement impact: {:.3f}".format(ej_metadata.get('enhancement_impact', 0)))
                else:
                    print("⚠️  EJ contextual labeler not active")
                    
            else:
                print("⚠️  No processed text found in response")
                
        else:
            print("❌ API Error: {}".format(response.status_code))
            print("Response: {}".format(response.text))
            
    except requests.exceptions.ConnectionError as e:
        print("❌ Connection Error: {}".format(e))
        print("Make sure the API service is running")
    except Exception as e:
        print("❌ Error: {}".format(e))

if __name__ == "__main__":
    print("🧹 EJ TEXT CLEANING VERIFICATION")
    print("Testing EJ timestamp pattern removal...")
    
    test_regex_patterns()
    test_api_integration()
    
    print("\n" + "=" * 60)
    print("🏁 TEST COMPLETE")
    print("=" * 60)
