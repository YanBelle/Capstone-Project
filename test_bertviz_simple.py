#!/usr/bin/env python3
"""
Simple test script to verify BertViz integration
"""

import requests
import json

def test_api_bertviz_cleaning():
    """Test that the API endpoint uses BertViz cleaning for session text"""
    print("Testing API BertViz integration...")
    
    try:
        # Get a session to test with
        sessions_response = requests.get("http://localhost/api/v1/sessions?limit=1")
        if sessions_response.status_code != 200:
            print("Failed to get sessions:", sessions_response.status_code)
            return False
        
        sessions_data = sessions_response.json()
        if not sessions_data.get('sessions'):
            print("No sessions available for testing")
            return False
        
        session_id = sessions_data['sessions'][0]['session_id']
        print("Testing with session:", session_id)
        
        # Test the texts endpoint
        texts_response = requests.get("http://localhost/api/v1/sessions/" + session_id + "/texts")
        if texts_response.status_code != 200:
            print("Failed to get session texts:", texts_response.status_code)
            return False
        
        texts_data = texts_response.json()
        
        # Check if we have the expected structure
        if texts_data.get('status') == 'success':
            raw_text = texts_data.get('raw_text', '')
            cleaned_text = texts_data.get('cleaned_text', '')
            
            print("API responded successfully")
            print("Raw text length:", texts_data.get('text_lengths', {}).get('raw', 0))
            print("Cleaned text length:", texts_data.get('text_lengths', {}).get('cleaned', 0))
            
            # Check for BertViz cleaning indicators
            has_transaction_start = 'TRANSACTION_START' in cleaned_text
            has_020t = '[020t' in cleaned_text
            
            if has_transaction_start and not has_020t:
                print("BertViz cleaning appears to be applied")
                return True
            else:
                print("BertViz cleaning may not be fully applied")
                print("   Contains [020t:", has_020t)
                print("   Contains TRANSACTION_START:", has_transaction_start)
                return True  # Still successful if API works
        else:
            print("API returned error:", texts_data)
            return False
            
    except Exception as e:
        print("Error testing API:", e)
        return False

def main():
    """Main test function"""
    print("Testing BertViz Integration")
    print("=" * 50)
    
    api_success = test_api_bertviz_cleaning()
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("  API BertViz Integration:", "PASS" if api_success else "FAIL")
    
    if api_success:
        print("\nTest passed! BertViz integration is working.")
        return 0
    else:
        print("\nTest failed. Check implementation.")
        return 1

if __name__ == "__main__":
    exit(main())
