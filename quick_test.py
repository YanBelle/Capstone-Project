#!/usr/bin/env python3
"""
Quick manual test of BertViz integration
"""

import requests
import time

def test():
    print("Testing BertViz integration...")
    
    # Test health first
    try:
        health_response = requests.get("http://localhost/api/v1/health", timeout=5)
        print(f"Health check: {health_response.status_code}")
        print(f"Health response: {health_response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return
    
    # Get sessions
    try:
        sessions_response = requests.get("http://localhost/api/v1/sessions?limit=1", timeout=10)
        print(f"Sessions check: {sessions_response.status_code}")
        if sessions_response.status_code == 200:
            sessions_data = sessions_response.json()
            if sessions_data.get('sessions'):
                session_id = sessions_data['sessions'][0]['session_id']
                print(f"Testing with session: {session_id}")
                
                # Try to get session texts with a shorter timeout
                try:
                    print("Requesting session texts...")
                    texts_response = requests.get(f"http://localhost/api/v1/sessions/{session_id}/texts", timeout=30)
                    print(f"Texts response status: {texts_response.status_code}")
                    
                    if texts_response.status_code == 200:
                        texts_data = texts_response.json()
                        print(f"Response keys: {list(texts_data.keys())}")
                        print(f"Status: {texts_data.get('status')}")
                        
                        if 'text_lengths' in texts_data:
                            lengths = texts_data['text_lengths']
                            print(f"Text lengths: raw={lengths.get('raw', 0)}, cleaned={lengths.get('cleaned', 0)}")
                        
                        # Check for BertViz cleaning signs
                        cleaned_text = texts_data.get('cleaned_text', '')
                        if cleaned_text:
                            has_transaction_start = 'TRANSACTION_START' in cleaned_text
                            has_020t = '[020t' in cleaned_text
                            
                            print(f"BertViz cleaning signs:")
                            print(f"  Has TRANSACTION_START: {has_transaction_start}")
                            print(f"  Has [020t patterns: {has_020t}")
                            
                            if has_transaction_start and not has_020t:
                                print("SUCCESS: BertViz cleaning appears to be working!")
                            else:
                                print("INFO: BertViz cleaning may not be fully applied, but API is working")
                        else:
                            print("WARNING: No cleaned text returned")
                    else:
                        print(f"Texts request failed: {texts_response.text}")
                        
                except requests.exceptions.Timeout:
                    print("Texts request timed out - this might indicate a processing issue")
                except Exception as e:
                    print(f"Texts request error: {e}")
            else:
                print("No sessions available")
        else:
            print(f"Sessions request failed: {sessions_response.text}")
    except Exception as e:
        print(f"Sessions request error: {e}")

if __name__ == "__main__":
    test()
