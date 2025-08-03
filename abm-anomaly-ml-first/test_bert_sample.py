#!/usr/bin/env python3

import json
import requests

# Sample ABM log data
sample_log = """[020t*629*06/18/2025*00:46*
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

def test_bert_analysis():
    url = "http://localhost:8000/api/v1/bert/analyze"
    
    payload = {
        "text": sample_log,
        "analysis_type": "analyze_session"
    }
    
    try:
        print("Testing BERT analysis endpoint...")
        print(f"URL: {url}")
        print(f"Payload text length: {len(payload['text'])} characters")
        
        response = requests.post(url, json=payload, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response Keys: {list(data.keys())}")
            
            if 'results' in data:
                results = data['results']
                print(f"Results Keys: {list(results.keys())}")
                
                # Check token importance
                if 'token_importance' in results:
                    token_imp = results['token_importance']
                    print(f"Token importance type: {type(token_imp)}")
                    if isinstance(token_imp, list):
                        print(f"Token importance length: {len(token_imp)}")
                        if len(token_imp) > 0:
                            print(f"First token: {token_imp[0]}")
                    else:
                        print(f"Token importance content: {token_imp}")
                else:
                    print("No token_importance in results")
                
                # Check detected patterns
                if 'detected_patterns' in results:
                    patterns = results['detected_patterns']
                    print(f"Detected patterns type: {type(patterns)}")
                    if isinstance(patterns, list):
                        print(f"Detected patterns length: {len(patterns)}")
                        if len(patterns) > 0:
                            print(f"First pattern: {patterns[0]}")
                    else:
                        print(f"Detected patterns content: {patterns}")
                else:
                    print("No detected_patterns in results")
                
                # Check attention analysis
                if 'attention_analysis' in results:
                    attention = results['attention_analysis']
                    print(f"Attention analysis type: {type(attention)}")
                    print(f"Attention analysis keys: {list(attention.keys()) if isinstance(attention, dict) else 'Not a dict'}")
                else:
                    print("No attention_analysis in results")
                    
                # Save full response for inspection
                with open('/tmp/bert_response.json', 'w') as f:
                    json.dump(data, f, indent=2)
                print("Full response saved to /tmp/bert_response.json")
                
            else:
                print("No 'results' key in response")
                print(f"Response content: {data}")
        else:
            print(f"Error response: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_bert_analysis()
