#!/usr/bin/env python3

import requests
import json

def test_bert_with_ej():
    url = 'http://localhost:8000/api/v1/bert/visualize'
    test_data = {'text': 'DEVICE ERROR', 'max_length': 20}
    
    print("🧪 Testing BERT API with EJ Contextual Enhancement")
    print(f"📝 Text: {test_data['text']}")
    print("=" * 50)
    
    try:
        response = requests.post(url, json=test_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS!")
            
            # Check metadata for EJ enhancement indicators
            metadata = result.get('metadata', {})
            print(f"\n🔍 Metadata ({len(metadata)} keys):")
            for key, value in metadata.items():
                if 'ej' in key.lower() or 'contextual' in key.lower():
                    print(f"  🎯 {key}: {value}")
                else:
                    print(f"     {key}: {value}")
            
            # Check token importance
            importance = result.get('token_importance', [])
            if importance:
                print(f"\n📊 Token Importance ({len(importance)} tokens):")
                sorted_tokens = sorted(importance, key=lambda x: x['importance'], reverse=True)
                
                for token in sorted_tokens:
                    token_text = token.get('token', 'N/A')
                    importance_score = token.get('importance', 0)
                    
                    if any(keyword.lower() in token_text.lower() for keyword in ['device', 'error']):
                        print(f"  🔴 {token_text}: {importance_score:.4f} ← PRIORITY TERM")
                    else:
                        print(f"     {token_text}: {importance_score:.4f}")
                        
                # Check if special tokens are filtered
                special_in_top_3 = [t for t in sorted_tokens[:3] if t.get('token', '') in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']]
                if special_in_top_3:
                    print(f"\n⚠️  Special tokens still in top 3: {[t['token'] for t in special_in_top_3]}")
                else:
                    print(f"\n✅ No special tokens in top 3 - filtering working!")
                    
            else:
                print("❌ No token importance data found")
                
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_bert_with_ej()
