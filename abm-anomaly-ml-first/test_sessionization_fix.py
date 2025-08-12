#!/usr/bin/env python3
"""
Test script to verify sessionization logic works correctly
"""

import re

def test_sessionization():
    """Test the sessionization logic with sample data"""
    
    # Read the actual EJ file
    file_path = "data/input/processed/ABM25EJ_20250613_20250613.txt"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()
        
        print(f"✅ Successfully read file: {len(raw_content)} characters")
        
        # Test transaction boundary detection
        transaction_pattern = r'(\*(?:TRANSACTION|CARDLESS TRANSACTION)\s+START\*)'
        matches = list(re.finditer(transaction_pattern, raw_content, re.IGNORECASE))
        
        print(f"🔍 Found {len(matches)} transaction boundaries using regex")
        
        if matches:
            print("\n📊 Session boundaries found:")
            for i, match in enumerate(matches[:10]):  # Show first 10
                session_start = match.start()
                context = raw_content[max(0, session_start-50):session_start+100]
                print(f"  Session {i+1}: Position {session_start}")
                print(f"    Context: {repr(context)}")
            
            if len(matches) > 10:
                print(f"    ... and {len(matches) - 10} more")
            
            # Test session splitting
            sessions = []
            for i, match in enumerate(matches):
                session_start = match.start()
                if i < len(matches) - 1:
                    session_end = matches[i + 1].start()
                else:
                    session_end = len(raw_content)
                
                session_content = raw_content[session_start:session_end].strip()
                sessions.append({
                    'id': i + 1,
                    'start': session_start,
                    'end': session_end,
                    'length': len(session_content),
                    'preview': session_content[:100] + "..." if len(session_content) > 100 else session_content
                })
            
            print(f"\n✅ Successfully split into {len(sessions)} sessions:")
            for session in sessions[:5]:  # Show first 5
                print(f"  Session {session['id']}: {session['length']} chars")
                print(f"    Preview: {repr(session['preview'])}")
            
            if len(sessions) > 5:
                print(f"    ... and {len(sessions) - 5} more sessions")
            
            return True, len(sessions)
        
        else:
            print("❌ No transaction boundaries found!")
            return False, 0
    
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False, 0

if __name__ == "__main__":
    print("🧪 Testing Sessionization Logic")
    print("=" * 50)
    
    success, session_count = test_sessionization()
    
    print("\n" + "=" * 50)
    if success:
        print(f"✅ SESSIONIZATION TEST PASSED")
        print(f"✅ Expected to create {session_count} individual sessions")
        print(f"✅ Logic is working correctly!")
    else:
        print(f"❌ SESSIONIZATION TEST FAILED")
        print(f"❌ Could not split file into sessions")
    
    print("\nNext step: Apply this logic to the API batch_process_ej_files function")
