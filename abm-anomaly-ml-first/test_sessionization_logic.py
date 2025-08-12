#!/usr/bin/env python3
"""
Test script to verify sessionization logic works
"""

def test_fallback_sessionization():
    """Test the fallback sessionization logic"""
    print("Testing fallback sessionization...")
    
    # Sample content with 3 transactions
    raw_content = """Some header stuff
*TRANSACTION START*
[020t CARD INSERTED
 06:10:47 ATR RECEIVED T=0
[020t 06:10:50 OPCODE = FI      
[020t 06:11:03 PIN ENTERED
[020t 06:11:10 OPCODE = IB      
Some footer
*TRANSACTION START*
[020t CARD INSERTED
 07:10:47 ATR RECEIVED T=0
[020t 07:10:50 OPCODE = FI      
[020t 07:11:03 PIN ENTERED
[020t 07:11:10 OPCODE = IB      
More data
*TRANSACTION START*
[020t CARD INSERTED
 08:10:47 ATR RECEIVED T=0
[020t 08:10:50 OPCODE = FI      
[020t 08:11:03 PIN ENTERED
[020t 08:11:10 OPCODE = IB      
Final data"""
    
    # Fallback sessionization logic (copied from main.py)
    lines = raw_content.split('\n')
    current_session = []
    sessions_from_file = 0
    all_sessions = []
    
    for line in lines:
        if line.strip():
            # Start new session on transaction start indicators
            if any(keyword in line.upper() for keyword in ['*TRANSACTION START*', '*CARDLESS TRANSACTION START*']):
                if current_session:
                    # Process previous session
                    session_text = '\n'.join(current_session)
                    session_id = f"test_{sessions_from_file+1:04d}"
                    all_sessions.append({
                        'id': session_id,
                        'text': session_text,
                        'length': len(session_text)
                    })
                    print(f"Created session {session_id} with {len(session_text)} chars")
                    sessions_from_file += 1
                    current_session = []
            
            current_session.append(line)
    
    # Process final session
    if current_session:
        session_text = '\n'.join(current_session)
        session_id = f"test_{sessions_from_file+1:04d}"
        all_sessions.append({
            'id': session_id,
            'text': session_text,
            'length': len(session_text)
        })
        print(f"Created final session {session_id} with {len(session_text)} chars")
    
    total_sessions = sessions_from_file + (1 if current_session else 0)
    print(f"\nTotal sessions created: {total_sessions}")
    
    return all_sessions

if __name__ == "__main__":
    sessions = test_fallback_sessionization()
    for session in sessions:
        print(f"\nSession {session['id']}:")
        print(f"Length: {session['length']} chars")
        print(f"Preview: {session['text'][:100]}...")
