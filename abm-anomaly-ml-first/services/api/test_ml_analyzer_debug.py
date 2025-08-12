#!/usr/bin/env python3
"""
Test script to debug ML analyzer import and sessionization
"""

def test_ml_analyzer_import():
    """Test importing and using the ML analyzer"""
    print("Testing ML analyzer import...")
    
    try:
        from ml_analyzer import MLFirstAnomalyDetector
        print("✓ Successfully imported MLFirstAnomalyDetector")
        
        # Test instantiation
        analyzer = MLFirstAnomalyDetector()
        print("✓ Successfully created MLFirstAnomalyDetector instance")
        
        # Test sessionization with sample data
        sample_content = """*TRANSACTION START*
[020t CARD INSERTED
 06:10:47 ATR RECEIVED T=0
[020t 06:10:50 OPCODE = FI      
[020t 06:11:03 PIN ENTERED
[020t 06:11:10 OPCODE = IB      
*TRANSACTION START*
[020t CARD INSERTED
 07:10:47 ATR RECEIVED T=0
[020t 07:10:50 OPCODE = FI      
[020t 07:11:03 PIN ENTERED
[020t 07:11:10 OPCODE = IB      
*TRANSACTION START*
[020t CARD INSERTED
 08:10:47 ATR RECEIVED T=0
[020t 08:10:50 OPCODE = FI      
[020t 08:11:03 PIN ENTERED
[020t 08:11:10 OPCODE = IB      """
        
        print(f"Testing with sample content that should create 3 sessions...")
        sessions = analyzer.split_into_sessions(sample_content, "test_file.txt")
        print(f"✓ split_into_sessions returned {len(sessions)} sessions")
        
        for i, session in enumerate(sessions):
            print(f"  Session {i+1}: ID={session.session_id}, Length={len(session.text)} chars")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except AttributeError as e:
        print(f"✗ Attribute error: {e}")
        return False
    except Exception as e:
        print(f"✗ Other error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_ml_analyzer_import()
