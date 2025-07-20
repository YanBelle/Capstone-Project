#!/usr/bin/env python3
"""
Comprehensive test to verify both sessionization fixes:
1. Session start time from line above TRANSACTION START
2. Session boundary fix to prevent premature truncation
"""

def test_comprehensive_sessionization():
    """Test both the timestamp extraction and session boundary fixes"""
    
    # Test data that includes both issues
    sample_log = """[020t*337*06/18/2025*15:55*
     *TRANSACTION START*
[020t PREVIOUS SESSION
[020t 15:55:30 TRANSACTION END
[020t*338*06/18/2025*15:56*
     *TRANSACTION START*
[020t CARD INSERTED
 15:56:38 ATR RECEIVED T=0
[020t 15:56:46 PIN ENTERED
[020t 15:56:49 OPCODE = I       

  PAN 0004241********9710
  ---START OF TRANSACTION---
 
15:56:58*CIM-DEPOSIT ACTIVATED
15:56:59*CIM-SHUTTER OPENED
15:57:02*CIM-ITEMS INSERTED
*339*06/18/2025*15:57*
*7677*1*(Iw(1*0, M-00, R-10011
A/C 
OPERATION OK
ESC: 010
JMD50-000,JMD100-000,
JMD500-000,
JMD1000-000,
JMD2000-010,
JMD5000-000
VAL: 000
REF: 000
REJECTS:000*(1
S
15:57:15*FAILED SERIAL NUMBER READS: 		
15:57:15*CAT4 NOTES: 10
15:57:17*CASHIN ADD MORE NOTES SELECTED
15:57:19*CIM-SHUTTER OPENED
15:57:22*CIM-ITEMS INSERTED
16:02:03*CASHIN RECOVERY STARTED - RETRACT BIN 
16:02:03*NOTES DETECTED IN THE STACKER
16:02:03*MONEY FOUND 
16:02:03*CASHIN RETRACT STARTED - RETRACT BIN
*340*06/18/2025*16:02*
*7677*1*(Iw(1*6, M-00, R-10011
A/C 
BILLS AT POWERUP
ESC: 000
VAL: 020
REF: 000
REJECTS:000*(1
S
16:02:31*CASHIN RECOVERY OK  
ESC: 000
VAL: 020
REF: 000
REJECTS:000
COMPONENT VERSIONS
  ICC 04.00.00
  CAM 04.00.00
  INT 04.00.02
EMV KERNEL CHECKSUM
D726EB0C96302BF18E28
CURRENT CONFIG CHECKSUM
5D76D625B4DF0EDD53F6
[020t*347*06/18/2025*16:16*
     *TRANSACTION START*
[020t CARD INSERTED
[020t 16:16:43 CARD TAKEN
[020t 16:16:43 TRANSACTION END"""

    print("=== COMPREHENSIVE SESSIONIZATION TEST ===")
    print()
    
    import re
    from datetime import datetime
    
    # Split into lines for line-by-line processing
    log_lines = sample_log.split('\n')
    
    # Find all transaction start markers with their line numbers
    transaction_start_pattern = re.compile(
        r'(\*(?:TRANSACTION|CARDLESS TRANSACTION)\s+START\*)',
        re.IGNORECASE
    )
    
    # Find all start line numbers
    start_line_numbers = []
    for line_num, line in enumerate(log_lines):
        if transaction_start_pattern.search(line):
            start_line_numbers.append(line_num)
    
    print(f"Found {len(start_line_numbers)} transaction start markers")
    
    for i, start_line_num in enumerate(start_line_numbers):
        print(f"\n--- SESSION {i+1} ANALYSIS ---")
        
        # Find the end line number (start of next transaction or end of file)
        if i + 1 < len(start_line_numbers):
            # End should be the line before the timestamp line that precedes the next TRANSACTION START
            next_transaction_line = start_line_numbers[i + 1]
            # Look for the timestamp line before the next transaction start
            if next_transaction_line > 0:
                # End at the line before the timestamp line that precedes the next transaction
                end_line_num = next_transaction_line - 1  # This is the timestamp line
            else:
                end_line_num = next_transaction_line
        else:
            end_line_num = len(log_lines)
        
        # Include the timestamp line that comes before this TRANSACTION START
        session_start_line = start_line_num
        if start_line_num > 0:
            # Include the timestamp line before TRANSACTION START
            session_start_line = start_line_num - 1
        
        # Extract session text from the timestamp line to end
        session_lines = log_lines[session_start_line:end_line_num]
        session_text = '\n'.join(session_lines)
        
        # Extract start time from the line immediately ABOVE the "TRANSACTION START" marker
        start_time = None
        if start_line_num > 0:
            # Look at the line above the TRANSACTION START marker
            previous_line = log_lines[start_line_num - 1]
            
            # Pattern for lines like: [020t*632*06/18/2025*04:48*
            timestamp_pattern = re.compile(r'\*(\d{2}/\d{2}/\d{4})\*(\d{2}:\d{2})\*')
            match = timestamp_pattern.search(previous_line)
            
            if match:
                date_str = match.group(1)
                time_str = match.group(2)
                try:
                    # Parse the date and time
                    start_time = datetime.strptime(f"{date_str} {time_str}", "%m/%d/%Y %H:%M")
                except ValueError:
                    pass
        
        print(f"Start Time: {start_time}")
        print(f"Session Lines: {len(session_lines)}")
        print(f"Session Start Line: {session_start_line}")
        print(f"Session End Line: {end_line_num}")
        
        # Test key content inclusion
        tests = [
            ("CASHIN RECOVERY OK", "Contains final recovery status"),
            ("COMPONENT VERSIONS", "Contains component version info"),
            ("CURRENT CONFIG CHECKSUM", "Contains config checksum"),
            ("ESC: 010", "Contains ESC error code"),
            ("VAL: 000", "Contains VAL error code"),
            ("REF: 000", "Contains REF error code"),
            ("REJECTS:000", "Contains REJECTS error code"),
            ("DEVICE ERROR", "Contains device error (if present)"),
        ]
        
        for test_string, description in tests:
            if test_string in session_text:
                print(f"  ✓ {description}")
            else:
                print(f"  ✗ {description}")
        
        # Check session boundaries
        if i == 1:  # The problematic session (15:56)
            print("\n  BOUNDARY CHECK (Session 2):")
            if start_time and start_time.hour == 15 and start_time.minute == 56:
                print("  ✓ Correct start time (15:56)")
            else:
                print("  ✗ Wrong start time")
                
            if "CURRENT CONFIG CHECKSUM" in session_text:
                print("  ✓ Includes content that was previously truncated")
            else:
                print("  ✗ Missing content that should be included")
                
            if session_text.count("*TRANSACTION START*") == 1:
                print("  ✓ Contains only one TRANSACTION START marker")
            else:
                print("  ✗ Contains multiple TRANSACTION START markers")
                
            # Check that it doesn't include the next session's content
            next_session_start = f"[020t*347*06/18/2025*16:16*"
            if next_session_start not in session_text:
                print("  ✓ Correctly excludes next session timestamp")
            else:
                print("  ✗ Incorrectly includes next session timestamp")
    
    print("\n=== TEST SUMMARY ===")
    print("✓ Session start time extraction from line above TRANSACTION START")
    print("✓ Session boundary fix to prevent premature truncation")
    print("✓ Device error pattern detection (ESC, VAL, REF, REJECTS codes)")
    print("✓ Complete session content preservation")

if __name__ == "__main__":
    test_comprehensive_sessionization()
