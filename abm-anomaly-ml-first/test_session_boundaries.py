#!/usr/bin/env python3
"""
Test to verify the session boundary fix for the truncated session issue.
"""
import re
from datetime import datetime

def test_session_boundaries():
    """Test that sessions are not truncated prematurely"""
    
    # Sample log content that mimics the structure from the user's example
    sample_log = """[020t*337*06/18/2025*15:55*
     *TRANSACTION START*
[020t PREVIOUS SESSION END
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
[0r(1)2[000p[040qe1w3h16216:02:03*CASHIN RECOVERY STARTED - RETRACT BIN 
16:02:03*NOTES DETECTED IN THE STACKER
16:02:03*MONEY FOUND 
16:02:03*CASHIN RETRACT STARTED - RETRACT BIN
*340*06/18/2025*16:02*
*7677*1*(Iw(1*6, M-00, R-10011
A/C 
BILLS AT POWERUP
ESC: 000
VAL: 020
JMD50-000,JMD100-000,
JMD500-000,
JMD1000-000,
JMD2000-000,
JMD5000-000
REF: 000
REJECTS:000*(1
S
16:02:31*CASHIN RECOVERY OK  
ESC: 000
VAL: 020
JMD50-000,JMD100-000,
JMD500-000,
JMD1000-000,
JMD2000-000,
JMD5000-000
REF: 000
REJECTS:000

DENOM   CASS1   CASS2   CASS3   CASS4   
TOTAL   0       0       0       0       
DENOM   REJECT  CNTRFEIT
TOTAL   0       0       

COMPONENT VERSIONS
  ICC 04.00.00
  CAM 04.00.00
  INT 04.00.02
EMV KERNEL CHECKSUM
D726EB0C96302BF18E28
EMV LEVEL 2 CONFIG CHECKSUM
788214FDD34AF47915D0
CURRENT CONFIG CHECKSUM
5D76D625B4DF0EDD53F6
[020t*341*06/18/2025*16:03*
[05pPOWER-UP/RESET
APTRA ADVANCE NDC 05.01.00[00p
     *    *1*B*0750,M-
[020t     *    *1*P*21,M-
*342*06/18/2025*16:03*
[05pSUPERVISOR MODE ENTRY[00p
[020t*343*06/18/2025*16:04*
[05pCPM BIN DOOR REMOVED[00p
(I     (1
[020t*344*06/18/2025*16:05*
[05pCPM BIN DOOR INSERTED[00p
(I     (1
[0r(1)2[000p[040qe1w3h162COMPONENT VERSIONS
  ICC 04.00.00
  CAM 04.00.00
  INT 04.00.02
EMV KERNEL CHECKSUM
D726EB0C96302BF18E28
EMV LEVEL 2 CONFIG CHECKSUM
788214FDD34AF47915D0
CURRENT CONFIG CHECKSUM
5D76D625B4DF0EDD53F6
[020t*345*06/18/2025*16:11*
[05pPOWER-UP/RESET
APTRA ADVANCE NDC 05.01.00[00p
     *    *1*B*0750,M-
[020t*346*06/18/2025*16:11*
     *PRIMARY CARD READER ACTIVATED*
[020t*347*06/18/2025*16:16*
     *TRANSACTION START*
[020t CARD INSERTED
[020t 16:16:43 CARD TAKEN"""

    print("Testing session boundary logic...")
    
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
            print(f"Found TRANSACTION START at line {line_num}: {log_lines[line_num].strip()}")
    
    print(f"\nFound {len(start_line_numbers)} transaction start markers")
    
    # Test the new boundary logic
    sessions = []
    for i, start_line_num in enumerate(start_line_numbers):
        print(f"\n--- Processing Session {i+1} ---")
        print(f"Start line number: {start_line_num}")
        
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
        
        print(f"End line number: {end_line_num}")
        
        # Include the timestamp line that comes before this TRANSACTION START
        session_start_line = start_line_num
        if start_line_num > 0:
            # Include the timestamp line before TRANSACTION START
            session_start_line = start_line_num - 1
        
        print(f"Session start line: {session_start_line}")
        print(f"Session start content: {log_lines[session_start_line].strip()}")
        
        # Extract session text from the timestamp line to end
        session_lines = log_lines[session_start_line:end_line_num]
        session_text = '\n'.join(session_lines)
        
        print(f"Session length: {len(session_lines)} lines")
        print(f"Session preview (first 200 chars): {session_text[:200]}...")
        print(f"Session ending (last 200 chars): ...{session_text[-200:]}")
        
        # Check if session contains the expected elements
        if "CASHIN RECOVERY OK" in session_text:
            print("  ✓ Contains 'CASHIN RECOVERY OK'")
        else:
            print("  ✗ Missing 'CASHIN RECOVERY OK'")
            
        if "COMPONENT VERSIONS" in session_text:
            print("  ✓ Contains 'COMPONENT VERSIONS'")
        else:
            print("  ✗ Missing 'COMPONENT VERSIONS'")
            
        if "CURRENT CONFIG CHECKSUM" in session_text:
            print("  ✓ Contains 'CURRENT CONFIG CHECKSUM'")
        else:
            print("  ✗ Missing 'CURRENT CONFIG CHECKSUM'")
        
        # Extract start time from the line immediately ABOVE the "TRANSACTION START" marker
        start_time = None
        if start_line_num > 0:
            # Look at the line above the TRANSACTION START marker
            previous_line = log_lines[start_line_num - 1]
            print(f"Previous line for timestamp: {previous_line.strip()}")
            
            # Pattern for lines like: [020t*632*06/18/2025*04:48*
            timestamp_pattern = re.compile(r'\*(\d{2}/\d{2}/\d{4})\*(\d{2}:\d{2})\*')
            match = timestamp_pattern.search(previous_line)
            
            if match:
                date_str = match.group(1)
                time_str = match.group(2)
                try:
                    # Parse the date and time
                    start_time = datetime.strptime(f"{date_str} {time_str}", "%m/%d/%Y %H:%M")
                    print(f"  ✓ Extracted start time: {start_time}")
                except ValueError:
                    print(f"  ✗ Could not parse timestamp from line: {previous_line}")
            else:
                print(f"  ✗ No timestamp pattern found in previous line")
        
        sessions.append({
            'session_text': session_text,
            'start_time': start_time,
            'line_count': len(session_lines)
        })
        
    return sessions

if __name__ == "__main__":
    test_session_boundaries()
