#!/usr/bin/env python3
"""
Test to analyze the specific session mentioned by the user.
"""

def find_missing_session():
    """Find why the specific session wasn't picked up"""
    
    # Sample session from the user's request
    sample_session = """     *PRIMARY CARD READER ACTIVATED*
[020t*602*06/18/2025*21:07*
     *TRANSACTION START*
[020t CARD INSERTED
[020t 21:07:15 CARD TAKEN
[000p[040q(I     *    *1*D(1*0,M-090B0210B9,R-4S
[000p[040q(I     *    *1*D(1*0,M-00,R-4S
[000p[040q(I     *    *1*D(1*0,M-070C000090,R-4S
[020t 21:07:15 TRANSACTION END
[000p[040q(I     *    *1*D(1*0,M-1704000190,R-4S
[000p[040q(I     *    *1*D(1*0,M-2006000190,R-4S
[000p[040q(I     *    *1*D(1*1,M-2006000190,R-4S
[020t     *    *1*P*21,M-
*603*06/18/2025*21:15*
[05pSUPERVISOR MODE ENTRY[00p
[020t*604*06/18/2025*21:15*
[05pSUPERVISOR MODE EXIT[00p
     *    *1*R*09
     *    *1*P*20,M-"""

    print("Analyzing the specific session mentioned by the user...")
    
    # Check for transaction start pattern
    import re
    
    transaction_start_pattern = re.compile(
        r'(\*(?:TRANSACTION|CARDLESS TRANSACTION)\s+START\*)',
        re.IGNORECASE
    )
    
    matches = list(transaction_start_pattern.finditer(sample_session))
    print(f"Found {len(matches)} TRANSACTION START markers")
    
    if matches:
        for i, match in enumerate(matches):
            print(f"Match {i+1}: {match.group(0)} at position {match.start()}-{match.end()}")
            
        # Split into lines and find the line before the transaction start
        lines = sample_session.split('\n')
        
        for line_num, line in enumerate(lines):
            if transaction_start_pattern.search(line):
                print(f"TRANSACTION START found at line {line_num}: {line.strip()}")
                if line_num > 0:
                    previous_line = lines[line_num - 1]
                    print(f"Previous line: {previous_line.strip()}")
                    
                    # Check if previous line has timestamp pattern
                    timestamp_pattern = re.compile(r'\*(\d{2}/\d{2}/\d{4})\*(\d{2}:\d{2})\*')
                    timestamp_match = timestamp_pattern.search(previous_line)
                    
                    if timestamp_match:
                        date_str = timestamp_match.group(1)
                        time_str = timestamp_match.group(2)
                        print(f"Found timestamp: {date_str} {time_str}")
                        
                        # Parse the timestamp
                        from datetime import datetime
                        try:
                            session_time = datetime.strptime(f"{date_str} {time_str}", "%m/%d/%Y %H:%M")
                            print(f"Parsed session time: {session_time}")
                        except ValueError as e:
                            print(f"Error parsing timestamp: {e}")
                    else:
                        print("No timestamp pattern found in previous line")
                        
                        # Try other timestamp patterns
                        alt_patterns = [
                            r'(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2}:\d{2})',
                            r'(\d{4}/\d{2}/\d{2})\s+(\d{2}:\d{2}:\d{2})',
                            r'(\d{2}:\d{2}:\d{2})'
                        ]
                        
                        for pattern in alt_patterns:
                            alt_match = re.search(pattern, previous_line)
                            if alt_match:
                                print(f"Found alternative timestamp pattern: {alt_match.group(0)}")
                                break
    else:
        print("No TRANSACTION START markers found!")
        print("Sample text preview:")
        print(sample_session[:200])

if __name__ == "__main__":
    find_missing_session()
