import re
from datetime import datetime

def test_sessionization_simple():
    # Read a small sample from the actual file
    sample_lines = """[020t*601*06/18/2025*20:49*
     *PRIMARY CARD READER ACTIVATED*
[020t*602*06/18/2025*21:07*
     *TRANSACTION START*
[020t CARD INSERTED
[020t 21:07:15 CARD TAKEN
[000p[040q(I     *    *1*D(1*0,M-090B0210B9,R-4S
[000p[040q(I     *    *1*D(1*0,M-00,R-4S
[000p[040q(I     *    *1*D(1*0,M-070C000090,R-4S
[020t 21:07:15 TRANSACTION END"""

    print("Testing sessionization on sample...")
    
    # Split into lines for line-by-line processing
    log_lines = sample_lines.split('\n')
    
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
        print(f"\nSession {i+1}:")
        print(f"  Start line: {start_line_num}")
        print(f"  Start line content: {log_lines[start_line_num]}")
        
        # Extract start time from the line immediately ABOVE the "TRANSACTION START" marker
        start_time = None
        if start_line_num > 0:
            # Look at the line above the TRANSACTION START marker
            previous_line = log_lines[start_line_num - 1]
            print(f"  Previous line: {previous_line}")
            
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

if __name__ == "__main__":
    test_sessionization_simple()
