#!/usr/bin/env python3
"""
Test machine status detection on the session file
"""
import re

def test_machine_status_detection(text: str):
    """Test the machine status detection logic"""
    
    # Regex pattern to extract machine status codes
    # Updated pattern to handle the actual format with control characters
    machine_status_pattern = re.compile(
        r'\*(\d+)\*(\d+)\*([A-Z]*)\(?(\d*)\*([^,]*),M-([^,]+),R-(\d+)',
        re.IGNORECASE
    )
    
    # Also try a simpler pattern that focuses on the M- and R- parts
    simple_pattern = re.compile(
        r'M-([^,\s]+).*?R-(\d+)',
        re.IGNORECASE
    )
    
    machine_status_matches = machine_status_pattern.findall(text)
    simple_matches = simple_pattern.findall(text)
    
    print(f"Complex pattern found {len(machine_status_matches)} matches")
    print(f"Simple pattern found {len(simple_matches)} matches")
    
    # Try the simple pattern if complex one fails
    if not machine_status_matches and simple_matches:
        print("Using simple pattern results:")
        for module_code, retry_count in simple_matches:
            print(f"  Module Code: {module_code}, Retry Count: {retry_count}")
    
    # Let's also print the raw text to debug
    print(f"\nRaw text for debugging:")
    for line in text.split('\n'):
        if 'M-' in line:
            print(f"  '{line}'")
    
    print(f"Found {len(machine_status_matches)} machine status codes:")
    
    # Define module code classifications
    error_module_codes = {
        '02': 'Communication Error',
        '03': 'Hardware Fault', 
        '04': 'Cash Dispenser Error',
        '05': 'Card Reader Error',
        '06': 'Receipt Printer Error',
        '07': 'Cash Cassette Error',
        '08': 'Security Module Error',
        '09': 'Pin Pad Error',
        '10': 'Display Error',
        '11': 'Network Communication Error',
        '12': 'Transaction Processing Error'
    }
    
    warning_module_codes = {
        '01': 'Minor Warning',
        '20': 'Maintenance Required',
        '21': 'Low Cash Warning',
        '22': 'Paper Low Warning'
    }
    
    # Module codes to ignore (known non-critical issues)
    ignored_module_codes = {
        '81': 'Chip Read Failure (Normal)',
        '00': 'Status OK',
        '090B0210B9': 'Diagnostic Status'
    }
    
    error_modules = []
    warning_modules = []
    ignored_modules = []
    
    # Debug: print what we captured
    print(f"Machine status matches: {machine_status_matches}")
    
    if machine_status_matches:
        for match in machine_status_matches:
            print(f"  Match groups: {match} (length: {len(match)})")
            # Extract the relevant parts based on actual capture groups
            if len(match) >= 7:
                trans_no, device_id, status_type, error_code, unknown, module_code, retry_count = match
            elif len(match) >= 6:
                trans_no, device_id, status_type, error_code, module_code, retry_count = match
            else:
                print(f"  Unexpected match format: {match}")
                continue
                
            retry_count_int = int(retry_count) if retry_count.isdigit() else 0
            
            print(f"  Transaction {trans_no}: Device {device_id}, Status {status_type}, Error {error_code}, Module {module_code}, Retries {retry_count}")
            
            # Classify module codes
            if module_code in error_module_codes:
                error_modules.append((module_code, error_module_codes[module_code]))
            elif module_code in warning_module_codes:
                warning_modules.append((module_code, warning_module_codes[module_code]))
            elif module_code in ignored_module_codes:
                ignored_modules.append((module_code, ignored_module_codes[module_code]))
            else:
                error_modules.append((module_code, f'Unknown Module Code: {module_code}'))
    
    else:
        # Use simple pattern results
        print("Using simple pattern results:")
        for module_code, retry_count in simple_matches:
            print(f"  Module Code: {module_code}, Retry Count: {retry_count}")
            if module_code in ignored_module_codes:
                ignored_modules.append((module_code, ignored_module_codes[module_code]))
            else:
                error_modules.append((module_code, f'Module Code: {module_code}'))
    
    print(f"\nClassification:")
    print(f"  Errors: {error_modules}")
    print(f"  Warnings: {warning_modules}")
    print(f"  Ignored: {ignored_modules}")
    
    return len(error_modules) > 0, len(warning_modules) > 0

# Test with the sample session content
sample_session = """[020t*726*06/18/2025*07:49*
     *TRANSACTION START*
[020t CARD INSERTED
[020t 07:49:48 CARD TAKEN
[000p[040q(I     *7289*1*D(1*0,M-090B0210B9,R-4S
[000p[040q(I     *7289*1*D(1*0,M-00,R-4S
[020t 07:49:49 TRANSACTION END"""

print("=== Testing Machine Status Detection ===")
has_errors, has_warnings = test_machine_status_detection(sample_session)
print(f"\nResult: Has Errors: {has_errors}, Has Warnings: {has_warnings}")
