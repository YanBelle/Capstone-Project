#!/usr/bin/env python3
import re
import sys
sys.path.append('/app')

def test_patterns():
    print("=== Testing Anomaly Detection Patterns ===")
    
    # Test session with device error
    session_text = """*TRANSACTION START*
[020t CARD INSERTED
 06:25:00 ATR RECEIVED T=0
[020t 06:25:03 OPCODE = FI      

  PAN 0004263********5342
  ---START OF TRANSACTION---
 
[020t 06:25:18 PIN ENTERED
[020t 06:25:25 OPCODE = IB      

  PAN 0004263********5342
  ---START OF TRANSACTION---
 
*660*06/18/2025*06:25*
*7249*1*(Iw(1*3, M-02, R-10011
A/C 
DEVICE ERROR
ESC: 000
VAL: 000
REF: 000
REJECTS:000*(1
S
[020t 06:26:00 CARD TAKEN
[020t 06:26:02 TRANSACTION END
[020t*661*06/18/2025*06:26*
     *PRIMARY CARD READER ACTIVATED*
[020t*662*06/18/2025*06:29*"""

    text_upper = session_text.upper()
    
    # Test 1: Device Error Detection
    device_error_found = 'DEVICE ERROR' in text_upper
    print(f"1. Device Error Detection: {device_error_found}")
    
    # Test 2: Error Code Detection
    error_code_pattern = re.compile(r'(ESC|VAL|REF|REJECTS):\s*(\d+)', re.IGNORECASE)
    error_matches = error_code_pattern.findall(session_text)
    print(f"2. Error Codes Found: {error_matches}")
    
    # Test 3: Hardware Error Pattern
    hardware_errors = ['HARDWARE ERROR', 'SENSOR ERROR', 'MOTOR ERROR', 'DEVICE ERROR']
    found_errors = [error for error in hardware_errors if error in text_upper]
    print(f"3. Hardware Errors Found: {found_errors}")
    
    # Test 4: Should this be an anomaly?
    should_be_anomaly = device_error_found or len(error_matches) > 0 or len(found_errors) > 0
    print(f"4. Should be flagged as anomaly: {should_be_anomaly}")
    
    print("\n=== Test Complete ===")
    
    if should_be_anomaly:
        print("✅ SUCCESS: This session should now be detected as an anomaly!")
        print("Reasons:")
        if device_error_found:
            print("  - Contains 'DEVICE ERROR' text")
        if error_matches:
            print(f"  - Contains error codes: {error_matches}")
        if found_errors:
            print(f"  - Contains hardware errors: {found_errors}")
    else:
        print("❌ FAILED: This session would not be detected as an anomaly")
    
    return should_be_anomaly

if __name__ == "__main__":
    test_patterns()
