#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple EJ Anomaly Detection Testing Script
==========================================
This script tests individual EJ sessions using pattern matching to see if they
should be flagged as anomalies.

Usage:
    python3 simple_test.py
"""

import re
from datetime import datetime

def test_ej_session_patterns():
    """Test EJ sessions using pattern matching"""
    print("=" * 60)
    print("[TEST] EJ ANOMALY DETECTION PATTERN TEST")
    print("=" * 60)
    
    # Test sessions with expected results
    test_sessions = [
        {
            "name": "Device Error Session",
            "description": "Session with DEVICE ERROR - should be flagged",
            "should_detect": True,
            "session_text": """*TRANSACTION START*
[020t*630*06/18/2025*06:25*
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
[020t*661*06/18/2025*06:26*"""
        },
        {
            "name": "Invalid Amount Session", 
            "description": "Session with INVALID AMOUNT - should be flagged",
            "should_detect": True,
            "session_text": """*TRANSACTION START*
[020t*1085*06/18/2025*09:42*
[020t CARD INSERTED
 09:42:15 ATR RECEIVED T=0
[020t 09:42:18 OPCODE = WD      

  PAN 0004263********5342
  ---START OF TRANSACTION---
 
[020t 09:42:30 PIN ENTERED
[020t 09:42:35 AMOUNT ENTERED: $200.00

*1095*06/18/2025*09:42*
*7249*1*(Iw(1*3, M-02, R-10011
A/C 
   INVALID AMOUNT
ESC: 000
VAL: 000
REF: 000
REJECTS:000*(1
S
[020t 09:43:00 CARD TAKEN
[020t 09:43:02 TRANSACTION END
[020t*1096*06/18/2025*09:43*"""
        },
        {
            "name": "Successful Withdrawal",
            "description": "Normal successful withdrawal - should NOT be flagged", 
            "should_detect": False,
            "session_text": """*TRANSACTION START*
[020t*500*06/18/2025*14:30*
[020t CARD INSERTED
 14:30:15 ATR RECEIVED T=0
[020t 14:30:18 OPCODE = WD      

  PAN 0004263********5342
  ---START OF TRANSACTION---
 
[020t 14:30:30 PIN ENTERED
[020t 14:30:35 AMOUNT ENTERED: $100.00
[020t 14:30:40 NOTES STACKED
[020t 14:30:42 NOTES PRESENTED
[020t 14:30:50 NOTES TAKEN
[020t 14:30:55 RECEIPT PRINTED
[020t 14:31:00 CARD TAKEN
[020t 14:31:02 TRANSACTION END
[020t*501*06/18/2025*14:31*"""
        },
        {
            "name": "Unable to Process Session",
            "description": "Host communication failure - should be flagged",
            "should_detect": True,
            "session_text": """*TRANSACTION START*
[020t*180*06/18/2025*11:15*
[020t CARD INSERTED
 11:15:00 ATR RECEIVED T=0
[020t 11:15:03 OPCODE = WD      

  PAN 0004263********5342
  ---START OF TRANSACTION---
 
[020t 11:15:18 PIN ENTERED
[020t 11:15:25 AMOUNT ENTERED: $50.00

*190*06/18/2025*11:15*
*7249*1*(Iw(1*3, M-02, R-10011
A/C 
   UNABLE TO PROCESS
ESC: 000
VAL: 000
REF: 000
REJECTS:000*(1
S
[020t 11:16:00 CARD TAKEN
[020t 11:16:02 TRANSACTION END
[020t*191*06/18/2025*11:16*"""
        }
    ]
    
    results = []
    
    for i, test_session in enumerate(test_sessions, 1):
        print(f"\n[TEST {i}] {test_session['name']}")
        print(f"[DESC] {test_session['description']}")
        print("-" * 50)
        
        # Analyze the session text
        text_upper = test_session['session_text'].upper()
        
        # Pattern checks for anomalies
        device_error = 'DEVICE ERROR' in text_upper
        invalid_amount = 'INVALID AMOUNT' in text_upper  
        unable_to_process = 'UNABLE TO PROCESS' in text_upper
        hardware_fault = 'HARDWARE FAULT' in text_upper or 'SENSOR ERROR' in text_upper
        
        # Error codes
        error_pattern = re.compile(r'(ESC|VAL|REF|REJECTS):\s*(\d+)', re.IGNORECASE)
        error_codes = error_pattern.findall(test_session['session_text'])
        
        # Check for positive transaction indicators (should NOT be anomalies)
        notes_taken = 'NOTES TAKEN' in text_upper
        successful_completion = notes_taken and 'CARD TAKEN' in text_upper
        
        # Determine if anomaly detected
        has_error_patterns = any([device_error, invalid_amount, unable_to_process, hardware_fault])
        has_error_codes = len(error_codes) > 0
        
        # Override: if it's a successful transaction, don't flag as anomaly
        if successful_completion and not has_error_patterns:
            detected_anomaly = False
        else:
            detected_anomaly = has_error_patterns or has_error_codes
        
        # Check result
        correct = detected_anomaly == test_session['should_detect']
        
        # Display analysis
        print(f"[PATTERNS] Pattern Analysis:")
        print(f"   - Device Error: {device_error}")
        print(f"   - Invalid Amount: {invalid_amount}")
        print(f"   - Unable to Process: {unable_to_process}")
        print(f"   - Hardware Fault: {hardware_fault}")
        print(f"   - Error Codes: {error_codes}")
        print(f"   - Notes Taken: {notes_taken}")
        print(f"   - Successful Completion: {successful_completion}")
        
        print(f"[RESULT] Anomaly Detected: {'YES' if detected_anomaly else 'NO'}")
        print(f"[EXPECTED] Should Detect: {'YES' if test_session['should_detect'] else 'NO'}")
        print(f"[STATUS] {'CORRECT' if correct else 'INCORRECT'}")
        
        results.append({
            'name': test_session['name'],
            'detected': detected_anomaly,
            'expected': test_session['should_detect'],
            'correct': correct
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("[SUMMARY] TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    correct_detections = sum(1 for r in results if r['correct'])
    
    print(f"Total Tests: {total_tests}")
    print(f"Correct Detections: {correct_detections}")
    print(f"Accuracy: {(correct_detections/total_tests)*100:.1f}%")
    
    print(f"\n[DETAILS] Detailed Results:")
    for result in results:
        status = "[CORRECT]" if result['correct'] else "[INCORRECT]"
        detected = "ANOMALY" if result['detected'] else "NORMAL"
        expected = "ANOMALY" if result['expected'] else "NORMAL"
        print(f"{status} {result['name']}: Detected={detected}, Expected={expected}")
    
    return correct_detections == total_tests

def test_custom_session():
    """Test a custom EJ session provided by user"""
    print("\n" + "=" * 60)
    print("[CUSTOM] Test Your Own EJ Session")
    print("=" * 60)
    print("You can modify this function to test your own EJ session text")
    
    # Example: Test the exact session from the original file
    custom_session = """*TRANSACTION START*
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
    
    print("[ANALYZING] Custom session...")
    text_upper = custom_session.upper()
    
    # Run the same analysis
    device_error = 'DEVICE ERROR' in text_upper
    invalid_amount = 'INVALID AMOUNT' in text_upper  
    unable_to_process = 'UNABLE TO PROCESS' in text_upper
    hardware_fault = 'HARDWARE FAULT' in text_upper or 'SENSOR ERROR' in text_upper
    
    error_pattern = re.compile(r'(ESC|VAL|REF|REJECTS):\s*(\d+)', re.IGNORECASE)
    error_codes = error_pattern.findall(custom_session)
    
    notes_taken = 'NOTES TAKEN' in text_upper
    successful_completion = notes_taken and 'CARD TAKEN' in text_upper
    
    has_error_patterns = any([device_error, invalid_amount, unable_to_process, hardware_fault])
    has_error_codes = len(error_codes) > 0
    
    if successful_completion and not has_error_patterns:
        detected_anomaly = False
    else:
        detected_anomaly = has_error_patterns or has_error_codes
    
    print(f"[ANALYSIS] Results:")
    print(f"   Device Error Found: {device_error}")
    print(f"   Invalid Amount Found: {invalid_amount}")
    print(f"   Unable to Process Found: {unable_to_process}")
    print(f"   Hardware Fault Found: {hardware_fault}")
    print(f"   Error Codes Found: {error_codes}")
    print(f"   Notes Taken: {notes_taken}")
    print(f"   Successful Completion: {successful_completion}")
    print(f"   ANOMALY DETECTED: {'YES' if detected_anomaly else 'NO'}")
    
    if detected_anomaly:
        print(f"[CONCLUSION] This session SHOULD be flagged as an anomaly!")
        reasons = []
        if device_error:
            reasons.append("Contains DEVICE ERROR")
        if invalid_amount:
            reasons.append("Contains INVALID AMOUNT")
        if unable_to_process:
            reasons.append("Contains UNABLE TO PROCESS")
        if hardware_fault:
            reasons.append("Contains hardware fault indicators")
        if error_codes:
            reasons.append(f"Contains error codes: {error_codes}")
        
        print(f"[REASONS]:")
        for reason in reasons:
            print(f"   - {reason}")
    else:
        print(f"[CONCLUSION] This session should NOT be flagged as an anomaly")
    
    return detected_anomaly

def main():
    """Main test function"""
    print("[START] Starting Simple EJ Anomaly Detection Test...")
    
    try:
        # Run the standard tests
        success = test_ej_session_patterns()
        
        # Run custom session test
        custom_anomaly = test_custom_session()
        
        print(f"\n[FINAL] Standard Tests: {'PASSED' if success else 'FAILED'}")
        print(f"[FINAL] Custom Session: {'ANOMALY DETECTED' if custom_anomaly else 'NO ANOMALY'}")
        
        return success
        
    except Exception as e:
        print(f"[ERROR] Test failed with error: {e}")
        return False

if __name__ == "__main__":
    main()
