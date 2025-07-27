#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive EJ Anomaly Testing Tool
===================================
This script allows you to test any EJ session by pasting the text and getting
immediate feedback on whether it should be flagged as an anomaly.

Usage:
    python3 interactive_test.py
"""

import re
from datetime import datetime

def analyze_ej_session(session_text):
    """Analyze an EJ session and determine if it's an anomaly"""
    if not session_text.strip():
        return False, []
    
    text_upper = session_text.upper()
    reasons = []
    
    # Pattern checks for anomalies
    device_error = 'DEVICE ERROR' in text_upper
    invalid_amount = 'INVALID AMOUNT' in text_upper  
    unable_to_process = 'UNABLE TO PROCESS' in text_upper
    hardware_fault = 'HARDWARE FAULT' in text_upper or 'SENSOR ERROR' in text_upper
    connection_timeout = 'TIMEOUT' in text_upper or 'CONNECTION' in text_upper
    
    # Error codes
    error_pattern = re.compile(r'(ESC|VAL|REF|REJECTS):\s*(\d+)', re.IGNORECASE)
    error_codes = error_pattern.findall(session_text)
    
    # Transaction failure indicators
    transaction_failed = any([
        'TRANSACTION FAILED' in text_upper,
        'FAILED TO' in text_upper,
        'ERROR' in text_upper and 'TRANSACTION' in text_upper
    ])
    
    # Check for positive transaction indicators (should NOT be anomalies)
    notes_taken = 'NOTES TAKEN' in text_upper
    receipt_printed = 'RECEIPT PRINTED' in text_upper
    successful_completion = (notes_taken or receipt_printed) and 'CARD TAKEN' in text_upper
    
    # Build list of anomaly indicators
    if device_error:
        reasons.append("Contains DEVICE ERROR")
    if invalid_amount:
        reasons.append("Contains INVALID AMOUNT")
    if unable_to_process:
        reasons.append("Contains UNABLE TO PROCESS")
    if hardware_fault:
        reasons.append("Contains hardware fault indicators")
    if connection_timeout:
        reasons.append("Contains timeout/connection issues")
    if transaction_failed:
        reasons.append("Contains transaction failure indicators")
    if error_codes:
        reasons.append(f"Contains error codes: {error_codes}")
    
    # Determine if anomaly detected
    has_error_patterns = any([device_error, invalid_amount, unable_to_process, 
                             hardware_fault, connection_timeout, transaction_failed])
    has_error_codes = len(error_codes) > 0
    
    # Override: if it's a successful transaction, don't flag as anomaly
    if successful_completion and not has_error_patterns:
        detected_anomaly = False
        if reasons:
            reasons.append("BUT: Transaction completed successfully - NOT flagged as anomaly")
    else:
        detected_anomaly = has_error_patterns or has_error_codes
    
    return detected_anomaly, reasons

def print_analysis(session_text, is_anomaly, reasons):
    """Print detailed analysis of the session"""
    print("\n" + "=" * 60)
    print("[ANALYSIS] EJ SESSION ANALYSIS RESULTS")
    print("=" * 60)
    
    # Basic stats
    lines = session_text.strip().split('\n')
    print(f"[INFO] Session Length: {len(lines)} lines")
    print(f"[INFO] Character Count: {len(session_text)} characters")
    
    # Check for common indicators
    text_upper = session_text.upper()
    has_card_insert = 'CARD INSERTED' in text_upper
    has_pin_entry = 'PIN ENTERED' in text_upper
    has_amount = 'AMOUNT ENTERED' in text_upper
    has_card_taken = 'CARD TAKEN' in text_upper
    
    print(f"\n[TRANSACTION FLOW]")
    print(f"   Card Inserted: {'YES' if has_card_insert else 'NO'}")
    print(f"   PIN Entered: {'YES' if has_pin_entry else 'NO'}")
    print(f"   Amount Entered: {'YES' if has_amount else 'NO'}")
    print(f"   Card Taken: {'YES' if has_card_taken else 'NO'}")
    
    print(f"\n[RESULT] ANOMALY DETECTED: {'YES' if is_anomaly else 'NO'}")
    
    if reasons:
        print(f"\n[REASONS] Detection Reasons:")
        for reason in reasons:
            print(f"   - {reason}")
    else:
        print(f"\n[REASONS] No anomaly indicators found - appears to be normal transaction")
    
    if is_anomaly:
        print(f"\n[RECOMMENDATION] This session SHOULD be flagged for review")
    else:
        print(f"\n[RECOMMENDATION] This session appears normal - no action needed")

def interactive_mode():
    """Interactive mode for testing EJ sessions"""
    print("=" * 60)
    print("[INTERACTIVE] EJ ANOMALY DETECTION TOOL")
    print("=" * 60)
    print("Paste your EJ session text below.")
    print("When finished, type 'END' on a new line and press Enter.")
    print("Type 'QUIT' to exit the program.")
    print("-" * 60)
    
    while True:
        print("\n[INPUT] Paste EJ session text (type 'END' when finished):")
        
        lines = []
        while True:
            try:
                line = input()
                if line.upper() == 'END':
                    break
                elif line.upper() == 'QUIT':
                    print("[EXIT] Goodbye!")
                    return
                else:
                    lines.append(line)
            except (EOFError, KeyboardInterrupt):
                print("\n[EXIT] Goodbye!")
                return
        
        session_text = '\n'.join(lines)
        
        if not session_text.strip():
            print("[ERROR] No session text provided. Please try again.")
            continue
        
        # Analyze the session
        is_anomaly, reasons = analyze_ej_session(session_text)
        
        # Print results
        print_analysis(session_text, is_anomaly, reasons)
        
        print(f"\n[CONTINUE] Test another session? (Enter to continue, 'QUIT' to exit)")
        response = input().strip().upper()
        if response == 'QUIT':
            print("[EXIT] Goodbye!")
            break

def test_with_sample():
    """Test with a sample session for demonstration"""
    print("=" * 60)
    print("[DEMO] Testing with Sample EJ Session")
    print("=" * 60)
    
    sample_session = """*TRANSACTION START*
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
    
    print("[SAMPLE] Analyzing this sample session:")
    print("-" * 40)
    print(sample_session[:200] + "..." if len(sample_session) > 200 else sample_session)
    print("-" * 40)
    
    is_anomaly, reasons = analyze_ej_session(sample_session)
    print_analysis(sample_session, is_anomaly, reasons)

def main():
    """Main function"""
    print("[START] EJ Anomaly Detection Interactive Tool")
    
    # First show the demo
    test_with_sample()
    
    print(f"\n[MENU] What would you like to do?")
    print("1. Test your own EJ session (interactive)")
    print("2. Exit")
    
    try:
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == '1':
            interactive_mode()
        elif choice == '2':
            print("[EXIT] Goodbye!")
        else:
            print("[ERROR] Invalid choice. Exiting.")
            
    except (EOFError, KeyboardInterrupt):
        print("\n[EXIT] Goodbye!")

if __name__ == "__main__":
    main()
