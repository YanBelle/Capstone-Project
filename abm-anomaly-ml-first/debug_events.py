#!/usr/bin/env python3
"""
Test event extraction for the problematic sessions
"""
import re

def extract_key_events(session_text: str) -> list:
    """Extract key events from session text for expert analysis"""
    events = []
    
    event_patterns = {
        'CARD_INSERTED': r'CARD INSERTED',
        'PIN_ENTERED': r'PIN ENTERED', 
        'NOTES_PRESENTED': r'NOTES PRESENTED',
        'NOTES_TAKEN': r'NOTES TAKEN',
        'NOTES_STACKED': r'NOTES STACKED',
        'CARD_TAKEN': r'CARD TAKEN',
        'UNABLE_TO_DISPENSE': r'UNABLE TO DISPENSE',
        'DEVICE_ERROR': r'DEVICE ERROR',
        'TIMEOUT': r'TIMEOUT',
        'NOTES_RETRACTED': r'NOTES RETRACTED',
        'RECEIPT_PRINTED': r'RECEIPT PRINTED',
        'BALANCE_INQUIRY': r'BALANCE INQUIRY',
        'SUPERVISOR_MODE': r'SUPERVISOR MODE',
        'POWER_RESET': r'POWER-UP/RESET'
    }
    
    for event_name, pattern in event_patterns.items():
        if re.search(pattern, session_text, re.IGNORECASE):
            events.append(event_name)
    
    return events

def is_successful_inquiry_old(session_text: str, events: list) -> bool:
    """OLD logic - Check if this is a successful inquiry transaction"""
    return (("CARD_INSERTED" in events or "BALANCE_INQUIRY" in events) and
            "CARD_TAKEN" in events and
            "UNABLE_TO_DISPENSE" not in events and
            "DEVICE_ERROR" not in events)

def is_successful_inquiry_new(session_text: str, events: list) -> bool:
    """NEW logic - Check if this is a successful inquiry transaction"""
    # A successful inquiry should have:
    # 1. Card inserted and taken
    # 2. Some form of authentication or transaction activity
    # 3. No errors
    
    basic_card_flow = ("CARD_INSERTED" in events and "CARD_TAKEN" in events)
    no_errors = ("UNABLE_TO_DISPENSE" not in events and "DEVICE_ERROR" not in events)
    
    # Must have some indication of actual transaction processing
    has_transaction_activity = (
        "PIN_ENTERED" in events or
        "BALANCE_INQUIRY" in events or 
        "RECEIPT_PRINTED" in events or
        re.search(r'AUTHORIZATION', session_text, re.IGNORECASE) or
        re.search(r'ACCOUNT', session_text, re.IGNORECASE) or
        re.search(r'BALANCE.*\d+', session_text, re.IGNORECASE)
    )
    
    return basic_card_flow and no_errors and has_transaction_activity

def check_incomplete_transaction_logic(events, text):
    """Test the incomplete transaction detection logic"""
    # Pattern 1: Card inserted and immediately taken without PIN (suspicious)
    pattern1 = ("CARD_INSERTED" in events and "CARD_TAKEN" in events and 
               "PIN_ENTERED" not in events and
               not re.search(r'AUTHORIZATION', text, re.IGNORECASE) and
               not re.search(r'BALANCE.*\d+', text, re.IGNORECASE))
    
    return pattern1

# Test the three problematic sessions
session_files = ['282', '305', '357']

for session_id in session_files:
    try:
        # Find the actual session file
        import glob
        pattern = f'data/sessions/AB/ABM250_20250618_SESSION_{session_id}_*.txt'
        files = glob.glob(pattern)
        if not files:
            print(f"No file found for SESSION_{session_id}")
            continue
        
        with open(files[0], 'r') as f:
            content = f.read()
        
        events = extract_key_events(content)
        would_detect = check_incomplete_transaction_logic(events, content)
        old_inquiry_logic = is_successful_inquiry_old(content, events)
        new_inquiry_logic = is_successful_inquiry_new(content, events)
        
        print(f"\n=== SESSION_{session_id} ===")
        print(f"Extracted events: {events}")
        print(f"Would detect as incomplete: {would_detect}")
        print(f"OLD: Classified as successful inquiry: {old_inquiry_logic}")
        print(f"NEW: Classified as successful inquiry: {new_inquiry_logic}")
        print(f"Has CARD_INSERTED: {'CARD_INSERTED' in events}")
        print(f"Has CARD_TAKEN: {'CARD_TAKEN' in events}")
        print(f"Has PIN_ENTERED: {'PIN_ENTERED' in events}")
        
        # Show first 300 characters for debugging
        print(f"Content sample: {content[:300]}...")
        
    except Exception as e:
        print(f"Error processing SESSION_{session_id}: {e}")
