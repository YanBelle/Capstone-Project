#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple test script for cassette counter parsing functionality (Python 2.7 compatible).
"""

import re
from datetime import datetime

def test_cassette_parsing():
    """Test the cassette counter parsing functionality"""
    
    print("Testing Cassette Counter Parsing Functionality")
    print("=" * 60)
    
    # Sample EJ session text with cassette information
    sample_ej_text = """
*416*12/25/2024*14:30*
*TRANSACTION START*
12/25/2024 14:30:25 CARD INSERTED
12/25/2024 14:30:28 PIN ENTERED
12/25/2024 14:30:30 WITHDRAWAL REQUEST $200
12/25/2024 14:30:32 AUTHORIZATION APPROVED
12/25/2024 14:30:35 NOTES PRESENTED
12/25/2024 14:30:40 NOTES TAKEN
MACHINE 416
DATE TIME 2025/01/15 14:30:25
DENOMINATION    20    50   100    20
DISPENSED        2     1     0     3
REJECTED         0     0     0     0
REMAINING      498   799   300   597
12/25/2024 14:30:45 CARD TAKEN
*TRANSACTION END*
"""
    
    def test_parse_cassette_data(text):
        """Test version of the cassette parsing logic"""
        
        # Only parse cassette data for sessions with "NOTES PRESENTED" (successful withdrawals)
        if "NOTES PRESENTED" not in text.upper():
            print("No 'NOTES PRESENTED' found - skipping cassette parsing")
            return None
        
        try:
            # Extract machine number
            machine_match = re.search(r"MACHINE\s+(\d+)", text, re.IGNORECASE)
            
            # Extract date/time
            datetime_match = re.search(r"DATE\s+TIME\s+(\d{4}/\d{2}/\d{2})\s+(\d{2}:\d{2}:\d{2})", text, re.IGNORECASE)
            
            # Extract cassette information - more precise regex
            denom_match = re.search(r"DENOMINATION\s+((?:\d+\s*){4})", text, re.IGNORECASE)
            dispensed_match = re.search(r"DISPENSED\s+((?:\d+\s*){4})", text, re.IGNORECASE)
            rejected_match = re.search(r"REJECTED\s+((?:\d+\s*){4})", text, re.IGNORECASE)
            remaining_match = re.search(r"REMAINING\s+((?:\d+\s*){4})", text, re.IGNORECASE)
            
            print("Machine match: " + (machine_match.group(1) if machine_match else 'None'))
            print("DateTime match: " + str(datetime_match.groups() if datetime_match else 'None'))
            print("Denomination match: " + (denom_match.group(1) if denom_match else 'None'))
            print("Dispensed match: " + (dispensed_match.group(1) if dispensed_match else 'None'))
            print("Rejected match: " + (rejected_match.group(1) if rejected_match else 'None'))
            print("Remaining match: " + (remaining_match.group(1) if remaining_match else 'None'))
            
            # Verify all required data is present
            if not all([denom_match, dispensed_match, rejected_match, remaining_match]):
                print("Missing required cassette data")
                return None
            
            # Parse the numeric data
            denominations = [int(x) for x in denom_match.group(1).split()]
            dispensed = [int(x) for x in dispensed_match.group(1).split()]
            rejected = [int(x) for x in rejected_match.group(1).split()]
            remaining = [int(x) for x in remaining_match.group(1).split()]
            
            print("Parsed denominations: " + str(denominations))
            print("Parsed dispensed: " + str(dispensed))
            print("Parsed rejected: " + str(rejected))
            print("Parsed remaining: " + str(remaining))
            
            # Verify we have data for exactly 4 cassettes
            lengths = [len(lst) for lst in [denominations, dispensed, rejected, remaining]]
            if not all(length == 4 for length in lengths):
                print("Incorrect cassette count. Expected 4 cassettes, got lengths: " + str(lengths))
                return None
            
            # Extract machine and datetime
            machine = machine_match.group(1) if machine_match else "UNKNOWN"
            
            if datetime_match:
                dt_str = datetime_match.group(1) + " " + datetime_match.group(2)
                transaction_datetime = datetime.strptime(dt_str, "%Y/%m/%d %H:%M:%S")
            else:
                transaction_datetime = datetime.now()
            
            # Calculate total amounts
            total_dispensed = sum(dispensed[i] * denominations[i] for i in range(4))
            total_rejected = sum(rejected[i] * denominations[i] for i in range(4))
            
            print("Total dispensed: $" + str(total_dispensed))
            print("Total rejected: $" + str(total_rejected))
            
            cassette_data = {
                "session_id": "test_session_123",
                "terminal_id": machine,  # Use terminal_id (same as machine number)
                "transaction_datetime": transaction_datetime,
                "total_dispensed_amount": total_dispensed,
                "total_rejected_amount": total_rejected,
                "withdrawal_successful": total_dispensed > 0,
                "cassette_1_remaining": remaining[0],
                "cassette_2_remaining": remaining[1],
                "cassette_3_remaining": remaining[2],
                "cassette_4_remaining": remaining[3],
                "cassette_1_denomination": denominations[0],
                "cassette_2_denomination": denominations[1],
                "cassette_3_denomination": denominations[2],
                "cassette_4_denomination": denominations[3]
            }
            
            return cassette_data
            
        except Exception as e:
            print("Error parsing cassette counters: " + str(e))
            return None
    
    # Test with sample data
    print("\nTesting with sample EJ session data:")
    result = test_parse_cassette_data(sample_ej_text)
    
    if result:
        print("\nCassette parsing successful!")
        print("Cassette Data Summary:")
        print("   Terminal ID: " + result['terminal_id'])
        print("   Transaction Time: " + str(result['transaction_datetime']))
        print("   Total Dispensed: $" + str(result['total_dispensed_amount']))
        print("   Withdrawal Successful: " + str(result['withdrawal_successful']))
        print("\nCassette Status After Transaction:")
        for i in range(1, 5):
            denom = result['cassette_' + str(i) + '_denomination']
            remaining = result['cassette_' + str(i) + '_remaining']
            value = remaining * denom
            print("   Cassette " + str(i) + ": " + str(remaining) + " x $" + str(denom) + " = $" + str(value))
        
        # Calculate total cash remaining
        total_cash = sum(result['cassette_' + str(i) + '_remaining'] * result['cassette_' + str(i) + '_denomination'] for i in range(1, 5))
        print("\nTotal Cash Remaining: $" + str(total_cash))
        
    else:
        print("\nCassette parsing failed!")
    
    print("\n" + "="*60)
    print("Cassette Counter Parsing Test Complete")

if __name__ == "__main__":
    test_cassette_parsing()
