#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for cassette counter parsing and storage functionality.
This script validates the cassette counter extraction from EJ sessions for cash forecasting.
"""

import sys
import os
import re
from datetime import datetime
from typing import Dict, Any, Optional

# Add the services directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'abm-anomaly-ml-first', 'services', 'anomaly-detector'))

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
    
    # Test the regex patterns used in the parsing function
    def test_parse_cassette_data(text: str) -> Optional[Dict[str, Any]]:
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
            
            # Extract cassette information
            denom_match = re.search(r"DENOMINATION\s+([\d\s]+)", text, re.IGNORECASE)
            dispensed_match = re.search(r"DISPENSED\s+([\d\s]+)", text, re.IGNORECASE)
            rejected_match = re.search(r"REJECTED\s+([\d\s]+)", text, re.IGNORECASE)
            remaining_match = re.search(r"REMAINING\s+([\d\s]+)", text, re.IGNORECASE)
            
            print(f"Machine match: {machine_match.group(1) if machine_match else 'None'}")
            print(f"DateTime match: {datetime_match.groups() if datetime_match else 'None'}")
            print(f"Denomination match: {denom_match.group(1) if denom_match else 'None'}")
            print(f"Dispensed match: {dispensed_match.group(1) if dispensed_match else 'None'}")
            print(f"Rejected match: {rejected_match.group(1) if rejected_match else 'None'}")
            print(f"Remaining match: {remaining_match.group(1) if remaining_match else 'None'}")
            
            # Verify all required data is present
            if not all([denom_match, dispensed_match, rejected_match, remaining_match]):
                print("Missing required cassette data")
                return None
            
            # Parse the numeric data
            denominations = [int(x) for x in denom_match.group(1).split()]
            dispensed = [int(x) for x in dispensed_match.group(1).split()]
            rejected = [int(x) for x in rejected_match.group(1).split()]
            remaining = [int(x) for x in remaining_match.group(1).split()]
            
            print(f"Parsed denominations: {denominations}")
            print(f"Parsed dispensed: {dispensed}")
            print(f"Parsed rejected: {rejected}")
            print(f"Parsed remaining: {remaining}")
            
            # Verify we have data for exactly 4 cassettes
            if not all(len(lst) == 4 for lst in [denominations, dispensed, rejected, remaining]):
                print(f"Incorrect cassette count. Expected 4 cassettes, got lengths: {[len(lst) for lst in [denominations, dispensed, rejected, remaining]]}")
                return None
            
            # Extract machine and datetime
            machine = machine_match.group(1) if machine_match else "UNKNOWN"
            
            if datetime_match:
                dt_str = f"{datetime_match.group(1)} {datetime_match.group(2)}"
                transaction_datetime = datetime.strptime(dt_str, "%Y/%m/%d %H:%M:%S")
            else:
                transaction_datetime = datetime.now()
            
            # Calculate total amounts
            total_dispensed = sum(dispensed[i] * denominations[i] for i in range(4))
            total_rejected = sum(rejected[i] * denominations[i] for i in range(4))
            
            print(f"Total dispensed: ${total_dispensed}")
            print(f"Total rejected: ${total_rejected}")
            
            # Extract raw cassette section for debugging
            cassette_section_match = re.search(
                r"(DENOMINATION.*?REMAINING\s+[\d\s]+)", 
                text, 
                re.IGNORECASE | re.DOTALL
            )
            raw_cassette_data = cassette_section_match.group(1) if cassette_section_match else ""
            
            cassette_data = {
                "session_id": "test_session_123",
                "terminal_id": "416",
                "machine_number": machine,
                "transaction_datetime": transaction_datetime,
                
                # Remaining counts after withdrawal
                "cassette_1_remaining": remaining[0],
                "cassette_2_remaining": remaining[1],
                "cassette_3_remaining": remaining[2],
                "cassette_4_remaining": remaining[3],
                
                # Denominations
                "cassette_1_denomination": denominations[0],
                "cassette_2_denomination": denominations[1],
                "cassette_3_denomination": denominations[2],
                "cassette_4_denomination": denominations[3],
                
                # Dispensed amounts for this transaction
                "cassette_1_dispensed": dispensed[0],
                "cassette_2_dispensed": dispensed[1],
                "cassette_3_dispensed": dispensed[2],
                "cassette_4_dispensed": dispensed[3],
                
                # Rejected amounts for this transaction
                "cassette_1_rejected": rejected[0],
                "cassette_2_rejected": rejected[1],
                "cassette_3_rejected": rejected[2],
                "cassette_4_rejected": rejected[3],
                
                # Totals
                "total_dispensed_amount": total_dispensed,
                "total_rejected_amount": total_rejected,
                "withdrawal_successful": total_dispensed > 0,
                
                # Metadata
                "raw_cassette_data": raw_cassette_data
            }
            
            return cassette_data
            
        except Exception as e:
            print(f"Error parsing cassette counters: {str(e)}")
            return None
    
    # Test with sample data
    print("\nTesting with sample EJ session data:")
    result = test_parse_cassette_data(sample_ej_text)
    
    if result:
        print("\nCassette parsing successful!")
        print(f"Cassette Data Summary:")
        print(f"   Terminal ID: {result['terminal_id']}")
        print(f"   Machine: {result['machine_number']}")
        print(f"   Transaction Time: {result['transaction_datetime']}")
        print(f"   Total Dispensed: ${result['total_dispensed_amount']}")
        print(f"   Withdrawal Successful: {result['withdrawal_successful']}")
        print(f"\nCassette Status After Transaction:")
        for i in range(1, 5):
            denom = result[f'cassette_{i}_denomination']
            remaining = result[f'cassette_{i}_remaining']
            dispensed = result[f'cassette_{i}_dispensed']
            value = remaining * denom
            print(f"   Cassette {i}: {remaining} x ${denom} = ${value} (dispensed: {dispensed})")
        
        # Calculate total cash remaining
        total_cash = sum(result[f'cassette_{i}_remaining'] * result[f'cassette_{i}_denomination'] for i in range(1, 5))
        print(f"\nTotal Cash Remaining: ${total_cash}")
        
    else:
        print("\nCassette parsing failed!")
    
    # Test with non-withdrawal session (should be skipped)
    print("\n" + "="*60)
    print("Testing with non-withdrawal session (should be skipped):")
    
    non_withdrawal_text = """
*416*12/25/2024*14:30*
*TRANSACTION START*
12/25/2024 14:30:25 CARD INSERTED
12/25/2024 14:30:28 PIN ENTERED
12/25/2024 14:30:30 BALANCE INQUIRY
12/25/2024 14:30:32 AUTHORIZATION APPROVED
12/25/2024 14:30:35 BALANCE: $1,250.00
12/25/2024 14:30:40 RECEIPT PRINTED
12/25/2024 14:30:45 CARD TAKEN
*TRANSACTION END*
"""
    
    result2 = test_parse_cassette_data(non_withdrawal_text)
    if result2 is None:
        print("Correctly skipped non-withdrawal session")
    else:
        print("Should have skipped non-withdrawal session")
    
    print("\n" + "="*60)
    print("Cassette Counter Parsing Test Complete")

if __name__ == "__main__":
    test_cassette_parsing()
