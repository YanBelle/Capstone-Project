#!/usr/bin/env python3
"""
Direct test of EJ cleaning functionality within the API container
"""

import sys
sys.path.append('/app')
import re
from bertviz_analyzer import BertVisualizationAnalyzer

def test_ej_cleaning_direct():
    """Test EJ cleaning directly within the container"""
    
    print("Testing EJ timestamp cleaning directly...")
    print("=" * 50)
    
    # Test text with EJ timestamp pattern
    test_text = "[020t 00:47:13 DEVICE ERROR detected REJECTS:000"
    print(f"Original text: '{test_text}'")
    print(f"Original length: {len(test_text)}")
    
    # Create analyzer and test preprocessing
    analyzer = BertVisualizationAnalyzer()
    processed_text = analyzer._preprocess_text(test_text)
    
    print(f"Processed text: '{processed_text}'")
    print(f"Processed length: {len(processed_text)}")
    
    # Check results
    if "[020t" in processed_text:
        print("❌ Timestamp pattern still present!")
    else:
        print("✅ Timestamp pattern successfully removed!")
        
    if "DEVICE ERROR" in processed_text and "REJECTS:000" in processed_text:
        print("✅ Critical content preserved!")
    else:
        print("❌ Critical content missing!")
    
    # Test the regex patterns directly
    print("\nTesting regex patterns directly:")
    test_pattern1 = r'\[020t\s+\d{2}:\d{2}:\d{2}'
    test_pattern2 = r'\[020t\s+'
    
    result1 = re.sub(test_pattern1, '', test_text)
    print(f"After pattern 1: '{result1}'")
    
    result2 = re.sub(test_pattern2, '', result1) 
    print(f"After pattern 2: '{result2}'")

if __name__ == "__main__":
    test_ej_cleaning_direct()
