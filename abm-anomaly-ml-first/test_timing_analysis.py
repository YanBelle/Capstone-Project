#!/usr/bin/env python3
"""Test the cash dispensing timing analysis functionality"""

import sys
import os

# Add the anomaly-detector path
sys.path.append('/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/services/anomaly-detector')

def test_cash_timing_analysis():
    """Test timing analysis for cash dispensing sequence"""
    
    try:
        from ej_contextual_labeler import EJLogLabeler, EventType
        
        labeler = EJLogLabeler()
        
        # Sample EJ log with timing before events
        timed_cash_log = """
07:07:56 NOTES STACKED
07:07:58 NOTES PRESENTED
07:08:15 NOTES TAKEN
        """.strip()
        
        print("Testing cash dispensing timing analysis...")
        labels = labeler.label_log(timed_cash_log)
        
        print("Total labels: " + str(len(labels)))
        
        # Look for timing analysis in metadata
        timing_found = False
        timing_data = None
        
        for i, label in enumerate(labels):
            print(str(i+1) + ". " + label.event_type.value + " at " + 
                  (label.timestamp.strftime('%H:%M:%S') if label.timestamp else 'No time'))
            
            if 'cash_timing_analysis' in label.metadata:
                timing_found = True
                timing_data = label.metadata['cash_timing_analysis']
                sequence_role = label.metadata.get('cash_sequence_role', 'unknown')
                
                print("   --> Timing analysis found (role: " + sequence_role + ")")
                print("       Timestamps:")
                timestamps = timing_data.get('timestamps', {})
                for event, time_str in timestamps.items():
                    print("         " + event.upper() + ": " + time_str)
                
                print("       Timing intervals:")
                print("         STACKED->PRESENTED: " + str(timing_data.get('stacked_to_presented_seconds', 0)) + "s")
                print("         PRESENTED->TAKEN: " + str(timing_data.get('presented_to_taken_seconds', 0)) + "s")
                print("         TOTAL TIME: " + str(timing_data.get('total_dispensing_seconds', 0)) + "s")
                
                print("       Performance:")
                print("         Presentation: " + timing_data.get('presentation_performance', 'UNKNOWN'))
                print("         Customer Response: " + timing_data.get('customer_response', 'UNKNOWN'))
                print("         Overall Efficiency: " + timing_data.get('overall_efficiency', 'UNKNOWN'))
                print("         Sequence Health: " + timing_data.get('sequence_health', 'UNKNOWN'))
                
                insights = timing_data.get('insights', [])
                if insights:
                    print("       Insights:")
                    for insight in insights:
                        print("         - " + insight)
        
        return timing_found and timing_data is not None
        
    except Exception as e:
        print("[FAIL] Timing analysis test failed: " + str(e))
        import traceback
        traceback.print_exc()
        return False

def test_slow_dispensing_detection():
    """Test detection of slow dispensing performance"""
    
    try:
        from ej_contextual_labeler import EJLogLabeler
        
        labeler = EJLogLabeler()
        
        # Simulate slow dispensing (6 second delay)
        slow_dispensing_log = """
09:15:30 NOTES STACKED
09:15:36 NOTES PRESENTED
09:15:50 NOTES TAKEN
        """.strip()
        
        print("\nTesting slow dispensing detection...")
        labels = labeler.label_log(slow_dispensing_log)
        
        # Look for performance warnings
        slow_detected = False
        for label in labels:
            if 'cash_timing_analysis' in label.metadata:
                timing_data = label.metadata['cash_timing_analysis']
                performance = timing_data.get('presentation_performance', '')
                insights = timing_data.get('insights', [])
                
                print("Presentation performance: " + performance)
                print("Insights detected: " + str(len(insights)))
                for insight in insights:
                    print("  - " + insight)
                
                # Check if slow presentation was detected
                if performance == 'SLOW' or any('slow' in insight.lower() for insight in insights):
                    slow_detected = True
                    print("[PASS] Slow dispensing detected")
        
        return slow_detected
        
    except Exception as e:
        print("[FAIL] Slow dispensing test failed: " + str(e))
        return False

def test_fast_efficient_dispensing():
    """Test normal/fast dispensing performance"""
    
    try:
        from ej_contextual_labeler import EJLogLabeler
        
        labeler = EJLogLabeler()
        
        # Simulate fast, efficient dispensing
        fast_dispensing_log = """
10:30:15 NOTES STACKED
10:30:17 NOTES PRESENTED
10:30:22 NOTES TAKEN
        """.strip()
        
        print("\nTesting efficient dispensing detection...")
        labels = labeler.label_log(fast_dispensing_log)
        
        # Look for good performance metrics
        efficient_detected = False
        for label in labels:
            if 'cash_timing_analysis' in label.metadata:
                timing_data = label.metadata['cash_timing_analysis']
                presentation = timing_data.get('presentation_performance', '')
                efficiency = timing_data.get('overall_efficiency', '')
                health = timing_data.get('sequence_health', '')
                
                print("Presentation: " + presentation)
                print("Overall efficiency: " + efficiency)
                print("Sequence health: " + health)
                
                # Check for good performance
                if (presentation in ['FAST', 'MODERATE'] and 
                    efficiency in ['EXCELLENT', 'MODERATE'] and 
                    health == 'HEALTHY'):
                    efficient_detected = True
                    print("[PASS] Efficient dispensing detected")
        
        return efficient_detected
        
    except Exception as e:
        print("[FAIL] Efficient dispensing test failed: " + str(e))
        return False

def test_timestamp_extraction():
    """Test timestamp extraction from time-prefixed events"""
    
    try:
        from ej_contextual_labeler import EJLogLabeler
        
        labeler = EJLogLabeler()
        
        # Test individual timestamp extraction
        test_lines = [
            "07:07:56 NOTES STACKED",
            "12:34:59 NOTES PRESENTED", 
            "23:59:01 NOTES TAKEN"
        ]
        
        print("\nTesting timestamp extraction...")
        
        all_extracted = True
        for line in test_lines:
            timestamp = labeler._extract_timestamp(line)
            if timestamp:
                print("Line: '" + line + "' -> " + timestamp.strftime('%H:%M:%S'))
            else:
                print("Line: '" + line + "' -> NO TIMESTAMP EXTRACTED")
                all_extracted = False
        
        return all_extracted
        
    except Exception as e:
        print("[FAIL] Timestamp extraction test failed: " + str(e))
        return False

def main():
    """Run all timing analysis tests"""
    print("Cash Dispensing Timing Analysis Test Suite")
    print("=" * 60)
    
    tests = [
        ("Timestamp Extraction", test_timestamp_extraction),
        ("Cash Timing Analysis", test_cash_timing_analysis),
        ("Slow Dispensing Detection", test_slow_dispensing_detection),
        ("Efficient Dispensing Detection", test_fast_efficient_dispensing)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print("\n" + "=" * 20 + " " + test_name + " " + "=" * 20)
        if test_func():
            passed += 1
            print("RESULT: PASS")
        else:
            print("RESULT: FAIL")
    
    print("\n" + "=" * 60)
    print("Timing Analysis Tests: " + str(passed) + "/" + str(len(tests)) + " passed")
    
    if passed == len(tests):
        print("\n[SUCCESS] Cash Dispensing Timing Analysis Working!")
        print("[SUCCESS] Timestamps properly extracted from hh:mm:ss format")
        print("[SUCCESS] STACKED->PRESENTED->TAKEN timing calculated")
        print("[SUCCESS] Performance metrics and insights generated")
        print("[SUCCESS] Slow dispensing detection functional")
        print("[SUCCESS] Efficient operations properly classified")
        print("\nMetadata Available:")
        print("- stacked_to_presented_seconds")
        print("- presented_to_taken_seconds") 
        print("- total_dispensing_seconds")
        print("- presentation_performance (FAST/MODERATE/SLOW)")
        print("- customer_response (FAST/MODERATE/SLOW)")
        print("- overall_efficiency (EXCELLENT/MODERATE/POOR)")
        print("- sequence_health (HEALTHY/ISSUES_DETECTED)")
        print("- operational insights and warnings")
    else:
        print("\n[FAIL] Some timing analysis tests failed")

if __name__ == "__main__":
    main()
