#!/usr/bin/env python3
"""
Test the enhanced CIM status block parsing functionality
"""

import sys
import os
sys.path.append('services/anomaly-detector')

from ej_contextual_labeler import EJLogLabeler

# Test CIM status block examples from the real EJ deposit log
test_cim_status_lines = [
    # Initial status: 2 notes in escrow, no validation yet
    "A/C OPERATION OK ESC: 2 VAL: 0 REF: 0 REJECTS: 1 JMD$5000: 2",
    
    # Final status: 2 notes validated, deposit complete
    "A/C OPERATION OK ESC: 0 VAL: 2 REF: 0 REJECTS: 1 JMD$5000: 2",
    
    # High rejection scenario
    "A/C OPERATION OK ESC: 1 VAL: 3 REF: 2 REJECTS: 15 JMD$1000: 2 JMD$5000: 2",
    
    # Large denomination mixed deposit
    "A/C OPERATION OK ESC: 0 VAL: 5 REF: 1 REJECTS: 3 JMD$1000: 1 JMD$2000: 2 JMD$5000: 2 JMD$10000: 1",
]

def test_cim_status_parsing():
    """Test CIM status block parsing functionality"""
    labeler = EJLogLabeler()
    
    print("🧪 Testing CIM Status Block Parsing\n")
    
    for i, test_line in enumerate(test_cim_status_lines, 1):
        print(f"Test {i}: {test_line}")
        
        # Extract CIM status data
        cim_status = labeler._extract_cim_status_block(test_line)
        
        if cim_status:
            print("✅ CIM Status Data Extracted:")
            
            # Display parsed data
            print(f"   📊 Escrow Count: {cim_status.get('escrow_count', 'N/A')}")
            print(f"   ✅ Validated Count: {cim_status.get('validated_count', 'N/A')}")
            print(f"   ❌ Refused Count: {cim_status.get('refused_count', 'N/A')}")
            print(f"   🔄 Total Rejects: {cim_status.get('total_rejects', 'N/A')}")
            print(f"   💰 Currency: {cim_status.get('currency', 'N/A')}")
            
            if 'total_deposit_value' in cim_status:
                print(f"   💵 Total Value: {cim_status['currency']}{cim_status['total_deposit_value']:,}")
            
            if 'denominations' in cim_status:
                print(f"   🏦 Denominations: {cim_status['denominations']}")
            
            if 'validation_rate' in cim_status:
                print(f"   📈 Validation Rate: {cim_status['validation_rate']:.1%}")
            
            if 'rejection_rate' in cim_status:
                print(f"   ⚠️ Rejection Rate: {cim_status['rejection_rate']:.1%}")
            
            if 'deposit_status' in cim_status:
                print(f"   🎯 Deposit Status: {cim_status['deposit_status']}")
            
            if 'rejection_severity' in cim_status:
                print(f"   🚨 Rejection Severity: {cim_status['rejection_severity']}")
            
            # Test deposit classification
            classification = labeler._classify_cim_deposit(cim_status)
            print(f"   🏷️ Deposit Classification: {classification}")
            
        else:
            print("❌ Failed to extract CIM status data")
        
        print("-" * 60)
    
    print("✅ CIM status block parsing tests completed!")

def test_full_labeling():
    """Test complete labeling with CIM status blocks"""
    labeler = EJLogLabeler()
    
    print("\n🔬 Testing Full EJ Labeling with CIM Status\n")
    
    # Simulate a complete deposit transaction log
    sample_log = """
07:45:12 CIM-DEPOSIT ACTIVATED
07:45:15 CIM-SHUTTER OPENED
07:45:18 CIM-ITEMS INSERTED
07:45:20 A/C OPERATION OK ESC: 2 VAL: 0 REF: 0 REJECTS: 1 JMD$5000: 2
07:45:25 CIM-INPUT REFUSED,REASON-INVALID MEDIA
07:45:30 A/C OPERATION OK ESC: 1 VAL: 1 REF: 1 REJECTS: 2 JMD$5000: 1
07:45:35 CIM-ITEMS PRESENTED
07:45:38 CIM-ITEMS TAKEN
07:45:40 A/C OPERATION OK ESC: 0 VAL: 2 REF: 0 REJECTS: 2 JMD$5000: 2
07:45:45 CIM-DEPOSIT COMPLETED
"""
    
    labels = labeler.label_log(sample_log)
    
    print(f"📝 Generated {len(labels)} labels from sample deposit log:")
    
    for label in labels:
        print(f"\n🏷️ Line {label.line_number}: {label.event_type.value}")
        print(f"   ⏰ Phase: {label.phase.value}")
        print(f"   📊 Severity: {label.severity.value}")
        
        if label.cim_status:
            print(f"   🏦 CIM Status: ESC:{label.cim_status.get('escrow_count', 0)} "
                  f"VAL:{label.cim_status.get('validated_count', 0)} "
                  f"REJ:{label.cim_status.get('total_rejects', 0)}")
            
            if 'total_deposit_value' in label.cim_status:
                print(f"   💰 Deposit Value: {label.cim_status['currency']}{label.cim_status['total_deposit_value']:,}")
        
        if label.rejected_reason:
            print(f"   ❌ Rejection Reason: {label.rejected_reason}")
        
        if label.metadata and 'contextual_anomalies' in label.metadata:
            print(f"   🚨 Anomalies: {len(label.metadata['contextual_anomalies'])}")
            for anomaly in label.metadata['contextual_anomalies']:
                print(f"      - {anomaly}")
        
        if label.metadata and 'cim_deposit_classification' in label.metadata:
            print(f"   🏷️ Classification: {label.metadata['cim_deposit_classification']}")

if __name__ == "__main__":
    try:
        test_cim_status_parsing()
        test_full_labeling()
        print("\n🎉 All CIM status block tests passed!")
        
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
