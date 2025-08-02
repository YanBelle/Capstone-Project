#!/usr/bin/env python3
"""
Simple validation test for enhanced preprocessing patterns
Tests locally without needing the API service
"""

import sys
import os

# Add the anomaly-detector path to import the bertviz_analyzer
sys.path.append('/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/services/anomaly-detector')

try:
    from bertviz_analyzer import BertVisualizationAnalyzer
    BERT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Could not import BertVisualizationAnalyzer: {e}")
    BERT_AVAILABLE = False

def test_preprocessing_directly():
    """Test the enhanced preprocessing logic directly"""
    
    # The exact EJ sample that had tokenization issues
    ej_sample = """[020t*629*06/18/2025*00:46*
     *TRANSACTION START*
[020t CARD INSERTED
 00:46:27 ATR RECEIVED T=0
[020t 00:46:30 OPCODE = FI      

  PAN 0004263********1897
  ---START OF TRANSACTION---
 
[020t 00:46:42 PIN ENTERED
[020t 00:46:47 OPCODE = IB      

  PAN 0004263********1897
  ---START OF TRANSACTION---
 
*630*06/18/2025*00:46*
*7231*1*(Iw(1*3, M-02, R-10011
A/C 
DEVICE ERROR
ESC: 000
VAL: 000
REF: 000
REJECTS:000*(1
S
[020t 00:47:13 CARD TAKEN
[020t 00:47:15 TRANSACTION END
[020t*631*06/18/2025*00:47*"""

    print("🔬 Testing Enhanced Preprocessing Directly")
    print("=" * 60)
    
    if BERT_AVAILABLE:
        try:
            # Initialize the BERT analyzer (this will use our enhanced preprocessing)
            analyzer = BertVisualizationAnalyzer()
            
            # Apply preprocessing
            print("📝 Applying enhanced preprocessing...")
            processed_text = analyzer._preprocess_text(ej_sample)
            
            print("✅ Preprocessing completed!")
            print()
            print("📋 Results:")
            print("-" * 40)
            print(f"Processed Text: {processed_text}")
            print()
            
            # Validate specific improvements
            validation_checks = [
                ("ESC_000 token present", "ESC_000" in processed_text),
                ("VAL_000 token present", "VAL_000" in processed_text),
                ("REF_000 token present", "REF_000" in processed_text),
                ("REJECTS_000 token present", "REJECTS_000" in processed_text),
                ("ATR_RECEIVED_T_0 present", "ATR_RECEIVED_T_0" in processed_text),
                ("No isolated '1' tokens", " 1 " not in processed_text and not processed_text.endswith(" 1")),
                ("No isolated 'S' tokens", " S " not in processed_text and not processed_text.endswith(" S")),
                ("TRANSACTION_START present", "TRANSACTION_START" in processed_text),
                ("DEVICE_ERROR present", "DEVICE_ERROR" in processed_text),
                ("CARD_INSERTED present", "CARD_INSERTED" in processed_text),
                ("CARD_TAKEN present", "CARD_TAKEN" in processed_text),
                ("PIN_ENTERED present", "PIN_ENTERED" in processed_text),
                ("TRANSACTION_END present", "TRANSACTION_END" in processed_text),
                ("M_02 present", "M_02" in processed_text),
                ("R_10011 present", "R_10011" in processed_text),
                ("OPCODE_FI present", "OPCODE_FI" in processed_text),
                ("OPCODE_IB present", "OPCODE_IB" in processed_text),
                ("CardNumber present", "CardNumber" in processed_text),
                ("No noise patterns", "*7231*1*(Iw(1*3," not in processed_text),
                ("No timestamps", "00:46:27" not in processed_text),
            ]
            
            print("🔍 Validation Results:")
            print("-" * 40)
            passed = 0
            total = len(validation_checks)
            
            for check_name, check_result in validation_checks:
                status = "✅ PASS" if check_result else "❌ FAIL"
                print(f"  {check_name}: {status}")
                if check_result:
                    passed += 1
            
            print()
            print("📊 Summary:")
            print(f"  Validation Score: {passed}/{total} checks passed")
            
            if passed == total:
                print("🎉 Perfect! All enhanced preprocessing patterns working correctly!")
                print("   ✓ Isolated noise tokens eliminated")
                print("   ✓ ESC/VAL/REF properly combined")
                print("   ✓ ATR patterns correctly formatted")
                print("   ✓ Transaction codes cleaned up")
                print("   ✓ Compound tokens preserved")
                return True
            else:
                print(f"⚠️  {total - passed} validation checks failed")
                print("   Some preprocessing patterns need further refinement")
                return False
            
        except Exception as e:
            print(f"❌ Error during preprocessing test: {e}")
            return False
    else:
        print("❌ BERT analyzer not available - cannot test preprocessing")
        return False

if __name__ == "__main__":
    success = test_preprocessing_directly()
    if success:
        print("\n🚀 Enhanced preprocessing is working perfectly!")
        print("The BERT tokenization issues have been resolved.")
    else:
        print("\n⚠️  Some issues remain with the preprocessing patterns.")
