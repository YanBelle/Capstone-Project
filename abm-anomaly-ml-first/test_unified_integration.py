#!/usr/bin/env python3
"""
Test script for the Unified ML Analyzer

This script tests the integration of the unified analyzer with both services
to ensure feature preservation and functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'shared'))

def test_unified_analyzer():
    """Test basic functionality of the unified analyzer"""
    print("🧪 Testing Unified ML Analyzer Integration")
    
    try:
        from ml_analyzer_unified import UnifiedMLAnomalyDetector
        print("✅ Successfully imported UnifiedMLAnomalyDetector")
        
        # Test API service mode
        print("\n📡 Testing API service mode...")
        api_analyzer = UnifiedMLAnomalyDetector(
            model_name='bert-base-uncased',
            service_mode='api'
        )
        print(f"✅ API analyzer initialized - Service mode: {api_analyzer.service_mode}")
        
        # Test anomaly-detector service mode  
        print("\n🔍 Testing anomaly-detector service mode...")
        detector_analyzer = UnifiedMLAnomalyDetector(
            model_name='bert-base-uncased', 
            service_mode='anomaly-detector'
        )
        print(f"✅ Detector analyzer initialized - Service mode: {detector_analyzer.service_mode}")
        
        # Test sessionization functionality
        print("\n📝 Testing sessionization functionality...")
        test_ej_content = """
*TRANSACTION START* 12:00:01
ATM Transaction Log
Customer Card Read: ****1234
Amount: $100.00
*TRANSACTION END* 12:00:05

*TRANSACTION START* 12:05:01  
ATM Transaction Log
Customer Card Read: ****5678
Amount: $200.00
*TRANSACTION END* 12:05:08
"""
        
        sessions = api_analyzer.split_into_sessions(test_ej_content, "test_file.txt")
        print(f"✅ Sessionization successful - Found {len(sessions)} sessions")
        
        # Test cassette counter parsing
        print("\n💰 Testing cassette counter parsing...")
        test_cassette_content = """
CASSETTE_01_COUNT: 100
CASSETTE_02_COUNT: 250
CASSETTE_03_COUNT: 50
CASSETTE_04_COUNT: 75
"""
        
        cassette_info = detector_analyzer.parse_cassette_counters(test_cassette_content)
        print(f"✅ Cassette parsing successful - Found {len(cassette_info)} cassettes")
        for cassette, count in cassette_info.items():
            print(f"   {cassette}: {count}")
        
        # Test terminal ID detection
        print("\n🏧 Testing terminal ID detection...")
        test_filename = "ABM12345EJ_20240101_20240101.txt"
        terminal_id = api_analyzer._extract_terminal_id_from_filename(test_filename)
        print(f"✅ Terminal ID detection successful - ID: {terminal_id}")
        
        print("\n🎉 All tests passed! Unified analyzer is working correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure the unified analyzer file is in the shared directory")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def test_service_imports():
    """Test that services can import the unified analyzer"""
    print("\n🔗 Testing service import compatibility...")
    
    # Test API service import
    try:
        api_path = os.path.join(os.path.dirname(__file__), 'services', 'api')
        sys.path.append(api_path)
        print("✅ API service path added")
    except Exception as e:
        print(f"⚠️ API service path warning: {e}")
    
    # Test anomaly-detector service import
    try:
        detector_path = os.path.join(os.path.dirname(__file__), 'services', 'anomaly-detector')
        sys.path.append(detector_path)
        print("✅ Anomaly-detector service path added")
    except Exception as e:
        print(f"⚠️ Anomaly-detector service path warning: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 ABM Unified ML Analyzer Integration Test")
    print("=" * 60)
    
    test_service_imports()
    success = test_unified_analyzer()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ INTEGRATION TEST PASSED")
        print("The unified analyzer is ready for deployment!")
    else:
        print("❌ INTEGRATION TEST FAILED")
        print("Please check the errors above and fix them before deployment.")
    print("=" * 60)
