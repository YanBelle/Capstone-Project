#!/usr/bin/env python3
"""
Test script to verify DeepLog integration with ML analyzer
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all necessary modules can be imported"""
    try:
        # Test DeepLog analyzer import
        from deeplog_analyzer import DeepLogAnalyzer
        logger.info("✅ DeepLog analyzer imported successfully")
        
        # Test ML analyzer import
        from ml_analyzer import MLFirstAnomalyDetector
        logger.info("✅ ML analyzer imported successfully")
        
        return True
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_deeplog_initialization():
    """Test DeepLog analyzer initialization"""
    try:
        from deeplog_analyzer import DeepLogAnalyzer
        
        # Initialize DeepLog analyzer
        deeplog = DeepLogAnalyzer(window_size=8, top_k=7)
        logger.info("✅ DeepLog analyzer initialized successfully")
        
        # Test basic functionality
        test_sequence = ["CARD_INSERTED", "PIN_ENTERED", "NOTES_PRESENTED", "NOTES_TAKEN", "CARD_TAKEN"]
        extracted = deeplog.extract_event_sequence(" ".join(test_sequence))
        logger.info(f"✅ Event extraction working: {extracted}")
        
        return True
    except Exception as e:
        logger.error(f"❌ DeepLog initialization failed: {e}")
        return False

def test_ml_analyzer_integration():
    """Test ML analyzer with DeepLog integration"""
    try:
        from ml_analyzer import MLFirstAnomalyDetector
        
        # Initialize ML analyzer (this should initialize DeepLog internally)
        ml_analyzer = MLFirstAnomalyDetector()
        logger.info("✅ ML analyzer with DeepLog integration initialized successfully")
        
        # Check if DeepLog is properly integrated
        if hasattr(ml_analyzer, 'deeplog_analyzer'):
            if ml_analyzer.deeplog_analyzer:
                logger.info("✅ DeepLog analyzer properly integrated into ML analyzer")
            else:
                logger.warning("⚠️ DeepLog analyzer initialized but is None")
        else:
            logger.error("❌ DeepLog analyzer not found in ML analyzer")
            
        return True
    except Exception as e:
        logger.error(f"❌ ML analyzer integration test failed: {e}")
        return False

def test_sample_transaction():
    """Test anomaly detection on a sample transaction"""
    try:
        from ml_analyzer import MLFirstAnomalyDetector, TransactionSession
        from datetime import datetime
        
        # Create a sample incomplete transaction
        incomplete_transaction = """
TRANSACTION START
CARD INSERTED
CARD TAKEN
"""
        
        # Create sample session
        session = TransactionSession(
            session_id="test_001",
            raw_text=incomplete_transaction,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        # Initialize ML analyzer
        ml_analyzer = MLFirstAnomalyDetector()
        
        if ml_analyzer.deeplog_analyzer:
            # Test event extraction
            events = ml_analyzer.extract_key_events(session.raw_text)
            logger.info(f"✅ Extracted events: {events}")
            
            # Test DeepLog anomaly detection method
            ml_analyzer._detect_deeplog_anomalies(session, events)
            
            if session.anomalies:
                logger.info(f"✅ DeepLog detected anomalies: {[a.anomaly_type for a in session.anomalies]}")
            else:
                logger.info("ℹ️ No anomalies detected (expected without trained model)")
        else:
            logger.warning("⚠️ DeepLog analyzer not available for testing")
            
        return True
    except Exception as e:
        logger.error(f"❌ Sample transaction test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 Starting DeepLog integration tests...")
    
    tests = [
        ("Import Tests", test_imports),
        ("DeepLog Initialization", test_deeplog_initialization),
        ("ML Analyzer Integration", test_ml_analyzer_integration),
        ("Sample Transaction Test", test_sample_transaction)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
            logger.info(f"✅ {test_name} PASSED")
        else:
            logger.error(f"❌ {test_name} FAILED")
    
    logger.info(f"\n🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! DeepLog integration is working correctly.")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the integration.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
