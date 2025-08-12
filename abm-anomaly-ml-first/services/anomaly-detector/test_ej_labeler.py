#!/usr/bin/env python3
"""
Test script to verify EJLogLabeler functionality
"""

import sys
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_ej_labeler():
    """Test EJLogLabeler import and method availability"""
    
    logger.info(f"Python version: {sys.version}")
    
    try:
        # Test import
        from ej_contextual_labeler import EJLogLabeler
        logger.info("Successfully imported EJLogLabeler")
        
        # Test instantiation
        labeler = EJLogLabeler()
        logger.info("Successfully instantiated EJLogLabeler")
        
        # Test method availability
        if hasattr(labeler, 'process_transaction_session'):
            logger.info("process_transaction_session method is available")
        else:
            logger.error("process_transaction_session method is NOT available")
            logger.info(f"Available methods: {[method for method in dir(labeler) if not method.startswith('_')]}")
            return False
        
        # Test simple method call
        sample_text = "TXN Start\nCard Insert\nTXN End"
        result = labeler.process_transaction_session(sample_text)
        logger.info(f"Method call successful. Result type: {type(result)}")
        
        if isinstance(result, dict):
            logger.info(f"Result keys: {list(result.keys())}")
        
        return True
        
    except ImportError as e:
        logger.error(f"Import error: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        return False

if __name__ == "__main__":
    success = test_ej_labeler()
    if success:
        logger.info("EJLogLabeler test PASSED")
        sys.exit(0)
    else:
        logger.error("EJLogLabeler test FAILED")
        sys.exit(1)
