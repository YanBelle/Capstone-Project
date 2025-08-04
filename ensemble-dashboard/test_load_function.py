#!/usr/bin/env python3
"""
Direct test of the load_ej_sessions function
"""

import sys
import os
import asyncio

# Add the backend app to the path
sys.path.append('/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/ensemble-dashboard/backend')

async def test_load_function():
    try:
        from app.main import load_ej_sessions
        
        print("Testing load_ej_sessions function directly...")
        print("=" * 50)
        
        # Test the function with no file/text (should load from processed data)
        result = await load_ej_sessions(
            file=None,
            text=None,
            include_errors=False,
            limit=10
        )
        
        print("Result:")
        print(f"  Success: {result.get('success', False)}")
        print(f"  Message: {result.get('message', 'No message')}")
        print(f"  Count: {result.get('count', 0)}")
        print(f"  Data Source: {result.get('data_source', 'Unknown')}")
        
        if 'suggestions' in result:
            print("  Suggestions:")
            for suggestion in result['suggestions']:
                print(f"    - {suggestion}")
        
        return result.get('success', False)
        
    except Exception as e:
        print(f"Error testing function: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_load_function())
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
