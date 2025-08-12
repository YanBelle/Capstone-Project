#!/usr/bin/env python3
"""
Test script to verify unified analyzer imports work in container environment
"""

import sys
import os

def test_unified_analyzer_import():
    """Test importing the unified analyzer with various path configurations"""
    print("🧪 Testing Unified ML Analyzer Import in Container Environment")
    print("=" * 60)
    
    # Test different paths that might exist in container
    shared_paths = [
        '/app/shared',  # Docker volume mount
        '/app/../shared',  # Relative to container
        './shared',  # Current directory relative
        os.path.join(os.path.dirname(__file__), 'shared'),  # Script relative
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'shared'),  # Parent relative
    ]
    
    print("🔍 Checking available paths:")
    for path in shared_paths:
        exists = os.path.exists(path)
        print(f"   {path}: {'✅ EXISTS' if exists else '❌ NOT FOUND'}")
        if exists:
            files = os.listdir(path)
            print(f"      Contents: {files}")
    
    print("\n📦 Testing imports:")
    
    # Test unified analyzer import
    unified_imported = False
    for shared_path in shared_paths:
        try:
            if os.path.exists(shared_path):
                sys.path.insert(0, shared_path)
                from ml_analyzer_unified import UnifiedMLAnomalyDetector
                print(f"✅ Successfully imported UnifiedMLAnomalyDetector from {shared_path}")
                unified_imported = True
                
                # Test initialization
                analyzer = UnifiedMLAnomalyDetector(
                    model_name='bert-base-uncased',
                    service_mode='anomaly-detector'
                )
                print(f"✅ Successfully initialized unified analyzer with service_mode")
                break
        except ImportError as e:
            print(f"❌ Import failed from {shared_path}: {e}")
        except Exception as e:
            print(f"⚠️ Initialization failed from {shared_path}: {e}")
    
    if not unified_imported:
        print("❌ Could not import unified analyzer from any path")
        
        # Test fallback import
        try:
            from ml_analyzer import MLFirstAnomalyDetector
            print("✅ Successfully imported original MLFirstAnomalyDetector (fallback)")
            
            # Test initialization without service_mode
            analyzer = MLFirstAnomalyDetector(model_name='bert-base-uncased')
            print("✅ Successfully initialized original analyzer")
        except Exception as e:
            print(f"❌ Fallback import failed: {e}")
    
    print("\n" + "=" * 60)
    return unified_imported

def test_shared_directory_contents():
    """Test the contents of the shared directory"""
    print("📁 Shared Directory Analysis:")
    
    shared_dirs = ['/app/shared', './shared']
    for shared_dir in shared_dirs:
        if os.path.exists(shared_dir):
            print(f"\n📂 {shared_dir}:")
            try:
                for item in os.listdir(shared_dir):
                    item_path = os.path.join(shared_dir, item)
                    if os.path.isfile(item_path):
                        size = os.path.getsize(item_path)
                        print(f"   📄 {item} ({size} bytes)")
                    else:
                        print(f"   📁 {item}/")
            except PermissionError:
                print(f"   ❌ Permission denied")

if __name__ == "__main__":
    print("🐳 Container Environment ML Analyzer Test")
    print("=" * 60)
    
    # Show Python path
    print("🐍 Python path:")
    for path in sys.path:
        print(f"   {path}")
    
    print("\n" + "=" * 60)
    
    # Test shared directory
    test_shared_directory_contents()
    
    print("\n" + "=" * 60)
    
    # Test imports
    success = test_unified_analyzer_import()
    
    print("\n" + "=" * 60)
    print("✅ TEST PASSED" if success else "❌ TEST FAILED")
    print("=" * 60)
