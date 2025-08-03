#!/usr/bin/env python3
"""
Simple EJ Rule-Based Processor Runner
=====================================

A lightweight runner for the EJ processor that handles dependencies gracefully.
Can run with or without pandas, using standard library CSV output.
"""

import sys
import subprocess
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import pandas
        return True, "pandas available"
    except ImportError:
        return False, "pandas not installed"

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing required dependencies...")
    requirements_file = Path(__file__).parent / "requirements_ej_processor.txt"
    
    if requirements_file.exists():
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    else:
        print(f"❌ Requirements file not found: {requirements_file}")
        return False

def create_data_directories():
    """Create necessary data directories"""
    input_dir = Path("./data/input")
    output_dir = Path("./data/processed")
    
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Data directories ready:")
    print(f"   Input: {input_dir.absolute()}")
    print(f"   Output: {output_dir.absolute()}")
    
    return input_dir, output_dir

def run_processor():
    """Run the EJ processor"""
    try:
        from ej_rule_based_processor import main
        main()
        return True
    except Exception as e:
        print(f"❌ Error running processor: {e}")
        return False

def main():
    """Main runner function"""
    print("🚀 EJ Rule-Based Processor Runner")
    print("=" * 40)
    
    # Check if data directories exist, create if needed
    input_dir, output_dir = create_data_directories()
    
    # Check for EJ files
    ej_files = list(input_dir.glob("*.txt"))
    if not ej_files:
        print(f"\n⚠️  No EJ files found in {input_dir}")
        print("   Please place your EJ .txt files in the input directory")
        print(f"   Expected location: {input_dir.absolute()}")
        return
    
    print(f"✅ Found {len(ej_files)} EJ files to process")
    
    # Check dependencies
    has_pandas, status = check_dependencies()
    if not has_pandas:
        print(f"📦 {status}")
        user_input = input("Install dependencies automatically? (y/n): ").lower().strip()
        
        if user_input == 'y':
            if not install_dependencies():
                print("❌ Failed to install dependencies. Exiting.")
                return
        else:
            print("❌ Dependencies required. Exiting.")
            return
    
    print("✅ All dependencies available")
    
    # Run the processor
    print("\n🔄 Starting EJ processing...")
    if run_processor():
        print("\n🎉 Processing completed successfully!")
        print(f"📁 Check output files in: {output_dir.absolute()}")
    else:
        print("❌ Processing failed")

if __name__ == "__main__":
    main()
