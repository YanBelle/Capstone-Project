#!/usr/bin/env python3

import subprocess
import sys
import time
import os

def run_backend():
    """Run the backend directly with Python"""
    print("Installing minimal requirements...")
    subprocess.run([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn", "python-multipart"], check=True)
    
    print("Starting FastAPI server...")
    
    # Change to backend directory
    backend_dir = os.path.join(os.getcwd(), "backend")
    os.chdir(backend_dir)
    
    # Run uvicorn directly
    subprocess.run([
        sys.executable, "-m", "uvicorn", 
        "app.main_working:app", 
        "--host", "0.0.0.0", 
        "--port", "8000", 
        "--reload"
    ])

if __name__ == "__main__":
    try:
        run_backend()
    except KeyboardInterrupt:
        print("\nStopping server...")
    except Exception as e:
        print(f"Error: {e}")
