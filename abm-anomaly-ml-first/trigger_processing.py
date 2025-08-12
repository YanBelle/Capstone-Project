#!/usr/bin/env python3
"""
Script to manually trigger EJ file processing.
This bypasses the scheduled processing and forces immediate processing of files.
"""

import os
import sys
import docker
import requests
import time

def trigger_processing_via_container():
    """Trigger processing by executing Python code inside the container"""
    try:
        client = docker.from_env()
        container = client.containers.get('abm-ml-anomaly-detector')
        
        # Execute processing command
        cmd = """python -c "
from main import MLFirstEJProcessor
import os
processor = MLFirstEJProcessor()
input_dir = '/app/input'
for filename in os.listdir(input_dir):
    if filename.endswith('.txt') and not filename.startswith('.'):
        file_path = os.path.join(input_dir, filename)
        print(f'Processing {filename}...')
        try:
            processor.process_ej_file(file_path)
            print(f'Successfully processed {filename}')
        except Exception as e:
            print(f'Error processing {filename}: {e}')
"
"""
        
        result = container.exec_run(cmd, workdir='/app')
        print("Container execution result:")
        print(result.output.decode())
        
        return result.exit_code == 0
        
    except Exception as e:
        print(f"Container execution failed: {e}")
        return False

def trigger_processing_via_api():
    """Try to trigger processing via API endpoints"""
    try:
        # Check if there are specific endpoints for triggering processing
        api_base = "http://localhost:8000"
        
        # Try common processing endpoints
        endpoints_to_try = [
            "/api/v1/process/scan",
            "/api/v1/anomaly/process",
            "/api/v1/ml/process",
            "/api/v1/upload",
            "/process"
        ]
        
        for endpoint in endpoints_to_try:
            try:
                response = requests.post(f"{api_base}{endpoint}", timeout=5)
                if response.status_code == 200:
                    print(f"Successfully triggered processing via {endpoint}")
                    print(response.json())
                    return True
                elif response.status_code != 404:
                    print(f"Endpoint {endpoint} returned status {response.status_code}: {response.text}")
            except Exception as e:
                print(f"Endpoint {endpoint} failed: {e}")
        
        return False
        
    except Exception as e:
        print(f"API approach failed: {e}")
        return False

def check_file_status():
    """Check what files are available for processing"""
    try:
        input_dir = "/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/data/input"
        processed_dir = f"{input_dir}/processed"
        
        print("Files in input directory:")
        if os.path.exists(input_dir):
            for file in os.listdir(input_dir):
                if file.endswith('.txt'):
                    filepath = os.path.join(input_dir, file)
                    size = os.path.getsize(filepath)
                    print(f"  - {file} ({size} bytes)")
        
        print("\nFiles in processed directory:")
        if os.path.exists(processed_dir):
            for file in os.listdir(processed_dir):
                if file.endswith('.txt'):
                    print(f"  - {file}")
        
    except Exception as e:
        print(f"Error checking file status: {e}")

def main():
    print("=== ABM Anomaly Detection - Manual Processing Trigger ===\n")
    
    # Check file status first
    check_file_status()
    
    print("\n=== Attempting to trigger processing ===")
    
    # Try container approach first
    print("\n1. Attempting direct container execution...")
    if trigger_processing_via_container():
        print("✓ Container approach succeeded")
        return
    
    # Try API approach
    print("\n2. Attempting API approach...")
    if trigger_processing_via_api():
        print("✓ API approach succeeded")
        return
    
    print("\n❌ All approaches failed. Manual intervention required.")
    print("\nSuggestions:")
    print("1. Check if the anomaly detector container is running: docker compose ps")
    print("2. Check container logs: docker compose logs anomaly-detector")
    print("3. Restart the anomaly detector: docker compose restart anomaly-detector")
    
if __name__ == "__main__":
    main()
