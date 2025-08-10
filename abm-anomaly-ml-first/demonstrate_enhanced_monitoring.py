#!/usr/bin/env python3
"""
Enhanced Monitoring Demonstration Script
Tests the new progress tracking features for EJ loading and model training
"""

import requests
import time
import json
import tempfile
import os
from datetime import datetime

API_BASE_URL = "http://localhost:8000"

def create_sample_ej_files(num_files=3):
    """Create sample EJ files for testing"""
    temp_dir = tempfile.mkdtemp()
    
    sample_ej_content = """
0001|2024-01-15 10:30:15|TXN_START|SESSION_001
0002|2024-01-15 10:30:16|CARD_INSERT|CHIP
0003|2024-01-15 10:30:17|PIN_ENTRY|****
0004|2024-01-15 10:30:18|AUTH_REQUEST|BANK_A
0005|2024-01-15 10:30:19|AUTH_RESPONSE|APPROVED
0006|2024-01-15 10:30:20|CASH_DISPENSE|$100.00
0007|2024-01-15 10:30:21|RECEIPT_PRINT|SUCCESS
0008|2024-01-15 10:30:22|CARD_EJECT|SUCCESS
0009|2024-01-15 10:30:23|TXN_END|SUCCESS
    """.strip()
    
    created_files = []
    for i in range(num_files):
        filename = f"test_session_{i+1}_{int(time.time())}.txt"
        filepath = os.path.join(temp_dir, filename)
        
        with open(filepath, 'w') as f:
            # Add some variation to each file
            content = sample_ej_content.replace("SESSION_001", f"SESSION_{i+1:03d}")
            content = content.replace("BANK_A", f"BANK_{chr(65+i)}")
            f.write(content)
        
        created_files.append(filepath)
    
    return created_files, temp_dir

def upload_files_to_api(file_paths):
    """Upload EJ files to the API"""
    print("📁 Uploading EJ files...")
    
    upload_results = []
    for file_path in file_paths:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'text/plain')}
            response = requests.post(f"{API_BASE_URL}/api/v1/upload-ej", files=files)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Uploaded {os.path.basename(file_path)}: {result.get('message', 'Success')}")
                upload_results.append(result)
            else:
                print(f"❌ Failed to upload {os.path.basename(file_path)}: {response.text}")
    
    return upload_results

def trigger_processing():
    """Trigger EJ file processing"""
    print("\n🔄 Triggering EJ file processing...")
    
    response = requests.post(f"{API_BASE_URL}/api/v1/process-input")
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Processing started: {result.get('message', 'Success')}")
        return result
    else:
        print(f"❌ Failed to trigger processing: {response.text}")
        return None

def monitor_progress(duration=30):
    """Monitor processing progress for a specified duration"""
    print(f"\n📊 Monitoring progress for {duration} seconds...")
    
    start_time = time.time()
    last_parsing_progress = 0
    last_training_progress = 0
    
    while time.time() - start_time < duration:
        try:
            response = requests.get(f"{API_BASE_URL}/api/v1/monitoring/status")
            if response.status_code == 200:
                data = response.json()
                
                # Display parsing progress
                parsing = data.get('parsing', {})
                parsing_progress = parsing.get('progress_percent', 0)
                parsing_status = parsing.get('status', 'unknown')
                current_file = parsing.get('current_file', 'N/A')
                
                if parsing_progress != last_parsing_progress or parsing_progress > 0:
                    print(f"📁 Parsing: {parsing_progress:.1f}% ({parsing_status}) - {current_file}")
                    last_parsing_progress = parsing_progress
                
                # Display training progress
                training = data.get('ml_training', {})
                training_progress = training.get('training_progress', 0)
                training_status = training.get('status', 'unknown')
                model_type = training.get('model_type', 'N/A')
                current_accuracy = training.get('current_accuracy', 0)
                
                if training_progress != last_training_progress or training_progress > 0:
                    print(f"🧠 Training: {training_progress:.1f}% ({training_status}) - {model_type} - Acc: {current_accuracy:.3f}")
                    last_training_progress = training_progress
                
                # Display system resources
                system = data.get('system', {})
                cpu = system.get('cpu_usage', 0)
                memory = system.get('memory_usage', 0)
                
                if time.time() % 10 < 1:  # Show every 10 seconds
                    print(f"💻 System: CPU {cpu:.1f}%, Memory {memory:.1f}%")
            
            time.sleep(2)
            
        except Exception as e:
            print(f"❌ Error monitoring: {e}")
            time.sleep(2)

def add_expert_labels():
    """Add some expert labels for testing supervised training"""
    print("\n🏷️ Adding expert labels...")
    
    # Get anomalies to label
    response = requests.get(f"{API_BASE_URL}/api/v1/expert/anomalies?limit=5")
    if response.status_code == 200:
        anomalies = response.json().get('anomalies', [])
        
        for i, anomaly in enumerate(anomalies[:3]):  # Label first 3
            session_id = anomaly['session_id']
            label = 'anomaly' if i % 2 == 0 else 'normal'  # Alternate labels
            
            label_data = {
                "session_id": session_id,
                "anomaly_label": label,
                "confidence": 0.8,
                "notes": f"Test label for monitoring demo"
            }
            
            response = requests.post(f"{API_BASE_URL}/api/v1/expert/label", json=label_data)
            if response.status_code == 200:
                print(f"✅ Labeled {session_id} as {label}")
            else:
                print(f"❌ Failed to label {session_id}: {response.text}")
    
    return len(anomalies)

def trigger_supervised_training():
    """Trigger supervised model training"""
    print("\n🧠 Triggering supervised model training...")
    
    response = requests.post(f"{API_BASE_URL}/api/v1/expert/train-supervised")
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Training started: {result.get('message', 'Success')}")
        return result
    else:
        print(f"❌ Failed to start training: {response.text}")
        return None

def main():
    """Main demonstration function"""
    print("🚀 Enhanced Monitoring Demonstration")
    print("=" * 50)
    
    try:
        # Step 1: Create and upload sample files
        file_paths, temp_dir = create_sample_ej_files(5)
        print(f"📝 Created {len(file_paths)} sample EJ files in {temp_dir}")
        
        upload_results = upload_files_to_api(file_paths)
        
        # Step 2: Trigger processing and monitor
        processing_result = trigger_processing()
        if processing_result:
            monitor_progress(15)  # Monitor for 15 seconds
        
        # Step 3: Add expert labels
        labeled_count = add_expert_labels()
        if labeled_count > 0:
            print(f"📊 Added labels to demonstrate training progress")
            
            # Step 4: Trigger training and monitor
            training_result = trigger_supervised_training()
            if training_result:
                monitor_progress(20)  # Monitor training for 20 seconds
        
        print("\n✅ Demonstration completed!")
        print("\n📖 What was demonstrated:")
        print("- EJ file upload and processing with progress tracking")
        print("- Real-time progress bars for file processing")
        print("- Model training progress with accuracy updates")
        print("- System resource monitoring")
        print("- Enhanced WebSocket updates")
        
        print(f"\n🌐 Visit http://localhost/dashboard/realtime to see the enhanced monitoring interface")
        
    except Exception as e:
        print(f"❌ Error in demonstration: {e}")
    
    finally:
        # Cleanup
        try:
            import shutil
            shutil.rmtree(temp_dir)
            print(f"🧹 Cleaned up temporary files from {temp_dir}")
        except:
            pass

if __name__ == "__main__":
    main()
