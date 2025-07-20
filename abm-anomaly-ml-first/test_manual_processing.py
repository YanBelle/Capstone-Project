#!/usr/bin/env python3

import os
import sys
import psycopg2
import json
from datetime import datetime

# Add the services directory to the path
sys.path.append('/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/services/anomaly-detector')

# Import the main processor
from main import MLFirstEJProcessor

def check_database_connection():
    """Check if we can connect to the database"""
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="abm_db",
            user="abm_user",
            password="abm_pass"
        )
        cursor = conn.cursor()
        
        # Check if tables exist
        cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
        tables = cursor.fetchall()
        print(f"Database tables: {[table[0] for table in tables]}")
        
        # Check ml_sessions count
        cursor.execute("SELECT COUNT(*) FROM ml_sessions;")
        sessions_count = cursor.fetchone()[0]
        print(f"ML sessions count: {sessions_count}")
        
        # Check labeled_anomalies count
        cursor.execute("SELECT COUNT(*) FROM labeled_anomalies;")
        labeled_count = cursor.fetchone()[0]
        print(f"Labeled anomalies count: {labeled_count}")
        
        # If we have labeled anomalies, show some examples
        if labeled_count > 0:
            cursor.execute("SELECT session_id, anomaly_type, label FROM labeled_anomalies LIMIT 5;")
            examples = cursor.fetchall()
            print("Sample labeled anomalies:")
            for example in examples:
                print(f"  Session: {example[0]}, Type: {example[1]}, Label: {example[2]}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"Database connection error: {e}")
        return False

def process_file_manually():
    """Manually process the EJ file"""
    try:
        # Create processor instance
        processor = MLFirstEJProcessor()
        
        # Path to the file
        file_path = "/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/data/input/ABM250EJ_20250618_20250618.txt"
        
        print(f"Processing file: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return
        
        # Process the file
        processor.process_ej_file(file_path)
        print("File processing completed")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("=== Manual Processing Test ===")
    print(f"Current time: {datetime.now()}")
    print()
    
    # Check database connection
    print("1. Checking database connection...")
    if check_database_connection():
        print("Database connection successful!")
    else:
        print("Database connection failed!")
    print()
    
    # Process file manually
    print("2. Processing file manually...")
    process_file_manually()
    print()
    
    # Check database again
    print("3. Checking database after processing...")
    check_database_connection()

if __name__ == "__main__":
    main()
