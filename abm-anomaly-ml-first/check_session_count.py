#!/usr/bin/env python3
"""
Simple script to check session count in database
"""
import os
import psycopg2
from datetime import datetime

def check_sessions():
    """Check the session count in the database"""
    try:
        # Database connection parameters
        db_params = {
            'host': 'localhost',
            'port': '5433',
            'database': 'ejdb',
            'user': 'user',
            'password': 'password'
        }
        
        print(f"[INFO] Connecting to database at {db_params['host']}:{db_params['port']}")
        
        # Connect to database
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        
        # Count total sessions
        cursor.execute("SELECT COUNT(*) FROM ml_sessions")
        total_sessions = cursor.fetchone()[0]
        
        # Count recent sessions (last hour)
        cursor.execute("""
            SELECT COUNT(*) FROM ml_sessions 
            WHERE created_at > NOW() - INTERVAL '1 hour'
        """)
        recent_sessions = cursor.fetchone()[0]
        
        # Get session details
        cursor.execute("""
            SELECT session_id, created_at, 
                   SUBSTRING(session_text, 1, 100) as text_preview
            FROM ml_sessions 
            ORDER BY created_at DESC 
            LIMIT 10
        """)
        recent_session_details = cursor.fetchall()
        
        print(f"\n[RESULTS] Session Count Summary:")
        print(f"  Total Sessions: {total_sessions}")
        print(f"  Recent Sessions (last hour): {recent_sessions}")
        
        print(f"\n[DETAILS] Recent Sessions:")
        for session_id, created_at, text_preview in recent_session_details:
            print(f"  ID: {session_id}")
            print(f"  Created: {created_at}")
            print(f"  Preview: {text_preview}...")
            print(f"  ---")
        
        cursor.close()
        conn.close()
        
        return total_sessions, recent_sessions
        
    except Exception as e:
        print(f"[ERROR] Failed to check sessions: {e}")
        return None, None

def check_files_processed():
    """Check files that have been processed"""
    try:
        processed_dir = "/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/data/input/processed"
        
        if os.path.exists(processed_dir):
            files = [f for f in os.listdir(processed_dir) if f.endswith('.txt')]
            print(f"\n[FILES] Processed Files ({len(files)}):")
            for f in files:
                file_path = os.path.join(processed_dir, f)
                if os.path.exists(file_path):
                    size = os.path.getsize(file_path)
                    print(f"  - {f} ({size} bytes)")
        else:
            print(f"[ERROR] Processed directory not found: {processed_dir}")
            
    except Exception as e:
        print(f"[ERROR] Failed to check processed files: {e}")

def main():
    """Main function"""
    print("=" * 60)
    print("[CHECK] Session Count Checker")
    print("=" * 60)
    
    # Check database sessions
    total, recent = check_sessions()
    
    # Check processed files
    check_files_processed()
    
    # Summary
    print(f"\n[SUMMARY]")
    if total is not None:
        print(f"  Database Status: Connected")
        print(f"  Total Sessions: {total}")
        print(f"  Recent Sessions: {recent}")
        
        if total > 0:
            print(f"  Sessionization Status: WORKING ✓")
            if total > 5:
                print(f"  Multiple Sessions: SUCCESS ✓")
            else:
                print(f"  Multiple Sessions: Limited ({total} sessions)")
        else:
            print(f"  Sessionization Status: NO SESSIONS")
    else:
        print(f"  Database Status: Connection Failed")
    
    return total

if __name__ == "__main__":
    main()
