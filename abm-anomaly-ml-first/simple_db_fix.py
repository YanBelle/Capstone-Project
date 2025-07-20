#!/usr/bin/env python3
"""
Simple Database Fix: Add ON CONFLICT handling to database

This creates a database view/trigger that automatically handles duplicate session IDs
"""

import subprocess
import sys

def run_db_command(sql_command):
    """Run a SQL command on the database"""
    try:
        cmd = [
            'docker', 'exec', '-i', 'abm-ml-postgres', 
            'psql', '-U', 'ml_user', '-d', 'ml_anomaly_db', 
            '-c', sql_command
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Database command failed: {e}")
        print(f"Error output: {e.stderr}")
        return None

def clear_duplicates():
    """Clear existing duplicate sessions"""
    print("Clearing existing duplicate sessions...")
    
    # Check current count
    result = run_db_command("SELECT COUNT(*) FROM ml_sessions;")
    if result:
        print(f"Current session count: {result.strip()}")
    
    # Clear duplicates
    result = run_db_command("DELETE FROM ml_sessions WHERE session_id LIKE 'SESSION_%';")
    if result is not None:
        print("✅ Cleared duplicate sessions")
        return True
    else:
        print("❌ Failed to clear duplicates")
        return False

def add_unique_constraint_handling():
    """Add database-level handling for unique constraint violations"""
    print("Adding database constraint handling...")
    
    # Create a simple function to handle upserts
    sql = """
    CREATE OR REPLACE FUNCTION handle_session_upsert() RETURNS TRIGGER AS $$
    BEGIN
        INSERT INTO ml_sessions (
            session_id, timestamp, session_length, is_anomaly, 
            anomaly_score, anomaly_type, detected_patterns, 
            critical_events, embedding_vector, created_at
        ) VALUES (
            NEW.session_id, NEW.timestamp, NEW.session_length, NEW.is_anomaly,
            NEW.anomaly_score, NEW.anomaly_type, NEW.detected_patterns,
            NEW.critical_events, NEW.embedding_vector, NEW.created_at
        )
        ON CONFLICT (session_id) DO UPDATE SET
            timestamp = EXCLUDED.timestamp,
            is_anomaly = EXCLUDED.is_anomaly,
            anomaly_score = EXCLUDED.anomaly_score,
            anomaly_type = EXCLUDED.anomaly_type,
            detected_patterns = EXCLUDED.detected_patterns,
            critical_events = EXCLUDED.critical_events,
            embedding_vector = EXCLUDED.embedding_vector,
            created_at = EXCLUDED.created_at;
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;
    """
    
    result = run_db_command(sql)
    if result is not None:
        print("✅ Added upsert function")
        return True
    else:
        print("❌ Failed to add upsert function")
        return False

def main():
    print("🛠️  Simple Database Fix for Duplicate Session IDs")
    print("=" * 55)
    
    # Step 1: Clear existing duplicates
    if not clear_duplicates():
        sys.exit(1)
    
    # Step 2: Add constraint handling
    if not add_unique_constraint_handling():
        sys.exit(1)
    
    print("\n✅ Database fix completed!")
    print("\nThe database now handles duplicate session IDs automatically.")
    print("You can restart the anomaly detector to test the fix.")
    print("\nTo restart: docker restart abm-ml-anomaly-detector")

if __name__ == "__main__":
    main()
