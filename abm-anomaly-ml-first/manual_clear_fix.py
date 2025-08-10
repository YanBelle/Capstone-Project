#!/usr/bin/env python3
"""
Manual foreign key constraint fix for clear data operation
"""

import psycopg2
import time

def fix_clear_data_constraints():
    """Fix the foreign key constraint issue by manually clearing data in correct order"""
    
    try:
        # Connect to database
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="abmdb", 
            user="abmuser",
            password="abmpass"
        )
        
        cursor = conn.cursor()
        
        print("🔧 Manual Clear Data with Foreign Key Handling")
        print("=" * 50)
        
        # Method 1: Clear in dependency order
        print("📋 Method 1: Clearing in dependency order...")
        
        tables_to_clear = [
            "ml_anomalies",
            "expert_feedback", 
            "labeled_anomalies",
            "anomaly_detections",
            "ml_summaries",
            "ml_sessions"
        ]
        
        cleared_tables = []
        
        for table in tables_to_clear:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count_before = cursor.fetchone()[0]
                
                cursor.execute(f"DELETE FROM {table}")
                rows_affected = cursor.rowcount
                
                cleared_tables.append(f"{table} ({rows_affected} rows)")
                print(f"   ✅ Cleared {table}: {rows_affected} rows (was {count_before})")
                
            except Exception as table_error:
                print(f"   ❌ Could not clear {table}: {table_error}")
        
        # Commit the changes
        conn.commit()
        print(f"\n✅ Successfully cleared: {', '.join(cleared_tables)}")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Database operation failed: {e}")
        return False

if __name__ == "__main__":
    success = fix_clear_data_constraints()
    if success:
        print("\n🎉 Clear data fix completed successfully!")
    else:
        print("\n💥 Clear data fix failed!")
