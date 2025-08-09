#!/usr/bin/env python3
"""
Direct training script for the enhanced ensemble model
"""

import requests
import json
import psycopg2
import os
from datetime import datetime

# Database connection
DB_CONFIG = {
    'host': 'localhost',
    'port': '5434',
    'database': 'abmdb_dev',
    'user': 'abmuser',
    'password': 'abmpass123'
}

API_BASE = 'http://localhost:8001'

def get_sessions_from_db():
    """Get sessions from the database for training"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Get sessions with their text
        query = """
        SELECT session_id, session_text, anomaly_score, is_anomaly, 
               session_length, unique_events_count, event_frequency
        FROM ml_sessions 
        WHERE session_text IS NOT NULL 
        AND session_text != ''
        ORDER BY created_at DESC 
        LIMIT 100
        """
        
        cur.execute(query)
        rows = cur.fetchall()
        
        sessions = []
        for row in rows:
            session_data = {
                'session_id': row[0],
                'session_text': row[1] or '',
                'raw_text': row[1] or '',
                'anomaly_score': float(row[2]) if row[2] else 0.0,
                'is_anomaly': bool(row[3]) if row[3] is not None else False,
                'session_length': int(row[4]) if row[4] else 0,
                'unique_events_count': int(row[5]) if row[5] else 0,
                'event_frequency': float(row[6]) if row[6] else 0.0
            }
            sessions.append(session_data)
        
        cur.close()
        conn.close()
        
        print(f"Retrieved {len(sessions)} sessions from database")
        return sessions
        
    except Exception as e:
        print(f"Error getting sessions from database: {e}")
        return []

def train_ensemble_model(sessions):
    """Train the enhanced ensemble model"""
    try:
        training_data = {'sessions': sessions}
        
        print(f"Training enhanced ensemble with {len(sessions)} sessions...")
        
        response = requests.post(
            f"{API_BASE}/api/train_enhanced_ensemble",
            json=training_data,
            headers={'Content-Type': 'application/json'},
            timeout=300  # 5 minute timeout for training
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Enhanced ensemble training completed successfully!")
            print(f"Training result: {json.dumps(result, indent=2)}")
            return True
        else:
            print(f"❌ Training failed with status {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return False

def check_model_status():
    """Check the current model status"""
    try:
        response = requests.get(f"{API_BASE}/api/model_info")
        if response.status_code == 200:
            model_info = response.json()
            print(f"Model status: {json.dumps(model_info, indent=2)}")
            return model_info
        else:
            print(f"Failed to get model info: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error getting model info: {e}")
        return None

def main():
    print("🚀 Enhanced Ensemble Model Training Script")
    print("==========================================")
    
    # Check initial model status
    print("\n📊 Checking initial model status...")
    initial_status = check_model_status()
    
    if initial_status and initial_status.get('is_trained'):
        print("✅ Model is already trained!")
        return
    
    # Get sessions from database
    print("\n📂 Getting sessions from database...")
    sessions = get_sessions_from_db()
    
    if not sessions:
        print("❌ No sessions found for training")
        return
    
    # Train the model
    print(f"\n🎯 Training model with {len(sessions)} sessions...")
    success = train_ensemble_model(sessions)
    
    if success:
        print("\n📊 Checking final model status...")
        check_model_status()
        print("\n🎉 Training completed! The isolation forest dashboard should now show real data.")
    else:
        print("\n❌ Training failed")

if __name__ == "__main__":
    main()
