#!/usr/bin/env python3
"""
Train the enhanced ensemble model with sessions from the database
"""
import requests
import psycopg2
import json
from datetime import datetime

# Database connection parameters
DB_CONFIG = {
    'host': 'localhost',
    'port': 5433,  # Mapped port from docker-compose
    'database': 'abm_ml_db',
    'user': 'abm_user',
    'password': 'secure_ml_password123'
}

API_BASE = 'http://localhost:8000'

def fetch_sessions_from_db(limit=200):
    """Fetch sessions from the database"""
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Fetch sessions with their data
        query = """
        SELECT session_id, session_length, is_anomaly, anomaly_score, 
               anomaly_type, detected_patterns, critical_events, timestamp
        FROM ml_sessions 
        ORDER BY RANDOM()
        LIMIT %s
        """
        
        cursor.execute(query, (limit,))
        rows = cursor.fetchall()
        
        sessions = []
        for row in rows:
            # Create session text from available data
            session_text = f"Session {row[0]} with {row[1]} events"
            if row[4]:  # anomaly_type
                session_text += f" type:{row[4]}"
            if row[5]:  # detected_patterns
                patterns = json.loads(row[5]) if isinstance(row[5], str) else row[5]
                if patterns:
                    session_text += f" patterns:{','.join(patterns) if isinstance(patterns, list) else str(patterns)}"
            if row[6]:  # critical_events
                events = json.loads(row[6]) if isinstance(row[6], str) else row[6]
                if events:
                    session_text += f" events:{','.join(events) if isinstance(events, list) else str(events)}"
            
            session = {
                'session_id': row[0],
                'session_text': session_text,
                'raw_text': session_text,
                'is_anomaly': row[2],
                'anomaly_score': float(row[3]) if row[3] else 0.0,
                'session_length': row[1] or 0,
                'unique_events_count': row[1] or 0,  # Use session_length as proxy
                'event_frequency': float(row[3]) if row[3] else 1.0  # Use anomaly_score as proxy
            }
            sessions.append(session)
        
        cursor.close()
        conn.close()
        
        print(f"Fetched {len(sessions)} sessions from database")
        return sessions
        
    except Exception as e:
        print(f"Error fetching sessions: {e}")
        return []

def train_model(sessions):
    """Train the enhanced ensemble model"""
    try:
        url = f"{API_BASE}/api/train_enhanced_ensemble"
        payload = {
            "sessions": sessions
        }
        
        print(f"Training model with {len(sessions)} sessions...")
        response = requests.post(url, json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Model training completed successfully!")
            print(f"Training result: {result}")
            return True
        else:
            print(f"❌ Training failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error training model: {e}")
        return False

def check_model_status():
    """Check if the model is trained and ready"""
    try:
        url = f"{API_BASE}/api/model_info"
        response = requests.get(url)
        
        if response.status_code == 200:
            info = response.json()
            print(f"Model status: {info}")
            return info.get('model_loaded', False)
        else:
            print(f"Error checking model status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Error checking model status: {e}")
        return False

def test_isolation_forest():
    """Test the isolation forest endpoint after training"""
    try:
        url = f"{API_BASE}/api/v1/isolation-forest/analysis"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Isolation Forest analysis working!")
            print(f"Total sessions: {data.get('total_sessions', 'N/A')}")
            print(f"Model info: {data.get('model_info', 'N/A')}")
            return True
        else:
            print(f"❌ Isolation Forest test failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Error testing isolation forest: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting model training process...")
    
    # Step 1: Check initial model status
    print("\n1. Checking initial model status...")
    check_model_status()
    
    # Step 2: Fetch sessions from database
    print("\n2. Fetching sessions from database...")
    sessions = fetch_sessions_from_db(200)
    
    if not sessions:
        print("❌ No sessions found, cannot train model")
        exit(1)
    
    # Step 3: Train the model
    print("\n3. Training the enhanced ensemble model...")
    if train_model(sessions):
        print("✅ Training successful!")
        
        # Step 4: Check model status after training
        print("\n4. Checking model status after training...")
        check_model_status()
        
        # Step 5: Test isolation forest
        print("\n5. Testing isolation forest analysis...")
        test_isolation_forest()
        
    else:
        print("❌ Training failed!")
        exit(1)
    
    print("\n🎉 Model training and testing completed!")
