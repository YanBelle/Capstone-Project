#!/usr/bin/env python3
"""
Populate the database with sample ML sessions for testing
"""
import psycopg2
import json
import random
from datetime import datetime, timedelta

# Database connection parameters
DB_CONFIG = {
    'host': 'localhost',
    'port': 5433,
    'database': 'abm_ml_db',
    'user': 'abm_user',
    'password': 'secure_ml_password123'
}

def create_sample_session(session_id):
    """Create a sample session with realistic ABM data"""
    # Sample session types
    session_types = ['deposit', 'withdrawal', 'transfer', 'balance_inquiry', 'card_transaction']
    anomaly_types = ['unusual_amount', 'frequency_spike', 'location_anomaly', 'time_anomaly', 'pattern_deviation']
    
    session_type = random.choice(session_types)
    is_anomaly = random.random() < 0.15  # 15% anomaly rate
    
    # Session characteristics
    base_length = 10 if session_type == 'balance_inquiry' else random.randint(15, 45)
    session_length = base_length + random.randint(-5, 15) if is_anomaly else base_length
    
    # Anomaly score
    if is_anomaly:
        anomaly_score = random.uniform(0.7, 0.95)
        anomaly_type = random.choice(anomaly_types)
    else:
        anomaly_score = random.uniform(0.1, 0.6)
        anomaly_type = None
    
    # Sample patterns and events
    patterns = []
    if session_type == 'deposit':
        patterns = ['amount_verification', 'account_check', 'receipt_print']
    elif session_type == 'withdrawal':
        patterns = ['pin_entry', 'amount_select', 'cash_dispense'] 
    elif session_type == 'transfer':
        patterns = ['account_verify', 'amount_input', 'confirmation']
    
    if is_anomaly:
        patterns.append('unusual_sequence')
        
    critical_events = []
    if is_anomaly:
        critical_events = ['security_flag', 'multiple_retries'] if random.random() < 0.5 else ['timeout_warning']
    
    return {
        'session_id': f'sess_{session_id:06d}',
        'session_length': session_length,
        'is_anomaly': is_anomaly,
        'anomaly_score': round(anomaly_score, 3),
        'anomaly_type': anomaly_type,
        'detected_patterns': json.dumps(patterns),
        'critical_events': json.dumps(critical_events),
        'timestamp': datetime.now() - timedelta(hours=random.randint(1, 168))  # Last week
    }

def populate_database(num_sessions=250):
    """Populate database with sample sessions"""
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        print(f"✓ Connected to database")
        
        # Clear existing data
        cursor.execute("DELETE FROM ml_sessions")
        print(f"✓ Cleared existing sessions")
        
        # Insert sample sessions - match the actual schema
        insert_query = """
        INSERT INTO ml_sessions (
            session_id, session_length, is_anomaly, anomaly_score, 
            anomaly_type, detected_patterns, critical_events, timestamp
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        sessions_created = 0
        for i in range(1, num_sessions + 1):
            session = create_sample_session(i)
            cursor.execute(insert_query, (
                session['session_id'],
                session['session_length'],
                session['is_anomaly'],
                session['anomaly_score'],
                session['anomaly_type'],
                session['detected_patterns'],  # Will be stored as JSON string
                session['critical_events'],   # Will be stored as JSON string  
                session['timestamp']
            ))
            sessions_created += 1
            
            if i % 50 == 0:
                print(f"  Created {i} sessions...")
        
        # Commit all changes
        conn.commit()
        
        # Get statistics
        cursor.execute("SELECT COUNT(*) FROM ml_sessions WHERE is_anomaly = true")
        anomaly_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM ml_sessions WHERE is_anomaly = false")
        normal_count = cursor.fetchone()[0]
        
        print(f"\n🎉 Successfully populated database:")
        print(f"   Total sessions: {sessions_created}")
        print(f"   Normal sessions: {normal_count}")
        print(f"   Anomalous sessions: {anomaly_count}")
        print(f"   Anomaly rate: {anomaly_count/sessions_created*100:.1f}%")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Error populating database: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Populating database with sample ML sessions...")
    success = populate_database(250)
    if success:
        print("\n✓ Database population complete! Ready for model training.")
    else:
        print("\n❌ Database population failed.")
