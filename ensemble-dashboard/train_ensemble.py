#!/usr/bin/env python3
import requests
import json

# Load sessions first
print("Loading EJ sessions...")
load_response = requests.post('http://localhost:8001/api/load_ej_sessions')
if load_response.status_code == 200:
    data = load_response.json()
    if data.get('success'):
        sessions = data.get('sessions', [])
        print(f"Loaded {len(sessions)} sessions")
        
        if len(sessions) >= 3:  # Need at least 3 for training
            # Train with loaded sessions
            print("Training model...")
            train_data = {
                "sessions": sessions[:50],  # Use first 50 sessions
                "text_weight": 0.4,
                "statistical_weight": 0.3,
                "threshold": 0.5
            }
            
            train_response = requests.post(
                'http://localhost:8001/api/train',
                headers={'Content-Type': 'application/json'},
                json=train_data
            )
            
            if train_response.status_code == 200:
                train_result = train_response.json()
                print("Training successful!")
                print(json.dumps(train_result, indent=2))
            else:
                print(f"Training failed: {train_response.status_code}")
                print(train_response.text)
        else:
            print(f"Not enough sessions for training. Got {len(sessions)}, need at least 3")
    else:
        print(f"Failed to load sessions: {data}")
else:
    print(f"Failed to load sessions: {load_response.status_code}")
    print(load_response.text)
