#!/usr/bin/env python3
import json
import requests

# Read the sample training data
with open('sample_training_data.json', 'r') as f:
    data = json.load(f)

sessions = data['sessions']

# Create training request
training_data = {
    "sessions": sessions,
    "text_weight": 0.6,
    "statistical_weight": 0.4,
    "threshold": 0.5
}

# Send training request
response = requests.post(
    'http://localhost:8001/api/train',
    headers={'Content-Type': 'application/json'},
    json=training_data
)

print(f"Status Code: {response.status_code}")
print(f"Response: {response.text}")
