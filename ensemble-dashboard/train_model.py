#!/usr/bin/env python3
"""
Script to load and train the ensemble model with real EJ session data
"""
import requests
import json

def main():
    base_url = "http://localhost:8001"
    
    print("1. Loading EJ sessions...")
    # Load EJ sessions from processed data
    load_response = requests.post(base_url + "/api/load_ej_sessions")
    if load_response.status_code != 200:
        print("Failed to load sessions: " + load_response.text)
        return
    
    load_data = load_response.json()
    print("Loaded " + str(load_data['count']) + " sessions from " + load_data['data_source'])
    
    sessions = load_data.get('sessions', [])
    if len(sessions) < 10:
        print("Not enough sessions (" + str(len(sessions)) + ") for training. Need at least 10.")
        return
    
    # Use a reasonable subset for training (take first 50 sessions)
    training_sessions = sessions[:50]
    print("Using " + str(len(training_sessions)) + " sessions for training...")
    
    print("2. Training ensemble model...")
    # Train the model
    train_data = {
        "sessions": training_sessions,
        "text_weight": 0.4,
        "statistical_weight": 0.3,
        "threshold": 0.5
    }
    
    train_response = requests.post(
        base_url + "/api/train",
        headers={"Content-Type": "application/json"},
        json=train_data
    )
    
    if train_response.status_code == 200:
        train_result = train_response.json()
        print("Training successful!")
        print("Training stats: " + json.dumps(train_result.get('training_stats', {}), indent=2))
    else:
        print("Training failed: " + train_response.text)
        return
    
    print("3. Testing cluster sessions endpoint...")
    # Test the cluster sessions endpoint
    cluster_response = requests.post(
        base_url + "/api/cluster_sessions",
        headers={"Content-Type": "application/json"},
        json={"cluster_id": 0}
    )
    
    if cluster_response.status_code == 200:
        cluster_result = cluster_response.json()
        print("Cluster sessions endpoint working!")
        print("Cluster 0 has " + str(cluster_result.get('count', 0)) + " sessions")
    else:
        print("Cluster sessions failed: " + cluster_response.text)
    
    print("4. Getting cluster insights...")
    # Get cluster insights
    insights_response = requests.get(base_url + "/api/cluster_insights")
    if insights_response.status_code == 200:
        insights_result = insights_response.json()
        print("Cluster insights available!")
        print("Insights: " + json.dumps(insights_result.get('insights', {}), indent=2))
    else:
        print("Cluster insights: " + insights_response.text)

if __name__ == "__main__":
    main()
