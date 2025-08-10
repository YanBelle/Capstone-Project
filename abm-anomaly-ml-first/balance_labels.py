#!/usr/bin/env python
"""
Script to balance labels by consolidating similar categories or adding more samples
to ensure minimum 2 samples per class for supervised training.
"""

import requests
import json

# Configuration
API_BASE_URL = "http://localhost:8000"

# Label consolidation mapping - merge similar labels together
LABEL_CONSOLIDATION = {
    "Supervisor Mode Anomaly": "System Administrative Issue",
    "System Recovery Failure": "System Administrative Issue", 
    "Incomplete Transaction": "Transaction Failure",
    "Card Retention Issue": "Transaction Failure",
    "Authentication Failure": "Security Issue",
    "Device Hardware Error": "Hardware Issue",
    "Cash Retraction Error": "Hardware Issue",
    "Communication Timeout": "System Administrative Issue",
    "Note Handling Error": "Hardware Issue"
}

def get_labeled_anomalies():
    """Fetch all labeled anomalies."""
    print("Fetching labeled anomalies...")
    try:
        response = requests.get(API_BASE_URL + "/api/v1/expert/anomalies?filter=labeled&limit=10000")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print("Error fetching anomalies: {}".format(e))
        return {"sessions": []}

def count_label_distribution(sessions):
    """Count current label distribution."""
    distribution = {}
    for session in sessions:
        label = session.get("expert_label", "Unknown")
        distribution[label] = distribution.get(label, 0) + 1
    return distribution

def consolidate_labels(sessions):
    """Consolidate similar labels to ensure minimum samples per class."""
    consolidated_labels = []
    
    for session in sessions:
        old_label = session.get("expert_label", "Unknown")
        
        # Check if this label should be consolidated
        if old_label in LABEL_CONSOLIDATION:
            new_label = LABEL_CONSOLIDATION[old_label]
        else:
            new_label = old_label
            
        consolidated_labels.append({
            "session_id": session["session_id"],
            "label": new_label,
            "is_excluded": False
        })
    
    return consolidated_labels

def save_labels(labels):
    """Save updated labels to the API."""
    if not labels:
        return True
        
    payload = {"labels": labels}
    
    try:
        response = requests.post(
            API_BASE_URL + "/api/v1/expert/save-labels",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()
        print("Successfully updated {} labels".format(result.get('saved_count', len(labels))))
        return True
    except requests.exceptions.RequestException as e:
        print("Error saving labels: {}".format(e))
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print("Response: {}".format(e.response.text))
        return False

def main():
    """Main function to balance labels."""
    print("Label Balancing Script")
    print("=" * 30)
    
    # Fetch labeled anomalies
    data = get_labeled_anomalies()
    sessions = data.get("sessions", [])
    
    if not sessions:
        print("No labeled anomalies found!")
        return
    
    print("Found {} labeled anomalies".format(len(sessions)))
    
    # Show current distribution
    current_distribution = count_label_distribution(sessions)
    print("\nCurrent label distribution:")
    for label, count in sorted(current_distribution.items(), key=lambda x: x[1]):
        print("  {}: {} samples".format(label, count))
    
    # Identify problematic labels (< 2 samples)
    problematic_labels = [label for label, count in current_distribution.items() if count < 2]
    if problematic_labels:
        print("\nProblematic labels (< 2 samples): {}".format(problematic_labels))
    
    # Consolidate labels
    print("\nConsolidating similar labels...")
    consolidated_labels = consolidate_labels(sessions)
    
    # Show new distribution
    new_sessions = []
    for session in sessions:
        session_copy = session.copy()
        # Find the new label for this session
        for cons_label in consolidated_labels:
            if cons_label["session_id"] == session["session_id"]:
                session_copy["expert_label"] = cons_label["label"]
                break
        new_sessions.append(session_copy)
    
    new_distribution = count_label_distribution(new_sessions)
    print("\nNew label distribution after consolidation:")
    for label, count in sorted(new_distribution.items(), key=lambda x: x[1]):
        print("  {}: {} samples".format(label, count))
    
    # Check if all classes now have >= 2 samples
    min_samples = min(new_distribution.values()) if new_distribution else 0
    print("\nMinimum samples per class: {}".format(min_samples))
    
    if min_samples >= 2:
        print("All classes now have >= 2 samples - ready for training!")
        
        # Save the consolidated labels
        if save_labels(consolidated_labels):
            print("\nLabel consolidation completed successfully!")
            print("You can now retry supervised training.")
        else:
            print("\nFailed to save consolidated labels.")
    else:
        remaining_problematic = [label for label, count in new_distribution.items() if count < 2]
        print("Still problematic labels: {}".format(remaining_problematic))
        print("Consider further consolidation or collecting more data.")

if __name__ == "__main__":
    main()
