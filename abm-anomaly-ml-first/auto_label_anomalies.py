#!/usr/bin/env python3
"""
Standalone script to automatically label anomalies based on their critical events.
This script fetches unlabeled anomalies and assigns appropriate labels based on detected patterns.
"""

import requests
import json
import time

# Configuration
API_BASE_URL = "http://localhost:8000"
BATCH_SIZE = 10  # Process anomalies in batches

# Label mapping based on critical events and detected patterns
LABEL_MAPPING = {
    "unable_to_dispense": "Dispense Failure",
    "card_retained": "Card Retention Issue", 
    "supervisor_mode": "Supervisor Mode Anomaly",
    "device_error": "Device Hardware Error",
    "cash_retract": "Cash Retraction Error",
    "error_codes": "System Error",
    "incorrect_pin": "Authentication Failure",
    "customer_cancelled": "Customer Cancellation",
    "cardless_transaction": "Cardless Transaction Pattern",
    "incomplete_transaction": "Incomplete Transaction",
    "statistical_outlier": "Statistical Anomaly",
    "power_reset": "Power Reset Issue",
    "communication_timeout": "Communication Timeout",
    "note_handling": "Note Handling Error"
}

# Default labels for anomaly types when no specific critical events are found
ANOMALY_TYPE_LABELS = {
    "statistical_outlier_svm": "Statistical Anomaly - SVM",
    "statistical_outlier_isolation": "Statistical Anomaly - Isolation Forest", 
    "incomplete_transaction": "Incomplete Transaction",
    "card_retained": "Card Retention Issue",
    "dispense_failure": "Dispense Failure"
}

def get_unlabeled_anomalies():
    """Fetch unlabeled anomalies from the API."""
    print("Fetching unlabeled anomalies...")
    try:
        response = requests.get(API_BASE_URL + "/api/v1/expert/anomalies?filter=unlabeled&limit=100")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print("Error fetching anomalies: {}".format(e))
        return {"sessions": [], "stats": {"total": 0}}

def determine_label(session):
    """Determine the appropriate label for a session based on critical events and patterns."""
    critical_events = session.get("critical_events", [])
    detected_patterns = session.get("detected_patterns", [])
    anomaly_type = session.get("anomaly_type", "")
    raw_text = session.get("raw_text", "").upper()
    
    # Priority 1: Check critical events
    for event in critical_events:
        event_lower = event.lower()
        for pattern, label in LABEL_MAPPING.items():
            if pattern in event_lower:
                return label
    
    # Priority 2: Check detected patterns
    for pattern in detected_patterns:
        pattern_lower = pattern.lower()
        if pattern_lower in LABEL_MAPPING:
            return LABEL_MAPPING[pattern_lower]
    
    # Priority 3: Check raw text for specific indicators
    if "UNABLE TO DISPENSE" in raw_text:
        return "Dispense Failure"
    elif "CARD RETAINED" in raw_text:
        return "Card Retention Issue"
    elif "INCORRECT PIN" in raw_text:
        return "Authentication Failure"
    elif "CUSTOMER CANCELLED" in raw_text:
        return "Customer Cancellation"
    elif "DEVICE ERROR" in raw_text:
        return "Device Hardware Error"
    elif "SUPERVISOR MODE" in raw_text:
        return "Supervisor Mode Anomaly"
    elif "CARDLESS TRANSACTION" in raw_text:
        return "Cardless Transaction Pattern"
    elif "SYSTEM ERROR" in raw_text:
        return "System Recovery Failure"
    elif "TIMEOUT" in raw_text:
        return "Communication Timeout"
    elif "RETRACT" in raw_text:
        return "Cash Retraction Error"
    
    # Priority 4: Use anomaly type as fallback
    if anomaly_type in ANOMALY_TYPE_LABELS:
        return ANOMALY_TYPE_LABELS[anomaly_type]
    
    # Default fallback
    return "Suspicious Transaction Pattern"

def save_labels(labels):
    """Save labels to the API."""
    if not labels:
        return True
        
    payload = {"labels": labels}
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/expert/save-labels",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()
        print(f"Successfully saved {result.get('saved_count', len(labels))} labels")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error saving labels: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        return False

def main():
    """Main function to auto-label anomalies."""
    print("🔍 Auto-Labeling Anomalies Script")
    print("=" * 40)
    
    # Fetch unlabeled anomalies
    data = get_unlabeled_anomalies()
    sessions = data.get("sessions", [])
    total_anomalies = len(sessions)
    
    if total_anomalies == 0:
        print("✅ No unlabeled anomalies found!")
        return
    
    print(f"📊 Found {total_anomalies} unlabeled anomalies")
    print("\n🏷️  Starting auto-labeling process...")
    
    # Process sessions in batches
    labeled_count = 0
    label_distribution = {}
    
    for i in range(0, total_anomalies, BATCH_SIZE):
        batch = sessions[i:i + BATCH_SIZE]
        batch_labels = []
        
        print(f"\n📦 Processing batch {i//BATCH_SIZE + 1} ({len(batch)} sessions)...")
        
        for session in batch:
            session_id = session["session_id"]
            label = determine_label(session)
            
            # Track label distribution
            label_distribution[label] = label_distribution.get(label, 0) + 1
            
            batch_labels.append({
                "session_id": session_id,
                "label": label,
                "is_excluded": False  # Mark as valid anomaly, not excluded
            })
            
            print(f"  🔖 {session_id[-12:]}: {label}")
        
        # Save this batch
        if save_labels(batch_labels):
            labeled_count += len(batch_labels)
            time.sleep(0.5)  # Small delay between batches
        else:
            print(f"❌ Failed to save batch {i//BATCH_SIZE + 1}")
            break
    
    # Print summary
    print("\n" + "=" * 50)
    print("📋 LABELING SUMMARY")
    print("=" * 50)
    print(f"✅ Successfully labeled: {labeled_count}/{total_anomalies} anomalies")
    
    if label_distribution:
        print(f"\n🏷️  Label Distribution:")
        for label, count in sorted(label_distribution.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / labeled_count) * 100
            print(f"   • {label}: {count} ({percentage:.1f}%)")
    
    if labeled_count == total_anomalies:
        print(f"\n🎉 All anomalies have been labeled successfully!")
        print(f"💡 You can now proceed to test supervised model training.")
    else:
        print(f"\n⚠️  {total_anomalies - labeled_count} anomalies were not labeled due to errors.")

if __name__ == "__main__":
    main()
