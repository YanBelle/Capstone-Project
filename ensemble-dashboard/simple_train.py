import requests
import json

# Sample ATM sessions with clear semantic categories
sample_sessions = [
    # Authentication Issues
    "TRANSACTION_START CARD_INSERTED PIN_ENTERED PIN_VERIFICATION_FAILED authentication failure",
    "CARD_INSERTED PIN_ENTERED PIN_VERIFICATION_FAILED E-45 authentication error card retained",
    "PIN_ENTERED PIN_VERIFICATION_FAILED timeout authentication failure security response",
    
    # Hardware Failures  
    "DEVICE_ERROR M-65 SUPERVISOR_MODE device initialization failure maintenance required",
    "DEVICE_ERROR M-01 critical system error hardware malfunction service required",
    "DEVICE_ERROR M-15 dispenser mechanism fault cash jam service needed",
    
    # Successful Transactions
    "TRANSACTION_START CARD_INSERTED PIN_ENTERED CASH_DISPENSED 100 RECEIPT_PRINTED successful",
    "CARD_INSERTED PIN_ENTERED AMOUNT_SELECTED CASH_DISPENSED 200 RECEIPT_PRINTED completed",
    "PIN_ENTERED CASH_DISPENSED 50 RECEIPT_PRINTED CARD_EJECTED transaction successful",
    
    # Communication Issues
    "COMMUNICATION_FAILURE M-23 timeout network error server unreachable",
    "COMMUNICATION_FAILURE network timeout authentication server connectivity issues", 
    "TRANSACTION_START COMMUNICATION_FAILURE timeout network error unable to complete"
]

print("Training semantic clustering model...")
print(f"Sample sessions: {len(sample_sessions)}")

try:
    response = requests.post(
        "http://localhost:8001/api/train",
        json={"sessions": sample_sessions},
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print("Training successful!")
        print(json.dumps(result, indent=2))
    else:
        print(f"Training failed: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"Error: {e}")
