# Fix False Positive: NOTES PRESENTED + NOTES TAKEN Pattern

## Problem Analysis

The current ML model is incorrectly flagging normal successful transactions as anomalies. Specifically, the pattern:
1. `NOTES PRESENTED` 
2. `NOTES TAKEN`

This is actually the **expected successful transaction flow**:
- `NOTES PRESENTED`: ATM has successfully dispensed cash from cassettes
- `NOTES TAKEN`: Customer has successfully collected the cash

## Root Cause

The unsupervised learning model (Isolation Forest/One-Class SVM) has learned from a dataset where this pattern appears less frequently, causing it to be classified as an outlier/anomaly when it's actually normal behavior.

## Solution: Supervised Learning with Proper Labeling

### 1. Create Training Labels

```python
# Correct labeling for transaction patterns
training_labels = {
    # NORMAL SUCCESSFUL TRANSACTIONS
    "notes_presented_then_taken": {
        "pattern": ["NOTES PRESENTED", "NOTES TAKEN"],
        "label": "normal_transaction",
        "description": "Successful cash dispensing and collection"
    },
    
    # ACTUAL ANOMALIES
    "notes_presented_not_taken": {
        "pattern": ["NOTES PRESENTED", "timeout", "NOTES RETRACTED"],
        "label": "customer_abandon",
        "description": "Customer did not collect dispensed cash"
    },
    
    "unable_to_dispense": {
        "pattern": ["UNABLE TO DISPENSE"],
        "label": "dispense_failure", 
        "description": "ATM could not dispense requested amount"
    },
    
    "device_error": {
        "pattern": ["DEVICE ERROR"],
        "label": "hardware_fault",
        "description": "Hardware malfunction detected"
    }
}
```

### 2. Update ML Model Training

The model needs to be retrained with expert-labeled data to distinguish between:
- **Normal Operations**: Successful transaction patterns
- **True Anomalies**: Actual problems requiring attention

### 3. Immediate Fix

For the current session `SESSION_13_3239535f`, this should be reclassified as:
- **Status**: Normal Transaction ✅
- **Pattern**: Successful Withdrawal
- **Action Required**: None

## Implementation Steps

1. **Create expert-labeled training dataset**
2. **Retrain supervised classifier**  
3. **Update anomaly detection rules**
4. **Validate against known good transactions**

## Expected Outcome

After retraining:
- `NOTES PRESENTED → NOTES TAKEN` = **Normal** (no alert)
- `NOTES PRESENTED → [timeout/no collection]` = **Anomaly** (alert required)
- `UNABLE TO DISPENSE` = **Anomaly** (alert required)

This will significantly reduce false positives and improve the system's accuracy for real anomaly detection.
