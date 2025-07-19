# Continuous Learning Interface - User Guide

## Overview
The Continuous Learning Interface allows domain experts to provide feedback on ML model predictions, enabling the system to improve its anomaly detection accuracy over time.

## Key Features

### 1. **Real-time Feedback Collection**
- Expert can review ML predictions and provide corrections
- System tracks confidence levels and feedback types
- Immediate incorporation into learning buffer

### 2. **Automatic Model Retraining**
- Triggers when feedback buffer reaches threshold (default: 50 samples)
- Updates Isolation Forest, One-Class SVM, and Supervised Classifier
- Preserves model performance history

### 3. **Expert Override Validation**
- Reviews sessions where expert rules overrode ML predictions
- Allows validation or correction of override decisions
- Improves expert rule accuracy

## How to Use

### Step 1: Access the Interface
1. Navigate to the dashboard
2. Click on the "ML Training" tab
3. View current learning status and feedback buffer

### Step 2: Provide Expert Feedback
1. Select a session from the feedback queue
2. Review the transaction details and ML prediction
3. Choose appropriate feedback:
   - **Confirmation**: ML prediction is correct
   - **Correction**: ML prediction is wrong
   - **New Discovery**: New type of anomaly identified
4. Set confidence level (0.1 to 1.0)
5. Provide explanation for the decision
6. Submit feedback

### Step 3: Monitor Learning Progress
- Track feedback buffer size
- View retraining cycles completed
- Monitor performance improvements
- Review feedback type distribution

### Step 4: Trigger Manual Retraining (Optional)
- Click "Trigger Retraining" when ready
- Requires minimum 5 feedback samples
- System will automatically retrain at threshold

## Feedback Types Explained

### **Confirmation**
Use when ML prediction is correct:
- Anomaly correctly identified as anomaly
- Normal transaction correctly identified as normal
- Helps reinforce correct model behavior

### **Correction**
Use when ML prediction is wrong:
- False Positive: ML said anomaly, but it's normal
- False Negative: ML said normal, but it's an anomaly
- Most valuable for model improvement

### **New Discovery**
Use when identifying new anomaly patterns:
- Previously unknown failure modes
- New attack patterns
- Novel hardware issues
- Helps expand model knowledge

## Expert Labels Guide

### Common Anomaly Types:
- `dispense_failure` - ATM unable to dispense cash
- `hardware_error` - Device malfunction detected
- `communication_error` - Network/communication issues
- `supervisor_activity` - Maintenance or supervisor actions
- `cash_handling_issue` - Problems with cash cassettes
- `card_reader_error` - Card insertion/reading problems
- `timeout_issue` - Transaction timed out
- `normal` - Regular successful transaction

### Custom Labels:
You can create custom labels for new anomaly types:
- `new_fraud_pattern`
- `specific_hardware_model_issue`
- `environmental_failure`

## Best Practices

### 1. **Be Consistent**
- Use same labels for similar issues
- Maintain consistent confidence levels
- Provide clear explanations

### 2. **Focus on Edge Cases**
- Prioritize unclear or borderline cases
- Review high-confidence ML predictions that seem wrong
- Examine expert-overridden sessions

### 3. **Provide Context**
- Explain reasoning in the explanation field
- Reference specific parts of the transaction log
- Note any external factors (time, location, etc.)

### 4. **Monitor Impact**
- Review performance improvements after retraining
- Validate that corrections are being learned
- Adjust feedback strategy based on results

## System Behavior

### **Feedback Buffer**
- Collects expert feedback until threshold reached
- Weights feedback based on type and confidence
- Triggers automatic retraining at capacity

### **Model Updates**
- **Isolation Forest**: Adjusts contamination parameter
- **One-Class SVM**: Modifies decision boundary
- **Supervised Classifier**: Learns from labeled examples
- **Expert Rules**: Updated based on feedback patterns

### **Performance Tracking**
- Measures accuracy improvement after retraining
- Tracks false positive/negative reduction
- Maintains history of all retraining cycles

## Troubleshooting

### **No Sessions Available**
- Check filter settings (Recent, High Confidence, Overridden)
- Verify that new anomalies are being detected
- Ensure sessions haven't already been labeled

### **Retraining Button Disabled**
- Need minimum 5 feedback samples
- Check feedback buffer size in status panel
- Wait for more expert feedback to accumulate

### **Feedback Not Submitting**
- Verify all required fields are filled
- Check network connection to API
- Ensure session ID is valid

## Expected Results

### **Short Term (After 50+ Feedback Samples)**
- Reduced false positive rate
- Better detection of previously missed anomalies
- More accurate anomaly type classification

### **Long Term (After Multiple Retraining Cycles)**
- Models adapted to specific ATM environment
- Fewer false alerts requiring human review
- Discovery of new anomaly patterns
- Improved overall system reliability

## API Integration

The continuous learning system exposes these endpoints:

```
POST /api/v1/continuous-learning/feedback
GET  /api/v1/continuous-learning/status
POST /api/v1/continuous-learning/trigger-retraining
GET  /api/v1/continuous-learning/feedback-sessions
GET  /api/v1/continuous-learning/session-details/{session_id}
```

For programmatic integration or custom tools.
