# Model Recommendations for EJ Hardware Anomaly Detection

## Problem Analysis
Current BERT-DeepLog model returns 0.0% anomaly probability for obvious hardware errors like "POWER-UP/RESET". This suggests:
1. Training data doesn't contain sufficient hardware error examples
2. BERT tokenization may not properly handle hardware-specific terminology
3. The model architecture may not be optimal for this specific domain

## Recommended Model Architectures

### 1. Domain-Adapted BERT with Hardware Vocabulary
**Approach**: Extend BERT's vocabulary with hardware-specific terms and fine-tune
```python
# Custom vocabulary extension
hardware_terms = [
    "POWER-UP/RESET", "HARDWARE-ERROR", "CIM-RESET", 
    "CAPTURE-FAILED", "RECOVERY-FAILED", "MALFUNCTION"
]
```
**Advantages**: 
- Preserves BERT's contextual understanding
- Better token representation for hardware terms
- Can be fine-tuned on ATM/banking domain data

### 2. Isolation Forest for Multivariate Anomaly Detection
**Approach**: Extract multiple features from EJ sessions and use ensemble isolation
```python
from sklearn.ensemble import IsolationForest
# Features: error_count, session_duration, hardware_keywords, transaction_success_rate
```
**Advantages**:
- Excellent for detecting outliers in multidimensional space
- No need for labeled anomaly data
- Handles mixed data types well

### 3. LSTM Autoencoder for Sequence Reconstruction
**Approach**: Train on normal sequences, detect anomalies by reconstruction error
```python
# Architecture: Encoder-Decoder LSTM
# Input: Sequence of EJ events
# Output: Reconstructed sequence
# Anomaly Score: Reconstruction error
```
**Advantages**:
- Learns normal patterns without needing anomaly examples
- Good for sequential data like EJ logs
- Reconstruction error naturally indicates anomalies

### 4. One-Class SVM with TF-IDF Features
**Approach**: Learn decision boundary around normal sessions using text features
```python
from sklearn.svm import OneClassSVM
from sklearn.feature_extraction.text import TfidfVectorizer
```
**Advantages**:
- Only needs normal data for training
- TF-IDF captures importance of rare terms like "HARDWARE ERROR"
- Robust to variations in normal behavior

### 5. Hybrid CNN-LSTM for Pattern Recognition
**Approach**: CNN for local pattern detection + LSTM for sequence modeling
```python
# CNN layers: Detect local error patterns
# LSTM layers: Model temporal dependencies
# Output: Anomaly probability
```
**Advantages**:
- CNN good at detecting specific error patterns
- LSTM captures sequence dependencies
- Can learn complex hierarchical features

## Recommended Implementation Strategy

### Phase 1: Quick Win - One-Class SVM
Start with One-Class SVM as it's:
- Fast to implement and train
- Only requires normal data
- Good baseline performance
- Interpretable results

### Phase 2: Deep Learning - LSTM Autoencoder
- Better performance on sequential data
- Learns complex normal patterns
- Provides reconstruction visualization

### Phase 3: Advanced - Domain-Adapted Transformer
- State-of-the-art performance
- Handles context and semantics
- Can be fine-tuned continuously

## Recommended Ensemble Approach

### **Primary Ensemble: One-Class SVM + Isolation Forest**
**Why this combination works perfectly:**

```python
# Ensemble voting system
class EnsembleAnomalyDetector:
    def __init__(self):
        self.svm_detector = OneClassSVMAnomalyDetector()
        self.isolation_detector = IsolationForestDetector()
        
    def predict_anomaly(self, session_text):
        # Get predictions from both models
        svm_result = self.svm_detector.predict_anomaly(session_text)
        iso_result = self.isolation_detector.predict_anomaly(session_text)
        
        # Ensemble voting with weights
        ensemble_score = (0.6 * svm_result['anomaly_probability'] + 
                         0.4 * iso_result['anomaly_probability'])
        
        return {
            'is_anomaly': ensemble_score > 0.5,
            'ensemble_probability': ensemble_score,
            'svm_probability': svm_result['anomaly_probability'],
            'isolation_probability': iso_result['anomaly_probability'],
            'agreement': abs(svm_result['anomaly_probability'] - 
                           iso_result['anomaly_probability']) < 0.3
        }
```

**Strengths of this ensemble:**
- ✅ **Complementary approaches**: SVM (text-focused) + Isolation Forest (feature-focused)
- ✅ **Different math**: Support vectors vs tree-based isolation
- ✅ **Both unsupervised**: Only need normal training data
- ✅ **Fast inference**: Both models are lightweight
- ✅ **High confidence**: When both agree, very reliable
- ✅ **Robust to false positives**: Reduces individual model weaknesses

### **Advanced Ensemble: Add LSTM Autoencoder**
**For even stronger detection:**

```python
# Three-model ensemble
class AdvancedEnsembleDetector:
    def __init__(self):
        self.svm_detector = OneClassSVMAnomalyDetector()      # Text patterns
        self.isolation_detector = IsolationForestDetector()   # Feature outliers  
        self.lstm_detector = LSTMAutoencoderDetector()        # Sequence patterns
        
    def predict_anomaly(self, session_text):
        results = {}
        results['svm'] = self.svm_detector.predict_anomaly(session_text)
        results['isolation'] = self.isolation_detector.predict_anomaly(session_text)
        results['lstm'] = self.lstm_detector.predict_anomaly(session_text)
        
        # Weighted ensemble (adjust weights based on your validation)
        ensemble_score = (0.4 * results['svm']['anomaly_probability'] +
                         0.3 * results['isolation']['anomaly_probability'] +
                         0.3 * results['lstm']['anomaly_probability'])
        
        # Consensus voting
        votes = sum([r['is_anomaly'] for r in results.values()])
        
        return {
            'is_anomaly': ensemble_score > 0.5 or votes >= 2,
            'ensemble_probability': ensemble_score,
            'consensus_votes': f"{votes}/3",
            'individual_results': results,
            'high_confidence': votes == 3 or votes == 0  # All agree
        }
```

## Implementation Priority

### **Phase 1: Dual Ensemble (RECOMMENDED)**
1. **One-Class SVM** (Text-based anomaly detection)
2. **Isolation Forest** (Feature-based anomaly detection)
3. **Ensemble Voting** (Combine predictions with weights)

### **Phase 2: Triple Ensemble (ADVANCED)**
3. **LSTM Autoencoder** (Sequence-based anomaly detection)
4. **Advanced Ensemble** (Three-model consensus)

### **Why NOT Include:**
- **Domain-Adapted BERT**: Too complex, defeats the purpose of lightweight ensemble
- **CNN-LSTM**: Redundant with LSTM Autoencoder
