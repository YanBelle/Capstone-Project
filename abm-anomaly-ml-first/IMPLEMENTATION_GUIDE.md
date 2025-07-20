# Implementation Guide: Knowledge-Guided ML Enhancement

## Overview

This guide provides step-by-step instructions for implementing the enhanced knowledge-guided ML anomaly detection system that addresses the current rigid rule interference issues.

## Current Problem Analysis

### Issues with Current System

1. **Binary Override Decisions**: Current expert rules completely override ML predictions
2. **No Learning from Known Patterns**: System doesn't use known error patterns as training signals
3. **Rigid Rule Application**: Rules are applied inflexibly without considering confidence levels
4. **Limited Pattern Discovery**: ML can't build on known patterns to find variations

### Example of Current Problematic Code

```python
# Current problematic approach in ml_analyzer.py
def apply_expert_override(self, session: TransactionSession) -> bool:
    if self.is_successful_withdrawal(session.raw_text, events):
        session.is_anomaly = False  # COMPLETE OVERRIDE - prevents ML learning
        session.anomaly_score = 0.0  # ZERO SCORE - loses all ML insights
        return True
```

## Recommended Implementation Strategy

### Phase 1: Replace Binary Overrides with Confidence Adjustment

#### Step 1.1: Update Expert Override Logic

Replace the current `apply_expert_override()` method with a confidence adjustment approach:

```python
def adjust_ml_confidence_with_domain_knowledge(self, session, ml_score):
    """Adjust ML confidence using domain knowledge rather than hard overrides"""
    
    original_score = ml_score
    adjustment_factor = 1.0
    knowledge_signals = []
    
    # Known normal patterns - reduce but don't eliminate confidence
    if self.is_successful_withdrawal(session.raw_text):
        adjustment_factor *= 0.2  # Reduce to 20% but keep some signal
        knowledge_signals.append({
            'type': 'known_normal_pattern',
            'pattern': 'successful_withdrawal',
            'confidence': 0.9,
            'adjustment': 0.2
        })
    
    # Known error patterns - boost confidence
    if self.has_genuine_hardware_error(session.raw_text):
        adjustment_factor *= 1.5  # Increase confidence by 50%
        knowledge_signals.append({
            'type': 'known_error_pattern', 
            'pattern': 'hardware_error',
            'confidence': 0.95,
            'adjustment': 1.5
        })
    
    adjusted_score = min(1.0, original_score * adjustment_factor)
    
    return {
        'original_ml_score': original_score,
        'adjusted_score': adjusted_score,
        'adjustment_factor': adjustment_factor,
        'knowledge_signals': knowledge_signals,
        'confidence_in_adjustment': self.calculate_adjustment_confidence(knowledge_signals)
    }
```

#### Step 1.2: Modify the Detection Pipeline

Update the main detection logic in `detect_anomalies_unsupervised()`:

```python
def detect_anomalies_unsupervised(self) -> Dict[str, np.ndarray]:
    # ... existing ML detection code ...
    
    for i, session in enumerate(self.sessions):
        # Get ML scores
        if_score_norm = self.normalize_score(if_scores[i], if_scores)
        svm_score_norm = self.normalize_score(svm_scores[i], svm_scores)
        combined_ml_score = max(if_score_norm, svm_score_norm)
        
        # Apply knowledge-guided adjustment instead of override
        knowledge_adjustment = self.adjust_ml_confidence_with_domain_knowledge(
            session, combined_ml_score
        )
        
        # Set both original and adjusted scores
        session.ml_anomaly_score = combined_ml_score
        session.knowledge_adjusted_score = knowledge_adjustment['adjusted_score']
        session.anomaly_score = knowledge_adjustment['adjusted_score']
        session.is_anomaly = knowledge_adjustment['adjusted_score'] > 0.5
        
        # Store knowledge signals for transparency
        session.knowledge_signals = knowledge_adjustment['knowledge_signals']
        session.confidence_level = knowledge_adjustment['confidence_in_adjustment']
```

### Phase 2: Implement Pattern Learning System

#### Step 2.1: Create Pattern Similarity Analyzer

```python
class PatternLearningSystem:
    """Learn from known patterns to detect similar variations"""
    
    def __init__(self):
        self.known_patterns = self.load_base_patterns()
        self.pattern_embeddings = {}
        self.similarity_threshold = 0.75
        
    def load_base_patterns(self):
        return {
            'successful_withdrawal': {
                'core_sequence': ['CARD_INSERTED', 'PIN_ENTERED', 'NOTES_PRESENTED', 'NOTES_TAKEN'],
                'variations': ['receipt_printed', 'balance_inquiry', 'timeout_occurred'],
                'embedding': None  # Will be computed
            },
            'dispense_failure': {
                'core_sequence': ['DISPENSE_COMMAND', 'UNABLE_TO_DISPENSE'],
                'variations': ['retry_attempt', 'cassette_empty', 'mechanical_jam'],
                'embedding': None
            }
        }
    
    def learn_pattern_variations(self, session):
        """Learn new variations of known patterns"""
        
        session_embedding = self.get_session_embedding(session)
        
        for pattern_name, pattern_def in self.known_patterns.items():
            similarity = self.calculate_pattern_similarity(
                session_embedding, 
                pattern_def['embedding']
            )
            
            if similarity > self.similarity_threshold:
                # This is a variation of a known pattern
                return {
                    'base_pattern': pattern_name,
                    'similarity': similarity,
                    'is_variation': True,
                    'confidence_adjustment': self.calculate_variation_adjustment(similarity)
                }
        
        return {
            'base_pattern': None,
            'similarity': 0.0,
            'is_variation': False,
            'confidence_adjustment': 1.0  # No adjustment for novel patterns
        }
```

#### Step 2.2: Integrate Pattern Learning into Detection

```python
def detect_with_pattern_learning(self, session):
    """Enhanced detection that learns from known patterns"""
    
    # Get base ML score
    ml_score = self.get_ml_anomaly_score(session)
    
    # Check for pattern similarity
    pattern_analysis = self.pattern_learning_system.learn_pattern_variations(session)
    
    if pattern_analysis['is_variation']:
        # Adjust confidence based on pattern similarity
        adjusted_score = ml_score * pattern_analysis['confidence_adjustment']
        
        session.pattern_match = {
            'base_pattern': pattern_analysis['base_pattern'],
            'similarity': pattern_analysis['similarity'],
            'type': 'pattern_variation'
        }
    else:
        # Novel pattern - keep original ML score
        adjusted_score = ml_score
        session.pattern_match = {
            'base_pattern': None,
            'similarity': 0.0,
            'type': 'novel_pattern'
        }
    
    return adjusted_score
```

### Phase 3: Leverage Log Analysis Data Science Models

#### Step 3.1: Temporal Anomaly Detection

```python
class TemporalAnomalyDetector:
    """Detect temporal anomalies using time series analysis"""
    
    def __init__(self):
        from sklearn.ensemble import IsolationForest
        self.duration_detector = IsolationForest(contamination=0.1)
        self.timing_detector = IsolationForest(contamination=0.05)
        
    def detect_temporal_anomalies(self, session):
        """Detect timing-based anomalies"""
        
        temporal_features = self.extract_temporal_features(session)
        
        results = {
            'duration_anomaly': False,
            'timing_anomaly': False,
            'temporal_score': 0.0
        }
        
        # Check session duration
        if temporal_features['duration'] is not None:
            duration_score = self.duration_detector.decision_function([[temporal_features['duration']]])[0]
            if duration_score < -0.5:  # Threshold for anomaly
                results['duration_anomaly'] = True
                results['temporal_score'] += 0.3
        
        # Check inter-event timing
        if temporal_features['event_intervals']:
            timing_scores = self.timing_detector.decision_function(
                temporal_features['event_intervals'].reshape(-1, 1)
            )
            anomalous_intervals = (timing_scores < -0.5).sum()
            if anomalous_intervals > 0:
                results['timing_anomaly'] = True
                results['temporal_score'] += min(0.4, anomalous_intervals * 0.1)
        
        return results
    
    def extract_temporal_features(self, session):
        """Extract timing features from session"""
        
        features = {
            'duration': None,
            'event_intervals': np.array([])
        }
        
        if session.start_time and session.end_time:
            features['duration'] = (session.end_time - session.start_time).total_seconds()
        
        # Extract event timestamps and calculate intervals
        timestamps = self.extract_event_timestamps(session.raw_text)
        if len(timestamps) > 1:
            intervals = np.diff(timestamps)
            features['event_intervals'] = intervals
        
        return features
```

#### Step 3.2: Sequence Anomaly Detection with LSTM

```python
class SequenceAnomalyDetector:
    """Detect sequence anomalies using LSTM models"""
    
    def __init__(self):
        self.sequence_model = self.build_lstm_model()
        self.event_encoder = self.build_event_encoder()
        
    def build_lstm_model(self):
        """Build LSTM model for sequence anomaly detection"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(None, 50)),  # 50-dim event encoding
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')  # Anomaly probability
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def detect_sequence_anomalies(self, session):
        """Detect sequence-based anomalies"""
        
        # Extract and encode event sequence
        event_sequence = self.extract_event_sequence(session.raw_text)
        encoded_sequence = self.event_encoder.encode_sequence(event_sequence)
        
        if len(encoded_sequence) == 0:
            return {'sequence_anomaly_score': 0.0, 'sequence_type': 'no_events'}
        
        # Predict using LSTM
        sequence_array = np.array(encoded_sequence).reshape(1, -1, 50)
        anomaly_probability = self.sequence_model.predict(sequence_array)[0][0]
        
        return {
            'sequence_anomaly_score': anomaly_probability,
            'sequence_length': len(event_sequence),
            'sequence_type': 'lstm_detected'
        }
```

### Phase 4: Implement Continuous Learning Feedback

#### Step 4.1: Expert Feedback Integration

```python
class ContinuousLearningSystem:
    """System for continuous learning from expert feedback"""
    
    def __init__(self):
        self.feedback_buffer = []
        self.learning_threshold = 50  # Retrain after 50 feedback instances
        
    def process_expert_feedback(self, session_id, expert_label, confidence):
        """Process expert feedback for continuous learning"""
        
        feedback = {
            'session_id': session_id,
            'expert_label': expert_label,
            'expert_confidence': confidence,
            'timestamp': datetime.now(),
            'original_ml_score': self.get_session_ml_score(session_id),
            'knowledge_adjusted_score': self.get_session_adjusted_score(session_id)
        }
        
        self.feedback_buffer.append(feedback)
        
        # Trigger retraining if threshold reached
        if len(self.feedback_buffer) >= self.learning_threshold:
            self.retrain_with_feedback()
    
    def retrain_with_feedback(self):
        """Retrain models using expert feedback"""
        
        # Prepare training data from feedback
        X_feedback = []
        y_feedback = []
        
        for feedback in self.feedback_buffer:
            session = self.get_session_by_id(feedback['session_id'])
            if session:
                X_feedback.append(session.embedding)
                y_feedback.append(1 if feedback['expert_label'] == 'anomaly' else 0)
        
        if len(X_feedback) > 10:  # Minimum samples for retraining
            # Update supervised model
            self.update_supervised_model(X_feedback, y_feedback)
            
            # Update knowledge base with new patterns
            self.update_knowledge_base_from_feedback()
            
            # Clear feedback buffer
            self.feedback_buffer = []
            
            logger.info(f"Models retrained with {len(X_feedback)} expert feedback samples")
```

## Integration Steps

### Step 1: Backup Current Implementation

```bash
cp services/anomaly-detector/ml_analyzer.py services/anomaly-detector/ml_analyzer_backup.py
```

### Step 2: Implement Confidence Adjustment

Replace the rigid override logic with confidence adjustment:

1. Modify `apply_expert_override()` to return adjustment factors instead of boolean overrides
2. Update `detect_anomalies_unsupervised()` to apply adjustments
3. Add knowledge signal tracking

### Step 3: Add Pattern Learning

1. Implement `PatternLearningSystem` class
2. Integrate pattern similarity analysis
3. Add variation detection logic

### Step 4: Integrate Advanced Models

1. Add temporal anomaly detection
2. Implement sequence analysis
3. Create ensemble scoring system

### Step 5: Enable Continuous Learning

1. Implement feedback collection system
2. Add retraining capabilities
3. Create knowledge base updates

## Expected Benefits

1. **Reduced False Positives**: Knowledge guides ML without completely overriding it
2. **Pattern Discovery**: System can find variations of known patterns
3. **Continuous Improvement**: Learning from expert feedback
4. **Better Transparency**: Clear insight into why decisions were made
5. **Balanced Approach**: Combines domain expertise with ML discovery

## Testing Strategy

1. **A/B Testing**: Compare current system vs enhanced system
2. **Expert Validation**: Have domain experts review results
3. **Performance Metrics**: Track precision, recall, and false positive rates
4. **Pattern Discovery Validation**: Verify that new pattern variations are meaningful

## Rollout Plan

1. **Phase 1**: Deploy confidence adjustment (2 weeks)
2. **Phase 2**: Add pattern learning (3 weeks)
3. **Phase 3**: Integrate advanced models (4 weeks)
4. **Phase 4**: Enable continuous learning (2 weeks)
5. **Phase 5**: Full production deployment (1 week)

Total implementation time: ~12 weeks with proper testing and validation.
