#!/usr/bin/env python3

"""
Retrain BERT-DeepLog model with cleaned training data
"""

import sys
import os
sys.path.append('/app')

from services.anomaly_detector.bert_deeplog_model import BertDeepLogAnalyzer
import json

def retrain_with_cleaned_data():
    """Retrain the model using cleaned normal sessions"""
    
    print("=== RETRAINING WITH CLEANED DATA ===")
    
    # Load cleaned normal sessions
    with open('/app/data/processed/normal_sessions_cleaned.json', 'r') as f:
        cleaned_normal = json.load(f)
    
    # Load error sessions  
    with open('/app/data/processed/error_sessions_full_20250803_102920.json', 'r') as f:
        error_sessions = json.load(f)
    
    print("Cleaned normal sessions:", len(cleaned_normal))
    print("Error sessions:", len(error_sessions))
    
    # Initialize analyzer
    analyzer = BertDeepLogAnalyzer()
    
    # Prepare training data (normal sessions only for unsupervised learning)
    analyzer.prepare_training_data(cleaned_normal, normal_sessions_only=True)
    
    # Train model
    training_history = analyzer.train_model(validation_split=0.2)
    
    print("Retraining completed!")
    print("Training history:", len(training_history))
    
    return analyzer

if __name__ == "__main__":
    retrained_analyzer = retrain_with_cleaned_data()
