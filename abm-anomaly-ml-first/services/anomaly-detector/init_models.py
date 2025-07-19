#!/usr/bin/env python3
"""Initialize placeholder models for first-time setup"""

import os
import json
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
import joblib

def initialize_models():
    """Initialize placeholder models and configuration files"""
    models_dir = "/app/models"
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"Initializing models in {models_dir}...")
    
    # Create dummy data for initial model fitting
    dummy_data = np.random.random((100, 20))
    
    # Initialize StandardScaler
    scaler = StandardScaler()
    scaler.fit(dummy_data)
    joblib.dump(scaler, f"{models_dir}/scaler.pkl")
    print("✓ Created scaler.pkl")
    
    # Initialize PCA
    pca = PCA(n_components=10)
    pca.fit(dummy_data)
    joblib.dump(pca, f"{models_dir}/pca.pkl")
    print("✓ Created pca.pkl")
    
    # Initialize Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(dummy_data)
    joblib.dump(iso_forest, f"{models_dir}/isolation_forest.pkl")
    print("✓ Created isolation_forest.pkl")
    
    # Initialize One-Class SVM
    svm_model = OneClassSVM(gamma='scale', nu=0.1)
    svm_model.fit(dummy_data)
    joblib.dump(svm_model, f"{models_dir}/one_class_svm.pkl")
    print("✓ Created one_class_svm.pkl")
    
    # Create enhanced expert rules
    expert_rules = {
        "rules": [
            {
                "name": "incomplete_transaction_no_pin",
                "pattern": "CARD INSERTED.*CARD TAKEN",
                "exclusion_patterns": ["PIN ENTERED", "OPCODE"],
                "confidence": 0.95,
                "severity": "high",
                "description": "Card inserted and taken without PIN entry or transaction processing"
            },
            {
                "name": "incomplete_transaction_pin_no_completion",
                "pattern": "PIN ENTERED.*OPCODE.*CARD TAKEN",
                "exclusion_patterns": ["NOTES PRESENTED", "RECEIPT PRINTED", "TRANSACTION COMPLETED"],
                "confidence": 0.90,
                "severity": "high",
                "description": "PIN entered and transaction initiated but not completed"
            },
            {
                "name": "supervisor_mode",
                "pattern": "SUPERVISOR MODE",
                "confidence": 0.9,
                "severity": "medium",
                "description": "Supervisor mode entry or exit"
            }
        ]
    }
    
    with open(f"{models_dir}/expert_rules.json", "w") as f:
        json.dump(expert_rules, f, indent=2)
    print("✓ Created expert_rules.json")
    
    print("Models initialized successfully!")
    return True

if __name__ == "__main__":
    initialize_models()
