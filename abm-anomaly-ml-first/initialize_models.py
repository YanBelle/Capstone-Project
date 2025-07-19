#!/usr/bin/env python3
"""
Script to initialize ML models directory with empty placeholder files.
This will prevent errors when the ML services try to load models that don't exist yet.
"""
import os
import pickle
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

def create_directory(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def initialize_models(model_dir="/app/models"):
    """Initialize empty ML models for first start"""
    create_directory(model_dir)
    
    # Define the models to create
    models = {
        "isolation_forest": IsolationForest(contamination=0.1, random_state=42),
        "one_class_svm": OneClassSVM(nu=0.05, gamma='auto'),
        "scaler": StandardScaler(),
        "pca": PCA(n_components=20)
    }
    
    # Generate some dummy data to fit the models
    dummy_data = np.random.random((100, 20))
    
    # Fit and save initial models
    for name, model in models.items():
        model_path = os.path.join(model_dir, f"{name}.pkl")
        
        if os.path.exists(model_path):
            print(f"Model {name}.pkl already exists, skipping...")
            continue
            
        try:
            # Fit simple models on dummy data
            if name == "scaler" or name == "pca":
                model.fit(dummy_data)
            else:
                model.fit(dummy_data)
                
            # Save the model
            joblib.dump(model, model_path)
            print(f"Created initial {name}.pkl")
        except Exception as e:
            print(f"Error creating {name}.pkl: {str(e)}")
    
    # Create expert rules file if it doesn't exist
    expert_rules_path = os.path.join(model_dir, "expert_rules.json")
    if not os.path.exists(expert_rules_path):
        with open(expert_rules_path, 'w') as f:
            f.write('{\n  "rules": [\n    {\n      "name": "initial_rule",\n      "pattern": "SUPERVISOR MODE",\n      "confidence": 0.9,\n      "severity": "medium"\n    }\n  ]\n}')
        print(f"Created initial expert_rules.json")

if __name__ == "__main__":
    # Initialize models in local development directory
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "models")
    initialize_models(model_dir)
    print("Model initialization complete. You can now start the services.")
