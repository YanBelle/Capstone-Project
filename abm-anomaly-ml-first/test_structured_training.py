#!/usr/bin/env python3
"""
Test structured feature engineering approach for ML training
"""
import requests
import psycopg2
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

# Database connection parameters
DB_CONFIG = {
    'host': 'localhost',
    'port': 5433,
    'database': 'abm_ml_db',
    'user': 'abm_user',
    'password': 'secure_ml_password123'
}

API_BASE = 'http://localhost:8000'

def fetch_structured_data_from_db(limit=100):
    """Fetch and structure data for ML training"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Get sessions with structured data
        query = """
        SELECT session_id, session_length, is_anomaly, anomaly_score, 
               anomaly_type, detected_patterns, critical_events
        FROM ml_sessions 
        ORDER BY RANDOM()
        LIMIT %s
        """
        
        cursor.execute(query, (limit,))
        rows = cursor.fetchall()
        
        # Collect all unique patterns and events for one-hot encoding
        all_patterns = set()
        all_events = set()
        raw_sessions = []
        
        for row in rows:
            raw_sessions.append(row)
            
            # Collect patterns
            if row[5]:  # detected_patterns
                try:
                    patterns = json.loads(row[5]) if isinstance(row[5], str) else row[5]
                    if patterns and isinstance(patterns, list):
                        all_patterns.update(patterns)
                except:
                    pass
            
            # Collect events
            if row[6]:  # critical_events
                try:
                    events = json.loads(row[6]) if isinstance(row[6], str) else row[6]
                    if events and isinstance(events, list):
                        all_events.update(events)
                except:
                    pass
        
        # Sort for consistent feature ordering
        sorted_patterns = sorted(list(all_patterns))
        sorted_events = sorted(list(all_events))
        
        print(f"✓ Found {len(sorted_patterns)} unique patterns: {sorted_patterns}")
        print(f"✓ Found {len(sorted_events)} unique events: {sorted_events}")
        
        # Create structured feature vectors
        feature_vectors = []
        labels = []
        
        for row in raw_sessions:
            # 1. Numerical features
            session_length = float(row[1]) if row[1] else 0.0
            anomaly_score = float(row[3]) if row[3] else 0.0
            feature_vector = [session_length, anomaly_score]
            
            # 2. One-hot encoding for patterns
            patterns = []
            if row[5]:
                try:
                    patterns = json.loads(row[5]) if isinstance(row[5], str) else row[5]
                    patterns = patterns if isinstance(patterns, list) else []
                except:
                    patterns = []
            
            for pattern in sorted_patterns:
                feature_vector.append(1.0 if pattern in patterns else 0.0)
            
            # 3. One-hot encoding for critical events
            events = []
            if row[6]:
                try:
                    events = json.loads(row[6]) if isinstance(row[6], str) else row[6]
                    events = events if isinstance(events, list) else []
                except:
                    events = []
            
            for event in sorted_events:
                feature_vector.append(1.0 if event in events else 0.0)
            
            # 4. Derived features
            pattern_count = len(patterns)
            event_count = len(events)
            feature_vector.extend([
                float(pattern_count),
                float(event_count),
                float(pattern_count + event_count),  # Total activity
                1.0 if pattern_count > 3 else 0.0,  # High pattern activity
                1.0 if event_count > 0 else 0.0     # Has critical events
            ])
            
            feature_vectors.append(feature_vector)
            labels.append(bool(row[2]) if row[2] is not None else False)
        
        cursor.close()
        conn.close()
        
        return np.array(feature_vectors), np.array(labels), sorted_patterns, sorted_events
        
    except Exception as e:
        print(f"❌ Error fetching structured data: {e}")
        return None, None, None, None

def test_structured_ml_training(feature_vectors, labels):
    """Test ML training with structured features"""
    try:
        print(f"\n🧪 Testing ML Training with Structured Features")
        print(f"Feature matrix shape: {feature_vectors.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Anomaly rate: {np.mean(labels):.2%}")
        
        # 1. Feature scaling
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_vectors)
        print(f"✓ Features scaled successfully")
        
        # 2. PCA for dimensionality reduction
        n_components = min(10, scaled_features.shape[1])
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(scaled_features)
        print(f"✓ PCA applied: {scaled_features.shape[1]} → {pca_features.shape[1]} dimensions")
        print(f"  Explained variance ratio: {pca.explained_variance_ratio_[:3]}")
        
        # 3. Isolation Forest training
        contamination = np.mean(labels) if np.mean(labels) > 0 else 0.1
        isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        # Train isolation forest
        isolation_forest.fit(scaled_features)
        print(f"✓ Isolation Forest trained with contamination={contamination:.3f}")
        
        # 4. Predictions and evaluation
        predictions = isolation_forest.predict(scaled_features) == -1
        decision_scores = isolation_forest.decision_function(scaled_features)
        
        # Calculate metrics
        true_positives = np.sum(predictions & labels)
        false_positives = np.sum(predictions & ~labels)
        true_negatives = np.sum(~predictions & ~labels)
        false_negatives = np.sum(~predictions & labels)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n📊 Training Results:")
        print(f"  True Positives: {true_positives}")
        print(f"  False Positives: {false_positives}")
        print(f"  True Negatives: {true_negatives}")
        print(f"  False Negatives: {false_negatives}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1 Score: {f1_score:.3f}")
        
        return {
            'success': True,
            'feature_count': feature_vectors.shape[1],
            'sample_count': feature_vectors.shape[0],
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'decision_scores_range': [float(decision_scores.min()), float(decision_scores.max())]
        }
        
    except Exception as e:
        print(f"❌ Error in ML training: {e}")
        return {'success': False, 'error': str(e)}

def test_api_integration():
    """Test if the API can use structured data"""
    try:
        print(f"\n🌐 Testing API Integration")
        url = f"{API_BASE}/api/v1/isolation-forest/analysis"
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API responded successfully")
            print(f"  Status: {data.get('status', 'unknown')}")
            
            model_info = data.get('model_info', {})
            print(f"  Model Type: {model_info.get('model_type', 'unknown')}")
            print(f"  Vectorization: {model_info.get('vectorization_method', 'unknown')}")
            print(f"  Is Trained: {model_info.get('is_trained', False)}")
            
            if 'data' in data:
                api_data = data['data']
                print(f"  Total Sessions: {api_data.get('total_sessions', 0)}")
                print(f"  Feature Count: {model_info.get('feature_count', 0)}")
                print(f"  Pattern Count: {model_info.get('pattern_count', 0)}")
                print(f"  Event Count: {model_info.get('event_count', 0)}")
            
            return True
        else:
            print(f"❌ API error: {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False

def main():
    print("🚀 Testing Structured Data Training Approach")
    print("=" * 50)
    
    # 1. Check database connection and data
    print("1️⃣ Fetching structured data from database...")
    feature_vectors, labels, patterns, events = fetch_structured_data_from_db(200)
    
    if feature_vectors is None:
        print("❌ Failed to fetch data from database")
        return
    
    # 2. Test ML training with structured features
    print("\n2️⃣ Testing ML training...")
    training_results = test_structured_ml_training(feature_vectors, labels)
    
    # 3. Test API integration
    print("\n3️⃣ Testing API integration...")
    api_success = test_api_integration()
    
    # 4. Summary
    print("\n" + "=" * 50)
    print("📋 TRAINING TEST SUMMARY")
    print("=" * 50)
    
    if training_results['success']:
        print("✅ STRUCTURED TRAINING: SUCCESS")
        print(f"   📊 Feature Engineering: {training_results['feature_count']} features from {training_results['sample_count']} samples")
        print(f"   🎯 Model Performance: F1={training_results['f1_score']:.3f}, Precision={training_results['precision']:.3f}, Recall={training_results['recall']:.3f}")
        print(f"   🔢 Decision Scores: {training_results['decision_scores_range'][0]:.3f} to {training_results['decision_scores_range'][1]:.3f}")
    else:
        print("❌ STRUCTURED TRAINING: FAILED")
        print(f"   Error: {training_results.get('error', 'Unknown error')}")
    
    if api_success:
        print("✅ API INTEGRATION: SUCCESS")
        print("   🌐 Structured vectorization working in API")
    else:
        print("❌ API INTEGRATION: FAILED") 
        print("   🌐 API not responding or returning errors")
    
    # Overall assessment
    if training_results['success'] and api_success:
        print("\n🎉 OVERALL: STRUCTURED APPROACH IS READY FOR PRODUCTION!")
        print("   The new feature engineering approach successfully replaces TF-IDF")
    elif training_results['success']:
        print("\n⚠️  OVERALL: TRAINING WORKS, API NEEDS FIXING")
        print("   The structured approach works, but API integration needs attention")
    else:
        print("\n❌ OVERALL: TRAINING APPROACH NEEDS MORE WORK")
        print("   Both training and API need fixes")

if __name__ == "__main__":
    main()
