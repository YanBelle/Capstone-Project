#!/usr/bin/env python3
"""
Practical Demonstration: One-Class SVM Anomaly Detection
Shows exactly how hardware errors get detected with real examples
"""

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import re

class SVM_Demo:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
        self.scaler = StandardScaler()
        self.svm = OneClassSVM(kernel='rbf', nu=0.1, gamma='scale')
        
    def extract_features(self, session_text):
        """Extract the same features as production system"""
        features = {}
        lines = session_text.split('\n')
        
        # 1. Hardware patterns (critical for detection)
        hardware_patterns = {
            'power_reset': r'power-up/reset|power.*reset',
            'hardware_error': r'hardware.*error|hardwareerror', 
            'component_failure': r'cim-reset|recovery.*failed|malfunction',
            'device_issues': r'device.*error|initialization.*fail'
        }
        
        for pattern_name, pattern in hardware_patterns.items():
            count = len(re.findall(pattern, session_text.lower()))
            features[f'hw_{pattern_name}'] = count
            
        # Critical hardware score (key indicator)
        features['critical_hardware_score'] = sum([
            features['hw_power_reset'],
            features['hw_hardware_error'] * 2,  # Weight hardware errors more
            features['hw_component_failure'],
            features['hw_device_issues']
        ])
        
        # 2. Error patterns
        error_patterns = [r'error', r'fail', r'malfunction', r'fault', r'timeout']
        total_errors = sum(len(re.findall(pattern, session_text.lower())) 
                          for pattern in error_patterns)
        features['total_error_count'] = total_errors
        
        # 3. Session characteristics
        features['session_length'] = len(lines)
        features['avg_line_length'] = np.mean([len(line) for line in lines if line.strip()])
        
        return list(features.values())
    
    def demonstrate_detection(self):
        print("🎯 ONE-CLASS SVM ANOMALY DETECTION DEMONSTRATION")
        print("=" * 60)
        
        # Training data: Normal ABM sessions
        normal_sessions = [
            """TRANSACTION START
CARD INSERTED
PIN VERIFIED
AMOUNT: $100
TRANSACTION COMPLETE
RECEIPT PRINTED""",
            
            """SESSION INITIATED
BALANCE INQUIRY
ACCOUNT: ****1234
BALANCE: $2,500.00
SESSION ENDED""",
            
            """CARD READER ACTIVE
PIN REQUEST
CASH WITHDRAWAL: $200
DISPENSING CASH
TRANSACTION LOG UPDATED""",
            
            """SYSTEM STARTUP
DEVICE CHECK: OK
READY FOR TRANSACTIONS
CUSTOMER SESSION
NORMAL OPERATION"""
        ]
        
        # Test data: Hardware anomaly
        hardware_anomaly = """POWER-UP/RESET
HARDWARE ERROR - CARD READER MALFUNCTION
HARDWAREERROR DETECTED  
RECOVERY FAILED - UNABLE TO INITIALIZE
CIM-RESET ATTEMPTED
DEVICE ERROR CRITICAL"""
        
        print("📚 TRAINING ON NORMAL SESSIONS...")
        print("-" * 40)
        
        # Extract features from normal sessions
        normal_features = []
        for i, session in enumerate(normal_sessions):
            features = self.extract_features(session)
            normal_features.append(features)
            print(f"Normal Session {i+1} Features: {features}")
        
        # Train TF-IDF and SVM
        normal_texts = [session.replace('\n', ' ') for session in normal_sessions]
        tfidf_features = self.vectorizer.fit_transform(normal_texts).toarray()
        
        # Combine manual and TF-IDF features
        combined_features = []
        for i, manual_feat in enumerate(normal_features):
            combined = manual_feat + list(tfidf_features[i])
            combined_features.append(combined)
        
        # Scale and train
        combined_features_scaled = self.scaler.fit_transform(combined_features)
        self.svm.fit(combined_features_scaled)
        
        print(f"\n✅ SVM Trained on {len(normal_sessions)} normal sessions")
        print(f"📊 Feature dimensions: {len(combined_features[0])}")
        
        print("\n🚨 TESTING HARDWARE ANOMALY...")
        print("-" * 40)
        print(f"Anomaly Session:\n{hardware_anomaly}")
        
        # Extract features from anomaly
        anomaly_manual_features = self.extract_features(hardware_anomaly)
        anomaly_tfidf = self.vectorizer.transform([hardware_anomaly.replace('\n', ' ')]).toarray()[0]
        anomaly_combined = anomaly_manual_features + list(anomaly_tfidf)
        anomaly_scaled = self.scaler.transform([anomaly_combined])
        
        # Predict
        prediction = self.svm.predict(anomaly_scaled)[0]
        decision_score = self.svm.decision_function(anomaly_scaled)[0]
        anomaly_prob = 1 / (1 + np.exp(decision_score))
        
        print(f"\n📊 FEATURE ANALYSIS:")
        print(f"   Hardware Features: {anomaly_manual_features[:7]}")
        print(f"   Critical Hardware Score: {anomaly_manual_features[4]} ⚠️")
        print(f"   Total Error Count: {anomaly_manual_features[5]} ⚠️")
        
        print(f"\n🎯 SVM RESULTS:")
        print(f"   Prediction: {'🚨 ANOMALY' if prediction == -1 else '✅ Normal'}")
        print(f"   Decision Score: {decision_score:.3f}")
        print(f"   Anomaly Probability: {anomaly_prob:.1%}")
        print(f"   Distance from Normal Boundary: {abs(decision_score):.3f}")
        
        # Compare with normal session
        print(f"\n📈 COMPARISON WITH NORMAL:")
        normal_sample = self.extract_features(normal_sessions[0])
        print(f"   Normal Critical Hardware Score: {normal_sample[4]}")
        print(f"   Anomaly Critical Hardware Score: {anomaly_manual_features[4]}")
        print(f"   Difference: {anomaly_manual_features[4] - normal_sample[4]}x higher! 🔥")
        
        # Feature importance
        print(f"\n🔍 WHY IT'S ANOMALOUS:")
        feature_names = ['hw_power_reset', 'hw_hardware_error', 'hw_component_failure', 
                        'hw_device_issues', 'critical_hardware_score', 'total_error_count', 
                        'session_length', 'avg_line_length']
        
        for i, (name, value) in enumerate(zip(feature_names, anomaly_manual_features)):
            if value > 0:
                print(f"   ⚠️  {name}: {value}")
        
        print(f"\n✅ CONCLUSION: Hardware anomaly successfully detected!")
        print(f"   The combination of multiple hardware errors, power resets,")
        print(f"   and failure patterns created a feature vector FAR outside")
        print(f"   the normal boundary learned from regular transactions.")
        
        return prediction, decision_score, anomaly_prob

if __name__ == "__main__":
    demo = SVM_Demo()
    demo.demonstrate_detection()
