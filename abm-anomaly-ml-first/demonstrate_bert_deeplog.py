#!/usr/bin/env python3
"""
BERT-DeepLog Demonstration Script
================================

This script demonstrates the BERT-DeepLog anomaly detection system
with sample EJ log data and real-time prediction capabilities.

Usage:
    python demonstrate_bert_deeplog.py
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from bert_deeplog_trainer import BERTDeepLogTrainer, BERTDeepLogConfig, EJLogProcessor
except ImportError as e:
    print(f"Error importing BERT-DeepLog modules: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install -r bert_deeplog_requirements.txt")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BERTDeepLogDemo:
    """Demonstration class for BERT-DeepLog system"""
    
    def __init__(self):
        self.config = BERTDeepLogConfig()
        self.trainer = BERTDeepLogTrainer(self.config)
        self.model_path = "./models/bert_deeplog_demo.pth"
        
    def create_sample_data(self) -> tuple:
        """Create sample EJ log data for demonstration"""
        
        # Normal EJ log samples
        normal_logs = [
            """TRANSACTION START
            ATM ID: ATM001
            SESSION: sess_000200
            TIMESTAMP: 2025-01-15 10:30:45
            CARD INSERTED: ****1234
            PIN VERIFICATION: SUCCESS
            ACCOUNT BALANCE: $1,250.00
            WITHDRAWAL REQUEST: $100.00
            CASH DISPENSED: $100.00
            RECEIPT PRINTED: YES
            TRANSACTION COMPLETE""",
            
            """TRANSACTION START
            ATM ID: ATM002
            SESSION: sess_000201
            TIMESTAMP: 2025-01-15 10:35:12
            CARD INSERTED: ****5678
            PIN VERIFICATION: SUCCESS
            ACCOUNT BALANCE: $2,850.00
            BALANCE INQUIRY: SUCCESS
            RECEIPT PRINTED: NO
            TRANSACTION COMPLETE""",
            
            """TRANSACTION START
            ATM ID: ATM003
            SESSION: sess_000202
            TIMESTAMP: 2025-01-15 10:40:33
            CARD INSERTED: ****9012
            PIN VERIFICATION: SUCCESS
            ACCOUNT BALANCE: $450.00
            DEPOSIT: $200.00
            NEW BALANCE: $650.00
            RECEIPT PRINTED: YES
            TRANSACTION COMPLETE""",
            
            """TRANSACTION START
            ATM ID: ATM001
            SESSION: sess_000203
            TIMESTAMP: 2025-01-15 10:45:22
            CARD INSERTED: ****3456
            PIN VERIFICATION: SUCCESS
            ACCOUNT BALANCE: $3,200.00
            TRANSFER TO SAVINGS: $500.00
            CONFIRMATION: SUCCESS
            RECEIPT PRINTED: YES
            TRANSACTION COMPLETE""",
            
            """TRANSACTION START
            ATM ID: ATM004
            SESSION: sess_000204
            TIMESTAMP: 2025-01-15 10:50:11
            CARD INSERTED: ****7890
            PIN VERIFICATION: SUCCESS
            ACCOUNT BALANCE: $1,750.00
            WITHDRAWAL REQUEST: $300.00
            CASH DISPENSED: $300.00
            RECEIPT PRINTED: YES
            TRANSACTION COMPLETE"""
        ]
        
        # Anomalous EJ log samples
        anomalous_logs = [
            """TRANSACTION START
            ATM ID: ATM001
            SESSION: sess_000205
            TIMESTAMP: 2025-01-15 10:55:44
            CARD INSERTED: ****1111
            PIN VERIFICATION: FAILED
            PIN VERIFICATION: FAILED
            PIN VERIFICATION: FAILED
            CARD RETAINED: YES
            ERROR: SUSPICIOUS ACTIVITY DETECTED
            SECURITY ALERT TRIGGERED
            TRANSACTION TERMINATED""",
            
            """TRANSACTION START
            ATM ID: ATM002
            SESSION: sess_000206
            TIMESTAMP: 2025-01-15 11:00:15
            CARD INSERTED: ****2222
            PIN VERIFICATION: SUCCESS
            ACCOUNT BALANCE: $50.00
            WITHDRAWAL REQUEST: $10,000.00
            ERROR: INSUFFICIENT FUNDS
            DISPENSER ERROR: JAM DETECTED
            MAINTENANCE REQUIRED
            TRANSACTION FAILED""",
            
            """TRANSACTION START
            ATM ID: ATM003
            SESSION: sess_000207
            TIMESTAMP: 2025-01-15 11:05:33
            SYSTEM ERROR: CONNECTION TIMEOUT
            DATABASE ERROR: UNABLE TO CONNECT
            NETWORK ERROR: HOST UNREACHABLE
            CRITICAL ERROR: SYSTEM FAILURE
            EMERGENCY SHUTDOWN INITIATED
            TRANSACTION ABORTED""",
            
            """TRANSACTION START
            ATM ID: ATM004
            SESSION: sess_000208
            TIMESTAMP: 2025-01-15 11:10:22
            CARD INSERTED: ****9999
            PIN VERIFICATION: SUCCESS
            ACCOUNT BALANCE: ERROR
            WITHDRAWAL REQUEST: $500.00
            CASH DISPENSED: $0.00
            ERROR: DISPENSER MALFUNCTION
            TRANSACTION FAILED""",
            
            """TRANSACTION START
            ATM ID: ATM001
            SESSION: sess_000209
            TIMESTAMP: 2025-01-15 11:15:11
            CARD INSERTED: ****0000
            PIN VERIFICATION: BYPASS DETECTED
            UNAUTHORIZED ACCESS ATTEMPT
            SECURITY BREACH: ALERT LEVEL 5
            EMERGENCY PROTOCOLS ACTIVATED
            LAW ENFORCEMENT NOTIFIED
            TRANSACTION BLOCKED"""
        ]
        
        # Combine data
        all_logs = normal_logs + anomalous_logs
        labels = [0] * len(normal_logs) + [1] * len(anomalous_logs)
        
        # Shuffle data
        indices = np.arange(len(all_logs))
        np.random.shuffle(indices)
        
        shuffled_logs = [all_logs[i] for i in indices]
        shuffled_labels = [labels[i] for i in indices]
        
        return shuffled_logs, shuffled_labels
    
    def run_training_demo(self):
        """Demonstrate the training process"""
        logger.info("=== BERT-DeepLog Training Demonstration ===")
        
        # Create sample data
        logs, labels = self.create_sample_data()
        logger.info(f"Created {len(logs)} sample log entries")
        logger.info(f"Normal transactions: {labels.count(0)}")
        logger.info(f"Anomalous transactions: {labels.count(1)}")
        
        # Split data
        split_idx = int(0.8 * len(logs))
        train_logs, val_logs = logs[:split_idx], logs[split_idx:]
        train_labels, val_labels = labels[:split_idx], labels[split_idx:]
        
        logger.info(f"Training on {len(train_logs)} samples")
        logger.info(f"Validating on {len(val_logs)} samples")
        
        # Train model
        try:
            os.makedirs("./models", exist_ok=True)
            self.trainer.train(train_logs, train_labels, val_logs, val_labels)
            self.trainer.save_model(self.model_path)
            logger.info("Training completed successfully!")
            return True
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def run_prediction_demo(self):
        """Demonstrate real-time prediction"""
        logger.info("=== BERT-DeepLog Prediction Demonstration ===")
        
        # Check if model exists
        if not os.path.exists(self.model_path):
            logger.warning("No trained model found. Running training first...")
            if not self.run_training_demo():
                logger.error("Cannot proceed without trained model")
                return
        
        # Load model
        try:
            self.trainer.load_model(self.model_path)
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return
        
        # Test samples for prediction
        test_samples = [
            {
                "description": "Normal withdrawal transaction",
                "log": """TRANSACTION START
                ATM ID: ATM005
                SESSION: sess_000250
                TIMESTAMP: 2025-01-15 14:30:45
                CARD INSERTED: ****4567
                PIN VERIFICATION: SUCCESS
                ACCOUNT BALANCE: $800.00
                WITHDRAWAL REQUEST: $60.00
                CASH DISPENSED: $60.00
                RECEIPT PRINTED: YES
                TRANSACTION COMPLETE"""
            },
            {
                "description": "Suspicious multiple PIN failures",
                "log": """TRANSACTION START
                ATM ID: ATM005
                SESSION: sess_000251
                TIMESTAMP: 2025-01-15 14:35:22
                CARD INSERTED: ****8888
                PIN VERIFICATION: FAILED
                PIN VERIFICATION: FAILED
                PIN VERIFICATION: FAILED
                CARD RETAINED: YES
                SECURITY ALERT: MULTIPLE FAILURES
                TRANSACTION TERMINATED"""
            },
            {
                "description": "System error with connection issues",
                "log": """TRANSACTION START
                ATM ID: ATM005
                SESSION: sess_000252
                TIMESTAMP: 2025-01-15 14:40:11
                NETWORK ERROR: CONNECTION LOST
                DATABASE ERROR: TIMEOUT
                SYSTEM ERROR: CRITICAL FAILURE
                EMERGENCY SHUTDOWN
                TRANSACTION ABORTED"""
            },
            {
                "description": "Normal balance inquiry",
                "log": """TRANSACTION START
                ATM ID: ATM005
                SESSION: sess_000253
                TIMESTAMP: 2025-01-15 14:45:33
                CARD INSERTED: ****1357
                PIN VERIFICATION: SUCCESS
                ACCOUNT BALANCE: $1,200.00
                BALANCE INQUIRY: COMPLETE
                RECEIPT PRINTED: NO
                TRANSACTION COMPLETE"""
            }
        ]
        
        # Run predictions
        logger.info("Running predictions on test samples...")
        print("\n" + "="*80)
        
        for i, sample in enumerate(test_samples, 1):
            print(f"\nTest Sample {i}: {sample['description']}")
            print("-" * 60)
            
            # Get prediction
            scores, predictions = self.trainer.predict([sample['log']])
            
            anomaly_score = scores[0]
            is_anomaly = predictions[0]
            confidence = anomaly_score if is_anomaly else (1 - anomaly_score)
            
            # Display results
            print(f"Anomaly Score: {anomaly_score:.4f}")
            print(f"Prediction: {'🚨 ANOMALY DETECTED' if is_anomaly else '✅ Normal Transaction'}")
            print(f"Confidence: {confidence:.2%}")
            
            # Risk assessment
            if anomaly_score > 0.8:
                risk_level = "🔴 HIGH RISK"
            elif anomaly_score > 0.6:
                risk_level = "🟡 MEDIUM RISK"
            elif anomaly_score > 0.4:
                risk_level = "🟠 LOW RISK"
            else:
                risk_level = "🟢 NORMAL"
                
            print(f"Risk Level: {risk_level}")
            
            # Sample log preview
            log_preview = sample['log'][:100].replace('\n', ' ').strip() + "..."
            print(f"Log Preview: {log_preview}")
    
    def run_batch_analysis_demo(self):
        """Demonstrate batch analysis capabilities"""
        logger.info("=== Batch Analysis Demonstration ===")
        
        # Check if model exists
        if not os.path.exists(self.model_path):
            logger.warning("No trained model found. Running training first...")
            if not self.run_training_demo():
                logger.error("Cannot proceed without trained model")
                return
        
        # Load model
        self.trainer.load_model(self.model_path)
        
        # Create a batch of mixed logs
        batch_logs, true_labels = self.create_sample_data()
        
        # Get predictions
        scores, predictions = self.trainer.predict(batch_logs)
        
        # Analyze results
        true_positives = sum(1 for true, pred in zip(true_labels, predictions) if true == 1 and pred == 1)
        false_positives = sum(1 for true, pred in zip(true_labels, predictions) if true == 0 and pred == 1)
        true_negatives = sum(1 for true, pred in zip(true_labels, predictions) if true == 0 and pred == 0)
        false_negatives = sum(1 for true, pred in zip(true_labels, predictions) if true == 1 and pred == 0)
        
        accuracy = (true_positives + true_negatives) / len(true_labels)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print("\n" + "="*60)
        print("BATCH ANALYSIS RESULTS")
        print("="*60)
        print(f"Total Samples: {len(batch_logs)}")
        print(f"True Anomalies: {sum(true_labels)}")
        print(f"Predicted Anomalies: {sum(predictions)}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall: {recall:.2%}")
        print(f"F1-Score: {f1_score:.2%}")
        print("-" * 60)
        print(f"True Positives: {true_positives}")
        print(f"False Positives: {false_positives}")
        print(f"True Negatives: {true_negatives}")
        print(f"False Negatives: {false_negatives}")
        
        # Show some example predictions
        print("\nSample Predictions:")
        for i in range(min(3, len(batch_logs))):
            score = scores[i]
            pred = predictions[i]
            true_label = true_labels[i]
            status = "✅ Correct" if pred == true_label else "❌ Incorrect"
            
            print(f"\nSample {i+1}: {status}")
            print(f"  True Label: {'Anomaly' if true_label else 'Normal'}")
            print(f"  Predicted: {'Anomaly' if pred else 'Normal'}")
            print(f"  Score: {score:.4f}")
    
    def run_full_demo(self):
        """Run the complete demonstration"""
        print("🤖 BERT-DeepLog Anomaly Detection System")
        print("=" * 80)
        print("This demonstration shows how BERT embeddings can be combined")
        print("with DeepLog LSTM models for EJ log anomaly detection.")
        print("=" * 80)
        
        try:
            # Run training demo
            self.run_training_demo()
            print("\n")
            
            # Run prediction demo
            self.run_prediction_demo()
            print("\n")
            
            # Run batch analysis demo
            self.run_batch_analysis_demo()
            
            print("\n" + "="*80)
            print("✅ BERT-DeepLog demonstration completed successfully!")
            print("The model can now be used for real-time anomaly detection.")
            print("Model saved at:", self.model_path)
            print("="*80)
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            print("\n❌ Demo failed. Please check the logs for details.")

def main():
    """Main function to run the demonstration"""
    demo = BERTDeepLogDemo()
    demo.run_full_demo()

if __name__ == "__main__":
    main()
