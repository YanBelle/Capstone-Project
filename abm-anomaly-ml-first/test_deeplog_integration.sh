#!/bin/bash

# DeepLog + BERT Training and Testing Script
# Tests the new DeepLog integration with your ABM anomaly detection system

echo "🚀 Testing DeepLog + BERT Integration for ABM Anomaly Detection"
echo "=============================================================="

# Test 1: Train DeepLog model on existing sessions
echo "📊 Test 1: Training DeepLog model..."
python3 -c "
import sys
sys.path.append('/app')
from deeplog_service_integration import DeepLogServiceIntegration

try:
    service = DeepLogServiceIntegration()
    
    # Get sample sessions from database
    sessions = service.get_training_sessions(limit=50)
    print(f'Retrieved {len(sessions)} sessions for training')
    
    if len(sessions) >= 5:
        results = service.train_deeplog_model()
        print(f'Training completed successfully!')
        print(f'Vocab size: {results.get(\"vocab_size\", \"N/A\")}')
        print(f'Number of sequences: {results.get(\"num_sequences\", \"N/A\")}')
    else:
        print('Insufficient training data, using sample data...')
        # Use sample data for demonstration
        sample_sessions = [
            'TRANSACTION START CARD INSERTED PIN ENTERED BALANCE INQUIRY CARD TAKEN TRANSACTION END',
            'TRANSACTION START CARD INSERTED CARD TAKEN TRANSACTION END',
            'TRANSACTION START CARD INSERTED PIN ENTERED CASH WITHDRAWAL CASH DISPENSED CARD TAKEN TRANSACTION END'
        ]
        results = service.trainer.train_on_sessions(sample_sessions)
        print(f'Training completed on sample data!')
        
except Exception as e:
    print(f'Training error: {e}')
"

echo ""

# Test 2: Test anomaly detection on your specific examples
echo "🔍 Test 2: Testing anomaly detection on your transaction examples..."
python3 -c "
import sys
sys.path.append('/app')
from deeplog_service_integration import test_transaction_examples

try:
    test_transaction_examples()
except Exception as e:
    print(f'Testing error: {e}')
"

echo ""

# Test 3: Integration with existing ML system
echo "🔗 Test 3: Integration test with existing ML system..."
python3 -c "
import sys
sys.path.append('/app')
sys.path.append('/app/shared')

try:
    from deeplog_service_integration import DeepLogServiceIntegration
    
    service = DeepLogServiceIntegration()
    
    # Test transaction examples
    test_sessions = [
        # Should be detected as anomaly (immediate card removal)
        '''TRANSACTION START
        CARD INSERTED
        CARD TAKEN
        TRANSACTION END
        PRIMARY CARD READER ACTIVATED''',
        
        # Should be detected as anomaly (incomplete transaction)
        '''TRANSACTION START
        CARD INSERTED
        ATR RECEIVED T=0
        OPCODE = FI
        PAN 0004263********6687
        START OF TRANSACTION
        PIN ENTERED
        CARD TAKEN
        TRANSACTION END''',
        
        # Should be normal
        '''TRANSACTION START
        CARD INSERTED
        ATR RECEIVED T=0
        PIN ENTERED
        BALANCE INQUIRY SELECTED
        BALANCE DISPLAYED \$1,250.00
        CARD TAKEN
        TRANSACTION END'''
    ]
    
    predictions = service.predict_session_anomalies(test_sessions)
    
    print('DeepLog Predictions:')
    for i, pred in enumerate(predictions):
        print(f'Session {i+1}:')
        print(f'  Is Anomaly: {pred.get(\"is_anomaly\", False)}')
        print(f'  Anomaly Score: {pred.get(\"anomaly_score\", 0.0):.4f}')
        print(f'  Model Type: {pred.get(\"model_type\", \"unknown\")}')
        print()
        
except Exception as e:
    print(f'Integration test error: {e}')
"

echo ""
echo "✅ DeepLog + BERT Integration Testing Complete!"
echo ""
echo "📋 Summary:"
echo "- DeepLog model combines BERT embeddings with LSTM sequence modeling"
echo "- Detects anomalies based on sequence prediction errors"
echo "- Integrates with existing ABM ML system database"
echo "- Specifically designed to catch transaction pattern anomalies"
echo ""
echo "🎯 Your Transaction Examples:"
echo "- Transaction 1 (immediate card removal) → Should be detected as anomaly"
echo "- Transaction 2 (incomplete after PIN) → Should be detected as anomaly"
echo "- Normal transactions → Should be classified as normal"
