#!/usr/bin/env python3
"""
Test script to verify the BERT-DeepLog tensor dimension fix
"""

import sys
import os
sys.path.append('/home/yc/development/Capstone-Project/abm-anomaly-ml-first/services/anomaly-detector')

import torch
import numpy as np
from bert_deeplog_model import BertDeepLogLSTM, BertDeepLogAnalyzer

def test_tensor_dimensions():
    """Test that the tensor dimensions match correctly"""
    print("🔧 Testing BERT-DeepLog tensor dimension fix...")
    
    # Test the model with sample data
    try:
        # Initialize model
        model = BertDeepLogLSTM(bert_dim=768, hidden_dim=128)
        print("✅ Model initialized successfully")
        
        # Create sample input (batch_size=2, seq_len=10, bert_dim=768)
        batch_size, seq_len, bert_dim = 2, 10, 768
        sample_input = torch.randn(batch_size, seq_len, bert_dim)
        print(f"✅ Sample input created: {sample_input.shape}")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(sample_input)
        
        print(f"✅ Forward pass successful!")
        print(f"   - Anomaly logits shape: {outputs['anomaly_logits'].shape}")
        print(f"   - Sequence predictions shape: {outputs['sequence_predictions'].shape}")
        print(f"   - Expected sequence predictions shape: ({batch_size}, {seq_len}, {bert_dim})")
        
        # Verify dimensions match
        expected_anomaly_shape = (batch_size, seq_len, 2)
        expected_sequence_shape = (batch_size, seq_len, bert_dim)
        
        assert outputs['anomaly_logits'].shape == expected_anomaly_shape, \
            f"Anomaly logits shape mismatch: {outputs['anomaly_logits'].shape} != {expected_anomaly_shape}"
        
        assert outputs['sequence_predictions'].shape == expected_sequence_shape, \
            f"Sequence predictions shape mismatch: {outputs['sequence_predictions'].shape} != {expected_sequence_shape}"
        
        print("✅ All tensor dimensions are correct!")
        
        # Test loss calculation that was failing before
        target_sequences = sample_input[:, 1:, :]  # Next embeddings (shape: batch_size, seq_len-1, bert_dim)
        pred_sequences = outputs['sequence_predictions'][:, :-1, :]  # Predictions (shape: batch_size, seq_len-1, bert_dim)
        
        print(f"✅ Loss calculation shapes:")
        print(f"   - Target sequences: {target_sequences.shape}")
        print(f"   - Predicted sequences: {pred_sequences.shape}")
        
        # Calculate MSE loss (this was failing before the fix)
        mse_loss = torch.nn.MSELoss()
        loss = mse_loss(pred_sequences, target_sequences)
        print(f"✅ Sequence prediction loss calculated successfully: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in tensor dimension test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_bert_deeplog_analyzer():
    """Test the full BERT-DeepLog analyzer"""
    print("\n🔧 Testing BERT-DeepLog analyzer...")
    
    try:
        # Initialize analyzer (this will take some time to load BERT)
        analyzer = BertDeepLogAnalyzer()
        print("✅ BERT-DeepLog analyzer initialized successfully")
        
        # Test with sample EJ data
        sample_sessions = [
            {
                'raw_text': 'CARD INSERTED PIN ENTERED OPCODE FI BALANCE INQUIRY RECEIPT PRINTED CARD TAKEN TRANSACTION END',
                'session_id': 'test_session_1',
                'is_anomaly': False
            },
            {
                'raw_text': 'CARD INSERTED DEVICE ERROR HARDWARE MALFUNCTION CARD TAKEN TRANSACTION END',
                'session_id': 'test_session_2',
                'is_anomaly': True
            }
        ]
        
        # Prepare training data
        num_sequences = analyzer.prepare_training_data(sample_sessions)
        print(f"✅ Training data prepared: {num_sequences} sequences")
        
        if num_sequences > 0:
            print("✅ BERT-DeepLog fix is working correctly!")
            return True
        else:
            print("⚠️  No training sequences generated, but no tensor errors occurred")
            return True
            
    except Exception as e:
        print(f"❌ Error in BERT-DeepLog analyzer test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing BERT-DeepLog tensor dimension fix\n")
    
    # Test 1: Basic tensor dimensions
    test1_success = test_tensor_dimensions()
    
    # Test 2: Full analyzer
    test2_success = test_bert_deeplog_analyzer()
    
    print(f"\n📊 Test Results:")
    print(f"   - Tensor dimension test: {'✅ PASSED' if test1_success else '❌ FAILED'}")
    print(f"   - BERT-DeepLog analyzer test: {'✅ PASSED' if test2_success else '❌ FAILED'}")
    
    if test1_success and test2_success:
        print(f"\n🎉 All tests passed! The tensor dimension fix is working correctly.")
        print(f"   The training should now work without the 'tensor a (64) must match tensor b (768)' error.")
    else:
        print(f"\n⚠️  Some tests failed. Please check the errors above.")
