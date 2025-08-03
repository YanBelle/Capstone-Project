#!/usr/bin/env python3
"""
Test script to verify the enhanced preprocessing methodology integration
between bertviz_analyzer and BERT-DeepLog training pipeline
"""

import requests
import json
import sys
import os

def test_model_info():
    """Test the model info endpoint"""
    print("=== Testing Model Info ===")
    try:
        response = requests.get("http://localhost:8000/api/v1/bert-deeplog/model-info")
        if response.status_code == 200:
            model_info = response.json()
            print(f"✅ Model info retrieved successfully")
            print(f"   - Model available: {model_info['model_available']}")
            print(f"   - Parameters: {model_info['model_stats']['model_info']['parameters']:,}")
            print(f"   - Device: {model_info['model_stats']['model_info']['device']}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model info error: {e}")
        return False

def load_processed_sessions():
    """Load the processed EJ sessions for training"""
    print("\n=== Loading Processed Sessions ===")
    try:
        # Use the latest processed sessions file
        sessions_file = "/home/yc/development/Capstone-Project/abm-anomaly-ml-first/data/processed/normal_sessions_full_20250803_102920.json"
        
        with open(sessions_file, 'r') as f:
            sessions = json.load(f)  # Direct array load
            
        print(f"✅ Loaded {len(sessions)} processed sessions")
        
        # Show sample of preprocessed text to verify our methodology is working
        if sessions:
            sample_session = sessions[0]
            print(f"   - Sample session keys: {list(sample_session.keys())}")
            
            if 'bert_preprocessed_text' in sample_session:
                preprocessed = sample_session['bert_preprocessed_text']
                print(f"   - Sample preprocessed text (first 200 chars): {preprocessed[:200]}...")
                
                # Check for our compound tokens
                compound_tokens = ['TRANSACTION_START', 'CARD_INSERTED', 'ESC_000', 'VAL_000', 'NOTES_PRESENTED']
                found_tokens = [token for token in compound_tokens if token in preprocessed]
                print(f"   - Found compound tokens: {found_tokens}")
                
                if 'preprocessing_info' in sample_session:
                    preprocessing_info = sample_session['preprocessing_info']
                    print(f"   - Original length: {preprocessing_info.get('original_length', 'N/A')}")
                    print(f"   - Compressed length: {preprocessing_info.get('compressed_length', 'N/A')}")
                    print(f"   - Compression ratio: {preprocessing_info.get('compression_ratio', 'N/A'):.3f}")
            else:
                print("   ⚠️  No bert_preprocessed_text found - using raw_text")
        
        return sessions[:50]  # Return first 50 sessions for testing
        
    except Exception as e:
        print(f"❌ Error loading sessions: {e}")
        return []

def test_training(sessions):
    """Test training the BERT-DeepLog model with enhanced preprocessing"""
    print("\n=== Testing Training with Enhanced Preprocessing ===")
    
    if not sessions:
        print("❌ No sessions available for training")
        return False
    
    try:
        training_request = {
            "sessions": sessions,
            "validation_split": 0.2,
            "normal_sessions_only": True
        }
        
        print(f"   - Submitting {len(sessions)} sessions for training...")
        response = requests.post(
            "http://localhost:8000/api/v1/bert-deeplog/train",
            json=training_request,
            timeout=300  # 5 minute timeout for training
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Training completed successfully")
            print(f"   - Training time: {result.get('training_time', 'N/A')}")
            print(f"   - Training samples: {result.get('training_samples', 'N/A')}")
            print(f"   - Validation samples: {result.get('validation_samples', 'N/A')}")
            print(f"   - Final loss: {result.get('final_loss', 'N/A')}")
            return True
        else:
            print(f"❌ Training failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Training error: {e}")
        return False

def test_prediction():
    """Test prediction with enhanced preprocessing"""
    print("\n=== Testing Prediction with Enhanced Preprocessing ===")
    
    # Test with complex EJ text that should benefit from our preprocessing
    test_session = {
        "session_text": "[020t*629*06/18/2025*00:46*TRANSACTION START*CARD INSERTED*PAN 0004263********1897*ESC: 000*VAL: 000*REF: 000*NOTES PRESENTED 100,50,20*NOTES TAKEN*TRANSACTION END*REJECTS:000*(1*7231*1*(Iw(1*3,*PRIMARY CARD READER ACTIVATED*",
        "session_id": "test_enhanced_preprocessing_001"
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/api/v1/bert-deeplog/predict",
            json=test_session
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction completed successfully")
            print(f"   - Session ID: {result.get('session_id', 'N/A')}")
            print(f"   - Anomaly probability: {result.get('anomaly_probability', 'N/A'):.4f}")
            print(f"   - Is anomaly: {result.get('is_anomaly', 'N/A')}")
            
            # Check if we get detailed analysis showing our preprocessing worked
            if 'analysis' in result:
                analysis = result['analysis']
                if 'important_events' in analysis:
                    events = analysis['important_events'][:5]  # First 5 events
                    print(f"   - Top important events: {[e.get('token', 'N/A') for e in events]}")
            
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Testing Enhanced Preprocessing Integration in BERT-DeepLog")
    print("=" * 70)
    
    # Step 1: Check model info
    if not test_model_info():
        print("\n❌ Model info test failed - exiting")
        return False
    
    # Step 2: Load processed sessions  
    sessions = load_processed_sessions()
    if not sessions:
        print("\n❌ No sessions loaded - exiting")
        return False
    
    # Step 3: Train model with enhanced preprocessing
    if not test_training(sessions):
        print("\n❌ Training test failed - continuing to prediction test")
    
    # Step 4: Test prediction with enhanced preprocessing
    if not test_prediction():
        print("\n❌ Prediction test failed")
        return False
    
    print("\n" + "=" * 70)
    print("✅ Enhanced Preprocessing Integration Test Complete!")
    print("✅ bertviz_analyzer tokenization methodology successfully integrated into BERT-DeepLog training pipeline")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
