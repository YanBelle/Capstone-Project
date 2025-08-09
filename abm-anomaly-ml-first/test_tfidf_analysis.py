#!/usr/bin/env python3
"""
Test script for TF-IDF analysis with One-Class SVM
"""

import requests
import json
import sys

# API endpoint
API_URL = "http://localhost:8000"

# Test sessions
test_sessions = {
    "power_reset_anomaly": """[020t15706/18/202513:39
TRANSACTION START
[020t CARD INSERTED
POWER-UP/RESET OCCURRED
HARDWARE ERROR DETECTED
RECOVERY FAILED
[020t 13:39:56 CARD TAKEN
[020t 13:39:56 TRANSACTION END""",
    
    "incomplete_transaction": """[020t*209*06/18/2025*14:23*
TRANSACTION START
[020t CARD INSERTED
14:23:03 ATR RECEIVED T=0
[020t 14:23:06 OPCODE = FI
PIN ENTERED
DEVICE MALFUNCTION
[020t CARD TAKEN
[020t TRANSACTION END""",
    
    "normal_transaction": """[020t*209*06/18/2025*14:23*
TRANSACTION START
[020t CARD INSERTED
14:23:03 ATR RECEIVED T=0
[020t 14:23:06 OPCODE = FI
PAN 0004263********6687
PIN ENTERED
[020t 14:23:36 OPCODE = BC
CASH DISPENSED SUCCESSFULLY
[020t 14:24:28 CARD TAKEN
[020t 14:24:29 TRANSACTION END"""
}

def test_enhanced_ensemble_training():
    """Test training the enhanced ensemble detector"""
    print("🚀 Testing Enhanced Ensemble Training...")
    
    # Create training data
    training_sessions = []
    for session_id, text in test_sessions.items():
        training_sessions.append({
            'session_id': session_id,
            'raw_text': text,
            'transactions': []
        })
    
    # Train the model
    try:
        response = requests.post(
            f"{API_URL}/api/train_enhanced_ensemble",
            json={'sessions': training_sessions},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Training successful!")
            print(f"   Training stats: {json.dumps(result.get('training_stats', {}), indent=2)}")
            return True
        else:
            print(f"❌ Training failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Training error: {e}")
        return False

def test_tfidf_analysis():
    """Test TF-IDF analysis for each session"""
    print("\n🔍 Testing TF-IDF Analysis...")
    
    for session_name, session_text in test_sessions.items():
        print(f"\n📊 Analyzing: {session_name}")
        
        try:
            response = requests.post(
                f"{API_URL}/api/v1/svm-tfidf/analyze-session",
                json={
                    'session_id': session_name,
                    'raw_text': session_text
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                prediction = result.get('prediction_result', {})
                tfidf_analysis = result.get('tfidf_analysis', [])
                word_categories = result.get('word_categories', {})
                
                print(f"   Prediction: {'ANOMALY' if prediction.get('is_anomaly') else 'NORMAL'}")
                print(f"   Ensemble Score: {prediction.get('ensemble_score', 0):.3f}")
                print(f"   Top TF-IDF words: {len(tfidf_analysis)}")
                
                if tfidf_analysis:
                    print("   Top 5 words:")
                    for i, word_data in enumerate(tfidf_analysis[:5]):
                        print(f"     {i+1}. {word_data['word']} (score: {word_data['tfidf_score']:.4f})")
                
                if word_categories:
                    print("   Word categories:")
                    for category, words in word_categories.items():
                        if words:
                            print(f"     {category}: {len(words)} words")
                
            else:
                print(f"   ❌ Analysis failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"   ❌ Analysis error: {e}")

def test_vocabulary():
    """Test vocabulary retrieval"""
    print("\n📚 Testing Vocabulary Retrieval...")
    
    try:
        response = requests.get(f"{API_URL}/api/v1/svm-tfidf/vocabulary", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Vocabulary retrieved successfully!")
            print(f"   Vocabulary size: {result.get('vocabulary_size', 0)}")
            print(f"   Model trained: {result.get('model_trained', False)}")
            print(f"   Max features: {result.get('feature_extraction_config', {}).get('max_features', 'N/A')}")
            
            top_words = result.get('top_100_words', [])[:10]
            if top_words:
                print(f"   Sample words: {', '.join(top_words)}")
                
        else:
            print(f"❌ Vocabulary retrieval failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Vocabulary error: {e}")

def main():
    """Run all tests"""
    print("🧪 TF-IDF Analysis Test Suite")
    print("="*50)
    
    # Test 1: Train the enhanced ensemble
    training_success = test_enhanced_ensemble_training()
    
    if not training_success:
        print("\n❌ Training failed. Cannot proceed with analysis tests.")
        return
    
    # Test 2: Test TF-IDF analysis
    test_tfidf_analysis()
    
    # Test 3: Test vocabulary
    test_vocabulary()
    
    print("\n✅ All tests completed!")
    print("\n🎯 Next steps:")
    print("   1. Open the dashboard at http://localhost:3000")
    print("   2. Navigate to 'TF-IDF Analysis' tab")
    print("   3. Try analyzing the sample sessions")

if __name__ == "__main__":
    main()
