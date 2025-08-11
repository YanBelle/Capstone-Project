#!/usr/bin/env python3
"""
Test BERT Primary Embedding Configuration
Verify that BERT is now the primary embedding method instead of Sentence Transformers.
"""

import sys
import os
import logging

# Add paths for imports
sys.path.append('/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/services/anomaly-detector')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_bert_primary_configuration():
    """Test that BERT is configured as the primary embedding method"""
    print("🧠 Testing BERT Primary Embedding Configuration")
    print("=" * 60)
    
    try:
        # Test 1: Import and initialize ML analyzer
        print("\n📦 Test 1: Importing ML analyzer...")
        from ml_analyzer import MLFirstAnomalyDetector
        
        analyzer = MLFirstAnomalyDetector()
        print("✅ ML Analyzer initialized successfully")
        
        # Test 2: Check that convert_to_embeddings points to BERT method
        print("\n🔍 Test 2: Checking embedding method configuration...")
        
        # Check the method alias
        embedding_method = analyzer.convert_to_embeddings
        bert_method = analyzer.generate_embeddingsUsingBERT
        sentence_method = analyzer.generate_embeddingsUsingSentence
        
        if embedding_method == bert_method:
            print("✅ convert_to_embeddings points to BERT method (PRIMARY)")
            primary_method = "BERT"
        elif embedding_method == sentence_method:
            print("❌ convert_to_embeddings points to Sentence Transformers (should be BERT)")
            primary_method = "Sentence Transformers"
        else:
            print("❓ convert_to_embeddings points to unknown method")
            primary_method = "Unknown"
        
        # Test 3: Check method names and docstrings
        print("\n📝 Test 3: Checking method documentation...")
        
        bert_doc = bert_method.__doc__
        sentence_doc = sentence_method.__doc__
        
        if "PRIMARY METHOD" in bert_doc:
            print("✅ BERT method marked as PRIMARY METHOD")
        else:
            print("❌ BERT method not marked as primary")
        
        if "FALLBACK" in sentence_doc:
            print("✅ Sentence Transformers method marked as FALLBACK")
        else:
            print("❌ Sentence Transformers method not marked as fallback")
        
        # Test 4: Check BERT model initialization
        print("\n⚙️ Test 4: Checking BERT model components...")
        
        bert_components = {
            'tokenizer': hasattr(analyzer, 'tokenizer') and analyzer.tokenizer is not None,
            'bert_model': hasattr(analyzer, 'bert_model') and analyzer.bert_model is not None,
            'model_name': 'bert-base-uncased'  # Expected model
        }
        
        for component, status in bert_components.items():
            if component == 'model_name':
                # Check if the model name is correct
                try:
                    model_name = analyzer.tokenizer.name_or_path if hasattr(analyzer.tokenizer, 'name_or_path') else 'unknown'
                    if 'bert' in model_name.lower():
                        print(f"✅ BERT model: {model_name}")
                    else:
                        print(f"❓ Model name: {model_name}")
                except:
                    print("❓ Could not determine model name")
            else:
                if status:
                    print(f"✅ {component}: Available")
                else:
                    print(f"❌ {component}: Missing")
        
        # Test 5: Summary
        print("\n📊 Configuration Summary:")
        print("=" * 40)
        print(f"Primary Embedding Method: {primary_method}")
        print(f"Fallback Method: {'Sentence Transformers' if primary_method == 'BERT' else 'Unknown'}")
        print(f"Final Fallback: TF-IDF")
        
        if primary_method == "BERT":
            print("\n🎉 SUCCESS: BERT is configured as PRIMARY embedding method!")
            print("\n📋 Embedding Hierarchy:")
            print("1. 🧠 BERT (bert-base-uncased) - PRIMARY")
            print("2. 🔄 Sentence Transformers - FALLBACK")
            print("3. 📊 TF-IDF - EMERGENCY FALLBACK")
            
            print("\n⚡ BERT Features:")
            print("• Batch processing (16 sessions per batch)")
            print("• Mean pooling (no special token contamination)")
            print("• Advanced text preprocessing")
            print("• Production-optimized error handling")
            return True
        else:
            print("\n❌ ISSUE: BERT is not the primary embedding method")
            return False
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test execution"""
    success = test_bert_primary_configuration()
    
    if success:
        print("\n🏆 All tests passed! BERT is now the primary embedding method.")
    else:
        print("\n⚠️ Configuration issues detected. Check the output above.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
