#!/usr/bin/env python3
"""
Script to download ML models during Docker build
"""

def download_bert_models():
    """Download BERT models"""
    try:
        print("Downloading BERT models...")
        from transformers import BertTokenizer, BertModel
        
        # Download tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        print("✓ BERT tokenizer downloaded")
        
        # Download model
        model = BertModel.from_pretrained('bert-base-uncased')
        print("✓ BERT model downloaded")
        
        print("✅ BERT models downloaded successfully")
        return True
    except Exception as e:
        print(f"❌ BERT download failed: {e}")
        return False

def download_sentence_transformers():
    """Download SentenceTransformer model (optional)"""
    try:
        print("Downloading SentenceTransformer model...")
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ SentenceTransformer model downloaded successfully")
        return True
    except Exception as e:
        print(f"⚠️  SentenceTransformer download failed: {e}")
        print("Will fallback to BERT during runtime")
        return False

if __name__ == "__main__":
    import sys
    
    # Download BERT models (required)
    if not download_bert_models():
        print("❌ Failed to download required BERT models")
        sys.exit(1)
    
    # Download SentenceTransformer models (optional)
    download_sentence_transformers()
    
    print("🎉 Model download process completed!")
