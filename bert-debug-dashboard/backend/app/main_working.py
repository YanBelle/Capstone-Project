from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
import time
import random
import asyncio

app = FastAPI(title="BERT Debug Dashboard API - Working Version")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3333", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "BERT Debug Dashboard API - Working Version", 
        "status": "running",
        "timestamp": time.time()
    }

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": "mock-fast-analyzer",
        "timestamp": time.time()
    }

@app.get("/api/model_info")
async def model_info():
    return {
        "model_loaded": True,
        "device": "cpu",
        "num_labels": 4,
        "max_length": 128,
        "model_type": "mock-fast-analyzer"
    }

@app.post("/api/analyze")
async def analyze_text(text: str = Form(...)):
    """Fast mock analyze endpoint that simulates real analysis"""
    print(f"=== ANALYZE ENDPOINT CALLED ===")
    print(f"Received text: {text[:100]}...")
    
    # Simulate some processing time
    start_time = time.time()
    await asyncio.sleep(0.1)  # Very fast response
    
    # Generate realistic mock data
    words = text.split()[:10] if text else ["mock", "text"]
    tokens = ["[CLS]"] + words + ["[SEP]"]
    
    # Generate random but realistic probabilities
    probs = [random.uniform(0.1, 0.9) for _ in range(4)]
    prob_sum = sum(probs)
    probs = [p/prob_sum for p in probs]  # Normalize
    
    predicted_class = probs.index(max(probs))
    
    # Generate attention weights
    seq_len = len(tokens)
    attention_matrix = [[random.uniform(0.1, 0.9) for _ in range(seq_len)] for _ in range(seq_len)]
    
    result = {
        "text": text,
        "tokens": tokens,
        "predicted_class": predicted_class,
        "probabilities": probs,
        "attention_weights": [{
            "layer": 5,  # Last layer of DistilBERT
            "heads": [{
                "head": 0,
                "attention": attention_matrix
            }]
        }],
        "token_importance": [random.uniform(0.2, 0.8) for _ in range(len(tokens))],
        "hidden_states": {"cls_embeddings": []},
        "analysis_time": f"{time.time() - start_time:.2f}s",
        "analyzer_type": "MockFastAnalyzer"
    }
    
    print(f"Analysis completed in {time.time() - start_time:.2f}s")
    print(f"Predicted class: {predicted_class}")
    return result
