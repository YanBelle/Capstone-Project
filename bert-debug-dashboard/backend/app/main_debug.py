from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time

app = FastAPI(title="BERT Debug Dashboard API - Debug Mode")

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
        "message": "BERT Debug Dashboard API - Debug Mode", 
        "status": "running",
        "timestamp": time.time()
    }

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": False,
        "mode": "debug",
        "timestamp": time.time()
    }

@app.post("/api/analyze")
async def analyze_debug(text: str = ""):
    """Debug analyze endpoint"""
    return {
        "text": text or "debug text",
        "tokens": ["debug", "tokens"],
        "predicted_class": 1,
        "probabilities": [0.1, 0.7, 0.15, 0.05],
        "attention_weights": [],
        "token_importance": [0.5, 0.5],
        "hidden_states": {"cls_embeddings": []},
        "analysis_time": "0.1s",
        "mode": "debug"
    }
