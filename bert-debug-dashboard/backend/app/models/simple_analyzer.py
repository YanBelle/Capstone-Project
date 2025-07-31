import time
from typing import Dict, Any

class SimpleBERTAnalyzer:
    """Simple fallback analyzer for debugging"""
    
    def __init__(self, model_path: str):
        print(f"Initializing SimpleBERTAnalyzer (fallback mode)")
        self.model_path = model_path
        self.device = "cpu"
        print("SimpleBERTAnalyzer initialized successfully")
        
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Fast mock analysis for debugging"""
        start_time = time.time()
        print(f"SimpleBERTAnalyzer: Analyzing '{text[:50]}...'")
        
        # Simulate some processing time
        time.sleep(0.5)
        
        # Mock tokenization
        tokens = text.split()[:10]  # First 10 words
        if not tokens:
            tokens = ["[MOCK]", "TOKENS"]
        
        result = {
            "text": text,
            "tokens": tokens,
            "predicted_class": 1,
            "probabilities": [0.1, 0.7, 0.15, 0.05],
            "attention_weights": [{
                "layer": 0,
                "heads": [{
                    "head": 0,
                    "attention": [[0.5] * len(tokens) for _ in range(len(tokens))]
                }]
            }],
            "token_importance": [0.5] * len(tokens),
            "hidden_states": {"cls_embeddings": []},
            "analysis_time": f"{time.time() - start_time:.2f}s",
            "analyzer_type": "SimpleBERTAnalyzer (fallback)"
        }
        
        print(f"SimpleBERTAnalyzer: Analysis completed in {time.time() - start_time:.2f}s")
        return result
