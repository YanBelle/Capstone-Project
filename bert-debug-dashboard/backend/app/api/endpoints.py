from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import List, Dict, Any, Optional
import json
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import umap
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

from app.models.fast_analyzer import FastAnalyzer
from app.models.bert_analyzer import BERTAnalyzer
from app.models.simple_analyzer import SimpleBERTAnalyzer
from app.utils.metrics import MetricsCalculator

router = APIRouter()

# Initialize model (you'll need to update the path)
MODEL_PATH = "/app/models/bert_ej_model"
analyzer = None

def initialize_model():
    """Initialize the model - called from main.py"""
    global analyzer
    try:
        print("Starting with FastAnalyzer for instant responses...")
        analyzer = FastAnalyzer(MODEL_PATH)
        print("FastAnalyzer loaded successfully - ready for instant analysis")
        return True
    except Exception as e:
        print(f"FastAnalyzer failed: {e}")
        print("Falling back to BERTAnalyzer...")
        try:
            analyzer = BERTAnalyzer(MODEL_PATH)
            print("BERTAnalyzer loaded successfully")
            return True
        except Exception as e2:
            print(f"BERTAnalyzer failed: {e2}")
            print("Final fallback to SimpleBERTAnalyzer...")
            try:
                analyzer = SimpleBERTAnalyzer(MODEL_PATH)
                print("SimpleBERTAnalyzer loaded successfully")
                return True
            except Exception as e3:
                print(f"All analyzers failed: {e3}")
                return False

@router.post("/analyze")
async def analyze_text(text: str = Form(...)):
    """Analyze a single text input"""
    print(f"=== ANALYZE ENDPOINT CALLED ===")
    print(f"Received text: {text[:100]}...")
    print(f"Analyzer status: {analyzer is not None}")
    
    if not analyzer:
        print("ERROR: Model not loaded!")
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not text or text.strip() == "":
        print("ERROR: Empty text input")
        raise HTTPException(status_code=400, detail="Text input is required")
    
    try:
        print("Starting analysis...")
        results = analyzer.analyze_text(text)
        print("Analysis completed successfully")
        print(f"Results keys: {list(results.keys())}")
        return results
    except Exception as e:
        print(f"ANALYSIS ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/batch_analyze")
async def batch_analyze(file: UploadFile = File(...)):
    """Analyze batch of EJ logs from uploaded file"""
    if not analyzer:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read the file
        content = await file.read()
        df = pd.read_csv(BytesIO(content))
        
        # Analyze each text
        results = []
        metrics_calc = MetricsCalculator()
        
        for idx, row in df.iterrows():
            text = row.get('text', '')
            true_label = row.get('label', -1)
            
            if not text:
                continue
                
            analysis = analyzer.analyze_text(text)
            analysis['true_label'] = true_label
            results.append(analysis)
            
            if true_label != -1:
                metrics_calc.add_batch(
                    [analysis['predicted_class']],
                    [true_label],
                    [analysis['probabilities']]
                )
        
        # Calculate metrics if labels were provided
        metrics = {}
        if metrics_calc.true_labels:
            metrics = metrics_calc.calculate_metrics()
            
        return {
            "results": results,
            "metrics": metrics,
            "total_samples": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/embeddings")
async def get_embeddings(
    method: str = Form("tsne"),
    layer: int = Form(-1),
    file: UploadFile = File(...)
):
    """Get embeddings visualization using t-SNE or UMAP"""
    if not analyzer:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        content = await file.read()
        df = pd.read_csv(BytesIO(content))
        
        embeddings = []
        labels = []
        
        for idx, row in df.iterrows():
            text = row.get('text', '')
            if not text:
                continue
                
            analysis = analyzer.analyze_text(text)
            # Get CLS embedding from specified layer
            cls_embedding = analysis['hidden_states']['cls_embeddings'][layer]['embedding']
            embeddings.append(cls_embedding)
            labels.append(analysis['predicted_class'])
        
        embeddings = np.array(embeddings)
        
        # Reduce dimensionality
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42)
        else:
            reducer = umap.UMAP(n_components=2, random_state=42)
        
        reduced_embeddings = reducer.fit_transform(embeddings)
        
        return {
            "embeddings": reduced_embeddings.tolist(),
            "labels": labels,
            "method": method
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/token_bias")
async def analyze_token_bias(file: UploadFile = File(...)):
    """Analyze token bias across different classes"""
    if not analyzer:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        content = await file.read()
        df = pd.read_csv(BytesIO(content))
        
        # Collect tokens by class
        class_tokens = {}
        
        for idx, row in df.iterrows():
            text = row.get('text', '')
            label = row.get('label', -1)
            
            if label == -1 or not text:
                continue
                
            analysis = analyzer.analyze_text(text)
            tokens = analysis['tokens']
            
            if label not in class_tokens:
                class_tokens[label] = []
            
            class_tokens[label].extend(tokens)
        
        # Calculate token frequencies per class
        token_frequencies = {}
        for label, tokens in class_tokens.items():
            freq = {}
            for token in tokens:
                if token not in ['[CLS]', '[SEP]', '[PAD]']:
                    freq[token] = freq.get(token, 0) + 1
            
            # Sort by frequency
            sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:50]
            token_frequencies[label] = sorted_freq
        
        return {
            "token_frequencies": token_frequencies,
            "class_distribution": {str(k): len(v) for k, v in class_tokens.items()}
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/experiment")
async def experiment_with_text(
    text: str = Form(...),
    mask_tokens: Optional[str] = Form(None),
    substitute_tokens: Optional[str] = Form(None)
):
    """Experiment with text modifications"""
    if not analyzer:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Original analysis
        original_analysis = analyzer.analyze_text(text)
        
        # Modified text
        modified_text = text
        
        # Apply masking
        if mask_tokens:
            tokens_to_mask = json.loads(mask_tokens)
            for token in tokens_to_mask:
                modified_text = modified_text.replace(token, "[MASK]")
        
        # Apply substitutions
        if substitute_tokens:
            substitutions = json.loads(substitute_tokens)
            for old_token, new_token in substitutions.items():
                modified_text = modified_text.replace(old_token, new_token)
        
        # Analyze modified text
        modified_analysis = analyzer.analyze_text(modified_text)
        
        return {
            "original": original_analysis,
            "modified": modified_analysis,
            "modified_text": modified_text
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": analyzer is not None,
        "model_type": "FastAnalyzer (instant response)" if analyzer else "none",
        "timestamp": pd.Timestamp.now().isoformat()
    }

@router.get("/model_info")
async def get_model_info():
    """Get model information"""
    if not analyzer:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_loaded": True,
        "device": str(analyzer.device),
        "num_labels": analyzer.model.config.num_labels,
        "max_length": analyzer.tokenizer.model_max_length
    }
