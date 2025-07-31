#!/bin/bash

# BERT Debugging Dashboard Setup Script
# This script creates the complete project structure and files

set -e  # Exit on error

echo "🚀 Setting up BERT Debugging Dashboard..."

# Create main project directory
PROJECT_DIR="bert-debug-dashboard"
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

echo "📁 Creating directory structure..."

# Create all necessary directories
mkdir -p backend/app/{models,utils,api}
mkdir -p frontend/{public,src/{components,styles}}
mkdir -p models/bert_ej_model
mkdir -p data

# Create docker-compose.yml with updated port
echo "📝 Creating docker-compose.yml..."
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  backend:
    build: ./backend
    container_name: bert-debug-backend
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - PYTHONUNBUFFERED=1
      - MODEL_PATH=/app/models/bert_ej_model
    networks:
      - bert-network

  frontend:
    build: ./frontend
    container_name: bert-debug-frontend
    ports:
      - "3333:3000"
    depends_on:
      - backend
    environment:
      - REACT_APP_API_URL=http://localhost:8000
    networks:
      - bert-network

networks:
  bert-network:
    driver: bridge
EOF

# Backend Files
echo "📝 Creating backend files..."

# Backend Dockerfile
cat > backend/Dockerfile << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create directories for models and data
RUN mkdir -p /app/models /app/data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
EOF

# Backend requirements.txt
cat > backend/requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn==0.24.0
torch==2.1.0
transformers==4.35.0
numpy==1.24.3
pandas==2.1.3
scikit-learn==1.3.2
matplotlib==3.8.0
seaborn==0.13.0
plotly==5.18.0
captum==0.6.0
shap==0.43.0
lime==0.2.0.1
umap-learn==0.5.4
bertviz==1.4.0
pydantic==2.5.0
python-multipart==0.0.6
EOF

# Create __init__.py files
touch backend/app/__init__.py
touch backend/app/models/__init__.py
touch backend/app/utils/__init__.py
touch backend/app/api/__init__.py

# Backend main.py
cat > backend/app/main.py << 'EOF'
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import json

from app.api.endpoints import router

app = FastAPI(title="BERT Debug Dashboard API")

# Configure CORS - Updated to allow port 3333
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3333", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "BERT Debug Dashboard API"}
EOF

# Backend bert_analyzer.py
cat > backend/app/models/bert_analyzer.py << 'EOF'
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import List, Dict, Tuple, Any
from captum.attr import IntegratedGradients, LayerIntegratedGradients
import shap

class BERTAnalyzer:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        except:
            # Fallback to default BERT if custom model not found
            print(f"Warning: Could not load model from {model_path}, using default bert-base-uncased")
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
        
        self.model.to(self.device)
        self.model.eval()
        
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Comprehensive analysis of input text"""
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", 
                               padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model outputs with attention
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True, 
                               output_hidden_states=True)
        
        # Get predictions
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
        
        # Extract attention weights
        attention_weights = self._extract_attention_weights(outputs.attentions)
        
        # Get token importance using Integrated Gradients
        token_importance = self._get_token_importance(inputs, predicted_class)
        
        # Get hidden states for embedding analysis
        hidden_states = self._extract_hidden_states(outputs.hidden_states)
        
        return {
            "text": text,
            "tokens": self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]),
            "predicted_class": predicted_class,
            "probabilities": probs[0].tolist(),
            "attention_weights": attention_weights,
            "token_importance": token_importance,
            "hidden_states": hidden_states,
        }
    
    def _extract_attention_weights(self, attentions: Tuple) -> List[Dict]:
        """Extract attention weights for each layer and head"""
        attention_data = []
        for layer_idx, layer_attention in enumerate(attentions):
            layer_data = {
                "layer": layer_idx,
                "heads": []
            }
            # Shape: (batch, heads, seq_len, seq_len)
            for head_idx in range(layer_attention.shape[1]):
                head_attention = layer_attention[0, head_idx].cpu().numpy()
                layer_data["heads"].append({
                    "head": head_idx,
                    "attention": head_attention.tolist()
                })
            attention_data.append(layer_data)
        return attention_data
    
    def _get_token_importance(self, inputs: Dict, target_class: int) -> List[float]:
        """Calculate token importance using Integrated Gradients"""
        try:
            ig = IntegratedGradients(self._forward_func)
            
            # Prepare baseline (all padding tokens)
            baseline_ids = torch.full_like(inputs["input_ids"], 
                                          self.tokenizer.pad_token_id)
            
            # Calculate attributions
            attributions = ig.attribute(
                inputs["input_ids"],
                baselines=baseline_ids,
                target=target_class,
                n_steps=50
            )
            
            # Aggregate attributions
            token_importance = attributions.sum(dim=-1).squeeze(0).cpu().numpy()
            
            # Normalize
            if token_importance.max() - token_importance.min() > 0:
                token_importance = (token_importance - token_importance.min()) / \
                                  (token_importance.max() - token_importance.min())
            else:
                token_importance = np.zeros_like(token_importance)
            
            return token_importance.tolist()
        except:
            # Return uniform importance if IG fails
            return [0.5] * len(inputs["input_ids"][0])
    
    def _forward_func(self, input_ids):
        """Forward function for Integrated Gradients"""
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
    
    def _extract_hidden_states(self, hidden_states: Tuple) -> Dict[str, List]:
        """Extract hidden states for embedding analysis"""
        # Get CLS token embeddings from each layer
        cls_embeddings = []
        for layer_idx, layer_hidden in enumerate(hidden_states):
            cls_embedding = layer_hidden[0, 0].cpu().numpy()  # [CLS] token
            cls_embeddings.append({
                "layer": layer_idx,
                "embedding": cls_embedding.tolist()
            })
        
        return {"cls_embeddings": cls_embeddings}
EOF

# Backend metrics.py
cat > backend/app/utils/metrics.py << 'EOF'
import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                           confusion_matrix, roc_curve, auc, precision_recall_curve)
from typing import List, Dict, Tuple

class MetricsCalculator:
    def __init__(self):
        self.predictions = []
        self.true_labels = []
        self.probabilities = []
        
    def add_batch(self, predictions: List[int], true_labels: List[int], 
                  probabilities: List[List[float]]):
        """Add a batch of predictions for metric calculation"""
        self.predictions.extend(predictions)
        self.true_labels.extend(true_labels)
        self.probabilities.extend(probabilities)
    
    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive metrics"""
        if not self.predictions:
            return {}
            
        predictions = np.array(self.predictions)
        true_labels = np.array(self.true_labels)
        probabilities = np.array(self.probabilities)
        
        # Basic metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # ROC and PR curves (for binary classification)
        roc_data = {}
        pr_data = {}
        
        if probabilities.shape[1] == 2:  # Binary classification
            fpr, tpr, _ = roc_curve(true_labels, probabilities[:, 1])
            roc_auc = auc(fpr, tpr)
            
            precision_curve, recall_curve, _ = precision_recall_curve(
                true_labels, probabilities[:, 1]
            )
            
            roc_data = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": roc_auc
            }
            
            pr_data = {
                "precision": precision_curve.tolist(),
                "recall": recall_curve.tolist()
            }
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist(),
            "roc_curve": roc_data,
            "pr_curve": pr_data
        }
    
    def get_misclassifications(self) -> List[Dict]:
        """Get misclassified examples"""
        misclassified = []
        for i, (pred, true) in enumerate(zip(self.predictions, self.true_labels)):
            if pred != true:
                misclassified.append({
                    "index": i,
                    "predicted": pred,
                    "true_label": true,
                    "confidence": max(self.probabilities[i])
                })
        return misclassified
EOF

# Backend endpoints.py
cat > backend/app/api/endpoints.py << 'EOF'
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

from app.models.bert_analyzer import BERTAnalyzer
from app.utils.metrics import MetricsCalculator

router = APIRouter()

# Initialize model (you'll need to update the path)
MODEL_PATH = "/app/models/bert_ej_model"
analyzer = None

@router.on_event("startup")
async def load_model():
    global analyzer
    try:
        analyzer = BERTAnalyzer(MODEL_PATH)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")

@router.post("/analyze")
async def analyze_text(text: str = Form(...)):
    """Analyze a single text input"""
    if not analyzer:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        results = analyzer.analyze_text(text)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
EOF

# Frontend Files
echo "📝 Creating frontend files..."

# Frontend Dockerfile
cat > frontend/Dockerfile << 'EOF'
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy application code
COPY . .

# Build the app (if needed)
# RUN npm run build

# Use development server
EXPOSE 3000

CMD ["npm", "start"]
EOF

# Frontend package.json
cat > frontend/package.json << 'EOF'
{
  "name": "bert-debug-dashboard",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "axios": "^1.6.0",
    "recharts": "^2.8.0",
    "plotly.js": "^2.27.0",
    "react-plotly.js": "^2.6.0",
    "@mui/material": "^5.14.0",
    "@emotion/react": "^11.11.0",
    "@emotion/styled": "^11.11.0",
    "d3": "^7.8.0",
    "react-dropzone": "^14.2.0",
    "react-tabs": "^6.0.0"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "eslintConfig": {
    "extends": [
      "react-app"
    ]
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  }
}
EOF

# Frontend public/index.html
cat > frontend/public/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta name="description" content="BERT Debugging Dashboard" />
    <title>BERT Debug Dashboard</title>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>
EOF

# Frontend src/index.js
cat > frontend/src/index.js << 'EOF'
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
EOF

# Frontend src/App.js
cat > frontend/src/App.js << 'EOF'
import React, { useState } from 'react';
import Dashboard from './components/Dashboard';
import './styles/Dashboard.css';

function App() {
  const [modelInfo, setModelInfo] = useState(null);

  React.useEffect(() => {
    // Check if model is loaded
    fetch(`${process.env.REACT_APP_API_URL}/api/model_info`)
      .then(res => res.json())
      .then(data => setModelInfo(data))
      .catch(err => console.error('Failed to load model info:', err));
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>BERT EJ Log Debugging Dashboard</h1>
        {modelInfo && (
          <div className="model-info">
            <span>Model: Loaded on {modelInfo.device}</span>
            <span>Classes: {modelInfo.num_labels}</span>
          </div>
        )}
      </header>
      <Dashboard />
    </div>
  );
}

export default App;
EOF

# Create all component files
echo "📝 Creating component files..."

# Dashboard.js
cat > frontend/src/components/Dashboard.js << 'EOF'
import React, { useState } from 'react';
import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';
import 'react-tabs/style/react-tabs.css';
import AttentionVisualizer from './AttentionVisualizer';
import PredictionPanel from './PredictionPanel';
import SaliencyMap from './SaliencyMap';
import EmbeddingAnalysis from './EmbeddingAnalysis';
import MetricsPanel from './MetricsPanel';
import MisclassificationExplorer from './MisclassificationExplorer';
import TokenBiasInsights from './TokenBiasInsights';
import { TextField, Button, CircularProgress, Paper } from '@mui/material';
import axios from 'axios';

const Dashboard = () => {
  const [inputText, setInputText] = useState('');
  const [analysisResult, setAnalysisResult] = useState(null);
  const [batchResults, setBatchResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const analyzeText = async () => {
    if (!inputText) return;
    
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('text', inputText);
      
      const response = await axios.post(
        `${process.env.REACT_APP_API_URL}/api/analyze`,
        formData
      );
      
      setAnalysisResult(response.data);
    } catch (error) {
      console.error('Analysis failed:', error);
      alert('Analysis failed. Please check the console for details.');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (file) => {
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await axios.post(
        `${process.env.REACT_APP_API_URL}/api/batch_analyze`,
        formData
      );
      
      setBatchResults(response.data);
    } catch (error) {
      console.error('Batch analysis failed:', error);
      alert('Batch analysis failed. Please check the console for details.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboard">
      <Paper className="input-section" elevation={3}>
        <h2>Input EJ Log</h2>
        <TextField
          multiline
          rows={4}
          fullWidth
          variant="outlined"
          placeholder="Enter EJ log text here..."
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
        />
        <Button 
          variant="contained" 
          color="primary" 
          onClick={analyzeText}
          disabled={loading}
          style={{ marginTop: 10 }}
        >
          {loading ? <CircularProgress size={24} /> : 'Analyze'}
        </Button>
      </Paper>

      {analysisResult && (
        <Tabs>
          <TabList>
            <Tab>Prediction</Tab>
            <Tab>Attention</Tab>
            <Tab>Saliency</Tab>
            <Tab>Embeddings</Tab>
            <Tab>Metrics</Tab>
            <Tab>Misclassifications</Tab>
            <Tab>Token Bias</Tab>
          </TabList>

          <TabPanel>
            <PredictionPanel result={analysisResult} />
          </TabPanel>

          <TabPanel>
            <AttentionVisualizer 
              tokens={analysisResult.tokens}
              attentionWeights={analysisResult.attention_weights}
            />
          </TabPanel>

          <TabPanel>
            <SaliencyMap 
              tokens={analysisResult.tokens}
              importance={analysisResult.token_importance}
            />
          </TabPanel>

          <TabPanel>
            <EmbeddingAnalysis 
              hiddenStates={analysisResult.hidden_states}
              onFileUpload={handleFileUpload}
            />
          </TabPanel>

          <TabPanel>
            <MetricsPanel 
              metrics={batchResults?.metrics}
              onFileUpload={handleFileUpload}
            />
          </TabPanel>

          <TabPanel>
            <MisclassificationExplorer 
              results={batchResults?.results}
              onFileUpload={handleFileUpload}
            />
          </TabPanel>

          <TabPanel>
            <TokenBiasInsights 
              onFileUpload={handleFileUpload}
            />
          </TabPanel>
        </Tabs>
      )}
    </div>
  );
};

export default Dashboard;
EOF

# AttentionVisualizer.js
cat > frontend/src/components/AttentionVisualizer.js << 'EOF'
import React, { useState } from 'react';
import Plot from 'react-plotly.js';
import { FormControl, InputLabel, Select, MenuItem } from '@mui/material';

const AttentionVisualizer = ({ tokens, attentionWeights }) => {
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [selectedHead, setSelectedHead] = useState(0);

  if (!attentionWeights || attentionWeights.length === 0) {
    return <div>No attention data available</div>;
  }

  const currentAttention = attentionWeights[selectedLayer].heads[selectedHead].attention;

  // Create heatmap data
  const heatmapData = [{
    z: currentAttention,
    x: tokens,
    y: tokens,
    type: 'heatmap',
    colorscale: 'Viridis',
    showscale: true
  }];

  const layout = {
    title: `Attention Weights - Layer ${selectedLayer}, Head ${selectedHead}`,
    xaxis: { 
      title: 'Keys',
      tickangle: -45
    },
    yaxis: { 
      title: 'Queries',
      autorange: 'reversed'
    },
    width: 800,
    height: 600
  };

  return (
    <div className="attention-visualizer">
      <div className="controls">
        <FormControl style={{ marginRight: 20 }}>
          <InputLabel>Layer</InputLabel>
          <Select
            value={selectedLayer}
            onChange={(e) => setSelectedLayer(e.target.value)}
          >
            {attentionWeights.map((_, idx) => (
              <MenuItem key={idx} value={idx}>Layer {idx}</MenuItem>
            ))}
          </Select>
        </FormControl>

        <FormControl>
          <InputLabel>Head</InputLabel>
          <Select
            value={selectedHead}
            onChange={(e) => setSelectedHead(e.target.value)}
          >
            {attentionWeights[selectedLayer].heads.map((_, idx) => (
              <MenuItem key={idx} value={idx}>Head {idx}</MenuItem>
            ))}
          </Select>
        </FormControl>
      </div>

      <Plot data={heatmapData} layout={layout} />
    </div>
  );
};

export default AttentionVisualizer;
EOF

# PredictionPanel.js
cat > frontend/src/components/PredictionPanel.js << 'EOF'
import React from 'react';
import { Paper, Typography, LinearProgress } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

const PredictionPanel = ({ result }) => {
  if (!result) return null;

  const classNames = ['Normal', 'Anomaly', 'Fault', 'Warning']; // Adjust based on your model
  
  const predictionData = result.probabilities.map((prob, idx) => ({
    class: classNames[idx] || `Class ${idx}`,
    probability: (prob * 100).toFixed(2)
  }));

  const predictedClassName = classNames[result.predicted_class] || `Class ${result.predicted_class}`;
  const confidence = Math.max(...result.probabilities) * 100;

  return (
    <Paper className="prediction-panel" elevation={2} style={{ padding: 20 }}>
      <Typography variant="h5" gutterBottom>
        Prediction Results
      </Typography>
      
      <div className="prediction-summary">
        <Typography variant="h6">
          Predicted: <strong>{predictedClassName}</strong>
        </Typography>
        <Typography variant="body1">
          Confidence: {confidence.toFixed(2)}%
        </Typography>
        <LinearProgress 
          variant="determinate" 
          value={confidence} 
          style={{ marginTop: 10, marginBottom: 20 }}
        />
      </div>

      <Typography variant="h6" gutterBottom>
        Class Probabilities
      </Typography>
      <BarChart width={600} height={300} data={predictionData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="class" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Bar dataKey="probability" fill="#8884d8" />
      </BarChart>
    </Paper>
  );
};

export default PredictionPanel;
EOF

# SaliencyMap.js
cat > frontend/src/components/SaliencyMap.js << 'EOF'
import React from 'react';
import { Paper, Typography } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const SaliencyMap = ({ tokens, importance }) => {
  if (!tokens || !importance) return null;

  // Prepare data for bar chart
  const data = tokens.map((token, idx) => ({
    token: token,
    importance: importance[idx] * 100
  })).filter(item => item.token !== '[PAD]');

  // Color tokens based on importance
  const getImportanceClass = (value) => {
    if (value < 0.2) return 'token-very-low';
    if (value < 0.4) return 'token-low';
    if (value < 0.6) return 'token-medium';
    if (value < 0.8) return 'token-high';
    return 'token-very-high';
  };

  return (
    <Paper className="saliency-map" elevation={2} style={{ padding: 20 }}>
      <Typography variant="h5" gutterBottom>
        Token Importance Analysis
      </Typography>
      
      <div className="token-importance">
        {tokens.map((token, idx) => (
          token !== '[PAD]' && (
            <span 
              key={idx} 
              className={`token ${getImportanceClass(importance[idx])}`}
              title={`Importance: ${(importance[idx] * 100).toFixed(2)}%`}
            >
              {token}
            </span>
          )
        ))}
      </div>

      <Typography variant="h6" style={{ marginTop: 30 }}>
        Importance Scores
      </Typography>
      <ResponsiveContainer width="100%" height={400}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="token" angle={-45} textAnchor="end" height={100} />
          <YAxis />
          <Tooltip />
          <Bar dataKey="importance" fill="#82ca9d" />
        </BarChart>
      </ResponsiveContainer>
    </Paper>
  );
};

export default SaliencyMap;
EOF

# EmbeddingAnalysis.js
cat > frontend/src/components/EmbeddingAnalysis.js << 'EOF'
import React, { useState } from 'react';
import { Paper, Typography, Button, FormControl, InputLabel, Select, MenuItem } from '@mui/material';
import { useDropzone } from 'react-dropzone';
import Plot from 'react-plotly.js';
import axios from 'axios';

const EmbeddingAnalysis = ({ onFileUpload }) => {
  const [embeddings, setEmbeddings] = useState(null);
  const [method, setMethod] = useState('tsne');
  const [layer, setLayer] = useState(-1);
  const [loading, setLoading] = useState(false);

  const onDrop = async (acceptedFiles) => {
    if (acceptedFiles.length === 0) return;
    
    setLoading(true);
    const file = acceptedFiles[0];
    onFileUpload(file);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('method', method);
      formData.append('layer', layer);

      const response = await axios.post(
        `${process.env.REACT_APP_API_URL}/api/embeddings`,
        formData
      );

      setEmbeddings(response.data);
    } catch (error) {
      console.error('Embedding analysis failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const { getRootProps, getInputProps } = useDropzone({ 
    onDrop,
    accept: {
      'text/csv': ['.csv']
    }
  });

  const plotData = embeddings ? [{
    x: embeddings.embeddings.map(e => e[0]),
    y: embeddings.embeddings.map(e => e[1]),
    mode: 'markers',
    type: 'scatter',
    marker: {
      color: embeddings.labels,
      colorscale: 'Viridis',
      size: 8
    },
    text: embeddings.labels.map(l => `Class: ${l}`)
  }] : [];

  const layout = {
    title: `${method.toUpperCase()} Visualization - Layer ${layer}`,
    xaxis: { title: 'Component 1' },
    yaxis: { title: 'Component 2' },
    width: 800,
    height: 600
  };

  return (
    <Paper className="embedding-analysis" elevation={2} style={{ padding: 20 }}>
      <Typography variant="h5" gutterBottom>
        Embedding Analysis
      </Typography>

      <div style={{ marginBottom: 20 }}>
        <FormControl style={{ marginRight: 20 }}>
          <InputLabel>Method</InputLabel>
          <Select value={method} onChange={(e) => setMethod(e.target.value)}>
            <MenuItem value="tsne">t-SNE</MenuItem>
            <MenuItem value="umap">UMAP</MenuItem>
          </Select>
        </FormControl>

        <FormControl>
          <InputLabel>Layer</InputLabel>
          <Select value={layer} onChange={(e) => setLayer(e.target.value)}>
            <MenuItem value={-1}>Last Layer</MenuItem>
            <MenuItem value={0}>Layer 0</MenuItem>
            <MenuItem value={6}>Layer 6</MenuItem>
            <MenuItem value={11}>Layer 11</MenuItem>
          </Select>
        </FormControl>
      </div>

      <div {...getRootProps()} className="upload-zone">
        <input {...getInputProps()} />
        <p>Drop a CSV file here or click to upload</p>
        <p>File should contain 'text' column</p>
      </div>

      {loading && <Typography>Processing embeddings...</Typography>}

      {embeddings && (
        <Plot data={plotData} layout={layout} />
      )}
    </Paper>
  );
};

export default EmbeddingAnalysis;
EOF

# MetricsPanel.js
cat > frontend/src/components/MetricsPanel.js << 'EOF'
import React from 'react';
import { Paper, Typography } from '@mui/material';
import { useDropzone } from 'react-dropzone';
import Plot from 'react-plotly.js';

const MetricsPanel = ({ metrics, onFileUpload }) => {
  const { getRootProps, getInputProps } = useDropzone({ 
    onDrop: (files) => files.length > 0 && onFileUpload(files[0]),
    accept: {
      'text/csv': ['.csv']
    }
  });

  if (!metrics || Object.keys(metrics).length === 0) {
    return (
      <Paper elevation={2} style={{ padding: 20 }}>
        <Typography variant="h5" gutterBottom>
          Performance Metrics
        </Typography>
        <div {...getRootProps()} className="upload-zone">
          <input {...getInputProps()} />
          <p>Upload a CSV file with 'text' and 'label' columns to see metrics</p>
        </div>
      </Paper>
    );
  }

  // Confusion Matrix Plot
  const confusionData = [{
    z: metrics.confusion_matrix,
    type: 'heatmap',
    colorscale: 'Blues',
    showscale: true
  }];

  const confusionLayout = {
    title: 'Confusion Matrix',
    xaxis: { title: 'Predicted' },
    yaxis: { title: 'True', autorange: 'reversed' },
    width: 500,
    height: 500
  };

  return (
    <Paper elevation={2} style={{ padding: 20 }}>
      <Typography variant="h5" gutterBottom>
        Performance Metrics
      </Typography>

      <div className="metrics-grid">
        <div className="metric-card">
          <h3>Accuracy</h3>
          <div className="value">{(metrics.accuracy * 100).toFixed(2)}%</div>
        </div>
        <div className="metric-card">
          <h3>Precision</h3>
          <div className="value">{(metrics.precision * 100).toFixed(2)}%</div>
        </div>
        <div className="metric-card">
          <h3>Recall</h3>
          <div className="value">{(metrics.recall * 100).toFixed(2)}%</div>
        </div>
        <div className="metric-card">
          <h3>F1 Score</h3>
          <div className="value">{(metrics.f1_score * 100).toFixed(2)}%</div>
        </div>
      </div>

      <div className="confusion-matrix">
        <Plot data={confusionData} layout={confusionLayout} />
      </div>

      {metrics.roc_curve && metrics.roc_curve.fpr && (
        <Plot
          data={[{
            x: metrics.roc_curve.fpr,
            y: metrics.roc_curve.tpr,
            type: 'scatter',
            mode: 'lines',
            name: `ROC (AUC = ${metrics.roc_curve.auc.toFixed(3)})`
          }]}
          layout={{
            title: 'ROC Curve',
            xaxis: { title: 'False Positive Rate' },
            yaxis: { title: 'True Positive Rate' },
            width: 600,
            height: 500
          }}
        />
      )}
    </Paper>
  );
};

export default MetricsPanel;
EOF

# MisclassificationExplorer.js
cat > frontend/src/components/MisclassificationExplorer.js << 'EOF'
import React, { useState } from 'react';
import { Paper, Typography, TextField } from '@mui/material';
import { useDropzone } from 'react-dropzone';

const MisclassificationExplorer = ({ results, onFileUpload }) => {
  const [filter, setFilter] = useState('');
  
  const { getRootProps, getInputProps } = useDropzone({ 
    onDrop: (files) => files.length > 0 && onFileUpload(files[0]),
    accept: {
      'text/csv': ['.csv']
    }
  });

  if (!results || results.length === 0) {
    return (
      <Paper elevation={2} style={{ padding: 20 }}>
        <Typography variant="h5" gutterBottom>
          Misclassification Explorer
        </Typography>
        <div {...getRootProps()} className="upload-zone">
          <input {...getInputProps()} />
          <p>Upload a CSV file with 'text' and 'label' columns to explore misclassifications</p>
        </div>
      </Paper>
    );
  }

  const misclassified = results.filter(r => 
    r.true_label !== -1 && r.true_label !== r.predicted_class
  );

  const filtered = filter 
    ? misclassified.filter(r => r.text.toLowerCase().includes(filter.toLowerCase()))
    : misclassified;

  return (
    <Paper elevation={2} style={{ padding: 20 }}>
      <Typography variant="h5" gutterBottom>
        Misclassification Explorer
      </Typography>
      
      <Typography variant="body1" gutterBottom>
        Found {misclassified.length} misclassifications out of {results.length} samples
      </Typography>

      <TextField
        fullWidth
        variant="outlined"
        placeholder="Filter by text content..."
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        style={{ marginBottom: 20 }}
      />

      <div className="misclassification-list">
        {filtered.map((item, idx) => (
          <div key={idx} className="misclassification-item">
            <Typography variant="body2" style={{ fontFamily: 'monospace' }}>
              {item.text}
            </Typography>
            <div style={{ marginTop: 10 }}>
              <span style={{ marginRight: 20 }}>
                True: <strong>{item.true_label}</strong>
              </span>
              <span style={{ marginRight: 20 }}>
                Predicted: <strong>{item.predicted_class}</strong>
              </span>
              <span>
                Confidence: <strong>{(Math.max(...item.probabilities) * 100).toFixed(2)}%</strong>
              </span>
            </div>
          </div>
        ))}
      </div>
    </Paper>
  );
};

export default MisclassificationExplorer;
EOF

# TokenBiasInsights.js
cat > frontend/src/components/TokenBiasInsights.js << 'EOF'
import React, { useState } from 'react';
import { Paper, Typography, CircularProgress } from '@mui/material';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';

const TokenBiasInsights = ({ onFileUpload }) => {
  const [tokenData, setTokenData] = useState(null);
  const [loading, setLoading] = useState(false);

  const onDrop = async (acceptedFiles) => {
    if (acceptedFiles.length === 0) return;
    
    setLoading(true);
    const file = acceptedFiles[0];
    onFileUpload(file);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post(
        `${process.env.REACT_APP_API_URL}/api/token_bias`,
        formData
      );

      setTokenData(response.data);
    } catch (error) {
      console.error('Token bias analysis failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const { getRootProps, getInputProps } = useDropzone({ 
    onDrop,
    accept: {
      'text/csv': ['.csv']
    }
  });

  return (
    <Paper elevation={2} style={{ padding: 20 }}>
      <Typography variant="h5" gutterBottom>
        Token Bias Insights
      </Typography>

      <div {...getRootProps()} className="upload-zone">
        <input {...getInputProps()} />
        <p>Upload a CSV file with 'text' and 'label' columns to analyze token bias</p>
      </div>

      {loading && <CircularProgress />}

      {tokenData && (
        <div className="token-bias-container">
          {Object.entries(tokenData.token_frequencies).map(([label, tokens]) => (
            <div key={label} className="class-tokens">
              <Typography variant="h6">
                Class {label} - Top Tokens
              </Typography>
              <div className="token-cloud">
                {tokens.slice(0, 20).map(([token, count], idx) => (
                  <span 
                    key={idx} 
                    style={{ 
                      fontSize: Math.min(20, 12 + Math.log(count)),
                      opacity: 0.7 + (idx === 0 ? 0.3 : 0)
                    }}
                  >
                    {token} ({count})
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </Paper>
  );
};

export default TokenBiasInsights;
EOF

# CSS file
cat > frontend/src/styles/Dashboard.css << 'EOF'
.App {
  text-align: center;
  background-color: #f5f5f5;
  min-height: 100vh;
}

.App-header {
  background-color: #1976d2;
  padding: 20px;
  color: white;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.App-header h1 {
  margin: 0;
  font-size: 2.5rem;
}

.model-info {
  margin-top: 10px;
  font-size: 0.9rem;
}

.model-info span {
  margin: 0 15px;
  opacity: 0.9;
}

.dashboard {
  padding: 20px;
  max-width: 1400px;
  margin: 0 auto;
}

.input-section {
  padding: 20px;
  margin-bottom: 30px;
}

.input-section h2 {
  margin-top: 0;
  color: #333;
}

.react-tabs {
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.react-tabs__tab-list {
  border-bottom: 2px solid #e0e0e0;
  margin: 0;
  padding: 0;
}

.react-tabs__tab {
  padding: 12px 24px;
  font-weight: 500;
  color: #666;
  transition: all 0.3s ease;
}

.react-tabs__tab--selected {
  color: #1976d2;
  border-bottom: 3px solid #1976d2;
}

.react-tabs__tab-panel {
  padding: 20px;
}

.attention-visualizer .controls {
  margin-bottom: 20px;
  display: flex;
  gap: 20px;
}

.prediction-panel {
  max-width: 800px;
  margin: 0 auto;
}

.prediction-summary {
  background: #f0f7ff;
  padding: 15px;
  border-radius: 8px;
  margin-bottom: 20px;
}

.token-importance {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  padding: 20px;
  background: #fafafa;
  border-radius: 8px;
}

.token {
  padding: 4px 8px;
  border-radius: 4px;
  font-family: monospace;
  font-size: 14px;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
  margin-bottom: 30px;
}

.metric-card {
  background: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  text-align: center;
}

.metric-card h3 {
  margin: 0;
  color: #666;
  font-size: 0.9rem;
  text-transform: uppercase;
}

.metric-card .value {
  font-size: 2rem;
  font-weight: bold;
  color: #1976d2;
  margin: 10px 0;
}

.confusion-matrix {
  margin-top: 30px;
}

.misclassification-list {
  max-height: 600px;
  overflow-y: auto;
}

.misclassification-item {
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  padding: 15px;
  margin-bottom: 10px;
  transition: all 0.3s ease;
}

.misclassification-item:hover {
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.token-bias-container {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
}

.class-tokens {
  background: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.token-cloud {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 10px;
}

.token-cloud span {
  padding: 4px 12px;
  background: #e3f2fd;
  border-radius: 16px;
  font-size: 14px;
}

.upload-zone {
  border: 2px dashed #1976d2;
  border-radius: 8px;
  padding: 40px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
}

.upload-zone:hover {
  background: #f0f7ff;
}

/* Token importance colors */
.token-very-low { background-color: #e8f5e9; }
.token-low { background-color: #c8e6c9; }
.token-medium { background-color: #81c784; }
.token-high { background-color: #4caf50; }
.token-very-high { background-color: #2e7d32; color: white; }

/* Responsive design */
@media (max-width: 768px) {
  .dashboard {
    padding: 10px;
  }
  
  .App-header h1 {
    font-size: 1.8rem;
  }
  
  .metrics-grid {
    grid-template-columns: 1fr;
  }
}
EOF

# Create sample data file
echo "📝 Creating sample data file..."
cat > data/sample_ej_logs.csv << 'EOF'
text,label
"EJ001 TRANS START CARD_READ SUCCESS AMT=50.00 RESP=00",0
"EJ002 TRANS FAILED CARD_READ ERROR CODE=51 INSUFFICIENT_FUNDS",1
"EJ003 DEVICE FAULT PRINTER_JAM SEVERITY=HIGH ACTION=SERVICE_REQUIRED",2
"EJ004 TRANS COMPLETE WITHDRAWAL AMT=100.00 BAL=450.00",0
"EJ005 NETWORK ERROR TIMEOUT HOST_UNREACHABLE RETRY=3",2
"EJ006 SECURITY ALERT CARD_RETAINED REASON=SUSPECTED_FRAUD",3
EOF

# Create README
echo "📝 Creating README..."
cat > README.md << 'EOF'
# BERT Debugging Dashboard

## Overview
This is a Dockerized web application for debugging and optimizing BERT models trained on Electronic Journal (EJ) logs.

## Features
- Token Attention Visualization
- Prediction Analysis
- Saliency Maps
- Embedding Visualization (t-SNE/UMAP)
- Performance Metrics
- Misclassification Explorer
- Token Bias Analysis

## Setup

### Prerequisites
- Docker and Docker Compose installed
- BERT model files (place in `models/bert_ej_model/`)

### Quick Start
1. Run the setup script:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

2. Place your BERT model files in `models/bert_ej_model/`

3. Start the application:
   ```bash
   docker-compose up --build
   ```

4. Access the dashboard at http://localhost:3333

### Model Requirements
Your BERT model directory should contain:
- `config.json`
- `pytorch_model.bin` or `model.safetensors`
- `tokenizer_config.json`
- `vocab.txt`

### Sample Data Format
CSV files should have the following columns:
- `text`: The EJ log text
- `label` (optional): True label for metrics calculation

## Port Configuration
- Frontend: 3333
- Backend API: 8000

## Troubleshooting
- If the model fails to load, the application will use a default BERT model
- Check Docker logs: `docker-compose logs -f`
- Ensure all ports are free before starting
EOF

echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Navigate to the project directory: cd $PROJECT_DIR"
echo "2. Place your BERT model files in: models/bert_ej_model/"
echo "3. Build and run: docker-compose up --build"
echo "4. Access the dashboard at: http://localhost:3333"
echo ""
echo "🎯 The application is configured to run on port 3333 to avoid conflicts."
EOF

# Make the script executable
chmod +x setup-bert-dashboard.sh

echo "✅ Setup script created successfully!"
echo ""
echo "To run the setup:"
echo "1. chmod +x setup-bert-dashboard.sh"
echo "2. ./setup-bert-dashboard.sh"
