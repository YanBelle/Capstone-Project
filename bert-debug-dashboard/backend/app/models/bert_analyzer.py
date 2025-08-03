import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import List, Dict, Tuple, Any
import time

class BERTAnalyzer:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        except:
            # Use DistilBERT for faster inference - much smaller and faster than BERT
            print(f"Warning: Could not load model from {model_path}, using distilbert-base-uncased for faster debugging")
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4)
        
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")
        
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Fast analysis of input text for debugging"""
        start_time = time.time()
        try:
            print(f"Starting analysis for text: {text[:50]}...")
            
            # Tokenize with shorter max length for speed
            inputs = self.tokenizer(text, return_tensors="pt", 
                                   padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            print(f"Tokenization completed in {time.time() - start_time:.2f}s")
            
            # Get model outputs - only get what we need for speed
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
            
            print(f"Model inference completed in {time.time() - start_time:.2f}s")
            
            # Get predictions
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
            
            print(f"Predictions computed: class={predicted_class}")
            
            # Fast attention extraction - only last layer, first head
            attention_weights = self._extract_attention_fast(outputs.attentions)
            
            print(f"Attention weights extracted in {time.time() - start_time:.2f}s")
            
            # Simple token importance - just use attention weights
            token_importance = self._get_token_importance_fast(outputs.attentions)
            
            print(f"Token importance computed in {time.time() - start_time:.2f}s")
            
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            result = {
                "text": text,
                "tokens": tokens,
                "predicted_class": predicted_class,
                "probabilities": probs[0].tolist(),
                "attention_weights": attention_weights,
                "token_importance": token_importance,
                "hidden_states": {"cls_embeddings": []},  # Empty for speed
                "analysis_time": f"{time.time() - start_time:.2f}s"
            }
            
            print(f"Analysis completed successfully in {time.time() - start_time:.2f}s")
            return result
            
        except Exception as e:
            print(f"Error in analyze_text: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return a basic response if analysis fails
            try:
                tokens = self.tokenizer.tokenize(text)
            except:
                tokens = text.split()
            
            return {
                "text": text,
                "tokens": tokens,
                "predicted_class": 0,
                "probabilities": [0.25, 0.25, 0.25, 0.25],
                "attention_weights": [],
                "token_importance": [0.5] * len(tokens),
                "hidden_states": {"cls_embeddings": []},
                "error": str(e),
                "analysis_time": f"{time.time() - start_time:.2f}s"
            }
    
    def _extract_attention_fast(self, attentions: Tuple) -> List[Dict]:
        """Fast attention extraction - only last layer, first head"""
        try:
            last_layer_attention = attentions[-1]
            head_attention = last_layer_attention[0, 0].cpu().numpy()  # First head only
            
            return [{
                "layer": len(attentions) - 1,
                "heads": [{
                    "head": 0,
                    "attention": head_attention.tolist()
                }]
            }]
        except Exception as e:
            print(f"Error extracting attention weights: {e}")
            return []
    
    def _get_token_importance_fast(self, attentions: Tuple) -> List[float]:
        """Fast token importance using attention weights"""
        try:
            # Use attention weights from last layer, first head as importance
            last_layer_attention = attentions[-1]
            attention_weights = last_layer_attention[0, 0].cpu().numpy()
            
            # Average attention to each token (excluding [CLS] attention to itself)
            importance = np.mean(attention_weights[1:], axis=0)  # Skip [CLS] token's attention
            
            # Normalize
            if importance.max() > importance.min():
                importance = (importance - importance.min()) / (importance.max() - importance.min())
            else:
                importance = np.ones_like(importance) * 0.5
                
            return importance.tolist()
        except Exception as e:
            print(f"Error in token importance calculation: {e}")
            return [0.5] * 10  # Default fallback
