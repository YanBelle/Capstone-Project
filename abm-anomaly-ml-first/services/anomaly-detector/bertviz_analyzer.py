"""
BertViz Integration for ABM Anomaly Detection
Visualizes BERT attention patterns to understand token importance and model behavior
"""

import torch
import numpy as np
from transformers import BertTokenizer, BertModel, BertConfig
from bertviz import head_view, model_view, neuron_view
from bertviz.transformers_neuron_view import BertNeuronView
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import json
import base64
from io import BytesIO
import logging
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

class BertVisualizationAnalyzer:
    """
    Analyzes BERT attention patterns and token importance for ABM anomaly detection
    """
    
    def __init__(self, model_name: str = 'bert-base-uncased', device: str = None):
        """
        Initialize the BERT visualization analyzer
        
        Args:
            model_name: BERT model name/path
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize BERT components
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name, output_attentions=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Store analysis results
        self.attention_cache = {}
        self.token_importance_cache = {}
        
        logger.info(f"BertViz analyzer initialized with {model_name} on {self.device}")
    
    def analyze_session_text(self, session_text: str, session_id: str = None) -> Dict[str, Any]:
        """
        Comprehensive analysis of session text including attention patterns and token importance
        
        Args:
            session_text: Raw session text to analyze
            session_id: Optional session identifier
            
        Returns:
            Dictionary containing all analysis results
        """
        try:
            # Preprocess text
            processed_text = self._preprocess_text(session_text)
            
            # Get BERT outputs with attention
            inputs, attention_weights, hidden_states = self._get_bert_outputs(processed_text)
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            # Perform various analyses
            analysis_results = {
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'text_length': len(session_text),
                'token_count': len(tokens),
                'processed_text': processed_text,
                'tokens': tokens,
                
                # Attention analysis
                'attention_analysis': self._analyze_attention_patterns(attention_weights, tokens),
                
                # Token importance
                'token_importance': self._calculate_token_importance(attention_weights, tokens),
                
                # Layer-wise analysis
                'layer_analysis': self._analyze_layers(attention_weights, hidden_states),
                
                # Head analysis
                'head_analysis': self._analyze_attention_heads(attention_weights, tokens),
                
                # Pattern detection
                'patterns': self._detect_attention_patterns(attention_weights, tokens),
                
                # Visualizations (base64 encoded)
                'visualizations': self._generate_visualizations(attention_weights, tokens, session_text)
            }
            
            # Cache results
            if session_id:
                self.attention_cache[session_id] = analysis_results
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing session text: {e}")
            return {'error': str(e), 'session_id': session_id}
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess ABM log text for BERT analysis"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Truncate to BERT's max length (512 tokens minus special tokens)
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > 510:  # Leave room for [CLS] and [SEP]
            tokens = tokens[:510]
            text = self.tokenizer.convert_tokens_to_string(tokens)
        
        return text
    
    def _get_bert_outputs(self, text: str) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
        """Get BERT outputs including attention weights and hidden states"""
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            attention_weights = outputs.attentions  # Tuple of attention tensors for each layer
            hidden_states = outputs.last_hidden_state
        
        return inputs, attention_weights, hidden_states
    
    def _analyze_attention_patterns(self, attention_weights: Tuple, tokens: List[str]) -> Dict[str, Any]:
        """Analyze attention patterns across layers and heads"""
        num_layers = len(attention_weights)
        num_heads = attention_weights[0].shape[1]
        seq_len = len(tokens)
        
        # Average attention across all heads and layers
        avg_attention = torch.stack(attention_weights).mean(dim=(0, 1)).squeeze().cpu().numpy()
        
        # Find tokens with highest attention
        token_attention_scores = avg_attention.sum(axis=0)  # Sum attention received by each token
        top_tokens_idx = np.argsort(token_attention_scores)[-10:]  # Top 10 tokens
        
        # Attention distribution analysis
        attention_entropy = self._calculate_attention_entropy(avg_attention)
        
        return {
            'num_layers': num_layers,
            'num_heads': num_heads,
            'sequence_length': seq_len,
            'average_attention_matrix': avg_attention.tolist(),
            'token_attention_scores': token_attention_scores.tolist(),
            'top_attended_tokens': [
                {'token': tokens[idx], 'score': float(token_attention_scores[idx]), 'position': int(idx)}
                for idx in top_tokens_idx[::-1]  # Reverse for descending order
            ],
            'attention_entropy': float(attention_entropy),
            'attention_concentration': float(np.max(token_attention_scores) / np.mean(token_attention_scores))
        }
    
    def _calculate_token_importance(self, attention_weights: Tuple, tokens: List[str]) -> Dict[str, Any]:
        """Calculate importance scores for each token"""
        # Method 1: Attention-based importance
        attention_importance = self._attention_based_importance(attention_weights)
        
        # Method 2: Gradient-based importance (approximated)
        gradient_importance = self._gradient_based_importance(attention_weights)
        
        # Method 3: Layer-wise importance
        layer_importance = self._layer_wise_importance(attention_weights)
        
        # Combine importance scores
        combined_importance = (
            0.4 * attention_importance + 
            0.3 * gradient_importance + 
            0.3 * layer_importance
        )
        
        # Create token importance rankings
        token_rankings = [
            {
                'token': token,
                'position': idx,
                'attention_importance': float(attention_importance[idx]),
                'gradient_importance': float(gradient_importance[idx]),
                'layer_importance': float(layer_importance[idx]),
                'combined_importance': float(combined_importance[idx])
            }
            for idx, token in enumerate(tokens)
        ]
        
        # Sort by combined importance
        token_rankings.sort(key=lambda x: x['combined_importance'], reverse=True)
        
        return {
            'token_rankings': token_rankings,
            'importance_statistics': {
                'mean_importance': float(np.mean(combined_importance)),
                'std_importance': float(np.std(combined_importance)),
                'max_importance': float(np.max(combined_importance)),
                'min_importance': float(np.min(combined_importance))
            }
        }
    
    def _attention_based_importance(self, attention_weights: Tuple) -> np.ndarray:
        """Calculate token importance based on attention patterns"""
        # Sum attention received from all other tokens across all layers and heads
        total_attention = torch.stack(attention_weights).sum(dim=(0, 1)).squeeze()
        return total_attention.sum(dim=0).cpu().numpy()  # Sum along attention-to dimension
    
    def _gradient_based_importance(self, attention_weights: Tuple) -> np.ndarray:
        """Approximate gradient-based importance using attention variance"""
        # Use attention variance as a proxy for gradient-based importance
        attention_stack = torch.stack(attention_weights)
        attention_var = attention_stack.var(dim=(0, 1)).squeeze()
        return attention_var.sum(dim=0).cpu().numpy()
    
    def _layer_wise_importance(self, attention_weights: Tuple) -> np.ndarray:
        """Calculate importance based on layer-wise attention evolution"""
        layer_attentions = []
        for layer_attention in attention_weights:
            layer_avg = layer_attention.mean(dim=1).squeeze()  # Average across heads
            layer_attentions.append(layer_avg.sum(dim=0).cpu().numpy())  # Sum across attention-to
        
        # Calculate importance as the change in attention across layers
        layer_changes = np.diff(layer_attentions, axis=0)
        return np.abs(layer_changes).sum(axis=0)
    
    def _analyze_layers(self, attention_weights: Tuple, hidden_states: torch.Tensor) -> Dict[str, Any]:
        """Analyze attention patterns across different BERT layers"""
        layer_analysis = []
        
        for layer_idx, layer_attention in enumerate(attention_weights):
            layer_avg = layer_attention.mean(dim=1).squeeze().cpu().numpy()  # Average across heads
            
            # Calculate layer-specific metrics
            layer_entropy = self._calculate_attention_entropy(layer_avg)
            layer_sparsity = np.sum(layer_avg < 0.01) / layer_avg.size  # Percentage of near-zero attention
            layer_max_attention = np.max(layer_avg)
            
            layer_analysis.append({
                'layer': layer_idx,
                'entropy': float(layer_entropy),
                'sparsity': float(layer_sparsity),
                'max_attention': float(layer_max_attention),
                'attention_matrix': layer_avg.tolist()
            })
        
        return {
            'layers': layer_analysis,
            'layer_progression': self._analyze_layer_progression(attention_weights)
        }
    
    def _analyze_attention_heads(self, attention_weights: Tuple, tokens: List[str]) -> Dict[str, Any]:
        """Analyze individual attention heads to understand their specialization"""
        head_analysis = []
        
        for layer_idx, layer_attention in enumerate(attention_weights):
            layer_attention = layer_attention.squeeze().cpu().numpy()  # Shape: [num_heads, seq_len, seq_len]
            
            for head_idx in range(layer_attention.shape[0]):
                head_attention = layer_attention[head_idx]
                
                # Analyze head behavior
                head_type = self._classify_attention_head(head_attention, tokens)
                head_entropy = self._calculate_attention_entropy(head_attention)
                
                head_analysis.append({
                    'layer': layer_idx,
                    'head': head_idx,
                    'type': head_type,
                    'entropy': float(head_entropy),
                    'max_attention': float(np.max(head_attention)),
                    'primary_focus': self._get_head_primary_focus(head_attention, tokens)
                })
        
        return {'heads': head_analysis}
    
    def _classify_attention_head(self, head_attention: np.ndarray, tokens: List[str]) -> str:
        """Classify attention head type based on its attention pattern"""
        # Simple heuristics to classify attention heads
        diagonal_strength = np.trace(head_attention) / np.sum(head_attention)
        
        if diagonal_strength > 0.3:
            return "self-attention"
        elif np.max(head_attention[0, :]) > 0.5:  # High attention to [CLS]
            return "classification-focused"
        elif np.max(head_attention[:, -1]) > 0.5:  # High attention to [SEP]
            return "delimiter-focused"
        else:
            return "content-focused"
    
    def _get_head_primary_focus(self, head_attention: np.ndarray, tokens: List[str]) -> Dict[str, Any]:
        """Get the primary focus of an attention head"""
        # Find the token that receives the most attention overall
        total_attention_received = head_attention.sum(axis=0)
        max_idx = np.argmax(total_attention_received)
        
        return {
            'token': tokens[max_idx] if max_idx < len(tokens) else '<UNK>',
            'position': int(max_idx),
            'attention_score': float(total_attention_received[max_idx])
        }
    
    def _detect_attention_patterns(self, attention_weights: Tuple, tokens: List[str]) -> Dict[str, Any]:
        """Detect specific attention patterns relevant to ABM anomaly detection"""
        patterns = {}
        
        # Average attention across all layers and heads
        avg_attention = torch.stack(attention_weights).mean(dim=(0, 1)).squeeze().cpu().numpy()
        
        # Pattern 1: Sequential attention (following word order)
        sequential_score = self._calculate_sequential_attention(avg_attention)
        patterns['sequential_attention'] = {
            'score': float(sequential_score),
            'description': 'How much the model follows sequential word order'
        }
        
        # Pattern 2: Error keyword attention
        error_keywords = ['error', 'fail', 'timeout', 'reject', 'abort', 'exception']
        error_attention = self._calculate_keyword_attention(avg_attention, tokens, error_keywords)
        patterns['error_attention'] = {
            'score': float(error_attention),
            'description': 'Attention to error-related keywords',
            'keywords_found': [token for token in tokens if any(keyword in token.lower() for keyword in error_keywords)]
        }
        
        # Pattern 3: Temporal attention (time-related tokens)
        temporal_keywords = ['time', 'date', 'hour', 'minute', 'second', 'am', 'pm']
        temporal_attention = self._calculate_keyword_attention(avg_attention, tokens, temporal_keywords)
        patterns['temporal_attention'] = {
            'score': float(temporal_attention),
            'description': 'Attention to temporal information'
        }
        
        # Pattern 4: Transaction attention
        transaction_keywords = ['withdraw', 'deposit', 'balance', 'card', 'account', 'pin']
        transaction_attention = self._calculate_keyword_attention(avg_attention, tokens, transaction_keywords)
        patterns['transaction_attention'] = {
            'score': float(transaction_attention),
            'description': 'Attention to transaction-related terms'
        }
        
        return patterns
    
    def _calculate_sequential_attention(self, attention_matrix: np.ndarray) -> float:
        """Calculate how much attention follows sequential order"""
        seq_len = attention_matrix.shape[0]
        sequential_weight = 0.0
        
        for i in range(seq_len):
            for j in range(max(0, i-2), min(seq_len, i+3)):  # Window of ±2 positions
                weight = 1.0 / (abs(i - j) + 1)  # Higher weight for closer positions
                sequential_weight += attention_matrix[i, j] * weight
        
        return sequential_weight / (seq_len * seq_len)
    
    def _calculate_keyword_attention(self, attention_matrix: np.ndarray, tokens: List[str], keywords: List[str]) -> float:
        """Calculate total attention to specific keywords"""
        keyword_attention = 0.0
        keyword_count = 0
        
        for idx, token in enumerate(tokens):
            if any(keyword in token.lower() for keyword in keywords):
                keyword_attention += attention_matrix[:, idx].sum()  # Sum attention to this token
                keyword_count += 1
        
        return keyword_attention / max(keyword_count, 1)  # Average attention per keyword
    
    def _calculate_attention_entropy(self, attention_matrix: np.ndarray) -> float:
        """Calculate entropy of attention distribution"""
        # Flatten and normalize attention weights
        attention_flat = attention_matrix.flatten()
        attention_prob = attention_flat / attention_flat.sum()
        
        # Calculate entropy
        entropy = -np.sum(attention_prob * np.log(attention_prob + 1e-10))
        return entropy
    
    def _analyze_layer_progression(self, attention_weights: Tuple) -> Dict[str, Any]:
        """Analyze how attention patterns evolve across layers"""
        layer_entropies = []
        layer_concentrations = []
        
        for layer_attention in attention_weights:
            layer_avg = layer_attention.mean(dim=1).squeeze().cpu().numpy()
            entropy = self._calculate_attention_entropy(layer_avg)
            concentration = np.max(layer_avg) / np.mean(layer_avg)
            
            layer_entropies.append(float(entropy))
            layer_concentrations.append(float(concentration))
        
        return {
            'entropy_progression': layer_entropies,
            'concentration_progression': layer_concentrations,
            'entropy_trend': 'increasing' if layer_entropies[-1] > layer_entropies[0] else 'decreasing',
            'concentration_trend': 'increasing' if layer_concentrations[-1] > layer_concentrations[0] else 'decreasing'
        }
    
    def _generate_visualizations(self, attention_weights: Tuple, tokens: List[str], original_text: str) -> Dict[str, str]:
        """Generate BertViz visualizations and encode as base64"""
        visualizations = {}
        
        try:
            # Prepare inputs for BertViz
            inputs = self.tokenizer(original_text, return_tensors='pt', truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate head view (attention between tokens)
            try:
                head_view_html = head_view(
                    self.model, 
                    self.tokenizer, 
                    original_text,
                    html_action='return'
                )
                visualizations['head_view'] = self._encode_html_to_base64(head_view_html)
            except Exception as e:
                logger.warning(f"Could not generate head view: {e}")
            
            # Generate model view (aggregated attention)
            try:
                model_view_html = model_view(
                    self.model,
                    self.tokenizer,
                    original_text,
                    html_action='return'
                )
                visualizations['model_view'] = self._encode_html_to_base64(model_view_html)
            except Exception as e:
                logger.warning(f"Could not generate model view: {e}")
            
            # Generate custom attention heatmap
            visualizations['attention_heatmap'] = self._generate_attention_heatmap(attention_weights, tokens)
            
            # Generate token importance visualization
            visualizations['token_importance'] = self._generate_token_importance_plot(attention_weights, tokens)
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            visualizations['error'] = str(e)
        
        return visualizations
    
    def _generate_attention_heatmap(self, attention_weights: Tuple, tokens: List[str]) -> str:
        """Generate custom attention heatmap"""
        try:
            # Average attention across all layers and heads
            avg_attention = torch.stack(attention_weights).mean(dim=(0, 1)).squeeze().cpu().numpy()
            
            # Create heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                avg_attention, 
                xticklabels=tokens, 
                yticklabels=tokens,
                cmap='Blues',
                cbar_kws={'label': 'Attention Weight'}
            )
            plt.title('BERT Attention Heatmap (Averaged across all layers and heads)')
            plt.xlabel('Tokens (To)')
            plt.ylabel('Tokens (From)')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Error generating attention heatmap: {e}")
            return ""
    
    def _generate_token_importance_plot(self, attention_weights: Tuple, tokens: List[str]) -> str:
        """Generate token importance bar plot"""
        try:
            # Calculate token importance
            importance_scores = self._attention_based_importance(attention_weights)
            
            # Create DataFrame for plotting
            df = pd.DataFrame({
                'Token': tokens,
                'Importance': importance_scores,
                'Position': range(len(tokens))
            })
            
            # Sort by importance and take top 20
            df_top = df.nlargest(20, 'Importance')
            
            # Create bar plot
            plt.figure(figsize=(12, 8))
            bars = plt.bar(range(len(df_top)), df_top['Importance'])
            plt.xticks(range(len(df_top)), df_top['Token'], rotation=45, ha='right')
            plt.ylabel('Attention Importance Score')
            plt.title('Top 20 Most Important Tokens (by Attention)')
            
            # Color bars by importance level
            max_importance = df_top['Importance'].max()
            for i, bar in enumerate(bars):
                importance_ratio = df_top.iloc[i]['Importance'] / max_importance
                if importance_ratio > 0.8:
                    bar.set_color('red')
                elif importance_ratio > 0.6:
                    bar.set_color('orange')
                elif importance_ratio > 0.4:
                    bar.set_color('yellow')
                else:
                    bar.set_color('green')
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Error generating token importance plot: {e}")
            return ""
    
    def _encode_html_to_base64(self, html_content: str) -> str:
        """Encode HTML content to base64"""
        try:
            html_bytes = html_content.encode('utf-8')
            return base64.b64encode(html_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding HTML to base64: {e}")
            return ""
    
    def compare_sessions(self, session1_id: str, session2_id: str) -> Dict[str, Any]:
        """Compare attention patterns between two sessions"""
        if session1_id not in self.attention_cache or session2_id not in self.attention_cache:
            return {'error': 'One or both sessions not found in cache'}
        
        analysis1 = self.attention_cache[session1_id]
        analysis2 = self.attention_cache[session2_id]
        
        # Compare key metrics
        comparison = {
            'session1_id': session1_id,
            'session2_id': session2_id,
            'attention_entropy_diff': analysis1['attention_analysis']['attention_entropy'] - analysis2['attention_analysis']['attention_entropy'],
            'attention_concentration_diff': analysis1['attention_analysis']['attention_concentration'] - analysis2['attention_analysis']['attention_concentration'],
            'token_count_diff': analysis1['token_count'] - analysis2['token_count'],
            'pattern_differences': {}
        }
        
        # Compare attention patterns
        for pattern_name in analysis1['patterns']:
            if pattern_name in analysis2['patterns']:
                comparison['pattern_differences'][pattern_name] = {
                    'session1_score': analysis1['patterns'][pattern_name]['score'],
                    'session2_score': analysis2['patterns'][pattern_name]['score'],
                    'difference': analysis1['patterns'][pattern_name]['score'] - analysis2['patterns'][pattern_name]['score']
                }
        
        return comparison
    
    def get_anomaly_attention_insights(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights about anomaly detection based on attention analysis"""
        insights = {
            'attention_quality': 'good',  # good, medium, poor
            'key_indicators': [],
            'recommendations': [],
            'confidence_factors': []
        }
        
        # Analyze attention quality
        entropy = analysis_results['attention_analysis']['attention_entropy']
        concentration = analysis_results['attention_analysis']['attention_concentration']
        
        if entropy < 2.0:  # Low entropy suggests focused attention
            insights['key_indicators'].append('Model shows focused attention (low entropy)')
            insights['confidence_factors'].append('High confidence due to focused attention')
        else:
            insights['key_indicators'].append('Model shows distributed attention (high entropy)')
            insights['attention_quality'] = 'medium'
        
        # Check error attention
        error_attention = analysis_results['patterns']['error_attention']['score']
        if error_attention > 0.1:
            insights['key_indicators'].append(f'High attention to error keywords (score: {error_attention:.3f})')
            insights['confidence_factors'].append('Model focusing on error-related terms')
        
        # Check temporal attention
        temporal_attention = analysis_results['patterns']['temporal_attention']['score']
        if temporal_attention > 0.05:
            insights['key_indicators'].append('Model paying attention to temporal information')
        
        # Generate recommendations
        if error_attention < 0.05:
            insights['recommendations'].append('Consider fine-tuning to increase attention to error keywords')
        
        if entropy > 3.0:
            insights['recommendations'].append('Attention appears too distributed - consider input preprocessing')
            insights['attention_quality'] = 'poor'
        
        if concentration < 2.0:
            insights['recommendations'].append('Low attention concentration - model may benefit from focused training')
        
        return insights
