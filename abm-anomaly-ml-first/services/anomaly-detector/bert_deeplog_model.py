http://localhost/dashboard/deeplog"""
BERT-Enhanced DeepLog Model for ABM EJ Log Anomaly Detection
Combines BERT embeddings with DeepLog sequential pattern learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import os
import pickle
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
import joblib
from sklearn.preprocessing import StandardScaler
import re

# Import BERT components
from transformers import BertTokenizer, BertModel
from bertviz_analyzer import BertVisualizationAnalyzer

logger = logging.getLogger(__name__)

class BertDeepLogLSTM(nn.Module):
    """
    Enhanced DeepLog LSTM that processes BERT embeddings instead of simple event tokens
    """
    
    def __init__(self, bert_dim=768, hidden_dim=128, num_layers=2, dropout=0.3):
        """
        Initialize BERT-enhanced DeepLog LSTM
        
        Args:
            bert_dim: BERT embedding dimension (768 for base, 1024 for large)
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(BertDeepLogLSTM, self).__init__()
        
        self.bert_dim = bert_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Project BERT embeddings to smaller dimension for efficiency
        self.bert_projection = nn.Linear(bert_dim, hidden_dim // 2)
        
        # LSTM for sequential pattern learning
        self.lstm = nn.LSTM(
            input_size=hidden_dim // 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layers for different prediction tasks
        self.anomaly_classifier = nn.Linear(hidden_dim, 2)  # Binary anomaly classification

        self.sequence_predictor = nn.Linear(hidden_dim, bert_dim)  # Next event prediction - match BERT embedding size
        self.attention_layer = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Dropout and normalization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, bert_embeddings, lengths=None):
        """
        Forward pass through the model
        
        Args:
            bert_embeddings: Tensor of shape (batch_size, seq_len, bert_dim)
            lengths: Optional sequence lengths for padding
            
        Returns:
            Dictionary containing various outputs
        """
        batch_size, seq_len, _ = bert_embeddings.shape
        
        # Project BERT embeddings
        projected = self.bert_projection(bert_embeddings)  # (batch_size, seq_len, hidden_dim//2)
        projected = torch.relu(projected)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(projected)  # (batch_size, seq_len, hidden_dim)
        
        # Apply attention mechanism
        attended_out, attention_weights = self.attention_layer(lstm_out, lstm_out, lstm_out)
        
        # Combine LSTM and attention outputs
        combined = lstm_out + attended_out
        combined = self.layer_norm(combined)
        combined = self.dropout(combined)
        
        # Generate predictions
        anomaly_logits = self.anomaly_classifier(combined)  # (batch_size, seq_len, 2)
        sequence_pred = self.sequence_predictor(combined)   # (batch_size, seq_len, bert_dim)
        
        return {
            'anomaly_logits': anomaly_logits,
            'sequence_predictions': sequence_pred,
            'lstm_hidden': hidden,
            'lstm_cell': cell,
            'attention_weights': attention_weights,
            'combined_features': combined
        }

class BertDeepLogAnalyzer:
    """
    Main analyzer that combines BERT preprocessing with DeepLog sequential learning
    """
    
    def __init__(self, model_dir="/app/data/models", bert_model_name='bert-base-uncased'):
        """
        Initialize the BERT-DeepLog analyzer
        
        Args:
            model_dir: Directory to store trained models
            bert_model_name: BERT model name for embeddings
        """
        self.model_dir = model_dir
        self.bert_model_name = bert_model_name
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize BERT components
        self.bert_analyzer = BertVisualizationAnalyzer(model_name=bert_model_name)
        
        # Initialize DeepLog model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BertDeepLogLSTM().to(self.device)
        self.model_trained = False
        
        # Training parameters
        self.window_size = 10  # Sequence window for training
        self.batch_size = 16
        self.learning_rate = 0.001
        self.num_epochs = 50
        
        # Scaler for embeddings
        self.scaler = StandardScaler()
        self.scaler_fitted = False
        
        # Training data storage
        self.training_sequences = []
        self.training_labels = []
        self.event_vocabulary = {}
        self.reverse_vocabulary = {}
        
        # Anomaly detection parameters
        self.anomaly_threshold = 0.7
        self.sequence_threshold = 0.5
        
        # Performance tracking
        self.training_history = []
        self.prediction_cache = {}
        
        logger.info(f"BertDeepLogAnalyzer initialized with device: {self.device}")
    
    def prepare_training_data(self, ej_sessions: List[Dict], normal_sessions_only=True):
        """
        Prepare training data from EJ sessions using BERT embeddings
        
        Args:
            ej_sessions: List of EJ session dictionaries
            normal_sessions_only: If True, only use sessions labeled as normal for training
        """
        logger.info(f"Preparing training data from {len(ej_sessions)} sessions")
        
        sequences = []
        labels = []
        all_embeddings = []
        
        for session in ej_sessions:
            # Skip anomalous sessions if only training on normal data
            if normal_sessions_only and session.get('is_anomaly', False):
                continue
            
            # Get BERT embeddings for the session
            session_text = session.get('raw_text', session.get('text', ''))
            if not session_text.strip():
                continue
            
            try:
                # Use BERT analyzer to get cleaned text and embeddings
                analysis_result = self.bert_analyzer.analyze_session_text(
                    session_text, 
                    session_id=session.get('session_id', f'session_{len(sequences)}')
                )
                
                if 'error' in analysis_result:
                    logger.warning(f"Failed to analyze session: {analysis_result['error']}")
                    continue
                
                # Extract event embeddings from token importance
                token_rankings = analysis_result['token_importance']['token_rankings']
                
                # Create sequence of important event embeddings
                event_sequence = []
                for token_info in token_rankings[:self.window_size]:  # Take top window_size tokens
                    # Create a simple embedding representation for the token
                    embedding = self._create_token_embedding(
                        token_info['token'], 
                        token_info['combined_importance']
                    )
                    event_sequence.append(embedding)
                
                if len(event_sequence) >= 3:  # Minimum sequence length
                    # Pad sequence to window_size
                    while len(event_sequence) < self.window_size:
                        event_sequence.append(np.zeros(768))  # BERT dimension
                    
                    sequences.append(np.array(event_sequence))
                    labels.append(0 if not session.get('is_anomaly', False) else 1)
                    all_embeddings.extend(event_sequence)
                
            except Exception as e:
                logger.error(f"Error processing session {session.get('session_id', 'unknown')}: {e}")
                continue
        
        # Fit scaler on all embeddings
        if all_embeddings and not self.scaler_fitted:
            all_embeddings_array = np.array(all_embeddings)
            self.scaler.fit(all_embeddings_array)
            self.scaler_fitted = True
            logger.info("Fitted scaler on embedding data")
        
        # Scale sequences
        scaled_sequences = []
        for seq in sequences:
            scaled_seq = self.scaler.transform(seq)
            scaled_sequences.append(scaled_seq)
        
        self.training_sequences = scaled_sequences
        self.training_labels = labels
        
        logger.info(f"Prepared {len(self.training_sequences)} training sequences")
        return len(self.training_sequences)
    
    def _create_token_embedding(self, token: str, importance: float) -> np.ndarray:
        """
        Create a simple embedding for a token with importance weighting
        """
        # Use BERT to get token embedding
        try:
            inputs = self.bert_analyzer.tokenizer(
                token, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=8
            )
            
            with torch.no_grad():
                outputs = self.bert_analyzer.model(**inputs)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[0, 0, :].numpy()
                
                # Weight by importance
                embedding = embedding * importance
                
                return embedding
        except:
            # Fallback to random embedding
            return np.random.normal(0, 0.1, 768)
    
    def train_model(self, validation_split=0.2):
        """
        Train the BERT-DeepLog model
        
        Args:
            validation_split: Fraction of data to use for validation
        """
        if not self.training_sequences:
            raise ValueError("No training data available. Call prepare_training_data() first.")
        
        logger.info("Starting BERT-DeepLog model training")
        
        # Prepare data
        X = np.array(self.training_sequences)
        y = np.array(self.training_labels)
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        # Initialize optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        sequence_criterion = nn.MSELoss()
        
        # Training loop
        self.training_history = []
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            self.model.train()
            train_losses = []
            
            # Mini-batch training
            for i in range(0, len(X_train), self.batch_size):
                batch_X = X_train_tensor[i:i+self.batch_size]
                batch_y = y_train_tensor[i:i+self.batch_size]
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X)
                
                # Calculate losses
                # Anomaly classification loss
                anomaly_loss = criterion(
                    outputs['anomaly_logits'].view(-1, 2),
                    batch_y.unsqueeze(1).expand(-1, batch_X.size(1)).contiguous().view(-1)
                )
                
                # Sequence prediction loss (predict next embedding)
                if batch_X.size(1) > 1:
                    # Project the target embeddings to match prediction dimensions (64-dim)
                    sequence_targets = self.model.bert_projection(batch_X[:, 1:, :])
                    sequence_preds = outputs['sequence_predictions'][:, :-1, :]
                    sequence_loss = sequence_criterion(sequence_preds, sequence_targets)
                else:
                    sequence_loss = torch.tensor(0.0).to(self.device)
                
                # Combined loss
                total_loss = anomaly_loss + 0.3 * sequence_loss
                
                total_loss.backward()
                optimizer.step()
                
                train_losses.append(total_loss.item())
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_anomaly_loss = criterion(
                    val_outputs['anomaly_logits'].view(-1, 2),
                    y_val_tensor.unsqueeze(1).expand(-1, X_val_tensor.size(1)).contiguous().view(-1)
                )
                val_loss = val_anomaly_loss.item()
            
            # Record history
            epoch_history = {
                'epoch': epoch + 1,
                'train_loss': np.mean(train_losses),
                'val_loss': val_loss,
                'timestamp': datetime.now().isoformat()
            }
            self.training_history.append(epoch_history)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.num_epochs}, "
                          f"Train Loss: {np.mean(train_losses):.4f}, "
                          f"Val Loss: {val_loss:.4f}")
        
        self.model_trained = True
        logger.info("BERT-DeepLog model training completed")
        
        return self.training_history
    
    def predict_anomaly(self, session_text: str, session_id: str = None) -> Dict[str, Any]:
        """
        Predict if a session is anomalous using the trained model
        
        Args:
            session_text: Raw EJ session text
            session_id: Optional session identifier
            
        Returns:
            Dictionary containing prediction results and explanations
        """
        if not self.model_trained:
            try:
                self.load_model()
            except:
                raise ValueError("Model not trained. Call train_model() first or ensure saved model exists.")
        
        try:
            # Analyze session with BERT
            analysis_result = self.bert_analyzer.analyze_session_text(session_text, session_id)
            
            if 'error' in analysis_result:
                return {'error': analysis_result['error'], 'session_id': session_id}
            
            # Extract event sequence
            token_rankings = analysis_result['token_importance']['token_rankings']
            
            event_sequence = []
            important_events = []
            
            for token_info in token_rankings[:self.window_size]:
                embedding = self._create_token_embedding(
                    token_info['token'],
                    token_info['combined_importance']
                )
                event_sequence.append(embedding)
                important_events.append({
                    'token': token_info['token'],
                    'importance': token_info['combined_importance'],
                    'position': token_info['position'],
                    'attention_importance': token_info['attention_importance'],
                    'contextual_importance': token_info['contextual_importance']
                })
            
            # Pad sequence
            while len(event_sequence) < self.window_size:
                event_sequence.append(np.zeros(768))
            
            # Scale and convert to tensor
            sequence_array = np.array([event_sequence])
            if self.scaler_fitted:
                sequence_scaled = self.scaler.transform(sequence_array.reshape(-1, 768)).reshape(sequence_array.shape)
            else:
                sequence_scaled = sequence_array
            
            sequence_tensor = torch.FloatTensor(sequence_scaled).to(self.device)
            
            # Predict
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(sequence_tensor)
                
                # Get anomaly probabilities
                anomaly_probs = torch.softmax(outputs['anomaly_logits'], dim=-1)
                anomaly_prob = anomaly_probs[0, :, 1].mean().item()  # Average over sequence
                
                # Get attention weights for explanation
                attention_weights = outputs['attention_weights'][0].cpu().numpy()
                
                # Determine if anomalous
                is_anomaly = anomaly_prob > self.anomaly_threshold
                
                # Create detailed prediction result
                prediction_result = {
                    'session_id': session_id,
                    'is_anomaly': bool(is_anomaly),
                    'anomaly_probability': float(anomaly_prob),
                    'confidence': float(abs(anomaly_prob - 0.5) * 2),  # Distance from 0.5
                    'threshold_used': self.anomaly_threshold,
                    'prediction_timestamp': datetime.now().isoformat(),
                    
                    # Explanation data
                    'important_events': important_events,
                    'sequence_length': len([e for e in important_events if e['importance'] > 0]),
                    'attention_patterns': attention_weights.tolist(),
                    
                    # BERT analysis data
                    'bert_analysis': {
                        'token_count': analysis_result['token_count'],
                        'attention_entropy': analysis_result['attention_analysis']['attention_entropy'],
                        'attention_concentration': analysis_result['attention_analysis']['attention_concentration'],
                        'error_attention_score': analysis_result['patterns']['error_attention']['score'],
                        'transaction_attention_score': analysis_result['patterns']['transaction_attention']['score']
                    },
                    
                    # Model internals for debugging
                    'model_outputs': {
                        'raw_anomaly_logits': outputs['anomaly_logits'][0].cpu().numpy().tolist(),
                        'sequence_predictions_norm': torch.norm(outputs['sequence_predictions'][0], dim=-1).cpu().numpy().tolist()
                    }
                }
                
                # Cache prediction
                if session_id:
                    self.prediction_cache[session_id] = prediction_result
                
                return prediction_result
                
        except Exception as e:
            logger.error(f"Error predicting anomaly for session {session_id}: {e}")
            return {
                'error': str(e),
                'session_id': session_id,
                'is_anomaly': False,
                'anomaly_probability': 0.0
            }
    
    def explain_prediction(self, session_id: str) -> Dict[str, Any]:
        """
        Provide detailed explanation for a prediction
        """
        if session_id not in self.prediction_cache:
            return {'error': 'Session not found in prediction cache'}
        
        prediction = self.prediction_cache[session_id]
        
        # Generate explanation
        explanation = {
            'session_id': session_id,
            'prediction_summary': {
                'is_anomaly': prediction['is_anomaly'],
                'confidence': prediction['confidence'],
                'key_factors': []
            },
            'event_analysis': [],
            'attention_analysis': {},
            'model_reasoning': []
        }
        
        # Analyze important events
        for event in prediction['important_events']:
            event_analysis = {
                'event': event['token'],
                'importance_score': event['importance'],
                'contribution_type': self._classify_event_contribution(event),
                'explanation': self._explain_event_importance(event)
            }
            explanation['event_analysis'].append(event_analysis)
        
        # Add model reasoning
        if prediction['is_anomaly']:
            explanation['model_reasoning'].extend([
                f"Anomaly probability ({prediction['anomaly_probability']:.3f}) exceeds threshold ({prediction['threshold_used']})",
                f"Model confidence: {prediction['confidence']:.3f}",
                f"Key contributing events: {', '.join([e['token'] for e in prediction['important_events'][:3]])}"
            ])
        else:
            explanation['model_reasoning'].extend([
                f"Anomaly probability ({prediction['anomaly_probability']:.3f}) below threshold ({prediction['threshold_used']})",
                f"Session appears to follow normal patterns",
                f"Model confidence: {prediction['confidence']:.3f}"
            ])
        
        return explanation
    
    def _classify_event_contribution(self, event: Dict) -> str:
        """Classify how an event contributes to the prediction"""
        if event['importance'] > 0.8:
            return "Critical"
        elif event['importance'] > 0.5:
            return "High"
        elif event['importance'] > 0.3:
            return "Medium"
        else:
            return "Low"
    
    def _explain_event_importance(self, event: Dict) -> str:
        """Generate human-readable explanation for event importance"""
        token = event['token'].lower()
        importance = event['importance']
        
        if 'error' in token or 'fail' in token:
            return f"Error-related event with high importance ({importance:.3f})"
        elif 'card' in token:
            return f"Card-related event contributing to transaction pattern ({importance:.3f})"
        elif 'pin' in token:
            return f"PIN-related event in authentication sequence ({importance:.3f})"
        elif 'cash' in token or 'dispense' in token:
            return f"Cash dispensing event affecting transaction outcome ({importance:.3f})"
        else:
            return f"Event contributing to sequence pattern ({importance:.3f})"
    
    def save_model(self, model_path: str = None):
        """Save the trained model and associated data"""
        if model_path is None:
            model_path = os.path.join(self.model_dir, 'bert_deeplog_model.pth')
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_trained': self.model_trained,
            'training_history': self.training_history,
            'window_size': self.window_size,
            'anomaly_threshold': self.anomaly_threshold,
            'sequence_threshold': self.sequence_threshold,
            'scaler_fitted': self.scaler_fitted
        }, model_path)
        
        # Save scaler separately
        if self.scaler_fitted:
            scaler_path = os.path.join(self.model_dir, 'bert_deeplog_scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
        
        # Save configuration
        config = {
            'bert_model_name': self.bert_model_name,
            'model_architecture': {
                'bert_dim': self.model.bert_dim,
                'hidden_dim': self.model.hidden_dim,
                'num_layers': self.model.num_layers
            },
            'training_params': {
                'window_size': self.window_size,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'num_epochs': self.num_epochs
            },
            'save_timestamp': datetime.now().isoformat()
        }
        
        config_path = os.path.join(self.model_dir, 'bert_deeplog_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str = None):
        """Load a saved model"""
        if model_path is None:
            model_path = os.path.join(self.model_dir, 'bert_deeplog_model.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model state
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model_trained = checkpoint['model_trained']
        self.training_history = checkpoint.get('training_history', [])
        self.window_size = checkpoint.get('window_size', self.window_size)
        self.anomaly_threshold = checkpoint.get('anomaly_threshold', self.anomaly_threshold)
        self.sequence_threshold = checkpoint.get('sequence_threshold', self.sequence_threshold)
        self.scaler_fitted = checkpoint.get('scaler_fitted', False)
        
        # Load scaler
        if self.scaler_fitted:
            scaler_path = os.path.join(self.model_dir, 'bert_deeplog_scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
        
        logger.info(f"Model loaded from {model_path}")
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get comprehensive model statistics"""
        return {
            'model_info': {
                'trained': self.model_trained,
                'device': str(self.device),
                'parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            },
            'training_data': {
                'num_sequences': len(self.training_sequences),
                'sequence_length': self.window_size,
                'scaler_fitted': self.scaler_fitted
            },
            'hyperparameters': {
                'window_size': self.window_size,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'num_epochs': self.num_epochs,
                'anomaly_threshold': self.anomaly_threshold
            },
            'performance': {
                'training_history_length': len(self.training_history),
                'cached_predictions': len(self.prediction_cache)
            }
        }
