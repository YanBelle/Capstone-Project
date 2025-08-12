"""
DeepLog + BERT Training Solution for ABM Anomaly Detection
Combines BERT token embeddings with DeepLog sequence modeling for enhanced anomaly detection
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import pickle
import joblib
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DeepLogConfig:
    """Configuration for DeepLog training"""
    sequence_length: int = 64
    embedding_dim: int = 768  # BERT base embedding dimension
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    patience: int = 10
    anomaly_threshold: float = 0.5
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class DeepLogLSTM(nn.Module):
    """
    DeepLog LSTM model that accepts BERT embeddings as input
    Predicts next token/event in sequence for anomaly detection
    """
    
    def __init__(self, config: DeepLogConfig, vocab_size: int):
        super(DeepLogLSTM, self).__init__()
        self.config = config
        self.vocab_size = vocab_size
        
        # LSTM layers for sequence modeling
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Output projection layer
        self.output_projection = nn.Linear(config.hidden_dim, vocab_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, embeddings: torch.Tensor, hidden: Optional[Tuple] = None):
        """
        Forward pass through the DeepLog model
        
        Args:
            embeddings: BERT embeddings [batch_size, seq_len, embedding_dim]
            hidden: Hidden state for LSTM
            
        Returns:
            predictions: Next token predictions [batch_size, seq_len, vocab_size]
            hidden: Updated hidden state
        """
        # Apply dropout to embeddings
        embeddings = self.dropout(embeddings)
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(embeddings, hidden)
        
        # Apply dropout to LSTM output
        lstm_out = self.dropout(lstm_out)
        
        # Project to vocabulary size
        predictions = self.output_projection(lstm_out)
        
        return predictions, hidden
    
    def predict_next_token(self, embeddings: torch.Tensor):
        """
        Predict the next token given a sequence of embeddings
        
        Args:
            embeddings: Input embeddings [batch_size, seq_len, embedding_dim]
            
        Returns:
            predictions: Next token probabilities [batch_size, vocab_size]
        """
        self.eval()
        with torch.no_grad():
            predictions, _ = self.forward(embeddings)
            # Return the last timestep prediction
            return torch.softmax(predictions[:, -1, :], dim=-1)
    
    def compute_anomaly_score(self, embeddings: torch.Tensor, target_tokens: torch.Tensor):
        """
        Compute anomaly score based on prediction error
        
        Args:
            embeddings: Input embeddings [batch_size, seq_len, embedding_dim]
            target_tokens: Target token indices [batch_size, seq_len]
            
        Returns:
            anomaly_scores: Anomaly scores for each sequence [batch_size]
        """
        self.eval()
        with torch.no_grad():
            predictions, _ = self.forward(embeddings)
            
            # Compute cross-entropy loss for each sequence
            criterion = nn.CrossEntropyLoss(reduction='none')
            losses = []
            
            for i in range(predictions.size(1)):
                loss = criterion(predictions[:, i, :], target_tokens[:, i])
                losses.append(loss)
            
            # Average loss across sequence
            sequence_losses = torch.stack(losses, dim=1).mean(dim=1)
            
            return sequence_losses.cpu().numpy()

class BERTDeepLogTrainer:
    """
    Trainer class for BERT + DeepLog integration
    Handles data preprocessing, training, and model persistence
    """
    
    def __init__(self, config: DeepLogConfig, model_save_path: str = "/app/models"):
        self.config = config
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(exist_ok=True)
        
        # Initialize BERT tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model.eval()
        
        # Initialize components
        self.token_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.deeplog_model = None
        self.vocab_size = 0
        
        # Move BERT to device
        self.device = torch.device(config.device)
        self.bert_model.to(self.device)
        
        logger.info(f"Initialized BERT-DeepLog trainer on device: {self.device}")
    
    def preprocess_sessions(self, sessions: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess session texts into BERT embeddings and token sequences
        
        Args:
            sessions: List of session text strings
            
        Returns:
            embeddings: BERT embeddings array [num_sequences, seq_len, embedding_dim]
            token_sequences: Token ID sequences [num_sequences, seq_len]
        """
        logger.info(f"Preprocessing {len(sessions)} sessions for BERT-DeepLog training")
        
        all_embeddings = []
        all_token_sequences = []
        
        for session_text in sessions:
            # Tokenize session text
            tokens = self.tokenizer.tokenize(session_text.lower())
            
            # Create sliding windows of sequences
            sequences = self._create_sequences(tokens)
            
            for sequence in sequences:
                # Convert to token IDs
                token_ids = self.tokenizer.convert_tokens_to_ids(sequence)
                
                # Get BERT embeddings
                embeddings = self._get_bert_embeddings(sequence)
                
                if embeddings is not None and len(token_ids) == self.config.sequence_length:
                    all_embeddings.append(embeddings)
                    all_token_sequences.append(token_ids)
        
        if not all_embeddings:
            raise ValueError("No valid sequences found for training")
        
        # Convert to numpy arrays
        embeddings_array = np.stack(all_embeddings)
        token_sequences_array = np.array(all_token_sequences)
        
        # Fit token encoder on all unique tokens
        all_tokens = token_sequences_array.flatten()
        self.token_encoder.fit(all_tokens)
        self.vocab_size = len(self.token_encoder.classes_)
        
        # Encode token sequences
        encoded_sequences = np.array([
            self.token_encoder.transform(seq) for seq in token_sequences_array
        ])
        
        logger.info(f"Created {len(embeddings_array)} training sequences")
        logger.info(f"Vocabulary size: {self.vocab_size}")
        
        return embeddings_array, encoded_sequences
    
    def _create_sequences(self, tokens: List[str]) -> List[List[str]]:
        """Create sliding window sequences from tokens"""
        sequences = []
        seq_len = self.config.sequence_length
        
        if len(tokens) < seq_len:
            # Pad short sequences
            padded_tokens = tokens + ['[PAD]'] * (seq_len - len(tokens))
            sequences.append(padded_tokens)
        else:
            # Create sliding windows
            for i in range(len(tokens) - seq_len + 1):
                sequences.append(tokens[i:i + seq_len])
        
        return sequences
    
    def _get_bert_embeddings(self, tokens: List[str]) -> Optional[np.ndarray]:
        """Get BERT embeddings for a sequence of tokens"""
        try:
            # Convert tokens to input IDs
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            
            # Pad or truncate to sequence length
            if len(input_ids) < self.config.sequence_length:
                input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.config.sequence_length - len(input_ids))
            else:
                input_ids = input_ids[:self.config.sequence_length]
            
            # Convert to tensor
            input_tensor = torch.tensor([input_ids]).to(self.device)
            
            # Get BERT embeddings
            with torch.no_grad():
                outputs = self.bert_model(input_tensor)
                embeddings = outputs.last_hidden_state[0]  # [seq_len, embedding_dim]
            
            return embeddings.cpu().numpy()
        
        except Exception as e:
            logger.warning(f"Failed to get BERT embeddings: {e}")
            return None
    
    def train_deeplog_model(self, embeddings: np.ndarray, token_sequences: np.ndarray) -> Dict[str, Any]:
        """
        Train the DeepLog model on BERT embeddings
        
        Args:
            embeddings: BERT embeddings [num_sequences, seq_len, embedding_dim]
            token_sequences: Encoded token sequences [num_sequences, seq_len]
            
        Returns:
            training_history: Dictionary with training metrics
        """
        logger.info("Starting DeepLog model training")
        
        # Initialize model
        self.deeplog_model = DeepLogLSTM(self.config, self.vocab_size)
        self.deeplog_model.to(self.device)
        
        # Initialize optimizer and loss function
        optimizer = optim.Adam(self.deeplog_model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Convert to tensors
        embeddings_tensor = torch.FloatTensor(embeddings).to(self.device)
        sequences_tensor = torch.LongTensor(token_sequences).to(self.device)
        
        # Training history
        history = {
            'train_loss': [],
            'best_loss': float('inf'),
            'patience_counter': 0
        }
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.deeplog_model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            # Mini-batch training
            for i in range(0, len(embeddings_tensor), self.config.batch_size):
                batch_embeddings = embeddings_tensor[i:i + self.config.batch_size]
                batch_sequences = sequences_tensor[i:i + self.config.batch_size]
                
                # Input: all but last token, Target: all but first token
                input_embeddings = batch_embeddings[:, :-1, :]
                target_sequences = batch_sequences[:, 1:]
                
                # Forward pass
                optimizer.zero_grad()
                predictions, _ = self.deeplog_model(input_embeddings)
                
                # Compute loss
                loss = 0
                for t in range(predictions.size(1)):
                    loss += criterion(predictions[:, t, :], target_sequences[:, t])
                loss = loss / predictions.size(1)  # Average over sequence length
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            history['train_loss'].append(avg_loss)
            
            # Early stopping
            if avg_loss < history['best_loss']:
                history['best_loss'] = avg_loss
                history['patience_counter'] = 0
                # Save best model
                self.save_model()
            else:
                history['patience_counter'] += 1
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
            
            if history['patience_counter'] >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        logger.info(f"Training completed. Best loss: {history['best_loss']:.4f}")
        return history
    
    def predict_anomalies(self, sessions: List[str]) -> List[Dict[str, Any]]:
        """
        Predict anomalies in new sessions using trained DeepLog model
        
        Args:
            sessions: List of session text strings
            
        Returns:
            predictions: List of prediction dictionaries
        """
        if self.deeplog_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info(f"Predicting anomalies for {len(sessions)} sessions")
        
        predictions = []
        
        for i, session_text in enumerate(sessions):
            try:
                # Preprocess session
                tokens = self.tokenizer.tokenize(session_text.lower())
                sequences = self._create_sequences(tokens)
                
                session_anomaly_scores = []
                
                for sequence in sequences:
                    # Get BERT embeddings
                    embeddings = self._get_bert_embeddings(sequence)
                    
                    if embeddings is not None:
                        # Convert to tensor
                        embeddings_tensor = torch.FloatTensor(embeddings).unsqueeze(0).to(self.device)
                        
                        # Get token sequence
                        token_ids = self.tokenizer.convert_tokens_to_ids(sequence)
                        
                        # Ensure sequence length matches
                        if len(token_ids) == self.config.sequence_length:
                            # Encode tokens
                            try:
                                encoded_tokens = self.token_encoder.transform(token_ids)
                                target_tensor = torch.LongTensor([encoded_tokens]).to(self.device)
                                
                                # Compute anomaly score
                                anomaly_score = self.deeplog_model.compute_anomaly_score(
                                    embeddings_tensor, target_tensor
                                )[0]
                                
                                session_anomaly_scores.append(anomaly_score)
                            
                            except ValueError as e:
                                # Handle unseen tokens
                                logger.warning(f"Unseen tokens in sequence: {e}")
                                continue
                
                # Aggregate anomaly scores for session
                if session_anomaly_scores:
                    max_score = max(session_anomaly_scores)
                    avg_score = np.mean(session_anomaly_scores)
                    is_anomaly = max_score > self.config.anomaly_threshold
                else:
                    max_score = avg_score = 0.0
                    is_anomaly = False
                
                prediction = {
                    'session_id': f"session_{i}",
                    'is_anomaly': is_anomaly,
                    'anomaly_score': max_score,
                    'avg_anomaly_score': avg_score,
                    'num_sequences': len(session_anomaly_scores),
                    'anomaly_type': 'deeplog_sequence' if is_anomaly else None,
                    'confidence': 1.0 - min(avg_score, 1.0)
                }
                
                predictions.append(prediction)
            
            except Exception as e:
                logger.error(f"Error processing session {i}: {e}")
                predictions.append({
                    'session_id': f"session_{i}",
                    'is_anomaly': False,
                    'anomaly_score': 0.0,
                    'error': str(e)
                })
        
        return predictions
    
    def save_model(self):
        """Save the trained model and associated components"""
        model_dir = self.model_save_path / "deeplog_bert"
        model_dir.mkdir(exist_ok=True)
        
        # Save DeepLog model
        if self.deeplog_model is not None:
            torch.save(
                self.deeplog_model.state_dict(),
                model_dir / "deeplog_model.pt"
            )
        
        # Save token encoder
        joblib.dump(self.token_encoder, model_dir / "token_encoder.pkl")
        
        # Save configuration
        config_dict = {
            'sequence_length': self.config.sequence_length,
            'embedding_dim': self.config.embedding_dim,
            'hidden_dim': self.config.hidden_dim,
            'num_layers': self.config.num_layers,
            'dropout': self.config.dropout,
            'vocab_size': self.vocab_size,
            'anomaly_threshold': self.config.anomaly_threshold
        }
        
        with open(model_dir / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save training metadata
        metadata = {
            'trained_at': datetime.now().isoformat(),
            'bert_model': 'bert-base-uncased',
            'vocab_size': self.vocab_size,
            'device': str(self.device)
        }
        
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_dir}")
    
    def load_model(self):
        """Load a previously trained model"""
        model_dir = self.model_save_path / "deeplog_bert"
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Load configuration
        with open(model_dir / "config.json", 'r') as f:
            config_dict = json.load(f)
        
        self.vocab_size = config_dict['vocab_size']
        self.config.anomaly_threshold = config_dict['anomaly_threshold']
        
        # Load token encoder
        self.token_encoder = joblib.load(model_dir / "token_encoder.pkl")
        
        # Initialize and load DeepLog model
        self.deeplog_model = DeepLogLSTM(self.config, self.vocab_size)
        self.deeplog_model.load_state_dict(
            torch.load(model_dir / "deeplog_model.pt", map_location=self.device)
        )
        self.deeplog_model.to(self.device)
        self.deeplog_model.eval()
        
        logger.info(f"Model loaded from {model_dir}")
    
    def train_on_sessions(self, sessions: List[str]) -> Dict[str, Any]:
        """
        Complete training pipeline on session data
        
        Args:
            sessions: List of session text strings
            
        Returns:
            training_results: Dictionary with training metrics and statistics
        """
        logger.info("Starting BERT-DeepLog training pipeline")
        
        # Preprocess sessions
        embeddings, token_sequences = self.preprocess_sessions(sessions)
        
        # Train model
        history = self.train_deeplog_model(embeddings, token_sequences)
        
        # Training results
        results = {
            'training_history': history,
            'num_sessions': len(sessions),
            'num_sequences': len(embeddings),
            'vocab_size': self.vocab_size,
            'model_saved': True,
            'trained_at': datetime.now().isoformat()
        }
        
        logger.info("BERT-DeepLog training completed successfully")
        return results

def demo_deeplog_bert_training():
    """
    Demonstration of BERT + DeepLog training on sample ABM session data
    """
    # Sample ABM session data (similar to your examples)
    sample_sessions = [
        # Normal transaction
        """
        TRANSACTION START
        CARD INSERTED
        ATR RECEIVED T=0
        OPCODE = FI
        PAN 0004263********6687
        START OF TRANSACTION
        PIN ENTERED
        OPCODE = BC
        BALANCE INQUIRY SELECTED
        BALANCE DISPLAYED $1,250.00
        TRANSACTION COMPLETED
        CARD TAKEN
        TRANSACTION END
        PRIMARY CARD READER ACTIVATED
        """,
        
        # Anomalous transaction (card taken immediately)
        """
        TRANSACTION START
        CARD INSERTED
        CARD TAKEN
        TRANSACTION END
        PRIMARY CARD READER ACTIVATED
        """,
        
        # Anomalous transaction (missing completion)
        """
        TRANSACTION START
        CARD INSERTED
        ATR RECEIVED T=0
        OPCODE = FI
        PAN 0004263********6687
        START OF TRANSACTION
        PIN ENTERED
        OPCODE = BC
        CARD TAKEN
        TRANSACTION END
        PRIMARY CARD READER ACTIVATED
        """,
        
        # Normal cash withdrawal
        """
        TRANSACTION START
        CARD INSERTED
        ATR RECEIVED T=0
        OPCODE = FI
        PAN 0004263********6687
        START OF TRANSACTION
        PIN ENTERED
        CASH WITHDRAWAL SELECTED
        AMOUNT ENTERED $100.00
        CASH DISPENSED $100.00
        RECEIPT PRINTED
        CARD TAKEN
        TRANSACTION END
        PRIMARY CARD READER ACTIVATED
        """
    ]
    
    # Initialize trainer
    config = DeepLogConfig(
        sequence_length=32,
        num_epochs=50,
        batch_size=16
    )
    
    trainer = BERTDeepLogTrainer(config)
    
    # Train on sample sessions
    results = trainer.train_on_sessions(sample_sessions)
    print(f"Training Results: {results}")
    
    # Test anomaly detection
    test_sessions = [
        # Should be detected as anomaly (incomplete transaction)
        """
        TRANSACTION START
        CARD INSERTED
        CARD TAKEN
        TRANSACTION END
        """,
        
        # Should be normal
        """
        TRANSACTION START
        CARD INSERTED
        PIN ENTERED
        BALANCE INQUIRY
        BALANCE DISPLAYED $500.00
        CARD TAKEN
        TRANSACTION END
        """
    ]
    
    predictions = trainer.predict_anomalies(test_sessions)
    
    for i, pred in enumerate(predictions):
        print(f"Session {i}: {pred}")

if __name__ == "__main__":
    demo_deeplog_bert_training()
