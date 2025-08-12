"""
DeepLog Training Integration for ABM Anomaly Detection
====================================================

This module integrates DeepLog functionality to accept BERT tokens and train models
for anomaly prediction in ABM transaction logs.

DeepLog is a log-based anomaly detection system that uses deep learning to model
log sequences and detect anomalies based on deviation from normal patterns.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime
import re
from collections import defaultdict, Counter
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
import joblib

logger = logging.getLogger(__name__)

class DeepLogLSTM(nn.Module):
    """
    DeepLog LSTM model that processes BERT token sequences for anomaly detection.
    
    Architecture:
    - Input: BERT token sequences (from EJ log sessions)
    - LSTM layers for sequence modeling
    - Output: Probability distribution over next token (for sequence prediction)
    - Anomaly score based on prediction confidence
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 64, 
                 num_layers: int = 2, dropout: float = 0.1):
        super(DeepLogLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer for BERT tokens
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers for sequence modeling
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer for next token prediction
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden=None):
        """
        Forward pass through the model.
        
        Args:
            x: Input token sequences [batch_size, seq_length]
            hidden: Optional hidden state for LSTM
            
        Returns:
            output: Token predictions [batch_size, seq_length, vocab_size]
            hidden: Updated hidden state
        """
        # Embedding
        embedded = self.embedding(x)  # [batch_size, seq_length, embedding_dim]
        
        # LSTM
        lstm_out, hidden = self.lstm(embedded, hidden)  # [batch_size, seq_length, hidden_dim]
        
        # Dropout
        lstm_out = self.dropout(lstm_out)
        
        # Output layer
        output = self.fc(lstm_out)  # [batch_size, seq_length, vocab_size]
        
        return output, hidden


class EJLogDataset(Dataset):
    """
    Dataset class for EJ log sequences compatible with DeepLog training.
    
    Processes BERT tokenized EJ sessions into sequences suitable for LSTM training.
    """
    
    def __init__(self, sessions: List[str], tokenizer, sequence_length: int = 50, 
                 stride: int = 25):
        """
        Initialize dataset from EJ log sessions.
        
        Args:
            sessions: List of raw EJ session texts
            tokenizer: BERT tokenizer for text processing
            sequence_length: Length of input sequences
            stride: Stride for sliding window over sessions
        """
        self.sessions = sessions
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.stride = stride
        
        # Process sessions into sequences
        self.sequences = []
        self.labels = []
        
        self._process_sessions()
        
    def _process_sessions(self):
        """Process sessions into training sequences."""
        logger.info(f"Processing {len(self.sessions)} sessions into sequences")
        
        for session_idx, session_text in enumerate(self.sessions):
            # Tokenize the session text
            tokens = self.tokenizer.encode(
                session_text, 
                add_special_tokens=True,
                max_length=512,  # BERT limit
                truncation=True,
                return_tensors='pt'
            ).squeeze()
            
            # Create sliding window sequences
            for i in range(0, len(tokens) - self.sequence_length, self.stride):
                sequence = tokens[i:i + self.sequence_length]
                next_token = tokens[i + self.sequence_length] if i + self.sequence_length < len(tokens) else tokens[-1]
                
                self.sequences.append(sequence)
                self.labels.append(next_token)
        
        logger.info(f"Created {len(self.sequences)} training sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],
            'label': self.labels[idx]
        }


class DeepLogBERTTrainer:
    """
    Trainer class for DeepLog model using BERT tokenized EJ logs.
    
    Handles model training, validation, and anomaly prediction.
    """
    
    def __init__(self, model_dir: str = "/app/models/deeplog", device: str = None):
        """
        Initialize trainer.
        
        Args:
            model_dir: Directory to save/load models
            device: Device for training (cuda/cpu)
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.vocab_size = self.tokenizer.vocab_size
        
        # Model and training components
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Training statistics
        self.training_history = []
        
    def prepare_model(self, embedding_dim: int = 128, hidden_dim: int = 64, 
                     num_layers: int = 2, dropout: float = 0.1):
        """
        Initialize the DeepLog model.
        
        Args:
            embedding_dim: Dimension of token embeddings
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        self.model = DeepLogLSTM(
            vocab_size=self.vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        logger.info(f"Initialized DeepLog model with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train_from_sessions(self, sessions: List[str], validation_split: float = 0.2,
                           epochs: int = 50, batch_size: int = 32, 
                           sequence_length: int = 50) -> Dict[str, float]:
        """
        Train DeepLog model from EJ session data.
        
        Args:
            sessions: List of EJ session texts
            validation_split: Fraction of data for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            sequence_length: Length of input sequences
            
        Returns:
            Training metrics dictionary
        """
        if self.model is None:
            self.prepare_model()
        
        logger.info(f"Training DeepLog on {len(sessions)} sessions")
        
        # Split data
        split_idx = int(len(sessions) * (1 - validation_split))
        train_sessions = sessions[:split_idx]
        val_sessions = sessions[split_idx:]
        
        # Create datasets
        train_dataset = EJLogDataset(train_sessions, self.tokenizer, sequence_length)
        val_dataset = EJLogDataset(val_sessions, self.tokenizer, sequence_length)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training loop
        best_val_loss = float('inf')
        training_start = datetime.now()
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_samples = 0
            
            for batch in train_loader:
                sequences = batch['sequence'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs, _ = self.model(sequences)
                
                # Calculate loss (predict next token)
                loss = self.criterion(outputs[:, -1, :], labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * sequences.size(0)
                train_samples += sequences.size(0)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_samples = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    sequences = batch['sequence'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs, _ = self.model(sequences)
                    loss = self.criterion(outputs[:, -1, :], labels)
                    
                    val_loss += loss.item() * sequences.size(0)
                    val_samples += sequences.size(0)
            
            # Calculate average losses
            avg_train_loss = train_loss / train_samples
            avg_val_loss = val_loss / val_samples
            
            # Log progress
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model()
            
            # Record training history
            self.training_history.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'timestamp': datetime.now().isoformat()
            })
        
        training_time = (datetime.now() - training_start).total_seconds()
        
        # Save training history
        with open(self.model_dir / 'training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        metrics = {
            'final_train_loss': avg_train_loss,
            'best_val_loss': best_val_loss,
            'training_time': training_time,
            'epochs': epochs,
            'total_sequences': len(train_dataset) + len(val_dataset)
        }
        
        logger.info(f"Training completed in {training_time:.2f}s. Best val loss: {best_val_loss:.4f}")
        return metrics
    
    def predict_anomaly(self, session_text: str, threshold: float = 0.1) -> Dict[str, float]:
        """
        Predict anomaly score for a single EJ session.
        
        Args:
            session_text: Raw EJ session text
            threshold: Anomaly threshold (lower = more sensitive)
            
        Returns:
            Dictionary with anomaly prediction results
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_from_sessions() first.")
        
        self.model.eval()
        
        with torch.no_grad():
            # Tokenize input
            tokens = self.tokenizer.encode(
                session_text,
                add_special_tokens=True,
                max_length=512,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            # Get model predictions
            outputs, _ = self.model(tokens)
            
            # Calculate prediction confidence (lower = more anomalous)
            probs = torch.softmax(outputs, dim=-1)
            
            # Calculate anomaly score based on prediction uncertainty
            # Higher entropy = more uncertain = more anomalous
            log_probs = torch.log(probs + 1e-8)
            entropy = -torch.sum(probs * log_probs, dim=-1)
            avg_entropy = torch.mean(entropy).item()
            
            # Also calculate perplexity
            avg_log_prob = torch.mean(log_probs).item()
            perplexity = torch.exp(-torch.mean(log_probs)).item()
            
            # Anomaly score (normalized entropy)
            anomaly_score = min(1.0, avg_entropy / 5.0)  # Normalize to 0-1
            
            is_anomaly = anomaly_score > threshold
            
            return {
                'anomaly_score': anomaly_score,
                'is_anomaly': is_anomaly,
                'entropy': avg_entropy,
                'perplexity': perplexity,
                'confidence': 1.0 - anomaly_score,
                'threshold': threshold,
                'prediction_method': 'deeplog_bert'
            }
    
    def save_model(self):
        """Save the trained model and tokenizer."""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save model state
        torch.save(self.model.state_dict(), self.model_dir / 'deeplog_model.pth')
        
        # Save tokenizer
        self.tokenizer.save_pretrained(self.model_dir / 'tokenizer')
        
        # Save model config
        config = {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.model.embedding_dim,
            'hidden_dim': self.model.hidden_dim,
            'num_layers': self.model.num_layers,
            'device': self.device,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(self.model_dir / 'model_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {self.model_dir}")
    
    def load_model(self):
        """Load a previously trained model."""
        config_path = self.model_dir / 'model_config.json'
        model_path = self.model_dir / 'deeplog_model.pth'
        tokenizer_path = self.model_dir / 'tokenizer'
        
        if not all(p.exists() for p in [config_path, model_path, tokenizer_path]):
            raise FileNotFoundError("Model files not found. Train a model first.")
        
        # Load config
        with open(config_path) as f:
            config = json.load(f)
        
        # Load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.vocab_size = config['vocab_size']
        
        # Initialize model
        self.model = DeepLogLSTM(
            vocab_size=config['vocab_size'],
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers']
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        logger.info(f"Model loaded from {self.model_dir}")


def train_deeplog_from_database(db_engine, model_dir: str = "/app/models/deeplog") -> Dict[str, float]:
    """
    Train DeepLog model using sessions from the database.
    
    Args:
        db_engine: Database engine for loading sessions
        model_dir: Directory to save the trained model
        
    Returns:
        Training metrics
    """
    logger.info("Training DeepLog from database sessions")
    
    # Load sessions from database
    query = """
    SELECT raw_text, is_anomaly 
    FROM ml_sessions 
    WHERE raw_text IS NOT NULL 
    AND LENGTH(raw_text) > 50
    ORDER BY created_at DESC
    """
    
    sessions_df = pd.read_sql(query, db_engine)
    
    if len(sessions_df) < 10:
        raise ValueError(f"Insufficient training data: {len(sessions_df)} sessions found (minimum 10 required)")
    
    logger.info(f"Loaded {len(sessions_df)} sessions from database")
    
    # Extract session texts
    session_texts = sessions_df['raw_text'].tolist()
    
    # Initialize trainer
    trainer = DeepLogBERTTrainer(model_dir)
    
    # Train model
    metrics = trainer.train_from_sessions(
        sessions=session_texts,
        epochs=100,
        batch_size=16,
        sequence_length=32
    )
    
    # Test model on some sessions
    test_sessions = session_texts[:5]
    test_results = []
    
    for i, session in enumerate(test_sessions):
        result = trainer.predict_anomaly(session)
        test_results.append(result)
        logger.info(f"Test session {i+1}: Anomaly score = {result['anomaly_score']:.3f}")
    
    # Save test results
    with open(Path(model_dir) / 'test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    metrics['test_results'] = test_results
    return metrics


# Example usage and integration
if __name__ == "__main__":
    # Example of how to use the DeepLog trainer
    
    # Sample EJ session data (replace with actual data)
    sample_sessions = [
        """[020t*209*06/18/2025*14:23*
        *TRANSACTION START*
        [020t CARD INSERTED
        14:23:03 ATR RECEIVED T=0
        [020t 14:23:06 OPCODE = FI      
        PAN 0004263********6687
        ---START OF TRANSACTION---
        [020t 14:23:22 PIN ENTERED
        [020t 14:23:36 OPCODE = BC      
        [020t 14:24:28 CARD TAKEN
        [020t 14:24:29 TRANSACTION END""",
        
        """[020t15706/18/202513:39
        TRANSACTION START
        [020t CARD INSERTED
        [020t 13:39:56 CARD TAKEN
        [000p[040q(I 75561D(10,M-090B0210B9,R-4S
        [020t 13:39:56 TRANSACTION END"""
    ]
    
    # Initialize trainer
    trainer = DeepLogBERTTrainer()
    
    # Train model
    metrics = trainer.train_from_sessions(sample_sessions, epochs=10)
    print(f"Training metrics: {metrics}")
    
    # Test anomaly detection
    test_session = sample_sessions[0]
    result = trainer.predict_anomaly(test_session)
    print(f"Anomaly prediction: {result}")
