"""
DeepLog + BERT Integration for Sequence Anomaly Detection
Combines BERT token embeddings with DeepLog's sequential pattern learning
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pickle
import logging
from typing import List, Dict, Any, Tuple, Optional
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict, Counter
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class BertTokenizer4DeepLog:
    """
    BERT tokenizer specifically designed for DeepLog integration
    Converts ABM transaction text into BERT tokens that DeepLog can process
    """
    
    def __init__(self, model_name: str = 'bert-base-uncased', max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert_model = BertModel.from_pretrained(model_name)
        
        # Special tokens for ABM transactions
        self.special_tokens = {
            '[TXN_START]': '[unused1]',
            '[TXN_END]': '[unused2]',
            '[CARD_IN]': '[unused3]',
            '[CARD_OUT]': '[unused4]',
            '[PIN_ENTERED]': '[unused5]',
            '[CASH_DISPENSED]': '[unused6]',
            '[ERROR]': '[unused7]',
            '[TIMEOUT]': '[unused8]'
        }
        
        # Add special tokens to tokenizer
        self.tokenizer.add_tokens(list(self.special_tokens.values()))
        self.bert_model.resize_token_embeddings(len(self.tokenizer))
        
        logger.info(f"BertTokenizer4DeepLog initialized with {model_name}")
    
    def preprocess_abm_text(self, raw_text: str) -> str:
        """
        Preprocess ABM transaction text to enhance tokenization
        """
        # Replace common ABM patterns with special tokens
        text = raw_text
        
        # Transaction boundaries
        text = text.replace('TRANSACTION START', '[TXN_START]')
        text = text.replace('TRANSACTION END', '[TXN_END]')
        
        # Card operations
        text = text.replace('CARD INSERTED', '[CARD_IN]')
        text = text.replace('CARD TAKEN', '[CARD_OUT]')
        
        # User interactions
        text = text.replace('PIN ENTERED', '[PIN_ENTERED]')
        
        # Financial operations
        text = text.replace('CASH DISPENSED', '[CASH_DISPENSED]')
        
        # Error conditions
        if 'ERROR' in text or 'UNABLE' in text or 'FAIL' in text:
            text = text + ' [ERROR]'
        
        if 'TIMEOUT' in text or 'TIME OUT' in text:
            text = text + ' [TIMEOUT]'
        
        return text
    
    def tokenize_for_deeplog(self, text: str) -> List[str]:
        """
        Tokenize text into BERT tokens suitable for DeepLog sequence learning
        """
        # Preprocess ABM text
        preprocessed = self.preprocess_abm_text(text)
        
        # Tokenize with BERT
        tokens = self.tokenizer.tokenize(preprocessed, max_length=self.max_length, truncation=True)
        
        # Filter out special BERT tokens for cleaner sequences
        filtered_tokens = [token for token in tokens if not token.startswith('[unused')]
        
        return filtered_tokens
    
    def get_token_embeddings(self, tokens: List[str]) -> torch.Tensor:
        """
        Get BERT embeddings for tokens
        """
        # Convert tokens to input IDs
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor([token_ids])
        
        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.bert_model(input_ids)
            embeddings = outputs.last_hidden_state.squeeze(0)  # Remove batch dimension
        
        return embeddings


class DeepLogBertModel(nn.Module):
    """
    DeepLog model enhanced with BERT token embeddings
    Predicts next tokens in sequence and detects anomalies
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 768, hidden_dim: int = 256, 
                 num_layers: int = 2, window_size: int = 10):
        super(DeepLogBertModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.window_size = window_size
        
        # Embedding layer (can be pre-initialized with BERT embeddings)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers for sequence modeling
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        
        # Output layers
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.2)
        
        # Anomaly detection threshold
        self.anomaly_threshold = 0.1
        
    def forward(self, x):
        """
        Forward pass through the model
        """
        # Embedding lookup
        embedded = self.embedding(x)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Output projection
        output = self.fc_out(lstm_out)
        
        return output
    
    def predict_next_token(self, sequence: torch.Tensor) -> Tuple[int, float]:
        """
        Predict the next token in sequence and return confidence
        """
        self.eval()
        with torch.no_grad():
            # Get model output
            output = self.forward(sequence.unsqueeze(0))  # Add batch dimension
            
            # Get last timestep prediction
            last_output = output[0, -1, :]  # [vocab_size]
            
            # Apply softmax to get probabilities
            probs = torch.softmax(last_output, dim=0)
            
            # Get most likely token
            predicted_token = torch.argmax(probs).item()
            confidence = probs[predicted_token].item()
            
            return predicted_token, confidence
    
    def detect_anomaly(self, sequence: torch.Tensor, actual_next_token: int) -> Tuple[bool, float]:
        """
        Detect if the actual next token is anomalous given the sequence
        """
        predicted_token, confidence = self.predict_next_token(sequence)
        
        # Calculate anomaly score based on prediction confidence
        if predicted_token == actual_next_token:
            anomaly_score = 1.0 - confidence  # Low confidence in correct prediction = potential anomaly
        else:
            anomaly_score = confidence  # High confidence in wrong prediction = definite anomaly
        
        is_anomaly = anomaly_score > self.anomaly_threshold
        
        return is_anomaly, anomaly_score


class DeepLogBertTrainer:
    """
    Trainer for DeepLog + BERT model on ABM transaction sequences
    """
    
    def __init__(self, model_save_path: str = "/app/models/deeplog_bert_model.pt"):
        self.model_save_path = model_save_path
        self.tokenizer_bert = BertTokenizer4DeepLog()
        self.token_encoder = LabelEncoder()
        self.model = None
        self.vocab_size = 0
        self.window_size = 10
        
        # Training parameters
        self.learning_rate = 0.001
        self.batch_size = 32
        self.epochs = 50
        
        logger.info("DeepLogBertTrainer initialized")
    
    def prepare_training_data(self, transaction_sessions: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare training data from transaction sessions
        
        Args:
            transaction_sessions: List of raw transaction text
            
        Returns:
            input_sequences, target_sequences for training
        """
        logger.info(f"Preparing training data from {len(transaction_sessions)} sessions")
        
        # Tokenize all sessions
        all_tokens = []
        for session in transaction_sessions:
            tokens = self.tokenizer_bert.tokenize_for_deeplog(session)
            all_tokens.extend(tokens)
        
        logger.info(f"Total tokens extracted: {len(all_tokens)}")
        
        # Encode tokens to integers
        self.token_encoder.fit(all_tokens)
        self.vocab_size = len(self.token_encoder.classes_)
        encoded_tokens = self.token_encoder.transform(all_tokens)
        
        logger.info(f"Vocabulary size: {self.vocab_size}")
        
        # Create sliding window sequences
        input_sequences = []
        target_sequences = []
        
        for i in range(len(encoded_tokens) - self.window_size):
            input_seq = encoded_tokens[i:i + self.window_size]
            target_seq = encoded_tokens[i + 1:i + self.window_size + 1]
            
            input_sequences.append(input_seq)
            target_sequences.append(target_seq)
        
        input_tensor = torch.tensor(input_sequences, dtype=torch.long)
        target_tensor = torch.tensor(target_sequences, dtype=torch.long)
        
        logger.info(f"Created {len(input_sequences)} training sequences")
        
        return input_tensor, target_tensor
    
    def train_model(self, transaction_sessions: List[str], labels: List[int] = None) -> Dict[str, Any]:
        """
        Train the DeepLog + BERT model
        
        Args:
            transaction_sessions: List of transaction text
            labels: Optional labels for supervised learning (1 = anomaly, 0 = normal)
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting DeepLog + BERT training")
        
        # Prepare training data
        input_sequences, target_sequences = self.prepare_training_data(transaction_sessions)
        
        # Initialize model
        self.model = DeepLogBertModel(
            vocab_size=self.vocab_size,
            window_size=self.window_size
        )
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        training_losses = []
        
        for epoch in range(self.epochs):\n            epoch_loss = 0.0\n            \n            # Mini-batch training\n            for i in range(0, len(input_sequences), self.batch_size):\n                batch_input = input_sequences[i:i + self.batch_size]\n                batch_target = target_sequences[i:i + self.batch_size]\n                \n                # Forward pass\n                optimizer.zero_grad()\n                output = self.model(batch_input)\n                \n                # Reshape for loss calculation\n                output = output.view(-1, self.vocab_size)\n                batch_target = batch_target.view(-1)\n                \n                # Calculate loss\n                loss = criterion(output, batch_target)\n                \n                # Backward pass\n                loss.backward()\n                optimizer.step()\n                \n                epoch_loss += loss.item()\n            \n            avg_loss = epoch_loss / (len(input_sequences) // self.batch_size)\n            training_losses.append(avg_loss)\n            \n            if epoch % 10 == 0:\n                logger.info(f"Epoch {epoch}/{self.epochs}, Loss: {avg_loss:.4f}")\n        \n        # Save model\n        self.save_model()\n        \n        return {\n            "status": "success",\n            "epochs_trained": self.epochs,\n            "final_loss": training_losses[-1],\n            "vocab_size": self.vocab_size,\n            "sequences_trained": len(input_sequences),\n            "model_path": self.model_save_path\n        }\n    \n    def save_model(self):\n        """Save the trained model and metadata"""\n        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)\n        \n        # Save model state\n        torch.save({\n            'model_state_dict': self.model.state_dict(),\n            'vocab_size': self.vocab_size,\n            'window_size': self.window_size,\n            'token_encoder': self.token_encoder,\n            'model_config': {\n                'embedding_dim': self.model.embedding_dim,\n                'hidden_dim': self.model.hidden_dim,\n                'num_layers': self.model.num_layers\n            }\n        }, self.model_save_path)\n        \n        logger.info(f"Model saved to {self.model_save_path}")\n    \n    def load_model(self) -> bool:\n        """Load a pre-trained model"""\n        try:\n            checkpoint = torch.load(self.model_save_path)\n            \n            self.vocab_size = checkpoint['vocab_size']\n            self.window_size = checkpoint['window_size']\n            self.token_encoder = checkpoint['token_encoder']\n            \n            # Reconstruct model\n            config = checkpoint['model_config']\n            self.model = DeepLogBertModel(\n                vocab_size=self.vocab_size,\n                embedding_dim=config['embedding_dim'],\n                hidden_dim=config['hidden_dim'],\n                num_layers=config['num_layers'],\n                window_size=self.window_size\n            )\n            \n            self.model.load_state_dict(checkpoint['model_state_dict'])\n            self.model.eval()\n            \n            logger.info(f"Model loaded from {self.model_save_path}")\n            return True\n            \n        except Exception as e:\n            logger.error(f"Failed to load model: {e}")\n            return False\n    \n    def predict_anomaly(self, session_text: str) -> Dict[str, Any]:\n        """Predict if a session is anomalous"""\n        if self.model is None:\n            if not self.load_model():\n                return {"error": "Model not available"}\n        \n        # Tokenize session\n        tokens = self.tokenizer_bert.tokenize_for_deeplog(session_text)\n        \n        if len(tokens) < self.window_size:\n            return {\n                "is_anomaly": False,\n                "confidence": 0.5,\n                "reason": "Session too short for analysis"\n            }\n        \n        # Encode tokens\n        try:\n            encoded_tokens = self.token_encoder.transform(tokens)\n        except ValueError as e:\n            # Unknown tokens\n            return {\n                "is_anomaly": True,\n                "confidence": 0.8,\n                "reason": f"Unknown tokens detected: {e}"\n            }\n        \n        # Analyze sequences\n        anomaly_scores = []\n        \n        for i in range(len(encoded_tokens) - self.window_size):\n            sequence = torch.tensor(encoded_tokens[i:i + self.window_size], dtype=torch.long)\n            actual_next = encoded_tokens[i + self.window_size]\n            \n            is_anomaly, score = self.model.detect_anomaly(sequence, actual_next)\n            anomaly_scores.append(score)\n        \n        # Overall anomaly decision\n        avg_anomaly_score = np.mean(anomaly_scores)\n        max_anomaly_score = np.max(anomaly_scores)\n        is_session_anomaly = max_anomaly_score > self.model.anomaly_threshold\n        \n        return {\n            "is_anomaly": is_session_anomaly,\n            "confidence": max_anomaly_score,\n            "average_anomaly_score": avg_anomaly_score,\n            "max_anomaly_score": max_anomaly_score,\n            "anomalous_sequences": sum(1 for score in anomaly_scores if score > self.model.anomaly_threshold),\n            "total_sequences": len(anomaly_scores)\n        }\n\n\n# Factory function for easy integration\ndef create_deeplog_bert_trainer(model_path: str = None) -> DeepLogBertTrainer:\n    """Create a DeepLog + BERT trainer instance"""\n    if model_path is None:\n        model_path = "/app/models/deeplog_bert_model.pt"\n    \n    return DeepLogBertTrainer(model_save_path=model_path)\n\n\n# Training script for command line usage\ndef train_deeplog_on_abm_data(data_file: str, model_output: str = None):\n    """Train DeepLog model on ABM transaction data"""\n    \n    # Load training data\n    with open(data_file, 'r') as f:\n        sessions = f.read().split('\\n\\n')  # Assuming sessions are separated by double newlines\n    \n    # Create trainer\n    trainer = create_deeplog_bert_trainer(model_output)\n    \n    # Train model\n    results = trainer.train_model(sessions)\n    \n    print(f"Training completed: {results}")\n    \n    return results\n\n\nif __name__ == "__main__":\n    # Example usage\n    import sys\n    \n    if len(sys.argv) > 1:\n        data_file = sys.argv[1]\n        model_output = sys.argv[2] if len(sys.argv) > 2 else None\n        train_deeplog_on_abm_data(data_file, model_output)\n    else:\n        print("Usage: python deeplog_bert_training.py <data_file> [model_output]")\n
