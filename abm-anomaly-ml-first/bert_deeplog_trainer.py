#!/usr/bin/env python3
"""
BERT-DeepLog Anomaly Detection System
=====================================

This script provides a complete solution for training DeepLog with BERT tokens
to predict anomalies in EJ (Electronic Journal) logs.

Features:
- BERT tokenization and embedding generation
- DeepLog LSTM model training
- Anomaly detection and scoring
- Model persistence and loading
- Real-time prediction capability

Usage:
    python bert_deeplog_trainer.py --mode train --data_path ./data/ej_logs.txt
    python bert_deeplog_trainer.py --mode predict --model_path ./models/bert_deeplog.pkl --text "sample log text"
"""

import argparse
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BERTDeepLogConfig:
    """Configuration class for BERT-DeepLog model"""
    def __init__(self):
        self.bert_model_name = "bert-base-uncased"
        self.max_sequence_length = 512
        self.hidden_size = 768  # BERT hidden size
        self.lstm_hidden_size = 128
        self.lstm_num_layers = 2
        self.dropout_rate = 0.1
        self.window_size = 10  # DeepLog window size
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_epochs = 50
        self.early_stopping_patience = 10
        self.anomaly_threshold = 0.5

class EJLogDataset(Dataset):
    """Dataset class for EJ log sequences"""
    
    def __init__(self, sequences: List[List[torch.Tensor]], labels: List[int] = None):
        self.sequences = sequences
        self.labels = labels
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # Pad sequence to window size
        if len(sequence) < BERTDeepLogConfig().window_size:
            padding = [torch.zeros(BERTDeepLogConfig().hidden_size) for _ in range(BERTDeepLogConfig().window_size - len(sequence))]
            sequence = padding + sequence
        elif len(sequence) > BERTDeepLogConfig().window_size:
            sequence = sequence[-BERTDeepLogConfig().window_size:]
        
        sequence_tensor = torch.stack(sequence)
        
        if self.labels is not None:
            return sequence_tensor, torch.tensor(self.labels[idx], dtype=torch.float32)
        return sequence_tensor

class BERTEmbeddingExtractor:
    """Extract BERT embeddings from text"""
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"BERT model loaded on {self.device}")
    
    def extract_embeddings(self, texts: List[str], max_length: int = 512) -> List[torch.Tensor]:
        """Extract BERT embeddings from a list of texts"""
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize text
                tokens = self.tokenizer.encode(
                    text,
                    max_length=max_length,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                ).to(self.device)
                
                # Get BERT embeddings
                outputs = self.model(tokens)
                # Use [CLS] token embedding as sentence representation
                embedding = outputs.last_hidden_state[:, 0, :].squeeze()  # Shape: (768,)
                embeddings.append(embedding.cpu())
        
        return embeddings

class DeepLogLSTM(nn.Module):
    """DeepLog LSTM model for anomaly detection"""
    
    def __init__(self, config: BERTDeepLogConfig):
        super(DeepLogLSTM, self).__init__()
        self.config = config
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            dropout=config.dropout_rate if config.lstm_num_layers > 1 else 0,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.lstm_hidden_size, config.lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.lstm_hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # Use the last output
        last_output = lstm_out[:, -1, :]  # (batch_size, lstm_hidden_size)
        
        # Classification
        output = self.classifier(last_output)
        return output.squeeze()

class BERTDeepLogTrainer:
    """Training class for BERT-DeepLog model"""
    
    def __init__(self, config: BERTDeepLogConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_extractor = BERTEmbeddingExtractor(config.bert_model_name)
        self.model = None
        self.scaler = StandardScaler()
        
    def preprocess_logs(self, log_texts: List[str], labels: List[int] = None) -> Tuple[List[List[torch.Tensor]], List[int]]:
        """Preprocess log texts into BERT embedding sequences"""
        logger.info(f"Processing {len(log_texts)} log entries...")
        
        # Extract BERT embeddings
        embeddings = self.bert_extractor.extract_embeddings(log_texts)
        
        # Create sequences using sliding window
        sequences = []
        sequence_labels = []
        
        for i in range(len(embeddings) - self.config.window_size + 1):
            sequence = embeddings[i:i + self.config.window_size]
            sequences.append(sequence)
            
            if labels is not None:
                # Label the sequence as anomalous if any log in the window is anomalous
                window_labels = labels[i:i + self.config.window_size]
                sequence_label = 1 if any(window_labels) else 0
                sequence_labels.append(sequence_label)
        
        logger.info(f"Created {len(sequences)} sequences")
        return sequences, sequence_labels if labels is not None else None
    
    def train(self, train_texts: List[str], train_labels: List[int], 
              val_texts: List[str] = None, val_labels: List[int] = None):
        """Train the BERT-DeepLog model"""
        logger.info("Starting training...")
        
        # Preprocess training data
        train_sequences, train_seq_labels = self.preprocess_logs(train_texts, train_labels)
        train_dataset = EJLogDataset(train_sequences, train_seq_labels)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # Preprocess validation data if provided
        val_loader = None
        if val_texts is not None and val_labels is not None:
            val_sequences, val_seq_labels = self.preprocess_logs(val_texts, val_labels)
            val_dataset = EJLogDataset(val_sequences, val_seq_labels)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        # Initialize model
        self.model = DeepLogLSTM(self.config).to(self.device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_predictions = []
            train_targets = []
            
            for batch_sequences, batch_labels in train_loader:
                batch_sequences = batch_sequences.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_sequences)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_predictions.extend(outputs.detach().cpu().numpy())
                train_targets.extend(batch_labels.detach().cpu().numpy())
            
            train_loss /= len(train_loader)
            
            # Validation phase
            val_loss = 0.0
            val_predictions = []
            val_targets = []
            
            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    for batch_sequences, batch_labels in val_loader:
                        batch_sequences = batch_sequences.to(self.device)
                        batch_labels = batch_labels.to(self.device)
                        
                        outputs = self.model(batch_sequences)
                        loss = criterion(outputs, batch_labels)
                        
                        val_loss += loss.item()
                        val_predictions.extend(outputs.cpu().numpy())
                        val_targets.extend(batch_labels.cpu().numpy())
                
                val_loss /= len(val_loader)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.save_model("best_model.pth")
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        logger.info("Early stopping triggered")
                        break
            
            # Calculate metrics
            train_binary_preds = (np.array(train_predictions) > self.config.anomaly_threshold).astype(int)
            train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
                train_targets, train_binary_preds, average='binary', zero_division=0
            )
            
            if val_loader is not None:
                val_binary_preds = (np.array(val_predictions) > self.config.anomaly_threshold).astype(int)
                val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
                    val_targets, val_binary_preds, average='binary', zero_division=0
                )
                
                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
                logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                logger.info(f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
                logger.info(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
        
        logger.info("Training completed!")
    
    def predict(self, texts: List[str]) -> Tuple[List[float], List[int]]:
        """Predict anomalies for given texts"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        sequences, _ = self.preprocess_logs(texts)
        dataset = EJLogDataset(sequences)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch_sequences in loader:
                batch_sequences = batch_sequences.to(self.device)
                outputs = self.model(batch_sequences)
                predictions.extend(outputs.cpu().numpy())
        
        # Convert to binary predictions
        binary_predictions = (np.array(predictions) > self.config.anomaly_threshold).astype(int)
        
        return predictions, binary_predictions.tolist()
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'scaler': self.scaler,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        model_data = torch.load(filepath, map_location=self.device)
        
        # Recreate config
        config_dict = model_data['config']
        self.config = BERTDeepLogConfig()
        for key, value in config_dict.items():
            setattr(self.config, key, value)
        
        # Load model
        self.model = DeepLogLSTM(self.config).to(self.device)
        self.model.load_state_dict(model_data['model_state_dict'])
        self.scaler = model_data['scaler']
        
        logger.info(f"Model loaded from {filepath}")

class EJLogProcessor:
    """Process EJ log files for training"""
    
    @staticmethod
    def parse_log_file(filepath: str) -> Tuple[List[str], List[int]]:
        """Parse EJ log file and return texts and labels"""
        texts = []
        labels = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by transaction boundaries
        transactions = content.split('TRANSACTION START')
        
        for transaction in transactions[1:]:  # Skip empty first split
            # Clean up the transaction text
            text = 'TRANSACTION START' + transaction
            text = text.strip()
            
            if text:
                texts.append(text)
                # Simple heuristic for labeling - mark short transactions as potential anomalies
                # In practice, you would have labeled data
                if len(text) < 200 or 'ERROR' in text.upper() or 'FAIL' in text.upper():
                    labels.append(1)  # Anomaly
                else:
                    labels.append(0)  # Normal
        
        return texts, labels
    
    @staticmethod
    def create_synthetic_anomalies(texts: List[str], labels: List[int], anomaly_ratio: float = 0.1) -> Tuple[List[str], List[int]]:
        """Create synthetic anomalies for training"""
        anomaly_patterns = [
            "ERROR: Connection timeout",
            "CARD ERROR - INVALID",
            "DISPENSER JAM",
            "INSUFFICIENT FUNDS",
            "TRANSACTION DECLINED",
            "SYSTEM ERROR 500"
        ]
        
        num_anomalies = int(len(texts) * anomaly_ratio)
        indices = np.random.choice(len(texts), num_anomalies, replace=False)
        
        augmented_texts = texts.copy()
        augmented_labels = labels.copy()
        
        for idx in indices:
            # Insert anomaly pattern
            anomaly_pattern = np.random.choice(anomaly_patterns)
            augmented_texts[idx] = augmented_texts[idx] + f"\n{anomaly_pattern}"
            augmented_labels[idx] = 1
        
        return augmented_texts, augmented_labels

def main():
    parser = argparse.ArgumentParser(description='BERT-DeepLog Anomaly Detection')
    parser.add_argument('--mode', choices=['train', 'predict', 'evaluate'], required=True,
                       help='Mode to run the script')
    parser.add_argument('--data_path', type=str, help='Path to EJ log data file')
    parser.add_argument('--model_path', type=str, help='Path to save/load model')
    parser.add_argument('--text', type=str, help='Text to predict (for predict mode)')
    parser.add_argument('--output_dir', type=str, default='./models/', 
                       help='Directory to save models and outputs')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    config = BERTDeepLogConfig()
    trainer = BERTDeepLogTrainer(config)
    
    if args.mode == 'train':
        if not args.data_path:
            raise ValueError("--data_path required for training")
        
        logger.info(f"Loading data from {args.data_path}")
        texts, labels = EJLogProcessor.parse_log_file(args.data_path)
        
        # Augment with synthetic anomalies
        texts, labels = EJLogProcessor.create_synthetic_anomalies(texts, labels)
        
        # Split data (80% train, 20% validation)
        split_idx = int(0.8 * len(texts))
        train_texts, val_texts = texts[:split_idx], texts[split_idx:]
        train_labels, val_labels = labels[:split_idx], labels[split_idx:]
        
        logger.info(f"Training on {len(train_texts)} samples, validating on {len(val_texts)} samples")
        
        # Train model
        trainer.train(train_texts, train_labels, val_texts, val_labels)
        
        # Save final model
        model_path = os.path.join(args.output_dir, 'bert_deeplog_final.pth')
        trainer.save_model(model_path)
        
        logger.info(f"Training completed. Model saved to {model_path}")
    
    elif args.mode == 'predict':
        if not args.model_path or not args.text:
            raise ValueError("--model_path and --text required for prediction")
        
        # Load model
        trainer.load_model(args.model_path)
        
        # Predict
        scores, predictions = trainer.predict([args.text])
        
        result = {
            'text': args.text,
            'anomaly_score': float(scores[0]),
            'is_anomaly': bool(predictions[0]),
            'threshold': config.anomaly_threshold
        }
        
        print(json.dumps(result, indent=2))
    
    elif args.mode == 'evaluate':
        if not args.model_path or not args.data_path:
            raise ValueError("--model_path and --data_path required for evaluation")
        
        # Load model and test data
        trainer.load_model(args.model_path)
        texts, true_labels = EJLogProcessor.parse_log_file(args.data_path)
        
        # Predict
        scores, predictions = trainer.predict(texts)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary', zero_division=0
        )
        auc = roc_auc_score(true_labels, scores)
        
        results = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'threshold': config.anomaly_threshold,
            'total_samples': len(texts),
            'true_anomalies': sum(true_labels),
            'predicted_anomalies': sum(predictions)
        }
        
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
