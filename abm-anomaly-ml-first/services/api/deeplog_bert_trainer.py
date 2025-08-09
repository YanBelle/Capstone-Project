"""
DeepLog Training Module with BERT Token Support
Integrates BERT embeddings with DeepLog for anomaly detection
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
from loguru import logger
from typing import List, Dict, Tuple, Optional
import pickle
import os
from collections import defaultdict
import asyncio

class BERTDeepLogModel(nn.Module):
    """DeepLog model that accepts BERT token embeddings as input"""
    
    def __init__(self, bert_dim=768, hidden_dim=128, num_layers=2, num_classes=None):
        super(BERTDeepLogModel, self).__init__()
        
        self.bert_dim = bert_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # LSTM layers for sequence modeling
        self.lstm = nn.LSTM(
            input_size=bert_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Classification head
        if num_classes:
            self.classifier = nn.Linear(hidden_dim, num_classes)
        else:
            # For anomaly detection (binary classification)
            self.classifier = nn.Linear(hidden_dim, 2)
        
        # Attention mechanism for important event focus
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, bert_embeddings, attention_mask=None):
        """
        Args:
            bert_embeddings: (batch_size, seq_len, bert_dim)
            attention_mask: (batch_size, seq_len)
        """
        batch_size, seq_len, _ = bert_embeddings.shape
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(bert_embeddings)
        
        # Apply attention mechanism
        if attention_mask is not None:
            # Convert attention mask for multihead attention
            # (batch_size, seq_len) -> (batch_size, seq_len, seq_len)
            attn_mask = attention_mask.unsqueeze(1).repeat(1, seq_len, 1)
            attn_mask = attn_mask.bool()
        else:
            attn_mask = None
            
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out, attn_mask=attn_mask)
        
        # Use the last hidden state for classification
        if attention_mask is not None:
            # Get the last non-padded position for each sequence
            lengths = attention_mask.sum(dim=1) - 1
            last_hidden = attended_out[range(batch_size), lengths]
        else:
            last_hidden = attended_out[:, -1, :]
        
        # Apply dropout and classify
        features = self.dropout(last_hidden)
        logits = self.classifier(features)
        
        return logits, features

class BERTDeepLogTrainer:
    """Trainer for BERT-enhanced DeepLog model"""
    
    def __init__(self, 
                 bert_model_name="distilbert-base-uncased",
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 max_sequence_length=512,
                 model_save_path="/app/models/deeplog_bert"):
        
        self.device = device
        self.max_sequence_length = max_sequence_length
        self.model_save_path = model_save_path
        
        # Initialize BERT components
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert_model = AutoModel.from_pretrained(bert_model_name).to(device)
        self.bert_model.eval()
        
        # DeepLog model (will be initialized during training)
        self.deeplog_model = None
        self.label_encoder = {}
        self.reverse_label_encoder = {}
        
        # Training history
        self.training_history = []
        
        logger.info(f"Initialized BERTDeepLogTrainer with device: {device}")
    
    def extract_bert_embeddings(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract BERT embeddings from raw text sessions"""
        all_embeddings = []
        all_masks = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize text
                tokens = self.tokenizer(
                    text,
                    max_length=self.max_sequence_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = tokens['input_ids'].to(self.device)
                attention_mask = tokens['attention_mask'].to(self.device)
                
                # Get BERT embeddings
                outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state.squeeze(0)  # (seq_len, bert_dim)
                
                all_embeddings.append(embeddings.cpu())
                all_masks.append(attention_mask.squeeze(0).cpu())
        
        # Stack embeddings
        embeddings_tensor = torch.stack(all_embeddings)  # (batch_size, seq_len, bert_dim)
        masks_tensor = torch.stack(all_masks)  # (batch_size, seq_len)
        
        return embeddings_tensor, masks_tensor
    
    def prepare_training_data(self, sessions_data: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare training data from session data"""
        texts = []
        labels = []
        
        # Build label encoder
        unique_labels = set()
        for session in sessions_data:
            label = session.get('anomaly_type', 'normal')
            if session.get('is_anomaly', False):
                unique_labels.add(label)
            else:
                unique_labels.add('normal')
        
        self.label_encoder = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        self.reverse_label_encoder = {idx: label for label, idx in self.label_encoder.items()}
        
        logger.info(f"Label encoder: {self.label_encoder}")
        
        # Extract texts and encode labels
        for session in sessions_data:
            raw_text = session.get('raw_text', '')
            if not raw_text or raw_text == "Raw text not available":
                continue
                
            texts.append(raw_text)
            
            if session.get('is_anomaly', False):
                label = session.get('anomaly_type', 'unknown_anomaly')
            else:
                label = 'normal'
            
            labels.append(self.label_encoder[label])
        
        # Extract BERT embeddings
        embeddings, masks = self.extract_bert_embeddings(texts)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        logger.info(f"Prepared {len(texts)} training samples with {len(self.label_encoder)} classes")
        
        return embeddings, masks, labels_tensor
    
    def train_model(self, 
                   sessions_data: List[Dict],
                   epochs: int = 10,
                   batch_size: int = 8,
                   learning_rate: float = 1e-4) -> Dict:
        """Train the DeepLog model on session data"""
        
        # Prepare data
        embeddings, masks, labels = self.prepare_training_data(sessions_data)
        
        # Initialize model
        num_classes = len(self.label_encoder)
        self.deeplog_model = BERTDeepLogModel(
            bert_dim=self.bert_model.config.hidden_size,
            hidden_dim=128,
            num_layers=2,
            num_classes=num_classes
        ).to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.deeplog_model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
        
        dataset_size = len(embeddings)
        indices = torch.randperm(dataset_size)
        train_split = int(0.8 * dataset_size)
        
        train_indices = indices[:train_split]
        val_indices = indices[train_split:]
        
        best_val_accuracy = 0.0
        training_history = []
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            self.deeplog_model.train()
            
            # Training loop
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for i in range(0, len(train_indices), batch_size):
                batch_indices = train_indices[i:i+batch_size]
                
                batch_embeddings = embeddings[batch_indices].to(self.device)
                batch_masks = masks[batch_indices].to(self.device)
                batch_labels = labels[batch_indices].to(self.device)
                
                optimizer.zero_grad()
                
                logits, _ = self.deeplog_model(batch_embeddings, batch_masks)
                loss = criterion(logits, batch_labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()
            
            # Validation loop
            self.deeplog_model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for i in range(0, len(val_indices), batch_size):
                    batch_indices = val_indices[i:i+batch_size]
                    
                    batch_embeddings = embeddings[batch_indices].to(self.device)
                    batch_masks = masks[batch_indices].to(self.device)
                    batch_labels = labels[batch_indices].to(self.device)
                    
                    logits, _ = self.deeplog_model(batch_embeddings, batch_masks)
                    loss = criterion(logits, batch_labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(logits.data, 1)
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels).sum().item()
            
            train_accuracy = 100 * train_correct / train_total
            val_accuracy = 100 * val_correct / val_total
            
            epoch_stats = {
                'epoch': epoch + 1,
                'train_loss': train_loss / (len(train_indices) // batch_size),
                'train_accuracy': train_accuracy,
                'val_loss': val_loss / (len(val_indices) // batch_size),
                'val_accuracy': val_accuracy,
                'learning_rate': scheduler.get_last_lr()[0]
            }
            
            training_history.append(epoch_stats)
            
            logger.info(f"Epoch {epoch+1}/{epochs}: "
                       f"Train Loss: {epoch_stats['train_loss']:.4f}, "
                       f"Train Acc: {train_accuracy:.2f}%, "
                       f"Val Loss: {epoch_stats['val_loss']:.4f}, "
                       f"Val Acc: {val_accuracy:.2f}%")
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.save_model()
                logger.info(f"New best model saved with validation accuracy: {val_accuracy:.2f}%")
            
            scheduler.step()
        
        self.training_history = training_history
        
        return {
            'status': 'success',
            'training_history': training_history,
            'best_val_accuracy': best_val_accuracy,
            'num_classes': num_classes,
            'label_encoder': self.label_encoder
        }
    
    def predict_anomaly(self, raw_text: str) -> Dict:
        """Predict if a session is anomalous using the trained model"""
        if self.deeplog_model is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        self.deeplog_model.eval()
        
        # Extract BERT embeddings
        embeddings, masks = self.extract_bert_embeddings([raw_text])
        embeddings = embeddings.to(self.device)
        masks = masks.to(self.device)
        
        with torch.no_grad():
            logits, features = self.deeplog_model(embeddings, masks)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        predicted_label = self.reverse_label_encoder[predicted_class]
        is_anomaly = predicted_label != 'normal'
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_type': predicted_label if is_anomaly else None,
            'confidence': confidence,
            'probabilities': {
                self.reverse_label_encoder[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            }
        }
    
    def save_model(self):
        """Save the trained model and related artifacts"""
        os.makedirs(self.model_save_path, exist_ok=True)
        
        if self.deeplog_model is not None:
            # Save model state
            torch.save(self.deeplog_model.state_dict(), 
                      os.path.join(self.model_save_path, 'deeplog_bert_model.pth'))
            
            # Save model config
            model_config = {
                'bert_dim': self.deeplog_model.bert_dim,
                'hidden_dim': self.deeplog_model.hidden_dim,
                'num_layers': self.deeplog_model.num_layers,
                'num_classes': self.deeplog_model.num_classes,
                'label_encoder': self.label_encoder,
                'reverse_label_encoder': self.reverse_label_encoder
            }
            
            with open(os.path.join(self.model_save_path, 'model_config.pkl'), 'wb') as f:
                pickle.dump(model_config, f)
            
            # Save training history
            with open(os.path.join(self.model_save_path, 'training_history.pkl'), 'wb') as f:
                pickle.dump(self.training_history, f)
            
            logger.info(f"Model saved to {self.model_save_path}")
    
    def load_model(self):
        """Load a previously trained model"""
        try:
            # Load model config
            with open(os.path.join(self.model_save_path, 'model_config.pkl'), 'rb') as f:
                config = pickle.load(f)
            
            # Initialize model with saved config
            self.deeplog_model = BERTDeepLogModel(
                bert_dim=config['bert_dim'],
                hidden_dim=config['hidden_dim'],
                num_layers=config['num_layers'],
                num_classes=config['num_classes']
            ).to(self.device)
            
            # Load model weights
            self.deeplog_model.load_state_dict(
                torch.load(os.path.join(self.model_save_path, 'deeplog_bert_model.pth'),
                          map_location=self.device)
            )
            
            # Load encoders
            self.label_encoder = config['label_encoder']
            self.reverse_label_encoder = config['reverse_label_encoder']
            
            # Load training history
            try:
                with open(os.path.join(self.model_save_path, 'training_history.pkl'), 'rb') as f:
                    self.training_history = pickle.load(f)
            except FileNotFoundError:
                self.training_history = []
            
            self.deeplog_model.eval()
            logger.info("Model loaded successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

# Global trainer instance
deeplog_trainer = None

def get_deeplog_trainer():
    """Get or create the global DeepLog trainer instance"""
    global deeplog_trainer
    if deeplog_trainer is None:
        deeplog_trainer = BERTDeepLogTrainer()
        # Try to load existing model
        deeplog_trainer.load_model()
    return deeplog_trainer
