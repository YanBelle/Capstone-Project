"""
Enhanced DeepLog Integration with BERT Tokens for ABM Anomaly Detection
Combines BERT semantic understanding with DeepLog sequential pattern learning
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel
from typing import List, Dict, Tuple, Optional
import joblib
import json
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class BERTDeepLogModel(nn.Module):
    """Enhanced DeepLog model that accepts BERT token embeddings"""
    
    def __init__(self, bert_dim: int = 768, hidden_dim: int = 256, 
                 num_layers: int = 2, vocab_size: int = 1000):
        super(BERTDeepLogModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Process BERT embeddings
        self.bert_projection = nn.Linear(bert_dim, hidden_dim)
        
        # LSTM for sequential pattern learning
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.2)
        
        # Output layers for anomaly detection
        self.anomaly_classifier = nn.Linear(hidden_dim, 2)  # Normal vs Anomaly
        self.confidence_estimator = nn.Linear(hidden_dim, 1)  # Confidence score
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, bert_embeddings):
        """
        Forward pass with BERT embeddings
        Args:
            bert_embeddings: (batch_size, sequence_length, 768)
        Returns:
            anomaly_logits: (batch_size, sequence_length, 2)
            confidence_scores: (batch_size, sequence_length, 1)
        """
        # Project BERT embeddings to hidden dimension
        projected = self.relu(self.bert_projection(bert_embeddings))
        
        # LSTM processing for sequential patterns
        lstm_out, (hidden, cell) = self.lstm(projected)
        lstm_out = self.dropout(lstm_out)
        
        # Anomaly classification
        anomaly_logits = self.anomaly_classifier(lstm_out)
        confidence_scores = torch.sigmoid(self.confidence_estimator(lstm_out))
        
        return anomaly_logits, confidence_scores

class BERTDeepLogTrainer:
    """Trainer for BERT + DeepLog integrated anomaly detection"""
    
    def __init__(self, model_save_path: str = "/app/models"):
        self.model_save_path = model_save_path
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.deeplog_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training parameters
        self.max_sequence_length = 512
        self.learning_rate = 0.001
        self.batch_size = 16
        
        os.makedirs(model_save_path, exist_ok=True)
        
    def prepare_training_data(self, sessions: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare training data from transaction sessions
        Args:
            sessions: List of transaction sessions with labels
        Returns:
            bert_embeddings: BERT embeddings for sessions
            labels: Anomaly labels (0=normal, 1=anomaly) 
            confidence_targets: Target confidence scores
        """
        logger.info(f"Preparing training data from {len(sessions)} sessions")
        
        bert_embeddings = []
        labels = []
        confidence_targets = []
        
        for session in sessions:
            # Get session text and label
            session_text = session.get('raw_text', '')
            is_anomaly = session.get('is_anomaly', False)
            anomaly_score = session.get('anomaly_score', 0.0)
            
            if not session_text:
                continue
                
            # Generate BERT embeddings for session
            bert_embedding = self._get_bert_embedding(session_text)
            if bert_embedding is not None:
                bert_embeddings.append(bert_embedding)
                labels.append(1 if is_anomaly else 0)
                confidence_targets.append(anomaly_score)
        
        if not bert_embeddings:
            raise ValueError("No valid training data generated")
            
        # Convert to tensors
        bert_embeddings = torch.stack(bert_embeddings)
        labels = torch.tensor(labels, dtype=torch.long)
        confidence_targets = torch.tensor(confidence_targets, dtype=torch.float32)
        
        logger.info(f"Generated training data: {bert_embeddings.shape[0]} samples")
        return bert_embeddings, labels, confidence_targets
    
    def _get_bert_embedding(self, text: str) -> Optional[torch.Tensor]:
        """Generate BERT embedding for transaction text"""
        try:
            # Tokenize and encode
            inputs = self.bert_tokenizer(
                text, 
                return_tensors='pt', 
                max_length=self.max_sequence_length,
                padding='max_length',
                truncation=True
            )
            
            # Get BERT embeddings
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # Use [CLS] token embedding as session representation
                session_embedding = outputs.last_hidden_state[:, 0, :]  # (1, 768)
                
            return session_embedding.squeeze(0)  # (768,)
            
        except Exception as e:
            logger.error(f"Error generating BERT embedding: {e}")
            return None
    
    def train_model(self, training_sessions: List[Dict], 
                   validation_sessions: List[Dict] = None,
                   epochs: int = 50) -> Dict[str, float]:
        """
        Train the BERT + DeepLog model
        Args:
            training_sessions: Training data sessions
            validation_sessions: Optional validation data
            epochs: Number of training epochs
        Returns:
            Training metrics dictionary
        """
        logger.info("Starting BERT + DeepLog model training")
        
        # Prepare training data
        train_embeddings, train_labels, train_confidence = self.prepare_training_data(training_sessions)
        
        # Initialize model
        self.deeplog_model = BERTDeepLogModel(
            bert_dim=768,
            hidden_dim=256,
            num_layers=2
        ).to(self.device)
        
        # Training setup
        criterion_classification = nn.CrossEntropyLoss()
        criterion_confidence = nn.MSELoss()
        optimizer = torch.optim.Adam(self.deeplog_model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        # Training loop
        training_metrics = {
            'training_loss': [],
            'classification_accuracy': [],
            'confidence_mse': []
        }
        
        self.deeplog_model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            # Process in batches
            for i in range(0, len(train_embeddings), self.batch_size):
                batch_embeddings = train_embeddings[i:i+self.batch_size].to(self.device)
                batch_labels = train_labels[i:i+self.batch_size].to(self.device)
                batch_confidence = train_confidence[i:i+self.batch_size].to(self.device)
                
                # Add sequence dimension for LSTM
                batch_embeddings = batch_embeddings.unsqueeze(1)  # (batch, 1, 768)
                
                optimizer.zero_grad()
                
                # Forward pass
                anomaly_logits, confidence_scores = self.deeplog_model(batch_embeddings)
                
                # Remove sequence dimension for loss calculation
                anomaly_logits = anomaly_logits.squeeze(1)  # (batch, 2)
                confidence_scores = confidence_scores.squeeze()  # (batch,)
                
                # Calculate losses
                classification_loss = criterion_classification(anomaly_logits, batch_labels)
                confidence_loss = criterion_confidence(confidence_scores, batch_confidence)
                total_loss = classification_loss + 0.5 * confidence_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(anomaly_logits.data, 1)
                total_predictions += batch_labels.size(0)
                correct_predictions += (predicted == batch_labels).sum().item()
            
            scheduler.step()
            
            # Calculate epoch metrics
            avg_loss = epoch_loss / (len(train_embeddings) // self.batch_size + 1)
            accuracy = correct_predictions / total_predictions
            
            training_metrics['training_loss'].append(avg_loss)
            training_metrics['classification_accuracy'].append(accuracy)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
        
        # Save trained model
        self.save_model()
        
        final_metrics = {
            'final_loss': training_metrics['training_loss'][-1],
            'final_accuracy': training_metrics['classification_accuracy'][-1],
            'training_epochs': epochs,
            'model_parameters': sum(p.numel() for p in self.deeplog_model.parameters()),
            'training_samples': len(training_sessions)
        }
        
        logger.info(f"Training completed. Final accuracy: {final_metrics['final_accuracy']:.4f}")
        return final_metrics
    
    def predict_anomalies(self, sessions: List[Dict]) -> List[Dict]:
        """
        Predict anomalies for new sessions using trained model
        Args:
            sessions: List of transaction sessions to analyze
        Returns:
            List of anomaly predictions with confidence scores
        """
        if self.deeplog_model is None:
            raise ValueError("Model not trained or loaded")
            
        logger.info(f"Predicting anomalies for {len(sessions)} sessions")
        
        self.deeplog_model.eval()
        predictions = []
        
        with torch.no_grad():
            for session in sessions:
                session_text = session.get('raw_text', '')
                if not session_text:
                    predictions.append({
                        'session_id': session.get('session_id', 'unknown'),
                        'is_anomaly': False,
                        'anomaly_score': 0.0,
                        'prediction_method': 'bert_deeplog',
                        'error': 'No text available'
                    })
                    continue
                
                # Get BERT embedding
                bert_embedding = self._get_bert_embedding(session_text)
                if bert_embedding is None:
                    predictions.append({
                        'session_id': session.get('session_id', 'unknown'),
                        'is_anomaly': False,
                        'anomaly_score': 0.0,
                        'prediction_method': 'bert_deeplog',
                        'error': 'BERT embedding failed'
                    })
                    continue
                
                # Prepare for model input
                bert_embedding = bert_embedding.unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, 768)
                
                # Predict
                anomaly_logits, confidence_scores = self.deeplog_model(bert_embedding)
                
                # Process results
                probabilities = torch.softmax(anomaly_logits.squeeze(), dim=-1)
                anomaly_probability = probabilities[1].item()  # Probability of being anomaly
                confidence = confidence_scores.squeeze().item()
                
                # Determine final prediction
                is_anomaly = anomaly_probability > 0.5
                final_score = anomaly_probability * confidence
                
                predictions.append({
                    'session_id': session.get('session_id', 'unknown'),
                    'is_anomaly': is_anomaly,
                    'anomaly_score': final_score,
                    'anomaly_probability': anomaly_probability,
                    'model_confidence': confidence,
                    'prediction_method': 'bert_deeplog'
                })
        
        logger.info(f"Anomaly prediction completed. Found {sum(1 for p in predictions if p['is_anomaly'])} anomalies")
        return predictions
    
    def save_model(self):
        """Save the trained model and metadata"""
        if self.deeplog_model is None:
            raise ValueError("No model to save")
            
        model_path = os.path.join(self.model_save_path, "bert_deeplog_model.pt")
        metadata_path = os.path.join(self.model_save_path, "bert_deeplog_metadata.json")
        
        # Save model state
        torch.save({
            'model_state_dict': self.deeplog_model.state_dict(),
            'model_config': {
                'bert_dim': 768,
                'hidden_dim': 256,
                'num_layers': 2
            }
        }, model_path)
        
        # Save metadata
        metadata = {
            'model_type': 'BERTDeepLog',
            'training_date': datetime.now().isoformat(),
            'max_sequence_length': self.max_sequence_length,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self):
        """Load a previously trained model"""
        model_path = os.path.join(self.model_save_path, "bert_deeplog_model.pt")
        metadata_path = os.path.join(self.model_save_path, "bert_deeplog_metadata.json")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No trained model found at {model_path}")
            
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        model_config = checkpoint['model_config']
        
        self.deeplog_model = BERTDeepLogModel(**model_config).to(self.device)
        self.deeplog_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load metadata
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.max_sequence_length = metadata.get('max_sequence_length', 512)
                self.learning_rate = metadata.get('learning_rate', 0.001)
                self.batch_size = metadata.get('batch_size', 16)
        
        logger.info(f"Model loaded from {model_path}")

# Usage example for integration with existing system
def integrate_bert_deeplog_training():
    """Example integration with existing ML analyzer"""
    
    # Initialize trainer
    trainer = BERTDeepLogTrainer(model_save_path="/app/models")
    
    # Prepare training data from existing sessions (example)
    training_sessions = [
        {
            'session_id': 'session_1',
            'raw_text': 'TRANSACTION START\nCARD INSERTED\nNOTES PRESENTED\nNOTES TAKEN\nTRANSACTION END',
            'is_anomaly': False,
            'anomaly_score': 0.1
        },
        {
            'session_id': 'session_2', 
            'raw_text': 'TRANSACTION START\nCARD INSERTED\nUNABLE TO DISPENSE\nTRANSACTION END',
            'is_anomaly': True,
            'anomaly_score': 0.9
        }
        # Add more training sessions...
    ]
    
    # Train the model
    metrics = trainer.train_model(training_sessions, epochs=50)
    print(f"Training completed with accuracy: {metrics['final_accuracy']:.4f}")
    
    # Use for prediction
    new_sessions = [
        {
            'session_id': 'new_session_1',
            'raw_text': 'TRANSACTION START\nCARD INSERTED\nCARD TAKEN\nTRANSACTION END'
        }
    ]
    
    predictions = trainer.predict_anomalies(new_sessions)
    for pred in predictions:
        print(f"Session {pred['session_id']}: Anomaly={pred['is_anomaly']}, Score={pred['anomaly_score']:.3f}")

if __name__ == "__main__":
    integrate_bert_deeplog_training()
