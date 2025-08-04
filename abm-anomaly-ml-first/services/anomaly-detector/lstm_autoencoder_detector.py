"""
LSTM Autoencoder for EJ Session Anomaly Detection
Learns to reconstruct normal sequences; anomalies have high reconstruction error
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
import re

logger = logging.getLogger(__name__)

class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for sequence anomaly detection
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        """
        Initialize LSTM Autoencoder
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(LSTMAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer to reconstruct input
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass through autoencoder
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Reconstructed sequence
        """
        batch_size, seq_len, _ = x.shape
        
        # Encoder
        encoded, (hidden, cell) = self.encoder(x)
        
        # Use last hidden state as context vector
        context = encoded[:, -1, :].unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # Repeat context for decoder sequence
        decoder_input = context.repeat(1, seq_len, 1)  # (batch_size, seq_len, hidden_dim)
        
        # Decoder
        decoded, _ = self.decoder(decoder_input, (hidden, cell))
        decoded = self.dropout(decoded)
        
        # Reconstruct input
        reconstructed = self.output_layer(decoded)
        
        return reconstructed

class LSTMAutoencoderAnomalyDetector:
    """
    LSTM Autoencoder-based anomaly detector for EJ sessions
    """
    
    def __init__(self, model_dir="/app/data/models", sequence_length=20):
        """
        Initialize LSTM Autoencoder Anomaly Detector
        
        Args:
            model_dir: Directory to save models
            sequence_length: Length of input sequences
        """
        self.model_dir = model_dir
        self.sequence_length = sequence_length
        os.makedirs(model_dir, exist_ok=True)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Feature extraction
        self.vectorizer = TfidfVectorizer(
            max_features=100,  # Smaller for sequence processing
            ngram_range=(1, 2),
            lowercase=True,
            token_pattern=r'\b\w+(?:[-/]\w+)*\b'
        )
        
        self.scaler = StandardScaler()
        
        # Model parameters
        self.feature_dim = None
        self.model = None
        self.model_trained = False
        
        # Training parameters
        self.learning_rate = 0.001
        self.num_epochs = 100
        self.batch_size = 32
        
        # Anomaly detection threshold (will be calculated from training data)
        self.threshold = None
        
    def extract_sequence_features(self, session_text: str) -> np.ndarray:
        """
        Extract features for sequence processing
        """
        # Split session into lines/events
        lines = [line.strip() for line in session_text.split('\n') if line.strip()]
        
        # Extract features for each line
        line_features = []
        
        for line in lines:
            # Text features (TF-IDF for this line)
            if self.model_trained:
                text_vec = self.vectorizer.transform([line]).toarray()[0]
            else:
                text_vec = None
            
            # Manual features for this line
            features = {
                'line_length': len(line),
                'has_error': int(bool(re.search(r'error|fail|malfunction', line.lower()))),
                'has_hardware': int(bool(re.search(r'hardware|power|reset|cim', line.lower()))),
                'has_transaction': int(bool(re.search(r'card|pin|cash|withdraw|deposit', line.lower()))),
                'has_numbers': int(bool(re.search(r'\d+', line))),
                'has_uppercase': int(any(c.isupper() for c in line)),
                'word_count': len(line.split()),
                'is_critical': int(bool(re.search(r'power-up/reset|hardware.*error|recovery.*failed', line.lower())))
            }
            
            if text_vec is not None:
                # Combine text and manual features
                combined = np.concatenate([text_vec, list(features.values())])
            else:
                combined = np.array(list(features.values()))
            
            line_features.append(combined)
        
        # Pad or truncate to sequence_length
        if len(line_features) > self.sequence_length:
            line_features = line_features[:self.sequence_length]
        elif len(line_features) < self.sequence_length:
            # Pad with zeros
            if line_features:
                feature_dim = len(line_features[0])
                padding = [np.zeros(feature_dim) for _ in range(self.sequence_length - len(line_features))]
                line_features.extend(padding)
            else:
                # Empty session
                feature_dim = self.feature_dim if self.feature_dim else 8  # fallback
                line_features = [np.zeros(feature_dim) for _ in range(self.sequence_length)]
        
        return np.array(line_features)
    
    def prepare_training_data(self, ej_sessions: List[Dict]) -> np.ndarray:
        """
        Prepare training data from normal sessions only
        """
        logger.info(f"Preparing LSTM Autoencoder training data from {len(ej_sessions)} sessions")
        
        # Filter normal sessions
        normal_sessions = [
            session for session in ej_sessions 
            if not session.get('is_anomaly', False)
        ]
        
        logger.info(f"Using {len(normal_sessions)} normal sessions for training")
        
        # Collect all session texts for TF-IDF fitting
        all_lines = []
        for session in normal_sessions:
            session_text = session.get('raw_text', session.get('text', ''))
            lines = [line.strip() for line in session_text.split('\n') if line.strip()]
            all_lines.extend(lines)
        
        # Fit TF-IDF vectorizer
        if all_lines:
            self.vectorizer.fit(all_lines)
        
        # Extract sequence features
        sequences = []
        for session in normal_sessions:
            session_text = session.get('raw_text', session.get('text', ''))
            if session_text.strip():
                sequence = self.extract_sequence_features(session_text)
                sequences.append(sequence)
        
        if not sequences:
            raise ValueError("No valid training sequences found")
        
        sequences_array = np.array(sequences)
        
        # Store feature dimension
        self.feature_dim = sequences_array.shape[2]
        
        # Scale features
        original_shape = sequences_array.shape
        sequences_flat = sequences_array.reshape(-1, self.feature_dim)
        sequences_scaled = self.scaler.fit_transform(sequences_flat)
        sequences_array = sequences_scaled.reshape(original_shape)
        
        logger.info(f"Prepared {len(sequences)} sequences of shape {sequences_array.shape}")
        
        return sequences_array
    
    def train_model(self, ej_sessions: List[Dict]):
        """
        Train the LSTM Autoencoder
        """
        logger.info("Starting LSTM Autoencoder training")
        
        # Prepare training data
        X_train = self.prepare_training_data(ej_sessions)
        
        # Initialize model
        self.model = LSTMAutoencoder(
            input_dim=self.feature_dim,
            hidden_dim=64,
            num_layers=2,
            dropout=0.1
        ).to(self.device)
        
        # Training setup
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Convert to tensor
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        
        # Training loop
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            
            for i in range(0, len(X_train), self.batch_size):
                batch = X_train_tensor[i:i+self.batch_size]
                
                optimizer.zero_grad()
                
                # Forward pass
                reconstructed = self.model(batch)
                loss = criterion(reconstructed, batch)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / (len(X_train) // self.batch_size)
                logger.info(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.6f}")
        
        # Calculate threshold from training data
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_train_tensor)
            reconstruction_errors = torch.mean((X_train_tensor - reconstructed) ** 2, dim=(1, 2))
            
            # Set threshold as 95th percentile of training errors
            self.threshold = float(torch.quantile(reconstruction_errors, 0.95))
        
        self.model_trained = True
        logger.info(f"Training completed. Anomaly threshold: {self.threshold:.6f}")
        
        return {
            'training_samples': len(X_train),
            'feature_dim': self.feature_dim,
            'final_loss': total_loss / (len(X_train) // self.batch_size),
            'anomaly_threshold': self.threshold
        }
    
    def predict_anomaly(self, session_text: str, session_id: str = None) -> Dict[str, Any]:
        """
        Predict anomaly using reconstruction error
        """
        if not self.model_trained:
            raise ValueError("Model not trained. Call train_model() first.")
        
        try:
            # Extract sequence features
            sequence = self.extract_sequence_features(session_text)
            
            # Scale features
            sequence_flat = sequence.reshape(-1, self.feature_dim)
            sequence_scaled = self.scaler.transform(sequence_flat)
            sequence = sequence_scaled.reshape(1, self.sequence_length, self.feature_dim)
            
            # Convert to tensor
            sequence_tensor = torch.FloatTensor(sequence).to(self.device)
            
            # Predict
            self.model.eval()
            with torch.no_grad():
                reconstructed = self.model(sequence_tensor)
                
                # Calculate reconstruction error
                reconstruction_error = torch.mean((sequence_tensor - reconstructed) ** 2).item()
                
                # Determine if anomalous
                is_anomaly = reconstruction_error > self.threshold
                
                # Convert to probability-like score
                if self.threshold > 0:
                    anomaly_probability = min(reconstruction_error / self.threshold, 1.0)
                else:
                    anomaly_probability = 1.0 if is_anomaly else 0.0
                
                # Calculate confidence
                confidence = abs(reconstruction_error - self.threshold) / max(self.threshold, reconstruction_error)
            
            result = {
                'session_id': session_id,
                'is_anomaly': bool(is_anomaly),
                'anomaly_probability': float(anomaly_probability),
                'confidence': float(confidence),
                'reconstruction_error': float(reconstruction_error),
                'threshold': float(self.threshold),
                'prediction_timestamp': datetime.now().isoformat(),
                'detection_method': 'lstm_autoencoder'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting anomaly for session {session_id}: {e}")
            return {
                'error': str(e),
                'session_id': session_id,
                'is_anomaly': False,
                'anomaly_probability': 0.0
            }
    
    def save_model(self, model_path: str = None):
        """Save the trained model"""
        if model_path is None:
            model_path = os.path.join(self.model_dir, 'lstm_autoencoder_model.pth')
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vectorizer': self.vectorizer,
            'scaler': self.scaler,
            'feature_dim': self.feature_dim,
            'threshold': self.threshold,
            'sequence_length': self.sequence_length,
            'model_trained': self.model_trained
        }, model_path)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str = None):
        """Load a saved model"""
        if model_path is None:
            model_path = os.path.join(self.model_dir, 'lstm_autoencoder_model.pth')
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.feature_dim = checkpoint['feature_dim']
        self.threshold = checkpoint['threshold']
        self.sequence_length = checkpoint['sequence_length']
        self.model_trained = checkpoint['model_trained']
        self.vectorizer = checkpoint['vectorizer']
        self.scaler = checkpoint['scaler']
        
        # Recreate model
        self.model = LSTMAutoencoder(
            input_dim=self.feature_dim,
            hidden_dim=64,
            num_layers=2,
            dropout=0.1
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Model loaded from {model_path}")

# Alias for compatibility  
BERTDeepLogAnomalyDetector = LSTMAutoencoderAnomalyDetector
