"""
DeepLog Sequential Anomaly Detection for ABM Transaction Logs
Implements LSTM-based sequence modeling for detecting anomalous transaction patterns
"""
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import pickle
from typing import List, Dict, Tuple, Optional
import re
import os
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DeepLogLSTM(nn.Module):
    """DeepLog LSTM model for log sequence anomaly detection"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, 
                 hidden_dim: int = 256, num_layers: int = 2):
        super(DeepLogLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        output = self.fc(self.dropout(lstm_out))
        return output

class DeepLogAnalyzer:
    """Enhanced anomaly detection using DeepLog sequential modeling"""
    
    def __init__(self, window_size: int = 10, top_k: int = 9):
        self.window_size = window_size
        self.top_k = top_k
        self.event_vocabulary = {}
        self.reverse_vocab = {}
        self.model = None
        self.transaction_patterns = defaultdict(list)
        self.model_path = "/app/models/deeplog_model.pt"
        self.vocab_path = "/app/models/deeplog_vocab.json"
        
    def extract_log_events(self, session_text: str) -> List[str]:
        """Extract structured events from raw log text"""
        events = []
        lines = session_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Extract key events using patterns
            event = self._classify_log_event(line)
            if event:
                events.append(event)
                
        return events
    
    def _classify_log_event(self, line: str) -> Optional[str]:
        """Classify log line into semantic event types"""
        line_upper = line.upper()
        
        # Transaction boundaries
        if 'TRANSACTION START' in line_upper:
            return 'TXN_START'
        elif 'TRANSACTION END' in line_upper:
            return 'TXN_END'
            
        # Card operations
        elif 'CARD INSERTED' in line_upper:
            return 'CARD_INSERT'
        elif 'CARD TAKEN' in line_upper or 'CARD REMOVED' in line_upper:
            return 'CARD_REMOVE'
        elif 'CARD READ' in line_upper:
            return 'CARD_READ'
            
        # Authentication
        elif 'PIN ENTERED' in line_upper:
            return 'PIN_ENTRY'
        elif 'ATR RECEIVED' in line_upper:
            return 'CARD_ATR'
        elif 'PIN VERIFIED' in line_upper:
            return 'PIN_VERIFIED'
            
        # Transaction operations
        elif 'OPCODE = FI' in line_upper:
            return 'FINANCIAL_INQUIRY'
        elif 'OPCODE = BC' in line_upper:
            return 'BALANCE_CHECK'
        elif 'OPCODE = WD' in line_upper:
            return 'WITHDRAWAL'
        elif 'CASH DISPENSED' in line_upper:
            return 'CASH_DISPENSED'
        elif 'BALANCE DISPLAYED' in line_upper:
            return 'BALANCE_DISPLAYED'
            
        # System events
        elif 'SUPERVISOR MODE' in line_upper:
            return 'SUPERVISOR_MODE'
        elif 'UNABLE TO DISPENSE' in line_upper or 'DISPENSE ERROR' in line_upper:
            return 'DISPENSE_ERROR'
        elif 'POWER-UP' in line_upper or 'RESET' in line_upper:
            return 'SYSTEM_RESET'
        elif 'TIMEOUT' in line_upper:
            return 'TIMEOUT'
            
        # Communication events
        elif 'HOST RESPONSE' in line_upper:
            return 'HOST_RESPONSE'
        elif 'AUTHORIZATION' in line_upper:
            return 'AUTHORIZATION'
        elif 'DECLINED' in line_upper:
            return 'DECLINED'
            
        # Diagnostic messages
        elif re.match(r'\[000p\[040q', line):
            return 'DIAGNOSTIC_MSG'
        elif 'PRIMARY CARD READER ACTIVATED' in line_upper:
            return 'READER_ACTIVATED'
        elif 'RECEIPT PRINTED' in line_upper:
            return 'RECEIPT_PRINTED'
            
        # Error conditions
        elif 'ERROR' in line_upper:
            return 'ERROR_EVENT'
        elif 'FAILED' in line_upper:
            return 'FAILED_EVENT'
            
        return None  # Don't include unclassified events
    
    def build_vocabulary(self, sessions: List[str]) -> None:
        """Build event vocabulary from training sessions"""
        all_events = set()
        
        for session in sessions:
            events = self.extract_log_events(session)
            all_events.update(events)
            
        # Add special tokens
        all_events.add('<UNK>')
        all_events.add('<PAD>')
        
        # Create vocabulary mapping
        self.event_vocabulary = {event: idx for idx, event in enumerate(sorted(all_events))}
        self.reverse_vocab = {idx: event for event, idx in self.event_vocabulary.items()}
        
        logger.info(f"Built vocabulary with {len(self.event_vocabulary)} unique events")
        logger.info(f"Events: {list(self.event_vocabulary.keys())}")
        
    def events_to_sequence(self, events: List[str]) -> List[int]:
        """Convert event list to integer sequence"""
        return [self.event_vocabulary.get(event, self.event_vocabulary.get('<UNK>', 0)) for event in events]
    
    def create_training_sequences(self, sessions: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Create windowed sequences for training"""
        X, y = [], []
        
        for session in sessions:
            events = self.extract_log_events(session)
            if len(events) < self.window_size + 1:
                continue  # Skip sessions too short
                
            sequence = self.events_to_sequence(events)
            
            # Create sliding windows
            for i in range(len(sequence) - self.window_size):
                X.append(sequence[i:i + self.window_size])
                y.append(sequence[i + self.window_size])
                
        return np.array(X), np.array(y)
    
    def train_deeplog_model(self, sessions: List[str], epochs: int = 50, save_model: bool = True):
        """Train DeepLog LSTM model"""
        logger.info("Training DeepLog model...")
        
        # Build vocabulary
        self.build_vocabulary(sessions)
        
        # Create training data
        X_train, y_train = self.create_training_sequences(sessions)
        
        if len(X_train) == 0:
            logger.warning("No training sequences created - sessions may be too short")
            return
        
        logger.info(f"Created {len(X_train)} training sequences")
        
        # Initialize model
        vocab_size = len(self.event_vocabulary)
        self.model = DeepLogLSTM(vocab_size)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Convert to tensors
        X_tensor = torch.LongTensor(X_train)
        y_tensor = torch.LongTensor(y_train)
        
        # Training loop
        self.model.train()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            outputs = self.model(X_tensor)
            loss = criterion(outputs[:, -1, :], y_tensor)
            
            loss.backward()
            optimizer.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
        logger.info(f"DeepLog training completed! Best loss: {best_loss:.4f}")
        
        if save_model:
            self.save_model()
    
    def save_model(self):
        """Save trained model and vocabulary"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Save model
            if self.model:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'vocab_size': len(self.event_vocabulary),
                    'window_size': self.window_size,
                    'top_k': self.top_k
                }, self.model_path)
            
            # Save vocabulary
            with open(self.vocab_path, 'w') as f:
                json.dump({
                    'event_vocabulary': self.event_vocabulary,
                    'reverse_vocab': self.reverse_vocab
                }, f, indent=2)
            
            logger.info(f"Model and vocabulary saved to {self.model_path} and {self.vocab_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self) -> bool:
        """Load trained model and vocabulary"""
        try:
            if not os.path.exists(self.model_path) or not os.path.exists(self.vocab_path):
                logger.warning("Model or vocabulary files not found")
                return False
            
            # Load vocabulary
            with open(self.vocab_path, 'r') as f:
                vocab_data = json.load(f)
                self.event_vocabulary = vocab_data['event_vocabulary']
                self.reverse_vocab = {int(k): v for k, v in vocab_data['reverse_vocab'].items()}
            
            # Load model
            checkpoint = torch.load(self.model_path, map_location='cpu')
            vocab_size = checkpoint['vocab_size']
            
            self.model = DeepLogLSTM(vocab_size)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.window_size = checkpoint.get('window_size', self.window_size)
            self.top_k = checkpoint.get('top_k', self.top_k)
            
            logger.info("DeepLog model and vocabulary loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def detect_sequence_anomalies(self, session_text: str) -> Dict:
        """Detect anomalies using trained DeepLog model"""
        if not self.model:
            if not self.load_model():
                return {"error": "Model not available"}
            
        events = self.extract_log_events(session_text)
        sequence = self.events_to_sequence(events)
        
        if len(sequence) < self.window_size:
            return {
                "anomaly_detected": False,
                "reason": "Sequence too short for analysis",
                "events": events,
                "sequence_length": len(sequence),
                "required_length": self.window_size
            }
        
        anomalies = []
        self.model.eval()
        
        with torch.no_grad():
            for i in range(len(sequence) - self.window_size):
                window = sequence[i:i + self.window_size]
                expected_next = sequence[i + self.window_size]
                
                # Predict next event
                input_tensor = torch.LongTensor([window])
                output = self.model(input_tensor)
                predictions = torch.softmax(output[0, -1, :], dim=0)
                
                # Get top-k predictions
                top_k_values, top_k_indices = torch.topk(predictions, min(self.top_k, len(predictions)))
                top_k_indices = top_k_indices.tolist()
                
                # Check if actual next event is in top-k
                if expected_next not in top_k_indices:
                    confidence = 1.0 - predictions[expected_next].item()
                    anomalies.append({
                        "position": i + self.window_size,
                        "window_events": [self.reverse_vocab.get(idx, '<UNK>') for idx in window],
                        "expected_events": [self.reverse_vocab.get(idx, '<UNK>') for idx in top_k_indices[:3]],
                        "actual_event": self.reverse_vocab.get(expected_next, '<UNK>'),
                        "confidence": confidence,
                        "prediction_probability": predictions[expected_next].item()
                    })
        
        # Analyze transaction completeness
        transaction_analysis = self._analyze_transaction_completeness(events)
        
        return {
            "anomaly_detected": len(anomalies) > 0 or transaction_analysis["incomplete"],
            "sequence_anomalies": anomalies,
            "transaction_analysis": transaction_analysis,
            "events_sequence": events,
            "total_events": len(events),
            "analyzed_windows": len(sequence) - self.window_size if len(sequence) >= self.window_size else 0
        }
    
    def check_transaction_completeness(self, events: List[str]) -> Tuple[bool, float, List[str]]:
        """
        Public method to check transaction completeness
        Returns: (is_complete, completeness_score, missing_events)
        """
        analysis = self._analyze_transaction_completeness(events)
        completeness_score = self._calculate_completeness_score(events)
        
        is_complete = not analysis["incomplete"]
        missing_events = []
        
        if analysis["incomplete"]:
            missing_events = analysis["patterns"]
        
        return is_complete, completeness_score, missing_events
    
    def _analyze_transaction_completeness(self, events: List[str]) -> Dict:
        """Analyze if transaction completed normally"""
        has_start = 'TXN_START' in events
        has_end = 'TXN_END' in events
        has_card_insert = 'CARD_INSERT' in events
        has_card_remove = 'CARD_REMOVE' in events
        has_pin = 'PIN_ENTRY' in events
        has_operation = any(op in events for op in ['WITHDRAWAL', 'BALANCE_CHECK', 'FINANCIAL_INQUIRY'])
        has_cash_dispensed = 'CASH_DISPENSED' in events
        has_authorization = any(auth in events for auth in ['AUTHORIZATION', 'HOST_RESPONSE'])
        
        # Define incomplete patterns
        incomplete_patterns = []
        
        # Pattern 1: Card inserted/removed without PIN entry
        if has_card_insert and has_card_remove and not has_pin:
            incomplete_patterns.append("Card inserted/removed without PIN entry")
            
        # Pattern 2: PIN entered but no transaction operation
        if has_pin and not has_operation:
            incomplete_patterns.append("PIN entered but no transaction operation")
            
        # Pattern 3: Transaction started but not ended
        if has_start and not has_end:
            incomplete_patterns.append("Transaction started but not ended")
            
        # Pattern 4: Withdrawal requested but no cash dispensed
        if 'WITHDRAWAL' in events and not has_cash_dispensed:
            incomplete_patterns.append("Withdrawal requested but no cash dispensed")
            
        # Pattern 5: No authorization for financial operations
        if has_operation and not has_authorization:
            incomplete_patterns.append("Financial operation without authorization")
        
        # Check for quick card removal (minimal events between insert/remove)
        card_insert_idx = events.index('CARD_INSERT') if 'CARD_INSERT' in events else -1
        card_remove_idx = events.index('CARD_REMOVE') if 'CARD_REMOVE' in events else -1
        
        events_between = 0
        if card_insert_idx >= 0 and card_remove_idx >= 0:
            events_between = card_remove_idx - card_insert_idx
            if events_between <= 2:  # Very few events between insert/remove
                incomplete_patterns.append("Very quick card removal with minimal activity")
        
        return {
            "incomplete": len(incomplete_patterns) > 0,
            "patterns": incomplete_patterns,
            "transaction_metrics": {
                "has_start": has_start,
                "has_end": has_end,
                "has_pin": has_pin,
                "has_operation": has_operation,
                "has_authorization": has_authorization,
                "events_between_card_ops": events_between
            },
            "completeness_score": self._calculate_completeness_score(events)
        }
    
    def _calculate_completeness_score(self, events: List[str]) -> float:
        """Calculate a completeness score (0-1) for the transaction"""
        required_events = ['TXN_START', 'CARD_INSERT', 'PIN_ENTRY', 'TXN_END', 'CARD_REMOVE']
        optional_events = ['CARD_ATR', 'HOST_RESPONSE', 'AUTHORIZATION']
        
        required_count = sum(1 for event in required_events if event in events)
        optional_count = sum(1 for event in optional_events if event in events)
        
        # Base score from required events
        base_score = required_count / len(required_events)
        
        # Bonus from optional events
        bonus_score = min(0.2, optional_count * 0.1)
        
        return min(1.0, base_score + bonus_score)

def create_normal_training_patterns() -> List[str]:
    """Create normal transaction patterns for DeepLog training"""
    
    normal_patterns = [
        # Complete cash withdrawal
        """TXN_START
CARD_INSERT
CARD_ATR
PIN_ENTRY
WITHDRAWAL
AUTHORIZATION
HOST_RESPONSE
CASH_DISPENSED
RECEIPT_PRINTED
CARD_REMOVE
TXN_END""",
        
        # Balance inquiry
        """TXN_START
CARD_INSERT
CARD_ATR
PIN_ENTRY
BALANCE_CHECK
AUTHORIZATION
HOST_RESPONSE
BALANCE_DISPLAYED
CARD_REMOVE
TXN_END""",
        
        # Financial inquiry
        """TXN_START
CARD_INSERT
CARD_ATR
PIN_ENTRY
FINANCIAL_INQUIRY
AUTHORIZATION
HOST_RESPONSE
BALANCE_DISPLAYED
CARD_REMOVE
TXN_END""",
        
        # Declined transaction
        """TXN_START
CARD_INSERT
CARD_ATR
PIN_ENTRY
WITHDRAWAL
AUTHORIZATION
DECLINED
CARD_REMOVE
TXN_END""",
        
        # PIN verification failure
        """TXN_START
CARD_INSERT
CARD_ATR
PIN_ENTRY
PIN_ENTRY
PIN_ENTRY
FAILED_EVENT
CARD_REMOVE
TXN_END""",
    ]
    
    # Create variations with different orders and additional events
    variations = []
    for pattern in normal_patterns:
        # Add diagnostic messages
        events = pattern.strip().split('\n')
        
        # Variation 1: Add diagnostic messages
        enhanced = []
        for i, event in enumerate(events):
            enhanced.append(event)
            if i % 3 == 0 and i > 0:
                enhanced.append('DIAGNOSTIC_MSG')
        variations.append('\n'.join(enhanced))
        
        # Variation 2: Add reader activation
        if 'CARD_INSERT' in events:
            idx = events.index('CARD_INSERT')
            events_copy = events.copy()
            events_copy.insert(idx + 1, 'READER_ACTIVATED')
            variations.append('\n'.join(events_copy))
    
    return normal_patterns + variations

if __name__ == "__main__":
    # Test the DeepLog analyzer
    analyzer = DeepLogAnalyzer()
    
    # Create training data
    training_patterns = create_normal_training_patterns()
    
    # Train the model
    analyzer.train_deeplog_model(training_patterns, epochs=30)
    
    print("DeepLog analyzer trained and ready for use!")
