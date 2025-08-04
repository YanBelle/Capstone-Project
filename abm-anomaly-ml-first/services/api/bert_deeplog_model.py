"""
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
try:
    from bertviz_analyzer import BertVisualizationAnalyzer
    BERTVIZ_AVAILABLE = True
except ImportError:
    BertVisualizationAnalyzer = None
    BERTVIZ_AVAILABLE = False

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
        self.sequence_predictor = nn.Linear(hidden_dim, hidden_dim // 2)  # Next event prediction
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
        sequence_pred = self.sequence_predictor(combined)   # (batch_size, seq_len, hidden_dim//2)
        
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
        
        # Initialize device first before any model operations
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize BERT components
        if BERTVIZ_AVAILABLE:
            self.bert_analyzer = BertVisualizationAnalyzer(model_name=bert_model_name)
        else:
            # Fallback: use basic BERT tokenization without BertVisualizationAnalyzer
            from transformers import BertTokenizer, BertModel
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
            self.bert_model = BertModel.from_pretrained(bert_model_name)
            self.bert_analyzer = None
            logger.warning("BertVisualizationAnalyzer not available, using basic BERT tokenizer")
            
            # Add custom ATM/EJ domain tokens to prevent splitting (same as bertviz_analyzer.py)
            custom_tokens = [
                # Core ATM events - compound terms
                "DEVICE_ERROR", "CARD_INSERTED", "CARD_TAKEN", "PIN_ENTERED", 
                "ATR_RECEIVED", "TRANSACTION_START", "TRANSACTION_END",
                "CASH_DISPENSED", "BALANCE_INQUIRY", "RECEIPT_PRINTED", 
                "CARD_RETAINED", "CARD_EJECTED", "CARD_READ",
                
                # Error states
                "TIMEOUT_ERROR", "COMMUNICATION_ERROR", "NETWORK_ERROR", 
                "CASH_DISPENSER_ERROR", "READ_ERROR", "WRITE_ERROR",
                
                # Account and validation
                "ACCOUNT_VALIDATION", "PIN_VALIDATION", "INSUFFICIENT_FUNDS", 
                "INVALID_PIN", "CARD_EXPIRED",
                
                # Transaction types
                "WITHDRAWAL_TRANSACTION", "DEPOSIT_TRANSACTION", "TRANSFER_TRANSACTION",
                
                # Status indicators
                "OUT_OF_SERVICE", "OUT_OF_CASH", "OUT_OF_ORDER", 
                "SERVICE_MODE", "DIAGNOSTIC_MODE",
                
                # Specific patterns that appear in EJ logs
                "CardNumber", "REF", "VAL", "ESC", "REJECTS",
                
                # Common combined patterns
                "VAL_000", "ESC_000", "REF_000", "REJECTS_000",
                "OPCODE_FI", "OPCODE_IB", "OPCODE_IC", "OPCODE_ID", "OPCODE_BBC",
                "ATR_RECEIVED_T_0", "ATR_RECEIVED_T_1",
                
                # Cash handling events that should remain as single tokens
                "NOTES_STACKED", "NOTES_PRESENTED", "NOTES_TAKEN",
                "CASH_DISPENSED_SUMMARY", "PRIMARY_CARD_READER_ACTIVATED",
                
                # Common Machine and R status patterns to prevent fragmentation
                "M_00", "M_01", "M_02", "M_03", "M_04", "M_05", "M_10", "M_15", "M_20", "M_99",
                "R_0000", "R_5005", "R_10011", "R_20001", "R_30015", "R_40000", "R_50000"
            ]
            
            # Add tokens to tokenizer vocabulary
            num_added_tokens = self.bert_tokenizer.add_tokens(custom_tokens)
            logger.info(f"Added {num_added_tokens} custom ATM domain tokens to BERT tokenizer")
            
            # Resize model embeddings to accommodate new tokens
            self.bert_model.resize_token_embeddings(len(self.bert_tokenizer))
            self.bert_model.to(self.device)
            self.bert_model.eval()
        
        # Initialize DeepLog model
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
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess ABM log text for BERT analysis with enhanced pattern cleaning
        Uses the same methodology as bertviz_analyzer.py for consistency
        """
        # Enhanced EJ pattern cleaning with specific fixes for BERT attention optimization
        
        # NEW: NOISE REDUCTION - Replace verbose sections with concise event labels
        # 1. Replace Cash Dispensing Summary with concise event
        # Enhanced pattern to match various cash dispensing table formats
        cash_summary_pattern = r'CASH\s+TOTAL\s+TYPE\d+.*?REMAINING\s+\d+(?:\s+\d+)*'
        text = re.sub(cash_summary_pattern, 'CASH_DISPENSED_SUMMARY', text, flags=re.DOTALL)
        
        # 2. Replace Receipt section with concise event - ENHANCED for NCB format
        # Pattern 1: NCB MIDAS format - Bank name + branch + detailed receipt ending with THANK YOU
        receipt_pattern1 = r'N\.C\.B\.\s+MIDAS\s+NCB\s+[A-Z\s\.]+BRANCH.*?THANK YOU'
        text = re.sub(receipt_pattern1, ' RECEIPT_PRINTED ', text, flags=re.DOTALL)
        
        # Pattern 2: General bank name + receipt content ending with THANK YOU (with proper spacing)
        receipt_pattern2 = r'([A-Z][A-Z\s\.]+(?:BANK|CREDIT UNION|ATM)[^\n]*\n(?:.*?\n)*?.*?THANK YOU)'
        text = re.sub(receipt_pattern2, ' RECEIPT_PRINTED ', text, flags=re.DOTALL)
        
        # Pattern 3: DATE/TIME/MACHINE format receipts  
        receipt_pattern3 = r'(DATE:\s*[^\n]*\n(?:.*?\n)*?.*?THANK YOU)'
        text = re.sub(receipt_pattern3, ' RECEIPT_PRINTED ', text, flags=re.DOTALL)
        
        # Pattern 4: Simple receipt format with institution names
        receipt_pattern4 = r'(\s+[A-Z][A-Z\s]+\n\s*(?:ATM|RECEIPT)[^\n]*\n(?:.*?\n)*?.*?THANK YOU)'
        text = re.sub(receipt_pattern4, ' RECEIPT_PRINTED ', text, flags=re.DOTALL)
        
        # CRITICAL FIRST: Handle ESC/VAL/REF patterns BEFORE any other cleanup removes the values
        # Convert VAL: 000, ESC: 000, REF: 000 patterns to compound tokens
        text = re.sub(r'\b(VAL|ESC|REF):\s*(\d+)\b', r'\1_\2', text)
        # Handle cases like "ESC 000" (without colon), "VAL   000" (multiple spaces)
        text = re.sub(r'\b(VAL|ESC|REF)\s+(\d+)\b', r'\1_\2', text)
        
        # CRITICAL SECOND: Handle ATR pattern IMMEDIATELY after ESC/VAL/REF
        text = re.sub(r'\bATR\s+RECEIVED\s+T=(\d+)\b', r'ATR_RECEIVED_T_\1', text)
        
        # CRITICAL THIRD: Handle REJECTS patterns early to prevent "1" token isolation
        # Clean up "REJECTS:000*(1" patterns that create isolated "1" tokens
        text = re.sub(r'REJECTS:(\d+)\*\([^)]*\)', r'REJECTS_\1', text)
        text = re.sub(r'REJECTS:(\d+)', r'REJECTS_\1', text)
        text = re.sub(r'REJECTS\s+(\d+)', r'REJECTS_\1', text)
        
        # NEW: Handle specific patterns from current EJ sample
        # 1. Remove asterisks around PRIMARY CARD READER ACTIVATED
        text = re.sub(r'\*PRIMARY CARD READER ACTIVATED\*', 'PRIMARY_CARD_READER_ACTIVATED', text)
        
        # 2. Handle NOTES patterns - convert to compound tokens and remove comma-separated numbers
        # NOTES PRESENTED followed by comma-separated numbers -> NOTES_PRESENTED
        text = re.sub(r'\bNOTES\s+PRESENTED\s+[\d,\s]+', 'NOTES_PRESENTED', text)
        # NOTES STACKED -> NOTES_STACKED
        text = re.sub(r'\bNOTES\s+STACKED\b', 'NOTES_STACKED', text)
        # NOTES TAKEN -> NOTES_TAKEN  
        text = re.sub(r'\bNOTES\s+TAKEN\b', 'NOTES_TAKEN', text)
        
        # 3. Handle additional OPCODE patterns
        text = re.sub(r'\bOPCODE\s*=\s*(BBC)\b', r'OPCODE_\1', text)
        
        # 1. Remove EJ header patterns: [020t*629*06/18/2025*00:46*
        # Pattern: [020t*<sequence>*<mm/dd/yyyy>*<hh:mm>*
        text = re.sub(r'\[020t\*\d+\*\d{2}/\d{2}/\d{4}\*\d{2}:\d{2}\*', '', text)
        
        # 1b. Remove standalone date/time patterns that don't start with [020t
        # Pattern: *630*06/18/2025*00:46* (removes patterns like "*630*06/18/2025*00:46*")
        text = re.sub(r'\*\d+\*\d{2}/\d{2}/\d{4}\*\d{2}:\d{2}\*', '', text)
        
        # 1c. ENHANCED: Remove complex transaction code patterns that cause fragmentation
        # Pattern: *7231*1*(Iw(1*3, (removes patterns like "*7231*1*(Iw(1*3,")
        # This is the main source of ##31, ##1, ##w noise tokens
        # IMPROVED: More aggressive pattern to catch the full "*7231*1*(Iw(1*3," structure
        text = re.sub(r'\*\d+\*\d+\*\([^,)]*,?\s*', '', text)
        
        # 1d. Remove any remaining complex patterns with asterisks and parentheses
        # This catches patterns like "*7231*1*(Iw(1*3," more aggressively with better coverage
        text = re.sub(r'\*\d+\*\d+\*\([^)]*\)[^,]*,?\s*', '', text)
        
        # 1e. AGGRESSIVE CLEANUP: Remove any remaining transaction code fragments
        # Enhanced to catch more fragment patterns that create isolated tokens
        text = re.sub(r'\*\d+\*', '', text)  # Remove *digits*
        text = re.sub(r'\*\([^)]*\)', '', text)  # Remove *(content)
        text = re.sub(r'\([^)]*\*\d+', '', text)  # Remove (content*digits
        text = re.sub(r'\(Iw\([^)]*\)', '', text)  # Remove (Iw(content) pattern more completely
        text = re.sub(r'\(\d+\*\d+[^)]*\)', '', text)  # Remove (digits*digits*content) patterns
        
        # 1f. SPECIFIC FIX: Remove the exact "*7231*1*(Iw(1*3," pattern completely
        # This targets the specific pattern causing the isolated "1" token
        text = re.sub(r'\*7231\*1\*\(Iw\(1\*3,?\s*', '', text)
        
        # 2. Remove remaining [020t patterns with any following content
        # This catches patterns like "[020t CARD INSERTED", "[020t 00:47:13", etc.
        text = re.sub(r'\[020t\s+', '', text)
        
        # 3. Remove standalone timestamps in format hh:mm:ss (before main events)
        # Matches patterns like " 00:46:27 ", " 00:46:30 ", etc.
        text = re.sub(r'\s*\d{2}:\d{2}:\d{2}\s+', ' ', text)
        
        # 3b. Remove standalone timestamps in format hh:mm (without seconds)
        # Matches patterns like " 00:47 ", " 00:46 ", etc.
        text = re.sub(r'\s*\d{2}:\d{2}\s+', ' ', text)
        
        # 3b2. Remove partial timestamps that remain after aggressive cleanup
        # Catches patterns like "05:50:" or "00::" left behind
        text = re.sub(r'\d{2}::\s*', '', text)  # Remove xx:: patterns
        text = re.sub(r'\d{2}:\d{2}:\s*', '', text)  # Remove xx:xx: patterns
        
        # 3c. ENHANCED PATTERN: Aggressively remove isolated numeric fragments that are likely noise
        # Uses context-aware removal - preserves meaningful amounts/counts but removes ALL isolated noise digits
        # First, protect meaningful numeric contexts by temporarily marking them with placeholder tokens
        text = re.sub(r'(AMOUNT)\s+(\d+)', r'PROTECTED_\1_\2', text)
        text = re.sub(r'(COUNT)\s+(\d+)', r'PROTECTED_\1_\2', text)
        text = re.sub(r'(TOTAL)\s+(\d+)', r'PROTECTED_\1_\2', text)
        text = re.sub(r'(BALANCE)\s+(\d+)', r'PROTECTED_\1_\2', text)
        text = re.sub(r'(STEP)_(\d+)', r'PROTECTED_\1_\2', text)  # Protect STEP_1 patterns
        text = re.sub(r'(T)_(\d+)', r'PROTECTED_\1_\2', text)      # Protect T_1 patterns
        
        # AGGRESSIVE: Remove ALL isolated single digits that appear between words or at boundaries
        # This will catch the isolated "1" token regardless of context
        text = re.sub(r'\b\d\b', '', text)  # Remove any single isolated digit
        
        # Also remove isolated multi-digit fragments that are likely noise (2-4 digits)
        text = re.sub(r'\b\d{2,4}\b(?=\s+(?:[A-Z][A-Z_]+|[a-z]+)|\s*$)', '', text)
        
        # Restore protected meaningful numbers
        text = re.sub(r'PROTECTED_(AMOUNT|COUNT|TOTAL|BALANCE|STEP|T)_(\d+)', r'\1_\2', text)
        
        # 3d. CONTEXTUAL FRAGMENT REMOVAL: Remove isolated single chars/digits between meaningful terms
        # Targets fragments like "w", "i", "1", "3" that appear isolated between proper ATM terms
        text = re.sub(r'(?<=\s)[a-zA-Z0-9](?=\s+[A-Z_]|\s*$)', '', text)
        
        # 4. Remove transaction start markers
        text = re.sub(r'\s*---START OF TRANSACTION---\s*', ' ', text)
        
        # 5. ENHANCED PATTERN CLEANING - Fix specific issues with punctuation and compound words
        
        # Replace *TRANSACTION START* with TRANSACTION START (remove asterisks)
        text = re.sub(r'\*TRANSACTION START\*', 'TRANSACTION_START', text)
        
        # Replace PAN patterns with simplified CardNumber label
        # Matches: "PAN 0004263********1897" or similar patterns
        text = re.sub(r'PAN\s+\d{4}\d+\*+\d+', 'CardNumber', text)
        
        # Remove complex transaction codes like "*7231*1*(Iw(1*3," but keep meaningful parts
        # Pattern: *digits*digits*(complex_chars*digits, -> keep what follows after comma
        text = re.sub(r'\*\d+\*\d+\*\([^,]+,\s*', '', text)
        
        # Remove "A/C" as requested
        text = re.sub(r'\bA/C\b', '', text)
        
        # Clean up "REJECTS:000*(1\nS" to just "REJECTS_000"
        text = re.sub(r'REJECTS:(\d+)\*\([^)]*\)\s*S?', r'REJECTS_\1', text)
        
        # Additional cleanup for any remaining REJECTS fragments
        text = re.sub(r'REJECTS:(\d+)\*\([^)]*\)[^A-Z]*', r'REJECTS_\1', text)
        
        # Handle remaining REJECTS:000 patterns that don't have the full pattern
        text = re.sub(r'REJECTS:(\d+)', r'REJECTS_\1', text)
        
        # Remove standalone "S" that might be left from REJECTS patterns
        text = re.sub(r'\bS\b(?=\s|$)', '', text)
        
        # 8. Convert specific transaction elements to single tokens (AFTER all value processing is complete)
        # These patterns should only run after all value extraction/cleaning above
        text = re.sub(r'\bTRACK\s+\d+\s+DATA\b', 'TRACK_DATA', text)
        text = re.sub(r'\bT=(\d+)\b', r'T_\1', text)
        text = re.sub(r'\bLEN=(\d+)\b', r'LEN_\1', text)
        text = re.sub(r'\bSTEP\s+(\d+)\b', r'STEP_\1', text)
        text = re.sub(r'\bTIME\s*=\s*(\d+)\b', r'TIME_\1', text)
        text = re.sub(r'\bCOUNT\s*=\s*(\d+)\b', r'COUNT_\1', text)
        text = re.sub(r'\bDATA\s+LEN\s*=\s*(\d+)\b', r'DATA_LEN_\1', text)
        
        # 9. Handle specific values: Convert hex patterns and structured values
        
        # ENHANCED: Handle REJECTS patterns with different formatting
        text = re.sub(r'REJECTS\s+(\d+)', r'REJECTS_\1', text)
        
        # Convert OPCODE = <code> to OPCODE_<code>
        text = re.sub(r'\bOPCODE\s*=\s*([A-Z]+)\b', r'OPCODE_\1', text)
        
        # Additional noise cleanup - remove isolated asterisks and punctuation fragments
        text = re.sub(r'\*+', '', text)
        text = re.sub(r'[()]+', '', text)
        
        # Clean specific EJ patterns that cause fragmentation
        # Convert M-<digits>, R-<digits> to compound tokens to prevent BERT fragmentation
        # Machine status: M-02, M-15, etc. -> M_02, M_15, etc.
        text = re.sub(r'\bM-(\d+),?\s*', r'M_\1 ', text)
        # R status: R-10011, R-5005, etc. -> R_10011, R_5005, etc.
        text = re.sub(r'\bR-(\d+)\b', r'R_\1', text)
        
        # Create compound tokens for ATM events that should stay together
        # This prevents BERT from splitting important multi-word terms
        compound_patterns = {
            # Core ATM events
            r'\bDEVICE\s+ERROR\b': 'DEVICE_ERROR',
            r'\bCARD\s+INSERTED\b': 'CARD_INSERTED', 
            r'\bCARD\s+TAKEN\b': 'CARD_TAKEN',
            r'\bPIN\s+ENTERED\b': 'PIN_ENTERED',
            r'\bTRANSACTION\s+END\b': 'TRANSACTION_END',
            r'\bTRANSACTION\s+START\b': 'TRANSACTION_START',
            
            # Additional ATM operations
            r'\bCASH\s+DISPENSED\b': 'CASH_DISPENSED',
            r'\bBALANCE\s+INQUIRY\b': 'BALANCE_INQUIRY',
            r'\bRECEIPT\s+PRINTED\b': 'RECEIPT_PRINTED',
            r'\bCARD\s+RETAINED\b': 'CARD_RETAINED',
            r'\bCARD\s+EJECTED\b': 'CARD_EJECTED',
            r'\bCARD\s+read\b': 'CARD_READ',
            
            # NEW: Cash handling events (these were handled earlier but ensure consistency)
            r'\bNOTES\s+STACKED\b': 'NOTES_STACKED',
            r'\bNOTES\s+PRESENTED\b': 'NOTES_PRESENTED', 
            r'\bNOTES\s+TAKEN\b': 'NOTES_TAKEN',
            r'\bPRIMARY\s+CARD\s+READER\s+ACTIVATED\b': 'PRIMARY_CARD_READER_ACTIVATED',
            
            # Error states and conditions
            r'\bTIMEOUT\s+ERROR\b': 'TIMEOUT_ERROR',
            r'\bCOMMUNICATION\s+ERROR\b': 'COMMUNICATION_ERROR',
            r'\bNETWORK\s+ERROR\b': 'NETWORK_ERROR',
            r'\bCASH\s+DISPENSER\s+ERROR\b': 'CASH_DISPENSER_ERROR',
            r'\bREAD\s+ERROR\b': 'READ_ERROR',
            r'\bWRITE\s+ERROR\b': 'WRITE_ERROR',
            
            # Account and validation
            r'\bACCOUNT\s+VALIDATION\b': 'ACCOUNT_VALIDATION',
            r'\bPIN\s+VALIDATION\b': 'PIN_VALIDATION',
            r'\bINSUFFICIENT\s+FUNDS\b': 'INSUFFICIENT_FUNDS',
            r'\bINVALID\s+PIN\b': 'INVALID_PIN',
            r'\bCARD\s+EXPIRED\b': 'CARD_EXPIRED',
            
            # Transaction types
            r'\bWITHDRAWAL\s+TRANSACTION\b': 'WITHDRAWAL_TRANSACTION',
            r'\bDEPOSIT\s+TRANSACTION\b': 'DEPOSIT_TRANSACTION',
            r'\bTRANSFER\s+TRANSACTION\b': 'TRANSFER_TRANSACTION',
            
            # Status indicators
            r'\bOUT\s+OF\s+SERVICE\b': 'OUT_OF_SERVICE',
            r'\bOUT\s+OF\s+CASH\b': 'OUT_OF_CASH',
            r'\bOUT\s+OF\s+ORDER\b': 'OUT_OF_ORDER',
            r'\bSERVICE\s+MODE\b': 'SERVICE_MODE',
            r'\bDIAGNOSTIC\s+MODE\b': 'DIAGNOSTIC_MODE',
        }
        
        for pattern, replacement in compound_patterns.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Reduce excessive punctuation that gets high attention scores
        # Replace multiple asterisks with single underscore
        text = re.sub(r'\*+', '_', text)
        
        # Clean up excessive parentheses and commas that fragment attention
        text = re.sub(r'[(),]+', ' ', text)
        
        # Additional punctuation cleaning for better BERT focus
        # Remove excessive colons that don't add semantic value
        text = re.sub(r':(\s*\d{3})\b', r' \1', text)  # Convert "ESC: 000" to "ESC 000"
        
        # Normalize numeric patterns to reduce fragmentation
        # Keep amounts as single tokens
        text = re.sub(r'\$(\d+)\.(\d{2})', r'AMOUNT_\1_\2', text)  # $100.00 -> AMOUNT_100_00
        
        # Simplify reference numbers while preserving meaning
        text = re.sub(r'\b(REF|ESC|VAL):\s*(\d+)\b', r'\1_\2', text)  # REF: 000 -> REF_000
        
        # Clean up excessive whitespace around punctuation
        text = re.sub(r'\s*[=:]\s*', ' ', text)  # Remove = and : with spaces
        
        # 6. Remove excessive whitespace and clean up
        text = ' '.join(text.split())
        
        # Truncate to BERT's max length (512 tokens minus special tokens)
        if hasattr(self, 'bert_tokenizer'):
            tokens = self.bert_tokenizer.tokenize(text)
            if len(tokens) > 510:  # Leave room for [CLS] and [SEP]
                tokens = tokens[:510]
                text = self.bert_tokenizer.convert_tokens_to_string(tokens)
        
        return text
    
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
                # Use our preprocessed text if available, otherwise preprocess it now
                if 'bert_preprocessed_text' in session and session['bert_preprocessed_text']:
                    cleaned_text = session['bert_preprocessed_text']
                else:
                    # Apply the same preprocessing methodology as bertviz_analyzer
                    cleaned_text = self._preprocess_text(session_text)
                
                # Get BERT embeddings using our integrated preprocessing
                if self.bert_analyzer is not None:
                    analysis_result = self.bert_analyzer.analyze_session_text(
                        session_text, 
                        session_id=session.get('session_id', f'session_{len(sequences)}')
                    )
                    
                    if 'error' in analysis_result:
                        logger.warning(f"Failed to analyze session: {analysis_result['error']}")
                        continue
                    
                    # Extract event embeddings from token importance
                    token_rankings = analysis_result['token_importance']['token_rankings']
                else:
                    # Fallback: basic preprocessing without BERT analysis
                    # Use the preprocessed text if available, otherwise raw text
                    cleaned_text = session.get('bert_preprocessed_text', session_text)
                    # Create simple token rankings for basic operation
                    tokens = cleaned_text.split()[:50]  # Limit tokens
                    token_rankings = [{'token': token, 'importance': 0.5} for token in tokens]
                
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
            if self.bert_analyzer is not None:
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
            else:
                # Fallback: use basic tokenizer with random embedding
                inputs = self.bert_tokenizer(
                    token, 
                    return_tensors='pt', 
                    padding=True, 
                    truncation=True, 
                    max_length=8
                )
                # Create a simple random embedding (768 dimensions for BERT)
                embedding = np.random.normal(0, 0.1, (768,)).astype(np.float32)
                
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
            # Apply the same preprocessing methodology as bertviz_analyzer
            cleaned_text = self._preprocess_text(session_text)
            
            # Analyze session with BERT using preprocessed text
            if self.bert_analyzer is not None:
                analysis_result = self.bert_analyzer.analyze_session_text(cleaned_text, session_id)
                
                if 'error' in analysis_result:
                    return {'error': analysis_result['error'], 'session_id': session_id}
                
                # Extract event sequence
                token_rankings = analysis_result['token_importance']['token_rankings']
            else:
                # Fallback: Use integrated preprocessing and basic BERT tokenization
                if hasattr(self, 'bert_tokenizer'):
                    tokens = self.bert_tokenizer.tokenize(cleaned_text)
                    # Create token rankings with uniform importance for fallback
                    token_rankings = []
                    for i, token in enumerate(tokens[:50]):  # Limit tokens
                        token_rankings.append({
                            'token': token,
                            'importance': 0.5,
                            'combined_importance': 0.5,
                            'position': i,
                            'attention_importance': 0.5,
                            'contextual_importance': 0.5
                        })
                else:
                    # Ultimate fallback: basic text processing on cleaned text
                    tokens = cleaned_text.split()[:50]  # Limit tokens
                    token_rankings = []
                    for i, token in enumerate(tokens):
                        token_rankings.append({
                            'token': token,
                            'importance': 0.5,
                            'combined_importance': 0.5,
                            'position': i,
                            'attention_importance': 0.5,
                            'contextual_importance': 0.5
                        })
            
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
