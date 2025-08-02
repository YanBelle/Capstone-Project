"""
BertViz Integration for ABM Anomaly Detection
Visualizes BERT attention patterns to understand token importance and model behavior
"""

import torch
import numpy as np
from transformers import BertTokenizer, BertModel, BertConfig
from bertviz import head_view, model_view, neuron_view
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Docker
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import json
import base64
from io import BytesIO
import logging
from datetime import datetime
import pandas as pd
import re

# Import contextual labeling system
try:
    from ej_contextual_labeler import EJLogLabeler, EventType, TransactionPhase
    EJ_LABELER_AVAILABLE = True
except ImportError:
    EJ_LABELER_AVAILABLE = False
    logging.warning("EJ Contextual Labeler not available - using basic BERT attention")

# Import expert labeling system
try:
    from expert_labeling_system import ExpertLabelingSystem
    EXPERT_LABELER_AVAILABLE = True
except ImportError:
    EXPERT_LABELER_AVAILABLE = False
    logging.warning("Expert Labeling System not available - using basic importance")

logger = logging.getLogger(__name__)

class BertVisualizationAnalyzer:
    """
    Analyzes BERT attention patterns and token importance for ABM anomaly detection
    """
    
    def __init__(self, model_name: str = 'bert-base-uncased', device: str = None):
        """
        Initialize the BERT visualization analyzer with EJ contextual enhancement
        
        Args:
            model_name: BERT model name/path
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize BERT components
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name, output_attentions=True)
        
        # Add custom ATM/EJ domain tokens to prevent splitting
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
            "OPCODE_FI", "OPCODE_IB", "OPCODE_IC", "OPCODE_ID",
            "ATR_RECEIVED_T_0", "ATR_RECEIVED_T_1",
            
            # Common Machine and R status patterns to prevent fragmentation
            "M_00", "M_01", "M_02", "M_03", "M_04", "M_05", "M_10", "M_15", "M_20", "M_99",
            "R_0000", "R_5005", "R_10011", "R_20001", "R_30015", "R_40000", "R_50000"
        ]
        
        # Add tokens to tokenizer vocabulary
        num_added_tokens = self.tokenizer.add_tokens(custom_tokens)
        logger.info(f"Added {num_added_tokens} custom ATM domain tokens to tokenizer vocabulary")
        
        # Resize model embeddings to accommodate new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize EJ contextual labeler for domain-specific enhancement
        if EJ_LABELER_AVAILABLE:
            try:
                self.ej_labeler = EJLogLabeler()
                logger.info("EJ Contextual Labeler initialized successfully - domain-specific analysis enabled")
            except Exception as e:
                self.ej_labeler = None
                logger.error(f"Failed to initialize EJ Contextual Labeler: {e}")
        else:
            self.ej_labeler = None
            logger.warning("EJ Contextual Labeler unavailable - using basic analysis")
        
        if EXPERT_LABELER_AVAILABLE:
            self.expert_labeler = ExpertLabelingSystem()
            logger.info("Expert Labeling System initialized - expert knowledge enhanced")
        else:
            self.expert_labeler = None
            logger.warning("Expert Labeling System unavailable - using basic patterns")
        
        # Store analysis results
        self.attention_cache = {}
        # Cache for performance
        self.token_importance_cache = {}
        self.attention_cache = {}
        
        # Store original session text for EJ labeler
        self.original_session_text = None
        
        logger.info(f"BertViz analyzer initialized with {model_name} on {self.device}")
    
    def analyze_session_text(self, session_text: str, session_id: str = None) -> Dict[str, Any]:
        """
        Comprehensive analysis of session text including attention patterns and token importance
        
        Args:
            session_text: Raw session text to analyze
            session_id: Optional session identifier
            
        Returns:
            Dictionary containing all analysis results
        """
        try:
            # Store original text for EJ contextual labeler (needs timestamps/patterns for feature extraction)
            self.original_session_text = session_text
            
            # Preprocess text for BERT (removes noise patterns that interfere with attention)
            processed_text = self._preprocess_text(session_text)
            
            # Get BERT outputs with attention
            inputs, attention_weights, hidden_states = self._get_bert_outputs(processed_text)
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            # Fix for scalar error - ensure tensors are properly shaped
            # Check attention_weights shape and convert as needed
            try:
                # Stack for consistent processing
                stacked_attention = torch.stack(attention_weights)
                logger.info(f"Initial attention weights shape: {stacked_attention.shape}")
                
                # Handle the case that causes the scalar error
                if len(stacked_attention.shape) == 5 and stacked_attention.shape[1] == 1:
                    # Remove batch dimension if it's 1
                    stacked_attention = stacked_attention.squeeze(1)
                    logger.info(f"Removed batch dimension: {stacked_attention.shape}")
                    # Create a new tuple of attention tensors
                    attention_weights = tuple(stacked_attention[i] for i in range(stacked_attention.shape[0]))
            except Exception as shape_error:
                logger.error(f"Error processing attention shape: {shape_error}")
            
            # Perform various analyses
            analysis_results = {
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'text_length': len(session_text),
                'token_count': len(tokens),
                'processed_text': processed_text,
                'tokens': tokens,
            }
            
            # Add components one by one to isolate any issues
            try:
                analysis_results['attention_analysis'] = self._analyze_attention_patterns(attention_weights, tokens)
            except Exception as e:
                logger.error(f"Error in attention analysis: {e}")
                analysis_results['attention_analysis'] = {'error': str(e)}
            
            try:
                analysis_results['token_importance'] = self._calculate_token_importance(attention_weights, tokens)
            except Exception as e:
                logger.error(f"Error in token importance: {e}")
                analysis_results['token_importance'] = {'error': str(e)}
            
            try:
                analysis_results['layer_analysis'] = self._analyze_layers(attention_weights, hidden_states)
            except Exception as e:
                logger.error(f"Error in layer analysis: {e}")
                analysis_results['layer_analysis'] = {'error': str(e)}
            
            try:
                analysis_results['head_analysis'] = self._analyze_attention_heads(attention_weights, tokens)
            except Exception as e:
                logger.error(f"Error in head analysis: {e}")
                analysis_results['head_analysis'] = {'error': str(e)}
            
            try:
                analysis_results['patterns'] = self._detect_attention_patterns(attention_weights, tokens)
            except Exception as e:
                logger.error(f"Error in pattern detection: {e}")
                analysis_results['patterns'] = {'error': str(e)}
            
            try:
                analysis_results['visualizations'] = self._generate_visualizations(attention_weights, tokens, session_text)
            except Exception as e:
                logger.error(f"Error in visualizations: {e}")
                analysis_results['visualizations'] = {'error': str(e)}
            
            # Cache results
            if session_id:
                self.attention_cache[session_id] = analysis_results
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing session text: {e}")
            return {'error': str(e), 'session_id': session_id}
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess ABM log text for BERT analysis with enhanced pattern cleaning"""
        # Enhanced EJ pattern cleaning with specific fixes for BERT attention optimization
        
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
        
        # ATR pattern was already handled above - don't repeat it here to avoid conflicts
        
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
            # REMOVED: r'\bATR\s+RECEIVED\b': 'ATR_RECEIVED',  # This would break ATR_RECEIVED_T_0!
            r'\bTRANSACTION\s+END\b': 'TRANSACTION_END',
            r'\bTRANSACTION\s+START\b': 'TRANSACTION_START',
            
            # Additional ATM operations
            r'\bCASH\s+DISPENSED\b': 'CASH_DISPENSED',
            r'\bBALANCE\s+INQUIRY\b': 'BALANCE_INQUIRY',
            r'\bRECEIPT\s+PRINTED\b': 'RECEIPT_PRINTED',
            r'\bCARD\s+RETAINED\b': 'CARD_RETAINED',
            r'\bCARD\s+EJECTED\b': 'CARD_EJECTED',
            r'\bCARD\s+READ\b': 'CARD_READ',
            
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
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > 510:  # Leave room for [CLS] and [SEP]
            tokens = tokens[:510]
            text = self.tokenizer.convert_tokens_to_string(tokens)
        
        return text
    
    def _get_bert_outputs(self, text: str) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
        """Get BERT outputs including attention weights and hidden states"""
        # CRITICAL FIX: Tokenize WITHOUT special tokens to prevent ML pipeline contamination
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=512,
            add_special_tokens=False  # Prevent [CLS]/[SEP] injection into ML pipeline
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            attention_weights = outputs.attentions  # Tuple of attention tensors for each layer
            hidden_states = outputs.last_hidden_state
        
        return inputs, attention_weights, hidden_states
    
    def _analyze_attention_patterns(self, attention_weights: Tuple, tokens: List[str]) -> Dict[str, Any]:
        """Analyze attention patterns across layers and heads"""
        try:
            num_layers = len(attention_weights)
            if num_layers == 0:
                return {'error': 'No attention weights provided'}
            
            num_heads = attention_weights[0].shape[1] if len(attention_weights[0].shape) > 1 else 1
            seq_len = len(tokens)
            
            # Average attention across all heads and layers
            # attention_weights is tuple of [batch_size, num_heads, seq_len, seq_len] tensors
            stacked_attention = torch.stack(attention_weights)  # [num_layers, batch_size, num_heads, seq_len, seq_len]
            logger.info(f"Stacked attention shape: {stacked_attention.shape}")
            
            # Handle different tensor shapes dynamically
            if len(stacked_attention.shape) == 5:  # [layers, batch, heads, seq, seq]
                avg_attention = stacked_attention.mean(dim=(0, 1, 2)).cpu().numpy()
            elif len(stacked_attention.shape) == 4:  # [layers, heads, seq, seq] 
                avg_attention = stacked_attention.mean(dim=(0, 1)).cpu().numpy()
            elif len(stacked_attention.shape) == 3:  # [layers, seq, seq]
                avg_attention = stacked_attention.mean(dim=0).cpu().numpy()
            else:
                logger.error(f"Unexpected stacked attention shape: {stacked_attention.shape}")
                return {'error': 'Invalid attention tensor shape'}
            
            logger.info(f"Averaged attention shape: {avg_attention.shape}")
            
            # Ensure attention matrix dimensions match token length
            if avg_attention.shape[0] != seq_len or avg_attention.shape[1] != seq_len:
                logger.warning(f"Attention matrix shape {avg_attention.shape} doesn't match token length {seq_len}")
                # Truncate or pad to match
                min_dim = min(avg_attention.shape[0], avg_attention.shape[1], seq_len)
                avg_attention = avg_attention[:min_dim, :min_dim]
                tokens = tokens[:min_dim]
                seq_len = min_dim
            
            # Find tokens with highest attention
            token_attention_scores = avg_attention.sum(axis=0)  # Sum attention received by each token
            top_k = min(10, len(token_attention_scores))  # Don't try to get more tokens than we have
            top_tokens_idx = np.argsort(token_attention_scores)[-top_k:]  # Top k tokens
            
            # Attention distribution analysis
            attention_entropy = self._calculate_attention_entropy(avg_attention)
            
            # Safe token access
            top_attended_tokens = []
            for idx in top_tokens_idx[::-1]:  # Reverse for descending order
                if 0 <= idx < len(tokens):
                    top_attended_tokens.append({
                        'token': tokens[idx], 
                        'score': float(token_attention_scores[idx]), 
                        'position': int(idx)
                    })
            
            return {
                'num_layers': num_layers,
                'num_heads': num_heads,
                'sequence_length': seq_len,
                'average_attention_matrix': avg_attention.tolist(),
                'token_attention_scores': token_attention_scores.tolist(),
                'top_attended_tokens': top_attended_tokens,
                'attention_entropy': float(attention_entropy),
                'attention_concentration': float(np.max(token_attention_scores) / (np.mean(token_attention_scores) + 1e-8))
            }
        except Exception as e:
            logger.error(f"Error analyzing attention patterns: {e}")
            return {'error': str(e)}
    
    def _calculate_token_importance(self, attention_weights: Tuple, tokens: List[str]) -> Dict[str, Any]:
        """Calculate importance scores for each token with EJ contextual enhancement"""
        # Method 1: Attention-based importance
        attention_importance = self._attention_based_importance(attention_weights)
        
        # Method 2: Gradient-based importance (approximated)
        gradient_importance = self._gradient_based_importance(attention_weights)
        
        # Method 3: Layer-wise importance
        layer_importance = self._layer_wise_importance(attention_weights)
        
        # Method 4: Contextual importance (EJ DOMAIN KNOWLEDGE!)
        contextual_importance = self._contextual_importance(tokens)
        
        # Method 5: Expert knowledge importance
        expert_importance = self._expert_knowledge_importance(tokens)
        
        # Enhanced combination with contextual weights - EJ domain knowledge gets 35% weight!
        combined_importance = (
            0.25 * attention_importance + 
            0.20 * gradient_importance + 
            0.20 * layer_importance +
            0.25 * contextual_importance +  # EJ contextual labeler
            0.10 * expert_importance        # Expert knowledge patterns
        )
        
        # CRITICAL FIX: Suppress special tokens [CLS] and [SEP] that BERT automatically adds
        for idx, token in enumerate(tokens):
            if token in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']:
                combined_importance[idx] *= 0.01  # Reduce to 1% of original importance
                attention_importance[idx] *= 0.01
                gradient_importance[idx] *= 0.01
                layer_importance[idx] *= 0.01
                # Don't suppress contextual and expert importance as they should be zero anyway
        
        # Create token importance rankings
        token_rankings = [
            {
                'token': token,
                'position': idx,
                'attention_importance': float(attention_importance[idx]),
                'gradient_importance': float(gradient_importance[idx]),
                'layer_importance': float(layer_importance[idx]),
                'contextual_importance': float(contextual_importance[idx]),
                'expert_importance': float(expert_importance[idx]),
                'combined_importance': float(combined_importance[idx]),
                'is_special_token': token in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']
            }
            for idx, token in enumerate(tokens)
        ]
        
        # Sort by combined importance (special tokens should now be at the bottom)
        token_rankings.sort(key=lambda x: x['combined_importance'], reverse=True)
        
        # Filter out special tokens from top rankings for cleaner display
        content_token_rankings = [r for r in token_rankings if not r['is_special_token']]
        
        return {
            'token_rankings': content_token_rankings,  # Only content tokens for main display
            'all_token_rankings': token_rankings,      # All tokens including special ones
            'importance_statistics': {
                'mean_importance': float(np.mean(combined_importance)),
                'std_importance': float(np.std(combined_importance)),
                'max_importance': float(np.max(combined_importance)),
                'min_importance': float(np.min(combined_importance))
            },
            'contextual_enhancement': {
                'ej_labeler_used': self.ej_labeler is not None,
                'expert_labeler_used': self.expert_labeler is not None,
                'enhancement_impact': float(np.mean(contextual_importance + expert_importance)),
                'special_tokens_suppressed': True
            }
        }
    
    def _attention_based_importance(self, attention_weights: Tuple) -> np.ndarray:
        """Calculate token importance based on attention patterns"""
        try:
            # Stack attention weights
            attention_stack = torch.stack(attention_weights)
            logger.info(f"Attention stack shape: {attention_stack.shape}")
            
            # Handle different possible tensor shapes:
            # Case 1: [num_layers, batch_size, num_heads, seq_len, seq_len]
            # Case 2: [num_layers, num_heads, seq_len, seq_len] (no batch dim)  
            # Case 3: [num_layers, seq_len, seq_len] (already averaged across heads)
            
            if len(attention_stack.shape) == 5:  # [layers, batch, heads, seq, seq]
                # Average across layers (0), batch (1), and heads (2)
                avg_attention = attention_stack.mean(dim=(0, 1, 2))
            elif len(attention_stack.shape) == 4:  # [layers, heads, seq, seq] - no batch dim
                # Average across layers (0) and heads (1)
                avg_attention = attention_stack.mean(dim=(0, 1))
            elif len(attention_stack.shape) == 3:  # [layers, seq, seq] - already averaged
                # Just average across layers (0)
                avg_attention = attention_stack.mean(dim=0)
            else:
                logger.error(f"Unexpected attention shape: {attention_stack.shape}")
                # Fallback: just use zeros
                seq_len = attention_weights[0].shape[-1]  # Get sequence length
                return np.zeros(seq_len)
            
            logger.info(f"Averaged attention shape: {avg_attention.shape}")
            
            # Sum attention received by each token (sum along the 'from' dimension)
            # This gives us the total attention each token receives
            if len(avg_attention.shape) == 2:
                token_importance = avg_attention.sum(dim=0).cpu().numpy()
            else:
                logger.error(f"Unexpected final attention shape: {avg_attention.shape}")
                # Fallback: just use zeros
                seq_len = attention_weights[0].shape[-1]  # Get sequence length
                token_importance = np.zeros(seq_len)
            
            logger.info(f"Token importance shape: {token_importance.shape}")
            return token_importance
            
        except Exception as e:
            logger.error(f"Error in _attention_based_importance: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Fallback: return zeros
            try:
                seq_len = attention_weights[0].shape[-1] if attention_weights else 10
            except:
                seq_len = 10
            return np.zeros(seq_len)
    
    def _gradient_based_importance(self, attention_weights: Tuple) -> np.ndarray:
        """Approximate gradient-based importance using attention variance"""
        # Use attention variance as a proxy for gradient-based importance
        attention_stack = torch.stack(attention_weights)
        attention_var = attention_stack.var(dim=(0, 1)).squeeze()
        return attention_var.sum(dim=0).cpu().numpy()
    
    def _layer_wise_importance(self, attention_weights: Tuple) -> np.ndarray:
        """Calculate importance based on layer-wise attention evolution"""
        layer_attentions = []
        for layer_attention in attention_weights:
            layer_avg = layer_attention.mean(dim=1).squeeze()  # Average across heads
            layer_attentions.append(layer_avg.sum(dim=0).cpu().numpy())  # Sum across attention-to
        
        # Calculate importance as the change in attention across layers
        layer_changes = np.diff(layer_attentions, axis=0)
        return np.abs(layer_changes).sum(axis=0)
    
    def _analyze_layers(self, attention_weights: Tuple, hidden_states: torch.Tensor) -> Dict[str, Any]:
        """Analyze attention patterns across different BERT layers"""
        layer_analysis = []
        
        try:
            for layer_idx, layer_attention in enumerate(attention_weights):
                # Safely process layer attention
                try:
                    layer_avg = layer_attention.mean(dim=1).squeeze().cpu().numpy()  # Average across heads
                    
                    # Ensure we have a valid 2D array
                    if len(layer_avg.shape) != 2:
                        logger.warning(f"Layer {layer_idx} attention has unexpected shape: {layer_avg.shape}")
                        continue
                    
                    # Calculate layer-specific metrics
                    layer_entropy = self._calculate_attention_entropy(layer_avg)
                    layer_sparsity = np.sum(layer_avg < 0.01) / layer_avg.size  # Percentage of near-zero attention
                    layer_max_attention = np.max(layer_avg)
                    
                    layer_analysis.append({
                        'layer': layer_idx,
                        'entropy': float(layer_entropy),
                        'sparsity': float(layer_sparsity),
                        'max_attention': float(layer_max_attention),
                        'attention_matrix': layer_avg.tolist()
                    })
                except Exception as layer_error:
                    logger.error(f"Error processing layer {layer_idx}: {layer_error}")
                    continue
        except Exception as e:
            logger.error(f"Error analyzing layers: {e}")
            return {'layers': [], 'error': str(e)}
        
        try:
            layer_progression = self._analyze_layer_progression(attention_weights)
        except Exception as e:
            logger.error(f"Error analyzing layer progression: {e}")
            layer_progression = {'error': str(e)}
        
        return {
            'layers': layer_analysis,
            'layer_progression': layer_progression
        }
    
    def _analyze_attention_heads(self, attention_weights: Tuple, tokens: List[str]) -> Dict[str, Any]:
        """Analyze individual attention heads to understand their specialization"""
        head_analysis = []
        
        try:
            for layer_idx, layer_attention in enumerate(attention_weights):
                # Ensure we have the right shape and convert to numpy
                layer_attention = layer_attention.squeeze().cpu().numpy()
                
                # Handle different possible shapes
                if len(layer_attention.shape) == 3:  # [num_heads, seq_len, seq_len]
                    num_heads = layer_attention.shape[0]
                elif len(layer_attention.shape) == 2:  # [seq_len, seq_len] - single head
                    layer_attention = layer_attention[np.newaxis, :, :]  # Add head dimension
                    num_heads = 1
                else:
                    continue  # Skip malformed attention tensors
                
                for head_idx in range(num_heads):
                    head_attention = layer_attention[head_idx]
                    
                    # Ensure attention matrix is square and matches token length
                    if head_attention.shape[0] != head_attention.shape[1]:
                        continue  # Skip malformed attention matrices
                    
                    # Analyze head behavior
                    head_type = self._classify_attention_head(head_attention, tokens)
                    head_entropy = self._calculate_attention_entropy(head_attention)
                    
                    head_analysis.append({
                        'layer': layer_idx,
                        'head': head_idx,
                        'type': head_type,
                        'entropy': float(head_entropy),
                        'max_attention': float(np.max(head_attention)),
                        'primary_focus': self._get_head_primary_focus(head_attention, tokens)
                    })
        except Exception as e:
            logger.error(f"Error analyzing attention heads: {e}")
            return {'heads': [], 'error': str(e)}
        
        return {'heads': head_analysis}
    
    def _classify_attention_head(self, head_attention: np.ndarray, tokens: List[str]) -> str:
        """Classify attention head type based on its attention pattern"""
        try:
            # Ensure we have a square matrix
            if head_attention.shape[0] != head_attention.shape[1] or head_attention.size == 0:
                return "unknown"
            
            # Simple heuristics to classify attention heads
            diagonal_strength = np.trace(head_attention) / (np.sum(head_attention) + 1e-8)  # Add small epsilon
            
            if diagonal_strength > 0.3:
                return "self-attention"
            elif head_attention.shape[0] > 0 and np.max(head_attention[0, :]) > 0.5:  # High attention to [CLS]
                return "classification-focused"
            elif head_attention.shape[1] > 1 and np.max(head_attention[:, -1]) > 0.5:  # High attention to [SEP]
                return "delimiter-focused"
            else:
                return "content-focused"
        except Exception as e:
            return f"error: {str(e)}"
    
    def _get_head_primary_focus(self, head_attention: np.ndarray, tokens: List[str]) -> Dict[str, Any]:
        """Get the primary focus of an attention head"""
        try:
            # Find the token that receives the most attention overall
            total_attention_received = head_attention.sum(axis=0)
            max_idx = np.argmax(total_attention_received)
            
            # Ensure the index is within bounds
            if max_idx >= len(tokens) or max_idx < 0:
                return {
                    'token': '<UNK>',
                    'position': -1,
                    'attention_score': 0.0
                }
            
            return {
                'token': tokens[max_idx],
                'position': int(max_idx),
                'attention_score': float(total_attention_received[max_idx])
            }
        except Exception as e:
            return {
                'token': '<ERROR>',
                'position': -1,
                'attention_score': 0.0,
                'error': str(e)
            }
    
    def _detect_attention_patterns(self, attention_weights: Tuple, tokens: List[str]) -> Dict[str, Any]:
        """Detect specific attention patterns relevant to ABM anomaly detection"""
        patterns = {}
        
        # Average attention across all layers and heads
        avg_attention = torch.stack(attention_weights).mean(dim=(0, 1)).squeeze().cpu().numpy()
        
        # Pattern 1: Sequential attention (following word order)
        sequential_score = self._calculate_sequential_attention(avg_attention)
        patterns['sequential_attention'] = {
            'score': float(sequential_score),
            'description': 'How much the model follows sequential word order'
        }
        
        # Pattern 2: Error keyword attention
        error_keywords = ['error', 'fail', 'timeout', 'reject', 'abort', 'exception']
        error_attention = self._calculate_keyword_attention(avg_attention, tokens, error_keywords)
        patterns['error_attention'] = {
            'score': float(error_attention),
            'description': 'Attention to error-related keywords',
            'keywords_found': [token for token in tokens if any(keyword in token.lower() for keyword in error_keywords)]
        }
        
        # Pattern 3: Temporal attention (time-related tokens)
        temporal_keywords = ['time', 'date', 'hour', 'minute', 'second', 'am', 'pm']
        temporal_attention = self._calculate_keyword_attention(avg_attention, tokens, temporal_keywords)
        patterns['temporal_attention'] = {
            'score': float(temporal_attention),
            'description': 'Attention to temporal information'
        }
        
        # Pattern 4: Transaction attention
        transaction_keywords = ['withdraw', 'deposit', 'balance', 'card', 'account', 'pin']
        transaction_attention = self._calculate_keyword_attention(avg_attention, tokens, transaction_keywords)
        patterns['transaction_attention'] = {
            'score': float(transaction_attention),
            'description': 'Attention to transaction-related terms'
        }
        
        return patterns
    
    def _calculate_sequential_attention(self, attention_matrix: np.ndarray) -> float:
        """Calculate how much attention follows sequential order"""
        seq_len = attention_matrix.shape[0]
        sequential_weight = 0.0
        
        for i in range(seq_len):
            for j in range(max(0, i-2), min(seq_len, i+3)):  # Window of ±2 positions
                weight = 1.0 / (abs(i - j) + 1)  # Higher weight for closer positions
                sequential_weight += attention_matrix[i, j] * weight
        
        return sequential_weight / (seq_len * seq_len)
    
    def _calculate_keyword_attention(self, attention_matrix: np.ndarray, tokens: List[str], keywords: List[str]) -> float:
        """Calculate total attention to specific keywords"""
        keyword_attention = 0.0
        keyword_count = 0
        
        for idx, token in enumerate(tokens):
            if any(keyword in token.lower() for keyword in keywords):
                keyword_attention += attention_matrix[:, idx].sum()  # Sum attention to this token
                keyword_count += 1
        
        return keyword_attention / max(keyword_count, 1)  # Average attention per keyword
    
    def _calculate_attention_entropy(self, attention_matrix: np.ndarray) -> float:
        """Calculate entropy of attention distribution"""
        # Flatten and normalize attention weights
        attention_flat = attention_matrix.flatten()
        attention_prob = attention_flat / attention_flat.sum()
        
        # Calculate entropy
        entropy = -np.sum(attention_prob * np.log(attention_prob + 1e-10))
        return entropy
    
    def _analyze_layer_progression(self, attention_weights: Tuple) -> Dict[str, Any]:
        """Analyze how attention patterns evolve across layers"""
        layer_entropies = []
        layer_concentrations = []
        
        for layer_attention in attention_weights:
            layer_avg = layer_attention.mean(dim=1).squeeze().cpu().numpy()
            entropy = self._calculate_attention_entropy(layer_avg)
            concentration = np.max(layer_avg) / np.mean(layer_avg)
            
            layer_entropies.append(float(entropy))
            layer_concentrations.append(float(concentration))
        
        return {
            'entropy_progression': layer_entropies,
            'concentration_progression': layer_concentrations,
            'entropy_trend': 'increasing' if layer_entropies[-1] > layer_entropies[0] else 'decreasing',
            'concentration_trend': 'increasing' if layer_concentrations[-1] > layer_concentrations[0] else 'decreasing'
        }
    
    def _generate_visualizations(self, attention_weights: Tuple, tokens: List[str], original_text: str) -> Dict[str, str]:
        """Generate BertViz visualizations and encode as base64"""
        visualizations = {}
        
        try:
            logger.info(f"Starting visualization generation for text length: {len(original_text)}")
            logger.info(f"Attention weights tuple length: {len(attention_weights)}")
            logger.info(f"Number of tokens: {len(tokens)}")
            
            # Prepare inputs for BertViz
            inputs = self.tokenizer(original_text, return_tensors='pt', truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate head view (attention between tokens)
            try:
                logger.info("Attempting to generate head view...")
                # BertViz may have compatibility issues with newer transformers versions
                # Skip BertViz for now and focus on custom visualizations
                logger.warning("Skipping BertViz head_view due to compatibility issues")
                visualizations['head_view'] = ""
            except Exception as e:
                logger.warning(f"Could not generate head view: {e}")
                visualizations['head_view'] = ""
            
            # Generate model view (aggregated attention)
            try:
                logger.info("Attempting to generate model view...")
                # BertViz may have compatibility issues with newer transformers versions
                # Skip BertViz for now and focus on custom visualizations
                logger.warning("Skipping BertViz model_view due to compatibility issues")
                visualizations['model_view'] = ""
            except Exception as e:
                logger.warning(f"Could not generate model view: {e}")
                visualizations['model_view'] = ""
            
            # Calculate averaged attention matrix first
            logger.info("Computing averaged attention matrix for visualizations...")
            try:
                stacked_attention = torch.stack(attention_weights)
                logger.info(f"Stacked attention shape: {stacked_attention.shape}")
                
                # Handle different tensor shapes dynamically
                if len(stacked_attention.shape) == 5:  # [layers, batch, heads, seq, seq]
                    avg_attention = stacked_attention.mean(dim=(0, 1, 2)).cpu().numpy()
                elif len(stacked_attention.shape) == 4:  # [layers, heads, seq, seq] 
                    avg_attention = stacked_attention.mean(dim=(0, 1)).cpu().numpy()
                elif len(stacked_attention.shape) == 3:  # [layers, seq, seq]
                    avg_attention = stacked_attention.mean(dim=0).cpu().numpy()
                else:
                    logger.error(f"Unexpected stacked attention shape: {stacked_attention.shape}")
                    return {'error': 'Invalid attention tensor shape for visualization'}
                
                logger.info(f"Averaged attention shape: {avg_attention.shape}")
                
                # Ensure we have a valid 2D matrix
                if len(avg_attention.shape) != 2:
                    logger.error(f"Averaged attention is not 2D: {avg_attention.shape}")
                    return {'error': 'Could not create 2D attention matrix'}
                    
            except Exception as avg_error:
                logger.error(f"Error computing averaged attention: {avg_error}")
                return {'error': f'Attention averaging failed: {str(avg_error)}'}
            
            # Generate custom attention heatmap
            logger.info("Generating custom attention heatmap...")
            visualizations['attention_heatmap'] = self._generate_attention_heatmap_from_matrix(avg_attention, tokens)
            
            # Generate token importance visualization  
            logger.info("Generating token importance visualization...")
            visualizations['token_importance'] = self._generate_token_importance_from_matrix(avg_attention, tokens)
            
            # Log final results
            for viz_name, viz_data in visualizations.items():
                logger.info(f"{viz_name}: {'Generated' if viz_data else 'Empty'} (length: {len(viz_data)})")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            visualizations['error'] = str(e)
        
        return visualizations
    
    def _generate_attention_heatmap_from_matrix(self, avg_attention: np.ndarray, tokens: List[str]) -> str:
        """Generate custom attention heatmap from pre-averaged 2D attention matrix"""
        try:
            logger.info(f"Generating attention heatmap from matrix shape: {avg_attention.shape}, tokens: {len(tokens)}")
            
            # Validate input
            if len(avg_attention.shape) != 2:
                logger.error(f"Expected 2D attention matrix, got shape: {avg_attention.shape}")
                return ""
            
            if avg_attention.shape[0] == 0 or avg_attention.shape[1] == 0:
                logger.error(f"Empty attention matrix: {avg_attention.shape}")
                return ""
            
            # Check for invalid values (NaN, infinity)
            if not np.isfinite(avg_attention).all():
                logger.warning("Non-finite values in attention matrix, replacing with zeros")
                avg_attention = np.nan_to_num(avg_attention, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Ensure dimensions match token count
            seq_len = avg_attention.shape[0]
            if seq_len != len(tokens):
                min_len = min(seq_len, len(tokens))
                avg_attention = avg_attention[:min_len, :min_len]
                tokens = tokens[:min_len]
                logger.info(f"Adjusted dimensions to {min_len}x{min_len}")
            
            # CRITICAL FIX: Filter out special tokens from visualization like [CLS], [SEP], etc.
            special_tokens = ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[', ']', '=']
            
            # Create filtered indices (exclude special tokens)
            filtered_indices = []
            for i, token in enumerate(tokens):
                if token not in special_tokens and not any(st in token for st in special_tokens):
                    filtered_indices.append(i)
            
            logger.info(f"Filtered out special tokens: {len(tokens) - len(filtered_indices)} removed")
            
            # Limit tokens for readability (max 25 tokens - smaller for readability) 
            max_tokens = 25
            if len(filtered_indices) > max_tokens:
                # Take most important content tokens based on attention received
                token_importance = avg_attention.sum(axis=0)
                
                # Get importance scores for filtered indices only
                filtered_importance = [(i, token_importance[i]) for i in filtered_indices]
                # Sort by importance and take top max_tokens
                filtered_importance.sort(key=lambda x: x[1], reverse=True)
                top_indices = [x[0] for x in filtered_importance[:max_tokens]]
                top_indices.sort()  # Keep in original order for readability
                
                logger.info(f"Selected {max_tokens} most important content tokens for visualization")
            else:
                top_indices = filtered_indices
                logger.info(f"Using all {len(top_indices)} content tokens for visualization")
            
            # Filter attention matrix and tokens
            avg_attention = avg_attention[top_indices, :][:, top_indices]
            tokens = [tokens[i] for i in top_indices]
            
            # Create heatmap with comprehensive error handling
            try:
                plt.figure(figsize=(12, 10))
                
                logger.info(f"Creating heatmap with attention shape: {avg_attention.shape}, tokens: {len(tokens)}")
                
                # Use matplotlib directly for more control
                im = plt.imshow(avg_attention, cmap='Blues')
                plt.colorbar(im, label='Attention Weight')
                
                # Add token labels
                plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
                plt.yticks(range(len(tokens)), tokens)
                
                # Add grid lines for clarity
                plt.grid(False)
                
                # Add title and labels
                plt.title('BERT Attention Heatmap')
                plt.xlabel('Tokens (To)')
                plt.ylabel('Tokens (From)')
                
                plt.tight_layout()
                
            except Exception as heatmap_error:
                logger.error(f"Error creating heatmap: {heatmap_error}")
                # Try with simpler parameters - absolute minimum
                try:
                    logger.info("Attempting simplified heatmap...")
                    plt.figure(figsize=(8, 6))
                    plt.imshow(avg_attention, cmap='Blues')
                    plt.colorbar(label='Attention')
                    plt.title('BERT Attention Heatmap')
                    plt.tight_layout()
                except Exception as simple_error:
                    logger.error(f"Even simplified heatmap failed: {simple_error}")
                    plt.close()
                    return ""
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            logger.info(f"Generated attention heatmap, base64 length: {len(image_base64)}")
            return image_base64
            
        except Exception as e:
            logger.error(f"Error generating attention heatmap from matrix: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return ""
    
    def _generate_token_importance_from_matrix(self, avg_attention: np.ndarray, tokens: List[str]) -> str:
        """Generate token importance bar plot from pre-averaged 2D attention matrix"""
        try:
            logger.info(f"Generating token importance from matrix shape: {avg_attention.shape}, tokens: {len(tokens)}")
            
            # Validate input
            if len(avg_attention.shape) != 2:
                logger.error(f"Expected 2D attention matrix, got shape: {avg_attention.shape}")
                return ""
            
            # Calculate token importance (sum of attention received by each token)
            importance_scores = avg_attention.sum(axis=0)  # Sum along 'from' dimension
            logger.info(f"Importance scores shape: {importance_scores.shape}")
            
            # Ensure we have valid numeric data
            if importance_scores.size == 0:
                logger.error("Empty importance scores array")
                return ""
            
            # Handle length mismatch between tokens and scores
            if len(importance_scores) != len(tokens):
                min_len = min(len(importance_scores), len(tokens))
                logger.warning(f"Length mismatch: tokens={len(tokens)}, scores={len(importance_scores)}, using min={min_len}")
                importance_scores = importance_scores[:min_len]
                tokens = tokens[:min_len]
            
            # CRITICAL FIX: Filter out special tokens from token importance visualization
            special_tokens = ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[', ']', '=']
            
            # Create filtered lists excluding special tokens
            filtered_tokens = []
            filtered_scores = []
            for i, token in enumerate(tokens):
                if token not in special_tokens and not any(st in token for st in special_tokens):
                    filtered_tokens.append(token)
                    filtered_scores.append(importance_scores[i])
            
            logger.info(f"Filtered token importance: {len(filtered_tokens)} tokens after removing {len(tokens) - len(filtered_tokens)} special tokens")
            
            # Final validation
            if len(filtered_tokens) == 0 or len(filtered_scores) == 0:
                logger.error("No content tokens remaining after filtering special tokens")
                return ""
            
            # Convert to basic Python types for pandas compatibility
            tokens_list = [str(token) for token in filtered_tokens]  # Ensure strings
            importance_list = [float(score) for score in filtered_scores]  # Ensure floats
            
            # Validate all importance values are finite
            if not all(np.isfinite(importance_list)):
                logger.warning("Non-finite values in importance scores, replacing with zeros")
                importance_list = [0.0 if not np.isfinite(x) else x for x in importance_list]
            
            logger.info(f"Final data: {len(tokens_list)} tokens, {len(importance_list)} scores")
            
            # Create DataFrame for plotting
            df = pd.DataFrame({
                'Token': tokens_list,
                'Importance': importance_list
            })
            
            # Sort by importance
            df = df.sort_values('Importance', ascending=False)  # Changed to descending order
            
            # Limit to top 15 for readability (was 20)
            if len(df) > 15:
                df = df.head(15)  # Get top items instead of tail since we sort descending now
            
            # Create bar plot with horizontal bars for better readability
            plt.figure(figsize=(10, max(6, len(df) * 0.4)))  # Increased vertical space per item
            
            try:
                # Use safer matplotlib approach instead of seaborn
                plt.barh(range(len(df)), df['Importance'], color='steelblue')
                plt.yticks(range(len(df)), df['Token'])
                plt.xlabel('Attention Score')
                plt.ylabel('Tokens')
                plt.title('Token Importance (Based on Attention)')
                plt.grid(axis='x', linestyle='--', alpha=0.7)
                plt.tight_layout()
                
            except Exception as plot_error:
                logger.error(f"Error creating importance plot: {plot_error}")
                # Try even simpler version without pandas
                try:
                    # Sort tokens and scores together
                    items = list(zip(tokens_list, importance_list))
                    items.sort(key=lambda x: x[1], reverse=True)
                    top_items = items[:10]  # Limit to top 10
                    
                    # Plot directly with matplotlib
                    labels = [item[0] for item in top_items]
                    values = [item[1] for item in top_items]
                    
                    plt.figure(figsize=(8, 6))
                    plt.barh(range(len(labels)), values)
                    plt.yticks(range(len(labels)), labels)
                    plt.xlabel('Importance')
                    plt.title('Token Importance')
                    plt.tight_layout()
                except Exception as simple_error:
                    logger.error(f"Even simplified plot failed: {simple_error}")
                    plt.close()
                    return ""
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            logger.info(f"Generated token importance plot, base64 length: {len(image_base64)}")
            return image_base64
            
        except Exception as e:
            logger.error(f"Error generating token importance from matrix: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return ""
    def _encode_html_to_base64(self, html_content: str) -> str:
        """Encode HTML content to base64"""
        try:
            html_bytes = html_content.encode('utf-8')
            return base64.b64encode(html_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding HTML to base64: {e}")
            return ""
    
    def compare_sessions(self, session1_id: str, session2_id: str) -> Dict[str, Any]:
        """Compare attention patterns between two sessions"""
        if session1_id not in self.attention_cache or session2_id not in self.attention_cache:
            return {'error': 'One or both sessions not found in cache'}
        
        analysis1 = self.attention_cache[session1_id]
        analysis2 = self.attention_cache[session2_id]
        
        # Compare key metrics
        comparison = {
            'session1_id': session1_id,
            'session2_id': session2_id,
            'attention_entropy_diff': analysis1['attention_analysis']['attention_entropy'] - analysis2['attention_analysis']['attention_entropy'],
            'attention_concentration_diff': analysis1['attention_analysis']['attention_concentration'] - analysis2['attention_analysis']['attention_concentration'],
            'token_count_diff': analysis1['token_count'] - analysis2['token_count'],
            'pattern_differences': {}
        }
        
        # Compare attention patterns
        for pattern_name in analysis1['patterns']:
            if pattern_name in analysis2['patterns']:
                comparison['pattern_differences'][pattern_name] = {
                    'session1_score': analysis1['patterns'][pattern_name]['score'],
                    'session2_score': analysis2['patterns'][pattern_name]['score'],
                    'difference': analysis1['patterns'][pattern_name]['score'] - analysis2['patterns'][pattern_name]['score']
                }
        
        return comparison
    
    def get_anomaly_attention_insights(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights about anomaly detection based on attention analysis"""
        insights = {
            'attention_quality': 'good',  # good, medium, poor
            'key_indicators': [],
            'recommendations': [],
            'confidence_factors': []
        }
        
        # Analyze attention quality
        entropy = analysis_results['attention_analysis']['attention_entropy']
        concentration = analysis_results['attention_analysis']['attention_concentration']
        
        if entropy < 2.0:  # Low entropy suggests focused attention
            insights['key_indicators'].append('Model shows focused attention (low entropy)')
            insights['confidence_factors'].append('High confidence due to focused attention')
        else:
            insights['key_indicators'].append('Model shows distributed attention (high entropy)')
            insights['attention_quality'] = 'medium'
        
        # Check error attention
        error_attention = analysis_results['patterns']['error_attention']['score']
        if error_attention > 0.1:
            insights['key_indicators'].append(f'High attention to error keywords (score: {error_attention:.3f})')
            insights['confidence_factors'].append('Model focusing on error-related terms')
        
        # Check temporal attention
        temporal_attention = analysis_results['patterns']['temporal_attention']['score']
        if temporal_attention > 0.05:
            insights['key_indicators'].append('Model paying attention to temporal information')
        
        # Generate recommendations
        if error_attention < 0.05:
            insights['recommendations'].append('Consider fine-tuning to increase attention to error keywords')
        
        if entropy > 3.0:
            insights['recommendations'].append('Attention appears too distributed - consider input preprocessing')
            insights['attention_quality'] = 'poor'
        
        if concentration < 2.0:
            insights['recommendations'].append('Low attention concentration - model may benefit from focused training')
        
        return insights
    
    def _contextual_importance(self, tokens: List[str]) -> np.ndarray:
        """Calculate importance based on EJ contextual labels - THIS IS THE KEY METHOD!"""
        logger.info(f"_contextual_importance called with tokens: {tokens}")
        logger.info(f"EJ labeler available: {self.ej_labeler is not None}")
        
        if not self.ej_labeler:
            # Fallback: basic keyword-based importance
            logger.warning("Using keyword-based importance fallback - EJ labeler not available")
            return self._keyword_based_importance(tokens)
        
        # CRITICAL: Use original text for EJ labeler, not the cleaned tokens
        # The EJ labeler needs original timestamps and patterns for proper feature extraction
        text = ' '.join(tokens)
        
        try:
            # Get contextual labels from EJ labeler using ORIGINAL session text
            # This ensures the EJ labeler can extract event times and other features properly
            logger.info(f"Calling EJ labeler with ORIGINAL text for feature extraction")
            labels = self.ej_labeler.label_log(self.original_session_text)  # Use original text!
            logger.info(f"EJ labeler returned {len(labels)} labels: {labels}")
            
            # Create importance array
            importance = np.zeros(len(tokens))
            logger.info(f"Initialized importance array with {len(tokens)} tokens")
            
            # Boost importance for tokens associated with important events
            for i, token in enumerate(tokens):
                for label in labels:
                    # HIGH IMPORTANCE for anomaly-related events - DEVICE ERROR, REJECTS, etc.
                    if label['event_type'] in [EventType.ERROR, EventType.WARNING, 
                                             EventType.DEVICE_RECOVERY, EventType.SUPERVISOR_ENTRY]:
                        if any(keyword in token.lower() for keyword in 
                              ['error', 'device', 'supervisor', 'malfunction', 'timeout', 'fail', 'reject']):
                            importance[i] += 2.0
                            logger.info(f"HIGH IMPORTANCE boost for token '{token}': {importance[i]}")
                    
                    # Medium importance for transaction events
                    elif label['event_type'] in [EventType.CASH_DISPENSE, EventType.PIN_ENTRY,
                                                EventType.CARD_INSERT, EventType.CARD_REMOVE]:
                        if any(keyword in token.lower() for keyword in 
                              ['card', 'pin', 'cash', 'notes', 'dispense', 'taken', 'inserted']):
                            importance[i] += 1.0
                            logger.info(f"MEDIUM IMPORTANCE boost for token '{token}': {importance[i]}")
                    
                    # Transaction phase importance
                    if hasattr(label, 'transaction_phase'):
                        if label['transaction_phase'] in [TransactionPhase.ERROR_HANDLING, 
                                                        TransactionPhase.EXCEPTION]:
                            if any(keyword in token.lower() for keyword in 
                                  ['reject', 'timeout', 'cancel', 'abort']):
                                importance[i] += 1.5
                                logger.info(f"PHASE IMPORTANCE boost for token '{token}': {importance[i]}")
            
            # Normalize
            if importance.max() > 0:
                importance = importance / importance.max()
                logger.info(f"Normalized importance scores: {dict(zip(tokens, importance))}")
            else:
                logger.warning("No importance scores assigned by EJ contextual labeler")
            
            return importance
            
        except Exception as e:
            logger.warning(f"Contextual labeling failed: {e}, falling back to keyword-based")
            return self._keyword_based_importance(tokens)
    
    def _expert_knowledge_importance(self, tokens: List[str]) -> np.ndarray:
        """Calculate importance based on expert knowledge patterns"""
        if not self.expert_labeler:
            return np.zeros(len(tokens))
        
        text = ' '.join(tokens)
        importance = np.zeros(len(tokens))
        
        try:
            # Get expert labels for the text
            expert_labels = self.expert_labeler.expert_labels
            
            # Check for patterns in expert knowledge
            for label_type, label_info in expert_labels.items():
                patterns = label_info.get('patterns', [])
                confidence = label_info.get('confidence', 0.5)
                action_required = label_info.get('action_required', False)
                
                # Boost importance for tokens matching expert patterns
                for pattern in patterns:
                    if isinstance(pattern, list):
                        for pattern_term in pattern:
                            for i, token in enumerate(tokens):
                                if pattern_term.lower() in token.lower():
                                    # Higher importance for action-required patterns
                                    boost = confidence * (2.0 if action_required else 1.0)
                                    importance[i] += boost
            
            # Special boost for known anomaly indicators
            anomaly_keywords = [
                'device', 'error', 'malfunction', 'timeout', 'fail', 'reject',
                'supervisor', 'intervention', 'manual', 'override', 'exception'
            ]
            
            for i, token in enumerate(tokens):
                if any(keyword in token.lower() for keyword in anomaly_keywords):
                    importance[i] += 1.0
            
            # Normalize
            if importance.max() > 0:
                importance = importance / importance.max()
            
            return importance
            
        except Exception as e:
            logger.warning(f"Expert knowledge importance calculation failed: {e}")
            return np.zeros(len(tokens))
    
    def _keyword_based_importance(self, tokens: List[str]) -> np.ndarray:
        """Fallback keyword-based importance when contextual labeler unavailable"""
        importance = np.zeros(len(tokens))
        
        # Define keyword categories with different importance weights
        high_importance_keywords = [
            'device', 'error', 'reject', 'fail', 'timeout', 'malfunction',
            'supervisor', 'intervention', 'manual', 'exception', 'abort'
        ]
        
        medium_importance_keywords = [
            'card', 'pin', 'cash', 'dispense', 'notes', 'taken', 'inserted',
            'transaction', 'journal', 'voucher'
        ]
        
        # Apply importance weights
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            if any(keyword in token_lower for keyword in high_importance_keywords):
                importance[i] = 1.0
            elif any(keyword in token_lower for keyword in medium_importance_keywords):
                importance[i] = 0.5
        
        return importance
