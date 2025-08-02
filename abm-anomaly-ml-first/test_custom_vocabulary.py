#!/usr/bin/env python3
"""
Test script to validate custom vocabulary additions to BERT tokenizer
"""

import sys
import os
sys.path.append('services/api')

from transformers import BertTokenizer, BertModel
import torch

def test_custom_vocabulary():
    """Test that custom tokens are properly added to BERT's vocabulary"""
    
    print("=== BERT Custom Vocabulary Test ===")
    print()
    
    # Test sample EJ text with the patterns we want to handle
    sample_text = """
    TRANSACTION_START CARD_INSERTED ATR_RECEIVED T 0 OPCODE FI CardNumber 
    PIN_ENTERED OPCODE IB CardNumber M-02 R-10011 DEVICE_ERROR 
    ESC 000 VAL 000 REF 000 REJECTS 000 CARD_TAKEN TRANSACTION_END
    """
    
    print("Sample EJ text:")
    print(sample_text.strip())
    print()
    
    # Test with original BERT tokenizer (without custom tokens)
    print("=== ORIGINAL BERT TOKENIZER ===")
    original_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    original_tokens = original_tokenizer.tokenize(sample_text)
    print(f"Original tokenization ({len(original_tokens)} tokens):")
    print(original_tokens)
    print()
    
    # Count subword tokens (starting with ##)
    original_subwords = [token for token in original_tokens if token.startswith('##')]
    print(f"Subword tokens in original: {len(original_subwords)}")
    print(f"Subwords: {original_subwords}")
    print()
    
    # Test with custom vocabulary
    print("=== CUSTOM VOCABULARY BERT TOKENIZER ===")
    custom_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Add the same custom tokens as in our implementation
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
        "CardNumber", "R-10011", "M-02", "REF", "VAL", "ESC", "REJECTS",
        
        # Common combined patterns
        "VAL_000", "ESC_000", "REF_000", "REJECTS_000",
        "OPCODE_FI", "OPCODE_IB", "OPCODE_IC", "OPCODE_ID",
        "ATR_RECEIVED_T_0", "ATR_RECEIVED_T_1"
    ]
    
    num_added_tokens = custom_tokenizer.add_tokens(custom_tokens)
    print(f"Added {num_added_tokens} custom tokens to vocabulary")
    print()
    
    custom_tokens_tokenized = custom_tokenizer.tokenize(sample_text)
    print(f"Custom tokenization ({len(custom_tokens_tokenized)} tokens):")
    print(custom_tokens_tokenized)
    print()
    
    # Count subword tokens (starting with ##)
    custom_subwords = [token for token in custom_tokens_tokenized if token.startswith('##')]
    print(f"Subword tokens in custom: {len(custom_subwords)}")
    print(f"Subwords: {custom_subwords}")
    print()
    
    # Comparison
    print("=== COMPARISON ===")
    print(f"Token reduction: {len(original_tokens)} -> {len(custom_tokens_tokenized)} ({len(original_tokens) - len(custom_tokens_tokenized)} fewer tokens)")
    print(f"Subword reduction: {len(original_subwords)} -> {len(custom_subwords)} ({len(original_subwords) - len(custom_subwords)} fewer subwords)")
    print()
    
    # Test specific tokens
    print("=== SPECIFIC TOKEN TESTS ===")
    test_phrases = [
        "DEVICE_ERROR",
        "CARD_INSERTED", 
        "PIN_ENTERED",
        "VAL_000",
        "ESC_000",
        "REF_000",
        "REJECTS_000",
        "OPCODE_FI",
        "ATR_RECEIVED_T_0",
        "CardNumber"
    ]
    
    for phrase in test_phrases:
        original_result = original_tokenizer.tokenize(phrase)
        custom_result = custom_tokenizer.tokenize(phrase)
        
        print(f"{phrase:20} | Original: {original_result} | Custom: {custom_result}")
        
        # Check if custom tokenizer treats it as single token
        is_single_token = len(custom_result) == 1 and custom_result[0] == phrase.lower()
        print(f"{' ':20} | Single token: {is_single_token}")
    
    print()
    print("=== PREPROCESSING TEST ===")
    
    # Test the preprocessing that creates these patterns
    original_ej_text = """
    *TRANSACTION START*
    CARD INSERTED
    ATR RECEIVED T=0
    OPCODE = FI
    PAN 0004263********1897
    PIN ENTERED
    OPCODE = IB
    DEVICE ERROR
    ESC: 000
    VAL: 000
    REF: 000
    REJECTS:000*(1
    S
    CARD TAKEN
    TRANSACTION END
    """
    
    print("Original EJ patterns:")
    print(original_ej_text.strip())
    print()
    
    # Apply preprocessing transformations
    import re
    
    processed_text = original_ej_text
    
    # Apply same preprocessing as in our code
    processed_text = re.sub(r'\*TRANSACTION START\*', 'TRANSACTION_START', processed_text)
    processed_text = re.sub(r'PAN\s+\d{4}\d+\*+\d+', 'CardNumber', processed_text)
    processed_text = re.sub(r'\bA/C\b', '', processed_text)
    processed_text = re.sub(r'REJECTS:(\d+)\*\([^)]*\)\s*S?', r'REJECTS_\1', processed_text)
    processed_text = re.sub(r'\b(VAL|ESC|REF):\s*(\d+)\b', r'\1_\2', processed_text)
    processed_text = re.sub(r'\bOPCODE\s*=\s*([A-Z]+)\b', r'OPCODE_\1', processed_text)
    processed_text = re.sub(r'\bATR\s+RECEIVED\s+T=(\d+)\b', r'ATR_RECEIVED_T_\1', processed_text)
    
    # Apply compound patterns
    compound_patterns = {
        r'\bDEVICE\s+ERROR\b': 'DEVICE_ERROR',
        r'\bCARD\s+INSERTED\b': 'CARD_INSERTED', 
        r'\bCARD\s+TAKEN\b': 'CARD_TAKEN',
        r'\bPIN\s+ENTERED\b': 'PIN_ENTERED',
        r'\bTRANSACTION\s+END\b': 'TRANSACTION_END',
        r'\bTRANSACTION\s+START\b': 'TRANSACTION_START',
    }
    
    for pattern, replacement in compound_patterns.items():
        processed_text = re.sub(pattern, replacement, processed_text, flags=re.IGNORECASE)
    
    # Clean up whitespace
    processed_text = ' '.join(processed_text.split())
    
    print("After preprocessing:")
    print(processed_text)
    print()
    
    # Test tokenization of processed text
    processed_original_tokens = original_tokenizer.tokenize(processed_text)
    processed_custom_tokens = custom_tokenizer.tokenize(processed_text)
    
    print(f"Processed text tokenization:")
    print(f"Original BERT: {len(processed_original_tokens)} tokens")
    print(f"Custom BERT:   {len(processed_custom_tokens)} tokens")
    print()
    print(f"Original tokens: {processed_original_tokens}")
    print(f"Custom tokens:   {processed_custom_tokens}")
    
    # Count improvements
    original_subwords_processed = [token for token in processed_original_tokens if token.startswith('##')]
    custom_subwords_processed = [token for token in processed_custom_tokens if token.startswith('##')]
    
    print()
    print(f"Subword comparison:")
    print(f"Original subwords: {len(original_subwords_processed)}")
    print(f"Custom subwords:   {len(custom_subwords_processed)}")
    print(f"Subword reduction: {len(original_subwords_processed) - len(custom_subwords_processed)}")


if __name__ == "__main__":
    test_custom_vocabulary()
