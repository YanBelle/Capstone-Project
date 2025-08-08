"""
EJ Log Cleaning and Preprocessing Module
Handles cleaning, normalization and preprocessing of raw EJ logs for analysis
"""

import re
import json
from typing import List, Dict, Tuple, Optional
from loguru import logger
from datetime import datetime
import asyncio

class EJLogCleaner:
    """
    Comprehensive EJ log cleaning and preprocessing system
    """
    
    def __init__(self):
        """Initialize the EJ log cleaner with predefined patterns and rules"""
        
        # Common EJ tokens and their standardized forms
        self.token_standardization = {
            # Card reader events
            'CARD_INSERTED': 'CARD_INSERTED',
            'CARD_TAKEN': 'CARD_TAKEN',
            'CARD_REMOVED': 'CARD_TAKEN',
            'CARD_EJECTED': 'CARD_TAKEN',
            
            # Transaction events
            'TRANSACTION_START': 'TRANSACTION_START',
            'TRANSACTION_BEGIN': 'TRANSACTION_START',
            'TRANSACTION_END': 'TRANSACTION_END',
            'TRANSACTION_COMPLETE': 'TRANSACTION_END',
            
            # Authentication events
            'PIN_ENTERED': 'PIN_ENTERED',
            'PIN_INPUT': 'PIN_ENTERED',
            'PIN_VERIFICATION': 'PIN_ENTERED',
            
            # Cash dispensing
            'NOTES_STACKED': 'NOTES_STACKED',
            'NOTES_PRESENTED': 'NOTES_PRESENTED',
            'NOTES_TAKEN': 'NOTES_TAKEN',
            'CASH_DISPENSED': 'CASH_DISPENSED',
            'CASH_DISPENSED_SUMMARY': 'CASH_DISPENSED_SUMMARY',
            
            # Receipt events
            'RECEIPT_PRINTED': 'RECEIPT_PRINTED',
            'RECEIPT_PRINT': 'RECEIPT_PRINTED',
            'RECEIPT_GENERATED': 'RECEIPT_PRINTED',
            
            # EMV/Chip events
            'GENAC_1_ARQC': 'GENAC_1_ARQC',
            'GENAC_2_TC': 'GENAC_2_TC',
            'GENAC_2_AAC': 'GENAC_2_AAC',
            'GENAC_2_ARQC': 'GENAC_2_ARQC',
            
            # ATR events
            'ATR_RECEIVED_T_0': 'ATR_RECEIVED_T_0',
            'ATR_RECEIVED_T_1': 'ATR_RECEIVED_T_1',
            
            # Operation codes
            'OPCODE_FI': 'OPCODE_FI',
            'OPCODE_BBC': 'OPCODE_BBC',
            'OPCODE_IB': 'OPCODE_IB',
            'OPCODE_DAAC': 'OPCODE_DAAC'
        }
        
        # Patterns for noise removal
        self.noise_patterns = [
            r'\b\d{13,19}\b',  # Card numbers (13-19 digits)
            r'\b\d{4}\s*\d{4}\s*\d{4}\s*\d{4}\b',  # Formatted card numbers
            r'\b[A-Z0-9]{32,}\b',  # Long hex strings
            r'\b\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\b',  # Timestamps
            r'\b\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}\b',  # Alternative timestamps
            r'\b[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}\b',  # UUIDs
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b',  # IP addresses
            r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b',  # Email addresses
        ]
        
        # Device error patterns
        self.device_error_patterns = [
            r'M_\d+',  # Device error codes
            r'ERROR_\w+',
            r'TIMEOUT_\w+',
            r'FAILURE_\w+',
            r'EXCEPTION_\w+'
        ]
        
        # Sequence markers
        self.sequence_markers = [
            'TRANSACTION_START',
            'TRANSACTION_END',
            'SESSION_START',
            'SESSION_END',
            'CARD_INSERTED',
            'CARD_TAKEN'
        ]
        
        logger.info("EJLogCleaner initialized with standardization rules")
    
    def clean_ej_log(self, raw_ej: str, preserve_structure: bool = True) -> Dict[str, str]:
        """
        Comprehensive cleaning of raw EJ log
        
        Args:
            raw_ej: Raw EJ log content
            preserve_structure: Whether to preserve original structure
            
        Returns:
            Dictionary with cleaned versions and metadata
        """
        try:
            if not raw_ej or not raw_ej.strip():
                return {
                    'cleaned_text': '',
                    'normalized_tokens': '',
                    'structured_events': '[]',
                    'cleaning_stats': json.dumps({
                        'original_length': 0,
                        'cleaned_length': 0,
                        'tokens_removed': 0,
                        'tokens_standardized': 0
                    })
                }
            
            # Step 1: Basic cleaning
            cleaned = self._basic_clean(raw_ej)
            
            # Step 2: Remove sensitive data
            cleaned, pii_removed = self._remove_sensitive_data(cleaned)
            
            # Step 3: Standardize tokens
            cleaned, tokens_standardized = self._standardize_tokens(cleaned)
            
            # Step 4: Remove noise
            cleaned, noise_removed = self._remove_noise(cleaned)
            
            # Step 5: Normalize spacing and format
            cleaned = self._normalize_format(cleaned)
            
            # Step 6: Extract structured events
            structured_events = self._extract_structured_events(cleaned)
            
            # Step 7: Create normalized token sequence
            normalized_tokens = self._create_normalized_sequence(cleaned)
            
            # Generate cleaning statistics
            cleaning_stats = {
                'original_length': len(raw_ej),
                'cleaned_length': len(cleaned),
                'pii_items_removed': pii_removed,
                'tokens_standardized': tokens_standardized,
                'noise_items_removed': noise_removed,
                'structured_events_count': len(structured_events),
                'cleaning_timestamp': datetime.now().isoformat()
            }
            
            result = {
                'cleaned_text': cleaned,
                'normalized_tokens': normalized_tokens,
                'structured_events': json.dumps(structured_events),
                'cleaning_stats': json.dumps(cleaning_stats)
            }
            
            logger.debug(f"EJ cleaning completed: {len(raw_ej)} -> {len(cleaned)} chars")
            return result
            
        except Exception as e:
            logger.error(f"Error cleaning EJ log: {e}")
            return {
                'cleaned_text': raw_ej,  # Fallback to original
                'normalized_tokens': raw_ej,
                'structured_events': '[]',
                'cleaning_stats': json.dumps({'error': str(e)})
            }
    
    def _basic_clean(self, text: str) -> str:
        """Basic text cleaning operations"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove non-printable characters except newlines
        text = re.sub(r'[^\x20-\x7E\n\r\t]', '', text)
        
        # Convert to uppercase for consistency
        text = text.upper()
        
        # Remove multiple consecutive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _remove_sensitive_data(self, text: str) -> Tuple[str, int]:
        """Remove personally identifiable information"""
        pii_removed = 0
        
        for pattern in self.noise_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            pii_removed += len(matches)
            
            if 'card' in pattern.lower() or r'\d{13,19}' in pattern:
                text = re.sub(pattern, 'CARDNUMBER', text, flags=re.IGNORECASE)
            elif 'timestamp' in pattern.lower() or r'\d{4}-\d{2}-\d{2}' in pattern:
                text = re.sub(pattern, 'TIMESTAMP', text, flags=re.IGNORECASE)
            elif 'uuid' in pattern.lower() or r'[0-9A-F]{8}-' in pattern:
                text = re.sub(pattern, 'UUID', text, flags=re.IGNORECASE)
            elif 'ip' in pattern.lower() or r'\d{1,3}\.' in pattern:
                text = re.sub(pattern, 'IPADDRESS', text, flags=re.IGNORECASE)
            elif 'email' in pattern.lower() or '@' in pattern:
                text = re.sub(pattern, 'EMAIL', text, flags=re.IGNORECASE)
            else:
                text = re.sub(pattern, 'REDACTED', text, flags=re.IGNORECASE)
        
        return text, pii_removed
    
    def _standardize_tokens(self, text: str) -> Tuple[str, int]:
        """Standardize EJ tokens to consistent forms"""
        standardized_count = 0
        
        # Apply token standardization
        for original, standard in self.token_standardization.items():
            if original in text and original != standard:
                text = text.replace(original, standard)
                standardized_count += text.count(standard) - text.count(original)
        
        # Standardize device error patterns
        for pattern in self.device_error_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                text = text.replace(match, 'DEVICE_ERROR')
                standardized_count += 1
        
        return text, standardized_count
    
    def _remove_noise(self, text: str) -> Tuple[str, int]:
        """Remove noisy elements that don't contribute to analysis"""
        noise_removed = 0
        
        # Remove isolated numbers (likely counters or IDs)
        isolated_numbers = re.findall(r'\b\d{1,6}\b', text)
        noise_removed += len(isolated_numbers)
        text = re.sub(r'\b\d{1,6}\b', '', text)
        
        # Remove very short isolated tokens (likely artifacts)
        short_tokens = re.findall(r'\b[A-Z]{1,2}\b', text)
        noise_removed += len([t for t in short_tokens if t not in ['TC', 'AC', 'FI', 'IB']])
        text = re.sub(r'\b[A-Z]{1,2}\b(?![A-Z_])', lambda m: m.group() if m.group() in ['TC', 'AC', 'FI', 'IB'] else '', text)
        
        # Remove repeated identical tokens (likely logging artifacts)
        words = text.split()
        cleaned_words = []
        last_word = None
        repeat_count = 0
        
        for word in words:
            if word == last_word:
                repeat_count += 1
                if repeat_count < 3:  # Allow up to 2 repetitions
                    cleaned_words.append(word)
                else:
                    noise_removed += 1
            else:
                cleaned_words.append(word)
                repeat_count = 0
            last_word = word
        
        text = ' '.join(cleaned_words)
        
        return text, noise_removed
    
    def _normalize_format(self, text: str) -> str:
        """Normalize text formatting"""
        # Ensure single spaces between tokens
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Ensure proper line breaks for readability
        # Add line breaks after transaction boundaries
        for marker in self.sequence_markers:
            text = text.replace(marker, f'\n{marker}')
        
        # Clean up multiple newlines
        text = re.sub(r'\n{2,}', '\n', text)
        
        return text.strip()
    
    def _extract_structured_events(self, text: str) -> List[Dict]:
        """Extract structured events from cleaned text"""
        events = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            tokens = line.split()
            
            for j, token in enumerate(tokens):
                # Identify event type
                event_type = self._classify_event_type(token)
                
                if event_type != 'UNKNOWN':
                    event = {
                        'sequence': i,
                        'position': j,
                        'token': token,
                        'event_type': event_type,
                        'context': tokens[max(0, j-2):j+3]  # 2 tokens before/after for context
                    }
                    events.append(event)
        
        return events
    
    def _classify_event_type(self, token: str) -> str:
        """Classify a token into an event type"""
        token = token.upper()
        
        if token in ['TRANSACTION_START', 'TRANSACTION_END']:
            return 'TRANSACTION_BOUNDARY'
        elif token in ['CARD_INSERTED', 'CARD_TAKEN']:
            return 'CARD_EVENT'
        elif token in ['PIN_ENTERED']:
            return 'AUTHENTICATION'
        elif token.startswith('GENAC'):
            return 'EMV_COMMAND'
        elif token.startswith('OPCODE'):
            return 'OPERATION_CODE'
        elif token in ['NOTES_STACKED', 'NOTES_PRESENTED', 'NOTES_TAKEN', 'CASH_DISPENSED']:
            return 'CASH_HANDLING'
        elif token in ['RECEIPT_PRINTED']:
            return 'RECEIPT_EVENT'
        elif token.startswith('ATR_'):
            return 'ATR_EVENT'
        elif token == 'DEVICE_ERROR':
            return 'ERROR_EVENT'
        elif token in ['CARDNUMBER', 'TIMESTAMP', 'UUID', 'REDACTED']:
            return 'REDACTED_DATA'
        else:
            return 'UNKNOWN'
    
    def _create_normalized_sequence(self, text: str) -> str:
        """Create a normalized token sequence optimized for ML analysis"""
        lines = text.split('\n')
        normalized_tokens = []
        
        for line in lines:
            tokens = line.split()
            
            # Filter out redacted data and very common tokens
            filtered_tokens = []
            for token in tokens:
                event_type = self._classify_event_type(token)
                if event_type not in ['REDACTED_DATA', 'UNKNOWN']:
                    filtered_tokens.append(token)
            
            if filtered_tokens:
                normalized_tokens.extend(filtered_tokens)
        
        # Join with spaces for ML analysis
        return ' '.join(normalized_tokens)
    
    def clean_batch(self, raw_ej_list: List[str]) -> List[Dict[str, str]]:
        """Clean a batch of EJ logs"""
        results = []
        
        for raw_ej in raw_ej_list:
            cleaned = self.clean_ej_log(raw_ej)
            results.append(cleaned)
        
        logger.info(f"Batch cleaning completed for {len(raw_ej_list)} EJ logs")
        return results
    
    def get_cleaning_summary(self, cleaning_stats_list: List[str]) -> Dict:
        """Generate summary statistics from cleaning operations"""
        try:
            total_stats = {
                'total_logs_processed': len(cleaning_stats_list),
                'total_original_chars': 0,
                'total_cleaned_chars': 0,
                'total_pii_removed': 0,
                'total_tokens_standardized': 0,
                'total_noise_removed': 0,
                'total_events_extracted': 0,
                'compression_ratio': 0.0
            }
            
            for stats_json in cleaning_stats_list:
                try:
                    stats = json.loads(stats_json)
                    total_stats['total_original_chars'] += stats.get('original_length', 0)
                    total_stats['total_cleaned_chars'] += stats.get('cleaned_length', 0)
                    total_stats['total_pii_removed'] += stats.get('pii_items_removed', 0)
                    total_stats['total_tokens_standardized'] += stats.get('tokens_standardized', 0)
                    total_stats['total_noise_removed'] += stats.get('noise_items_removed', 0)
                    total_stats['total_events_extracted'] += stats.get('structured_events_count', 0)
                except:
                    continue
            
            if total_stats['total_original_chars'] > 0:
                total_stats['compression_ratio'] = (
                    total_stats['total_cleaned_chars'] / total_stats['total_original_chars']
                )
            
            return total_stats
            
        except Exception as e:
            logger.error(f"Error generating cleaning summary: {e}")
            return {'error': str(e)}

# Global cleaner instance
ej_cleaner = EJLogCleaner()
