#!/usr/bin/env python3
"""
Intelligent Sessionizer - NER + Regex Hybrid
===========================================

Drop-in replacement for regex sessionization with ML enhancement.
Maintains same output format for seamless pipeline integration.
"""

import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib
import os
from dataclasses import dataclass

@dataclass
class TransactionSession:
    """Same format as existing TransactionSession"""
    session_id: str
    raw_text: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]

class IntelligentSessionizer:
    """
    Hybrid sessionizer that combines NER with regex fallback.
    Maintains compatibility with existing pipeline.
    """
    
    def __init__(self, use_ner: bool = True, confidence_threshold: float = 0.8):
        self.use_ner = use_ner
        self.confidence_threshold = confidence_threshold
        
        # Initialize NER model for ABM log parsing
        if self.use_ner:
            try:
                self.ner_pipeline = pipeline(
                    "ner",
                    model="bert-base-uncased",  # Can be fine-tuned for ABM logs
                    aggregation_strategy="simple",
                    device=0 if torch.cuda.is_available() else -1
                )
                print("✅ NER model loaded successfully")
            except Exception as e:
                print(f"⚠️ NER model failed to load: {e}, falling back to regex")
                self.use_ner = False
        
        # Regex patterns (fallback)
        self.transaction_start_pattern = re.compile(
            r'(\*(?:TRANSACTION|CARDLESS TRANSACTION)\s+START\*)',
            re.IGNORECASE
        )
        self.timestamp_pattern = re.compile(r'\*(\d{2}/\d{2}/\d{4})\*(\d{2}:\d{2})\*')
    
    def split_into_sessions(self, raw_logs: str, file_path: str = None) -> List[TransactionSession]:
        """
        Main sessionization method - same signature as original.
        Returns same TransactionSession format for pipeline compatibility.
        """
        if self.use_ner:
            # Try NER-based sessionization first
            ner_sessions = self._ner_sessionize(raw_logs, file_path)
            if self._validate_sessions(ner_sessions):
                print(f"✅ NER sessionization successful: {len(ner_sessions)} sessions")
                return ner_sessions
            else:
                print("⚠️ NER sessionization failed validation, falling back to regex")
        
        # Fallback to regex (same as original implementation)
        return self._regex_sessionize(raw_logs, file_path)
    
    def _ner_sessionize(self, raw_logs: str, file_path: str = None) -> List[TransactionSession]:
        """NER-based sessionization using transformer model"""
        sessions = []
        
        # Extract file identifier for unique session IDs
        file_identifier = self._extract_file_identifier(file_path)
        timestamp_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use NER to identify transaction boundaries and entities
        ner_results = self.ner_pipeline(raw_logs)
        
        # Process NER results to identify session boundaries
        log_lines = raw_logs.split('\n')
        session_boundaries = self._extract_session_boundaries_from_ner(ner_results, log_lines)
        
        # Create sessions from boundaries
        for i, (start_idx, end_idx) in enumerate(session_boundaries):
            session_lines = log_lines[start_idx:end_idx]
            session_text = '\n'.join(session_lines)
            
            if len(session_text.strip()) > 50:  # Filter short sessions
                # Generate unique session ID (same format as original)
                content_hash = hashlib.md5(session_text.encode()).hexdigest()[:8]
                session_id = f"{file_identifier}_NER_SESSION_{i+1}_{content_hash}_{timestamp_suffix}"
                
                # Extract timestamps using NER + regex hybrid
                start_time = self._extract_start_time_ner(session_text, ner_results)
                end_time = self._extract_end_time_ner(session_text, ner_results)
                
                session = TransactionSession(
                    session_id=session_id,
                    raw_text=session_text,
                    start_time=start_time,
                    end_time=end_time
                )
                sessions.append(session)
        
        return sessions
    
    def _extract_session_boundaries_from_ner(self, ner_results: List[Dict], log_lines: List[str]) -> List[tuple]:
        """Extract session boundaries using NER entity recognition"""
        boundaries = []
        
        # Look for entities that indicate transaction start/end
        transaction_start_entities = ['TRANSACTION_START', 'SESSION_BEGIN', 'TXN_START']
        
        current_start = 0
        for i, result in enumerate(ner_results):
            if any(entity in result.get('entity', '').upper() for entity in transaction_start_entities):
                if i > 0:  # Close previous session
                    boundaries.append((current_start, i))
                current_start = i
        
        # Add final session
        if current_start < len(log_lines):
            boundaries.append((current_start, len(log_lines)))
        
        return boundaries
    
    def _extract_start_time_ner(self, session_text: str, ner_results: List[Dict]) -> Optional[datetime]:
        """Extract start time using NER + regex hybrid approach"""
        # First try NER to find DATE and TIME entities
        for result in ner_results:
            if result.get('entity') in ['DATE', 'TIME', 'TIMESTAMP']:
                try:
                    # Parse the extracted entity value
                    timestamp_str = result.get('word', '')
                    # Try multiple timestamp formats
                    for fmt in ['%m/%d/%Y %H:%M', '%Y-%m-%d %H:%M:%S', '%H:%M']:
                        try:
                            return datetime.strptime(timestamp_str, fmt)
                        except ValueError:
                            continue
                except:
                    pass
        
        # Fallback to regex pattern matching
        timestamp_match = self.timestamp_pattern.search(session_text[:200])
        if timestamp_match:
            try:
                date_str = timestamp_match.group(1)
                time_str = timestamp_match.group(2)
                return datetime.strptime(f"{date_str} {time_str}", "%m/%d/%Y %H:%M")
            except ValueError:
                pass
        
        return None
    
    def _extract_end_time_ner(self, session_text: str, ner_results: List[Dict]) -> Optional[datetime]:
        """Extract end time using NER + regex hybrid approach"""
        # Similar to start time but look at end of session
        lines = session_text.split('\n')
        for line in reversed(lines[-10:]):  # Check last 10 lines
            timestamp_match = self.timestamp_pattern.search(line)
            if timestamp_match:
                try:
                    date_str = timestamp_match.group(1)
                    time_str = timestamp_match.group(2)
                    return datetime.strptime(f"{date_str} {time_str}", "%m/%d/%Y %H:%M")
                except ValueError:
                    pass
        return None
    
    def _regex_sessionize(self, raw_logs: str, file_path: str = None) -> List[TransactionSession]:
        """Original regex-based sessionization (unchanged)"""
        sessions = []
        file_identifier = self._extract_file_identifier(file_path)
        timestamp_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        log_lines = raw_logs.split('\n')
        start_line_numbers = []
        
        for line_num, line in enumerate(log_lines):
            if self.transaction_start_pattern.search(line):
                start_line_numbers.append(line_num)
        
        # Process sessions (same logic as original)
        for i, start_line_num in enumerate(start_line_numbers):
            if i + 1 < len(start_line_numbers):
                end_line_num = start_line_numbers[i + 1] - 1
            else:
                end_line_num = len(log_lines)
            
            session_start_line = max(0, start_line_num - 1) if start_line_num > 0 else start_line_num
            session_lines = log_lines[session_start_line:end_line_num]
            session_text = '\n'.join(session_lines)
            
            if len(session_text.strip()) > 50:
                content_hash = hashlib.md5(session_text.encode()).hexdigest()[:8]
                session_id = f"{file_identifier}_REGEX_SESSION_{i+1}_{content_hash}_{timestamp_suffix}"
                
                start_time = self._extract_start_time_regex(session_text)
                end_time = self._extract_end_time_regex(session_text)
                
                session = TransactionSession(
                    session_id=session_id,
                    raw_text=session_text,
                    start_time=start_time,
                    end_time=end_time
                )
                sessions.append(session)
        
        return sessions
    
    def _extract_start_time_regex(self, session_text: str) -> Optional[datetime]:
        """Extract start time using regex (original implementation)"""
        timestamp_match = self.timestamp_pattern.search(session_text[:200])
        if timestamp_match:
            try:
                date_str = timestamp_match.group(1)
                time_str = timestamp_match.group(2)
                return datetime.strptime(f"{date_str} {time_str}", "%m/%d/%Y %H:%M")
            except ValueError:
                pass
        return None
    
    def _extract_end_time_regex(self, session_text: str) -> Optional[datetime]:
        """Extract end time using regex (original implementation)"""
        lines = session_text.split('\n')
        for line in reversed(lines[-10:]):
            timestamp_match = self.timestamp_pattern.search(line)
            if timestamp_match:
                try:
                    date_str = timestamp_match.group(1)
                    time_str = timestamp_match.group(2)
                    return datetime.strptime(f"{date_str} {time_str}", "%m/%d/%Y %H:%M")
                except ValueError:
                    pass
        return None
    
    def _extract_file_identifier(self, file_path: str) -> str:
        """Extract file identifier (same as original)"""
        if not file_path:
            return "unknown"
        
        file_name = os.path.basename(file_path)
        file_match = re.search(r'ABM(\d+)EJ_(\d{8})_(\d{8})', file_name)
        if file_match:
            abm_num = file_match.group(1)
            start_date = file_match.group(2)
            return f"ABM{abm_num}_{start_date}"
        else:
            return file_name.replace('.txt', '').replace('.', '_')
    
    def _validate_sessions(self, sessions: List[TransactionSession]) -> bool:
        """Validate that NER sessionization produced reasonable results"""
        if not sessions:
            return False
        
        # Check if sessions have reasonable length
        avg_length = sum(len(s.raw_text) for s in sessions) / len(sessions)
        if avg_length < 100 or avg_length > 10000:  # Reasonable session length
            return False
        
        # Check if we have timestamps
        sessions_with_timestamps = sum(1 for s in sessions if s.start_time)
        if sessions_with_timestamps / len(sessions) < 0.5:  # At least 50% should have timestamps
            return False
        
        return True

# Usage example showing drop-in replacement
if __name__ == "__main__":
    # Initialize intelligent sessionizer
    sessionizer = IntelligentSessionizer(use_ner=True)
    
    # Same interface as original
    with open("sample_ej_log.txt", "r") as f:
        raw_logs = f.read()
    
    # Same method call, same output format
    sessions = sessionizer.split_into_sessions(raw_logs, "sample_ej_log.txt")
    
    print(f"Extracted {len(sessions)} sessions")
    for session in sessions[:3]:  # Show first 3
        print(f"Session ID: {session.session_id}")
        print(f"Start Time: {session.start_time}")
        print(f"Preview: {session.raw_text[:100]}...")
        print("-" * 50)
