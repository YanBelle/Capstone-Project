#!/usr/bin/env python3
"""
EJ Rule-Based Processor - Enhanced CSV-Safe Version
===================================================

A rule-based solution with proper CSV handling for complex data including:
1. Raw text with commas, quotes, and newlines
2. JSON data structures 
3. List fields that might contain commas
4. Proper escaping and quoting

Uses multiple output formats to avoid CSV corruption.
"""

import os
import re
import csv
import json
import base64
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class EJSession:
    """Represents a single EJ session with classification"""
    session_id: str
    file_source: str
    start_time: Optional[str]
    end_time: Optional[str]
    duration_seconds: Optional[float]
    raw_text: str
    
    # Classification fields
    has_errors: bool
    error_types: List[str]
    error_details: List[Dict[str, Any]]
    
    # Transaction details
    transaction_type: Optional[str]
    card_inserted: bool
    pin_entered: bool
    transaction_completed: bool
    notes_dispensed: bool
    notes_taken: bool
    card_taken: bool
    
    # Session metrics
    line_count: int
    character_count: int
    session_length_category: str  # 'short', 'normal', 'long'
    
    # Financial details
    withdrawal_amount: Optional[float]
    deposit_amount: Optional[float]
    account_balance: Optional[float]
    authorization_code: Optional[str]
    
    # BERT preprocessing (with default value - must come last)
    bert_preprocessed_text: Optional[str] = None  # BERT-preprocessed text for ML training

class EJRuleBasedProcessor:
    """Enhanced rule-based processor with CSV-safe output"""
    
    def __init__(self, input_dir: str = "./data/input", output_dir: str = "./data/processed"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize error detection rules
        self._setup_error_rules()
        self._setup_transaction_patterns()
        
        # Results storage
        self.normal_sessions: List[EJSession] = []
        self.error_sessions: List[EJSession] = []
        
    def _setup_error_rules(self):
        """Define comprehensive error detection rules"""
        self.error_rules = {
            # Critical operational errors
            'supervisor_mode_entry': {
                'pattern': re.compile(r'SUPERVISOR MODE ENTRY', re.IGNORECASE),
                'severity': 'critical',
                'description': 'Supervisor intervention required'
            },
            'power_reset': {
                'pattern': re.compile(r'Power-Up/Reset', re.IGNORECASE),
                'severity': 'critical',
                'description': 'System power reset during operation'
            },
            'unable_to_dispense': {
                'pattern': re.compile(r'UNABLE TO DISPENSE', re.IGNORECASE),
                'severity': 'critical',
                'description': 'Cash dispensing failure'
            },
            'deposit_error': {
                'pattern': re.compile(r'DEPOSIT ERROR', re.IGNORECASE),
                'severity': 'high',
                'description': 'Cash deposit processing error'
            },
            
            # Cash retract errors (customer funds retained)
            'cash_retract_bna': {
                'pattern': re.compile(r'INIT BNA STARTED - RETRACT BIN', re.IGNORECASE),
                'severity': 'critical',
                'description': 'BNA retract - customer cash retained'
            },
            'cash_retract_cashin': {
                'pattern': re.compile(r'CASHIN RETRACT STARTED - RETRACT BIN', re.IGNORECASE),
                'severity': 'critical', 
                'description': 'Cash-in retract - customer cash retained'
            },
            'cash_retract_cim': {
                'pattern': re.compile(r'CIM-RESET CALLED - RETRACT BIN', re.IGNORECASE),
                'severity': 'critical',
                'description': 'CIM reset retract - customer cash retained'
            },
            
            # Device errors
            'device_error': {
                'pattern': re.compile(r'DEVICE ERROR', re.IGNORECASE),
                'severity': 'high',
                'description': 'General device malfunction'
            },
            'communication_error': {
                'pattern': re.compile(r'COMMUNICATION ERROR', re.IGNORECASE),
                'severity': 'high',
                'description': 'Communication system failure'
            },
            'hardware_fault': {
                'pattern': re.compile(r'HARDWARE FAULT', re.IGNORECASE),
                'severity': 'high',
                'description': 'Hardware component failure'
            },
            
            # Error codes
            'error_codes': {
                'pattern': re.compile(r'(ESC|VAL|REF|REJECTS):\s*[1-9]\d*', re.IGNORECASE),
                'severity': 'medium',
                'description': 'Error codes detected'
            },
            'timeout_error': {
                'pattern': re.compile(r'TIMEOUT', re.IGNORECASE),
                'severity': 'medium',
                'description': 'Operation timeout'
            }
        }
        
        # Normal patterns that should NOT be flagged as errors
        self.normal_patterns = {
            'card_init_attempts': re.compile(r'CARD INITIALISE ATTEMPT = [123]', re.IGNORECASE),
            'customer_cancelled': re.compile(r'CUSTOMER CANCELLED', re.IGNORECASE),
            'successful_completion': re.compile(r'NOTES PRESENTED.*NOTES TAKEN', re.DOTALL | re.IGNORECASE)
        }
    
    def _setup_transaction_patterns(self):
        """Define transaction classification patterns"""
        self.transaction_patterns = {
            'withdrawal': re.compile(r'WITHDRAWAL|NOTES PRESENTED', re.IGNORECASE),
            'deposit': re.compile(r'DEPOSIT|CIM.*OPERATION', re.IGNORECASE),
            'balance_inquiry': re.compile(r'BALANCE.*INQUIRY|OPCODE.*=.*FI', re.IGNORECASE),
            'pin_change': re.compile(r'PIN.*CHANGE', re.IGNORECASE)
        }
        
        # Financial patterns
        self.financial_patterns = {
            'amount': re.compile(r'WITHDRAWAL\s+(\d+(?:\.\d{2})?)', re.IGNORECASE),
            'balance': re.compile(r'AVAILABLE\s+(\d+(?:\.\d{2})?)', re.IGNORECASE),
            'authorization': re.compile(r'AUTHORIZATION\s+(\d+)', re.IGNORECASE)
        }
    
    def process_all_files(self) -> Dict[str, Any]:
        """Process all EJ files in the input directory"""
        print(f"🔍 Scanning {self.input_dir} for EJ files...")
        
        ej_files = list(self.input_dir.glob("*.txt"))
        if not ej_files:
            print(f"❌ No .txt files found in {self.input_dir}")
            return {"status": "error", "message": "No EJ files found"}
        
        print(f"📁 Found {len(ej_files)} EJ files to process")
        
        total_sessions = 0
        for file_path in ej_files:
            print(f"\n📄 Processing: {file_path.name}")
            sessions = self._process_single_file(file_path)
            total_sessions += len(sessions)
            print(f"   ✅ Extracted {len(sessions)} sessions")
        
        # Generate summary report
        normal_count = len(self.normal_sessions)
        error_count = len(self.error_sessions)
        
        print(f"\n📊 Processing Summary:")
        print(f"   Total Sessions: {total_sessions}")
        if total_sessions > 0:
            print(f"   Normal Sessions: {normal_count} ({normal_count/total_sessions*100:.1f}%)")
            print(f"   Error Sessions: {error_count} ({error_count/total_sessions*100:.1f}%)")
        
        # Save results
        self._save_results()
        
        return {
            "status": "success",
            "total_sessions": total_sessions,
            "normal_sessions": normal_count,
            "error_sessions": error_count,
            "files_processed": len(ej_files)
        }
    
    def _process_single_file(self, file_path: Path) -> List[EJSession]:
        """Process a single EJ file and extract sessions"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_content = f.read()
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            return []
        
        # Sessionize the content
        sessions = self._sessionize_content(raw_content, file_path.name)
        
        # Classify each session
        for session in sessions:
            self._classify_session(session)
            
            # Store in appropriate list
            if session.has_errors:
                self.error_sessions.append(session)
            else:
                self.normal_sessions.append(session)
        
        return sessions
    
    def _sessionize_content(self, content: str, filename: str) -> List[EJSession]:
        """Split content into individual transaction sessions"""
        sessions = []
        
        # Split on transaction start markers
        transaction_pattern = re.compile(r'(\*(?:TRANSACTION|CARDLESS TRANSACTION)\s+START\*)', re.IGNORECASE)
        
        # Find all transaction boundaries
        lines = content.split('\n')
        session_boundaries = []
        
        for i, line in enumerate(lines):
            if transaction_pattern.search(line):
                session_boundaries.append(i)
        
        if not session_boundaries:
            # Alternative: Split on timestamp patterns
            timestamp_pattern = re.compile(r'\*(\d+)\*(\d{2}/\d{2}/\d{4})\*(\d{2}:\d{2})\*')
            for i, line in enumerate(lines):
                if timestamp_pattern.search(line):
                    session_boundaries.append(i)
        
        # Create sessions
        for i, start_idx in enumerate(session_boundaries):
            # Find session end (next session start or end of file)
            if i + 1 < len(session_boundaries):
                end_idx = session_boundaries[i + 1] - 1
            else:
                end_idx = len(lines)
            
            # Include timestamp line before transaction start if available
            actual_start = max(0, start_idx - 1) if start_idx > 0 else start_idx
            
            session_lines = lines[actual_start:end_idx]
            session_text = '\n'.join(session_lines)
            
            if len(session_text.strip()) > 50:  # Skip very short sessions
                session_id = f"{filename}_SESSION_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                session = EJSession(
                    session_id=session_id,
                    file_source=filename,
                    start_time=self._extract_start_time(session_text),
                    end_time=self._extract_end_time(session_text),
                    duration_seconds=None,  # Will calculate later
                    raw_text=session_text,
                    bert_preprocessed_text=None,  # Will be set during classification
                    has_errors=False,  # Will determine in classification
                    error_types=[],
                    error_details=[],
                    transaction_type=None,  # Will determine in classification
                    card_inserted=False,
                    pin_entered=False,
                    transaction_completed=False,
                    notes_dispensed=False,
                    notes_taken=False,
                    card_taken=False,
                    line_count=len(session_lines),
                    character_count=len(session_text),
                    session_length_category='normal',
                    withdrawal_amount=None,
                    deposit_amount=None,
                    account_balance=None,
                    authorization_code=None
                )
                
                sessions.append(session)
        
        return sessions
    
    def _classify_session(self, session: EJSession):
        """Classify session as normal or error-containing"""
        text = session.raw_text
        
        # Generate BERT-preprocessed text for ML training efficiency
        session.bert_preprocessed_text = self._bert_preprocess_text(text)
        
        # Check for normal patterns first (false positive prevention)
        for pattern_name, pattern in self.normal_patterns.items():
            if pattern.search(text):
                if pattern_name == 'card_init_attempts':
                    # Count attempts - up to 3 is normal for magstrip cards
                    attempts = len(re.findall(r'CARD INITIALISE ATTEMPT = [123]', text))
                    if attempts <= 3:
                        continue  # This is normal, don't flag as error
        
        # Check for specific incomplete transaction patterns
        detected_errors = []
        
        # Check for notes presented but not taken
        if re.search(r'NOTES PRESENTED', text, re.IGNORECASE) and not re.search(r'NOTES TAKEN', text, re.IGNORECASE):
            detected_errors.append({
                'type': 'notes_presented_not_taken',
                'severity': 'critical',
                'description': 'Cash presented but not taken by customer'
            })
        
        # Check for card inserted but not taken
        if re.search(r'CARD INSERTED', text, re.IGNORECASE) and not re.search(r'CARD TAKEN', text, re.IGNORECASE):
            detected_errors.append({
                'type': 'card_inserted_not_taken',
                'severity': 'high',
                'description': 'Card inserted but not retrieved'
            })
        
        # Check for other error patterns
        for error_type, rule in self.error_rules.items():
            pattern = rule['pattern']
            if pattern.search(text):
                detected_errors.append({
                    'type': error_type,
                    'severity': rule['severity'],
                    'description': rule['description']
                })
        
        # Set error status
        session.has_errors = len(detected_errors) > 0
        session.error_types = [e['type'] for e in detected_errors]
        session.error_details = detected_errors
        
        # Classify transaction type
        for trans_type, pattern in self.transaction_patterns.items():
            if pattern.search(text):
                session.transaction_type = trans_type
                break
        
        # Extract transaction flow details
        session.card_inserted = bool(re.search(r'CARD INSERTED', text, re.IGNORECASE))
        session.pin_entered = bool(re.search(r'PIN ENTERED', text, re.IGNORECASE))
        session.transaction_completed = bool(re.search(r'TRANSACTION END', text, re.IGNORECASE))
        session.notes_dispensed = bool(re.search(r'NOTES PRESENTED', text, re.IGNORECASE))
        session.notes_taken = bool(re.search(r'NOTES TAKEN', text, re.IGNORECASE))
        session.card_taken = bool(re.search(r'CARD TAKEN', text, re.IGNORECASE))
        
        # Extract financial details
        amount_match = self.financial_patterns['amount'].search(text)
        if amount_match:
            try:
                session.withdrawal_amount = float(amount_match.group(1))
            except ValueError:
                pass
        
        balance_match = self.financial_patterns['balance'].search(text)
        if balance_match:
            try:
                session.account_balance = float(balance_match.group(1))
            except ValueError:
                pass
        
        auth_match = self.financial_patterns['authorization'].search(text)
        if auth_match:
            session.authorization_code = auth_match.group(1)
        
        # Determine session length category
        if session.line_count < 10:
            session.session_length_category = 'short'
        elif session.line_count > 50:
            session.session_length_category = 'long'
        else:
            session.session_length_category = 'normal'
    
    def _extract_start_time(self, text: str) -> Optional[str]:
        """Extract session start time"""
        # Look for timestamp pattern in first few lines
        timestamp_pattern = re.compile(r'\*(\d{2}/\d{2}/\d{4})\*(\d{2}:\d{2})\*')
        match = timestamp_pattern.search(text[:200])
        if match:
            return f"{match.group(1)} {match.group(2)}"
        return None
    
    def _extract_end_time(self, text: str) -> Optional[str]:
        """Extract session end time"""
        # Look for transaction end or last timestamp
        lines = text.split('\n')
        for line in reversed(lines[-10:]):  # Check last 10 lines
            timestamp_pattern = re.compile(r'\*(\d{2}/\d{2}/\d{4})\*(\d{2}:\d{2})\*')
            match = timestamp_pattern.search(line)
            if match:
                return f"{match.group(1)} {match.group(2)}"
        return None
    
    def _bert_preprocess_text(self, text: str) -> str:
        """
        Preprocess EJ text using the same method as BertVisualizationAnalyzer._preprocess_text
        This ensures consistency with BERT model training and reduces preprocessing time during training
        """
        # Enhanced EJ pattern cleaning with specific fixes for BERT attention optimization
        
        # NEW: NOISE REDUCTION - Replace verbose sections with concise event labels
        # 1. Replace Cash Dispensing Summary with concise event
        # Enhanced pattern to match various cash dispensing table formats
        cash_summary_pattern = r'CASH\s+TOTAL\s+TYPE\d+.*?REMAINING\s+\d+(?:\s+\d+)*'
        text = re.sub(cash_summary_pattern, 'CASH_DISPENSED_SUMMARY', text, flags=re.DOTALL)
        
        # 2. Replace Receipt section with concise event - ENHANCED for NCB format
        # Pattern 1: NCB MIDAS format - Enhanced to capture complete receipt including THANK YOU FOR USING THE MULTILINK NETWORK
        receipt_pattern1 = r'N\.C\.B\.\s+MIDAS.*?(?:THANK YOU FOR USING\s+THE MULTILINK NETWORK|THANK YOU)'
        text = re.sub(receipt_pattern1, ' RECEIPT_PRINTED ', text, flags=re.DOTALL)
        
        # Pattern 2: General bank name + receipt content ending with THANK YOU (with proper spacing)
        receipt_pattern2 = r'([A-Z][A-Z\s\.]+(?:BANK|CREDIT UNION|ATM)[^\n]*\n(?:.*?\n)*?.*?(?:THANK YOU FOR USING\s+THE MULTILINK NETWORK|THANK YOU))'
        text = re.sub(receipt_pattern2, ' RECEIPT_PRINTED ', text, flags=re.DOTALL)
        
        # Pattern 3: DATE/TIME/MACHINE format receipts  
        receipt_pattern3 = r'(DATE:\s*[^\n]*\n(?:.*?\n)*?.*?(?:THANK YOU FOR USING\s+THE MULTILINK NETWORK|THANK YOU))'
        text = re.sub(receipt_pattern3, ' RECEIPT_PRINTED ', text, flags=re.DOTALL)
        
        # Pattern 4: Simple receipt format with institution names
        receipt_pattern4 = r'(\s+[A-Z][A-Z\s]+\n\s*(?:ATM|RECEIPT)[^\n]*\n(?:.*?\n)*?.*?(?:THANK YOU FOR USING\s+THE MULTILINK NETWORK|THANK YOU))'
        text = re.sub(receipt_pattern4, ' RECEIPT_PRINTED ', text, flags=re.DOTALL)
        
        # CUSTOM VOCABULARY ENHANCEMENTS - Convert specific patterns to compound tokens
        # 1. GENAC patterns: Convert "GENAC <digit> : <text>" to "GENAC_<digit>_<text>"
        text = re.sub(r'\bGENAC\s+(\d+)\s*:\s*([A-Z]+)\b', r'GENAC_\1_\2', text)
        
        # 2. CARD INITIALISE ATTEMPT with counter: Convert "CARD INITIALISE ATTEMPT = <digit>" to "CARD_INITIALISE_ATTEMPT_<digit>"
        text = re.sub(r'\bCARD\s+INITIALISE\s+ATTEMPT\s*=\s*(\d+)\b', r'CARD_INITIALISE_ATTEMPT_\1', text)
        
        # 3. CARD status patterns: Convert "*<sequence>*1*D*9,M-<Mstatus>,R-<Rstatus>" to "D_9 M_<Mstatus> R_<Rstatus>"
        # This pattern follows CARD INSERTED and CARD INITIALISE ATTEMPT
        text = re.sub(r'\*\d+\*1\*D\*9,M-(\d+),R-(\d+)', r'D_9 M_\1 R_\2', text)
        
        # 4. EXTERNAL AUTHENTICATE patterns: Convert "EXTERNAL AUTHENTICATE: NO ARPC" to "EXTERNAL_AUTHENTICATE_NO_ARPC"
        text = re.sub(r'\bEXTERNAL\s+AUTHENTICATE\s*:\s*NO\s+ARPC\b', 'EXTERNAL_AUTHENTICATE_NO_ARPC', text)
        text = re.sub(r'\bEXTERNAL\s+AUTHENTICATE\b', 'EXTERNAL_AUTHENTICATE', text)
        
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
        
        # ENHANCED: Handle REJECTS patterns more comprehensively
        # Clean up "REJECTS:000*(1\nS" to just "REJECTS_000"
        text = re.sub(r'REJECTS:(\d+)\*\([^)]*\)\s*S?', r'REJECTS_\1', text)
        
        # Additional cleanup for any remaining REJECTS fragments
        text = re.sub(r'REJECTS:(\d+)\*\([^)]*\)[^A-Z]*', r'REJECTS_\1', text)
        
        # Handle remaining REJECTS:000 patterns that don't have the full pattern
        text = re.sub(r'REJECTS:(\d+)', r'REJECTS_\1', text)
        
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
            
            # NEW: Cash handling events (these were handled earlier but ensure consistency)
            r'\bNOTES\s+STACKED\b': 'NOTES_STACKED',
            r'\bNOTES\s+PRESENTED\b': 'NOTES_PRESENTED', 
            r'\bNOTES\s+TAKEN\b': 'NOTES_TAKEN',
            r'\bPRIMARY\s+CARD\s+READER\s+ACTIVATED\b': 'PRIMARY_CARD_READER_ACTIVATED',
            
            # Custom vocabulary patterns (backup in case the earlier patterns didn't catch them)
            # Note: CARD_INITIALISE_ATTEMPT now handled with counters in earlier section
            r'\bEXTERNAL\s+AUTHENTICATE\b': 'EXTERNAL_AUTHENTICATE',
            
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
        
        # Note: BERT tokenization truncation would require importing tokenizer
        # For now, we'll do a simple character-based truncation to approximate BERT's 512 token limit
        # Rough approximation: average token length is ~4 characters, so 512 tokens ≈ 2048 characters
        if len(text) > 2048:
            text = text[:2048].rsplit(' ', 1)[0]  # Truncate at word boundary
        
        return text
    
    def _clean_text_for_csv(self, text: str) -> str:
        """Clean text to be CSV-safe"""
        if not text:
            return ""
        
        # Replace newlines with space to keep CSV structure
        cleaned = text.replace('\n', ' ').replace('\r', ' ')
        
        # Replace multiple spaces with single space
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Trim whitespace
        cleaned = cleaned.strip()
        
        # If still too long, truncate and add indicator
        if len(cleaned) > 32767:  # Excel cell limit
            cleaned = cleaned[:32760] + "...[TRUNCATED]"
        
        return cleaned
    
    def _save_results(self):
        """Save results to multiple formats for maximum compatibility"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save CSV summaries (safe for Excel/analysis)
        self._save_csv_summaries(timestamp)
        
        # 2. Save JSON with full data (preserves all structure)
        self._save_json_data(timestamp)
        
        # 3. Save raw text files separately
        self._save_raw_text_files(timestamp)
        
        # 4. Save detailed error report
        self._save_error_report(timestamp)
        
        print(f"\n📁 Output files saved in: {self.output_dir}")
        print("   CSV Summaries (Excel-safe):")
        print(f"     - normal_sessions_summary_{timestamp}.csv")
        print(f"     - error_sessions_summary_{timestamp}.csv")
        print("   Complete Data (JSON with BERT preprocessing):")
        print(f"     - normal_sessions_full_{timestamp}.json")
        print(f"     - error_sessions_full_{timestamp}.json")
        print("   Raw Session Text:")
        print(f"     - raw_sessions_{timestamp}.txt")
        print("   Analysis:")
        print(f"     - error_analysis_report_{timestamp}.json")
        print("\n🎯 BERT Preprocessing Benefits:")
        print("   ✅ Preprocessed text included in JSON for ML training efficiency")
        print("   ✅ Reduces model training time (no repeated preprocessing)")
        print("   ✅ Consistent with BertVisualizationAnalyzer preprocessing")
        print("   ✅ Optimized token patterns for ATM domain understanding")
    
    def _save_csv_summaries(self, timestamp: str):
        """Save CSV summaries without problematic raw text"""
        # Normal sessions summary
        if self.normal_sessions:
            normal_summary_file = self.output_dir / f"normal_sessions_summary_{timestamp}.csv"
            with open(normal_summary_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['session_id', 'file_source', 'start_time', 'end_time', 
                            'transaction_type', 'withdrawal_amount', 'deposit_amount',
                            'card_inserted', 'pin_entered', 'transaction_completed', 
                            'notes_dispensed', 'notes_taken', 'card_taken',
                            'line_count', 'character_count', 'session_length_category',
                            'authorization_code']
                writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
                writer.writeheader()
                
                for session in self.normal_sessions:
                    summary = {
                        'session_id': session.session_id,
                        'file_source': session.file_source,
                        'start_time': session.start_time or '',
                        'end_time': session.end_time or '',
                        'transaction_type': session.transaction_type or '',
                        'withdrawal_amount': session.withdrawal_amount or '',
                        'deposit_amount': session.deposit_amount or '',
                        'card_inserted': session.card_inserted,
                        'pin_entered': session.pin_entered,
                        'transaction_completed': session.transaction_completed,
                        'notes_dispensed': session.notes_dispensed,
                        'notes_taken': session.notes_taken,
                        'card_taken': session.card_taken,
                        'line_count': session.line_count,
                        'character_count': session.character_count,
                        'session_length_category': session.session_length_category,
                        'authorization_code': session.authorization_code or ''
                    }
                    writer.writerow(summary)
            
            print(f"💾 Normal sessions summary saved to: {normal_summary_file}")
        
        # Error sessions summary
        if self.error_sessions:
            error_summary_file = self.output_dir / f"error_sessions_summary_{timestamp}.csv"
            with open(error_summary_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['session_id', 'file_source', 'start_time', 'end_time',
                            'error_types_count', 'error_types_list', 'highest_severity',
                            'transaction_type', 'transaction_completed', 'line_count',
                            'withdrawal_amount', 'notes_presented_not_taken', 'card_not_taken']
                writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
                writer.writeheader()
                
                for session in self.error_sessions:
                    # Determine highest severity
                    severities = [detail.get('severity', 'unknown') for detail in session.error_details]
                    severity_order = ['critical', 'high', 'medium', 'low', 'info']
                    highest_severity = 'unknown'
                    for severity in severity_order:
                        if severity in severities:
                            highest_severity = severity
                            break
                    
                    # Safe error types list (using | separator to avoid CSV confusion)
                    error_types_safe = ' | '.join(session.error_types) if session.error_types else ''
                    
                    summary = {
                        'session_id': session.session_id,
                        'file_source': session.file_source,
                        'start_time': session.start_time or '',
                        'end_time': session.end_time or '',
                        'error_types_count': len(session.error_types),
                        'error_types_list': error_types_safe,
                        'highest_severity': highest_severity,
                        'transaction_type': session.transaction_type or '',
                        'transaction_completed': session.transaction_completed,
                        'line_count': session.line_count,
                        'withdrawal_amount': session.withdrawal_amount or '',
                        'notes_presented_not_taken': 'notes_presented_not_taken' in session.error_types,
                        'card_not_taken': 'card_inserted_not_taken' in session.error_types
                    }
                    writer.writerow(summary)
            
            print(f"💾 Error sessions summary saved to: {error_summary_file}")
    
    def _save_json_data(self, timestamp: str):
        """Save complete data as JSON (preserves all structure)"""
        # Convert sessions to JSON-serializable format
        def session_to_dict(session):
            return {
                'session_id': session.session_id,
                'file_source': session.file_source,
                'start_time': session.start_time,
                'end_time': session.end_time,
                'duration_seconds': session.duration_seconds,
                'has_errors': session.has_errors,
                'error_types': session.error_types,
                'error_details': session.error_details,
                'transaction_type': session.transaction_type,
                'card_inserted': session.card_inserted,
                'pin_entered': session.pin_entered,
                'transaction_completed': session.transaction_completed,
                'notes_dispensed': session.notes_dispensed,
                'notes_taken': session.notes_taken,
                'card_taken': session.card_taken,
                'line_count': session.line_count,
                'character_count': session.character_count,
                'session_length_category': session.session_length_category,
                'withdrawal_amount': session.withdrawal_amount,
                'deposit_amount': session.deposit_amount,
                'account_balance': session.account_balance,
                'authorization_code': session.authorization_code,
                'raw_text_base64': base64.b64encode(session.raw_text.encode('utf-8')).decode('ascii'),
                'bert_preprocessed_text': session.bert_preprocessed_text,
                'preprocessing_info': {
                    'raw_text_length': len(session.raw_text),
                    'preprocessed_text_length': len(session.bert_preprocessed_text) if session.bert_preprocessed_text else 0,
                    'compression_ratio': (len(session.bert_preprocessed_text) / len(session.raw_text)) if session.bert_preprocessed_text and len(session.raw_text) > 0 else 0,
                    'preprocessing_method': 'BertVisualizationAnalyzer._preprocess_text',
                    'preprocessing_purpose': 'ML training optimization - reduces tokenization and preprocessing time'
                }
            }
        
        # Save normal sessions
        if self.normal_sessions:
            normal_data = [session_to_dict(session) for session in self.normal_sessions]
            normal_json_file = self.output_dir / f"normal_sessions_full_{timestamp}.json"
            with open(normal_json_file, 'w', encoding='utf-8') as f:
                json.dump(normal_data, f, indent=2, ensure_ascii=False)
            print(f"💾 Normal sessions (full) saved to: {normal_json_file}")
        
        # Save error sessions
        if self.error_sessions:
            error_data = [session_to_dict(session) for session in self.error_sessions]
            error_json_file = self.output_dir / f"error_sessions_full_{timestamp}.json"
            with open(error_json_file, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, indent=2, ensure_ascii=False)
            print(f"💾 Error sessions (full) saved to: {error_json_file}")
    
    def _save_raw_text_files(self, timestamp: str):
        """Save raw session text in separate file for easy reading"""
        raw_text_file = self.output_dir / f"raw_sessions_{timestamp}.txt"
        
        with open(raw_text_file, 'w', encoding='utf-8') as f:
            f.write("EJ SESSION RAW TEXT DUMP\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write normal sessions
            if self.normal_sessions:
                f.write("NORMAL SESSIONS\n")
                f.write("-" * 20 + "\n\n")
                for session in self.normal_sessions:
                    f.write(f"SESSION ID: {session.session_id}\n")
                    f.write(f"FILE: {session.file_source}\n")
                    f.write(f"TYPE: {session.transaction_type}\n")
                    f.write(f"START: {session.start_time}\n")
                    f.write("RAW TEXT:\n")
                    f.write("-" * 40 + "\n")
                    f.write(session.raw_text)
                    f.write("\n" + "=" * 50 + "\n\n")
            
            # Write error sessions
            if self.error_sessions:
                f.write("ERROR SESSIONS\n")
                f.write("-" * 20 + "\n\n")
                for session in self.error_sessions:
                    f.write(f"SESSION ID: {session.session_id}\n")
                    f.write(f"FILE: {session.file_source}\n")
                    f.write(f"ERRORS: {', '.join(session.error_types)}\n")
                    f.write(f"TYPE: {session.transaction_type}\n")
                    f.write(f"START: {session.start_time}\n")
                    f.write("RAW TEXT:\n")
                    f.write("-" * 40 + "\n")
                    f.write(session.raw_text)
                    f.write("\n" + "=" * 50 + "\n\n")
        
        print(f"💾 Raw session text saved to: {raw_text_file}")
    
    def _save_error_report(self, timestamp: str):
        """Save detailed error analysis report"""
        if not self.error_sessions:
            return
        
        error_summary = {}
        for session in self.error_sessions:
            for error_type in session.error_types:
                if error_type not in error_summary:
                    error_summary[error_type] = {
                        'count': 0,
                        'sessions': [],
                        'severity': 'unknown',
                        'description': 'Unknown error type'
                    }
                error_summary[error_type]['count'] += 1
                error_summary[error_type]['sessions'].append(session.session_id)
                
                # Get severity and description from error details
                for detail in session.error_details:
                    if detail['type'] == error_type:
                        error_summary[error_type]['severity'] = detail['severity']
                        error_summary[error_type]['description'] = detail['description']
                        break
        
        # Save error summary
        report_file = self.output_dir / f"error_analysis_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(error_summary, f, indent=2)
        print(f"📊 Error analysis report saved to: {report_file}")

def main():
    """Main execution function"""
    print("🚀 EJ Rule-Based Processor (CSV-Safe Version)")
    print("=" * 55)
    
    # Initialize processor
    processor = EJRuleBasedProcessor()
    
    # Process all files
    result = processor.process_all_files()
    
    print("\n" + "=" * 55)
    print("✅ Processing Complete!")
    print(f"Status: {result['status']}")
    
    if result['status'] == 'success':
        print(f"\n📈 Final Results:")
        print(f"  Files Processed: {result['files_processed']}")
        print(f"  Total Sessions: {result['total_sessions']}")
        print(f"  Normal Sessions: {result['normal_sessions']}")
        print(f"  Error Sessions: {result['error_sessions']}")
        
        if result['error_sessions'] > 0 and result['total_sessions'] > 0:
            error_rate = (result['error_sessions'] / result['total_sessions']) * 100
            print(f"  Error Rate: {error_rate:.1f}%")
        
        print(f"\n🎯 Output Strategy:")
        print("  📊 CSV Files: Clean summaries (Excel-compatible)")
        print("  📋 JSON Files: Complete data with BERT preprocessing for ML training") 
        print("  📄 TXT Files: Raw session text for review")
        print("  🚀 ML Optimization: BERT-preprocessed text reduces training time")

if __name__ == "__main__":
    main()
