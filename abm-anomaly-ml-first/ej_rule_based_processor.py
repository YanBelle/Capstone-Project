#!/usr/bin/env python3
"""
EJ Rule-Based Processor
========================

A rule-based solution to:
1. Read EJ logs from ./data/input
2. Sessionize the logs 
3. Classify sessions as normal or error-containing
4. Store results in tabular format for easy access

Features:
- Pattern-based error detection
- Comprehensive session classification
- CSV output for easy analysis
- Detailed error reporting
"""

import os
import re
import csv
import json
import pandas as pd
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

class EJRuleBasedProcessor:
    """Rule-based processor for EJ logs with comprehensive error detection"""
    
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
            
            # Incomplete transaction patterns
            'notes_presented_not_taken': {
                'pattern': lambda text: (
                    re.search(r'NOTES PRESENTED', text, re.IGNORECASE) and 
                    not re.search(r'NOTES TAKEN', text, re.IGNORECASE)
                ),
                'severity': 'critical',
                'description': 'Cash presented but not taken by customer'
            },
            'card_inserted_not_taken': {
                'pattern': lambda text: (
                    re.search(r'CARD INSERTED', text, re.IGNORECASE) and 
                    not re.search(r'CARD TAKEN', text, re.IGNORECASE)
                ),
                'severity': 'high',
                'description': 'Card inserted but not retrieved'
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
        
        # Check for normal patterns first (false positive prevention)
        for pattern_name, pattern in self.normal_patterns.items():
            if pattern.search(text):
                if pattern_name == 'card_init_attempts':
                    # Count attempts - up to 3 is normal for magstrip cards
                    attempts = len(re.findall(r'CARD INITIALISE ATTEMPT = [123]', text))
                    if attempts <= 3:
                        continue  # This is normal, don't flag as error
        
        # Check for error patterns
        detected_errors = []
        for error_type, rule in self.error_rules.items():
            pattern = rule['pattern']
            
            # Handle both regex and callable patterns
            if callable(pattern):
                if pattern(text):
                    detected_errors.append({
                        'type': error_type,
                        'severity': rule['severity'],
                        'description': rule['description']
                    })
            else:
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
            session.withdrawal_amount = float(amount_match.group(1))
        
        balance_match = self.financial_patterns['balance'].search(text)
        if balance_match:
            session.account_balance = float(balance_match.group(1))
        
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
    
    def _save_results(self):
        """Save results to CSV files for easy access"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save normal sessions
        normal_df = self._sessions_to_dataframe(self.normal_sessions)
        normal_file = self.output_dir / f"normal_sessions_{timestamp}.csv"
        normal_df.to_csv(normal_file, index=False)
        print(f"💾 Normal sessions saved to: {normal_file}")
        
        # Save error sessions
        error_df = self._sessions_to_dataframe(self.error_sessions)
        error_file = self.output_dir / f"error_sessions_{timestamp}.csv"
        error_df.to_csv(error_file, index=False)
        print(f"💾 Error sessions saved to: {error_file}")
        
        # Save detailed error report
        self._save_error_report(timestamp)
        
        # Save session summaries (without raw text for readability)
        self._save_session_summaries(timestamp)
    
    def _sessions_to_dataframe(self, sessions: List[EJSession]) -> pd.DataFrame:
        """Convert sessions to pandas DataFrame"""
        if not sessions:
            return pd.DataFrame()
        
        # Convert sessions to dictionaries
        session_dicts = []
        for session in sessions:
            session_dict = asdict(session)
            # Convert lists to strings for CSV compatibility
            session_dict['error_types'] = ', '.join(session_dict['error_types'])
            session_dict['error_details'] = json.dumps(session_dict['error_details'])
            session_dicts.append(session_dict)
        
        return pd.DataFrame(session_dicts)
    
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
                        'severity': 'unknown'
                    }
                error_summary[error_type]['count'] += 1
                error_summary[error_type]['sessions'].append(session.session_id)
                
                # Get severity from error details
                for detail in session.error_details:
                    if detail['type'] == error_type:
                        error_summary[error_type]['severity'] = detail['severity']
                        break
        
        # Save error summary
        report_file = self.output_dir / f"error_analysis_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(error_summary, f, indent=2)
        print(f"📊 Error analysis report saved to: {report_file}")
    
    def _save_session_summaries(self, timestamp: str):
        """Save session summaries without raw text for easy browsing"""
        # Normal sessions summary
        if self.normal_sessions:
            normal_summary = []
            for session in self.normal_sessions:
                summary = {
                    'session_id': session.session_id,
                    'file_source': session.file_source,
                    'start_time': session.start_time,
                    'transaction_type': session.transaction_type,
                    'withdrawal_amount': session.withdrawal_amount,
                    'card_inserted': session.card_inserted,
                    'pin_entered': session.pin_entered,
                    'transaction_completed': session.transaction_completed,
                    'notes_taken': session.notes_taken,
                    'card_taken': session.card_taken
                }
                normal_summary.append(summary)
            
            normal_summary_df = pd.DataFrame(normal_summary)
            normal_summary_file = self.output_dir / f"normal_sessions_summary_{timestamp}.csv"
            normal_summary_df.to_csv(normal_summary_file, index=False)
            print(f"📋 Normal sessions summary saved to: {normal_summary_file}")
        
        # Error sessions summary
        if self.error_sessions:
            error_summary = []
            for session in self.error_sessions:
                summary = {
                    'session_id': session.session_id,
                    'file_source': session.file_source,
                    'start_time': session.start_time,
                    'error_types': ', '.join(session.error_types),
                    'error_count': len(session.error_types),
                    'transaction_type': session.transaction_type,
                    'transaction_completed': session.transaction_completed,
                    'severity': 'critical' if any('critical' in str(detail) for detail in session.error_details) else 'high'
                }
                error_summary.append(summary)
            
            error_summary_df = pd.DataFrame(error_summary)
            error_summary_file = self.output_dir / f"error_sessions_summary_{timestamp}.csv"
            error_summary_df.to_csv(error_summary_file, index=False)
            print(f"📋 Error sessions summary saved to: {error_summary_file}")

def main():
    """Main execution function"""
    print("🚀 EJ Rule-Based Processor Starting...")
    print("=" * 50)
    
    # Initialize processor
    processor = EJRuleBasedProcessor()
    
    # Process all files
    result = processor.process_all_files()
    
    print("\n" + "=" * 50)
    print("✅ Processing Complete!")
    print(f"Status: {result['status']}")
    
    if result['status'] == 'success':
        print(f"\n📈 Final Results:")
        print(f"  Files Processed: {result['files_processed']}")
        print(f"  Total Sessions: {result['total_sessions']}")
        print(f"  Normal Sessions: {result['normal_sessions']}")
        print(f"  Error Sessions: {result['error_sessions']}")
        
        if result['error_sessions'] > 0:
            error_rate = (result['error_sessions'] / result['total_sessions']) * 100
            print(f"  Error Rate: {error_rate:.1f}%")
        
        print(f"\n📁 Output files saved in: ./data/processed/")
        print("  - normal_sessions_[timestamp].csv")
        print("  - error_sessions_[timestamp].csv") 
        print("  - normal_sessions_summary_[timestamp].csv")
        print("  - error_sessions_summary_[timestamp].csv")
        print("  - error_analysis_report_[timestamp].json")

if __name__ == "__main__":
    main()
