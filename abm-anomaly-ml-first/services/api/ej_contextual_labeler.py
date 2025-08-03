"""
EJ Log Contextual Labeling System
Enhanced BERT understanding for financial ATM transaction logs
"""

import re
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

# Set up logging
logger = logging.getLogger(__name__)

class EventType(Enum):
    """Enhanced event types for EJ logs"""
    # Existing types
    TXN_START = "transaction_start"
    TXN_END = "transaction_end"
    CARD_INSERT = "card_insert"
    CARD_REMOVE = "card_remove"
    PIN_ENTRY = "pin_entry"
    CASH_DISPENSE = "cash_dispense"
    RECEIPT_PRINT = "receipt_print"
    ERROR = "error"
    WARNING = "warning"
    
    # New contextual types from EJ analysis
    SUPERVISOR_ENTRY = "supervisor_entry"
    SUPERVISOR_EXIT = "supervisor_exit"
    DEVICE_RECOVERY = "device_recovery"
    CASH_RECONCILIATION = "cash_reconciliation"
    EXTERNAL_AUTH = "external_auth"
    NOTES_PRESENT = "notes_present"
    NOTES_TAKEN = "notes_taken"
    CIM_OPERATION = "cim_operation"
    BNA_OPERATION = "bna_operation"
    CHEQUE_OPERATION = "cheque_operation"
    
    # Enhanced ATM operational events
    ATM_IN_SERVICE = "atm_in_service"
    CARD_READER_ACTIVATED = "card_reader_activated"
    CASH_TOTAL_REPORT = "cash_total_report"
    NOTES_STACKED = "notes_stacked"
    
    # CIM (Cash-In Module) Deposit Operations
    CIM_DEPOSIT_ACTIVATED = "cim_deposit_activated"
    CIM_SHUTTER_OPENED = "cim_shutter_opened"
    CIM_ITEMS_INSERTED = "cim_items_inserted"
    CIM_INPUT_REFUSED = "cim_input_refused"
    CIM_ITEMS_PRESENTED = "cim_items_presented"
    CIM_ITEMS_TAKEN = "cim_items_taken"
    CIM_DEPOSIT_COMPLETED = "cim_deposit_completed"
    CASHIN_DEPOSIT_SELECTED = "cashin_deposit_selected"
    
    # Note Quality and Serial Number Analysis
    FAILED_SERIAL_READ = "failed_serial_read"
    NOTE_CATEGORIZATION = "note_categorization"
    
    # Enhanced Recovery Operations
    RETRACT_BIN_OPERATION = "retract_bin_operation"
    CASHIN_RETRACT_STARTED = "cashin_retract_started"

class TransactionPhase(Enum):
    """Transaction lifecycle phases"""
    INITIALIZATION = "initialization"
    CARD_AUTHENTICATION = "card_authentication"
    PIN_VERIFICATION = "pin_verification"
    ACCOUNT_SELECTION = "account_selection"
    TRANSACTION_SELECTION = "transaction_selection"
    AMOUNT_ENTRY = "amount_entry"
    PROCESSING = "processing"
    CASH_DISPENSING = "cash_dispensing"
    CASH_DEPOSITING = "cash_depositing"
    DEPOSIT_VERIFICATION = "deposit_verification"
    NOTE_QUALITY_CHECK = "note_quality_check"
    RECEIPT_PRINTING = "receipt_printing"
    COMPLETION = "completion"
    ERROR_HANDLING = "error_handling"

class OperationalMode(Enum):
    """Track operational context"""
    NORMAL = "normal"
    SUPERVISOR = "supervisor"
    RECOVERY = "recovery"
    MAINTENANCE = "maintenance"
    OUT_OF_SERVICE = "out_of_service"
    IN_SERVICE_WAITING = "in_service_waiting"  # ATM active and waiting for customers

class RecoveryType(Enum):
    """Types of recovery operations"""
    CIM_RESET = "cim_reset"
    BNA_INIT = "bna_init"
    CHEQUE_RECOVERY = "cheque_recovery"
    CASHIN_RETRACT = "cashin_retract"
    DEVICE_INIT = "device_init"
    COMMUNICATION_RESET = "communication_reset"
    RETRACT_BIN_INIT = "retract_bin_init"
    RETRACT_BIN_CASHIN = "retract_bin_cashin"
    RETRACT_BIN_CIM_RESET = "retract_bin_cim_reset"

class Severity(Enum):
    """Severity levels for events"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Categories of errors"""
    HARDWARE = "hardware"
    SOFTWARE = "software"
    NETWORK = "network"
    SECURITY = "security"
    CASH_HANDLING = "cash_handling"
    CARD_READER = "card_reader"
    RECEIPT_PRINTER = "receipt_printer"
    COMMUNICATION = "communication"

@dataclass
class EJLogLabel:
    """Enhanced contextual label for EJ log lines"""
    line_number: int
    timestamp: Optional[datetime] = None
    phase: TransactionPhase = TransactionPhase.INITIALIZATION
    event_type: EventType = EventType.TXN_START
    severity: Severity = Severity.INFO
    error_category: Optional[ErrorCategory] = None
    error_code: Optional[str] = None
    entity: Optional[str] = None  # Device/component involved
    amount: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Enhanced contextual fields
    operational_mode: OperationalMode = OperationalMode.NORMAL
    recovery_type: Optional[RecoveryType] = None
    denomination_data: Optional[Dict[str, int]] = None
    auth_failure_type: Optional[str] = None
    transaction_id: Optional[str] = None
    customer_present: bool = True
    confidence_score: float = 1.0
    
    # Note quality and deposit analysis
    note_categories: Optional[Dict[str, int]] = None  # CAT1, CAT2, CAT3, CAT4, CAT5 counts
    serial_read_failures: Optional[int] = None
    deposit_amount: Optional[float] = None
    rejected_reason: Optional[str] = None
    cim_status: Optional[Dict[str, Any]] = None  # CIM status block data (ESC, VAL, REF, etc.)

class EJLogLabeler:
    """Advanced EJ Log contextual labeling system"""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
        self.recovery_patterns = self._initialize_recovery_patterns()
        self.error_codes = self._initialize_error_codes()
        self.phase_transitions = self._initialize_phase_transitions()
        
        # State tracking
        self.current_transaction_id = None
        self.session_start_time = None
        self.last_transaction_end_time = None
        self.supervisor_mode_start_time = None
    
    def _initialize_patterns(self) -> Dict[str, Tuple[Optional[TransactionPhase], EventType]]:
        """Initialize pattern recognition for EJ logs"""
        return {
            # Transaction lifecycle
            r'\*TRANSACTION START\*': (TransactionPhase.INITIALIZATION, EventType.TXN_START),
            r'\*TRANSACTION END\*': (TransactionPhase.COMPLETION, EventType.TXN_END),
            r'CARD INSERTED': (TransactionPhase.CARD_AUTHENTICATION, EventType.CARD_INSERT),
            r'CARD REMOVED': (TransactionPhase.COMPLETION, EventType.CARD_REMOVE),
            r'PIN ENTRY': (TransactionPhase.PIN_VERIFICATION, EventType.PIN_ENTRY),
            r'NOTES DISPENSED': (TransactionPhase.CASH_DISPENSING, EventType.CASH_DISPENSE),
            
            # ATM operational status patterns
            r'PRIMARY CARD READER ACTIVATED': (None, EventType.CARD_READER_ACTIVATED),
            r'ATM IN SERVICE': (None, EventType.ATM_IN_SERVICE),
            
            # Cash handling events with precise meaning
            r'NOTES PRESENTED': (TransactionPhase.CASH_DISPENSING, EventType.NOTES_PRESENT),
            r'NOTES TAKEN': (TransactionPhase.COMPLETION, EventType.NOTES_TAKEN),
            r'NOTES STACKED': (TransactionPhase.CASH_DISPENSING, EventType.NOTES_STACKED),
            
            # Cash reconciliation reporting
            r'CASH TOTAL\s+TYPE1\s+TYPE2\s+TYPE3\s+TYPE4': (None, EventType.CASH_TOTAL_REPORT),
            
            # Receipt printing - detected by receipt content patterns
            r'^\s*N\.C\.B\.\s+MIDAS': (TransactionPhase.RECEIPT_PRINTING, EventType.RECEIPT_PRINT),
            r'^\s*NCB\s+.*\s+BRANCH': (TransactionPhase.RECEIPT_PRINTING, EventType.RECEIPT_PRINT),
            r'THANK YOU\s*$': (TransactionPhase.RECEIPT_PRINTING, EventType.RECEIPT_PRINT),
            
            # Supervisor mode patterns (CRITICAL for EJ understanding)
            r'SUPERVISOR MODE ENTRY': (None, EventType.SUPERVISOR_ENTRY),
            r'SUPERVISOR MODE EXIT': (None, EventType.SUPERVISOR_EXIT),
            
    # Enhanced retract bin operations (MUST come before general recovery patterns)
    r'INIT BNA STARTED - RETRACT BIN': (None, EventType.RETRACT_BIN_OPERATION),
    r'CASHIN RETRACT STARTED - RETRACT BIN': (None, EventType.CASHIN_RETRACT_STARTED),
    r'CIM-RESET CALLED - RETRACT BIN': (None, EventType.RETRACT_BIN_OPERATION),
    
    # CIM Status Block Recognition
    r'A/C OPERATION OK': (None, EventType.CIM_OPERATION),            # Recovery patterns (Device reliability indicators)
            r'INIT BNA STARTED': (None, EventType.DEVICE_RECOVERY),
            r'CIM-RESET CALLED': (None, EventType.DEVICE_RECOVERY),
            r'CASHIN RECOVERY OK': (None, EventType.DEVICE_RECOVERY),
            r'CHEQUE RECOVERY': (None, EventType.DEVICE_RECOVERY),
            r'DEVICE RESET': (None, EventType.DEVICE_RECOVERY),
            
            # CIM Deposit Operations (Cash-In Module)
            r'CIM-DEPOSIT ACTIVATED': (TransactionPhase.CASH_DEPOSITING, EventType.CIM_DEPOSIT_ACTIVATED),
            r'CIM-SHUTTER OPENED': (TransactionPhase.CASH_DEPOSITING, EventType.CIM_SHUTTER_OPENED),
            r'CIM-ITEMS INSERTED': (TransactionPhase.CASH_DEPOSITING, EventType.CIM_ITEMS_INSERTED),
            r'CIM-INPUT REFUSED': (TransactionPhase.NOTE_QUALITY_CHECK, EventType.CIM_INPUT_REFUSED),
            r'CIM-ITEMS PRESENTED': (TransactionPhase.DEPOSIT_VERIFICATION, EventType.CIM_ITEMS_PRESENTED),
            r'CIM-ITEMS TAKEN': (TransactionPhase.COMPLETION, EventType.CIM_ITEMS_TAKEN),
            r'CIM-DEPOSIT COMPLETED': (TransactionPhase.COMPLETION, EventType.CIM_DEPOSIT_COMPLETED),
            r'CASHIN DEPOSIT SELECTED': (TransactionPhase.TRANSACTION_SELECTION, EventType.CASHIN_DEPOSIT_SELECTED),
            
            # Note Quality and Serial Number Analysis
            r'FAILED SERIAL NUMBER READS': (TransactionPhase.NOTE_QUALITY_CHECK, EventType.FAILED_SERIAL_READ),
            r'CAT[1-5] NOTES:\s*\d+': (TransactionPhase.NOTE_QUALITY_CHECK, EventType.NOTE_CATEGORIZATION),
            
            # Cash handling patterns (Financial reconciliation)
            r'NOTES PRESENTED': (TransactionPhase.CASH_DISPENSING, EventType.NOTES_PRESENT),
            r'NOTES TAKEN': (TransactionPhase.COMPLETION, EventType.NOTES_TAKEN),
            r'NOTES STACKED': (TransactionPhase.CASH_DISPENSING, EventType.NOTES_STACKED),
            r'CASHIN STARTED': (TransactionPhase.PROCESSING, EventType.CIM_OPERATION),
            r'RETRACT BIN': (TransactionPhase.ERROR_HANDLING, EventType.BNA_OPERATION),
            
            # Authentication patterns (Security events)
            r'EXTERNAL AUTHENTICATE': (TransactionPhase.CARD_AUTHENTICATION, EventType.EXTERNAL_AUTH),
            r'PIN VERIFIED': (TransactionPhase.PIN_VERIFICATION, EventType.PIN_ENTRY),
            r'GENAC': (TransactionPhase.CARD_AUTHENTICATION, EventType.EXTERNAL_AUTH),
            
            # Error patterns
            r'ERROR': (TransactionPhase.ERROR_HANDLING, EventType.ERROR),
            r'FAILED': (TransactionPhase.ERROR_HANDLING, EventType.ERROR),
            r'TIMEOUT': (TransactionPhase.ERROR_HANDLING, EventType.ERROR),
            r'DEVICE ERROR': (TransactionPhase.ERROR_HANDLING, EventType.ERROR),
        }
    
    def _initialize_recovery_patterns(self) -> Dict[str, RecoveryType]:
        """Map recovery indicators to types"""
        return {
            'INIT BNA STARTED': RecoveryType.BNA_INIT,
            'CIM-RESET CALLED': RecoveryType.CIM_RESET,
            'CHEQUE RECOVERY': RecoveryType.CHEQUE_RECOVERY,
            'CASHIN RETRACT STARTED': RecoveryType.CASHIN_RETRACT,
            'DEVICE INIT': RecoveryType.DEVICE_INIT,
            'COMM RESET': RecoveryType.COMMUNICATION_RESET,
            
            # Enhanced retract bin operations
            'INIT BNA STARTED - RETRACT BIN': RecoveryType.RETRACT_BIN_INIT,
            'CASHIN RETRACT STARTED - RETRACT BIN': RecoveryType.RETRACT_BIN_CASHIN,
            'CIM-RESET CALLED - RETRACT BIN': RecoveryType.RETRACT_BIN_CIM_RESET,
        }
    
    def _initialize_error_codes(self) -> Dict[str, Tuple[str, Severity, ErrorCategory]]:
        """Map EJ error codes to descriptions"""
        return {
            'M-38': ('External authentication failure', Severity.ERROR, ErrorCategory.SECURITY),
            'M-01': ('Device communication error', Severity.CRITICAL, ErrorCategory.COMMUNICATION),
            'M-15': ('Cash dispenser error', Severity.ERROR, ErrorCategory.CASH_HANDLING),
            'M-23': ('Card reader error', Severity.ERROR, ErrorCategory.CARD_READER),
            'M-45': ('Receipt printer error', Severity.WARNING, ErrorCategory.RECEIPT_PRINTER),
            'M-67': ('Network communication failure', Severity.CRITICAL, ErrorCategory.NETWORK),
            'E-01': ('Hardware malfunction', Severity.CRITICAL, ErrorCategory.HARDWARE),
            'E-12': ('Software exception', Severity.ERROR, ErrorCategory.SOFTWARE),
            'W-05': ('Low cash warning', Severity.WARNING, ErrorCategory.CASH_HANDLING),
            'W-18': ('Maintenance required', Severity.WARNING, ErrorCategory.HARDWARE),
        }
    
    def _initialize_phase_transitions(self) -> Dict[TransactionPhase, List[TransactionPhase]]:
        """Define valid phase transitions for transaction flow validation"""
        return {
            TransactionPhase.INITIALIZATION: [TransactionPhase.CARD_AUTHENTICATION],
            TransactionPhase.CARD_AUTHENTICATION: [TransactionPhase.PIN_VERIFICATION, TransactionPhase.ERROR_HANDLING],
            TransactionPhase.PIN_VERIFICATION: [TransactionPhase.ACCOUNT_SELECTION, TransactionPhase.ERROR_HANDLING],
            TransactionPhase.ACCOUNT_SELECTION: [TransactionPhase.TRANSACTION_SELECTION, TransactionPhase.ERROR_HANDLING],
            TransactionPhase.TRANSACTION_SELECTION: [TransactionPhase.AMOUNT_ENTRY, TransactionPhase.PROCESSING, 
                                                   TransactionPhase.CASH_DEPOSITING, TransactionPhase.ERROR_HANDLING],
            TransactionPhase.AMOUNT_ENTRY: [TransactionPhase.PROCESSING, TransactionPhase.ERROR_HANDLING],
            TransactionPhase.PROCESSING: [TransactionPhase.CASH_DISPENSING, TransactionPhase.CASH_DEPOSITING, 
                                        TransactionPhase.RECEIPT_PRINTING, TransactionPhase.ERROR_HANDLING],
            TransactionPhase.CASH_DISPENSING: [TransactionPhase.RECEIPT_PRINTING, TransactionPhase.COMPLETION, TransactionPhase.ERROR_HANDLING],
            TransactionPhase.CASH_DEPOSITING: [TransactionPhase.NOTE_QUALITY_CHECK, TransactionPhase.DEPOSIT_VERIFICATION, 
                                             TransactionPhase.RECEIPT_PRINTING, TransactionPhase.ERROR_HANDLING],
            TransactionPhase.NOTE_QUALITY_CHECK: [TransactionPhase.DEPOSIT_VERIFICATION, TransactionPhase.CASH_DEPOSITING, 
                                                TransactionPhase.ERROR_HANDLING],
            TransactionPhase.DEPOSIT_VERIFICATION: [TransactionPhase.RECEIPT_PRINTING, TransactionPhase.COMPLETION, 
                                                  TransactionPhase.CASH_DEPOSITING, TransactionPhase.ERROR_HANDLING],
            TransactionPhase.RECEIPT_PRINTING: [TransactionPhase.COMPLETION, TransactionPhase.ERROR_HANDLING],
            TransactionPhase.ERROR_HANDLING: [TransactionPhase.COMPLETION, TransactionPhase.INITIALIZATION],
            TransactionPhase.COMPLETION: [TransactionPhase.INITIALIZATION],
        }
    
    def label_log(self, log_text: str) -> List[EJLogLabel]:
        """Enhanced labeling with deep EJ log context awareness"""
        lines = log_text.split('\n')
        labels = []
        
        # Enhanced state tracking for financial transaction context
        current_phase = TransactionPhase.INITIALIZATION
        transaction_active = False
        supervisor_mode = False
        recovery_active = False
        current_recovery_type = None
        transaction_context = {}
        cash_operations = []
        auth_sequence = []
        
        # Receipt detection state
        in_receipt_block = False
        receipt_start_line = None
        receipt_content = []
        
        for line_num, line in enumerate(lines):
            if not line.strip():
                continue
            
            # Check for receipt printing patterns first
            receipt_detected, receipt_info = self._detect_receipt_content(line, in_receipt_block, receipt_content)
            
            if receipt_detected:
                if not in_receipt_block:
                    # Starting a receipt block
                    in_receipt_block = True
                    receipt_start_line = line_num
                    receipt_content = [line.strip()]
                    current_phase = TransactionPhase.RECEIPT_PRINTING
                else:
                    # Continuing receipt block
                    receipt_content.append(line.strip())
                
                # Check if this is the end of receipt (multiple patterns)
                if ('THANK YOU' in line.upper() or 
                    re.search(r'CENTRE\s+1-888-622-3477', line, re.IGNORECASE) or
                    re.search(r'CONTACT.*CARE.*CENTRE', line, re.IGNORECASE)):
                    in_receipt_block = False
                    # Create receipt label with full content
                    receipt_label = self._create_receipt_label(
                        line_num, receipt_start_line, receipt_content, receipt_info
                    )
                    labels.append(receipt_label)
                    receipt_content = []
                    receipt_start_line = None
                    continue
                elif in_receipt_block:
                    # Skip processing individual receipt lines as separate events
                    continue
            
            # Extract base information for non-receipt lines
            timestamp = self._extract_timestamp(line)
            phase, event_type = self._determine_phase_and_event(line, current_phase)
            severity, error_category, error_code = self._analyze_severity(line)
            entity = self._extract_entity(line)
            amount = self._extract_amount(line)
            
            # Track ATM service status (CRITICAL for operational understanding)
            if event_type == EventType.CARD_READER_ACTIVATED:
                logger.info(f"ATM in service - waiting for customers at line {line_num}")
            elif event_type == EventType.ATM_IN_SERVICE:
                logger.info(f"ATM service mode activated at line {line_num}")
            
            # Track supervisor mode (CRITICAL for operational context)
            if event_type == EventType.SUPERVISOR_ENTRY:
                supervisor_mode = True
                self.supervisor_mode_start_time = timestamp
                logger.info(f"Supervisor mode entered at line {line_num}")
                
                # Check for suspicious supervisor mode timing
                if transaction_active:
                    logger.warning(f"ANOMALY: Supervisor mode entered during active transaction at line {line_num}")
                elif (self.last_transaction_end_time and timestamp and 
                      (timestamp - self.last_transaction_end_time).total_seconds() < 30):
                    logger.warning(f"ANOMALY: Supervisor mode entered {(timestamp - self.last_transaction_end_time).total_seconds():.1f}s after transaction end")
                    
            elif event_type == EventType.SUPERVISOR_EXIT:
                supervisor_mode = False
                
                # Check for unusually short supervisor sessions
                if self.supervisor_mode_start_time and timestamp:
                    supervisor_duration = (timestamp - self.supervisor_mode_start_time).total_seconds()
                    if supervisor_duration < 60:  # Less than 1 minute
                        logger.warning(f"ANOMALY: Very short supervisor session ({supervisor_duration:.1f}s) at line {line_num}")
                    elif supervisor_duration < 120:  # Less than 2 minutes
                        logger.info(f"Short supervisor session ({supervisor_duration:.1f}s) at line {line_num}")
                
                # Don't reset supervisor_mode_start_time yet - need it for contextual anomaly analysis
                logger.info(f"Supervisor mode exited at line {line_num}")
            
            # Track recovery operations (Device reliability)
            if event_type == EventType.DEVICE_RECOVERY:
                recovery_active = True
                current_recovery_type = self._identify_recovery_type(line)
                logger.warning(f"Recovery operation detected: {current_recovery_type}")
            elif 'RECOVERY OK' in line or 'INIT COMPLETE' in line:
                recovery_active = False
                current_recovery_type = None
            
            # Track transaction lifecycle
            if event_type == EventType.TXN_START:
                transaction_active = True
                self.current_transaction_id = self._extract_transaction_id(line)
                transaction_context = {'start_time': timestamp, 'events': []}
            elif event_type == EventType.TXN_END:
                transaction_active = False
                self.last_transaction_end_time = timestamp
                transaction_context = {}
            
            # Determine operational mode
            operational_mode = self._determine_operational_mode(
                supervisor_mode, recovery_active, transaction_active, event_type
            )
            
            # Extract denomination data for cash reconciliation
            denomination_data = self._extract_denomination_data(line)
            if denomination_data:
                cash_operations.append(denomination_data)
            
            # Check for authentication failures (Security context)
            auth_failure = self._check_auth_failure(line, lines, line_num)
            if auth_failure:
                auth_sequence.append(auth_failure)
            
            # Phase transition validation and assignment
            final_phase = current_phase  # Default to current phase
            
            if phase and phase != current_phase:
                if self._is_valid_transition(current_phase, phase):
                    current_phase = phase
                    final_phase = phase
                else:
                    # Allow certain direct transitions for cash dispensing events
                    if (event_type in [EventType.NOTES_STACKED, EventType.NOTES_PRESENT, EventType.NOTES_TAKEN] and
                        phase == TransactionPhase.CASH_DISPENSING):
                        current_phase = phase
                        final_phase = phase
                    elif (event_type == EventType.NOTES_TAKEN and 
                          phase == TransactionPhase.COMPLETION):
                        current_phase = phase
                        final_phase = phase
                    else:
                        logger.warning(f"Invalid phase transition: {current_phase} -> {phase} at line {line_num}")
                        # Use the suggested phase for this specific event even if transition is invalid
                        final_phase = phase if phase else current_phase
            elif phase:
                # Phase is specified and matches current, use it
                final_phase = phase
            else:
                # No phase specified, infer from event type
                if event_type in [EventType.NOTES_STACKED, EventType.NOTES_PRESENT]:
                    final_phase = TransactionPhase.CASH_DISPENSING
                elif event_type == EventType.NOTES_TAKEN:
                    final_phase = TransactionPhase.COMPLETION
                elif event_type == EventType.RECEIPT_PRINT:
                    final_phase = TransactionPhase.RECEIPT_PRINTING
            
            # Extract note quality and deposit analysis data
            note_categories = self._extract_note_categories(line)
            serial_failures = self._extract_serial_failures(line)
            deposit_amount = self._extract_deposit_amount(line)
            refusal_reason = self._extract_cim_refusal_reason(line)
            cim_status = self._extract_cim_status_block(line)
            
            # Create enhanced label with financial context
            label = EJLogLabel(
                line_number=line_num,
                timestamp=timestamp,
                phase=final_phase,
                event_type=event_type,
                severity=severity,
                error_category=error_category,
                error_code=error_code,
                entity=entity,
                amount=amount,
                metadata=self._create_metadata(line, transaction_context),
                operational_mode=operational_mode,
                recovery_type=current_recovery_type if recovery_active else None,
                denomination_data=denomination_data,
                auth_failure_type=auth_failure,
                transaction_id=self.current_transaction_id,
                customer_present=self._determine_customer_presence(operational_mode, event_type),
                confidence_score=self._calculate_confidence(line, event_type),
                # Enhanced note quality and deposit fields
                note_categories=note_categories,
                serial_read_failures=serial_failures,
                deposit_amount=deposit_amount,
                rejected_reason=refusal_reason,
                cim_status=cim_status
            )
            
            # Add context-specific anomaly detection
            label = self._add_contextual_anomalies(label, supervisor_mode, recovery_active, transaction_active)
            
            labels.append(label)
        
        # Post-process for transaction flow analysis
        labels = self._analyze_transaction_flows(labels)
        labels = self._detect_pattern_anomalies(labels)
        labels = self._analyze_cash_reports(labels)
        labels = self._analyze_cash_dispensing_timing(labels)
        
        return labels
    
    def _extract_timestamp(self, line: str) -> Optional[datetime]:
        """Extract timestamp from EJ log line"""
        # EJ timestamp patterns: [020t*629*06/18/2025*00:46*] or hh:mm:ss before event
        patterns = [
            # Time before event: "07:07:56 NOTES STACKED"
            r'^(\d{2}:\d{2}:\d{2})\s+',
            # EJ bracket format: [020t*629*06/18/2025*00:46*
            r'\[.*?(\d{2}/\d{2}/\d{4})\*(\d{2}:\d{2})',
            # Standard formats
            r'(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})',
            r'(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2})'
        ]
        
        for i, pattern in enumerate(patterns):
            match = re.search(pattern, line)
            if match:
                try:
                    if i == 0:  # Time-only format before event
                        time_str = match.group(1)
                        # Use today's date with the extracted time
                        today = datetime.now().date()
                        return datetime.combine(today, datetime.strptime(time_str, "%H:%M:%S").time())
                    elif i == 1:  # EJ bracket format
                        date_str = match.group(1)
                        time_str = match.group(2)
                        return datetime.strptime(f"{date_str} {time_str}", "%m/%d/%Y %H:%M")
                    else:  # Standard formats
                        date_str = match.group(1)
                        time_str = match.group(2)
                        if len(date_str) == 10:  # YYYY-MM-DD format
                            return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S" if len(time_str) == 8 else "%Y-%m-%d %H:%M")
                        else:  # MM/DD/YYYY format
                            return datetime.strptime(f"{date_str} {time_str}", "%m/%d/%Y %H:%M")
                except ValueError:
                    continue
        return None
    
    def _determine_phase_and_event(self, line: str, current_phase: TransactionPhase) -> Tuple[Optional[TransactionPhase], EventType]:
        """Determine transaction phase and event type with EJ context"""
        for pattern, (phase, event_type) in self.patterns.items():
            if re.search(pattern, line, re.IGNORECASE):
                return phase, event_type
        
        # Default event type based on content analysis
        if any(error in line.upper() for error in ['ERROR', 'FAILED', 'FAULT', 'TIMEOUT']):
            return TransactionPhase.ERROR_HANDLING, EventType.ERROR
        elif any(warn in line.upper() for warn in ['WARNING', 'WARN', 'LOW']):
            return current_phase, EventType.WARNING
        
        return None, EventType.TXN_START
    
    def _analyze_severity(self, line: str) -> Tuple[Severity, Optional[ErrorCategory], Optional[str]]:
        """Analyze severity with EJ-specific error codes"""
        line_upper = line.upper()
        
        # Check for specific error codes
        for code, (description, severity, category) in self.error_codes.items():
            if code in line:
                return severity, category, code
        
        # Pattern-based severity analysis
        if any(critical in line_upper for critical in ['CRITICAL', 'FATAL', 'COMMUNICATION FAILURE', 'DEVICE OFFLINE']):
            return Severity.CRITICAL, ErrorCategory.HARDWARE, None
        elif any(error in line_upper for error in ['ERROR', 'FAILED', 'FAULT', 'AAC', 'NO ARPC']):
            return Severity.ERROR, ErrorCategory.SOFTWARE, None
        elif any(warn in line_upper for warn in ['WARNING', 'WARN', 'LOW', 'MAINTENANCE']):
            return Severity.WARNING, ErrorCategory.HARDWARE, None
        
        return Severity.INFO, None, None
    
    def _extract_entity(self, line: str) -> Optional[str]:
        """Extract device/component entity from line"""
        entities = {
            'CIM': 'cash_in_module',
            'BNA': 'bill_note_acceptor', 
            'CARD': 'card_reader',
            'RECEIPT': 'receipt_printer',
            'CASH': 'cash_dispenser',
            'CHEQUE': 'cheque_module',
            'COMM': 'communication',
            'HOST': 'host_system'
        }
        
        line_upper = line.upper()
        for key, entity in entities.items():
            if key in line_upper:
                return entity
        return None
    
    def _extract_amount(self, line: str) -> Optional[float]:
        """Extract monetary amount from line"""
        # Look for amount patterns: $100.00, 100.00, etc.
        patterns = [
            r'\$(\d+\.?\d*)',
            r'AMOUNT[:\s]+(\d+\.?\d*)',
            r'(\d+\.\d{2})\s*(?:DOLLARS?|USD)',
            r'DISPENSE[:\s]+(\d+\.?\d*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return None
    
    def _extract_note_categories(self, line: str) -> Optional[Dict[str, int]]:
        """Extract note categorization data (CAT1-CAT5)"""
        categories = {}
        
        # Pattern: "CAT4 NOTES: 1" or "CAT1 NOTES: 5"
        cat_pattern = r'CAT([1-5])\s+NOTES:\s*(\d+)'
        match = re.search(cat_pattern, line, re.IGNORECASE)
        if match:
            cat_level = int(match.group(1))
            count = int(match.group(2))
            categories[f'CAT{cat_level}'] = count
            return categories
        
        # Multi-category line: "CAT1: 3, CAT2: 1, CAT4: 1"
        multi_cat_pattern = r'CAT([1-5]):\s*(\d+)'
        matches = re.findall(multi_cat_pattern, line, re.IGNORECASE)
        if matches:
            for cat_level, count in matches:
                categories[f'CAT{cat_level}'] = int(count)
            return categories
        
        return None
    
    def _extract_serial_failures(self, line: str) -> Optional[int]:
        """Extract failed serial number read count"""
        # Pattern: "FAILED SERIAL NUMBER READS and CAT4 NOTES: 1"
        serial_pattern = r'FAILED SERIAL NUMBER READS.*?(\d+)'
        match = re.search(serial_pattern, line, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        # Direct pattern: "FAILED SERIAL NUMBER READS: 3"
        direct_pattern = r'FAILED SERIAL NUMBER READS:\s*(\d+)'
        match = re.search(direct_pattern, line, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        return None
    
    def _extract_cim_refusal_reason(self, line: str) -> Optional[str]:
        """Extract CIM input refusal reason"""
        # Pattern: "CIM-INPUT REFUSED,REASON-INVALID MEDIA"
        if 'CIM-INPUT REFUSED' in line.upper():
            reason_match = re.search(r'REASON-(.+?)(?:\s|$)', line, re.IGNORECASE)
            if reason_match:
                return reason_match.group(1).strip()
            return 'unspecified_reason'
        return None
    
    def _extract_cim_status_block(self, line: str) -> Optional[Dict[str, Any]]:
        """Extract CIM status block information from A/C OPERATION OK lines"""
        if 'A/C OPERATION OK' not in line:
            return None
        
        status_data = {}
        
        # Parse ESC (Escrow) count - notes currently in escrow
        esc_match = re.search(r'ESC:\s*(\d+)', line, re.IGNORECASE)
        if esc_match:
            status_data['escrow_count'] = int(esc_match.group(1))
        
        # Parse VAL (Validated) count - notes successfully validated
        val_match = re.search(r'VAL:\s*(\d+)', line, re.IGNORECASE)
        if val_match:
            status_data['validated_count'] = int(val_match.group(1))
        
        # Parse REF (Refused) count - notes refused this cycle
        ref_match = re.search(r'REF:\s*(\d+)', line, re.IGNORECASE)
        if ref_match:
            status_data['refused_count'] = int(ref_match.group(1))
        
        # Parse REJECTS total - cumulative rejections
        rejects_match = re.search(r'REJECTS:\s*(\d+)', line, re.IGNORECASE)
        if rejects_match:
            status_data['total_rejects'] = int(rejects_match.group(1))
        
        # Parse JMD denomination counts (e.g., "JMD$5000: 2")
        jmd_pattern = r'JMD\$(\d+):\s*(\d+)'
        jmd_matches = re.findall(jmd_pattern, line, re.IGNORECASE)
        if jmd_matches:
            denominations = {}
            total_value = 0
            for denom_str, count_str in jmd_matches:
                denomination = int(denom_str)
                count = int(count_str)
                denominations[f'JMD_{denomination}'] = count
                total_value += denomination * count
            
            status_data['denominations'] = denominations
            status_data['total_deposit_value'] = total_value
        
        # Parse currency code if different from JMD
        currency_match = re.search(r'([A-Z]{3})\$', line)
        if currency_match:
            status_data['currency'] = currency_match.group(1)
        else:
            status_data['currency'] = 'JMD'  # Default to JMD
        
        # Calculate deposit progress metrics
        if 'escrow_count' in status_data and 'validated_count' in status_data:
            total_notes = status_data['escrow_count'] + status_data['validated_count']
            if total_notes > 0:
                validation_rate = status_data['validated_count'] / total_notes
                status_data['validation_rate'] = validation_rate
                
                if validation_rate == 1.0:
                    status_data['deposit_status'] = 'FULLY_VALIDATED'
                elif validation_rate > 0.5:
                    status_data['deposit_status'] = 'MOSTLY_VALIDATED'
                elif validation_rate > 0:
                    status_data['deposit_status'] = 'PARTIALLY_VALIDATED'
                else:
                    status_data['deposit_status'] = 'PENDING_VALIDATION'
        
        # Calculate rejection rate if we have reject data
        if 'total_rejects' in status_data and 'total_deposit_value' in status_data:
            # Estimate notes processed (this is approximate)
            if 'denominations' in status_data:
                total_notes_attempted = sum(status_data['denominations'].values()) + status_data['total_rejects']
                if total_notes_attempted > 0:
                    rejection_rate = status_data['total_rejects'] / total_notes_attempted
                    status_data['rejection_rate'] = rejection_rate
                    
                    # Classify rejection severity
                    if rejection_rate > 0.3:
                        status_data['rejection_severity'] = 'HIGH'
                    elif rejection_rate > 0.1:
                        status_data['rejection_severity'] = 'MODERATE'
                    elif rejection_rate > 0:
                        status_data['rejection_severity'] = 'LOW'
                    else:
                        status_data['rejection_severity'] = 'NONE'
        
        return status_data if status_data else None
    
    def _classify_cim_deposit(self, cim_status: Dict[str, Any]) -> str:
        """Classify the CIM deposit transaction based on status data"""
        if not cim_status:
            return 'UNKNOWN'
        
        # Determine deposit complexity
        rejection_rate = cim_status.get('rejection_rate', 0)
        total_rejects = cim_status.get('total_rejects', 0)
        validation_rate = cim_status.get('validation_rate', 0)
        denomination_count = len(cim_status.get('denominations', {}))
        total_value = cim_status.get('total_deposit_value', 0)
        
        # High-risk classification
        if rejection_rate > 0.3 or total_rejects > 10:
            return 'HIGH_REJECTION_RISK'
        elif total_value > 100000:  # Very large deposit
            return 'LARGE_VALUE_DEPOSIT'
        elif denomination_count > 3:
            return 'MIXED_DENOMINATION_COMPLEX'
        elif total_rejects > 5:
            return 'MULTIPLE_RETRY_DEPOSIT'
        elif validation_rate == 1.0 and total_rejects == 0:
            return 'CLEAN_SUCCESSFUL_DEPOSIT'
        elif validation_rate > 0.8:
            return 'MOSTLY_SUCCESSFUL_DEPOSIT'
        elif validation_rate > 0.5:
            return 'MODERATE_SUCCESS_DEPOSIT'
        elif validation_rate > 0:
            return 'CHALLENGING_DEPOSIT'
        else:
            return 'PENDING_VALIDATION_DEPOSIT'
    
    def _extract_deposit_amount(self, line: str) -> Optional[float]:
        """Extract deposit amount from CIM operations"""
        # Look for deposit-specific amount patterns
        patterns = [
            r'DEPOSIT[:\s]+\$?(\d+\.?\d*)',
            r'CIM.*?AMOUNT[:\s]+\$?(\d+\.?\d*)',
            r'DEPOSITED[:\s]+\$?(\d+\.?\d*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return None
    
    def _extract_denomination_data(self, line: str) -> Optional[Dict[str, int]]:
        """Extract cash denomination data for reconciliation"""
        if 'NOTES PRESENTED' in line:
            # Parse "NOTES PRESENTED 1,1,1,1"
            match = re.search(r'NOTES PRESENTED ([\d,]+)', line)
            if match:
                counts = match.group(1).split(',')
                try:
                    return {f'type_{i+1}': int(count) for i, count in enumerate(counts)}
                except ValueError:
                    return None
        
        # Enhanced cash total parsing for the specific EJ format
        elif 'CASH TOTAL' in line and 'TYPE1' in line:
            # This is the header line - extract the structure but don't parse numbers yet
            return {'cash_total_header': True, 'format': 'TYPE1_TYPE2_TYPE3_TYPE4'}
        
        elif 'DENOMINATION' in line and re.search(r'\d{4}', line):
            # Parse denomination values: "DENOMINATION      1000  2000  5000  5000"
            numbers = re.findall(r'\d+', line)
            if len(numbers) >= 4:
                return {
                    'denomination_values': {
                        'type_1': int(numbers[0]),
                        'type_2': int(numbers[1]), 
                        'type_3': int(numbers[2]),
                        'type_4': int(numbers[3])
                    }
                }
        
        elif 'DISPENSED' in line and re.search(r'\d{5}', line):
            # Parse dispensed counts: "DISPENSED        00271 00243 00621 00540"
            numbers = re.findall(r'\d{5}', line)
            if len(numbers) >= 4:
                return {
                    'dispensed_counts': {
                        'type_1': int(numbers[0]),
                        'type_2': int(numbers[1]),
                        'type_3': int(numbers[2]),
                        'type_4': int(numbers[3])
                    }
                }
        
        elif 'REJECTED' in line and re.search(r'\d{5}', line):
            # Parse rejected counts: "REJECTED         00003 00001 00010 00003"
            numbers = re.findall(r'\d{5}', line)
            if len(numbers) >= 4:
                return {
                    'rejected_counts': {
                        'type_1': int(numbers[0]),
                        'type_2': int(numbers[1]),
                        'type_3': int(numbers[2]),
                        'type_4': int(numbers[3])
                    }
                }
        
        elif 'REMAINING' in line and re.search(r'\d{5}', line):
            # Parse remaining counts: "REMAINING        01729 01757 01379 01460"
            numbers = re.findall(r'\d{5}', line)
            if len(numbers) >= 4:
                return {
                    'remaining_counts': {
                        'type_1': int(numbers[0]),
                        'type_2': int(numbers[1]),
                        'type_3': int(numbers[2]),
                        'type_4': int(numbers[3])
                    }
                }
        
        return None
    
    def _check_auth_failure(self, line: str, lines: List[str], line_num: int) -> Optional[str]:
        """Check for authentication failures with context"""
        if 'NO ARPC' in line:
            return 'no_arpc'
        elif 'GENAC 2 : AAC' in line:
            # Check previous lines for context
            context_lines = lines[max(0, line_num-5):line_num]
            if any('NO ARPC' in context_line for context_line in context_lines):
                return 'external_auth_failed'
            return 'chip_declined'
        elif 'PIN FAILED' in line:
            return 'pin_verification_failed'
        elif 'CARD DECLINED' in line:
            return 'card_declined'
        
        return None
    
    def _identify_recovery_type(self, line: str) -> Optional[RecoveryType]:
        """Identify specific recovery operation type"""
        for pattern, recovery_type in self.recovery_patterns.items():
            if pattern in line:
                return recovery_type
        return None
    
    def _determine_operational_mode(self, supervisor_mode: bool, recovery_active: bool, 
                                  transaction_active: bool, event_type: EventType = None) -> OperationalMode:
        """Determine current operational mode"""
        if supervisor_mode:
            return OperationalMode.SUPERVISOR
        elif recovery_active:
            return OperationalMode.RECOVERY
        elif event_type == EventType.CARD_READER_ACTIVATED or event_type == EventType.ATM_IN_SERVICE:
            return OperationalMode.IN_SERVICE_WAITING
        elif not transaction_active and self.session_start_time and any(keyword in str(self.session_start_time) 
                                          for keyword in ['MAINTENANCE', 'OUT OF SERVICE']):
            return OperationalMode.OUT_OF_SERVICE
        else:
            return OperationalMode.NORMAL
    
    def _determine_customer_presence(self, operational_mode: OperationalMode, 
                                   event_type: EventType) -> bool:
        """Determine if customer is likely present"""
        if operational_mode in [OperationalMode.SUPERVISOR, OperationalMode.MAINTENANCE]:
            return False
        if event_type in [EventType.DEVICE_RECOVERY, EventType.SUPERVISOR_ENTRY]:
            return False
        return True
    
    def _calculate_confidence(self, line: str, event_type: EventType) -> float:
        """Calculate confidence score for the labeling"""
        base_confidence = 0.8
        
        # Increase confidence for clear patterns
        if any(clear_pattern in line.upper() for clear_pattern in [
            'TRANSACTION START', 'TRANSACTION END', 'SUPERVISOR MODE', 
            'NOTES PRESENTED', 'CARD INSERTED'
        ]):
            base_confidence = 0.95
        
        # Decrease for ambiguous lines
        if len(line.split()) < 3:
            base_confidence *= 0.7
        
        return min(1.0, base_confidence)
    
    def _create_metadata(self, line: str, transaction_context: Dict) -> Dict[str, Any]:
        """Create metadata dictionary with EJ-specific information"""
        metadata = {}
        
        # Add transaction context
        if transaction_context:
            metadata['transaction_context'] = transaction_context
        
        # Add technical details
        if 'TIMEOUT' in line:
            timeout_match = re.search(r'TIMEOUT.*?(\d+)', line)
            if timeout_match:
                metadata['timeout_seconds'] = int(timeout_match.group(1))
        
        # Add communication details
        if 'HOST' in line:
            metadata['involves_host_communication'] = True
        
        # Add cash handling details
        if any(cash_term in line for cash_term in ['DISPENSE', 'RETRACT', 'STACKED']):
            metadata['cash_operation'] = True
        
        return metadata
    
    def _is_valid_transition(self, current_phase: TransactionPhase, 
                           next_phase: TransactionPhase) -> bool:
        """Validate transaction phase transitions"""
        valid_transitions = self.phase_transitions.get(current_phase, [])
        return next_phase in valid_transitions
    
    def _add_contextual_anomalies(self, label: EJLogLabel, supervisor_mode: bool, 
                                recovery_active: bool, transaction_active: bool) -> EJLogLabel:
        """Add context-specific anomaly detection"""
        anomalies = []
        
        # Enhanced supervisor mode anomaly detection
        if supervisor_mode and transaction_active:
            anomalies.append('Transaction during supervisor mode - highly suspicious')
            label.severity = Severity.CRITICAL
        
        # Supervisor mode entry timing anomalies
        if label.event_type == EventType.SUPERVISOR_ENTRY:
            if transaction_active:
                anomalies.append('Supervisor mode entered during active transaction - security concern')
                label.severity = Severity.CRITICAL
            elif (self.last_transaction_end_time and label.timestamp and 
                  (label.timestamp - self.last_transaction_end_time).total_seconds() < 30):
                time_gap = (label.timestamp - self.last_transaction_end_time).total_seconds()
                anomalies.append(f'Supervisor mode entered {time_gap:.1f}s after transaction - suspicious timing')
                label.severity = Severity.ERROR
            elif (self.last_transaction_end_time and label.timestamp and 
                  (label.timestamp - self.last_transaction_end_time).total_seconds() < 120):
                time_gap = (label.timestamp - self.last_transaction_end_time).total_seconds()
                anomalies.append(f'Supervisor mode entered {time_gap:.1f}s after transaction - unusual timing')
                label.severity = Severity.WARNING
        
        # Supervisor mode exit timing anomalies  
        if label.event_type == EventType.SUPERVISOR_EXIT:
            if self.supervisor_mode_start_time and label.timestamp:
                duration = (label.timestamp - self.supervisor_mode_start_time).total_seconds()
                if duration < 60:
                    anomalies.append(f'Very short supervisor session ({duration:.1f}s) - suspicious activity')
                    label.severity = Severity.ERROR
                elif duration < 120:
                    anomalies.append(f'Short supervisor session ({duration:.1f}s) - unusual for admin tasks')
                    label.severity = Severity.WARNING
                
                # Add duration metadata for analysis
                label.metadata['supervisor_session_duration'] = duration
                label.metadata['supervisor_session_classification'] = (
                    'VERY_SHORT' if duration < 60 else
                    'SHORT' if duration < 120 else
                    'BRIEF' if duration < 300 else
                    'NORMAL' if duration < 1800 else
                    'EXTENDED'
                )
                
                # Reset supervisor mode start time after analysis
                self.supervisor_mode_start_time = None
        
        # Recovery anomalies
        if recovery_active and label.event_type == EventType.TXN_START:
            anomalies.append('Transaction started during recovery')
            label.severity = Severity.CRITICAL
        
        # Authentication anomalies
        if label.auth_failure_type == 'external_auth_failed':
            anomalies.append('External authentication failure - possible EMV issue')
            label.severity = Severity.ERROR
        
        # Cash handling anomalies
        if label.denomination_data:
            try:
                # Convert denomination values to integers before summing
                total_values = 0
                for key, value in label.denomination_data.items():
                    if isinstance(value, dict):
                        # Handle nested denomination data
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (int, float)):
                                total_values += sub_value
                            elif isinstance(sub_value, str) and sub_value.isdigit():
                                total_values += int(sub_value)
                    elif isinstance(value, (int, float)):
                        total_values += value
                    elif isinstance(value, str) and value.isdigit():
                        total_values += int(value)
                
                if total_values == 0:
                    anomalies.append('Zero notes dispensed - possible jam')
                    label.severity = Severity.ERROR
            except (ValueError, TypeError):
                # Skip anomaly detection if denomination data format is unexpected
                pass
        
        # Note Quality and Deposit Anomalies
        if label.note_categories:
            total_notes = sum(label.note_categories.values())
            cat4_count = label.note_categories.get('CAT4', 0)
            cat5_count = label.note_categories.get('CAT5', 0)
            
            # High rejection rate (CAT4 + CAT5)
            rejection_rate = (cat4_count + cat5_count) / total_notes if total_notes > 0 else 0
            if rejection_rate > 0.3:  # More than 30% rejected
                anomalies.append(f'High note rejection rate ({rejection_rate:.1%}) - possible counterfeit or poor quality notes')
                label.severity = Severity.ERROR
            elif rejection_rate > 0.1:  # More than 10% rejected
                anomalies.append(f'Elevated note rejection rate ({rejection_rate:.1%}) - note quality concerns')
                label.severity = Severity.WARNING
            
            # Specific CAT4 (rejected) note warnings
            if cat4_count > 0:
                anomalies.append(f'{cat4_count} notes rejected (CAT4) - authenticity or readability issues')
                if cat4_count > 2:
                    label.severity = Severity.ERROR
                else:
                    label.severity = Severity.WARNING
        
        # Serial Number Read Failures
        if label.serial_read_failures and label.serial_read_failures > 0:
            if label.serial_read_failures >= 3:
                anomalies.append(f'{label.serial_read_failures} serial number read failures - note condition or scanner issues')
                label.severity = Severity.ERROR
            else:
                anomalies.append(f'{label.serial_read_failures} serial number read failures - note quality concern')
                label.severity = Severity.WARNING
        
        # CIM Input Refusal Analysis
        if label.rejected_reason:
            if 'INVALID MEDIA' in label.rejected_reason.upper():
                anomalies.append('CIM rejected deposit - invalid media detected (possible counterfeit)')
                label.severity = Severity.ERROR
            elif 'DOUBLE FEED' in label.rejected_reason.upper():
                anomalies.append('CIM rejected deposit - double feed detected (mechanical issue)')
                label.severity = Severity.WARNING
            elif 'JAM' in label.rejected_reason.upper():
                anomalies.append('CIM rejected deposit - jam detected (requires maintenance)')
                label.severity = Severity.CRITICAL
            else:
                anomalies.append(f'CIM rejected deposit - {label.rejected_reason}')
                label.severity = Severity.WARNING
        
        # Deposit Transaction Anomalies
        if label.event_type in [EventType.CIM_DEPOSIT_ACTIVATED, EventType.CIM_ITEMS_INSERTED]:
            # Check for unusually small or large deposits
            if label.deposit_amount:
                if label.deposit_amount < 10:
                    anomalies.append(f'Unusually small deposit amount (${label.deposit_amount:.2f}) - possible test transaction')
                    label.severity = Severity.WARNING
                elif label.deposit_amount > 10000:
                    anomalies.append(f'Large deposit amount (${label.deposit_amount:.2f}) - requires additional scrutiny')
                    label.severity = Severity.WARNING
        
        # CIM Status Block Anomalies
        if label.cim_status:
            cim_status = label.cim_status
            
            # High rejection rate anomalies
            if 'rejection_rate' in cim_status:
                rejection_rate = cim_status['rejection_rate']
                if rejection_rate > 0.5:  # More than 50% rejection rate
                    anomalies.append(f'Very high deposit rejection rate ({rejection_rate:.1%}) - possible counterfeit or damaged notes')
                    label.severity = Severity.CRITICAL
                elif rejection_rate > 0.3:  # More than 30% rejection rate
                    anomalies.append(f'High deposit rejection rate ({rejection_rate:.1%}) - note quality issues')
                    label.severity = Severity.ERROR
                elif rejection_rate > 0.15:  # More than 15% rejection rate
                    anomalies.append(f'Elevated deposit rejection rate ({rejection_rate:.1%}) - review note condition')
                    label.severity = Severity.WARNING
            
            # Escrow vs Validation anomalies
            if 'escrow_count' in cim_status and 'validated_count' in cim_status:
                escrow_count = cim_status['escrow_count']
                validated_count = cim_status['validated_count']
                
                # High escrow with low validation (stuck in verification)
                if escrow_count > 0 and validated_count == 0:
                    anomalies.append(f'{escrow_count} notes in escrow but none validated - verification issues')
                    label.severity = Severity.WARNING
                
                # Large denomination deposits
                if 'total_deposit_value' in cim_status:
                    total_value = cim_status['total_deposit_value']
                    if total_value > 50000:  # Large deposit
                        anomalies.append(f'Large deposit transaction ({cim_status["currency"]}{total_value:,}) - enhanced monitoring required')
                        label.severity = Severity.WARNING
                    
                    # Unusual denomination patterns
                    if 'denominations' in cim_status:
                        denoms = cim_status['denominations']
                        # Check for unusual denomination combinations
                        if len(denoms) > 3:  # More than 3 different denominations
                            anomalies.append(f'Mixed denomination deposit ({len(denoms)} different denominations) - verify authenticity')
                            label.severity = Severity.WARNING
            
            # Deposit progress anomalies
            if 'deposit_status' in cim_status:
                deposit_status = cim_status['deposit_status']
                if deposit_status == 'PENDING_VALIDATION':
                    anomalies.append('Deposit pending validation - customer may retry with additional notes')
                    label.severity = Severity.INFO
                elif deposit_status == 'PARTIALLY_VALIDATED':
                    anomalies.append('Partial deposit validation - some notes may have been rejected')
                    label.severity = Severity.WARNING
            
            # Multiple reject cycles (indicated by high total rejects vs current transaction)
            if 'total_rejects' in cim_status and cim_status['total_rejects'] > 5:
                anomalies.append(f'{cim_status["total_rejects"]} total rejections - customer making multiple retry attempts')
                label.severity = Severity.WARNING
            
            # Add CIM status classification to metadata
            label.metadata['cim_deposit_classification'] = self._classify_cim_deposit(cim_status)
        
        if anomalies:
            label.metadata['contextual_anomalies'] = anomalies
        
        return label
    
    def _analyze_transaction_flows(self, labels: List[EJLogLabel]) -> List[EJLogLabel]:
        """Analyze complete transaction flows for anomalies"""
        # Group by transaction ID
        transactions = defaultdict(list)
        for label in labels:
            if label.transaction_id:
                transactions[label.transaction_id].append(label)
        
        # Analyze each transaction
        for txn_id, txn_labels in transactions.items():
            # Check for incomplete transactions
            has_start = any(l.event_type == EventType.TXN_START for l in txn_labels)
            has_end = any(l.event_type == EventType.TXN_END for l in txn_labels)
            
            if has_start and not has_end:
                for label in txn_labels:
                    if 'flow_anomalies' not in label.metadata:
                        label.metadata['flow_anomalies'] = []
                    label.metadata['flow_anomalies'].append('Incomplete transaction')
        
        return labels
    
    def _detect_pattern_anomalies(self, labels: List[EJLogLabel]) -> List[EJLogLabel]:
        """Detect unusual patterns in the log sequence"""
        for i in range(1, len(labels)):
            current_label = labels[i]
            prev_label = labels[i-1]
            
            # Detect rapid repeated events
            if (current_label.event_type == prev_label.event_type and 
                current_label.timestamp and prev_label.timestamp):
                time_diff = (current_label.timestamp - prev_label.timestamp).total_seconds()
                if time_diff < 1.0:  # Same event within 1 second
                    if 'pattern_anomalies' not in current_label.metadata:
                        current_label.metadata['pattern_anomalies'] = []
                    current_label.metadata['pattern_anomalies'].append('Rapid repeated event')
        
        return labels
    
    def _analyze_cash_reports(self, labels: List[EJLogLabel]) -> List[EJLogLabel]:
        """Analyze cash total reports and add comprehensive cash metrics"""
        cash_report_groups = []
        current_group = []
        
        # Group consecutive cash-related labels
        for label in labels:
            if (label.event_type == EventType.CASH_TOTAL_REPORT or 
                (label.denomination_data and any(key in label.denomination_data for key in 
                ['cash_total_header', 'denomination_values', 'dispensed_counts', 'rejected_counts', 'remaining_counts']))):
                current_group.append(label)
            else:
                if current_group:
                    cash_report_groups.append(current_group)
                    current_group = []
        
        # Don't forget the last group
        if current_group:
            cash_report_groups.append(current_group)
        
        # Analyze each complete cash report group
        for group in cash_report_groups:
            if len(group) >= 3:  # At least header + some data
                cash_analysis = self._analyze_cash_total_block(labels, group)
                
                # Add analysis to the last label in the group (summary)
                if cash_analysis:
                    summary_label = group[-1]
                    summary_label.metadata['cash_analysis'] = cash_analysis
                    
                    # Update severity based on cash health
                    health_score = cash_analysis.get('cash_health_score', 1.0)
                    if health_score < 0.5:
                        summary_label.severity = Severity.WARNING
                        summary_label.metadata['cash_health_warning'] = 'Poor cash handling performance detected'
                    elif health_score < 0.3:
                        summary_label.severity = Severity.ERROR
                        summary_label.metadata['cash_health_warning'] = 'Critical cash handling issues detected'
        
        return labels
    
    def _detect_receipt_content(self, line: str, in_receipt_block: bool, receipt_content: List[str]) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect receipt printing based on actual receipt content patterns
        Returns (is_receipt_line, receipt_info)
        """
        line_clean = line.strip()
        receipt_info = {}
        
        # Receipt start patterns - NCB MIDAS header
        if re.match(r'^\s*N\.C\.B\.\s+MIDAS\s*$', line, re.IGNORECASE):
            receipt_info['receipt_type'] = 'NCB_MIDAS'
            receipt_info['bank'] = 'NCB'
            return True, receipt_info
        
        # Branch information
        if re.match(r'^\s*NCB\s+.*\s+BRANCH\s*$', line, re.IGNORECASE):
            branch_match = re.search(r'NCB\s+(.*?)\s+BRANCH', line, re.IGNORECASE)
            if branch_match:
                receipt_info['branch'] = branch_match.group(1).strip()
            return True, receipt_info
        
        # Date and time pattern
        if in_receipt_block and re.match(r'^\s*\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\s*$', line):
            receipt_info['receipt_timestamp'] = line.strip()
            return True, receipt_info
        
        # Machine number
        if in_receipt_block and re.match(r'^\s*MACHINE\s+\d+\s*$', line, re.IGNORECASE):
            machine_match = re.search(r'MACHINE\s+(\d+)', line, re.IGNORECASE)
            if machine_match:
                receipt_info['machine_id'] = machine_match.group(1)
            return True, receipt_info
        
        # Transaction number
        if in_receipt_block and re.match(r'^\s*TRAN NO\s+\d+\s*$', line, re.IGNORECASE):
            tran_match = re.search(r'TRAN NO\s+(\d+)', line, re.IGNORECASE)
            if tran_match:
                receipt_info['transaction_number'] = tran_match.group(1)
            return True, receipt_info
        
        # Authorization code
        if in_receipt_block and re.match(r'^\s*AUTHORIZATION\s+\d+\s*$', line, re.IGNORECASE):
            auth_match = re.search(r'AUTHORIZATION\s+(\d+)', line, re.IGNORECASE)
            if auth_match:
                receipt_info['authorization_code'] = auth_match.group(1)
            return True, receipt_info
        
        # Card number (masked) - handles both *****1234 and ************6595 formats
        if in_receipt_block and re.match(r'^\s*\*+\d{4}\s*$', line):
            receipt_info['masked_card'] = line.strip()
            return True, receipt_info
        
        # Deposit information - number of bills by denomination
        if in_receipt_block and re.search(r'\d+\s*X\s*\$\s*\d+', line):
            bill_match = re.search(r'(\d+)\s*X\s*\$\s*(\d+)', line)
            if bill_match:
                count = int(bill_match.group(1))
                denomination = int(bill_match.group(2))
                if 'deposit_bills' not in receipt_info:
                    receipt_info['deposit_bills'] = {}
                receipt_info['deposit_bills'][f'JMD_{denomination}'] = count
            return True, receipt_info
        
        # Deposit account and value
        if in_receipt_block and re.search(r'DEPOSIT AC.*VALUE', line, re.IGNORECASE):
            receipt_info['deposit_account_line'] = line.strip()
            return True, receipt_info
        
        # Available balance
        if in_receipt_block and re.search(r'AVAILABLE\s+[\d,]+\.\d{2}', line, re.IGNORECASE):
            balance_match = re.search(r'AVAILABLE\s+([\d,]+\.\d{2})', line, re.IGNORECASE)
            if balance_match:
                balance_str = balance_match.group(1).replace(',', '')
                receipt_info['available_balance'] = float(balance_str)
            return True, receipt_info
        
        # Transaction result messages
        if in_receipt_block and any(msg in line.upper() for msg in [
            'UNABLE TO PROCESS', 'TRANSACTION APPROVED', 'DECLINED', 'COMPLETED SUCCESSFULLY'
        ]):
            receipt_info['transaction_result'] = line.strip()
            return True, receipt_info
        
        # Amount information
        if in_receipt_block and re.search(r'\$\d+\.\d{2}', line):
            amount_match = re.search(r'\$(\d+\.\d{2})', line)
            if amount_match:
                receipt_info['amount'] = float(amount_match.group(1))
            return True, receipt_info
        
        # Receipt end patterns - multiple ways receipts can end
        if re.match(r'^\s*THANK YOU\s*$', line, re.IGNORECASE):
            receipt_info['receipt_end'] = True
            receipt_info['receipt_end_type'] = 'THANK_YOU'
            return True, receipt_info
        
        # NCB contact information pattern (alternative receipt ending)
        if in_receipt_block and re.search(r'CENTRE\s+1-888-622-3477', line, re.IGNORECASE):
            receipt_info['receipt_end'] = True
            receipt_info['receipt_end_type'] = 'CONTACT_INFO'
            receipt_info['contact_number'] = '1-888-622-3477'
            return True, receipt_info
        
        # General contact/care centre patterns
        if in_receipt_block and re.search(r'CONTACT.*CARE.*CENTRE', line, re.IGNORECASE):
            receipt_info['receipt_end'] = True
            receipt_info['receipt_end_type'] = 'CARE_CENTRE'
            return True, receipt_info
        
        # Continue receipt block for other content if we're already in one
        if in_receipt_block and line_clean:
            # Generic receipt content
            return True, {'content_line': line_clean}
        
        return False, {}
    
    def _create_receipt_label(self, end_line_num: int, start_line_num: int, 
                            receipt_content: List[str], receipt_info: Dict[str, Any]) -> EJLogLabel:
        """Create a comprehensive receipt label"""
        
        # Parse receipt content for key information
        parsed_info = self._parse_receipt_content(receipt_content)
        
        # Extract timestamp from receipt if available
        receipt_timestamp = None
        for line in receipt_content:
            if re.match(r'^\s*\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\s*$', line):
                try:
                    receipt_timestamp = datetime.strptime(line.strip(), '%Y/%m/%d %H:%M:%S')
                except ValueError:
                    pass
                break
        
        # Determine transaction result and severity
        severity = Severity.INFO
        transaction_result = parsed_info.get('transaction_result', '')
        
        if 'UNABLE TO PROCESS' in transaction_result.upper():
            severity = Severity.WARNING
        elif 'DECLINED' in transaction_result.upper():
            severity = Severity.WARNING
        elif 'ERROR' in transaction_result.upper():
            severity = Severity.ERROR
        
        # Create enhanced metadata
        metadata = {
            'receipt_content': receipt_content,
            'receipt_line_count': len(receipt_content),
            'receipt_start_line': start_line_num,
            'receipt_end_line': end_line_num,
            **parsed_info
        }
        
        # Determine if this indicates a failed transaction
        failed_transaction = any(fail_indicator in transaction_result.upper() for fail_indicator in [
            'UNABLE TO PROCESS', 'DECLINED', 'FAILED', 'ERROR'
        ])
        
        if failed_transaction:
            metadata['transaction_failed'] = True
            metadata['failure_reason'] = transaction_result
        
        return EJLogLabel(
            line_number=end_line_num,
            timestamp=receipt_timestamp,
            phase=TransactionPhase.RECEIPT_PRINTING,
            event_type=EventType.RECEIPT_PRINT,
            severity=severity,
            error_category=ErrorCategory.SOFTWARE if failed_transaction else None,
            entity='receipt_printer',
            amount=parsed_info.get('amount'),
            metadata=metadata,
            operational_mode=OperationalMode.NORMAL,
            transaction_id=parsed_info.get('transaction_number'),
            customer_present=True,
            confidence_score=0.95  # High confidence for receipt detection
        )
    
    def _parse_receipt_content(self, receipt_content: List[str]) -> Dict[str, Any]:
        """Parse receipt content for structured information"""
        parsed = {}
        
        for line in receipt_content:
            line_upper = line.upper()
            line_clean = line.strip()
            
            # Extract bank and branch
            if 'N.C.B. MIDAS' in line_upper:
                parsed['bank'] = 'NCB'
                parsed['system'] = 'MIDAS'
            elif 'NCB' in line_upper and 'BRANCH' in line_upper:
                branch_match = re.search(r'NCB\s+(.*?)\s+BRANCH', line, re.IGNORECASE)
                if branch_match:
                    parsed['branch'] = branch_match.group(1).strip()
            
            # Extract machine ID
            elif line_upper.startswith('MACHINE'):
                machine_match = re.search(r'MACHINE\s+(\d+)', line)
                if machine_match:
                    parsed['machine_id'] = machine_match.group(1)
            
            # Extract transaction number
            elif line_upper.startswith('TRAN NO'):
                tran_match = re.search(r'TRAN NO\s+(\d+)', line)
                if tran_match:
                    parsed['transaction_number'] = tran_match.group(1)
            
            # Extract authorization code
            elif line_upper.startswith('AUTHORIZATION'):
                auth_match = re.search(r'AUTHORIZATION\s+(\d+)', line)
                if auth_match:
                    parsed['authorization_code'] = auth_match.group(1)
            
            # Extract masked card number
            elif re.match(r'^\*+\d{4}$', line_clean):
                parsed['masked_card'] = line_clean
            
            # Extract deposit bill counts (e.g., "02  X $ 5000")
            elif re.search(r'\d+\s*X\s*\$\s*\d+', line_clean):
                bill_match = re.search(r'(\d+)\s*X\s*\$\s*(\d+)', line_clean)
                if bill_match:
                    count = int(bill_match.group(1))
                    denomination = int(bill_match.group(2))
                    if 'deposit_bills' not in parsed:
                        parsed['deposit_bills'] = {}
                    parsed['deposit_bills'][f'JMD_{denomination}'] = count
            
            # Extract deposit value (look for standalone amounts after VALUE JMD)
            elif re.search(r'VALUE JMD', line_upper):
                parsed['deposit_section'] = True  # Mark that we're in deposit section
            elif ('deposit_section' in parsed and 
                  re.match(r'^\s*([\d,]+\.\d{2})\s*$', line_clean)):
                value_str = line_clean.replace(',', '')
                try:
                    parsed['deposit_value'] = float(value_str)
                except ValueError:
                    pass
            
            # Extract available balance
            elif line_upper.startswith('AVAILABLE'):
                balance_match = re.search(r'AVAILABLE\s+([\d,]+\.\d{2})', line)
                if balance_match:
                    balance_str = balance_match.group(1).replace(',', '')
                    parsed['available_balance'] = float(balance_str)
            
            # Extract standard amount (for non-deposit transactions)
            elif '$' in line and 'deposit_bills' not in parsed:
                amount_match = re.search(r'\$(\d+\.\d{2})', line)
                if amount_match:
                    parsed['amount'] = float(amount_match.group(1))
            
            # Extract transaction result
            elif any(result in line_upper for result in [
                'UNABLE TO PROCESS', 'TRANSACTION APPROVED', 'DECLINED', 
                'COMPLETED SUCCESSFULLY', 'CANCELLED'
            ]):
                parsed['transaction_result'] = line_clean
            
            # Extract contact information
            elif re.search(r'1-888-622-3477', line):
                parsed['contact_number'] = '1-888-622-3477'
                parsed['support_available'] = True
        
        # Calculate total deposit amount from bill breakdown if not found directly
        if 'deposit_bills' in parsed and 'deposit_value' not in parsed:
            total_value = 0
            for denom_key, count in parsed['deposit_bills'].items():
                if count > 0:  # Only count denominations with actual bills
                    denom_value = int(denom_key.split('_')[1])  # Extract value from JMD_5000
                    total_value += denom_value * count
            if total_value > 0:
                parsed['calculated_deposit_value'] = total_value
        
        # Determine receipt type
        if 'deposit_bills' in parsed or 'deposit_value' in parsed:
            parsed['receipt_type'] = 'CASH_DEPOSIT'
            parsed['transaction_category'] = 'DEPOSIT'
        elif 'transaction_result' in parsed and 'UNABLE TO PROCESS' in parsed['transaction_result'].upper():
            parsed['receipt_type'] = 'FAILED_TRANSACTION'
            parsed['transaction_category'] = 'FAILED'
        else:
            parsed['receipt_type'] = 'STANDARD_TRANSACTION'
            parsed['transaction_category'] = 'STANDARD'
        
        return parsed
    
    def _analyze_cash_total_block(self, labels: List[EJLogLabel], cash_data_labels: List[EJLogLabel]) -> Dict[str, Any]:
        """Analyze complete cash total report block for comprehensive cash metrics"""
        if len(cash_data_labels) < 4:  # Need header, denomination, dispensed, rejected, remaining
            return {}
        
        cash_analysis = {
            'denominations': {},
            'dispensed_totals': {},
            'rejected_totals': {},
            'remaining_totals': {},
            'rejection_rates': {},
            'utilization_rates': {},
            'cash_health_score': 0.0
        }
        
        # Extract data from each cash report label
        for label in cash_data_labels:
            if not label.denomination_data:
                continue
                
            if 'denomination_values' in label.denomination_data:
                cash_analysis['denominations'] = label.denomination_data['denomination_values']
            elif 'dispensed_counts' in label.denomination_data:
                cash_analysis['dispensed_totals'] = label.denomination_data['dispensed_counts']
            elif 'rejected_counts' in label.denomination_data:
                cash_analysis['rejected_totals'] = label.denomination_data['rejected_counts']
            elif 'remaining_counts' in label.denomination_data:
                cash_analysis['remaining_totals'] = label.denomination_data['remaining_counts']
        
        # Calculate derived metrics if we have complete data
        if all(key in cash_analysis for key in ['dispensed_totals', 'rejected_totals', 'remaining_totals']):
            
            # Calculate rejection rates
            for type_key in ['type_1', 'type_2', 'type_3', 'type_4']:
                dispensed = cash_analysis['dispensed_totals'].get(type_key, 0)
                rejected = cash_analysis['rejected_totals'].get(type_key, 0)
                
                if dispensed + rejected > 0:
                    rejection_rate = rejected / (dispensed + rejected)
                    cash_analysis['rejection_rates'][type_key] = rejection_rate
                else:
                    cash_analysis['rejection_rates'][type_key] = 0.0
            
            # Calculate utilization rates (how much of original stock has been used)
            for type_key in ['type_1', 'type_2', 'type_3', 'type_4']:
                dispensed = cash_analysis['dispensed_totals'].get(type_key, 0)
                rejected = cash_analysis['rejected_totals'].get(type_key, 0)
                remaining = cash_analysis['remaining_totals'].get(type_key, 0)
                
                original_count = dispensed + rejected + remaining
                if original_count > 0:
                    utilization_rate = (dispensed + rejected) / original_count
                    cash_analysis['utilization_rates'][type_key] = utilization_rate
                else:
                    cash_analysis['utilization_rates'][type_key] = 0.0
            
            # Calculate overall cash health score (1.0 = perfect, 0.0 = poor)
            avg_rejection_rate = sum(cash_analysis['rejection_rates'].values()) / 4
            avg_utilization_rate = sum(cash_analysis['utilization_rates'].values()) / 4
            
            # Good health = low rejection rate, moderate utilization (not too high = low stock)
            health_score = (1.0 - avg_rejection_rate) * 0.7 + (1.0 - min(avg_utilization_rate, 0.8)) * 0.3
            cash_analysis['cash_health_score'] = max(0.0, min(1.0, health_score))
            
            # Add operational insights
            cash_analysis['insights'] = []
            
            # Check for high rejection rates
            high_rejection_types = [k for k, v in cash_analysis['rejection_rates'].items() if v > 0.05]
            if high_rejection_types:
                cash_analysis['insights'].append(f"High rejection rates detected for: {', '.join(high_rejection_types)}")
            
            # Check for low stock warnings
            low_stock_types = [k for k, v in cash_analysis['utilization_rates'].items() if v > 0.8]
            if low_stock_types:
                cash_analysis['insights'].append(f"Low stock warning for: {', '.join(low_stock_types)}")
            
            # Check for uneven utilization
            utilization_values = list(cash_analysis['utilization_rates'].values())
            if max(utilization_values) - min(utilization_values) > 0.3:
                cash_analysis['insights'].append("Uneven cassette utilization detected")
        
        return cash_analysis
    
    def _analyze_cash_dispensing_timing(self, labels: List[EJLogLabel]) -> List[EJLogLabel]:
        """Analyze timing between cash dispensing sequence events"""
        cash_sequences = []
        current_sequence = {}
        
        # Find cash dispensing sequences
        for label in labels:
            if label.event_type == EventType.NOTES_STACKED and label.timestamp:
                current_sequence = {
                    'stacked': {'label': label, 'timestamp': label.timestamp},
                    'presented': None,
                    'taken': None
                }
            elif label.event_type == EventType.NOTES_PRESENT and label.timestamp and current_sequence:
                current_sequence['presented'] = {'label': label, 'timestamp': label.timestamp}
            elif label.event_type == EventType.NOTES_TAKEN and label.timestamp and current_sequence:
                current_sequence['taken'] = {'label': label, 'timestamp': label.timestamp}
                
                # Complete sequence found, analyze timing
                if all(key in current_sequence and current_sequence[key] for key in ['stacked', 'presented', 'taken']):
                    timing_analysis = self._calculate_dispensing_timing(current_sequence)
                    
                    # Add timing metadata to all events in the sequence
                    for event_key, event_data in current_sequence.items():
                        if event_data and event_data['label']:
                            event_data['label'].metadata['cash_timing_analysis'] = timing_analysis
                            event_data['label'].metadata['cash_sequence_role'] = event_key
                    
                    cash_sequences.append(current_sequence)
                
                # Reset for next sequence
                current_sequence = {}
        
        return labels
    
    def _calculate_dispensing_timing(self, sequence: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate timing metrics for cash dispensing sequence"""
        stacked_time = sequence['stacked']['timestamp']
        presented_time = sequence['presented']['timestamp']
        taken_time = sequence['taken']['timestamp']
        
        # Calculate time differences in seconds
        stacked_to_presented = (presented_time - stacked_time).total_seconds()
        presented_to_taken = (taken_time - presented_time).total_seconds()
        total_dispensing_time = (taken_time - stacked_time).total_seconds()
        
        # Determine performance classification
        performance_metrics = {
            'stacked_to_presented_seconds': stacked_to_presented,
            'presented_to_taken_seconds': presented_to_taken,
            'total_dispensing_seconds': total_dispensing_time,
            'timestamps': {
                'stacked': stacked_time.strftime('%H:%M:%S'),
                'presented': presented_time.strftime('%H:%M:%S'),
                'taken': taken_time.strftime('%H:%M:%S')
            }
        }
        
        # Performance analysis
        performance_insights = []
        
        # Analyze stacking to presentation time (should be quick, <3 seconds typical)
        if stacked_to_presented > 5.0:
            performance_insights.append('Slow cash presentation - possible mechanical issue')
            performance_metrics['presentation_performance'] = 'SLOW'
        elif stacked_to_presented > 3.0:
            performance_insights.append('Moderate delay in cash presentation')
            performance_metrics['presentation_performance'] = 'MODERATE'
        else:
            performance_metrics['presentation_performance'] = 'FAST'
        
        # Analyze presentation to taken time (customer response time)
        if presented_to_taken > 30.0:
            performance_insights.append('Customer took long time to collect cash')
            performance_metrics['customer_response'] = 'SLOW'
        elif presented_to_taken > 15.0:
            performance_metrics['customer_response'] = 'MODERATE'
        else:
            performance_metrics['customer_response'] = 'FAST'
        
        # Overall dispensing efficiency
        if total_dispensing_time > 45.0:
            performance_insights.append('Long total dispensing time')
            performance_metrics['overall_efficiency'] = 'POOR'
        elif total_dispensing_time > 20.0:
            performance_metrics['overall_efficiency'] = 'MODERATE'
        else:
            performance_metrics['overall_efficiency'] = 'EXCELLENT'
        
        # Detect potential issues
        if stacked_to_presented > 10.0:
            performance_insights.append('Critical delay - possible jam during presentation')
        
        if presented_to_taken < 2.0:
            performance_insights.append('Very fast customer response - possible automated transaction')
        
        performance_metrics['insights'] = performance_insights
        performance_metrics['sequence_health'] = 'HEALTHY' if len(performance_insights) == 0 else 'ISSUES_DETECTED'
        
        return performance_metrics
    
    def _extract_transaction_id(self, line: str) -> Optional[str]:
        """Extract transaction ID from line"""
        # Look for transaction ID patterns
        patterns = [
            r'TXN[:\s]+(\w+)',
            r'TRANSACTION[:\s]+(\w+)',
            r'ID[:\s]+(\w+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
