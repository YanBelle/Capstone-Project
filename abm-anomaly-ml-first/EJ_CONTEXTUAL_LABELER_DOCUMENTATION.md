# EJ Contextual Labeler - Complete Documentation

## Overview
The EJ Contextual Labeler is a sophisticated financial domain intelligence system specifically designed for NCB MIDAS ATM operations. It provides comprehensive labeling, anomaly detection, and contextual analysis of Electronic Journal (EJ) logs.

## 🎯 Core Capabilities

### 1. Event Type Recognition (35 Types)
The labeler recognizes 35 distinct event types across ATM operations:

#### **Transaction Lifecycle Events**
- `transaction_started` - Customer initiates transaction
- `card_inserted` - Card insertion detected
- `card_removed` - Card removal detected
- `pin_entered` - PIN verification
- `transaction_completed` - Transaction finished
- `customer_timeout` - Customer inactivity timeout

#### **ATM Operational Events**
- `atm_in_service` - ATM becomes operational
- `card_reader_activated` - Card reader initialization
- `device_activated` - General device activation
- `device_deactivated` - General device deactivation

#### **Cash Handling Events**
- `cash_dispensed` - Notes dispensed to customer
- `notes_presented` - Notes ready for collection
- `notes_not_taken` - Customer didn't collect notes
- `notes_retracted` - Notes pulled back into dispenser
- `notes_stacked` - Notes stored in cash cassette
- `cash_count` - Cash counting operation
- `dispenser_error` - Cash dispenser malfunction
- `notes_presented_timeout` - Notes collection timeout
- `notes_removed_by_customer` - Customer collected notes
- `cash_validation` - Cash validation process

#### **CIM Deposit Operations** (Enhanced Financial Intelligence)
- `cim_deposit_activated` - CIM deposit session begins
- `cim_shutter_opened` - Deposit slot opens
- `cim_items_inserted` - Customer inserts cash
- `cim_shutter_closed` - Deposit slot closes
- `cim_items_validated` - Cash validation complete
- `cim_operation` - General CIM operation status
- `cim_input_refused` - Deposit rejected
- `cim_deposit_completed` - Deposit finalized

#### **Recovery & Maintenance Events**
- `supervisor_entry` - Supervisor mode activated
- `supervisor_exit` - Supervisor mode deactivated
- `init_bna_started` - Bill note acceptor initialization
- `cim_reset_called` - CIM reset operation
- `cheque_recovery` - Check handling recovery
- `cashin_retract_started` - Cash retraction begun
- `device_init` - Device initialization
- `communication_reset` - Communication restart
- `cash_deposit_retract` - Cash deposit failed, money retained ⭐

#### **Note Quality Analysis Events**
- `failed_serial_read` - Serial number read failure
- `note_categorization` - Note fitness assessment
- `validation_failure` - Note validation failed

#### **Authentication Events**
- `authentication_failure` - Login/access failure

#### **Receipt & Communication Events**
- `receipt_printed` - Receipt successfully printed
- `ncb_midas_header` - Receipt header detected
- `receipt_content` - Receipt body content

### 2. Transaction Phases (14 Phases)
Complete transaction lifecycle with proper flow validation:

1. **initialization** - System startup/ready state
2. **card_authentication** - Card validation
3. **pin_verification** - PIN entry and validation
4. **account_selection** - Account type selection
5. **transaction_selection** - Transaction type selection
6. **amount_entry** - Amount specification
7. **processing** - Transaction processing
8. **cash_dispensing** - Withdrawal operations
9. **cash_depositing** - Deposit operations ⭐
10. **note_quality_check** - Note fitness analysis ⭐
11. **deposit_verification** - Deposit confirmation ⭐
12. **receipt_printing** - Receipt generation
13. **error_handling** - Error recovery
14. **completion** - Transaction finalization

### 3. Operational Modes (6 Modes)
System operational state awareness:

- **normal** - Standard customer operation
- **supervisor** - Administrative/maintenance mode
- **recovery** - Error recovery mode
- **maintenance** - Scheduled maintenance
- **out_of_service** - System unavailable
- **in_service_waiting** - Ready for customer

### 4. Recovery Types (9 Types)
Enhanced recovery operation classification:

- **bna_init** - Bill note acceptor initialization
- **cim_reset** - CIM system reset
- **cheque_recovery** - Check handling recovery
- **cashin_retract** - General cash retraction
- **device_init** - Device initialization
- **communication_reset** - Communication restart
- **retract_bin_init** - Retract bin initialization ⭐
- **retract_bin_cashin** - Retract bin cash handling ⭐
- **retract_bin_cim_reset** - Retract bin CIM reset ⭐
- **cash_deposit_retract** - Cash deposit failed, money retained by ATM ⭐

### 5. Error Categories (8 Categories)
Comprehensive error classification:

- **hardware** - Physical device failures
- **software** - Software exceptions
- **network** - Communication failures
- **security** - Authentication/access issues
- **cash_handling** - Cash mechanism problems
- **card_reader** - Card reading issues
- **receipt_printer** - Printing problems
- **communication** - General communication errors

### 6. Severity Levels (4 Levels)
Risk assessment classification:

- **info** - Informational events
- **warning** - Potential issues
- **error** - Operational problems
- **critical** - System failures

## 🎛️ Advanced Features

### CIM Status Block Parsing ⭐
Comprehensive financial transaction analysis:

```python
# Parses CIM status blocks like:
# "A/C OPERATION OK ESC: 2 VAL: 0 REF: 0 REJECTS: 1 JMD$5000: 2"

cim_status = {
    'escrow_count': 2,           # Notes in escrow
    'validated_count': 0,        # Validated notes
    'refused_count': 0,          # Refused notes
    'total_rejects': 1,          # Total rejections
    'denominations': {           # Denomination breakdown
        'JMD_5000': 2
    },
    'total_deposit_value': 10000,  # Total value
    'currency': 'JMD',            # Currency type
    'validation_rate': 0.0,       # Success rate
    'rejection_rate': 0.33,       # Rejection rate
    'deposit_status': 'PENDING_VALIDATION'
}
```

### Note Quality Analysis ⭐
CAT1-CAT5 categorization and serial number analysis:

```python
# Analyzes patterns like:
# "FAILED SERIAL NUMBER READS and CAT4 NOTES: 1"

note_analysis = {
    'CAT1': 0,  # Fit notes
    'CAT2': 0,  # Good notes  
    'CAT3': 0,  # Fair notes
    'CAT4': 1,  # Poor notes
    'CAT5': 0,  # Unfit notes
    'serial_read_failures': 1
}
```

### Receipt Intelligence ⭐
Multi-format receipt detection and parsing:

- **NCB MIDAS Headers**: `N.C.B. MIDAS` detection
- **Multiple Endings**: Support for "THANK YOU" and contact center information
- **Deposit Breakdowns**: Bill denomination analysis
- **Authorization Codes**: Transaction reference extraction

### Anomaly Detection ⭐
15+ contextual anomaly types:

1. **Financial Anomalies**
   - High rejection rates (>30% critical, >10% warning)
   - Unusual deposit amounts
   - Currency mismatches
   - Cash deposit retracts (customer money retained) ⭐

2. **Operational Anomalies**
   - Invalid phase transitions
   - Supervisor mode timing issues
   - Recovery operation patterns

3. **Quality Anomalies**
   - High CAT4/CAT5 note concentrations
   - Excessive serial read failures
   - Note fitness degradation

## 📊 Pattern Recognition

### 45+ Active Patterns
The labeler uses 45+ regex patterns for:

- **7 CIM Deposit Operations** - Complete deposit flow
- **10 Cash Operations** - Withdrawal and handling
- **8 Recovery Operations** - System recovery (including retract bins) ⭐
- **6 Transaction Lifecycle** - Customer journey
- **3 Note Quality Analysis** - Note assessment
- **3 Receipt Printing** - Receipt handling
- **3 Error Patterns** - Error detection
- **3 Cash Retract Patterns** - Deposit failure detection ⭐
- **2 Supervisor Mode** - Administrative access
- **2 ATM Operations** - System state
- **1 Authentication** - Security events

### Error Code Recognition
10 predefined error codes with severity mapping:

| Code | Description | Severity | Category |
|------|-------------|----------|----------|
| M-38 | External authentication failure | ERROR | security |
| M-01 | Device communication error | CRITICAL | communication |
| M-15 | Cash dispenser error | ERROR | cash_handling |
| M-23 | Card reader error | ERROR | card_reader |
| M-45 | Receipt printer error | WARNING | receipt_printer |
| M-67 | Network communication failure | CRITICAL | network |
| E-01 | Hardware malfunction | CRITICAL | hardware |
| E-12 | Software exception | ERROR | software |
| W-05 | Low cash warning | WARNING | cash_handling |
| W-18 | Maintenance required | WARNING | hardware |
| **RETRACT** | **Cash deposit retract failure** | **CRITICAL** | **cash_handling** ⭐ |

## 🏗️ Data Structure

### EJLogLabel Fields
Complete label structure with 16 fields:

#### Core Fields
- `line_number`: int - Log line reference
- `timestamp`: Optional[datetime] - Extracted timestamp
- `phase`: TransactionPhase - Current transaction phase
- `event_type`: EventType - Classified event type
- `severity`: Severity - Risk level assessment
- `error_category`: Optional[ErrorCategory] - Error classification
- `error_code`: Optional[str] - Specific error code
- `entity`: Optional[str] - Related entity/component
- `amount`: Optional[float] - Financial amounts
- `metadata`: Dict[str, Any] - Additional context

#### Contextual Intelligence Fields
- `operational_mode`: OperationalMode - System operation state
- `recovery_type`: Optional[RecoveryType] - Recovery classification
- `denomination_data`: Optional[Dict[str, int]] - Currency breakdown
- `auth_failure_type`: Optional[str] - Authentication failure type
- `transaction_id`: Optional[str] - Transaction correlation
- `customer_present`: bool - Customer presence inference
- `confidence_score`: float - Labeling confidence (0.0-1.0)

#### Financial Analysis Fields ⭐
- `note_categories`: Optional[Dict[str, int]] - CAT1-CAT5 breakdown
- `serial_read_failures`: Optional[int] - Serial number failures
- `deposit_amount`: Optional[float] - Deposit transaction amounts
- `rejected_reason`: Optional[str] - Rejection reasoning
- `cim_status`: Optional[Dict[str, Any]] - Complete CIM status

## 🔄 Phase Transition Validation

The labeler enforces proper transaction flow with 36 transition rules:

```
initialization → card_authentication
card_authentication → pin_verification, error_handling
pin_verification → account_selection, error_handling
account_selection → transaction_selection, error_handling
transaction_selection → amount_entry, processing, cash_depositing, error_handling
amount_entry → processing, error_handling
processing → cash_dispensing, cash_depositing, receipt_printing, error_handling
cash_dispensing → receipt_printing, completion, error_handling
cash_depositing → note_quality_check, deposit_verification, receipt_printing, error_handling
note_quality_check → deposit_verification, cash_depositing, error_handling
deposit_verification → receipt_printing, completion, cash_depositing, error_handling
receipt_printing → completion, error_handling
error_handling → completion, initialization
completion → initialization
```

## 🎯 Use Cases

### 1. Real-time Anomaly Detection
- Monitor deposit rejection rates
- Detect supervisor mode anomalies
- Track note quality degradation
- Identify authentication failures
- Detect cash deposit retract failures ⭐

### 2. Financial Transaction Analysis
- CIM deposit flow tracking
- Note categorization monitoring
- Cash reconciliation
- Denomination analysis

### 3. Operational Intelligence
- Device performance monitoring
- Recovery operation tracking
- Customer experience analysis
- System availability assessment

### 4. Compliance & Reporting
- Transaction audit trails
- Error classification
- Performance metrics
- Quality assurance

## 🚀 Performance Characteristics

- **Pattern Matching**: O(n) linear time complexity
- **Memory Usage**: Minimal state retention
- **Accuracy**: High confidence scoring with contextual validation
- **Coverage**: 35 event types across complete ATM operation spectrum
- **Extensibility**: Modular design for easy pattern addition

## 📈 Success Metrics

The EJ Contextual Labeler provides:

✅ **Complete transaction lifecycle tracking** (14 phases)  
✅ **Comprehensive CIM deposit operation support** (8 specific events)  
✅ **Advanced supervisor mode anomaly detection**  
✅ **CIM status block parsing with financial metrics**  
✅ **Multi-pattern receipt recognition** (NCB MIDAS + contact endings)  
✅ **Note quality analysis** (CAT1-CAT5 + serial failures)  
✅ **Enhanced recovery operation classification** (9 types)  
✅ **Contextual anomaly detection** across multiple dimensions  
✅ **Cash reconciliation and denomination tracking**  
✅ **Authentication failure analysis** with context  
✅ **Operational mode awareness** (6 modes)  
✅ **Error categorization** with severity mapping  
✅ **Phase transition validation** with flow rules  
✅ **Customer presence inference**  
✅ **Confidence scoring** for labeling accuracy

---

**The EJ Contextual Labeler represents a comprehensive financial domain intelligence system with deep understanding of NCB MIDAS operations, CIM deposits, and multi-dimensional anomaly detection.**
