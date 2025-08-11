# Cassette Counter Integration for Cash Forecasting

## Overview
This document describes the implementation of cassette counter parsing and storage functionality for ABM EJ log sessions. This feature extracts cash withdrawal information from each transaction session and stores it in a dedicated database table for cash forecasting purposes.

## Implementation Summary

### 1. Database Schema
**New Table**: `cassette_counters`

```sql
CREATE TABLE cassette_counters (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,
    terminal_id VARCHAR(50),
    transaction_datetime TIMESTAMP NOT NULL,
    
    -- Cassette remaining counts (after withdrawal)
    cassette_1_remaining INTEGER,
    cassette_2_remaining INTEGER,
    cassette_3_remaining INTEGER,
    cassette_4_remaining INTEGER,
    
    -- Cassette denominations (note values)
    cassette_1_denomination INTEGER,
    cassette_2_denomination INTEGER,
    cassette_3_denomination INTEGER,
    cassette_4_denomination INTEGER,
    
    -- Dispensed/rejected amounts for this transaction
    cassette_1_dispensed INTEGER DEFAULT 0,
    cassette_2_dispensed INTEGER DEFAULT 0,
    cassette_3_dispensed INTEGER DEFAULT 0,
    cassette_4_dispensed INTEGER DEFAULT 0,
    
    cassette_1_rejected INTEGER DEFAULT 0,
    cassette_2_rejected INTEGER DEFAULT 0,
    cassette_3_rejected INTEGER DEFAULT 0,
    cassette_4_rejected INTEGER DEFAULT 0,
    
    -- Total transaction amounts
    total_dispensed_amount INTEGER,
    total_rejected_amount INTEGER,
    withdrawal_successful BOOLEAN DEFAULT TRUE,
    
    -- Metadata
    source_file VARCHAR(255),
    raw_cassette_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (session_id) REFERENCES ml_sessions(session_id)
);
```

### 2. Database Views for Analytics

#### `cassette_forecasting_view`
Provides comprehensive cash forecasting data including:
- Terminal cash levels per transaction
- Total cash remaining calculations
- Withdrawal amounts and patterns
- Anomaly correlation data

#### `terminal_cash_status`
Provides current cash status summary per terminal:
- Latest cassette levels
- Total transactions
- Current total cash available
- Last transaction time

### 3. Code Implementation

#### New Method: `parse_cassette_counters()` in ml_analyzer.py
```python
def parse_cassette_counters(self, session: TransactionSession) -> Optional[Dict[str, Any]]:
    """
    Parse cassette counter information from EJ session for cash forecasting.
    
    Returns cassette counter data if the session contains a successful withdrawal,
    None otherwise.
    """
```

**Key Features:**
- Only processes sessions with "NOTES PRESENTED" (successful withdrawals)
- Extracts machine number, transaction datetime, and cassette data
- Validates data integrity (exactly 4 cassettes required)
- Calculates total dispensed/rejected amounts
- Returns structured dictionary for database storage

#### Enhanced Storage Methods in main.py
- Updated `store_sessions()` to include cassette counter extraction and storage
- Updated `store_production_sessions()` to handle production mode cassette data
- New `store_cassette_counters()` method for dedicated cassette data storage

### 4. Data Extraction Pattern

The system looks for this pattern in EJ session logs:
```
MACHINE 416
DATE TIME 2025/01/15 14:30:25
DENOMINATION    20    50   100    20
DISPENSED        2     1     0     3
REJECTED         0     0     0     0
REMAINING      498   799   300   597
```

**Parsing Logic:**
- **MACHINE**: Terminal/machine identifier (mapped to `terminal_id`)
- **DATE TIME**: Transaction timestamp
- **DENOMINATION**: Note values for each cassette ($20, $50, $100, $20)
- **DISPENSED**: Number of notes dispensed from each cassette
- **REJECTED**: Number of notes rejected from each cassette  
- **REMAINING**: Number of notes remaining in each cassette after transaction

**Note**: The `MACHINE` field in the EJ logs represents the same value as `terminal_id` extracted from the filename. Both identify the ABM terminal number.

### 5. Cash Forecasting Benefits

#### Real-time Cash Monitoring
- Track cash levels per terminal in real-time
- Monitor dispensing patterns and rates
- Identify low cash situations before they occur

#### Historical Analysis
- Analyze withdrawal patterns by time, terminal, amount
- Predict cash depletion based on historical usage
- Optimize cash replenishment schedules

#### Integration with Anomaly Detection
- Correlate cash handling anomalies with technical issues
- Track relationships between low cash and transaction failures
- Monitor excessive dispensing or unusual patterns

### 6. Testing and Validation

**Test File**: `test_cassette_py27.py`
- Validates regex patterns for cassette data extraction
- Tests successful withdrawal session parsing
- Verifies data integrity and calculations
- Confirms non-withdrawal sessions are skipped

**Test Results:**
```
✓ Successfully extracts machine number (416)
✓ Correctly parses transaction datetime
✓ Extracts 4 cassette denominations: [20, 50, 100, 20]
✓ Parses dispensed amounts: [2, 1, 0, 3]
✓ Calculates total dispensed: $150
✓ Computes remaining cash: $91,850
```

### 7. Usage Examples

#### Query Current Terminal Cash Status
```sql
SELECT * FROM terminal_cash_status WHERE terminal_id = '416';
```

#### Get Cash Forecasting Data for Date Range
```sql
SELECT * FROM cassette_forecasting_view 
WHERE terminal_id = '416' 
AND transaction_datetime BETWEEN '2025-01-01' AND '2025-01-31'
ORDER BY transaction_datetime;
```

#### Monitor Low Cash Terminals
```sql
SELECT terminal_id, current_total_cash, last_transaction_time
FROM terminal_cash_status 
WHERE current_total_cash < 10000
ORDER BY current_total_cash;
```

### 8. Integration Points

#### With Existing Session Processing
- Cassette data extraction occurs during normal session processing
- No impact on existing anomaly detection functionality
- Data stored alongside session data with foreign key relationship

#### With Dashboard Systems
- New cassette counter data available via existing database connections
- Real-time updates through Redis publishing (if needed)
- Compatible with existing reporting infrastructure

### 9. Files Modified/Created

#### Database Migrations
- `005_add_cassette_counters_table.sql` - Complete database schema

#### Core Implementation
- `ml_analyzer.py` - Added `parse_cassette_counters()` method
- `main.py` - Updated session storage methods, added cassette storage

#### Testing
- `test_cassette_py27.py` - Validation test script

### 10. Next Steps

1. **Deploy Database Migration**: Run `005_add_cassette_counters_table.sql`
2. **Test with Real EJ Data**: Validate parsing with actual EJ log files
3. **Dashboard Integration**: Add cash forecasting views to existing dashboards
4. **Alerting Setup**: Configure alerts for low cash situations
5. **Analytics Development**: Create cash forecasting models and reports

## Conclusion

The cassette counter integration provides comprehensive cash tracking capabilities while maintaining compatibility with existing anomaly detection systems. The implementation follows established patterns and includes robust error handling, data validation, and testing coverage.

This enhancement enables proactive cash management, reduces ATM downtime due to cash depletion, and provides valuable insights into terminal usage patterns.
