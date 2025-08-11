# ABM Terminal ID Integration

## Overview

The ABM anomaly detection system now automatically extracts and stores the terminal ID for each session during the sessionisation process. This enhancement allows for better tracking and analysis of anomalies by specific ATM terminals.

## Terminal ID Extraction

### Filename Format
The system expects EJ log files to follow this naming convention:
```
ABM{terminal_id}EJ_{start_date}_{end_date}.txt
```

### Examples
- `ABM416EJ_20250101_20250630.txt` → Terminal ID: `416`
- `ABM175EJ_20250624_20250624.txt` → Terminal ID: `175`
- `ABM001EJ_20241201_20241231.txt` → Terminal ID: `001`

### Extraction Process
1. **During sessionisation** in `split_into_sessions()` method:
   - Extract terminal ID using regex: `r'ABM(\d+)EJ_(\d{8})_(\d{8})'`
   - Log successful extraction or warnings for invalid formats
   - Assign terminal ID to all sessions created from that file

2. **Session creation**: Each `TransactionSession` object includes:
   ```python
   terminal_id: Optional[str] = None  # ABM Terminal ID extracted from filename
   ```

3. **Database storage**: Terminal ID is stored in the `ml_sessions` table:
   ```sql
   terminal_id VARCHAR(20)
   ```

## Database Schema Updates

### New Columns
- `ml_sessions.terminal_id` - Stores the ABM terminal ID
- Indexed for performance: `idx_ml_sessions_terminal_id`

### New Views and Functions

#### Terminal Statistics View
```sql
SELECT * FROM terminal_statistics;
```
Shows:
- Total sessions per terminal
- Anomaly count and rate
- Average anomaly score
- First and last session times

#### Get Sessions by Terminal Function
```sql
SELECT * FROM get_sessions_by_terminal('416', 50);
```
Returns recent sessions for a specific terminal.

### Migration Files
- `003_add_location_support.sql` - Initial terminal_id column
- `004_ensure_terminal_id_support.sql` - Ensures proper setup and indexing

## Code Changes Summary

### 1. TransactionSession Class Updates
**Files**: 
- `services/api/ml_analyzer.py`
- `services/anomaly-detector/ml_analyzer.py`

**Changes**:
```python
@dataclass
class TransactionSession:
    session_id: str
    raw_text: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    terminal_id: Optional[str] = None  # NEW: ABM Terminal ID
    # ... other fields
```

### 2. Sessionisation Logic Updates
**Method**: `split_into_sessions()`

**Changes**:
- Extract terminal ID from filename
- Log extraction results
- Assign terminal ID to all sessions
- Enhanced documentation

### 3. Database Storage Updates
**File**: `services/anomaly-detector/main.py`

**Methods Updated**:
- `store_sessions()` - Include terminal_id in session data
- `store_production_sessions()` - Include terminal_id for production sessions
- `store_sessions_with_conflict_resolution()` - Update SQL queries

**SQL Changes**:
```sql
-- INSERT statement
INSERT INTO ml_sessions 
(session_id, timestamp, ..., terminal_id, created_at)
VALUES (:session_id, :timestamp, ..., :terminal_id, :created_at)

-- UPDATE statement  
UPDATE ml_sessions SET 
    timestamp = :timestamp,
    ...
    terminal_id = :terminal_id,
    created_at = :created_at
WHERE session_id = :session_id
```

### 4. Real-time Processing Updates
**Method**: `process_realtime_session()`

**Changes**:
- Added optional `terminal_id` parameter
- Include terminal_id in session creation

## Usage Examples

### 1. Processing EJ Files
```python
# File: ABM416EJ_20250101_20250630.txt
processor = MLFirstEJProcessor()
processor.process_ej_file("/path/to/ABM416EJ_20250101_20250630.txt")

# All sessions will automatically have terminal_id = "416"
```

### 2. Database Queries
```sql
-- Get all anomalies for terminal 416
SELECT * FROM ml_sessions 
WHERE terminal_id = '416' AND is_anomaly = true;

-- Get terminal statistics
SELECT * FROM terminal_statistics 
WHERE terminal_id = '416';

-- Most problematic terminals
SELECT * FROM terminal_statistics 
ORDER BY anomaly_rate_percent DESC 
LIMIT 10;
```

### 3. Real-time Processing
```python
# Process with known terminal ID
result = processor.process_realtime_session(
    session_text="[020t CARD INSERTED...",
    terminal_id="416"
)
```

## Error Handling

### Invalid Filenames
If the filename doesn't match the expected pattern:
- Terminal ID will be `None`
- Warning logged: "Could not extract terminal ID from filename"
- Processing continues normally

### Missing Terminal ID
Sessions without terminal_id:
- Can still be processed and stored
- Database queries can filter out `NULL` terminal_id values
- Migration script attempts to extract from existing session_id patterns

## Testing

### Test Script
Run the terminal ID extraction test:
```bash
python test_terminal_id_extraction.py
```

Tests:
- Filename pattern matching
- Sessionisation with terminal ID
- Database schema validation

### Expected Output
```
✓ PASS | ABM416EJ_20250101_20250630.txt | Expected: 416  | Got: 416
✓ PASS | ABM175EJ_20250624_20250624.txt | Expected: 175  | Got: 175
✓ Terminal ID extraction successful!
```

## Benefits

1. **Terminal-Specific Analysis**: Track anomalies per ATM terminal
2. **Performance Optimization**: Indexed queries by terminal_id
3. **Maintenance Insights**: Identify problematic terminals
4. **Historical Tracking**: Monitor terminal performance over time
5. **Automated Extraction**: No manual configuration required

## Migration Path

1. **Existing Data**: Migration script attempts to extract terminal IDs from existing session_id patterns
2. **New Data**: Automatically extracted during processing
3. **Backwards Compatibility**: All existing functionality remains unchanged
4. **Gradual Adoption**: Systems work with or without terminal_id

## Monitoring

### Log Messages
- `"Extracted terminal ID: 416 from filename: ABM416EJ_20250101_20250630.txt"`
- `"Could not extract terminal ID from filename: invalid_file.txt"`

### Database Queries
Monitor terminal_id extraction success:
```sql
SELECT 
    COUNT(*) as total_sessions,
    COUNT(terminal_id) as sessions_with_terminal_id,
    ROUND((COUNT(terminal_id)::decimal / COUNT(*)) * 100, 2) as extraction_rate_percent
FROM ml_sessions;
```
