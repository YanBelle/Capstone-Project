# Force Process Input Directory - Updated with Sessionization ✅

## Summary of Changes

The `force_process_input_directory` method has been successfully updated to use the same sessionization approach as the `process_input` method, ensuring consistent processing with ML analyzer capabilities.

## Key Changes Made

### ✅ **Replaced Manual Processing with ML Analyzer Sessionization**

**Before (Manual Processing)**:
```python
# Old approach - manual file processing
for file_path in ej_files:
    # Read file content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create simple session ID
    session_id = f"force_processed_{int(time.time())}_{filename}"
    
    # Count lines for transaction count (inaccurate)
    lines = content.split('\n')
    transaction_count = len([line for line in lines if line.strip()])
    
    # Insert single session record directly into database
    # No ML analysis, no transaction boundary detection
```

**After (ML Analyzer Sessionization)**:
```python
# New approach - use batch_process_ej_files with ML analyzer
logger.info("Starting force processing with ML analyzer sessionization")
logger.info("Calling batch_process_ej_files for sessionization and processing")
processing_result = batch_process_ej_files(input_dir)
```

### ✅ **Enhanced Response with Sessionization Details**

**New Response Fields**:
```python
{
    "sessionization_enabled": True,
    "ml_analyzer_used": True/False,
    "total_sessions_created": X,
    "average_sessions_per_file": X.X,
    "processing_summary": {...},
    "detailed_results": [...]
}
```

### ✅ **Consistent Processing Pipeline**

Both endpoints now use the same processing pipeline:

1. **File Discovery**: Find EJ files in input directory
2. **ML Sessionization**: Use `batch_process_ej_files()` which calls ML analyzer
3. **Transaction Boundary Detection**: "*TRANSACTION START*" / "*TRANSACTION END*" patterns
4. **Individual Session Creation**: Multiple sessions per file instead of one
5. **Database Storage**: Clean text, raw text, terminal IDs, cassette counters
6. **Cache Management**: Clear Redis cache for dashboard refresh

## Benefits Achieved

### 🎯 **Accurate Transaction Detection**
- **Before**: Simple line counting (inaccurate)
- **After**: ML analyzer transaction boundary detection (accurate)
- **Result**: Proper individual transaction sessions

### 🔧 **Feature Preservation**
- **Cassette Counter Parsing**: Now available in force processing
- **Terminal ID Detection**: Now extracted from EJ filenames  
- **BertViz Cleaning**: Raw text preprocessing applied
- **Anomaly Detection**: Sessions ready for ML analysis

### 🚀 **Consistency**
- **Same Pipeline**: Both `/process-input` and `/process/force-input` use identical logic
- **Same Output**: Consistent session creation and database storage
- **Same Quality**: ML-driven sessionization for both endpoints

## Technical Implementation

### Method Signature
```python
@app.post("/api/v1/process/force-input")
async def force_process_input_directory():
    """Force the anomaly detection system to process any EJ files in the input directory with sessionization"""
```

### Core Processing Logic
```python
# Use the same sessionization approach as process_input method
logger.info("Starting force processing with ML analyzer sessionization")
processing_result = batch_process_ej_files(input_dir)

if processing_result['status'] == 'success':
    return {
        "sessionization_enabled": True,
        "ml_analyzer_used": processing_result['summary'].get('ml_processed_files', 0) > 0,
        "total_sessions_created": processing_result['summary'].get('total_sessions_created', 0),
        # ... additional sessionization metrics
    }
```

### Sessionization Features Now Available
1. **Transaction Boundary Detection**: Splits files into individual transactions
2. **Cassette Counter Parsing**: Extracts cassette information during processing
3. **Terminal ID Detection**: Extracts terminal IDs from EJ filenames
4. **Text Cleaning**: BertViz preprocessing applied to content
5. **Database Integration**: Proper storage with clean_text and raw_text fields

## Testing the Update

### API Call
```bash
POST /api/v1/process/force-input
```

### Expected Response
```json
{
    "status": "success",
    "message": "EJ files force processed with sessionization successfully",
    "files_found": 3,
    "sessionization_enabled": true,
    "ml_analyzer_used": true,
    "total_sessions_created": 15,
    "average_sessions_per_file": 5.0,
    "processing_summary": {
        "ml_processed_files": 3,
        "total_sessions_created": 15,
        "total_original_chars": 45000,
        "total_cleaned_chars": 38000
    }
}
```

## Comparison with process_input

Both methods now use identical core logic:

### `/api/v1/process-input`
```python
def process_input():
    processing_result = batch_process_ej_files("/app/input")
    # Return processing_result details
```

### `/api/v1/process/force-input` (Updated)
```python
async def force_process_input_directory():
    processing_result = batch_process_ej_files(input_dir)  
    # Return enhanced processing_result details
```

## Deployment Status

### ✅ **Ready for Production**
- Updated method maintains backward compatibility
- Enhanced response provides more detailed information
- Same reliable processing pipeline as standard input processing
- Comprehensive error handling preserved

### ✅ **User Benefits**
- Force processing now creates proper individual transaction sessions
- Cassette counter parsing available in force processing
- Terminal ID detection works in force processing
- ML analysis ready sessions for anomaly detection

The `force_process_input_directory` method now provides the same high-quality sessionization capabilities as the standard `process_input` method, ensuring consistent and accurate transaction boundary detection across all processing endpoints!
