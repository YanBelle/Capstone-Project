# EJ Rule-Based Processor
## Automated EJ Log Sessionization and Error Detection

This solution provides a comprehensive rule-based approach to process Electronic Journal (EJ) logs, automatically sessionizing them and separating normal transactions from those containing errors.

## 🎯 Features

- **Automated Sessionization**: Intelligently splits EJ logs into individual transaction sessions
- **Error Detection**: Comprehensive rule-based error detection with 15+ error patterns
- **False Positive Prevention**: Smart rules to avoid flagging normal operations as errors
- **Tabular Output**: Results stored in CSV format for easy analysis
- **Multiple Output Formats**: Full data, summaries, and error analysis reports
- **No Dependencies**: Lightweight version uses only Python standard library

## 🚀 Quick Start

### Option 1: Simple Shell Script (Recommended)
```bash
# 1. Place your EJ .txt files in ./data/input/
cp /path/to/your/ej_files/*.txt ./data/input/

# 2. Run the processor
./run_ej_processor.sh
```

### Option 2: Direct Python Execution
```bash
# Run the CSV-safe processor (recommended)
python3 ej_rule_processor_csv_safe.py

# Or run the lightweight processor (basic version)
python3 ej_rule_processor_lightweight.py
```

### Option 3: Full-Featured Version (Requires pandas)
```bash
# Install dependencies
pip install -r requirements_ej_processor.txt

# Run full version
python3 ej_rule_based_processor.py
```

## 📁 Directory Structure

```
./
├── data/
│   ├── input/          # Place your EJ .txt files here
│   └── processed/      # Output files will be generated here
├── ej_rule_processor_lightweight.py    # Main processor (no dependencies)
├── ej_rule_based_processor.py          # Full version (requires pandas)
├── run_ej_processor.sh                 # Simple runner script
├── ej_processor_config.json            # Configuration file
└── requirements_ej_processor.txt       # Python dependencies
```

## 📊 Output Files

After processing, you'll find these files in `./data/processed/`:

### CSV Files (Excel-Compatible) 📊
1. **`normal_sessions_summary_[timestamp].csv`** - Clean summary of normal sessions
2. **`error_sessions_summary_[timestamp].csv`** - Clean summary of error sessions

### Complete Data Files (JSON) 📋
3. **`normal_sessions_full_[timestamp].json`** - Complete data with BERT preprocessing
4. **`error_sessions_full_[timestamp].json`** - Complete error data with BERT preprocessing

### Raw Text Files (Human Readable) 📄
5. **`raw_sessions_[timestamp].txt`** - All session raw text for review

### Analysis Reports 📊
6. **`error_analysis_report_[timestamp].json`** - Detailed error analysis

### 🎯 Multi-Format Strategy
To avoid CSV corruption from commas and special characters in EJ logs:
- **CSV files**: Clean, structured summaries perfect for Excel analysis
- **JSON files**: Complete data preservation with BERT-preprocessed text for ML training
- **TXT files**: Human-readable raw session text for detailed review

### 🚀 BERT Preprocessing Integration
Each JSON session now includes:
- **`raw_text_base64`**: Original EJ text (Base64 encoded)
- **`bert_preprocessed_text`**: BERT-optimized text for ML training
- **`preprocessing_info`**: Compression ratios and processing details

This significantly reduces model training time by eliminating repeated preprocessing!

## 🔍 Error Detection Rules

### Critical Errors (High Priority)
- **Cash Retract Scenarios**: Customer money retained by ATM
  - `INIT BNA STARTED - RETRACT BIN`
  - `CASHIN RETRACT STARTED - RETRACT BIN`
  - `CIM-RESET CALLED - RETRACT BIN`
- **Supervisor Mode Entry**: Manual intervention required
- **Power Reset**: System reset during operation
- **Unable to Dispense**: Cash dispensing failure
- **Incomplete Transactions**: 
  - Notes presented but not taken
  - Card inserted but not retrieved

### High Priority Errors
- **Deposit Errors**: Cash deposit processing failures
- **Device Errors**: Hardware malfunctions
- **Communication Errors**: System communication failures

### Medium Priority Errors
- **Error Codes**: ESC, VAL, REF, REJECTS codes
- **Timeouts**: Operation timeouts

### False Positive Prevention
The system intelligently ignores:
- **Card initialization attempts** (up to 3 tries normal for magstrip cards)
- **Customer cancellations** (normal user behavior)
- **Successful completions** (complete transaction flows)

## 📋 CSV Output Schema

### Session Data Fields
| Field | Description |
|-------|-------------|
| `session_id` | Unique session identifier |
| `file_source` | Source EJ file name |
| `start_time` | Session start timestamp |
| `end_time` | Session end timestamp |
| `has_errors` | Boolean: true if errors detected |
| `error_types` | List of detected error types |
| `error_details` | Detailed error information |
| `transaction_type` | withdrawal, deposit, balance_inquiry, etc. |
| `card_inserted` | Boolean: card insertion detected |
| `pin_entered` | Boolean: PIN entry detected |
| `transaction_completed` | Boolean: transaction completion |
| `notes_dispensed` | Boolean: cash dispensed |
| `notes_taken` | Boolean: cash collected by customer |
| `card_taken` | Boolean: card retrieved by customer |
| `withdrawal_amount` | Amount withdrawn (if applicable) |
| `authorization_code` | Transaction authorization code |
| `raw_text` | Complete session text |

## � BERT Preprocessing Integration

### **Why BERT Preprocessing Matters:**
- **Training Speed**: Eliminates repeated preprocessing during model training
- **Consistency**: Uses exact same preprocessing as BertVisualizationAnalyzer
- **Optimization**: ATM domain-specific token patterns and noise reduction
- **Memory Efficiency**: Compressed text reduces storage and processing overhead

### **Preprocessing Benefits:**
```
Original EJ Text (2,450 chars):
*[020t*629*06/18/2025*00:46* TRANSACTION START *PRIMARY CARD READER ACTIVATED* 
CARD INSERTED ATR RECEIVED T=0 *7231*1*(Iw(1*3, PIN ENTERED NOTES PRESENTED 50,000 
NOTES TAKEN CARD TAKEN TRANSACTION END...

↓ BERT Preprocessing ↓

Preprocessed Text (856 chars):
TRANSACTION_START PRIMARY_CARD_READER_ACTIVATED CARD_INSERTED ATR_RECEIVED_T_0 
PIN_ENTERED NOTES_PRESENTED NOTES_TAKEN CARD_TAKEN TRANSACTION_END...

Compression: 65% size reduction with preserved semantic meaning!
```

### **Preprocessing Features:**
✅ **Noise Removal**: Timestamps, transaction codes, formatting artifacts  
✅ **Compound Tokens**: Multi-word ATM terms become single tokens  
✅ **Pattern Optimization**: ESC_000, VAL_000, REJECTS_000 format  
✅ **Receipt Compression**: Long receipts → RECEIPT_PRINTED  
✅ **Domain Intelligence**: ATM-specific token preservation  

## �🛠️ CSV Structure Protection

### The Problem You Identified ✅
EJ logs contain:
- **Commas in transaction data**: `"WITHDRAWAL, AMOUNT: 5,000"` 
- **Quotes in messages**: `'Customer said "cancel transaction"'`
- **Newlines in raw text**: Multi-line EJ entries
- **JSON structures**: Complex error details

### Our Solution Strategy 🎯

#### 1. **CSV Summaries** (Excel-Safe)
- **No raw text**: Only clean, structured fields
- **Safe separators**: Uses `|` instead of `,` for lists
- **Proper quoting**: `csv.QUOTE_MINIMAL` with escaping
- **Truncation**: Limits field lengths for Excel compatibility

#### 2. **JSON Complete Data** (Full Preservation)  
- **Base64 encoding**: Raw text encoded to prevent structure corruption
- **Native JSON**: Lists and objects preserved as-is
- **Full metadata**: All session details included

#### 3. **Raw Text Files** (Human Review)
- **Readable format**: Plain text with clear separators
- **Session boundaries**: Clear visual separation
- **All content**: Unmodified EJ log text

### Example Safe CSV Structure:
```csv
session_id,file_source,transaction_type,error_types_list,highest_severity
SESSION_1,ATM_001.txt,withdrawal,timeout | device_error,critical
SESSION_2,ATM_001.txt,deposit,cash_retract_bna,critical
```

### Example JSON Structure with BERT Preprocessing:
```json
{
  "session_id": "ATM_001_SESSION_1_20241203_142301",
  "raw_text_base64": "KlRSQU5TQUNUSU9OIFNUQVJUK...",
  "bert_preprocessed_text": "TRANSACTION_START CARD_INSERTED PIN_ENTERED WITHDRAWAL...",
  "preprocessing_info": {
    "raw_text_length": 2450,
    "preprocessed_text_length": 856,
    "compression_ratio": 0.35,
    "preprocessing_method": "BertVisualizationAnalyzer._preprocess_text"
  }
}
```

## 💡 Usage Examples

### Example 1: Basic Processing
```bash
# Copy EJ files
cp /path/to/ATM_logs/*.txt ./data/input/

# Process
./run_ej_processor.sh

# Results available in ./data/processed/
```

### Example 2: Review Error Sessions
```bash
# After processing, review error sessions safely
cat ./data/processed/error_sessions_summary_*.csv

# Or view complete data
cat ./data/processed/error_sessions_full_*.json

# Or read raw text
less ./data/processed/raw_sessions_*.txt
```

### Example 3: Analyze Specific Error Types
```bash
# View error analysis report
cat ./data/processed/error_analysis_report_*.json
```

## 📈 Sample Output

```
🚀 EJ Rule-Based Processor Starting...
🔍 Scanning ./data/input for EJ files...
📁 Found 3 EJ files to process

📄 Processing: ATM_001_20241203.txt
   ✅ Extracted 45 sessions

📄 Processing: ATM_002_20241203.txt
   ✅ Extracted 32 sessions

📊 Processing Summary:
   Total Sessions: 77
   Normal Sessions: 72 (93.5%)
   Error Sessions: 5 (6.5%)

💾 Normal sessions saved to: ./data/processed/normal_sessions_20241203_140523.csv
💾 Error sessions saved to: ./data/processed/error_sessions_20241203_140523.csv
📋 Normal sessions summary saved to: ./data/processed/normal_sessions_summary_20241203_140523.csv
📋 Error sessions summary saved to: ./data/processed/error_sessions_summary_20241203_140523.csv
📊 Error analysis report saved to: ./data/processed/error_analysis_report_20241203_140523.json

✅ Processing Complete!
```

## 🔧 Advanced Usage

### Sessionization Logic
The processor uses multiple patterns to identify session boundaries:
1. **Transaction Start Markers**: `*TRANSACTION START*`
2. **Timestamp Patterns**: `*123*03/12/2024*14:05*`
3. **Context-aware boundaries**: Intelligent session splitting

### Error Classification Hierarchy
1. **Critical**: Immediate attention required (customer impact)
2. **High**: Operational issues (system reliability)
3. **Medium**: Monitoring alerts (performance impact)

### Data Integration
The CSV outputs can be easily imported into:
- **Excel**: For manual review and analysis
- **Database Systems**: For automated monitoring
- **BI Tools**: For dashboard creation
- **Python/R**: For advanced analytics

## 🚨 Important Notes

1. **File Placement**: EJ files must be placed in `./data/input/` directory
2. **File Format**: Only `.txt` files are processed
3. **Memory Usage**: Large EJ files are processed efficiently
4. **Encoding**: UTF-8 encoding with error handling for corrupted files
5. **Timestamps**: All output files include timestamp for version control

## 🆘 Troubleshooting

### No EJ Files Found
```bash
# Ensure files are in correct location
ls -la ./data/input/*.txt

# Copy files if needed
cp /your/path/*.txt ./data/input/
```

### Permission Issues
```bash
# Make script executable
chmod +x run_ej_processor.sh
```

### Python Issues
```bash
# Check Python version
python3 --version

# Should be Python 3.6 or higher
```

## 📞 Support

For questions or issues with the EJ Rule-Based Processor, check:
1. **Output logs**: Error messages are detailed
2. **Configuration**: Verify `ej_processor_config.json`
3. **File format**: Ensure EJ files are properly formatted
4. **Dependencies**: Use lightweight version if pandas issues occur

---

**The EJ Rule-Based Processor provides comprehensive, automated analysis of ATM transaction logs with intelligent error detection and prevention of false positives.**
