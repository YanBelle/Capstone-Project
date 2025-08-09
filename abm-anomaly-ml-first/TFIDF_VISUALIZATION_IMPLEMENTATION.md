# One-Class SVM TF-IDF Visualization Implementation

## 🎯 **What's Been Implemented**

### **1. Enhanced One-Class SVM Detector** (`services/anomaly-detector/oneclass_svm_detector.py`)

#### **TF-IDF Feature Analysis:**
- **`get_tfidf_feature_importance()`** - Extracts top TF-IDF words with importance scores
- **`get_outlier_analysis()`** - Complete analysis combining prediction + TF-IDF features
- **`_categorize_important_words()`** - Groups words into logical categories:
  - `error_terms` - error, fail, timeout, abort, reject, invalid, malfunction
  - `hardware_terms` - power, reset, hardware, device, component, cim, recovery
  - `transaction_terms` - transaction, withdraw, deposit, balance, pin, card, cash
  - `status_terms` - start, end, success, complete, activated, taken, inserted
  - `other_terms` - uncategorized words

#### **Advanced Feature Extraction:**
```python
TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 3),  # Capture phrases like "POWER-UP/RESET"
    stop_words=None,     # Keep all words for technical logs
    token_pattern=r'\b\w+(?:[-/]\w+)*\b'  # Handle hyphenated terms
)
```

### **2. Enhanced Ensemble Detector** (`enhanced_ensemble_detector.py`)

#### **New TF-IDF Methods:**
- **`get_tfidf_analysis_for_session()`** - Detailed TF-IDF analysis for single sessions
- **`predict_single_session()`** - Ensemble prediction for individual sessions
- **`_categorize_tfidf_words()`** - Word categorization for visualization

#### **Integration with Ensemble:**
- Uses TF-IDF as text feature source for One-Class SVM
- Combines with numerical features and DBSCAN clustering
- Provides word-level explanations for anomaly decisions

### **3. API Endpoints** (`services/api/main.py` & `services/api/svm_debug_api.py`)

#### **New TF-IDF Endpoints:**
```
POST /api/v1/svm-tfidf/analyze-session
- Analyzes session text using enhanced ensemble TF-IDF
- Returns prediction + detailed word analysis + categories

GET /api/v1/svm-tfidf/vocabulary
- Returns TF-IDF vocabulary information
- Model configuration and sample words

POST /api/v1/svm-debug/tfidf-analysis
- Legacy endpoint for One-Class SVM specific analysis
```

### **4. React Dashboard Component** (`services/dashboard/src/TFIDFVisualization.js`)

#### **Visualization Features:**
- **📊 Bar Chart** - Top TF-IDF words contributing to outlier detection
- **🥧 Pie Chart** - Distribution of word categories
- **📋 Detailed Table** - Word-by-word analysis with scores
- **📈 Summary Cards** - Key metrics (anomaly status, scores, word counts)
- **🔍 Category Filtering** - View words by category (error, hardware, transaction, etc.)

#### **Interactive Elements:**
- Sample session selector (power reset, incomplete transaction, normal)
- Real-time text input for custom analysis
- Category-based filtering
- Responsive design with Tailwind CSS

#### **Sample Test Cases:**
```javascript
'power_reset_anomaly': // POWER-UP/RESET, HARDWARE ERROR patterns
'incomplete_transaction': // DEVICE MALFUNCTION, incomplete flow
'normal_transaction': // CASH DISPENSED SUCCESSFULLY, complete flow
```

### **5. Dashboard Integration** (`services/dashboard/src/Dashboard.js`)

#### **New Tab Added:**
- **"TF-IDF Analysis"** tab in main dashboard navigation
- Integrated with existing SVM Debug and other analysis tools
- Consistent styling with rest of dashboard

### **6. Test Suite** (`test_tfidf_analysis.py`)

#### **Automated Testing:**
- **Training Test** - Trains enhanced ensemble with sample data
- **Analysis Test** - Tests TF-IDF analysis on different session types
- **Vocabulary Test** - Verifies model vocabulary and configuration
- **Integration Test** - End-to-end workflow validation

## 🚀 **How to Use**

### **1. Start the System:**
```bash
cd /home/yc/development/Capstone-Project/abm-anomaly-ml-first
docker-compose up -d
```

### **2. Train the Model (if needed):**
```bash
python test_tfidf_analysis.py
```

### **3. Access the Dashboard:**
- Navigate to `http://localhost:3000`
- Click on "TF-IDF Analysis" tab
- Select a sample session or enter custom text
- Click "Analyze TF-IDF Features"

### **4. Interpret Results:**

#### **For Anomalous Sessions:**
- **High TF-IDF scores** for error/hardware terms
- **Words like:** "power", "reset", "error", "malfunction", "recovery", "failed"
- **Categories:** Dominated by error_terms and hardware_terms
- **Bar chart:** Shows clear peaks for problematic words

#### **For Normal Sessions:**
- **High TF-IDF scores** for transaction terms
- **Words like:** "transaction", "cash", "dispensed", "successfully", "card", "pin"
- **Categories:** Dominated by transaction_terms and status_terms
- **Bar chart:** Shows balanced distribution of normal operation words

## 🎯 **Key Benefits**

### **1. Explainable AI:**
- Shows **exactly which words** triggered anomaly detection
- **Categorizes words** for better understanding
- **Visual representation** of decision factors

### **2. Domain-Specific Analysis:**
- **ABM-specific terminology** (POWER-UP/RESET, CIM-RESET, etc.)
- **Transaction flow understanding** (CARD INSERTED → PIN → CASH → CARD TAKEN)
- **Hardware error patterns** (RECOVERY FAILED, DEVICE MALFUNCTION)

### **3. Interactive Debugging:**
- **Real-time analysis** of any session text
- **Category filtering** to focus on specific word types
- **Sample sessions** for quick testing and validation

### **4. Integration with Ensemble:**
- **Not just SVM** - integrated with full ensemble (Isolation Forest + DBSCAN)
- **Multiple feature types** - text (TF-IDF) + numerical + clustering
- **Confidence scoring** and consensus analysis

## 📊 **Example Analysis Output**

For a session with `POWER-UP/RESET OCCURRED HARDWARE ERROR DETECTED RECOVERY FAILED`:

```json
{
  "prediction_result": {
    "is_anomaly": true,
    "ensemble_score": 0.847
  },
  "tfidf_analysis": [
    {"word": "power", "tfidf_score": 0.4521, "importance": 100.0},
    {"word": "reset", "tfidf_score": 0.4521, "importance": 100.0},
    {"word": "hardware", "tfidf_score": 0.3876, "importance": 85.7},
    {"word": "error", "tfidf_score": 0.3654, "importance": 80.8},
    {"word": "recovery", "tfidf_score": 0.3098, "importance": 68.5},
    {"word": "failed", "tfidf_score": 0.3098, "importance": 68.5}
  ],
  "word_categories": {
    "error_terms": [{"word": "error", ...}, {"word": "failed", ...}],
    "hardware_terms": [{"word": "power", ...}, {"word": "reset", ...}, {"word": "hardware", ...}, {"word": "recovery", ...}],
    "transaction_terms": [],
    "status_terms": [],
    "other_terms": []
  }
}
```

This implementation provides **comprehensive TF-IDF visualization** specifically designed for **One-Class SVM anomaly detection** in **ABM transaction logs**, with **interactive dashboards** and **explainable AI capabilities**.
