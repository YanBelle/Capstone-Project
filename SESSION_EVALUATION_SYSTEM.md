# Session-Level Model Evaluation System

## Overview

This system provides comprehensive session-level evaluation capabilities for the ABM ML-First Anomaly Detection ensemble. It allows you to analyze individual EJ sessions across all models with detailed visualizations and explanations.

## Components

### 1. SessionModelEvaluator (`session_evaluation.py`)
- **Purpose**: Evaluates individual EJ sessions against specific models in the ensemble
- **Models Supported**:
  - Isolation Forest (unsupervised anomaly detection)
  - One-Class SVM (support vector-based anomaly detection)
  - DBSCAN (clustering-based anomaly detection)
  - DeepLog LSTM (sequential event analysis)
  - Sentiment Analysis (VADER + TextBlob combination)
  - Preprocessing Analysis (StandardScaler + PCA impact)

### 2. EnsembleVisualizationEngine (`model_visualization.py`)
- **Purpose**: Creates comprehensive visualizations for ensemble model performance
- **Visualizations Available**:
  - Isolation Forest: Score distributions, PCA projections, ROC curves
  - One-Class SVM: Support vector analysis, decision boundaries
  - DBSCAN: Cluster analysis, silhouette plots
  - Ensemble Dashboard: 12 comprehensive visualizations

### 3. Interactive Session Evaluation Page (`templates/session_evaluation.html`)
- **Purpose**: Web interface for real-time session evaluation
- **Features**:
  - Direct session ID input
  - Model-specific or all-models evaluation
  - URL parameter support for direct links
  - Real-time visualizations
  - Detailed explanations for each model's decision

## API Endpoints

### Session Evaluation
```
GET /api/v1/session/evaluate/{session_id}
```
- Evaluates session across all models
- Returns comprehensive analysis with overall assessment

```
GET /api/v1/session/evaluate/{session_id}/{model_name}
```
- Evaluates session with specific model
- Available models: isolation_forest, one_class_svm, dbscan, deeplog_lstm, sentiment_analysis, preprocessing

### Visualizations
```
GET /api/v1/visualization/ensemble/dashboard
```
- Returns comprehensive ensemble dashboard with 12 visualizations

```
GET /api/v1/visualization/model/{model_name}
```
- Returns model-specific visualization
- Available for: isolation_forest, one_class_svm, dbscan

### Interactive Page
```
GET /session-evaluation
```
- Serves the interactive HTML page for session evaluation
- Supports URL parameters: `?session_id=SESSION_ID&model=MODEL_NAME`

## Usage Examples

### 1. Direct URL Access
Navigate to: `http://localhost:8000/session-evaluation?session_id=ABC123&model=all`
- This will automatically load and evaluate session ABC123 across all models

### 2. API Usage
```python
import requests

# Evaluate session across all models
response = requests.get("http://localhost:8000/api/v1/session/evaluate/ABC123")
result = response.json()

# Get overall assessment
overall = result['overall_assessment']
print(f"Prediction: {overall['overall_prediction']}")
print(f"Anomaly Probability: {overall['anomaly_probability']:.2%}")
print(f"Model Agreement: {overall['model_agreement']:.2%}")

# Check individual model results
for model_name, model_result in result['models'].items():
    print(f"{model_name}: {model_result['prediction']} (confidence: {model_result.get('confidence', 'N/A')})")
```

### 3. Model-Specific Evaluation
```python
# Evaluate with Isolation Forest only
response = requests.get("http://localhost:8000/api/v1/session/evaluate/ABC123/isolation_forest")
result = response.json()

print(f"Anomaly Score: {result['result']['anomaly_score']}")
print(f"Explanation: {result['result']['explanation']}")

# Get visualization if available
if 'visualization' in result['result']:
    import base64
    img_data = base64.b64decode(result['result']['visualization'])
    with open('isolation_forest_viz.png', 'wb') as f:
        f.write(img_data)
```

## Model-Specific Features

### Isolation Forest
- **Metrics**: Anomaly score, decision function, confidence
- **Visualization**: Score distribution, PCA projection, feature importance
- **Explanation**: Split-based anomaly detection reasoning

### One-Class SVM
- **Metrics**: Decision score, distance to boundary, support vector info
- **Visualization**: Support vector analysis, decision boundaries
- **Explanation**: Support vector machine boundary analysis

### DBSCAN
- **Metrics**: Cluster assignment, distances to cluster centers
- **Visualization**: Cluster plots, silhouette analysis
- **Explanation**: Density-based clustering results

### DeepLog LSTM
- **Metrics**: Sequence analysis, event patterns, completion status
- **Visualization**: Event sequence timeline, frequency distribution
- **Explanation**: Sequential pattern anomaly detection
- **Features**: 
  - Event sequence extraction
  - Transaction completeness checking
  - Pattern repetition analysis
  - Transition analysis

### Sentiment Analysis (VADER + TextBlob)
- **Metrics**: VADER score, TextBlob score, combined score, severity level
- **Visualization**: Sentiment breakdown, negative phrases, component analysis
- **Explanation**: Negative sentiment detection for technical failures
- **Features**:
  - Dual sentiment engine combination
  - Negative phrase extraction
  - Technical failure pattern recognition
  - Confidence scoring

### Preprocessing Analysis
- **Metrics**: Scaling impact, PCA dimension reduction, information retention
- **Visualization**: Feature distribution before/after processing
- **Explanation**: Data transformation impact analysis

## Integration with Existing System

### Requirements
- FastAPI application with existing ML analyzer
- Session data available in database or cache
- Dependencies: matplotlib, seaborn, numpy, pandas, sklearn

### Setup
1. Place `session_evaluation.py` and `model_visualization.py` in the API services directory
2. Place `session_evaluation.html` in `templates/` directory
3. Add endpoints to `main.py` (already implemented)
4. Ensure ML analyzer is available globally or modify endpoint code

### Session Data Format
The system expects session data with:
```python
{
    'session_id': str,
    'raw_text': str,
    'cleaned_text': str,
    'created_at': datetime
}
```

## Error Handling

The system includes comprehensive error handling:
- **Missing Session**: 404 error with clear message
- **Model Unavailable**: 500 error with specific model status
- **Processing Errors**: Graceful degradation with partial results
- **Visualization Failures**: Continues without visualization component

## Performance Considerations

- **Caching**: Session data is cached when possible
- **Lazy Loading**: Visualizations generated on-demand
- **Graceful Degradation**: Functions without full dependency stack
- **Memory Management**: Large visualizations are base64 encoded and cleaned up

## Future Enhancements

1. **Batch Evaluation**: Process multiple sessions simultaneously
2. **Comparison Mode**: Compare multiple sessions side-by-side
3. **Historical Analysis**: Track session evaluation changes over time
4. **Export Features**: PDF reports, CSV exports
5. **Real-time Monitoring**: WebSocket-based live evaluation
6. **Custom Thresholds**: User-configurable anomaly thresholds
7. **Model Explanation**: SHAP/LIME integration for deeper insights

## Security Notes

- Input validation on all session IDs
- SQL injection protection in database queries
- Rate limiting recommended for production
- Authentication should be added for sensitive environments

## Dependencies Status

```python
# Core dependencies (required)
fastapi
pydantic
sqlalchemy

# ML dependencies (required for full functionality)
scikit-learn
numpy
pandas

# Visualization dependencies (optional - graceful degradation)
matplotlib
seaborn
plotly

# Additional dependencies
redis (optional - for caching)
loguru (optional - for logging)
```

This comprehensive session evaluation system provides deep insights into how each model in the ensemble processes individual EJ sessions, enabling detailed forensic analysis and model performance understanding.
