# DeepLog Training Solution with BERT Token Integration

## Overview

This document outlines a complete DeepLog training solution that integrates BERT embeddings for enhanced anomaly detection in log sequences. The solution combines the sequential modeling capabilities of DeepLog with the semantic understanding of BERT transformers.

## Architecture

### Components

1. **BERTDeepLogModel**: Neural network that accepts BERT embeddings as input
2. **BERTDeepLogTrainer**: Training and inference pipeline
3. **API Integration**: FastAPI endpoints for training and prediction
4. **Database Integration**: Stores raw text and training results

### Model Architecture

```
Input: Raw Text Sessions
    ↓
BERT Tokenizer & Model
    ↓
BERT Embeddings (768-dim)
    ↓
LSTM Layers (128 hidden units, 2 layers)
    ↓
Multi-Head Attention (8 heads)
    ↓
Classification Head (2-N classes)
    ↓
Output: Anomaly Prediction + Confidence
```

## Files Created

### 1. deeplog_bert_trainer.py
- **BERTDeepLogModel**: PyTorch model combining BERT + LSTM + Attention
- **BERTDeepLogTrainer**: Complete training pipeline
- **Features**:
  - BERT embedding extraction from raw text
  - LSTM sequence modeling
  - Multi-head attention mechanism
  - Model saving/loading
  - Training history tracking

### 2. deeplog_api.py
- API integration functions for FastAPI
- Functions for training, prediction, and status checking
- Database integration for training data collection
- File system fallback for data collection

## API Endpoints

### POST /api/v1/deeplog/retrain
Retrains the DeepLog model using available session data.

**Features**:
- Collects training data from database first
- Falls back to file system if insufficient database data
- Automatic label encoding based on anomaly scores
- Configurable training parameters

**Response**:
```json
{
  "status": "success",
  "message": "DeepLog model retrained successfully on 150 sessions",
  "training_results": {
    "training_history": [...],
    "best_val_accuracy": 87.5,
    "num_classes": 3,
    "label_encoder": {...}
  }
}
```

### POST /api/v1/deeplog/predict
Makes anomaly predictions on new session data.

**Request**:
```json
{
  "raw_text": "Session log content here..."
}
```

**Response**:
```json
{
  "status": "success",
  "prediction": {
    "is_anomaly": true,
    "anomaly_type": "high_anomaly",
    "confidence": 0.89,
    "probabilities": {
      "normal": 0.11,
      "medium_anomaly": 0.23,
      "high_anomaly": 0.66
    }
  }
}
```

### GET /api/v1/deeplog/status
Returns model status and training information.

**Response**:
```json
{
  "status": "ready",
  "message": "DeepLog model trained and ready for predictions",
  "available": true,
  "num_classes": 3,
  "labels": ["normal", "medium_anomaly", "high_anomaly"],
  "training_history": [...]
}
```

## Training Process

### Data Collection
1. **Database Priority**: Fetches sessions from `ml_sessions` table with `raw_text` column
2. **File System Fallback**: Scans `/app/input/processed/*.txt` files
3. **Label Generation**: Creates labels based on anomaly scores:
   - `normal`: anomaly_score ≤ 0.5
   - `medium_anomaly`: 0.5 < anomaly_score ≤ 0.8
   - `high_anomaly`: anomaly_score > 0.8

### Model Training
1. **BERT Embedding**: Converts raw text to BERT embeddings (768-dim)
2. **Sequence Modeling**: LSTM processes embedding sequences
3. **Attention**: Multi-head attention focuses on important patterns
4. **Classification**: Final layer predicts anomaly type
5. **Validation**: 80/20 train/validation split

### Training Parameters
- **Epochs**: 10 (configurable)
- **Batch Size**: 4 (optimized for limited data)
- **Learning Rate**: 1e-4
- **BERT Model**: distilbert-base-uncased (lightweight)
- **Max Sequence Length**: 512 tokens

## Integration with Existing System

### Database Schema Requirements
The system requires the `ml_sessions` table to have:
- `raw_text TEXT` column for storing original session content
- `anomaly_score FLOAT` for training label generation

### Dependencies
Required Python packages:
```
torch>=1.12.0
transformers>=4.21.0
numpy>=1.21.0
```

### Model Storage
- Model weights: `/app/models/deeplog_bert/deeplog_bert_model.pth`
- Configuration: `/app/models/deeplog_bert/model_config.pkl`
- Training history: `/app/models/deeplog_bert/training_history.pkl`

## Usage Examples

### 1. Train Model
```bash
curl -X POST "http://localhost/api/v1/deeplog/retrain"
```

### 2. Make Prediction
```bash
curl -X POST "http://localhost/api/v1/deeplog/predict" \
  -H "Content-Type: application/json" \
  -d '{"raw_text": "ERROR: Database connection failed at 2024-01-15 10:30:15"}'
```

### 3. Check Status
```bash
curl "http://localhost/api/v1/deeplog/status"
```

## Benefits Over Traditional Approaches

### 1. Semantic Understanding
- BERT embeddings capture semantic meaning beyond keyword matching
- Understands context and relationships between log elements
- Better generalization to unseen log patterns

### 2. Sequence Modeling
- LSTM captures temporal dependencies in log sequences
- Attention mechanism highlights important log events
- Handles variable-length sequences effectively

### 3. Multi-Class Classification
- Distinguishes between different types of anomalies
- Provides confidence scores for predictions
- Enables fine-grained anomaly categorization

### 4. Continuous Learning
- Model can be retrained with new data
- Adapts to evolving system behavior
- Incorporates expert feedback through labeled data

## Performance Considerations

### Memory Usage
- BERT model: ~250MB
- DeepLog model: ~50MB
- Training data: Depends on session count and length

### Training Time
- 100 sessions: ~5-10 minutes
- 1000 sessions: ~30-60 minutes
- GPU recommended for large datasets

### Inference Speed
- Single prediction: ~100-500ms
- Batch predictions: ~50ms per session

## Error Handling

### Common Issues
1. **Insufficient Training Data**: Requires minimum 5 sessions
2. **Missing Dependencies**: Falls back gracefully without PyTorch
3. **Memory Limitations**: Automatic batch size adjustment
4. **Model Not Trained**: Clear error messages with guidance

### Fallback Behavior
- If PyTorch unavailable: Returns dependency error
- If insufficient data: Collects from file system
- If model untrained: Provides training instructions

## Future Enhancements

### 1. Advanced Features
- Custom BERT fine-tuning on log data
- Ensemble methods combining multiple models
- Active learning for optimal training data selection

### 2. Performance Optimizations
- Model quantization for faster inference
- Caching of BERT embeddings
- Distributed training for large datasets

### 3. Monitoring Integration
- Real-time prediction metrics
- Model drift detection
- Automated retraining triggers

## Conclusion

This DeepLog training solution provides a sophisticated anomaly detection system that combines the power of modern transformer models with traditional sequence modeling approaches. It offers superior performance compared to rule-based systems while maintaining interpretability and ease of use.

The solution is designed to integrate seamlessly with the existing ABM anomaly detection system, providing enhanced capabilities for identifying complex patterns in transaction logs and other sequential data.
