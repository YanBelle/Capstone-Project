import React, { useState } from 'react';

const Training = ({ onTrainingComplete }) => {
  const [ejSessions, setEjSessions] = useState([]);
  const [trainingStatus, setTrainingStatus] = useState('idle');
  const [trainingResults, setTrainingResults] = useState(null);
  const [loadingStatus, setLoadingStatus] = useState('');
  const [config, setConfig] = useState({
    textWeight: 0.6,
    statisticalWeight: 0.4,
    threshold: 0.5
  });

  const loadFromProcessedData = async () => {
    setLoadingStatus('Loading from processed EJ data...');
    try {
      // Load from processed data without sending file or text
      const response = await fetch(`${process.env.REACT_APP_API_URL || 'http://localhost:8001'}/api/load_ej_sessions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          include_errors: false,  // Start with normal sessions only
          limit: 100  // Limit for testing
        })
      });

      const data = await response.json();
      if (data.success) {
        setEjSessions(data.sessions);
        setLoadingStatus(`✅ Loaded ${data.count} EJ sessions from ${data.data_source}`);
        alert(`Successfully loaded EJ session data!
        
📊 Source: ${data.data_source}
📝 Sessions: ${data.count}
🎯 Ready for ensemble training!`);
      } else {
        throw new Error(data.message || 'No processed data found');
      }
    } catch (error) {
      console.error('Loading from processed data failed:', error);
      setLoadingStatus(`Error: ${error.message}`);
      alert(`Could not load from processed data: ${error.message}

💡 Tip: This loads from processed EJ data files. Ensure the data processing has been completed to generate the required files.`);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoadingStatus('Loading EJ sessions...');
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${process.env.REACT_APP_API_URL || 'http://localhost:8001'}/api/load_ej_sessions`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (data.success) {
        setEjSessions(data.sessions);
        setLoadingStatus(`Loaded ${data.count} EJ sessions from ${data.source}`);
      } else {
        throw new Error(data.error || 'Failed to load sessions');
      }
    } catch (error) {
      console.error('File upload failed:', error);
      setLoadingStatus(`Error: ${error.message}`);
    }
  };

  const handleTextInput = async (text) => {
    if (!text.trim()) return;

    setLoadingStatus('Sessionizing EJ log text...');
    try {
      const formData = new FormData();
      formData.append('text', text);

      const response = await fetch(`${process.env.REACT_APP_API_URL || 'http://localhost:8001'}/api/sessionize`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (data.success) {
        setEjSessions(data.sessions);
        setLoadingStatus(`Sessionized into ${data.count} EJ sessions`);
      } else {
        throw new Error(data.error || 'Failed to sessionize');
      }
    } catch (error) {
      console.error('Sessionization failed:', error);
      setLoadingStatus(`Error: ${error.message}`);
    }
  };

  const trainModel = async () => {
    if (ejSessions.length === 0) {
      alert('Please load EJ sessions first');
      return;
    }

    setTrainingStatus('training');
    try {
      const response = await fetch(`${process.env.REACT_APP_API_URL || 'http://localhost:8001'}/api/train`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sessions: ejSessions,
          text_weight: config.textWeight,
          statistical_weight: config.statisticalWeight,
          threshold: config.threshold
        }),
      });

      const data = await response.json();
      if (data.success) {
        setTrainingResults(data.training_stats);
        setTrainingStatus('completed');
        onTrainingComplete && onTrainingComplete();
      } else {
        throw new Error(data.detail || 'Training failed');
      }
    } catch (error) {
      console.error('Training failed:', error);
      setTrainingStatus('failed');
      setLoadingStatus(`Training failed: ${error.message}`);
    }
  };

  const updateConfig = (key, value) => {
    const newConfig = { ...config, [key]: parseFloat(value) };
    
    // Auto-adjust weights to sum to 1
    if (key === 'textWeight') {
      newConfig.statisticalWeight = 1.0 - newConfig.textWeight;
    } else if (key === 'statisticalWeight') {
      newConfig.textWeight = 1.0 - newConfig.statisticalWeight;
    }
    
    setConfig(newConfig);
  };

  const loadSampleData = () => {
    const sampleSessions = [
      `SESSION START
CARD INSERTED
PIN ENTERED
PIN VERIFIED
BALANCE INQUIRY SELECTED
ACCOUNT BALANCE: $1,250.45
RECEIPT PRINTED
CARD EJECTED
SESSION END`,
      `SESSION START
CARD INSERTED
PIN ENTERED
PIN VERIFIED
WITHDRAW SELECTED
AMOUNT ENTERED: $100
CASH DISPENSED
RECEIPT PRINTED
CARD EJECTED
SESSION END`,
      `SESSION START
CARD INSERTED
PIN ENTERED
PIN VERIFIED
DEPOSIT SELECTED
ENVELOPE INSERTED
DEPOSIT AMOUNT: $500
DEPOSIT COMPLETED
RECEIPT PRINTED
CARD EJECTED
SESSION END`,
      `SESSION START
CARD INSERTED
PIN ENTERED
PIN VERIFIED
TRANSFER SELECTED
ACCOUNT FROM: CHECKING
ACCOUNT TO: SAVINGS
AMOUNT: $200
TRANSFER COMPLETED
RECEIPT PRINTED
CARD EJECTED
SESSION END`,
      `SESSION START
CARD INSERTED
PIN ENTERED
PIN INCORRECT
PIN ENTERED
PIN VERIFIED
CUSTOMER CANCELLED
CARD EJECTED
SESSION END`
    ];
    
    setEjSessions(sampleSessions);
    setLoadingStatus(`Loaded ${sampleSessions.length} sample normal EJ sessions`);
  };

  return (
    <div className="training-container">
      {/* Header Section */}
      <div className="section-header">
        <h2>🚀 Train Ensemble Model</h2>
        <p className="section-description">
          Load normal EJ sessions to train the ensemble anomaly detection model combining TF-IDF text analysis with statistical features for comprehensive pattern learning
        </p>
      </div>

      {/* Data Loading Section */}
      <div className="section">
        <div className="section-title">
          <div>
            <h3>📁 Load Training Data</h3>
            <p>Choose your preferred method to load EJ session data for comprehensive model training</p>
          </div>
        </div>
        
        <div className="data-source-grid">
          {/* Recommended Option */}
          <div className="data-source-card recommended">
            <div className="card-header">
              <div className="card-icon">🎯</div>
              <div className="card-title">
                <h4>Real EJ Data</h4>
                <span className="badge recommended-badge">RECOMMENDED</span>
              </div>
            </div>
            <p className="card-description">
              Load real EJ sessions from processed data files for comprehensive training with actual transaction patterns
            </p>
            <button 
              className="btn btn-primary full-width" 
              onClick={loadFromProcessedData}
              disabled={loadingStatus && loadingStatus.includes('Loading')}
            >
              {loadingStatus && loadingStatus.includes('Loading') ? '🔄 Loading...' : '🎯 Load from Processed Data'}
            </button>
          </div>

          {/* File Upload Option */}
          <div className="data-source-card">
            <div className="card-header">
              <div className="card-icon">📂</div>
              <div className="card-title">
                <h4>CSV Upload</h4>
              </div>
            </div>
            <p className="card-description">
              Upload a CSV file with 'text' column containing EJ sessions for custom training datasets
            </p>
            <div className="file-upload-wrapper">
              <input
                type="file"
                accept=".csv"
                onChange={handleFileUpload}
                className="file-input"
                id="csv-upload"
              />
              <label htmlFor="csv-upload" className="file-label">
                📂 Choose CSV File
              </label>
            </div>
          </div>

          {/* Raw Text Option */}
          <div className="data-source-card">
            <div className="card-header">
              <div className="card-icon">📝</div>
              <div className="card-title">
                <h4>Raw EJ Log</h4>
              </div>
            </div>
            <p className="card-description">
              Paste raw EJ log text - will be automatically sessionized into individual transaction flows
            </p>
            <textarea
              rows={4}
              placeholder="Paste raw EJ log text here..."
              className="text-input"
              onBlur={(e) => handleTextInput(e.target.value)}
            />
          </div>

          {/* Sample Data Option */}
          <div className="data-source-card">
            <div className="card-header">
              <div className="card-icon">🧪</div>
              <div className="card-title">
                <h4>Sample Data</h4>
              </div>
            </div>
            <p className="card-description">
              Use sample normal EJ sessions for testing and demonstration of the ensemble training process
            </p>
            <button className="btn btn-secondary full-width" onClick={loadSampleData}>
              🧪 Load Sample Data
            </button>
          </div>
        </div>

        {/* Loading Status */}
        {loadingStatus && (
          <div className={`status-alert ${loadingStatus.includes('✅') ? 'success' : loadingStatus.includes('Error') ? 'error' : 'info'}`}>
            <div className="status-content">
              <div className="status-icon">
                {loadingStatus.includes('✅') ? '✅' : 
                 loadingStatus.includes('Error') ? '❌' : 
                 loadingStatus.includes('Loading') ? '🔄' : 'ℹ️'}
              </div>
              <div className="status-text">{loadingStatus}</div>
            </div>
          </div>
        )}
      </div>

      {/* Loaded Sessions Preview */}
      {ejSessions.length > 0 && (
        <div className="section">
          <div className="section-title">
            <div>
              <h3>📊 Loaded Training Data</h3>
              <p>Preview of loaded EJ sessions ready for ensemble model training</p>
            </div>
            <div className="session-count-badge">{ejSessions.length} sessions loaded</div>
          </div>
          
          <div className="sessions-preview-container">
            <div className="sessions-grid">
              {ejSessions.slice(0, 3).map((session, index) => (
                <div key={index} className="session-preview-card">
                  <div className="session-header">
                    <span className="session-number">📄 Session {index + 1}</span>
                    <span className="session-length">{session.length} chars</span>
                  </div>
                  <div className="session-content">
                    <pre>{session.substring(0, 150)}...</pre>
                  </div>
                </div>
              ))}
            </div>
            
            {ejSessions.length > 3 && (
              <div className="more-sessions-indicator">
                <span>📚 + {ejSessions.length - 3} more sessions ready for training</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Ensemble Configuration */}
      <div className="section">
        <div className="section-title">
          <div>
            <h3>⚙️ Model Configuration</h3>
            <p>Configure the ensemble model parameters for optimal anomaly detection performance</p>
          </div>
        </div>
        
        <div className="config-panel">
          <div className="config-group">
            <div className="config-item">
              <div className="config-header">
                <label>📝 Text Analysis Weight</label>
                <span className="config-value">{config.textWeight.toFixed(1)}</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={config.textWeight}
                onChange={(e) => updateConfig('textWeight', e.target.value)}
                className="range-slider"
              />
              <div className="config-description">
                Weight for TF-IDF text analysis and pattern recognition in error terminology and transaction flows
              </div>
            </div>

            <div className="config-item">
              <div className="config-header">
                <label>📊 Statistical Analysis Weight</label>
                <span className="config-value">{config.statisticalWeight.toFixed(1)}</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={config.statisticalWeight}
                onChange={(e) => updateConfig('statisticalWeight', e.target.value)}
                className="range-slider"
              />
              <div className="config-description">
                Weight for session length, timing patterns, error counts, and numerical features analysis
              </div>
            </div>

            <div className="config-item">
              <div className="config-header">
                <label>🎯 Anomaly Threshold</label>
                <span className="config-value">{config.threshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={config.threshold}
                onChange={(e) => updateConfig('threshold', e.target.value)}
                className="range-slider"
              />
              <div className="config-description">
                Decision threshold for classifying sessions as anomalous - higher values reduce false positives
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Training Section */}
      <div className="section">
        <div className="section-title">
          <div>
            <h3>🚀 Model Training</h3>
            <p>Train the ensemble model with your loaded data using advanced machine learning techniques</p>
          </div>
        </div>
        
        <div className="training-panel">
          {ejSessions.length === 0 ? (
            <div className="training-placeholder">
              <div className="placeholder-icon">📥</div>
              <h4>No Training Data Loaded</h4>
              <p>Please load EJ sessions first to begin training the ensemble anomaly detection model</p>
              <div className="training-steps">
                <div className="training-step">
                  <span>1️⃣</span>
                  <span>Load training data from any source above</span>
                </div>
                <div className="training-step">
                  <span>2️⃣</span>
                  <span>Configure ensemble parameters</span>
                </div>
                <div className="training-step">
                  <span>3️⃣</span>
                  <span>Start training process</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="training-ready">
              <div className="training-info">
                <div className="info-item">
                  <span className="info-label">📄 Sessions Loaded:</span>
                  <span className="info-value">{ejSessions.length}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">📝 Text Weight:</span>
                  <span className="info-value">{config.textWeight.toFixed(1)}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">📊 Statistical Weight:</span>
                  <span className="info-value">{config.statisticalWeight.toFixed(1)}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">🎯 Decision Threshold:</span>
                  <span className="info-value">{config.threshold.toFixed(2)}</span>
                </div>
              </div>
              
              <button
                className={`btn btn-primary large-btn ${trainingStatus}`}
                onClick={trainModel}
                disabled={trainingStatus === 'training'}
              >
                {trainingStatus === 'training' ? '🔄 Training Ensemble Model...' : 
                 trainingStatus === 'completed' ? '✅ Training Complete - Model Ready' :
                 trainingStatus === 'failed' ? '❌ Training Failed - Retry' :
                 '🚀 Start Ensemble Training'}
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Training Results */}
      {trainingResults && (
        <div className="section">
          <div className="section-title">
            <div>
              <h3>📈 Training Results</h3>
              <p>Comprehensive analysis of the ensemble model training performance and capabilities</p>
            </div>
            <div className="success-badge">✅ Training Successful</div>
          </div>
          
          <div className="results-dashboard">
            <div className="results-grid">
              <div className="result-card primary">
                <div className="result-icon">📁</div>
                <div className="result-content">
                  <div className="result-value">{trainingResults.num_training_sessions}</div>
                  <div className="result-label">Training Sessions</div>
                </div>
              </div>

              <div className="result-card">
                <div className="result-icon">📝</div>
                <div className="result-content">
                  <div className="result-value">{trainingResults.text_feature_dims}</div>
                  <div className="result-label">Text Features</div>
                </div>
              </div>

              <div className="result-card">
                <div className="result-icon">📊</div>
                <div className="result-content">
                  <div className="result-value">{trainingResults.numerical_feature_dims}</div>
                  <div className="result-label">Statistical Features</div>
                </div>
              </div>

              <div className="result-card">
                <div className="result-icon">🎯</div>
                <div className="result-content">
                  <div className="result-value">{(trainingResults.avg_ensemble_score * 100).toFixed(1)}%</div>
                  <div className="result-label">Baseline Score</div>
                </div>
              </div>
            </div>

            <div className="training-summary">
              <div className="summary-header">
                <h4>🎉 Ensemble Model Training Complete!</h4>
              </div>
              <div className="summary-content">
                <p>
                  Your ensemble model has been successfully trained on <strong>{trainingResults.num_training_sessions}</strong> normal EJ sessions.
                  The model combines TF-IDF text analysis with statistical features for comprehensive anomaly detection across multiple dimensions.
                </p>
                <div className="next-steps">
                  <h5>📋 Next Steps:</h5>
                  <ul>
                    <li>✅ Switch to the <strong>Prediction</strong> tab to analyze new sessions for anomalies</li>
                    <li>🧪 Test with both normal and anomalous EJ sessions to validate detection accuracy</li>
                    <li>📊 Evaluate ensemble detection performance across different anomaly types</li>
                    <li>🔄 Monitor model performance and retrain as needed with new data</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Training;
