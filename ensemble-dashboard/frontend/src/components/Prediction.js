import React, { useState } from 'react';

const Prediction = ({ modelInfo }) => {
  const [inputText, setInputText] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [batchResults, setBatchResults] = useState(null);
  const [predictionStatus, setPredictionStatus] = useState('idle');

  const analyzeSingleSession = async () => {
    if (!inputText.trim()) {
      alert('Please enter EJ session text');
      return;
    }

    if (!modelInfo?.is_trained) {
      alert('Model not trained. Please train the model first.');
      return;
    }

    setPredictionStatus('analyzing');
    try {
      const formData = new FormData();
      formData.append('text', inputText);

      const response = await fetch(`${process.env.REACT_APP_API_URL || 'http://localhost:8001'}/api/predict`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (data.success) {
        setPrediction(data.prediction);
        setPredictionStatus('completed');
      } else {
        throw new Error(data.detail || 'Prediction failed');
      }
    } catch (error) {
      console.error('Prediction failed:', error);
      setPredictionStatus('failed');
      alert(`Prediction failed: ${error.message}`);
    }
  };

  const analyzeBatchSessions = async (file) => {
    if (!modelInfo?.is_trained) {
      alert('Model not trained. Please train the model first.');
      return;
    }

    setPredictionStatus('analyzing');
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${process.env.REACT_APP_API_URL || 'http://localhost:8001'}/api/batch_predict`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (data.success) {
        setBatchResults(data);
        setPredictionStatus('completed');
      } else {
        throw new Error(data.detail || 'Batch prediction failed');
      }
    } catch (error) {
      console.error('Batch prediction failed:', error);
      setPredictionStatus('failed');
      alert(`Batch prediction failed: ${error.message}`);
    }
  };

  const loadSampleAnomaly = () => {
    const sampleAnomaly = `SESSION START
POWER-UP/RESET
HARDWARE ERROR - CARD READER MALFUNCTION  
HARDWAREERROR DETECTED
RECOVERY FAILED - UNABLE TO INITIALIZE
CAPTURE FAILED - CARD TRAPPED
CIM-RESET INITIATED
CUSTOMER CANCELLED
TRANSACTION TERMINATED
DEVICE OFFLINE
SESSION END`;
    
    setInputText(sampleAnomaly);
  };

  const loadSampleNormal = () => {
    const sampleNormal = `SESSION START
CARD INSERTED
PIN ENTERED
PIN VERIFIED
WITHDRAW SELECTED
AMOUNT ENTERED: $200
CASH DISPENSED
RECEIPT PRINTED
CARD EJECTED
SESSION END`;
    
    setInputText(sampleNormal);
  };

  const renderPredictionResult = () => {
    if (!prediction) return null;

    const isAnomaly = prediction.is_anomaly;
    const score = prediction.ensemble_score;
    const confidence = prediction.confidence;

    return (
      <div className={`prediction-result ${isAnomaly ? 'anomaly' : 'normal'}`}>
        <div className="prediction-header">
          <h3>
            {isAnomaly ? '🚨 ANOMALY DETECTED' : '✅ NORMAL SESSION'}
          </h3>
          <div className="prediction-score">
            Ensemble Score: <strong>{(score * 100).toFixed(1)}%</strong>
          </div>
          <div className="prediction-confidence">
            Confidence: <span className={`confidence ${confidence.toLowerCase()}`}>{confidence}</span>
          </div>
        </div>

        <div className="prediction-breakdown">
          <h4>🔍 Analysis Breakdown</h4>
          
          <div className="component-scores">
            <div className="component-score">
              <h5>📝 Text Analysis Component</h5>
              <div className="score-bar">
                <div 
                  className="score-fill text" 
                  style={{ width: `${prediction.text_anomaly_score * 100}%` }}
                ></div>
              </div>
              <div className="score-details">
                <span>Score: {(prediction.text_anomaly_score * 100).toFixed(1)}%</span>
                <span>Weight: {(prediction.prediction_breakdown.text_component.weight * 100).toFixed(0)}%</span>
                <span>Contribution: {(prediction.prediction_breakdown.text_component.contribution * 100).toFixed(1)}%</span>
              </div>
            </div>

            <div className="component-score">
              <h5>📊 Statistical Analysis Component</h5>
              <div className="score-bar">
                <div 
                  className="score-fill statistical" 
                  style={{ width: `${prediction.statistical_anomaly_score * 100}%` }}
                ></div>
              </div>
              <div className="score-details">
                <span>Score: {(prediction.statistical_anomaly_score * 100).toFixed(1)}%</span>
                <span>Weight: {(prediction.prediction_breakdown.statistical_component.weight * 100).toFixed(0)}%</span>
                <span>Contribution: {(prediction.prediction_breakdown.statistical_component.contribution * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>

          <div className="feature-analysis">
            <h5>📈 Key Features Detected</h5>
            <div className="features-grid">
              {Object.entries(prediction.text_features).map(([key, value]) => (
                <div key={key} className="feature-item">
                  <span className="feature-name">{key.replace('_', ' ').toUpperCase()}</span>
                  <span className="feature-value">{value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="prediction-interpretation">
          <h4>💡 Interpretation</h4>
          {isAnomaly ? (
            <div className="anomaly-explanation">
              <p><strong>This session shows anomalous patterns:</strong></p>
              <ul>
                {prediction.text_anomaly_score > 0.6 && (
                  <li>🔤 Unusual text patterns detected (error terminology, hardware failures)</li>
                )}
                {prediction.statistical_anomaly_score > 0.6 && (
                  <li>📊 Statistical outliers in session structure and error counts</li>
                )}
                {prediction.text_features.critical_hardware_patterns > 0 && (
                  <li>⚠️ Critical hardware patterns found</li>
                )}
                {prediction.text_features.error_count > 2 && (
                  <li>🚫 High error count detected</li>
                )}
              </ul>
            </div>
          ) : (
            <div className="normal-explanation">
              <p><strong>This session appears normal:</strong></p>
              <ul>
                <li>✅ Text patterns match normal transaction flow</li>
                <li>📊 Statistical features within normal ranges</li>
                <li>🎯 Ensemble score below anomaly threshold ({prediction.threshold})</li>
              </ul>
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderBatchResults = () => {
    if (!batchResults) return null;

    const anomalies = batchResults.predictions.filter(p => p.is_anomaly);
    const normals = batchResults.predictions.filter(p => !p.is_anomaly);

    return (
      <div className="batch-results">
        <div className="batch-summary">
          <h3>📊 Batch Analysis Results</h3>
          <div className="summary-stats">
            <div className="stat">
              <span className="stat-value">{batchResults.total_sessions}</span>
              <span className="stat-label">Total Sessions</span>
            </div>
            <div className="stat anomaly">
              <span className="stat-value">{anomalies.length}</span>
              <span className="stat-label">Anomalies</span>
            </div>
            <div className="stat normal">
              <span className="stat-value">{normals.length}</span>
              <span className="stat-label">Normal</span>
            </div>
            {batchResults.metrics && (
              <div className="stat">
                <span className="stat-value">{(batchResults.metrics.accuracy * 100).toFixed(1)}%</span>
                <span className="stat-label">Accuracy</span>
              </div>
            )}
          </div>
        </div>

        <div className="batch-details">
          <div className="session-list">
            <h4>🚨 Detected Anomalies ({anomalies.length})</h4>
            {anomalies.length === 0 ? (
              <p>No anomalies detected in the batch</p>
            ) : (
              <div className="sessions">
                {anomalies.map((pred, index) => (
                  <div key={index} className="session-item anomaly">
                    <div className="session-header">
                      <span className="session-id">Session {index + 1}</span>
                      <span className="session-score">{(pred.ensemble_score * 100).toFixed(1)}%</span>
                      <span className={`session-confidence ${pred.confidence.toLowerCase()}`}>
                        {pred.confidence}
                      </span>
                    </div>
                    <div className="session-preview">
                      {pred.session_text.substring(0, 150)}...
                    </div>
                    <div className="session-components">
                      <span>Text: {(pred.text_anomaly_score * 100).toFixed(1)}%</span>
                      <span>Statistical: {(pred.statistical_anomaly_score * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="session-list">
            <h4>✅ Normal Sessions ({normals.length})</h4>
            {normals.slice(0, 5).map((pred, index) => (
              <div key={index} className="session-item normal">
                <div className="session-header">
                  <span className="session-id">Session {index + 1}</span>
                  <span className="session-score">{(pred.ensemble_score * 100).toFixed(1)}%</span>
                </div>
                <div className="session-preview">
                  {pred.session_text.substring(0, 100)}...
                </div>
              </div>
            ))}
            {normals.length > 5 && (
              <div className="more-sessions">... and {normals.length - 5} more normal sessions</div>
            )}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="prediction-container">
      <div className="prediction-header">
        <h2>🔍 Anomaly Prediction</h2>
        <p>Analyze EJ sessions for anomalies using the trained ensemble model</p>
      </div>

      {!modelInfo?.is_trained && (
        <div className="warning-message">
          <h3>⚠️ Model Not Trained</h3>
          <p>Please train the ensemble model first in the Training tab before making predictions.</p>
        </div>
      )}

      {modelInfo?.is_trained && (
        <>
          <div className="section">
            <h3>📝 Single Session Analysis</h3>
            
            <div className="sample-buttons">
              <button className="sample-btn normal" onClick={loadSampleNormal}>
                Load Normal Sample
              </button>
              <button className="sample-btn anomaly" onClick={loadSampleAnomaly}>
                Load Anomaly Sample
              </button>
            </div>

            <textarea
              className="session-input"
              rows={12}
              placeholder="Enter EJ session text here..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
            />

            <div className="prediction-controls">
              <button
                className={`analyze-btn ${predictionStatus}`}
                onClick={analyzeSingleSession}
                disabled={predictionStatus === 'analyzing'}
              >
                {predictionStatus === 'analyzing' ? '🔄 Analyzing...' : '🔍 Analyze Session'}
              </button>
            </div>

            {prediction && renderPredictionResult()}
          </div>

          <div className="section">
            <h3>📁 Batch Analysis</h3>
            <p>Upload a CSV file or raw EJ log text for batch anomaly detection</p>
            
            <input
              type="file"
              accept=".csv,.txt"
              onChange={(e) => e.target.files[0] && analyzeBatchSessions(e.target.files[0])}
              className="file-input"
            />

            {batchResults && renderBatchResults()}
          </div>
        </>
      )}
    </div>
  );
};

export default Prediction;
