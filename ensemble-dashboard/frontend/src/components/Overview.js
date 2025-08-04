import React, { useState, useEffect } from 'react';

const Overview = ({ modelInfo, onRefresh }) => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (modelInfo?.is_trained) {
      fetchTrainingStats();
    }
  }, [modelInfo]);

  const fetchTrainingStats = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${process.env.REACT_APP_API_URL || 'http://localhost:8001'}/api/training_stats`);
      const data = await response.json();
      if (data.success) {
        setStats(data.stats);
      }
    } catch (error) {
      console.error('Failed to fetch training stats:', error);
    } finally {
      setLoading(false);
    }
  };

  const renderModelStatus = () => {
    if (!modelInfo) {
      return (
        <div className="status-card error">
          <h3>❌ Model Status</h3>
          <p>Unable to connect to ensemble model</p>
        </div>
      );
    }

    if (!modelInfo.is_trained) {
      return (
        <div className="status-card warning">
          <h3>⚠️ Model Status</h3>
          <p>Ensemble model is not trained yet</p>
          <p>Go to the Training tab to train the model with EJ session data</p>
        </div>
      );
    }

    return (
      <div className="status-card success">
        <h3>✅ Model Status</h3>
        <p>Ensemble model is trained and ready for predictions</p>
        <div className="model-config">
          <h4>Current Configuration:</h4>
          <ul>
            <li>Text Weight: {modelInfo.ensemble_config?.text_weight || 0.6}</li>
            <li>Statistical Weight: {modelInfo.ensemble_config?.statistical_weight || 0.4}</li>
            <li>Threshold: {modelInfo.ensemble_config?.threshold || 0.5}</li>
          </ul>
        </div>
      </div>
    );
  };

  const renderTrainingStats = () => {
    if (!stats) return null;

    return (
      <div className="stats-grid">
        <div className="stat-card">
          <h3>📁 Training Data</h3>
          <div className="stat-value">{stats.num_training_sessions}</div>
          <div className="stat-label">Normal Sessions</div>
        </div>

        <div className="stat-card">
          <h3>📝 Text Features</h3>
          <div className="stat-value">{stats.text_feature_dims}</div>
          <div className="stat-label">Dimensions</div>
        </div>

        <div className="stat-card">
          <h3>📊 Statistical Features</h3>
          <div className="stat-value">{stats.numerical_feature_dims}</div>
          <div className="stat-label">Features</div>
        </div>

        <div className="stat-card">
          <h3>🎯 Ensemble Score</h3>
          <div className="stat-value">{(stats.avg_ensemble_score * 100).toFixed(1)}%</div>
          <div className="stat-label">Avg Normal Score</div>
        </div>
      </div>
    );
  };

  const renderFeatureBreakdown = () => {
    if (!stats?.feature_names) return null;

    return (
      <div className="feature-breakdown">
        <h3>🔍 Feature Analysis</h3>
        <div className="feature-grid">
          <div className="feature-category">
            <h4>Session Structure</h4>
            <ul>
              {stats.feature_names.filter(name => 
                ['line_count', 'total_chars', 'avg_line_length', 'empty_lines'].includes(name)
              ).map(name => (
                <li key={name}>{name.replace('_', ' ').toUpperCase()}</li>
              ))}
            </ul>
          </div>

          <div className="feature-category">
            <h4>Error Patterns</h4>
            <ul>
              {stats.feature_names.filter(name => 
                ['error_count', 'fail_count', 'malfunction_count', 'timeout_count'].includes(name)
              ).map(name => (
                <li key={name}>{name.replace('_', ' ').toUpperCase()}</li>
              ))}
            </ul>
          </div>

          <div className="feature-category">
            <h4>Hardware Patterns</h4>
            <ul>
              {stats.feature_names.filter(name => 
                ['hardware_mentions', 'power_reset_count', 'critical_hardware_patterns'].includes(name)
              ).map(name => (
                <li key={name}>{name.replace('_', ' ').toUpperCase()}</li>
              ))}
            </ul>
          </div>

          <div className="feature-category">
            <h4>Success Indicators</h4>
            <ul>
              {stats.feature_names.filter(name => 
                ['success_indicators', 'network_errors', 'cash_errors'].includes(name)
              ).map(name => (
                <li key={name}>{name.replace('_', ' ').toUpperCase()}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="overview-container">
      <div className="overview-header">
        <h2>📊 Ensemble Model Overview</h2>
        <button className="refresh-btn" onClick={onRefresh}>
          🔄 Refresh Status
        </button>
      </div>

      {renderModelStatus()}

      {loading && (
        <div className="loading-container">
          <div className="spinner"></div>
          <p>Loading training statistics...</p>
        </div>
      )}

      {modelInfo?.is_trained && stats && (
        <>
          <div className="section">
            <h3>📈 Training Statistics</h3>
            {renderTrainingStats()}
          </div>

          <div className="section">
            {renderFeatureBreakdown()}
          </div>

          <div className="section">
            <h3>ℹ️ How It Works</h3>
            <div className="info-grid">
              <div className="info-card">
                <h4>🔤 Text Analysis Component</h4>
                <p>
                  Uses TF-IDF vectorization to analyze text patterns, error terminology, 
                  and linguistic anomalies in EJ sessions. Weighted at {(modelInfo.ensemble_config?.text_weight || 0.6) * 100}%.
                </p>
              </div>

              <div className="info-card">
                <h4>📊 Statistical Analysis Component</h4>
                <p>
                  Analyzes numerical patterns like error counts, session structure, 
                  and hardware failure indicators. Weighted at {(modelInfo.ensemble_config?.statistical_weight || 0.4) * 100}%.
                </p>
              </div>

              <div className="info-card">
                <h4>🎯 Ensemble Decision</h4>
                <p>
                  Combines both components using weighted voting. Sessions with ensemble 
                  scores above {modelInfo.ensemble_config?.threshold || 0.5} are flagged as anomalies.
                </p>
              </div>

              <div className="info-card">
                <h4>🔍 Unknown Anomaly Detection</h4>
                <p>
                  The model can detect completely new anomaly types it has never seen before 
                  by identifying deviations from learned normal patterns.
                </p>
              </div>
            </div>
          </div>
        </>
      )}

      {!modelInfo?.is_trained && (
        <div className="getting-started">
          <h3>🚀 Getting Started</h3>
          <div className="steps">
            <div className="step">
              <span className="step-number">1</span>
              <div>
                <h4>Go to Training Tab</h4>
                <p>Click on the "Training" tab above to start training the ensemble model</p>
              </div>
            </div>
            <div className="step">
              <span className="step-number">2</span>
              <div>
                <h4>Load EJ Sessions</h4>
                <p>Upload a CSV file with EJ session data or paste raw EJ logs</p>
              </div>
            </div>
            <div className="step">
              <span className="step-number">3</span>
              <div>
                <h4>Train Model</h4>
                <p>Train the ensemble on normal EJ sessions to learn baseline patterns</p>
              </div>
            </div>
            <div className="step">
              <span className="step-number">4</span>
              <div>
                <h4>Make Predictions</h4>
                <p>Use the Prediction tab to analyze new EJ sessions for anomalies</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Overview;
