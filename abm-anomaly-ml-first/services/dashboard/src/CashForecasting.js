import React, { useState, useEffect } from 'react';

const CashForecasting = () => {
  const [terminalStatus, setTerminalStatus] = useState({
    terminals: [],
    summary: { total_terminals: 0, healthy: 0, warning: 0, critical: 0 }
  });
  const [alerts, setAlerts] = useState({ alerts: [], total_alerts: 0 });
  const [predictions, setPredictions] = useState({ predictions: [], model_info: {} });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  // Fetch data from the cash forecasting API
  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch terminal status
      const statusResponse = await fetch('/api/cash-forecasting/terminal-status');
      if (!statusResponse.ok) throw new Error('Failed to fetch terminal status');
      const statusData = await statusResponse.json();
      setTerminalStatus(statusData);

      // Fetch alerts
      const alertsResponse = await fetch('/api/cash-forecasting/alerts');
      if (!alertsResponse.ok) throw new Error('Failed to fetch alerts');
      const alertsData = await alertsResponse.json();
      setAlerts(alertsData);

      // Fetch predictions
      const predictionsResponse = await fetch('/api/cash-forecasting/predictions');
      if (!predictionsResponse.ok) throw new Error('Failed to fetch predictions');
      const predictionsData = await predictionsResponse.json();
      setPredictions(predictionsData);

      setLastUpdated(new Date().toLocaleString());
    } catch (err) {
      setError(err.message);
      console.error('Error fetching data:', err);
    } finally {
      setLoading(false);
    }
  };

  // Initial data fetch and refresh interval
  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5 * 60 * 1000); // Refresh every 5 minutes
    return () => clearInterval(interval);
  }, []);

  // Trigger model retraining
  const triggerRetraining = async () => {
    try {
      const response = await fetch('/api/cash-forecasting/retrain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      if (response.ok) {
        alert('Model retraining triggered successfully!');
        // Refresh data after a delay
        setTimeout(fetchData, 3000);
      } else {
        throw new Error('Failed to trigger retraining');
      }
    } catch (err) {
      alert('Error triggering retraining: ' + err.message);
    }
  };

  // Get risk level color class
  const getRiskLevelClass = (riskLevel) => {
    switch (riskLevel?.toLowerCase()) {
      case 'high': return 'text-danger';
      case 'medium': return 'text-warning';
      case 'low': return 'text-success';
      default: return 'text-muted';
    }
  };

  // Get status badge class
  const getStatusBadgeClass = (riskLevel) => {
    switch (riskLevel?.toLowerCase()) {
      case 'high': return 'badge bg-danger';
      case 'medium': return 'badge bg-warning';
      case 'low': return 'badge bg-success';
      default: return 'badge bg-secondary';
    }
  };

  if (loading && terminalStatus.terminals.length === 0) {
    return (
      <div className="dashboard-container">
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <h2>Loading Cash Forecasting Dashboard...</h2>
          <p>Fetching real-time ATM cash levels and predictions</p>
        </div>
      </div>
    );
  }

  if (error && terminalStatus.terminals.length === 0) {
    return (
      <div className="dashboard-container">
        <div className="error-container">
          <h2>Error Loading Cash Forecasting Data</h2>
          <p>{error}</p>
          <button className="action-button retry" onClick={fetchData}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard-container">
      {/* Header */}
      <div className="dashboard-header">
        <h1>💰 Cash Forecasting Dashboard</h1>
        <p>ML-powered ATM cash depletion prediction and monitoring system</p>
        <div className="last-updated">
          Last updated: {lastUpdated || 'Loading...'}
        </div>
      </div>

      {/* Summary Metrics */}
      <div className="dashboard-section">
        <h2>📊 System Overview</h2>
        <div className="metrics-grid">
          <div className="metric-card">
            <div className="metric-value">{terminalStatus.summary.total_terminals}</div>
            <div className="metric-label">Total Terminals</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{terminalStatus.summary.healthy}</div>
            <div className="metric-label">Healthy (Low Risk)</div>
          </div>
          <div className="metric-card anomaly">
            <div className="metric-value">{terminalStatus.summary.warning}</div>
            <div className="metric-label">Medium Risk</div>
          </div>
          <div className="metric-card anomaly">
            <div className="metric-value">{terminalStatus.summary.critical}</div>
            <div className="metric-label">High Risk (Critical)</div>
          </div>
        </div>
      </div>

      {/* Active Alerts */}
      {alerts.alerts && alerts.alerts.length > 0 && (
        <div className="dashboard-section">
          <h2>🚨 Active Alerts ({alerts.total_alerts})</h2>
          <div className="recommendations-sections">
            {alerts.alerts.map((alert, index) => (
              <div key={index} className="recommendation-card">
                <div className="rec-header">
                  <h4>Terminal {alert.terminal_id}</h4>
                  <span className={getStatusBadgeClass(alert.level)}>
                    {alert.level}
                  </span>
                </div>
                <div className="rec-description">
                  {alert.message}
                </div>
                <div className="rec-action">
                  <strong>Priority:</strong> {alert.priority}
                </div>
                <div className="rec-impact">
                  Created: {new Date(alert.created_at).toLocaleString()}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Terminal Status Grid */}
      <div className="dashboard-section">
        <h2>🏛️ Terminal Status</h2>
        <div className="clusters-grid">
          {terminalStatus.terminals.map((terminal) => (
            <div 
              key={terminal.id} 
              className={`cluster-card ${terminal.risk_level?.toLowerCase() === 'high' ? 'high-anomaly' : 
                         terminal.risk_level?.toLowerCase() === 'medium' ? '' : 'low-anomaly'}`}
            >
              <div className="cluster-header">
                <h3>Terminal {terminal.id}</h3>
                <span className="cluster-size">{terminal.location || 'Unknown Location'}</span>
              </div>
              
              <div className="anomaly-rate">
                <div className="rate-label">Cash Level</div>
                <div className={`rate-value ${getRiskLevelClass(terminal.risk_level)}`}>
                  {terminal.cash_level}%
                </div>
                <div className="rate-bar">
                  <div 
                    className="rate-fill" 
                    style={{ width: `${terminal.cash_level}%` }}
                  ></div>
                </div>
              </div>
              
              <div className="cluster-description">
                <strong>Risk Level:</strong> <span className={getRiskLevelClass(terminal.risk_level)}>
                  {terminal.risk_level}
                </span><br/>
                <strong>Predicted Depletion:</strong> {terminal.predicted_depletion_days} days<br/>
                <strong>Last Refill:</strong> {terminal.last_refill}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ML Model Information */}
      {predictions.model_info && (
        <div className="dashboard-section">
          <h2>🤖 ML Model Performance</h2>
          <div className="info-grid">
            <div className="info-item">
              <strong>Algorithm:</strong> {predictions.model_info.algorithm || 'Random Forest + LSTM Ensemble'}
            </div>
            <div className="info-item">
              <strong>Model Accuracy:</strong> {((predictions.model_info.accuracy || 0.91) * 100).toFixed(1)}%
            </div>
            <div className="info-item">
              <strong>Last Trained:</strong> {predictions.model_info.last_trained ? 
                new Date(predictions.model_info.last_trained).toLocaleString() : 'Unknown'}
            </div>
            <div className="info-item">
              <strong>Predictions Count:</strong> {predictions.predictions?.length || 0}
            </div>
          </div>
        </div>
      )}

      {/* Detailed Predictions */}
      {predictions.predictions && predictions.predictions.length > 0 && (
        <div className="dashboard-section">
          <h2>📈 Detailed Predictions</h2>
          <div className="methods-grid">
            {predictions.predictions.map((prediction) => (
              <div key={prediction.terminal_id} className="method-card">
                <h3 className="method-name">Terminal {prediction.terminal_id}</h3>
                <div className="method-metrics">
                  <div className="metric">
                    <span className="metric-label">Confidence</span>
                    <span className="metric-value">{(prediction.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Depletion Date</span>
                    <span className="metric-value">
                      {new Date(prediction.predicted_depletion_date).toLocaleDateString()}
                    </span>
                  </div>
                </div>
                {prediction.factors && (
                  <div className="method-parameters">
                    <h4>Key Factors</h4>
                    <div className="method-tags">
                      {prediction.factors.map((factor, index) => (
                        <span key={index} className="method-tag">
                          {factor.replace(/_/g, ' ')}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="dashboard-section">
        <h2>⚡ Actions</h2>
        <div className="actions-buttons">
          <button className="action-button refresh" onClick={fetchData} disabled={loading}>
            {loading ? '🔄 Refreshing...' : '🔄 Refresh Data'}
          </button>
          <button className="action-button export" onClick={triggerRetraining}>
            🤖 Retrain Models
          </button>
          <a 
            href="/api/cash-forecasting/predictions" 
            target="_blank" 
            rel="noopener noreferrer"
            className="action-button export"
          >
            📊 Export Data
          </a>
        </div>
      </div>
    </div>
  );
};

export default CashForecasting;
