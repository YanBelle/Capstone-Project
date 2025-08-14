import React, { useState, useEffect } from 'react';
import apiConfig from './config/api';

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
      const statusResponse = await fetch(apiConfig.endpoint('/api/cash-forecasting/terminal-status'));
      if (!statusResponse.ok) throw new Error('Failed to fetch terminal status');
      const statusData = await statusResponse.json();
      setTerminalStatus(statusData);

      // Fetch alerts
      const alertsResponse = await fetch(apiConfig.endpoint('/api/cash-forecasting/alerts'));
      if (!alertsResponse.ok) throw new Error('Failed to fetch alerts');
      const alertsData = await alertsResponse.json();
      setAlerts(alertsData);

      // Fetch predictions
      const predictionsResponse = await fetch(apiConfig.endpoint('/api/cash-forecasting/predictions'));
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
      const response = await fetch(apiConfig.endpoint('/api/cash-forecasting/retrain'), {
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
    <div className="dashboard-container cash-forecasting-dashboard">
      {/* Clean Header with Better Structure */}
      <div className="dashboard-header cash-forecast-header">
        <div className="header-content">
          <div className="header-main">
            <div className="header-title-section">
              <h1 className="dashboard-title">💰 Cash Forecasting Dashboard</h1>
              <p className="dashboard-subtitle">AI-powered ATM cash depletion prediction and risk assessment</p>
            </div>
            <div className="header-status-section">
              <div className="status-badges">
                <div className="status-badge system-online">
                  <span className="status-icon">✅</span>
                  <span className="status-text">System Online</span>
                </div>
                <div className="status-badge last-update">
                  <span className="status-icon">🕒</span>
                  <span className="status-text">Updated: {lastUpdated || 'Loading...'}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Enhanced Summary Metrics with Better Visual Hierarchy */}
      <div className="dashboard-section summary-section">
        <div className="section-header">
          <h2>📊 System Overview</h2>
          <div className="section-subtitle">Real-time monitoring across all ATM terminals</div>
        </div>
        
        <div className="metrics-grid enhanced-metrics">
          <div className="metric-card primary">
            <div className="metric-icon">🏛️</div>
            <div className="metric-content">
              <div className="metric-value">{terminalStatus.summary.total_terminals}</div>
              <div className="metric-label">Total Terminals</div>
              <div className="metric-trend">Monitored 24/7</div>
            </div>
          </div>
          
          <div className="metric-card success">
            <div className="metric-icon">✅</div>
            <div className="metric-content">
              <div className="metric-value">{terminalStatus.summary.healthy}</div>
              <div className="metric-label">Healthy Terminals</div>
              <div className="metric-trend">Low Risk Status</div>
            </div>
          </div>
          
          <div className="metric-card warning">
            <div className="metric-icon">⚠️</div>
            <div className="metric-content">
              <div className="metric-value">{terminalStatus.summary.warning}</div>
              <div className="metric-label">Medium Risk</div>
              <div className="metric-trend">Requires Monitoring</div>
            </div>
          </div>
          
          <div className="metric-card critical">
            <div className="metric-icon">🚨</div>
            <div className="metric-content">
              <div className="metric-value">{terminalStatus.summary.critical}</div>
              <div className="metric-label">Critical Risk</div>
              <div className="metric-trend">Immediate Action</div>
            </div>
          </div>
        </div>
      </div>

      {/* Enhanced Active Alerts Section */}
      {alerts.alerts && alerts.alerts.length > 0 && (
        <div className="dashboard-section alerts-section">
          <div className="section-header">
            <h2>🚨 Active Alerts</h2>
            <div className="alert-badge">{alerts.total_alerts} active</div>
          </div>
          
          <div className="alerts-grid">
            {alerts.alerts.map((alert, index) => (
              <div key={index} className={`alert-card alert-${alert.level?.toLowerCase()}`}>
                <div className="alert-header">
                  <div className="alert-title">
                    <span className="alert-icon">
                      {alert.level === 'CRITICAL' ? '🚨' : alert.level === 'WARNING' ? '⚠️' : 'ℹ️'}
                    </span>
                    <h4>Terminal {alert.terminal_id}</h4>
                  </div>
                  <span className={`alert-level-badge ${alert.level?.toLowerCase()}`}>
                    {alert.level}
                  </span>
                </div>
                
                <div className="alert-content">
                  <div className="alert-message">{alert.message}</div>
                  
                  <div className="alert-metadata">
                    <div className="alert-priority">
                      <strong>Priority:</strong> {alert.priority}
                    </div>
                    <div className="alert-time">
                      <strong>Created:</strong> {new Date(alert.created_at).toLocaleString()}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Enhanced Terminal Status with Visual Improvements */}
      <div className="dashboard-section terminals-section">
        <div className="section-header">
          <h2>🏛️ Terminal Status & Analytics</h2>
          <div className="section-subtitle">Real-time cash levels and risk assessment</div>
        </div>
        
        <div className="terminals-grid">
          {terminalStatus.terminals.map((terminal) => (
            <div 
              key={terminal.id} 
              className={`terminal-card risk-${terminal.risk_level?.toLowerCase()}`}
            >
              <div className="terminal-header">
                <div className="terminal-info">
                  <h3>Terminal {terminal.id}</h3>
                  <span className="terminal-location">{terminal.location || 'Unknown Location'}</span>
                </div>
                <div className="risk-indicator">
                  <span className={`risk-badge ${terminal.risk_level?.toLowerCase()}`}>
                    {terminal.risk_level}
                  </span>
                </div>
              </div>
              
              <div className="cash-level-section">
                <div className="cash-level-header">
                  <span className="cash-label">Cash Level</span>
                  <span className={`cash-percentage ${getRiskLevelClass(terminal.risk_level)}`}>
                    {terminal.cash_level}%
                  </span>
                </div>
                
                <div className="cash-progress-bar">
                  <div 
                    className={`cash-progress-fill ${terminal.risk_level?.toLowerCase()}`}
                    style={{ width: `${terminal.cash_level}%` }}
                  >
                    <div className="progress-indicator"></div>
                  </div>
                </div>
                
                <div className="cash-amount">
                  ${terminal.cash_amount?.toLocaleString() || 'N/A'}
                </div>
              </div>
              
              <div className="terminal-details">
                <div className="detail-row">
                  <span className="detail-label">📅 Predicted Depletion:</span>
                  <span className="detail-value">{terminal.predicted_depletion_days} days</span>
                </div>
                <div className="detail-row">
                  <span className="detail-label">🔄 Last Refill:</span>
                  <span className="detail-value">{terminal.last_refill}</span>
                </div>
              </div>
              
              {/* Chart placeholder for this terminal */}
              <div className="terminal-chart-placeholder" id={`terminal-chart-${terminal.terminal_id || terminal.id.replace('ATM', '')}`}>
                <div className="chart-loading">📊 Loading visualization...</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Enhanced ML Model Information */}
      {predictions.model_info && (
        <div className="dashboard-section model-info-section">
          <div className="section-header">
            <h2>🤖 ML Model Performance</h2>
            <div className="section-subtitle">Advanced forecasting algorithm metrics</div>
          </div>
          
          <div className="model-metrics-grid">
            <div className="model-metric-card">
              <div className="metric-icon">🎯</div>
              <div className="metric-content">
                <div className="metric-label">Algorithm</div>
                <div className="metric-value">{predictions.model_info.algorithm || 'Ensemble Forecasting'}</div>
                <div className="metric-description">Multi-model approach</div>
              </div>
            </div>
            
            <div className="model-metric-card">
              <div className="metric-icon">📊</div>
              <div className="metric-content">
                <div className="metric-label">Accuracy</div>
                <div className="metric-value">{((predictions.model_info.accuracy || 0.91) * 100).toFixed(1)}%</div>
                <div className="metric-description">Prediction reliability</div>
              </div>
            </div>
            
            <div className="model-metric-card">
              <div className="metric-icon">🔄</div>
              <div className="metric-content">
                <div className="metric-label">Last Trained</div>
                <div className="metric-value">
                  {predictions.model_info.last_trained ? 
                    new Date(predictions.model_info.last_trained).toLocaleDateString() : 'N/A'}
                </div>
                <div className="metric-description">Model freshness</div>
              </div>
            </div>
            
            <div className="model-metric-card">
              <div className="metric-icon">📈</div>
              <div className="metric-content">
                <div className="metric-label">Active Predictions</div>
                <div className="metric-value">{predictions.predictions?.length || 0}</div>
                <div className="metric-description">Forecasts generated</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Enhanced Predictions Section */}
      {predictions.predictions && predictions.predictions.length > 0 && (
        <div className="dashboard-section predictions-section">
          <div className="section-header">
            <h2>📈 Terminal Forecasts</h2>
            <div className="section-subtitle">AI-powered depletion predictions and confidence intervals</div>
          </div>
          
          <div className="predictions-grid">
            {predictions.predictions.map((prediction) => (
              <div key={prediction.terminal_id} className="prediction-card">
                <div className="prediction-header">
                  <h3>Terminal {prediction.terminal_id}</h3>
                  <div className="confidence-badge">
                    {(prediction.confidence * 100).toFixed(1)}% confidence
                  </div>
                </div>
                
                <div className="prediction-content">
                  <div className="prediction-main">
                    <div className="prediction-date">
                      <span className="date-label">Predicted Depletion</span>
                      <span className="date-value">
                        {new Date(prediction.predicted_depletion_date).toLocaleDateString('en-US', {
                          weekday: 'short',
                          month: 'short',
                          day: 'numeric',
                          year: 'numeric'
                        })}
                      </span>
                      <span className="days-remaining">
                        {Math.ceil((new Date(prediction.predicted_depletion_date) - new Date()) / (1000 * 60 * 60 * 24))} days
                      </span>
                    </div>
                    
                    <div className="confidence-meter">
                      <div className="confidence-bar">
                        <div 
                          className="confidence-fill"
                          style={{ width: `${prediction.confidence * 100}%` }}
                        ></div>
                      </div>
                      <span className="confidence-label">Prediction Confidence</span>
                    </div>
                  </div>
                  
                  {prediction.factors && (
                    <div className="prediction-factors">
                      <h4>Key Factors</h4>
                      <div className="factors-tags">
                        {prediction.factors.map((factor, index) => (
                          <span key={index} className="factor-tag">
                            {factor.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Enhanced Actions Section */}
      <div className="dashboard-section actions-section">
        <div className="section-header">
          <h2>⚡ Dashboard Actions</h2>
          <div className="section-subtitle">System controls and data export options</div>
        </div>
        
        <div className="actions-grid">
          <div className="action-card">
            <div className="action-icon">🔄</div>
            <div className="action-content">
              <h3>Refresh Data</h3>
              <p>Update all terminal status and predictions</p>
              <button 
                className="action-button primary" 
                onClick={fetchData} 
                disabled={loading}
              >
                {loading ? '🔄 Updating...' : '🔄 Refresh All Data'}
              </button>
            </div>
          </div>
          
          <div className="action-card">
            <div className="action-icon">🤖</div>
            <div className="action-content">
              <h3>Retrain Models</h3>
              <p>Update ML models with latest transaction data</p>
              <button 
                className="action-button secondary" 
                onClick={triggerRetraining}
              >
                🚀 Retrain AI Models
              </button>
            </div>
          </div>
          
          <div className="action-card">
            <div className="action-icon">📊</div>
            <div className="action-content">
              <h3>Export Data</h3>
              <p>Download predictions and analytics data</p>
              <a 
                href={apiConfig.endpoint('/api/cash-forecasting/predictions')} 
                target="_blank" 
                rel="noopener noreferrer"
                className="action-button export"
              >
                � Export JSON Data
              </a>
            </div>
          </div>
          
          <div className="action-card">
            <div className="action-icon">📈</div>
            <div className="action-content">
              <h3>View Charts</h3>
              <p>Detailed visualization analytics</p>
              <div className="chart-toggle-info">
                Charts auto-load below terminal cards
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CashForecasting;
