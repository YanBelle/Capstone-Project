import React, { useState, useEffect } from 'react';
import './Dashboard.css';
import apiConfig from './config/api';

const UnsupervisedAnalysisDashboard = () => {
  const [analysisData, setAnalysisData] = useState(null);
  const [methodComparison, setMethodComparison] = useState(null);
  const [clusteringResults, setClusteringResults] = useState(null);
  const [recommendations, setRecommendations] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = async () => {
    try {
      setLoading(true);
      
      // Fetch all unsupervised analysis data
      const [analysisRes, methodRes, clusterRes, recRes] = await Promise.all([
        fetch(apiConfig.endpoint('/api/v1/unsupervised/analysis-overview')),
        fetch(apiConfig.endpoint('/api/v1/unsupervised/method-comparison')),
        fetch(apiConfig.endpoint('/api/v1/unsupervised/clustering-results')),
        fetch(apiConfig.endpoint('/api/v1/unsupervised/recommendations'))
      ]);

      if (!analysisRes.ok || !methodRes.ok || !clusterRes.ok || !recRes.ok) {
        throw new Error('Failed to fetch unsupervised analysis data');
      }

      const [analysis, methods, clusters, recs] = await Promise.all([
        analysisRes.json(),
        methodRes.json(),
        clusterRes.json(),
        recRes.json()
      ]);

      setAnalysisData(analysis);
      setMethodComparison(methods);
      setClusteringResults(clusters);
      setRecommendations(recs);
      setError(null);
    } catch (err) {
      console.error('Error fetching unsupervised analysis data:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    
    // Refresh data every 30 seconds
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="dashboard-container">
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading Unsupervised Analysis Dashboard...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="dashboard-container">
        <div className="error-container">
          <h2>Error Loading Dashboard</h2>
          <p>{error}</p>
          <button onClick={fetchData} className="retry-button">
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <h1>🔍 Unsupervised Analysis Dashboard</h1>
        <p>Comprehensive analysis of unsupervised machine learning results</p>
        <div className="last-updated">
          Last Updated: {analysisData?.last_updated ? new Date(analysisData.last_updated).toLocaleString() : 'Unknown'}
        </div>
      </div>

      {/* Analysis Overview Section */}
      {analysisData && (
        <div className="dashboard-section">
          <h2>📊 Analysis Overview</h2>
          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-value">{analysisData.total_sequences?.toLocaleString()}</div>
              <div className="metric-label">Total Sequences</div>
            </div>
            <div className="metric-card anomaly">
              <div className="metric-value">{analysisData.anomalies_detected}</div>
              <div className="metric-label">Anomalies Detected</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">{analysisData.clusters_identified}</div>
              <div className="metric-label">Clusters Identified</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">{analysisData.confidence_score}</div>
              <div className="metric-label">Confidence Score</div>
            </div>
          </div>
          
          <div className="dataset-info">
            <h3>Dataset Information</h3>
            <div className="info-grid">
              <div className="info-item">
                <strong>Source:</strong> {analysisData.dataset_info?.source}
              </div>
              <div className="info-item">
                <strong>Sessions Processed:</strong> {analysisData.dataset_info?.sessions_processed?.toLocaleString()}
              </div>
              <div className="info-item">
                <strong>Time Range:</strong> {analysisData.dataset_info?.time_range}
              </div>
              <div className="info-item">
                <strong>Processing Time:</strong> {analysisData.processing_time}s
              </div>
            </div>
          </div>

          <div className="methods-used">
            <h3>Analysis Methods</h3>
            <div className="method-tags">
              {analysisData.analysis_methods?.map((method, index) => (
                <span key={index} className="method-tag">{method}</span>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Method Comparison Section */}
      {methodComparison && (
        <div className="dashboard-section">
          <h2>⚖️ Method Performance Comparison</h2>
          <div className="best-method">
            <strong>Best Performing Method:</strong> {methodComparison.best_method} 
            <span className="ensemble-score">(Ensemble Score: {methodComparison.ensemble_score})</span>
          </div>
          
          <div className="methods-grid">
            {methodComparison.methods?.map((method, index) => (
              <div key={index} className="method-card">
                <h3 className="method-name">{method.name}</h3>
                <div className="method-metrics">
                  <div className="metric">
                    <span className="metric-label">Precision:</span>
                    <span className="metric-value">{method.precision}</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Recall:</span>
                    <span className="metric-value">{method.recall}</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">F1 Score:</span>
                    <span className="metric-value">{method.f1_score}</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Anomalies:</span>
                    <span className="metric-value">{method.anomalies_detected}</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Time:</span>
                    <span className="metric-value">{method.processing_time}s</span>
                  </div>
                </div>
                
                <div className="method-parameters">
                  <h4>Parameters:</h4>
                  <div className="parameters-list">
                    {Object.entries(method.parameters || {}).map(([key, value]) => (
                      <div key={key} className="parameter">
                        <span className="param-name">{key}:</span>
                        <span className="param-value">{value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Clustering Results Section */}
      {clusteringResults && (
        <div className="dashboard-section">
          <h2>🎯 Clustering Analysis Results</h2>
          
          <div className="clustering-overview">
            <div className="clustering-metrics">
              <div className="metric-item">
                <strong>Silhouette Score:</strong> {clusteringResults.silhouette_score}
              </div>
              <div className="metric-item">
                <strong>Optimal Clusters:</strong> {clusteringResults.optimal_clusters}
              </div>
              <div className="metric-item">
                <strong>Algorithm:</strong> {clusteringResults.clustering_algorithm}
              </div>
              <div className="metric-item">
                <strong>Inertia:</strong> {clusteringResults.inertia}
              </div>
            </div>
          </div>

          <div className="clusters-grid">
            {clusteringResults.clusters?.map((cluster, index) => (
              <div key={cluster.id} className={`cluster-card ${cluster.anomaly_rate > 0.5 ? 'high-anomaly' : 'low-anomaly'}`}>
                <div className="cluster-header">
                  <h3>Cluster {cluster.id}</h3>
                  <div className="cluster-size">Size: {cluster.size}</div>
                </div>
                <div className="anomaly-rate">
                  <div className="rate-label">Anomaly Rate</div>
                  <div className="rate-value">{(cluster.anomaly_rate * 100).toFixed(1)}%</div>
                  <div className="rate-bar">
                    <div 
                      className="rate-fill" 
                      style={{ width: `${cluster.anomaly_rate * 100}%` }}
                    ></div>
                  </div>
                </div>
                <div className="cluster-description">{cluster.description}</div>
              </div>
            ))}
          </div>

          <div className="feature-importance">
            <h3>Feature Importance</h3>
            <div className="features-list">
              {clusteringResults.feature_importance?.map((feature, index) => (
                <div key={index} className="feature-item">
                  <div className="feature-name">{feature.feature}</div>
                  <div className="feature-bar">
                    <div 
                      className="feature-fill" 
                      style={{ width: `${feature.importance * 100}%` }}
                    ></div>
                  </div>
                  <div className="feature-value">{(feature.importance * 100).toFixed(1)}%</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Recommendations Section */}
      {recommendations && (
        <div className="dashboard-section">
          <h2>💡 Actionable Recommendations</h2>
          
          <div className="recommendations-summary">
            <div className="summary-metrics">
              <div className="summary-item">
                <strong>Total Recommendations:</strong> {recommendations.summary?.total_recommendations}
              </div>
              <div className="summary-item">
                <strong>Critical Issues:</strong> {recommendations.summary?.critical_issues}
              </div>
              <div className="summary-item">
                <strong>Est. Resolution Time:</strong> {recommendations.summary?.estimated_resolution_time}
              </div>
              <div className="summary-item">
                <strong>Potential Impact Reduction:</strong> {recommendations.summary?.potential_impact_reduction}
              </div>
            </div>
          </div>

          <div className="recommendations-sections">
            {/* High Priority */}
            {recommendations.high_priority?.length > 0 && (
              <div className="priority-section high-priority">
                <h3>🚨 High Priority</h3>
                <div className="recommendations-list">
                  {recommendations.high_priority.map((rec, index) => (
                    <div key={rec.id} className="recommendation-card">
                      <div className="rec-header">
                        <h4>{rec.title}</h4>
                        <div className="confidence-badge">
                          Confidence: {(rec.confidence * 100).toFixed(0)}%
                        </div>
                      </div>
                      <div className="rec-description">{rec.description}</div>
                      <div className="rec-action">
                        <strong>Action:</strong> {rec.action}
                      </div>
                      <div className="rec-impact">Impact: {rec.impact}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Medium Priority */}
            {recommendations.medium_priority?.length > 0 && (
              <div className="priority-section medium-priority">
                <h3>⚠️ Medium Priority</h3>
                <div className="recommendations-list">
                  {recommendations.medium_priority.map((rec, index) => (
                    <div key={rec.id} className="recommendation-card">
                      <div className="rec-header">
                        <h4>{rec.title}</h4>
                        <div className="confidence-badge">
                          Confidence: {(rec.confidence * 100).toFixed(0)}%
                        </div>
                      </div>
                      <div className="rec-description">{rec.description}</div>
                      <div className="rec-action">
                        <strong>Action:</strong> {rec.action}
                      </div>
                      <div className="rec-impact">Impact: {rec.impact}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Low Priority */}
            {recommendations.low_priority?.length > 0 && (
              <div className="priority-section low-priority">
                <h3>📋 Low Priority</h3>
                <div className="recommendations-list">
                  {recommendations.low_priority.map((rec, index) => (
                    <div key={rec.id} className="recommendation-card">
                      <div className="rec-header">
                        <h4>{rec.title}</h4>
                        <div className="confidence-badge">
                          Confidence: {(rec.confidence * 100).toFixed(0)}%
                        </div>
                      </div>
                      <div className="rec-description">{rec.description}</div>
                      <div className="rec-action">
                        <strong>Action:</strong> {rec.action}
                      </div>
                      <div className="rec-impact">Impact: {rec.impact}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Actions Section */}
      <div className="dashboard-section">
        <h2>🔄 Actions</h2>
        <div className="actions-buttons">
          <button onClick={fetchData} className="action-button refresh">
            🔄 Refresh Data
          </button>
          <button 
            onClick={() => window.open('/api/v1/unsupervised/export-visualizations', '_blank')} 
            className="action-button export"
          >
            📊 Export Visualizations
          </button>
        </div>
      </div>
    </div>
  );
};

export default UnsupervisedAnalysisDashboard;
