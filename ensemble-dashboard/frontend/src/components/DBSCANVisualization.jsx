import React, { useState, useEffect } from 'react';
import './DBSCANVisualization.css';

// Utility function to safely format numbers
const safeToFixed = (value, decimals = 2) => {
  console.log(`[safeToFixed] Called with value: ${value}, type: ${typeof value}, decimals: ${decimals}`);
  
  if (value === null || value === undefined || isNaN(value) || !isFinite(value)) {
    console.log(`[safeToFixed] Returning 'N/A' for invalid value: ${value}`);
    return 'N/A';
  }
  if (typeof value !== 'number') {
    const num = parseFloat(value);
    if (isNaN(num) || !isFinite(num)) {
      console.log(`[safeToFixed] Returning 'N/A' for unparseable value: ${value}`);
      return 'N/A';
    }
    console.log(`[safeToFixed] Converted string to number: ${value} -> ${num}`);
    return num.toFixed(decimals);
  }
  console.log(`[safeToFixed] Valid number, returning: ${value.toFixed(decimals)}`);
  return value.toFixed(decimals);
};

const DBSCANVisualization = () => {
  console.log('[DBSCANVisualization] Component loaded with enhanced error handling and logging - Version 2.0');
  console.log('[DBSCANVisualization] safeToFixed function available:', typeof safeToFixed === 'function');
  
  const [clusterInsights, setClusterInsights] = useState(null);
  const [visualizationData, setVisualizationData] = useState(null);
  const [performanceData, setPerformanceData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('insights');
  
  // New state for cluster interaction
  const [selectedCluster, setSelectedCluster] = useState(null);
  const [clusterSessions, setClusterSessions] = useState([]);
  const [showSessionModal, setShowSessionModal] = useState(false);
  const [labelingCluster, setLabelingCluster] = useState(null);
  const [showLabelModal, setShowLabelModal] = useState(false);
  const [labelForm, setLabelForm] = useState({
    labelName: '',
    labelDescription: '',
    confidence: 0.8
  });
  const [submittingLabel, setSubmittingLabel] = useState(false);
  const [clusterLabels, setClusterLabels] = useState({});
  const [supervisedPrediction, setSupervisedPrediction] = useState(null);
  const [testSession, setTestSession] = useState('');
  const [showPredictionModal, setShowPredictionModal] = useState(false);

  useEffect(() => {
    fetchDBSCANData();
    fetchClusterLabels();
  }, []);

  const fetchDBSCANData = async () => {
    try {
      console.log('[fetchDBSCANData] Starting to fetch DBSCAN data...');
      setLoading(true);
      setError(null);

      // Fetch cluster insights
      console.log('[fetchDBSCANData] Fetching cluster insights...');
      const insightsResponse = await fetch('http://localhost:8001/api/cluster_insights');
      console.log('[fetchDBSCANData] Insights response status:', insightsResponse.status);
      console.log('[fetchDBSCANData] Insights response headers:', Object.fromEntries(insightsResponse.headers.entries()));
      
      if (insightsResponse.ok) {
        const insightsData = await insightsResponse.json();
        console.log('[fetchDBSCANData] Insights data received:', insightsData);
        setClusterInsights(insightsData.insights);
      } else {
        console.error('[fetchDBSCANData] Insights request failed:', insightsResponse.status, insightsResponse.statusText);
      }

      // Fetch visualization data
      console.log('[fetchDBSCANData] Fetching visualization data...');
      const vizResponse = await fetch('http://localhost:8001/api/cluster_visualization_data', { method: 'POST' });
      console.log('[fetchDBSCANData] Viz response status:', vizResponse.status);
      
      if (vizResponse.ok) {
        const vizData = await vizResponse.json();
        console.log('[fetchDBSCANData] Viz data received:', vizData);
        setVisualizationData(vizData.visualization_data);
      } else {
        console.error('[fetchDBSCANData] Viz request failed:', vizResponse.status, vizResponse.statusText);
      }

      // Fetch performance comparison
      console.log('[fetchDBSCANData] Fetching performance comparison...');
      const perfResponse = await fetch('http://localhost:8001/api/performance_comparison', { method: 'POST' });
      console.log('[fetchDBSCANData] Perf response status:', perfResponse.status);
      
      if (perfResponse.ok) {
        const perfData = await perfResponse.json();
        console.log('[fetchDBSCANData] Perf data received:', perfData);
        setPerformanceData(perfData.comparison_data);
      } else {
        console.error('[fetchDBSCANData] Perf request failed:', perfResponse.status, perfResponse.statusText);
      }

    } catch (err) {
      console.error('[fetchDBSCANData] Error occurred:', err);
      setError(err.message);
    } finally {
      console.log('[fetchDBSCANData] Finished fetching data');
      setLoading(false);
    }
  };

  const fetchClusterLabels = async () => {
    try {
      const response = await fetch('http://localhost:8001/api/cluster_labels');
      if (response.ok) {
        const data = await response.json();
        setClusterLabels(data.labels || {});
      }
    } catch (err) {
      console.error('Error fetching cluster labels:', err);
    }
  };

  const fetchClusterSessions = async (clusterId, featureType) => {
    try {
      console.log(`[fetchClusterSessions] Starting to fetch sessions for cluster ${clusterId}, type ${featureType}`);
      setLoading(true);
      const response = await fetch('http://localhost:8001/api/cluster_sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          cluster_id: clusterId,
          feature_type: featureType
        })
      });

      console.log(`[fetchClusterSessions] Response status: ${response.status}`);
      console.log(`[fetchClusterSessions] Response headers:`, Object.fromEntries(response.headers.entries()));

      if (response.ok) {
        const data = await response.json();
        console.log(`[fetchClusterSessions] Received ${data.sessions?.length || 0} sessions:`, data);
        setClusterSessions(data.sessions || []);
        setSelectedCluster({ id: clusterId, type: featureType });
        setShowSessionModal(true);
      } else {
        const errorText = await response.text();
        console.error(`[fetchClusterSessions] Request failed: ${response.status} ${response.statusText}`, errorText);
        throw new Error('Failed to fetch cluster sessions');
      }
    } catch (err) {
      console.error(`[fetchClusterSessions] Error occurred:`, err);
      setError(err.message);
    } finally {
      console.log('[fetchClusterSessions] Finished');
      setLoading(false);
    }
  };

  const trainSupervisedClassifier = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:8001/api/train_supervised_classifier', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      if (response.ok) {
        const data = await response.json();
        console.log('Supervised classifier trained:', data);
        alert(`Supervised classifier trained successfully!\nAccuracy: ${(data.training_result.accuracy * 100).toFixed(1)}%\nClasses: ${data.training_result.classes.join(', ')}`);
      } else {
        throw new Error('Failed to train supervised classifier');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const predictWithSupervised = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:8001/api/predict_with_supervised', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_text: testSession })
      });

      if (response.ok) {
        const data = await response.json();
        setSupervisedPrediction(data.prediction);
        setShowPredictionModal(true);
      } else {
        throw new Error('Failed to get supervised prediction');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const submitClusterLabel = async () => {
    try {
      setSubmittingLabel(true);
      
      const response = await fetch('http://localhost:8001/api/label_cluster', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          cluster_id: labelingCluster.id,
          feature_type: labelingCluster.type,
          label_name: labelForm.labelName,
          label_description: labelForm.labelDescription,
          confidence: labelForm.confidence
        })
      });

      if (response.ok) {
        setShowLabelModal(false);
        fetchClusterLabels(); // Refresh labels
        alert('Cluster labeled successfully!');
      } else {
        setError('Failed to label cluster');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmittingLabel(false);
    }
  };

  const renderScatterPlot = (clusters, title, width = 400, height = 300) => {
    if (!clusters || clusters.length === 0) return null;

    const margin = { top: 20, right: 20, bottom: 40, left: 40 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;

    // Calculate bounds
    const xValues = clusters.map(c => c.x);
    const yValues = clusters.map(c => c.y);
    const xMin = Math.min(...xValues) - 0.1;
    const xMax = Math.max(...xValues) + 0.1;
    const yMin = Math.min(...yValues) - 0.1;
    const yMax = Math.max(...yValues) + 0.1;

    const xScale = (x) => ((x - xMin) / (xMax - xMin)) * plotWidth;
    const yScale = (y) => plotHeight - ((y - yMin) / (yMax - yMin)) * plotHeight;

    const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FCEA2B', '#FF8E53', '#6C5CE7', '#FDCB6E'];

    // Extract feature type from title
    const featureType = title.toLowerCase().includes('text') ? 'text' : 
                       title.toLowerCase().includes('numerical') ? 'numerical' : 'combined';

    const handleClusterClick = (cluster) => {
      const clusterId = parseInt(cluster.name.replace('cluster_', ''));
      fetchClusterSessions(clusterId, featureType);
    };

    const handleClusterRightClick = (e, cluster) => {
      e.preventDefault();
      const clusterId = parseInt(cluster.name.replace('cluster_', ''));
      setLabelingCluster({ id: clusterId, type: featureType, name: cluster.name });
      setShowLabelModal(true);
    };

    const getClusterLabel = (cluster) => {
      const clusterId = parseInt(cluster.name.replace('cluster_', ''));
      const labelKey = `${featureType}_${clusterId}`;
      return clusterLabels[labelKey]?.label_name || null;
    };

    return (
      <div className="scatter-plot-container">
        <h4>{title}</h4>
        <svg width={width} height={height} className="scatter-plot">
          {/* Background */}
          <rect width={width} height={height} fill="#f8f9fa" stroke="#dee2e6" strokeWidth="1"/>
          
          {/* Plot area */}
          <g transform={`translate(${margin.left}, ${margin.top})`}>
            {/* Grid lines */}
            {[0, 0.25, 0.5, 0.75, 1].map(t => (
              <g key={t}>
                <line
                  x1={t * plotWidth}
                  y1={0}
                  x2={t * plotWidth}
                  y2={plotHeight}
                  stroke="#e9ecef"
                  strokeWidth="1"
                />
                <line
                  x1={0}
                  y1={t * plotHeight}
                  x2={plotWidth}
                  y2={t * plotHeight}
                  stroke="#e9ecef"
                  strokeWidth="1"
                />
              </g>
            ))}

            {/* Data points */}
            {clusters.map((cluster, index) => {
              const clusterLabel = getClusterLabel(cluster);
              const isLabeled = clusterLabel !== null;
              
              return (
                <g key={cluster.name}>
                  <circle
                    cx={xScale(cluster.x)}
                    cy={yScale(cluster.y)}
                    r={Math.max(5, Math.min(20, cluster.size * 2))}
                    fill={colors[index % colors.length]}
                    fillOpacity={isLabeled ? "0.9" : "0.7"}
                    stroke={isLabeled ? "#000" : colors[index % colors.length]}
                    strokeWidth={isLabeled ? "3" : "2"}
                    style={{ cursor: 'pointer' }}
                    onClick={() => handleClusterClick(cluster)}
                    onContextMenu={(e) => handleClusterRightClick(e, cluster)}
                  />
                  <text
                    x={xScale(cluster.x)}
                    y={yScale(cluster.y) - Math.max(5, Math.min(20, cluster.size * 2)) - 5}
                    textAnchor="middle"
                    fontSize="10"
                    fill="#333"
                    style={{ cursor: 'pointer' }}
                    onClick={() => handleClusterClick(cluster)}
                  >
                    {isLabeled ? clusterLabel : cluster.name}
                  </text>
                  {isLabeled && (
                    <text
                      x={xScale(cluster.x)}
                      y={yScale(cluster.y) - Math.max(5, Math.min(20, cluster.size * 2)) - 18}
                      textAnchor="middle"
                      fontSize="8"
                      fill="#666"
                      style={{ cursor: 'pointer' }}
                      onClick={() => handleClusterClick(cluster)}
                    >
                      📋 Labeled
                    </text>
                  )}
                </g>
              );
            })}

            {/* Axes */}
            <line x1={0} y1={plotHeight} x2={plotWidth} y2={plotHeight} stroke="#333" strokeWidth="2"/>
            <line x1={0} y1={0} x2={0} y2={plotHeight} stroke="#333" strokeWidth="2"/>
            
            {/* Axis labels */}
            <text x={plotWidth/2} y={plotHeight + 35} textAnchor="middle" fontSize="12" fill="#666">
              Principal Component 1
            </text>
            <text x={-plotHeight/2} y={-25} textAnchor="middle" fontSize="12" fill="#666" transform={`rotate(-90, -${plotHeight/2}, -25)`}>
              Principal Component 2
            </text>
          </g>
        </svg>
        
        {/* Legend */}
        <div className="cluster-legend">
          {clusters.map((cluster, index) => {
            const clusterLabel = getClusterLabel(cluster);
            const isLabeled = clusterLabel !== null;
            
            return (
              <div key={cluster.name} className="legend-item" style={{ cursor: 'pointer' }} onClick={() => handleClusterClick(cluster)}>
                <div 
                  className="legend-color" 
                  style={{ 
                    backgroundColor: colors[index % colors.length],
                    border: isLabeled ? '2px solid #000' : 'none'
                  }}
                ></div>
                <span>
                  {isLabeled ? `${clusterLabel} (${cluster.name})` : cluster.name} (size: {cluster.size})
                  {isLabeled && ' 📋'}
                </span>
              </div>
            );
          })}
        </div>
        
        <div className="cluster-instructions" style={{ marginTop: '10px', fontSize: '12px', color: '#666' }}>
          💡 <strong>Click</strong> cluster to view sessions • <strong>Right-click</strong> to label cluster
        </div>
      </div>
    );
  };

  const renderInsights = () => {
    if (!clusterInsights) return <div>No cluster insights available.</div>;

    return (
      <div className="insights-section">
        <h3>Cluster Analysis Summary</h3>
        
        <div className="overview-grid">
          {Object.entries(clusterInsights.cluster_summary).map(([type, summary]) => (
            <div key={type} className="summary-card">
              <h4>{type.charAt(0).toUpperCase() + type.slice(1)} Features</h4>
              <div className="summary-stats">
                <div className="stat">
                  <span className="stat-label">Clusters:</span>
                  <span className="stat-value">{summary.n_clusters}</span>
                </div>
                <div className="stat">
                  <span className="stat-label">Noise Ratio:</span>
                  <span className="stat-value">
                    {safeToFixed(summary.noise_ratio * 100, 1)}%
                  </span>
                </div>
                <div className="stat">
                  <span className="stat-label">Largest Cluster:</span>
                  <span className="stat-value">{summary.largest_cluster_size}</span>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="anomaly-patterns">
          <h4>Anomaly Pattern Analysis</h4>
          {Object.entries(clusterInsights.anomaly_patterns).map(([type, patterns]) => (
            <div key={type} className="pattern-analysis">
              <h5>{type.charAt(0).toUpperCase() + type.slice(1)} Patterns</h5>
              <div className="pattern-indicators">
                {Object.entries(patterns).map(([pattern, value]) => (
                  <div key={pattern} className={`indicator ${value ? 'warning' : 'good'}`}>
                    <span className="indicator-label">
                      {pattern.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:
                    </span>
                    <span className="indicator-value">{value ? 'Yes' : 'No'}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderVisualization = () => {
    if (!visualizationData) return <div>No visualization data available.</div>;

    return (
      <div className="visualization-section">
        <h3>DBSCAN Cluster Visualizations</h3>
        
        <div className="plots-grid">
          {visualizationData.text_clusters && (
            <div className="plot-container">
              {renderScatterPlot(visualizationData.text_clusters, 'Text Feature Clusters')}
            </div>
          )}
          
          {visualizationData.numerical_clusters && (
            <div className="plot-container">
              {renderScatterPlot(visualizationData.numerical_clusters, 'Numerical Feature Clusters')}
            </div>
          )}
          
          {visualizationData.combined_clusters && (
            <div className="plot-container">
              {renderScatterPlot(visualizationData.combined_clusters, 'Combined Feature Clusters')}
            </div>
          )}
        </div>

        {visualizationData.cluster_statistics && (
          <div className="cluster-statistics">
            <h4>Cluster Statistics</h4>
            <div className="stats-grid">
              {Object.entries(visualizationData.cluster_statistics).map(([type, stats]) => (
                <div key={type} className="stat-card">
                  <h5>{type.charAt(0).toUpperCase() + type.slice(1)}</h5>
                  <div className="stat-list">
                    <div>Clusters: {stats.n_clusters}</div>
                    <div>Noise Ratio: {safeToFixed(stats.noise_ratio * 100, 1)}%</div>
                    <div>Total Points: {stats.total_points}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderPerformance = () => {
    if (!performanceData) return <div>No performance data available.</div>;

    return (
      <div className="performance-section">
        <h3>Performance Comparison</h3>
        
        <div className="performance-overview">
          <div className="overview-card">
            <h4>Model Configuration</h4>
            <div className="config-details">
              <div>Type: {performanceData.model_type}</div>
              <div>Training Sessions: {performanceData.training_sessions}</div>
              <div>Text Features: {performanceData.feature_dimensions?.text}</div>
              <div>Numerical Features: {performanceData.feature_dimensions?.numerical}</div>
            </div>
          </div>

          <div className="overview-card">
            <h4>Ensemble Weights</h4>
            <div className="weights-details">
              <div>Text Weight: {performanceData.weights?.text_weight}</div>
              <div>Statistical Weight: {performanceData.weights?.statistical_weight}</div>
              <div>Density Weight: {performanceData.weights?.density_weight}</div>
            </div>
          </div>
        </div>

        <div className="scores-section">
          <h4>Average Component Scores</h4>
          <div className="scores-grid">
            {Object.entries(performanceData.average_scores || {}).map(([component, score]) => (
              <div key={component} className="score-card">
                <div className="score-label">{component.charAt(0).toUpperCase() + component.slice(1)}</div>
                <div className="score-value">{score !== null ? score.toFixed(4) : 'N/A'}</div>
                <div className="score-bar">
                  <div 
                    className="score-fill" 
                    style={{ width: `${(score || 0) * 100}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {performanceData.dbscan_parameters && (
          <div className="dbscan-params">
            <h4>DBSCAN Parameters</h4>
            <div className="params-grid">
              {Object.entries(performanceData.dbscan_parameters).map(([param, value]) => (
                <div key={param} className="param-item">
                  <span className="param-label">{param.replace(/_/g, ' ')}:</span>
                  <span className="param-value">
                    {typeof value === 'number' ? value.toFixed(4) : value}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  if (loading) {
    return (
      <div className="dbscan-container">
        <div className="loading-state">
          <div className="loading-spinner"></div>
          <p>Loading DBSCAN analysis...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="dbscan-container">
        <div className="error-state">
          <h3>Error Loading DBSCAN Analysis</h3>
          <p>{error}</p>
          <button onClick={fetchDBSCANData} className="retry-button">
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="dbscan-container">
      <div className="dbscan-header">
        <h2>🔬 DBSCAN Cluster Analysis</h2>
        <p>Density-based spatial clustering analysis and visualization</p>
      </div>

      <div className="tab-navigation">
        <button 
          className={`tab-button ${activeTab === 'insights' ? 'active' : ''}`}
          onClick={() => setActiveTab('insights')}
        >
          Cluster Insights
        </button>
        <button 
          className={`tab-button ${activeTab === 'visualization' ? 'active' : ''}`}
          onClick={() => setActiveTab('visualization')}
        >
          Scatter Plots
        </button>
        <button 
          className={`tab-button ${activeTab === 'performance' ? 'active' : ''}`}
          onClick={() => setActiveTab('performance')}
        >
          Performance
        </button>
      </div>

      <div className="tab-content">
        {activeTab === 'insights' && renderInsights()}
        {activeTab === 'visualization' && renderVisualization()}
        {activeTab === 'performance' && renderPerformance()}
      </div>

      <div className="refresh-section">
        <button onClick={fetchDBSCANData} className="refresh-button">
          🔄 Refresh Data
        </button>
        <button onClick={trainSupervisedClassifier} className="train-button">
          🤖 Train Supervised Classifier
        </button>
        <button onClick={() => setShowPredictionModal(true)} className="predict-button">
          🔮 Test Prediction
        </button>
      </div>

      {/* Session Details Modal */}
      {showSessionModal && (
        <div className="modal-overlay" onClick={() => setShowSessionModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>🔍 Cluster Sessions: {selectedCluster?.type} cluster {selectedCluster?.id}</h3>
              <button className="close-button" onClick={() => setShowSessionModal(false)}>✕</button>
            </div>
            
            <div className="modal-body">
              <div className="cluster-info">
                <p><strong>Sessions in cluster:</strong> {clusterSessions.length}</p>
                <p><strong>Feature type:</strong> {selectedCluster?.type}</p>
              </div>
              
              <div className="sessions-list">
                {clusterSessions.map((session, index) => {
                  console.log(`[Session Render] Session ${index}:`, session);
                  console.log(`[Session Render] anomaly_score:`, session?.anomaly_score, `type:`, typeof session?.anomaly_score);
                  
                  return (
                    <div key={index} className="session-item">
                      <div className="session-header">
                        <span className="session-id">📄 {session.session_id}</span>
                        {session.expert_label && (
                          <span className="expert-label">🏷️ {session.expert_label}</span>
                        )}
                      </div>
                      
                      <div className="session-details">
                      {/* ULTRA SAFE VERSION - Using utility function to prevent any toFixed() errors */}
                      {session && session.anomaly_score !== undefined && (
                        <div className="detail-row">
                          <span>Anomaly Score:</span>
                          <span className={session.anomaly_score > 0.5 ? 'high-score' : 'low-score'}>
                            {(() => {
                              console.log(`[Session Render] About to call safeToFixed for session ${index}`);
                              const result = safeToFixed(session.anomaly_score, 3);
                              console.log(`[Session Render] safeToFixed result: ${result}`);
                              return result;
                            })()}
                          </span>
                        </div>
                      )}
                      
                      {session.cluster_id !== undefined && (
                        <div className="detail-row">
                          <span>Cluster ID:</span>
                          <span>{session.cluster_id}</span>
                        </div>
                      )}
                      
                      {session.feature_type && (
                        <div className="detail-row">
                          <span>Feature Type:</span>
                          <span>{session.feature_type}</span>
                        </div>
                      )}
                      
                      {session.has_errors && session.error_types && (
                        <div className="detail-row">
                          <span>Error Types:</span>
                          <span className="error-types">{session.error_types.join(', ')}</span>
                        </div>
                      )}
                      
                      {session.transaction_type && (
                        <div className="detail-row">
                          <span>Transaction Type:</span>
                          <span>{session.transaction_type}</span>
                        </div>
                      )}
                      </div>
                      
                      <div className="session-preview">
                        <strong>Session Preview:</strong>
                        <pre className="session-text">
                          {session.raw_text_preview || session.session_text || 'No text available'}
                        </pre>
                      </div>
                    </div>
                  );
                })}
              </div>
              
              <div className="modal-actions">
                <button 
                  className="label-cluster-button"
                  onClick={() => {
                    setLabelingCluster(selectedCluster);
                    setShowLabelModal(true);
                    setShowSessionModal(false);
                  }}
                >
                  🏷️ Label This Cluster
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Label Cluster Modal */}
      {showLabelModal && (
        <div className="modal-overlay" onClick={() => setShowLabelModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>🏷️ Label Cluster: {labelingCluster?.type} cluster {labelingCluster?.id}</h3>
              <button className="close-button" onClick={() => setShowLabelModal(false)}>✕</button>
            </div>
            
            <div className="modal-body">
              <div className="form-group">
                <label htmlFor="labelName">Label Name:</label>
                <input
                  id="labelName"
                  type="text"
                  value={labelForm.labelName}
                  onChange={(e) => setLabelForm({...labelForm, labelName: e.target.value})}
                  placeholder="e.g., 'Device Errors', 'Communication Timeouts', 'Normal Transactions'"
                  className="form-input"
                />
              </div>
              
              <div className="form-group">
                <label htmlFor="labelDescription">Description (optional):</label>
                <textarea
                  id="labelDescription"
                  value={labelForm.labelDescription}
                  onChange={(e) => setLabelForm({...labelForm, labelDescription: e.target.value})}
                  placeholder="Describe the pattern or characteristics of this cluster..."
                  className="form-textarea"
                  rows="3"
                />
              </div>
              
              <div className="form-group">
                <label htmlFor="confidence">Confidence Level:</label>
                <input
                  id="confidence"
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.1"
                  value={labelForm.confidence}
                  onChange={(e) => setLabelForm({...labelForm, confidence: parseFloat(e.target.value)})}
                  className="form-range"
                />
                <span className="confidence-value">{(labelForm.confidence * 100).toFixed(0)}%</span>
              </div>
            </div>
            
            <div className="modal-actions">
              <button 
                className="cancel-button"
                onClick={() => setShowLabelModal(false)}
              >
                Cancel
              </button>
              <button 
                className="submit-button"
                onClick={submitClusterLabel}
                disabled={!labelForm.labelName.trim() || submittingLabel}
              >
                {submittingLabel ? 'Labeling...' : 'Apply Label'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Prediction Test Modal */}
      {showPredictionModal && (
        <div className="modal-overlay" onClick={() => setShowPredictionModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>🔮 Test Supervised Prediction</h3>
              <button className="close-button" onClick={() => setShowPredictionModal(false)}>✕</button>
            </div>
            
            <div className="modal-body">
              <div className="form-group">
                <label htmlFor="testSession">EJ Session Text:</label>
                <textarea
                  id="testSession"
                  value={testSession}
                  onChange={(e) => setTestSession(e.target.value)}
                  placeholder="Paste an EJ session text here to test the supervised classifier..."
                  className="form-textarea"
                  rows="10"
                />
              </div>
              
              {supervisedPrediction && (
                <div className="prediction-results">
                  <h4>Prediction Results:</h4>
                  <div className="prediction-item">
                    <span className="prediction-label">Predicted Label:</span>
                    <span className="prediction-value">
                      {supervisedPrediction.predictions?.combined?.predicted_label || 
                       supervisedPrediction.predictions?.text?.predicted_label || 'N/A'}
                    </span>
                  </div>
                  <div className="prediction-item">
                    <span className="prediction-label">Confidence:</span>
                    <span className="prediction-value">
                      {((supervisedPrediction.predictions?.combined?.confidence || 
                         supervisedPrediction.predictions?.text?.confidence || 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                  
                  {supervisedPrediction.predictions?.combined?.all_probabilities && (
                    <div className="all-probabilities">
                      <h5>All Class Probabilities:</h5>
                      {Object.entries(supervisedPrediction.predictions.combined.all_probabilities).map(([label, prob]) => (
                        <div key={label} className="probability-item">
                          <span>{label}:</span>
                          <span>{(prob * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
            
            <div className="modal-actions">
              <button 
                className="cancel-button"
                onClick={() => setShowPredictionModal(false)}
              >
                Close
              </button>
              <button 
                className="submit-button"
                onClick={predictWithSupervised}
                disabled={!testSession.trim() || loading}
              >
                {loading ? 'Predicting...' : 'Predict'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DBSCANVisualization;
