import React, { useState, useEffect } from 'react';

const DBSCANVisualization = () => {
  const [analysisData, setAnalysisData] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('text');

  const fetchModelInfo = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/model_info');
      const data = await response.json();
      setModelInfo(data);
      return data.is_trained;
    } catch (err) {
      console.error('Error fetching model info:', err);
      setError('Failed to fetch model information');
      return false;
    }
  };

  const fetchAnalysisData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:8000/api/dbscan_analysis');
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      setAnalysisData(data);
    } catch (err) {
      console.error('Error fetching DBSCAN analysis:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const initializeData = async () => {
      const isModelTrained = await fetchModelInfo();
      if (isModelTrained) {
        await fetchAnalysisData();
      }
    };
    
    initializeData();
  }, []);

  const renderScatterPlot = (featureType) => {
    if (!analysisData || !analysisData[featureType]) {
      return <div className="text-gray-500">No data available for {featureType} features</div>;
    }

    const data = analysisData[featureType];
    const { features_2d, labels } = data.visualization_data;
    const { n_clusters, n_noise } = data.cluster_statistics;

    // Generate colors for clusters
    const colors = [
      '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
      '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
    ];
    const noiseColor = '#95A5A6';

    // Find data bounds for scaling
    const xValues = features_2d.map(point => point[0]);
    const yValues = features_2d.map(point => point[1]);
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);

    const svgWidth = 500;
    const svgHeight = 400;
    const margin = 40;
    const plotWidth = svgWidth - 2 * margin;
    const plotHeight = svgHeight - 2 * margin;

    // Scale functions
    const scaleX = (x) => margin + ((x - xMin) / (xMax - xMin)) * plotWidth;
    const scaleY = (y) => margin + ((yMax - y) / (yMax - yMin)) * plotHeight;

    // Get unique clusters for legend
    const uniqueClusters = [...new Set(labels)].sort((a, b) => a - b);

    return (
      <div className="bg-white p-6 rounded-lg shadow-lg">
        <h3 className="text-xl font-semibold mb-4 text-gray-800">
          {featureType.charAt(0).toUpperCase() + featureType.slice(1)} Features Clustering
        </h3>
        
        <div className="flex flex-wrap gap-6">
          {/* Scatter Plot */}
          <div className="flex-1 min-w-[500px]">
            <svg width={svgWidth} height={svgHeight} className="border border-gray-300 rounded">
              {/* Background */}
              <rect width={svgWidth} height={svgHeight} fill="#f8f9fa" />
              
              {/* Grid lines */}
              {[0, 0.25, 0.5, 0.75, 1].map(ratio => (
                <g key={`grid-${ratio}`}>
                  <line
                    x1={margin + ratio * plotWidth}
                    y1={margin}
                    x2={margin + ratio * plotWidth}
                    y2={margin + plotHeight}
                    stroke="#e9ecef"
                    strokeWidth="1"
                  />
                  <line
                    x1={margin}
                    y1={margin + ratio * plotHeight}
                    x2={margin + plotWidth}
                    y2={margin + ratio * plotHeight}
                    stroke="#e9ecef"
                    strokeWidth="1"
                  />
                </g>
              ))}
              
              {/* Axes */}
              <line x1={margin} y1={margin} x2={margin} y2={margin + plotHeight} stroke="#333" strokeWidth="2" />
              <line x1={margin} y1={margin + plotHeight} x2={margin + plotWidth} y2={margin + plotHeight} stroke="#333" strokeWidth="2" />
              
              {/* Data points */}
              {features_2d.map((point, index) => {
                const cluster = labels[index];
                const color = cluster === -1 ? noiseColor : colors[cluster % colors.length];
                
                return (
                  <circle
                    key={index}
                    cx={scaleX(point[0])}
                    cy={scaleY(point[1])}
                    r="4"
                    fill={color}
                    stroke="#fff"
                    strokeWidth="1"
                    opacity="0.8"
                  />
                );
              })}
              
              {/* Axis labels */}
              <text x={svgWidth / 2} y={svgHeight - 5} textAnchor="middle" className="text-sm fill-gray-600">
                Principal Component 1
              </text>
              <text
                x="15"
                y={svgHeight / 2}
                textAnchor="middle"
                transform={`rotate(-90, 15, ${svgHeight / 2})`}
                className="text-sm fill-gray-600"
              >
                Principal Component 2
              </text>
            </svg>
          </div>

          {/* Legend and Statistics */}
          <div className="w-64 space-y-4">
            {/* Cluster Legend */}
            <div className="bg-gray-50 p-4 rounded-lg">
              <h4 className="font-semibold text-gray-800 mb-3">Clusters</h4>
              <div className="space-y-2">
                {uniqueClusters.map(cluster => {
                  const color = cluster === -1 ? noiseColor : colors[cluster % colors.length];
                  const count = labels.filter(label => label === cluster).length;
                  const percentage = ((count / labels.length) * 100).toFixed(1);
                  
                  return (
                    <div key={cluster} className="flex items-center space-x-2">
                      <div
                        className="w-4 h-4 rounded-full border border-white"
                        style={{ backgroundColor: color }}
                      ></div>
                      <span className="text-sm text-gray-700">
                        {cluster === -1 ? 'Noise' : `Cluster ${cluster}`}: {count} ({percentage}%)
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Statistics */}
            <div className="bg-gray-50 p-4 rounded-lg">
              <h4 className="font-semibold text-gray-800 mb-3">Statistics</h4>
              <div className="space-y-2 text-sm text-gray-700">
                <div>Total Points: {data.cluster_statistics.total_points}</div>
                <div>Clusters Found: {n_clusters}</div>
                <div>Noise Points: {n_noise}</div>
                <div>Noise Ratio: {data.cluster_statistics.noise_percentage.toFixed(1)}%</div>
                <div>Silhouette Score: {data.cluster_statistics.silhouette_score.toFixed(3)}</div>
                <div>Explained Variance: {(data.visualization_data.explained_variance * 100).toFixed(1)}%</div>
              </div>
            </div>

            {/* Parameters */}
            <div className="bg-gray-50 p-4 rounded-lg">
              <h4 className="font-semibold text-gray-800 mb-3">DBSCAN Parameters</h4>
              <div className="space-y-2 text-sm text-gray-700">
                <div>eps: {data.clustering_results.eps.toFixed(3)}</div>
                <div>min_samples: {data.clustering_results.min_samples}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  if (!modelInfo) {
    return (
      <div className="p-6">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading model information...</p>
        </div>
      </div>
    );
  }

  if (!modelInfo.is_trained) {
    return (
      <div className="p-6">
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 text-center">
          <h2 className="text-xl font-semibold text-yellow-800 mb-2">Model Not Trained</h2>
          <p className="text-yellow-700 mb-4">
            The enhanced ensemble model needs to be trained before DBSCAN analysis can be performed.
          </p>
          <p className="text-sm text-yellow-600">
            {modelInfo.message || 'Please train the model first using the training interface.'}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h1 className="text-2xl font-bold text-gray-800 mb-2">DBSCAN Clustering Analysis</h1>
        <p className="text-gray-600 mb-6">
          Density-based clustering analysis across different feature spaces for anomaly detection
        </p>

        {/* Model Status */}
        <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-6">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            <span className="font-medium text-green-800">Model Trained</span>
          </div>
          <p className="text-sm text-green-700 mt-1">
            Training completed: {new Date(modelInfo.training_timestamp).toLocaleString()}
          </p>
          <p className="text-sm text-green-700">
            Training sessions: {modelInfo.training_stats?.n_sessions || 'Unknown'}
          </p>
        </div>

        {/* Tab Navigation */}
        <div className="border-b border-gray-200 mb-6">
          <nav className="-mb-px flex space-x-8">
            {['text', 'numerical', 'combined'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)} Features
              </button>
            ))}
          </nav>
        </div>

        {/* Content */}
        {loading ? (
          <div className="text-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
            <p className="mt-4 text-gray-600">Loading DBSCAN analysis...</p>
          </div>
        ) : error ? (
          <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
            <h3 className="text-lg font-semibold text-red-800 mb-2">Error Loading Analysis</h3>
            <p className="text-red-700 mb-4">{error}</p>
            <button
              onClick={fetchAnalysisData}
              className="bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700 transition-colors"
            >
              Retry
            </button>
          </div>
        ) : (
          <div>
            {renderScatterPlot(activeTab)}
            
            {/* Refresh Button */}
            <div className="mt-6 text-center">
              <button
                onClick={fetchAnalysisData}
                className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors"
                disabled={loading}
              >
                {loading ? 'Refreshing...' : 'Refresh Analysis'}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DBSCANVisualization;
