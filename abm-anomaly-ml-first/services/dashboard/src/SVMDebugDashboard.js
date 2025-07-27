import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ScatterChart, Scatter, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { AlertCircle, Activity, TrendingUp, Settings, Eye, Download, RefreshCw } from 'lucide-react';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const SVMDebugDashboard = () => {
  const [svmData, setSvmData] = useState(null);
  const [selectedSession, setSelectedSession] = useState('');
  const [debugResults, setDebugResults] = useState(null);
  const [performanceMetrics, setPerformanceMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    fetchSVMModelInfo();
    fetchPerformanceMetrics();
  }, []);

  const fetchSVMModelInfo = async () => {
    try {
      const response = await fetch(`${API_URL}/api/v1/svm-debug/model-info`);
      if (!response.ok) throw new Error('Failed to fetch model info');
      const data = await response.json();
      setSvmData(data);
    } catch (error) {
      console.error('Error fetching SVM model info:', error);
      setError('Failed to load SVM model information');
    }
  };

  const fetchPerformanceMetrics = async () => {
    try {
      const response = await fetch(`${API_URL}/api/v1/svm-debug/performance-metrics`);
      if (!response.ok) throw new Error('Failed to fetch performance metrics');
      const data = await response.json();
      setPerformanceMetrics(data);
    } catch (error) {
      console.error('Error fetching performance metrics:', error);
    }
  };

  const debugSession = async (sessionId, rawText) => {
    if (!sessionId.trim()) {
      setError('Please enter a session ID');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_URL}/api/v1/svm-debug/analyze-session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          raw_text: rawText || "Sample transaction text for analysis",
          include_visualization: true
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const results = await response.json();
      setDebugResults(results);
    } catch (error) {
      console.error('Error debugging session:', error);
      setError(`Failed to analyze session: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const tuneParameters = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/api/v1/svm-debug/tune-parameters`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      });
      
      if (!response.ok) throw new Error('Parameter tuning failed');
      
      const tuningResults = await response.json();
      alert(`Parameter tuning completed! Recommended: nu=${tuningResults.recommended_parameters.nu}, gamma=${tuningResults.recommended_parameters.gamma}`);
    } catch (error) {
      console.error('Error tuning parameters:', error);
      setError('Parameter tuning failed');
    } finally {
      setLoading(false);
    }
  };

  const DecisionScorePlot = ({ data }) => {
    if (!data || !data.decision_score_stats) return null;
    
    const chartData = [
      { name: 'Mean', value: data.decision_score_stats.mean },
      { name: 'Std Dev', value: data.decision_score_stats.std },
      { name: 'Min', value: data.decision_score_stats.min },
      { name: 'Max', value: data.decision_score_stats.max }
    ];

    return (
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">Decision Score Statistics</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" fill="#3b82f6" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  };

  const FeatureContributions = ({ contributions }) => {
    if (!contributions || Object.keys(contributions).length === 0) return null;

    const data = Object.entries(contributions).map(([feature, value]) => ({
      feature,
      contribution: Math.abs(value),
      positive: value > 0,
      value: value
    })).sort((a, b) => b.contribution - a.contribution);

    return (
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">Feature Contributions to SVM Decision</h3>
        <div className="space-y-2">
          {data.slice(0, 10).map(({ feature, contribution, positive, value }) => (
            <div key={feature} className="flex items-center">
              <div className="w-32 text-sm truncate" title={feature}>{feature}</div>
              <div className="flex-1 bg-gray-200 rounded-full h-4 mr-2">
                <div 
                  className={`h-4 rounded-full ${positive ? 'bg-blue-500' : 'bg-red-500'}`}
                  style={{ width: `${(contribution / Math.max(...data.map(d => d.contribution))) * 100}%` }}
                />
              </div>
              <div className="w-20 text-xs text-right" title={`Raw value: ${value.toFixed(6)}`}>
                {contribution.toFixed(3)}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const ModelParametersCard = ({ data }) => (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold mb-4 flex items-center">
        <Settings className="w-5 h-5 mr-2" />
        Model Parameters
      </h3>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div>
          <div className="text-sm font-medium text-gray-500">Nu Parameter</div>
          <div className="text-2xl font-bold text-gray-900">{data?.parameters?.nu || 'N/A'}</div>
        </div>
        <div>
          <div className="text-sm font-medium text-gray-500">Gamma</div>
          <div className="text-2xl font-bold text-gray-900">{data?.parameters?.gamma || 'N/A'}</div>
        </div>
        <div>
          <div className="text-sm font-medium text-gray-500">Support Vectors</div>
          <div className="text-2xl font-bold text-gray-900">{data?.support_vectors_count || 0}</div>
        </div>
        <div>
          <div className="text-sm font-medium text-gray-500">Feature Dimensions</div>
          <div className="text-2xl font-bold text-gray-900">{data?.feature_dimensions || 'N/A'}</div>
        </div>
      </div>
      <div className="mt-4">
        <span className={`px-2 py-1 text-xs font-semibold rounded-full ${
          data?.is_fitted ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
        }`}>
          {data?.is_fitted ? 'Model Fitted' : 'Model Not Fitted'}
        </span>
      </div>
    </div>
  );

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 flex items-center">
          <Eye className="w-8 h-8 mr-3 text-purple-600" />
          One-Class SVM Debug Dashboard
        </h1>
        <p className="text-gray-600 mt-1">Real-time monitoring and debugging of SVM anomaly detection</p>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center">
          <AlertCircle className="w-5 h-5 text-red-500 mr-2" />
          <span className="text-red-700">{error}</span>
          <button 
            onClick={() => setError(null)}
            className="ml-auto text-red-500 hover:text-red-700"
          >
            ×
          </button>
        </div>
      )}

      {/* Navigation Tabs */}
      <div className="bg-white border-b">
        <div className="flex space-x-8">
          {['overview', 'debug-session', 'performance', 'tuning'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`py-3 px-1 border-b-2 font-medium text-sm capitalize ${
                activeTab === tab
                  ? 'border-purple-600 text-purple-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              {tab.replace('-', ' ')}
            </button>
          ))}
        </div>
      </div>

      {/* Overview Tab */}
      {activeTab === 'overview' && (
        <div className="space-y-6">
          {/* Model Information */}
          <ModelParametersCard data={svmData} />
          
          {/* Performance Overview */}
          {performanceMetrics && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-white rounded-lg shadow p-6">
                <div className="text-sm font-medium text-gray-500">Model Status</div>
                <div className={`text-2xl font-bold ${
                  performanceMetrics.model_status === 'active' ? 'text-green-600' : 'text-red-600'
                }`}>
                  {performanceMetrics.model_status}
                </div>
              </div>
              <div className="bg-white rounded-lg shadow p-6">
                <div className="text-sm font-medium text-gray-500">Sessions Analyzed</div>
                <div className="text-2xl font-bold text-gray-900">
                  {performanceMetrics.total_sessions_analyzed || 0}
                </div>
              </div>
              <div className="bg-white rounded-lg shadow p-6">
                <div className="text-sm font-medium text-gray-500">Anomaly Rate</div>
                <div className="text-2xl font-bold text-gray-900">
                  {performanceMetrics.anomaly_rate ? 
                    `${(performanceMetrics.anomaly_rate * 100).toFixed(1)}%` : '0%'}
                </div>
              </div>
            </div>
          )}

          {/* Decision Score Statistics */}
          {performanceMetrics && <DecisionScorePlot data={performanceMetrics} />}
        </div>
      )}

      {/* Debug Session Tab */}
      {activeTab === 'debug-session' && (
        <div className="space-y-6">
          {/* Session Debug Input */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold mb-4">Debug Specific Session</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="md:col-span-2">
                <label className="block text-sm font-medium text-gray-700 mb-2">Session ID</label>
                <input
                  type="text"
                  className="w-full border border-gray-300 rounded-md px-3 py-2"
                  placeholder="Enter session ID to analyze"
                  value={selectedSession}
                  onChange={(e) => setSelectedSession(e.target.value)}
                />
              </div>
              <div className="flex items-end">
                <button
                  onClick={() => debugSession(selectedSession)}
                  disabled={loading || !selectedSession.trim()}
                  className="w-full bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center justify-center"
                >
                  {loading ? (
                    <RefreshCw className="w-4 h-4 animate-spin mr-2" />
                  ) : (
                    <Activity className="w-4 h-4 mr-2" />
                  )}
                  {loading ? 'Analyzing...' : 'Debug Session'}
                </button>
              </div>
            </div>
          </div>

          {/* Debug Results */}
          {debugResults && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="bg-white rounded-lg shadow p-6">
                  <div className="text-sm font-medium text-gray-500">Decision Score</div>
                  <div className={`text-2xl font-bold ${
                    debugResults.decision_score < 0 ? 'text-red-600' : 'text-green-600'
                  }`}>
                    {debugResults.decision_score.toFixed(3)}
                  </div>
                </div>
                <div className="bg-white rounded-lg shadow p-6">
                  <div className="text-sm font-medium text-gray-500">Prediction</div>
                  <div className={`text-2xl font-bold ${
                    debugResults.prediction === 'Anomaly' ? 'text-red-600' : 'text-green-600'
                  }`}>
                    {debugResults.prediction}
                  </div>
                </div>
                <div className="bg-white rounded-lg shadow p-6">
                  <div className="text-sm font-medium text-gray-500">Confidence</div>
                  <div className="text-2xl font-bold text-gray-900">
                    {debugResults.confidence.toFixed(3)}
                  </div>
                </div>
                <div className="bg-white rounded-lg shadow p-6">
                  <div className="text-sm font-medium text-gray-500">Processing Time</div>
                  <div className="text-2xl font-bold text-gray-900">
                    {debugResults.processing_time_ms.toFixed(0)}ms
                  </div>
                </div>
              </div>

              {/* Feature Contributions */}
              <FeatureContributions contributions={debugResults.feature_contributions} />

              {/* Visualization Link */}
              {debugResults.visualization_url && (
                <div className="bg-white rounded-lg shadow p-6">
                  <h3 className="text-lg font-semibold mb-4 flex items-center">
                    <Download className="w-5 h-5 mr-2" />
                    Detailed Visualization
                  </h3>
                  <a
                    href={debugResults.visualization_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600 inline-flex items-center"
                  >
                    <Eye className="w-4 h-4 mr-2" />
                    Open Detailed Analysis
                  </a>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Performance Tab */}
      {activeTab === 'performance' && (
        <div className="space-y-6">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">Performance Monitoring</h3>
              <button
                onClick={fetchPerformanceMetrics}
                className="bg-gray-500 text-white px-4 py-2 rounded-md hover:bg-gray-600 flex items-center"
              >
                <RefreshCw className="w-4 h-4 mr-2" />
                Refresh
              </button>
            </div>
            
            {performanceMetrics ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <div className="text-sm font-medium text-gray-500">Last Updated</div>
                    <div className="text-sm text-gray-900">
                      {new Date(performanceMetrics.timestamp).toLocaleString()}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm font-medium text-gray-500">Support Vectors</div>
                    <div className="text-sm text-gray-900">{performanceMetrics.support_vector_count}</div>
                  </div>
                  <div>
                    <div className="text-sm font-medium text-gray-500">Anomalies Detected</div>
                    <div className="text-sm text-gray-900">{performanceMetrics.anomalies_detected || 0}</div>
                  </div>
                  <div>
                    <div className="text-sm font-medium text-gray-500">Detection Rate</div>
                    <div className="text-sm text-gray-900">
                      {performanceMetrics.anomaly_rate ? 
                        `${(performanceMetrics.anomaly_rate * 100).toFixed(2)}%` : '0%'}
                    </div>
                  </div>
                </div>
                
                {performanceMetrics.decision_score_stats && (
                  <DecisionScorePlot data={performanceMetrics} />
                )}
              </div>
            ) : (
              <p className="text-gray-500">Loading performance metrics...</p>
            )}
          </div>
        </div>
      )}

      {/* Parameter Tuning Tab */}
      {activeTab === 'tuning' && (
        <div className="space-y-6">
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <TrendingUp className="w-5 h-5 mr-2" />
              Parameter Optimization
            </h3>
            <p className="text-gray-600 mb-4">
              Automatically tune SVM parameters for optimal performance based on current data.
            </p>
            
            <div className="space-y-4">
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <div className="flex items-center">
                  <AlertCircle className="w-5 h-5 text-yellow-500 mr-2" />
                  <span className="text-yellow-700 font-medium">Warning</span>
                </div>
                <p className="text-yellow-700 mt-1">
                  Parameter tuning will analyze your current dataset and may take several minutes to complete.
                </p>
              </div>
              
              <button
                onClick={tuneParameters}
                disabled={loading}
                className="bg-green-500 text-white px-6 py-3 rounded-md hover:bg-green-600 disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center"
              >
                {loading ? (
                  <RefreshCw className="w-4 h-4 animate-spin mr-2" />
                ) : (
                  <Settings className="w-4 h-4 mr-2" />
                )}
                {loading ? 'Tuning Parameters...' : 'Start Parameter Tuning'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SVMDebugDashboard;
