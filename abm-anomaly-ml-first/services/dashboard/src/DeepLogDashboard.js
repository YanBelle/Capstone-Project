import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, ScatterChart, Scatter, PieChart, Pie, Cell, Area, AreaChart
} from 'recharts';
import { 
  AlertCircle, Activity, TrendingUp, Settings, Eye, Download, RefreshCw, 
  Brain, Layers, Target, Clock, Database, Cpu, BarChart3, Zap
} from 'lucide-react';

const apiEndpoint = (path) => {
  // Handle different API URL configurations
  const baseURL = process.env.REACT_APP_API_URL || '';
  
  if (!baseURL) {
    // No base URL set, use relative path through nginx proxy
    return `/api/v1/bert-deeplog${path}`;
  } else if (baseURL.endsWith('/api')) {
    // Base URL already includes /api, just append the deeplog path
    return `${baseURL}/v1/bert-deeplog${path}`;
  } else {
    // Base URL doesn't include /api, append full path
    return `${baseURL}/api/v1/bert-deeplog${path}`;
  }
};

const DeepLogDashboard = () => {
  // State management
  const [modelInfo, setModelInfo] = useState(null);
  const [trainingHistory, setTrainingHistory] = useState([]);
  const [predictionCache, setPredictionCache] = useState(null);
  const [selectedSession, setSelectedSession] = useState('');
  const [sessionText, setSessionText] = useState('');
  const [predictionResult, setPredictionResult] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  
  // Training state
  const [trainingData, setTrainingData] = useState([]);
  const [trainingParams, setTrainingParams] = useState({
    window_size: 10,
    anomaly_threshold: 0.7,
    learning_rate: 0.001,
    num_epochs: 50
  });
  const [isTraining, setIsTraining] = useState(false);

  useEffect(() => {
    fetchModelInfo();
    fetchTrainingHistory();
    fetchPredictionCache();
  }, []);

  const fetchModelInfo = async () => {
    try {
      const response = await fetch(apiEndpoint('/model-info'));
      if (!response.ok) throw new Error('Failed to fetch model info');
      const data = await response.json();
      setModelInfo(data);
    } catch (error) {
      console.error('Error fetching model info:', error);
      setError('Failed to load model information');
    }
  };

  const fetchTrainingHistory = async () => {
    try {
      const response = await fetch(apiEndpoint('/training-history'));
      if (!response.ok) throw new Error('Failed to fetch training history');
      const data = await response.json();
      setTrainingHistory(data.training_history || []);
    } catch (error) {
      console.error('Error fetching training history:', error);
    }
  };

  const fetchPredictionCache = async () => {
    try {
      const response = await fetch(apiEndpoint('/prediction-cache'));
      if (!response.ok) throw new Error('Failed to fetch prediction cache');
      const data = await response.json();
      setPredictionCache(data);
    } catch (error) {
      console.error('Error fetching prediction cache:', error);
    }
  };

  const handlePredict = async () => {
    if (!sessionText.trim()) {
      setError('Please enter session text');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(apiEndpoint('/predict'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_text: sessionText,
          session_id: selectedSession || `session_${Date.now()}`
        })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }
      
      const result = await response.json();
      setPredictionResult(result);
      
      // Fetch explanation if available
      if (result.session_id) {
        try {
          const explainResponse = await fetch(apiEndpoint(`/explanation/${result.session_id}`));
          if (explainResponse.ok) {
            const explanationData = await explainResponse.json();
            setExplanation(explanationData);
          }
        } catch (e) {
          console.warn('Could not fetch explanation:', e);
        }
      }
      
    } catch (error) {
      console.error('Error making prediction:', error);
      setError(`Failed to analyze session: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleTrain = async () => {
    if (trainingData.length < 10) {
      setError('Need at least 10 training sessions');
      return;
    }

    setIsTraining(true);
    setError(null);
    
    try {
      const response = await fetch(apiEndpoint('/train'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessions: trainingData,
          validation_split: 0.2,
          normal_sessions_only: true
        })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }
      
      const result = await response.json();
      alert(`Training ${result.training_stats.training_status}: ${result.message}`);
      
      // Refresh data
      await fetchModelInfo();
      await fetchTrainingHistory();
      
    } catch (error) {
      console.error('Error training model:', error);
      setError(`Training failed: ${error.message}`);
    } finally {
      setIsTraining(false);
    }
  };

  const handleLoadEJSessions = async () => {
    try {
      setLoading(true);
      
      // Load processed EJ sessions from the EJ Rule-Based Processor
      const response = await fetch('/api/v1/bert-deeplog/load-ej-sessions?include_errors=true&limit=100');
      
      if (!response.ok) {
        throw new Error(`Failed to load EJ sessions: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.success && data.sessions) {
        setTrainingData(data.sessions);
        
        // Show detailed information about loaded data
        const stats = data.data_sources;
        const preprocessing = data.preprocessing_stats;
        
        alert(`✅ Loaded ${data.sessions.length} real EJ sessions!
        
📊 Data Sources:
• Normal sessions: ${stats.total_normal}
• Error sessions: ${stats.total_errors}
• Files: ${stats.normal_file}${stats.error_file ? `, ${stats.error_file}` : ''}

🚀 BERT Preprocessing:
• Sessions with preprocessing: ${preprocessing.sessions_with_bert_preprocessing}
• Average compression: ${(preprocessing.average_compression_ratio * 100).toFixed(1)}%

Ready for BERT-DeepLog training with real ATM transaction data!`);
      } else {
        throw new Error(data.message || 'Failed to load sessions');
      }
    } catch (error) {
      console.error('Error loading EJ sessions:', error);
      alert(`Failed to load EJ sessions: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const ModelOverview = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {/* Model Status */}
      <div className="bg-white p-6 rounded-lg shadow">
        <div className="flex items-center mb-4">
          <Brain className="h-6 w-6 text-blue-500 mr-2" />
          <h3 className="text-lg font-semibold">Model Status</h3>
        </div>
        {modelInfo ? (
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-600">Trained:</span>
              <span className={`font-medium ${modelInfo.model_stats.model_info.trained ? 'text-green-600' : 'text-red-600'}`}>
                {modelInfo.model_stats.model_info.trained ? 'Yes' : 'No'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Parameters:</span>
              <span className="font-medium">{modelInfo.model_stats.model_info.parameters?.toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Device:</span>
              <span className="font-medium">{modelInfo.model_stats.model_info.device}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Sequences:</span>
              <span className="font-medium">{modelInfo.model_stats.training_data.num_sequences}</span>
            </div>
          </div>
        ) : (
          <div className="text-gray-500">Loading model info...</div>
        )}
      </div>

      {/* Training Progress */}
      <div className="bg-white p-6 rounded-lg shadow">
        <div className="flex items-center mb-4">
          <TrendingUp className="h-6 w-6 text-green-500 mr-2" />
          <h3 className="text-lg font-semibold">Training Progress</h3>
        </div>
        {trainingHistory.length > 0 ? (
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-600">Total Epochs:</span>
              <span className="font-medium">{trainingHistory.length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Final Train Loss:</span>
              <span className="font-medium">{trainingHistory[trainingHistory.length - 1]?.train_loss?.toFixed(4)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Final Val Loss:</span>
              <span className="font-medium">{trainingHistory[trainingHistory.length - 1]?.val_loss?.toFixed(4)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Last Trained:</span>
              <span className="font-medium text-sm">
                {new Date(trainingHistory[trainingHistory.length - 1]?.timestamp).toLocaleDateString()}
              </span>
            </div>
          </div>
        ) : (
          <div className="text-gray-500">No training history available</div>
        )}
      </div>

      {/* Prediction Cache */}
      <div className="bg-white p-6 rounded-lg shadow">
        <div className="flex items-center mb-4">
          <Database className="h-6 w-6 text-purple-500 mr-2" />
          <h3 className="text-lg font-semibold">Prediction Cache</h3>
        </div>
        {predictionCache ? (
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-600">Total Predictions:</span>
              <span className="font-medium">{predictionCache.total_cached_predictions}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Anomalies Found:</span>
              <span className="font-medium text-red-600">{predictionCache.anomalies_in_cache}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Anomaly Rate:</span>
              <span className="font-medium">
                {predictionCache.total_cached_predictions > 0 
                  ? ((predictionCache.anomalies_in_cache / predictionCache.total_cached_predictions) * 100).toFixed(1)
                  : 0}%
              </span>
            </div>
          </div>
        ) : (
          <div className="text-gray-500">No predictions cached</div>
        )}
      </div>
    </div>
  );

  const TrainingLossChart = () => {
    if (trainingHistory.length === 0) {
      return (
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="text-center text-gray-500">No training history available</div>
        </div>
      );
    }

    return (
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">Training Loss History</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={trainingHistory}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="epoch" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="train_loss" stroke="#8884d8" name="Training Loss" />
            <Line type="monotone" dataKey="val_loss" stroke="#82ca9d" name="Validation Loss" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  };

  const PredictionInterface = () => (
    <div className="space-y-6">
      {/* Input Section */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">Session Analysis</h3>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Session ID (optional)
            </label>
            <input
              type="text"
              value={selectedSession}
              onChange={(e) => setSelectedSession(e.target.value)}
              className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="Enter session ID or leave blank"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              EJ Session Text
            </label>
            <textarea
              value={sessionText}
              onChange={(e) => setSessionText(e.target.value)}
              className="w-full h-32 p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="Enter ABM EJ session text for analysis..."
            />
          </div>
          
          <div className="flex space-x-4">
            <button
              onClick={handlePredict}
              disabled={loading || !sessionText.trim()}
              className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400"
            >
              {loading ? <RefreshCw className="animate-spin h-4 w-4 mr-2" /> : <Eye className="h-4 w-4 mr-2" />}
              {loading ? 'Analyzing...' : 'Analyze Session'}
            </button>
            
            <button
              onClick={() => {
                setSessionText('CARD INSERTED DEVICE ERROR M_02 SUPERVISOR ENTRY CARD TAKEN');
                setSelectedSession('sample_anomaly');
              }}
              className="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700"
            >
              Load Sample
            </button>
          </div>
        </div>
      </div>

      {/* Results Section */}
      {predictionResult && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Prediction Summary */}
          <div className="bg-white p-6 rounded-lg shadow">
            <h3 className="text-lg font-semibold mb-4">Prediction Results</h3>
            <div className="space-y-3">
              <div className={`p-4 rounded-lg ${predictionResult.is_anomaly ? 'bg-red-50 border border-red-200' : 'bg-green-50 border border-green-200'}`}>
                <div className="flex items-center">
                  <AlertCircle className={`h-5 w-5 mr-2 ${predictionResult.is_anomaly ? 'text-red-500' : 'text-green-500'}`} />
                  <span className={`font-semibold ${predictionResult.is_anomaly ? 'text-red-700' : 'text-green-700'}`}>
                    {predictionResult.is_anomaly ? 'ANOMALY DETECTED' : 'NORMAL SESSION'}
                  </span>
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <span className="text-gray-600">Anomaly Probability:</span>
                  <div className="font-medium text-lg">{(predictionResult.anomaly_probability * 100).toFixed(1)}%</div>
                </div>
                <div>
                  <span className="text-gray-600">Confidence:</span>
                  <div className="font-medium text-lg">{(predictionResult.confidence * 100).toFixed(1)}%</div>
                </div>
              </div>
              
              <div>
                <span className="text-gray-600">Processing Time:</span>
                <div className="font-medium">{predictionResult.processing_time_ms.toFixed(1)}ms</div>
              </div>
            </div>
          </div>

          {/* Important Events */}
          <div className="bg-white p-6 rounded-lg shadow">
            <h3 className="text-lg font-semibold mb-4">Important Events Detected</h3>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {predictionResult.important_events.map((event, index) => (
                <div key={index} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                  <span className="font-medium">{event.token}</span>
                  <div className="text-sm text-gray-600">
                    <span className="mr-2">Importance: {(event.importance * 100).toFixed(1)}%</span>
                    <span>Pos: {event.position}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Explanation Section */}
      {explanation && (
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">Prediction Explanation</h3>
          <div className="space-y-4">
            {/* Model Reasoning */}
            <div>
              <h4 className="font-medium text-gray-700 mb-2">Model Reasoning:</h4>
              <ul className="list-disc list-inside space-y-1">
                {explanation.model_reasoning.map((reason, index) => (
                  <li key={index} className="text-gray-600">{reason}</li>
                ))}
              </ul>
            </div>
            
            {/* Event Analysis */}
            <div>
              <h4 className="font-medium text-gray-700 mb-2">Event Analysis:</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {explanation.event_analysis.slice(0, 6).map((event, index) => (
                  <div key={index} className="p-3 bg-gray-50 rounded">
                    <div className="font-medium">{event.event}</div>
                    <div className="text-sm text-gray-600">
                      <span className={`inline-block px-2 py-1 rounded text-xs mr-2 ${
                        event.contribution_type === 'Critical' ? 'bg-red-100 text-red-700' :
                        event.contribution_type === 'High' ? 'bg-orange-100 text-orange-700' :
                        event.contribution_type === 'Medium' ? 'bg-yellow-100 text-yellow-700' :
                        'bg-gray-100 text-gray-700'
                      }`}>
                        {event.contribution_type}
                      </span>
                      {event.explanation}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  const TrainingInterface = () => (
    <div className="space-y-6">
      {/* Training Configuration */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">Training Configuration</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Window Size</label>
            <input
              type="number"
              value={trainingParams.window_size}
              onChange={(e) => setTrainingParams({...trainingParams, window_size: parseInt(e.target.value)})}
              className="w-full p-2 border border-gray-300 rounded-md"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Anomaly Threshold</label>
            <input
              type="number"
              step="0.1"
              value={trainingParams.anomaly_threshold}
              onChange={(e) => setTrainingParams({...trainingParams, anomaly_threshold: parseFloat(e.target.value)})}
              className="w-full p-2 border border-gray-300 rounded-md"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Learning Rate</label>
            <input
              type="number"
              step="0.001"
              value={trainingParams.learning_rate}
              onChange={(e) => setTrainingParams({...trainingParams, learning_rate: parseFloat(e.target.value)})}
              className="w-full p-2 border border-gray-300 rounded-md"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Epochs</label>
            <input
              type="number"
              value={trainingParams.num_epochs}
              onChange={(e) => setTrainingParams({...trainingParams, num_epochs: parseInt(e.target.value)})}
              className="w-full p-2 border border-gray-300 rounded-md"
            />
          </div>
        </div>
      </div>

      {/* Training Data */}
      <div className="bg-white p-6 rounded-lg shadow">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold">Training Data</h3>
          <div className="space-x-2">
            <button
              onClick={handleLoadEJSessions}
              disabled={loading}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400"
            >
              {loading ? 'Loading...' : 'Load EJ Sessions'}
            </button>
            <button
              onClick={handleTrain}
              disabled={isTraining || trainingData.length < 10}
              className="flex items-center px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:bg-gray-400"
            >
              {isTraining ? <RefreshCw className="animate-spin h-4 w-4 mr-2" /> : <Zap className="h-4 w-4 mr-2" />}
              {isTraining ? 'Training...' : 'Start Training'}
            </button>
          </div>
        </div>
        
        <div className="space-y-2">
          <div className="flex justify-between text-sm text-gray-600">
            <span>Total Sessions: {trainingData.length}</span>
            <span>Normal: {trainingData.filter(s => !s.is_anomaly).length}</span>
            <span>Anomalies: {trainingData.filter(s => s.is_anomaly).length}</span>
          </div>
          
          {trainingData.length > 0 && (
            <div className="max-h-40 overflow-y-auto border border-gray-200 rounded">
              <table className="w-full text-sm">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="p-2 text-left">Session ID</th>
                    <th className="p-2 text-left">Type</th>
                    <th className="p-2 text-left">Text Preview</th>
                  </tr>
                </thead>
                <tbody>
                  {trainingData.slice(0, 10).map((session, index) => (
                    <tr key={index} className="border-t">
                      <td className="p-2">{session.session_id}</td>
                      <td className="p-2">
                        <span className={`px-2 py-1 rounded text-xs ${
                          session.is_anomaly ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
                        }`}>
                          {session.is_anomaly ? 'Anomaly' : 'Normal'}
                        </span>
                      </td>
                      <td className="p-2 truncate max-w-xs">{session.raw_text}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {trainingData.length > 10 && (
                <div className="p-2 text-center text-gray-500 bg-gray-50">
                  ... and {trainingData.length - 10} more sessions
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Training History Chart */}
      <TrainingLossChart />
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-100">
      <div className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center">
              <Brain className="h-8 w-8 text-blue-500 mr-3" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">BERT-Enhanced DeepLog Dashboard</h1>
                <p className="text-gray-600">Advanced sequential anomaly detection with BERT embeddings</p>
              </div>
            </div>
            <div className="flex space-x-2">
              <button
                onClick={fetchModelInfo}
                className="flex items-center px-3 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
              >
                <RefreshCw className="h-4 w-4 mr-2" />
                Refresh
              </button>
            </div>
          </div>
          
          {/* Tab Navigation */}
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-8">
              {[
                { id: 'overview', label: 'Overview', icon: BarChart3 },
                { id: 'prediction', label: 'Prediction', icon: Target },
                { id: 'training', label: 'Training', icon: Settings }
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center py-2 px-1 border-b-2 font-medium text-sm ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <tab.icon className="h-4 w-4 mr-2" />
                  {tab.label}
                </button>
              ))}
            </nav>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="flex">
              <AlertCircle className="h-5 w-5 text-red-400 mt-0.5 mr-2" />
              <div className="text-red-700">{error}</div>
            </div>
          </div>
        )}

        {activeTab === 'overview' && <ModelOverview />}
        {activeTab === 'prediction' && <PredictionInterface />}
        {activeTab === 'training' && <TrainingInterface />}
      </div>
    </div>
  );
};

export default DeepLogDashboard;
