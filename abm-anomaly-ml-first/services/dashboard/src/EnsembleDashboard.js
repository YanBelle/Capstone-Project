import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';
import { AlertCircle, Activity, TrendingUp, Clock, Shield, Database, Brain, AlertTriangle, Target, Zap } from 'lucide-react';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const EnsembleDashboard = () => {
  const [ensembleStats, setEnsembleStats] = useState({
    training_stats: null,
    model_status: null,
    recent_predictions: [],
    performance_metrics: null
  });
  const [loading, setLoading] = useState(true);
  const [trainingSessions, setTrainingSessions] = useState([]);
  const [isTraining, setIsTraining] = useState(false);
  const [testSession, setTestSession] = useState('');
  const [testResult, setTestResult] = useState(null);

  // Fetch ensemble status and stats
  const fetchEnsembleStats = async () => {
    try {
      setLoading(true);
      
      // Get training stats if model is trained
      const statsResponse = await fetch(`${API_URL}/api/ensemble_status`);
      if (statsResponse.ok) {
        const stats = await statsResponse.json();
        setEnsembleStats(stats);
      }
      
    } catch (error) {
      console.error('Error fetching ensemble stats:', error);
    } finally {
      setLoading(false);
    }
  };

  // Train ensemble model
  const trainEnsemble = async () => {
    if (trainingSessions.length === 0) {
      alert('Please add some training sessions first');
      return;
    }

    try {
      setIsTraining(true);
      
      const response = await fetch(`${API_URL}/api/train_enhanced_ensemble`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sessions: trainingSessions
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      console.log('Training result:', result);
      
      // Refresh stats after training
      await fetchEnsembleStats();
      alert('Ensemble model trained successfully!');
      
    } catch (error) {
      console.error('Error training ensemble:', error);
      alert('Error training ensemble: ' + error.message);
    } finally {
      setIsTraining(false);
    }
  };

  // Test session with ensemble
  const testSessionWithEnsemble = async () => {
    if (!testSession.trim()) {
      alert('Please enter a test session');
      return;
    }

    try {
      const response = await fetch(`${API_URL}/api/predict_enhanced`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sessions: [testSession]
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      setTestResult(result);
      
    } catch (error) {
      console.error('Error testing session:', error);
      alert('Error testing session: ' + error.message);
    }
  };

  // Add sample training sessions
  const addSampleTrainingSessions = () => {
    const sampleSessions = [
      "SESSION START\nCARD INSERTED\nPIN ENTERED\nPIN VERIFIED\nBALANCE INQUIRY\nRECEIPT PRINTED\nCARD EJECTED\nSESSION END",
      "SESSION START\nCARD INSERTED\nPIN ENTERED\nPIN VERIFIED\nWITHDRAW SELECTED\nAMOUNT ENTERED: $100\nCASH DISPENSED\nRECEIPT PRINTED\nCARD EJECTED\nSESSION END",
      "SESSION START\nCARD INSERTED\nPIN ENTERED\nPIN VERIFIED\nTRANSACTION MENU\nDEPOSIT SELECTED\nDEPOSIT COMPLETED\nRECEIPT PRINTED\nCARD EJECTED\nSESSION END"
    ];
    
    setTrainingSessions([...trainingSessions, ...sampleSessions]);
  };

  useEffect(() => {
    fetchEnsembleStats();
  }, []);

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Ensemble Model Dashboard</h1>
        <p className="text-gray-600">Advanced multi-model anomaly detection system</p>
      </div>

      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div className="bg-white p-6 rounded-lg shadow-lg border-l-4 border-blue-500">
          <div className="flex items-center">
            <Brain className="h-8 w-8 text-blue-500" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Model Status</p>
              <p className="text-2xl font-bold text-gray-900">
                {ensembleStats.model_status?.is_trained ? 'Trained' : 'Not Trained'}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-lg border-l-4 border-green-500">
          <div className="flex items-center">
            <Target className="h-8 w-8 text-green-500" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Training Sessions</p>
              <p className="text-2xl font-bold text-gray-900">
                {ensembleStats.training_stats?.n_sessions || 0}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-lg border-l-4 border-yellow-500">
          <div className="flex items-center">
            <Zap className="h-8 w-8 text-yellow-500" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Features Extracted</p>
              <p className="text-2xl font-bold text-gray-900">
                {ensembleStats.training_stats?.combined_features_shape?.[1] || 0}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-lg border-l-4 border-purple-500">
          <div className="flex items-center">
            <Activity className="h-8 w-8 text-purple-500" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">PCA Variance</p>
              <p className="text-2xl font-bold text-gray-900">
                {ensembleStats.training_stats?.pca_explained_variance ? 
                  `${(ensembleStats.training_stats.pca_explained_variance * 100).toFixed(1)}%` : 'N/A'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Training Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        {/* Training Control */}
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h2 className="text-xl font-bold text-gray-900 mb-4">Model Training</h2>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Training Sessions ({trainingSessions.length})
              </label>
              <div className="max-h-32 overflow-y-auto border rounded p-2 bg-gray-50">
                {trainingSessions.length === 0 ? (
                  <p className="text-gray-500 text-sm">No training sessions added</p>
                ) : (
                  trainingSessions.map((session, index) => (
                    <div key={index} className="text-xs mb-1 p-1 border-b">
                      Session {index + 1}: {session.substring(0, 50)}...
                    </div>
                  ))
                )}
              </div>
            </div>

            <div className="flex space-x-2">
              <button
                onClick={addSampleTrainingSessions}
                className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 text-sm"
              >
                Add Sample Sessions
              </button>
              
              <button
                onClick={trainEnsemble}
                disabled={isTraining || trainingSessions.length === 0}
                className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 disabled:bg-gray-400 text-sm"
              >
                {isTraining ? 'Training...' : 'Train Ensemble'}
              </button>
            </div>
          </div>
        </div>

        {/* Test Session */}
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h2 className="text-xl font-bold text-gray-900 mb-4">Test Session</h2>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Enter ABM Session Text
              </label>
              <textarea
                value={testSession}
                onChange={(e) => setTestSession(e.target.value)}
                className="w-full h-24 p-2 border rounded text-sm"
                placeholder="POWER-UP/RESET&#10;HARDWARE ERROR - CARD READER MALFUNCTION&#10;RECOVERY FAILED - UNABLE TO INITIALIZE"
              />
            </div>

            <button
              onClick={testSessionWithEnsemble}
              disabled={!ensembleStats.model_status?.is_trained}
              className="px-4 py-2 bg-purple-500 text-white rounded hover:bg-purple-600 disabled:bg-gray-400 text-sm"
            >
              Test with Ensemble
            </button>

            {testResult && (
              <div className="mt-4 p-3 border rounded bg-gray-50">
                <h3 className="font-medium text-gray-900 mb-2">Test Results:</h3>
                <pre className="text-xs overflow-auto">
                  {JSON.stringify(testResult, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Model Architecture */}
      <div className="bg-white p-6 rounded-lg shadow-lg mb-8">
        <h2 className="text-xl font-bold text-gray-900 mb-4">Ensemble Architecture</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* One-Class SVM */}
          <div className="border rounded-lg p-4 bg-blue-50">
            <h3 className="font-medium text-blue-900 mb-2">One-Class SVM</h3>
            <ul className="text-sm text-blue-800 space-y-1">
              <li>• Text feature specialist</li>
              <li>• TF-IDF vectorization</li>
              <li>• Hardware term detection</li>
              <li>• RBF kernel boundary</li>
            </ul>
            <div className="mt-2 text-xs text-blue-600">
              Weight: 40% in ensemble
            </div>
          </div>

          {/* Isolation Forest */}
          <div className="border rounded-lg p-4 bg-green-50">
            <h3 className="font-medium text-green-900 mb-2">Isolation Forest</h3>
            <ul className="text-sm text-green-800 space-y-1">
              <li>• Multivariate outlier detection</li>
              <li>• Combined feature analysis</li>
              <li>• Isolation tree ensemble</li>
              <li>• PCA dimensionality reduction</li>
            </ul>
            <div className="mt-2 text-xs text-green-600">
              Weight: 35% in ensemble
            </div>
          </div>

          {/* DBSCAN Clustering */}
          <div className="border rounded-lg p-4 bg-purple-50">
            <h3 className="font-medium text-purple-900 mb-2">DBSCAN Clustering</h3>
            <ul className="text-sm text-purple-800 space-y-1">
              <li>• Density-based clustering</li>
              <li>• Pattern discovery</li>
              <li>• Outlier identification</li>
              <li>• Feature space analysis</li>
            </ul>
            <div className="mt-2 text-xs text-purple-600">
              Weight: 25% in ensemble
            </div>
          </div>
        </div>
      </div>

      {/* Training Statistics */}
      {ensembleStats.training_stats && (
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h2 className="text-xl font-bold text-gray-900 mb-4">Training Statistics</h2>
          
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {ensembleStats.training_stats.text_features_shape?.[1] || 0}
              </div>
              <div className="text-sm text-gray-600">Text Features</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {ensembleStats.training_stats.numerical_features_shape?.[1] || 0}
              </div>
              <div className="text-sm text-gray-600">Numerical Features</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {ensembleStats.training_stats.combined_features_shape?.[1] || 0}
              </div>
              <div className="text-sm text-gray-600">Combined Features</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">
                {ensembleStats.training_stats.n_sessions || 0}
              </div>
              <div className="text-sm text-gray-600">Training Sessions</div>
            </div>
          </div>
        </div>
      )}

      {loading && (
        <div className="text-center py-8">
          <div className="text-gray-600">Loading ensemble dashboard...</div>
        </div>
      )}
    </div>
  );
};

export default EnsembleDashboard;
