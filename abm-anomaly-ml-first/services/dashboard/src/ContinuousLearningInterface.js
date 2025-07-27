import React, { useState, useEffect } from 'react';
import { RefreshCw, CheckCircle, XCircle, AlertTriangle, TrendingUp, Database, Play, Pause } from 'lucide-react';
import Layout from './Layout';

const ContinuousLearningInterface = () => {
  const [learningStatus, setLearningStatus] = useState({
    isActive: false,
    lastUpdate: null,
    totalFeedback: 0,
    processedFeedback: 0,
    modelAccuracy: 0.85,
    pendingRetraining: false
  });
  
  const [recentFeedback, setRecentFeedback] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchLearningStatus();
    fetchRecentFeedback();
  }, []);

  const fetchLearningStatus = async () => {
    try {
      // Mock data for demonstration
      setLearningStatus({
        isActive: true,
        lastUpdate: new Date().toISOString(),
        totalFeedback: 156,
        processedFeedback: 142,
        modelAccuracy: 0.87,
        pendingRetraining: false
      });
    } catch (err) {
      console.error('Error fetching learning status:', err);
      setError('Failed to fetch learning status');
    }
  };

  const fetchRecentFeedback = async () => {
    try {
      // Mock data for demonstration
      const mockFeedback = [
        {
          id: 1,
          sessionId: 'ABM250_20250726_001',
          originalPrediction: 'anomaly',
          expertLabel: 'normal',
          confidence: 0.92,
          timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
          processed: true,
          anomalyType: 'dispense_failure'
        },
        {
          id: 2,
          sessionId: 'ABM250_20250726_002',
          originalPrediction: 'normal',
          expertLabel: 'anomaly',
          confidence: 0.45,
          timestamp: new Date(Date.now() - 4 * 60 * 60 * 1000).toISOString(),
          processed: false,
          anomalyType: 'timeout_error'
        },
        {
          id: 3,
          sessionId: 'ABM250_20250726_003',
          originalPrediction: 'anomaly',
          expertLabel: 'anomaly',
          confidence: 0.88,
          timestamp: new Date(Date.now() - 6 * 60 * 60 * 1000).toISOString(),
          processed: true,
          anomalyType: 'hardware_error'
        }
      ];
      setRecentFeedback(mockFeedback);
    } catch (err) {
      console.error('Error fetching recent feedback:', err);
      setError('Failed to fetch recent feedback');
    }
  };

  const toggleLearning = async () => {
    setLoading(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      setLearningStatus(prev => ({
        ...prev,
        isActive: !prev.isActive,
        lastUpdate: new Date().toISOString()
      }));
    } catch (err) {
      console.error('Error toggling learning:', err);
      setError('Failed to toggle learning status');
    } finally {
      setLoading(false);
    }
  };

  const triggerRetraining = async () => {
    setLoading(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 2000));
      setLearningStatus(prev => ({
        ...prev,
        pendingRetraining: true,
        lastUpdate: new Date().toISOString()
      }));
    } catch (err) {
      console.error('Error triggering retraining:', err);
      setError('Failed to trigger retraining');
    } finally {
      setLoading(false);
    }
  };

  const processPendingFeedback = async () => {
    setLoading(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1500));
      setLearningStatus(prev => ({
        ...prev,
        processedFeedback: prev.totalFeedback,
        lastUpdate: new Date().toISOString()
      }));
      // Mark all feedback as processed
      setRecentFeedback(prev => prev.map(item => ({ ...item, processed: true })));
    } catch (err) {
      console.error('Error processing feedback:', err);
      setError('Failed to process feedback');
    } finally {
      setLoading(false);
    }
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'Never';
    return new Date(timestamp).toLocaleString();
  };

  const getFeedbackStatusIcon = (originalPrediction, expertLabel) => {
    if (originalPrediction === expertLabel) {
      return <CheckCircle className="h-5 w-5 text-green-500" />;
    } else {
      return <XCircle className="h-5 w-5 text-red-500" />;
    }
  };

  return (
    <Layout>
      <div className="p-6 space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Continuous Learning</h1>
          <p className="text-gray-600 mt-1">Monitor and control the ML model's continuous learning process</p>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="flex">
              <AlertTriangle className="h-5 w-5 text-red-400" />
              <div className="ml-3">
                <h3 className="text-sm font-medium text-red-800">Error</h3>
                <p className="text-sm text-red-700 mt-1">{error}</p>
              </div>
              <button
                onClick={() => setError(null)}
                className="ml-auto text-red-400 hover:text-red-600"
              >
                <XCircle className="h-5 w-5" />
              </button>
            </div>
          </div>
        )}

        {/* Status Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className={`h-8 w-8 rounded-full flex items-center justify-center ${learningStatus.isActive ? 'bg-green-100' : 'bg-gray-100'}`}>
                  {learningStatus.isActive ? (
                    <Play className="h-5 w-5 text-green-600" />
                  ) : (
                    <Pause className="h-5 w-5 text-gray-600" />
                  )}
                </div>
              </div>
              <div className="ml-5 w-0 flex-1">
                <dl>
                  <dt className="text-sm font-medium text-gray-500 truncate">Learning Status</dt>
                  <dd className="text-lg font-medium text-gray-900">
                    {learningStatus.isActive ? 'Active' : 'Inactive'}
                  </dd>
                </dl>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <Database className="h-8 w-8 text-blue-600" />
              </div>
              <div className="ml-5 w-0 flex-1">
                <dl>
                  <dt className="text-sm font-medium text-gray-500 truncate">Feedback Count</dt>
                  <dd className="text-lg font-medium text-gray-900">
                    {learningStatus.processedFeedback} / {learningStatus.totalFeedback}
                  </dd>
                </dl>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <TrendingUp className="h-8 w-8 text-green-600" />
              </div>
              <div className="ml-5 w-0 flex-1">
                <dl>
                  <dt className="text-sm font-medium text-gray-500 truncate">Model Accuracy</dt>
                  <dd className="text-lg font-medium text-gray-900">
                    {(learningStatus.modelAccuracy * 100).toFixed(1)}%
                  </dd>
                </dl>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <RefreshCw className={`h-8 w-8 ${learningStatus.pendingRetraining ? 'text-orange-600' : 'text-gray-400'}`} />
              </div>
              <div className="ml-5 w-0 flex-1">
                <dl>
                  <dt className="text-sm font-medium text-gray-500 truncate">Retraining</dt>
                  <dd className="text-lg font-medium text-gray-900">
                    {learningStatus.pendingRetraining ? 'Pending' : 'Up to date'}
                  </dd>
                </dl>
              </div>
            </div>
          </div>
        </div>

        {/* Control Panel */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Learning Controls</h3>
          <div className="flex flex-wrap gap-4">
            <button
              onClick={toggleLearning}
              disabled={loading}
              className={`inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white ${
                learningStatus.isActive 
                  ? 'bg-red-600 hover:bg-red-700' 
                  : 'bg-green-600 hover:bg-green-700'
              } disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              {loading ? (
                <RefreshCw className="animate-spin -ml-1 mr-2 h-4 w-4" />
              ) : learningStatus.isActive ? (
                <Pause className="-ml-1 mr-2 h-4 w-4" />
              ) : (
                <Play className="-ml-1 mr-2 h-4 w-4" />
              )}
              {learningStatus.isActive ? 'Stop Learning' : 'Start Learning'}
            </button>

            <button
              onClick={processPendingFeedback}
              disabled={loading || learningStatus.processedFeedback >= learningStatus.totalFeedback}
              className="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Database className="-ml-1 mr-2 h-4 w-4" />
              Process Pending Feedback
            </button>

            <button
              onClick={triggerRetraining}
              disabled={loading}
              className="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <RefreshCw className="-ml-1 mr-2 h-4 w-4" />
              Trigger Retraining
            </button>
          </div>
          
          <div className="mt-4 text-sm text-gray-600">
            <p><strong>Last Update:</strong> {formatTimestamp(learningStatus.lastUpdate)}</p>
          </div>
        </div>

        {/* Recent Feedback */}
        <div className="bg-white shadow rounded-lg">
          <div className="px-4 py-5 sm:p-6">
            <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">
              Recent Expert Feedback
            </h3>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Session ID</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Original Prediction</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Expert Label</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Confidence</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Timestamp</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Match</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {recentFeedback.map((feedback) => (
                    <tr key={feedback.id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {feedback.sessionId}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                          feedback.originalPrediction === 'anomaly' 
                            ? 'bg-red-100 text-red-800' 
                            : 'bg-green-100 text-green-800'
                        }`}>
                          {feedback.originalPrediction}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                          feedback.expertLabel === 'anomaly' 
                            ? 'bg-red-100 text-red-800' 
                            : 'bg-green-100 text-green-800'
                        }`}>
                          {feedback.expertLabel}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {(feedback.confidence * 100).toFixed(1)}%
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                          feedback.processed 
                            ? 'bg-green-100 text-green-800' 
                            : 'bg-yellow-100 text-yellow-800'
                        }`}>
                          {feedback.processed ? 'Processed' : 'Pending'}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {formatTimestamp(feedback.timestamp)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        {getFeedbackStatusIcon(feedback.originalPrediction, feedback.expertLabel)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            {recentFeedback.length === 0 && (
              <div className="text-center py-8 text-gray-500">
                No recent feedback available
              </div>
            )}
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default ContinuousLearningInterface;
