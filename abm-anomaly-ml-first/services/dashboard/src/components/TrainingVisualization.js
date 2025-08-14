import React, { useState, useEffect } from 'react';
import { BarChart3, Target, Clock, Check, AlertTriangle, RefreshCw, Eye } from 'lucide-react';
import apiConfig from '../config/api';

const TrainingVisualization = () => {
  const [trainingResults, setTrainingResults] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [trainingInfo, setTrainingInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');
  const [isTraining, setIsTraining] = useState(false);

  useEffect(() => {
    fetchTrainingData();
    fetchTrainingStatus();
    fetchTrainingInfo();
  }, []);

  const fetchTrainingData = async () => {
    try {
      const response = await fetch(apiConfig.endpoint('/api/v1/models/training-results'));
      const data = await response.json();
      console.log('Training data response:', data); // Debug log
      if (data.status === 'success') {
        setTrainingResults(data);
        console.log('Training results set:', data); // Debug log
      } else {
        console.error('Training data fetch failed:', data);
      }
    } catch (error) {
      console.error('Error fetching training results:', error);
    }
  };

  const fetchTrainingStatus = async () => {
    try {
      const response = await fetch(apiConfig.endpoint('/api/v1/expert/training-status'));
      const data = await response.json();
      setTrainingStatus(data);
    } catch (error) {
      console.error('Error fetching training status:', error);
    }
  };

  const fetchTrainingInfo = async () => {
    try {
      const response = await fetch(apiConfig.endpoint('/api/v1/expert/training-data-info'));
      const data = await response.json();
      setTrainingInfo(data);
    } catch (error) {
      console.error('Error fetching training info:', error);
    } finally {
      setLoading(false);
    }
  };

  const startTraining = async () => {
    setIsTraining(true);
    try {
      const response = await fetch(apiConfig.endpoint('/api/v1/expert/train-supervised'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      const data = await response.json();
      if (data.status === 'success') {
        // Start polling for status updates
        const pollInterval = setInterval(async () => {
          await fetchTrainingStatus();
          await fetchTrainingData();
        }, 2000);
        
        // Stop polling after 30 seconds
        setTimeout(() => {
          clearInterval(pollInterval);
          setIsTraining(false);
        }, 30000);
      }
    } catch (error) {
      console.error('Error starting training:', error);
      setIsTraining(false);
    }
  };

  const PerformanceMetricsCard = ({ model }) => {
    if (!model || !model.performance_metrics) return null;

    const metrics = model.performance_metrics;
    
    return (
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Performance Metrics</h3>
          <span className={`px-3 py-1 rounded-full text-sm font-medium ${
            model.is_active ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
          }`}>
            {model.is_active ? 'Active Model' : 'Previous Version'}
          </span>
        </div>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">
              {(metrics.accuracy * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-gray-600">Accuracy</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">
              {(metrics.precision * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-gray-600">Precision</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">
              {(metrics.recall * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-gray-600">Recall</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-orange-600">
              {(metrics.f1_score * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-gray-600">F1-Score</div>
          </div>
        </div>

        {/* Training Details */}
        <div className="border-t pt-4">
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Training Samples:</span>
              <span className="font-medium ml-2">{model.training_samples?.toLocaleString()}</span>
            </div>
            <div>
              <span className="text-gray-600">Training Date:</span>
              <span className="font-medium ml-2">
                {new Date(model.training_date).toLocaleString()}
              </span>
            </div>
            <div>
              <span className="text-gray-600">Model Version:</span>
              <span className="font-medium ml-2">{model.model_version}</span>
            </div>
            <div>
              <span className="text-gray-600">Model Type:</span>
              <span className="font-medium ml-2">Random Forest</span>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const ClassDistributionChart = ({ model }) => {
    if (!model || !model.performance_metrics || !model.performance_metrics.class_distribution) return null;

    const classData = model.performance_metrics.class_distribution;
    const totalSamples = Object.values(classData).reduce((sum, count) => sum + count, 0);
    
    return (
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Class Distribution</h3>
        <div className="space-y-3">
          {Object.entries(classData)
            .sort(([,a], [,b]) => b - a)
            .map(([className, count]) => {
              const percentage = (count / totalSamples) * 100;
              return (
                <div key={className} className="flex items-center">
                  <div className="w-40 text-sm text-gray-700 truncate">{className}</div>
                  <div className="flex-1 mx-3">
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-blue-500 h-2 rounded-full" 
                        style={{width: `${percentage}%`}}
                      ></div>
                    </div>
                  </div>
                  <div className="w-20 text-sm text-gray-600 text-right">
                    {count} ({percentage.toFixed(1)}%)
                  </div>
                </div>
              );
            })}
        </div>
      </div>
    );
  };

  const ConfusionMatrixView = ({ model }) => {
    if (!model || !model.performance_metrics || !model.performance_metrics.confusion_matrix) return null;

    const matrix = model.performance_metrics.confusion_matrix;
    const classNames = Object.keys(model.performance_metrics.class_distribution || {});
    
    return (
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Confusion Matrix</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr>
                <th className="border border-gray-300 p-2 bg-gray-50">Actual \ Predicted</th>
                {classNames.slice(0, matrix[0]?.length || 0).map((name, i) => (
                  <th key={i} className="border border-gray-300 p-2 bg-gray-50 text-xs">
                    {name.substring(0, 10)}...
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {matrix.map((row, i) => (
                <tr key={i}>
                  <td className="border border-gray-300 p-2 bg-gray-50 font-medium text-xs">
                    {classNames[i]?.substring(0, 15)}...
                  </td>
                  {row.map((value, j) => (
                    <td 
                      key={j} 
                      className={`border border-gray-300 p-2 text-center ${
                        i === j ? 'bg-green-100' : value > 0 ? 'bg-red-50' : 'bg-white'
                      }`}
                    >
                      {value}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  const TrainingStatusCard = () => {
    if (!trainingStatus) return null;

    const statusIcon = {
      'idle': <Clock className="w-5 h-5 text-gray-500" />,
      'starting': <RefreshCw className="w-5 h-5 text-blue-500 animate-spin" />,
      'running': <RefreshCw className="w-5 h-5 text-blue-500 animate-spin" />,
      'completed': <Check className="w-5 h-5 text-green-500" />,
      'error': <AlertTriangle className="w-5 h-5 text-red-500" />
    }[trainingStatus.status] || <Clock className="w-5 h-5 text-gray-500" />;

    const statusColor = {
      'idle': 'text-gray-600',
      'starting': 'text-blue-600',
      'running': 'text-blue-600', 
      'completed': 'text-green-600',
      'error': 'text-red-600'
    }[trainingStatus.status] || 'text-gray-600';

    return (
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Training Status</h3>
          <button
            onClick={startTraining}
            disabled={isTraining || trainingStatus.status === 'running'}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
          >
            <Target className="w-4 h-4" />
            <span>{isTraining ? 'Training...' : 'Start Training'}</span>
          </button>
        </div>

        <div className="flex items-center space-x-3 mb-3">
          {statusIcon}
          <span className={`font-medium ${statusColor}`}>
            {trainingStatus.status.charAt(0).toUpperCase() + trainingStatus.status.slice(1)}
          </span>
        </div>

        <p className="text-gray-600 mb-2">{trainingStatus.message}</p>
        
        {trainingStatus.progress > 0 && (
          <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
            <div 
              className="bg-blue-500 h-2 rounded-full transition-all duration-300" 
              style={{width: `${trainingStatus.progress}%`}}
            ></div>
          </div>
        )}

        <div className="text-xs text-gray-500">
          Last updated: {new Date(trainingStatus.timestamp).toLocaleString()}
        </div>
      </div>
    );
  };

  const DataInfoCard = () => {
    if (!trainingInfo) return null;

    return (
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Training Data Overview</h3>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">
              {trainingInfo.ml_sessions?.total?.toLocaleString() || 0}
            </div>
            <div className="text-sm text-gray-600">Total Sessions</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">
              {trainingInfo.labeled_anomalies?.total?.toLocaleString() || 0}
            </div>
            <div className="text-sm text-gray-600">Labeled Anomalies</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">
              {trainingInfo.training_ready?.candidates?.toLocaleString() || 0}
            </div>
            <div className="text-sm text-gray-600">Training Ready</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-orange-600">
              {trainingInfo.training_ready?.unique_labels || 0}
            </div>
            <div className="text-sm text-gray-600">Unique Labels</div>
          </div>
        </div>

        <div className="border-t pt-4">
          <div className="flex items-center space-x-2 mb-2">
            <span className="text-sm font-medium text-gray-700">Available Labels:</span>
            <span className={`px-2 py-1 rounded text-xs ${
              trainingInfo.training_possible ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'
            }`}>
              {trainingInfo.training_possible ? 'Ready for Training' : 'Needs More Data'}
            </span>
          </div>
          <div className="text-sm text-gray-600">
            {trainingInfo.training_ready?.available_labels || 'No labels available'}
          </div>
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <RefreshCw className="w-8 h-8 animate-spin text-blue-500" />
        <span className="ml-2 text-gray-600">Loading training data...</span>
      </div>
    );
  }

  const latestModel = trainingResults?.models?.[0];
  console.log('Latest model:', latestModel, 'Training results:', trainingResults); // Debug log

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900 flex items-center">
          <BarChart3 className="w-8 h-8 mr-3 text-blue-600" />
          Model Training Dashboard
        </h2>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {[
            { id: 'overview', name: 'Overview', icon: Eye },
            { id: 'performance', name: 'Performance', icon: Target },
            { id: 'analysis', name: 'Analysis', icon: BarChart3 }
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
              <tab.icon className="w-4 h-4 mr-2" />
              {tab.name}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <TrainingStatusCard />
          <DataInfoCard />
          {latestModel && <PerformanceMetricsCard model={latestModel} />}
          {/* Debug: Always show performance metrics if we have models */}
          {!latestModel && trainingResults?.models?.length > 0 && (
            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
              <strong>Debug:</strong> Models found ({trainingResults.models.length}) but latestModel is null
            </div>
          )}
        </div>
      )}

      {activeTab === 'performance' && latestModel && (
        <div className="space-y-6">
          <PerformanceMetricsCard model={latestModel} />
          <ClassDistributionChart model={latestModel} />
        </div>
      )}

      {activeTab === 'analysis' && latestModel && (
        <div className="space-y-6">
          <ConfusionMatrixView model={latestModel} />
          {trainingResults?.models?.length > 1 && (
            <div className="bg-white rounded-lg shadow-sm border p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Training History</h3>
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2">Version</th>
                      <th className="text-left py-2">Date</th>
                      <th className="text-left py-2">Accuracy</th>
                      <th className="text-left py-2">Samples</th>
                      <th className="text-left py-2">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {trainingResults.models.map((model, index) => (
                      <tr key={index} className="border-b">
                        <td className="py-2">{model.model_version}</td>
                        <td className="py-2">{new Date(model.training_date).toLocaleDateString()}</td>
                        <td className="py-2">{(model.performance_metrics.accuracy * 100).toFixed(1)}%</td>
                        <td className="py-2">{model.training_samples?.toLocaleString()}</td>
                        <td className="py-2">
                          <span className={`px-2 py-1 rounded text-xs ${
                            model.is_active ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                          }`}>
                            {model.is_active ? 'Active' : 'Archived'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Debug Section - Remove in production */}
      <div className="mt-6 bg-gray-100 p-4 rounded-lg">
        <h3 className="text-lg font-semibold mb-2">Debug Information</h3>
        <div className="text-sm space-y-2">
          <div><strong>Loading:</strong> {loading ? 'true' : 'false'}</div>
          <div><strong>Training Results:</strong> {trainingResults ? 'Available' : 'null'}</div>
          <div><strong>Models Count:</strong> {trainingResults?.models?.length || 0}</div>
          <div><strong>Latest Model:</strong> {latestModel ? 'Available' : 'null'}</div>
          <div><strong>Active Tab:</strong> {activeTab}</div>
          {trainingResults && (
            <div className="bg-white p-2 rounded border">
              <strong>Training Results Status:</strong> {trainingResults.status}
            </div>
          )}
          {latestModel && (
            <div className="bg-white p-2 rounded border">
              <strong>Latest Model Version:</strong> {latestModel.model_version}
            </div>
          )}
        </div>
      </div>

      {!latestModel && (
        <div className="text-center py-12">
          <BarChart3 className="w-16 h-16 mx-auto text-gray-400 mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Training Results Yet</h3>
          <p className="text-gray-500 mb-4">
            Start training your first model to see performance metrics and analysis.
          </p>
          <button
            onClick={startTraining}
            disabled={!trainingInfo?.training_possible}
            className="px-6 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2 mx-auto"
          >
            <Target className="w-5 h-5" />
            <span>Start Training</span>
          </button>
        </div>
      )}
    </div>
  );
};

export default TrainingVisualization;
