import React, { useState, useEffect } from 'react';
import { ChevronLeft, ChevronRight, Save, RefreshCw, CheckCircle, XCircle, AlertTriangle, Brain, Tag, Filter } from 'lucide-react';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const ExpertLabelingInterface = () => {
  const [sessions, setSessions] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [labels, setLabels] = useState({});
  const [predefinedLabels, setPredefinedLabels] = useState([]);
  const [customLabel, setCustomLabel] = useState('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [filter, setFilter] = useState('unlabeled');
  const [stats, setStats] = useState({
    total: 0,
    labeled: 0,
    excluded: 0
  });
  const [trainingStatus, setTrainingStatus] = useState(null);

  const fetchAnomalies = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/api/v1/expert/anomalies?filter=${filter}`);
      const data = await response.json();
      setSessions(data.sessions);
      setStats(data.stats);
      
      const existingLabels = {};
      data.sessions.forEach(session => {
        if (session.expert_label) {
          existingLabels[session.session_id] = {
            label: session.expert_label,
            excluded: session.is_excluded || false
          };
        }
      });
      setLabels(existingLabels);
    } catch (error) {
      console.error('Error fetching anomalies:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchPredefinedLabels = async () => {
    try {
      const response = await fetch(`${API_URL}/api/v1/expert/labels`);
      const data = await response.json();
      setPredefinedLabels(data.labels);
    } catch (error) {
      console.error('Error fetching labels:', error);
    }
  };

  useEffect(() => {
    fetchAnomalies();
    fetchPredefinedLabels();
  }, [filter]);

  const currentSession = sessions[currentIndex] || null;

  const handleLabelChange = (sessionId, label, excluded = false) => {
    setLabels(prev => ({
      ...prev,
      [sessionId]: { label, excluded }
    }));
  };

  const handleMultiLabelToggle = (sessionId, label) => {
    setLabels(prev => {
      const currentLabels = prev[sessionId]?.labels || [];
      const isSelected = currentLabels.includes(label);
      
      const updatedLabels = isSelected 
        ? currentLabels.filter(l => l !== label)
        : [...currentLabels, label];
      
      return {
        ...prev,
        [sessionId]: { 
          ...prev[sessionId], // Preserve existing properties including multiMode
          labels: updatedLabels,
          label: updatedLabels.join(', '), // For backward compatibility
          excluded: false 
        }
      };
    });
  };

  const handleNext = () => {
    if (currentIndex < sessions.length - 1) {
      setCurrentIndex(currentIndex + 1);
    }
  };

  const handlePrevious = () => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1);
    }
  };

  const handleSaveLabels = async () => {
    setSaving(true);
    try {
      const labelData = Object.entries(labels).map(([sessionId, data]) => ({
        session_id: sessionId,
        label: data.labels?.length > 0 ? data.labels.join(', ') : data.label,
        labels: data.labels || (data.label ? [data.label] : []), // Multi-label array
        is_excluded: data.excluded,
        is_multi_label: data.labels?.length > 1 || false
      }));

      const response = await fetch(`${API_URL}/api/v1/expert/save-labels`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ labels: labelData })
      });

      if (response.ok) {
        alert('Labels saved successfully!');
        fetchAnomalies();
      }
    } catch (error) {
      console.error('Error saving labels:', error);
      alert('Failed to save labels');
    } finally {
      setSaving(false);
    }
  };

  const handleAddCustomLabel = () => {
    if (customLabel && !predefinedLabels.includes(customLabel)) {
      setPredefinedLabels([...predefinedLabels, customLabel]);
      setCustomLabel('');
    }
  };

  const handleTrainModel = async () => {
    if (stats.labeled < 10) {
      alert('Please label at least 10 anomalies before training');
      return;
    }

    setTrainingStatus('training');
    try {
      const response = await fetch(`${API_URL}/api/v1/expert/train-supervised`, {
        method: 'POST'
      });

      if (response.ok) {
        const result = await response.json();
        setTrainingStatus('completed');
        alert(`Training started! ${result.training_samples} samples, ${result.unique_labels} unique labels`);
      }
    } catch (error) {
      console.error('Error training model:', error);
      setTrainingStatus('error');
      alert('Training failed');
    }
  };

  const formatPatterns = (patterns) => {
    if (!patterns || patterns.length === 0) return 'None detected';
    return patterns.map(p => p.replace(/_/g, ' ').toUpperCase()).join(', ');
  };

  const formatEvents = (events) => {
    if (!events || events.length === 0) return 'No critical events';
    return events.join('; ');
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <RefreshCw className="w-12 h-12 animate-spin mx-auto mb-4" />
          <p>Loading anomalies for review...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex justify-between items-center mb-4">
          <div>
            <h1 className="text-2xl font-bold flex items-center">
              <Brain className="w-8 h-8 mr-2 text-purple-600" />
              Expert Anomaly Labeling
            </h1>
            <p className="text-gray-600 mt-1">Review and label ML-detected anomalies for supervised learning</p>
          </div>
          <button
            onClick={handleTrainModel}
            disabled={stats.labeled < 10 || trainingStatus === 'training'}
            className={`px-6 py-3 rounded-lg font-medium flex items-center ${
              stats.labeled >= 10 
                ? 'bg-purple-600 text-white hover:bg-purple-700' 
                : 'bg-gray-300 text-gray-500 cursor-not-allowed'
            }`}
          >
            <Brain className="w-5 h-5 mr-2" />
            {trainingStatus === 'training' ? 'Training...' : 'Train Supervised Model'}
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-4 gap-4">
          <div className="bg-gray-50 p-4 rounded">
            <p className="text-sm text-gray-600">Total Anomalies</p>
            <p className="text-2xl font-bold">{stats.total}</p>
          </div>
          <div className="bg-green-50 p-4 rounded">
            <p className="text-sm text-gray-600">Labeled</p>
            <p className="text-2xl font-bold text-green-600">{stats.labeled}</p>
          </div>
          <div className="bg-red-50 p-4 rounded">
            <p className="text-sm text-gray-600">Excluded</p>
            <p className="text-2xl font-bold text-red-600">{stats.excluded}</p>
          </div>
          <div className="bg-blue-50 p-4 rounded">
            <p className="text-sm text-gray-600">Progress</p>
            <p className="text-2xl font-bold text-blue-600">
              {stats.total > 0 ? ((stats.labeled / stats.total) * 100).toFixed(0) : 0}%
            </p>
          </div>
        </div>
      </div>

      {/* Filter Controls */}
      <div className="bg-white rounded-lg shadow-md p-4">
        <div className="flex items-center space-x-4">
          <Filter className="w-5 h-5 text-gray-600" />
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="px-4 py-2 border rounded-lg"
          >
            <option value="all">All Anomalies</option>
            <option value="unlabeled">Unlabeled Only</option>
            <option value="labeled">Labeled Only</option>
          </select>
          <div className="flex-1" />
          <button
            onClick={handleSaveLabels}
            disabled={saving || Object.keys(labels).length === 0}
            className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 flex items-center"
          >
            <Save className="w-4 h-4 mr-2" />
            {saving ? 'Saving...' : 'Save All Labels'}
          </button>
        </div>
      </div>

      {/* Main Content */}
      {currentSession ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Session Details */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg font-semibold">
                Session {currentIndex + 1} of {sessions.length}
              </h2>
              <span className={`px-3 py-1 rounded text-sm font-medium ${
                currentSession.anomaly_type?.startsWith('cluster_') 
                  ? 'bg-purple-100 text-purple-800'
                  : 'bg-gray-100 text-gray-800'
              }`}>
                {currentSession.anomaly_type || 'Unclassified'}
              </span>
            </div>

            <div className="space-y-4">
              <div>
                <p className="text-sm text-gray-600">Session ID</p>
                <p className="font-mono">{currentSession.session_id}</p>
              </div>

              <div>
                <p className="text-sm text-gray-600">Anomaly Score</p>
                <div className="flex items-center">
                  <div className="flex-1 bg-gray-200 rounded-full h-4 mr-3">
                    <div 
                      className="bg-red-500 h-4 rounded-full"
                      style={{ width: `${(currentSession.anomaly_score || 0) * 100}%` }}
                    />
                  </div>
                  <span className="font-medium">{(currentSession.anomaly_score || 0).toFixed(3)}</span>
                </div>
              </div>

              <div>
                <p className="text-sm text-gray-600">Detected Patterns</p>
                <p className="text-sm">{formatPatterns(currentSession.detected_patterns)}</p>
              </div>

              <div>
                <p className="text-sm text-gray-600">Critical Events</p>
                <p className="text-sm text-red-600">{formatEvents(currentSession.critical_events)}</p>
              </div>

              <div>
                <p className="text-sm text-gray-600 mb-2">Raw Log Preview</p>
                <pre 
                  className="bg-gray-50 p-4 rounded text-xs overflow-x-auto"
                  style={{ height: 'fit-content' }}
                >
                  {currentSession.raw_text || 'No log data available'}
                </pre>
              </div>
            </div>
          </div>

          {/* Labeling Controls */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-lg font-semibold mb-4 flex items-center">
              <Tag className="w-5 h-5 mr-2" />
              Assign Label
            </h2>

            {/* Current Label Status */}
            {currentSession && labels[currentSession.session_id] && (
              <div className={`mb-4 p-4 rounded-lg ${
                labels[currentSession.session_id].excluded 
                  ? 'bg-red-50 border border-red-200' 
                  : 'bg-green-50 border border-green-200'
              }`}>
                <p className="text-sm font-medium">
                  {labels[currentSession.session_id].excluded 
                    ? 'Marked as NOT an anomaly' 
                    : labels[currentSession.session_id].labels?.length > 0
                      ? `Labeled as: ${labels[currentSession.session_id].labels.join(', ')}`
                      : `Labeled as: ${labels[currentSession.session_id].label || 'None'}`}
                </p>
                {labels[currentSession.session_id].labels?.length > 1 && (
                  <p className="text-xs text-green-600 mt-1">
                    Multi-label assignment: {labels[currentSession.session_id].labels.length} anomaly types
                  </p>
                )}
              </div>
            )}

            {/* Multi-Label Selection Mode Toggle */}
            {currentSession && (
              <div className="mb-4 p-3 bg-blue-50 rounded-lg">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={labels[currentSession.session_id]?.multiMode || false}
                    onChange={(e) => {
                      const sessionId = currentSession.session_id;
                      setLabels(prev => ({
                        ...prev,
                        [sessionId]: {
                          ...prev[sessionId],
                          multiMode: e.target.checked,
                          labels: prev[sessionId]?.labels || []
                        }
                      }));
                    }}
                    className="mr-2"
                  />
                  <span className="text-sm font-medium text-blue-700">
                    Multi-label mode (select multiple anomaly types)
                  </span>
                </label>
                <p className="text-xs text-blue-600 mt-1">
                  Enable this when a session contains multiple different types of anomalies
                </p>
              </div>
            )}

            {/* Predefined Labels */}
            {currentSession && (
              <div className="space-y-2 mb-6">
                <p className="text-sm font-medium text-gray-700">
                  {labels[currentSession.session_id]?.multiMode ? 'Select Anomaly Types:' : 'Select Single Anomaly Type:'}
                </p>
                {predefinedLabels.map(label => {
                  const isMultiMode = labels[currentSession.session_id]?.multiMode;
                  const currentLabels = labels[currentSession.session_id]?.labels || [];
                  const isSelected = isMultiMode 
                    ? currentLabels.includes(label)
                    : labels[currentSession.session_id]?.label === label && !labels[currentSession.session_id]?.excluded;

                  return (
                    <div key={label} className="flex items-center">
                      {isMultiMode ? (
                        <label className="flex items-center w-full px-4 py-3 rounded-lg border transition-colors cursor-pointer hover:border-gray-400">
                          <input
                            type="checkbox"
                            checked={isSelected}
                            onChange={() => handleMultiLabelToggle(currentSession.session_id, label)}
                            className="mr-3"
                          />
                          <span className={`flex-1 ${isSelected ? 'text-purple-700 font-medium' : ''}`}>
                            {label}
                          </span>
                          {isSelected && (
                            <CheckCircle className="w-5 h-5 text-purple-600 ml-2" />
                          )}
                        </label>
                      ) : (
                        <button
                          onClick={() => handleLabelChange(currentSession.session_id, label, false)}
                          className={`w-full text-left px-4 py-3 rounded-lg border transition-colors ${
                            isSelected
                              ? 'border-purple-500 bg-purple-50 text-purple-700'
                              : 'border-gray-300 hover:border-gray-400'
                          }`}
                        >
                          {label}
                        </button>
                      )}
                    </div>
                  );
                })}
              </div>
            )}

            {/* Custom Label */}
            <div className="mb-6">
              <p className="text-sm font-medium text-gray-700 mb-2">Add custom anomaly type:</p>
              <div className="flex space-x-2">
                <input
                  type="text"
                  value={customLabel}
                  onChange={(e) => setCustomLabel(e.target.value)}
                  onKeyPress={(e) => {
                    if (e.key === 'Enter') {
                      if (labels[currentSession.session_id]?.multiMode) {
                        handleMultiLabelToggle(currentSession.session_id, customLabel);
                      } else {
                        handleLabelChange(currentSession.session_id, customLabel, false);
                      }
                      handleAddCustomLabel();
                    }
                  }}
                  placeholder="Enter custom anomaly type"
                  className="flex-1 px-4 py-2 border rounded-lg"
                />
                <button
                  onClick={() => {
                    if (customLabel) {
                      if (labels[currentSession.session_id]?.multiMode) {
                        handleMultiLabelToggle(currentSession.session_id, customLabel);
                      } else {
                        handleLabelChange(currentSession.session_id, customLabel, false);
                      }
                      handleAddCustomLabel();
                    }
                  }}
                  disabled={!customLabel}
                  className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50"
                >
                  {labels[currentSession.session_id]?.multiMode ? 'Add to Selection' : 'Set Label'}
                </button>
              </div>
              <p className="text-xs text-gray-500 mt-1">
                {labels[currentSession.session_id]?.multiMode 
                  ? 'Will be added to the current selection of anomaly types'
                  : 'Will replace any existing label'
                }
              </p>
            </div>

            {/* Exclude Option */}
            <div className="border-t pt-4">
              <button
                onClick={() => {
                  setLabels(prev => ({
                    ...prev,
                    [currentSession.session_id]: {
                      label: 'not_anomaly',
                      labels: [],
                      excluded: true,
                      multiMode: false
                    }
                  }));
                }}
                className={`w-full px-4 py-3 rounded-lg border transition-colors flex items-center justify-center ${
                  labels[currentSession.session_id]?.excluded
                    ? 'border-red-500 bg-red-50 text-red-700'
                    : 'border-gray-300 hover:border-red-400'
                }`}
              >
                <XCircle className="w-5 h-5 mr-2" />
                Mark as NOT an Anomaly (Exclude)
              </button>
            </div>

            {/* Navigation */}
            <div className="flex justify-between mt-6">
              <button
                onClick={handlePrevious}
                disabled={currentIndex === 0}
                className="px-4 py-2 bg-gray-200 rounded-lg hover:bg-gray-300 disabled:opacity-50 flex items-center"
              >
                <ChevronLeft className="w-5 h-5 mr-1" />
                Previous
              </button>
              <span className="text-sm text-gray-600 py-2">
                {currentIndex + 1} / {sessions.length}
              </span>
              <button
                onClick={handleNext}
                disabled={currentIndex === sessions.length - 1}
                className="px-4 py-2 bg-gray-200 rounded-lg hover:bg-gray-300 disabled:opacity-50 flex items-center"
              >
                Next
                <ChevronRight className="w-5 h-5 ml-1" />
              </button>
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-white rounded-lg shadow-md p-12 text-center">
          <AlertTriangle className="w-16 h-16 text-yellow-500 mx-auto mb-4" />
          <p className="text-lg text-gray-600">No anomalies to review</p>
        </div>
      )}
    </div>
  );
};

export default ExpertLabelingInterface;
