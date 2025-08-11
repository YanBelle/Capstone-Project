import React, { useState, useEffect } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { 
  ArrowLeft, 
  FileText, 
  Brain, 
  Target, 
  AlertTriangle,
  RefreshCw,
  Eye
} from 'lucide-react';

const SessionEvaluation = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const sessionId = searchParams.get('session_id') || searchParams.get('id');
  
  const [sessionData, setSessionData] = useState(null);
  const [evaluationData, setEvaluationData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('session-data');
  const [selectedModel, setSelectedModel] = useState('All Models');

  const fetchSessionData = async () => {
    try {
      const response = await fetch(`/api/v1/sessions/${sessionId}`);
      const data = await response.json();
      setSessionData(data);
    } catch (error) {
      console.error('Error fetching session data:', error);
      // Mock data for development
      setSessionData({
        session_id: sessionId,
        status: 'anomaly',
        anomaly_type: 'Dispense Failure',
        start_time: new Date().toISOString(),
        raw_text: 'No raw text available for this session.',
        cleaned_text: '',
        confidence_score: 0.85
      });
    }
  };

  const fetchEvaluationData = async () => {
    try {
      const response = await fetch(`/api/v1/evaluation/${sessionId}`);
      const data = await response.json();
      setEvaluationData(data);
    } catch (error) {
      console.error('Error fetching evaluation data:', error);
      // Mock data for development
      setEvaluationData({
        bert_analysis: {
          token_importance: [],
          detected_patterns: [],
          attention_analysis: [],
          attention_heatmap: []
        }
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (sessionId) {
      fetchSessionData();
      fetchEvaluationData();
    }
  }, [sessionId]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleEvaluateSession = async () => {
    try {
      setLoading(true);
      const response = await fetch(`/api/v1/evaluate-session`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          model: selectedModel
        })
      });
      
      if (response.ok) {
        await fetchEvaluationData();
      }
    } catch (error) {
      console.error('Error evaluating session:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyzeAttention = async () => {
    try {
      setLoading(true);
      const response = await fetch(`/api/v1/bert/analyze-attention`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId
        })
      });
      
      if (response.ok) {
        await fetchEvaluationData();
      }
    } catch (error) {
      console.error('Error analyzing attention:', error);
    } finally {
      setLoading(false);
    }
  };

  if (!sessionId) {
    return (
      <div className="text-center py-12">
        <AlertTriangle className="w-16 h-16 mx-auto text-yellow-500 mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">No Session Selected</h3>
        <p className="text-gray-500 mb-4">Please select a session from the Session Review page.</p>
        <button
          onClick={() => navigate('/dashboard/session-review')}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
        >
          Go to Session Review
        </button>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="w-8 h-8 animate-spin text-blue-500" />
        <span className="ml-2 text-gray-600">Loading session evaluation...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-indigo-600 rounded-lg p-6 text-white">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center mb-2">
              <button
                onClick={() => navigate('/dashboard/session-review')}
                className="mr-3 p-1 hover:bg-white/20 rounded"
              >
                <ArrowLeft className="w-5 h-5" />
              </button>
              <Brain className="w-6 h-6 mr-2" />
              <h1 className="text-xl font-bold">EJ Session Model Evaluation</h1>
            </div>
            <p className="text-purple-200">Analyze individual EJ sessions across ensemble models with detailed visualizations</p>
          </div>
        </div>
      </div>

      {/* Session ID and Model Selection */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Session ID:</label>
            <input
              type="text"
              value={sessionId}
              readOnly
              className="w-full px-3 py-2 border border-gray-300 rounded-md bg-gray-50 font-mono text-sm"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Model:</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md"
            >
              <option>All Models</option>
              <option>BERT Model</option>
              <option>Random Forest</option>
              <option>Isolation Forest</option>
            </select>
          </div>
        </div>
        <div className="mt-4 flex space-x-3">
          <button
            onClick={handleEvaluateSession}
            className="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 flex items-center"
            disabled={loading}
          >
            <Target className="w-4 h-4 mr-2" />
            Evaluate Session
          </button>
        </div>
      </div>

      {/* Session Data Card */}
      <div className="bg-purple-600 rounded-lg text-white p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <FileText className="w-5 h-5 mr-2" />
            <span className="font-medium">EJ Session Data</span>
          </div>
          <span className="text-purple-200 font-mono text-sm">{sessionId}</span>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          <button
            onClick={() => setActiveTab('session-data')}
            className={`py-2 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'session-data'
                ? 'border-purple-500 text-purple-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Raw EJ Text
          </button>
          <button
            onClick={() => setActiveTab('cleaned-text')}
            className={`py-2 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'cleaned-text'
                ? 'border-purple-500 text-purple-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Cleaned EJ Text
          </button>
        </nav>
      </div>

      {/* Tab Content */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        {activeTab === 'session-data' && (
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Raw EJ Text</h3>
            <div className="bg-gray-50 rounded-md p-4 font-mono text-sm">
              {sessionData?.raw_text || 'No raw text available for this session.'}
            </div>
          </div>
        )}

        {activeTab === 'cleaned-text' && (
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Cleaned EJ Text</h3>
            <div className="bg-gray-50 rounded-md p-4 font-mono text-sm">
              {sessionData?.cleaned_text || 'No cleaned text available for this session.'}
            </div>
          </div>
        )}
      </div>

      {/* BERT Attention Analysis */}
      <div className="bg-white rounded-lg shadow-sm border">
        <div className="border-b border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <Brain className="w-5 h-5 mr-2 text-purple-600" />
              <h3 className="text-lg font-semibold text-gray-900">BERT Attention Analysis</h3>
            </div>
            <button
              onClick={handleAnalyzeAttention}
              className="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 flex items-center text-sm"
              disabled={loading}
            >
              <Eye className="w-4 h-4 mr-2" />
              Analyze Attention
            </button>
          </div>
        </div>

        {/* Sub-tabs for BERT Analysis */}
        <div className="border-b border-gray-200 px-4">
          <nav className="-mb-px flex space-x-8">
            {[
              { id: 'token-importance', name: 'Token Importance' },
              { id: 'detected-patterns', name: 'Detected Patterns' },
              { id: 'attention-analysis', name: 'Attention Analysis' },
              { id: 'attention-heatmap', name: 'Attention Heatmap' }
            ].map((tab) => (
              <button
                key={tab.id}
                className="py-2 px-1 border-b-2 border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 font-medium text-sm"
              >
                {tab.name}
              </button>
            ))}
          </nav>
        </div>

        <div className="p-6">
          <div className="text-center py-8 text-gray-500">
            <Brain className="w-12 h-12 mx-auto text-gray-400 mb-2" />
            <p>Click "Analyze Attention" to generate BERT token importance analysis</p>
          </div>
        </div>
      </div>

      {/* Direct Link Warning */}
      <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4">
        <div className="flex">
          <AlertTriangle className="w-5 h-5 text-yellow-400 mr-2" />
          <div className="text-sm text-yellow-700">
            <strong>Direct Link:</strong> 
            <span className="ml-2 font-mono text-xs">
              http://localhost:8000/session-evaluation?session_id={sessionId}&model=all
            </span>
          </div>
        </div>
      </div>

      {/* Error Message */}
      <div className="bg-red-50 border border-red-200 rounded-md p-4">
        <div className="flex">
          <AlertTriangle className="w-5 h-5 text-red-400 mr-2" />
          <div className="text-sm text-red-700">
            <strong>Error:</strong> Evaluation failed
          </div>
        </div>
      </div>
    </div>
  );
};

export default SessionEvaluation;
