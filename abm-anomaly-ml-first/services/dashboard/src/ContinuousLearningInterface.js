import React, { useState, useEffect } from 'react';
import { 
  Brain, 
  RefreshCw, 
  CheckCircle, 
  XCircle, 
  AlertTriangle, 
  TrendingUp, 
  Database, 
  Clock,
  Users,
  Target,
  Award,
  Activity
} from 'lucide-react';
import apiConfig from './config/api';

const ContinuousLearningInterface = () => {
  const [learningStatus, setLearningStatus] = useState(null);
  const [feedbackSessions, setFeedbackSessions] = useState([]);
  const [selectedSession, setSelectedSession] = useState(null);
  const [loading, setLoading] = useState(true);
  const [retraining, setRetraining] = useState(false);
  const [feedbackFilter, setFeedbackFilter] = useState('recent_anomalies');
  const [submittingFeedback, setSubmittingFeedback] = useState(false);
  const [feedbackForm, setFeedbackForm] = useState({
    expert_label: '',
    expert_confidence: 0.8,
    feedback_type: 'confirmation',
    expert_explanation: ''
  });

  // Fetch continuous learning status
  const fetchLearningStatus = async () => {
    try {
      const response = await fetch(apiConfig.endpoint("/api/v1/continuous-learning/status"));
      if (response.ok) {
        const data = await response.json();
        setLearningStatus(data.learning_status);
      }
    } catch (error) {
      console.error('Error fetching learning status:', error);
    }
  };

  // Fetch sessions for feedback
  const fetchFeedbackSessions = async () => {
    try {
      setLoading(true);
      const response = await fetch(apiConfig.endpoint("/api/v1/continuous-learning/feedback-sessions?filter_type=${feedbackFilter}&limit=50"));
      if (response.ok) {
        const data = await response.json();
        setFeedbackSessions(data.sessions);
      }
    } catch (error) {
      console.error('Error fetching feedback sessions:', error);
    } finally {
      setLoading(false);
    }
  };

  // Fetch detailed session information
  const fetchSessionDetails = async (sessionId) => {
    try {
      const response = await fetch(apiConfig.endpoint("/api/v1/continuous-learning/session-details/${sessionId}"));
      if (response.ok) {
        const data = await response.json();
        setSelectedSession(data.session);
        
        // Pre-fill form if existing feedback
        if (data.session.existing_feedback) {
          const feedback = data.session.existing_feedback;
          setFeedbackForm({
            expert_label: feedback.expert_label || '',
            expert_confidence: feedback.expert_confidence || 0.8,
            feedback_type: feedback.feedback_type || 'confirmation',
            expert_explanation: feedback.expert_explanation || ''
          });
        } else {
          // Reset form for new feedback
          setFeedbackForm({
            expert_label: data.session.is_anomaly ? (data.session.anomaly_type || 'anomaly') : 'normal',
            expert_confidence: 0.8,
            feedback_type: 'confirmation',
            expert_explanation: ''
          });
        }
      }
    } catch (error) {
      console.error('Error fetching session details:', error);
    }
  };

  // Submit expert feedback
  const submitFeedback = async () => {
    if (!selectedSession) return;
    
    try {
      setSubmittingFeedback(true);
      
      const response = await fetch(apiConfig.endpoint("/api/v1/continuous-learning/feedback"), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: selectedSession.session_id,
          expert_label: feedbackForm.expert_label,
          expert_confidence: feedbackForm.expert_confidence,
          feedback_type: feedbackForm.feedback_type,
          expert_explanation: feedbackForm.expert_explanation
        })
      });

      if (response.ok) {
        alert('Feedback submitted successfully!');
        fetchLearningStatus(); // Refresh status
        fetchFeedbackSessions(); // Refresh session list
        setSelectedSession(null); // Close details panel
      } else {
        throw new Error('Failed to submit feedback');
      }
    } catch (error) {
      console.error('Error submitting feedback:', error);
      alert('Failed to submit feedback. Please try again.');
    } finally {
      setSubmittingFeedback(false);
    }
  };

  // Trigger manual retraining
  const triggerRetraining = async () => {
    try {
      setRetraining(true);
      
      const response = await fetch(apiConfig.endpoint("/api/v1/continuous-learning/trigger-retraining"), {
        method: 'POST'
      });

      if (response.ok) {
        alert('Continuous retraining triggered successfully! This may take a few minutes.');
        setTimeout(() => {
          fetchLearningStatus(); // Refresh status after delay
        }, 5000);
      } else {
        throw new Error('Failed to trigger retraining');
      }
    } catch (error) {
      console.error('Error triggering retraining:', error);
      alert('Failed to trigger retraining. Please try again.');
    } finally {
      setRetraining(false);
    }
  };

  useEffect(() => {
    fetchLearningStatus();
    fetchFeedbackSessions();
    
    // Refresh status every 30 seconds
    const interval = setInterval(fetchLearningStatus, 30000);
    return () => clearInterval(interval);
  }, [feedbackFilter]);

  const StatusCard = ({ title, value, icon: Icon, color, subtitle }) => (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-600">{title}</p>
          <p className="text-2xl font-bold mt-1">{value}</p>
          {subtitle && <p className="text-xs text-gray-500 mt-1">{subtitle}</p>}
        </div>
        <div className={`p-3 rounded-full ${color}`}>
          <Icon className="w-6 h-6 text-white" />
        </div>
      </div>
    </div>
  );

  const FeedbackTypeSelect = ({ value, onChange }) => (
    <select 
      value={value} 
      onChange={(e) => onChange(e.target.value)}
      className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
    >
      <option value="confirmation">Confirmation</option>
      <option value="correction">Correction</option>
      <option value="new_discovery">New Discovery</option>
    </select>
  );

  if (loading && !learningStatus) {
    return (
      <div className="flex justify-center items-center h-64">
        <RefreshCw className="w-8 h-8 animate-spin text-blue-500" />
        <span className="ml-2 text-gray-600">Loading continuous learning interface...</span>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Continuous Learning</h1>
          <p className="text-gray-600 mt-1">Improve ML models through expert feedback</p>
        </div>
        <button
          onClick={triggerRetraining}
          disabled={retraining || (learningStatus && learningStatus.feedback_buffer_size < 5)}
          className={`px-6 py-2 rounded-lg font-medium flex items-center space-x-2 ${
            retraining || (learningStatus && learningStatus.feedback_buffer_size < 5)
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-blue-600 text-white hover:bg-blue-700'
          }")}
        >
          {retraining ? (
            <RefreshCw className="w-4 h-4 animate-spin" />
          ) : (
            <Brain className="w-4 h-4" />
          )}
          <span>{retraining ? 'Retraining...' : 'Trigger Retraining'}</span>
        </button>
      </div>

      {/* Status Cards */}
      {learningStatus && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <StatusCard
            title="Feedback Buffer"
            value={learningStatus.feedback_buffer_size}
            icon={Database}
            color="bg-blue-500"
            subtitle={")Threshold: ${learningStatus.learning_threshold}`}
          />
          <StatusCard
            title="Retraining Cycles"
            value={learningStatus.retraining_cycles}
            icon={RefreshCw}
            color="bg-green-500"
            subtitle="Completed cycles"
          />
          <StatusCard
            title="Total Feedback"
            value={learningStatus.total_feedback_processed}
            icon={Users}
            color="bg-purple-500"
            subtitle="All-time processed"
          />
          <StatusCard
            title="Last Performance"
            value={`${(learningStatus.last_performance_improvement * 100).toFixed(1)}%`}
            icon={TrendingUp}
            color="bg-orange-500"
            subtitle="Improvement"
          />
        </div>
      )}

      {/* Feedback Summary */}
      {learningStatus && learningStatus.feedback_types_summary && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Current Feedback Buffer</h3>
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {learningStatus.feedback_types_summary.confirmations}
              </div>
              <div className="text-sm text-gray-600">Confirmations</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">
                {learningStatus.feedback_types_summary.corrections}
              </div>
              <div className="text-sm text-gray-600">Corrections</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {learningStatus.feedback_types_summary.new_discoveries}
              </div>
              <div className="text-sm text-gray-600">New Discoveries</div>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Sessions List */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold">Sessions for Feedback</h3>
            <select
              value={feedbackFilter}
              onChange={(e) => setFeedbackFilter(e.target.value)}
              className="p-2 border border-gray-300 rounded-md"
            >
              <option value="recent_anomalies">Recent Anomalies</option>
              <option value="high_confidence_anomalies">High Confidence</option>
              <option value="overridden_sessions">Expert Overridden</option>
            </select>
          </div>

          <div className="space-y-3 max-h-96 overflow-y-auto">
            {feedbackSessions.map((session) => (
              <div
                key={session.session_id}
                onClick={() => fetchSessionDetails(session.session_id)}
                className="p-3 border border-gray-200 rounded-lg cursor-pointer hover:bg-gray-50 transition-colors"
              >
                <div className="flex justify-between items-start">
                  <div>
                    <p className="font-medium text-sm">{session.session_id}</p>
                    <p className="text-xs text-gray-600">
                      {session.anomaly_type} • Score: {session.anomaly_score.toFixed(3)}
                    </p>
                    <p className="text-xs text-gray-500">
                      {new Date(session.start_time).toLocaleString()}
                    </p>
                  </div>
                  <div className="flex items-center space-x-1">
                    {session.expert_override_applied && (
                      <AlertTriangle className="w-4 h-4 text-orange-500" />
                    )}
                    {session.anomaly_score > 0.8 ? (
                      <AlertTriangle className="w-4 h-4 text-red-500" />
                    ) : (
                      <CheckCircle className="w-4 h-4 text-green-500" />
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Session Details & Feedback Form */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Expert Feedback</h3>
          
          {selectedSession ? (
            <div className="space-y-4">
              {/* Session Info */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="font-medium mb-2">Session: {selectedSession.session_id}</h4>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <span>Type: {selectedSession.anomaly_type}</span>
                  <span>Score: {selectedSession.anomaly_score.toFixed(3)}</span>
                  <span>ML Prediction: {selectedSession.is_anomaly ? 'Anomaly' : 'Normal'}</span>
                  <span>Override: {selectedSession.expert_override_applied ? 'Yes' : 'No'}</span>
                </div>
                
                {selectedSession.detected_patterns.length > 0 && (
                  <div className="mt-2">
                    <span className="text-sm font-medium">Patterns:</span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {selectedSession.detected_patterns.map((pattern, idx) => (
                        <span key={idx} className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded">
                          {pattern}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Raw Text Preview */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Transaction Log (Preview)
                </label>
                <textarea
                  value={selectedSession.raw_text}
                  readOnly
                  className="w-full p-2 border border-gray-300 rounded-md bg-gray-50 text-xs"
                  rows={6}
                />
              </div>

              {/* Feedback Form */}
              <div className="space-y-3">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Expert Label
                  </label>
                  <input
                    type="text"
                    value={feedbackForm.expert_label}
                    onChange={(e) => setFeedbackForm({...feedbackForm, expert_label: e.target.value})}
                    placeholder="e.g., normal, dispense_failure, hardware_error"
                    className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Confidence: {feedbackForm.expert_confidence}
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="1.0"
                    step="0.1"
                    value={feedbackForm.expert_confidence}
                    onChange={(e) => setFeedbackForm({...feedbackForm, expert_confidence: parseFloat(e.target.value)})}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Feedback Type
                  </label>
                  <FeedbackTypeSelect
                    value={feedbackForm.feedback_type}
                    onChange={(type) => setFeedbackForm({...feedbackForm, feedback_type: type})}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Explanation
                  </label>
                  <textarea
                    value={feedbackForm.expert_explanation}
                    onChange={(e) => setFeedbackForm({...feedbackForm, expert_explanation: e.target.value})}
                    placeholder="Explain your decision..."
                    className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    rows={3}
                  />
                </div>

                <div className="flex space-x-2">
                  <button
                    onClick={submitFeedback}
                    disabled={submittingFeedback}
                    className="flex-1 bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 disabled:opacity-50 flex items-center justify-center space-x-2"
                  >
                    {submittingFeedback ? (
                      <RefreshCw className="w-4 h-4 animate-spin" />
                    ) : (
                      <CheckCircle className="w-4 h-4" />
                    )}
                    <span>{submittingFeedback ? 'Submitting...' : 'Submit Feedback'}</span>
                  </button>
                  <button
                    onClick={() => setSelectedSession(null)}
                    className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50"
                  >
                    Cancel
                  </button>
                </div>
              </div>

              {/* Existing Feedback */}
              {selectedSession.existing_feedback && (
                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
                  <h5 className="font-medium text-yellow-800 mb-2">Previous Feedback</h5>
                  <div className="text-sm text-yellow-700">
                    <p>Label: {selectedSession.existing_feedback.expert_label}</p>
                    <p>Confidence: {selectedSession.existing_feedback.expert_confidence}</p>
                    <p>Type: {selectedSession.existing_feedback.feedback_type}</p>
                    {selectedSession.existing_feedback.expert_explanation && (
                      <p>Explanation: {selectedSession.existing_feedback.expert_explanation}</p>
                    )}
                    <p className="text-xs mt-1">
                      Submitted: {new Date(selectedSession.existing_feedback.created_at).toLocaleString()}
                    </p>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              <Brain className="w-12 h-12 mx-auto mb-2 opacity-50" />
              <p>Select a session to provide expert feedback</p>
              <p className="text-sm">Your feedback helps improve the ML models</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ContinuousLearningInterface;
