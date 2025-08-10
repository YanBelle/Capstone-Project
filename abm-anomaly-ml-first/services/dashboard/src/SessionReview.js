import React, { useState, useEffect } from 'react';
import { Search, FileText, AlertTriangle, CheckCircle, Clock, Filter, ThumbsUp, ThumbsDown, Edit3, Save, X } from 'lucide-react';

const SessionReview = () => {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [selectedSession, setSelectedSession] = useState(null);
  const [detailedSession, setDetailedSession] = useState(null);
  const [sessionTextsLoading, setSessionTextsLoading] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [sessionsPerPage] = useState(20);
  const [feedbackSubmitting, setFeedbackSubmitting] = useState(false);
  const [feedbackStatus, setFeedbackStatus] = useState('');

  useEffect(() => {
    fetchSessions();
  }, []);

  const submitFeedback = async (sessionId, feedbackType, expertLabel, confidence = 1.0, explanation = '') => {
    setFeedbackSubmitting(true);
    setFeedbackStatus('');
    
    try {
      const response = await fetch('/api/v1/continuous-learning/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          expert_label: expertLabel,
          expert_confidence: confidence,
          feedback_type: feedbackType,
          expert_explanation: explanation
        })
      });

      if (response.ok) {
        setFeedbackStatus(`✅ Feedback submitted successfully for ${sessionId}`);
        // Refresh sessions to show updated status
        fetchSessions();
      } else {
        const errorData = await response.json();
        setFeedbackStatus(`❌ Error: ${errorData.detail || 'Failed to submit feedback'}`);
      }
    } catch (error) {
      console.error('Error submitting feedback:', error);
      setFeedbackStatus(`❌ Error: ${error.message}`);
    } finally {
      setFeedbackSubmitting(false);
      // Clear status after 5 seconds
      setTimeout(() => setFeedbackStatus(''), 5000);
    }
  };

  const confirmCorrect = (session) => {
    const feedbackType = 'confirmation';
    const expertLabel = session.is_anomaly ? (session.anomaly_type || 'anomaly') : 'normal';
    const explanation = session.is_anomaly 
      ? `Expert confirmed anomaly detection for ${session.anomaly_type}`
      : 'Expert confirmed normal transaction';
    
    submitFeedback(session.session_id, feedbackType, expertLabel, 1.0, explanation);
  };

  const markAsNormal = (session) => {
    const feedbackType = 'correction';
    const expertLabel = 'normal';
    const explanation = `Expert corrected false positive - this is actually a normal transaction`;
    
    submitFeedback(session.session_id, feedbackType, expertLabel, 1.0, explanation);
  };

  const markAsAnomaly = (session, anomalyType = 'unknown_anomaly') => {
    const feedbackType = 'correction';
    const expertLabel = anomalyType;
    const explanation = `Expert corrected false negative - this is actually an anomaly of type: ${anomalyType}`;
    
    submitFeedback(session.session_id, feedbackType, expertLabel, 1.0, explanation);
  };

  const fetchSessions = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/v1/sessions?limit=1000');
      if (response.ok) {
        const data = await response.json();
        setSessions(data.sessions || []);
      } else {
        console.error('Failed to fetch sessions');
        setSessions([]);
      }
    } catch (error) {
      console.error('Error fetching sessions:', error);
      setSessions([]);
    } finally {
      setLoading(false);
    }
  };

  // Fetch detailed session data including raw and cleaned text
  const fetchSessionDetails = async (sessionId) => {
    setSessionTextsLoading(true);
    try {
      const response = await fetch(`/api/v1/sessions/${sessionId}/texts`);
      if (response.ok) {
        const data = await response.json();
        setDetailedSession(data);
      } else {
        console.error('Failed to fetch session details');
        setDetailedSession(null);
      }
    } catch (error) {
      console.error('Error fetching session details:', error);
      setDetailedSession(null);
    } finally {
      setSessionTextsLoading(false);
    }
  };

  const filteredSessions = sessions.filter(session => {
    const matchesSearch = session.session_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         (session.anomaly_type && session.anomaly_type.toLowerCase().includes(searchTerm.toLowerCase()));
    
    const matchesFilter = filterType === 'all' || 
                         (filterType === 'anomalies' && session.is_anomaly) ||
                         (filterType === 'normal' && !session.is_anomaly);
    
    return matchesSearch && matchesFilter;
  });

  // Pagination
  const indexOfLastSession = currentPage * sessionsPerPage;
  const indexOfFirstSession = indexOfLastSession - sessionsPerPage;
  const currentSessions = filteredSessions.slice(indexOfFirstSession, indexOfLastSession);
  const totalPages = Math.ceil(filteredSessions.length / sessionsPerPage);

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'N/A';
    return new Date(timestamp).toLocaleString();
  };

  const getStatusIcon = (session) => {
    if (session.is_anomaly) {
      return <AlertTriangle className="w-5 h-5 text-red-500" />;
    }
    return <CheckCircle className="w-5 h-5 text-green-500" />;
  };

  const getStatusColor = (session) => {
    if (session.is_anomaly) {
      if (session.anomaly_score > 0.8) return 'bg-red-100 text-red-800';
      if (session.anomaly_score > 0.6) return 'bg-orange-100 text-orange-800';
      return 'bg-yellow-100 text-yellow-800';
    }
    return 'bg-green-100 text-green-800';
  };

  const SessionModal = ({ session, onClose }) => {
    const [feedbackForm, setFeedbackForm] = useState({
      feedbackType: 'confirmation',
      expertLabel: session?.is_anomaly ? (session.anomaly_type || 'anomaly') : 'normal',
      confidence: 1.0,
      explanation: ''
    });

    if (!session) return null;

    const handleCustomFeedback = () => {
      submitFeedback(
        session.session_id,
        feedbackForm.feedbackType,
        feedbackForm.expertLabel,
        feedbackForm.confidence,
        feedbackForm.explanation
      );
      onClose();
    };

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg max-w-4xl w-full mx-4 max-h-[90vh] overflow-y-auto">
          <div className="p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold">Session Details: {session.session_id}</h2>
              <button
                onClick={onClose}
                className="text-gray-500 hover:text-gray-700 text-2xl"
              >
                ×
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
              <div>
                <h3 className="font-semibold mb-2">Session Information</h3>
                <div className="space-y-2 text-sm">
                  <div><strong>Session ID:</strong> {session.session_id}</div>
                  <div><strong>Start Time:</strong> {formatTimestamp(session.start_time)}</div>
                  <div><strong>End Time:</strong> {formatTimestamp(session.end_time)}</div>
                  <div><strong>Status:</strong> 
                    <span className={`ml-2 px-2 py-1 rounded text-xs ${getStatusColor(session)}`}>
                      {session.is_anomaly ? 'Anomaly' : 'Normal'}
                    </span>
                  </div>
                </div>
              </div>

              {session.is_anomaly && (
                <div>
                  <h3 className="font-semibold mb-2">Anomaly Details</h3>
                  <div className="space-y-2 text-sm">
                    <div><strong>Type:</strong> {session.anomaly_type || 'Unknown'}</div>
                    <div><strong>Score:</strong> {session.anomaly_score?.toFixed(3) || 'N/A'}</div>
                    <div><strong>Patterns:</strong> {session.detected_patterns?.join(', ') || 'None'}</div>
                  </div>
                </div>
              )}
            </div>

            {/* Session Text Data */}
            <div className="space-y-4">
              {sessionTextsLoading ? (
                <div className="bg-gray-100 p-8 rounded text-center">
                  <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                  <p className="mt-2 text-gray-600">Loading session details...</p>
                </div>
              ) : (
                <>
                  {/* Raw EJ Text */}
                  <div>
                    <h3 className="font-semibold mb-2">Raw EJ Log</h3>
                    <div className="bg-gray-100 p-4 rounded max-h-96 overflow-y-auto">
                      <pre className="text-xs whitespace-pre-wrap font-mono">
                        {detailedSession?.raw_text || session.raw_text || 'No raw EJ data available'}
                      </pre>
                    </div>
                    {detailedSession?.text_lengths?.raw && (
                      <p className="text-xs text-gray-500 mt-1">
                        Raw text length: {detailedSession.text_lengths.raw.toLocaleString()} characters
                      </p>
                    )}
                  </div>

                  {/* Cleaned EJ Text */}
                  <div>
                    <h3 className="font-semibold mb-2">Cleaned EJ Text</h3>
                    <div className="bg-blue-50 p-4 rounded max-h-96 overflow-y-auto">
                      <pre className="text-xs whitespace-pre-wrap font-mono">
                        {detailedSession?.cleaned_text || 'Cleaned text not available'}
                      </pre>
                    </div>
                    {detailedSession?.text_lengths?.cleaned && (
                      <p className="text-xs text-gray-500 mt-1">
                        Cleaned text length: {detailedSession.text_lengths.cleaned.toLocaleString()} characters
                      </p>
                    )}
                  </div>

                  {/* Structured Events (if available) */}
                  {detailedSession?.structured_events && detailedSession.structured_events.length > 0 && (
                    <div>
                      <h3 className="font-semibold mb-2">Structured Events ({detailedSession.structured_events.length})</h3>
                      <div className="bg-green-50 p-4 rounded max-h-96 overflow-y-auto">
                        <pre className="text-xs whitespace-pre-wrap font-mono">
                          {JSON.stringify(detailedSession.structured_events, null, 2)}
                        </pre>
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>

            {/* Expert Feedback Section */}
            <div className="border-t pt-6">
              <h3 className="font-semibold mb-4">Expert Feedback</h3>
              
              {/* Quick Actions */}
              <div className="mb-4">
                <h4 className="font-medium mb-2">Quick Actions:</h4>
                <div className="flex space-x-3">
                  <button
                    onClick={() => { 
                      confirmCorrect(session); 
                      onClose(); 
                      setDetailedSession(null);
                    }}
                    disabled={feedbackSubmitting}
                    className="inline-flex items-center px-3 py-2 border border-green-300 text-sm leading-4 font-medium rounded text-green-700 bg-green-50 hover:bg-green-100 disabled:opacity-50"
                  >
                    <ThumbsUp className="w-4 h-4 mr-2" />
                    Confirm Correct
                  </button>
                  
                  {session.is_anomaly ? (
                    <button
                      onClick={() => { 
                        markAsNormal(session); 
                        onClose(); 
                        setDetailedSession(null);
                      }}
                      disabled={feedbackSubmitting}
                      className="inline-flex items-center px-3 py-2 border border-red-300 text-sm leading-4 font-medium rounded text-red-700 bg-red-50 hover:bg-red-100 disabled:opacity-50"
                    >
                      <X className="w-4 h-4 mr-2" />
                      Mark as Normal
                    </button>
                  ) : (
                    <button
                      onClick={() => { 
                        markAsAnomaly(session, 'expert_identified_anomaly'); 
                        onClose(); 
                        setDetailedSession(null);
                      }}
                      disabled={feedbackSubmitting}
                      className="inline-flex items-center px-3 py-2 border border-orange-300 text-sm leading-4 font-medium rounded text-orange-700 bg-orange-50 hover:bg-orange-100 disabled:opacity-50"
                    >
                      <ThumbsDown className="w-4 h-4 mr-2" />
                      Mark as Anomaly
                    </button>
                  )}
                </div>
              </div>

              {/* Custom Feedback Form */}
              <div className="border-t pt-4">
                <h4 className="font-medium mb-3">Custom Feedback:</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Feedback Type
                    </label>
                    <select
                      value={feedbackForm.feedbackType}
                      onChange={(e) => setFeedbackForm({...feedbackForm, feedbackType: e.target.value})}
                      className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-blue-500 focus:border-blue-500"
                    >
                      <option value="confirmation">Confirmation</option>
                      <option value="correction">Correction</option>
                      <option value="new_discovery">New Discovery</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Expert Label
                    </label>
                    <select
                      value={feedbackForm.expertLabel}
                      onChange={(e) => setFeedbackForm({...feedbackForm, expertLabel: e.target.value})}
                      className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-blue-500 focus:border-blue-500"
                    >
                      <option value="normal">Normal</option>
                      <option value="hardware_error">Hardware Error</option>
                      <option value="dispense_failure">Dispense Failure</option>
                      <option value="host_decline">Host Decline</option>
                      <option value="incomplete_transaction">Incomplete Transaction</option>
                      <option value="timeout_error">Timeout Error</option>
                      <option value="card_retained">Card Retained</option>
                      <option value="supervisor_activity">Supervisor Activity</option>
                      <option value="cash_handling_issue">Cash Handling Issue</option>
                      <option value="system_reset">System Reset</option>
                      <option value="unknown_anomaly">Unknown Anomaly</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Confidence
                    </label>
                    <input
                      type="range"
                      min="0.1"
                      max="1.0"
                      step="0.1"
                      value={feedbackForm.confidence}
                      onChange={(e) => setFeedbackForm({...feedbackForm, confidence: parseFloat(e.target.value)})}
                      className="w-full"
                    />
                    <div className="text-sm text-gray-500 mt-1">
                      {(feedbackForm.confidence * 100).toFixed(0)}%
                    </div>
                  </div>
                  
                  <div className="md:col-span-2">
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Explanation
                    </label>
                    <textarea
                      value={feedbackForm.explanation}
                      onChange={(e) => setFeedbackForm({...feedbackForm, explanation: e.target.value})}
                      placeholder="Explain your classification decision..."
                      rows={3}
                      className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>
                </div>
                
                <div className="mt-4">
                  <button
                    onClick={handleCustomFeedback}
                    disabled={feedbackSubmitting}
                    className="inline-flex items-center px-4 py-2 border border-blue-300 text-sm font-medium rounded text-blue-700 bg-blue-50 hover:bg-blue-100 disabled:opacity-50"
                  >
                    <Save className="w-4 h-4 mr-2" />
                    Submit Custom Feedback
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading sessions...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Feedback Status */}
      {feedbackStatus && (
        <div className={`p-4 rounded-lg ${feedbackStatus.includes('✅') ? 'bg-green-50 text-green-800' : 'bg-red-50 text-red-800'}`}>
          {feedbackStatus}
        </div>
      )}

      {/* Header */}
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-900">Session Review</h2>
        <button
          onClick={fetchSessions}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          Refresh
        </button>
      </div>

      {/* Filters */}
      <div className="bg-white p-4 rounded-lg shadow">
        <div className="flex flex-col md:flex-row gap-4">
          <div className="flex-1">
            <div className="relative">
              <Search className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search sessions by ID or anomaly type..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10 pr-4 py-2 w-full border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Filter className="h-4 w-4 text-gray-500" />
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              className="border border-gray-300 rounded-lg px-3 py-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="all">All Sessions</option>
              <option value="anomalies">Anomalies Only</option>
              <option value="normal">Normal Only</option>
            </select>
          </div>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white p-4 rounded-lg shadow">
          <div className="flex items-center">
            <FileText className="h-8 w-8 text-blue-500 mr-3" />
            <div>
              <p className="text-sm text-gray-600">Total Sessions</p>
              <p className="text-2xl font-bold">{sessions.length}</p>
            </div>
          </div>
        </div>
        <div className="bg-white p-4 rounded-lg shadow">
          <div className="flex items-center">
            <AlertTriangle className="h-8 w-8 text-red-500 mr-3" />
            <div>
              <p className="text-sm text-gray-600">Anomalies</p>
              <p className="text-2xl font-bold">{sessions.filter(s => s.is_anomaly).length}</p>
            </div>
          </div>
        </div>
        <div className="bg-white p-4 rounded-lg shadow">
          <div className="flex items-center">
            <CheckCircle className="h-8 w-8 text-green-500 mr-3" />
            <div>
              <p className="text-sm text-gray-600">Normal</p>
              <p className="text-2xl font-bold">{sessions.filter(s => !s.is_anomaly).length}</p>
            </div>
          </div>
        </div>
        <div className="bg-white p-4 rounded-lg shadow">
          <div className="flex items-center">
            <Clock className="h-8 w-8 text-purple-500 mr-3" />
            <div>
              <p className="text-sm text-gray-600">Filtered</p>
              <p className="text-2xl font-bold">{filteredSessions.length}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Sessions Table */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Session ID
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Start Time
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Type
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Score
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Feedback
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {currentSessions.map((session, index) => (
                <tr key={session.session_id || index} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    {getStatusIcon(session)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-gray-900">
                    {session.session_id}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {formatTimestamp(session.start_time)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {session.anomaly_type || 'Normal'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {session.is_anomaly ? (
                      <span className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${getStatusColor(session)}`}>
                        {session.anomaly_score?.toFixed(3) || 'N/A'}
                      </span>
                    ) : (
                      <span className="px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800">
                        Normal
                      </span>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    <button
                      onClick={() => {
                        setSelectedSession(session);
                        fetchSessionDetails(session.session_id);
                      }}
                      className="text-blue-600 hover:text-blue-900"
                    >
                      View Details
                    </button>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    <div className="flex space-x-2">
                      <button
                        onClick={() => confirmCorrect(session)}
                        disabled={feedbackSubmitting}
                        className="inline-flex items-center px-2 py-1 border border-green-300 text-xs leading-4 font-medium rounded text-green-700 bg-green-50 hover:bg-green-100 disabled:opacity-50"
                        title="Confirm this classification is correct"
                      >
                        <ThumbsUp className="w-3 h-3 mr-1" />
                        Correct
                      </button>
                      
                      {session.is_anomaly ? (
                        <button
                          onClick={() => markAsNormal(session)}
                          disabled={feedbackSubmitting}
                          className="inline-flex items-center px-2 py-1 border border-red-300 text-xs leading-4 font-medium rounded text-red-700 bg-red-50 hover:bg-red-100 disabled:opacity-50"
                          title="Mark as normal (false positive)"
                        >
                          <X className="w-3 h-3 mr-1" />
                          Normal
                        </button>
                      ) : (
                        <button
                          onClick={() => markAsAnomaly(session, 'expert_identified_anomaly')}
                          disabled={feedbackSubmitting}
                          className="inline-flex items-center px-2 py-1 border border-orange-300 text-xs leading-4 font-medium rounded text-orange-700 bg-orange-50 hover:bg-orange-100 disabled:opacity-50"
                          title="Mark as anomaly (false negative)"
                        >
                          <ThumbsDown className="w-3 h-3 mr-1" />
                          Anomaly
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="bg-white px-4 py-3 flex items-center justify-between border-t border-gray-200">
            <div className="flex-1 flex justify-between sm:hidden">
              <button
                onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                disabled={currentPage === 1}
                className="relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:bg-gray-100 disabled:text-gray-400"
              >
                Previous
              </button>
              <button
                onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                disabled={currentPage === totalPages}
                className="ml-3 relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:bg-gray-100 disabled:text-gray-400"
              >
                Next
              </button>
            </div>
            <div className="hidden sm:flex-1 sm:flex sm:items-center sm:justify-between">
              <div>
                <p className="text-sm text-gray-700">
                  Showing <span className="font-medium">{indexOfFirstSession + 1}</span> to{' '}
                  <span className="font-medium">{Math.min(indexOfLastSession, filteredSessions.length)}</span> of{' '}
                  <span className="font-medium">{filteredSessions.length}</span> results
                </p>
              </div>
              <div>
                <nav className="relative z-0 inline-flex rounded-md shadow-sm -space-x-px">
                  <button
                    onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                    disabled={currentPage === 1}
                    className="relative inline-flex items-center px-2 py-2 rounded-l-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:bg-gray-100 disabled:text-gray-400"
                  >
                    Previous
                  </button>
                  {[...Array(Math.min(totalPages, 5))].map((_, i) => {
                    const pageNumber = i + 1;
                    return (
                      <button
                        key={pageNumber}
                        onClick={() => setCurrentPage(pageNumber)}
                        className={`relative inline-flex items-center px-4 py-2 border text-sm font-medium ${
                          currentPage === pageNumber
                            ? 'z-10 bg-blue-50 border-blue-500 text-blue-600'
                            : 'bg-white border-gray-300 text-gray-500 hover:bg-gray-50'
                        }`}
                      >
                        {pageNumber}
                      </button>
                    );
                  })}
                  <button
                    onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                    disabled={currentPage === totalPages}
                    className="relative inline-flex items-center px-2 py-2 rounded-r-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:bg-gray-100 disabled:text-gray-400"
                  >
                    Next
                  </button>
                </nav>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Session Details Modal */}
      {selectedSession && (
        <SessionModal 
          session={selectedSession} 
          onClose={() => {
            setSelectedSession(null);
            setDetailedSession(null);
          }} 
        />
      )}
    </div>
  );
};

export default SessionReview;
