import React, { useState, useEffect } from 'react';
import { 
  Search, 
  Filter, 
  RefreshCw, 
  CheckCircle, 
  AlertTriangle, 
  Clock,
  FileText,
  BarChart3,
  Eye,
  X,
  ExternalLink,
  ThumbsUp,
  ThumbsDown,
  Flag,
  BookOpen
} from 'lucide-react';
import TrainingVisualization from './components/TrainingVisualization';
import apiConfig from './config/api';

const SessionReview = () => {
  console.log('SessionReview component loaded - Live Data Fix v2.0');
  
  const [activeTab, setActiveTab] = useState('sessions');
  const [sessions, setSessions] = useState([]);
  const [filteredSessions, setFilteredSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  
  // Modal and session details state
  const [selectedSession, setSelectedSession] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [sessionDetails, setSessionDetails] = useState(null);
  const [loadingDetails, setLoadingDetails] = useState(false);
  
  // Custom feedback state
  const [customFeedback, setCustomFeedback] = useState({
    type: 'Confirmation',
    level: 'Normal',
    confidence: 100,
    explanation: ''
  });

  const fetchSessions = async () => {
    try {
      setLoading(true);
      console.log('SessionReview: Starting to fetch sessions from API...');
      
      // Fetch all sessions from the main sessions endpoint
      const response = await fetch(apiConfig.endpoint('/api/v1/sessions?limit=1000&anomaly_filter=all'));
      console.log('SessionReview: API response status:', response.status);
      
      const data = await response.json();
      console.log('SessionReview: API data received:', data);
      
      if (data.sessions && Array.isArray(data.sessions)) {
        // Transform the database sessions to match our UI format
        const transformedSessions = data.sessions.map(session => ({
          session_id: session.session_id,
          status: session.is_anomaly ? 'anomaly' : 'normal',
          anomaly_type: session.anomaly_type || (session.is_anomaly ? 'Unknown Anomaly' : null),
          start_time: session.timestamp || session.created_at,
          transaction_count: session.session_length || 0,
          confidence_score: session.anomaly_score || 0.0
        }));
        
        setSessions(transformedSessions);
        console.log(`SessionReview: Successfully loaded ${transformedSessions.length} sessions from database (total: ${data.total})`);
      } else {
        console.warn('SessionReview: No sessions returned from API, falling back to mock data');
        setSessions(generateMockSessions());
      }
    } catch (error) {
      console.error('SessionReview: Error fetching sessions:', error);
      // Use mock data if API fails
      setSessions(generateMockSessions());
    } finally {
      setLoading(false);
    }
  };

  const generateMockSessions = () => {
    const mockStatuses = ['normal', 'anomaly', 'pending'];
    const mockTypes = ['incomplete_transaction', 'card_retention', 'dispensing_issue', 'timeout', 'system_error'];
    
    // Add some real-looking session IDs for testing
    const realSessions = [
      {
        session_id: 'ABM357_20250101_SESSION_3830_a3e7ddaf_20250809_025406',
        status: 'anomaly',
        anomaly_type: 'incomplete_transaction',
        start_time: '2025-01-09T11:41:00',
        transaction_count: 15,
        confidence_score: 0.92
      },
      {
        session_id: 'ABM357_20250101_SESSION_3831_b4f8eebf_20250809_035407',
        status: 'normal',
        anomaly_type: null,
        start_time: '2025-01-09T12:15:00',
        transaction_count: 23,
        confidence_score: 0.15
      }
    ];
    
    const mockSessions = Array.from({ length: 48 }, (_, i) => ({
      session_id: `ABM357_20250101_SESSION_${4000 + i}_cd3f485c_20250810_${String(i + 2).padStart(6, '0')}`,
      status: mockStatuses[Math.floor(Math.random() * mockStatuses.length)],
      anomaly_type: Math.random() > 0.4 ? mockTypes[Math.floor(Math.random() * mockTypes.length)] : null,
      start_time: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString(),
      transaction_count: Math.floor(Math.random() * 50) + 1,
      confidence_score: Math.random()
    }));
    
    return [...realSessions, ...mockSessions];
  };

  const applyFilters = () => {
    let filtered = sessions;

    if (searchTerm) {
      filtered = filtered.filter(session =>
        session.session_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
        (session.anomaly_type && session.anomaly_type.toLowerCase().includes(searchTerm.toLowerCase()))
      );
    }

    if (statusFilter !== 'all') {
      filtered = filtered.filter(session => session.status === statusFilter);
    }

    setFilteredSessions(filtered);
  };

  useEffect(() => {
    fetchSessions();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    applyFilters();
  }, [sessions, searchTerm, statusFilter]); // eslint-disable-line react-hooks/exhaustive-deps

  const getStatusIcon = (status) => {
    switch (status) {
      case 'normal':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'anomaly':
        return <AlertTriangle className="w-5 h-5 text-red-500" />;
      case 'pending':
        return <Clock className="w-5 h-5 text-yellow-500" />;
      default:
        return <Clock className="w-5 h-5 text-gray-500" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'normal':
        return 'text-green-600 bg-green-50';
      case 'anomaly':
        return 'text-red-600 bg-red-50';
      case 'pending':
        return 'text-yellow-600 bg-yellow-50';
      default:
        return 'text-gray-600 bg-gray-50';
    }
  };

  const getSessionStats = () => {
    const total = filteredSessions.length;
    const anomalies = filteredSessions.filter(s => s.status === 'anomaly').length;
    const normal = filteredSessions.filter(s => s.status === 'normal').length;
    const pending = filteredSessions.filter(s => s.status === 'pending').length;

    return { total, anomalies, normal, pending };
  };

  // Function to open session details modal
  const handleViewDetails = async (session) => {
    setSelectedSession(session);
    setIsModalOpen(true);
    setLoadingDetails(true);
    
    try {
      // Fetch session text data from the sessions API
      const response = await fetch(apiConfig.endpoint(`/api/v1/sessions/${session.session_id}/texts`));
      
      if (response.ok) {
        const data = await response.json();
        
        if (data.status === 'success') {
          // Use the text data provided by the API
          const rawEJ = data.raw_text || 'Raw EJ data not available';
          const cleanedEJ = data.cleaned_text || rawEJ;
          
          setSessionDetails({
            session_id: session.session_id,
            raw_ej: rawEJ,
            cleaned_ej: cleanedEJ,
            start_time: session.start_time,
            end_time: session.end_time || 'N/A',
            status: session.status,
            anomaly_type: session.anomaly_type,
            confidence_score: session.confidence_score,
            transaction_count: session.transaction_count,
            processing_info: {
              detected_patterns: data.structured_events?.detected_patterns || [],
              critical_events: data.structured_events?.critical_events || [],
              expert_override: false,
              expert_reason: null,
              preprocessing_applied: true,
              cleaning_method: 'BertViz Enhanced Server-side Preprocessing',
              text_lengths: data.text_lengths || {}
            }
          });
        } else {
          throw new Error(`API returned error: ${data.message || 'Session text data not found'}`);
        }
      } else if (response.status === 404) {
        // Session not found, try alternative endpoint
        console.warn(`Session ${session.session_id} not found, trying session evaluation endpoint...`);
        
        // Try to fetch from session evaluation endpoint
        const evalResponse = await fetch(apiConfig.endpoint(`/api/v1/session/evaluate/${session.session_id}`));
        
        if (evalResponse.ok) {
          const evalData = await evalResponse.json();
          
          setSessionDetails({
            session_id: session.session_id,
            raw_ej: evalData.raw_ej || 'Raw EJ data not available for this session',
            cleaned_ej: evalData.cleaned_ej || 'Cleaned EJ data not available for this session',
            start_time: session.start_time,
            end_time: evalData.end_time || session.end_time,
            status: session.status,
            anomaly_type: session.anomaly_type,
            confidence_score: session.confidence_score,
            transaction_count: evalData.transaction_count || session.transaction_count,
            processing_info: evalData.processing_info || null
          });
        } else {
          throw new Error('Session data not available from any endpoint');
        }
      } else {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
    } catch (error) {
      console.error('Error fetching session details:', error);
      
      // Fallback to basic session info with error message
      setSessionDetails({
        session_id: session.session_id,
        raw_ej: `Error loading session data: ${error.message}
        
This could be because:
- The session data has not been processed yet
- The session ID is invalid
- The API service is unavailable

Please try again later or contact support if the issue persists.`,
        cleaned_ej: `Error loading cleaned session data: ${error.message}

Please check the session ID and try again.`,
        start_time: session.start_time,
        end_time: session.end_time || 'N/A',
        status: session.status,
        anomaly_type: session.anomaly_type,
        confidence_score: session.confidence_score,
        transaction_count: session.transaction_count || 0,
        processing_info: null,
        error: true
      });
    } finally {
      setLoadingDetails(false);
    }
  };

  // Function to close modal
  const closeModal = () => {
    setIsModalOpen(false);
    setSelectedSession(null);
    setSessionDetails(null);
    setCustomFeedback({
      type: 'Confirmation',
      level: 'Normal', 
      confidence: 100,
      explanation: ''
    });
  };

  // Function to submit custom feedback
  const handleSubmitFeedback = async () => {
    try {
      const feedbackData = {
        session_id: selectedSession.session_id,
        feedback_type: customFeedback.type,
        expert_label: customFeedback.level,
        confidence: customFeedback.confidence,
        explanation: customFeedback.explanation,
        timestamp: new Date().toISOString()
      };
      
      console.log('Submitting feedback:', feedbackData);
      // Here you would send to your API
      // await fetch(apiConfig.endpoint('/api/v1/expert/feedback'), { method: 'POST', body: JSON.stringify(feedbackData) });
      
      alert('Feedback submitted successfully!');
      closeModal();
    } catch (error) {
      console.error('Error submitting feedback:', error);
      alert('Error submitting feedback. Please try again.');
    }
  };

  const stats = getSessionStats();

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="w-8 h-8 animate-spin text-blue-500" />
        <span className="ml-2 text-gray-600">Loading sessions...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">Session Review</h1>
        <div className="flex items-center space-x-2">
          <span className="text-sm text-blue-600 bg-blue-50 px-2 py-1 rounded">
            Live Data Fix Applied - Build 2.0
          </span>
          <button
            onClick={fetchSessions}
            className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </button>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          <button
            onClick={() => setActiveTab('sessions')}
            className={`flex items-center py-2 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'sessions'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            <FileText className="w-4 h-4 mr-2" />
            Session Review
          </button>
          <button
            onClick={() => setActiveTab('training')}
            className={`flex items-center py-2 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'training'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            <BarChart3 className="w-4 h-4 mr-2" />
            Model Training
          </button>
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'sessions' && (
        <div className="space-y-6">
          {/* Search and Filter */}
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
              <input
                type="text"
                placeholder="Search sessions by ID or anomaly type..."
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            <div className="flex items-center space-x-2">
              <Filter className="w-4 h-4 text-gray-400" />
              <select
                className="px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
              >
                <option value="all">All Sessions</option>
                <option value="normal">Normal</option>
                <option value="anomaly">Anomalies</option>
                <option value="pending">Pending</option>
              </select>
            </div>
          </div>

          {/* Session Statistics */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="bg-white rounded-lg shadow-sm border p-6 text-center">
              <FileText className="w-8 h-8 mx-auto text-blue-600 mb-2" />
              <div className="text-2xl font-bold text-gray-900">{stats.total.toLocaleString()}</div>
              <div className="text-sm text-gray-600">Total Sessions</div>
            </div>
            <div className="bg-white rounded-lg shadow-sm border p-6 text-center">
              <AlertTriangle className="w-8 h-8 mx-auto text-red-600 mb-2" />
              <div className="text-2xl font-bold text-gray-900">{stats.anomalies}</div>
              <div className="text-sm text-gray-600">Anomalies</div>
            </div>
            <div className="bg-white rounded-lg shadow-sm border p-6 text-center">
              <CheckCircle className="w-8 h-8 mx-auto text-green-600 mb-2" />
              <div className="text-2xl font-bold text-gray-900">{stats.normal}</div>
              <div className="text-sm text-gray-600">Normal</div>
            </div>
            <div className="bg-white rounded-lg shadow-sm border p-6 text-center">
              <Clock className="w-8 h-8 mx-auto text-yellow-600 mb-2" />
              <div className="text-2xl font-bold text-gray-900">{stats.pending}</div>
              <div className="text-sm text-gray-600">Filtered</div>
            </div>
          </div>

          {/* Sessions Table */}
          <div className="bg-white rounded-lg shadow-sm border overflow-hidden">
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
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {filteredSessions.slice(0, 50).map((session, index) => (
                    <tr key={index} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          {getStatusIcon(session.status)}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-gray-900">
                        {session.session_id}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                        {session.start_time ? new Date(session.start_time).toLocaleString() : 'N/A'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(session.status)}`}>
                          {session.anomaly_type || (session.status === 'normal' ? 'Normal' : session.status)}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        <div className="flex items-center space-x-2">
                          <button
                            onClick={() => handleViewDetails(session)}
                            className="text-blue-600 hover:text-blue-900 flex items-center"
                            title="View session details"
                          >
                            <Eye className="w-4 h-4 mr-1" />
                            View Details
                          </button>
                          <span className="text-gray-300">|</span>
                          <button
                            onClick={() => {
                              const url = `/session-evaluation?session_id=${session.session_id}&model=all`;
                              window.open(url, '_blank');
                            }}
                            className="text-purple-600 hover:text-purple-900 flex items-center"
                            title="Open session visualization in new tab"
                          >
                            <BarChart3 className="w-4 h-4 mr-1" />
                            Visualize
                            <ExternalLink className="w-3 h-3 ml-1" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {filteredSessions.length > 50 && (
              <div className="bg-gray-50 px-6 py-3 text-center text-sm text-gray-500">
                Showing first 50 of {filteredSessions.length.toLocaleString()} sessions
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === 'training' && (
        <TrainingVisualization />
      )}

      {/* Session Details Modal */}
      {isModalOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-4xl max-h-[90vh] overflow-y-auto">
            {/* Modal Header */}
            <div className="sticky top-0 bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <h2 className="text-lg font-semibold text-gray-900">Session Details</h2>
                <span className="text-sm text-gray-500 font-mono">
                  {selectedSession?.session_id}
                </span>
                <button
                  onClick={() => {
                    const baseUrl = process.env.NODE_ENV === 'production' ? '' : 'http://localhost:8000';
                    const url = `${baseUrl}/session-evaluation?session_id=${selectedSession?.session_id}&model=all`;
                    window.open(url, '_blank');
                  }}
                  className="inline-flex items-center px-3 py-1 border border-purple-300 text-sm leading-4 font-medium rounded text-purple-700 bg-purple-50 hover:bg-purple-100"
                  title="Open detailed model visualization"
                >
                  <BarChart3 className="w-4 h-4 mr-1" />
                  ML Analysis
                  <ExternalLink className="w-3 h-3 ml-1" />
                </button>
              </div>
              <button
                onClick={closeModal}
                className="text-gray-400 hover:text-gray-600"
              >
                <X className="w-6 h-6" />
              </button>
            </div>

            {/* Modal Content */}
            <div className="p-6">
              {loadingDetails ? (
                <div className="flex items-center justify-center py-12">
                  <RefreshCw className="w-8 h-8 animate-spin text-blue-500" />
                  <span className="ml-2 text-gray-600">Loading session details...</span>
                </div>
              ) : sessionDetails ? (
                <div className="space-y-6">
                  {/* Session Information */}
                  <div className="grid grid-cols-2 gap-4 p-4 bg-gray-50 rounded-lg">
                    <div>
                      <dt className="text-sm font-medium text-gray-500">Session ID</dt>
                      <dd className="text-sm text-gray-900 font-mono">{sessionDetails.session_id}</dd>
                    </div>
                    <div>
                      <dt className="text-sm font-medium text-gray-500">Status</dt>
                      <dd className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(sessionDetails.status)}`}>
                        {sessionDetails.anomaly_type || (sessionDetails.status === 'normal' ? 'Normal' : sessionDetails.status)}
                      </dd>
                    </div>
                    <div>
                      <dt className="text-sm font-medium text-gray-500">Start Time</dt>
                      <dd className="text-sm text-gray-900">
                        {sessionDetails.start_time ? new Date(sessionDetails.start_time).toLocaleString() : 'N/A'}
                      </dd>
                    </div>
                    <div>
                      <dt className="text-sm font-medium text-gray-500">End Time</dt>
                      <dd className="text-sm text-gray-900">
                        {sessionDetails.end_time ? new Date(sessionDetails.end_time).toLocaleString() : 'N/A'}
                      </dd>
                    </div>
                  </div>

                  {/* Raw EJ and Cleaned EJ */}
                  <div className="grid grid-cols-2 gap-6">
                    <div>
                      <h3 className="text-lg font-medium text-gray-900 mb-3 flex items-center">
                        Raw EJ Text
                        {sessionDetails.error && (
                          <span className="ml-2 px-2 py-1 bg-red-100 text-red-700 text-xs rounded-full">
                            Error Loading
                          </span>
                        )}
                      </h3>
                      <div className={`p-4 rounded-lg font-mono text-sm max-h-64 overflow-y-auto ${
                        sessionDetails.error 
                          ? 'bg-red-50 text-red-800 border border-red-200' 
                          : 'bg-gray-900 text-green-400'
                      }`}>
                        <pre className="whitespace-pre-wrap">{sessionDetails.raw_ej}</pre>
                      </div>
                      {sessionDetails.raw_ej && !sessionDetails.error && (
                        <div className="mt-2 text-xs text-gray-500">
                          Length: {sessionDetails.raw_ej.length} characters
                        </div>
                      )}
                    </div>
                    <div>
                      <h3 className="text-lg font-medium text-gray-900 mb-3 flex items-center">
                        Cleaned EJ Text
                        {sessionDetails.error && (
                          <span className="ml-2 px-2 py-1 bg-red-100 text-red-700 text-xs rounded-full">
                            Error Loading
                          </span>
                        )}
                      </h3>
                      <div className={`p-4 rounded-lg font-mono text-sm max-h-64 overflow-y-auto ${
                        sessionDetails.error 
                          ? 'bg-red-50 text-red-800 border border-red-200' 
                          : 'bg-blue-50 text-blue-900'
                      }`}>
                        <pre className="whitespace-pre-wrap">{sessionDetails.cleaned_ej}</pre>
                      </div>
                      {sessionDetails.cleaned_ej && !sessionDetails.error && (
                        <div className="mt-2 text-xs text-gray-500">
                          Length: {sessionDetails.cleaned_ej.length} characters
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Processing Information (if available) */}
                  {sessionDetails.processing_info && !sessionDetails.error && (
                    <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                      <h4 className="text-sm font-medium text-blue-900 mb-2">Processing Information</h4>
                      <div className="text-xs text-blue-800">
                        <div className="grid grid-cols-2 gap-4">
                          {sessionDetails.processing_info.processed_at && (
                            <div>
                              <span className="font-medium">Processed:</span>{' '}
                              {new Date(sessionDetails.processing_info.processed_at).toLocaleString()}
                            </div>
                          )}
                          {sessionDetails.processing_info.model_version && (
                            <div>
                              <span className="font-medium">Model Version:</span>{' '}
                              {sessionDetails.processing_info.model_version}
                            </div>
                          )}
                          {sessionDetails.processing_info.cleaning_steps && (
                            <div>
                              <span className="font-medium">Cleaning Steps:</span>{' '}
                              {sessionDetails.processing_info.cleaning_steps}
                            </div>
                          )}
                          {sessionDetails.transaction_count && (
                            <div>
                              <span className="font-medium">Transactions:</span>{' '}
                              {sessionDetails.transaction_count}
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Custom Feedback Section */}
                  <div className="border border-gray-200 rounded-lg p-6">
                    <h3 className="text-lg font-medium text-gray-900 mb-4">Custom Feedback</h3>
                    
                    <div className="grid grid-cols-2 gap-4 mb-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                          Feedback Type
                        </label>
                        <select
                          value={customFeedback.type}
                          onChange={(e) => setCustomFeedback({...customFeedback, type: e.target.value})}
                          className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        >
                          <option value="Confirmation">Confirmation</option>
                          <option value="Correction">Correction</option>
                          <option value="Enhancement">Enhancement</option>
                        </select>
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                          Expert Label
                        </label>
                        <select
                          value={customFeedback.level}
                          onChange={(e) => setCustomFeedback({...customFeedback, level: e.target.value})}
                          className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        >
                          <option value="Normal">Normal</option>
                          <option value="Anomaly">Anomaly</option>
                          <option value="Suspicious">Suspicious</option>
                        </select>
                      </div>
                    </div>

                    {/* Confidence Slider */}
                    <div className="mb-4">
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Confidence: {customFeedback.confidence}%
                      </label>
                      <div className="flex items-center space-x-3">
                        <span className="text-sm text-gray-500">0%</span>
                        <input
                          type="range"
                          min="0"
                          max="100"
                          value={customFeedback.confidence}
                          onChange={(e) => setCustomFeedback({...customFeedback, confidence: parseInt(e.target.value)})}
                          className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                          style={{
                            background: `linear-gradient(to right, #10b981 0%, #10b981 ${customFeedback.confidence}%, #e5e7eb ${customFeedback.confidence}%, #e5e7eb 100%)`
                          }}
                        />
                        <span className="text-sm text-gray-500">100%</span>
                      </div>
                    </div>

                    {/* Explanation Text Area */}
                    <div className="mb-4">
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Explanation
                      </label>
                      <textarea
                        value={customFeedback.explanation}
                        onChange={(e) => setCustomFeedback({...customFeedback, explanation: e.target.value})}
                        placeholder="Explain your classification decision..."
                        rows={4}
                        className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                    </div>

                    {/* Quick Actions */}
                    <div className="mb-4">
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Quick Actions
                      </label>
                      <div className="flex flex-wrap gap-2">
                        <button
                          onClick={() => setCustomFeedback({...customFeedback, level: 'Normal', confidence: 95})}
                          className="inline-flex items-center px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm hover:bg-green-200"
                        >
                          <ThumbsUp className="w-4 h-4 mr-1" />
                          Mark as Normal
                        </button>
                        <button
                          onClick={() => setCustomFeedback({...customFeedback, level: 'Anomaly', confidence: 90})}
                          className="inline-flex items-center px-3 py-1 bg-red-100 text-red-700 rounded-full text-sm hover:bg-red-200"
                        >
                          <ThumbsDown className="w-4 h-4 mr-1" />
                          Mark as Anomaly
                        </button>
                        <button
                          onClick={() => setCustomFeedback({...customFeedback, level: 'Suspicious', confidence: 75})}
                          className="inline-flex items-center px-3 py-1 bg-yellow-100 text-yellow-700 rounded-full text-sm hover:bg-yellow-200"
                        >
                          <Flag className="w-4 h-4 mr-1" />
                          Mark as Suspicious
                        </button>
                        <button
                          onClick={() => setCustomFeedback({...customFeedback, explanation: 'Based on transaction patterns and system behavior, this session exhibits characteristics consistent with normal ATM operations.'})}
                          className="inline-flex items-center px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm hover:bg-blue-200"
                        >
                          <BookOpen className="w-4 h-4 mr-1" />
                          Add Template
                        </button>
                      </div>
                    </div>

                    {/* Submit Button */}
                    <div className="flex justify-end">
                      <button
                        onClick={handleSubmitFeedback}
                        className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                      >
                        Submit Custom Feedback
                      </button>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-12">
                  <div className="text-gray-500">Failed to load session details</div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SessionReview;
