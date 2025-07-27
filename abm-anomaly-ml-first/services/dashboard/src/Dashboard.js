import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { AlertCircle, Activity, TrendingUp, Clock, Shield, Database, Brain, AlertTriangle } from 'lucide-react';
import { useLocation } from 'react-router-dom';
import Layout from './Layout';
import apiConfig from './config/api';
import ExpertLabelingInterface from './ExpertLabelingInterface';
import ContinuousLearningInterface from './ContinuousLearningInterface';
import MultiAnomalyView from './MultiAnomalyView';
import RealtimeMonitoringInterface from './RealtimeMonitoringInterface';
import SVMDebugDashboard from './SVMDebugDashboard';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const ATMDashboard = () => {
  const location = useLocation();
  const [activeTab, setActiveTab] = useState('overview');
  const [stats, setStats] = useState({
    total_transactions: 0,
    total_anomalies: 0,
    anomaly_rate: 0,
    high_risk_count: 0,
    recent_alerts: [],
    hourly_trend: [],
    problematic_terminals: []
  });
  
  const [anomalies, setAnomalies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [realTimeAlerts, setRealTimeAlerts] = useState([]);

  // Get current tab from URL or state
  const getCurrentTab = () => {
    if (activeTab && activeTab !== 'overview') return activeTab;
    
    const path = location.pathname;
    if (path === '/dashboard' || path === '/dashboard/') return 'overview';
    if (path.includes('/dashboard/anomalies')) return 'anomalies';
    if (path.includes('/dashboard/alerts')) return 'alerts';
    if (path.includes('/dashboard/analytics')) return 'analytics';
    return 'overview';
  };

  // Handle file upload
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      setLoading(true);
      console.log('Uploading file to:', apiConfig.endpoint('/api/v1/upload'));
      const response = await fetch(apiConfig.endpoint('/api/v1/upload'), {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      console.log('File upload result:', result);
      
      // Refresh data after upload
      await fetchStats();
      await fetchAnomalies();
      alert('File uploaded successfully!');
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('Error uploading file: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  // Fetch dashboard stats
  const fetchStats = async () => {
    try {
      console.log('Fetching dashboard stats from:', apiConfig.endpoint('/api/v1/dashboard/stats'));
      const response = await fetch(apiConfig.endpoint('/api/v1/dashboard/stats'));
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log('Dashboard stats received:', data);
      setStats(data);
    } catch (error) {
      console.error('Error fetching stats:', error);
      setStats({
        total_transactions: 0,
        total_anomalies: 0,
        anomaly_rate: 0,
        high_risk_count: 0,
        recent_alerts: [],
        hourly_trend: [],
        problematic_terminals: []
      });
    }
  };

  // Fetch anomalies
  const fetchAnomalies = async () => {
    try {
      console.log('Fetching anomalies from:', apiConfig.endpoint('/api/v1/anomalies?limit=50'));
      const response = await fetch(apiConfig.endpoint('/api/v1/anomalies?limit=50'));
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log('Anomalies received:', data);
      setAnomalies(data.anomalies || []);
    } catch (error) {
      console.error('Error fetching anomalies:', error);
      setAnomalies([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
    fetchAnomalies();
    
    // Refresh every 30 seconds
    const interval = setInterval(() => {
      fetchStats();
      fetchAnomalies();
    }, 30000);

    return () => {
      clearInterval(interval);
    };
  }, []);

  const anomalyRatePercent = (stats.anomaly_rate * 100).toFixed(2);

  const pieData = [
    { name: 'Normal', value: stats.total_transactions - stats.total_anomalies, fill: '#10b981' },
    { name: 'Anomalies', value: stats.total_anomalies, fill: '#ef4444' }
  ];

  const StatCard = ({ title, value, icon: Icon, color, subtitle }) => (
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

  const AlertItem = ({ alert }) => (
    <div className={`p-4 rounded-lg border-l-4 ${
      alert.level === 'HIGH' ? 'border-red-500 bg-red-50' : 'border-yellow-500 bg-yellow-50'
    }`}>
      <div className="flex items-start justify-between">
        <div>
          <p className="font-semibold text-sm">
            {alert.level} Risk - {alert.details?.anomaly_type || 'Anomaly Detected'}
          </p>
          <p className="text-sm text-gray-600 mt-1">
            Session: {alert.details?.session_id || 'Unknown'}
          </p>
          <p className="text-xs text-gray-500 mt-1">
            Score: {alert.details?.anomaly_score?.toFixed(3) || 'N/A'}
          </p>
        </div>
        <p className="text-xs text-gray-500">
          {new Date(alert.timestamp).toLocaleTimeString()}
        </p>
      </div>
    </div>
  );

  if (loading && getCurrentTab() === 'overview') {
    return (
      <Layout>
        <div className="min-h-screen bg-gray-100 flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
            <p className="mt-4 text-gray-600">Loading dashboard...</p>
          </div>
        </div>
      </Layout>
    );
  }

  // Check if we're using Layout or direct rendering based on tabs
  const shouldUseLayout = location.pathname.includes('/dashboard');

  const DashboardContent = () => (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <div className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <Brain className="w-8 h-8 text-purple-600 mr-3" />
              <h1 className="text-2xl font-bold text-gray-900">ML-First ABM Anomaly Detection</h1>
            </div>
            <div className="flex items-center space-x-4">
              <input
                type="file"
                id="file-upload"
                className="hidden"
                accept=".txt,.log"
                onChange={handleFileUpload}
              />
              <label
                htmlFor="file-upload"
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 cursor-pointer"
              >
                Upload EJournal
              </label>
              <div className="flex items-center text-sm text-gray-500">
                <Clock className="w-4 h-4 mr-1" />
                Last updated: {new Date().toLocaleTimeString()}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {['overview', 'anomalies', 'multi-anomaly', 'alerts', 'expert-labeling', 'continuous-learning', 'monitoring', 'analytics', 'svm-debug'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`py-3 px-1 border-b-2 font-medium text-sm capitalize ${
                  activeTab === tab
                    ? 'border-purple-600 text-purple-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                {tab === 'expert-labeling' ? 'Expert Review' : 
                 tab === 'continuous-learning' ? 'ML Training' : 
                 tab === 'multi-anomaly' ? 'Multi-Anomaly' : 
                 tab === 'monitoring' ? 'Real-time Monitor' : 
                 tab === 'svm-debug' ? 'SVM Debug' : tab}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <StatCard
                title="Total Sessions"
                value={stats.total_transactions.toLocaleString()}
                icon={Activity}
                color="bg-blue-600"
                subtitle="Processed today"
              />
              <StatCard
                title="Anomalies Detected"
                value={stats.total_anomalies.toLocaleString()}
                icon={AlertCircle}
                color="bg-red-600"
                subtitle={`${anomalyRatePercent}% anomaly rate`}
              />
              <StatCard
                title="High Risk Alerts"
                value={stats.high_risk_count.toLocaleString()}
                icon={TrendingUp}
                color="bg-yellow-600"
                subtitle="Requires immediate attention"
              />
              <StatCard
                title="Active Alerts"
                value={stats.recent_alerts.length}
                icon={Database}
                color="bg-purple-600"
                subtitle="Unresolved issues"
              />
            </div>

            {/* Charts Row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Hourly Trend Chart */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold mb-4">24-Hour Transaction Trend</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={stats.hourly_trend}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="hour" 
                      tickFormatter={(value) => new Date(value).getHours() + ':00'}
                    />
                    <YAxis />
                    <Tooltip 
                      labelFormatter={(value) => new Date(value).toLocaleString()}
                    />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="transactions" 
                      stroke="#8b5cf6" 
                      name="Sessions"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="anomalies" 
                      stroke="#ef4444" 
                      name="Anomalies"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Pie Chart */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold mb-4">Session Distribution</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={pieData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {pieData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Most Affected ATMs/Locations */}
            {stats.problematic_terminals && stats.problematic_terminals.length > 0 && (
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold mb-4">Most Affected ATMs/Locations</h3>
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">ATM ID</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Location</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Anomalies</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Critical</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Risk Score</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {stats.problematic_terminals.map((terminal, idx) => (
                        <tr key={idx}>
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{terminal.terminal_id}</td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{terminal.location}</td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{terminal.total_anomalies}</td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{terminal.critical_count}</td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <span className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${
                              terminal.avg_score > 0.8 
                                ? 'bg-red-100 text-red-800'
                                : terminal.avg_score > 0.6
                                ? 'bg-yellow-100 text-yellow-800'
                                : 'bg-green-100 text-green-800'
                            }`}>
                              {terminal.avg_score.toFixed(2)}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Recent Alerts */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold mb-4">Recent Alerts</h3>
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {stats.recent_alerts.length === 0 ? (
                  <p className="text-gray-500 text-center py-8">No active alerts</p>
                ) : (
                  stats.recent_alerts.map((alert, index) => (
                    <AlertItem key={alert.id || index} alert={alert} />
                  ))
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'anomalies' && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold mb-4">Detected Anomalies</h3>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Time
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Session ID
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Type
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Patterns
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Score
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {anomalies.map((anomaly, index) => (
                    <tr key={anomaly.id || index} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {anomaly.timestamp ? new Date(anomaly.timestamp).toLocaleString() : 'N/A'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-gray-900">
                        {anomaly.session_id}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {anomaly.anomaly_type || 'Unknown'}
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-900">
                        {anomaly.detected_patterns?.join(', ') || 'None'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${
                          anomaly.anomaly_score > 0.8 
                            ? 'bg-red-100 text-red-800'
                            : anomaly.anomaly_score > 0.6
                            ? 'bg-yellow-100 text-yellow-800'
                            : 'bg-green-100 text-green-800'
                        }`}>
                          {anomaly.anomaly_score.toFixed(3)}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {activeTab === 'multi-anomaly' && (
          <MultiAnomalyView />
        )}

        {activeTab === 'alerts' && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold mb-4">Active Alerts</h3>
              <div className="space-y-3">
                {stats.recent_alerts.length === 0 ? (
                  <p className="text-gray-500 text-center py-8">No active alerts</p>
                ) : (
                  stats.recent_alerts.map((alert, index) => (
                    <AlertItem key={alert.id || index} alert={alert} />
                  ))
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'expert-labeling' && (
          <ExpertLabelingInterface />
        )}

        {activeTab === 'continuous-learning' && (
          <ContinuousLearningInterface />
        )}

        {activeTab === 'monitoring' && (
          <RealtimeMonitoringInterface />
        )}

        {activeTab === 'svm-debug' && (
          <SVMDebugDashboard />
        )}

        {activeTab === 'analytics' && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold mb-4">ML Analytics Dashboard</h3>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                  <h4 className="text-md font-medium mb-3">Model Status</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Isolation Forest</span>
                      <span className="text-sm font-medium text-green-600">Active</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">One-Class SVM</span>
                      <span className="text-sm font-medium text-green-600">Active</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Autoencoder</span>
                      <span className="text-sm font-medium text-green-600">Active</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">BERT Embeddings</span>
                      <span className="text-sm font-medium text-green-600">Active</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Supervised Classifier</span>
                      <span className="text-sm font-medium text-gray-400">Not Trained</span>
                    </div>
                  </div>
                </div>
                <div>
                  <h4 className="text-md font-medium mb-3">Processing Stats</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Processing Mode</span>
                      <span className="text-sm font-medium">ML-First</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Average Processing Time</span>
                      <span className="text-sm font-medium">1.2s/session</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Model Update Interval</span>
                      <span className="text-sm font-medium">1 hour</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Debug Info for Development */}
            {process.env.NODE_ENV === 'development' && (
              <div className="bg-gray-100 border-l-4 border-blue-500 p-4">
                <p className="text-sm text-gray-700">
                  <strong>Debug Info:</strong> API URL: {API_URL}
                </p>
                <p className="text-sm text-gray-600">
                  Last fetch: {new Date().toLocaleString()}
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );

  // Return with or without Layout based on routing
  if (shouldUseLayout) {
    return (
      <Layout>
        <DashboardContent />
      </Layout>
    );
  }

  return <DashboardContent />;
};

export default ATMDashboard;
