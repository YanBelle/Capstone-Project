import React, { useState, useEffect } from 'react';
import { LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { AlertCircle, Activity, TrendingUp, Clock, Brain, Database } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import apiConfig from './config/api';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const SimpleDashboard = () => {
  const navigate = useNavigate();
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

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  return (
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
            {[
              { key: 'overview', label: 'Overview', path: '/dashboard' },
              { key: 'anomalies', label: 'Anomalies', path: '/dashboard/anomalies' },
              { key: 'multi-anomaly', label: 'Multi-Anomaly', path: '/dashboard/multi-anomaly' },
              { key: 'alerts', label: 'Alerts', path: '/dashboard/alerts' },
              { key: 'expert-labeling', label: 'Expert Review', path: '/dashboard/expert-labeling' },
              { key: 'continuous-learning', label: 'ML Training', path: '/dashboard/continuous-learning' },
              { key: 'bert-analysis', label: 'BERT Analysis', path: '/dashboard/bert-analysis' },
              { key: 'deeplog', label: 'DeepLog', path: '/dashboard/deeplog' },
              { key: 'monitoring', label: 'Real-time Monitor', path: '/dashboard/realtime' },
              { key: 'analytics', label: 'Analytics', path: '/dashboard/analytics' }
            ].map((tab) => (
              <button
                key={tab.key}
                onClick={() => navigate(tab.path)}
                className={`py-3 px-1 border-b-2 font-medium text-sm ${
                  tab.key === 'overview'
                    ? 'border-purple-600 text-purple-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
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
        </div>
      </div>
    </div>
  );
};

export default SimpleDashboard;
