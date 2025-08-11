import React, { useState, useEffect } from 'react';
import { LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { AlertCircle, Activity, TrendingUp, Database } from 'lucide-react';
import { useLocation } from 'react-router-dom';
import apiConfig from './config/api';
import ExpertLabelingInterface from './ExpertLabelingInterface';
import ContinuousLearningInterface from './ContinuousLearningInterface';
import MultiAnomalyView from './MultiAnomalyView';
import RealtimeMonitoringInterface from './RealtimeMonitoringInterface';
import SVMDebugDashboard from './SVMDebugDashboard';

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

  // Get current tab from URL
  const getCurrentTab = () => {
    const path = location.pathname;
    if (path === '/dashboard' || path === '/dashboard/') return 'overview';
    if (path.includes('/dashboard/anomalies')) return 'anomalies';
    if (path.includes('/dashboard/multi-anomaly')) return 'multi-anomaly';
    if (path.includes('/dashboard/alerts')) return 'alerts';
    if (path.includes('/dashboard/expert-labeling')) return 'expert-labeling';
    if (path.includes('/dashboard/continuous-learning')) return 'continuous-learning';
    if (path.includes('/dashboard/session-review')) return 'session-review';
    if (path.includes('/dashboard/realtime')) return 'monitoring';
    if (path.includes('/dashboard/analytics')) return 'analytics';
    if (path.includes('/dashboard/svm-debug')) return 'svm-debug';
    return 'overview';
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

  if (loading && getCurrentTab() === 'overview') {
    return (
      <div className="flex items-center justify-center min-h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading dashboard...</p>
        </div>
      </div>
    );
  }

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
      <>
        <div className="min-h-screen bg-gray-100 flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
            <p className="mt-4 text-gray-600">Loading dashboard...</p>
          </div>
        </div>
      </>
    );
  }

  // Check if we're using Layout or direct rendering based on tabs

  return (
    <>
      <div className="space-y-6">
        {getCurrentTab() === 'overview' && (
          <div>
            <h2 className="text-2xl font-bold mb-6">Overview Dashboard</h2>
            <p>Dashboard content will be here</p>
          </div>
        )}

        {getCurrentTab() === 'session-review' && (
          <div>
            <h2 className="text-2xl font-bold mb-6">Session Review</h2>
            <p>Session review content will be here</p>
          </div>
        )}

        {getCurrentTab() === 'multi-anomaly' && <MultiAnomalyView />}
        {getCurrentTab() === 'anomalies' && <AnomaliesPage />}
        {getCurrentTab() === 'alerts' && <AlertsPage />}
        {getCurrentTab() === 'expert-labeling' && <ExpertLabelingInterface />}
        {getCurrentTab() === 'continuous-learning' && <ContinuousLearningInterface />}
        {getCurrentTab() === 'monitoring' && <RealtimeMonitoringInterface />}
        {getCurrentTab() === 'svm-debug' && <SVMDebugDashboard />}
        {getCurrentTab() === 'analytics' && (
          <div>
            <h2 className="text-2xl font-bold mb-6">Analytics</h2>
            <p>Analytics content will be here</p>
          </div>
        )}
      </div>
    </>
  );
};

export default ATMDashboard;
