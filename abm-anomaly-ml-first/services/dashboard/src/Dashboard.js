import React, { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { AlertCircle, Activity, TrendingUp, Clock, Database, Brain } from 'lucide-react';
import { useLocation } from 'react-router-dom';
import apiConfig from './config/api';
import ExpertLabelingInterface from './ExpertLabelingInterface';
import ContinuousLearningInterface from './ContinuousLearningInterface';
import MultiAnomalyView from './MultiAnomalyView';
import RealtimeMonitoringInterface from './RealtimeMonitoringInterface';
import SVMDebugDashboard from './SVMDebugDashboard';
import AnomaliesPage from './AnomaliesPage';
import AlertsPage from './AlertsPage';
import OverviewPage from './OverviewPage';
import AnalyticsPage from './AnalyticsPage';

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
  
  const [loading, setLoading] = useState(true);
  const [processing, setProcessing] = useState(false);
  const [message, setMessage] = useState('');

  // Get current tab from URL or state
  const getCurrentTab = useCallback(() => {
    if (activeTab && activeTab !== 'overview') return activeTab;
    
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
  }, [activeTab, location.pathname]);

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
      alert('File uploaded successfully!');
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('Error uploading file: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  // Handle force process input directory
  const handleForceProcessInput = async () => {
    try {
      setProcessing(true);
      setMessage('Processing input files...');
      
      const response = await fetch(`${apiConfig.BASE_URL}/process/force-input`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      const data = await response.json();
      
      if (response.ok) {
        if (data.files_processed > 0) {
          setMessage(`Successfully processed ${data.files_processed} EJ files`);
          // Refresh dashboard data
          fetchStats();
        } else {
          setMessage(data.message || 'No files found to process');
        }
      } else {
        setMessage(`Error: ${data.detail || 'Failed to process input'}`);
      }
    } catch (error) {
      console.error('Force process input error:', error);
      setMessage('Error processing input files. Please check the console for details.');
    } finally {
      setProcessing(false);
      // Clear message after 5 seconds
      setTimeout(() => setMessage(''), 5000);
    }
  };  // Handle clear all data
  const handleClearAllData = async () => {
    if (!window.confirm('⚠️ WARNING: This will permanently delete ALL transactions, sessions, and training data.\n\nThis action cannot be undone!\n\nAre you sure you want to proceed?')) {
      return;
    }

    if (!window.confirm('🔴 FINAL CONFIRMATION: Are you absolutely certain you want to clear all data? This will remove everything and reset the system.')) {
      return;
    }

    try {
      setLoading(true);
      console.log('Clearing all data...');
      
      const response = await fetch(apiConfig.endpoint('/api/v1/data/clear-all?confirm=true'), {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errorText}`);
      }

      const result = await response.json();
      console.log('Clear data result:', result);
      
      // Refresh data after clearing
      await fetchStats();
      
      if (result.status === 'success') {
        alert(`✅ Data cleared successfully!\n\nRecords deleted: ${result.total_records_deleted || 'N/A'}\nTables cleared: ${result.deleted_counts ? Object.keys(result.deleted_counts).length : 'N/A'}\nRedis cleared: ${result.redis_cleared ? 'Yes' : 'No'}\n\nYou can now upload new sessions for training.`);
      } else {
        alert(`⚠️ Clear operation completed with issues:\n\n${result.message}\n\nSome data may not have been cleared properly.`);
      }
    } catch (error) {
      console.error('Error clearing data:', error);
      alert('❌ Error clearing data: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  // Fetch dashboard stats
  const fetchStats = async () => {
    try {
      setLoading(true);
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
      // Show actual error instead of misleading mock data
      alert(`Failed to fetch dashboard data: ${error.message}`);
      // Use empty data structure to reflect actual state
      const emptyData = {
        total_transactions: 0,
        total_anomalies: 0,
        anomaly_rate: 0.0,
        high_risk_count: 0,
        recent_alerts: [],
        hourly_trend: [],
        problematic_terminals: []
      };
      setStats(emptyData);
    } finally {
      setLoading(false);
    }
  };

  // Fetch anomalies - removed as it's now handled by individual components

  useEffect(() => {
    fetchStats();
    
    // Refresh every 5 minutes instead of 30 seconds to reduce API calls
    const interval = setInterval(() => {
      fetchStats();
    }, 300000); // 5 minutes = 300000ms

    return () => {
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    // Update active tab based on current location
    const path = location.pathname;
    let currentTab = 'overview';
    
    if (path === '/dashboard' || path === '/dashboard/') currentTab = 'overview';
    else if (path.includes('/dashboard/anomalies')) currentTab = 'anomalies';
    else if (path.includes('/dashboard/alerts')) currentTab = 'alerts';
    else if (path.includes('/dashboard/analytics')) currentTab = 'analytics';
    
    if (currentTab !== activeTab) {
      setActiveTab(currentTab);
    }
  }, [location.pathname, activeTab]); // Removed getCurrentTab dependency

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
        {getCurrentTab() === 'overview' && <OverviewPage />}

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
        {getCurrentTab() === 'analytics' && <AnalyticsPage />}
      </div>
    </>
  );
};

export default ATMDashboard;
