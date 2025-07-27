import React, { useState, useEffect } from 'react';
import { AlertCircle, Activity, TrendingUp, Database } from 'lucide-react';
import { useLocation } from 'react-router-dom';
import Layout from './Layout';
import apiConfig from './config/api';

const ATMDashboard = () => {
  const location = useLocation();
  const [stats, setStats] = useState({
    total_transactions: 0,
    total_anomalies: 0,
    anomaly_rate: 0,
    high_risk_count: 0,
    recent_alerts: [],
    hourly_trend: []
  });
  
  const [anomalies, setAnomalies] = useState([]);
  const [loading, setLoading] = useState(true);

  // Get current tab from URL
  const getCurrentTab = () => {
    const path = location.pathname;
    if (path === '/dashboard' || path === '/dashboard/') return 'overview';
    if (path.includes('/dashboard/anomalies')) return 'anomalies';
    if (path.includes('/dashboard/alerts')) return 'alerts';
    if (path.includes('/dashboard/analytics')) return 'analytics';
    return 'overview';
  };

  // Mock fetch for now - will replace with real API later
  const fetchStats = async () => {
    try {
      console.log('Would fetch dashboard stats from:', apiConfig.endpoint('/api/v1/dashboard/stats'));
      // Mock data for testing
      setStats({
        total_transactions: 1250,
        total_anomalies: 23,
        anomaly_rate: 1.84,
        high_risk_count: 5,
        recent_alerts: [],
        hourly_trend: []
      });
      setLoading(false);
    } catch (error) {
      console.error('Error fetching stats:', error);
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
  }, []);

  const renderOverview = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
      <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-100">
        <div className="flex items-center">
          <div className="flex-shrink-0">
            <Database className="h-8 w-8 text-blue-500" />
          </div>
          <div className="ml-5 w-0 flex-1">
            <dl>
              <dt className="text-sm font-medium text-gray-500 truncate">
                Total Transactions
              </dt>
              <dd className="text-lg font-medium text-gray-900">
                {stats.total_transactions.toLocaleString()}
              </dd>
            </dl>
          </div>
        </div>
      </div>

      <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-100">
        <div className="flex items-center">
          <div className="flex-shrink-0">
            <AlertCircle className="h-8 w-8 text-red-500" />
          </div>
          <div className="ml-5 w-0 flex-1">
            <dl>
              <dt className="text-sm font-medium text-gray-500 truncate">
                Total Anomalies
              </dt>
              <dd className="text-lg font-medium text-gray-900">
                {stats.total_anomalies.toLocaleString()}
              </dd>
            </dl>
          </div>
        </div>
      </div>

      <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-100">
        <div className="flex items-center">
          <div className="flex-shrink-0">
            <TrendingUp className="h-8 w-8 text-yellow-500" />
          </div>
          <div className="ml-5 w-0 flex-1">
            <dl>
              <dt className="text-sm font-medium text-gray-500 truncate">
                Anomaly Rate
              </dt>
              <dd className="text-lg font-medium text-gray-900">
                {stats.anomaly_rate.toFixed(2)}%
              </dd>
            </dl>
          </div>
        </div>
      </div>

      <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-100">
        <div className="flex items-center">
          <div className="flex-shrink-0">
            <Activity className="h-8 w-8 text-green-500" />
          </div>
          <div className="ml-5 w-0 flex-1">
            <dl>
              <dt className="text-sm font-medium text-gray-500 truncate">
                High Risk Count
              </dt>
              <dd className="text-lg font-medium text-gray-900">
                {stats.high_risk_count}
              </dd>
            </dl>
          </div>
        </div>
      </div>
    </div>
  );

  const renderContent = () => {
    const tab = getCurrentTab();
    
    switch (tab) {
      case 'overview':
        return (
          <div>
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Dashboard Overview</h2>
            {renderOverview()}
            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-100">
              <h3 className="text-lg font-medium text-gray-900 mb-4">System Status</h3>
              <p className="text-gray-600">All anomaly detection systems are operational.</p>
            </div>
          </div>
        );
      
      case 'anomalies':
        return (
          <div>
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Anomaly Detection</h2>
            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-100">
              <p className="text-gray-600">Anomaly detection results will be displayed here.</p>
            </div>
          </div>
        );
      
      case 'alerts':
        return (
          <div>
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Alerts & Notifications</h2>
            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-100">
              <p className="text-gray-600">Recent alerts and notifications will be shown here.</p>
            </div>
          </div>
        );
      
      case 'analytics':
        return (
          <div>
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Analytics & Reports</h2>
            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-100">
              <p className="text-gray-600">Analytics and detailed reports will be available here.</p>
            </div>
          </div>
        );
      
      default:
        return renderOverview();
    }
  };

  if (loading) {
    return (
      <Layout>
        <div className="flex justify-center items-center h-64">
          <div className="text-gray-500">Loading dashboard...</div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="p-6">
        {renderContent()}
      </div>
    </Layout>
  );
};

export default ATMDashboard;
