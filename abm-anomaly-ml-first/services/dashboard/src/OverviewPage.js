import React, { useState, useEffect } from 'react';
import { XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { 
  AlertCircle, 
  Activity, 
  TrendingUp, 
  Clock, 
  Database, 
  Server, 
  DollarSign,
  MonitorSpeaker,
  ShieldAlert,
  CheckCircle,
  XCircle
} from 'lucide-react';
import apiConfig from './config/api';

const OverviewPage = () => {
  const [overviewData, setOverviewData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchOverviewData = async () => {
    try {
      setLoading(true);
      const response = await fetch(apiConfig.endpoint('api/v1/overview/stats'));
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setOverviewData(data);
      setError(null);
    } catch (err) {
      console.error('Error fetching overview data:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchOverviewData();
    // Refresh every 5 minutes
    const interval = setInterval(fetchOverviewData, 300000);
    return () => clearInterval(interval);
  }, []);

  const StatCard = ({ title, value, icon: Icon, color, subtitle, trend }) => (
    <div className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
      <div className="flex items-center">
        <div className={`flex-shrink-0 ${color} rounded-md p-3`}>
          <Icon className="h-6 w-6 text-white" />
        </div>
        <div className="ml-5 w-0 flex-1">
          <dl>
            <dt className="text-sm font-medium text-gray-500 truncate">{title}</dt>
            <dd className="flex items-baseline">
              <div className="text-2xl font-semibold text-gray-900">{value}</div>
              {trend && (
                <div className={`ml-2 flex items-baseline text-sm font-semibold ${
                  trend > 0 ? 'text-green-600' : trend < 0 ? 'text-red-600' : 'text-gray-500'
                }`}>
                  {trend > 0 ? '+' : ''}{trend}%
                </div>
              )}
            </dd>
            {subtitle && <div className="text-sm text-gray-500 mt-1">{subtitle}</div>}
          </dl>
        </div>
      </div>
    </div>
  );

  const ActivityItem = ({ activity }) => (
    <div className="flex items-center space-x-3 p-3 bg-gray-50 rounded-md">
      <div className={`flex-shrink-0 h-2 w-2 rounded-full ${
        activity.type === 'anomaly_detected' ? 'bg-red-500' : 'bg-blue-500'
      }`}></div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-gray-900 truncate">
          {activity.description}
        </p>
        <p className="text-sm text-gray-500">
          Session: {activity.session_id} • Score: {activity.score?.toFixed(3)}
        </p>
      </div>
      <div className="text-xs text-gray-400">
        {new Date(activity.timestamp).toLocaleTimeString()}
      </div>
    </div>
  );

  const SystemHealthIndicator = ({ label, value, status }) => (
    <div className="flex items-center justify-between py-2">
      <span className="text-sm text-gray-600">{label}</span>
      <div className="flex items-center space-x-2">
        <span className="text-sm font-medium">{value}</span>
        {status === 'healthy' ? (
          <CheckCircle className="h-4 w-4 text-green-500" />
        ) : status === 'warning' ? (
          <AlertCircle className="h-4 w-4 text-yellow-500" />
        ) : (
          <XCircle className="h-4 w-4 text-red-500" />
        )}
      </div>
    </div>
  );

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading overview dashboard...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-center">
          <XCircle className="h-12 w-12 text-red-600 mx-auto mb-4" />
          <p className="text-red-600 mb-4">Error loading overview: {error}</p>
          <button 
            onClick={fetchOverviewData}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  const anomalyRatePercent = (overviewData.anomaly_rate * 100).toFixed(2);

  return (
    <div className="min-h-screen bg-gray-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">System Overview</h1>
          <p className="mt-2 text-gray-600">
            Real-time monitoring of anomaly detection and cash forecasting systems
          </p>
          <div className="mt-2 text-sm text-gray-500">
            Last updated: {new Date().toLocaleString()}
          </div>
        </div>

        {/* Key Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <StatCard
            title="Total Sessions"
            value={overviewData.total_sessions.toLocaleString()}
            icon={Activity}
            color="bg-blue-600"
            subtitle="Processed sessions"
          />
          <StatCard
            title="Anomalies Detected"
            value={overviewData.total_anomalies.toLocaleString()}
            icon={AlertCircle}
            color="bg-red-600"
            subtitle={`${anomalyRatePercent}% anomaly rate`}
          />
          <StatCard
            title="High Risk Alerts"
            value={overviewData.high_risk_count.toLocaleString()}
            icon={ShieldAlert}
            color="bg-orange-600"
            subtitle="Immediate attention required"
          />
          <StatCard
            title="Critical Alerts"
            value={overviewData.critical_alerts.toLocaleString()}
            icon={AlertCircle}
            color="bg-red-700"
            subtitle="Urgent response needed"
          />
        </div>

        {/* Second Row - Terminal and Cash Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <StatCard
            title="Active Terminals"
            value={overviewData.terminal_summary.active_terminals}
            icon={MonitorSpeaker}
            color="bg-green-600"
            subtitle={`${overviewData.terminal_summary.total_terminals} total terminals`}
          />
          <StatCard
            title="Terminals at Risk"
            value={overviewData.terminal_summary.terminals_at_risk}
            icon={TrendingUp}
            color="bg-yellow-600"
            subtitle="Low cash levels"
          />
          <StatCard
            title="Total Cash Monitored"
            value={`$${(overviewData.cash_summary.total_cash_monitored / 1000000).toFixed(1)}M`}
            icon={DollarSign}
            color="bg-green-700"
            subtitle="Across all terminals"
          />
          <StatCard
            title="Predicted Depletions"
            value={overviewData.cash_summary.predicted_depletions_24h}
            icon={Clock}
            color="bg-red-500"
            subtitle="Next 24 hours"
          />
        </div>

        {/* Charts and System Health Row */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* Hourly Activity Trend */}
          <div className="lg:col-span-2 bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold mb-4">24-Hour Activity Trend</h3>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={overviewData.hourly_trend}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="hour" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Area 
                  type="monotone" 
                  dataKey="total_sessions" 
                  stackId="1"
                  stroke="#3B82F6" 
                  fill="#3B82F6" 
                  name="Total Sessions"
                  fillOpacity={0.6}
                />
                <Area 
                  type="monotone" 
                  dataKey="anomalies" 
                  stackId="2"
                  stroke="#EF4444" 
                  fill="#EF4444" 
                  name="Anomalies"
                  fillOpacity={0.8}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* System Health */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <Server className="w-5 h-5 mr-2" />
              System Health
            </h3>
            <div className="space-y-3">
              <SystemHealthIndicator 
                label="System Status" 
                value={overviewData.system_health.status}
                status="healthy"
              />
              <SystemHealthIndicator 
                label="Uptime" 
                value={`${overviewData.system_health.uptime_hours}h`}
                status="healthy"
              />
              <SystemHealthIndicator 
                label="CPU Usage" 
                value={`${overviewData.system_health.cpu_usage}%`}
                status={overviewData.system_health.cpu_usage > 80 ? "warning" : "healthy"}
              />
              <SystemHealthIndicator 
                label="Memory Usage" 
                value={`${overviewData.system_health.memory_usage}%`}
                status={overviewData.system_health.memory_usage > 85 ? "warning" : "healthy"}
              />
              <SystemHealthIndicator 
                label="Database" 
                value={overviewData.system_health.database_status}
                status={overviewData.system_health.database_status === "connected" ? "healthy" : "error"}
              />
              <SystemHealthIndicator 
                label="Redis Cache" 
                value={overviewData.system_health.redis_status}
                status={overviewData.system_health.redis_status === "connected" ? "healthy" : "error"}
              />
            </div>
          </div>
        </div>

        {/* Recent Activity and Cash Status */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Recent Activity */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <Clock className="w-5 h-5 mr-2" />
              Recent Activity
            </h3>
            <div className="space-y-3 max-h-80 overflow-y-auto">
              {overviewData.recent_activity.length === 0 ? (
                <p className="text-gray-500 text-center py-8">No recent activity</p>
              ) : (
                overviewData.recent_activity.map((activity, index) => (
                  <ActivityItem key={index} activity={activity} />
                ))
              )}
            </div>
          </div>

          {/* Terminal Status Summary */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <Database className="w-5 h-5 mr-2" />
              Terminal Status Summary
            </h3>
            <div className="space-y-4">
              <div className="flex justify-between items-center p-3 bg-green-50 rounded-md">
                <span className="text-sm font-medium text-green-800">Healthy Terminals</span>
                <span className="text-2xl font-bold text-green-600">
                  {overviewData.terminal_summary.terminals_healthy}
                </span>
              </div>
              <div className="flex justify-between items-center p-3 bg-yellow-50 rounded-md">
                <span className="text-sm font-medium text-yellow-800">At Risk Terminals</span>
                <span className="text-2xl font-bold text-yellow-600">
                  {overviewData.terminal_summary.terminals_at_risk}
                </span>
              </div>
              <div className="flex justify-between items-center p-3 bg-red-50 rounded-md">
                <span className="text-sm font-medium text-red-800">Critical Terminals</span>
                <span className="text-2xl font-bold text-red-600">
                  {overviewData.cash_summary.critical_terminals}
                </span>
              </div>
              <div className="mt-4 pt-4 border-t border-gray-200">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Average Cash Level</span>
                  <span className="font-medium">
                    {overviewData.terminal_summary.average_cash_level.toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default OverviewPage;
